"""PLE integration tests. Hardware runs are serialized by the stage owner.

The convolution test isolates request ownership, logical tails, and EOS-independent
history. The real-weight test loads the complete table, including rows not present
in the test prompts; its HF control reads requested checkpoint rows on the host.
The HF control is outside the measured/device implementation.
"""
import json
import os
from pathlib import Path

import torch
from safetensors import safe_open

import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tests.metrics import pcc
from models.autoports.qwen_qwen3_8_flash_next.tt.ple import PLE, _conv_taps, _mask_padding

CHECKPOINT = Path(
    os.environ.get(
        "QWEN38_CHECKPOINT",
        "/mnt/models/huggingface/hub/models--Qwen--Qwen3.8-Flash-Next/snapshots/de4b8e4d43b917e7706784d8bb445c9af86a3540",
    )
)


def controls(values, mesh):
    tensor = torch.zeros(len(values), 32, dtype=torch.int32)
    tensor[:, 0] = torch.tensor(values)
    return ttnn.from_torch(tensor, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh)


def test_conv_request_history_and_padding():
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    try:
        torch.manual_seed(20261014)
        cpu_history = torch.zeros(4, 9, 10240, dtype=torch.bfloat16)
        history = ttnn.from_torch(
            cpu_history.reshape(36, 10240), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh
        )
        # Changed batch ordering, short/fresh requests, nine-vector boundary,
        # non-aligned tails, and repeated request slots on one loaded history.
        cases = [
            (7, [7, 4], [0, 0], [3, 1]),
            (1, [1, 1], [4, 7], [1, 3]),
            (32, [32, 31], [8, 5], [3, 1]),
            (33, [33, 30], [0, 40], [1, 3]),
            (1, [1, 0], [33, 70], [1, 3]),
        ]
        for seq, lengths, positions, slots in cases:
            x = torch.randn(2 * seq, 10240, dtype=torch.bfloat16)
            device_x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh)
            pos, slot, valid = (controls(v, mesh) for v in (positions, slots, lengths))
            output = ttnn.allocate_tensor_on_device(
                (4, 2 * seq, 10240), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG
            )
            expected = torch.zeros(4, 2 * seq, 10240, dtype=torch.bfloat16)
            for b, (length, position, request) in enumerate(zip(lengths, positions, slots)):
                old = cpu_history[request].clone() if position else torch.zeros_like(cpu_history[request])
                joined = torch.cat((old, x[b * seq : b * seq + length]))
                for tap in range(4):
                    expected[tap, b * seq : b * seq + length] = joined[tap * 3 : tap * 3 + length]
                cpu_history[request] = joined[-9:]
            _conv_taps(device_x, pos, slot, valid, history, output=output, batch=2, seq=seq)
            torch.testing.assert_close(ttnn.to_torch(output), expected, rtol=0, atol=0)
            torch.testing.assert_close(ttnn.to_torch(history).reshape_as(cpu_history), cpu_history, rtol=0, atol=0)
            _mask_padding(device_x, valid, batch=2, seq=seq)
            for b, length in enumerate(lengths):
                x[b * seq + length : (b + 1) * seq] = 0
            torch.testing.assert_close(ttnn.to_torch(device_x), x, rtol=0, atol=0)
        print("PLE_CONV_REQUEST_HISTORY_AND_LOGICAL_PADDING_BIT_EXACT")
    finally:
        ttnn.close_mesh_device(mesh)


class LazyCheckpoint:
    def __init__(self, path=CHECKPOINT):
        self.path = Path(path)
        self.index = json.loads((self.path / "model.safetensors.index.json").read_text())["weight_map"]

    def __getitem__(self, name):
        with safe_open(self.path / self.index[name], framework="pt", device="cpu") as handle:
            return handle.get_tensor(name)

    def rows(self, name, start, end):
        with safe_open(self.path / self.index[name], framework="pt", device="cpu") as handle:
            return handle.get_slice(name)[start:end]


class ExactCheckpointTable(torch.nn.Module):
    def __init__(self, checkpoint):
        super().__init__()
        self.checkpoint = checkpoint
        self.weight = torch.empty(0, dtype=torch.bfloat16)

    def forward(self, ids):
        rows = []
        for value in ids.flatten().tolist():
            shard, offset = divmod(value, 2500012)
            key = f"model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_{shard}.weight"
            rows.append(self.checkpoint.rows(key, offset, offset + 1)[0])
        return torch.stack(rows).reshape(*ids.shape, 160)


def test_real_weights_full_table_ple():
    """Expensive full resident table test: ~29 GB TT memory, no subset loading."""
    from transformers import AutoConfig, AutoTokenizer
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextPLELayer

    checkpoint = LazyCheckpoint()
    cfg = AutoConfig.from_pretrained(checkpoint.path).text_config
    tokenizer = AutoTokenizer.from_pretrained(checkpoint.path)
    with torch.device("meta"):
        reference = Qwen4ExpTextPLELayer(cfg, 1, 0)
    prefix = "model.language_model.layers.1.ple."
    weights = {
        k[len(prefix) :]: checkpoint[k]
        for k in checkpoint.index
        if k.startswith(prefix) and "ngram_embedding.shard_" not in k
    }
    missing, extra = reference.load_state_dict(weights, strict=False, assign=True)
    assert missing == ["ple_embedding.ngram_embedding.weight"] and not extra
    reference.ple_embedding.ngram_embedding = ExactCheckpointTable(checkpoint)
    reference.eval()
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    try:
        module = PLE.from_state_dict(
            checkpoint,
            hf_config=cfg,
            mesh_device=mesh,
            cache_dir=Path("/home/vkovacevic/qwen38-flash/tt-metal-cache/qwen_qwen3_8_flash_next/de4b8e4_ple"),
        )
        from models.autoports.qwen_qwen3_8_flash_next.tt.ple_lookup import lookup_rows

        # This covers final bank addresses and every tile-face/shard boundary,
        # with all 128 tensor runtime arguments live in the same operation.
        probe_ids = [
            0,
            15,
            16,
            31,
            32,
            2500011,
            2500012,
            2500013,
            10 * 2500012 - 1,
            10 * 2500012,
            64 * 2500012,
            100 * 2500012,
            127 * 2500012,
            127 * 2500012 + 15,
            128 * 2500012 - 2,
            128 * 2500012 - 1,
        ]
        ids_device = ttnn.from_torch(
            torch.tensor([probe_ids], dtype=torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh
        )
        actual_rows = ttnn.to_torch(lookup_rows(module.tables, ids_device)).reshape(16, 160)
        expected_rows = []
        for row in probe_ids:
            shard, offset = divmod(row, 2500012)
            start = offset // 32 * 32
            block = checkpoint.rows(
                prefix + f"ple_embedding.ngram_embedding.shard_{shard}.weight", start, min(start + 32, 2500012)
            )
            encoded = ttnn.from_torch(block, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT)
            expected_rows.append(ttnn.to_torch(encoded)[offset - start].to(torch.bfloat16))
        torch.testing.assert_close(actual_rows, torch.stack(expected_rows), rtol=0, atol=0)
        print("PLE_FULL128_HIGH_ADDRESS_TILE_FACE_AND_SHARD_BOUNDARIES_BIT_EXACT")
        cache = module.make_cache(max_requests=2)
        texts = [
            "Explain why the sky appears blue and how this changes at sunset.",
            "Write a Python function to sort a list of integers and explain its complexity.",
        ]
        all_tokens = [tokenizer(t, return_tensors="pt").input_ids for t in texts]
        for ids in all_tokens:
            module.prepare(batch=1, physical_seq_len=ids.shape[-1])
        for ids in all_tokens:
            seq = ids.shape[-1]
            x = checkpoint["model.language_model.embed_tokens.weight"][ids].repeat(1, 1, 4)
            with torch.no_grad():
                expected = reference(x, ids, None)
            device_x = ttnn.from_torch(x.unsqueeze(0), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
            tokens = ttnn.from_torch(ids.to(torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh)
            kwargs = dict(
                tokens=tokens,
                positions=controls([0], mesh),
                slots=controls([1], mesh),
                valid_lengths=controls([seq], mesh),
                cache=cache,
            )
            output = module.forward(device_x, **kwargs)
            actual = ttnn.to_torch(output).squeeze(0)
            score = pcc(actual, expected)
            print(
                json.dumps(
                    {
                        "component": "PLE",
                        "full_table_shards": len(module.tables),
                        "length": seq,
                        "pcc": score,
                        "real_weights": True,
                    }
                )
            )
            # This is component qualification. Complete decoder acceptance must
            # still independently compare the real layer-1 output.
            assert score >= 0.99  # Diagnostic floor; complete decoder still requires PCC >= .995.
            repeated = ttnn.to_torch(module.forward(device_x, **kwargs)).squeeze(0)
            torch.testing.assert_close(actual, repeated, rtol=0, atol=0)
    finally:
        ttnn.close_mesh_device(mesh)


def test_conv_trace_changed_inputs_and_slots():
    """Persistent tap/history buffers and device controls change across replay."""
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    trace = None
    try:
        torch.manual_seed(151)
        cpu_history = torch.zeros(4, 9, 10240, dtype=torch.bfloat16)
        history = ttnn.from_torch(
            cpu_history.reshape(36, 10240), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh
        )
        x = ttnn.from_torch(
            torch.zeros(2, 10240, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh
        )
        pos, slot, valid = (controls(v, mesh) for v in ([0, 0], [3, 1], [1, 1]))
        output = ttnn.allocate_tensor_on_device(
            (4, 2, 10240), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG
        )

        def forward():
            _conv_taps(x, pos, slot, valid, history, output=output, batch=2, seq=1)

        forward()
        ttnn.synchronize_device(mesh)
        trace = ttnn.begin_trace_capture(mesh, cq_id=0)
        forward()
        ttnn.end_trace_capture(mesh, trace, cq_id=0)
        count = {3: 0, 1: 0}
        for step in range(13):
            order = [3, 1] if step % 2 == 0 else [1, 3]
            values = torch.randn(2, 10240, dtype=torch.bfloat16)
            if step == 11:
                count[1] = 0  # Reuse a formerly populated request slot.
                cpu_history[1].zero_()
            positions = [count[s] for s in order]
            lengths = [1, 0] if step == 4 else [1, 1]
            expected = torch.zeros(4, 2, 10240, dtype=torch.bfloat16)
            for lane, request in enumerate(order):
                if lengths[lane]:
                    joined = torch.cat((cpu_history[request], values[lane : lane + 1]))
                    expected[:, lane] = joined[[0, 3, 6, 9]]
                    cpu_history[request] = joined[-9:]
                    count[request] += 1

            def replace(dst, value, dtype):
                host = ttnn.from_torch(value, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT)
                ttnn.copy_host_to_device_tensor(host, dst)

            replace(x, values, ttnn.bfloat16)
            for dst, values32 in ((pos, positions), (slot, order), (valid, lengths)):
                host32 = torch.zeros(2, 32, dtype=torch.int32)
                host32[:, 0] = torch.tensor(values32)
                replace(dst, host32, ttnn.uint32)
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
            torch.testing.assert_close(ttnn.to_torch(output), expected, rtol=0, atol=0)
            torch.testing.assert_close(ttnn.to_torch(history).reshape_as(cpu_history), cpu_history, rtol=0, atol=0)
        print("PLE_CONV_CHANGED_INPUT_SLOT_REUSE_TRACE_BIT_EXACT")
    finally:
        if trace is not None:
            ttnn.release_trace(mesh, trace)
        ttnn.close_mesh_device(mesh)
