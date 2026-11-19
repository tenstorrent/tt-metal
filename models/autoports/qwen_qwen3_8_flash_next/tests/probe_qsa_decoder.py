# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Full real-weight layer-3 prefill and changed-input traced decode probe.

The reference is the installed HF Qwen4ExpTextDecoderLayer, including both
hyper-connections, QSA, actual top-10 routed experts and shared expert. Prefill
starts from empty state and returns every logical row. Explicit test boundaries
snapshot/restore K/V, raw indexer history, compressed keys and device position
around warmup/capture. No host-computed model intermediate is fed into TT.

Use --seq-lens 1 31 32 33 to cover small/tile-boundary/tail prompts on one
loaded decoder. Default is 33. --cpu-only prepares real HF fixtures without
importing TTNN or touching devices; it is not decoder acceptance evidence.
Long-context, batching and profiling require separate stage evidence.
"""

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import math
from pathlib import Path

import torch
from safetensors import safe_open

from models.autoports.qwen_qwen3_8_flash_next.tests.probe_qsa import ReferenceCache


def pcc(expected, actual):
    expected, actual = expected.float().flatten(), actual.float().flatten()
    if not torch.isfinite(expected).all() or not torch.isfinite(actual).all():
        return None
    result = torch.corrcoef(torch.stack([expected, actual]))[0, 1].item()
    return result if math.isfinite(result) else None


def metrics(expected, actual, residual):
    actual = actual.reshape(expected.shape)
    expected_delta = expected.float() - residual.float()
    actual_delta = actual.float() - residual.float()
    rows = [
        {
            "row": row,
            "pcc": pcc(expected[:, row], actual[:, row]),
            "contribution_pcc": pcc(expected_delta[:, row], actual_delta[:, row]),
        }
        for row in range(expected.shape[1])
    ]
    return {
        "pcc": pcc(expected, actual),
        "contribution_pcc": pcc(expected_delta, actual_delta),
        "finite": bool(torch.isfinite(actual).all()),
        "max_abs_error": (expected.float() - actual.float()).abs().max().item(),
        "per_row": rows,
    }


def metric_passes(row):
    def good(value):
        return value is not None and value >= 0.995

    return (
        row["finite"]
        and good(row["pcc"])
        and good(row["contribution_pcc"])
        and all(good(item["pcc"]) and good(item["contribution_pcc"]) for item in row["per_row"])
    )


def load_reference(snapshot):
    from transformers.models.qwen4_exp import modeling_qwen4_exp as hf
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig

    config = Qwen4ExpTextConfig(**json.loads((snapshot / "config.json").read_text())["text_config"])
    config._attn_implementation = "eager"
    assert config.layer_types[3] == "indexed_attention" and 4 not in config.ple_layer_ids
    index_path = snapshot / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())["weight_map"]
    prefix = "model.language_model.layers.3."
    shard_keys = {}
    for key, filename in index.items():
        if key.startswith(prefix):
            shard_keys.setdefault(filename, []).append(key)
    weights = {}
    for filename, keys in shard_keys.items():
        with safe_open(snapshot / filename, framework="pt") as shard:
            for key in keys:
                weights[key.removeprefix(prefix)] = shard.get_tensor(key)
    with torch.device("meta"):
        reference = hf.Qwen4ExpTextDecoderLayer(config, 3)
    reference.load_state_dict(weights, strict=True, assign=True)
    reference.eval()
    rope = hf.Qwen4ExpTextRotaryEmbedding(config)
    source = Path(inspect.getfile(hf))
    provenance = {
        "snapshot": str(snapshot),
        "transformers": importlib.metadata.version("transformers"),
        "reference_path": str(source),
        "reference_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "config_sha256": hashlib.sha256((snapshot / "config.json").read_bytes()).hexdigest(),
        "checkpoint_index_sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(),
        "checkpoint_shards": sorted(shard_keys),
        "weight_tensors": {
            name: {"shape": list(value.shape), "dtype": str(value.dtype)} for name, value in weights.items()
        },
        "advertised_context": config.max_position_embeddings,
    }
    return config, reference, rope, weights, provenance


@torch.inference_mode()
def make_fixture(reference, rope, *, length):
    generator = torch.Generator().manual_seed(1733 + length)
    prompt = (torch.randn(1, length, 10240, generator=generator) * 0.02).bfloat16()
    decode_inputs = [(torch.randn(1, 1, 10240, generator=generator) * 0.02).bfloat16() for _ in range(3)]
    assert all(not torch.equal(decode_inputs[i], decode_inputs[j]) for i in range(3) for j in range(i))
    all_hidden = torch.cat([prompt, *decode_inputs], dim=1)
    cos, sin = rope(all_hidden, torch.arange(length + 3).reshape(1, -1))
    causal = torch.zeros(1, 1, length, length, dtype=torch.bfloat16)
    causal.masked_fill_(torch.ones(length, length, dtype=torch.bool).triu(1), torch.finfo(torch.bfloat16).min)
    cache = ReferenceCache()
    expected_prefill = reference(
        prompt,
        position_embeddings=(cos[:, :length], sin[:, :length]),
        attention_mask=causal,
        past_key_values=cache,
    )
    expected_decode = []
    for step, inputs in enumerate(decode_inputs):
        visible = length + step + 1
        expected_decode.append(
            reference(
                inputs,
                position_embeddings=(cos[:, :visible], sin[:, :visible]),
                attention_mask=torch.zeros(1, 1, 1, visible, dtype=torch.bfloat16),
                past_key_values=cache,
            )
        )
    assert all(torch.isfinite(value).all() for value in [expected_prefill, *expected_decode])
    assert cache.k.shape[-2] == length + 3 and cache.raw.shape[1] == length + 3
    return {
        "prompt": prompt,
        "decode_inputs": decode_inputs,
        "expected_prefill": expected_prefill,
        "expected_decode": expected_decode,
    }


def device_case(model, mesh, fixture, *, length, capacity, report):
    from ttnn.tools import trace_allocation_tracker

    import ttnn

    permutation = torch.randperm(capacity // 32, generator=torch.Generator().manual_seed(1734 + length))
    # Guarantee a non-identity mapping, including a nonzero initial physical slot.
    if permutation[0] == 0:
        permutation = permutation.roll(1)
    permutation = permutation.reshape(1, -1).int()
    state = model.create_state(capacity=capacity, page_table=permutation)
    attention = state.attention
    state_names = ("k_cache", "v_cache", "raw_history", "compressed_keys", "position")
    mutable = {name: getattr(attention, name) for name in state_names}
    report.update({"seq_len": length, "physical_capacity": capacity, "page_table": permutation.tolist(), "decode": []})
    trace_id = None

    def host(value, *, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
        return ttnn.from_torch(value.contiguous(), dtype=dtype, layout=layout)

    def snapshot():
        return {
            name: host(ttnn.to_torch(tensor), dtype=tensor.dtype, layout=tensor.layout)
            for name, tensor in mutable.items()
        }

    def restore(saved):
        for name, tensor in mutable.items():
            ttnn.copy_host_to_device_tensor(saved[name], tensor)

    def position():
        return int(ttnn.to_torch(attention.position).reshape(-1)[0])

    prompt_device = ttnn.to_device(host(fixture["prompt"].unsqueeze(0)), mesh)
    refresh_hidden = [host(value.unsqueeze(0)) for value in fixture["decode_inputs"]]
    decode_device = ttnn.to_device(refresh_hidden[0], mesh)
    initial = snapshot()
    try:
        prefill = model.prefill_forward(prompt_device, state=state)
        ttnn.synchronize_device(mesh)
        actual_prefill = ttnn.to_torch(prefill).reshape(fixture["expected_prefill"].shape)
        report["prefill"] = metrics(fixture["expected_prefill"], actual_prefill, fixture["prompt"])
        report["position_after_prefill"] = position()
        assert report["position_after_prefill"] == length
        ttnn.deallocate(prefill)
        prefix_state = snapshot()

        # Compile the exact decode graph, then restore all request state. None
        # of the warmed/captured requests may consume the reference's prefix.
        warmed = model.decode_forward(decode_device, state=state)
        ttnn.synchronize_device(mesh)
        ttnn.deallocate(warmed)
        restore(prefix_state)
        trace_id = ttnn.begin_trace_capture(mesh, cq_id=0)
        output = model.decode_forward(decode_device, state=state)
        ttnn.end_trace_capture(mesh, trace_id, cq_id=0)
        # This exact output backing allocation belongs to the sole live trace.
        # Every replay fully overwrites it; synchronous readback completes before
        # another replay, and no device alias/consumer survives that ordering.
        trace_allocation_tracker.acknowledge_corruptible(output)

        replay_outputs, final_states = [], []
        for repetition in range(2):
            # Capture executes the forward and mutates state: restore AFTER it,
            # as well as before capture. The second round proves repeatability.
            restore(prefix_state)
            assert position() == length
            round_outputs = []
            for step, (inputs, expected) in enumerate(zip(fixture["decode_inputs"], fixture["expected_decode"])):
                ttnn.copy_host_to_device_tensor(refresh_hidden[step], decode_device)
                ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=True)
                actual = ttnn.to_torch(output).reshape(expected.shape)
                round_outputs.append(actual.clone())
                actual_position = position()
                assert actual_position == length + step + 1
                if repetition == 0:
                    report["decode"].append(
                        {"step": step, "position_after": actual_position, **metrics(expected, actual, inputs)}
                    )
            replay_outputs.append(round_outputs)
            final_states.append({name: ttnn.to_torch(tensor).clone() for name, tensor in mutable.items()})

        report["trace_id"] = str(trace_id)
        report["captures"] = 1
        report["replays"] = 6
        report["changed_inputs"] = 3
        report["reset_replay_equal"] = all(torch.equal(a, b) for a, b in zip(*replay_outputs))
        report["reset_state_equal"] = {
            name: torch.equal(final_states[0][name], final_states[1][name]) for name in state_names
        }
        ttnn.release_trace(mesh, trace_id)
        trace_id = None

        # Fresh request on the same loaded instance, with the identical page
        # permutation and restored initial state, after all traces are released.
        restore(initial)
        repeated = model.prefill_forward(prompt_device, state=state)
        report["prefill_reset_equal"] = torch.equal(
            actual_prefill, ttnn.to_torch(repeated).reshape(actual_prefill.shape)
        )
        ttnn.deallocate(repeated)
        report["passed"] = (
            all(metric_passes(row) for row in [report["prefill"], *report["decode"]])
            and report["reset_replay_equal"]
            and all(report["reset_state_equal"].values())
            and report["prefill_reset_equal"]
        )
    finally:
        if trace_id is not None:
            ttnn.release_trace(mesh, trace_id)
    return report["passed"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    lengths = parser.add_mutually_exclusive_group()
    lengths.add_argument("--seq-len", type=int, default=33)
    lengths.add_argument("--seq-lens", type=int, nargs="+")
    parser.add_argument("--capacity", type=int)
    parser.add_argument("--cpu-only", action="store_true")
    args = parser.parse_args()
    sequence_lengths = args.seq_lens or [args.seq_len]
    if any(length < 1 for length in sequence_lengths):
        parser.error("Logical prompt lengths must be positive")
    torch.set_num_threads(16)
    config, reference, rope, weights, provenance = load_reference(args.snapshot)
    if max(sequence_lengths) + 3 > config.max_position_embeddings:
        parser.error("Prompt plus three decode tokens exceeds the advertised context")
    capacity = args.capacity or max(128, math.ceil((max(sequence_lengths) + 3) / 128) * 128)
    if capacity < max(sequence_lengths) + 3 or capacity > config.max_position_embeddings or capacity % 128:
        parser.error("Physical capacity must cover all tokens and be a valid multiple of 128")
    fixtures = {length: make_fixture(reference, rope, length=length) for length in dict.fromkeys(sequence_lengths)}
    report = {
        "scope": "complete real-weight indexed-attention decoder layer 3",
        "layer_idx": 3,
        "weights": "real",
        "threshold": 0.995,
        "provenance": provenance,
        "cpu_reference_ready": True,
        "device_executed": not args.cpu_only,
        "sequence_lengths": sequence_lengths,
        "runtime_boundary": "device hidden states plus explicit request-owned paged/indexer/position state",
        "limitations": ["This probe is batch one; long-context, batched and performance gates require separate runs."],
        "cases": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.cpu_only:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(
            json.dumps(
                {"status": "HF_REFERENCE_READY", "sequence_lengths": sequence_lengths, "output": str(args.output)}
            )
        )
        return

    import ttnn
    from models.autoports.qwen_qwen3_8_flash_next.tt.functional_decoder import FunctionalDecoder

    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    try:
        model = FunctionalDecoder.from_state_dict(weights, hf_config=config, layer_idx=3, mesh_device=mesh)
        for length in sequence_lengths:
            case = {}
            report["cases"].append(case)
            device_case(model, mesh, fixtures[length], length=length, capacity=capacity, report=case)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(
                json.dumps({"seq_len": length, "passed": case["passed"], "prefill_pcc": case["prefill"]["pcc"]}),
                flush=True,
            )
        report["passed"] = all(case["passed"] for case in report["cases"])
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {str(exc).split('backtrace:')[0]}"
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        ttnn.close_mesh_device(mesh)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
