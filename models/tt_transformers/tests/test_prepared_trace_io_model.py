# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Full-weight, all-layer T3K validation of prepared prefill/decode I/O.

HF_MODEL=Qwen/Qwen3-32B MESH_DEVICE=T3K TT_METAL_TRACE_ALLOC_TRACKING=1
TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 pytest -s <this file>
"""

import os

import pytest
import torch
from ttnn.tools import trace_allocation_tracker

import ttnn
from models.common.sampling import SamplingParams, format_sampling_params
from models.tt_transformers.tt.common import Mode, PagedAttentionConfig, copy_host_to_device, create_tt_model
from models.tt_transformers.tt.generator import Generator
from models.tt_transformers.tt.model_config import DecodersPrecision


def _shards(tensor):
    return [ttnn.to_torch(shard).float() for shard in ttnn.get_device_tensors(tensor)]


def _assert_equal(actual, expected, active_rows=None):
    actual = _shards(actual)
    assert len(actual) == len(expected)
    for a, b in zip(actual, expected):
        if active_rows is not None:
            # Decode pads to 32 lanes. Only admitted requests have defined KV
            # positions; the generator discards the inactive lanes on readback.
            a, b = a[..., :active_rows, :], b[..., :active_rows, :]
        torch.testing.assert_close(a, b, atol=0, rtol=0)


@pytest.mark.skipif(os.getenv("HF_MODEL") != "Qwen/Qwen3-32B", reason="requires full Qwen3-32B weights")
@pytest.mark.skipif(not trace_allocation_tracker.TRACE_ALLOC_TRACKING, reason="requires trace allocation tracking")
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"trace_region_size": 96 * 1024 * 1024, "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}],
    indirect=True,
)
@pytest.mark.parametrize("decode_first", [False, True], ids=["prefill-first", "decode-first"])
@pytest.mark.timeout(3600)
@torch.no_grad()
def test_full_model_prepared_trace_io(mesh_device, device_params, decode_first, reset_seeds, ensure_gc):
    args, model, kv_cache, state_dict = create_tt_model(
        mesh_device,
        instruct=True,
        max_batch_size=1,
        max_seq_len=2048,
        paged_attention_config=PagedAttentionConfig(block_size=32, max_num_blocks=64),
        optimizations=lambda a: DecodersPrecision.performance(a.n_layers, a.model_name),
    )
    del state_dict
    assert args.n_layers == 64, "This regression must exercise all Qwen3-32B layers"
    generator = Generator([model], [args], mesh_device)
    page_table = torch.arange(64, dtype=torch.int32).reshape(1, -1)
    kv_cache = [kv_cache]
    model.switch_mode(Mode.DECODE)
    params = SamplingParams(temperature=0.0, top_k=1, top_p=1.0)
    model.sampling.reset_sampling_params(format_sampling_params(params, model.sampling.tt_sampling.max_batch_size))
    tokens = [torch.tensor([[123]], dtype=torch.int64)]
    positions = [torch.tensor([128], dtype=torch.int64)]
    decodes = {
        mode: generator._prepare_decode_trace_text(
            tokens, positions, page_table=[page_table], kv_cache=kv_cache, on_device_sampling=mode
        )
        for mode in (False, True)
    }
    prefills = {}
    expected_prefills = {}
    for length in (128, 1024):
        prompt = torch.arange(1, length + 1, dtype=torch.int64).reshape(1, -1)
        prepared = generator._prepare_trace_prefill(prompt, page_table=page_table, kv_cache=kv_cache[0], model_id=0)
        prefills[length] = (prompt, prepared)
        expected_prefills[length] = _shards(prepared.pop("compile_output").cpu())

    # Refresh the same KV prefix before obtaining eager decode references.
    prompt, prep = prefills[128]
    generator._prefill_trace_forward(prep, prep["device_inputs"])
    expected_decodes = {}
    for mode, prepared in decodes.items():
        host = model.prepare_decode_inputs_host(tokens[0], positions[0], page_table)
        copy_host_to_device(host, device_tensors=prepared["device_inputs"][0])
        result = generator._decode_trace_forward(prepared, 0)
        logits = result if mode else result[0]
        expected_decodes[mode] = _shards(logits.cpu())
        del result, logits
        copy_host_to_device(host, device_tensors=prepared["device_inputs"][0])

    expected_token = expected_decodes[False][0][0, 0, 0, : args.vocab_size].argmax().item()
    retained = ttnn.clone(prefills[128][1]["output"])
    ttnn.copy(prefills[128][1]["output"], retained)
    cache_entries = mesh_device.num_program_cache_entries()
    captures = {}
    order = [("prefill", 128), ("prefill", 1024), ("decode", False), ("decode", True)]
    if decode_first:
        order.reverse()
    try:
        for kind, key in order:
            if kind == "prefill":
                trace_id, output, *inputs = generator._record_trace_prefill(prefills[key][1])
            else:
                ids, outputs, *inputs = generator._record_decode_trace_text(decodes[key])
                trace_id, output = ids[0], outputs[0] if key else outputs[0][0]
            captures[kind, key] = (trace_id, output, inputs)

        queued = []
        addresses = {key: value[1].buffer_address() for key, value in captures.items()}
        for _ in range(2):
            for length in (1024, 128):
                trace_id, output, inputs = captures["prefill", length]
                prompt, _ = prefills[length]
                generator._prefill_forward_trace(trace_id, inputs, output, prompt, page_table=page_table, model_id=0)
                queued.append((output.cpu(blocking=False), expected_prefills[length], None))
                if length == 128:
                    ttnn.copy(output, retained)
            for mode in (True, False, True):
                trace_id, output, _ = captures["decode", mode]
                host = model.prepare_decode_inputs_host(tokens[0], positions[0], page_table)
                copy_host_to_device(host, device_tensors=decodes[mode]["device_inputs"][0])
                ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
                queued.append((output.cpu(blocking=False), expected_decodes[mode], 1))
                if mode:
                    sampled, _ = model.sampling.sample(
                        output, enable_trace=True, tt_out_tok=decodes[mode]["device_inputs"][0][0]
                    )
                    # Force a consumer before another graph writes shared model buffers.
                    assert int(_shards(sampled.cpu())[0].reshape(-1)[0]) == expected_token
        ttnn.synchronize_device(mesh_device)
        for snapshot, expected, active_rows in queued:
            _assert_equal(snapshot, expected, active_rows)
        _assert_equal(retained.cpu(), expected_prefills[128])
        # A later unrelated graph must not corrupt a borrowed output either.
        for (kind, key), (_, output, _) in captures.items():
            _assert_equal(
                output.cpu(),
                expected_prefills[key] if kind == "prefill" else expected_decodes[key],
                None if kind == "prefill" else 1,
            )
        assert mesh_device.num_program_cache_entries() == cache_entries
        assert {key: value[1].buffer_address() for key, value in captures.items()} == addresses
    finally:
        model.sampling.reset_trace()
        for trace_id, *_ in captures.values():
            ttnn.release_trace(mesh_device, trace_id)


@pytest.mark.skipif(os.getenv("HF_MODEL") != "Qwen/Qwen3-32B", reason="requires full Qwen3-32B weights")
@pytest.mark.skipif(not trace_allocation_tracker.TRACE_ALLOC_TRACKING, reason="requires trace allocation tracking")
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"trace_region_size": 128 * 1024 * 1024, "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}],
    indirect=True,
)
@pytest.mark.timeout(3600)
@torch.no_grad()
def test_full_model_public_warmup(mesh_device, device_params, reset_seeds, ensure_gc):
    args, model, cache, state_dict = create_tt_model(
        mesh_device,
        instruct=True,
        max_batch_size=2,
        max_seq_len=2048,
        paged_attention_config=PagedAttentionConfig(block_size=32, max_num_blocks=128),
        optimizations=lambda a: DecodersPrecision.performance(a.n_layers, a.model_name),
    )
    del state_dict
    assert args.n_layers == 64
    generator = Generator([model], [args], mesh_device)
    page_table = torch.arange(128, dtype=torch.int32).reshape(2, -1)
    cache = [cache]
    prompts = [torch.arange(1, n + 1).reshape(1, -1).repeat(2, 1) for n in (97, 777)]
    expected = [
        generator.prefill_forward_text(
            prompt, page_table=page_table, kv_cache=cache, warmup_prefill=False, enable_trace=False
        )
        for prompt in prompts
    ]
    greedy = SamplingParams(temperature=0.0, top_k=1, top_p=1.0)
    try:
        for repeat in range(2):
            for prompt, logits in zip(prompts, expected):
                traced_logits = generator.prefill_forward_text(
                    prompt, page_table=page_table, kv_cache=cache, enable_trace=True
                )
                torch.testing.assert_close(traced_logits, logits, atol=0, rtol=0)
                # Both modes were prepared by the first prefill call. Switching
                # later must capture only, without allocating persistent I/O.
                token = logits.argmax(-1)
                for params in (None, greedy):
                    result = generator.decode_forward(
                        token,
                        torch.full((2,), prompt.shape[1], dtype=torch.int64),
                        page_table=page_table,
                        kv_cache=cache,
                        enable_trace=True,
                        sampling_params=params,
                        reset_batch=True,
                    )
                    if params is None:
                        decode_logits = result[0] if isinstance(result, tuple) else result
                    else:
                        sampled = result[0] if isinstance(result, tuple) else result
                        torch.testing.assert_close(sampled.reshape(-1).long(), decode_logits.argmax(-1).reshape(-1))
                sampled, _ = generator.prefill_forward_text(
                    prompt, page_table=page_table, kv_cache=cache, enable_trace=True, sampling_params=greedy
                )
                torch.testing.assert_close(sampled.reshape(-1).long(), logits.argmax(-1).reshape(-1))
            if repeat == 0:
                entries = mesh_device.num_program_cache_entries()
        assert mesh_device.num_program_cache_entries() == entries
    finally:
        model.sampling.reset_trace()
        for table in (generator.trace_id_prefill, generator.trace_id_prefill_sampling):
            for trace_id in table.values():
                if trace_id is not None:
                    ttnn.release_trace(mesh_device, trace_id)
        for ids in generator.trace_ids_decode.values():
            if ids:
                for trace_id in ids.values():
                    ttnn.release_trace(mesh_device, trace_id)
