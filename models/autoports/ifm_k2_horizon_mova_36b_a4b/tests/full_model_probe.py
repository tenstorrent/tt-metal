# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Two real layer kinds and the real terminal path; no full-stack profiling."""

import argparse
import json
import time
from pathlib import Path

import torch

import ttnn
from models.common.sampling.generator import SamplingParams

from ..tt.generator import build_generator


def resident_contract(gen):
    batch = gen.max_batch_size
    lengths = [33 + 16 * (slot % 3) for slot in range(batch)]
    tokens = torch.zeros(batch, max(lengths), dtype=torch.int64)
    for slot, length in enumerate(lengths):
        tokens[slot, :length] = torch.arange(10, 10 + length)
    table = gen.page_table_host.clone()
    logits = gen.prefill_forward(tokens, page_table=table, kv_cache=gen.kv_cache, prompt_lens=lengths)
    expected = logits[:, -1].argmax(-1)
    first = gen.prefill_forward(
        tokens, page_table=table, kv_cache=gen.kv_cache, prompt_lens=lengths, sampling_mode="device"
    ).reshape(-1)
    assert torch.equal(first, expected), (first.tolist(), expected.tolist())
    positions = torch.tensor(lengths, dtype=torch.int32)
    if batch > 2:
        positions[1] = -1
    params = SamplingParams(0.0, 1, 1.0)
    output = gen.decode_forward(
        first[:, None],
        positions,
        page_table=table,
        kv_cache=gen.kv_cache,
        sampling_params=params,
        reload_sampling_params=True,
        reset_sampling_state=True,
        read_from_device=False,
    )
    first_host, events = gen.read_decode_output(output, async_read=True)
    for event in events:
        ttnn.event_synchronize(event)
    sampled = gen.process_decode_output_host(first_host).reshape(-1)
    resident = gen.read_tokens()
    assert torch.equal(sampled, resident)
    actual_positions = ttnn.to_torch(ttnn.get_device_tensors(gen.positions)[0]).reshape(-1)
    assert torch.equal(actual_positions, torch.where(positions >= 0, positions + 1, positions))
    baseline = gen.counters.copy()
    output = gen.decode_forward(
        torch.full_like(first[:, None], 999999),
        torch.full_like(positions, -999),
        page_table=table,
        kv_cache=gen.kv_cache,
        sampling_mode="device",
        reload_inputs=False,
        read_from_device=False,
    )
    second_host, events2 = gen.read_decode_output(output, async_read=True)
    for name in ("token_refreshes", "position_refreshes", "rope_refreshes", "page_table_refreshes", "active_refreshes"):
        assert gen.counters[name] == baseline[name], name
    changed = table.clone()
    changed[:, [-2, -1]] = changed[:, [-1, -2]]
    output = gen.decode_forward(
        torch.full_like(first[:, None], 999999),
        torch.full_like(positions, -999),
        page_table=changed,
        kv_cache=gen.kv_cache,
        sampling_mode="device",
        reload_inputs=False,
        reload_page_table=True,
        read_from_device=False,
    )
    third_host, events3 = gen.read_decode_output(output, async_read=True)
    for event in events2 + events3:
        ttnn.event_synchronize(event)
    second = gen.process_decode_output_host(second_host).reshape(-1)
    third = gen.process_decode_output_host(third_host).reshape(-1)
    assert gen.counters["page_table_refreshes"] == baseline["page_table_refreshes"] + 1
    for name in ("token_refreshes", "position_refreshes", "rope_refreshes", "active_refreshes"):
        assert gen.counters[name] == baseline[name], name
    expected_positions = torch.where(positions >= 0, positions + 3, positions)
    actual_positions = ttnn.to_torch(ttnn.get_device_tensors(gen.positions)[0]).reshape(-1)
    assert torch.equal(actual_positions, expected_positions)
    if batch > 2:
        assert sampled[1] == second[1] == third[1] == first[1]
    continuation_logits = gen.decode_forward(
        third[:, None], actual_positions, page_table=table, kv_cache=gen.kv_cache, sampling_mode="host"
    )
    # A new prefill in the final slot must preserve every other token/seed slot.
    old_tokens = gen.read_tokens().clone()
    old_seeds = ttnn.to_torch(ttnn.get_device_tensors(gen.sampler.tt_sampling.seeds_tt_tensor)[0]).clone()
    new_first = gen.prefill_forward(
        tokens[-1:],
        page_table=table[-1:],
        kv_cache=gen.kv_cache,
        prompt_lens=lengths[-1:],
        empty_slots=[batch - 1],
        sampling_mode="device",
    )
    new_tokens = gen.read_tokens()
    new_seeds = ttnn.to_torch(ttnn.get_device_tensors(gen.sampler.tt_sampling.seeds_tt_tensor)[0])
    assert torch.equal(new_tokens[: batch - 1], old_tokens[: batch - 1])
    assert torch.equal(new_seeds[: batch - 1], old_seeds[: batch - 1])
    resumed_tokens = third[:, None].clone()
    resumed_tokens[-1] = new_first[0]
    resumed_positions = actual_positions.clone()
    resumed_positions[-1] = lengths[-1]
    resumed = gen.decode_forward(
        resumed_tokens,
        resumed_positions,
        page_table=table,
        kv_cache=gen.kv_cache,
        sampling_mode="device",
        sampling_params=params,
        reload_sampling_params=True,
        reset_sampling_state=True,
    ).reshape(-1)
    for slot in range(batch - 1):
        if resumed_positions[slot] >= 0:
            assert resumed[slot] == continuation_logits[slot].argmax(), slot
    return dict(
        batch=batch,
        prompt_lengths=lengths,
        prefill_tokens=first.tolist(),
        decode_tokens=[sampled.tolist(), second.tolist(), third.tolist()],
        positions=actual_positions.tolist(),
        stale_inputs_ignored=True,
        unchanged_page_refreshes=0,
        changed_page_refreshes=1,
        prefill_preserves_other_slots=True,
        resumed_decode_preserves_other_requests=True,
        counters=dict(gen.counters),
    )


def continuation_contract(gen):
    batch = gen.max_batch_size
    start, end = gen.page_size + 1, gen.page_size + 33
    prompt = torch.arange(10, 10 + end).repeat(batch, 1)
    table = gen.page_table_host.clone()
    trace_before = (gen.model_trace, {k: v[0] for k, v in gen.sampling_traces.items()})
    full = gen.prefill_forward(prompt, page_table=table, kv_cache=gen.kv_cache, prompt_lens=[end] * batch)
    repeated = gen.prefill_forward(prompt, page_table=table, kv_cache=gen.kv_cache, prompt_lens=[end] * batch)
    assert torch.equal(full, repeated)
    assert all(torch.equal(full[0], full[i]) for i in range(batch))
    gen.prefill_forward(prompt, page_table=table, kv_cache=gen.kv_cache, prompt_lens=[start] * batch)
    chunked = gen.prefill_forward(
        prompt, page_table=table, kv_cache=gen.kv_cache, start_pos=[start] * batch, prompt_lens=[end] * batch
    )
    pcc = float(torch.corrcoef(torch.stack([full.flatten(), chunked.flatten()]))[0, 1])
    assert pcc >= 0.995, pcc
    assert torch.equal(full.argmax(-1), chunked.argmax(-1))
    first = chunked[:, -1].argmax(-1)[:, None]
    positions = torch.full((batch,), end, dtype=torch.int32)
    host = gen.decode_forward(first, positions, page_table=table, kv_cache=gen.kv_cache, sampling_mode="host")
    device = gen.decode_forward(
        first,
        positions,
        page_table=table,
        kv_cache=gen.kv_cache,
        sampling_mode="device",
        sampling_params=SamplingParams(0.0, 1, 1.0),
        reload_sampling_params=True,
        reset_sampling_state=True,
    )
    assert torch.equal(host.argmax(-1), device.reshape(-1))
    # Relocate a page containing the live prefix in the last slot. Host cache
    # reads/writes here construct an equivalent test fixture, never a generator
    # path. Only that row's page mapping changes on the next resident replay.
    old_page, new_page = int(table[-1, 0]), int(table[-1, -1])
    for pair in gen.kv_cache:
        for cache in pair:
            values = ttnn.to_torch(cache, mesh_composer=ttnn.ConcatMeshToTensor(gen.mesh_device, dim=1))
            values[[old_page, new_page]] = values[[new_page, old_page]].clone()
            staged = gen.model.put(values, dtype=gen.model.kv_cache_dtype, shard=1)
            ttnn.copy(staged, cache)
            del staged, values
    moved_table = table.clone()
    moved_table[-1, [0, -1]] = moved_table[-1, [-1, 0]]
    before = gen.counters.copy()
    moved_tokens = gen.decode_forward(
        torch.full_like(first, -999),
        torch.full_like(positions, -999),
        page_table=moved_table,
        kv_cache=gen.kv_cache,
        sampling_mode="device",
        reload_inputs=False,
        reload_page_table=True,
    )
    assert torch.all(moved_tokens == moved_tokens[0]), moved_tokens
    assert gen.counters["page_table_refreshes"] == before["page_table_refreshes"] + 1
    for key in ("token_refreshes", "position_refreshes", "rope_refreshes", "active_refreshes"):
        assert gen.counters[key] == before[key]
    table = moved_table
    remap = list(reversed(range(batch)))
    gen.decode_forward(
        first, positions, page_table=table.flip(0), kv_cache=gen.kv_cache, sampling_mode="host", slot_remap=remap
    )
    # A dormant sampler was invalidated by the host-mode remap. The plugin's
    # host->device transition commands rebuild authoritative state before use.
    assert gen.sampler._slot_state_requires_authoritative_reload
    gen.decode_forward(
        first,
        positions,
        page_table=table.flip(0),
        kv_cache=gen.kv_cache,
        sampling_mode="device",
        sampling_params=SamplingParams(0.0, 1, 1.0),
        reload_sampling_params=True,
        reset_sampling_state=True,
    )
    assert not gen.sampler._slot_state_requires_authoritative_reload
    inactive = positions.clone()
    inactive[-1] = -1
    sentinels = first.clone()
    sentinels[-1] = -1
    gen.decode_forward(sentinels, inactive, page_table=table.flip(0), kv_cache=gen.kv_cache, sampling_mode="host")
    assert int(gen.read_tokens()[-1]) == 0
    actual = ttnn.to_torch(ttnn.get_device_tensors(gen.positions)[0]).reshape(-1)
    assert actual[-1] == -1
    assert trace_before == (gen.model_trace, {k: v[0] for k, v in gen.sampling_traces.items()})
    return dict(
        batch=batch,
        repeat_logits_bitwise=True,
        batch_slot_logits_bitwise=True,
        unaligned_continuation=[start, end],
        continuation_pcc=pcc,
        continuation_top1_equal=True,
        host_device_greedy_equal=True,
        live_prefix_page_relocation_equal=True,
        live_prefix_page_refresh_only=True,
        inactive_sentinel_sanitized=True,
        dormant_sampler_remap_transition=True,
        trace_ids_unchanged=True,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--page-size", type=int, default=32)
    ap.add_argument("--output", default="models/autoports/ifm_k2_horizon_mova_36b_a4b/doc/full_model/probe.json")
    ap.add_argument("--eager", action="store_true")
    ap.add_argument("--contract", action="store_true")
    ap.add_argument("--extended", action="store_true")
    ap.add_argument("--external-cache", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(8)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=0)
    began = time.perf_counter()
    try:
        gen = build_generator(
            None,
            mesh,
            override_num_layers=args.layers,
            max_seq_len=1024,
            max_batch_size=args.batch,
            prepare_lengths=(32, 33, 128),
            allocate_cache=not args.external_cache,
            page_size=args.page_size,
        )
        if args.external_cache:
            assert gen.kv_cache is None
            blocks = (524288 + args.batch * args.page_size) // args.page_size
            gen.allocate_kv_cache((blocks, 2, args.page_size, 128))
            pages = 1024 // args.page_size
            gen.refresh_page_table(torch.arange(args.batch * pages, dtype=torch.int32).reshape(args.batch, pages))
        print("MODEL_READY", time.perf_counter() - began, flush=True)
        if args.extended:
            # Compile the test-only cache relocation copy before any trace.
            scratch = ttnn.clone(gen.kv_cache[0][0])
            ttnn.copy(scratch, gen.kv_cache[0][0])
            del scratch
        prompt = list(range(10, 43))
        if not args.eager:
            gen.capture()
            print("CAPTURE_READY", time.perf_counter() - began, flush=True)
        if args.contract:
            result = resident_contract(gen)
            if args.extended:
                result["continuation"] = continuation_contract(gen)
        else:
            out = gen.generate(prompt, 4, enable_trace=not args.eager)
            print("TOKENS", out, flush=True)
            result = dict(tokens=out, counters=dict(gen.counters))
        result["prefill_trace_ids"] = {str(k): str(v) for k, v in gen.prefill_traces.items()}
        result["final_counters"] = dict(gen.counters)
        assert gen.counters["prefill_captures"] == len(gen.prefill_traces)
        result["elapsed_s"] = time.perf_counter() - began
        result["page_size"] = args.page_size
        result["external_cache"] = args.external_cache
        result["precision_config_id"] = gen.model.precision["config_id"]
        result["cache_dtypes"] = sorted({str(t.dtype) for pair in gen.kv_cache for t in pair})
        result["physical_cache_tokens"] = gen.num_cache_blocks * args.page_size
        result["advertised_aggregate_tokens"] = gen.get_max_tokens_all_users()
        Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
        gen.close()
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
