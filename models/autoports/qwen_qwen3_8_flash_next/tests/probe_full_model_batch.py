# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""All-layer fixed-slot batch capability check, separate from max context.

Default batch32/capacity128 uses the real generator and its complete preparation.
Four distinct mixed-length prompts recur across slots, so four solo controls
cover all rows without thirty-two redundant all-layer solo generations.

    timeout 7200 python -m models.autoports.qwen_qwen3_8_flash_next.tests.probe_full_model_batch \
      --batch-size 32 --capacity 128 \
      --output models/autoports/qwen_qwen3_8_flash_next/doc/full_model/batch32.json

Add --shared-kv to bind12 independent global QSA pools before preparation,
using the public allocator and distinct per-layer slot mappings. At B32/C128
the BF16 K/V pools occupy48MiB per rank; private recurrence/indexer state stays
per slot. This mode also checks mixed inactive rows and global table ownership.

Run serially, after restoring the accepted mesh descriptor. No profiler.
This proves the selected batch at the selected cache capacity, not batch32
simultaneously at the advertised maximum context or a serving throughput claim.
"""

import argparse
import json
import os
import time
import traceback
from pathlib import Path

import torch

import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tests.run_full_model_accuracy import memory
from models.autoports.qwen_qwen3_8_flash_next.tt.cache_budget import plan_shared_kv_cache
from models.autoports.qwen_qwen3_8_flash_next.tt.generator import build_generator


def shared_pool_layout(batch_size, capacity, attention_layers):
    """Full physical slot capacity, with disjoint global IDs in each QSA pool."""
    if not 2 <= batch_size <= 32 or capacity < 35 or attention_layers != 12:
        raise ValueError("Expected full48 geometry, batch2..32 and context at least35")
    from .shared_pool_layout import shared_pool_layout as make_layout

    return make_layout(batch_size, capacity, attention_layers)


def comparison(reference, actual):
    reference, actual = reference.float(), actual.float()
    assert reference.shape == actual.shape
    row_pcc = []
    for left, right in zip(reference, actual, strict=True):
        a, b = left - left.mean(), right - right.mean()
        denominator = float(a.norm() * b.norm())
        row_pcc.append(float(a @ b) / denominator if denominator else float(torch.equal(left, right)))
    a, b = reference.flatten(), actual.flatten()
    a, b = a - a.mean(), b - b.mean()
    denominator = float(a.norm() * b.norm())
    whole_pcc = float(a @ b) / denominator if denominator else float(torch.equal(reference, actual))
    ref_top1 = reference.argmax(-1)
    candidates = actual.topk(100, dim=-1).indices
    return {
        "finite": bool(torch.isfinite(reference).all() and torch.isfinite(actual).all()),
        "exact": bool(torch.equal(reference, actual)),
        "whole_pcc": whole_pcc,
        "per_row_pcc": row_pcc,
        "minimum_row_pcc": min(row_pcc),
        "top1_agreement": float((candidates[:, 0] == ref_top1).float().mean()),
        "top5_agreement": float((candidates[:, :5] == ref_top1[:, None]).any(-1).float().mean()),
        "top100_agreement": float((candidates == ref_top1[:, None]).any(-1).float().mean()),
        "max_absolute_difference": float((reference - actual).abs().max()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--capacity", type=int, default=128)
    parser.add_argument("--trace-region-size", type=int, default=0)
    parser.add_argument("--shared-kv", action="store_true", help="Bind all12 global QSA pools before preparation")
    args = parser.parse_args()
    if not 2 <= args.batch_size <= 32 or args.capacity < 35:
        parser.error("Use batch2..32 and capacity at least35")
    model_dir = Path(__file__).resolve().parents[1]
    result = {
        "passed": False,
        "scope": "all_48_layer_batch_capability",
        "batch_size": args.batch_size,
        "capacity": args.capacity,
        "regime": "serial",
        "not_a_serving_performance_measurement": True,
        "shared_kv": args.shared_kv,
        "thresholds": {
            "prefill_exact": True,
            "decode_whole_pcc": 0.995,
            "decode_minimum_row_pcc": 0.99,
            "decode_top1_agreement": 1.0,
            "decode_top5_min": 0.98,
            "decode_top100": 1.0,
        },
    }
    generator = mesh = None
    phase = "setup"
    started = time.monotonic()

    def mark(name, **extra):
        nonlocal phase
        phase = name
        print(json.dumps({"phase": name, **extra}), flush=True)

    try:
        assert os.environ.get("TT_MESH_GRAPH_DESC_PATH"), "Restore accepted mesh descriptor after lane-env"
        for name in ("TT_METAL_DEVICE_PROFILER", "TT_METAL_TRACE_PROFILER", "TT_METAL_STREAMING_PROFILER"):
            assert os.environ.get(name) in (None, "", "0"), "Do not profile the all-layer stack"
        torch.set_num_threads(8)
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
        mesh = ttnn.open_mesh_device(
            ttnn.MeshShape(1, 4), trace_region_size=args.trace_region_size, l1_small_size=32768
        )
        mark("BATCH_MODEL_LOAD", batch=args.batch_size, capacity=args.capacity)
        generator = build_generator(
            model_dir,
            mesh,
            max_seq_len=args.capacity,
            max_batch_size=args.batch_size,
            defer_kv_cache=args.shared_kv,
        )
        assert len(generator.model.layers) == 48
        result["memory_loaded"] = memory(mesh)
        pool = layer_tables = pool_addresses = None
        attention_indices = [
            index
            for index, state in zip(generator.model.layer_indices, generator.caches[0].layers, strict=True)
            if state.attention is not None
        ]
        if args.shared_kv:
            mark("BATCH_BIND_SHARED_KV", attention_layers=attention_indices)
            specs, layer_tables = shared_pool_layout(args.batch_size, args.capacity, len(attention_indices))
            assert not generator.prepared and not generator.traces and not generator.prefill_traces
            assert all(
                state.attention.k_cache is None and state.attention.v_cache is None
                for state in generator.caches[0].layers
                if state.attention is not None
            ), "Deferred constructor allocated private K/V"
            pool = generator.allocate_kv_cache_per_layer(specs)
            assert len(pool.pairs) == 12
            caches_and_scratch = [*generator.caches, generator.slot_mover.scratch]
            for cache in caches_and_scratch:
                assert cache.shared_kv and cache.kv_bound
                for index, state in enumerate(
                    [layer.attention for layer in cache.layers if layer.attention is not None]
                ):
                    pair = pool.pair_for_attention_layer(index)
                    assert state.k_cache is pair[0] and state.v_cache is pair[1]
                    assert all(
                        tuple(tensor.shape) == specs[index][0] and tensor.dtype == ttnn.bfloat16 for tensor in pair
                    )
                for state in cache.layers:
                    if getattr(state, "recurrent", None) is not None:
                        assert state.recurrent.dtype == ttnn.float32, "Shared setup changed recurrent-state precision"
            pool_addresses = [
                part.buffer_address()
                for pair in pool.pairs.values()
                for tensor in pair
                for part in ttnn.get_device_tensors(tensor)
            ]
            budget = plan_shared_kv_cache(args.capacity, args.batch_size)
            assert pool.bytes_per_rank == len(specs) * specs[0][0][0] * 32768
            result["shared_pool"] = {
                "attention_layer_indices": attention_indices,
                "per_layer_local_shape": list(specs[0][0]),
                "unique_pairs": len(pool.pairs),
                "pool_bytes_per_rank": pool.bytes_per_rank,
                "warmup_blocks_required_per_pool": 2 * args.batch_size,
                "page_tables_per_layer": [table.tolist() for table in layer_tables],
                "cache_and_scratch_bindings_verified": len(caches_and_scratch),
                "addresses_all_ranks": pool_addresses,
                "planning_inputs_unmeasured": budget.to_dict(),
                "planned_resident_with_pool_bytes_per_rank": budget.known_resident_before_pool_bytes
                + pool.bytes_per_rank,
                "headroom_after_unmeasured_reserve_bytes_per_rank": (
                    budget.allocatable_bytes
                    - budget.known_resident_before_pool_bytes
                    - pool.bytes_per_rank
                    - budget.runtime_reserve_bytes
                ),
            }
            result["bound_cache_spec"] = generator.cache_spec()
            result["memory_pool_bound"] = memory(mesh)

        def cache_inputs(submitted_slots, *, include_tables=True):
            if pool is None:
                return {"page_table": None, "kv_cache": [generator.caches[slot] for slot in submitted_slots]}
            if not include_tables:
                return {"page_table": None, "kv_cache": pool}
            tables = [table[list(submitted_slots)] for table in layer_tables]
            return {"page_table": tables[0], "page_tables_per_layer": tables, "kv_cache": pool}

        mark("BATCH_PREPARE")
        if args.shared_kv:
            generator.warmup_model_prefill(kv_cache=pool)
            generator.warmup_model_decode(kv_cache=pool)
            assert pool.frozen, "Prepared shared pool must reject rebinding"
        else:
            generator.prepare()
        result["memory_prepared"] = memory(mesh)
        result["setup_seconds"] = time.monotonic() - started
        seed = [785, 13458, 13, 785, 13, 13458, 42]
        prompts = [(seed * 5)[:length] for length in (3, 7, 31, 33)]
        solo_prefill, solo_decode = [], []
        for kind, prompt in enumerate(prompts):
            mark("BATCH_SOLO_CONTROL", prompt_kind=kind, length=len(prompt))
            logits = generator.prefill_forward(
                torch.tensor([prompt]),
                **cache_inputs((0,)),
                prompt_lens=[len(prompt)],
                slots=(0,),
            )
            solo_prefill.append(logits[0, -1].clone())
            decoded = generator.decode_forward(
                torch.tensor([42 + kind]),
                [len(prompt)],
                **cache_inputs((0,)),
                slots=(0,),
                sample_on_device=False,
            )
            solo_decode.append(decoded[0].clone())
        kinds = [slot % len(prompts) for slot in range(args.batch_size)]
        lengths = [len(prompts[kind]) for kind in kinds]
        tokens = torch.zeros(args.batch_size, max(lengths), dtype=torch.long)
        for slot, kind in enumerate(kinds):
            tokens[slot, : lengths[slot]] = torch.tensor(prompts[kind])
        slots = tuple(range(args.batch_size))

        def batch_prefill():
            return generator.prefill_forward(tokens, **cache_inputs(slots), prompt_lens=lengths, slots=slots)

        mark("BATCH_MIXED_PREFILL", lengths=lengths)
        prefill = batch_prefill()[:, -1]
        expected_prefill = torch.stack([solo_prefill[kind] for kind in kinds])
        result["mixed_prefill"] = comparison(expected_prefill, prefill)
        assert result["mixed_prefill"]["exact"], "Mixed batched prefill must equal serial same-row prefill exactly"
        mark("BATCH_DECODE_LOGITS")
        forced = torch.tensor([42 + kind for kind in kinds])
        decoded = generator.decode_forward(forced, lengths, **cache_inputs(slots), slots=slots, sample_on_device=False)
        expected_decode = torch.stack([solo_decode[kind] for kind in kinds])
        result["decode"] = comparison(expected_decode, decoded)
        stats = result["decode"]
        assert stats["finite"] and stats["whole_pcc"] >= 0.995 and stats["minimum_row_pcc"] >= 0.99, stats
        assert (
            stats["top1_agreement"] == 1 and stats["top5_agreement"] >= 0.98 and stats["top100_agreement"] == 1
        ), stats
        # Numerical differences from batched MoE are exposed above; they never
        # excuse a changed greedy token or a failed agreement threshold.
        mark("BATCH_DEVICE_TOKEN_FEEDBACK")
        batch_prefill()
        sampled = generator.decode_forward(forced, lengths, **cache_inputs(slots), slots=slots, sample_on_device=True)
        assert tuple(sampled.shape) == (args.batch_size,)
        assert torch.equal(sampled.long(), expected_decode.argmax(-1).long()), "Batched greedy tokens differ from solo"
        for rank, part in enumerate(ttnn.get_device_tensors(generator.decode_tokens)):
            resident = ttnn.to_torch(part).flatten()[: args.batch_size].long()
            assert torch.equal(
                resident, sampled.long()
            ), f"Rank {rank}: sampled tokens not in persistent feedback input"
        qsa = []
        for slot, cache in enumerate(generator.caches):
            assert cache.position == lengths[slot] + 1
            attention_index = 0
            for layer_index, state in zip(generator.model.layer_indices, cache.layers, strict=True):
                if state.attention is None:
                    continue
                for rank, part in enumerate(ttnn.get_device_tensors(state.attention.position)):
                    value = ttnn.to_torch(part).flatten().tolist()
                    assert value == [lengths[slot] + 1], f"Slot{slot} layer{layer_index} rank{rank} position mismatch"
                if pool is not None:
                    expected = layer_tables[attention_index][slot : slot + 1]
                    for rank, part in enumerate(ttnn.get_device_tensors(state.attention.page_table)):
                        assert torch.equal(
                            ttnn.to_torch(part).int(), expected
                        ), f"Slot{slot} layer{layer_index} rank{rank} global page mapping mismatch"
                attention_index += 1
                qsa.append({"slot": slot, "layer": layer_index, "position": lengths[slot] + 1, "verified_ranks": 4})
        before = dict(generator.counters)
        mark("BATCH_RESIDENT_CONTINUATION")
        continued = generator.decode_forward(
            torch.full((args.batch_size,), 1234),
            [97] * args.batch_size,
            **cache_inputs(slots, include_tables=False),
            slots=slots,
            sample_on_device=True,
            reload_inputs=False,
        )
        assert tuple(continued.shape) == (args.batch_size,)
        assert bool(((continued >= 0) & (continued < generator.model.config.vocab_size)).all())
        for key in ("token_refreshes", "position_refreshes", "page_table_refreshes"):
            assert generator.counters[key] == before[key], f"Resident continuation rebuilt {key}"
        for rank, part in enumerate(ttnn.get_device_tensors(generator.decode_tokens)):
            assert torch.equal(
                ttnn.to_torch(part).flatten()[: args.batch_size].long(), continued.long()
            ), f"Rank{rank} feedback mismatch"
        if pool is not None:
            mark("BATCH_SHARED_INACTIVE_ROWS")
            inactive_slots = tuple(range(1, args.batch_size, 2))
            active_slots = tuple(range(0, args.batch_size, 2))
            representative = inactive_slots[-1]

            def dormant_state_snapshot(slot):
                # Snapshot one complete all48-layer dormant request, including
                # its logically owned shared KV pages. Other requests' pool
                # pages may change and are deliberately outside this snapshot.
                cache = generator.caches[slot]
                snapshot = {}
                for layer_index, state in zip(generator.model.layer_indices, cache.layers, strict=True):
                    for name in ("conv", "recurrent", "ple_conv", "token_history"):
                        tensor = getattr(state, name, None)
                        if tensor is not None:
                            snapshot[f"{layer_index}.{name}"] = [
                                ttnn.to_torch(part).clone() for part in ttnn.get_device_tensors(tensor)
                            ]
                    if state.attention is not None:
                        for name in ("position", "page_table", "raw_history", "compressed_keys"):
                            snapshot[f"{layer_index}.{name}"] = [
                                ttnn.to_torch(part).clone()
                                for part in ttnn.get_device_tensors(getattr(state.attention, name))
                            ]
                        for name in ("k_cache", "v_cache"):
                            snapshots = []
                            for rank, part in enumerate(ttnn.get_device_tensors(getattr(state.attention, name))):
                                ids = snapshot[f"{layer_index}.page_table"][rank].flatten()[
                                    : (cache.position + 31) // 32
                                ]
                                snapshots.append(ttnn.to_torch(part)[ids.long()].clone())
                            snapshot[f"{layer_index}.{name}.owned_pages"] = snapshots
                return snapshot

            dormant_before = dormant_state_snapshot(representative)
            before_positions = [cache.position for cache in generator.caches]
            before_tokens = [ttnn.to_torch(part).long() for part in ttnn.get_device_tensors(generator.decode_tokens)]
            before_seeds = [ttnn.to_torch(part).long() for part in ttnn.get_device_tensors(generator.seed_state.state)]
            before = dict(generator.counters)
            no_rows = generator.decode_forward(
                torch.full((args.batch_size,), 1234),
                [-1] * args.batch_size,
                **cache_inputs(slots, include_tables=False),
                slots=slots,
                sample_on_device=True,
                reload_inputs=False,
            )
            assert no_rows.tolist() == [-1] * args.batch_size and generator.counters == before
            inactive_result = generator.decode_forward(
                torch.full((args.batch_size,), 1234),
                [97 if slot in active_slots else -1 for slot in slots],
                **cache_inputs(slots, include_tables=False),
                slots=slots,
                sample_on_device=True,
                reload_inputs=False,
            )
            assert inactive_result[list(inactive_slots)].tolist() == [-1] * len(inactive_slots)
            assert bool(
                (
                    (inactive_result[list(active_slots)] >= 0)
                    & (inactive_result[list(active_slots)] < generator.model.config.vocab_size)
                ).all()
            )
            for key in ("token_refreshes", "position_refreshes", "page_table_refreshes"):
                assert generator.counters[key] == before[key], f"Mixed inactivity rebuilt {key}"
            retained_rows = [slot for slot in range(32) if slot not in active_slots]
            for rank, part in enumerate(ttnn.get_device_tensors(generator.decode_tokens)):
                assert torch.equal(
                    ttnn.to_torch(part).long().flatten()[retained_rows], before_tokens[rank].flatten()[retained_rows]
                ), f"Rank{rank} dormant feedback changed"
            for rank, part in enumerate(ttnn.get_device_tensors(generator.seed_state.state)):
                assert torch.equal(ttnn.to_torch(part).long()[retained_rows], before_seeds[rank][retained_rows])
            for slot, cache in enumerate(generator.caches):
                expected = before_positions[slot] + int(slot in active_slots)
                assert cache.position == expected
                for state in cache.layers:
                    if state.attention is not None:
                        for part in ttnn.get_device_tensors(state.attention.position):
                            assert ttnn.to_torch(part).flatten().tolist() == [expected]
            dormant_after = dormant_state_snapshot(representative)
            assert dormant_before.keys() == dormant_after.keys()
            for name in dormant_before:
                for rank, (old, new) in enumerate(zip(dormant_before[name], dormant_after[name], strict=True)):
                    assert torch.equal(old, new), f"Dormant slot{representative} {name} rank{rank} changed"
            assert pool_addresses == [
                part.buffer_address()
                for pair in pool.pairs.values()
                for tensor in pair
                for part in ttnn.get_device_tensors(tensor)
            ], "Shared pool allocations changed after capture"
            result["shared_inactive_rows"] = {
                "passed": True,
                "all_inactive_call_no_counter_change": True,
                "active_slots": list(active_slots),
                "inactive_slots": list(inactive_slots),
                "all_inactive_feedback_and_seed_rows_exact": True,
                "all_slot_positions_verified": True,
                "complete_dormant_cache_verified_slot": representative,
                "dormant_cache_tensor_fields": len(dormant_before),
                "global_pool_addresses_stable": True,
            }
            del dormant_before, dormant_after, before_tokens, before_seeds
        result["sampled_tokens"] = sampled.tolist()
        result["continued_tokens"] = continued.tolist()
        result["qsa_positions_after_first_decode"] = qsa
        result["memory_final"] = memory(mesh)
        result["counters"] = dict(generator.counters)
        result["passed"] = True
    except BaseException as exc:
        result["failed_phase"] = phase
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
        raise
    finally:
        result["wall_seconds"] = time.monotonic() - started
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        if generator is not None:
            generator.close()
        if mesh is not None:
            ttnn.close_mesh_device(mesh)
        print(json.dumps({"passed": result["passed"], "output": str(args.output), "phase": phase}), flush=True)


if __name__ == "__main__":
    main()
