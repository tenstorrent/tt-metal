# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Clean full48/B32 shared-pool resident decode regression row.

After sourcing doc/optimized_full_model/environment.sh, run serially::

    TT_METAL_TRACE_ALLOC_TRACKING=0 timeout 7200 python_env/bin/python -m \
      models.autoports.qwen_qwen3_8_flash_next.tests.benchmark_full_model_batch_resident \
      --output models/autoports/qwen_qwen3_8_flash_next/doc/optimized_full_model/batch32_resident_v1.json

Four natural-text prefix lengths125/126/127/128 repeat over32 fixed slots.
Each request generates one initial prefill sample plus127 timed decode samples.
Two full-active public warmup steps precede actual prefill, which resets request
contents in place. Assert that this preserves captured buffers and the dispatch
plan. No reset, warmup token or discarded generation enters the measured sequence.
This is a performance/regression probe; full HF B32 accuracy is a separate gate.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import torch
from ttnn.tools import trace_allocation_tracker

import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tests.run_full_model_acceptance import (
    counter_delta,
    natural_prompt,
    trace_ids,
)
from models.autoports.qwen_qwen3_8_flash_next.tests.shared_pool_layout import shared_pool_layout, validate_pool_aliases
from models.autoports.qwen_qwen3_8_flash_next.tt.generator import build_generator


def persistent_identity(generator):
    """Metadata-only identity check; no host tensor conversion or device read."""
    tensors = {
        name: getattr(generator, name)
        for name in ("decode_tokens", "sample_logits", "saved_tokens", "seed_skip", "word_suffix")
    }
    owner = generator.factored_decode
    for name in ("residual", "mixed", "blocks", "injection"):
        tensors.update({f"factored.{name}.{i}": value for i, value in enumerate(getattr(owner, name))})
    tensors.update({f"factored.packed.{k}": value for k, value in owner.packed.items()})
    tensors["factored.slot_indices"] = owner.slot_indices
    tensors["seed.state"], tensors["seed.active"] = generator.seed_state.state, generator.seed_state.active
    for slot, cache in enumerate(generator.caches):
        for layer, state in enumerate(cache.layers):
            for name in ("conv", "recurrent", "ple_conv", "token_history"):
                value = getattr(state, name, None)
                if value is not None:
                    tensors[f"cache.{slot}.{layer}.{name}"] = value
            if state.attention is not None:
                for name in ("position", "page_table", "k_cache", "v_cache", "raw_history", "compressed_keys"):
                    tensors[f"cache.{slot}.{layer}.{name}"] = getattr(state.attention, name)
    for i, value in enumerate(generator.model.workspace.rope or ()):
        tensors[f"rope.{i}"] = value
    return tuple(
        (name, id(value), tuple(part.buffer_address() for part in ttnn.get_device_tensors(value)))
        for name, value in sorted(tensors.items())
    )


def check_positions_and_tables(generator, expected, tables):
    rows = []
    for slot, cache in enumerate(generator.caches):
        assert cache.position == expected[slot]
        attention_index = 0
        for layer, state in zip(generator.model.layer_indices, cache.layers, strict=True):
            if state.attention is None:
                continue
            positions = ttnn.get_device_tensors(state.attention.position)
            pages = ttnn.get_device_tensors(state.attention.page_table)
            assert len(positions) == len(pages) == 4
            for rank, (position, page) in enumerate(zip(positions, pages, strict=True)):
                actual = ttnn.to_torch(position).flatten().tolist()
                assert actual == [expected[slot]], (slot, layer, rank, actual, expected[slot])
                assert torch.equal(ttnn.to_torch(page).int(), tables[attention_index][slot : slot + 1])
            rows.append({"slot": slot, "layer": layer, "position": expected[slot], "verified_ranks": 4})
            attention_index += 1
        assert attention_index == 12
    return rows


def resident_step(generator, placeholder, slots, pool):
    return generator.decode_forward(
        placeholder,
        [cache.position for cache in generator.caches],
        page_table=None,
        kv_cache=pool,
        slots=slots,
        sample_on_device=True,
        reload_inputs=False,
        reload_page_table=False,
        read_from_device=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--capacity", type=int, default=256)
    args = parser.parse_args()
    if args.output.exists() or args.capacity < 256:
        parser.error("Use a fresh output path and capacity at least256")
    model_dir = Path(__file__).resolve().parents[1]
    sources = sorted(
        p
        for p in (model_dir / "tt").rglob("*")
        if p.is_file() and p.suffix in (".py", ".cpp", ".h", ".hpp", ".cc", ".cxx")
    )
    sources += [
        Path(__file__),
        Path(__file__).with_name("shared_pool_layout.py"),
        Path(__file__).with_name("run_full_model_acceptance.py"),
    ]
    sources += sorted(Path("models/common/sampling").rglob("*.py"))
    report = {
        "passed": False,
        "scope": "clean_full48_batch32_shared_pool_performance_regression",
        "stage_acceptance_claim": False,
        "hf_correctness_gate": "Separate full48 B32 HF gate; this probe checks contract/state/token ranges only",
        "execution_geometry": {
            "decode_attention_and_lm_head": "Q1 per active fixed slot in the factored dispatch plan; not Q32 attention/head decode",
            "decode_moe": "Packed32 active rows per layer",
            "prefill_physical_rows": [1, 2, 4, 8, 16, 32],
            "prefill_rows_scope": "Prepared dyadic signatures; logical nonaligned prompts are decomposed without padding state advance",
            "sampler": "Physical32 rows, all32 fixed slots active in the timed window",
            "inactive_rows_scope": "Not exercised in this throughput window; separate B32 HF/control gate covers shrinking, migration and inactive feedback",
        },
        "argv": sys.argv,
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "capacity": args.capacity,
        "mesh": [1, 4],
        "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        "environment": {k: v for k, v in os.environ.items() if k.startswith(("TT_", "QWEN_"))},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    generator = mesh = None
    started = time.monotonic()

    def mark(phase):
        report["phase"] = phase
        print(json.dumps({"phase": phase}), flush=True)

    try:
        assert not trace_allocation_tracker.TRACE_ALLOC_TRACKING, "This is a clean timing harness"
        assert os.environ.get("TT_MESH_GRAPH_DESC_PATH")
        assert "TT_METAL_WATCHER" not in os.environ, "Unset Watcher entirely; even TT_METAL_WATCHER=0 enables it"
        for key in (
            "TT_METAL_DEVICE_PROFILER",
            "TT_METAL_TRACE_PROFILER",
            "TT_METAL_STREAMING_PROFILER",
            "TT_METAL_PROFILE_PERF_COUNTERS",
        ):
            assert os.environ.get(key) in (None, "", "0"), f"Disable {key} for clean full48 performance"
        torch.set_num_threads(8)
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=0, l1_small_size=32768)
        mark("B32_MODEL_LOAD")
        generator = build_generator(model_dir, mesh, max_seq_len=args.capacity, max_batch_size=32, defer_kv_cache=True)
        assert generator.model.layer_indices == tuple(range(48)) and generator.factored_decode is not None
        assert not generator.sampler.tt_sampling.force_argmax_sampling
        specs, tables = shared_pool_layout(32, args.capacity, 12)
        pool = generator.allocate_kv_cache_per_layer(specs)
        validate_pool_aliases([*generator.caches, generator.slot_mover.scratch], pool, specs)
        report["shared_pool"] = {
            "bytes_per_rank": pool.bytes_per_rank,
            "per_layer_shape": list(specs[0][0]),
            "unique_pairs": len(pool.pairs),
            "tables_per_layer": [table.tolist() for table in tables],
        }
        mark("B32_PREPARE")
        generator.warmup_model_prefill(kv_cache=pool)
        generator.warmup_model_decode(kv_cache=pool)
        assert pool.frozen
        slots = tuple(range(32))
        placeholder = torch.zeros(32, 1, dtype=torch.int64)
        prompts = {length: natural_prompt(generator, model_dir, length) for length in (125, 126, 127, 128)}
        lengths = [125 + slot % 4 for slot in slots]
        prompt_tensor = torch.zeros(32, 128, dtype=torch.int64)
        for slot, length in enumerate(lengths):
            assert len(prompts[length]) == length
            prompt_tensor[slot, :length] = torch.tensor(prompts[length])
        report["prompts"] = {
            "helper": "run_full_model_acceptance.natural_prompt",
            "kind": "Readiness natural-text prefixes; four lengths repeat across fixed slots; not a full chat-quality evaluation",
            "tokens_by_length": prompts,
            "lengths_by_slot": lengths,
        }

        mark("B32_FULL_ACTIVE_WARMUP")
        # Build and execute the full32 dispatch plan before the actual request.
        # Public prefill below resets contents in place, not tensor bindings.
        generator.decode_forward(
            placeholder,
            [0] * 32,
            page_table=tables[0],
            page_tables_per_layer=tables,
            kv_cache=pool,
            slots=slots,
            sample_on_device=True,
            read_from_device=False,
        )
        resident_step(generator, placeholder, slots, pool)
        ttnn.synchronize_device(mesh)
        owner = generator.factored_decode
        assert owner._plan_slots == slots and owner.active_slots == slots
        plan = owner._plan
        assert len(plan) == 1 + 48 * (32 + 1) + 32
        identity = persistent_identity(generator)
        model_trace_ids, sample_trace_ids = dict(generator.traces), dict(generator.sampling_traces)

        mark("B32_ACTUAL_MIXED_PREFILL")
        generator.prefill_forward(
            prompt_tensor,
            page_table=tables[0],
            page_tables_per_layer=tables,
            kv_cache=pool,
            prompt_lens=lengths,
            slots=slots,
            return_device=True,
        )
        generator._sample(slots)
        ttnn.synchronize_device(mesh)
        assert owner._plan is plan and owner._plan_slots == slots
        assert persistent_identity(generator) == identity
        assert generator.traces == model_trace_ids and generator.sampling_traces == sample_trace_ids
        initial_tokens = [
            ttnn.to_torch(part).flatten().long() for part in ttnn.get_device_tensors(generator.decode_tokens)
        ]
        assert len(initial_tokens) == 4 and all(torch.equal(initial_tokens[0], x) for x in initial_tokens[1:])
        assert all(cache.position == length for cache, length in zip(generator.caches, lengths, strict=True))
        report["positions_before"] = check_positions_and_tables(generator, lengths, tables)
        report["setup_seconds"] = time.monotonic() - started
        ids = trace_ids(generator)
        before = dict(generator.counters)
        before_segments, before_uploads = owner.segment_replays, owner.metadata_uploads
        programs_before = mesh.num_program_cache_entries()
        assert generator.seed_state._active_slots == slots
        mark("B32_RESIDENT_127_START")

        def forbidden(*args, **kwargs):
            raise AssertionError("Resident timed loop attempted a host boundary, upload or trace lifecycle operation")

        # Negative guards have no per-replay call overhead: these entry points
        # must never execute. The explicit benchmark boundaries retain the real
        # synchronize function. This also catches seed uploads outside counters.
        synchronize = ttnn.synchronize_device
        with ExitStack() as guards:
            for name in (
                "to_torch",
                "from_torch",
                "copy_host_to_device_tensor",
                "synchronize_device",
                "begin_trace_capture",
                "end_trace_capture",
                "release_trace",
            ):
                guards.enter_context(patch.object(ttnn, name, side_effect=forbidden))
            for obj, name in (
                (generator, "_refresh"),
                (generator, "_read_tokens"),
                (generator, "_read_logits"),
                (generator.seed_state, "_upload"),
            ):
                guards.enter_context(patch.object(obj, name, side_effect=forbidden))
            synchronize(mesh)
            timer = time.perf_counter()
            for _ in range(127):
                submission = resident_step(generator, placeholder, slots, pool)
            synchronize(mesh)
            elapsed = time.perf_counter() - timer
        delta = counter_delta(generator.counters, before)
        assert delta["model_replays"] == delta["sampling_replays"] == 127
        assert all(
            value == 0 for key, value in delta.items() if key not in ("model_replays", "sampling_replays")
        ), delta
        assert owner.segment_replays - before_segments == 127 * len(plan)
        assert owner.metadata_uploads == before_uploads and owner._plan is plan and owner._plan_slots == slots
        assert owner.active_slots == generator.seed_state._active_slots == slots
        assert trace_ids(generator) == ids and persistent_identity(generator) == identity
        programs_after = mesh.num_program_cache_entries()
        assert programs_after == programs_before, "Resident decode introduced an unprepared program signature"
        assert submission.tensor is generator.decode_tokens and submission.is_tokens
        assert submission.slots == submission.active_rows == slots
        report["performance"] = {
            "decode_steps": 127,
            "batch": 32,
            "generated_tokens_per_request": 128,
            "elapsed_seconds": elapsed,
            "ms_per_step": elapsed * 1000 / 127,
            "tokens_per_second_per_user": 127 / elapsed,
            "aggregate_tokens_per_second": 32 * 127 / elapsed,
            "initial_sample_in_timing": False,
            "timed_tokens": 32 * 127,
            "boundary": "Public decode_forward -> factored model traces -> split sampling; resident feedback; final synchronization included",
            "public_kwargs": {
                "page_table": None,
                "kv_cache": "fixed shared pool",
                "slots": list(slots),
                "sample_on_device": True,
                "reload_inputs": False,
                "reload_page_table": False,
                "read_from_device": False,
            },
            "positions": {
                "first_input_by_slot": lengths,
                "last_input_by_slot": [n + 126 for n in lengths],
                "final_state_by_slot": [n + 127 for n in lengths],
            },
            "counter_delta": delta,
            "segment_replays": owner.segment_replays - before_segments,
            "segments_per_model_replay": len(plan),
            "metadata_uploads": 0,
            "program_cache_entries_before": programs_before,
            "program_cache_entries_after": programs_after,
            "host_boundary_and_seed_upload_negative_guards": "passed; restored after timing, no per-replay instrumentation",
            "trace_and_dispatch_plan_identity_unchanged": True,
            "persistent_binding_count": len(identity),
            "persistent_identity_sha256": hashlib.sha256(repr(identity).encode()).hexdigest(),
            "window_boundary_syncs": 2,
            "timed_boundary_syncs": 1,
            "counter_note": "Generator counters exclude benchmark boundary synchronizations and final direct validation reads",
            "warmup_note": "Two full-active steps before public prefill; no global reset. Actual prefill resets contents in place; identity and full-active plan remain unchanged. Timed requests start at125..128, not warmup-advanced positions.",
        }
        mark("B32_FINAL_STATE_VALIDATION")
        final_tokens = [
            ttnn.to_torch(part).flatten().long() for part in ttnn.get_device_tensors(generator.decode_tokens)
        ]
        assert len(final_tokens) == 4 and all(torch.equal(final_tokens[0], x) for x in final_tokens[1:])
        for tokens in (*initial_tokens, *final_tokens):
            assert tokens.numel() == 32 and bool(((tokens >= 0) & (tokens < generator.model.config.vocab_size)).all())
        report["initial_tokens"] = initial_tokens[0].tolist()
        report["final_tokens"] = final_tokens[0].tolist()
        report["positions_after"] = check_positions_and_tables(generator, [n + 127 for n in lengths], tables)
        validate_pool_aliases([*generator.caches, generator.slot_mover.scratch], pool, specs)
        report["trace_ids"] = ids
        report["passed"] = True
        mark("B32_RESIDENT_PASSED")
    except BaseException:
        report["error"] = traceback.format_exc()
        raise
    finally:
        report["wall_seconds"] = time.monotonic() - started
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        if generator is not None:
            generator.close()
        if mesh is not None:
            ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
