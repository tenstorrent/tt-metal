# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Public non-aligned long prefill after trace capture; externally serialize/timeout."""

import argparse
import json
import os
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--all-layers", action="store_true")
    parser.add_argument("--context", type=int, default=262144)
    parser.add_argument("--length", type=int)
    parser.add_argument("--also-context", action="store_true")
    parser.add_argument("--build-kwargs-file", type=Path)
    args = parser.parse_args()
    length = args.context - 1 if args.length is None else args.length
    if args.context < 64 or not 0 < length <= args.context:
        parser.error("Require context >=64 and 0 < length <= context")
    os.environ.setdefault("TT_METAL_TRACE_ALLOC_TRACKING", "1")
    import torch
    from ttnn.tools import trace_allocation_tracker

    import ttnn
    from models.autoports.ornith_ai_ornith_1_5_9b.tt.generator import OrnithGenerator
    from models.autoports.ornith_ai_ornith_1_5_9b.tt.model import OrnithModel

    if not trace_allocation_tracker.TRACE_ALLOC_TRACKING:
        raise ValueError("Capacity evidence requires TT_METAL_TRACE_ALLOC_TRACKING=1")
    torch.set_num_threads(8)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=256 * 1024 * 1024)
    gen = None
    result = {
        "status": "running",
        "context": args.context,
        "logical_prefill_length": length,
        "all_layers": args.all_layers,
        "trace_allocation_tracking": True,
        "memory": {},
        "scope": "Public last-logit prefill after live trace creation; finite logits and capacity, not HF accuracy",
    }
    try:
        result["phase"] = "load_weights"
        kwargs = json.loads(args.build_kwargs_file.read_text()) if args.build_kwargs_file else {}
        model = OrnithModel(mesh, layer_indices=None if args.all_layers else [0, 3], **kwargs)
        result["runtime_policy"] = model.precision_summary()
        result["layers"] = model.layer_indices
        result["memory"]["weights_loaded"] = model.memory_view()
        if args.all_layers and not kwargs:
            from models.autoports.ornith_ai_ornith_1_5_9b.tt.cache_contract import plan_cache_specs

            blocks = (args.context + 31) // 32
            specs = [((blocks, 1, 32, 256), model.precision_policy["kv_cache_dtype"], index) for index in range(8)]
            plan = plan_cache_specs(specs, batch=1, context=args.context)
            cache = model.allocate_cache(**plan.model_kwargs())
            gen = OrnithGenerator(
                model,
                max_batch_size=1,
                max_seq_len=args.context,
                kv_cache=cache,
                page_table=torch.arange(blocks, dtype=torch.int32).reshape(1, blocks),
            )
            result["allocation_triples"] = specs
        else:
            gen = OrnithGenerator(model, max_batch_size=1, max_seq_len=args.context)
        result["memory"]["cache_allocated"] = model.memory_view()
        result["phase"] = "prepare_and_capture"
        warm_tokens = gen.generate(list(range(100, 133)), 4)
        assert len(warm_tokens) == 4
        ttnn.synchronize_device(mesh)
        state = gen.states[(id(gen.cache), (0,))]
        assert state.model_traces and state.sample_traces
        result["memory"]["prepared_live_traces"] = model.memory_view()
        programs = mesh.num_program_cache_entries()
        counters = dict(gen.counters)
        mesh.set_program_cache_misses_allowed(False)
        result["phase"] = "long_public_prefill"
        # Synthetic valid IDs exercise length/layout/capacity, without a quality claim.
        prompt = (torch.arange(length, dtype=torch.int64) % 1000 + 100).reshape(1, -1)
        begin = time.perf_counter()
        logits = gen.prefill_forward(
            prompt,
            page_table=gen.page_table,
            kv_cache=gen.cache,
            prompt_lens=[length],
            start_pos=[0],
            return_all_logits=False,
        )
        ttnn.synchronize_device(mesh)
        elapsed = time.perf_counter() - begin
        assert tuple(logits.shape) == (1, 1, model.vocab_size), tuple(logits.shape)
        assert torch.isfinite(logits).all(), "Long prefill returned non-finite final logits"
        assert mesh.num_program_cache_entries() == programs, "Long request created unprepared programs"
        assert (
            gen.counters == counters
        ), "Prefill unexpectedly rebuilt/replayed decode or refreshed its persistent inputs"
        result.update(
            status="pass",
            phase="complete",
            last_logits_shape=list(logits.shape),
            logits_finite=True,
            elapsed_seconds=elapsed,
            input_tokens_per_second=length / elapsed,
            program_count_before=programs,
            program_count_after=mesh.num_program_cache_entries(),
            counters_before=counters,
            counters_after=dict(gen.counters),
        )
        result["memory"]["after_long_prefill"] = model.memory_view()
        result["live_model_traces"] = len(state.model_traces)
        result["live_sampling_traces"] = len(state.sample_traces)
        if length < args.context:
            terminal = gen.decode_forward(
                torch.tensor([[777]]),
                [length],
                page_table=gen.page_table,
                kv_cache=gen.cache,
                reload_inputs=True,
            )
            assert torch.isfinite(terminal).all()
            result["terminal_decode_position"] = length
            result["terminal_decode_finite"] = True
        if args.also_context and length != args.context:
            entire = (torch.arange(args.context, dtype=torch.int64) % 1000 + 100).reshape(1, -1)
            last = gen.prefill_forward(
                entire,
                page_table=gen.page_table,
                kv_cache=gen.cache,
                prompt_lens=[args.context],
                return_all_logits=False,
            )
            assert torch.isfinite(last).all()
            result["full_context_prefill_length"] = args.context
            result["full_context_prefill_finite"] = True
        assert mesh.num_program_cache_entries() == programs
        if args.all_layers:
            from dataclasses import asdict

            from models.autoports.ornith_ai_ornith_1_5_9b.tt.cache_contract import PERSISTENT_CATEGORIES, DramBudget

            view = model.memory_view()
            binding = gen.cache_ownership.register_inferred(gen.cache, context=args.context)
            budget = DramBudget(
                measured_free_bytes_per_device=(view["total_bytes_free_per_bank"] * view["num_banks"],) * 4,
                measurement_artifact=str(args.output),
                included_categories=tuple(sorted(PERSISTENT_CATEGORIES)),
                pending_bytes_per_device={},
                reserve_bytes_per_device=1024**3,
                existing_target_pool_bytes_per_device=(binding.kv_bytes_per_device,) * 4,
            )
            result["cache_spec"] = {str(i): asdict(spec) for i, spec in gen.get_kv_cache_spec().items()}
            result["validated_total_token_capacity"] = gen.get_max_tokens_all_users(
                budget=budget,
                validated_num_blocks=binding.num_blocks,
                validation_artifact=str(args.output),
                max_model_len=args.context,
                max_num_seqs=1,
                specs=gen.get_kv_cache_spec(),
            )
            result["budget_reserve_bytes_per_device"] = budget.reserve_bytes_per_device
            result["allocator_scope"] = "MeshDevice lockstep allocator across all four physical chips"
        print("CAPACITY_PROBE_PASS", length, elapsed, flush=True)
    except Exception as error:
        result.update(status="fail", error=repr(error))
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        mesh.set_program_cache_misses_allowed(True)
        if gen is not None:
            gen.close()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
