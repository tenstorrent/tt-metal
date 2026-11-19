# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Reduced real0/1/3 or full48 prefill signature/canonical-state correctness.

Device owner only; no launch by author. Bound each invocation with timeout1800.
Correctness: TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0.
--generator-source tests/diagnostic_sources/prefill_trace_v1/generator.py selects
an isolated frozen original, never patches production. --timing-only separates
untracked warmed B1/B2 tail timings from correctness evidence. Run --shared-kv
as well to cover canonical scratch state with the original global KV alias.

Example (from repo root after accepted lane setup):
  timeout --signal=INT --kill-after=60 1800 python_env/bin/python -m \
    models.autoports.qwen_qwen3_8_flash_next.tests.probe_prefill_trace_signatures \
    --shared-kv --output models/autoports/qwen_qwen3_8_flash_next/doc/full_model/prefill_signatures_v1.json
"""

import argparse
import ast
import hashlib
import importlib.util
import inspect
import json
import os
import statistics
import sys
import textwrap
import time
import traceback
from contextlib import contextmanager
from pathlib import Path

import torch


@contextmanager
def prefill_reduction_counterfactual(mode):
    """Diagnostic only: change one prefill reduction boundary, never decode.

    The actual grouped body is derived from the current method, not reimplemented.
    The second mode reuses the prior BF16 batched-matmul diagnostic helper for
    n>1 and retains the original n1 path. No production file is changed.
    """
    if mode is None:
        yield None
        return
    from models.autoports.qwen_qwen3_8_flash_next.tt.multichip_decoder import MeshMoe, MultichipDecoder

    original_reduce, original_prefill = MeshMoe._reduce, MultichipDecoder.prefill_forward
    source = textwrap.dedent(inspect.getsource(original_reduce))
    tree = ast.parse(source)
    function = tree.body[0]
    first = function.body[0]
    expected = ast.parse("if n == 1:\n    return self._reduce_single(combined, weights, ids_rm)").body[0]
    assert ast.dump(first, include_attributes=False) == ast.dump(
        expected, include_attributes=False
    ), "Unexpected reduction dispatch; re-review instead of rewriting a different function"
    function.body = function.body[1:]
    namespace = dict(original_reduce.__globals__)
    exec(compile(ast.fix_missing_locations(tree), "<diagnostic-native-prefill-n1>", "exec"), namespace)
    grouped = namespace[function.name]
    import ttnn as tt
    from models.autoports.qwen_qwen3_8_flash_next.tests.probe_batch_reduction_counterfactual import batched_reduction

    configs = {}
    state = {
        "prefill_depth": 0,
        "overridden_calls": 0,
        "decode_overrides": 0,
        "original_reduce_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "derived_ast_sha256": hashlib.sha256(ast.dump(tree).encode()).hexdigest(),
        "mode": mode,
        "unchanged_outside_prefill_calls": 0,
        "batched_helper_sha256": hashlib.sha256(inspect.getsource(batched_reduction).encode()).hexdigest(),
    }

    def prefill(self, *args, **kwargs):
        state["prefill_depth"] += 1
        try:
            return original_prefill(self, *args, **kwargs)
        finally:
            state["prefill_depth"] -= 1

    def reduce(self, combined, weights, ids_rm, n):
        if state["prefill_depth"] and mode == "native_single" and n == 1:
            state["overridden_calls"] += 1
            return grouped(self, combined, weights, ids_rm, n)
        if state["prefill_depth"] and mode == "bf16_batched" and n > 1:
            state["overridden_calls"] += 1
            if id(self) not in configs:
                configs[id(self)] = tt.init_device_compute_kernel_config(
                    self.loader.mesh_device.arch(),
                    math_fidelity=tt.MathFidelity.HiFi4,
                    math_approx_mode=False,
                    fp32_dest_acc_en=False,
                    dst_full_sync_en=False,
                    packer_l1_acc=False,
                )
            return batched_reduction(tt, self, combined, weights, ids_rm, n, configs[id(self)])
        if not state["prefill_depth"]:
            state["unchanged_outside_prefill_calls"] += 1
        return original_reduce(self, combined, weights, ids_rm, n)

    MeshMoe._reduce, MultichipDecoder.prefill_forward = reduce, prefill
    try:
        yield state
    finally:
        MeshMoe._reduce, MultichipDecoder.prefill_forward = original_reduce, original_prefill


def comparison(expected, actual, thresholds):
    assert expected.shape == actual.shape
    a, b = expected.double(), actual.double()
    assert bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    ac, bc = a - a.mean(-1, keepdim=True), b - b.mean(-1, keepdim=True)
    denom = ac.norm(dim=-1) * bc.norm(dim=-1)
    pcc = (ac * bc).sum(-1) / denom.clamp_min(1e-100)
    pcc = torch.where(denom == 0, (a == b).all(-1).double(), pcc)
    result = {
        "exact": bool(torch.equal(expected, actual)),
        "pcc_mean": float(pcc.mean()),
        "pcc_min": float(pcc.min()),
        "top1": float((a.argmax(-1) == b.argmax(-1)).double().mean()),
        "max_abs": float((a - b).abs().max()),
    }
    result["passed"] = (
        result["pcc_mean"] >= thresholds["pcc_mean_min"]
        and result["pcc_min"] >= thresholds["pcc_min_min"]
        and result["top1"] >= thresholds["top1_min"]
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--generator-source", type=Path)
    parser.add_argument("--shared-kv", action="store_true")
    parser.add_argument("--max-batch-size", type=int, choices=(1, 2), default=2)
    parser.add_argument(
        "--all-layers",
        action="store_true",
        help="Full48 correctness confirmation only; no counterfactual, timing or profiler",
    )
    parser.add_argument("--timing-only", action="store_true")
    parser.add_argument("--baseline-tensors", type=Path)
    parser.add_argument("--check-decode-submission-order", action="store_true")
    counterfactual = parser.add_mutually_exclusive_group()
    counterfactual.add_argument(
        "--prefill-native-single-counterfactual",
        action="store_true",
        help="Diagnostic policy change only: prefill n1 grouped BF16, decode unchanged",
    )
    counterfactual.add_argument(
        "--prefill-bf16-batched-counterfactual",
        action="store_true",
        help="Diagnostic schedule change only: prefill n>1 existing BF16 batched matmul",
    )
    counterfactual.add_argument(
        "--prefill-tail-checkpoint-diagnostic",
        action="store_true",
        help="Reduced diagnostic only: compact odd-tail checkpoint, unchanged arithmetic",
    )
    args = parser.parse_args()
    counterfactual_mode = (
        "native_single"
        if args.prefill_native_single_counterfactual
        else "bf16_batched"
        if args.prefill_bf16_batched_counterfactual
        else None
    )
    layer_indices = list(range(48)) if args.all_layers else [0, 1, 3]
    capacity = 384
    if args.all_layers and (
        counterfactual_mode is not None or args.prefill_tail_checkpoint_diagnostic or args.timing_only
    ):
        parser.error("--all-layers is an unmodified-policy correctness confirmation only")
    if args.max_batch_size == 1 and not args.timing_only:
        parser.error("Mixed-row correctness requires batch2; batch1 is available for --timing-only")
    if args.check_decode_submission_order and args.timing_only:
        parser.error("Submission-order correctness must run tracked, outside timing-only mode")
    if args.output.exists() or args.output.with_suffix(".pt").exists():
        parser.error("Use a new artifact path")
    if counterfactual_mode and "counterfactual" not in args.output.name:
        parser.error("Counterfactual output filename must explicitly contain counterfactual")
    if args.prefill_tail_checkpoint_diagnostic and args.timing_only:
        parser.error("Tail checkpoint diagnosis is never accepted performance evidence")
    if counterfactual_mode and args.timing_only:
        parser.error("Counterfactual is correctness diagnosis, never accepted performance evidence")
    baseline_fixture = None
    if args.baseline_tensors:
        baseline_fixture = torch.load(args.baseline_tensors, weights_only=True, map_location="cpu")
        if baseline_fixture.get("format") != "prefill_trace_snapshots_v2":
            parser.error("Baseline lacks explicit layer-depth provenance; cannot compare old unlabelled tensors")
        if baseline_fixture["layer_indices"] != layer_indices:
            parser.error("Baseline/model layer depths differ")
    model_dir = Path(__file__).resolve().parents[1]
    source = model_dir / "tt/generator.py" if args.generator_source is None else args.generator_source
    if not source.exists() and args.generator_source is not None:
        source = model_dir / args.generator_source
    tracking = os.environ.get("TT_METAL_TRACE_ALLOC_TRACKING") == "1"
    assert args.timing_only or tracking
    assert os.environ.get("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE", "0") == "0"
    assert os.environ.get("TT_MESH_GRAPH_DESC_PATH")
    for name in ("TT_METAL_DEVICE_PROFILER", "TT_METAL_TRACE_PROFILER", "TT_METAL_STREAMING_PROFILER"):
        assert os.environ.get(name) in (None, "", "0")
    from ttnn.tools import trace_allocation_tracker

    import ttnn

    if args.timing_only:
        assert not trace_allocation_tracker.TRACE_ALLOC_TRACKING
        assert not trace_allocation_tracker.TRACE_ALLOC_DIAGNOSTICS
        assert os.environ.get("TT_METAL_TRACE_ALLOC_TRACEBACKS") in (None, "", "0")
        assert "TT_METAL_WATCHER" not in os.environ, "Even watcher=0 enables watcher"
        assert int(os.environ.get("TT_METAL_PROFILE_PERF_COUNTERS") or "0") == 0
    from readiness_check.run_prefill_resume_check import DEFAULT_THRESHOLDS

    from models.autoports.qwen_qwen3_8_flash_next.tests.run_full_model_accuracy import memory

    if args.generator_source is None:
        from models.autoports.qwen_qwen3_8_flash_next.tt.generator import build_generator
    else:
        name = "models.autoports.qwen_qwen3_8_flash_next.tt._frozen_prefill_diagnostic"
        spec = importlib.util.spec_from_file_location(name, source)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        build_generator = module.build_generator
    ref = torch.load(model_dir / "readiness_aime24_chat.refpt", weights_only=True, map_location="cpu")["entries"][0]
    sequence = torch.cat((ref["prompt_tokens"].flatten(), ref["generated_tokens"].flatten())).long()
    report = {
        "passed": False,
        "scope": (
            "full48_prefill_signature_correctness_diagnostic"
            if args.all_layers
            else "reduced_real0_1_3_prefill_signature_diagnostic"
        ),
        "layer_indices": layer_indices,
        "stage_acceptance_claim": False,
        "model_sha256": hashlib.sha256((model_dir / "tt/model.py").read_bytes()).hexdigest(),
        "multichip_decoder_sha256": hashlib.sha256((model_dir / "tt/multichip_decoder.py").read_bytes()).hexdigest(),
        "tt_source_hashes": {
            str(p.relative_to(model_dir)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((model_dir / "tt").rglob("*"))
            if p.is_file() and p.suffix in (".py", ".cpp", ".h", ".hpp", ".cc", ".cxx")
        },
        "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "generator_source": str(source),
        "generator_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "shared_kv": args.shared_kv,
        "tracking": tracking,
        "timing_only": args.timing_only,
        "configured_batch_size": args.max_batch_size,
        "acceptance_eligible": not args.timing_only
        and counterfactual_mode is None
        and not args.prefill_tail_checkpoint_diagnostic,
        "thresholds": dict(DEFAULT_THRESHOLDS),
        "checks": {},
    }
    mesh = gen = checkpoint = None
    started = time.monotonic()
    phase = "setup"
    snapshots = {}
    override = prefill_reduction_counterfactual(counterfactual_mode)
    override_state = override.__enter__()
    if override_state is not None:
        report["counterfactual"] = {
            "mode": counterfactual_mode,
            "precision_changed": counterfactual_mode == "native_single",
            "change": (
                "prefill n1 FP32 DEST -> existing grouped native BF16 DEST"
                if counterfactual_mode == "native_single"
                else "prefill n>1 vectorized BF16/HiFi4 matmul; n1 FP32 unchanged"
            ),
            "decode_policy_changed": False,
            "acceptance_claim": False,
            "statistics": override_state,
        }

    def mark(label):
        nonlocal phase
        phase = label
        print(json.dumps({"phase": label, "wall_seconds": time.monotonic() - started}), flush=True)

    try:
        torch.set_num_threads(8)
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=0, l1_small_size=32768)
        gen = build_generator(
            model_dir,
            mesh,
            layer_indices=tuple(layer_indices),
            max_seq_len=capacity,
            max_batch_size=args.max_batch_size,
            defer_kv_cache=args.shared_kv,
        )
        assert list(gen.model.layer_indices) == layer_indices
        from .shared_pool_layout import indexed_attention_layout, validate_pool_aliases

        specs, layer_tables = indexed_attention_layout(gen.caches[0], args.max_batch_size, capacity)
        # Each layer owns a distinct pool, so the same physical page IDs are legal
        # across layers. Keep one common table to match the reduced baseline.
        table = layer_tables[0]
        pages_per_slot = table.shape[-1]
        private_table = torch.arange(pages_per_slot, dtype=torch.int32)
        report["pool_layout"] = {
            "indexed_attention_layers": len(specs),
            "helper_sha256": hashlib.sha256(Path(inspect.getfile(indexed_attention_layout)).read_bytes()).hexdigest(),
            "pages_per_slot": pages_per_slot,
            "specs": [{"shape": list(shape), "tensor_idx": index} for shape, _, index in specs],
            "common_page_table": table.tolist(),
        }
        pool = gen.allocate_kv_cache_per_layer(specs) if args.shared_kv else None
        if pool is not None:
            validate_pool_aliases([*gen.caches, gen.slot_mover.scratch], pool, specs)
            for index, spec in enumerate(specs):
                assert all(
                    t.dtype == ttnn.bfloat16 and tuple(t.shape) == spec[0] for t in pool.pair_for_attention_layer(index)
                )
        if args.prefill_tail_checkpoint_diagnostic:
            from .prefill_tail_checkpoint_diagnostic import TailCheckpointDiagnostic

            checkpoint = TailCheckpointDiagnostic(gen, memory)
            report["tail_checkpoint_diagnostic"] = checkpoint.report
            report["negative_control_artifacts"] = [
                "doc/full_model/prefill_signatures_original_v1.json",
                "doc/full_model/prefill_signatures_full48_v2.json",
            ]
        gen.prepare()
        report["setup_seconds"] = time.monotonic() - started
        report["memory_prepared"] = memory(mesh)
        report["programs_after_prepare"] = mesh.num_program_cache_entries()
        report["first_capture_memory"] = []
        original_chunk = gen._prefill_chunk

        def observed_chunk(count, slot):
            previous = set(gen.prefill_traces)
            before = memory(mesh) if not args.timing_only else None
            original_chunk(count, slot)
            added = set(gen.prefill_traces) - previous
            if added and not args.timing_only:
                after = memory(mesh)
                report["first_capture_memory"].append(
                    {
                        "count": count,
                        "submitted_slot": slot,
                        "new_keys": [repr(key) for key in added],
                        "memory_before": before,
                        "memory_after": after,
                        "allocated_delta_bytes_per_rank": after["num_banks"]
                        * (after["total_bytes_allocated_per_bank"] - before["total_bytes_allocated_per_bank"]),
                        "scope": "Net allocator growth across capture/replay; not isolated command-buffer bytes",
                    }
                )

        if not args.timing_only:
            gen._prefill_chunk = observed_chunk

        def prefill(lengths, slots=(0, 1), starts=None, all_logits=True):
            tokens = torch.nn.utils.rnn.pad_sequence([sequence[:end] for end in lengths], batch_first=True)
            return gen.prefill_forward(
                tokens,
                prompt_lens=list(lengths),
                start_pos=starts,
                slots=slots,
                return_all_logits=all_logits,
                page_table=table[list(slots)] if pool is not None else private_table.repeat(len(slots), 1),
                kv_cache=pool if pool is not None else [gen.caches[slot] for slot in slots],
            )

        def ids():
            return {
                name: {repr(key): str(value) for key, value in getattr(gen, name).items()}
                for name in ("prefill_traces", "traces", "sampling_traces")
            }

        def selected(values, length):
            indices = sorted(set((0, min(15, length - 1), min(31, length - 1), length - 1)))
            return values[indices].clone()

        if not args.timing_only:
            mark("MIXED_ALL_LOGITS")
            solos = [prefill((length,), slots=(0,))[0, :length].clone() for length in (33, 47)]
            mixed = prefill((33, 47))
            mixed_exact = True
            for row, length in enumerate((33, 47)):
                mixed_exact = bool(torch.equal(solos[row], mixed[row, :length])) and mixed_exact
                snapshots[f"mixed{length}"] = selected(mixed[row], length)
            reverse = prefill((47, 33), slots=(1, 0))
            for row, reference in enumerate(reversed(solos)):
                mixed_exact = bool(torch.equal(reference, reverse[row, : len(reference)])) and mixed_exact
            del mixed, reverse, solos
            report["checks"]["mixed_and_reversed_all_logits_exact"] = mixed_exact
            mark("MIXED_NONALIGNED_RESUME")
            lengths, starts = (257, 265), (68, 31)
            full = prefill(lengths)
            prefill(starts, all_logits=False)
            continuation = prefill(lengths, starts=starts)
            resume = []
            for row, (end, start) in enumerate(zip(lengths, starts, strict=True)):
                stats = comparison(full[row, start:end], continuation[row, : end - start], DEFAULT_THRESHOLDS)
                resume.append({"slot": row, "start": start, "end": end, **stats})
                snapshots[f"long{end}"] = selected(full[row], end)
            report["checks"]["resume"] = resume
            if checkpoint is not None:
                mark("MISSING_CHECKPOINT_NEGATIVE_CONTROL")
                prefill(starts, all_logits=False)
                checkpoint.valid.pop(1, None)
                negative = prefill(lengths, starts=starts)
                stats = comparison(
                    full[1, starts[1] : lengths[1]], negative[1, : lengths[1] - starts[1]], DEFAULT_THRESHOLDS
                )
                report["missing_checkpoint_negative_control"] = {
                    "scope": "Intentionally invalidate slot1 checkpoint; original failure must remain",
                    "slot": 1,
                    "start": starts[1],
                    "end": lengths[1],
                    **stats,
                }
                report["checks"]["missing_checkpoint_preserves_original_failure"] = not stats["passed"]
                del negative
            del full, continuation
            before = ids()
            prefill((47, 33), slots=(1, 0), all_logits=False)
            prefill((265, 257), slots=(1, 0), all_logits=False)
            report["checks"]["retained_trace_ids_reused"] = ids() == before
            report["checks"]["zero_request_time_releases"] = gen.counters["releases"] == 0
            report["checks"]["bounded_candidate_trace_count"] = (
                args.generator_source is not None or len(gen.prefill_traces) <= 6
            )
            report["trace_ids"] = ids()
            if args.baseline_tensors:
                original = baseline_fixture["snapshots"]
                assert original.keys() == snapshots.keys()
                report["original_comparison"] = {
                    name: comparison(original[name], snapshots[name], DEFAULT_THRESHOLDS) for name in snapshots
                }
            if args.check_decode_submission_order:
                from models.autoports.qwen_qwen3_8_flash_next.tests.decode_submission_order_check import check

                mark("MATCHED_RESET_DECODE_SUBMISSION_ORDER")
                report["decode_submission_order"] = check(gen, prefill, pool, table)
                report["checks"]["decode_submission_order"] = report["decode_submission_order"]["passed"]
            # End with a decode replay to expose allocations made since capture.
            current = [gen.caches[slot].position for slot in (0, 1)]
            gen.decode_forward(
                torch.tensor([42, 43]),
                current,
                slots=(0, 1),
                page_table=table if pool is not None else private_table.repeat(2, 1),
                kv_cache=pool if pool is not None else gen.caches,
                sample_on_device=True,
            )
        else:
            mark("WARMED_TAIL_TIMING")
            timings = []
            for width in range(1, args.max_batch_size + 1):
                for length in (31, 33, 47):
                    slots = tuple(range(width))
                    prefill((length,) * width, slots=slots, all_logits=False)
                    samples = []
                    for _ in range(3):
                        start = time.perf_counter()
                        prefill((length,) * width, slots=slots, all_logits=False)
                        samples.append((time.perf_counter() - start) * 1000)
                    timings.append(
                        {
                            "batch": width,
                            "length": length,
                            "samples_ms": samples,
                            "median_ms": statistics.median(samples),
                        }
                    )
            report["timings"] = timings
            report[
                "timing_boundary"
            ] = "Warmed public prefill including reset, device work and last-logit host read; no sampling"
        report["programs_final"] = mesh.num_program_cache_entries()
        report["checks"]["no_late_program_compilation"] = report["programs_final"] == report["programs_after_prepare"]
        report["memory_final"] = memory(mesh)
        checks_passed = all(
            all(row["passed"] for row in value) if isinstance(value, list) else value
            for value in report["checks"].values()
        ) and all(value["passed"] for value in report.get("original_comparison", {}).values())
        report["diagnostic_checks_passed"] = checks_passed
        if override_state is not None:
            assert override_state["overridden_calls"] > 0 and override_state["unchanged_outside_prefill_calls"] > 0
        report["passed"] = checks_passed and counterfactual_mode is None and not args.prefill_tail_checkpoint_diagnostic
        assert checks_passed, "One or more unchanged correctness gates failed; all collected diagnostics retained"
    except BaseException as error:
        report.update(error=f"{type(error).__name__}: {error}", failed_phase=phase, traceback=traceback.format_exc())
        raise
    finally:
        override.__exit__(None, None, None)
        report["wall_seconds"] = time.monotonic() - started
        if gen is not None:
            report["counters"] = dict(gen.counters)
            report["trace_keys_at_exit"] = {
                name: [repr(key) for key in getattr(gen, name)]
                for name in ("prefill_traces", "traces", "sampling_traces")
            }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if snapshots:
            tensor_path = args.output.with_suffix(".pt")
            torch.save(
                {
                    "format": "prefill_trace_snapshots_v2",
                    "layer_indices": layer_indices,
                    "generator_sha256": report["generator_sha256"],
                    "snapshots": snapshots,
                },
                tensor_path,
            )
            report["snapshot_artifact"] = str(tensor_path)
            report["snapshot_sha256"] = hashlib.sha256(tensor_path.read_bytes()).hexdigest()
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        if checkpoint is not None:
            checkpoint.close()
        if gen is not None:
            gen.close()
        if mesh is not None:
            ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
