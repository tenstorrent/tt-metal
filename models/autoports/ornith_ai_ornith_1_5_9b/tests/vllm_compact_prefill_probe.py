# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Discriminate canonical versus compact execution padding on real layers 0/3.

This numerical diagnostic is not a serving benchmark. After policy adoption the
candidate observes the production selector; the baseline disables compaction.
Before adoption it substitutes height 16384/32768 with8192 for valid<=8192.
Planner horizons, profile tags, chunk caps, and precision configuration stay fixed.
Run serially under an external timeout, with trace allocation tracking enabled.
"""

import argparse
import copy
import hashlib
import json
import os
import time
from contextlib import ExitStack, nullcontext
from pathlib import Path
from unittest.mock import patch


def chunk_schedules(chunk_tokens):
    if chunk_tokens <= 0 or chunk_tokens % 32 or 2 * (chunk_tokens + 9) >= 32768:
        raise ValueError("chunk-tokens must be a positive multiple of 32 with 2*(n+9) < 32768")
    tail = chunk_tokens + 9
    return (
        (f"aligned_{chunk_tokens}_then_{chunk_tokens}_then_completion", [chunk_tokens, 2 * chunk_tokens, 32768]),
        (f"nonaligned_{tail}_then_{tail}_then_completion", [tail, 2 * tail, 32768]),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--all-layers", action="store_true", help="Compare the complete selected model")
    parser.add_argument(
        "--chunk-tokens", type=int, default=1024, help="Aligned fragment length for both comparison schedules"
    )
    parser.add_argument(
        "--experimental-height", type=int, choices=(6144,), help="Diagnostic-only extra prepared extent"
    )
    parser.add_argument(
        "--refresh-cache-control",
        action="store_true",
        help="Compare forced refresh against cached refresh with production math in both arms",
    )
    args = parser.parse_args()
    try:
        schedules = chunk_schedules(args.chunk_tokens)
    except ValueError as error:
        parser.error(str(error))
    if args.refresh_cache_control and args.experimental_height:
        parser.error("Refresh-cache comparison must retain production execution geometry")
    context = 262144 if args.all_layers else 65536
    # Check before TTNN import: a falsely untracked success is not useful evidence.
    if os.environ.get("TT_METAL_TRACE_ALLOC_TRACKING") != "1":
        parser.error("Set TT_METAL_TRACE_ALLOC_TRACKING=1 before starting this process")
    if os.environ.get("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE", "0") != "0":
        parser.error("Program-cache allocations must remain included in tracking")

    import torch
    from ttnn.tools import trace_allocation_tracker

    import ttnn
    from models.autoports.ornith_ai_ornith_1_5_9b.tt import prefill_ops, prefill_runtime
    from models.autoports.ornith_ai_ornith_1_5_9b.tt.generator import build_generator
    from models.autoports.ornith_ai_ornith_1_5_9b.tt.prefill_runtime import PrefillRuntime
    from models.common.sampling.generator import SamplingParams

    if args.refresh_cache_control and not hasattr(prefill_ops.NormalizedLinearPrefill, "_refresh_profile"):
        parser.error("Apply prefill_refresh_candidate.patch before running the refresh-cache comparison")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "running",
        "scope": "Full-stack B1 TP4 comparison" if args.all_layers else "Reduced layers 0/3, B1, TP4 discriminator",
        "serving_acceptance": False,
        "context": context,
        "context_scope": (
            "Full context contract 262144"
            if args.all_layers
            else "Reduced diagnostic allocation 65536; original horizon 32768. Serving/eval context stays 262144."
        ),
        "canonical_horizon": 32768,
        "chunk_tokens": args.chunk_tokens,
        "candidate_height": args.experimental_height or 8192,
        "experimental": args.experimental_height is not None,
        "refresh_cache_control": args.refresh_cache_control,
        "tracking": trace_allocation_tracker.TRACE_ALLOC_TRACKING,
        "program_cache_tracking_included": True,
        "cases": [],
    }

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    def compare(actual, expected):
        assert actual.shape == expected.shape, (actual.shape, expected.shape)
        a, b = actual.double().reshape(-1), expected.double().reshape(-1)
        finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
        exact = torch.equal(actual, expected)
        ac, bc = a - a.mean(), b - b.mean()
        denom = torch.linalg.vector_norm(ac) * torch.linalg.vector_norm(bc)
        pcc = float(torch.dot(ac, bc) / denom) if finite and denom > 0 else None
        return {
            "exact": exact,
            "finite": finite,
            "pcc_diagnostic_only": pcc,
            "normalized_pcc_ge_0_995_diagnostic_only": finite and (exact or (pcc is not None and pcc >= 0.995)),
            "max_abs": float((a - b).abs().max()) if finite else None,
            "different_elements": int(torch.count_nonzero(actual != expected)),
            "elements": actual.numel(),
        }

    def logit_diagnostics(actual_rows, expected_rows):
        # Only fragment-terminal/decode logits, not a full resume corpus.
        rows = [compare(a, b) for a, b in zip(actual_rows, expected_rows, strict=True)]
        pccs = [1.0 if row["exact"] else row["pcc_diagnostic_only"] for row in rows]
        top1 = [bool(torch.equal(a.argmax(-1), b.argmax(-1))) for a, b in zip(actual_rows, expected_rows, strict=True)]
        top5 = [
            float((a.topk(5, dim=-1).indices == b.argmax(-1, keepdim=True)).any(-1).float().mean())
            for a, b in zip(actual_rows, expected_rows, strict=True)
        ]
        finite_pccs = all(value is not None for value in pccs)
        return {
            "scope": "Fragment-terminal and next-decode logits only; not a full resume or HF accuracy gate",
            "mean_pcc": sum(pccs) / len(pccs) if finite_pccs else None,
            "min_pcc": min(pccs) if finite_pccs else None,
            "top1_agreement": sum(top1) / len(top1),
            "baseline_top1_in_candidate_top5": sum(top5) / len(top5),
            "existing_resume_thresholds_diagnostic_only": finite_pccs
            and sum(pccs) / len(pccs) >= 0.99
            and min(pccs) >= 0.98
            and sum(top1) / len(top1) >= 0.98,
        }

    torch.set_num_threads(8)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=256 * 1024 * 1024)
    gen = None
    catalogue_patches = ExitStack()
    try:
        if args.experimental_height:
            # Patch only preparation/validation catalogues. Planner canonical
            # choices, caps and profile identities stay unchanged. Linear ops
            # also validate their explicit bucket list against a local import.
            for module, name in (
                (prefill_runtime, "LINEAR_BUCKETS"),
                (prefill_runtime, "FULL_PREFILL_BUCKETS"),
                (prefill_ops, "LINEAR_BUCKETS"),
                (prefill_ops, "FULL_PREFILL_BUCKETS"),
            ):
                choices = tuple(sorted(set(getattr(module, name)) | {args.experimental_height}))
                catalogue_patches.enter_context(patch.object(module, name, choices))
            report["experimental_catalogues"] = [
                "prefill_runtime.LINEAR_BUCKETS",
                "prefill_runtime.FULL_PREFILL_BUCKETS",
                "prefill_ops.LINEAR_BUCKETS",
                "prefill_ops.FULL_PREFILL_BUCKETS",
            ]
        policy = Path(__file__).resolve().parents[1] / "doc/datatype_sweep/configs/bfp8_all_lofi.json"
        gen = build_generator(
            None,
            mesh,
            layer_indices=None if args.all_layers else [0, 3],
            max_batch_size=1,
            max_seq_len=context,
            precision_config=policy,
        )
        report["precision"] = gen.model.precision_summary()
        text = (
            "A careful engineer measures the same computation before changing its implementation. "
            "The river crosses the valley, while the observatory records the changing evening sky. "
            "Explain how attention and recurrent memory preserve information across a sequence. "
        )
        seed_ids = gen.tokenizer.encode(text, add_special_tokens=False)
        assert seed_ids
        ids = (seed_ids * ((32769 + len(seed_ids) - 1) // len(seed_ids)))[:32769]
        report["prompt"] = {
            "source": text,
            "construction": "Repeated tokenizer IDs: original prompt 32768 plus one teacher-forced decode token; no quality claim",
            "sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
        }
        params = SamplingParams(temperature=0.0, top_k=1, top_p=1.0, seed=0)
        report["phase"] = "prepare_all_profiles_and_capture_decode_first"
        save()
        gen.generate(ids[:128], 2, sampling_params=params)
        runtime = gen.prefill_runtime
        assert 8192 in runtime.profile_spec["linear"] and 8192 in runtime.profile_spec["full"]
        assert all(report["candidate_height"] in runtime.profile_spec[key] for key in ("linear", "full"))

        def trace_ids():
            return {
                str(key): {
                    "model": {str(k): str(v) for k, v in state.model_traces.items()},
                    "sampling": {str(k): str(v) for k, v in state.sample_traces.items()},
                }
                for key, state in gen.states.items()
            }

        captured_ids = trace_ids()
        captured_count = gen.counters["captures"]
        assert captured_count and all(value["model"] and value["sampling"] for value in captured_ids.values())
        report["trace_ids_before"] = captured_ids
        report["programs_after_capture"] = mesh.num_program_cache_entries()
        original_execute = PrefillRuntime._execute
        original_select = getattr(PrefillRuntime, "_execution_height", None)
        report["candidate_policy_route"] = (
            "production8192_cached_refresh_vs_forced_refresh"
            if args.refresh_cache_control
            else "experimental_6144_after_production_helper"
            if args.experimental_height
            else "production_helper_callthrough"
            if original_select
            else "experimental_execute_override"
        )
        executed = []
        refresh_targets = {
            id(profile[name]): "linear_profile"
            for adapter in runtime.linear.values()
            for profile in adapter.profiles.values()
            for name in ("g_mask", "beta_mask", "history_indices")
        }
        refresh_targets.update({id(tensor): "row_indices" for tensor in runtime.row_indices.values()})
        attention_cache_enabled = hasattr(prefill_ops, "_cached_attention_inputs")
        report["attention_plan_cache_enabled"] = attention_cache_enabled
        if attention_cache_enabled:
            for bound in runtime.bindings.values():
                for layer, item in zip(gen.model.layers, bound["layers"], strict=True):
                    if layer.kind == "full_attention":
                        for adapter in item["adapters"].values():
                            refresh_targets.update(
                                {id(tensor): "attention_input" for tensor in adapter.inputs.values()}
                            )

        def observe_production_height(self, kind, height, valid):
            production = original_select(self, kind, height, valid)
            effective = (
                args.experimental_height
                if args.experimental_height and height in (16384, 32768) and valid <= args.experimental_height
                else production
            )
            executed.append({"kind": kind, "valid": valid, "canonical": height, "effective": effective})
            return effective

        def compact_execute(self, index, bound, source, destination, *, absolute, valid, height, mapping):
            candidate_height = report["candidate_height"]
            effective = candidate_height if height in (16384, 32768) and valid <= candidate_height else height
            executed.append(
                {"layer": index, "absolute": absolute, "valid": valid, "canonical": height, "effective": effective}
            )
            return original_execute(
                self,
                index,
                bound,
                source,
                destination,
                absolute=absolute,
                valid=valid,
                height=effective,
                mapping=mapping,
            )

        def state_snapshot(end):
            snapshot = {}
            for index, (layer, states) in enumerate(zip(gen.model.layers, gen.cache, strict=True)):
                names = ("conv", "recurrent") if layer.kind == "linear_attention" else ("key", "value")
                for name in names:
                    for shard, tensor in enumerate(ttnn.get_device_tensors(getattr(states[0], name))):
                        host = ttnn.to_torch(tensor)
                        if name in ("key", "value"):
                            # Contiguous B1 logical pages; omit invalid rows in the partial final page.
                            host = (
                                host[: (end + 31) // 32]
                                .permute(0, 2, 1, 3)
                                .reshape(-1, host.shape[1], host.shape[3])[:end]
                            )
                        snapshot[f"layer{index}/{name}/shard{shard}"] = host.clone()
            return snapshot

        def run(ends, candidate):
            gen.reset(clear_kv=True)
            executed.clear()
            outputs, snapshots, metadata, timing, programs = [], [], [], [], []
            refresh_copies = dict.fromkeys(set(refresh_targets.values()), 0)
            if args.refresh_cache_control:
                scope = ExitStack()
                scope.enter_context(patch.object(PrefillRuntime, "_execution_height", observe_production_height))
                original_copy = ttnn.copy_host_to_device_tensor

                def count_refresh_copy(host, destination, *positional, **keywords):
                    kind = refresh_targets.get(id(destination))
                    if kind:
                        refresh_copies[kind] += 1
                    return original_copy(host, destination, *positional, **keywords)

                scope.enter_context(patch.object(ttnn, "copy_host_to_device_tensor", side_effect=count_refresh_copy))
                if not candidate:
                    original_refresh = prefill_ops.NormalizedLinearPrefill._refresh_profile
                    original_read = PrefillRuntime._read
                    if attention_cache_enabled:
                        original_attention_refresh = prefill_ops.NormalizedAttentionPrefill.refresh
                        original_attention_plan = prefill_ops._cached_attention_inputs

                        def force_attention_refresh(self, **kwargs):
                            self._input_values.clear()
                            return original_attention_refresh(self, **kwargs)

                        def force_attention_plan(*positional, **keywords):
                            original_attention_plan.cache_clear()
                            return original_attention_plan(*positional, **keywords)

                        scope.enter_context(
                            patch.object(prefill_ops.NormalizedAttentionPrefill, "refresh", force_attention_refresh)
                        )
                        scope.enter_context(patch.object(prefill_ops, "_cached_attention_inputs", force_attention_plan))

                    def force_profile_refresh(self, *, valid_count, bucket):
                        self.profiles[bucket]["valid_count"] = None
                        return original_refresh(self, valid_count=valid_count, bucket=bucket)

                    def force_row_refresh(self, source, absolute, height):
                        self._row_indices_absolute.pop(height, None)
                        return original_read(self, source, absolute, height)

                    scope.enter_context(
                        patch.object(prefill_ops.NormalizedLinearPrefill, "_refresh_profile", force_profile_refresh)
                    )
                    scope.enter_context(patch.object(PrefillRuntime, "_read", force_row_refresh))
            elif candidate and original_select is not None:
                # Observe the adopted decision without supplying the answer or
                # changing _execute. The numerical candidate must exercise it.
                scope = patch.object(PrefillRuntime, "_execution_height", observe_production_height)
            elif candidate:
                scope = patch.object(PrefillRuntime, "_execute", compact_execute)
            elif original_select is not None:
                # Preserve a real canonical control after the candidate is adopted.
                scope = patch.object(PrefillRuntime, "_execution_height", lambda self, kind, height, valid: height)
            else:
                scope = nullcontext()
            with scope:
                start = 0
                for end in ends:
                    ttnn.synchronize_device(mesh)
                    before = time.perf_counter()
                    logits = gen.prefill_forward(
                        torch.tensor([ids[:end]], dtype=torch.long),
                        page_table=gen.page_table,
                        kv_cache=gen.cache,
                        prompt_lens=[end],
                        original_prompt_lens=[32768],
                        request_lens=[32768],
                        start_pos=[start],
                    )
                    ttnn.synchronize_device(mesh)
                    timing.append((time.perf_counter() - before) * 1000)
                    outputs.append(logits.clone())
                    snapshots.append(state_snapshot(end))
                    metadata.append(copy.deepcopy(runtime.events[-1]))
                    programs.append(mesh.num_program_cache_entries())
                    start = end
            # Decode is legal only after the authoritative original prompt has
            # completed. Preserve its 32768 horizon through all partial chunks.
            assert end == 32768
            # End every mixed workload with a real replay so allocation safety is exercised.
            # A fixed teacher-forced token isolates state differences from sampled-token divergence.
            decode = gen.decode_forward(
                torch.tensor([[ids[end]]]),
                [end],
                page_table=gen.page_table,
                kv_cache=gen.cache,
                sampling_params=None,
                read_from_device=True,
                reload_inputs=True,
            )
            assert trace_ids() == captured_ids and gen.counters["captures"] == captured_count
            assert all(count == report["programs_after_capture"] for count in programs), "Late program compilation"
            if candidate:
                assert any(
                    item["canonical"] in (16384, 32768) and item["effective"] == report["candidate_height"]
                    for item in executed
                ), "Candidate did not exercise compact production execution"
            return {
                "logits": outputs,
                "states": snapshots,
                "plans": metadata,
                "decode": decode.clone(),
                "timing_ms_diagnostic": timing,
                "program_counts": programs,
                "executed": list(executed),
                "refresh_copies": refresh_copies,
            }

        for label, ends in schedules:
            report["phase"] = label
            save()
            baseline = run(ends, False)
            candidate = run(ends, True)
            assert baseline["plans"] == candidate["plans"], "Candidate altered canonical planning metadata"
            if args.refresh_cache_control:
                assert baseline["executed"] == candidate["executed"], "Refresh-only control changed execution geometry"
                assert all(
                    candidate["refresh_copies"][kind] < baseline["refresh_copies"][kind]
                    for kind in set(refresh_targets.values())
                ), "Refresh candidate did not eliminate redundant copies"
            comparisons = {
                f"logits/end{end}": compare(a, b)
                for end, a, b in zip(ends, candidate["logits"], baseline["logits"], strict=True)
            }
            for end, actual, expected in zip(ends, candidate["states"], baseline["states"], strict=True):
                comparisons.update(
                    {f"state/end{end}/{key}": compare(value, expected[key]) for key, value in actual.items()}
                )
            comparisons["decode_logits"] = compare(candidate["decode"], baseline["decode"])
            exact = all(item["exact"] and item["finite"] for item in comparisons.values())
            report["cases"].append(
                {
                    "label": label,
                    "ends": ends,
                    "status": "exact-pass" if exact else "numerical-difference",
                    "comparisons": comparisons,
                    "canonical_plans_unchanged": True,
                    "logit_numerical_diagnostics": logit_diagnostics(
                        candidate["logits"] + [candidate["decode"]], baseline["logits"] + [baseline["decode"]]
                    ),
                    "plans": baseline["plans"],
                    "candidate_executions": candidate["executed"],
                    "baseline_refresh_copies": baseline["refresh_copies"],
                    "candidate_refresh_copies": candidate["refresh_copies"],
                    "baseline_ms_diagnostic": baseline["timing_ms_diagnostic"],
                    "candidate_ms_diagnostic": candidate["timing_ms_diagnostic"],
                    "baseline_program_counts": baseline["program_counts"],
                    "candidate_program_counts": candidate["program_counts"],
                }
            )
            save()
        report["trace_ids_after"] = trace_ids()
        report["counters_after"] = dict(gen.counters)
        report["status"] = (
            "exact-pass" if all(case["status"] == "exact-pass" for case in report["cases"]) else "numerical-difference"
        )
        save()
        assert (
            report["status"] == "exact-pass"
        ), "Compact padding changes numerics; diagnostic PCC does not waive exact equality"
    except BaseException as exc:
        if report["status"] == "running":
            report["status"] = "error"
        report["error"] = repr(exc)
        raise
    finally:
        save()
        if gen is not None:
            gen.close()
        ttnn.close_mesh_device(mesh)
        catalogue_patches.close()
    print("COMPACT_PREFILL_EXACT_PASS", args.output, flush=True)


if __name__ == "__main__":
    main()
