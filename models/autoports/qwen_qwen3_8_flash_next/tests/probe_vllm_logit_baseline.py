# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Serialized all48/B2/C1024 selected-policy logit and prefill-mask diagnostic.

Hardware execution belongs to the root device owner. Run under bounded timeout,
with trace allocation tracking and no live server. Host vocabulary readbacks and
seed-state reads here are diagnostic only, never performance evidence.
"""

import argparse
import json
import os
import time
import traceback
from pathlib import Path

from models.autoports.qwen_qwen3_8_flash_next.tests.check_vllm_logit_determinism import (
    LAYOUTS,
    MODEL_DIR,
    PROMPTS,
    comparisons,
    source_manifest,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--top-logprobs", type=int, default=20)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".pt").exists():
        parser.error("Output JSON/tensor paths must be fresh")
    if args.repeats < 2 or not 2 <= args.top_logprobs <= 20:
        parser.error("Need repeats>=2 and 2..20 top logprobs")
    if os.environ.get("TT_METAL_TRACE_ALLOC_TRACKING") != "1":
        parser.error("Set TT_METAL_TRACE_ALLOC_TRACKING=1 before device imports")
    for flag in (
        "TT_METAL_WATCHER",
        "TT_METAL_DEVICE_PROFILER",
        "TT_METAL_TRACE_PROFILER",
        "TT_METAL_STREAMING_PROFILER",
    ):
        if (flag == "TT_METAL_WATCHER" and flag in os.environ) or os.environ.get(flag) not in (None, "", "0"):
            parser.error("Run without watcher/profiler")

    import torch

    import ttnn
    from models.autoports.qwen_qwen3_8_flash_next.tests.shared_pool_layout import shared_pool_layout
    from models.autoports.qwen_qwen3_8_flash_next.tt.generator import build_generator
    from models.common.sampling.sampling_params import SamplingParams

    source = source_manifest()
    report = dict(
        passed=False,
        scope="all48_B2_C1024_selected_policy_explicit_host_logit_diagnostic",
        precision=source["precision"],
        source=source,
        capacity=1024,
        batch_capacity=2,
        layer_indices=list(range(48)),
        prompts=PROMPTS,
        runs=[],
        checks={},
        mask_cases=[],
        raw_logits=str(args.output.with_suffix(".pt")),
        performance_claim=False,
    )
    raw, mesh, gen = {}, None, None
    started = time.monotonic()
    try:
        torch.set_num_threads(8)
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=0, l1_small_size=32768)
        gen = build_generator(
            MODEL_DIR,
            mesh,
            precision_config=source["precision"],
            layer_indices=tuple(range(48)),
            max_batch_size=2,
            max_seq_len=1024,
            defer_kv_cache=True,
        )
        report["runtime_precision"] = gen.model.precision_summary()
        if (
            list(gen.model.layer_indices) != list(range(48))
            or report["runtime_precision"]["requested"] != source["precision"]
        ):
            raise ValueError("Runtime model differs from all48 selected policy")
        specs, tables = shared_pool_layout(2, 1024, 12)
        pool = gen.allocate_kv_cache_per_layer(
            [(shape, source["precision"]["kv_cache_dtype"], i) for shape, _, i in specs]
        )
        gen.prepare()
        report["counters_prepared"] = dict(gen.counters)

        def prefill(prompt_ids, slots, ends=None, starts=None, mask=None, params=None):
            ends = [len(PROMPTS[key]) for key in prompt_ids] if ends is None else ends
            tokens = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(PROMPTS[key][:end]) for key, end in zip(prompt_ids, ends, strict=True)], batch_first=True
            )
            selected_tables = [table[list(slots)] for table in tables]
            return gen.prefill_forward(
                tokens,
                prompt_lens=ends,
                start_pos=starts,
                slots=tuple(slots),
                page_table=selected_tables[0],
                page_tables_per_layer=selected_tables,
                kv_cache=pool,
                sampling_params=params,
                prefill_sample_mask=mask,
            )

        for repeat in range(args.repeats):
            for layout, prompt_ids in LAYOUTS.items():
                # Reversed request order changes physical rows as it does in
                # a front-packed serving batch; solo always occupies slot zero.
                slots = list(range(len(prompt_ids)))
                print(f"LOGIT_BASELINE repeat={repeat} layout={layout}", flush=True)
                output = prefill(prompt_ids, slots).float().cpu()
                if output.shape != (len(slots), 1, gen.model.config.vocab_size) or not torch.isfinite(output).all():
                    raise ValueError("Malformed/nonfinite standalone logits")
                raw[f"{layout}:{repeat}"] = output.clone()
                rows = []
                for i, key in enumerate(prompt_ids):
                    lp = output[i, 0].log_softmax(-1)
                    values, ids = lp.topk(args.top_logprobs)
                    rows.append(
                        dict(
                            prompt_id=key,
                            prompt_token_ids=PROMPTS[key],
                            submitted_index=i,
                            sampled_token_id=int(output[i, 0].argmax()),
                            top_logprobs={
                                f"token_id:{token}": value
                                for token, value in zip(ids.tolist(), values.tolist(), strict=True)
                            },
                        )
                    )
                report["runs"].append(
                    dict(
                        layout=layout,
                        repeat=repeat,
                        prompt_ids=prompt_ids,
                        submitted_slots=slots,
                        shape=list(output.shape),
                        rows=rows,
                    )
                )
        report["comparisons"] = comparisons(report["runs"], 0)
        report["checks"]["selected_logprobs_exact"] = all(
            c["passed"] for group in report["comparisons"].values() for c in group
        )
        report["checks"]["full_logits_repeat_exact"] = all(
            torch.equal(raw[f"{layout}:0"], raw[f"{layout}:{repeat}"])
            for repeat in range(1, args.repeats)
            for layout in LAYOUTS
        )
        report["checks"]["full_logits_cross_position_exact"] = all(
            torch.equal(raw[f"ab:{repeat}"], raw[f"ba:{repeat}"].flip(0)) for repeat in range(args.repeats)
        )
        report["checks"]["full_logits_solo_vs_batch_exact"] = all(
            torch.equal(raw[f"ab:{repeat}"][i : i + 1], raw[f"{solo}:{repeat}"])
            for repeat in range(args.repeats)
            for i, solo in enumerate(("solo_a", "solo_b"))
        )

        # Canonical on-device _sample is reached only through public prefill.
        def seeds():
            ttnn.synchronize_device(mesh)
            return torch.stack(
                [
                    ttnn.to_torch(t).to(torch.int64)[:2, :5].clone()
                    for t in ttnn.get_device_tensors(gen.seed_state.state)
                ]
            )

        def params(ids, mask=None):
            mask = [True] * len(ids) if mask is None else mask
            return SamplingParams(
                temperature=[0.8] * len(ids),
                top_k=[20 if emit else 40 for emit in mask],
                top_p=[0.95] * len(ids),
                presence_penalty=[0.0 if emit else 1.0 for emit in mask],
                seed=[53 if key == "a33" else 79 for key in ids],
            )

        def counted(ids, slots, **kw):
            before = dict(gen.counters)
            out = prefill(ids, slots, params=params(ids, kw.get("mask")), **kw)
            return out, {k: gen.counters[k] - before[k] for k in before}

        gen.reset()
        reference, delta = counted(["a33", "b47"], [0, 1])
        reference_seeds = seeds()
        raw["mask_reference_tokens"], raw["mask_reference_seed_state"] = reference, reference_seeds
        report["mask_reference"] = dict(tokens=reference.tolist(), counters=delta, seed_state=reference_seeds.tolist())
        for name, order, mask in (
            ("all_intermediate", [0, 1], [False, False]),
            ("mixed_false_true", [0, 1], [False, True]),
            ("mixed_true_false", [1, 0], [True, False]),
        ):
            gen.reset()
            before_seed = seeds()
            ids = [["a33", "b47"][slot] for slot in order]
            ends = [len(PROMPTS[key]) if emit else 32 for key, emit in zip(ids, mask, strict=True)]
            out, first_delta = counted(ids, order, ends=ends, mask=mask)
            after_seed = seeds()
            inactive = [slot for slot, emit in zip(order, mask, strict=True) if not emit]
            true_rows = [(row, slot) for row, (slot, emit) in enumerate(zip(order, mask, strict=True)) if emit]
            checks = dict(
                inactive_seed_unchanged=torch.equal(before_seed[:, inactive], after_seed[:, inactive]),
                intermediate_no_output=(
                    (out is None)
                    if not any(mask)
                    else all(int(out[row, 0]) == -1 for row, emit in enumerate(mask) if not emit)
                ),
                expected_readbacks=first_delta["readbacks"] == int(any(mask)),
                expected_sampling_replays=first_delta["sampling_replays"] == int(any(mask)),
                final_tokens_equal=all(torch.equal(out[row], reference[slot]) for row, slot in true_rows),
            )
            rest_ids = [["a33", "b47"][slot] for slot in inactive]
            final, final_delta = counted(rest_ids, inactive, starts=[32] * len(inactive), mask=[True] * len(inactive))
            final_seed = seeds()
            checks["continued_tokens_equal"] = all(
                torch.equal(final[row], reference[slot]) for row, slot in enumerate(inactive)
            )
            checks["continued_readbacks"] = final_delta["readbacks"] == 1
            checks["continued_sampling_replays"] = final_delta["sampling_replays"] == 1
            checks["final_seed_state_equal"] = torch.equal(final_seed, reference_seeds)
            report["mask_cases"].append(
                dict(
                    name=name,
                    order=order,
                    sample_mask=mask,
                    checks=checks,
                    first_counters=first_delta,
                    final_counters=final_delta,
                    first_tokens=None if out is None else out.tolist(),
                    final_tokens=final.tolist(),
                    seed_before=before_seed.tolist(),
                    seed_after_first=after_seed.tolist(),
                    seed_final=final_seed.tolist(),
                )
            )
            raw[f"{name}:seeds"] = torch.stack((before_seed, after_seed, final_seed))
        report["checks"]["canonical_mask_tokens_rng_readbacks"] = all(
            all(case["checks"].values()) for case in report["mask_cases"]
        )
        report["counters_final"] = dict(gen.counters)
        report["passed"] = all(report["checks"].values())
    except Exception:
        report["error"] = traceback.format_exc()
    finally:
        try:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            if raw:
                torch.save(
                    dict(precision=source["precision"], prompts=PROMPTS, tensors=raw), args.output.with_suffix(".pt")
                )
        except Exception:
            report["passed"] = False
            report["artifact_error"] = traceback.format_exc()
        try:
            if gen is not None:
                gen.close()
        except Exception:
            report["passed"] = False
            report["generator_cleanup_error"] = traceback.format_exc()
        finally:
            try:
                if mesh is not None:
                    ttnn.close_mesh_device(mesh)
            except Exception:
                report["passed"] = False
                report["mesh_cleanup_error"] = traceback.format_exc()
            report["seconds"] = time.monotonic() - started
            args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            dict(passed=report["passed"], checks=report["checks"], error=report.get("error"), output=str(args.output))
        )
    )
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
