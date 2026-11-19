# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Observe the existing immediate-twice transition oracle without weakening it.

Extra blocking host snapshots deliberately change timing. This is a reduced
state-localization diagnostic, never serving, async-overlap, or performance
acceptance. No device tensors are allocated by this observer.
"""

import argparse
import json
import os
import runpy
import sys
import time
from pathlib import Path


def compare_tensor(left, right, torch):
    """Compact exact comparison; finite errors never hide NaNs or infinities."""
    if left.shape != right.shape or left.dtype != right.dtype:
        return dict(equal=False, left_shape=list(left.shape), right_shape=list(right.shape), dtype_mismatch=True)
    unequal = left != right
    result = dict(equal=bool(torch.equal(left, right)), elements=left.numel(), differing=int(unequal.sum()))
    if left.is_floating_point():
        lf, rf = torch.isfinite(left), torch.isfinite(right)
        finite = lf & rf
        result.update(
            finite_left=int(lf.sum()),
            finite_right=int(rf.sum()),
            nan_left=int(torch.isnan(left).sum()),
            nan_right=int(torch.isnan(right).sum()),
            signed_zero_differences=int(
                ((left == 0) & (right == 0) & (torch.signbit(left) != torch.signbit(right))).sum()
            ),
            max_abs_finite=(
                float((left[finite].double() - right[finite].double()).abs().max()) if finite.any() else None
            ),
        )
    else:
        result["max_abs_finite"] = float((left.double() - right.double()).abs().max()) if left.numel() else 0.0
    if not result["equal"]:
        indices = unequal.nonzero()
        if len(indices):
            index = tuple(indices[0].tolist())
            result.update(
                first_index=list(index), left_value=str(left[index].item()), right_value=str(right[index].item())
            )
    return result


def main(extra_fields=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Original transition oracle JSON path")
    parser.add_argument("--state-output", type=Path, help="Defaults to OUTPUT.with_suffix('.state.json')")
    parser.add_argument(
        "--through-sequence", type=int, default=13, help="Inclusive original ledger sequence; default covers A520/B135"
    )
    parser.add_argument(
        "--save-first-diff", action="store_true", help="Save only the first unequal field/rank pair as a local .pt"
    )
    args = parser.parse_args()
    state_output = args.state_output or args.output.with_suffix(".state.json")
    if args.output.exists() or state_output.exists():
        parser.error("Use fresh original and state artifact paths")
    if not 0 <= args.through_sequence <= 30:
        parser.error("through-sequence must be in0..30, before the scripted page-growth/remap/release transitions")

    import torch

    from models.autoports.qwen_qwen3_8_flash_next.tests.probe_vllm_async import read_ranks, snapshot
    from models.autoports.qwen_qwen3_8_flash_next.tt.generator_vllm import QwenFlashNextForCausalLM

    cls = QwenFlashNextForCausalLM
    original_prefill, original_decode = cls.prefill_forward, cls.decode_forward
    original_release = cls.__dict__.get("release_request")
    original_argv = sys.argv
    baseline = {}
    progress = dict(run=-1, sequence=0, feedback_valid=set())
    report = dict(
        passed=False,
        scope="blocking_host_state_diagnostic_immediate_twice_not_serving_async_or_performance_acceptance",
        original_oracle=str(args.output),
        through_sequence=args.through_sequence,
        sequence_contract="Adapter prefill/decode calls count from0 after each run's release_request(0); matches the original ledger before its later release entry",
        comparison_contract="Even/odd runs within each original history; all snapshots follow the original forward, before its ordinary token read",
        fields_contract="All private cache state and currently mapped K/V pages for live slots; feedback/RNG/membership and request seeds only after that request's first sampled output; sampler logits only for rows sampled by this call",
        environment={
            key: os.environ.get(key)
            for key in (
                "TT_METAL_WATCHER",
                "TT_METAL_TRACE_ALLOC_TRACKING",
                "TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE",
                "QWEN_VLLM_LAYER_INDICES",
            )
        },
        observations=[],
        comparisons=[],
        first_difference=None,
        extra_fields_observer=None if extra_fields is None else extra_fields.__doc__,
    )
    started = time.monotonic()

    def release(adapter, slot):
        if slot == 0:
            progress["run"] += 1
            progress["sequence"] = 0
            progress["feedback_valid"].clear()
            if progress["run"] % 2 == 0:
                baseline.clear()
        progress["feedback_valid"].discard(slot)
        return adapter.generator.release_request(slot)

    def observe(adapter, kind, kwargs):
        run, sequence = progress["run"], progress["sequence"]
        progress["sequence"] += 1
        if run < 0 or sequence > args.through_sequence:
            return
        gen = adapter.generator
        live = tuple(i for i, cache in enumerate(gen.caches) if cache.position > 0)
        if kind == "decode":
            submitted = kwargs.get("slots", tuple(range(len(kwargs["start_pos"]))))
            sampled = tuple(
                slot for slot, position in zip(submitted, kwargs["start_pos"], strict=True) if position >= 0
            )
        else:
            submitted = kwargs.get("empty_slots", kwargs.get("slots", tuple(range(len(kwargs["prompt_lens"])))))
            mask = kwargs.get("prefill_sample_mask", [True] * len(submitted))
            sampled = tuple(slot for slot, emit in zip(submitted, mask, strict=True) if emit)
            for slot, start in zip(submitted, kwargs.get("start_pos", [0] * len(submitted)), strict=True):
                if start == 0:
                    progress["feedback_valid"].discard(slot)
        progress["feedback_valid"].update(sampled)
        feedback_valid = tuple(slot for slot in live if slot in progress["feedback_valid"])
        values = {}
        for slot in live:
            # Read the actual bound device page table, rather than assuming the
            # submitted host table is still authoritative for every survivor.
            attention = next(layer.attention for layer in gen.caches[slot].layers if layer.attention is not None)
            pages_by_rank = read_ranks(attention.page_table)
            # This probe uses a common replicated table for its one QSA layer.
            if not all(torch.equal(pages_by_rank[0], p) for p in pages_by_rank[1:]):
                raise AssertionError("Diagnostic expected replicated page tables")
            pages = pages_by_rank[0].flatten().long().tolist()
            for name, ranks in snapshot(gen.caches[slot], pages).items():
                values[f"slot{slot}.{name}"] = ranks
        for name, tensor in (
            ("feedback", gen.decode_tokens),
            ("rng", gen.seed_state.state),
            ("active", gen.seed_state.active),
        ):
            ranks = read_ranks(tensor) if feedback_valid else ()
            for slot in feedback_valid:
                values[f"slot{slot}.{name}"] = [
                    (rank[slot : slot + 1] if name == "rng" else rank.flatten()[slot : slot + 1]).clone()
                    for rank in ranks
                ]
        if sampled:
            logits = read_ranks(gen.sample_logits)
            for slot in sampled:
                values[f"slot{slot}.sample_logits"] = [rank[..., slot : slot + 1, :].clone() for rank in logits]
        if extra_fields is not None:
            additional = extra_fields(gen, kind, kwargs)
            if values.keys() & additional.keys():
                raise AssertionError("Diagnostic extra fields must have unique names")
            values.update(additional)
        metadata = dict(
            run=run,
            history=("C_singleton", "A_extra_singleton")[run // 2],
            sequence=sequence,
            kind=kind,
            positions=[cache.position for cache in gen.caches],
            live_slots=live,
            feedback_valid_slots=feedback_valid,
            sampled_slots=sampled,
            captures=gen.counters["captures"],
            request_seeds={slot: gen.request_seeds[slot] for slot in feedback_valid},
        )
        report["observations"].append(metadata)
        if run % 2 == 0:
            baseline[sequence] = (metadata, values)
            return
        expected_meta, expected = baseline[sequence]
        comparison = dict(
            **metadata,
            metadata_equal=all(
                metadata[key] == expected_meta[key]
                for key in ("kind", "positions", "live_slots", "feedback_valid_slots", "sampled_slots", "request_seeds")
            ),
            fields_equal=expected.keys() == values.keys(),
            fields={},
        )
        for name in sorted(expected.keys() & values.keys()):
            for rank, (left, right) in enumerate(zip(expected[name], values[name], strict=True)):
                field = f"{name}.rank{rank}"
                result = compare_tensor(left, right, torch)
                comparison["fields"][field] = result
                if not result["equal"] and report["first_difference"] is None:
                    report["first_difference"] = dict(**metadata, field=field, comparison=result)
                    if args.save_first_diff:
                        path = state_output.with_suffix(".first_difference.pt")
                        path.parent.mkdir(parents=True, exist_ok=True)
                        torch.save(dict(expected=left, actual=right, metadata=report["first_difference"]), path)
                        report["first_difference"]["tensor_artifact"] = str(path)
        comparison["equal"] = (
            comparison["metadata_equal"]
            and comparison["fields_equal"]
            and all(v["equal"] for v in comparison["fields"].values())
        )
        report["comparisons"].append(comparison)
        print(
            "STATE_DETERMINISM "
            + json.dumps(
                dict(
                    run=run,
                    sequence=sequence,
                    kind=kind,
                    positions=metadata["positions"],
                    equal=comparison["equal"],
                    differing_fields=[key for key, value in comparison["fields"].items() if not value["equal"]],
                )
            ),
            flush=True,
        )

    def prefill(adapter, *positional, **kwargs):
        output = original_prefill(adapter, *positional, **kwargs)
        observe(adapter, "prefill", kwargs)
        return output

    def decode(adapter, *positional, **kwargs):
        output = original_decode(adapter, *positional, **kwargs)
        observe(adapter, "decode", kwargs)
        return output

    cls.prefill_forward, cls.decode_forward, cls.release_request = prefill, decode, release
    try:
        sys.argv = ["probe_vllm_async_transitions", "--output", str(args.output), "--read-order", "immediate-twice"]
        runpy.run_module(
            "models.autoports.qwen_qwen3_8_flash_next.tests.probe_vllm_async_transitions", run_name="__main__"
        )
        report["original_oracle_completed"] = True
        report["passed"] = bool(report["comparisons"]) and all(row["equal"] for row in report["comparisons"])
    finally:
        cls.prefill_forward, cls.decode_forward = original_prefill, original_decode
        if original_release is None:
            del cls.release_request
        else:
            cls.release_request = original_release
        sys.argv = original_argv
        report["seconds"] = time.monotonic() - started
        state_output.parent.mkdir(parents=True, exist_ok=True)
        state_output.write_text(json.dumps(report, indent=2) + "\n")
    if not report["passed"]:
        raise AssertionError(f"State differs; see {state_output}")


if __name__ == "__main__":
    main()
