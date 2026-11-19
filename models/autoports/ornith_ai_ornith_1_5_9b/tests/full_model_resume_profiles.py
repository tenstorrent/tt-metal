# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Diagnostic-only request-canonical physical profiles for resumed prefill.

Wrap the unmodified readiness runner; its comparisons and controls remain intact.
Original reference prefills are unchanged unless --normalized-reference is set.
By default only aligned resume requests of length <=1024 use the candidate.
--aligned-overlap retains at most31 per-layer input rows plus a linear state
snapshot at the previous tile boundary, then recomputes that bounded overlap.
Without that option unaligned requests run the original path as uncovered.

This experiment has eager host page-table readback and variable logical crops.
It tests numerical partition effects, NOT trace safety or a production API.
It neither changes precision configuration nor substitutes a decode loop.
"""

import argparse
import json
from pathlib import Path
from types import MethodType

import torch

import ttnn
from models.autoports.ornith_ai_ornith_1_5_9b.tests.full_model_normalized_attention import (
    FULL_PREFILL_BUCKETS,
    NormalizedAttentionPrefill,
)
from models.autoports.ornith_ai_ornith_1_5_9b.tests.full_model_normalized_linear import (
    NormalizedLinearPrefill,
    select_bucket,
)


class RequestProfileLinear(NormalizedLinearPrefill):
    """Use the request's physical bucket with the segment's actual valid rows."""

    def refresh_profile(self, *, valid_count, bucket):
        if not 0 < valid_count <= bucket or bucket not in self.profiles:
            raise ValueError("Invalid canonical profile segment")
        profile = self.profiles[bucket]
        mask = (torch.arange(bucket) < valid_count).float().reshape(1, bucket, 1).expand(1, bucket, 8).contiguous()
        for name, value, dtype, layout in (
            ("g_mask", mask, ttnn.float32, ttnn.TILE_LAYOUT),
            ("beta_mask", mask, ttnn.bfloat16, ttnn.TILE_LAYOUT),
            ("history_indices", self.history_indices(valid_count), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT),
        ):
            host = self.upload(value, dtype=dtype, layout=layout, device=False)
            ttnn.copy_host_to_device_tensor(host, profile[name])


def install_generator(generator, *, normalized_reference, aligned_overlap=False, report):
    """Patch the actually constructed instance, including dynamic runner imports."""
    model = generator.model
    original_continuation = generator.prefill_logits_continuation
    original_prefill = model.prefill
    adapters = {}
    overlap_tails = {}
    model._diagnostic_request_profile = None

    def canonical_prefill(self, tokens, *, cache, slot, start, page_table, last_only=False):
        total = self._diagnostic_request_profile
        if total is None:
            return original_prefill(
                tokens, cache=cache, slot=slot, start=start, page_table=page_table, last_only=last_only
            )
        valid = tokens.shape[-1]
        if (start % 32 and not aligned_overlap) or not 0 < valid <= total <= 1024 or start + valid > total:
            raise ValueError(f"Unsupported diagnostic segment start={start}, valid={valid}, total={total}")
        # Explicit diagnostic host boundary: preserve the actual caller mapping.
        # No synthetic identity map is substituted for a supplied page table.
        mapping = ttnn.to_torch(ttnn.get_device_tensors(page_table)[0]).reshape(-1).tolist()
        context = len(mapping) * 32
        hidden = self.embed(tokens)
        layer_profiles = []
        for layer, states in zip(self.layers, cache):
            state = states[slot]
            bucket = (
                select_bucket(total)
                if layer.kind == "linear_attention"
                else next(n for n in FULL_PREFILL_BUCKETS if n >= ((total + 31) // 32) * 32)
            )
            key = (id(layer), id(state), bucket, context)
            if key not in adapters:
                adapters[key] = (
                    RequestProfileLinear(layer, buckets=(bucket,))
                    if layer.kind == "linear_attention"
                    else NormalizedAttentionPrefill(
                        layer, state=state, physical_bucket=bucket, supported_context=context
                    )
                )
            adapter = adapters[key]

            def segment_forward(inputs, absolute_start, *, bucket=bucket, layer=layer, adapter=adapter, state=state):
                count = inputs.shape[2]
                assert absolute_start % 32 == 0 and 0 < count <= bucket
                physical = (
                    ttnn.pad(inputs, [(0, 0), (0, 0), (0, bucket - count), (0, 0)], 0.0) if bucket != count else inputs
                )
                if layer.kind == "linear_attention":
                    adapter.refresh_profile(valid_count=count, bucket=bucket)
                    physical = adapter.forward(physical, state=state)
                else:
                    adapter.refresh(start=absolute_start, valid_count=count, page_mapping=mapping)
                    physical = adapter.forward(physical)
                return physical[:, :, :count, :]

            action = "ordinary"
            overlap = 0
            if aligned_overlap and start == 0 and valid < total and valid % 32:
                # Keep layer-input rows, not the output of this layer: a resumed
                # invocation must recompute exactly the same overlap contraction.
                boundary = valid // 32 * 32
                overlap = valid - boundary
                layer_input = hidden
                prefix_output = segment_forward(layer_input[:, :, :boundary, :], 0) if boundary else None
                saved = {
                    "boundary": boundary,
                    "end": valid,
                    "inputs": ttnn.clone(layer_input[:, :, boundary:valid, :]),
                }
                if layer.kind == "linear_attention":
                    saved["conv"] = ttnn.clone(layer.read_linear_state(state, "conv"))
                    saved["recurrent"] = ttnn.clone(layer.read_linear_state(state, "recurrent"))
                overlap_tails[key] = saved
                tail_output = segment_forward(saved["inputs"], boundary)
                hidden = ttnn.concat([prefix_output, tail_output], dim=2) if boundary else tail_output
                action = "save_boundary_state_and_input_tail"
            elif start % 32:
                # No-history controls clear this dictionary before admission and
                # use the original path. Never borrow tails from another request.
                saved = overlap_tails[key]
                assert saved["end"] == start and saved["boundary"] == start // 32 * 32
                overlap = start - saved["boundary"]
                assert 0 < overlap <= 31 and saved["inputs"].shape[2] == overlap
                inputs = ttnn.concat([saved["inputs"], hidden], dim=2)
                if layer.kind == "linear_attention":
                    layer.write_linear_state(state, "conv", saved["conv"])
                    layer.write_linear_state(state, "recurrent", saved["recurrent"])
                # Full attention rewrites only the same overlap K/V using saved
                # layer inputs. Prefix pages before the boundary remain untouched.
                physical = segment_forward(inputs, saved["boundary"])
                hidden = physical[:, :, overlap : overlap + valid, :]
                action = "restore_boundary_and_recompute_overlap"
            else:
                hidden = segment_forward(hidden, start)
            layer_profiles.append({"kind": layer.kind, "bucket": bucket, "action": action, "overlap": overlap})
        report["normalized_model_calls"] += 1
        event = {"total": total, "start": start, "valid": valid, "profiles": layer_profiles}
        report["segments"].append(event)
        print("CANONICAL_PROFILE_SEGMENT " + json.dumps(event), flush=True)
        if last_only:
            hidden = hidden[:, :, -1:, :]
        return self.logits(hidden)

    def canonical_continuation(self, prompt_token_ids, start, *, positions=None, prefill_prefix=True):
        total = len(prompt_token_ids)
        if not 0 < total <= 1024:
            raise ValueError("The request-canonical profile experiment is limited to logical lengths <=1024")
        unaligned = bool(start % 32)
        # These are per-request snapshots. Even the negative control begins with
        # an empty dictionary; original_continuation also resets its device cache.
        overlap_tails.clear()
        overlap_candidate = unaligned and aligned_overlap and prefill_prefix
        active = (not unaligned or overlap_candidate) and (start != 0 or normalized_reference)
        if unaligned and not active:
            reason = (
                "no_history_original_no_saved_tail"
                if aligned_overlap and not prefill_prefix
                else "unaligned_not_covered"
            )
        else:
            reason = "overlap_candidate" if overlap_candidate else ("candidate" if active else "original_reference")
        event = {
            "total": total,
            "start": start,
            "prefill_prefix": prefill_prefix,
            "path": "normalized" if active else "original",
            "reason": reason,
        }
        report["requests"].append(event)
        report["activations"] += int(active)
        print("CANONICAL_PROFILE_REQUEST " + json.dumps(event), flush=True)
        previous = model._diagnostic_request_profile
        model._diagnostic_request_profile = total if active else None
        try:
            return original_continuation(prompt_token_ids, start, positions=positions, prefill_prefix=prefill_prefix)
        finally:
            model._diagnostic_request_profile = previous
            overlap_tails.clear()

    model.prefill = MethodType(canonical_prefill, model)
    generator.prefill_logits_continuation = MethodType(canonical_continuation, generator)
    report["generator_instances_patched"] += 1
    return generator


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--normalized-reference", action="store_true")
    parser.add_argument("--aligned-overlap", action="store_true")
    parser.add_argument("--profile-report", type=Path)
    args, rest = parser.parse_known_args()
    runner_args = argparse.ArgumentParser(add_help=False)
    runner_args.add_argument("--output", type=Path)
    runner_args.add_argument("--model-dir", type=Path, required=True)
    destinations, _ = runner_args.parse_known_args(rest)
    output = destinations.output or destinations.model_dir / "readiness/prefill_resume_check.json"
    report_path = args.profile_report or output.with_suffix(".profiles.json")
    report = {
        "diagnostic_only": True,
        "normalized_reference": args.normalized_reference,
        "aligned_overlap": args.aligned_overlap,
        "generator_instances_patched": 0,
        "activations": 0,
        "normalized_model_calls": 0,
        "limitations": [
            "Logical lengths >1024 are not covered by this diagnostic.",
            "Without --aligned-overlap, unaligned requests retain original decoder behavior.",
            "Unaligned no-history controls use the original empty-cache path and never access saved tails.",
            "Eager adapter allocation, page-table readback and variable crops are not trace-safety evidence.",
            "Canonical physical profiles are diagnostic; preserving original policy knobs does not establish equivalence.",
        ],
        "requests": [],
        "segments": [],
    }
    from readiness_check import run_prefill_resume_check as runner

    original_import = runner._import_build_generator

    def import_wrapped_builder(model_dir):
        builder = original_import(model_dir)

        def build(*positional, **kwargs):
            return install_generator(
                builder(*positional, **kwargs),
                normalized_reference=args.normalized_reference,
                aligned_overlap=args.aligned_overlap,
                report=report,
            )

        return build

    runner._import_build_generator = import_wrapped_builder
    try:
        status = runner._main(rest)
        assert report["activations"] > 0 and report["normalized_model_calls"] > 0, "Diagnostic did not activate"
        raise SystemExit(status)
    finally:
        runner._import_build_generator = original_import
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        print(f"CANONICAL_PROFILE_REPORT {report_path} activations={report['activations']}", flush=True)


if __name__ == "__main__":
    main()
