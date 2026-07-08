#!/usr/bin/env python3
"""Annotate a pi0.5 ops_perf_results CSV — v3: clean single-inference annotation.

Adds STAGE / LAYER / SUBSTAGE columns AND separates init from prefix_setup
even in single-inference CSVs (where N=1 makes positional separation
impossible). Uses op-signature heuristics for init detection.

Heuristics for INIT detection in the pre-SDPA-#1 region:
  - ConcatDeviceOperation with output X in (2560, 4608) and ≥3 inputs
    → QKV-weight-fusion concat (SigLIP, VLM, Expert)
  - Any op with FLOAT32 input or output dtype
    → RoPE inv-freq / cos / sin table builders (these are FP32 by design)

These cover ~95% of init ops at pi0.5's typical model load. Anything not
matched in the pre-SDPA-#1 region is labeled "prefix_setup" (real per-call).

For multi-inference CSVs (N >= 2), reuses v2 behavior: init is positional
(rows before SDPA #1) and per-call breakdown comes from the last inference.

Stage layout per inference (1-based SDPA #'s):
  SDPA #1..27   → siglip          27 layers
  SDPA #28..45  → vlm_prefill     18 layers
  SDPA #46..63  → denoise_step_1  18 layers
  ... → denoise_step_5            18 layers
"""

import csv
import re
import sys
from pathlib import Path

SDPA_PER_INFERENCE = 135


def _logical(s: str):
    if not s:
        return None
    m = re.match(r"^(\d+)", s)
    return int(m.group(1)) if m else None


def _is_init_by_signature(row, header_idx) -> bool:
    """Return True if this row matches an init op signature (QKV fusion or FP32 RoPE)."""
    op = row[header_idx["OP CODE"]]
    out0_x = _logical(row[header_idx["OUTPUT_0_X_PAD[LOGICAL]"]])
    in0_dtype = row[header_idx["INPUT_0_DATATYPE"]]
    out0_dtype = row[header_idx["OUTPUT_0_DATATYPE"]]
    in2_x = _logical(row[header_idx["INPUT_2_X_PAD[LOGICAL]"]]) if "INPUT_2_X_PAD[LOGICAL]" in header_idx else None
    # QKV-fusion concat: 3 inputs concat'd into width 2560 (Gemma) or 4608 (SigLIP)
    if op == "ConcatDeviceOperation" and in2_x is not None and out0_x in (2560, 4608):
        return True
    # FP32 ops: RoPE table builders (FILL(10000), POWER, RECIP, COS, SIN, MUL with scalar, etc.)
    if in0_dtype == "FLOAT32" or out0_dtype == "FLOAT32":
        return True
    return False


def annotate(in_csv: str, out_csv: str):
    with open(in_csv) as f:
        rows = list(csv.reader(f))
    header = rows[0]
    body = rows[1:]
    header_idx = {h: i for i, h in enumerate(header)}
    op_col = header_idx["OP CODE"]
    kd_col = header_idx["DEVICE KERNEL DURATION [ns]"]

    sdpa_idx = [i for i, r in enumerate(body) if r[op_col] == "SDPAOperation"]
    total_sdpa = len(sdpa_idx)
    assert (
        total_sdpa % SDPA_PER_INFERENCE == 0
    ), f"expected SDPA count divisible by {SDPA_PER_INFERENCE}; got {total_sdpa}"
    n_inferences = total_sdpa // SDPA_PER_INFERENCE

    stages = [
        ("siglip", 0, 26, 27),
        ("vlm_prefill", 27, 44, 18),
        ("denoise_step_1", 45, 62, 18),
        ("denoise_step_2", 63, 80, 18),
        ("denoise_step_3", 81, 98, 18),
        ("denoise_step_4", 99, 116, 18),
        ("denoise_step_5", 117, 134, 18),
    ]

    stage_col = [""] * len(body)
    layer_col = [""] * len(body)
    sub_col = [""] * len(body)

    def last_ln_before(sdpa_row, lower_bound):
        last_ln = None
        for j in range(lower_bound, sdpa_row):
            if body[j][op_col] == "LayerNormDeviceOperation":
                last_ln = j
        if last_ln is None:
            return lower_bound
        if last_ln > 0 and body[last_ln - 1][op_col] == "InterleavedToShardedDeviceOperation":
            return last_ln - 1
        return last_ln

    def annotate_inference(inf_num, suffix=""):
        offs = SDPA_PER_INFERENCE * (inf_num - 1)
        inf_start = 0 if inf_num == 1 else sdpa_idx[offs - 1] + 1

        boundaries = []
        prev_end = inf_start
        for stage_label, s_off, e_off, n_layers in stages:
            first_sdpa = sdpa_idx[offs + s_off]
            stage_start = last_ln_before(first_sdpa, prev_end)
            boundaries.append((stage_label, stage_start, s_off, e_off, n_layers))
            prev_end = sdpa_idx[offs + e_off] + 1

        last_sdpa = sdpa_idx[offs + 134]
        ln_seen = 0
        p_start = last_sdpa + 1
        for j in range(last_sdpa + 1, len(body)):
            if body[j][op_col] == "LayerNormDeviceOperation":
                ln_seen += 1
                if ln_seen == 2:
                    p_start = j - 1 if j > 0 and body[j - 1][op_col] == "InterleavedToShardedDeviceOperation" else j
                    break

        if inf_num < n_inferences:
            next_offs = SDPA_PER_INFERENCE * inf_num
            inf_end = last_ln_before(sdpa_idx[next_offs + 0], sdpa_idx[next_offs - 1] + 1)
        else:
            inf_end = len(body)

        prefix_start = inf_start
        prefix_end = boundaries[0][1]
        # Label prefix region — for inference 1 of single-inference CSV, we'll
        # OVERRIDE these labels with the signature-based init detection below.
        for i in range(prefix_start, prefix_end):
            stage_col[i] = "prefix_setup" + suffix

        per_stage = {}
        for stage_label, stage_start, s_off, e_off, n_layers in boundaries:
            idx_in = next(i for i, b in enumerate(boundaries) if b[0] == stage_label)
            stage_end = boundaries[idx_in + 1][1] if idx_in + 1 < len(boundaries) else p_start
            layer_start = stage_start
            for li in range(n_layers):
                layer_sdpa = sdpa_idx[offs + s_off + li]
                if li + 1 < n_layers:
                    next_layer_sdpa = sdpa_idx[offs + s_off + li + 1]
                    next_attn_start = last_ln_before(next_layer_sdpa, layer_sdpa + 1)
                else:
                    next_attn_start = stage_end
                for j in range(layer_start, layer_sdpa + 1):
                    stage_col[j] = stage_label + suffix
                    layer_col[j] = str(li + 1)
                    sub_col[j] = "attn"
                for j in range(layer_sdpa + 1, next_attn_start):
                    stage_col[j] = stage_label + suffix
                    layer_col[j] = str(li + 1)
                    sub_col[j] = "mlp"
                layer_start = next_attn_start
            per_stage[stage_label] = (stage_start, stage_end)

        for i in range(p_start, inf_end):
            stage_col[i] = "project_output" + suffix

        ms = {
            "prefix_setup": (
                prefix_start,
                prefix_end,
                sum(float(body[i][kd_col] or 0) for i in range(prefix_start, prefix_end)) / 1e6,
            )
        }
        for stage_label, (s, e) in per_stage.items():
            ms[stage_label] = (s, e, sum(float(body[i][kd_col] or 0) for i in range(s, e)) / 1e6)
        ms["project_output"] = (
            p_start,
            inf_end,
            sum(float(body[i][kd_col] or 0) for i in range(p_start, inf_end)) / 1e6,
        )
        return ms

    # === Annotation pass ===
    if n_inferences > 1:
        # Multi-inference: use positional init separation (v2 behavior)
        inf_breakdowns = {}
        for k in range(1, n_inferences + 1):
            if k == n_inferences:
                lbl = ""
            elif k == n_inferences - 1:
                lbl = "_trace_capture"
            else:
                lbl = f"_warmup{k}"
            inf_breakdowns[k] = annotate_inference(k, lbl)
        # Re-label inference 1's prefix_setup region as init_one_time
        end_inf1 = inf_breakdowns[1]["prefix_setup"][1]
        for i in range(0, end_inf1):
            stage_col[i] = "init_one_time"
        canonical = inf_breakdowns[n_inferences]
        init_count = end_inf1
        init_ms_val = sum(float(body[i][kd_col] or 0) for i in range(0, end_inf1)) / 1e6
        canonical_label = f"Inference {n_inferences} (canonical = trace replay/last)"
    else:
        # Single-inference: detect init region as a CONTIGUOUS BLOCK from row 0
        # to the LAST init-signature op. Everything after = real per-call work.
        # This respects chronological order: init runs once at model construction
        # (rows 0..N), then sample_actions begins (rows N+1..prefix_end).
        single = annotate_inference(1, "")
        prefix_end = single["prefix_setup"][1]
        last_init_row = -1
        for i in range(0, prefix_end):
            if _is_init_by_signature(body[i], header_idx):
                last_init_row = i
        init_count = last_init_row + 1  # rows 0..last_init_row (inclusive)
        init_ms_val = (
            sum(float(body[i][kd_col] or 0) for i in range(0, last_init_row + 1)) / 1e6 if last_init_row >= 0 else 0.0
        )
        real_count = prefix_end - (last_init_row + 1)
        real_ms_val = sum(float(body[i][kd_col] or 0) for i in range(last_init_row + 1, prefix_end)) / 1e6
        for i in range(0, last_init_row + 1):
            stage_col[i] = "init_one_time"
        for i in range(last_init_row + 1, prefix_end):
            stage_col[i] = "prefix_setup"
        # Patch the breakdown to reflect the heuristic split
        single["prefix_setup"] = (last_init_row + 1, prefix_end, real_ms_val)
        canonical = single
        canonical_label = "Inference 1 (single-inference CSV; init/prefix split at last init-signature op)"

    # === Write annotated CSV ===
    new_header = ["STAGE", "LAYER", "SUBSTAGE"] + header
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(new_header)
        for i, row in enumerate(body):
            w.writerow([stage_col[i], layer_col[i], sub_col[i]] + row)

    # === Print report ===
    print(f"Wrote {out_csv}")
    print(f"Total rows: {len(body)}")
    print(f"Inferences detected: {n_inferences}")
    print()

    print("=== Init (ONE-TIME setup; runs once at model construction) ===")
    print(f"  {init_count} ops, kernel sum {init_ms_val:.3f} ms")
    if n_inferences == 1:
        print("  (detected by op-signature heuristic in single-inference CSV)")
    print()

    print(f"=== {canonical_label} — per-call breakdown ===")
    stage_order = [
        "prefix_setup",
        "siglip",
        "vlm_prefill",
        "denoise_step_1",
        "denoise_step_2",
        "denoise_step_3",
        "denoise_step_4",
        "denoise_step_5",
        "project_output",
    ]
    print(f"  {'STAGE':<18} {'rows':<22} {'count':>6}  {'kernel_ms':>10}")
    print("  " + "-" * 64)
    total_ms = 0.0
    for s in stage_order:
        if s not in canonical:
            continue
        start, end, ms = canonical[s]
        n = end - start
        total_ms += ms
        # For multi-inference, the rows are still raw byte offsets; for single-inf
        # prefix_setup, we keep the same range but the count includes only ones
        # we labeled prefix_setup (heuristic-classified). Report effective counts.
        if n_inferences == 1 and s == "prefix_setup":
            # count only rows labeled "prefix_setup"
            n_real = sum(1 for j in range(start, end) if stage_col[j] == "prefix_setup")
            print(f"  {s:<18} {start + 2:5d}..{end + 1:5d}    {n_real:>6}  {ms:>9.3f}")
        else:
            print(f"  {s:<18} {start + 2:5d}..{end + 1:5d}    {n:>6}  {ms:>9.3f}")
    print("  " + "-" * 64)
    note = "  ← EXCLUDES init"
    print(f"  {'TOTAL/inference':<18} {'':<22} {'':>6}  {total_ms:>9.3f}{note}")
    return {
        "init_ms": init_ms_val,
        "init_count": init_count,
        "breakdown": canonical,
        "n_inferences": n_inferences,
    }


if __name__ == "__main__":
    in_csv = sys.argv[1] if len(sys.argv) > 1 else None
    if in_csv is None:
        print("usage: annotate_ops_csv_v3.py <input.csv> [output.csv]", file=sys.stderr)
        sys.exit(1)
    out_csv = sys.argv[2] if len(sys.argv) > 2 else str(Path(in_csv).with_name(Path(in_csv).stem + "_annotated_v3.csv"))
    annotate(in_csv, out_csv)
