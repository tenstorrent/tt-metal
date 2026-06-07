#!/usr/bin/env python3
"""Annotate a pi0.5 ops_perf_results CSV — v2: handles multi-inference tracy CSVs.

Adds three columns: STAGE / LAYER / SUBSTAGE.

Differences vs v1:
  - Auto-detects N inferences from total SDPA count (135 SDPAs / inference).
  - For multi-inference CSVs, the LAST inference (= trace replay) is the
    canonical per-call breakdown. Earlier inferences (warmup, capture) are
    tagged but the report focuses on the last.
  - Init ops (one-time setup; rows before SDPA #1 in inference 1) are
    labeled `init_one_time` and reported SEPARATELY from `prefix_setup`.
  - Uses the same "last LN before next SDPA" boundary heuristic as v1.

Stage layout per inference:
  SDPA #1..27   → siglip          27 layers
  SDPA #28..45  → vlm_prefill     18 layers
  SDPA #46..63  → denoise_step_1  18 layers
  SDPA #64..81  → denoise_step_2  18 layers
  SDPA #82..99  → denoise_step_3  18 layers
  SDPA #100..117→ denoise_step_4  18 layers
  SDPA #118..135→ denoise_step_5  18 layers
"""

import csv
import sys
from pathlib import Path

SDPA_PER_INFERENCE = 135


def annotate(in_csv: str, out_csv: str):
    with open(in_csv) as f:
        rows = list(csv.reader(f))
    header = rows[0]
    body = rows[1:]
    op_col = header.index("OP CODE")
    kd_col = header.index("DEVICE KERNEL DURATION [ns]")

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

    def last_ln_before(sdpa_row: int, lower_bound: int) -> int:
        last_ln = None
        for j in range(lower_bound, sdpa_row):
            if body[j][op_col] == "LayerNormDeviceOperation":
                last_ln = j
        if last_ln is None:
            return lower_bound
        if last_ln > 0 and body[last_ln - 1][op_col] == "InterleavedToShardedDeviceOperation":
            return last_ln - 1
        return last_ln

    def annotate_inference(inf_num: int, inf_label_suffix: str = "") -> dict:
        """Tag stages for one inference. Returns {stage_name: (start, end, ms)}."""
        offs = SDPA_PER_INFERENCE * (inf_num - 1)
        # inference start = (previous inference's last SDPA + 1) or 0 for inference 1
        inf_start = 0 if inf_num == 1 else sdpa_idx[offs - 1] + 1

        # Per-stage boundaries (start of each stage's first layer's attn)
        boundaries = []
        prev_end = inf_start
        for stage_label, s_off, e_off, n_layers in stages:
            first_sdpa_in_stage = sdpa_idx[offs + s_off]
            stage_start = last_ln_before(first_sdpa_in_stage, prev_end)
            boundaries.append((stage_label, stage_start, s_off, e_off, n_layers))
            # next stage's lower bound is the LAST SDPA of this stage + 1
            prev_end = sdpa_idx[offs + e_off] + 1

        # project_output: 2nd LN after the last denoise SDPA in this inference
        last_sdpa_idx_inf = sdpa_idx[offs + 134]
        ln_seen = 0
        p_start = last_sdpa_idx_inf + 1
        for j in range(last_sdpa_idx_inf + 1, len(body)):
            if body[j][op_col] == "LayerNormDeviceOperation":
                ln_seen += 1
                if ln_seen == 2:
                    p_start = j - 1 if j > 0 and body[j - 1][op_col] == "InterleavedToShardedDeviceOperation" else j
                    break

        # End of this inference: next inference's first stage start, or end of body
        if inf_num < n_inferences:
            next_offs = SDPA_PER_INFERENCE * inf_num
            inf_end = last_ln_before(sdpa_idx[next_offs + 0], sdpa_idx[next_offs - 1] + 1)
        else:
            inf_end = len(body)

        # Compute prefix_setup region for this inference.
        # For inference 1: prefix_setup = inf_start .. boundaries[0][1] - 1
        # But "init_one_time" subset for inference 1 is captured separately by caller.
        # Tag rows.
        # 1. prefix_setup
        prefix_start = inf_start
        prefix_end = boundaries[0][1]
        for i in range(prefix_start, prefix_end):
            stage_col[i] = "prefix_setup" + inf_label_suffix
        # 2. each stage's layers
        per_stage_ranges = {}
        for stage_label, stage_start, s_off, e_off, n_layers in boundaries:
            # Next stage's start (or p_start for the last stage)
            stage_idx_in_list = next(i for i, b in enumerate(boundaries) if b[0] == stage_label)
            stage_end = boundaries[stage_idx_in_list + 1][1] if stage_idx_in_list + 1 < len(boundaries) else p_start

            # Per-layer: attn = pre-attn LN .. SDPA; mlp = SDPA+1 .. next layer attn start (or stage_end)
            layer_start = stage_start
            for li in range(n_layers):
                layer_sdpa = sdpa_idx[offs + s_off + li]
                # Next layer attn start within this stage:
                if li + 1 < n_layers:
                    next_layer_sdpa = sdpa_idx[offs + s_off + li + 1]
                    next_layer_attn_start = last_ln_before(next_layer_sdpa, layer_sdpa + 1)
                else:
                    next_layer_attn_start = stage_end

                # Tag attn rows
                for j in range(layer_start, layer_sdpa + 1):
                    stage_col[j] = stage_label + inf_label_suffix
                    layer_col[j] = str(li + 1)
                    sub_col[j] = "attn"
                # Tag mlp rows
                for j in range(layer_sdpa + 1, next_layer_attn_start):
                    stage_col[j] = stage_label + inf_label_suffix
                    layer_col[j] = str(li + 1)
                    sub_col[j] = "mlp"

                layer_start = next_layer_attn_start

            per_stage_ranges[stage_label] = (stage_start, stage_end)

        # 3. project_output
        for i in range(p_start, inf_end):
            stage_col[i] = "project_output" + inf_label_suffix

        # Compute per-stage ms
        stage_ms = {}
        # prefix_setup
        stage_ms["prefix_setup"] = (
            prefix_start,
            prefix_end,
            sum(float(body[i][kd_col] or 0) for i in range(prefix_start, prefix_end)) / 1e6,
        )
        for stage_label, (s, e) in per_stage_ranges.items():
            stage_ms[stage_label] = (s, e, sum(float(body[i][kd_col] or 0) for i in range(s, e)) / 1e6)
        stage_ms["project_output"] = (
            p_start,
            inf_end,
            sum(float(body[i][kd_col] or 0) for i in range(p_start, inf_end)) / 1e6,
        )
        return stage_ms

    # === Annotation pass ===
    # For multi-inference CSVs, separate init from inference 1's prefix_setup.
    # Init = rows 0 .. (siglip_start_of_inference_1 - 1) DROPPED FROM PREFIX_SETUP,
    # and labeled "init_one_time".
    if n_inferences > 1:
        # Annotate each inference with a suffix tag.
        inf_breakdowns = {}
        for k in range(1, n_inferences + 1):
            if k == n_inferences:
                label = ""  # last inference = canonical (trace replay)
            elif k == n_inferences - 1:
                label = "_trace_capture"  # second to last = trace capture
            else:
                label = f"_warmup{k}"  # earlier = warmups
            inf_breakdowns[k] = annotate_inference(k, label)

        # For inference 1, re-tag the prefix_setup as init_one_time (it's mixed; we treat
        # the whole region before siglip as init).
        # Find prefix_setup rows of inference 1 and rename their STAGE label.
        inf1_prefix_start = 0
        inf1_prefix_end = inf_breakdowns[1]["prefix_setup"][1]
        for i in range(inf1_prefix_start, inf1_prefix_end):
            stage_col[i] = "init_one_time"

        canonical_breakdown = inf_breakdowns[n_inferences]
    else:
        # Single-inference CSV: keep v1 behavior, but flag prefix_setup as mixed.
        single = annotate_inference(1, "")
        # Rename inference 1's prefix_setup to flag the contamination
        inf1_prefix_start = 0
        inf1_prefix_end = single["prefix_setup"][1]
        # We KEEP "prefix_setup" label but warn
        canonical_breakdown = single

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
    print(f"Inferences detected: {n_inferences}  (135 SDPAs / inference; {total_sdpa} SDPAs total)")
    print()

    # Init region (only for multi-inference CSVs)
    init_ms = 0.0
    init_count = 0
    if n_inferences > 1:
        init_ms = sum(float(body[i][kd_col] or 0) for i in range(0, inf_breakdowns[1]["prefix_setup"][1])) / 1e6
        init_count = inf_breakdowns[1]["prefix_setup"][1]
        print(f"=== Init (ONE-TIME setup; appears once at model construction) ===")
        print(f"  {init_count} ops, kernel sum {init_ms:.3f} ms")
        print()
    else:
        print(f"=== Single-inference CSV — prefix_setup MAY include init contamination ===")
        print()

    # Canonical per-inference breakdown
    if n_inferences > 1:
        canon_name = "Inference %d (trace REPLAY)" % n_inferences
    else:
        canon_name = "Inference 1 (single-inference CSV)"
    print(f"=== {canon_name} — canonical per-call breakdown ===")
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
        if s not in canonical_breakdown:
            continue
        start, end, ms = canonical_breakdown[s]
        n = end - start
        total_ms += ms
        # +2 to convert to CSV 1-based row (header is row 1)
        print(f"  {s:<18} {start + 2:5d}..{end + 1:5d}    {n:>6}  {ms:>9.3f}")
    print("  " + "-" * 64)
    note = "  ← EXCLUDES init" if n_inferences > 1 else ""
    print(f"  {'TOTAL/inference':<18} {'':<22} {'':>6}  {total_ms:>9.3f}{note}")

    return {
        "init_ms": init_ms,
        "init_count": init_count,
        "breakdown": canonical_breakdown,
        "n_inferences": n_inferences,
    }


if __name__ == "__main__":
    in_csv = sys.argv[1] if len(sys.argv) > 1 else None
    if in_csv is None:
        print("usage: annotate_ops_csv_v2.py <input.csv> [output.csv]", file=sys.stderr)
        sys.exit(1)
    out_csv = sys.argv[2] if len(sys.argv) > 2 else str(Path(in_csv).with_name(Path(in_csv).stem + "_annotated.csv"))
    annotate(in_csv, out_csv)
