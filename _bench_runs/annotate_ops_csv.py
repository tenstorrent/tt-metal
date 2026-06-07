#!/usr/bin/env python3
"""Annotate a pi0.5 ops_perf_results CSV with STAGE / LAYER / SUBSTAGE columns.

Stage segmentation is anchored on SDPAOperation calls:
  SDPA #1..27   → SigLIP (Sq=Skv=256, bs=3)            27 layers
  SDPA #28..45  → VLM prefill (Sq=Skv=1024)            18 layers
  SDPA #46..63  → Denoise step 1 (Sq=32, Skv=1056)     18 layers
  SDPA #64..81  → Denoise step 2                       18 layers
  SDPA #82..99  → Denoise step 3                       18 layers
  SDPA #100..117→ Denoise step 4                       18 layers
  SDPA #118..135→ Denoise step 5                       18 layers
  rows before SDPA #1   → prefix_setup (embeddings, patch_conv, etc.)
  rows after SDPA #135  → project_output

Within a layer:
  Up to and including the SDPA      → SUBSTAGE = "attn"  (LN, QKV, RoPE, SDPA, concat, o_proj, residual_add)
  Between SDPA and next layer LN    → SUBSTAGE = "mlp"   (LN, gate/up, multiply, down, residual_add)

Output adds three columns: STAGE, LAYER (1..N within stage), SUBSTAGE.
"""

import csv
import sys
from pathlib import Path


def annotate(in_csv: str, out_csv: str):
    with open(in_csv) as f:
        rows = list(csv.reader(f))
    header = rows[0]
    body = rows[1:]

    # First pass: locate SDPA row indices and pre-tag stage/layer
    sdpa_idx = [i for i, r in enumerate(body) if r[0] == "SDPAOperation"]
    assert len(sdpa_idx) == 135, f"expected 135 SDPAs, got {len(sdpa_idx)}"

    # Stage definition: (label, start_sdpa#1based, end_sdpa#1based, layers_in_stage)
    stages = [
        ("siglip", 1, 27, 27),
        ("vlm_prefill", 28, 45, 18),
        ("denoise_step_1", 46, 63, 18),
        ("denoise_step_2", 64, 81, 18),
        ("denoise_step_3", 82, 99, 18),
        ("denoise_step_4", 100, 117, 18),
        ("denoise_step_5", 118, 135, 18),
    ]

    # Build per-row stage/layer/substage
    stage_col = [""] * len(body)
    layer_col = [""] * len(body)
    sub_col = [""] * len(body)

    # The pre-attn LN of a layer is the LAST LN before its SDPA in the search
    # range. This works for within-stage transitions (where between SDPAs we
    # see 2 LNs: prev mlp LN + curr attn LN) AND cross-stage transitions
    # (3+ LNs because of stage-final norms like vlm_norm — but the LAST one
    # is still the next layer's pre-attn LN).
    def pre_attn_ln_for(sdpa_body_idx: int, lower_bound: int) -> int:
        """Returns the row of the boundary where this layer's attn begins.
        That's the LAST LayerNormDeviceOperation in (lower_bound, sdpa_body_idx)
        — or its preceding InterleavedToSharded if present."""
        last_ln = None
        for j in range(lower_bound, sdpa_body_idx):
            if body[j][0] == "LayerNormDeviceOperation":
                last_ln = j
        if last_ln is None:
            return lower_bound
        if last_ln > 0 and body[last_ln - 1][0] == "InterleavedToShardedDeviceOperation":
            return last_ln - 1
        return last_ln

    # Find post-mlp boundary AFTER the very last SDPA = start of project_output.
    # Within the last layer's mlp we have: o_proj → BinaryNg(residual+gate) →
    # I2S → LN(pre-MLP) → S2I → gate/up matmul → multiply → down → BinaryNg.
    # Project_output starts at the FINAL LN of the model (post-block norm) —
    # find the 2nd LN after the last SDPA.
    def project_output_start(last_sdpa_body_idx: int) -> int:
        ln_seen = 0
        for j in range(last_sdpa_body_idx + 1, len(body)):
            if body[j][0] == "LayerNormDeviceOperation":
                ln_seen += 1
                if ln_seen == 2:
                    # If preceded by I2S, back up one
                    if j > 0 and body[j - 1][0] == "InterleavedToShardedDeviceOperation":
                        return j - 1
                    return j
        return len(body)  # no boundary found → no project_output

    project_start = project_output_start(sdpa_idx[-1])

    # Walk through stages, sdpas inside them
    for stage_label, s_start, s_end, n_layers in stages:
        # SDPA rows of this stage (0-based indices into body)
        for layer_i in range(n_layers):
            sdpa_num = s_start + layer_i
            sdpa_body_idx = sdpa_idx[sdpa_num - 1]

            # Attn-block boundary: walk back from THIS SDPA to find the last LN
            # in the gap since the previous SDPA (the layer's pre-attn LN).
            if layer_i == 0 and stage_label == "siglip":
                attn_start = pre_attn_ln_for(sdpa_body_idx, 0)
            elif layer_i == 0:
                attn_start = pre_attn_ln_for(sdpa_body_idx, sdpa_idx[s_start - 2] + 1)
            else:
                attn_start = pre_attn_ln_for(sdpa_body_idx, sdpa_idx[s_start - 1 + layer_i - 1] + 1)

            # attn ends at the SDPA row (inclusive)
            attn_end = sdpa_body_idx

            # mlp extends from current SDPA + 1 to (next layer's attn start - 1).
            # Next layer's attn starts at the LAST LayerNorm before its SDPA —
            # this handles both within-stage (LN-2 between SDPAs) and cross-stage
            # transitions (LN-3 due to a stage-final norm like vlm_norm sitting
            # between the layer's pre-mlp LN and the next layer's pre-attn LN).
            if layer_i + 1 < n_layers:
                next_sdpa_body_idx = sdpa_idx[s_start - 1 + layer_i + 1]
            elif (s_start, s_end) == (118, 135):
                mlp_end = project_start - 1
                next_sdpa_body_idx = None
            else:
                next_sdpa_body_idx = sdpa_idx[s_end]  # next stage's first SDPA
            if next_sdpa_body_idx is not None:
                # Find the LAST LayerNorm between this SDPA and next SDPA.
                # That LN is the next layer's pre-attn norm; everything before
                # it belongs to THIS layer's mlp (including any stage-final norm).
                last_ln = None
                for j in range(sdpa_body_idx + 1, next_sdpa_body_idx):
                    if body[j][0] == "LayerNormDeviceOperation":
                        last_ln = j
                if last_ln is not None:
                    # Boundary is the LN (or its preceding I2S). Next attn starts there.
                    attn_start_next = (
                        last_ln - 1
                        if last_ln > 0 and body[last_ln - 1][0] == "InterleavedToShardedDeviceOperation"
                        else last_ln
                    )
                    mlp_end = attn_start_next - 1
                else:
                    mlp_end = next_sdpa_body_idx - 1  # fallback

            # Tag attn rows
            for j in range(attn_start, attn_end + 1):
                stage_col[j] = stage_label
                layer_col[j] = str(layer_i + 1)
                sub_col[j] = "attn"
            # Tag mlp rows
            for j in range(attn_end + 1, mlp_end + 1):
                stage_col[j] = stage_label
                layer_col[j] = str(layer_i + 1)
                sub_col[j] = "mlp"

    # Fill in any leading unlabeled rows = "prefix_setup", trailing = "project_output"
    for i in range(len(body)):
        if stage_col[i] == "":
            if i < sdpa_idx[0]:
                stage_col[i] = "prefix_setup"
            else:
                stage_col[i] = "project_output"

    # Write annotated CSV
    new_header = ["STAGE", "LAYER", "SUBSTAGE"] + header
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(new_header)
        for i, row in enumerate(body):
            w.writerow([stage_col[i], layer_col[i], sub_col[i]] + row)

    # Print summary
    print(f"Wrote {out_csv}")
    print(f"Total rows: {len(body)}")
    print()
    print("Stage row ranges (CSV row #, 1-based with header as row 1):")
    print(f"  {'STAGE':<18} {'rows':<14} {'count':>6} {'kernel_ms':>10}")
    print("  " + "-" * 60)
    # Use device kernel duration column to compute ms per stage
    kd_col_idx = header.index("DEVICE KERNEL DURATION [ns]")
    by_stage = {}
    for i, s in enumerate(stage_col):
        by_stage.setdefault(s, []).append(i)
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
    total_ms = 0.0
    for s in stage_order:
        idxs = by_stage.get(s, [])
        if not idxs:
            continue
        start_row = min(idxs) + 2  # +2: CSV header + 1-based
        end_row = max(idxs) + 2
        total_ns = sum(float(body[i][kd_col_idx]) for i in idxs if body[i][kd_col_idx])
        ms = total_ns / 1e6
        total_ms += ms
        print(f"  {s:<18} {start_row:4d}..{end_row:4d}      {len(idxs):>6}  {ms:>9.3f}")
    print("  " + "-" * 60)
    print(f"  {'TOTAL':<18} {'':<14} {sum(len(v) for v in by_stage.values()):>6}  {total_ms:>9.3f}")


if __name__ == "__main__":
    in_csv = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "/home/tt-admin/sdawle/pi0/tt-metal/generated/profiler/reports/2026_06_07_04_52_56/ops_perf_results_2026_06_07_04_52_56.csv"
    )
    out_csv = sys.argv[2] if len(sys.argv) > 2 else str(Path(in_csv).with_name(Path(in_csv).stem + "_annotated.csv"))
    annotate(in_csv, out_csv)
