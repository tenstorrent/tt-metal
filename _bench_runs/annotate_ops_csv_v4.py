#!/usr/bin/env python3
"""Annotate a pi0.5 ops_perf_results CSV — v4: rebalanced per-step boundaries.

Same overall stage layout as v3 (siglip / vlm_prefill / denoise_step_1..5 /
project_output), with one fix: every denoise step's range now contains its
OWN head (x_t upload + embed_actions + per-chip adarms_cond build) + body
(18 expert layers) + tail (final adaRMS norm + action_out_proj for that
step). v3's boundaries put the head of step N+1 into step N's range, which
made step 5 look ~100 ops / ~0.85 ms shorter than steps 1–4 even though all
five steps do identical work.

What changed vs v3:
  - Each denoise step's stage_end is the op AFTER that step's action_out_proj
    (the project matmul that follows the 2nd LayerNorm past the last SDPA).
  - SUBSTAGE column for denoise steps now distinguishes `head`, `attn{li}`,
    `mlp{li}`, `tail` (instead of just `attn` / `mlp`).
  - project_output stage label is reserved for the final per-step tail of
    step 5 only when there is genuinely no following step — for v4 we fold
    it back into denoise_step_5/tail so all 5 steps look identical.

Stage layout per inference (1-based SDPA #'s):
  SDPA #1..27   → siglip          27 layers
  SDPA #28..45  → vlm_prefill     18 layers
  SDPA #46..63  → denoise_step_1  18 layers (head + body + tail)
  ... → denoise_step_5            18 layers (head + body + tail)

Init detection (single-inference CSVs) is unchanged from v3.
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
    op = row[header_idx["OP CODE"]]
    out0_x = _logical(row[header_idx["OUTPUT_0_X_PAD[LOGICAL]"]])
    in0_dtype = row[header_idx["INPUT_0_DATATYPE"]]
    out0_dtype = row[header_idx["OUTPUT_0_DATATYPE"]]
    in2_x = _logical(row[header_idx["INPUT_2_X_PAD[LOGICAL]"]]) if "INPUT_2_X_PAD[LOGICAL]" in header_idx else None
    if op == "ConcatDeviceOperation" and in2_x is not None and out0_x in (2560, 4608):
        return True
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

    def find_step_tail_end(last_sdpa_of_step, hard_limit):
        """Locate the index AFTER the action_out_proj matmul that follows the
        2nd LayerNorm after the step's last SDPA. Returns hard_limit if not
        found (e.g. truncated CSV)."""
        ln_seen = 0
        for j in range(last_sdpa_of_step + 1, hard_limit):
            if body[j][op_col] == "LayerNormDeviceOperation":
                ln_seen += 1
                if ln_seen == 2:
                    # The action_out_proj is the next Matmul AFTER this LN.
                    # Allow a few intermediary ops (sharded↔interleaved typecasts).
                    for k in range(j + 1, min(j + 20, hard_limit)):
                        if body[k][op_col] == "MatmulDeviceOperation":
                            return k + 1  # exclusive end
                    return j + 1  # didn't find matmul; clamp at LN
        return hard_limit

    def annotate_inference(inf_num, suffix=""):
        offs = SDPA_PER_INFERENCE * (inf_num - 1)
        inf_start = 0 if inf_num == 1 else sdpa_idx[offs - 1] + 1

        if inf_num < n_inferences:
            next_offs = SDPA_PER_INFERENCE * inf_num
            inf_end = sdpa_idx[next_offs - 1] + 1  # placeholder; refined below
            inf_end = last_ln_before(sdpa_idx[next_offs + 0], sdpa_idx[next_offs - 1] + 1)
        else:
            inf_end = len(body)

        # === Stage boundaries ===
        # siglip + vlm_prefill use the same "body-start = first attn LN of layer 1"
        # convention as v3. The denoise steps are the part that differs.
        siglip_first_sdpa = sdpa_idx[offs + 0]
        vlm_first_sdpa = sdpa_idx[offs + 27]
        denoise1_first_sdpa = sdpa_idx[offs + 45]

        siglip_start = last_ln_before(siglip_first_sdpa, inf_start)
        vlm_start = last_ln_before(vlm_first_sdpa, sdpa_idx[offs + 26] + 1)

        # vlm_prefill ENDS right after VLM's final RMS norm (the 2nd LN after
        # vlm's last SDPA #45). Everything after that (KV-migration typecasts +
        # x_t upload + embed_actions + per-chip adarms_cond) is the "head" of
        # denoise step 1. This keeps step 1 consistent with steps 2-5.
        vlm_last_sdpa = sdpa_idx[offs + 44]
        ln_seen = 0
        vlm_end = denoise1_first_sdpa  # safe fallback
        for j in range(vlm_last_sdpa + 1, denoise1_first_sdpa):
            if body[j][op_col] == "LayerNormDeviceOperation":
                ln_seen += 1
                if ln_seen == 2:
                    vlm_end = j + 1
                    break
        denoise1_start = vlm_end

        # For each denoise step, the TAIL END is right after that step's
        # action_out_proj. The HEAD of step N+1 begins at the TAIL END of step N.
        # Step 1's HEAD begins where vlm_prefill ends (vlm_end == denoise1_start),
        # i.e. at the first attn LN of layer 1 of step 1, which means step 1
        # has no "head" of its own — the loop iteration's setup (x_t upload,
        # embed_actions, adarms_cond) actually happens BEFORE that LN, which
        # falls into the vlm_prefill "tail". For consistency we move that
        # boundary back to "after vlm's project / its 2nd-LN" too. Easiest:
        # snap denoise step 1's head_start to the start of post-vlm activity.
        # But since SDPA #28 (vlm layer 1) is preceded by vlm prefill embedding,
        # we keep the current vlm_end == first attn LN of expert layer 1.

        denoise_starts = [denoise1_start]
        for n in range(1, 5):
            # last SDPA of step n is offs + 45 + (n-1)*18 + 17
            last_sdpa_of_step_n = sdpa_idx[offs + 45 + (n - 1) * 18 + 17]
            # search bound = the next step's first SDPA (or inf_end)
            next_first_sdpa = sdpa_idx[offs + 45 + n * 18] if n < 5 else None
            hard_limit = next_first_sdpa if next_first_sdpa is not None else inf_end
            tail_end = find_step_tail_end(last_sdpa_of_step_n, hard_limit)
            denoise_starts.append(tail_end)

        # Step 5's tail end is right after step 5's action_out_proj.
        last_sdpa_of_step_5 = sdpa_idx[offs + 134]
        step5_tail_end = find_step_tail_end(last_sdpa_of_step_5, inf_end)

        denoise_ranges = []
        for n in range(1, 6):
            start = denoise_starts[n - 1]
            end = denoise_starts[n] if n < 5 else step5_tail_end
            denoise_ranges.append((f"denoise_step_{n}", start, end))

        # === Label prefix_setup region (will be split into init / prefix_setup
        # later for single-inference CSVs via signature heuristic) ===
        prefix_start = inf_start
        prefix_end = siglip_start
        for i in range(prefix_start, prefix_end):
            stage_col[i] = "prefix_setup" + suffix

        # === Label siglip + vlm_prefill stages ===
        stages_with_attn_mlp = [
            ("siglip", siglip_start, vlm_start, 0, 26, 27),
            ("vlm_prefill", vlm_start, vlm_end, 27, 44, 18),
        ]
        for stage_label, s_start, s_end, s_off, e_off, n_layers in stages_with_attn_mlp:
            layer_start = s_start
            for li in range(n_layers):
                layer_sdpa = sdpa_idx[offs + s_off + li]
                if li + 1 < n_layers:
                    next_layer_sdpa = sdpa_idx[offs + s_off + li + 1]
                    next_attn_start = last_ln_before(next_layer_sdpa, layer_sdpa + 1)
                else:
                    next_attn_start = s_end
                for j in range(layer_start, layer_sdpa + 1):
                    stage_col[j] = stage_label + suffix
                    layer_col[j] = str(li + 1)
                    sub_col[j] = "attn"
                for j in range(layer_sdpa + 1, next_attn_start):
                    stage_col[j] = stage_label + suffix
                    layer_col[j] = str(li + 1)
                    sub_col[j] = "mlp"
                layer_start = next_attn_start

        # === Label denoise steps with head / attn{li} / mlp{li} / tail ===
        for step_idx, (stage_label, s_start, s_end) in enumerate(denoise_ranges):
            step_num = step_idx + 1
            s_off = 45 + step_idx * 18
            e_off = s_off + 17
            n_layers = 18

            # head = s_start .. first_attn_LN_of_layer_1 (exclusive)
            first_layer_sdpa = sdpa_idx[offs + s_off]
            head_end = last_ln_before(first_layer_sdpa, s_start)

            # body = head_end .. last_attn_LN_of_layer_18 then mlp..tail_start
            # tail = (after layer 18's MLP residual add) .. s_end
            last_layer_sdpa = sdpa_idx[offs + e_off]
            # find the SECOND LN after last_layer_sdpa — that's the final adaRMS norm
            ln_seen = 0
            tail_start = s_end  # fallback
            for j in range(last_layer_sdpa + 1, s_end):
                if body[j][op_col] == "LayerNormDeviceOperation":
                    ln_seen += 1
                    if ln_seen == 2:
                        tail_start = (
                            j - 1 if j > 0 and body[j - 1][op_col] == "InterleavedToShardedDeviceOperation" else j
                        )
                        break

            # head label
            for j in range(s_start, head_end):
                stage_col[j] = stage_label + suffix
                layer_col[j] = ""
                sub_col[j] = "head"

            # body label (per-layer attn/mlp)
            layer_start = head_end
            for li in range(n_layers):
                layer_sdpa = sdpa_idx[offs + s_off + li]
                if li + 1 < n_layers:
                    next_layer_sdpa = sdpa_idx[offs + s_off + li + 1]
                    next_attn_start = last_ln_before(next_layer_sdpa, layer_sdpa + 1)
                else:
                    next_attn_start = tail_start
                for j in range(layer_start, layer_sdpa + 1):
                    stage_col[j] = stage_label + suffix
                    layer_col[j] = str(li + 1)
                    sub_col[j] = "attn"
                for j in range(layer_sdpa + 1, next_attn_start):
                    stage_col[j] = stage_label + suffix
                    layer_col[j] = str(li + 1)
                    sub_col[j] = "mlp"
                layer_start = next_attn_start

            # tail label
            for j in range(tail_start, s_end):
                stage_col[j] = stage_label + suffix
                layer_col[j] = ""
                sub_col[j] = "tail"

        # === per-stage ms ===
        def ms_of(start, end):
            return sum(float(body[i][kd_col] or 0) for i in range(start, end)) / 1e6

        breakdown = {}
        breakdown["prefix_setup"] = (prefix_start, prefix_end, ms_of(prefix_start, prefix_end))
        breakdown["siglip"] = (siglip_start, vlm_start, ms_of(siglip_start, vlm_start))
        breakdown["vlm_prefill"] = (vlm_start, vlm_end, ms_of(vlm_start, vlm_end))
        for stage_label, s, e in denoise_ranges:
            breakdown[stage_label] = (s, e, ms_of(s, e))
        return breakdown

    # === Annotation pass ===
    if n_inferences > 1:
        inf_breakdowns = {}
        for k in range(1, n_inferences + 1):
            if k == n_inferences:
                lbl = ""
            elif k == n_inferences - 1:
                lbl = "_trace_capture"
            else:
                lbl = f"_warmup{k}"
            inf_breakdowns[k] = annotate_inference(k, lbl)
        end_inf1 = inf_breakdowns[1]["prefix_setup"][1]
        for i in range(0, end_inf1):
            stage_col[i] = "init_one_time"
        canonical = inf_breakdowns[n_inferences]
        init_count = end_inf1
        init_ms_val = sum(float(body[i][kd_col] or 0) for i in range(0, end_inf1)) / 1e6
        canonical_label = f"Inference {n_inferences} (canonical = trace replay/last)"
    else:
        single = annotate_inference(1, "")
        prefix_end = single["prefix_setup"][1]
        last_init_row = -1
        for i in range(0, prefix_end):
            if _is_init_by_signature(body[i], header_idx):
                last_init_row = i
        init_count = last_init_row + 1
        init_ms_val = (
            sum(float(body[i][kd_col] or 0) for i in range(0, last_init_row + 1)) / 1e6 if last_init_row >= 0 else 0.0
        )
        real_count = prefix_end - (last_init_row + 1)
        real_ms_val = sum(float(body[i][kd_col] or 0) for i in range(last_init_row + 1, prefix_end)) / 1e6
        for i in range(0, last_init_row + 1):
            stage_col[i] = "init_one_time"
        for i in range(last_init_row + 1, prefix_end):
            stage_col[i] = "prefix_setup"
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
        if n_inferences == 1 and s == "prefix_setup":
            n_real = sum(1 for j in range(start, end) if stage_col[j] == "prefix_setup")
            print(f"  {s:<18} {start + 2:5d}..{end + 1:5d}    {n_real:>6}  {ms:>9.3f}")
        else:
            print(f"  {s:<18} {start + 2:5d}..{end + 1:5d}    {n:>6}  {ms:>9.3f}")
    print("  " + "-" * 64)
    print(f"  {'TOTAL/inference':<18} {'':<22} {'':>6}  {total_ms:>9.3f}  ← EXCLUDES init")
    print()

    # === Per-denoise-step substage breakdown ===
    print("=== Denoise step substage breakdown (head + body + tail per step) ===")
    print(f"  {'STEP':<18} {'head ms':>10} {'body ms':>10} {'tail ms':>10} {'total ms':>10}")
    print("  " + "-" * 64)
    for n in range(1, 6):
        s_label = f"denoise_step_{n}"
        if s_label not in canonical:
            continue
        s_start, s_end, _ = canonical[s_label]
        head_ms = (
            sum(
                float(body[i][kd_col] or 0)
                for i in range(s_start, s_end)
                if stage_col[i] == s_label and sub_col[i] == "head"
            )
            / 1e6
        )
        tail_ms = (
            sum(
                float(body[i][kd_col] or 0)
                for i in range(s_start, s_end)
                if stage_col[i] == s_label and sub_col[i] == "tail"
            )
            / 1e6
        )
        body_ms = (
            sum(
                float(body[i][kd_col] or 0)
                for i in range(s_start, s_end)
                if stage_col[i] == s_label and sub_col[i] in ("attn", "mlp")
            )
            / 1e6
        )
        total = head_ms + body_ms + tail_ms
        print(f"  {s_label:<18} {head_ms:>10.3f} {body_ms:>10.3f} {tail_ms:>10.3f} {total:>10.3f}")

    return {
        "init_ms": init_ms_val,
        "init_count": init_count,
        "breakdown": canonical,
        "n_inferences": n_inferences,
    }


if __name__ == "__main__":
    in_csv = sys.argv[1] if len(sys.argv) > 1 else None
    if in_csv is None:
        print("usage: annotate_ops_csv_v4.py <input.csv> [output.csv]", file=sys.stderr)
        sys.exit(1)
    out_csv = sys.argv[2] if len(sys.argv) > 2 else str(Path(in_csv).with_name(Path(in_csv).stem + "_annotated_v4.csv"))
    annotate(in_csv, out_csv)
