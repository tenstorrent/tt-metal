#!/usr/bin/env python3
"""Annotate a pi0.5 ops_perf_results CSV — v5: per-device aware.

v4's boundary-detection walks the CSV linearly across all rows. On a
multi-device run the rows interleave by host time across all 8 chips, so
v4's "find 2nd LN after last SDPA" picks up LNs from OTHER devices, not
the same one whose SDPA it just saw. The result is mis-attribution of
~10% of ops between adjacent stages.

v5 fix: process each device's row subsequence INDEPENDENTLY. Each device
ran the same trace, so the per-device sequence is consistent.

Stage layout per inference (same as v4):
  SDPA #1..27   → siglip          27 layers
  SDPA #28..45  → vlm_prefill     18 layers
  SDPA #46..63  → denoise_step_1  18 layers (head + body + tail)
  ... → denoise_step_5            18 layers (head + body + tail)
"""

import csv
import re
import sys
from collections import defaultdict
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

    # Required columns
    header_idx = {h: i for i, h in enumerate(header)}
    op_col = header_idx["OP CODE"]
    device_col = header_idx["DEVICE ID"]
    kd_col = header_idx["DEVICE KERNEL DURATION [ns]"]

    # Group row indices by device (preserves order within each device).
    rows_per_device = defaultdict(list)
    for i, row in enumerate(body):
        rows_per_device[row[device_col]].append(i)

    # Output arrays.
    stage_col = [""] * len(body)
    layer_col = [""] * len(body)
    sub_col = [""] * len(body)

    # ===== Per-device annotation =====
    # We use device 0's row subsequence to detect inferences (number of full
    # 135-SDPA blocks per device). All devices have the same count.
    any_device = next(iter(rows_per_device.keys()))
    dev_rows = rows_per_device[any_device]
    dev_sdpa_count = sum(1 for i in dev_rows if body[i][op_col] == "SDPAOperation")
    n_inferences = dev_sdpa_count // SDPA_PER_INFERENCE
    if n_inferences < 1:
        print(f"WARN: only {dev_sdpa_count} SDPAs on device {any_device}; expected >= {SDPA_PER_INFERENCE}")
        n_inferences = 1

    print(f"Devices: {sorted(rows_per_device.keys())}")
    print(f"Per-device SDPAs: {dev_sdpa_count} → {n_inferences} inferences/device")

    def last_ln_before(dev_rows_list, sdpa_local_pos, lower_local_pos):
        """Find the last LayerNorm in dev_rows_list[lower_local_pos:sdpa_local_pos].
        Return its LOCAL POSITION (index into dev_rows_list)."""
        last_ln = None
        for j in range(lower_local_pos, sdpa_local_pos):
            if body[dev_rows_list[j]][op_col] == "LayerNormDeviceOperation":
                last_ln = j
        if last_ln is None:
            return lower_local_pos
        if last_ln > 0 and body[dev_rows_list[last_ln - 1]][op_col] == "InterleavedToShardedDeviceOperation":
            return last_ln - 1
        return last_ln

    def find_step_tail_end(dev_rows_list, last_sdpa_local_pos, hard_limit_local_pos):
        """Locate the local position AFTER the action_out_proj matmul that
        follows the 2nd LN after the step's last SDPA."""
        ln_seen = 0
        for j in range(last_sdpa_local_pos + 1, hard_limit_local_pos):
            if body[dev_rows_list[j]][op_col] == "LayerNormDeviceOperation":
                ln_seen += 1
                if ln_seen == 2:
                    for k in range(j + 1, min(j + 20, hard_limit_local_pos)):
                        if body[dev_rows_list[k]][op_col] == "MatmulDeviceOperation":
                            return k + 1
                    return j + 1
        return hard_limit_local_pos

    def annotate_device_inference(dev_rows_list, sdpa_local_idx, inf_num, suffix=""):
        """Annotate ONE inference on ONE device. dev_rows_list is the device's
        ordered row indices into body. sdpa_local_idx is positions of SDPAs
        within dev_rows_list."""
        offs = SDPA_PER_INFERENCE * (inf_num - 1)
        inf_start_local = 0 if inf_num == 1 else sdpa_local_idx[offs - 1] + 1

        if inf_num < n_inferences:
            next_offs = SDPA_PER_INFERENCE * inf_num
            inf_end_local = last_ln_before(
                dev_rows_list, sdpa_local_idx[next_offs + 0], sdpa_local_idx[next_offs - 1] + 1
            )
        else:
            inf_end_local = len(dev_rows_list)

        siglip_first_sdpa = sdpa_local_idx[offs + 0]
        vlm_first_sdpa = sdpa_local_idx[offs + 27]
        denoise1_first_sdpa = sdpa_local_idx[offs + 45]

        siglip_start = last_ln_before(dev_rows_list, siglip_first_sdpa, inf_start_local)
        vlm_start = last_ln_before(dev_rows_list, vlm_first_sdpa, sdpa_local_idx[offs + 26] + 1)

        # vlm_end = row after vlm's final RMS norm (2nd LN after vlm's last SDPA).
        vlm_last_sdpa = sdpa_local_idx[offs + 44]
        ln_seen = 0
        vlm_end = denoise1_first_sdpa
        for j in range(vlm_last_sdpa + 1, denoise1_first_sdpa):
            if body[dev_rows_list[j]][op_col] == "LayerNormDeviceOperation":
                ln_seen += 1
                if ln_seen == 2:
                    vlm_end = j + 1
                    break
        denoise1_start = vlm_end

        # Denoise step boundaries.
        denoise_starts = [denoise1_start]
        for n in range(1, 5):
            last_sdpa_of_step_n = sdpa_local_idx[offs + 45 + (n - 1) * 18 + 17]
            next_first_sdpa = sdpa_local_idx[offs + 45 + n * 18] if n < 5 else None
            hard_limit = next_first_sdpa if next_first_sdpa is not None else inf_end_local
            tail_end = find_step_tail_end(dev_rows_list, last_sdpa_of_step_n, hard_limit)
            denoise_starts.append(tail_end)

        last_sdpa_of_step_5 = sdpa_local_idx[offs + 134]
        step5_tail_end = find_step_tail_end(dev_rows_list, last_sdpa_of_step_5, inf_end_local)

        denoise_ranges = []
        for n in range(1, 6):
            start = denoise_starts[n - 1]
            end = denoise_starts[n] if n < 5 else step5_tail_end
            denoise_ranges.append((f"denoise_step_{n}", start, end))

        # === Label prefix_setup region ===
        prefix_start_local = inf_start_local
        prefix_end_local = siglip_start
        for j in range(prefix_start_local, prefix_end_local):
            stage_col[dev_rows_list[j]] = "prefix_setup" + suffix

        # === Label siglip + vlm_prefill stages with per-layer attn/mlp ===
        stages_with_attn_mlp = [
            ("siglip", siglip_start, vlm_start, 0, 27),
            ("vlm_prefill", vlm_start, vlm_end, 27, 18),
        ]
        for stage_label, s_start, s_end, s_off, n_layers in stages_with_attn_mlp:
            layer_start = s_start
            for li in range(n_layers):
                layer_sdpa = sdpa_local_idx[offs + s_off + li]
                if li + 1 < n_layers:
                    next_layer_sdpa = sdpa_local_idx[offs + s_off + li + 1]
                    next_attn_start = last_ln_before(dev_rows_list, next_layer_sdpa, layer_sdpa + 1)
                else:
                    next_attn_start = s_end
                for j in range(layer_start, layer_sdpa + 1):
                    gi = dev_rows_list[j]
                    stage_col[gi] = stage_label + suffix
                    layer_col[gi] = str(li + 1)
                    sub_col[gi] = "attn"
                for j in range(layer_sdpa + 1, next_attn_start):
                    gi = dev_rows_list[j]
                    stage_col[gi] = stage_label + suffix
                    layer_col[gi] = str(li + 1)
                    sub_col[gi] = "mlp"
                layer_start = next_attn_start

        # === Label denoise steps with head / attn{li} / mlp{li} / tail ===
        for step_idx, (stage_label, s_start, s_end) in enumerate(denoise_ranges):
            s_off = 45 + step_idx * 18
            n_layers = 18

            first_layer_sdpa = sdpa_local_idx[offs + s_off]
            head_end = last_ln_before(dev_rows_list, first_layer_sdpa, s_start)

            last_layer_sdpa = sdpa_local_idx[offs + s_off + 17]
            ln_seen = 0
            tail_start = s_end
            for j in range(last_layer_sdpa + 1, s_end):
                if body[dev_rows_list[j]][op_col] == "LayerNormDeviceOperation":
                    ln_seen += 1
                    if ln_seen == 2:
                        tail_start = (
                            j - 1
                            if j > 0 and body[dev_rows_list[j - 1]][op_col] == "InterleavedToShardedDeviceOperation"
                            else j
                        )
                        break

            for j in range(s_start, head_end):
                gi = dev_rows_list[j]
                stage_col[gi] = stage_label + suffix
                sub_col[gi] = "head"

            layer_start = head_end
            for li in range(n_layers):
                layer_sdpa = sdpa_local_idx[offs + s_off + li]
                if li + 1 < n_layers:
                    next_layer_sdpa = sdpa_local_idx[offs + s_off + li + 1]
                    next_attn_start = last_ln_before(dev_rows_list, next_layer_sdpa, layer_sdpa + 1)
                else:
                    next_attn_start = tail_start
                for j in range(layer_start, layer_sdpa + 1):
                    gi = dev_rows_list[j]
                    stage_col[gi] = stage_label + suffix
                    layer_col[gi] = str(li + 1)
                    sub_col[gi] = "attn"
                for j in range(layer_sdpa + 1, next_attn_start):
                    gi = dev_rows_list[j]
                    stage_col[gi] = stage_label + suffix
                    layer_col[gi] = str(li + 1)
                    sub_col[gi] = "mlp"
                layer_start = next_attn_start

            for j in range(tail_start, s_end):
                gi = dev_rows_list[j]
                stage_col[gi] = stage_label + suffix
                sub_col[gi] = "tail"

    # Run annotation per device.
    for dev, dev_rows_list in sorted(rows_per_device.items()):
        sdpa_local_idx = [i for i, ri in enumerate(dev_rows_list) if body[ri][op_col] == "SDPAOperation"]
        n_local_inf = len(sdpa_local_idx) // SDPA_PER_INFERENCE
        for k in range(1, n_local_inf + 1):
            if k == n_local_inf:
                lbl = ""
            elif k == n_local_inf - 1:
                lbl = "_trace_capture"
            else:
                lbl = f"_warmup{k}"
            annotate_device_inference(dev_rows_list, sdpa_local_idx, k, lbl)

    # Mark init_one_time rows for ALL devices: anything in dev_rows[0:first inference start].
    # First-inference start on each device = start of inference 1 = before first stage.
    # We've already set those rows to "prefix_setup" on the first inference. Mark
    # those that match init signatures as init_one_time.
    for i in range(len(body)):
        if stage_col[i] == "prefix_setup" and _is_init_by_signature(body[i], header_idx):
            stage_col[i] = "init_one_time"

    # === Write annotated CSV ===
    new_header = ["STAGE", "LAYER", "SUBSTAGE"] + header
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(new_header)
        for i, row in enumerate(body):
            w.writerow([stage_col[i], layer_col[i], sub_col[i]] + row)
    print(f"Wrote {out_csv}")
    print(f"Total rows: {len(body)}")

    # Print per-stage kernel ms on a sane device (device 1 if present, else any).
    sane_dev = "1" if "1" in rows_per_device else next(iter(rows_per_device.keys()))
    print(f"\n=== Per-stage kernel ms on device {sane_dev} (canonical inference) ===")
    stage_ms = defaultdict(float)
    stage_cnt = defaultdict(int)
    for i in rows_per_device[sane_dev]:
        s = stage_col[i]
        if not s or "_warmup" in s or "_trace_capture" in s or s == "init_one_time":
            continue
        try:
            d = float(body[i][kd_col]) if body[i][kd_col] else 0.0
        except ValueError:
            d = 0.0
        if 0 < d < 1e8:
            stage_ms[s] += d
            stage_cnt[s] += 1
    order = [
        "prefix_setup",
        "siglip",
        "vlm_prefill",
        "denoise_step_1",
        "denoise_step_2",
        "denoise_step_3",
        "denoise_step_4",
        "denoise_step_5",
    ]
    total = 0.0
    print(f"  {'STAGE':<22} {'count':>6} {'kernel_ms':>10}")
    print("  " + "-" * 50)
    for s in order:
        if s in stage_ms:
            print(f"  {s:<22} {stage_cnt[s]:>6}   {stage_ms[s]/1e6:>8.3f}")
            total += stage_ms[s]
    print("  " + "-" * 50)
    print(f"  {'TOTAL':<22} {'':>6}   {total/1e6:>8.3f}")


if __name__ == "__main__":
    in_csv = sys.argv[1] if len(sys.argv) > 1 else None
    if in_csv is None:
        print("usage: annotate_ops_csv_v5.py <in.csv> [out.csv]", file=sys.stderr)
        sys.exit(2)
    out_csv = sys.argv[2] if len(sys.argv) > 2 else str(Path(in_csv).with_name(Path(in_csv).stem + "_annotated_v5.csv"))
    annotate(in_csv, out_csv)
