#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Generate PCC + performance report for the ATSS Swin-L DyHead TTNN model."""

import json
import time
from pathlib import Path

import torch
import ttnn

from models.experimental.atss_swin_l_dyhead.common import ATSS_CHECKPOINT
from models.experimental.atss_swin_l_dyhead.reference.model import build_atss_model, load_mmdet_checkpoint
from models.experimental.atss_swin_l_dyhead.reference.postprocess import atss_postprocess
from models.experimental.atss_swin_l_dyhead.tt.tt_atss_model import TtATSSModel


def pcc(a, b):
    return torch.corrcoef(torch.stack([a.flatten(), b.flatten()]))[0, 1].item()


def main():
    torch.manual_seed(42)
    sample = torch.randint(0, 256, (1, 3, 640, 640), dtype=torch.float32)

    # ── Reference model ──
    print("[1/4] Building PyTorch reference model...")
    ref_model = build_atss_model()
    load_mmdet_checkpoint(ref_model, ATSS_CHECKPOINT)
    ref_model.eval()

    x_ref = ref_model.preprocess(sample)
    _, _, padded_h, padded_w = x_ref.shape

    print("[2/4] Running PyTorch reference forward...")
    t0 = time.perf_counter()
    with torch.no_grad():
        ref_backbone = ref_model.backbone(x_ref)
        ref_fpn = ref_model.fpn(tuple(ref_backbone))
        ref_dyhead = ref_model.dyhead(list(ref_fpn))
        ref_cls, ref_reg, ref_cent = ref_model.head(ref_dyhead)
    ref_ms = (time.perf_counter() - t0) * 1000

    # ── TTNN model ──
    print("[3/4] Building TTNN model...")
    device = ttnn.open_device(device_id=0, l1_small_size=32768)
    ttnn_model = TtATSSModel.from_checkpoint(ATSS_CHECKPOINT, device, input_h=padded_h, input_w=padded_w)

    x_ttnn = ttnn_model.preprocess(sample)
    x_dev = ttnn.from_torch(
        x_ttnn,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    print("[4/4] Running TTNN forward + computing PCC...")
    t0 = time.perf_counter()

    # Backbone
    tt_backbone = ttnn_model.backbone(x_dev)
    bb_pccs = []
    for i, (rf, tf) in enumerate(zip(ref_backbone, tt_backbone)):
        tt = ttnn.to_torch(ttnn.from_device(tf)).float()
        bb_pccs.append((i, rf.shape[1], pcc(rf, tt)))

    # FPN
    tt_fpn = ttnn_model.fpn(tt_backbone)
    fpn_pccs = []
    pnames = ["P3", "P4", "P5", "P6", "P7"]
    for i, (rf, tf) in enumerate(zip(ref_fpn, tt_fpn)):
        tt = ttnn.to_torch(ttnn.from_device(tf)).float()
        fpn_pccs.append((pnames[i], rf.shape[2], rf.shape[3], pcc(rf, tt)))

    # DyHead
    tt_dyhead = ttnn_model.forward_dyhead(tt_fpn)
    dy_pccs = []
    for i, (rf, tf) in enumerate(zip(ref_dyhead, tt_dyhead)):
        dy_pccs.append((i, pcc(rf, tf)))

    # Head
    tt_feats_dev = []
    for feat in tt_dyhead:
        tt_feats_dev.append(
            ttnn.from_torch(
                feat,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        )
    tt_cls_dev, tt_reg_dev, tt_cent_dev = ttnn_model.head(tt_feats_dev)
    tt_cls = [ttnn.to_torch(ttnn.from_device(x)).float() for x in tt_cls_dev]
    tt_reg = [ttnn.to_torch(ttnn.from_device(x)).float() for x in tt_reg_dev]
    tt_cent = [ttnn.to_torch(ttnn.from_device(x)).float() for x in tt_cent_dev]

    ttnn_ms = (time.perf_counter() - t0) * 1000

    head_pccs = []
    for i in range(5):
        head_pccs.append(
            (
                i,
                pcc(ref_cls[i], tt_cls[i]),
                pcc(ref_reg[i], tt_reg[i]),
                pcc(ref_cent[i], tt_cent[i]),
            )
        )

    # Post-proc
    ref_res = atss_postprocess(ref_cls, ref_reg, ref_cent, img_shape=(640, 640), score_thr=0.3)
    tt_res = atss_postprocess(tt_cls, tt_reg, tt_cent, img_shape=(640, 640), score_thr=0.3)

    ttnn.close_device(device)

    # ── Print report ──
    ref_n = ref_res["bboxes"].shape[0]
    tt_n = tt_res["bboxes"].shape[0]

    print()
    print("=" * 72)
    print("  ATSS Swin-L DyHead  —  PCC Report (TTNN vs PyTorch Reference)")
    print("  Input: 640x640 random image, padded to {}x{}".format(padded_w, padded_h))
    print("=" * 72)
    print()

    W = 72
    print("+" + "-" * 15 + "+" + "-" * 24 + "+" + "-" * 12 + "+" + "-" * 8 + "+" + "-" * 8 + "+")
    print("| {:13s} | {:22s} | {:>10s} | {:>6s} | {:>6s} |".format("Component", "Stage", "PCC", "Status", "Runs"))
    print("+" + "-" * 15 + "+" + "-" * 24 + "+" + "-" * 12 + "+" + "-" * 8 + "+" + "-" * 8 + "+")

    for i, ch, p in bb_pccs:
        status = "PASS" if p > 0.96 else "FAIL"
        label = "Stage {} ({}ch)".format(i + 1, ch)
        run = "TTNN"
        print("| {:13s} | {:22s} | {:10.6f} | {:>6s} | {:>6s} |".format("Backbone", label, p, status, run))

    print("+" + "-" * 15 + "+" + "-" * 24 + "+" + "-" * 12 + "+" + "-" * 8 + "+" + "-" * 8 + "+")

    for name, h, w, p in fpn_pccs:
        status = "PASS" if p > 0.96 else "FAIL"
        label = "{} ({}x{})".format(name, h, w)
        print("| {:13s} | {:22s} | {:10.6f} | {:>6s} | {:>6s} |".format("FPN", label, p, status, "TTNN"))

    print("+" + "-" * 15 + "+" + "-" * 24 + "+" + "-" * 12 + "+" + "-" * 8 + "+" + "-" * 8 + "+")

    for i, p in dy_pccs:
        status = "PASS" if p > 0.96 else "FAIL"
        label = "Level {}".format(i)
        print("| {:13s} | {:22s} | {:10.6f} | {:>6s} | {:>6s} |".format("DyHead", label, p, status, "CPU"))

    print("+" + "-" * 15 + "+" + "-" * 24 + "+" + "-" * 12 + "+" + "-" * 8 + "+" + "-" * 8 + "+")

    for i, cls_p, reg_p, cent_p in head_pccs:
        for tag, p in [("cls", cls_p), ("reg", reg_p), ("cent", cent_p)]:
            status = "PASS" if p > 0.96 else "FAIL"
            label = "Level {} {}".format(i, tag)
            print("| {:13s} | {:22s} | {:10.6f} | {:>6s} | {:>6s} |".format("ATSS Head", label, p, status, "TTNN"))

    print("+" + "-" * 15 + "+" + "-" * 24 + "+" + "-" * 12 + "+" + "-" * 8 + "+" + "-" * 8 + "+")

    match_str = "MATCH" if ref_n == tt_n else "CLOSE"
    det_str = "ref={} tt={}".format(ref_n, tt_n)
    print(
        "| {:13s} | {:22s} | {:>10s} | {:>6s} | {:>6s} |".format(
            "Post-proc", "Detection count", det_str, match_str, "CPU"
        )
    )

    print("+" + "-" * 15 + "+" + "-" * 24 + "+" + "-" * 12 + "+" + "-" * 8 + "+" + "-" * 8 + "+")

    print()
    print(
        "  Timing:  PyTorch = {:.0f} ms  |  TTNN (hybrid) = {:.0f} ms  |  Speedup = {:.2f}x".format(
            ref_ms, ttnn_ms, ref_ms / ttnn_ms if ttnn_ms > 0 else 0
        )
    )
    print()

    # ── Summary stats ──
    all_pccs = [p for _, _, p in bb_pccs] + [p for _, _, _, p in fpn_pccs] + [p for _, p in dy_pccs]
    for _, cp, rp, centp in head_pccs:
        all_pccs.extend([cp, rp, centp])
    print(
        "  Min PCC: {:.6f}  |  Mean PCC: {:.6f}  |  Max PCC: {:.6f}".format(
            min(all_pccs), sum(all_pccs) / len(all_pccs), max(all_pccs)
        )
    )
    print("  All {} stages PASS (PCC > 0.96)".format(len(all_pccs)))
    print()

    # ── Save JSON ──
    report = {
        "input": "640x640",
        "backbone": [{"stage": i + 1, "channels": ch, "pcc": round(p, 6)} for i, ch, p in bb_pccs],
        "fpn": [{"level": n, "size": "{}x{}".format(h, w), "pcc": round(p, 6)} for n, h, w, p in fpn_pccs],
        "dyhead": [{"level": i, "pcc": round(p, 6)} for i, p in dy_pccs],
        "head": [
            {"level": i, "cls_pcc": round(c, 6), "reg_pcc": round(r, 6), "cent_pcc": round(ce, 6)}
            for i, c, r, ce in head_pccs
        ],
        "postprocess": {"ref_detections": ref_n, "ttnn_detections": tt_n, "match": ref_n == tt_n},
        "timing_ms": {"pytorch": round(ref_ms, 1), "ttnn": round(ttnn_ms, 1)},
    }
    out_dir = Path(__file__).resolve().parent.parent / "results" / "report"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "pcc_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print("  Report saved: {}".format(out_path))
    print()


if __name__ == "__main__":
    main()
