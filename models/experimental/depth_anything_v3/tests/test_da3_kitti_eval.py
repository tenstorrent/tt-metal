# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""KITTI Eigen-split evaluation for Depth Anything V3 metric.

Three complementary measurements per image:

  - **Real GT** — KITTI's seven canonical metrics (AbsRel, SqRel, RMSE,
    RMSE-log, δ<1.25, δ<1.25², δ<1.25³) with median scaling and Garg crop,
    against the dense GT in `gt_depths.npy`. Reported for the canonical
    DA3-Metric, our CPU fp32 reference, and the chip pipeline.
  - **Relative PCC vs canonical** — confirms our reference + chip-pipeline
    outputs match the canonical raw DPT depth (post-`exp`, pre-sky-clip).
  - **Relative PCC chip vs reference** — preserves the chip-vs-CPU regression
    guard from earlier iterations.

Set `DA3_EVAL_LIMIT=N` to cap to N images (default 20).
Set `DA3_EVAL_SKIP_CHIP=1` to skip the chip path (CPU-only host).
Set `DA3_EVAL_SKIP_CANONICAL=1` to skip the canonical comparison."""

from __future__ import annotations

import os
import time

import numpy as np
import pytest

from models.experimental.depth_anything_v3.eval.kitti_eigen import (
    load_kitti_eigen_test,
)
from models.experimental.depth_anything_v3.eval.metrics import (
    aggregate, compute_depth_metrics, garg_crop_mask, pearson_pcc,
)
from models.experimental.depth_anything_v3.eval.runner import (
    chip_predict, reference_predict,
)


def test_da3_kitti_eval() -> None:
    limit = int(os.environ.get("DA3_EVAL_LIMIT", "20"))
    skip_chip = bool(int(os.environ.get("DA3_EVAL_SKIP_CHIP", "0")))
    skip_canon = bool(int(os.environ.get("DA3_EVAL_SKIP_CANONICAL", "0")))

    samples = list(load_kitti_eigen_test(limit=limit))
    print(f"loaded {len(samples)} KITTI Eigen samples", flush=True)

    canonical_predict = None
    if not skip_canon:
        from models.experimental.depth_anything_v3.eval.canonical import (
            canonical_predict as _canonical_predict,
        )
        canonical_predict = _canonical_predict

    canon_metrics, ref_metrics, chip_metrics = [], [], []
    pcc_ref_vs_canon, pcc_chip_vs_canon, pcc_chip_vs_ref = [], [], []

    t0 = time.perf_counter()
    for s in samples:
        ref_d = reference_predict(s.rgb)
        crop = garg_crop_mask(*s.gt_depth.shape)
        ref_metrics.append(compute_depth_metrics(ref_d, s.gt_depth))

        if canonical_predict is not None:
            canon_d = canonical_predict(s.rgb)
            canon_metrics.append(compute_depth_metrics(canon_d, s.gt_depth))
            pcc_ref_vs_canon.append(pearson_pcc(ref_d[crop], canon_d[crop]))

        if not skip_chip:
            chip_d = chip_predict(s.rgb)
            chip_metrics.append(compute_depth_metrics(chip_d, s.gt_depth))
            pcc_chip_vs_ref.append(pearson_pcc(chip_d[crop], ref_d[crop]))
            if canonical_predict is not None:
                pcc_chip_vs_canon.append(pearson_pcc(chip_d[crop], canon_d[crop]))

    elapsed = time.perf_counter() - t0
    inference_speed = len(samples) / max(elapsed, 1e-9)

    def fmt(d):
        return ", ".join(f"{k}={v:.4f}" for k, v in d.items())

    if canon_metrics:
        print("==== canonical DA3-Metric (Bytedance) vs GT ====", flush=True)
        print(fmt(aggregate(canon_metrics)), flush=True)
    print("==== our reference (CPU fp32) vs GT ====", flush=True)
    print(fmt(aggregate(ref_metrics)), flush=True)
    if chip_metrics:
        print("==== chip pipeline vs GT ====", flush=True)
        print(fmt(aggregate(chip_metrics)), flush=True)

    if pcc_ref_vs_canon:
        print(f"==== ours-ref vs canonical Pearson PCC ====", flush=True)
        print(f"pcc_mean={np.mean(pcc_ref_vs_canon):.6f}  pcc_min={min(pcc_ref_vs_canon):.6f}", flush=True)
    if pcc_chip_vs_canon:
        print(f"==== chip vs canonical Pearson PCC ====", flush=True)
        print(f"pcc_mean={np.mean(pcc_chip_vs_canon):.6f}  pcc_min={min(pcc_chip_vs_canon):.6f}", flush=True)
    if pcc_chip_vs_ref:
        print(f"==== chip vs reference Pearson PCC ====", flush=True)
        print(f"pcc_mean={np.mean(pcc_chip_vs_ref):.6f}  pcc_min={min(pcc_chip_vs_ref):.6f}", flush=True)
    print(f"inference_speed={inference_speed:.4f} frames/sec", flush=True)

    # Reference must reproduce canonical to high PCC if we trust the GT metrics.
    if pcc_ref_vs_canon:
        assert np.mean(pcc_ref_vs_canon) > 0.999, (
            f"reference vs canonical PCC = {np.mean(pcc_ref_vs_canon):.4f} — "
            "our reference no longer reproduces canonical DA3-Metric"
        )
    if pcc_chip_vs_ref:
        assert np.mean(pcc_chip_vs_ref) > 0.97, (
            f"chip-vs-ref PCC too low: {np.mean(pcc_chip_vs_ref):.4f}"
        )
