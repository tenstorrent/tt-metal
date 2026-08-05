#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Run the four-device COCO evaluator with the vanilla Swin-L precision path.

This wrapper accepts the same arguments as ``evaluate_coco_slice_4dev.py`` but
disables all ATSS-specific high-precision backbone stages for this process. It
does not modify the checked-in ATSS defaults.

Examples:

    # Test split, score threshold 0.3
    python models/experimental/atss_swin_l_dyhead/demo/evaluate_coco_slice_4dev_vanilla.py \
        --score-threshold 0.3 \
        --output-dir models/experimental/atss_swin_l_dyhead/results/coco_eval_slice_4dev/test_vanilla_score_0p3

    # Validation split
    python models/experimental/atss_swin_l_dyhead/demo/evaluate_coco_slice_4dev_vanilla.py \
        --annotations models/experimental/atss_swin_l_dyhead/boat-detection-marina.v2i.coco-segmentation/valid/_annotations.coco.json \
        --images-dir models/experimental/atss_swin_l_dyhead/boat-detection-marina.v2i.coco-segmentation/valid \
        --score-threshold 0.3 \
        --output-dir models/experimental/atss_swin_l_dyhead/results/coco_eval_slice_4dev/valid_vanilla_score_0p3
"""

from pathlib import Path

import models.experimental.atss_swin_l_dyhead.tt.tt_atss_model as tt_atss_model
from models.experimental.atss_swin_l_dyhead.demo.evaluate_coco_slice_4dev import evaluate, parse_args
from models.experimental.atss_swin_l_dyhead.tt.tt_swin_backbone import build_atss_backbone


def build_vanilla_atss_backbone(checkpoint_path, device, input_h=None, input_w=None):
    """Build ATSS's Swin-L backbone with all precision promotions disabled."""
    return build_atss_backbone(
        checkpoint_path,
        device,
        input_h=input_h,
        input_w=input_w,
        high_precision_mlp_stages=(),
        high_precision_attn_stages=(),
    )


def main():
    # TtATSSModel.from_checkpoint resolves this module-level symbol when the
    # model is constructed, so replacing it here affects only this process.
    tt_atss_model.build_atss_backbone = build_vanilla_atss_backbone

    args = parse_args()
    if args.pytorch_only:
        raise ValueError("--pytorch-only is not valid for the vanilla TTNN wrapper")
    if args.output_dir is None:
        model_root = Path(__file__).resolve().parent.parent
        args.output_dir = str(model_root / "results" / "coco_eval_slice_4dev" / "test_vanilla")
    evaluate(args)


if __name__ == "__main__":
    main()
