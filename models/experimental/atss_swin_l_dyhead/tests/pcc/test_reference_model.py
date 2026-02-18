# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Verify that the standalone reference model produces outputs matching
# the original mmdet model loaded via mmdet.apis.init_detector.

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

# Add tt-metal root to path
TT_METAL_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(TT_METAL_ROOT))

from models.experimental.atss_swin_l_dyhead.reference.model import (
    build_atss_model,
    load_mmdet_checkpoint,
)
from models.experimental.atss_swin_l_dyhead.common import (
    get_checkpoint_path,
    get_config_path,
)


@pytest.fixture(scope="module")
def reference_model():
    """Build and load the standalone reference model."""
    ckpt_path = get_checkpoint_path()
    model = build_atss_model()
    missing, unexpected = load_mmdet_checkpoint(model, ckpt_path)
    print(f"[reference] Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
    if missing:
        print(f"  Missing (first 10): {missing[:10]}")
    if unexpected:
        print(f"  Unexpected (first 10): {unexpected[:10]}")
    model.eval()
    return model


@pytest.fixture(scope="module")
def mmdet_model():
    """Load the mmdet model via init_detector for comparison."""
    try:
        from mmdet.apis import init_detector
    except ImportError:
        pytest.skip("mmdet not available for comparison test")
    cfg_path = get_config_path()
    ckpt_path = get_checkpoint_path()
    model = init_detector(cfg_path, ckpt_path, device="cpu")
    model.eval()
    return model


@pytest.fixture(scope="module")
def sample_input():
    """Create a deterministic sample input image."""
    torch.manual_seed(42)
    img = torch.randint(0, 256, (1, 3, 640, 640), dtype=torch.float32)
    return img


class TestReferenceModel:
    """Tests for the standalone reference model."""

    def test_model_builds(self, reference_model):
        """Verify model builds and all parameters are loaded."""
        assert reference_model is not None
        total_params = sum(p.numel() for p in reference_model.parameters())
        print(f"[info] Total parameters: {total_params:,}")
        assert total_params > 0

    def test_backbone_forward(self, reference_model, sample_input):
        """Verify backbone produces expected output shapes."""
        x = reference_model.preprocess(sample_input)
        with torch.no_grad():
            feats = reference_model.backbone(x)

        assert len(feats) == 3
        print(f"[backbone] Feature shapes: {[f.shape for f in feats]}")

        assert feats[0].shape[1] == 384
        assert feats[1].shape[1] == 768
        assert feats[2].shape[1] == 1536

    def test_fpn_forward(self, reference_model, sample_input):
        """Verify FPN produces 5 levels with 256 channels."""
        x = reference_model.preprocess(sample_input)
        with torch.no_grad():
            feats = reference_model.backbone(x)
            fpn_feats = reference_model.fpn(tuple(feats))

        assert len(fpn_feats) == 5
        for i, f in enumerate(fpn_feats):
            assert f.shape[1] == 256, f"FPN level {i} has {f.shape[1]} channels, expected 256"
        print(f"[fpn] Feature shapes: {[f.shape for f in fpn_feats]}")

    def test_dyhead_forward(self, reference_model, sample_input):
        """Verify DyHead produces 5 levels with 256 channels."""
        x = reference_model.preprocess(sample_input)
        with torch.no_grad():
            feats = reference_model.backbone(x)
            fpn_feats = reference_model.fpn(tuple(feats))
            dy_feats = reference_model.dyhead(list(fpn_feats))

        assert len(dy_feats) == 5
        for i, f in enumerate(dy_feats):
            assert f.shape[1] == 256
        print(f"[dyhead] Feature shapes: {[f.shape for f in dy_feats]}")

    def test_full_forward(self, reference_model, sample_input):
        """Verify full forward produces cls_scores, bbox_preds, centernesses."""
        x = reference_model.preprocess(sample_input)
        with torch.no_grad():
            cls_scores, bbox_preds, centernesses = reference_model(x)

        assert len(cls_scores) == 5
        assert len(bbox_preds) == 5
        assert len(centernesses) == 5

        for i in range(5):
            assert cls_scores[i].shape[1] == 80
            assert bbox_preds[i].shape[1] == 4
            assert centernesses[i].shape[1] == 1
        print(f"[head] cls shapes: {[s.shape for s in cls_scores]}")

    def test_predict(self, reference_model, sample_input):
        """Verify end-to-end prediction produces detections."""
        with torch.no_grad():
            results = reference_model.predict(
                sample_input,
                img_shape=(640, 640),
                score_thr=0.3,
            )

        assert "bboxes" in results
        assert "scores" in results
        assert "labels" in results
        print(f"[predict] Detections: {results['bboxes'].shape[0]}")
        if results["scores"].numel() > 0:
            print(f"  Top score: {results['scores'][0]:.4f}")


class TestPCCvsMMDet:
    """Compare standalone reference outputs against mmdet model."""

    def test_backbone_pcc(self, reference_model, mmdet_model, sample_input):
        """Compare backbone feature maps."""
        x_ref = reference_model.preprocess(sample_input)

        with torch.no_grad():
            ref_feats = reference_model.backbone(x_ref)

        with torch.no_grad():
            mmdet_feats = mmdet_model.backbone(x_ref)

        assert len(ref_feats) == len(mmdet_feats)
        for i, (rf, mf) in enumerate(zip(ref_feats, mmdet_feats)):
            assert rf.shape == mf.shape, f"Shape mismatch at level {i}: {rf.shape} vs {mf.shape}"
            pcc = torch.corrcoef(torch.stack([rf.flatten(), mf.flatten()]))[0, 1].item()
            print(f"  Backbone level {i}: PCC = {pcc:.6f}")
            assert pcc > 0.99, f"PCC too low at backbone level {i}: {pcc:.6f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
