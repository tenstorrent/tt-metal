# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

import torch
import ttnn
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

REPO_PATH = Path(__file__).parent.parent.parent / "RT-DETR" / "rtdetr_pytorch"
sys.path.insert(0, str(REPO_PATH))

from src.core import YAMLConfig

from tt.resnet_backbone import presnet50
from tt.weight_utils import get_backbone_parameters
from models.common.utility_functions import comp_pcc

REF      = Path(__file__).parent.parent / "reference_outputs.pt"
CFG_PATH = REPO_PATH / "configs/rtdetr/rtdetr_r50vd_6x_coco.yml"
CKPT     = Path(__file__).parent.parent.parent / "weights/rtdetr_r50vd.pth"

PCC_THRESHOLD = 0.97


@pytest.fixture(scope="module")
def ref():
    return torch.load(REF, map_location="cpu")


@pytest.fixture(scope="module")
def device():
    dev = ttnn.open_device(device_id=0, l1_small_size=16384)
    yield dev
    ttnn.close_device(dev)


@pytest.fixture(scope="module")
def torch_model():
    cfg   = YAMLConfig(str(CFG_PATH))
    model = cfg.model
    ckpt  = torch.load(str(CKPT), map_location="cpu")
    model.load_state_dict(ckpt["ema"]["module"])
    model.eval()
    return model


@pytest.fixture(scope="module")
def backbone_params(torch_model, device):

    return get_backbone_parameters(torch_model, device)


def _to_device(t, device):
    # ttnn.conv2d expects (N, H, W, C) ROW_MAJOR — permute on CPU first
    t_nhwc = t.permute(0, 2, 3, 1).contiguous()  # (1,3,640,640) -> (1,640,640,3)
    return ttnn.from_torch(
        t_nhwc,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _tt_to_nchw(tt_tensor, h, w):
    t = ttnn.to_torch(tt_tensor).squeeze(0).squeeze(0)  # (H*W, C)
    c = t.shape[-1]
    return t.reshape(h, w, c).permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)


class TestResNet50Backbone:
    
    def test_pcc_s3(self, ref, device, backbone_params):
        x_tt = _to_device(ref["backbone_input"], device)
        s3_tt, _, _ = presnet50(x_tt, backbone_params, device)
        s3 = _tt_to_nchw(s3_tt, 80, 80)

        # compare stats before PCC
        ref_s3 = ref["backbone_s3"]
        print(f"\nref  s3: shape={tuple(ref_s3.shape)} min={ref_s3.min():.3f} max={ref_s3.max():.3f} mean={ref_s3.mean():.3f}")
        print(f"ttnn s3: shape={tuple(s3.shape)}     min={s3.min():.3f}     max={s3.max():.3f}     mean={s3.mean():.3f}")

        passing, pcc_msg = comp_pcc(ref_s3, s3, PCC_THRESHOLD)
        print(f"backbone s3 PCC: {pcc_msg}")
        assert passing, f"s3 PCC below threshold — {pcc_msg}"

    def test_pcc_s4(self, ref, device, backbone_params):
        x_tt = _to_device(ref["backbone_input"], device)
        _, s4_tt, _ = presnet50(x_tt, backbone_params, device)
        s4 = _tt_to_nchw(s4_tt, 40, 40)

        passing, pcc_msg = comp_pcc(ref["backbone_s4"], s4, PCC_THRESHOLD)
        print(f"\nbackbone s4 PCC: {pcc_msg}")
        assert passing, f"s4 PCC below threshold — {pcc_msg}"

    def test_pcc_s5(self, ref, device, backbone_params):
        x_tt = _to_device(ref["backbone_input"], device)
        _, _, s5_tt = presnet50(x_tt, backbone_params, device)
        s5 = _tt_to_nchw(s5_tt, 20, 20)

        passing, pcc_msg = comp_pcc(ref["backbone_s5"], s5, PCC_THRESHOLD)
        print(f"\nbackbone s5 PCC: {pcc_msg}")
        assert passing, f"s5 PCC below threshold — {pcc_msg}"

    def test_pcc_s5(self, ref, device, backbone_params):
        x_tt = _to_device(ref["backbone_input"], device)
        _, _, s5_tt = presnet50(x_tt, backbone_params, device)
        s5 = _tt_to_nchw(s5_tt, 20, 20)

        passing, pcc_msg = comp_pcc(ref["backbone_s5"], s5, PCC_THRESHOLD)
        print(f"\nbackbone s5 PCC: {pcc_msg}")
        assert passing, f"s5 PCC below threshold — {pcc_msg}"
    
    def test_stage0_pcc(self, ref, device, backbone_params):
        from tt.resnet_backbone import _stem, _stage
        import torch.nn.functional as F

        # run TTNN stem + stage0
        x_tt = _to_device(ref["backbone_input"], device)
        x_tt, h, w = _stem(x_tt, backbone_params, device)
        x_tt, h, w = _stage(x_tt, backbone_params.stages[0], device, 1, h, w)
        stage0_out = _tt_to_nchw(x_tt, h, w)  # expect (1, 256, 160, 160)

        # CPU reference
        cfg_ref = YAMLConfig(str(CFG_PATH))
        m = cfg_ref.model
        ck = torch.load(str(CKPT), map_location="cpu")
        m.load_state_dict(ck["ema"]["module"])
        m.eval()

        with torch.no_grad():
            x = ref["backbone_input"]
            stem_ref = m.backbone.conv1(x)
            pool_ref = F.max_pool2d(stem_ref, kernel_size=3, stride=2, padding=1)
            stage0_ref = m.backbone.res_layers[0](pool_ref)  # (1, 256, 160, 160)

        print(f"\nref  stage0: shape={tuple(stage0_ref.shape)} "
            f"min={stage0_ref.min():.3f} max={stage0_ref.max():.3f} mean={stage0_ref.mean():.3f}")
        print(f"ttnn stage0: shape={tuple(stage0_out.shape)} "
            f"min={stage0_out.min():.3f} max={stage0_out.max():.3f} mean={stage0_out.mean():.3f}")

        diff = torch.abs(stage0_ref - stage0_out)
        print(f"Max difference:  {diff.max().item():.6f}")
        print(f"Mean difference: {diff.mean().item():.6f}")
        print(f"99% of errors are under: {torch.quantile(diff.float(), 0.99).item():.6f}")

        passing, pcc_msg = comp_pcc(stage0_ref, stage0_out, 0.97)
        print(f"stage0 PCC: {pcc_msg}")
        assert passing, f"stage0 PCC failed — {pcc_msg}"

    def test_stage1_pcc(self, ref, device, backbone_params):
        from tt.resnet_backbone import _stem, _stage
        import torch.nn.functional as F

        x_tt = _to_device(ref["backbone_input"], device)
        x_tt, h, w = _stem(x_tt, backbone_params, device)
        x_tt, h, w = _stage(x_tt, backbone_params.stages[0], device, 1, h, w)
        x_tt, h, w = _stage(x_tt, backbone_params.stages[1], device, 2, h, w)
        out = _tt_to_nchw(x_tt, h, w)  # expect (1, 512, 80, 80)

        cfg_ref = YAMLConfig(str(CFG_PATH))
        m = cfg_ref.model
        m.load_state_dict(torch.load(str(CKPT), map_location="cpu")["ema"]["module"])
        m.eval()
        with torch.no_grad():
            x = ref["backbone_input"]
            p = F.max_pool2d(m.backbone.conv1(x), 3, stride=2, padding=1)
            r0 = m.backbone.res_layers[0](p)
            ref_out = m.backbone.res_layers[1](r0)

        print(f"\nref  s1: min={ref_out.min():.3f} max={ref_out.max():.3f} mean={ref_out.mean():.3f}")
        print(f"ttnn s1: min={out.min():.3f} max={out.max():.3f} mean={out.mean():.3f}")
        passing, pcc_msg = comp_pcc(ref_out, out, 0.97)
        print(f"stage1 PCC: {pcc_msg}")
        assert passing, f"stage1 PCC failed — {pcc_msg}"


    def test_stage2_pcc(self, ref, device, backbone_params):
        from tt.resnet_backbone import _stem, _stage
        import torch.nn.functional as F

        x_tt = _to_device(ref["backbone_input"], device)
        x_tt, h, w = _stem(x_tt, backbone_params, device)
        x_tt, h, w = _stage(x_tt, backbone_params.stages[0], device, 1, h, w)
        x_tt, h, w = _stage(x_tt, backbone_params.stages[1], device, 2, h, w)
        x_tt, h, w = _stage(x_tt, backbone_params.stages[2], device, 2, h, w)
        out = _tt_to_nchw(x_tt, h, w)  # expect (1, 1024, 40, 40)

        cfg_ref = YAMLConfig(str(CFG_PATH))
        m = cfg_ref.model
        m.load_state_dict(torch.load(str(CKPT), map_location="cpu")["ema"]["module"])
        m.eval()
        with torch.no_grad():
            x = ref["backbone_input"]
            p  = F.max_pool2d(m.backbone.conv1(x), 3, stride=2, padding=1)
            r0 = m.backbone.res_layers[0](p)
            r1 = m.backbone.res_layers[1](r0)
            ref_out = m.backbone.res_layers[2](r1)

        print(f"\nref  s2: min={ref_out.min():.3f} max={ref_out.max():.3f} mean={ref_out.mean():.3f}")
        print(f"ttnn s2: min={out.min():.3f} max={out.max():.3f} mean={out.mean():.3f}")
        passing, pcc_msg = comp_pcc(ref_out, out, 0.97)
        print(f"stage2 PCC: {pcc_msg}")
        assert passing, f"stage2 PCC failed — {pcc_msg}"


    def test_stage3_pcc(self, ref, device, backbone_params):
        from tt.resnet_backbone import _stem, _stage
        import torch.nn.functional as F

        x_tt = _to_device(ref["backbone_input"], device)
        x_tt, h, w = _stem(x_tt, backbone_params, device)
        x_tt, h, w = _stage(x_tt, backbone_params.stages[0], device, 1, h, w)
        x_tt, h, w = _stage(x_tt, backbone_params.stages[1], device, 2, h, w)
        x_tt, h, w = _stage(x_tt, backbone_params.stages[2], device, 2, h, w)
        x_tt, h, w = _stage(x_tt, backbone_params.stages[3], device, 2, h, w)
        out = _tt_to_nchw(x_tt, h, w)  # expect (1, 2048, 20, 20)

        cfg_ref = YAMLConfig(str(CFG_PATH))
        m = cfg_ref.model
        m.load_state_dict(torch.load(str(CKPT), map_location="cpu")["ema"]["module"])
        m.eval()
        with torch.no_grad():
            x = ref["backbone_input"]
            p  = F.max_pool2d(m.backbone.conv1(x), 3, stride=2, padding=1)
            r0 = m.backbone.res_layers[0](p)
            r1 = m.backbone.res_layers[1](r0)
            r2 = m.backbone.res_layers[2](r1)
            ref_out = m.backbone.res_layers[3](r2)

        print(f"\nref  s3: min={ref_out.min():.3f} max={ref_out.max():.3f} mean={ref_out.mean():.3f}")
        print(f"ttnn s3: min={out.min():.3f} max={out.max():.3f} mean={out.mean():.3f}")
        passing, pcc_msg = comp_pcc(ref_out, out, 0.97)
        print(f"stage3 PCC: {pcc_msg}")
        assert passing, f"stage3 PCC failed — {pcc_msg}"


if __name__ == "__main__":
    dev = ttnn.open_device(device_id=0, l1_small_size=16384)
    try:
        r   = torch.load(REF, map_location="cpu")
        cfg = YAMLConfig(str(CFG_PATH))
        m   = cfg.model
        ck  = torch.load(str(CKPT), map_location="cpu")
        m.load_state_dict(ck["ema"]["module"])
        m.eval()

        params = get_backbone_parameters(m, dev)

        t = TestResNet50Backbone()
        t.test_output_shapes(r, dev, params)
        t.test_pcc_s3(r, dev, params)
        t.test_pcc_s4(r, dev, params)
        t.test_pcc_s5(r, dev, params)
        t.test_stages_have_distinct_shapes(r, dev, params)
        t.test_bn_folding_is_applied(m, dev, params)
        print("passed")
    finally:
        ttnn.close_device(dev)