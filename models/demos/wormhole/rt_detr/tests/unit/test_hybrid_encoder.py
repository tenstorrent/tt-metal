# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import sys
from pathlib import Path

import pytest
import torch

import ttnn

rt_detr_root = Path(__file__).parent.parent.parent

repo_path = rt_detr_root / "RT-DETR" / "rtdetr_pytorch"
sys.path.insert(0, str(repo_path))
sys.path.insert(0, str(rt_detr_root))

cfg_path = repo_path / "configs/rtdetr/rtdetr_r50vd_6x_coco.yml"

ckpt_path = rt_detr_root / "weights/rtdetr_r50vd.pth"

from src.core import YAMLConfig
from tt.hybrid_encoder import hybrid_encoder
from tt.weight_utils import get_encoder_parameters

from models.common.utility_functions import comp_pcc

pcc_threshold = 0.90  

_device = None
_pt_p3, _pt_p4, _pt_p5 = None, None, None
_tt_p3, _tt_p4, _tt_p5 = None, None, None


def _to_tt_flat(tensor_nchw, device):
    """Converts PyTorch (N, C, H, W) to TTNN (N, 1, H*W, C) for the encoder."""
    n, c, h, w = tensor_nchw.shape
    t = tensor_nchw.permute(0, 2, 3, 1).reshape(n, 1, h * w, c).contiguous()

    return ttnn.from_torch(
        t,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device) if hasattr(device, "get_num_devices") else None,
    )


def _pull_and_reshape(tt_tensor, device, h, w):
    """Pulls TTNN (N, 1, H*W, C) back to PyTorch (N, C, H, W)."""
    mesh_composer = ttnn.ConcatMeshToTensor(device, dim=0) if hasattr(device, "get_num_devices") else None
    pt_flat = ttnn.to_torch(
        tt_tensor,
        mesh_composer=mesh_composer,
    )[0:1].float()

    # pt_flat is (1, 1, H*W, C)
    return pt_flat.squeeze(1).reshape(1, h, w, -1).permute(0, 3, 1, 2)


def _setup_encoder_pipeline():
    global _device
    global _pt_p3, _pt_p4, _pt_p5
    global _tt_p3, _tt_p4, _tt_p5

    if _device is not None:
        return

    # 1. Initialize Device
    mesh_shape = ttnn.MeshShape(1, 2)
    _device = ttnn.open_mesh_device(mesh_shape, l1_small_size=16384)

    # 2. Load PyTorch Model & Weights
    cfg = YAMLConfig(str(cfg_path))
    torch_model = cfg.model
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    torch_model.load_state_dict(ckpt["ema"]["module"])
    torch_model.eval()

    torch_encoder = torch_model.encoder
    tt_params = get_encoder_parameters(torch_model, _device)

    # 3. Create dummy inputs representing backbone features (ResNet50 dims)
    s3_pt = torch.randn(1, 512, 80, 80)
    s4_pt = torch.randn(1, 1024, 40, 40)
    s5_pt = torch.randn(1, 2048, 20, 20)

    # 4. Run Reference PyTorch Encoder
    print("\n[setup] Running PyTorch reference encoder...")
    with torch.no_grad():
        pt_outs = torch_encoder([s3_pt, s4_pt, s5_pt])
        _pt_p3, _pt_p4, _pt_p5 = pt_outs

    # 5. Convert Inputs for TTNN
    print("[setup] Loading tensors to TT device...")
    s3_tt = _to_tt_flat(s3_pt, _device)
    s4_tt = _to_tt_flat(s4_pt, _device)
    s5_tt = _to_tt_flat(s5_pt, _device)

    # 6. Run TTNN Encoder
    print("[setup] Running TTNN hybrid_encoder...")
    tt_outs = hybrid_encoder(s3_tt, s4_tt, s5_tt, tt_params, _device)
    p3_tt_out, p4_tt_out, p5_tt_out = tt_outs

    # 7. Pull back to PyTorch for comparison
    _tt_p3 = _pull_and_reshape(p3_tt_out, _device, 80, 80)
    _tt_p4 = _pull_and_reshape(p4_tt_out, _device, 40, 40)
    _tt_p5 = _pull_and_reshape(p5_tt_out, _device, 20, 20)

    print("[setup] Done.\n")


@pytest.fixture(scope="module", autouse=True)
def force_pipeline_setup():
    """PyTest fixture to bootstrap device, model, and inferences before tests run."""
    _setup_encoder_pipeline()
    yield
    global _device
    if _device is not None:
        ttnn.close_mesh_device(_device)
        _device = None


def test_hybrid_encoder_p3_pcc():
    pcc, msg = comp_pcc(_pt_p3, _tt_p3, pcc_threshold)
    print(f"\nEncoder P3 (80x80) PCC: {msg}")
    assert pcc, f"P3 PCC below {pcc_threshold} - {msg}"


def test_hybrid_encoder_p4_pcc():
    pcc, msg = comp_pcc(_pt_p4, _tt_p4, pcc_threshold)
    print(f"\nEncoder P4 (40x40) PCC: {msg}")
    assert pcc, f"P4 PCC below {pcc_threshold} - {msg}"


def test_hybrid_encoder_p5_pcc():
    pcc, msg = comp_pcc(_pt_p5, _tt_p5, pcc_threshold)
    print(f"\nEncoder P5 (20x20) PCC: {msg}")
    assert pcc, f"P5 PCC below {pcc_threshold} - {msg}"


if __name__ == "__main__":
    _setup_encoder_pipeline()

    test_hybrid_encoder_p3_pcc()
    test_hybrid_encoder_p4_pcc()
    test_hybrid_encoder_p5_pcc()

    if _device is not None:
        ttnn.close_mesh_device(_device)
