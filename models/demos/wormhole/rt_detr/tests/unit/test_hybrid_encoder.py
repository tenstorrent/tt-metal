# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path
import gc

import torch
import ttnn
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))
REPO_PATH = Path(__file__).parent.parent.parent / "RT-DETR" / "rtdetr_pytorch"
sys.path.insert(0, str(REPO_PATH))

from src.core import YAMLConfig

from tt.hybrid_encoder import hybrid_encoder
from tt.weight_utils import get_encoder_parameters
from models.common.utility_functions import comp_pcc

REF = Path(__file__).parent.parent / "reference_outputs.pt"
CFG_PATH = REPO_PATH / "configs/rtdetr/rtdetr_r50vd_6x_coco.yml"
CKPT = Path(__file__).parent.parent.parent / "weights/rtdetr_r50vd.pth"
PCC_THRESHOLD = 0.97

@pytest.fixture(scope="module")
def ref():
    return torch.load(REF, map_location="cpu")

@pytest.fixture(scope="module")
def device():
    mesh_shape = ttnn.MeshShape(1, 2)
    dev = ttnn.open_mesh_device(mesh_shape, l1_small_size=16384)
    yield dev
    ttnn.close_mesh_device(dev)

@pytest.fixture(scope="module")
def torch_model():
    cfg = YAMLConfig(str(CFG_PATH))
    model = cfg.model
    ckpt = torch.load(str(CKPT), map_location="cpu")
    model.load_state_dict(ckpt["ema"]["module"])
    model.eval()
    return model

@pytest.fixture(scope="module")
def enc_params(torch_model, device):
    return get_encoder_parameters(torch_model, device)


def _to_device(t, device):
    # NCHW -> NHWC
    t_nhwc = t.permute(0, 2, 3, 1).contiguous()
    
    return ttnn.from_torch(
        t_nhwc, 
        dtype=ttnn.bfloat16, 
        layout=ttnn.TILE_LAYOUT,
        device=device, 
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )


class TestHybridEncoder:
    def teardown_method(self, method):
        """Force Python to clean up tensors to prevent TTNN memory fragmentation."""
        gc.collect()
    def _run(self, ref, enc_params, device):
        s3 = _to_device(ref["backbone_s3"], device)
        s4 = _to_device(ref["backbone_s4"], device)
        s5 = _to_device(ref["backbone_s5"], device)
        return hybrid_encoder(s3, s4, s5, enc_params, device)

    def test_output_count(self, ref, enc_params, device):
        out = self._run(ref, enc_params, device)
        assert len(out) == 3, f"expected 3 output feature maps, got {len(out)}"

    def test_pcc_p3(self, ref, enc_params, device):
        p3_tt, _, _ = self._run(ref, enc_params, device)
        
        # 1. Pull from mesh and isolate the batch
        p3 = ttnn.to_torch(p3_tt, mesh_composer=ttnn.ConcatMeshToTensor(device, dim=0))[0:1]
        
        # 2. Reshape from (1, 1, 6400, 256) to (1, 80, 80, 256)
        # 3. Permute from NHWC to NCHW (1, 256, 80, 80)
        p3 = p3.reshape(1, 80, 80, 256).permute(0, 3, 1, 2)

        pcc, msg = comp_pcc(ref["encoder_p3"], p3, PCC_THRESHOLD)
        print(f"\nencoder p3 PCC: {pcc:.6f}")
        assert pcc >= PCC_THRESHOLD, f"encoder p3 PCC {pcc:.4f} < {PCC_THRESHOLD} - {msg}"

    def test_pcc_p4(self, ref, enc_params, device):
        _, p4_tt, _ = self._run(ref, enc_params, device)
        
        p4 = ttnn.to_torch(p4_tt, mesh_composer=ttnn.ConcatMeshToTensor(device, dim=0))[0:1]
        
        # Reshape to 40x40 and permute to NCHW
        p4 = p4.reshape(1, 40, 40, 256).permute(0, 3, 1, 2)

        pcc, msg = comp_pcc(ref["encoder_p4"], p4, PCC_THRESHOLD)
        print(f"\nencoder p4 PCC: {pcc:.6f}")
        assert pcc >= PCC_THRESHOLD, f"encoder p4 PCC {pcc:.4f} < {PCC_THRESHOLD} - {msg}"

    def test_pcc_p5(self, ref, enc_params, device):
        _, _, p5_tt = self._run(ref, enc_params, device)
        
        p5 = ttnn.to_torch(p5_tt, mesh_composer=ttnn.ConcatMeshToTensor(device, dim=0))[0:1]
        
        # Reshape to 20x20 and permute to NCHW
        p5 = p5.reshape(1, 20, 20, 256).permute(0, 3, 1, 2)

        pcc, msg = comp_pcc(ref["encoder_p5"], p5, PCC_THRESHOLD)
        print(f"\nencoder p5 PCC: {pcc:.6f}")
        assert pcc >= PCC_THRESHOLD, f"encoder p5 PCC {pcc:.4f} < {PCC_THRESHOLD} - {msg}"

    def test_all_scales_differ(self, ref, enc_params, device):
        p3_tt, p4_tt, p5_tt = self._run(ref, enc_params, device)
        p3 = ttnn.to_torch(p3_tt, mesh_composer=ttnn.ConcatMeshToTensor(device, dim=0))[0:1]
        p4 = ttnn.to_torch(p4_tt, mesh_composer=ttnn.ConcatMeshToTensor(device, dim=0))[0:1]
        p5 = ttnn.to_torch(p5_tt, mesh_composer=ttnn.ConcatMeshToTensor(device, dim=0))[0:1]

        assert p3.shape != p4.shape, "p3 and p4 have the same shape - CCFM spatial dim bug"
        assert p4.shape != p5.shape, "p4 and p5 have the same shape - CCFM spatial dim bug"

if __name__ == "__main__":

    mesh_shape = ttnn.MeshShape(1, 2)
    dev = ttnn.open_mesh_device(mesh_shape, l1_small_size=16384)
    try:
        r = torch.load(REF, map_location="cpu")
        cfg = YAMLConfig(str(CFG_PATH))
        m = cfg.model
        m.load_state_dict(torch.load(str(CKPT), map_location="cpu")["ema"]["module"])
        m.eval()
        
        # Generate the params
        params = get_encoder_parameters(m, dev)
        
        t = TestHybridEncoder()
        t.test_output_count(r, params, dev)
        t.test_pcc_p3(r, dev)
        t.test_pcc_p4(r, dev)
        t.test_pcc_p5(r, dev)
        t.test_all_scales_differ(r, dev)
        print("passed")
    finally:
        ttnn.close_device(dev)