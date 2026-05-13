# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Tests the AIFI encoder stack (run_aifi) against reference_outputs.pt.

import sys
from pathlib import Path

import torch
import ttnn
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

REPO_PATH = Path(__file__).parent.parent.parent / "RT-DETR" / "rtdetr_pytorch"
sys.path.insert(0, str(REPO_PATH))

from src.core import YAMLConfig

from tt.rtdetr_encoder import run_aifi
from tt.weight_utils import get_encoder_parameters
from models.common.utility_functions import comp_pcc

REF      = Path(__file__).parent.parent / "reference_outputs.pt"
CFG_PATH = REPO_PATH / "configs/rtdetr/rtdetr_r50vd_6x_coco.yml"
CKPT     = Path(__file__).parent.parent.parent / "weights/rtdetr_r50vd.pth"

PCC_THRESHOLD = 0.99


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
    cfg   = YAMLConfig(str(CFG_PATH))
    model = cfg.model
    ckpt  = torch.load(str(CKPT), map_location="cpu")
    model.load_state_dict(ckpt["ema"]["module"])
    model.eval()
    return model


@pytest.fixture(scope="module")
def enc_params(torch_model, device):
    return get_encoder_parameters(torch_model, device)


def _to_device(t, device):
    return ttnn.from_torch(
        t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=device, memory_config=ttnn.L1_MEMORY_CONFIG,
    )


class TestEncoderStack:
    def test_aifi_no_pos_embed(self, ref, device, enc_params):
        x_tt = _to_device(ref["aifi_input"], device)

        out_tt = run_aifi(x_tt, enc_params.encoder_layers, device, pos_embed=None)
        out = ttnn.to_torch(out_tt)

        pcc, msg = comp_pcc(ref["aifi_output_no_pos"], out, PCC_THRESHOLD)
        print(f"\nAIFI (no pos_embed) PCC: {pcc:.6f}")
        assert pcc >= PCC_THRESHOLD, f"AIFI no-pos PCC {pcc:.4f} < {PCC_THRESHOLD} - {msg}"

    def test_aifi_with_pos_embed(self, ref, device, enc_params):
        x_tt   = _to_device(ref["aifi_input"],    device)
        pos_tt = _to_device(ref["aifi_pos_embed"], device)

        out_tt = run_aifi(x_tt, enc_params.encoder_layers, device, pos_embed=pos_tt)
        out = ttnn.to_torch(out_tt)

        pcc, msg = comp_pcc(ref["aifi_output_with_pos"], out, PCC_THRESHOLD)
        print(f"\nAIFI (with pos_embed) PCC: {pcc:.6f}")
        assert pcc >= PCC_THRESHOLD, f"AIFI with-pos PCC {pcc:.4f} < {PCC_THRESHOLD} - {msg}"

    def test_aifi_pos_embed_affects_output(self, ref, device, enc_params):
        # pos_embed must visibly change the output - if not it is being dropped
        x_tt   = _to_device(ref["aifi_input"],    device)
        pos_tt = _to_device(ref["aifi_pos_embed"], device)

        out_no_pos   = ttnn.to_torch(run_aifi(x_tt, enc_params.encoder_layers, device, pos_embed=None))
        out_with_pos = ttnn.to_torch(run_aifi(x_tt, enc_params.encoder_layers, device, pos_embed=pos_tt))

        max_diff = (out_no_pos - out_with_pos).abs().max().item()
        print(f"\npos_embed effect max |diff|: {max_diff:.6f}")
        assert max_diff > 1e-3, "pos_embed had no effect - likely not being applied to Q/K"


if __name__ == "__main__":
    mesh_shape = ttnn.MeshShape(1, 2)
    dev = ttnn.open_mesh_device(mesh_shape, l1_small_size=16384)
    try:
        r   = torch.load(REF, map_location="cpu")
        cfg = YAMLConfig(str(CFG_PATH))
        m   = cfg.model
        ck  = torch.load(str(CKPT), map_location="cpu")
        m.load_state_dict(ck["ema"]["module"])
        m.eval()
        from tt.weight_utils import get_encoder_parameters
        params = get_encoder_parameters(m, dev)

        t = TestEncoderStack()
        t.test_aifi_no_pos_embed(r, dev, params)
        t.test_aifi_with_pos_embed(r, dev, params)
        t.test_aifi_pos_embed_affects_output(r, dev, params)
        print("passed")
    finally:
        ttnn.close_mesh_device(dev)