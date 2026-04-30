# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Unit test for column-parallel (TP) MLP vs replicated MLP.

Tests the PRODUCTION TtMolmo2TextMLP (models/demos/molmo2/tt/mlp.py):
  1. PCC ≥ 0.99 and RMS ratio ≈ 1.0 vs float32 PyTorch reference (S=256)
  2. No OOM for S=16384 (replicated MLP OOMs at ~S=10k)

Run:
    MESH_DEVICE=T3K pytest models/demos/molmo2/tests/test_tp_mlp.py -v -s
"""

import os
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

import ttnn
from models.demos.molmo2.tt.mlp import TtMolmo2TextMLP
from models.tt_transformers.tt.common import Mode

HF_PATH = Path(
    os.path.expanduser(
        "~/.cache/huggingface/hub/models--allenai--Molmo2-8B/snapshots/e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b"
    )
)
WEIGHT_CACHE = Path("/tmp/molmo2_weight_cache")
_MESH_SHAPE = {"T3K": (1, 8)}.get(os.environ.get("MESH_DEVICE"), (1, 8))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mesh_device():
    rows, cols = _MESH_SHAPE
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    device = ttnn.open_mesh_device(ttnn.MeshShape(rows, cols))
    yield device
    ttnn.close_mesh_device(device)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


@pytest.fixture(scope="module")
def state_dict_and_cfg(mesh_device):
    from transformers import AutoModelForImageTextToText

    from models.demos.molmo2.tt.model_config import Molmo2Config

    sys.path.insert(0, str(HF_PATH))
    hf = AutoModelForImageTextToText.from_pretrained(
        str(HF_PATH), trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="cpu"
    )
    sd = hf.state_dict()
    del hf
    cfg = Molmo2Config(mesh_device=mesh_device)
    WEIGHT_CACHE.mkdir(parents=True, exist_ok=True)
    return sd, cfg


# ---------------------------------------------------------------------------
# Reference: pure PyTorch SwiGLU MLP
# ---------------------------------------------------------------------------


def ref_mlp(x_bfloat16, state_dict, layer_num):
    ln = f"model.transformer.blocks.{layer_num}.mlp"
    ff_proj = state_dict[f"{ln}.ff_proj.weight"].float()  # [24576, 4096]
    ff_out = state_dict[f"{ln}.ff_out.weight"].float()  # [4096, 12288]
    intermediate_size = 12288
    w_value = ff_proj[:intermediate_size]  # [12288, 4096]
    w_gate = ff_proj[intermediate_size:]  # [12288, 4096]
    x = x_bfloat16.float()
    gate = F.silu(x @ w_gate.T)
    value = x @ w_value.T
    return (gate * value) @ ff_out.T


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pcc(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_tp_mlp_pcc(mesh_device, state_dict_and_cfg):
    """Production TP MLP: PCC ≥ 0.99 AND RMS ratio ≈ 1.0 vs float32 reference (S=256)."""
    sd, cfg = state_dict_and_cfg
    layer_num = 0
    S = 256

    mlp = TtMolmo2TextMLP(
        mesh_device=mesh_device,
        tt_ccl=None,
        state_dict=sd,
        weight_cache_path=WEIGHT_CACHE,
        layer_num=layer_num,
        dtype=ttnn.bfloat16,
        configuration=cfg,
    )

    x_cpu = torch.randn(1, 1, S, 4096, dtype=torch.bfloat16)
    ref = ref_mlp(x_cpu, sd, layer_num)

    x_tt = ttnn.from_torch(
        x_cpu,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    out_tt = mlp.forward(x_tt, mode=Mode.PREFILL)
    out_cpu = ttnn.to_torch(ttnn.get_device_tensors(out_tt)[0]).float()
    ttnn.deallocate(out_tt)

    pcc = _pcc(out_cpu, ref)
    # RMS ratio check: catches the 8× scale bug where replicated weights +
    # AllReduce sums 8 copies (PCC would still be ~1.0 but values are 8× wrong)
    rms_ratio = (out_cpu.norm() / ref.norm()).item()
    print(f"\nTP MLP PCC (S={S}): {pcc:.6f}  RMS ratio: {rms_ratio:.4f}")
    assert pcc >= 0.99, f"PCC {pcc:.4f} < 0.99"
    assert 0.7 <= rms_ratio <= 1.3, f"RMS ratio {rms_ratio:.3f} not in [0.7, 1.3] — possible 8× scaling bug"


def test_tp_mlp_no_oom_large_s(mesh_device, state_dict_and_cfg):
    """Production TP MLP: S=16384 runs without DRAM OOM (replicated would OOM)."""
    sd, cfg = state_dict_and_cfg
    layer_num = 0
    S = 16384

    mlp = TtMolmo2TextMLP(
        mesh_device=mesh_device,
        tt_ccl=None,
        state_dict=sd,
        weight_cache_path=WEIGHT_CACHE,
        layer_num=layer_num,
        dtype=ttnn.bfloat16,
        configuration=cfg,
    )

    x_cpu = torch.zeros(1, 1, S, 4096, dtype=torch.bfloat16)
    x_tt = ttnn.from_torch(
        x_cpu,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    out_tt = mlp.forward(x_tt, mode=Mode.PREFILL)
    out_shape = out_tt.shape
    ttnn.deallocate(out_tt)

    print(f"\nTP MLP S={S}: output shape {out_shape}  — no OOM ✓")
    assert out_shape[-2] == S
    assert out_shape[-1] == cfg.dim
