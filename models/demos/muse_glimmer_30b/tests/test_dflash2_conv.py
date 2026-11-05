# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Device test: DFlash2 grouped dynamic causal conv vs the float32 torch reference (real drafter weights)."""

import os

import pytest
import torch
from loguru import logger
from safetensors import safe_open

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.muse_glimmer_30b.reference.dflash2 import GroupedDynamicCausalConv
from models.demos.muse_glimmer_30b.tt.mesh_utils import mesh_test_params, skip_and_submesh
from models.demos.muse_glimmer_30b.tt.model_config import dflash_checkpoint_dir

_CKPT = os.path.join(dflash_checkpoint_dir(), "model.safetensors")
pytestmark = pytest.mark.skipif(not os.path.exists(_CKPT), reason="DFlash2 checkpoint not available")

PCC = 0.999
H = 6656
ROWS = 16

_mesh_params = mesh_test_params()


def _to_tt(x, mesh_device):
    return ttnn.from_torch(
        x,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _to_torch(t, mesh_device):
    out = ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    return out[0:1].float()  # replicated: take rank 0 -> [1, 1, S, H]


@_mesh_params
@pytest.mark.timeout(900)
@pytest.mark.parametrize("prefix", ["layers.0.attention_conv", "layers.4.mlp_conv"])
def test_dynamic_causal_conv(mesh_device, tp, prefix):
    sub = skip_and_submesh(mesh_device, tp)
    from models.demos.muse_glimmer_30b.tt.dflash.conv import TtDynamicCausalConv

    with safe_open(_CKPT, framework="pt") as f:
        base = f.get_tensor(f"{prefix}.base_kernel").float()
        proj = f.get_tensor(f"{prefix}.kernel_projection.weight").float()

    ref = GroupedDynamicCausalConv(H, 2, 16).float()
    ref.load_state_dict({"base_kernel": base, "kernel_projection.weight": proj})
    ref.eval()

    torch.manual_seed(0)
    h = torch.randn(1, 32, H)
    x = torch.randn(1, 32, H)
    with torch.no_grad():
        ref_out, ref_dyn = ref.prepare(h[:, :ROWS])
        ref_fin = ref.finish(x[:, :ROWS], ref_dyn)

    conv = TtDynamicCausalConv(sub, base, proj, group_size=16, block_rows=ROWS)
    h_tt = _to_tt(h.bfloat16().view(1, 1, 32, H), sub)
    x_tt = _to_tt(x.bfloat16().view(1, 1, 32, H), sub)
    out, dyn = conv.prepare(h_tt)
    fin = conv.finish(x_tt, dyn)

    out_t = _to_torch(out, sub)[0, 0, :ROWS]
    fin_t = _to_torch(fin, sub)[0, 0, :ROWS]
    ok1, msg1 = comp_pcc(ref_out[0], out_t, PCC)
    ok2, msg2 = comp_pcc(ref_fin[0], fin_t, PCC)
    logger.info(f"{prefix} prepare: {msg1}")
    logger.info(f"{prefix} finish: {msg2}")
    assert ok1, f"prepare: {msg1}"
    assert ok2, f"finish: {msg2}"
