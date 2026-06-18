# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# PCC test for tt/moe/moe_parallel.py HunyuanTtMoEParallel — expert-parallel MoE.
# Compares the sharded MoE (experts split across a 1x2 mesh, resident) against
# the already-validated dense HunyuanTtMoE run replicated on the same mesh, with
# identical replicated input. Same gate/expert code on both sides, so any gap
# isolates the sharding + all-reduce combine. Both bf16 here (sharding check);
# bf8 residency is exercised separately.
#
# Run:
#   python_env/bin/python -m pytest \
#     models/experimental/hunyuan_image_3_0/tests/pcc/test_moe_parallel.py -v -s --timeout=900

import os, sys, json, glob
import torch
from safetensors import safe_open
import pytest
from loguru import logger

ROOT = "/home/iguser/Christy/tt-metal"
WEIGHTS = "/home/iguser/Christy/HunyuanImage-3"
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import ttnn
from models.common.utility_functions import comp_pcc
from models.tt_dit.parallel.manager import CCLManager
from models.experimental.hunyuan_image_3_0.tt.moe.moe import HunyuanTtMoE
from models.experimental.hunyuan_image_3_0.tt.moe.moe_parallel import HunyuanTtMoEParallel

LAYER = 0
PREFIX = f"model.layers.{LAYER}.mlp"
PCC_THR = 0.99

_WMAP = json.load(open(glob.glob(f"{WEIGHTS}/*.index.json")[0]))["weight_map"]
_OPEN = {}


def _load(key):
    shard = _WMAP[key]
    f = _OPEN.get(shard) or _OPEN.setdefault(shard, safe_open(f"{WEIGHTS}/{shard}", framework="pt"))
    return f.get_tensor(key)


def _layer_mlp_sd():
    return {k: _load(k) for k in _WMAP if k.startswith(PREFIX + ".")}


def _cfg():
    c = json.load(open(f"{WEIGHTS}/config.json"))
    first = lambda v: v if isinstance(v, int) else v[0]
    return dict(
        H=c["hidden_size"],
        E=first(c["num_experts"]),
        K=first(c["moe_topk"]),
        NORM=c.get("norm_topk_prob", True),
        MIXED=c.get("use_mixed_mlp_moe", True),
    )


@pytest.fixture(scope="function")
def device_params(request):
    return {"fabric_config": ttnn.FabricConfig.FABRIC_1D}


# 2x2 mesh: 4-way EP (16 experts/device across both axes), the recommended layout.
@pytest.mark.parametrize("mesh_device", [(2, 2)], indirect=True)
def test_moe_parallel_vs_dense(mesh_device):
    mesh_device.enable_program_cache()
    c = _cfg()
    sd = _layer_mlp_sd()
    H, E, K = c["H"], c["E"], c["K"]

    torch.manual_seed(0)
    B, S = 1, 32
    x = torch.randn(B, S, H) * 0.05
    x_tt = ttnn.from_torch(
        x,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    # Dense (replicated) reference on the mesh.
    dense = HunyuanTtMoE(
        mesh_device,
        H,
        E,
        K,
        sd,
        PREFIX,
        use_mixed_mlp_moe=c["MIXED"],
        norm_topk_prob=c["NORM"],
        weight_dtype=ttnn.bfloat16,
        stream_experts=False,
    )
    y_dense = dense.forward(x_tt)

    # Expert-parallel (sharded) on the mesh.
    # Sharded-expert dtype: bf16 isolates sharding correctness; bf8 (set
    # HY_MOE_DTYPE=bf8) exercises the realized residency win (experts ~1 byte,
    # sharded 4-way) and its accuracy cost vs the bf16 dense reference.
    par_dtype = ttnn.bfloat8_b if os.environ.get("HY_MOE_DTYPE", "bf16") == "bf8" else ttnn.bfloat16
    thr = 0.97 if par_dtype == ttnn.bfloat8_b else PCC_THR
    ccl = CCLManager(mesh_device, num_links=1, topology=ttnn.Topology.Linear)
    par = HunyuanTtMoEParallel(
        mesh_device,
        ccl,
        sd,
        PREFIX,
        num_experts=E,
        hidden_size=H,
        moe_topk=K,
        norm_topk_prob=c["NORM"],
        use_mixed_mlp_moe=c["MIXED"],
        mesh_axis=1,
        weight_dtype=par_dtype,
    )
    y_par = par.forward(x_tt)

    # Compare device-0 slices.
    d0 = ttnn.to_torch(y_dense, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[:B].float()
    p0 = ttnn.to_torch(y_par, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[:B].float()

    passing, pcc = comp_pcc(d0, p0, thr)
    logger.info(
        f"moe_parallel({par_dtype}) vs dense(bf16) PCC: {pcc:.6f}  shapes d={tuple(d0.shape)} p={tuple(p0.shape)}"
    )
    assert passing, f"PCC {pcc:.6f} < {thr}"
