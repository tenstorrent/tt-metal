# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# PCC test: a full mesh-resident decoder layer (expert-parallel bf8 MoE +
# replicated attention/norms) vs the dense decoder layer run replicated on the
# same mesh. Validates the layer integration of HunyuanTtMoEParallel — i.e. a
# decoder layer that fits via expert sharding + bf8. See MEMORY_FIT_PLAN.md.
#
# Run:
#   python_env/bin/python -m pytest \
#     models/experimental/hunyuan_image_3_0/tests/pcc/test_decoder_layer_parallel.py -v -s --timeout=1700

import sys
import json
import glob
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
from models.experimental.hunyuan_image_3_0.tt.transformer_layer import HunyuanTtDecoderLayer

LAYER = 0
PCC_THR = 0.97  # bf8 parallel vs bf16 dense

_WMAP = json.load(open(glob.glob(f"{WEIGHTS}/*.index.json")[0]))["weight_map"]
_OPEN = {}


def _load(key):
    shard = _WMAP[key]
    f = _OPEN.get(shard) or _OPEN.setdefault(shard, safe_open(f"{WEIGHTS}/{shard}", framework="pt"))
    return f.get_tensor(key)


def _layer_sd():
    pre = f"model.layers.{LAYER}."
    return {k: _load(k) for k in _WMAP if k.startswith(pre)}


def _cfg():
    c = json.load(open(f"{WEIGHTS}/config.json"))
    first = lambda v: v if isinstance(v, int) else v[0]
    return dict(
        H=c["hidden_size"],
        HEADS=c["num_attention_heads"],
        KV=c.get("num_key_value_heads", c["num_attention_heads"]),
        HD=c.get("attention_head_dim", c["hidden_size"] // c["num_attention_heads"]),
        E=first(c["num_experts"]),
        K=first(c["moe_topk"]),
        NORM=c.get("norm_topk_prob", True),
        MIXED=c.get("use_mixed_mlp_moe", True),
        QKN=c.get("use_qk_norm", True),
        EPS=c.get("rms_norm_eps", 1e-5),
    )


@pytest.fixture(scope="function")
def device_params(request):
    return {"fabric_config": ttnn.FabricConfig.FABRIC_1D}


def _build(device, sd, c, dtype, ccl, tp_factor=1):
    return HunyuanTtDecoderLayer(
        device,
        sd,
        layer_num=LAYER,
        hidden_size=c["H"],
        num_heads=c["HEADS"],
        num_kv_heads=c["KV"],
        head_dim=c["HD"],
        num_experts=c["E"],
        moe_topk=c["K"],
        use_qk_norm=c["QKN"],
        use_mixed_mlp_moe=c["MIXED"],
        norm_topk_prob=c["NORM"],
        rms_norm_eps=c["EPS"],
        weight_dtype=dtype,
        stream_experts=False,
        ccl_manager=ccl,
        expert_mesh_axis=1,
        tp_axis=1,
        tp_factor=tp_factor,
    )


# 2x2 sp0tp1 layout: TP=2 on axis 1 + 4-way EP across the mesh. (SP shards the
# sequence and lives in HunyuanTtModel.forward, so it is exercised by the full-model
# tests / test_parallel_2x2.py, not by this single-layer component test.)
@pytest.mark.parametrize("mesh_device", [(2, 2)], indirect=True)
def test_decoder_layer_parallel_vs_dense(mesh_device):
    mesh_device.enable_program_cache()
    c = _cfg()
    sd = _layer_sd()
    H = c["H"]
    B, S = 1, 32

    torch.manual_seed(0)
    x = torch.randn(B, S, H) * 0.05

    def _upload():
        # The layer's forward deallocates its input (residual), so each call needs
        # its own copy.
        return ttnn.from_torch(
            x,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    dense = _build(mesh_device, sd, c, ttnn.bfloat16, None)
    y_dense = dense.forward(_upload(), seq_len=S, image_infos=None, attention_mask=None)

    ccl = CCLManager(mesh_device, num_links=1, topology=ttnn.Topology.Linear)
    par = _build(mesh_device, sd, c, ttnn.bfloat8_b, ccl, tp_factor=2)
    y_par = par.forward(_upload(), seq_len=S, image_infos=None, attention_mask=None)

    d0 = ttnn.to_torch(y_dense, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[:B].float()
    p0 = ttnn.to_torch(y_par, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[:B].float()

    passing, pcc = comp_pcc(d0, p0, PCC_THR)
    logger.info(f"decoder-layer parallel(bf8) vs dense(bf16) PCC: {pcc:.6f}  shape={tuple(p0.shape)}")
    assert passing, f"PCC {pcc:.6f} < {PCC_THR}"
