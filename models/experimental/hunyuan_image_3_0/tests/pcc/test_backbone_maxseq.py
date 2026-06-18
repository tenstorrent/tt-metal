# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Max-sequence-length scale test: run the RESIDENT sharded backbone at the
# config's max_position_embeddings (22800, tile-aligned down). Validates that
# attention / 2D-RoPE / MoE / the resident bf8 sharded weights handle the model's
# full context window without OOM, producing finite output. Text-only causal path
# (no image span) — this is a scale/memory check, not a PCC gate.
#
# Run (default 2 layers; bump HY_NUM_LAYERS for depth):
#   python_env/bin/python -m pytest \
#     models/experimental/hunyuan_image_3_0/tests/pcc/test_backbone_maxseq.py -v -s --timeout=3000

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
from models.tt_dit.parallel.manager import CCLManager
from models.experimental.hunyuan_image_3_0.tt.model import HunyuanTtModel

NUM_LAYERS = int(os.environ.get("HY_NUM_LAYERS", "2"))

_WMAP = json.load(open(glob.glob(f"{WEIGHTS}/*.index.json")[0]))["weight_map"]
_OPEN = {}


def _load(key):
    shard = _WMAP[key]
    f = _OPEN.get(shard) or _OPEN.setdefault(shard, safe_open(f"{WEIGHTS}/{shard}", framework="pt"))
    return f.get_tensor(key)


def _load_prefix(prefix):
    return {k[len(prefix) + 1 :]: _load(k) for k in _WMAP if k.startswith(prefix + ".")}


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
        MAX_SEQ=int(c["max_position_embeddings"]),
    )


def _dram_gb(mesh_device):
    try:
        v = ttnn.get_memory_view(mesh_device, ttnn.BufferType.DRAM)
        return v.num_banks * v.total_bytes_allocated_per_bank / 1e9
    except Exception:
        return None


@pytest.fixture(scope="function")
def device_params(request):
    return {"fabric_config": ttnn.FabricConfig.FABRIC_1D}


# Recommended full-QB2 layout: 2x2 mesh, SP=2 (axis 0) + TP=2 (axis 1) + 4-way EP.
# Long-sequence stress test — the primary motivation for SP (sequence sharding).
@pytest.mark.parametrize("mesh_device", [(2, 2)], indirect=True)
def test_backbone_max_seq(mesh_device):
    mesh_device.enable_program_cache()
    c = _cfg()
    H = c["H"]
    B = 1
    S = (c["MAX_SEQ"] // 32) * 32  # tile-aligned <= max_position_embeddings
    logger.info(f"max_position_embeddings={c['MAX_SEQ']} -> testing S={S} ({NUM_LAYERS} layers, bf8 resident)")

    ccl = CCLManager(mesh_device, num_links=1, topology=ttnn.Topology.Linear)
    layer_loader = lambda i: {f"model.layers.{i}.{k}": v for k, v in _load_prefix(f"model.layers.{i}").items()}
    model = HunyuanTtModel(
        mesh_device,
        num_layers=NUM_LAYERS,
        hidden_size=H,
        num_heads=c["HEADS"],
        num_kv_heads=c["KV"],
        head_dim=c["HD"],
        num_experts=c["E"],
        moe_topk=c["K"],
        use_qk_norm=c["QKN"],
        use_mixed_mlp_moe=c["MIXED"],
        norm_topk_prob=c["NORM"],
        rms_norm_eps=c["EPS"],
        stream_experts=False,
        layer_loader=layer_loader,
        apply_final_norm=False,
        weight_dtype=ttnn.bfloat8_b,
        ccl_manager=ccl,
        expert_mesh_axis=1,
        tp_axis=1,
        tp_factor=2,
        sp_axis=0,
        sp_factor=2,
    )

    dram = _dram_gb(mesh_device)
    x = torch.randn(B, S, H) * 0.02
    x_tt = ttnn.from_torch(
        x,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    # SP needs an explicit (query-sharded) mask; supply a plain causal one.
    m = torch.triu(torch.full((S, S), -1.0e30), diagonal=1).reshape(B, 1, S, S)
    mask_tt = ttnn.from_torch(
        m,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    out = model.forward(inputs_embeds=x_tt, seq_len=S, image_infos=None, attention_mask=mask_tt)
    y = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[:B].float()

    msg = f"max-seq forward OK: S={S} out={tuple(y.shape)} finite={bool(torch.isfinite(y).all())}"
    if dram:
        msg += f"  DRAM/device~{dram:.2f}GB"
    logger.info(msg)
    assert y.shape == (B, S, H)
    assert torch.isfinite(y).all(), "non-finite output at max seq"
