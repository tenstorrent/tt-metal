# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Residency proof: build an N-layer backbone with expert-parallel bf8 MoE,
# RESIDENT (sharded across the (1,4) mesh, no host streaming), run it, and report
# per-device DRAM. A dense/replicated backbone of this depth would OOM
# (~4.8 GB/layer/device); the sharded bf8 version is ~0.6 GB/layer/device, so
# running it at depth IS the demonstration that the model path fits. See
# MEMORY_FIT_PLAN.md.
#
# Run (default 8 layers):
#   python_env/bin/python -m pytest \
#     models/experimental/hunyuan_image_3_0/tests/pcc/test_backbone_resident.py -v -s --timeout=1800
#   HY_NUM_LAYERS=16 python_env/bin/python -m pytest ... (deeper residency check)

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

NUM_LAYERS = int(os.environ.get("HY_NUM_LAYERS", "8"))

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
    )


def _dram_used_bytes(mesh_device):
    """Per-device allocated DRAM (bytes)."""
    try:
        v = ttnn.get_memory_view(mesh_device, ttnn.BufferType.DRAM)
        return int(v.num_banks) * int(v.total_bytes_allocated_per_bank)
    except Exception as e:  # API shape varies; fall back to analytical reporting
        logger.info(f"(DRAM view unavailable: {e})")
        return None


@pytest.fixture(scope="function")
def device_params(request):
    return {"fabric_config": ttnn.FabricConfig.FABRIC_1D}


@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
def test_backbone_resident(mesh_device):
    mesh_device.enable_program_cache()
    c = _cfg()
    H = c["H"]
    B, S = 1, 32
    ndev = mesh_device.shape[1]

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
        weight_dtype=ttnn.bfloat8_b,
        stream_experts=False,
        layer_loader=layer_loader,
        apply_final_norm=False,
        ccl_manager=ccl,
        expert_mesh_axis=1,
    )

    dram = _dram_used_bytes(mesh_device)
    exp_per_layer_gb = c["E"] * 3 * H * 3072 * 1 / ndev / 1e9  # bf8, sharded
    logger.info(
        f"RESIDENT backbone built: {NUM_LAYERS} layers, bf8, experts sharded {ndev}-way. "
        f"Analytical expert mem/device ~{exp_per_layer_gb * NUM_LAYERS:.2f} GB"
        + (f"; measured DRAM/device ~{dram / 1e9:.2f} GB" if dram else "")
    )

    x = torch.randn(B, S, H) * 0.02
    x_tt = ttnn.from_torch(
        x,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    out = model.forward(inputs_embeds=x_tt, seq_len=S, image_infos=None, attention_mask=None)
    y = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[:B].float()

    logger.info(f"forward OK: out shape={tuple(y.shape)} finite={bool(torch.isfinite(y).all())}")
    assert y.shape == (B, S, H)
    assert torch.isfinite(y).all(), "non-finite output"
