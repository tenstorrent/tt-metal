# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# On-device (QB2, 2x2 Blackhole) validation of the SP/TP/EP build-out.
#   A. sp_shard/sp_gather reshard primitives round-trip (reduce_scatter-as-scatter).
#   B. Phase 1 — MoE 4-way EP on a 2x2 vs dense replicated.
#   C. Phase 2 — TP=2 attention vs tp_factor=1 (replicated) reference.
#   D. Phase 3 — SP=2 small-stack model vs sp_factor=1 reference.
#
# Run (one at a time while bringing up):
#   python_env/bin/python -m pytest \
#     models/experimental/hunyuan_image_3_0/tests/pcc/test_parallel_2x2.py -v -s \
#     -k reshard --timeout=900

import sys
import json
import glob
import torch
import pytest
from loguru import logger

ROOT = "/home/iguser/Christy/tt-metal"
WEIGHTS = "/home/iguser/Christy/HunyuanImage-3"
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import ttnn
from models.common.utility_functions import comp_pcc
from models.tt_dit.parallel.manager import CCLManager


@pytest.fixture(scope="function")
def device_params(request):
    return {"fabric_config": ttnn.FabricConfig.FABRIC_1D}


_WMAP = json.load(open(glob.glob(f"{WEIGHTS}/*.index.json")[0]))["weight_map"]
_OPEN = {}


def _load(key):
    from safetensors import safe_open

    shard = _WMAP[key]
    f = _OPEN.get(shard) or _OPEN.setdefault(shard, safe_open(f"{WEIGHTS}/{shard}", framework="pt"))
    return f.get_tensor(key)


def _sd(prefix):
    return {k: _load(k) for k in _WMAP if k.startswith(prefix + ".")}


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
        QKN=c.get("use_qk_norm", True),
        NORM=c.get("norm_topk_prob", True),
        MIXED=c.get("use_mixed_mlp_moe", True),
        EPS=c.get("rms_norm_eps", 1e-5),
    )


# --------------------------------------------------------------------------- A
@pytest.mark.parametrize("mesh_device", [(2, 2)], indirect=True)
def test_reshard_roundtrip(mesh_device):
    """sp_shard (replicated->sharded) then sp_gather (->replicated) == identity."""
    from models.experimental.hunyuan_image_3_0.tt.parallel_utils import sp_shard, sp_gather

    mesh_device.enable_program_cache()
    ccl = CCLManager(mesh_device, num_links=1, topology=ttnn.Topology.Linear)
    n, ax = 2, 0  # SP axis 0, factor 2

    B, S, H = 1, 64, 128  # S/n = 32 = 1 tile (alignment OK)
    x = torch.randn(B, S, H)
    x_tt = ttnn.from_torch(
        x,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    sharded = sp_shard(ccl, x_tt, dim=1, mesh_axis=ax, n=n)  # [B, S/2, H] per row
    # Per-device shard should be the row's contiguous seq slice.
    sh = ttnn.to_torch(sharded, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=1))
    logger.info(f"sharded gather shape {tuple(sh.shape)} (expect seq concat across 2 rows + 2 tp replicas)")

    gathered = sp_gather(ccl, sharded, dim=1, mesh_axis=ax, n=n)  # back to [B, S, H]
    g0 = ttnn.to_torch(gathered, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[:B].float()

    passing, pcc = comp_pcc(x, g0, 0.999)
    logger.info(f"reshard round-trip PCC: {pcc:.6f}  shape {tuple(g0.shape)}")
    assert passing, f"reshard round-trip PCC {pcc:.6f} — reduce_scatter/n scatter semantic is wrong"


# --------------------------------------------------------------------------- B
@pytest.mark.parametrize("mesh_device", [(2, 2)], indirect=True)
def test_moe_ep_2x2_vs_dense(mesh_device):
    """Phase 1: 4-way EP (16 experts/device across both axes) vs dense replicated."""
    from models.experimental.hunyuan_image_3_0.tt.moe.moe import HunyuanTtMoE
    from models.experimental.hunyuan_image_3_0.tt.moe.moe_parallel import HunyuanTtMoEParallel

    mesh_device.enable_program_cache()
    c = _cfg()
    prefix = "model.layers.0.mlp"
    sd = _sd(prefix)
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

    dense = HunyuanTtMoE(
        mesh_device,
        H,
        E,
        K,
        sd,
        prefix,
        use_mixed_mlp_moe=c["MIXED"],
        norm_topk_prob=c["NORM"],
        weight_dtype=ttnn.bfloat16,
        stream_experts=False,
    )
    y_dense = dense.forward(x_tt)

    ccl = CCLManager(mesh_device, num_links=1, topology=ttnn.Topology.Linear)
    par = HunyuanTtMoEParallel(
        mesh_device,
        ccl,
        sd,
        prefix,
        num_experts=E,
        hidden_size=H,
        moe_topk=K,
        norm_topk_prob=c["NORM"],
        use_mixed_mlp_moe=c["MIXED"],
        weight_dtype=ttnn.bfloat16,
    )
    logger.info(f"EP: ndev={par.ndev} experts_per_dev={par.experts_per_dev} reduce_axes={par.ep_reduce_axes}")
    assert par.experts_per_dev == E // 4, f"expected {E//4} experts/device on 2x2, got {par.experts_per_dev}"
    y_par = par.forward(x_tt)

    d0 = ttnn.to_torch(y_dense, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[:B].float()
    p0 = ttnn.to_torch(y_par, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[:B].float()
    passing, pcc = comp_pcc(d0, p0, 0.99)
    logger.info(f"MoE EP(2x2) vs dense PCC: {pcc:.6f}")
    assert passing, f"PCC {pcc:.6f} < 0.99 — 2D EP all-reduce is wrong"


# --------------------------------------------------------------------------- C
@pytest.mark.parametrize("mesh_device", [(2, 2)], indirect=True)
def test_tp_attention_2x2_vs_replicated(mesh_device):
    """Phase 2: TP=2 attention (axis 1) vs tp_factor=1 replicated reference."""
    from models.experimental.hunyuan_image_3_0.tt.attention.attention import HunyuanTtAttention

    mesh_device.enable_program_cache()
    c = _cfg()
    prefix = "model.layers.0.self_attn"
    sd = _sd(prefix)
    H = c["H"]
    ccl = CCLManager(mesh_device, num_links=1, topology=ttnn.Topology.Linear)

    torch.manual_seed(0)
    B, S = 1, 256
    x = torch.randn(B, S, H) * 0.05
    x_tt = ttnn.from_torch(
        x,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    common = dict(
        hidden_size=H,
        num_heads=c["HEADS"],
        num_kv_heads=c["KV"],
        head_dim=c["HD"],
        use_qk_norm=c["QKN"],
        eps=c["EPS"],
        weight_dtype=ttnn.bfloat16,
    )
    ref = HunyuanTtAttention(mesh_device, sd, layer_num=0, **common)  # tp_factor=1
    tp = HunyuanTtAttention(mesh_device, sd, layer_num=0, ccl_manager=ccl, tp_axis=1, tp_factor=2, **common)
    logger.info(f"TP attn: local q-heads={tp.num_heads} kv-heads={tp.num_kv_heads}")
    assert tp.num_heads == c["HEADS"] // 2 and tp.num_kv_heads == c["KV"] // 2

    cos_tt, sin_tt = ref.rope.prepare_cos_sin(S, image_infos=None)
    y_ref = ttnn.to_torch(
        ref.forward(x_tt, cos_tt, sin_tt, attention_mask=None),
        mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0),
    )[:B].float()
    y_tp = ttnn.to_torch(
        tp.forward(x_tt, cos_tt, sin_tt, attention_mask=None), mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)
    )[:B].float()
    passing, pcc = comp_pcc(y_ref, y_tp, 0.99)
    logger.info(f"TP attn(2x2) vs replicated PCC: {pcc:.6f}")
    assert passing, f"PCC {pcc:.6f} < 0.99 — TP column/row split or all-reduce is wrong"


# --------------------------------------------------------------------------- D
@pytest.mark.parametrize("mesh_device", [(2, 2)], indirect=True)
def test_sp_model_2x2_vs_no_sp(mesh_device):
    """Phase 3: SP=2 small-stack backbone vs sp_factor=1 (both TP=1, EP on). Same
    replicated inputs + explicit causal mask; compare gathered [B,S,H] outputs."""
    from models.experimental.hunyuan_image_3_0.tt.model import HunyuanTtModel

    mesh_device.enable_program_cache()
    c = _cfg()
    H = c["H"]
    ccl = CCLManager(mesh_device, num_links=1, topology=ttnn.Topology.Linear)
    NL = 2

    def layer_loader(i):
        return _sd(f"model.layers.{i}")

    def build(sp_factor):
        return HunyuanTtModel(
            mesh_device,
            num_layers=NL,
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
            weight_dtype=ttnn.bfloat16,
            stream_experts=False,
            layer_loader=layer_loader,
            apply_final_norm=False,
            ccl_manager=ccl,
            tp_factor=1,
            sp_axis=0,
            sp_factor=sp_factor,
        )

    torch.manual_seed(0)
    B, S = 1, 256  # S/2 = 128 (tile-aligned)
    x = torch.randn(B, S, H) * 0.05
    # Additive causal mask [B,1,S,S]: 0 on/under diagonal, -1e30 above.
    m = torch.triu(torch.full((S, S), -1.0e30), diagonal=1).reshape(1, 1, S, S)

    def up(t):
        return ttnn.from_torch(
            t,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    def run(model):
        out = model.forward(inputs_embeds=up(x), seq_len=S, attention_mask=up(m))
        return ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[:B].float()

    y_ref = run(build(1))
    y_sp = run(build(2))
    passing, pcc = comp_pcc(y_ref, y_sp, 0.99)
    logger.info(f"SP model(2x2) sp=2 vs sp=1 PCC: {pcc:.6f}  shape {tuple(y_sp.shape)}")
    assert passing, f"PCC {pcc:.6f} < 0.99 — SP gather-KV / reshard / mask-slice is wrong"


# --------------------------------------------------------------------------- E
@pytest.mark.parametrize("mesh_device", [(2, 2)], indirect=True)
def test_sp_model_2x2_unaligned_seq(mesh_device):
    """Phase 3 padding: SP=2 with a non-tile-aligned S (200 -> pad 256) vs sp=1."""
    from models.experimental.hunyuan_image_3_0.tt.model import HunyuanTtModel

    mesh_device.enable_program_cache()
    c = _cfg()
    H = c["H"]
    ccl = CCLManager(mesh_device, num_links=1, topology=ttnn.Topology.Linear)
    NL = 2

    def layer_loader(i):
        return _sd(f"model.layers.{i}")

    def build(sp_factor):
        return HunyuanTtModel(
            mesh_device,
            num_layers=NL,
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
            weight_dtype=ttnn.bfloat16,
            stream_experts=False,
            layer_loader=layer_loader,
            apply_final_norm=False,
            ccl_manager=ccl,
            tp_factor=1,
            sp_axis=0,
            sp_factor=sp_factor,
        )

    torch.manual_seed(0)
    B, S = 1, 200  # odd-ish: not a multiple of 64; SP must pad to 256
    x = torch.randn(B, S, H) * 0.05
    m = torch.triu(torch.full((S, S), -1.0e30), diagonal=1).reshape(1, 1, S, S)

    def up(t):
        return ttnn.from_torch(
            t,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    def run(model):
        out = model.forward(inputs_embeds=up(x), seq_len=S, attention_mask=up(m))
        return ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[:B].float()

    y_ref = run(build(1))
    y_sp = run(build(2))
    assert tuple(y_sp.shape) == (B, S, H), f"SP output not unpadded to S: {tuple(y_sp.shape)}"
    passing, pcc = comp_pcc(y_ref, y_sp, 0.99)
    logger.info(f"SP unaligned(S=200->256) sp=2 vs sp=1 PCC: {pcc:.6f}  shape {tuple(y_sp.shape)}")
    assert passing, f"PCC {pcc:.6f} < 0.99 — SP sequence padding/unpad is wrong"
