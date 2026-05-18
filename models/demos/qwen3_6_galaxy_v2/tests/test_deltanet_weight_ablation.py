# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""V2-DEC — DeltaNet per-weight ablation to isolate which weight causes
the L4 prefill PCC catastrophe (PCC=0.6949 at baseline).

Setup:
  Start with all of L4's DeltaNet weights (the FAILING case).
  Swap ONE weight at a time from L4 → L0 (the PASSING case).
  Whichever swap restores PCC to ~0.9994 is the culprit weight.

Weights tested:
  A_log, conv1d.weight, dt_bias, in_proj_a.weight, in_proj_b.weight,
  in_proj_qkv.weight, in_proj_z.weight, norm.weight, out_proj.weight,
  input_layernorm.weight, post_attention_layernorm.weight,
  mlp.{gate,up,down}_proj.weight

Plus baseline: pure L4 (control) and pure L0 (sanity).
"""
from __future__ import annotations

import json
import pathlib

import pytest
import torch
from safetensors.torch import load_file as load_st

import ttnn

_SNAPSHOT = pathlib.Path(
    "/home/tt-admin/.cache/huggingface/hub/models--Qwen--Qwen3.6-27B"
    "/snapshots/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
)
_B = 1
_T_PREFILL = 128
_H = 5120

_DELTANET_WEIGHT_SUFFIXES = [
    "linear_attn.A_log",
    "linear_attn.conv1d.weight",
    "linear_attn.dt_bias",
    "linear_attn.in_proj_a.weight",
    "linear_attn.in_proj_b.weight",
    "linear_attn.in_proj_qkv.weight",
    "linear_attn.in_proj_z.weight",
    "linear_attn.norm.weight",
    "linear_attn.out_proj.weight",
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
]


@pytest.fixture(scope="module")
def bh_glx_mesh():
    ttnn.set_fabric_config(
        ttnn.FabricConfig.FABRIC_1D_RING,
        ttnn.FabricReliabilityMode.STRICT_INIT,
        None,
        ttnn.FabricTensixConfig.DISABLED,
    )
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(8, 4))
    yield mesh
    ttnn.close_mesh_device(mesh)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def _load_layer_sd(layer_idx: int) -> dict:
    with open(_SNAPSHOT / "model.safetensors.index.json") as f:
        idx = json.load(f)
    weight_map = idx["weight_map"]
    needed_prefixes = [
        "model.language_model.embed_tokens.",
        "model.language_model.norm.",
        "lm_head.",
        f"model.language_model.layers.{layer_idx}.",
    ]
    needed_keys = [k for k in weight_map if any(k.startswith(p) for p in needed_prefixes)]
    files = sorted({weight_map[k] for k in needed_keys})
    sd: dict[str, torch.Tensor] = {}
    for fn in files:
        shard = load_st(str(_SNAPSHOT / fn))
        for k in needed_keys:
            if k in shard:
                sd[k] = shard[k]
    return sd


def _relabel_to_layer_zero(sd: dict, src_layer_idx: int) -> dict:
    """Move src_layer_idx weight keys to layers.0.* slot."""
    if src_layer_idx == 0:
        return dict(sd)
    out = {}
    old_pfx = f"model.language_model.layers.{src_layer_idx}."
    new_pfx = "model.language_model.layers.0."
    for k, v in sd.items():
        if k.startswith(old_pfx):
            out[new_pfx + k[len(old_pfx) :]] = v
        else:
            out[k] = v
    return out


def _build_mixed_sd(l0_sd: dict, l4_sd: dict, swap_suffixes: list[str]) -> dict:
    """Build a state_dict where most weights are L4's but specific suffixes are L0's.

    Both l0_sd and l4_sd must already be relabeled to layers.0.*.
    """
    out = dict(l4_sd)
    pfx = "model.language_model.layers.0."
    for suffix in swap_suffixes:
        key = pfx + suffix
        if key in l0_sd:
            out[key] = l0_sd[key]
    return out


def _cpu_ref(sd: dict, x: torch.Tensor) -> torch.Tensor:
    from models.demos.qwen3_6_galaxy.reference.qwen36 import HybridDecoderLayer, Qwen36Config

    with open(_SNAPSHOT / "config.json") as f:
        cfg_dict = json.load(f)
    config = Qwen36Config(cfg_dict)
    layer = HybridDecoderLayer(config, 0).eval()
    pfx = "model.language_model.layers.0."
    layer_sd: dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        if k.startswith(pfx):
            short = k[len(pfx) :]
            if short.startswith("self_attn."):
                layer_sd["attention." + short[len("self_attn.") :]] = v.float()
            elif short.startswith("linear_attn."):
                layer_sd["attention." + short[len("linear_attn.") :]] = v.float()
            else:
                layer_sd[short] = v.float()
    layer.load_state_dict(layer_sd, strict=False)
    with torch.no_grad():
        out, _, _, _ = layer(x.float(), cos=None, sin=None, attention_mask=None)
    return out


def _build_tt_model(mesh, state_dict):
    from models.demos.qwen3_6_galaxy_v2.tt.llama_model import TtTransformer
    from models.demos.qwen3_6_galaxy_v2.tt.qwen36_model_config import TtQwen36ModelArgs

    args = TtQwen36ModelArgs(mesh)
    args.n_layers = 1
    args.linear_attention_pattern = ["linear_attention"]
    weight_cache_path = args.weight_cache_path(ttnn.bfloat8_b)
    weight_cache_path.mkdir(parents=True, exist_ok=True)
    model = TtTransformer(
        args=args,
        dtype=ttnn.bfloat8_b,
        mesh_device=mesh,
        state_dict=state_dict,
        weight_cache_path=weight_cache_path,
        paged_attention_config=None,
        use_paged_kv_cache=False,
    )
    return model, args


def _send_col_sharded(t: torch.Tensor, mesh, args):
    B, T, H = t.shape
    return ttnn.from_torch(
        t.reshape(1, 1, T, H),
        device=mesh,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, dims=(None, 3), mesh_shape=args.cluster_shape),
    )


def _gather_to_full(t, mesh, args, T):
    out = ttnn.to_torch(t, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh, dims=(0, 3), mesh_shape=args.cluster_shape))
    out = out[0:1]
    while out.dim() > 3 and out.shape[0] == 1:
        out = out.squeeze(0)
    if out.dim() == 3:
        out = out[:, :T, :]
    return out


def _build_partial_rope(mesh, T):
    cos = torch.zeros(T, 64, dtype=torch.bfloat16)
    sin = torch.zeros(T, 64, dtype=torch.bfloat16)
    cos_tt = ttnn.from_torch(
        cos.unsqueeze(0),
        device=mesh,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    sin_tt = ttnn.from_torch(
        sin.unsqueeze(0),
        device=mesh,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    return cos_tt, sin_tt


def _pcc(a, b):
    a = a.float().flatten()
    b = b.float().flatten()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def _run_one(mesh, sd, x_cpu, ref_out, tag):
    """Build a 1L model with sd, run prefill T=128, compare to ref_out (CPU)."""
    model, args = _build_tt_model(mesh, sd)
    x_tt = _send_col_sharded(x_cpu, mesh, args)
    cos_tt, sin_tt = _build_partial_rope(mesh, _T_PREFILL)
    chunk_start_idx_tt = ttnn.from_torch(
        torch.tensor([0], dtype=torch.int32),
        device=mesh,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    tt_out = model.forward(
        x_tt,
        current_pos=None,
        rot_mats=(cos_tt, sin_tt),
        user_id=0,
        mode="prefill",
        page_table=None,
        chunk_page_table=None,
        chunk_start_idx=chunk_start_idx_tt,
        start_pos=0,
        get_last_token=-1,
        kv_cache=None,
        batch_size=1,
    )
    tt_cpu = _gather_to_full(tt_out, mesh, args, T=_T_PREFILL)
    tt_cpu = tt_cpu.reshape(_B, _T_PREFILL, _H).float()
    pcc = _pcc(tt_cpu, ref_out[:, :_T_PREFILL, :])
    print(f"[{tag}] PCC = {pcc:.6f}")
    return pcc


@pytest.mark.hardware
@pytest.mark.parametrize("swap_suffix", ["__BASELINE_L4__", "__BASELINE_L0__"] + _DELTANET_WEIGHT_SUFFIXES)
def test_deltanet_per_weight_ablation(bh_glx_mesh, swap_suffix):
    """For each ablation case, build state_dict and measure TT vs CPU PCC."""
    torch.manual_seed(42)
    x_cpu = torch.randn(_B, _T_PREFILL, _H, dtype=torch.bfloat16)

    l0_sd = _relabel_to_layer_zero(_load_layer_sd(0), 0)
    l4_sd = _relabel_to_layer_zero(_load_layer_sd(4), 4)

    if swap_suffix == "__BASELINE_L4__":
        sd = l4_sd
        tag = "L4-baseline"
    elif swap_suffix == "__BASELINE_L0__":
        sd = l0_sd
        tag = "L0-baseline"
    else:
        sd = _build_mixed_sd(l0_sd, l4_sd, [swap_suffix])
        tag = f"L4-but-{swap_suffix.split('.')[-2]}=L0"

    ref_out = _cpu_ref(sd, x_cpu)
    pcc = _run_one(bh_glx_mesh, sd, x_cpu, ref_out, tag)
    # Don't assert — we want all variants to run; report PCC in tag.
    print(f"[ABLATION RESULT] {tag}: PCC={pcc:.6f}")
