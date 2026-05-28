# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Layer-3 full_attention prefill PCC — PAGED variant.

The existing ``test_layer3_full_attention_forward_pcc.py`` uses the unpaged
prefill path (``page_table=None``), which has bit-rotted: at T=128 it
returns PCC=0.7075, even though the production e2e prefill at ISL=128 (which
goes through the PAGED path) generates coherent text.

This test mirrors the e2e ``test_decode_perf_intrace.py`` prefill setup
exactly:

  - Builds a 1-layer ``TtTransformer`` with ``PagedAttentionConfig(block_size=32, ...)``
    and ``use_paged_kv_cache=False`` (just like the perf test).
  - Constructs a page table.
  - Runs ``model.forward(mode="prefill", page_table=page_table_tt, ...)``.

The layer-3 HF weights are relabeled to ``layers.0`` so the 1-layer model
picks them up at slot 0. The CPU reference uses the validated
``HybridDecoderLayer`` from ``models/demos/qwen3_6_galaxy/reference/qwen36.py``.

PCC threshold is taken from ``QWEN36_PCC_THRESH`` (default 0.99). Override
the ISL via ``QWEN36_PCC_T_PREFILL`` (default 128).

Run:

    export TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd) \\
        && source python_env/bin/activate \\
        && QWEN36_PCC_T_PREFILL=128 python -m pytest --noconftest \\
            models/demos/qwen3_6_galaxy_v2/tests/test_layer3_full_attention_paged_pcc.py \\
            -v -s
"""
from __future__ import annotations

import json
import os
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
_T_PREFILL = int(os.environ.get("QWEN36_PCC_T_PREFILL", "128"))
_LAYER_IDX = 3
_LAYER_TYPE = "full_attention"
_H = 5120
_PCC_THRESH = float(os.environ.get("QWEN36_PCC_THRESH", "0.99"))
_PAGED_BLOCK_SIZE = 32
_PAGED_MAX_NUM_BLOCKS = max(32, (_T_PREFILL + _PAGED_BLOCK_SIZE - 1) // _PAGED_BLOCK_SIZE + 4)


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


def _load_state_dict_for_layer(snapshot_dir: pathlib.Path, layer_idx: int) -> dict:
    with open(snapshot_dir / "model.safetensors.index.json") as f:
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
        shard = load_st(str(snapshot_dir / fn))
        for k in needed_keys:
            if k in shard:
                sd[k] = shard[k]
    return sd


def _relabel_layer_idx(sd: dict, src_idx: int, dst_idx: int) -> dict:
    src_prefix = f"model.language_model.layers.{src_idx}."
    dst_prefix = f"model.language_model.layers.{dst_idx}."
    out: dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        if k.startswith(src_prefix):
            out[dst_prefix + k[len(src_prefix) :]] = v
        else:
            out[k] = v
    return out


def _cpu_reference_layer(state_dict_full_hf: dict, layer_idx: int, x: torch.Tensor) -> torch.Tensor:
    """Run the validated HybridDecoderLayer (full decoder forward) at float32 for full_attention."""
    from models.demos.qwen3_6_galaxy.reference.qwen36 import HybridDecoderLayer, Qwen36Config, build_mrope_cos_sin

    with open(_SNAPSHOT / "config.json") as f:
        cfg_dict = json.load(f)
    config = Qwen36Config(cfg_dict)
    assert config.layer_types[layer_idx] == "full_attention"

    layer = HybridDecoderLayer(config, layer_idx).eval()

    pfx = f"model.language_model.layers.{layer_idx}."
    layer_sd: dict[str, torch.Tensor] = {}
    for k, v in state_dict_full_hf.items():
        if not k.startswith(pfx):
            continue
        short = k[len(pfx) :]
        if short.startswith("self_attn."):
            layer_sd["attention." + short[len("self_attn.") :]] = v.float()
        elif short.startswith("linear_attn."):
            layer_sd["attention." + short[len("linear_attn.") :]] = v.float()
        else:
            layer_sd[short] = v.float()
    layer.load_state_dict(layer_sd, strict=False)

    T = x.shape[1]
    positions = torch.arange(T, dtype=torch.long)
    positions_3d = torch.stack([positions, positions, positions], dim=0)
    cos, sin = build_mrope_cos_sin(
        positions_3d=positions_3d,
        head_dim=256,
        partial_rotary_factor=0.25,
        mrope_section=[11, 11, 10],
        theta=10_000_000.0,
    )
    causal_mask = torch.zeros(1, 1, T, T)
    causal_mask = causal_mask.masked_fill(torch.triu(torch.ones(T, T), diagonal=1).bool(), float("-inf"))
    with torch.no_grad():
        out, _, _, _ = layer(x.float(), cos, sin, attention_mask=causal_mask)
    return out


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().flatten()
    b = b.float().flatten()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def _build_tt_model_paged(mesh, state_dict, layer_type: str, paged_attention_config):
    """1-layer TtTransformer wired up with paged attention (mirrors decode_perf_intrace)."""
    from models.demos.qwen3_6_galaxy_v2.tt.llama_model import TtTransformer
    from models.demos.qwen3_6_galaxy_v2.tt.qwen36_model_config import TtQwen36ModelArgs

    args = TtQwen36ModelArgs(mesh)
    args.n_layers = 1
    args.linear_attention_pattern = [layer_type]
    weight_cache_path = args.weight_cache_path(ttnn.bfloat8_b)
    weight_cache_path.mkdir(parents=True, exist_ok=True)
    model = TtTransformer(
        args=args,
        dtype=ttnn.bfloat8_b,
        mesh_device=mesh,
        state_dict=state_dict,
        weight_cache_path=weight_cache_path,
        paged_attention_config=paged_attention_config,
        use_paged_kv_cache=False,
    )
    return model, args


def _build_paged_page_table(mesh, args, paged_attention_config):
    permutation = torch.randperm(paged_attention_config.max_num_blocks)
    reverse_permutation = torch.argsort(permutation)
    page_table = reverse_permutation.reshape(
        args.max_batch_size, paged_attention_config.max_num_blocks // args.max_batch_size
    )
    return ttnn.from_torch(
        page_table,
        device=mesh,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, dims=(None, None), mesh_shape=args.cluster_shape),
    )


def _build_partial_rope_cos_sin_tt(mesh, T: int):
    from models.demos.qwen3_6_galaxy.reference.qwen36 import build_mrope_cos_sin

    positions = torch.arange(T, dtype=torch.long)
    positions_3d = torch.stack([positions, positions, positions], dim=0)
    cos_ref, sin_ref = build_mrope_cos_sin(
        positions_3d=positions_3d,
        head_dim=256,
        partial_rotary_factor=0.25,
        mrope_section=[11, 11, 10],
        theta=10_000_000.0,
    )
    cos_tt = ttnn.from_torch(
        cos_ref.unsqueeze(0),
        device=mesh,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    sin_tt = ttnn.from_torch(
        sin_ref.unsqueeze(0),
        device=mesh,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    return cos_tt, sin_tt


def _send_col_sharded_hidden(t: torch.Tensor, mesh, args):
    B, T, H = t.shape
    t_4d = t.reshape(1, 1, T, H)
    return ttnn.from_torch(
        t_4d,
        device=mesh,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, dims=(None, 3), mesh_shape=args.cluster_shape),
    )


def _gather_col_sharded_to_full(tt_tensor, mesh, args, T: int):
    out = ttnn.to_torch(
        tt_tensor,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh, dims=(0, 3), mesh_shape=args.cluster_shape),
    )
    out = out[0:1]
    while out.dim() > 3 and out.shape[0] == 1:
        out = out.squeeze(0)
    if out.dim() == 3:
        out = out[:, :T, :]
    return out


@pytest.mark.hardware
def test_qwen36_layer3_full_attention_paged_prefill(bh_glx_mesh):
    """1L layer-3 full_attention PCC through the PRODUCTION paged prefill path."""
    from models.demos.llama3_70b_galaxy.tt.llama_common import PagedAttentionConfig

    state_dict_orig = _load_state_dict_for_layer(_SNAPSHOT, _LAYER_IDX)
    print(f"[Layer3/paged] loaded {len(state_dict_orig)} weights")

    state_dict_for_tt = _relabel_layer_idx(state_dict_orig, src_idx=_LAYER_IDX, dst_idx=0)
    paged_attention_config = PagedAttentionConfig(block_size=_PAGED_BLOCK_SIZE, max_num_blocks=_PAGED_MAX_NUM_BLOCKS)
    model, args = _build_tt_model_paged(bh_glx_mesh, state_dict_for_tt, _LAYER_TYPE, paged_attention_config)
    assert getattr(model.layers[0], "is_linear_attention_layer", True) is False
    print(f"[Layer3/paged] TT 1-layer full_attention built (paged, T={_T_PREFILL})")

    page_table_tt = _build_paged_page_table(bh_glx_mesh, args, paged_attention_config)

    torch.manual_seed(43)
    x_cpu = torch.randn(_B, _T_PREFILL, _H, dtype=torch.bfloat16)

    out_ref = _cpu_reference_layer(state_dict_orig, _LAYER_IDX, x_cpu)
    print(f"[Layer3/paged] CPU ref shape: {out_ref.shape}")

    x_tt = _send_col_sharded_hidden(x_cpu, bh_glx_mesh, args)
    cos_tt, sin_tt = _build_partial_rope_cos_sin_tt(bh_glx_mesh, _T_PREFILL)
    chunk_start_idx_tt = ttnn.from_torch(
        torch.tensor([0], dtype=torch.int32),
        device=bh_glx_mesh,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(bh_glx_mesh),
    )

    tt_out = model.forward(
        x_tt,
        current_pos=None,
        rot_mats=(cos_tt, sin_tt),
        user_id=0,
        mode="prefill",
        page_table=page_table_tt,
        chunk_page_table=None,
        chunk_start_idx=chunk_start_idx_tt,
        start_pos=0,
        get_last_token=-1,
        kv_cache=None,
        batch_size=1,
    )
    tt_out_cpu = (
        _gather_col_sharded_to_full(tt_out, bh_glx_mesh, args, T=_T_PREFILL).reshape(_B, _T_PREFILL, _H).float()
    )
    print(f"[Layer3/paged] TT out shape: {tt_out_cpu.shape}")

    pcc = _pcc(tt_out_cpu, out_ref[:, :_T_PREFILL, :])
    diff_flat = (tt_out_cpu.float() - out_ref[:, :_T_PREFILL, :].float()).abs().flatten()
    p99 = torch.kthvalue(diff_flat, int(0.99 * diff_flat.numel())).values.item()
    print(f"[Layer3/paged] PCC = {pcc:.6f} (thresh={_PCC_THRESH})  |  p99 abs-diff = {p99:.4f}")
    assert pcc > _PCC_THRESH, f"Layer3/paged PCC {pcc:.4f} < {_PCC_THRESH} (p99={p99:.4f})"
    print("[Layer3/paged] PASSED")
