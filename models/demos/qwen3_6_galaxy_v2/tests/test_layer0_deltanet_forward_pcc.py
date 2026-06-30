# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""V2-7b — Layer-0 DeltaNet prefill PCC test through TtTransformer.forward.

Builds a 1-layer ``TtTransformer`` whose only decoder block is layer 0
(``linear_attention``), feeds an already-embedded col-sharded ``[1, 1, T, H/4]``
hidden state directly into ``TtTransformer.forward(mode="prefill")`` (bypassing
the embedding), and compares the prefill output to the CPU reference
``HybridDecoderLayer`` from
``models/demos/qwen3_6_galaxy/reference/qwen36.py``. PCC threshold > 0.99.

This is the V2-7b retry: V2-7a's block-level forward bypassed the decoder
gather/scatter; we now exercise the full prefill path including pre-norm
+ DeltaNet + residual + post-norm + MLP + residual.

Run:

    export TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd) \\
        && source python_env/bin/activate \\
        && python -m pytest --noconftest \\
            models/demos/qwen3_6_galaxy_v2/tests/test_layer0_deltanet_forward_pcc.py \\
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
from models.demos.qwen3_6_galaxy_v2.tt.gdn_chunk_ops_seq import _gdn_subop_pcc_report

_SNAPSHOT = pathlib.Path(
    "/home/tt-admin/.cache/huggingface/hub/models--Qwen--Qwen3.6-27B"
    "/snapshots/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
)

_B = 1
# Override with ``QWEN36_PCC_T_PREFILL=4096`` (etc.) to PCC-check long-ISL
# shapes. Default 128 matches TT_CCL prefill ``support_seqlens`` minimum.
_T_PREFILL = int(os.environ.get("QWEN36_PCC_T_PREFILL", "128"))
_H = 5120
_PCC_THRESH = 0.99
_LAYER_IDX = 0
_LAYER_TYPE = "linear_attention"


# ---------------------------------------------------------------------------
# Mesh fixture
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Weight loading
# ---------------------------------------------------------------------------


def _load_state_dict_for_layer(snapshot_dir: pathlib.Path, layer_idx: int) -> dict:
    """Load the HF tensors required to construct a 1-layer TtTransformer."""
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


# ---------------------------------------------------------------------------
# CPU reference: full HybridDecoderLayer (norm + attn + residual + norm + MLP + residual)
# ---------------------------------------------------------------------------


def _cpu_reference_layer(state_dict_full_hf: dict, layer_idx: int, layer_type: str, x: torch.Tensor) -> torch.Tensor:
    """Run the validated ``HybridDecoderLayer`` (full decoder forward) at float32."""
    from models.demos.qwen3_6_galaxy.reference.qwen36 import HybridDecoderLayer, Qwen36Config

    with open(_SNAPSHOT / "config.json") as f:
        cfg_dict = json.load(f)
    config = Qwen36Config(cfg_dict)

    layer = HybridDecoderLayer(config, layer_idx).eval()

    # Load weights for this layer from the HF state_dict
    pfx = f"model.language_model.layers.{layer_idx}."
    layer_sd: dict[str, torch.Tensor] = {}
    for k, v in state_dict_full_hf.items():
        if k.startswith(pfx):
            short = k[len(pfx) :]
            # Map qwen3.6 HF names to the reference module's state_dict
            # (HybridDecoderLayer.attention is GatedAttention or GatedDeltaNet
            # — both consume short keys like ``q_proj.weight`` /
            # ``in_proj_qkv.weight`` respectively).
            if short.startswith("self_attn."):
                layer_sd["attention." + short[len("self_attn.") :]] = v.float()
            elif short.startswith("linear_attn."):
                layer_sd["attention." + short[len("linear_attn.") :]] = v.float()
            else:
                layer_sd[short] = v.float()

    missing, unexpected = layer.load_state_dict(layer_sd, strict=False)
    if missing:
        # Allow only the unused-attention branch's keys to be missing.
        unexpected_kinds = []
        for k in missing:
            unexpected_kinds.append(k)
        # Hard-fail on truly missing keys we did expect.
        for k in missing:
            if k.startswith("input_layernorm") or k.startswith("post_attention_layernorm") or k.startswith("mlp."):
                raise AssertionError(f"Missing reference weight: {k}")
        # The other branch's attention keys are missing — that's fine.

    # For linear_attention we don't need RoPE; for full_attention we'd build it.
    if layer_type == "full_attention":
        from models.demos.qwen3_6_galaxy.reference.qwen36 import build_mrope_cos_sin

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
    else:
        # linear_attention: no RoPE/mask. Pass dummy cos/sin (unused).
        with torch.no_grad():
            out, _, _, _ = layer(x.float(), cos=None, sin=None, attention_mask=None)
    return out  # [B, T, H]


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().flatten()
    b = b.float().flatten()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_tt_model(mesh, state_dict, layer_idx: int, layer_type: str):
    """Construct a 1-layer TtTransformer mirroring the chosen layer index."""
    from models.demos.qwen3_6_galaxy_v2.tt.llama_model import TtTransformer
    from models.demos.qwen3_6_galaxy_v2.tt.qwen36_model_config import TtQwen36ModelArgs

    args = TtQwen36ModelArgs(mesh)
    args.n_layers = 1
    args.linear_attention_pattern = [layer_type]

    # bfloat8_b model dtype works for DeltaNet (layer 0).  For full_attention
    # (layer 3) the 1L test required bfloat16 because bf8 w1/w3 dropped PCC.
    # The cleanest fix lives in llama_mlp.py (force is_qwen36 MLP weights to
    # bfloat16) so MLP precision is layer-independent; the test harness then
    # passes the canonical bfloat8_b model dtype.
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


def _build_partial_rope_cos_sin_tt(mesh, T: int):
    """Build qwen3.6 partial-RoPE cos/sin at rope_dim=64, replicated across mesh."""
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

    # cos_ref / sin_ref shape: [T, 64] each (rope_dim slice of head_dim).
    cos_tt = ttnn.from_torch(
        cos_ref.unsqueeze(0),  # [1, T, 64] — rank-3, matches the validated v2 block test
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
    """Upload a [B, T, H] hidden state as col-sharded [1, 1, T, H/4] across cols.

    Matches the post-embedding layout that TtTransformer.forward expects.
    Rows are replicated; cols shard dim 3 of the 4D tensor.
    """
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
    """Gather a col-sharded [1, 1, T, H/4] back to [B, T, H] on host (first row replica)."""
    out = ttnn.to_torch(
        tt_tensor,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh, dims=(0, 3), mesh_shape=args.cluster_shape),
    )
    # ConcatMesh2dToTensor stacks rows along dim 0 → take row 0 (rows are replicated).
    out = out[0:1]
    # Now out is [1, 1, T, H] approximately — squeeze leading singletons.
    while out.dim() > 3 and out.shape[0] == 1:
        out = out.squeeze(0)
    if out.dim() == 3:
        out = out[:, :T, :]
    return out


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.hardware
def test_qwen36_layer0_deltanet_through_transformer_forward(bh_glx_mesh):
    """1L TtTransformer.forward (mode=prefill) — layer 0 DeltaNet end-to-end PCC."""
    state_dict = _load_state_dict_for_layer(_SNAPSHOT, _LAYER_IDX)
    print(f"[Layer0/forward] loaded {len(state_dict)} weights")

    model, args = _build_tt_model(bh_glx_mesh, state_dict, _LAYER_IDX, _LAYER_TYPE)
    assert getattr(model.layers[0], "is_linear_attention_layer", False) is True
    print(f"[Layer0/forward] TT 1-layer DeltaNet built (H={args.dim}, n_layers={args.n_layers})")

    # --- Random hidden state (post-embedding stand-in) ---
    torch.manual_seed(42)
    x_cpu = torch.randn(_B, _T_PREFILL, _H, dtype=torch.bfloat16)
    print(f"[Layer0/forward] input shape: {x_cpu.shape}")

    # --- CPU reference (full HybridDecoderLayer forward) ---
    out_ref = _cpu_reference_layer(state_dict, _LAYER_IDX, _LAYER_TYPE, x_cpu)
    print(f"[Layer0/forward] CPU ref shape: {out_ref.shape}")

    # --- TT forward through TtTransformer.forward(mode="prefill") ---
    x_tt = _send_col_sharded_hidden(x_cpu, bh_glx_mesh, args)

    # Build partial-RoPE cos/sin at rope_dim=64; the DeltaNet branch ignores
    # rot_mats but we pass non-None to satisfy the assert in ttnn_prefill_forward.
    cos_tt, sin_tt = _build_partial_rope_cos_sin_tt(bh_glx_mesh, _T_PREFILL)

    # chunk_start_idx as device tensor (required by ttnn_prefill_forward assert).
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
        page_table=None,
        chunk_page_table=None,
        chunk_start_idx=chunk_start_idx_tt,
        start_pos=0,
        get_last_token=-1,
        kv_cache=None,
        batch_size=1,
    )
    tt_out_cpu = _gather_col_sharded_to_full(tt_out, bh_glx_mesh, args, T=_T_PREFILL)
    tt_out_cpu = tt_out_cpu.reshape(_B, _T_PREFILL, _H).float()
    print(f"[Layer0/forward] TT out shape: {tt_out_cpu.shape}")

    pcc = _pcc(tt_out_cpu, out_ref[:, :_T_PREFILL, :])
    # torch.quantile is capped at ~16M elements; use kthvalue for long-ISL shapes.
    diff_flat = (tt_out_cpu.float() - out_ref[:, :_T_PREFILL, :].float()).abs().flatten()
    p99 = torch.kthvalue(diff_flat, int(0.99 * diff_flat.numel())).values.item()
    print(f"[Layer0/forward] PCC = {pcc:.6f} (thresh={_PCC_THRESH})  |  p99 abs-diff = {p99:.4f}")
    _gdn_subop_pcc_report()
    assert pcc > _PCC_THRESH, f"Layer0/forward PCC {pcc:.4f} < {_PCC_THRESH} (p99={p99:.4f})"
    print("[Layer0/forward] PASSED")
