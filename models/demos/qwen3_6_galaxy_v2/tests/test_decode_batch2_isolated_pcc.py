# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Phase-0 batch-N ISOLATED-BLOCK decode PCC gate (N=2, bump to 32 later).

A minimal, robust per-user decode PCC gate for batch-N enablement, modeled
EXACTLY on the PROVEN ``test_deltanet_layer_isolated_pcc.py`` pattern (same
fixture / arg-build / weight loader / CPU reference / ``_pcc`` / col-sharded
send+gather helpers). It builds a 1-layer ``TtTransformer`` via
``TtQwen36ModelArgs(mesh)`` and drives ONE batched decode step for N DISTINCT
users through the full ``model.forward(mode="decode")`` path, comparing each
user's decode-step LOGITS (vocab-wide, via the model's final RMSNorm + lm_head)
to its OWN CPU reference LOGITS (per-user PCC > 0.99). ``model.forward`` in
decode mode ALWAYS returns vocab-wide logits (norm + lm_head, llama_model.py
~L1130); the lm_head batch-2 dtype bug that previously motivated dropping the
lm_head is now FIXED (tt/lm_head.py), so we compare LOGITS directly.

Two parametrized sub-tests (each at N=2):

  1. ``test_gdn_decode_batchN_pcc`` — GatedDeltaNet (layer 0, linear_attention).
  2. ``test_full_attn_decode_batchN_pcc`` — full_attention (layer 3, paged KV).

N is parametrized (``[2]``) so it is trivial to bump to 32 once Phase 1 lands.

The N users are carried in **dim-2** of the decode input (llama70b convention:
``[1, 1, N, H]``), tile-padded to ``tile_padded_batch_rows``.

EXPECTED RESULT PRE-PHASE-1 (this is a gate Phase 1 turns green)
----------------------------------------------------------------
The decode BACKBONE still collapses dim-2 to a single logical row:

  - full-attn ``_forward_decode_qwen36`` (llama_attention.py ~L1776-1779)
    slices ``x_3d`` dim-2 (T) back to 1 → only user 0 survives.
  - the decoder GDN/full-attn branches (llama_decoder.py ~L434-468) also slice
    ``attn_out`` dim-2 back to 1 in decode mode.

So at N=2, BOTH sub-tests currently collapse to user 0: **user 0 PASSES,
user 1 FAILS** (it was sliced off). That is the intended pre-Phase-1 state;
Phase 1 removes the dim-2 slicing and turns user 1 green. The GDN block math
itself is already batch-agnostic (``forward_decode`` relabels dim-2 rows into
the dim-0 user-batch), but the surrounding decoder backbone still slices, so
the END-TO-END ``model.forward`` decode used here collapses GDN too — hence
both sub-tests gate the SAME backbone fix.

STATE SEEDING (prefill code is UNTOUCHED)
-----------------------------------------
  - GDN: a TT prefill at batch-1 into a ``max_batch=N`` model hits a
    ``ttnn.copy`` shape mismatch (prefill ``new_state`` is ``[1,...]`` but
    ``dn_state_buffer`` is ``[N,...]``). So we seed the N-user DeltaNet
    recurrent + conv state DIRECTLY FROM THE CPU REFERENCE (same proven helpers
    as the prior batchN test), with the model's exact mesh sharding.
  - full-attn: the paged KV cache is sized for N. We run a TT prefill PER USER
    (user ``u``), passing that user's single page-table row so
    ``paged_fill_cache(batch_idx=0)`` writes into user ``u``'s pages.

Run (N=2):

    export TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd) \\
        && source python_env/bin/activate \\
        && QWEN36_TT_LANG_BETA_G=0 python -m pytest --noconftest \\
            models/demos/qwen3_6_galaxy_v2/tests/test_decode_batch2_isolated_pcc.py \\
            -v -s
"""
from __future__ import annotations

import json
import math
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

_T_PREFILL = 128
_H = 5120
_PCC_THRESH = 0.99

_GDN_LAYER_IDX = 0  # linear_attention (GatedDeltaNet)
_FA_LAYER_IDX = 3  # full_attention

# Paged-attention config for the full-attn sub-test (mirrors test_layer3_paged).
_PAGED_BLOCK_SIZE = 32


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
# Weight loading (copied from test_deltanet_layer_isolated_pcc.py)
# ---------------------------------------------------------------------------


def _load_state_dict_for_layer(snapshot_dir: pathlib.Path, layer_idx: int) -> dict:
    """Load embed/norm/lm_head + the one decoder layer, relabel layer→0."""
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
    if layer_idx != 0:
        relabeled: dict[str, torch.Tensor] = {}
        old_prefix = f"model.language_model.layers.{layer_idx}."
        new_prefix = "model.language_model.layers.0."
        for k, v in sd.items():
            if k.startswith(old_prefix):
                relabeled[new_prefix + k[len(old_prefix) :]] = v
            else:
                relabeled[k] = v
        sd = relabeled
    return sd


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().flatten()
    b = b.float().flatten()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


# ---------------------------------------------------------------------------
# CPU reference — full HybridDecoderLayer (T+1 forward) + final norm + lm_head,
# take position-T (decode-step) LOGITS [vocab]
# ---------------------------------------------------------------------------


def _cpu_reference_logits(state_dict_hf: dict, layer_type: str, x_full: torch.Tensor) -> torch.Tensor:
    """Run embed-free 1-layer HybridDecoderLayer over x_full [1, T+1, H], then
    the model's final RMSNorm + lm_head linear, and return the position-T
    (decode-step) LOGITS [vocab].

    ``model.forward(mode="decode")`` ALWAYS returns vocab-wide logits (norm +
    lm_head), so the CPU reference must mirror that: take the decode-step hidden,
    apply the final zero-centered RMSNorm, then the lm_head linear. Weights are
    relabeled to layer 0 in ``state_dict_hf`` already; the final ``norm`` and
    ``lm_head`` weights are also present in ``state_dict_hf``.
    """
    from models.demos.qwen3_6_galaxy.reference.qwen36 import (
        HybridDecoderLayer,
        Qwen36Config,
        RMSNorm,
        build_mrope_cos_sin,
    )

    with open(_SNAPSHOT / "config.json") as f:
        cfg_dict = json.load(f)
    config = Qwen36Config(cfg_dict)
    layer = HybridDecoderLayer(config, 0).eval()  # weights relabeled to layer 0

    pfx = "model.language_model.layers.0."
    layer_sd: dict[str, torch.Tensor] = {}
    for k, v in state_dict_hf.items():
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

    T = x_full.shape[1]
    if layer_type == "full_attention":
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
            block_out, _, _, _ = layer(x_full.float(), cos, sin, attention_mask=causal_mask)
    else:  # linear_attention — no RoPE / no mask
        with torch.no_grad():
            block_out, _, _, _ = layer(x_full.float(), cos=None, sin=None, attention_mask=None)

    # block_out: [1, T, H]. Take the decode-step (last position) hidden, then
    # apply the model's final RMSNorm + lm_head to produce vocab-wide LOGITS —
    # exactly what model.forward(mode="decode") returns.
    decode_hidden = block_out[:, T - 1 : T, :]  # [1, 1, H]

    final_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, zero_centered=True)
    final_norm.load_state_dict({"weight": state_dict_hf["model.language_model.norm.weight"].float()})
    lm_head_w = state_dict_hf["lm_head.weight"].float()  # [vocab, H]

    with torch.no_grad():
        normed = final_norm(decode_hidden)  # [1, 1, H]
        logits = torch.nn.functional.linear(normed, lm_head_w)  # [1, 1, vocab]
    return logits.reshape(-1)  # [vocab]


# ---------------------------------------------------------------------------
# Model build (copied from the isolated / paged tests; max_batch_size=N)
# ---------------------------------------------------------------------------


def _build_tt_model(mesh, state_dict, layer_type: str, N: int, paged_attention_config=None):
    from models.demos.qwen3_6_galaxy_v2.tt.llama_model import TtTransformer
    from models.demos.qwen3_6_galaxy_v2.tt.qwen36_model_config import TtQwen36ModelArgs

    args = TtQwen36ModelArgs(mesh)
    args.n_layers = 1
    args.linear_attention_pattern = [layer_type]
    # Size KV cache + DeltaNet state buffers for N users BEFORE construction.
    args.max_batch_size = N
    args.tile_padded_batch_rows = args.tile_size * int(math.ceil(N / args.tile_size))
    if hasattr(args, "num_device_groups") and args.num_device_groups:
        args.batch_size_per_device_group = max(N // args.num_device_groups, 1)
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


# ---------------------------------------------------------------------------
# Col-sharded send / logits gather (copied from the isolated + perf tests)
# ---------------------------------------------------------------------------


def _send_col_sharded_decode_rows(rows: torch.Tensor, mesh, args) -> ttnn.Tensor:
    """``rows`` [N, H] torch → col-sharded ``[1, 1, n_pad, H/4]`` per chip.

    Place the N users in dim-2 (llama70b ``tile_padded_batch_rows`` convention),
    pad to ``tile_padded_batch_rows``.
    """
    N, H = rows.shape
    n_pad = getattr(args, "tile_padded_batch_rows", 32)
    padded = torch.zeros(1, 1, n_pad, H, dtype=rows.dtype)
    padded[0, 0, :N, :] = rows
    return ttnn.from_torch(
        padded,
        device=mesh,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, dims=(None, 3), mesh_shape=args.cluster_shape),
    )


def _send_col_sharded_hidden(t: torch.Tensor, mesh, args) -> ttnn.Tensor:
    """``t`` [B, T, H] → col-sharded ``[1, 1, T, H/4]`` per chip (prefill input)."""
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


def _gather_decode_logits(tt_out, mesh, args) -> torch.Tensor:
    """Decode ``model.forward`` returns vocab-wide LOGITS (norm + lm_head),
    possibly a ``list[ttnn.Tensor]`` → take ``[0]``. The lm_head output is
    column-sharded over the vocab dim across the 4 mesh columns; gather it via
    ``ConcatMesh2dToTensor dims=(3, 0)`` (vocab in dim 3, mesh-rows in dim 0),
    keep the row-0 mesh-row copy, and slice to ``args.vocab_size``. The N users
    live in dim-2 (rows), tile-padded. Returns ``[N_pad, vocab]``.

    Mirrors ``test_decode_perf_intrace.py::_gather_prefill_logits_to_cpu``.
    """
    out0 = tt_out[0] if isinstance(tt_out, list) else tt_out
    logits = ttnn.to_torch(
        out0,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh, dims=(3, 0), mesh_shape=args.cluster_shape),
    )
    # post-all-gather the logits are replicated cluster_shape[1]× along dim-0;
    # keep one copy. (Mirror test_decode_perf_intrace::_gather_prefill_logits_to_cpu,
    # which divides by cluster_shape[1] — using cluster_shape[0] zeroed the row dim.)
    n_dup = args.cluster_shape[1]
    logits = logits[: logits.shape[0] // n_dup]
    while logits.dim() > 3 and logits.shape[0] == 1:
        logits = logits.squeeze(0)
    # logits: [1, N_pad, vocab_padded] (or [N_pad, vocab_padded]) → [N_pad, vocab]
    logits = logits.reshape(-1, logits.shape[-1])[:, : args.vocab_size].float()
    return logits  # [N_pad, vocab]


def _build_partial_rope_cos_sin_tt(mesh, positions: torch.Tensor):
    from models.demos.qwen3_6_galaxy.reference.qwen36 import build_mrope_cos_sin

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


def _build_zero_rope_cos_sin_tt(mesh, T: int):
    """Minimal zero cos/sin (GDN has no RoPE; rank-3 [1, T, 64])."""
    cos_ref = torch.zeros(T, 64, dtype=torch.bfloat16)
    sin_ref = torch.zeros(T, 64, dtype=torch.bfloat16)
    mk = lambda r: ttnn.from_torch(
        r.unsqueeze(0),
        device=mesh,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    return mk(cos_ref), mk(sin_ref)


# ---------------------------------------------------------------------------
# GDN per-user state seeding (from CPU ref) — copied from the prior batchN test
# ---------------------------------------------------------------------------


def _build_ref_gdn(state_dict_hf: dict):
    from models.demos.qwen3_6_galaxy.reference.qwen36 import GatedDeltaNet, Qwen36Config

    with open(_SNAPSHOT / "config.json") as f:
        cfg_dict = json.load(f)
    config = Qwen36Config(cfg_dict)
    gdn = GatedDeltaNet(config).eval()
    pfx = "model.language_model.layers.0.linear_attn."
    sd = {k[len(pfx) :]: v.float() for k, v in state_dict_hf.items() if k.startswith(pfx)}
    gdn.load_state_dict(sd, strict=False)
    return gdn


def _seed_dn_state(attn, recurrent_states_per_user: list[torch.Tensor]):
    """Write per-user recurrent state into ``attn.dn_state_buffer``: stack users
    into [N, 48, K, V], shard head axis (dim 1) → mesh rows, replicate cols."""
    state_nv = torch.cat(recurrent_states_per_user, dim=0).float()  # [N, 48, K, V]
    seed = ttnn.from_torch(
        state_nv,
        device=attn.mesh_device,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        memory_config=attn.dn_state_buffer.memory_config(),
        mesh_mapper=ttnn.ShardTensor2dMesh(attn.mesh_device, dims=(1, None), mesh_shape=attn.cluster_shape),
    )
    ttnn.copy(seed, attn.dn_state_buffer)
    seed.deallocate(True)


def _seed_conv_state(attn, conv_states_per_user: list[torch.Tensor]):
    """Write per-user conv state into ``attn.conv_state_buffer``: reorder each
    user's Q|K|V channels into per-mesh-row blocks, shard row axis → mesh rows."""
    mesh_rows = attn.mesh_rows
    q_per_row = attn.q_per_row
    v_per_row = attn.v_per_row
    n_k_hd = attn.n_k_heads * attn.head_dim
    Km1 = attn.conv_kernel - 1

    per_row_blocks = []
    for r in range(mesh_rows):
        users = []
        for cs in conv_states_per_user:  # cs: [1, conv_dim, Km1]
            cs_kt = cs[0].transpose(0, 1)  # [Km1, conv_dim]
            qc = cs_kt[:, r * q_per_row : (r + 1) * q_per_row]
            kc = cs_kt[:, n_k_hd + r * q_per_row : n_k_hd + (r + 1) * q_per_row]
            vc = cs_kt[:, 2 * n_k_hd + r * v_per_row : 2 * n_k_hd + (r + 1) * v_per_row]
            users.append(torch.cat([qc, kc, vc], dim=-1).unsqueeze(0))  # [1, Km1, conv_per_row]
        per_row_blocks.append(torch.cat(users, dim=0).unsqueeze(0))  # [1, N, Km1, conv_per_row]
    seed_rm = torch.cat(per_row_blocks, dim=0)  # [mesh_rows, N, Km1, conv_per_row]

    is_fp32 = attn.dtype == ttnn.float32
    seed = ttnn.from_torch(
        seed_rm.float() if is_fp32 else seed_rm.bfloat16(),
        device=attn.mesh_device,
        dtype=ttnn.float32 if is_fp32 else ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=attn.conv_state_buffer.memory_config(),
        mesh_mapper=ttnn.ShardTensor2dMesh(attn.mesh_device, dims=(0, None), mesh_shape=attn.cluster_shape),
    )
    N = len(conv_states_per_user)
    seed = ttnn.reshape(seed, [N, Km1, q_per_row * 2 + v_per_row])
    ttnn.copy(seed, attn.conv_state_buffer)
    seed.deallocate(True)


# ---------------------------------------------------------------------------
# Paged page-table (from test_decode_perf_intrace / test_layer3_paged)
# ---------------------------------------------------------------------------


def _build_paged_page_table(mesh, args, paged_attention_config):
    """Per-user page table [N, max_blocks/N], sharded replicate (dims None,None)."""
    permutation = torch.randperm(paged_attention_config.max_num_blocks)
    reverse_permutation = torch.argsort(permutation)
    page_table = reverse_permutation.reshape(
        args.max_batch_size, paged_attention_config.max_num_blocks // args.max_batch_size
    )
    page_table_tt = ttnn.from_torch(
        page_table,
        device=mesh,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, dims=(None, None), mesh_shape=args.cluster_shape),
    )
    return page_table, page_table_tt


def _user_page_table_tt(page_table_torch: torch.Tensor, user_idx: int, mesh, args):
    """Single-user page-table row [1, max_blocks/N] for that user's prefill fill
    (paged_fill_cache uses batch_idx=0 ⇒ writes the row-0 of the table given)."""
    row = page_table_torch[user_idx : user_idx + 1, :].contiguous()
    return ttnn.from_torch(
        row,
        device=mesh,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, dims=(None, None), mesh_shape=args.cluster_shape),
    )


# ===========================================================================
# Sub-test 1: GatedDeltaNet (linear_attention) decode at batch-N
# ===========================================================================


@pytest.mark.hardware
@pytest.mark.parametrize("N", [2], ids=lambda n: f"N{n}")
def test_gdn_decode_batchN_pcc(bh_glx_mesh, N):
    """Batch-N GDN decode through the full ``model.forward`` path.

    Pre-Phase-1: the decoder collapses dim-2 to user 0 (llama_decoder.py
    ~L434-441), so user 0 PASSES and user 1 FAILS. Phase 1 (remove the dim-2
    slice) turns user 1 green.
    """
    os.environ.setdefault("QWEN36_TT_LANG_BETA_G", "0")  # B>1 uses the 6-op chain

    state_dict = _load_state_dict_for_layer(_SNAPSHOT, _GDN_LAYER_IDX)
    print(f"[gdn-N{N}] loaded {len(state_dict)} weights")

    # ---- Per-user CPU reference: prefill T -> state, then 1 decode step,
    #      and full-model logits for the decode step.
    gdn_ref = _build_ref_gdn(state_dict)
    x_users = []
    ref_logits = []  # [N] of [vocab]
    conv_states = []
    recur_states = []
    for u in range(N):
        torch.manual_seed(44 + u)
        x_full = torch.randn(1, _T_PREFILL + 1, _H, dtype=torch.bfloat16).float()  # distinct stream/user
        x_users.append(x_full)
        with torch.no_grad():
            _, conv_s, recur_s = gdn_ref(x_full[:, :_T_PREFILL, :], conv_state=None, recurrent_state=None)
        conv_states.append(conv_s)
        recur_states.append(recur_s)
        ref_logits.append(_cpu_reference_logits(state_dict, "linear_attention", x_full))
    print(f"[gdn-N{N}] CPU references built (T_PREFILL={_T_PREFILL})")

    model, args = _build_tt_model(bh_glx_mesh, state_dict, "linear_attention", N)
    assert getattr(model.layers[0], "is_linear_attention_layer", False) is True
    attn = model.layers[0].attention
    print(f"[gdn-N{N}] TT model built; max_batch_size={attn.max_batch_size}")

    # ---- Seed per-user DeltaNet recurrent + conv state from CPU ref (prefill
    #      code untouched — TT prefill at B=1 into a max_batch=N model would hit
    #      a ttnn.copy [1,..] vs [N,..] shape mismatch).
    attn.clear_state()
    _seed_dn_state(attn, recur_states)
    _seed_conv_state(attn, conv_states)
    print(f"[gdn-N{N}] seeded recurrent + conv state for {N} users")

    # ---- ONE batched decode step (N users in dim-2).
    decode_rows = torch.cat([x_users[u][:, _T_PREFILL, :] for u in range(N)], dim=0).bfloat16()  # [N, H]
    x_decode_tt = _send_col_sharded_decode_rows(decode_rows, bh_glx_mesh, args)
    cos_tt, sin_tt = _build_zero_rope_cos_sin_tt(bh_glx_mesh, 1)
    cur_pos_tt = ttnn.from_torch(
        torch.tensor([_T_PREFILL] * N, dtype=torch.int32),
        device=bh_glx_mesh,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(bh_glx_mesh),
    )
    tt_out = model.forward(
        x_decode_tt,
        current_pos=cur_pos_tt,
        rot_mats=(cos_tt, sin_tt),
        user_id=0,
        mode="decode",
        page_table=None,
        chunk_page_table=None,
        chunk_start_idx=None,
        start_pos=_T_PREFILL,
        get_last_token=-1,
        kv_cache=None,
        batch_size=N,
    )
    tt_logits = _gather_decode_logits(tt_out, bh_glx_mesh, args)  # [N_pad, vocab]
    _assert_per_user(tt_logits, ref_logits, N, tag=f"gdn-N{N}")


# ===========================================================================
# Sub-test 2: full_attention decode at batch-N (paged KV)
# ===========================================================================


@pytest.mark.hardware
@pytest.mark.parametrize("N", [2], ids=lambda n: f"N{n}")
def test_full_attn_decode_batchN_pcc(bh_glx_mesh, N):
    """Batch-N full_attention decode through the full ``model.forward`` path
    (paged KV; per-user TT prefill).

    Pre-Phase-1: ``_forward_decode_qwen36`` slices x dim-2 (T) back to 1
    (llama_attention.py ~L1776-1779), so user 0 PASSES and user 1 FAILS.
    Phase 1 (remove the dim-2 slice + thread the batch through SDPA) turns
    user 1 green.
    """
    from models.demos.llama3_70b_galaxy.tt.llama_common import PagedAttentionConfig

    max_num_blocks = max(N * 8, (_T_PREFILL + _PAGED_BLOCK_SIZE - 1) // _PAGED_BLOCK_SIZE * N + 4 * N)
    # Round up to a multiple of N so it reshapes cleanly to [N, max_blocks/N].
    max_num_blocks = int(math.ceil(max_num_blocks / N) * N)
    paged_attention_config = PagedAttentionConfig(block_size=_PAGED_BLOCK_SIZE, max_num_blocks=max_num_blocks)

    state_dict = _load_state_dict_for_layer(_SNAPSHOT, _FA_LAYER_IDX)
    print(f"[fa-N{N}] loaded {len(state_dict)} weights (max_num_blocks={max_num_blocks})")

    # ---- Per-user CPU reference LOGITS.
    x_users = []
    ref_logits = []
    for u in range(N):
        torch.manual_seed(44 + u)
        x_full = torch.randn(1, _T_PREFILL + 1, _H, dtype=torch.bfloat16).float()  # distinct stream/user
        x_users.append(x_full)
        ref_logits.append(_cpu_reference_logits(state_dict, "full_attention", x_full))
    print(f"[fa-N{N}] CPU references built (T_PREFILL={_T_PREFILL})")

    model, args = _build_tt_model(bh_glx_mesh, state_dict, "full_attention", N, paged_attention_config)
    assert getattr(model.layers[0], "is_linear_attention_layer", True) is False
    print(f"[fa-N{N}] TT model built (paged KV); max_batch_size={args.max_batch_size}")

    page_table_torch, page_table_tt = _build_paged_page_table(bh_glx_mesh, args, paged_attention_config)

    # ---- TT prefill PER USER: pass that user's single page-table row so
    #      paged_fill_cache(batch_idx=0) writes into user u's pages.
    for u in range(N):
        x_tt = _send_col_sharded_hidden(x_users[u][:, :_T_PREFILL, :].bfloat16(), bh_glx_mesh, args)
        cos_tt, sin_tt = _build_partial_rope_cos_sin_tt(bh_glx_mesh, torch.arange(_T_PREFILL, dtype=torch.long))
        chunk_start_idx_tt = ttnn.from_torch(
            torch.tensor([0], dtype=torch.int32),
            device=bh_glx_mesh,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(bh_glx_mesh),
        )
        user_pt = _user_page_table_tt(page_table_torch, u, bh_glx_mesh, args)
        _ = model.forward(
            x_tt,
            current_pos=None,
            rot_mats=(cos_tt, sin_tt),
            user_id=u,
            mode="prefill",
            page_table=user_pt,
            chunk_page_table=None,
            chunk_start_idx=chunk_start_idx_tt,
            start_pos=0,
            get_last_token=-1,
            kv_cache=None,
            batch_size=1,
        )
        x_tt.deallocate(True)
    print(f"[fa-N{N}] per-user prefill complete; paged KV populated for {N} users")

    # ---- ONE batched decode step (N users in dim-2), full [N, max_blocks] page table.
    decode_rows = torch.cat([x_users[u][:, _T_PREFILL, :] for u in range(N)], dim=0).bfloat16()  # [N, H]
    x_decode_tt = _send_col_sharded_decode_rows(decode_rows, bh_glx_mesh, args)
    cos_dec, sin_dec = _build_partial_rope_cos_sin_tt(bh_glx_mesh, torch.tensor([_T_PREFILL], dtype=torch.long))
    cur_pos_tt = ttnn.from_torch(
        torch.tensor([_T_PREFILL] * N, dtype=torch.int32),
        device=bh_glx_mesh,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(bh_glx_mesh),
    )
    tt_out = model.forward(
        x_decode_tt,
        current_pos=cur_pos_tt,
        rot_mats=(cos_dec, sin_dec),
        user_id=0,
        mode="decode",
        page_table=page_table_tt,
        chunk_page_table=None,
        chunk_start_idx=None,
        start_pos=_T_PREFILL,
        get_last_token=-1,
        kv_cache=None,
        batch_size=N,
    )
    tt_logits = _gather_decode_logits(tt_out, bh_glx_mesh, args)  # [N_pad, vocab]
    _assert_per_user(tt_logits, ref_logits, N, tag=f"fa-N{N}")


# ---------------------------------------------------------------------------
# Per-user assertion helper
# ---------------------------------------------------------------------------


def _assert_per_user(tt_logits: torch.Tensor, ref_logits: list[torch.Tensor], N: int, tag: str):
    """Compare the per-user decode LOGITS (vocab-wide; norm + lm_head).
    ``tt_logits`` is the gathered ``[N_pad, vocab]``; the N users live in dim-0
    (rows). Slice user u's row → ``[vocab]`` and compare to its CPU-ref logits
    ``[vocab]``. Also report per-user argmax (predicted token) match.

    Pre-Phase-1 the decode backbone collapses the user dim to a single logical
    row, so only user 0's row carries that user's logits — user 0 PASSES, users
    1..N-1 FAIL (their row was sliced off). Phase 1 (remove the dim-2 slice)
    turns them green.
    """
    vocab = ref_logits[0].numel()
    flat = tt_logits.reshape(-1, vocab)  # [N_pad, vocab]
    failures = []
    for u in range(N):
        tt_u = flat[u].reshape(vocab)  # [vocab]
        ref_u = ref_logits[u].reshape(vocab)
        pcc_u = _pcc(tt_u, ref_u)
        argmax_match = int(tt_u.argmax().item()) == int(ref_u.argmax().item())
        ok = pcc_u > _PCC_THRESH
        print(
            f"[{tag}] user {u}: logits PCC = {pcc_u:.6f} (thresh={_PCC_THRESH})  "
            f"argmax_match={argmax_match} (tt={int(tt_u.argmax().item())}, "
            f"ref={int(ref_u.argmax().item())})  {'PASS' if ok else 'FAIL'}"
        )
        if not ok:
            failures.append((u, pcc_u))
    assert not failures, (
        f"[{tag}] batch-{N} decode LOGITS PCC < {_PCC_THRESH} for users "
        + ", ".join(f"{u}(pcc={p:.4f})" for u, p in failures)
        + "  (EXPECTED pre-Phase-1: only user 0 survives the decode-backbone dim-2 slice)"
    )
    print(f"[{tag}] PASSED (all {N} users PCC > {_PCC_THRESH})")
