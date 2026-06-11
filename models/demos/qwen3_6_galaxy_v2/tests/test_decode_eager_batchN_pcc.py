# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""BATCH-N GatedDeltaNet (GDN) DECODE per-user PCC gate.

This gates the batch-agnostic GDN decode path (``TtQwen36DeltaAttention.
forward_decode``): B users carried in dim-0 of the recurrent core, one batched
decode step, every user's GDN-block output PCC > 0.99 vs its own CPU reference.

WHY THIS IS A *BLOCK-LEVEL* GATE (not a full-model logits gate)
---------------------------------------------------------------
The qwen3.6 decode BACKBONE (llama_decoder.py + the full-attention decode path
``_forward_decode_qwen36``) is still single-user: it places the one decode token
in the dim-2 ``tile_padded_batch_rows`` slot and **slices that slot back to a
single row** (decoder lines ~434-441 for the GDN branch and ~461-468 for the
full-attn branch; ``_forward_decode_qwen36`` line ~1777). So a full-model
forward at batch-N collapses to user 0 *regardless* of the GDN fix. Making the
whole model batch-N is a separate, larger change (decoder backbone + full-attn
SDPA batch threading) that is intentionally OUT OF SCOPE here. This test
therefore exercises the GDN block in ISOLATION — exactly the code that was made
batch-agnostic — by calling ``forward_decode`` directly and comparing the GDN
block output per user.

SEEDING (test-side, prefill UNTOUCHED)
--------------------------------------
We do NOT run TT prefill to seed per-user state (TT prefill at batch-1 into a
``max_batch=N`` model hits a ``ttnn.copy`` shape mismatch: prefill ``new_state``
is ``[1,...]`` but ``dn_state_buffer`` is ``[N,...]`` — and we are not allowed to
touch prefill). Instead we seed the per-user DeltaNet recurrent + conv state
DIRECTLY FROM THE CPU REFERENCE: run the reference ``GatedDeltaNet`` over each
user's T-1 prefill tokens to get ``(conv_state, recurrent_state)``, assemble the
N-user state on host, and write it into the model's ``[N,...]`` buffers with the
SAME mesh sharding the model uses (recurrent state: heads → mesh rows, replicate
cols; conv state: per-row Q|K|V channel blocks → mesh rows, replicate cols).

Run (N=2):

    export TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd) \\
        && source python_env/bin/activate \\
        && QWEN36_TT_LANG_BETA_G=0 python -m pytest --noconftest \\
            models/demos/qwen3_6_galaxy_v2/tests/test_decode_eager_batchN_pcc.py \\
            -v -s

  QWEN36_TT_LANG_BETA_G=0 is no longer strictly required for N>1 (the fused
  beta/g kernel state now auto-skips at max_batch_size>1 and the 6-op chain runs
  instead), but it keeps the test deterministic across builds.
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

_T_PREFILL = 128
_H = 5120
_LAYER_IDX = 0  # a linear_attention (GDN) layer
_PCC_THRESH = 0.99


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
    prefix = f"model.language_model.layers.{layer_idx}."
    needed_keys = [k for k in weight_map if k.startswith(prefix)]
    files = sorted({weight_map[k] for k in needed_keys})
    sd: dict[str, torch.Tensor] = {}
    for fn in files:
        shard = load_st(str(snapshot_dir / fn))
        for k in needed_keys:
            if k in shard:
                sd[k] = shard[k]
    return sd


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().flatten()
    b = b.float().flatten()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def _build_ref_gdn(state_dict_hf: dict, layer_idx: int):
    """Construct the CPU-reference GatedDeltaNet for ``layer_idx`` with real weights."""
    from models.demos.qwen3_6_galaxy.reference.qwen36 import GatedDeltaNet, Qwen36Config

    with open(_SNAPSHOT / "config.json") as f:
        cfg_dict = json.load(f)
    config = Qwen36Config(cfg_dict)
    gdn = GatedDeltaNet(config).eval()

    pfx = f"model.language_model.layers.{layer_idx}.linear_attn."
    sd = {}
    for k, v in state_dict_hf.items():
        if k.startswith(pfx):
            sd[k[len(pfx) :]] = v.float()
    gdn.load_state_dict(sd, strict=False)
    return gdn, config


def _send_col_sharded_decode_rows(rows: torch.Tensor, mesh, cluster_shape, n_pad):
    """``rows``: [N, H] torch → col-sharded [1, 1, n_pad, H/4] per chip.

    Place the N users in the dim-2 tile-padded row slot (rows 0..N-1); pad to
    n_pad. This is the decode backbone's row layout; forward_decode now relabels
    the first ``max_batch_size`` rows into dim-0 as the user-batch.
    """
    N, H = rows.shape
    padded = torch.zeros(1, 1, n_pad, H, dtype=rows.dtype)
    padded[0, 0, :N, :] = rows
    return ttnn.from_torch(
        padded,
        device=mesh,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, dims=(None, 3), mesh_shape=cluster_shape),
    )


def _seed_dn_state(attn, recurrent_states_per_user: list[torch.Tensor]):
    """Write per-user recurrent state into ``attn.dn_state_buffer`` test-side.

    Each user state is CPU ref ``[1, n_v=48, K=128, V=128]``. The model's buffer
    is per-device ``[N, n_v_per_row=6, K, V]`` with the 48 heads sharded across
    the 8 mesh rows (replicated across the 4 cols). We stack the N users into
    ``[N, 48, K, V]`` and shard the head axis (dim 1) → mesh rows, replicate cols
    — exactly the model's recurrent-state sharding — then ttnn.copy into the
    persistent fp32 buffer.
    """
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
    """Write per-user conv state into ``attn.conv_state_buffer`` test-side.

    Each user conv state is CPU ref ``[1, conv_dim=10240, K-1=3]`` in the global
    Q[2048]|K[2048]|V[6144] channel layout. The model's per-device conv buffer is
    ``[N, K-1=3, conv_per_row=1280]`` where conv_per_row = [Q_row_256 | K_row_256
    | V_row_768] for that mesh row (see _build_conv_host_blocks). We reorder each
    user's channels into the per-row blocks, stack rows on a new axis, and shard
    that axis → mesh rows (replicate cols).
    """
    mesh_rows = attn.mesh_rows
    q_per_row = attn.q_per_row  # 256
    v_per_row = attn.v_per_row  # 768
    n_k_hd = attn.n_k_heads * attn.head_dim  # 2048 (global Q == global K width)
    Km1 = attn.conv_kernel - 1  # 3

    # Build [mesh_rows, N, Km1, conv_per_row] then shard dim 0 → rows.
    per_row_blocks = []
    for r in range(mesh_rows):
        users = []
        for cs in conv_states_per_user:  # cs: [1, 10240, 3]
            cs_kt = cs[0].transpose(0, 1)  # [3, 10240]
            qc = cs_kt[:, r * q_per_row : (r + 1) * q_per_row]  # [3, 256] (Q)
            kc = cs_kt[:, n_k_hd + r * q_per_row : n_k_hd + (r + 1) * q_per_row]  # [3, 256] (K)
            vc = cs_kt[:, 2 * n_k_hd + r * v_per_row : 2 * n_k_hd + (r + 1) * v_per_row]  # [3, 768] (V)
            users.append(torch.cat([qc, kc, vc], dim=-1).unsqueeze(0))  # [1, 3, 1280]
        per_row_blocks.append(torch.cat(users, dim=0).unsqueeze(0))  # [1, N, 3, 1280]
    seed_rm = torch.cat(per_row_blocks, dim=0)  # [mesh_rows, N, 3, 1280]

    is_fp32 = attn.dtype == ttnn.float32
    # Shard dim 0 across the 8 mesh rows (replicate the 4 cols) → per-device
    # [1, N, 3, 1280]; reshape away the leading 1 to match the buffer's per-device
    # [N, 3, 1280] before the in-place copy.
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


def _gdn_out_to_torch(out_tt, mesh, cluster_shape, N):
    """forward_decode output → torch [N, H].

    The GDN out_proj is 2D-TP: rows split the INPUT (head) dim and cols split the
    OUTPUT H dim (col-sharded H/4 per chip, see _build_weights w_out comment).
    After the row-axis all_reduce the per-chip output is the FULL reduced value
    over a disjoint H/4 col slice, replicated across the 8 mesh rows. So we
    concat the 4 col slices (mesh dim 1) along dim 3 to reconstruct full H, and
    take ONE row copy (mesh dim 0 — post-reduce all rows are identical).

    At batch-N forward_decode returns [1, 1, N, H/4] (users in dim-2); at B==1 it
    returns the legacy [1, 1, H/4] (3D). Handle both.
    """
    full = ttnn.to_torch(
        out_tt,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh, dims=(0, 3), mesh_shape=cluster_shape),
    )
    # dim 0 carries the 8 row replicas (identical post-reduce) → keep first.
    rows = cluster_shape[0]
    if full.shape[0] == rows:
        full = full[: full.shape[0] // rows]  # keep one row replica
    full = full.reshape(-1, _H)  # flatten leading dims, full H now in last dim
    return full[:N]


@pytest.mark.hardware
@pytest.mark.parametrize("N", [2], ids=lambda n: f"N{n}")
def test_qwen36_gdn_decode_batchN_pcc(bh_glx_mesh, N):
    """BATCH-N GDN decode block: seed N users' DeltaNet state from the CPU ref,
    run ONE batched GDN decode step, every user's GDN output PCC > 0.99."""
    from models.demos.qwen3_6_galaxy_v2.tt.llama_model import TtTransformer
    from models.demos.qwen3_6_galaxy_v2.tt.qwen36_model_config import TtQwen36ModelArgs

    os.environ.setdefault("QWEN36_TT_LANG_BETA_G", "0")

    state_dict = _load_state_dict_for_layer(_SNAPSHOT, _LAYER_IDX)
    print(f"[gdn-{N}u] loaded {len(state_dict)} weights for layer {_LAYER_IDX}")

    # ---- CPU reference: per-user GDN prefill (T_PREFILL) → state, then 1 decode.
    gdn_ref, _cfg = _build_ref_gdn(state_dict, _LAYER_IDX)
    x_users = []
    ref_decode_out = []  # [N] of [H]
    conv_states = []  # [N] of [1, conv_dim, K-1]
    recur_states = []  # [N] of [1, n_v, K, V]
    for u in range(N):
        torch.manual_seed(44 + u)
        x_full = torch.randn(1, _T_PREFILL + 1, _H, dtype=torch.bfloat16).float()
        x_users.append(x_full)
        with torch.no_grad():
            _, conv_s, recur_s = gdn_ref(x_full[:, :_T_PREFILL, :], conv_state=None, recurrent_state=None)
            dec_out, _, _ = gdn_ref(
                x_full[:, _T_PREFILL : _T_PREFILL + 1, :], conv_state=conv_s, recurrent_state=recur_s
            )
        conv_states.append(conv_s)
        recur_states.append(recur_s)
        ref_decode_out.append(dec_out[0, 0, :].float())  # [H]
    print(f"[gdn-{N}u] CPU references built (T_PREFILL={_T_PREFILL})")

    # ---- Build the TT model (max_batch_size=N) and grab the layer-0 GDN attn.
    args = TtQwen36ModelArgs(bh_glx_mesh)
    args.n_layers = 1
    args.linear_attention_pattern = ["linear_attention"]
    args.max_batch_size = N
    if hasattr(args, "tile_padded_batch_rows"):
        import math as _math

        args.tile_padded_batch_rows = args.tile_size * int(_math.ceil(N / args.tile_size))
    if hasattr(args, "batch_size_per_device_group"):
        args.batch_size_per_device_group = max(N // args.num_device_groups, 1)
    weight_cache_path = args.weight_cache_path(ttnn.bfloat8_b)
    weight_cache_path.mkdir(parents=True, exist_ok=True)
    model = TtTransformer(
        args=args,
        dtype=ttnn.bfloat8_b,
        mesh_device=bh_glx_mesh,
        state_dict=state_dict,
        weight_cache_path=weight_cache_path,
        paged_attention_config=None,
        use_paged_kv_cache=False,
    )
    attn = model.layers[0].attention
    print(f"[gdn-{N}u] TT model built; max_batch_size={attn.max_batch_size}")

    # ---- Seed per-user DeltaNet recurrent + conv state test-side.
    attn.clear_state()
    _seed_dn_state(attn, recur_states)
    _seed_conv_state(attn, conv_states)
    print(f"[gdn-{N}u] seeded recurrent + conv state for {N} users")

    # ---- Build the col-sharded decode input (N users in the dim-2 row slot).
    decode_rows = torch.cat([x_users[u][:, _T_PREFILL, :] for u in range(N)], dim=0).bfloat16()  # [N, H]
    n_pad = getattr(args, "tile_padded_batch_rows", 32)
    x_decode_tt = _send_col_sharded_decode_rows(decode_rows, bh_glx_mesh, args.cluster_shape, n_pad)

    # ---- ONE batched GDN decode step (call forward_decode directly).
    out_tt = attn.forward_decode(x_decode_tt, current_pos=None, rot_mats=None, kv_cache=None, page_table=None)
    print(f"[gdn-{N}u] forward_decode returned shape {list(out_tt.shape)}")

    tt_out = _gdn_out_to_torch(out_tt, bh_glx_mesh, args.cluster_shape, N)  # [N, H]

    # ---- Per-user PCC.
    failures = []
    for u in range(N):
        pcc_u = _pcc(tt_out[u], ref_decode_out[u])
        print(f"[gdn-{N}u] user {u}: GDN-out PCC = {pcc_u:.6f} (thresh={_PCC_THRESH})")
        if not (pcc_u > _PCC_THRESH):
            failures.append((u, pcc_u))

    assert not failures, f"batch-{N} GDN decode PCC < {_PCC_THRESH} for users " + ", ".join(
        f"{u}(pcc={p:.4f})" for u, p in failures
    )
    print(f"[gdn-{N}u] PASSED (all {N} users PCC > {_PCC_THRESH})")
