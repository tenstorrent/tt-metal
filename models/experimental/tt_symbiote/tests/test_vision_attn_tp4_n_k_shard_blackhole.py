# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""TP4 unit test for the dots.ocr vision attention on Blackhole 4×P300c.

Tests the canonical tensor-parallel MHA layout (head-parallel) on a Blackhole
4-chip (1×4) mesh with full 2D RoPE, matching TTNNDotsVisionAttention.forward.
All local activation tensors are in L1 interleaved memory; weights stay in DRAM.
At TP4 (3 heads/device) SDPA q/k/v I/O fits in L1 with q=128/k=1024 + BFP4 V.

TP4 sharding (head-parallel):
    qkv_proj   column-parallel  — N-shard (output head groups, dim=-1)
                                  each device: [H, QKV/TP] weights
                                  each device owns HEADS_PER_TP=3 heads
                                  no collective; x replicated
    RoPE       local per device — 2D factored (H×W positions, spatial_merge_size=2)
                                  ttnn.experimental.rotary_embedding (non-llama)
                                  preserves BFP8 dtype on Q and K
                                  cos/sin shape [1,1,S,128] replicated to all devices
    SDPA       local per device — [1, 3, S, 128] q/k/v per device
    o_proj     row-parallel     — K-shard (input head groups, dim=-2)
                                  each device: [HEADS_PER_TP×HEAD_DIM, H] weights
                                  reduce_scatter(dim=3) + all_gather(dim=3)
                                  (Ring topology; trace-compatible collective path)

Memory layout:
    x                 L1  bf16  (replicated)
    qkv               L1  bf8   (N-shard qkv_proj output; ~210KB/core at TP4)
    q, k              L1  bf8   (nlp_create_qkv_heads; ~72KB/core each)
    q, k after RoPE   L1  bf8   (rotary_embedding → L1)
    v                 L1  bf4  (typecast; halves V CB vs bf8 for k_chunk=1024)
    ctx_sdpa          L1  bf16  (SDPA output; ~135KB/core at TP4 with 3 heads)
    ctx_concat        L1  bf16  (nlp_concat_heads; [1,1,S,384] ~135KB/core)
    partial           L1  bf16  (K-shard o_proj output; [1,1,S,H] ~528KB/core)
    scattered         L1  bf16  (reduce_scatter; [1,1,S,H/TP] ~135KB/core)
    out               L1  bf16  (all_gather; [1,1,S,H] ~528KB/core)
    cos/sin           DRAM bf8  ([1,1,S,128] half-half layout, replicated)

Reference: non-TP float torch forward with the same 2D RoPE applied to Q and K.

Run::

    MESH_DEVICE=P150x4 pytest \\
        models/experimental/tt_symbiote/tests/test_vision_attn_tp4_n_k_shard_blackhole.py -s

    MESH_DEVICE=P300x2 pytest \\
        models/experimental/tt_symbiote/tests/test_vision_attn_tp4_n_k_shard_blackhole.py -s
"""

from __future__ import annotations

import os

import pytest
import torch
import torch.nn.functional as F
from loguru import logger
from ttnn.operations.transformer import SDPAProgramConfig

import ttnn
from models.common.utility_functions import comp_pcc, is_blackhole, run_for_blackhole

# ---------------------------------------------------------------------------
# dots.ocr model source + expected vision attention shapes.
# ---------------------------------------------------------------------------

DOTS_OCR_MODEL_ID = "rednote-hilab/dots.ocr"

HIDDEN = 1536  # hidden dimension H  (vision encoder width)
NUM_HEADS = 12  # total Q = KV heads (head-parallel → 3 heads per TP device)
HEAD_DIM = 128  # head dimension d
QKV_OUT = NUM_HEADS * 3 * HEAD_DIM  # 4608  (fused Q|K|V output)
TP = 4
HEADS_PER_TP = NUM_HEADS // TP  # 3
QKV_OUT_PER_TP = QKV_OUT // TP  # 1152  (tile-aligned: 1152/32=36)
SEQ_LEN = 11264  # vision bucket (352 tiles)

# 2D RoPE parameters (must match dots.ocr vision config).
# SEQ_LEN = GRID_H × GRID_W (T=1): 88 × 128 = 11264 ✓; both even (÷ SPATIAL_MERGE_SIZE=2).
GRID_H = 88
GRID_W = 128
SPATIAL_MERGE_SIZE = 2
ROPE_THETA = 10000.0

PCC_THRESHOLD = 0.95

SUPPORTED_MESH_ENVS = frozenset({"P150x4", "P300x2"})


# ---------------------------------------------------------------------------
# Model loading.
# ---------------------------------------------------------------------------


def _resolve_model_path() -> str:
    env_path = os.environ.get("DOTS_OCR_MODEL_PATH")
    if env_path and os.path.isdir(env_path):
        return env_path
    try:
        from huggingface_hub import snapshot_download

        return snapshot_download(DOTS_OCR_MODEL_ID)
    except Exception:
        return DOTS_OCR_MODEL_ID


def _load_vision_attn():
    """Return blocks[0].attn from the dots.ocr vision tower (bf16, eval mode).

    Uses ``from_config`` so the test runs without downloading pretrained weights.
    Set ``DOTS_OCR_MODEL_PATH`` to a local checkpoint to test real weights.
    Returns the HF attention module and the resolved qkv / o_proj sub-modules.
    """
    try:
        from transformers import AutoConfig, AutoModelForCausalLM
    except ImportError:
        pytest.skip("transformers required for dots.ocr weight loading")

    model_path = _resolve_model_path()
    try:
        model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    except Exception as exc:
        pytest.skip(f"dots.ocr config unavailable at {model_path!r}: {exc}")

    model_config.num_hidden_layers = 1
    vision_config = getattr(model_config, "vision_config", None)
    if vision_config is not None:
        for attr in ("num_hidden_layers", "num_layers", "depth"):
            if hasattr(vision_config, attr):
                setattr(vision_config, attr, 1)

    try:
        hf_model = (
            AutoModelForCausalLM.from_config(model_config, trust_remote_code=True).to(dtype=torch.bfloat16).eval()
        )
    except Exception as exc:
        pytest.skip(f"dots.ocr model init failed: {exc}")

    blocks = getattr(hf_model.vision_tower, "blocks", getattr(hf_model.vision_tower, "layers", None))
    assert blocks is not None, "vision tower must expose .blocks or .layers"
    attn = blocks[0].attn

    qkv = getattr(attn, "qkv", getattr(attn, "qkv_proj", None))
    proj = getattr(attn, "proj", getattr(attn, "o_proj", getattr(attn, "out_proj", None)))
    assert qkv is not None, "vision attn must expose .qkv or .qkv_proj"
    assert proj is not None, "vision attn must expose .proj, .o_proj, or .out_proj"

    # Verify expected production shapes.
    assert int(qkv.weight.shape[1]) == HIDDEN, f"expected qkv input={HIDDEN}, got {qkv.weight.shape[1]}"
    assert int(qkv.weight.shape[0]) == QKV_OUT, f"expected qkv output={QKV_OUT}, got {qkv.weight.shape[0]}"
    assert qkv.bias is None, "production vision attn qkv is bias-free"
    assert proj.bias is None, "production vision attn o_proj is bias-free"

    return qkv, proj


# ---------------------------------------------------------------------------
# Mesh helpers (mirrored from test_vision_mlp_tp4_n_k_shard_blackhole.py).
# ---------------------------------------------------------------------------


def _mesh_env() -> str:
    env = os.environ.get("MESH_DEVICE", "")
    if env in SUPPORTED_MESH_ENVS:
        return env
    return "P150x4" if is_blackhole() else ""


def _device_params():
    if not _mesh_env():
        pytest.skip(f"requires MESH_DEVICE ∈ {sorted(SUPPORTED_MESH_ENVS)} and Blackhole arch")
    return {
        "trace_region_size": 0,
        "num_command_queues": 1,
        "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
        "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
    }


def _open_tp4_mesh(updated_device_params):
    if len(ttnn.get_pcie_device_ids()) < TP:
        pytest.skip(f"TP={TP} requires at least {TP} PCIe devices")

    req_shape = (1, TP)
    sys_shape = tuple(ttnn._ttnn.multi_device.SystemMeshDescriptor().shape())
    sys_num = sys_shape[0] * sys_shape[1]

    if sys_num == TP:
        mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(*req_shape), **updated_device_params)
        if sys_shape == (2, 2):
            mesh.reshape(ttnn.MeshShape(*req_shape))
        return mesh, None

    parent_shape = sys_shape
    if not (req_shape[0] <= parent_shape[0] and req_shape[1] <= parent_shape[1]):
        rotated = (parent_shape[1], parent_shape[0])
        if req_shape[0] <= rotated[0] and req_shape[1] <= rotated[1]:
            parent_shape = rotated
        else:
            pytest.skip(f"TP mesh {req_shape} does not fit system mesh {sys_shape}")

    parent = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(*parent_shape), **updated_device_params)
    tp_mesh = parent.create_submesh(ttnn.MeshShape(*req_shape))
    tp_mesh.reshape(ttnn.MeshShape(*req_shape))
    return tp_mesh, parent


@pytest.fixture(scope="function")
def bh_tp4_mesh_device(device_params, silicon_arch_name, silicon_arch_blackhole):
    from tests.scripts.common import get_updated_device_params

    _ = silicon_arch_name, silicon_arch_blackhole
    if not is_blackhole():
        pytest.skip("requires Blackhole (P150x4 or P300x2)")
    if not _mesh_env():
        pytest.skip(f"requires MESH_DEVICE ∈ {sorted(SUPPORTED_MESH_ENVS)}")

    updated = get_updated_device_params(dict(device_params))
    fabric_config = updated.pop("fabric_config", None)
    fabric_tensix_config = updated.pop("fabric_tensix_config", None)
    reliability_mode = updated.pop("reliability_mode", None)
    if fabric_config:
        ttnn.set_fabric_config(
            fabric_config,
            reliability_mode or ttnn.FabricReliabilityMode.STRICT_INIT,
            None,
            fabric_tensix_config or ttnn.FabricTensixConfig.DISABLED,
            ttnn.FabricUDMMode.DISABLED,
            ttnn.FabricManagerMode.DEFAULT,
        )

    mesh_device, parent_mesh = _open_tp4_mesh(updated)
    if mesh_device.get_num_devices() != TP:
        pytest.skip(f"expected {TP} devices, got {mesh_device.get_num_devices()}")
    if mesh_device.arch() != ttnn.Arch.BLACKHOLE:
        pytest.skip(f"expected Blackhole, got {mesh_device.arch().name}")

    yield mesh_device

    ttnn.close_mesh_device(mesh_device)
    if parent_mesh is not None:
        ttnn.close_mesh_device(parent_mesh)
    if fabric_config:
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


# ---------------------------------------------------------------------------
# Program configs.
# ---------------------------------------------------------------------------

_DTYPE_TILE_BYTES = {
    ttnn.bfloat16: 2048,
    ttnn.bfloat8_b: 1088,
    ttnn.bfloat4_b: 544,
}

_L1_PER_CORE_BYTES = 1500 * 1024


def _largest_divisor_le(value: int, limit: int) -> int:
    for c in range(min(value, limit), 0, -1):
        if value % c == 0:
            return c
    return 1


def _best_dst_subblock(ob_h: int, per_core_n: int, *, dst_budget: int = 8) -> tuple[int, int, int]:
    """Return (out_subblock_h, out_subblock_w, dst_area) maximising DST register use."""
    best_area = 0
    best_h = best_w = 1
    for h in range(min(ob_h, dst_budget), 0, -1):
        if ob_h % h != 0:
            continue
        for w in range(min(per_core_n, dst_budget // h), 0, -1):
            if per_core_n % w != 0:
                continue
            area = h * w
            if area > best_area:
                best_area = area
                best_h = h
                best_w = w
    return best_h, best_w, best_area


def _bh_tp4_qkv_pc(device):
    """Hardware-swept QKV matmul for 11264×1536×1152 on BH P150 11×10 (~148 μs).

    Silicon sweep 2026-06-07: grid=(9,8) tm=False M=44 N=4 obh=22 ibw=8 sub=(2,4).
    """
    grid = device.compute_with_storage_grid_size()
    if int(grid.x) == 11 and int(grid.y) == 10:
        return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(9, 8),
            in0_block_w=8,
            out_subblock_h=2,
            out_subblock_w=4,
            out_block_h=22,
            out_block_w=4,
            per_core_M=44,
            per_core_N=4,
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=False,
        )
    return _bh_tp4_vision_attn_pc(
        device,
        SEQ_LEN,
        HIDDEN,
        QKV_OUT_PER_TP,
        l1_resident_bytes_per_core=0,
    )


def _bh_tp4_o_proj_pc(device):
    """Hardware-swept o_proj for 11264×384×1536, BFP8×BFP8→BFP8 L1 (~80 μs).

    Silicon sweep 2026-06-07: grid=(8,8) tm=False M=44 N=6 obh=22 ibw=6 sub=(2,3).
    """
    grid = device.compute_with_storage_grid_size()
    if int(grid.x) == 11 and int(grid.y) == 10:
        return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            in0_block_w=6,
            out_subblock_h=2,
            out_subblock_w=3,
            out_block_h=22,
            out_block_w=6,
            per_core_M=44,
            per_core_N=6,
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=False,
        )
    return _bh_tp4_vision_attn_pc(
        device,
        SEQ_LEN,
        HEADS_PER_TP * HEAD_DIM,
        HIDDEN,
        l1_resident_bytes_per_core=0,
    )


def _bh_tp4_vision_attn_pc(
    device,
    m_dim: int,
    k_dim: int,
    n_dim: int,
    *,
    l1_resident_bytes_per_core: int = 0,
):
    """Optimal 2D-mcast program config for BH TP4 vision attention matmuls (o_proj etc.).

    Probes both transpose_mcast orientations, M/N core factorisations, and all
    valid out_block_h values.  On BH P150x the grid is (grid_x=11, grid_y=10).

    Scoring (lexicographic, higher is better):
      1. dst_area = out_subblock_h × out_subblock_w  (primary compute lever)
      2. transpose_mcast=False preferred (M→rows; faster when in0 is L1)
      3. smaller out_block_h at equal dst_area (less L1 CB thrash)
      4. fewer total cores (less multicast overhead)

    Fixed CB costs (independent of out_block_h):
      • in1 double-buffered (bf8 weight):  2 × in0_block_w × per_N × 1088
      • partial-sums (bf16 packer acc):    per_M × per_N × 2048

    Variable CB costs (scale with out_block_h):
      • in0 double-buffered (bf16 act):    2 × ob_h × in0_block_w × 2048
      • interm + out (bf16):               2 × ob_h × per_N × 2048
    """
    grid = device.compute_with_storage_grid_size()
    grid_x, grid_y = int(grid.x), int(grid.y)
    tile = 32

    if m_dim % tile or k_dim % tile or n_dim % tile:
        return None

    m_tiles = m_dim // tile
    k_tiles = k_dim // tile
    n_tiles = n_dim // tile

    in0_block_w = _largest_divisor_le(k_tiles, 8)
    cb_budget_bytes = max(256 * 1024, _L1_PER_CORE_BYTES - l1_resident_bytes_per_core)

    def _per_core_n_candidates(n_max: int) -> list[int]:
        cands = {n for n in range(1, min(n_tiles, 24) + 1) if n_tiles % n == 0}
        cands.add((n_tiles + n_max - 1) // n_max)
        return sorted(cands)

    best_pc = None
    best_score = (-1, -1, -(2**31), -(2**31))  # (dst_area, tm_false, -ob_h, -cores)

    for transpose_mcast in (True, False):
        m_grid_max = grid_x if transpose_mcast else grid_y
        n_grid_max = grid_y if transpose_mcast else grid_x

        for eff_mg in range(min(m_tiles, m_grid_max), 0, -1):
            if m_tiles % eff_mg != 0:
                continue
            per_core_m = m_tiles // eff_mg
            if per_core_m > 64:
                continue

            for per_core_n in _per_core_n_candidates(n_grid_max):
                if per_core_n > 24:
                    continue
                actual_ng = (n_tiles + per_core_n - 1) // per_core_n
                if actual_ng > n_grid_max:
                    continue

                in1_fixed = 2 * in0_block_w * per_core_n * 1088
                partial_fixed = per_core_m * per_core_n * 2048
                fixed_cb_bytes = in1_fixed + partial_fixed
                if fixed_cb_bytes >= cb_budget_bytes:
                    continue

                remaining_bytes = cb_budget_bytes - fixed_cb_bytes
                best_ob_h = 0
                best_sub = (1, 1, 0)

                for ob_h in range(per_core_m, 0, -1):
                    if per_core_m % ob_h != 0:
                        continue
                    in0_bytes = 2 * ob_h * in0_block_w * 2048
                    interm_bytes = ob_h * per_core_n * 2048
                    out_bytes = ob_h * per_core_n * 2048
                    if in0_bytes + interm_bytes + out_bytes > remaining_bytes:
                        continue
                    sub = _best_dst_subblock(ob_h, per_core_n)
                    if sub[2] > best_sub[2] or (sub[2] == best_sub[2] and ob_h < best_ob_h):
                        best_ob_h = ob_h
                        best_sub = sub

                if best_ob_h == 0:
                    continue

                sub_h, sub_w, dst_area = best_sub
                total_cores = eff_mg * actual_ng
                score = (dst_area, int(not transpose_mcast), -best_ob_h, -total_cores)

                if score > best_score:
                    best_score = score
                    gx = eff_mg if transpose_mcast else actual_ng
                    gy = actual_ng if transpose_mcast else eff_mg
                    best_pc = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                        compute_with_storage_grid_size=(gx, gy),
                        in0_block_w=in0_block_w,
                        out_subblock_h=sub_h,
                        out_subblock_w=sub_w,
                        out_block_h=best_ob_h,
                        out_block_w=per_core_n,
                        per_core_M=per_core_m,
                        per_core_N=per_core_n,
                        transpose_mcast=transpose_mcast,
                        fused_activation=None,
                        fuse_batch=False,
                    )

    return best_pc


def _bh_tp4_sdpa_pc(device) -> SDPAProgramConfig:
    """Hardware-swept SDPA for TP4 vision attn on BH P150 11×10 (~1006 μs).

    Shape per device: [1, 3, 11264, 128] BFP8 Q/K + BFP4 V → L1 out, non-causal.
    Swept on silicon 2026-06-06; ~19% faster than q=256/k=1024 baseline (~1241 μs).

    Winner: full 11×10 grid, q_chunk=128, k_chunk=1024, exp_approx_mode=True.
    Smaller q_chunk improves core utilisation at S=11264 (88 q-blocks vs 44);
    k_chunk=1024 still needs BFP4 V to fit the scores CB in L1.
    """
    grid = device.compute_with_storage_grid_size()
    if int(grid.x) == 11 and int(grid.y) == 10:
        return SDPAProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(grid.x, grid.y),
            q_chunk_size=128,
            k_chunk_size=1024,
            exp_approx_mode=True,
        )
    return SDPAProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(grid.x, grid.y),
        q_chunk_size=256,
        k_chunk_size=1024,
        exp_approx_mode=True,
    )


# ---------------------------------------------------------------------------
# Weight upload helpers.
# ---------------------------------------------------------------------------


def _to_tt_weight(w: torch.Tensor) -> torch.Tensor:
    return w.t().contiguous()


def _rearrange_qkv_for_tp(qkv_weight: torch.Tensor) -> torch.Tensor:
    """Reorder fused QKV weight rows for head-parallel TP.

    A naïve N-shard of the flat [Q_all | K_all | V_all] weight gives each
    device a contiguous slice that mixes Q/K/V for different head groups:
        device 0 columns 0:1152  →  Q heads 0–8 only (no K or V!)

    We need each device to receive [Q_local | K_local | V_local] for its
    own head group.  Reordering the rows of the torch weight before upload
    achieves this: after transposing to TT layout [HIDDEN, 3H] and N-shard
    the columns are [Q_dev0|K_dev0|V_dev0 | Q_dev1|K_dev1|V_dev1 | ...].

    Input shape:  [3*HIDDEN, HIDDEN] = [Q_all | K_all | V_all]
    Output shape: [3*HIDDEN, HIDDEN] with rows reordered per-device block
    """
    feat = HEADS_PER_TP * HEAD_DIM  # features per device per Q/K/V = 384
    q_w = qkv_weight[:HIDDEN]  # [1536, HIDDEN]
    k_w = qkv_weight[HIDDEN : 2 * HIDDEN]
    v_w = qkv_weight[2 * HIDDEN :]
    chunks = []
    for i in range(TP):
        chunks += [
            q_w[i * feat : (i + 1) * feat],
            k_w[i * feat : (i + 1) * feat],
            v_w[i * feat : (i + 1) * feat],
        ]
    return torch.cat(chunks, dim=0)  # [3H, HIDDEN] reordered


def _upload(t: torch.Tensor, mesh, mapper, dtype=ttnn.bfloat8_b, *, mem=ttnn.DRAM_MEMORY_CONFIG) -> ttnn.Tensor:
    return ttnn.from_torch(t, device=mesh, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_config=mem, mesh_mapper=mapper)


# ---------------------------------------------------------------------------
# TP4 forward: all local activation tensors in L1.
# ---------------------------------------------------------------------------


def _attn_tp4_l1_forward(
    x_tt,  # [1,1,S,H]         bf16  L1    replicated
    qkv_w_tt,  # [H, QKV/TP]       bf8   DRAM  N-shard
    o_w_tt,  # [HEADS/TP*D, H]   bf8   DRAM  K-shard  (output → L1)
    *,
    qkv_pc,
    o_pc,
    compute_cfg,
    sdpa_cfg,
    cos_sin=None,  # (cos_tt, sin_tt) | None — [1,1,S,128] bf8 DRAM replicated
) -> ttnn.Tensor:
    """TP4 vision MHA — all local activation I/O in L1.

    Column-parallel qkv_proj (N-shard):
      qkv     = qkv_proj(x_rep)  →  [1,1,S,QKV/TP]  bf8   L1
      q,k,v   = nlp_create_qkv_heads(qkv, num_heads=3)
                                  →  [1,3,S,D]   bf8   L1  (per device)

    2D RoPE (non-llama kernel, preserves BFP8 dtype):
      q       = rotary_embedding(q, cos, sin)  →  L1 bf8
      k       = rotary_embedding(k, cos, sin)  →  L1 bf8

    SDPA (local, per device; all q/k/v I/O in L1):
      v       = typecast(v → bf4) →  L1  (bf4 halves V CB for k_chunk=1024)
      ctx     = SDPA(q, k, v)    →  L1    (TP4: ~831KB SDPA CBs + ~180KB q/k/v
                                            + ~79KB ctx < 1500KB/core)
      ctx     = concat_heads(ctx) →  [1,1,S,384]  bf16  L1

    Row-parallel o_proj (K-shard):
      partial = o_proj(ctx)               →  [1,1,S,H]      bf16  L1
      scattered = reduce_scatter(dim=3)   →  [1,1,S,H/TP]   bf16  L1  (~135KB/core)
      out       = all_gather(dim=3)       →  [1,1,S,H]      bf16  L1  (~528KB/core)

    reduce_scatter+all_gather decomposes the all-reduce into two trace-compatible
    Ring CCL ops (topology=Ring for BH).  dim=3 (H dimension) aligns each
    device's scatter shard with its head group, minimising off-chip traffic vs dim=2.
    """
    l1 = ttnn.L1_MEMORY_CONFIG

    # ---- column-parallel QKV (N-shard) ----------------------------------------
    qkv = ttnn.linear(
        x_tt,
        qkv_w_tt,
        bias=None,
        dtype=ttnn.bfloat8_b,
        memory_config=l1,
        compute_kernel_config=compute_cfg,
        program_config=qkv_pc,
    )
    ttnn.deallocate(x_tt)

    q, k, v = ttnn.experimental.nlp_create_qkv_heads(
        qkv,
        num_heads=HEADS_PER_TP,
        num_kv_heads=HEADS_PER_TP,
        transpose_k_heads=False,
        memory_config=l1,
    )
    ttnn.deallocate(qkv)

    # ---- 2D RoPE (non-llama kernel, preserves BFP8 dtype) ----------------------
    # Q/K stay L1 BFP8 through rotary so SDPA reads q/k/v from L1.
    if cos_sin is not None:
        cos, sin = cos_sin
        q = ttnn.experimental.rotary_embedding(q, cos, sin, memory_config=l1)
        k = ttnn.experimental.rotary_embedding(k, cos, sin, memory_config=l1)

    # ---- SDPA (local per device; all q/k/v I/O in L1) --------------------------
    # V → bf4 L1: halves V CB vs bf8 so k_chunk=1024 scores CB fits in L1.
    v = ttnn.typecast(v, ttnn.bfloat4_b, memory_config=l1)
    ctx = ttnn.transformer.scaled_dot_product_attention(
        q,
        k,
        v,
        is_causal=False,
        attn_mask=None,
        program_config=sdpa_cfg,
        compute_kernel_config=compute_cfg,
        memory_config=l1,
    )
    ttnn.deallocate(q)
    ttnn.deallocate(k)
    ttnn.deallocate(v)

    # nlp_concat_heads: [1,3,S,D] L1 → [1,1,S,3D] L1
    ctx = ttnn.experimental.nlp_concat_heads(ctx, memory_config=l1)

    # ---- row-parallel o_proj (K-shard) ----------------------------------------
    partial = ttnn.linear(
        ctx,
        o_w_tt,
        bias=None,
        dtype=ttnn.bfloat8_b,
        memory_config=l1,
        compute_kernel_config=compute_cfg,
        program_config=o_pc,
    )
    ttnn.deallocate(ctx)

    # Ring reduce_scatter (dim=3, H-dim): [1,1,S,H] → [1,1,S,H/TP] bf16 L1 (~135KB/core)
    # Scattering on dim=3 (H) means each device retains its own head-group slice
    # of the partial sum, minimising cross-chip traffic vs scattering on dim=2 (S).
    scattered = ttnn.reduce_scatter(
        partial,
        dim=3,
        num_links=1,
        cluster_axis=1,
        memory_config=l1,
        topology=ttnn.Topology.Ring,
    )
    ttnn.deallocate(partial)

    # Ring all_gather (dim=3): [1,1,S,H/TP] → [1,1,S,H] bf16 L1 (~528KB/core)
    out = ttnn.all_gather(
        scattered,
        dim=3,
        num_links=1,
        cluster_axis=1,
        memory_config=l1,
        topology=ttnn.Topology.Ring,
    )
    ttnn.deallocate(scattered)
    return out


# ---------------------------------------------------------------------------
# 2D RoPE helpers (CPU reference, matching TTNNDotsVision2DRoPE logic).
# ---------------------------------------------------------------------------


def _rope_cos_sin_torch() -> tuple[torch.Tensor, torch.Tensor]:
    """Compute 2D RoPE cos/sin tables for the test grid (T=1, H=88, W=128).

    Replicates TTNNDotsVision2DRoPE._compute_cos_sin_torch exactly:
      rotary_dim = HEAD_DIM // 2 = 64;  inv_freq shape [32]
      spatial-merge rearrangement (sms=2) on H/W position grids
      half-half layout: [cos_h|cos_w|cos_h|cos_w] → shape [1,1,S,128]

    The "half-half" duplication is what lets ttnn.experimental.rotary_embedding
    (non-llama kernel) preserve input dtype (BFP8) without extra typecasts.
    """
    sms = SPATIAL_MERGE_SIZE  # 2
    rotary_dim = HEAD_DIM // 2  # 64
    inv_freq = 1.0 / (ROPE_THETA ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim))  # [32]

    h, w = GRID_H, GRID_W  # 88, 128  (88*128 == SEQ_LEN ✓)

    h_ids = torch.arange(h, dtype=torch.float32)
    w_ids = torch.arange(w, dtype=torch.float32)
    h_grid = h_ids.unsqueeze(1).expand(h, w)  # [88, 128]  h_grid[i,j] = i
    w_grid = w_ids.unsqueeze(0).expand(h, w)  # [88, 128]  w_grid[i,j] = j

    # Spatial-merge rearrangement: groups sms×sms adjacent patches so that
    # merged tokens are contiguous in sequence order.
    h_grid = h_grid.reshape(h // sms, sms, w // sms, sms).permute(0, 2, 1, 3).reshape(-1)
    w_grid = w_grid.reshape(h // sms, sms, w // sms, sms).permute(0, 2, 1, 3).reshape(-1)

    freqs_h = h_grid.unsqueeze(1) * inv_freq.unsqueeze(0)  # [S, 32]
    freqs_w = w_grid.unsqueeze(1) * inv_freq.unsqueeze(0)  # [S, 32]

    cos_half = torch.cat([torch.cos(freqs_h), torch.cos(freqs_w)], dim=-1)  # [S, 64]
    sin_half = torch.cat([torch.sin(freqs_h), torch.sin(freqs_w)], dim=-1)  # [S, 64]

    cos_full = torch.cat([cos_half, cos_half], dim=-1)  # [S, 128]  half-half layout
    sin_full = torch.cat([sin_half, sin_half], dim=-1)  # [S, 128]

    cos = cos_full.unsqueeze(0).unsqueeze(0).to(torch.bfloat16)  # [1, 1, S, 128]
    sin = sin_full.unsqueeze(0).unsqueeze(0).to(torch.bfloat16)  # [1, 1, S, 128]
    return cos, sin


def _rotate_half_torch(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


# ---------------------------------------------------------------------------
# Torch reference (full forward: QKV → 2D RoPE → 12-head SDPA → o_proj).
# ---------------------------------------------------------------------------


def _torch_ref(qkv_weight, o_weight, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Non-TP float reference: QKV → 2D RoPE → 12-head SDPA → o_proj.

    cos/sin shapes: [1, 1, S, 128] — broadcast across all 12 heads.
    """
    S, H = int(x.shape[2]), int(x.shape[3])
    xf = x.reshape(S, H).float()

    # Fused QKV: weight [3H, H]
    qkv = F.linear(xf, qkv_weight.float())  # [S, 3H]
    q = qkv[:, :H].reshape(S, NUM_HEADS, HEAD_DIM).permute(1, 0, 2).unsqueeze(0)  # [1,12,S,128]
    k = qkv[:, H : 2 * H].reshape(S, NUM_HEADS, HEAD_DIM).permute(1, 0, 2).unsqueeze(0)
    v = qkv[:, 2 * H :].reshape(S, NUM_HEADS, HEAD_DIM).permute(1, 0, 2).unsqueeze(0)

    # 2D RoPE: q_embed = q*cos + rotate_half(q)*sin  (broadcast [1,1,S,128] over heads)
    cos_f = cos.float()  # [1, 1, S, 128]
    sin_f = sin.float()  # [1, 1, S, 128]
    q = q * cos_f + _rotate_half_torch(q) * sin_f
    k = k * cos_f + _rotate_half_torch(k) * sin_f

    ctx = F.scaled_dot_product_attention(q, k, v, is_causal=False)  # [1,12,S,128]
    ctx = ctx.permute(0, 2, 1, 3).reshape(1, 1, S, H)  # [1,1,S,H]
    out = F.linear(ctx.reshape(S, H), o_weight.float()).reshape(1, 1, S, H)
    return out.to(torch.float32)


# ---------------------------------------------------------------------------
# Test.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device_params", [_device_params()], indirect=True)
@run_for_blackhole()
def test_vision_attn_tp4_n_k_shard_blackhole(bh_tp4_mesh_device):
    """Vision MHA TP4 on Blackhole P300c: real dots.ocr weights, all-L1 activations.

    Head-parallel (3 heads/device).  Column-parallel qkv_proj (N-shard) +
    local SDPA + row-parallel o_proj (K-shard) + Ring reduce_scatter+all_gather.
    All local activation I/O is L1 interleaved; only weights and cos/sin tables are DRAM.
    PCC is checked against the float torch reference (no rotary) on all 4 devices.
    """
    mesh_device = bh_tp4_mesh_device
    mesh_env = _mesh_env()

    assert NUM_HEADS % TP == 0, f"NUM_HEADS={NUM_HEADS} must divide TP={TP}"
    assert QKV_OUT_PER_TP % 32 == 0, f"QKV_OUT_PER_TP={QKV_OUT_PER_TP} must be tile-aligned"
    assert GRID_H * GRID_W == SEQ_LEN, f"GRID_H×GRID_W={GRID_H*GRID_W} ≠ SEQ_LEN={SEQ_LEN}"

    # ---- load real dots.ocr vision attention weights ------------------------
    qkv_mod, proj_mod = _load_vision_attn()
    qkv_weight = qkv_mod.weight.data  # [4608, 1536]
    o_weight = proj_mod.weight.data  # [1536, 1536]

    torch.manual_seed(0xA77E_B1AC)
    x = torch.randn(1, 1, SEQ_LEN, HIDDEN, dtype=torch.bfloat16) * 0.1

    # ---- 2D RoPE cos/sin (CPU) -----------------------------------------------
    cos_torch, sin_torch = _rope_cos_sin_torch()  # [1, 1, S, 128] bfloat16

    # Torch reference (float, with 2D RoPE applied to Q and K).
    with torch.no_grad():
        ref = _torch_ref(qkv_weight, o_weight, x, cos_torch, sin_torch)

    # ---- mesh mappers -------------------------------------------------------
    rep = ttnn.ReplicateTensorToMesh(mesh_device)
    shard_n = ttnn.ShardTensorToMesh(mesh_device, dim=-1)  # column-parallel (N)
    shard_k = ttnn.ShardTensorToMesh(mesh_device, dim=-2)  # row-parallel    (K)

    # ---- weight upload (DRAM) -----------------------------------------------
    # qkv_proj: rearrange rows so each shard is [Q_local|K_local|V_local], then
    # N-shard on TT [HIDDEN, 3H].  Without the reorder, device 0 gets Q cols 0–8
    # (9 heads worth of Q only) and nlp_create_qkv_heads treats them as Q+K+V.
    qkv_w_tt = _upload(_to_tt_weight(_rearrange_qkv_for_tp(qkv_weight)), mesh_device, shard_n, dtype=ttnn.bfloat8_b)

    # o_proj: K-shard on TT [HEADS_TOT*D, H] → each device [HEADS/TP*D, H]
    o_w_tt = _upload(_to_tt_weight(o_weight), mesh_device, shard_k, dtype=ttnn.bfloat8_b)

    # ---- cos/sin upload (DRAM, replicated — same for all 4 TP devices) ------
    # Shape [1, 1, S, 128] — the non-llama rotary_embedding kernel broadcasts
    # these tables over the HEADS_PER_TP dimension.
    cos_tt = _upload(cos_torch, mesh_device, rep, dtype=ttnn.bfloat8_b)
    sin_tt = _upload(sin_torch, mesh_device, rep, dtype=ttnn.bfloat8_b)

    # ---- input upload (L1) --------------------------------------------------
    x_tt = ttnn.from_torch(
        x,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        mesh_mapper=rep,
    )

    # ---- program configs (computed once, logged, then passed to forward) ----
    grid = mesh_device.compute_with_storage_grid_size()
    nc = int(grid.x) * int(grid.y)
    tile = 32
    m_tiles = SEQ_LEN // tile
    h_tiles = HIDDEN // tile
    qkv_tiles = QKV_OUT_PER_TP // tile  # 36
    ctx_tiles = (HEADS_PER_TP * HEAD_DIM) // tile  # 12

    xB = ((m_tiles * h_tiles + nc - 1) // nc) * _DTYPE_TILE_BYTES[ttnn.bfloat16]
    qkvB = ((m_tiles * qkv_tiles + nc - 1) // nc) * _DTYPE_TILE_BYTES[ttnn.bfloat8_b]
    ctxB = ((m_tiles * ctx_tiles + nc - 1) // nc) * _DTYPE_TILE_BYTES[ttnn.bfloat16]
    partB = ((m_tiles * h_tiles + nc - 1) // nc) * _DTYPE_TILE_BYTES[ttnn.bfloat16]

    # Matmul CB budget: reserve only the in0 activation shard per core (x or ctx).
    # The output buffer is allocated separately and must not be double-counted —
    # doing so capped QKV out_block_h at 8 (4 outer-M iters) instead of 16 (2).
    qkv_pc = _bh_tp4_qkv_pc(mesh_device)
    o_pc = _bh_tp4_o_proj_pc(mesh_device)
    sdpa_cfg = _bh_tp4_sdpa_pc(mesh_device)

    compute_cfg = ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )

    def _log_pc(label, pc):
        if pc is not None:
            tm = "T" if pc.transpose_mcast else "F"
            outer_m = pc.per_core_M // pc.out_block_h
            logger.info(
                f"  {label}: grid={pc.compute_with_storage_grid_size} "
                f"tm={tm} per_core_M={pc.per_core_M} per_core_N={pc.per_core_N} "
                f"in0_block_w={pc.in0_block_w} out_block_h={pc.out_block_h} "
                f"outer_M_iters={outer_m} "
                f"sub=({pc.out_subblock_h},{pc.out_subblock_w}) "
                f"dst_area={pc.out_subblock_h * pc.out_subblock_w}"
            )
        else:
            logger.info(f"  {label}: None (auto-config)")

    logger.info(f"[vision_attn_tp4_bh grid=({grid.x},{grid.y}) num_cores={nc} " f"heads_per_tp={HEADS_PER_TP}]")
    logger.info(f"  x={xB//1024}KB  qkv={qkvB//1024}KB  ctx={ctxB//1024}KB  " f"partial={partB//1024}KB per core")
    _log_pc(f"qkv  [M={SEQ_LEN},K={HIDDEN},N={QKV_OUT_PER_TP}]", qkv_pc)
    _log_pc(f"o    [M={SEQ_LEN},K={HEADS_PER_TP*HEAD_DIM},N={HIDDEN}]", o_pc)
    logger.info(
        f"  sdpa: q_chunk={sdpa_cfg.q_chunk_size} k_chunk={sdpa_cfg.k_chunk_size} "
        f"grid=({sdpa_cfg.compute_with_storage_grid_size.x},"
        f"{sdpa_cfg.compute_with_storage_grid_size.y}) exp_approx={sdpa_cfg.exp_approx_mode}"
    )
    logger.info(
        f"  rope: grid=({GRID_H},{GRID_W}) sms={SPATIAL_MERGE_SIZE} theta={ROPE_THETA} "
        f"cos/sin=[1,1,{SEQ_LEN},{HEAD_DIM}] bf8 DRAM replicated"
    )

    # ---- TP4 forward --------------------------------------------------------
    out_tt = _attn_tp4_l1_forward(
        x_tt,
        qkv_w_tt,
        o_w_tt,
        qkv_pc=qkv_pc,
        o_pc=o_pc,
        compute_cfg=compute_cfg,
        sdpa_cfg=sdpa_cfg,
        cos_sin=(cos_tt, sin_tt),
    )
    ttnn.synchronize_device(mesh_device)
    ttnn.deallocate(cos_tt)
    ttnn.deallocate(sin_tt)

    # After all_gather every device holds the full [1,1,S,H] result in L1.
    # Check PCC on all 4 devices.
    device_tensors = ttnn.get_device_tensors(out_tt)
    assert len(device_tensors) == TP, f"expected {TP} device tensors, got {len(device_tensors)}"

    failures = []
    for dev_idx, dev_t in enumerate(device_tensors):
        out_dev = dev_t.cpu().to_torch().float().reshape(ref.shape)
        passed_dev, pcc_dev = comp_pcc(ref, out_dev, PCC_THRESHOLD)
        logger.info(
            f"[vision_attn_tp4_n_k_shard arch={mesh_device.arch().name} env={mesh_env} "
            f"TP={TP} S={SEQ_LEN} H={HIDDEN} heads={NUM_HEADS} heads/tp={HEADS_PER_TP} "
            f"grid=({GRID_H},{GRID_W}) "
            f"dev={dev_idx}] pcc={float(pcc_dev):.6f} (threshold {PCC_THRESHOLD})"
        )
        if not passed_dev:
            failures.append((dev_idx, float(pcc_dev)))

    ttnn.deallocate(out_tt)

    assert not failures, (
        "Vision Attn TP4 N/K-shard failed PCC on "
        + ", ".join(f"dev{i} pcc={p:.6f}" for i, p in failures)
        + f" (threshold {PCC_THRESHOLD})"
    )
