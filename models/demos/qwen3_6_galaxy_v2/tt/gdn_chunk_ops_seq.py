# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Chunked gated delta rule using the C++ `gated_delta_attn_seq` kernel (Path A).

Python preprocessing computes:
  - All cheap elementwise ops and two matmuls (kk, intra_attn).
  - L_inv: 4 diagonal block inverses of L_mat via _solve_lower_triangular_ttnn
    (D^{-1} Neumann + 2 Newton-Schulz steps — same path as parallel scan).
The C++ kernel performs blocked forward substitution + inter-chunk state scan.
"""

import os

import torch

import ttnn
from models.demos.qwen3_6_galaxy_v2.tt.gdn_chunk_ops import (
    _create_tril_ones,
    _create_triu_ones,
    _solve_lower_triangular_ttnn,
)

_TILE = 32


def _exact_block_inv_enabled():
    return os.environ.get("QWEN36_EXACT_BLOCK_INV") == "1"


def _exact_kk_enabled():
    return os.environ.get("QWEN36_EXACT_KK") == "1"


def _kk_fp32_enabled():
    # On-device fp32 matmul for kk — no CPU round-trip, trace-safe.
    return os.environ.get("QWEN36_KK_FP32") == "1"


def _exact_decay_enabled():
    return os.environ.get("QWEN36_EXACT_DECAY") == "1"


def _decay_fp32_enabled():
    # On-device fp32 cumsum for decay — no CPU round-trip, trace-safe.
    return os.environ.get("QWEN36_DECAY_FP32") == "1"


def _gdn_cpu_all_enabled():
    # All preprocessing ops A-G on CPU float32; only C++ kernel runs on device.
    return os.environ.get("QWEN36_GDN_CPU_ALL") == "1"


def _gdn_cpu_kernel_enabled():
    # Keep all TTNN preprocessing on device; replace only the C++ gated_delta_attn_seq
    # kernel with an exact CPU float32 inter-chunk scan.  Prefill-only diagnostic.
    return os.environ.get("QWEN36_GDN_CPU_KERNEL") == "1"


def _gdn_all_fp32_enabled():
    # Cast q, k, k_beta, v_beta to float32 before ALL device preprocessing ops so that
    # kk, L_unit, v_beta_sc, k_bd_sc, q_decay, k_decay, and intra_attn all run in fp32
    # instead of BF16.  Eliminates the dominant BF16 quantisation error in the GDN
    # preprocessing pipeline without any CPU round-trips (prefill + trace-safe).
    return os.environ.get("QWEN36_GDN_ALL_FP32") == "1"


# ---------------------------------------------------------------------------
# Sub-op PCC capture (QWEN36_GDN_SUBOP_PCC=1)
# ---------------------------------------------------------------------------
# After each device preprocessing op, gather to CPU and compare to CPU float32
# reference computed from the SAME gathered device inputs.  This answers: "does
# this specific op introduce error beyond what its inputs already have?"
# Only fires for the first chunk (ci=0) to avoid slowdown over 32 chunks.
# Call _gdn_subop_pcc_report() to see per-op PCC after a forward pass.
_GDN_SUBOP_LOG = []  # list of dicts, appended per layer call


def _gdn_subop_pcc_enabled():
    return os.environ.get("QWEN36_GDN_SUBOP_PCC") == "1"


def _gdn_subop_pcc_report():
    """Print a summary table of all captured sub-op PCC values."""
    if not _GDN_SUBOP_LOG:
        print("[subop_pcc] No captures recorded. Set QWEN36_GDN_SUBOP_PCC=1.")
        return
    ops = list(_GDN_SUBOP_LOG[0].keys())
    # Header
    print("\n[subop_pcc] Per-op PCC summary (device vs CPU, same BF16-precision device inputs)")
    print(f"{'Layer':>6}  " + "  ".join(f"{o:>14}" for o in ops))
    for li, entry in enumerate(_GDN_SUBOP_LOG):
        vals = "  ".join(f"{entry.get(o, float('nan')):>14.6f}" for o in ops)
        print(f"{li:>6}  {vals}")


def _pcc_scalar(dev_cpu: torch.Tensor, ref_cpu: torch.Tensor) -> float:
    d = dev_cpu.flatten().float()
    r = ref_cpu.flatten().float()
    if d.numel() != r.numel():
        r = r[: d.numel()]
    if d.std() < 1e-8 or r.std() < 1e-8:
        return float("nan")
    return torch.corrcoef(torch.stack([d, r]))[0, 1].item()


def _force_device_preprocess():
    # Sweep / device-only path: no CPU gather for decay or L_inv.
    return os.environ.get("QWEN36_FORCE_DEVICE_PREPROCESS") == "1"


def _kk_fp32_f32_enabled():
    # On-device fp32 matmul for kk; output stays float32 (no bf16 cast).
    return os.environ.get("QWEN36_KK_FP32_F32") == "1"


def _kk_hifi2_enabled():
    return os.environ.get("QWEN36_KK_HIFI2") == "1"


def _decay_hifi4_enabled():
    return os.environ.get("QWEN36_DECAY_HIFI4") == "1"


def _intra_fp32_f32_enabled():
    return os.environ.get("QWEN36_INTRA_FP32_F32") == "1"


def _intra_hifi2_enabled():
    return os.environ.get("QWEN36_INTRA_HIFI2") == "1"


def _gather_per_row(tt_tensor, batch, mesh_device):
    """Gather per-row-device data from a mesh tensor.

    For a MeshShape(R, C) mesh sharded with ShardTensor2dMesh(dims=(0, None)):
    - Row r, col 0 holds one independent copy of row r's data at index r*ncols in
      ttnn.get_device_tensors() (row-major order: (0,0),(0,1),...,(r,0),...).

    Returns CPU float32 tensor of shape [R * batch_per_row, ...] — all rows
    concatenated in order, col duplicates discarded.  batch arg unused internally
    but kept for call-site compatibility.
    """
    mshape = list(mesh_device.shape) if hasattr(mesh_device, "shape") else []
    is_mesh = len(mshape) > 1
    if not is_mesh:
        return ttnn.to_torch(tt_tensor).float()
    nrows, ncols = mshape[0], mshape[1]
    device_tensors = ttnn.get_device_tensors(tt_tensor)
    # Row r, col 0 is at flat index r * ncols in row-major order.
    row_tensors = [ttnn.to_torch(device_tensors[r * ncols]).float() for r in range(nrows)]
    return torch.cat(row_tensors, dim=0)  # [nrows * batch_per_row, ...]


def _upload_per_row(cpu_tensor, mesh_device, _cmc):
    """Upload a [nrows * batch, ...] CPU tensor to mesh.

    Uses ShardTensor2dMesh(dims=(0, None)): row r gets rows [r*batch : (r+1)*batch],
    replicated across all cols in that row.  Requires cpu_tensor.shape[0] % nrows == 0.
    """
    mshape = list(mesh_device.shape) if hasattr(mesh_device, "shape") else []
    is_mesh = len(mshape) > 1
    dmc = _cmc or ttnn.DRAM_MEMORY_CONFIG
    if is_mesh:
        return ttnn.from_torch(
            cpu_tensor,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(0, None), mesh_shape=mshape),
            memory_config=dmc,
        )
    else:
        return ttnn.from_torch(
            cpu_tensor,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=dmc,
        )


def _compute_decay_l_mask_exact_cpu(g_c, batch, C, mesh_device, _cmc=None):
    """CPU-exact decay and L_mask, bypassing the HiFi2 triu-matmul + exp chain.

    Avoids 2.81e-03 atol from HiFi2 cumsum.  Each row device's g values are
    gathered and processed independently on CPU, then re-distributed.

    g_c: [batch, C] float32 on mesh (each row device has its own BH*NC rows)
    Returns: decay_raw, decay, decay_exp, L_mask — all as device tensors sharded
             identically to the original g_c (ShardTensor2dMesh(dims=(0, None))).
    """
    mshape = list(mesh_device.shape) if hasattr(mesh_device, "shape") else []
    is_mesh = len(mshape) > 1
    nrows = mshape[0] if is_mesh else 1

    g_all_cpu = _gather_per_row(g_c, batch, mesh_device)  # [nrows * batch, C]
    total = nrows * batch

    triu = torch.triu(torch.ones(C, C, dtype=torch.float32))
    decay_raw_cpu = (g_all_cpu.unsqueeze(1) @ triu.unsqueeze(0)).squeeze(1)  # [total, C]
    decay_cpu = decay_raw_cpu - decay_raw_cpu[:, 0:1]
    decay_exp_cpu = torch.exp(decay_raw_cpu.clamp(-20, 0)).unsqueeze(-1)  # [total, C, 1]

    tril = torch.tril(torch.ones(C, C, dtype=torch.float32))
    L_diff = (decay_cpu.unsqueeze(-1) - decay_cpu.unsqueeze(-2)).clamp(-20, 0)
    L_mask_cpu = torch.exp(L_diff) * tril  # [total, C, C]

    decay_raw_tt = _upload_per_row(decay_raw_cpu, mesh_device, _cmc)
    decay_tt = _upload_per_row(decay_cpu, mesh_device, _cmc)
    decay_exp_tt = _upload_per_row(decay_exp_cpu, mesh_device, _cmc)
    L_mask_tt = _upload_per_row(L_mask_cpu, mesh_device, _cmc)
    return decay_raw_tt, decay_tt, decay_exp_tt, L_mask_tt


def _compute_decay_l_mask_device(g_c, batch, C, mesh_device, _cmc, triu_ones, tril_mask, hifi4_cfg, hifi2_cfg):
    """On-device decay and L_mask (no CPU round-trip)."""
    dmc = _cmc or ttnn.DRAM_MEMORY_CONFIG
    decay_cfg = hifi4_cfg if _decay_hifi4_enabled() else hifi2_cfg
    g_3d_tt = ttnn.reshape(g_c, [batch, 1, C], memory_config=dmc)
    decay_tt = ttnn.matmul(g_3d_tt, triu_ones, memory_config=dmc, compute_kernel_config=decay_cfg)
    decay_raw_tt = ttnn.reshape(decay_tt, [batch, C], memory_config=dmc)
    offset_tt = decay_raw_tt[:, 0:1]
    decay_norm_tt = ttnn.subtract(decay_raw_tt, offset_tt, memory_config=dmc)

    decay_col_tt = ttnn.reshape(decay_norm_tt, [batch, C, 1], memory_config=dmc)
    decay_row_tt = ttnn.reshape(decay_norm_tt, [batch, 1, C], memory_config=dmc)
    L_diff_tt = ttnn.subtract(decay_col_tt, decay_row_tt, memory_config=dmc)
    L_diff_masked_tt = ttnn.multiply(L_diff_tt, tril_mask, memory_config=dmc)
    L_diff_clamped_tt = ttnn.clip(L_diff_masked_tt, min=-20.0, max=0.0)
    decay_exp_tt = ttnn.reshape(
        ttnn.exp(ttnn.clip(decay_raw_tt, min=-20.0, max=0.0), memory_config=dmc),
        [batch, C, 1],
        memory_config=dmc,
    )
    L_mask_tt = ttnn.multiply(ttnn.exp(L_diff_clamped_tt, memory_config=dmc), tril_mask, memory_config=dmc)
    return decay_raw_tt, decay_norm_tt, decay_exp_tt, L_mask_tt


def _compute_kk_exact_cpu(k_beta_c, k_c_t, batch, C, K, mesh_device, _cmc=None):
    """CPU-exact kk = k_beta @ k^T, bypassing the HiFi2 float32 GEMM.

    k is col-replicated after the QKVZ all_reduce in qwen36_delta_attention.py.
    Each row device's data is gathered independently, kk is computed in float32
    on CPU, then re-distributed to the correct row devices.

    Avoids 6.84e-03 atol from HiFi2 K=128 float32 matmul accumulation error.
    """
    kb_all = _gather_per_row(k_beta_c, batch, mesh_device)  # [nrows * batch, C, K]
    kct_all = _gather_per_row(k_c_t, batch, mesh_device)  # [nrows * batch, K, C]
    kk_all = torch.bmm(kb_all, kct_all)  # [nrows * batch, C, C]
    return _upload_per_row(kk_all, mesh_device, _cmc)


def _compute_L_inv_exact_cpu(L_mat_4d, BH, NC, C, mesh_device, _cmc=None):
    """Exact diagonal block inversion via CPU torch.linalg.solve_triangular.

    Replaces the Neumann+Newton-Schulz TTNN path in _compute_L_inv_ttnn.
    Used via QWEN36_EXACT_BLOCK_INV=1 to test whether Neumann DRAM rounding
    contributes additional error beyond the 2D-TP all_reduce floor.

    L_mat_4d: [BH, NC, C, C] float32 on mesh (unit lower-triangular after D normalization)
    Returns:  [BH, NC, C, _TILE] float32 on mesh — C//_TILE block inverses stacked
    """
    mshape = list(mesh_device.shape) if hasattr(mesh_device, "shape") else []
    is_mesh = len(mshape) > 1
    dmc = _cmc or ttnn.DRAM_MEMORY_CONFIG

    if is_mesh:
        nrows, ncols = mshape[0], mshape[1]
        device_tensors = ttnn.get_device_tensors(L_mat_4d)
        # Row r, col 0 is at flat index r * ncols (row-major device ordering).
        # BH arg is the per-device (local) head count; gathering nrows devices gives
        # [nrows * BH_local, NC, C, C] total — use reshape(-1) not BH*NC.
        row_tensors = [ttnn.to_torch(device_tensors[r * ncols]).float() for r in range(nrows)]
        L_cpu = torch.cat(row_tensors, dim=0).reshape(-1, C, C)  # [nrows*BH_local*NC, C, C]
    else:
        L_cpu = ttnn.to_torch(L_mat_4d).reshape(-1, C, C).float()

    Ct = C // _TILE
    batch = L_cpu.shape[0]  # nrows * BH_local * NC (or BH * NC for single device)
    eye_t = torch.eye(_TILE, dtype=torch.float32).unsqueeze(0).expand(batch, -1, -1).contiguous()

    inv_blocks = []
    for b in range(Ct):
        r0 = b * _TILE
        block = L_cpu[:, r0 : r0 + _TILE, r0 : r0 + _TILE].contiguous()
        inv_blocks.append(torch.linalg.solve_triangular(block, eye_t, upper=False))

    # [batch, C, _TILE] → [nrows*BH_local, NC, C, _TILE] for mesh upload
    L_inv_cpu = torch.cat(inv_blocks, dim=1).reshape(-1, NC, C, _TILE)

    if is_mesh:
        return ttnn.from_torch(
            L_inv_cpu,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(0, None), mesh_shape=mshape),
            memory_config=dmc,
        )
    else:
        return ttnn.from_torch(
            L_inv_cpu,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=dmc,
        )


def _compute_L_inv_ttnn(L_mat_4d, BH, NC, C, mesh_device, _cmc=None, eye_32=None):
    """Compute diagonal block inverses of L_mat using _solve_lower_triangular_ttnn.

    L_mat_4d: [BH, NC, C, C] float32 lower-triangular, positive diagonal (~2)
    eye_32:   [1, _TILE, _TILE] float32 identity pre-allocated on device (required for trace compat)
    Returns:  [BH, NC, C, _TILE] float32 — (C//_TILE) diagonal block inverses stacked as [C, _TILE]

    Each _TILE x _TILE diagonal block B is inverted via the same D^{-1}-Neumann-NS path
    used by _solve_lower_triangular_ttnn in the parallel scan.
    C must be a multiple of _TILE (={_TILE}).
    """
    assert C % _TILE == 0, f"chunk_size ({C}) must be a multiple of _TILE ({_TILE})"

    if _exact_block_inv_enabled():
        return _compute_L_inv_exact_cpu(L_mat_4d, BH, NC, C, mesh_device, _cmc)

    if eye_32 is None:
        # Fallback for tests that don't pass cached_masks — not trace-compatible.
        eye_32 = ttnn.from_torch(
            torch.eye(_TILE, dtype=torch.float32).unsqueeze(0),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    Ct = C // _TILE
    batch = BH * NC
    L_flat = ttnn.reshape(L_mat_4d, [batch, C, C], memory_config=_cmc)

    inv_blocks = []
    for b in range(Ct):
        row_start = b * _TILE
        col_start = b * _TILE
        block = ttnn.slice(
            L_flat, [0, row_start, col_start], [batch, row_start + _TILE, col_start + _TILE], memory_config=_cmc
        )
        # _solve_lower_triangular_ttnn: D^{-1} normalization + Neumann + 2 NS steps
        block_inv = _solve_lower_triangular_ttnn(block, eye_32, mesh_device)
        ttnn.deallocate(block)
        inv_blocks.append(block_inv)  # [batch, _TILE, _TILE]

    # Do NOT deallocate L_flat: it is a reshape (view) of L_mat_4d which is the
    # same L_unit_4d tensor passed as a kernel input.  Freeing L_flat frees L_unit_4d's
    # buffer while the C++ kernel still needs to read from it.

    L_inv_flat = ttnn.concat(inv_blocks, dim=1, memory_config=_cmc)
    for blk in inv_blocks:
        ttnn.deallocate(blk)

    # Do NOT deallocate L_inv_flat here — ttnn.reshape returns a view that shares
    # the same DRAM buffer. Freeing L_inv_flat while L_inv_4d (the view) is still
    # in use causes the kernel to read from freed memory on later runs when the
    # allocator reuses that address. The caller's ttnn.deallocate(L_inv_4d) will
    # release the buffer when it is no longer needed.
    L_inv_4d = ttnn.reshape(L_inv_flat, [BH, NC, C, _TILE], memory_config=_cmc)
    return L_inv_4d


def _cpu_all_preprocess(
    q_c_tt,
    k_c_tt,
    k_beta_c_tt,
    v_beta_c_tt,
    g_c_tt,
    BH,
    num_chunks,
    chunk_size,
    K,
    V,
    mesh_device,
    _cmc,
):
    """Compute all GDN preprocessing ops A-G on CPU float32.

    Gathers chunked inputs from the mesh, runs ops A-G in torch.float32, and
    uploads the 8 C++ kernel inputs back to device.  Used by QWEN36_GDN_CPU_ALL=1.

    Ops:
      A — decay chain (cumsum via matmul + exp + L_mask)
      B — kk = k_beta @ k^T
      C — L_mat = I + kk * L_mask
      D — D normalisation → L_unit, v_beta_sc, k_bd_sc
      E — q_decay, k_decay, dl_exp preprocessing
      F — intra_attn = (q @ k^T) * L_mask * lower_causal
      G — diagonal block inversion via solve_triangular
    """
    C = chunk_size
    NC = num_chunks
    T = _TILE

    # Gather all chunked inputs from mesh: shape [nrows*BH*NC, C, ...]
    q_c = _gather_per_row(q_c_tt, BH * NC, mesh_device)  # [nrows*BH*NC, C, K]
    k_c = _gather_per_row(k_c_tt, BH * NC, mesh_device)
    kb_c = _gather_per_row(k_beta_c_tt, BH * NC, mesh_device)
    vb_c = _gather_per_row(v_beta_c_tt, BH * NC, mesh_device)
    g_c = _gather_per_row(g_c_tt, BH * NC, mesh_device)  # [nrows*BH*NC, C]

    batch = g_c.shape[0]  # nrows * BH * NC
    total_BH = batch // NC  # nrows * BH

    triu = torch.triu(torch.ones(C, C, dtype=torch.float32))
    tril = torch.tril(torch.ones(C, C, dtype=torch.float32))
    eye_cc = torch.eye(C, dtype=torch.float32).unsqueeze(0)  # [1, C, C]

    # A: decay chain
    decay_raw = (g_c.unsqueeze(1) @ triu.unsqueeze(0)).squeeze(1)  # [batch, C]
    decay = decay_raw - decay_raw[:, 0:1]
    decay_exp = decay_raw.clamp(-20, 0).exp().unsqueeze(-1)  # [batch, C, 1]
    L_diff = (decay.unsqueeze(-1) - decay.unsqueeze(-2)).clamp(-20, 0)
    L_mask = L_diff.exp() * tril  # [batch, C, C]

    # B: kk
    kk = torch.bmm(kb_c, k_c.transpose(1, 2))  # [batch, C, C]

    # C: L_mat
    L_mat = eye_cc + kk * L_mask  # [batch, C, C]

    # D: D normalisation
    D_diag = (L_mat * eye_cc).sum(-1)  # [batch, C]
    D_inv_r = (1.0 / D_diag).unsqueeze(-1)  # [batch, C, 1]
    L_unit = eye_cc + D_inv_r * (L_mat - L_mat * eye_cc)  # [batch, C, C]
    v_beta_sc = D_inv_r * vb_c  # [batch, C, V]
    k_bd_sc = D_inv_r * (kb_c * decay_exp)  # [batch, C, K]

    # E: q_decay, k_decay, dl_exp
    decay_3d = decay.reshape(total_BH, NC, C)
    decay_raw_3d = decay_raw.reshape(total_BH, NC, C)
    decay_last_raw = g_c.sum(-1).reshape(total_BH, NC, 1)  # [total_BH, NC, 1]
    decay_last_norm = decay_3d[:, :, -1:]  # [total_BH, NC, 1]

    q_c_4d = q_c.reshape(total_BH, NC, C, K)
    q_decay_4d = q_c_4d * decay_raw_3d.clamp(-20, 0).exp().unsqueeze(-1)  # [total_BH, NC, C, K]

    k_c_4d = k_c.reshape(total_BH, NC, C, K)
    decay_diff = (decay_last_norm - decay_3d).clamp(-20, 0)  # [total_BH, NC, C]
    k_decay_t_4d = (k_c_4d * decay_diff.exp().unsqueeze(-1)).transpose(2, 3)  # [total_BH, NC, K, C]

    dl_exp_4d = decay_last_raw.clamp(-20, 0).exp().reshape(total_BH, NC, 1, 1)

    # F: intra_attn
    lower_causal = torch.tril(torch.ones(C, C, dtype=torch.float32))
    combined_mask = L_mask.reshape(total_BH, NC, C, C) * lower_causal
    q_flat = q_c_4d.reshape(batch, C, K)
    k_flat = k_c_4d.reshape(batch, C, K)
    intra_attn_4d = torch.bmm(q_flat, k_flat.transpose(1, 2)).reshape(total_BH, NC, C, C) * combined_mask

    # Reshape L_unit, v_beta_sc, k_bd_sc to 4D
    L_unit_4d = L_unit.reshape(total_BH, NC, C, C)
    v_beta_sc_4d = v_beta_sc.reshape(total_BH, NC, C, V)
    k_bd_sc_4d = k_bd_sc.reshape(total_BH, NC, C, K)

    # G: diagonal block inversion
    Ct = C // T
    L_flat = L_unit.reshape(batch, C, C)
    eye_t = torch.eye(T, dtype=torch.float32).unsqueeze(0).expand(batch, -1, -1).contiguous()
    inv_blocks = []
    for b in range(Ct):
        r0 = b * T
        blk = L_flat[:, r0 : r0 + T, r0 : r0 + T].contiguous()
        inv_blocks.append(torch.linalg.solve_triangular(blk, eye_t, upper=False))
    L_inv_4d = torch.cat(inv_blocks, dim=1).reshape(total_BH, NC, C, T)

    def _up(t):
        return _upload_per_row(t, mesh_device, _cmc)

    return (
        _up(L_unit_4d),
        _up(v_beta_sc_4d),
        _up(k_bd_sc_4d),
        _up(intra_attn_4d),
        _up(q_decay_4d),
        _up(k_decay_t_4d),
        _up(dl_exp_4d),
        _up(L_inv_4d),
    )


def _cpu_scan_kernel(
    L_unit_4d,
    v_beta_sc_4d,
    k_bd_sc_4d,
    intra_attn_4d,
    q_decay_4d,
    k_decay_t_4d,
    dl_exp_4d,
    L_inv_4d,
    BH,
    NC,
    C,
    K,
    V,
    mesh_device,
    S0_tt=None,
):
    """CPU float32 replacement for gated_delta_attn_seq.

    Gathers TTNN-preprocessed tensors to CPU, runs the exact 7-step
    inter-chunk scan in float32, and uploads the result back to device.
    All preprocessing errors (kk, decay, L_inv) are preserved — only the
    kernel's internal matmul errors are eliminated.  Diagnostic only.
    """
    # Use _gather_per_row: handles mesh tensor gather (row-head-sharded, col-replicated)
    # Returns [nrows * BH_local, NC, ...] on CPU float32.
    BH_total = _gather_per_row(L_unit_4d, BH, mesh_device).shape[0]

    # Gather preprocessed tensors — shape [BH_total, NC, ...]
    L_unit = _gather_per_row(L_unit_4d, BH, mesh_device)  # [BH_total, NC, C, C]
    v_beta = _gather_per_row(v_beta_sc_4d, BH, mesh_device)  # [BH_total, NC, C, V]
    k_bd = _gather_per_row(k_bd_sc_4d, BH, mesh_device)  # [BH_total, NC, C, K]
    intra = _gather_per_row(intra_attn_4d, BH, mesh_device)  # [BH_total, NC, C, C]
    q_dec = _gather_per_row(q_decay_4d, BH, mesh_device)  # [BH_total, NC, C, K]
    k_dec_t = _gather_per_row(k_decay_t_4d, BH, mesh_device)  # [BH_total, NC, K, C]
    dl_exp = _gather_per_row(dl_exp_4d, BH, mesh_device)  # [BH_total, NC, 1, 1]
    L_inv = _gather_per_row(L_inv_4d, BH, mesh_device)  # [BH_total, NC, C, C]

    S = torch.zeros(BH_total, K, V, dtype=torch.float32)
    if S0_tt is not None:
        S = _gather_per_row(S0_tt, BH, mesh_device).reshape(BH_total, K, V)

    out_chunks = []
    for c in range(NC):
        Lu = L_unit[:, c]  # [BH, C, C]
        vb = v_beta[:, c]  # [BH, C, V]
        kb = k_bd[:, c]  # [BH, C, K]
        ia = intra[:, c]  # [BH, C, C]
        qd = q_dec[:, c]  # [BH, C, K]
        kdt = k_dec_t[:, c]  # [BH, K, C]
        dl = dl_exp[:, c]  # [BH, 1, 1]
        Li = L_inv[:, c]  # [BH, C, C]

        # Forward substitution using provided L_inv (matches kernel)
        v_cor = torch.linalg.solve_triangular(Lu, vb, upper=False)  # [BH, C, V]
        k_cum = torch.linalg.solve_triangular(Lu, kb, upper=False)  # [BH, C, K]

        # 7-step state update (mirrors gated_delta_attn.cpp)
        v_prime = torch.bmm(k_cum, S)  # [BH, C, V]
        v_new = v_cor - v_prime  # [BH, C, V]
        o_inter = torch.bmm(qd, S)  # [BH, C, V]
        intra_v = torch.bmm(ia, v_new)  # [BH, C, V]
        out_c = o_inter + intra_v  # [BH, C, V]
        s_upd = torch.bmm(kdt, v_new)  # [BH, K, V]
        S = S * dl + s_upd  # [BH, K, V]

        out_chunks.append(out_c)

    out_cpu = torch.stack(out_chunks, dim=1)  # [BH_total, NC, C, V]

    # Upload output back to device — row-head-sharded, col-replicated (matches kernel output layout)
    out_4d = _upload_per_row(out_cpu, mesh_device, ttnn.DRAM_MEMORY_CONFIG)

    # Upload final state: [BH_total, 1, K, V]
    final_state = _upload_per_row(S.unsqueeze(1), mesh_device, ttnn.DRAM_MEMORY_CONFIG)

    return out_4d, final_state


def chunk_gated_delta_rule_seq(
    q,  # [BH, T, K] float32 on mesh
    k,  # [BH, T, K] float32 on mesh
    v,  # [BH, T, V] float32 on mesh
    beta,  # [BH, T, 1] float32 on mesh
    g,  # [BH, T]    float32 on mesh
    chunk_size=128,
    scale=None,
    initial_state=None,  # [BH, K, V] float32 or None
    mesh_device=None,
    cached_masks=None,
):
    """Chunked gated delta rule using C++ sequential scan kernel (Path A).

    Python preprocessing: ~9ms (vs ~40ms in the previous Path A-lite version).
    C++ kernel: triangular solve + inter-chunk state scan (~1.5ms).
    Total per GDN layer: ~28ms (vs 57ms before).
    """
    _hifi_cfg = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )
    _hifi4_cfg = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )

    assert chunk_size % _TILE == 0, f"chunk_size ({chunk_size}) must be a multiple of _TILE ({_TILE})"

    BH = q.shape[0]
    T = q.shape[1]
    K = q.shape[2]
    V = v.shape[2]

    if scale is None:
        scale = K**-0.5

    q = ttnn.multiply(q, scale, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    pad_len = (chunk_size - (T % chunk_size)) % chunk_size
    L = T + pad_len
    num_chunks = L // chunk_size
    batch = BH * num_chunks

    beta_flat = beta
    if pad_len > 0:
        zeros_q = ttnn.zeros(
            [BH, pad_len, K],
            device=mesh_device,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        zeros_v = ttnn.zeros(
            [BH, pad_len, V],
            device=mesh_device,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        zeros_beta = ttnn.zeros(
            [BH, pad_len, 1],
            device=mesh_device,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        q = ttnn.concat([q, zeros_q], dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        k = ttnn.concat([k, zeros_q], dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        v = ttnn.concat([v, zeros_v], dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        beta_flat = ttnn.concat([beta_flat, zeros_beta], dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        g_3d = ttnn.reshape(g, [BH, T, 1])
        ttnn.deallocate(g)
        zeros_g = ttnn.zeros(
            [BH, pad_len, 1],
            device=mesh_device,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        g_3d = ttnn.concat([g_3d, zeros_g], dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        g = ttnn.reshape(g_3d, [BH, L])
        beta_flat = ttnn.reshape(beta_flat, [BH, L, 1])
    else:
        beta_flat = ttnn.reshape(beta_flat, [BH, L, 1])

    v_beta = ttnn.multiply(v, beta_flat, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    k_beta = ttnn.multiply(k, beta_flat, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    del beta_flat

    q_c = ttnn.reshape(q, [batch, chunk_size, K], memory_config=ttnn.DRAM_MEMORY_CONFIG)
    k_c = ttnn.reshape(k, [batch, chunk_size, K], memory_config=ttnn.DRAM_MEMORY_CONFIG)
    k_beta_c = ttnn.reshape(k_beta, [batch, chunk_size, K], memory_config=ttnn.DRAM_MEMORY_CONFIG)
    v_beta_c = ttnn.reshape(v_beta, [batch, chunk_size, V], memory_config=ttnn.DRAM_MEMORY_CONFIG)
    g_c = ttnn.reshape(g, [batch, chunk_size], memory_config=ttnn.DRAM_MEMORY_CONFIG)
    del q, v, k_beta, v_beta

    _cmc = ttnn.DRAM_MEMORY_CONFIG

    if _gdn_cpu_all_enabled():
        # ----------------------------------------------------------------
        # CPU-all mode: ops A-G on CPU float32 → upload 8 kernel inputs.
        # ----------------------------------------------------------------
        (
            L_unit_4d,
            v_beta_sc_4d,
            k_bd_sc_4d,
            intra_attn_4d,
            q_decay_4d,
            k_decay_t_4d,
            dl_exp_4d,
            L_inv_4d,
        ) = _cpu_all_preprocess(
            q_c,
            k_c,
            k_beta_c,
            v_beta_c,
            g_c,
            BH,
            num_chunks,
            chunk_size,
            K,
            V,
            mesh_device,
            _cmc,
        )
    else:
        _eye_32 = None
        if cached_masks is not None:
            triu_ones = cached_masks["triu_ones"]
            tril_mask = cached_masks["tril_mask"]
            _eye_1cc = cached_masks["eye"]
            lower_causal = cached_masks["lower_causal"]
            _eye_32 = cached_masks.get("eye_32")
        else:
            triu_ones = _create_triu_ones(chunk_size, mesh_device, dtype=ttnn.float32)
            triu_ones = ttnn.reshape(triu_ones, [1, chunk_size, chunk_size])
            tril_mask = _create_tril_ones(chunk_size, mesh_device, dtype=ttnn.float32)
            tril_mask = ttnn.reshape(tril_mask, [1, chunk_size, chunk_size])
            _eye_1cc = ttnn.from_torch(
                torch.eye(chunk_size, dtype=torch.float32).unsqueeze(0),
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            lower_causal = _create_tril_ones(chunk_size, mesh_device, dtype=ttnn.float32)

        # ----------------------------------------------------------------
        # Decay preprocessing — CPU-exact by default; device path when
        # QWEN36_FORCE_DEVICE_PREPROCESS=1 (sweep / no-CPU-forward builds).
        # ----------------------------------------------------------------
        if _force_device_preprocess():
            decay_raw, decay, decay_exp, L_mask = _compute_decay_l_mask_device(
                g_c, batch, chunk_size, mesh_device, _cmc, triu_ones, tril_mask, _hifi4_cfg, _hifi_cfg
            )
        else:
            decay_raw, decay, decay_exp, L_mask = _compute_decay_l_mask_exact_cpu(
                g_c, batch, chunk_size, mesh_device, _cmc
            )

        # ----------------------------------------------------------------
        # Sub-op PCC: gather inputs once, compute ALL CPU references up-front.
        # Inline captures below compare each device output to its CPU ref.
        # Only fires for the default (device-preprocessing) path.
        # ----------------------------------------------------------------
        _subop_pcc = _gdn_subop_pcc_enabled()
        _sop = {}  # per-op PCC for this layer call
        _sop_ref = {}  # CPU reference tensors, keyed by op name
        if _subop_pcc:
            _C = chunk_size
            _NC = num_chunks
            _eye_c = torch.eye(_C, dtype=torch.float32).unsqueeze(0)  # [1,C,C]
            _tril_c = torch.tril(torch.ones(_C, _C, dtype=torch.float32))

            # Gather device inputs to CPU (fp32 format, BF16-precision mantissa)
            _sp_q = _gather_per_row(q_c, batch, mesh_device)  # [nrows*BH*NC, C, K]
            _sp_k = _gather_per_row(k_c, batch, mesh_device)
            _sp_kb = _gather_per_row(k_beta_c, batch, mesh_device)
            _sp_vb = _gather_per_row(v_beta_c, batch, mesh_device)
            _sp_dr = _gather_per_row(decay_raw, batch, mesh_device)  # CPU-exact decay
            _sp_de = _gather_per_row(decay_exp, batch, mesh_device)  # CPU-exact decay_exp
            _sp_lm = _gather_per_row(L_mask, batch, mesh_device)  # CPU-exact L_mask

            # kk ref
            _kk_r = torch.bmm(_sp_kb, _sp_k.transpose(1, 2))
            _sop_ref["kk"] = _kk_r

            # L_unit, v_beta_sc, k_bd_sc refs
            _lm_r = _eye_c + _kk_r * _sp_lm
            _Dd_r = (_lm_r * _eye_c).sum(-1, keepdim=True)
            _Di_r = 1.0 / _Dd_r
            _sop_ref["L_unit"] = _eye_c + _Di_r * (_lm_r - _lm_r * _eye_c)
            _sop_ref["v_beta_sc"] = _Di_r * _sp_vb
            _sop_ref["k_bd_sc"] = _Di_r * (_sp_kb * _sp_de)

            # q_decay, k_decay_t, intra_attn refs
            _tbh = _sp_q.shape[0] // _NC
            _q4r = _sp_q.reshape(_tbh, _NC, _C, K)
            _k4r = _sp_k.reshape(_tbh, _NC, _C, K)
            _dr3 = _sp_dr.reshape(_tbh, _NC, _C)
            _d3 = _dr3 - _dr3[:, :, :1]
            _dlast = _d3[:, :, -1:]
            _sop_ref["q_decay"] = _q4r * _dr3.clamp(-20, 0).exp().unsqueeze(-1)
            _ddiff = (_dlast - _d3).clamp(-20, 0)
            _sop_ref["k_decay_t"] = (_k4r * _ddiff.exp().unsqueeze(-1)).transpose(2, 3)
            _cmask = _sp_lm.reshape(_tbh, _NC, _C, _C) * _tril_c
            _sop_ref["intra_attn"] = (
                torch.bmm(_q4r.reshape(-1, _C, K), _k4r.reshape(-1, _C, K).transpose(1, 2)).reshape(_tbh, _NC, _C, _C)
                * _cmask
            )

        # ----------------------------------------------------------------
        # (Optional) Full-precision preprocessing: cast q/k/k_beta/v_beta to
        # fp32 so that ALL subsequent device ops (kk, L_unit, v_beta_sc,
        # k_bd_sc, q_decay, k_decay, intra_attn) run in float32 rather than
        # BF16.  Eliminates BF16 quantisation as the dominant PCC bottleneck.
        # ----------------------------------------------------------------
        if _gdn_all_fp32_enabled():

            def _cast_fp32(t):
                return ttnn.typecast(t, ttnn.float32, memory_config=_cmc) if t.dtype != ttnn.float32 else t

            q_c = _cast_fp32(q_c)
            k_c = _cast_fp32(k_c)
            k_beta_c = _cast_fp32(k_beta_c)
            v_beta_c = _cast_fp32(v_beta_c)

        # ----------------------------------------------------------------
        # kk = k_beta @ k.T  [batch, C, C]  (1 matmul dispatch)
        # ----------------------------------------------------------------
        del k
        k_c = ttnn.move(k_c)
        k_c_t = ttnn.transpose(k_c, 1, 2, memory_config=_cmc)
        _kk_cfg = _hifi_cfg if _kk_hifi2_enabled() else _hifi4_cfg
        if _kk_fp32_f32_enabled():
            kb_fp32 = ttnn.typecast(k_beta_c, ttnn.float32) if k_beta_c.dtype != ttnn.float32 else k_beta_c
            kct_fp32 = ttnn.typecast(k_c_t, ttnn.float32) if k_c_t.dtype != ttnn.float32 else k_c_t
            kk = ttnn.matmul(kb_fp32, kct_fp32, memory_config=_cmc, compute_kernel_config=_kk_cfg)
            if kb_fp32 is not k_beta_c:
                ttnn.deallocate(kb_fp32)
            if kct_fp32 is not k_c_t:
                ttnn.deallocate(kct_fp32)
        elif _kk_fp32_enabled():
            # Legacy: fp32 matmul then cast to bf16 (hurts atol — prefer KK_FP32_F32).
            kb_fp32 = ttnn.typecast(k_beta_c, ttnn.float32)
            kct_fp32 = ttnn.typecast(k_c_t, ttnn.float32)
            kk_fp32 = ttnn.matmul(kb_fp32, kct_fp32, memory_config=_cmc, compute_kernel_config=_kk_cfg)
            ttnn.deallocate(kb_fp32)
            ttnn.deallocate(kct_fp32)
            kk = ttnn.typecast(kk_fp32, ttnn.bfloat16)
            ttnn.deallocate(kk_fp32)
        elif _exact_kk_enabled():
            kk = _compute_kk_exact_cpu(k_beta_c, k_c_t, batch, chunk_size, K, mesh_device, _cmc)
        else:
            kk = ttnn.matmul(k_beta_c, k_c_t, memory_config=_cmc, compute_kernel_config=_kk_cfg)
        ttnn.deallocate(k_c_t)
        if _subop_pcc:
            _sop["kk"] = _pcc_scalar(_gather_per_row(kk, batch, mesh_device), _sop_ref["kk"])

        # ----------------------------------------------------------------
        # Build L_mat = I + kk * L_mask  (2 cheap elementwise dispatches)
        # ----------------------------------------------------------------
        L_mat = ttnn.add(
            _eye_1cc,
            ttnn.multiply(kk, L_mask, memory_config=_cmc),
            memory_config=_cmc,
        )
        ttnn.deallocate(kk)

        # ----------------------------------------------------------------
        # Normalize L_mat to unit-diagonal form: L_unit = D^{-1} L_mat
        # Keeps off-diagonal correction values smaller → better float32 precision
        # in blocked forward substitution.
        # ----------------------------------------------------------------
        D_mat = ttnn.multiply(L_mat, _eye_1cc, memory_config=_cmc)
        D_diag = ttnn.sum(D_mat, dim=-1, memory_config=_cmc)
        D_inv = ttnn.reciprocal(D_diag, memory_config=_cmc)
        ttnn.deallocate(D_diag)
        D_inv_row = ttnn.reshape(D_inv, [batch, chunk_size, 1], memory_config=_cmc)

        L_strict = ttnn.subtract(L_mat, D_mat, memory_config=_cmc)
        ttnn.deallocate(D_mat)
        ttnn.deallocate(L_mat)
        N = ttnn.multiply(D_inv_row, L_strict, memory_config=_cmc)
        ttnn.deallocate(L_strict)
        L_unit = ttnn.add(_eye_1cc, N, memory_config=_cmc)
        ttnn.deallocate(N)
        if _subop_pcc:
            _sop["L_unit"] = _pcc_scalar(_gather_per_row(L_unit, batch, mesh_device), _sop_ref["L_unit"])

        v_beta_sc = ttnn.multiply(D_inv_row, v_beta_c, memory_config=_cmc)
        del v_beta_c
        k_beta_decay = ttnn.multiply(k_beta_c, decay_exp, memory_config=_cmc)
        k_bd_sc = ttnn.multiply(D_inv_row, k_beta_decay, memory_config=_cmc)
        ttnn.deallocate(k_beta_decay)
        ttnn.deallocate(D_inv_row)
        if _subop_pcc:
            _sop["v_beta_sc"] = _pcc_scalar(_gather_per_row(v_beta_sc, batch, mesh_device), _sop_ref["v_beta_sc"])
            _sop["k_bd_sc"] = _pcc_scalar(_gather_per_row(k_bd_sc, batch, mesh_device), _sop_ref["k_bd_sc"])

        # ----------------------------------------------------------------
        # Precompute intra_attn: q_decay @ k.T * L_mask * lower_causal
        # (1 matmul dispatch — still cheaper than full solve)
        # ----------------------------------------------------------------
        decay_3d = ttnn.reshape(decay, [BH, num_chunks, chunk_size], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        decay_raw_3d = ttnn.reshape(decay_raw, [BH, num_chunks, chunk_size], memory_config=ttnn.DRAM_MEMORY_CONFIG)

        decay_last_raw = ttnn.reshape(
            ttnn.sum(g_c, dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG),
            [BH, num_chunks, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        decay_last_normalized = ttnn.reshape(
            decay_3d[:, :, -1:], [BH, num_chunks, 1], memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

        decay_raw_exp_4d = ttnn.reshape(
            ttnn.exp(ttnn.clip(decay_raw_3d, min=-20.0, max=0.0), memory_config=_cmc),
            [BH, num_chunks, chunk_size, 1],
            memory_config=_cmc,
        )
        q_c_4d = ttnn.to_layout(
            ttnn.reshape(q_c, [BH, num_chunks, chunk_size, K], memory_config=ttnn.DRAM_MEMORY_CONFIG),
            ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        q_decay_4d = ttnn.multiply(q_c_4d, decay_raw_exp_4d, memory_config=_cmc)
        ttnn.deallocate(decay_raw_exp_4d)
        if _subop_pcc:
            _sop["q_decay"] = _pcc_scalar(
                _gather_per_row(q_decay_4d, BH * num_chunks, mesh_device), _sop_ref["q_decay"]
            )

        decay_last_norm_4d = ttnn.reshape(decay_last_normalized, [BH, num_chunks, 1], memory_config=_cmc)
        decay_diff_3d = ttnn.subtract(decay_last_norm_4d, decay_3d, memory_config=_cmc)
        decay_diff_exp_4d = ttnn.reshape(
            ttnn.exp(ttnn.clip(decay_diff_3d, min=-20.0, max=0.0), memory_config=_cmc),
            [BH, num_chunks, chunk_size, 1],
            memory_config=_cmc,
        )
        k_c_4d = ttnn.to_layout(
            ttnn.reshape(k_c, [BH, num_chunks, chunk_size, K], memory_config=ttnn.DRAM_MEMORY_CONFIG),
            ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        k_decay_4d = ttnn.multiply(k_c_4d, decay_diff_exp_4d, memory_config=_cmc)
        ttnn.deallocate(decay_diff_exp_4d)
        k_decay_t_4d = ttnn.transpose(k_decay_4d, 2, 3, memory_config=_cmc)
        ttnn.deallocate(k_decay_4d)
        if _subop_pcc:
            _sop["k_decay_t"] = _pcc_scalar(
                _gather_per_row(k_decay_t_4d, BH * num_chunks, mesh_device), _sop_ref["k_decay_t"]
            )

        dl_exp_3d = ttnn.exp(ttnn.clip(decay_last_raw, min=-20.0, max=0.0), memory_config=_cmc)
        dl_exp_4d = ttnn.reshape(
            ttnn.to_layout(
                ttnn.typecast(dl_exp_3d, ttnn.float32, memory_config=_cmc)
                if dl_exp_3d.dtype != ttnn.float32
                else dl_exp_3d,
                ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            ),
            [BH, num_chunks, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        L_mask_4d = ttnn.reshape(
            L_mask, [BH, num_chunks, chunk_size, chunk_size], memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        L_mask_4d = ttnn.to_layout(L_mask_4d, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        lower_causal_4d = ttnn.reshape(
            lower_causal, [1, 1, chunk_size, chunk_size], memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        combined_mask_4d = ttnn.multiply(L_mask_4d, lower_causal_4d, memory_config=_cmc)
        ttnn.deallocate(L_mask_4d)
        k_c_4d_t = ttnn.transpose(k_c_4d, 2, 3, memory_config=_cmc)
        _intra_cfg = _hifi_cfg if _intra_hifi2_enabled() else _hifi4_cfg
        if _intra_fp32_f32_enabled():
            q_intra = ttnn.typecast(q_c_4d, ttnn.float32) if q_c_4d.dtype != ttnn.float32 else q_c_4d
            k_intra_t = ttnn.typecast(k_c_4d_t, ttnn.float32) if k_c_4d_t.dtype != ttnn.float32 else k_c_4d_t
            qk_4d = ttnn.matmul(q_intra, k_intra_t, memory_config=_cmc, compute_kernel_config=_intra_cfg)
            if q_intra is not q_c_4d:
                ttnn.deallocate(q_intra)
            if k_intra_t is not k_c_4d_t:
                ttnn.deallocate(k_intra_t)
        else:
            qk_4d = ttnn.matmul(q_c_4d, k_c_4d_t, memory_config=_cmc, compute_kernel_config=_intra_cfg)
        ttnn.deallocate(k_c_4d_t)
        intra_attn_4d = ttnn.multiply(qk_4d, combined_mask_4d, memory_config=_cmc)
        ttnn.deallocate(qk_4d)
        ttnn.deallocate(combined_mask_4d)
        if _subop_pcc:
            _sop["intra_attn"] = _pcc_scalar(
                _gather_per_row(intra_attn_4d, BH * num_chunks, mesh_device), _sop_ref["intra_attn"]
            )

        # ----------------------------------------------------------------
        # Reshape preprocessing outputs to 4D for C++ kernel
        # ----------------------------------------------------------------
        def _to4d_f32(t, d1, d2):
            t4 = ttnn.reshape(t, [BH, num_chunks, d1, d2], memory_config=ttnn.DRAM_MEMORY_CONFIG)
            return ttnn.to_layout(t4, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        L_unit_4d = _to4d_f32(L_unit, chunk_size, chunk_size)
        # Do NOT deallocate L_unit/v_beta_sc/k_bd_sc here: _to4d_f32 returns a reshape
        # (view) followed by to_layout (no-op when already TILE+DRAM), so the 4D tensor
        # aliases the original buffer.  Calling ttnn.deallocate on the original while the
        # view is still in use as a kernel input causes use-after-free on the 3rd+ call.
        v_beta_sc_4d = _to4d_f32(v_beta_sc, chunk_size, V)
        k_bd_sc_4d = _to4d_f32(k_bd_sc, chunk_size, K)

        def _ensure_f32_dram(t):
            if t.dtype != ttnn.float32:
                t = ttnn.typecast(t, ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            elif t.memory_config() != ttnn.DRAM_MEMORY_CONFIG:
                t = ttnn.to_memory_config(t, ttnn.DRAM_MEMORY_CONFIG)
            return t

        L_unit_4d = _ensure_f32_dram(L_unit_4d)
        v_beta_sc_4d = _ensure_f32_dram(v_beta_sc_4d)
        k_bd_sc_4d = _ensure_f32_dram(k_bd_sc_4d)
        intra_attn_4d = _ensure_f32_dram(intra_attn_4d)
        q_decay_4d = _ensure_f32_dram(q_decay_4d)
        k_decay_t_4d = _ensure_f32_dram(k_decay_t_4d)

        # L_inv: CPU-exact by default; Neumann+NS on device when FORCE_DEVICE_PREPROCESS.
        if _force_device_preprocess():
            L_inv_4d = _compute_L_inv_ttnn(L_unit_4d, BH, num_chunks, chunk_size, mesh_device, _cmc, eye_32=_eye_32)
        else:
            L_inv_4d = _compute_L_inv_exact_cpu(L_unit_4d, BH, num_chunks, chunk_size, mesh_device, _cmc)

    # Initial state
    S0_tt = None
    if initial_state is not None:
        S0_tt = ttnn.typecast(
            ttnn.reshape(initial_state, [BH, K, V], memory_config=ttnn.DRAM_MEMORY_CONFIG),
            ttnn.float32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    # ----------------------------------------------------------------
    # C++ sequential scan kernel (Path A) — or CPU fallback for isolation
    # ----------------------------------------------------------------
    if _gdn_cpu_kernel_enabled():
        out_4d, final_state = _cpu_scan_kernel(
            L_unit_4d,
            v_beta_sc_4d,
            k_bd_sc_4d,
            intra_attn_4d,
            q_decay_4d,
            k_decay_t_4d,
            dl_exp_4d,
            L_inv_4d,
            BH,
            num_chunks,
            chunk_size,
            K,
            V,
            mesh_device,
            S0_tt,
        )
        ttnn.deallocate(L_inv_4d)
    else:
        out_4d, final_state = ttnn.transformer.gated_delta_attn_seq(
            L_unit_4d,
            v_beta_sc_4d,
            k_bd_sc_4d,
            intra_attn_4d,
            q_decay_4d,
            k_decay_t_4d,
            dl_exp_4d,
            L_inv_4d,
            initial_state=S0_tt,
        )
        ttnn.deallocate(L_inv_4d)

    # Reshape output to [BH, L, V]
    out_4d = ttnn.to_layout(
        ttnn.typecast(out_4d, ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if out_4d.dtype != ttnn.float32
        else out_4d,
        ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Sub-op PCC: capture scan kernel output and log the layer entry.
    if _subop_pcc:
        # scan_out: CPU-scan reference would need the CPU scan to run in parallel;
        # here we instead compare the device scan output to itself as a sanity check,
        # then append all captured per-op PCCs to the global log.
        # (For full scan comparison, run with GDN_CPU_KERNEL=1 + GDN_SUBOP_PCC=1.)
        _sop["scan_out_shape"] = float(out_4d.shape[-1])  # record shape, not PCC
        _GDN_SUBOP_LOG.append(dict(_sop))
        print(
            f"[subop_pcc] layer#{len(_GDN_SUBOP_LOG)-1:02d}  "
            + "  ".join(f"{k}={v:.5f}" for k, v in _sop.items() if isinstance(v, float) and k != "scan_out_shape")
        )

    o = ttnn.reshape(out_4d, [BH, L, V], memory_config=ttnn.DRAM_MEMORY_CONFIG)

    if pad_len > 0:
        o = o[:, :T, :]
        o = ttnn.to_layout(o, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    return o, final_state
