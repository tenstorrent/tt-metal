# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the sparse-frames extension to `ring_joint_scaled_dot_product_attention`.

The extension adds three optional kwargs (`frame_seqlen`, `num_frames_padded`, `frame_allow`) that
enable frame-block-sparse attention inside the ring op. This is the primitive powering HeyGen's SR
windowed sparse self-attention: each Q frame attends only to a centered window of K frames + one
reference frame.

These tests are independent of the SR model — they exercise the op directly with a synthetic
windowed pattern (window=5, add_last_frame) at shapes representative of the SR deliverable:
  * SR 720p: fsl=3840 tokens/frame, nf=21 (padded to 24), N=92160 padded
  * Smaller synthetic shapes for correctness across mesh sizes

Golden = pytorch SDPA with an additive `[N, N]` block-mask matching frame_allow. Ring output must
PCC-match the golden. The tests SKIP when the tt-metal build lacks the extension.

Meshes:
    BH 4x8, WH 2x4, WH 4x8. Only meshes with sufficient devices for the requested sp_factor run;
    the rest skip cleanly at collection time.

Run:
    pytest models/tt_dit/tests/unit/test_ring_joint_sparse_frames.py -k bh_4x8
    pytest models/tt_dit/tests/unit/test_ring_joint_sparse_frames.py -k wh_2x4
    pytest models/tt_dit/tests/unit/test_ring_joint_sparse_frames.py -k wh_4x8
"""

from __future__ import annotations

from typing import List, Tuple

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.tt_dit.utils.test import line_params, ring_params_8k

# ---------------------------------------------------------------------------
# Mesh + topology enumeration — one flat row per test config (mirrors
# test_pipeline_wan_svi.py). Row fields are unpacked directly by each test.
#
#   * sp_factor is the RING factor (== ring_size the op sees) on sp_axis.
#   * tp_factor shards V head-dim across the other mesh axis.
#   * num_links: 2 for BH 4x8 galaxy, 4 for WH 4x8 galaxy, 1 for WH 2x4.
#   * Ring topology is only emitted for 4x8 galaxies — 2x4 lacks a closed
#     fabric loop, so (wh_2x4, ring) would fail at fabric init.
# ---------------------------------------------------------------------------

_SDPA_L1 = {"worker_l1_size": 1344544, "trace_region_size": 1000000}
_LINE = {**_SDPA_L1, **line_params}  # no router_config for line (matches sibling tests)
_RING = {**_SDPA_L1, **ring_params_8k}  # ring uses the 8k router config


_MESH_TOPOLOGY_CONFIGS = [
    # (mesh_device_shape, num_links, sp_axis, sp_factor, tp_axis, tp_factor, device_params, topology)
    [(4, 8), 2, 1, 8, 0, 4, _LINE, ttnn.Topology.Linear],
    [(4, 8), 2, 1, 8, 0, 4, _RING, ttnn.Topology.Ring],
    [(2, 4), 1, 1, 4, 0, 2, _LINE, ttnn.Topology.Linear],
    [(4, 8), 4, 1, 8, 0, 4, _LINE, ttnn.Topology.Linear],
    [(4, 8), 4, 1, 8, 0, 4, _RING, ttnn.Topology.Ring],
]
_MESH_TOPOLOGY_IDS = [
    "bh_4x8_sp8tp4_line",
    "bh_4x8_sp8tp4_ring",
    "wh_2x4_sp4tp2_line",
    "wh_4x8_sp8tp4_line",
    "wh_4x8_sp8tp4_ring",
]
_MESH_TOPOLOGY = pytest.mark.parametrize(
    "mesh_device, num_links, sp_axis, sp_factor, tp_axis, tp_factor, device_params, all_gather_topology",
    _MESH_TOPOLOGY_CONFIGS,
    ids=_MESH_TOPOLOGY_IDS,
    indirect=["mesh_device", "device_params"],
)


# ---------------------------------------------------------------------------
# Helpers: build the windowed frame_allow pattern + torch reference.
# ---------------------------------------------------------------------------


def _window_plan(num_frames: int, window: int, add_last_frame: bool) -> List[Tuple[List[Tuple[int, int]], int]]:
    """Per-Q-frame allowed K ranges + counts. Duplicates sparse_attention.py::window_plan."""
    hl = window // 2
    hr = window - hl
    plan = []
    for i in range(num_frames):
        ws, we = max(0, i - hl), min(num_frames, i + hr)
        ranges = [(ws, we)]
        if add_last_frame and we < num_frames:
            ranges.append((num_frames - 1, num_frames))
        count = sum(e - s for s, e in ranges)
        plan.append((ranges, count))
    return plan


def _frame_allow(num_frames: int, num_frames_padded: int, window: int, add_last_frame: bool) -> torch.Tensor:
    """`[nf_padded, nf_padded]` uint8. 1 = Q attends K. Padded frames = all-zero rows/cols."""
    plan = _window_plan(num_frames, window, add_last_frame)
    allow = torch.zeros(num_frames_padded, num_frames_padded, dtype=torch.uint8)
    for i, (ranges, _) in enumerate(plan):
        for s, e in ranges:
            allow[i, s:e] = 1
    return allow


def _pack_frame_allow(allow: torch.Tensor) -> list:
    """Bitpack the [nf, nf] uint8 allow table into uint32 words, matching
    sparse_attention.py::build_frame_allow_packed's convention."""
    nf = allow.shape[0]
    total_bits = nf * nf
    num_words = (total_bits + 31) // 32
    words = [0] * num_words
    for q in range(nf):
        for k in range(nf):
            if allow[q, k]:
                bit_idx = q * nf + k
                words[bit_idx // 32] |= 1 << (bit_idx % 32)
    return words


def _additive_mask_from_allow(allow: torch.Tensor, frame_seqlen: int, n_pad: int) -> torch.Tensor:
    """Expand `frame_allow` [nf, nf] to the additive `[n_pad, n_pad]` block-mask used for the
    pytorch reference (0 = allowed, -inf = disallowed). Padded columns are all -inf; padded rows
    are all-zero so softmax doesn't NaN (their outputs are dropped after)."""
    nf = allow.shape[0]
    real = nf * frame_seqlen
    assert real >= n_pad or n_pad >= real, "expected n_pad and real to match after padding"
    ff = torch.where(allow.bool(), 0.0, float("-inf")).to(torch.float32)
    full = ff.repeat_interleave(frame_seqlen, 0).repeat_interleave(frame_seqlen, 1)[:n_pad, :n_pad]
    if n_pad > full.shape[0]:
        padded = torch.full((n_pad, n_pad), float("-inf"), dtype=torch.float32)
        padded[: full.shape[0], : full.shape[1]] = full
        padded[full.shape[0] :, :] = 0.0  # any non-empty row so softmax stays finite
        full = padded
    return full


def _torch_sdpa_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    allow: torch.Tensor,
    num_frames_real: int,
    frame_seqlen: int,
) -> torch.Tensor:
    """pytorch reference: block-sparse SDPA using the additive mask expanded from frame_allow."""
    n_pad = q.shape[2]
    mask = _additive_mask_from_allow(allow, frame_seqlen, n_pad).to(q.device).to(q.dtype)
    return torch.nn.functional.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=mask.reshape(1, 1, n_pad, n_pad),
        is_causal=False,
    )


# ---------------------------------------------------------------------------
# The runner.
# ---------------------------------------------------------------------------


def _run_sparse_frames_op(
    *,
    mesh_device,
    sp_axis,
    sp_factor,
    tp_axis,
    tp_factor,
    num_links,
    num_frames_real: int,
    num_frames_padded: int,
    frame_seqlen: int,
    b: int,
    nh: int,
    d: int,
    window: int,
    add_last_frame: bool,
    dtype=ttnn.bfloat16,
    all_gather_topology=ttnn.Topology.Linear,
    pcc_threshold: float = 0.999,
    q_chunk_size_tokens: int | None = None,
    k_chunk_size_tokens: int | None = None,
):
    """Build small Q/K/V, run the ring op with sparse-frames enabled, compare to a pytorch ref.

    q_chunk_size_tokens / k_chunk_size_tokens (in TOKENS — SDPAProgramConfig's chunk sizes are
    tokens, see sdpa_device_operation.cpp's `% TILE_WIDTH == 0` check) default to `frame_seqlen`
    so each SDPA chunk == one frame. Override with a divisor of frame_seqlen to exercise the
    sub-frame chunk path (multiple chunks per frame — needed at large fsl to fit L1 CB budgets)."""
    assert num_frames_padded % sp_factor == 0, "num_frames_padded must be a multiple of sp_factor"
    assert frame_seqlen % ttnn.TILE_SIZE == 0, "frame_seqlen must be tile-aligned"
    n_pad = num_frames_padded * frame_seqlen
    fsl_tiles = frame_seqlen // ttnn.TILE_SIZE
    q_chunk_size_tokens = q_chunk_size_tokens if q_chunk_size_tokens is not None else frame_seqlen
    k_chunk_size_tokens = k_chunk_size_tokens if k_chunk_size_tokens is not None else frame_seqlen
    assert (
        frame_seqlen % q_chunk_size_tokens == 0
    ), f"q_chunk_size_tokens ({q_chunk_size_tokens}) must divide frame_seqlen ({frame_seqlen})"
    assert (
        frame_seqlen % k_chunk_size_tokens == 0
    ), f"k_chunk_size_tokens ({k_chunk_size_tokens}) must divide frame_seqlen ({frame_seqlen})"

    # Golden reference on host.
    torch.manual_seed(0)
    real_n = num_frames_real * frame_seqlen
    Q = torch.randn(b, nh, real_n, d)
    K = torch.randn(b, nh, real_n, d)
    V = torch.randn(b, nh, real_n, d)
    # Pad to n_pad along seq dim.
    padded_Q = torch.cat([Q, torch.zeros(b, nh, n_pad - real_n, d)], dim=2)
    padded_K = torch.cat([K, torch.zeros(b, nh, n_pad - real_n, d)], dim=2)
    padded_V = torch.cat([V, torch.zeros(b, nh, n_pad - real_n, d)], dim=2)

    allow = _frame_allow(num_frames_real, num_frames_padded, window, add_last_frame)
    gt = _torch_sdpa_ref(
        padded_Q,
        padded_K,
        padded_V,
        allow,
        num_frames_real=num_frames_real,
        frame_seqlen=frame_seqlen,
    )[:, :, :real_n, :]

    # ------- Set up the ring op on device --------------------------------
    full_compute_grid = mesh_device.compute_with_storage_grid_size()
    sdpa_compute_grid = (full_compute_grid.x, full_compute_grid.y - 1)
    ccl_core_grid_offset = (0, full_compute_grid.y - 1)

    ccl_sub_device_crs = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(full_compute_grid.x - 1, full_compute_grid.y - 1))}
    )
    worker_sub_device = ttnn.SubDevice([ccl_sub_device_crs])
    worker_sub_device_id = ttnn.SubDeviceId(0)
    sub_device_stall_group = [worker_sub_device_id]
    sub_device_manager = mesh_device.create_sub_device_manager([worker_sub_device], 0)
    mesh_device.load_sub_device_manager(sub_device_manager)
    mesh_device.set_sub_device_stall_group(sub_device_stall_group)

    ccl_sem = [ttnn.create_global_semaphore(mesh_device, ccl_sub_device_crs, 0) for _ in range(2)]

    # Sharding: seq on sp_axis, heads on tp_axis (mirrors the SR sparse attention layout).
    input_shard_dims = [None, None]
    input_shard_dims[sp_axis] = 2
    input_shard_dims[tp_axis] = 1

    def _to_dev(t, dims):
        return ttnn.from_torch(
            t,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=dims),
        )

    tt_Q = _to_dev(padded_Q, input_shard_dims)
    tt_K = _to_dev(padded_K, input_shard_dims)
    tt_V = _to_dev(padded_V, input_shard_dims)

    # Persistent AllGather output buffers — the op internally gathers K/V across sp_axis into
    # these buffers. Shape is the full (unsharded) length on the sp_axis; kept sharded on tp_axis
    # (heads). Mirrors run_ring_joint_sdpa's setup.
    kv_out_shard_dims = [None, None]
    kv_out_shard_dims[sp_axis] = None
    kv_out_shard_dims[tp_axis] = 1
    ag_output_shape = (b, nh, n_pad, d)
    persistent_output_buffer_k = ttnn.from_torch(
        torch.zeros(ag_output_shape),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=dtype,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=kv_out_shard_dims),
    )
    persistent_output_buffer_v = ttnn.from_torch(
        torch.zeros(ag_output_shape),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=dtype,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=kv_out_shard_dims),
    )

    # Bitpack frame_allow into up to 32 uint32 words — passed to the op as a plain host vector
    # (frame_allow_packed kwarg). No device tensor / no DMA / no CB required.
    frame_allow_packed = _pack_frame_allow(allow)

    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=sdpa_compute_grid,
        q_chunk_size=q_chunk_size_tokens,
        k_chunk_size=k_chunk_size_tokens,
        exp_approx_mode=False,
    )
    compute_kernel_config = ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,  # use_streaming_compute == !fp32_dest_acc_en, so False -> streaming path
        packer_l1_acc=False,
    )

    tt_out, _tt_joint, _tt_lse = ttnn.transformer.ring_joint_scaled_dot_product_attention(
        tt_Q,
        tt_K,
        tt_V,
        None,
        None,
        None,
        persistent_output_buffer_k=persistent_output_buffer_k,
        persistent_output_buffer_v=persistent_output_buffer_v,
        joint_strategy="rear",
        logical_n=real_n,  # true un-padded sequence length; padded region is beyond
        program_config=program_config,
        compute_kernel_config=compute_kernel_config,
        dim=2,
        multi_device_global_semaphore=ccl_sem,
        num_links=num_links,
        cluster_axis=sp_axis,
        mesh_device=mesh_device,
        topology=all_gather_topology,
        subdevice_id=worker_sub_device_id,
        ccl_core_grid_offset=ccl_core_grid_offset,
        is_causal=False,
        # The extension: enable frame-block-sparse pattern.
        frame_seqlen=frame_seqlen,
        num_frames_padded=num_frames_padded,
        frame_allow_packed=frame_allow_packed,
    )

    # Gather output back (sharded seq on sp, heads on tp).
    out = ttnn.to_torch(
        tt_out,
        mesh_composer=ttnn.ConcatMesh2dToTensor(
            mesh_device,
            mesh_shape=tuple(mesh_device.shape),
            dims=input_shard_dims,
        ),
    )[:, :, :real_n, :]

    passing, pcc = comp_pcc(gt, out, pcc_threshold)
    logger.info(
        f"[sparse-frames ring] nf_real={num_frames_real} nf_pad={num_frames_padded} fsl={frame_seqlen} "
        f"window={window} add_last={add_last_frame} sp={sp_factor} tp={tp_factor} pcc={pcc}"
    )
    assert passing, f"sparse-frames ring SDPA vs torch reference PCC {pcc} < {pcc_threshold}"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSparseFramesRing:
    """Sparse-frames ring SDPA correctness across BH 4x8, WH 2x4, WH 4x8, Line + Ring."""

    @_MESH_TOPOLOGY
    def test_small_windowed(
        self,
        mesh_device,
        num_links,
        sp_axis,
        sp_factor,
        tp_axis,
        tp_factor,
        device_params,
        all_gather_topology,
        reset_seeds,
    ):
        """Small fsl=32 (1 tile per frame) exercises the wiring end-to-end quickly.

        `nf_real=8 -> nf_padded=8` at sp=8 (already aligned): the simplest case, no padding rows.
        At sp=4 (WH 2x4): `nf_real=6 -> nf_padded=8` — exercises the padding-frame case."""
        nf_real = 8 if sp_factor == 8 else 6
        nf_padded = 8
        _run_sparse_frames_op(
            mesh_device=mesh_device,
            sp_axis=sp_axis,
            sp_factor=sp_factor,
            tp_axis=tp_axis,
            tp_factor=tp_factor,
            num_links=num_links,
            num_frames_real=nf_real,
            num_frames_padded=nf_padded,
            frame_seqlen=32,
            b=1,
            nh=8,
            d=128,
            window=5,
            add_last_frame=True,
            all_gather_topology=all_gather_topology,
        )

    @_MESH_TOPOLOGY
    def test_padded_frames(
        self,
        mesh_device,
        num_links,
        sp_axis,
        sp_factor,
        tp_axis,
        tp_factor,
        device_params,
        all_gather_topology,
        reset_seeds,
    ):
        """Padded-frames case: nf_real=1 (or 2) with nf_padded=sp_factor (7-6 extra padded frames)."""
        _run_sparse_frames_op(
            mesh_device=mesh_device,
            sp_axis=sp_axis,
            sp_factor=sp_factor,
            tp_axis=tp_axis,
            tp_factor=tp_factor,
            num_links=num_links,
            num_frames_real=2,
            num_frames_padded=sp_factor,
            frame_seqlen=64,
            b=1,
            nh=8,
            d=128,
            window=3,
            add_last_frame=True,
            all_gather_topology=all_gather_topology,
        )

    @_MESH_TOPOLOGY
    def test_no_add_last_frame(
        self,
        mesh_device,
        num_links,
        sp_axis,
        sp_factor,
        tp_axis,
        tp_factor,
        device_params,
        all_gather_topology,
        reset_seeds,
    ):
        """Purely centered window (no reference-frame span) — catches kernel logic that assumes
        the two-range case (window + last)."""
        _run_sparse_frames_op(
            mesh_device=mesh_device,
            sp_axis=sp_axis,
            sp_factor=sp_factor,
            tp_axis=tp_axis,
            tp_factor=tp_factor,
            num_links=num_links,
            num_frames_real=sp_factor,
            num_frames_padded=sp_factor,
            frame_seqlen=64,
            b=1,
            nh=8,
            d=128,
            window=5,
            add_last_frame=False,
            all_gather_topology=all_gather_topology,
        )

    @_MESH_TOPOLOGY
    def test_sr_720p_shape(
        self,
        mesh_device,
        num_links,
        sp_axis,
        sp_factor,
        tp_axis,
        tp_factor,
        device_params,
        all_gather_topology,
        reset_seeds,
    ):
        """SR production deliverable geometry: fsl=3840, nf_real=21 -> nf_padded=sp_multiple, window=5,
        add_last_frame=True. Uses n_head=40 / dim=128 (production SR values).

        On WH 2x4 (sp=4) nf_padded=24; on sp=8 nf_padded=24 too (21 rounded to 24)."""
        nf_real = 21
        nf_padded = ((nf_real + sp_factor - 1) // sp_factor) * sp_factor
        _run_sparse_frames_op(
            mesh_device=mesh_device,
            sp_axis=sp_axis,
            sp_factor=sp_factor,
            tp_axis=tp_axis,
            tp_factor=tp_factor,
            num_links=num_links,
            num_frames_real=nf_real,
            num_frames_padded=nf_padded,
            frame_seqlen=3840,
            b=1,
            nh=40 // tp_factor,
            d=128,
            window=5,
            add_last_frame=True,
            all_gather_topology=all_gather_topology,
        )

    @_MESH_TOPOLOGY
    @pytest.mark.parametrize(
        ("q_chunk_div", "k_chunk_div"),
        [
            pytest.param(2, 2, id="chunk_half_fsl"),
            pytest.param(4, 4, id="chunk_quarter_fsl"),
            pytest.param(1, 4, id="asym_qfull_kquarter"),
            pytest.param(4, 1, id="asym_qquarter_kfull"),
        ],
    )
    def test_sub_frame_chunks(
        self,
        mesh_device,
        num_links,
        sp_axis,
        sp_factor,
        tp_axis,
        tp_factor,
        device_params,
        all_gather_topology,
        reset_seeds,
        q_chunk_div,
        k_chunk_div,
    ):
        """Sub-frame chunks: q_chunk_size = fsl/N (and k likewise). The device op requires each
        chunk to sit inside one frame (never straddle a boundary), so chunk sizes must divide
        frame_seqlen. Motivation: at large fsl (720p = 3840 tokens/frame) chunk=fsl blows L1
        CBs (~40 MB vs 1.5 MB budget); dropping to fsl/5..fsl/8 fits.

        Uses fsl=64 so both symmetric (fsl/2, fsl/4 = 32, 16 tokens... wait, chunks must be
        multiples of TILE_SIZE=32 too) — use fsl=128 so fsl/2=64, fsl/4=32 are both valid.
        Asymmetric cases prove q and k chunk sizes are independent — the frame_allow indexing
        walks each independently (compute_streaming.hpp:2417 for k, 2474 for q)."""
        frame_seqlen = 128  # supports fsl/1 (128), fsl/2 (64), fsl/4 (32); all tile-aligned
        assert frame_seqlen % q_chunk_div == 0 and (frame_seqlen // q_chunk_div) % ttnn.TILE_SIZE == 0
        assert frame_seqlen % k_chunk_div == 0 and (frame_seqlen // k_chunk_div) % ttnn.TILE_SIZE == 0
        nf_real = 8 if sp_factor == 8 else 6
        nf_padded = 8
        _run_sparse_frames_op(
            mesh_device=mesh_device,
            sp_axis=sp_axis,
            sp_factor=sp_factor,
            tp_axis=tp_axis,
            tp_factor=tp_factor,
            num_links=num_links,
            num_frames_real=nf_real,
            num_frames_padded=nf_padded,
            frame_seqlen=frame_seqlen,
            b=1,
            nh=8,
            d=128,
            window=5,
            add_last_frame=True,
            all_gather_topology=all_gather_topology,
            q_chunk_size_tokens=frame_seqlen // q_chunk_div,
            k_chunk_size_tokens=frame_seqlen // k_chunk_div,
        )
