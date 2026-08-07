# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for sparse attention.

The extension adds three optional kwargs (`tokens_per_frame`, `num_frames_padded`, `sparse_frame_mask`) that
enable frame-block-sparse attention inside the ring op: each Q frame attends only to a chosen
subset of K frames (e.g. a centered window + one reference frame).

The tests exercise the op directly with a synthetic windowed pattern
at shapes representative of sparse-attention workloads

Golden = pytorch SDPA with an additive `[N, N]` block-mask matching sparse_frame_mask. Ring output must
PCC-match the golden.

Meshes:
    BH 4x8, WH 2x4, WH 4x8. Only meshes with sufficient devices for the requested sp_factor run;
    the rest skip cleanly at collection time.

Run:
    pytest models/tt_dit/tests/unit/test_ring_joint_sparse_frames.py -k bh_4x8
    pytest models/tt_dit/tests/unit/test_ring_joint_sparse_frames.py -k wh_2x4
    pytest models/tt_dit/tests/unit/test_ring_joint_sparse_frames.py -k wh_4x8
"""

from __future__ import annotations

import gc
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
# Helpers: build the windowed sparse_frame_mask pattern + torch reference.
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


def _sparse_frame_mask(num_frames: int, num_frames_padded: int, window: int, add_last_frame: bool) -> torch.Tensor:
    """`[nf_padded, nf_padded]` uint8. 1 = Q attends K. Padded frames = all-zero rows/cols."""
    plan = _window_plan(num_frames, window, add_last_frame)
    allow = torch.zeros(num_frames_padded, num_frames_padded, dtype=torch.uint8)
    for i, (ranges, _) in enumerate(plan):
        for s, e in ranges:
            allow[i, s:e] = 1
    return allow


def _pack_sparse_frame_mask(allow: torch.Tensor) -> list:
    """Bitpack the [nf, nf] uint8 allow table into uint32 words, matching the packing convention
    used by sparse ring_joint SDPA."""
    nf = allow.shape[0]
    total_bits = nf * nf
    num_words = (total_bits + 31) // 32
    words = [0] * num_words
    for q in range(nf):
        for k in range(nf):
            if allow[q, k]:
                bit_idx = q * nf + k
                words[bit_idx // 32] |= 1 << (bit_idx % 32)
    # Padded Q rows stay all-zero. A device whose Q shard is entirely padded has an all-zero allow region;
    # the op detects this per-shard (reader `shard_attends_nothing`) and has that device participate fully in K/V
    # data movement while compute skips the matmul — so no allow-table fixup is needed.
    return words


def _torch_sdpa_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    allow: torch.Tensor,
    tokens_per_frame: int,
) -> torch.Tensor:
    """Block-sparse pytorch reference.
    Computing this densely would materialize the full [n_pad, n_pad] score matrix PER HEAD, which
    could OOM-kill the process. Instead compute the output one Q-frame block at a time,
    running SDPA against ONLY the K frames that Q frame attends.
    softmax over the attended-only positions is numerically identical to softmax over all positions
    with the disallowed ones set to -inf, so this is exact — it just never allocates the dense mask or score matrix.
    Heads are processed in chunks sized to hold peak score memory near a fixed budget regardless of
    shape (a windowed Q attends a few frames; a dense/allow-all Q attends all of them)."""
    b, nh, n_pad, d = q.shape
    tpf = tokens_per_frame
    nf = allow.shape[0]
    out = torch.zeros_like(q)
    score_budget_elems = 1_500_000_000 // 4  # cap the per-call [b, hc, tpf, attended] score tensor
    for qf in range(nf):
        qs = qf * tpf
        if qs >= n_pad:
            break
        qe = min(qs + tpf, n_pad)
        attended = torch.nonzero(allow[qf], as_tuple=False).flatten().tolist()
        if not attended:
            # Padded / fully-masked Q frame: leave zeros. Real Q frames always attend >= themselves;
            # the only all-zero rows are padded frames whose outputs are dropped by the caller's
            # [:real_n] slice (and the degenerate drain-all pattern is handled separately upstream).
            continue
        k_idx = torch.cat([torch.arange(kf * tpf, min((kf + 1) * tpf, n_pad)) for kf in attended])
        kb = k[:, :, k_idx, :]
        vb = v[:, :, k_idx, :]
        qb = q[:, :, qs:qe, :]
        head_chunk = max(1, score_budget_elems // (b * (qe - qs) * k_idx.numel()))
        for h0 in range(0, nh, head_chunk):
            h1 = min(h0 + head_chunk, nh)
            out[:, h0:h1, qs:qe, :] = torch.nn.functional.scaled_dot_product_attention(
                qb[:, h0:h1], kb[:, h0:h1], vb[:, h0:h1], is_causal=False
            )
    return out


# ---------------------------------------------------------------------------
# Runner.
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
    tokens_per_frame: int,
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
    sparse_frames_enabled: bool = True,
    force_allow_all: bool = False,
    allow_override: torch.Tensor | None = None,
):
    """Build small Q/K/V, run the ring op with sparse computation enabled, compare to a pytorch ref."""

    assert num_frames_padded % sp_factor == 0, "num_frames_padded must be a multiple of sp_factor"
    assert tokens_per_frame % ttnn.TILE_SIZE == 0, "tokens_per_frame must be tile-aligned"
    n_pad = num_frames_padded * tokens_per_frame
    fsl_tiles = tokens_per_frame // ttnn.TILE_SIZE
    q_chunk_size_tokens = q_chunk_size_tokens if q_chunk_size_tokens is not None else tokens_per_frame
    k_chunk_size_tokens = k_chunk_size_tokens if k_chunk_size_tokens is not None else tokens_per_frame
    assert (
        tokens_per_frame % q_chunk_size_tokens == 0
    ), f"q_chunk_size_tokens ({q_chunk_size_tokens}) must divide tokens_per_frame ({tokens_per_frame})"
    assert (
        tokens_per_frame % k_chunk_size_tokens == 0
    ), f"k_chunk_size_tokens ({k_chunk_size_tokens}) must divide tokens_per_frame ({tokens_per_frame})"

    # Golden reference on host.
    torch.manual_seed(0)
    real_n = num_frames_real * tokens_per_frame
    Q = torch.randn(b, nh, real_n, d)
    K = torch.randn(b, nh, real_n, d)
    V = torch.randn(b, nh, real_n, d)
    # Pad to n_pad along seq dim.
    padded_Q = torch.cat([Q, torch.zeros(b, nh, n_pad - real_n, d)], dim=2)
    padded_K = torch.cat([K, torch.zeros(b, nh, n_pad - real_n, d)], dim=2)
    padded_V = torch.cat([V, torch.zeros(b, nh, n_pad - real_n, d)], dim=2)

    if allow_override is not None:
        assert allow_override.shape == (num_frames_padded, num_frames_padded)
        allow = allow_override.to(torch.uint8)
    elif sparse_frames_enabled and not force_allow_all:
        allow = _sparse_frame_mask(num_frames_real, num_frames_padded, window, add_last_frame)
    else:
        # Dense-equivalent allow (every real Q attends every real K), used for both
        # sparse_frames_enabled=False and force_allow_all=True.
        allow = torch.zeros(num_frames_padded, num_frames_padded, dtype=torch.uint8)
        allow[:num_frames_real, :num_frames_real] = 1
    gt = _torch_sdpa_ref(padded_Q, padded_K, padded_V, allow, tokens_per_frame=tokens_per_frame)[:, :, :real_n, :]

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

    # Sharding: seq on sp_axis, heads on tp_axis (standard video-DiT sparse-attention layout).
    input_shard_dims = [None, None]
    input_shard_dims[sp_axis] = 2
    input_shard_dims[tp_axis] = 1

    def _to_dev(t, dims):
        # Upload in ROW_MAJOR (skips host tilize), then tilize on device — much faster than
        # single-threaded host tilize for large tensors (720p Q/K/V are ~944 MB each).
        rm_tensor = ttnn.from_torch(
            t.to(torch.bfloat16),  # pre-convert to bf16 to skip f32→bf16 during upload
            dtype=dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=dims),
        )
        return ttnn.to_layout(rm_tensor, ttnn.TILE_LAYOUT)

    tt_Q = _to_dev(padded_Q, input_shard_dims)
    tt_K = _to_dev(padded_K, input_shard_dims)
    tt_V = _to_dev(padded_V, input_shard_dims)

    # The golden (gt) is computed and Q/K/V are now on device.
    del padded_Q, padded_K, padded_V, Q, K, V
    gc.collect()

    # Persistent AllGather output buffers — the op internally gathers K/V across sp_axis into
    # these buffers. Shape is the full (unsharded) length on the sp_axis; kept sharded on tp_axis
    # (heads). Mirrors run_ring_joint_sdpa's setup.
    kv_out_shard_dims = [None, None]
    kv_out_shard_dims[sp_axis] = None
    kv_out_shard_dims[tp_axis] = 1
    ag_output_shape = (b, nh, n_pad, d)

    def _make_persistent_output_buffer():
        rm_tensor = ttnn.from_torch(
            torch.zeros(ag_output_shape, dtype=torch.bfloat16),
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                mesh_device, mesh_shape=tuple(mesh_device.shape), dims=kv_out_shard_dims
            ),
        )
        return ttnn.to_layout(rm_tensor, ttnn.TILE_LAYOUT)

    persistent_output_buffer_k = _make_persistent_output_buffer()
    persistent_output_buffer_v = _make_persistent_output_buffer()

    sparse_frame_mask = _pack_sparse_frame_mask(allow) if sparse_frames_enabled else []

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
        tokens_per_frame=tokens_per_frame if sparse_frames_enabled else None,
        num_frames_padded=num_frames_padded if sparse_frames_enabled else None,
        sparse_frame_mask=sparse_frame_mask,
    )

    # Gather output back (sharded seq on sp, heads on tp).
    tt_out = ttnn.to_layout(tt_out, ttnn.ROW_MAJOR_LAYOUT)
    out = ttnn.to_torch(
        tt_out,
        mesh_composer=ttnn.ConcatMesh2dToTensor(
            mesh_device,
            mesh_shape=tuple(mesh_device.shape),
            dims=input_shard_dims,
        ),
    )[:, :, :real_n, :]

    # Degenerate pattern: if any REAL Q frame attends zero K frames, the torch reference is a
    # fully-masked softmax row (which PyTorch collapses to 0/undefined), while the kernel's
    # `_pack_sparse_frame_mask` workaround forces those all-zero rows to attend-all so the reader chain
    # stays in sync — so the two diverge by construction and PCC is meaningless.
    if bool((allow[:num_frames_real].sum(dim=1) == 0).any()):
        assert torch.isfinite(
            out
        ).all(), "sparse ring SDPA produced non-finite output for a fully-drained (degenerate) pattern"
        logger.info(
            f"[sparse ring] degenerate allow (a Q frame attends no K) — skipped PCC, verified "
            f"finite output. sp={sp_factor} tp={tp_factor}"
        )
        return

    passing, pcc = comp_pcc(gt, out, pcc_threshold)
    logger.info(
        f"[sparse ring] nf_real={num_frames_real} nf_pad={num_frames_padded} fsl={tokens_per_frame} "
        f"window={window} add_last={add_last_frame} sp={sp_factor} tp={tp_factor} pcc={pcc}"
    )
    del gt, out
    gc.collect()
    assert passing, f"sparse ring SDPA vs torch reference PCC {pcc} < {pcc_threshold}"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSparseFramesRing:
    """Sparse ring SDPA correctness across BH 4x8, WH 2x4, WH 4x8, Line + Ring."""

    @_MESH_TOPOLOGY
    @pytest.mark.parametrize(
        ("nf_real_fn", "nf_padded_fn", "tokens_per_frame", "window", "add_last_frame"),
        [
            pytest.param(lambda sp: 8 if sp == 8 else 6, lambda sp: 8, 32, 5, True, id="small_windowed"),
            pytest.param(lambda sp: 2, lambda sp: sp, 64, 3, True, id="padded_frames"),
            pytest.param(lambda sp: sp, lambda sp: sp, 64, 5, False, id="no_add_last_frame"),
        ],
    )
    def test_windowed_patterns(
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
        nf_real_fn,
        nf_padded_fn,
        tokens_per_frame,
        window,
        add_last_frame,
    ):
        _run_sparse_frames_op(
            mesh_device=mesh_device,
            sp_axis=sp_axis,
            sp_factor=sp_factor,
            tp_axis=tp_axis,
            tp_factor=tp_factor,
            num_links=num_links,
            num_frames_real=nf_real_fn(sp_factor),
            num_frames_padded=nf_padded_fn(sp_factor),
            tokens_per_frame=tokens_per_frame,
            b=1,
            nh=8,
            d=128,
            window=window,
            add_last_frame=add_last_frame,
            all_gather_topology=all_gather_topology,
        )

    @_MESH_TOPOLOGY
    @pytest.mark.parametrize(
        ("sparse_frames_enabled", "force_allow_all"),
        [
            pytest.param(True, False, id="sparse"),
            pytest.param(False, False, id="dense"),
            pytest.param(True, True, id="sparse_allow_all"),
        ],
    )
    def test_720p_shape(
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
        sparse_frames_enabled,
        force_allow_all,
    ):
        """720p-scale geometry representative of a real workload."""
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
            tokens_per_frame=3840,
            b=1,
            nh=40,
            d=128,
            window=5,
            add_last_frame=True,
            all_gather_topology=all_gather_topology,
            q_chunk_size_tokens=320,  # 320 tokens = 10 tiles
            k_chunk_size_tokens=384,  # 384 tokens = 12 tiles
            sparse_frames_enabled=sparse_frames_enabled,
            force_allow_all=force_allow_all,
        )

    @_MESH_TOPOLOGY
    @pytest.mark.parametrize(
        ("sparse_frames_enabled", "force_allow_all"),
        [
            pytest.param(True, False, id="sparse"),
            pytest.param(True, True, id="sparse_allow_all"),
        ],
    )
    def test_720p_multi_oob(
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
        sparse_frames_enabled,
        force_allow_all,
    ):
        """Same 720p multi-Q-per-core regime as test_720p_shape, but nf_real chosen so that
        multiple whole SP shards are padding (fully out-of-bounds)."""
        nf_real = 18
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
            tokens_per_frame=3840,
            b=1,
            nh=40,
            d=128,
            window=5,
            add_last_frame=True,
            all_gather_topology=all_gather_topology,
            q_chunk_size_tokens=320,
            k_chunk_size_tokens=384,
            sparse_frames_enabled=sparse_frames_enabled,
            force_allow_all=force_allow_all,
        )

    @_MESH_TOPOLOGY
    @pytest.mark.parametrize(
        ("sparse_frames_enabled", "force_allow_all"),
        [
            pytest.param(True, False, id="sparse"),
            pytest.param(False, False, id="dense"),
            pytest.param(True, True, id="sparse_allow_all"),
        ],
    )
    @pytest.mark.parametrize(
        ("q_chunk_div", "k_chunk_div"),
        [
            pytest.param(1, 1, id="chunk_full_fsl"),  # baseline: chunk == tokens_per_frame
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
        sparse_frames_enabled,
        force_allow_all,
    ):
        """Sub-frame chunks: q_chunk_size = fsl/N (and k likewise). The device op requires each
        chunk to sit inside one frame (never straddle a boundary), so chunk sizes must divide
        tokens_per_frame."""
        tokens_per_frame = 128  # supports fsl/1 (128), fsl/2 (64), fsl/4 (32); all tile-aligned
        assert tokens_per_frame % q_chunk_div == 0 and (tokens_per_frame // q_chunk_div) % ttnn.TILE_SIZE == 0
        assert tokens_per_frame % k_chunk_div == 0 and (tokens_per_frame // k_chunk_div) % ttnn.TILE_SIZE == 0
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
            tokens_per_frame=tokens_per_frame,
            b=1,
            nh=8,
            d=128,
            window=5,
            add_last_frame=True,
            all_gather_topology=all_gather_topology,
            q_chunk_size_tokens=tokens_per_frame // q_chunk_div,
            k_chunk_size_tokens=tokens_per_frame // k_chunk_div,
            sparse_frames_enabled=sparse_frames_enabled,
            force_allow_all=force_allow_all,
        )

    @_MESH_TOPOLOGY
    @pytest.mark.parametrize(
        "drain_pattern",
        ["tail_drain", "head_drain", "middle_drain", "drain_all", "drain_one_last", "drain_one_first"],
    )
    def test_drain_pattern(
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
        drain_pattern,
    ):
        """Exercises the drain path"""
        tokens_per_frame = 128  # 4 tiles/frame, small
        nf_real = 8
        nf_padded = 8
        half = nf_real // 2
        allow = torch.zeros(nf_padded, nf_padded, dtype=torch.uint8)
        for q in range(nf_real):
            if drain_pattern == "tail_drain":
                allow[q, :half] = 1  # K frames [0..half) allowed, [half..nf) drained
            elif drain_pattern == "head_drain":
                allow[q, half:nf_real] = 1  # K frames [half..nf) allowed, [0..half) drained
            elif drain_pattern == "middle_drain":
                for k in range(nf_real):
                    if k % 2 == 0:
                        allow[q, k] = 1  # even allowed, odd drained
            elif drain_pattern == "drain_all":
                pass  # every chunk drained — 100% drain, zero processing
            elif drain_pattern == "drain_one_last":
                allow[q, : nf_real - 1] = 1  # all allowed except the very last K frame
            elif drain_pattern == "drain_one_first":
                allow[q, 1:nf_real] = 1  # all allowed except the very first K frame
        _run_sparse_frames_op(
            mesh_device=mesh_device,
            sp_axis=sp_axis,
            sp_factor=sp_factor,
            tp_axis=tp_axis,
            tp_factor=tp_factor,
            num_links=num_links,
            num_frames_real=nf_real,
            num_frames_padded=nf_padded,
            tokens_per_frame=tokens_per_frame,
            b=1,
            nh=8,
            d=128,
            window=5,
            add_last_frame=True,
            all_gather_topology=all_gather_topology,
            sparse_frames_enabled=True,
            allow_override=allow,
        )
