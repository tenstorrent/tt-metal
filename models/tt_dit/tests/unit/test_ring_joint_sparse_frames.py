# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for sparse attention, running the op with a synthetic windowed pattern
at shapes representative of real workloads
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
# Mesh + topology enumeration
# ---------------------------------------------------------------------------

_SDPA_L1 = {"worker_l1_size": 1344544, "trace_region_size": 1000000}
_LINE = {**_SDPA_L1, **line_params}  # no router_config for line (matches sibling tests)
_RING = {**_SDPA_L1, **ring_params_8k}  # ring uses the 8k router config


# Galaxy configs, running all tests.
_MESH_TOPOLOGY_CONFIGS = [
    # (mesh_device_shape, num_links, sp_axis, sp_factor, tp_axis, tp_factor, device_params, topology)
    [(4, 8), 2, 1, 8, 0, 4, _LINE, ttnn.Topology.Linear],
    [(4, 8), 2, 1, 8, 0, 4, _RING, ttnn.Topology.Ring],
    [(4, 8), 4, 1, 8, 0, 4, _LINE, ttnn.Topology.Linear],
    [(4, 8), 4, 1, 8, 0, 4, _RING, ttnn.Topology.Ring],
]
_MESH_TOPOLOGY_IDS = [
    "bh_4x8_sp8tp4_line",
    "bh_4x8_sp8tp4_ring",
    "wh_4x8_sp8tp4_line",
    "wh_4x8_sp8tp4_ring",
]

# Configs running only the cheaper tests.
_MESH_TOPOLOGY_SMALL_CONFIGS = [
    [(2, 4), 1, 1, 4, 0, 2, _LINE, ttnn.Topology.Linear],
    [(1, 4), 1, 1, 4, 0, 1, _LINE, ttnn.Topology.Linear],
]
_MESH_TOPOLOGY_SMALL_IDS = [
    "wh_2x4_sp4tp2_line",
    "bh_qb2_sp4tp1_line",
]

_MESH_TOPOLOGY_ARGS = (
    "mesh_device, num_links, sp_axis, sp_factor, tp_axis, tp_factor, device_params, all_gather_topology"
)

# Small tests.
_MESH_TOPOLOGY = pytest.mark.parametrize(
    _MESH_TOPOLOGY_ARGS,
    _MESH_TOPOLOGY_CONFIGS + _MESH_TOPOLOGY_SMALL_CONFIGS,
    ids=_MESH_TOPOLOGY_IDS + _MESH_TOPOLOGY_SMALL_IDS,
    indirect=["mesh_device", "device_params"],
)

# Large tests.
_MESH_TOPOLOGY_GALAXY = pytest.mark.parametrize(
    _MESH_TOPOLOGY_ARGS,
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
            # Only padding frames (q >= num_frames_real) are all-zero; their outputs are dropped by
            # the caller's [:real_n] slice. The op rejects all-zero rows for real frames, so a real Q
            # frame here would never have an empty attended set. Leave zeros for the padding rows.
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
    reference_as_extra_k: bool = False,
):
    """Build small Q/K/V, run the ring op with sparse computation enabled, compare to a pytorch ref.

    reference_as_extra_k: peel the reference frame (last real frame, attended by every query) out of the
    spatial mask and deliver it as a replicated stacked reference_kv buffer processed as one extra ring
    iteration. The spatial gather then only needs the window band. Golden is unchanged, so a match proves
    window-only spatial + reference-as-extra-K == full windowed+reference."""

    assert tokens_per_frame % ttnn.TILE_SIZE == 0, "tokens_per_frame must be tile-aligned"
    n_pad = num_frames_padded * tokens_per_frame
    # A shard may hold a fractional number of frames; only the padded sequence must shard evenly.
    assert n_pad % sp_factor == 0, "padded sequence (num_frames_padded * tokens_per_frame) must divide sp_factor"
    fsl_tiles = tokens_per_frame // ttnn.TILE_SIZE
    q_chunk_size_tokens = q_chunk_size_tokens if q_chunk_size_tokens is not None else tokens_per_frame
    k_chunk_size_tokens = k_chunk_size_tokens if k_chunk_size_tokens is not None else tokens_per_frame
    assert (
        tokens_per_frame % q_chunk_size_tokens == 0
    ), f"q_chunk_size_tokens ({q_chunk_size_tokens}) must divide tokens_per_frame ({tokens_per_frame})"
    assert (
        tokens_per_frame % k_chunk_size_tokens == 0
    ), f"k_chunk_size_tokens ({k_chunk_size_tokens}) must divide tokens_per_frame ({tokens_per_frame})"
    # No chunk may straddle a frame, so the per-device sequence must be a whole number of chunks.
    per_device_seq = n_pad // sp_factor
    assert (
        per_device_seq % q_chunk_size_tokens == 0
    ), f"per-device seq ({per_device_seq}) must be a whole number of q_chunks ({q_chunk_size_tokens})"
    assert (
        per_device_seq % k_chunk_size_tokens == 0
    ), f"per-device seq ({per_device_seq}) must be a whole number of k_chunks ({k_chunk_size_tokens})"

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

    # Peel the reference frame out of the op's spatial mask so the windowed gather no longer needs the
    # far reference shards; `allow` (hence `gt`) keeps the full pattern. The reference rides reference_kv.
    ref_frame = num_frames_real - 1  # the last real frame is the reference (see _window_plan add_last)
    if reference_as_extra_k:
        assert add_last_frame, "reference delivery expects the add-last-frame (reference) pattern"
        spatial_allow = allow.clone()
        spatial_allow[:, ref_frame] = 0  # every query gets the reference via reference_kv, not spatial
        # Reference frame's real tokens (< real_n), stacked [K frame | V frame] on the seq dim -> [b,nh,2*tpf,d].
        ref_slice = slice(ref_frame * tokens_per_frame, (ref_frame + 1) * tokens_per_frame)
        ref_K = padded_K[:, :, ref_slice, :].contiguous()
        ref_V = padded_V[:, :, ref_slice, :].contiguous()
        ref_kv_stacked = torch.cat([ref_K, ref_V], dim=2).contiguous()
    else:
        spatial_allow = allow

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

    sparse_frame_mask = _pack_sparse_frame_mask(spatial_allow) if sparse_frames_enabled else []

    # Extra-K path: stacked reference_kv replicated on sp, sharded on heads (tp), like the spatial inputs.
    tt_reference_kv = None
    if reference_as_extra_k:
        ref_shard_dims = [None, None]
        ref_shard_dims[sp_axis] = None  # replicated across the ring
        ref_shard_dims[tp_axis] = 1  # heads sharded
        tt_reference_kv = _to_dev(ref_kv_stacked, ref_shard_dims)
        del ref_K, ref_V, ref_kv_stacked
        gc.collect()

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
        None,  # no joint Q/K/V — reference rides reference_kv, not the joint path
        None,
        None,
        persistent_output_buffer_k=persistent_output_buffer_k,
        persistent_output_buffer_v=persistent_output_buffer_v,
        joint_strategy="rear",
        logical_n=real_n,  # true un-padded sequence length; padded region is beyond
        logical_l=0,
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
        reference_kv=tt_reference_kv,
        reference_frame_idx=(ref_frame if reference_as_extra_k else None),
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

    @_MESH_TOPOLOGY_GALAXY
    @pytest.mark.parametrize(
        ("sparse_frames_enabled", "force_allow_all"),
        [
            pytest.param(True, False, id="sparse"),
            pytest.param(False, False, id="dense"),
            pytest.param(True, True, id="sparse_allow_all"),
        ],
    )
    @pytest.mark.parametrize("tokens_per_frame", [pytest.param(3840, id="720p"), pytest.param(1920, id="480p")])
    def test_video_shape(
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
        tokens_per_frame,
    ):
        """Video-DiT-scale geometry. 720p uses frame-size=3840; 480p uses frame-size=1920.
        The 320/384-token chunk sizes divide both values."""
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
            tokens_per_frame=tokens_per_frame,
            b=1,
            nh=40,
            d=128,
            window=5,
            add_last_frame=True,
            all_gather_topology=all_gather_topology,
            q_chunk_size_tokens=320,  # 10 tiles
            k_chunk_size_tokens=384,  # 12 tiles
            sparse_frames_enabled=sparse_frames_enabled,
            force_allow_all=force_allow_all,
        )

    @_MESH_TOPOLOGY_GALAXY
    @pytest.mark.parametrize(
        ("sparse_frames_enabled", "force_allow_all"),
        [
            pytest.param(True, False, id="sparse"),
            pytest.param(True, True, id="sparse_allow_all"),
        ],
    )
    @pytest.mark.parametrize("tokens_per_frame", [pytest.param(3840, id="720p"), pytest.param(1920, id="480p")])
    def test_video_multi_oob(
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
        tokens_per_frame,
    ):
        """Same multi-Q-per-core regime as test_video_shape, but nf_real chosen so that multiple whole
        SP shards are padding (fully out-of-bounds). 720p: fsl=3840; 480p: fsl=1920."""
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
            tokens_per_frame=tokens_per_frame,
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

    @_MESH_TOPOLOGY_GALAXY
    @pytest.mark.parametrize(
        ("sparse_frames_enabled", "force_allow_all"),
        [
            pytest.param(True, False, id="sparse"),
            pytest.param(True, True, id="sparse_allow_all"),
        ],
    )
    @pytest.mark.parametrize("tokens_per_frame", [pytest.param(3840, id="720p"), pytest.param(1920, id="480p")])
    def test_video_fractional_frames(
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
        tokens_per_frame,
    ):
        """Video-scale geometry where num_frames_padded is not a multiple of sp_factor, so each shard
        holds a fractional number of frames and straddles frame boundaries."""
        nf_padded = sp_factor + sp_factor // 2
        nf_real = nf_padded - 2
        assert nf_padded % sp_factor != 0, "this test must exercise the fractional-frames-per-shard path"
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
            nh=40,
            d=128,
            window=5,
            add_last_frame=True,
            all_gather_topology=all_gather_topology,
            q_chunk_size_tokens=320,  # divides both the frame and the fractional per-device seq
            k_chunk_size_tokens=320,
            sparse_frames_enabled=sparse_frames_enabled,
            force_allow_all=force_allow_all,
        )

    @_MESH_TOPOLOGY
    @pytest.mark.parametrize(
        "drain_pattern",
        ["windowed", "tail_drain", "head_drain", "middle_drain"],
    )
    def test_fractional_frames_sub_frame(
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
        """Small-shape fractional-frames-per-shard (num_frames_padded not a multiple of sp_factor)."""
        tokens_per_frame = 64
        nf_padded = sp_factor + sp_factor // 2
        nf_real = nf_padded - 2
        assert nf_padded % sp_factor != 0, "this test must exercise the fractional-frames-per-shard path"

        allow_override = None
        if drain_pattern != "windowed":
            allow = torch.zeros(nf_padded, nf_padded, dtype=torch.uint8)
            half = nf_real // 2
            for q in range(nf_real):
                if drain_pattern == "tail_drain":
                    allow[q, :half] = 1
                elif drain_pattern == "head_drain":
                    allow[q, half:nf_real] = 1
                elif drain_pattern == "middle_drain":
                    for k in range(nf_real):
                        if k % 2 == 0:
                            allow[q, k] = 1
            allow_override = allow

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
            q_chunk_size_tokens=tokens_per_frame // 2,
            k_chunk_size_tokens=tokens_per_frame // 2,
            sparse_frames_enabled=True,
            allow_override=allow_override,
        )

    @_MESH_TOPOLOGY
    @pytest.mark.parametrize(
        ("tokens_per_frame", "window"),
        [
            # No reference frame (add_last=False), so the window alone drives W. ~1 frame/shard with a
            # +-2 window gives W=2 << ring_size -- exercises the genuinely-small-W windowed gather (the
            # path never hit by the add_last tests, which keep W=full via the far reference frame).
            pytest.param(128, 5, id="win5_w2"),
            pytest.param(128, 3, id="win3_w1"),
        ],
    )
    def test_windowed_small_radius(
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
        tokens_per_frame,
        window,
    ):
        """Windowed gather with a genuinely small radius (no joint). Validates the build_ring_work_plan
        window-bounding fix: without it, out-of-window active bits poison is_last_active_ring_iter and
        the ring deadlocks. 1 frame/shard so nf_padded == sp_factor keeps the padded seq shard-even."""
        _run_sparse_frames_op(
            mesh_device=mesh_device,
            sp_axis=sp_axis,
            sp_factor=sp_factor,
            tp_axis=tp_axis,
            tp_factor=tp_factor,
            num_links=num_links,
            num_frames_real=sp_factor,
            num_frames_padded=sp_factor,
            tokens_per_frame=tokens_per_frame,
            b=1,
            nh=8,
            d=128,
            window=window,
            add_last_frame=False,
            all_gather_topology=all_gather_topology,
            q_chunk_size_tokens=tokens_per_frame // 2,
            k_chunk_size_tokens=tokens_per_frame // 2,
            sparse_frames_enabled=True,
        )

    @_MESH_TOPOLOGY
    @pytest.mark.parametrize(
        ("tokens_per_frame", "nf_real_fn", "nf_padded_fn"),
        [
            # Fractional frames/shard (the real sp=32 regime): reference frame sits several shards away,
            # so the windowed spatial gather (reference peeled) collapses W to the window span.
            pytest.param(64, lambda sp: sp + sp // 2 - 2, lambda sp: sp + sp // 2, id="frac_1p5"),
            # Whole frame/shard: reference still far from the low devices.
            pytest.param(128, lambda sp: sp, lambda sp: sp, id="whole_1p0"),
        ],
    )
    def test_reference_as_extra_k(
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
        tokens_per_frame,
        nf_real_fn,
        nf_padded_fn,
    ):
        """Windowed-CCL Phase 2 (extra-K): deliver the reference frame as a replicated stacked reference_kv
        buffer processed as one extra ring iteration (no joint queries). Spatial gather is windowed (the
        reference column is peeled from the mask). Same windowed+reference golden -> PCC match proves
        window-only spatial gather + reference-as-extra-K == full windowed+reference."""
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
            window=5,
            add_last_frame=True,
            all_gather_topology=all_gather_topology,
            q_chunk_size_tokens=tokens_per_frame // 2,
            k_chunk_size_tokens=tokens_per_frame // 2,
            sparse_frames_enabled=True,
            reference_as_extra_k=True,
        )

    @_MESH_TOPOLOGY_GALAXY
    @pytest.mark.parametrize(
        "tokens_per_frame",
        # nf=6 frames on sp=8 -> 0.75 frames/shard (sub-frame).
        # The reference frame lands on the far shards (6-7 of 8),
        # so peeling it collapses the windowed spatial radius (W 7->4 here).
        [pytest.param(2560, id="tpf2560_3chunks"), pytest.param(5120, id="tpf5120_6chunks")],
    )
    def test_reference_as_extra_k_sub_frame(
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
        tokens_per_frame,
    ):
        """Sub-frame (frames/shard < 1)."""
        _run_sparse_frames_op(
            mesh_device=mesh_device,
            sp_axis=sp_axis,
            sp_factor=sp_factor,
            tp_axis=tp_axis,
            tp_factor=tp_factor,
            num_links=num_links,
            num_frames_real=6,
            num_frames_padded=6,
            tokens_per_frame=tokens_per_frame,
            b=1,
            nh=8,
            d=128,
            window=5,
            add_last_frame=True,
            all_gather_topology=all_gather_topology,
            # 160/160 matches the real 4x32 non-exp chunk sizes (ring_chunk_sizes(fsl, 32)); keeps the
            # QK score buffer (Sq*Sk = 5*5 tiles) small enough for L1, unlike 640/640.
            q_chunk_size_tokens=160,
            k_chunk_size_tokens=160,
            sparse_frames_enabled=True,
            reference_as_extra_k=True,
        )

    @_MESH_TOPOLOGY_GALAXY
    @pytest.mark.parametrize(
        ("tokens_per_frame", "num_frames_real", "num_frames_padded", "q_chunk", "k_chunk", "reference"),
        [
            pytest.param(3840, 21, 22, 320, 320, False, id="sp8native_fsl3840"),
            pytest.param(9600, 5, 6, 160, 160, True, id="sp32shapes_fsl9600"),
        ],
    )
    def test_single_galaxy_mesh_configs(
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
        tokens_per_frame,
        num_frames_real,
        num_frames_padded,
        q_chunk,
        k_chunk,
        reference,
    ):
        _run_sparse_frames_op(
            mesh_device=mesh_device,
            sp_axis=sp_axis,
            sp_factor=sp_factor,
            tp_axis=tp_axis,
            tp_factor=tp_factor,
            num_links=num_links,
            num_frames_real=num_frames_real,
            num_frames_padded=num_frames_padded,
            tokens_per_frame=tokens_per_frame,
            b=1,
            nh=8,
            d=128,
            window=5,
            add_last_frame=True,
            all_gather_topology=all_gather_topology,
            q_chunk_size_tokens=q_chunk,
            k_chunk_size_tokens=k_chunk,
            sparse_frames_enabled=True,
            reference_as_extra_k=reference,
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
        ["tail_drain", "head_drain", "middle_drain", "drain_one_last", "drain_one_first"],
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
