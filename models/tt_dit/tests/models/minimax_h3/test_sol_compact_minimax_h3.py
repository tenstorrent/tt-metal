# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Sol-Attn sparse ring SDPA: mode coverage and regression guards, on the 4x8 BH mesh.

Complements test_sol_ring_sdpa_minimax_h3.py, which takes its mode from $SOL_MODE (default "gather") and compares
against the out-of-repo sol_ref module. Here mode is a pytest parameter so a plain run covers gather/chunk/compact,
and the reference is the dense ring op on the same inputs, so the file is standalone.

Run everything:      pytest models/tt_dit/tests/models/minimax_h3/test_sol_compact_minimax_h3.py
One mode:            ... -k compact
Skip the slow shapes: they are already opt-in via SOL_COMPACT_FULL=1.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import pytest
import torch
from loguru import logger

import ttnn

from ....models.transformers.minimax_h3.sol_prepass import SolGeometry, SolPrepass
from ....parallel.manager import CCLManager
from ....utils.tensor import bf16_tensor, from_torch
from .common import GALAXY_RING

MODES = ["gather", "chunk", "compact"]


def logical_length_tensor(mesh_device: ttnn.MeshDevice, value: int) -> ttnn.Tensor:
    """The `logical_n` device tensor the op takes: [1, 1, 1, 1] uint32 ROW_MAJOR, replicated.

    Copied from test_transformer_minimax_h3 rather than imported: that module pulls in a MiniMax-H3-patched
    `diffusers`, which this file has no other use for and which is not installed everywhere. Keeping the file
    standalone is the point -- its reference is the dense ring op, not a model.
    """
    return from_torch(
        torch.tensor([value], dtype=torch.int64).reshape(1, 1, 1, 1),
        device=mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.Layout.ROW_MAJOR,
        mesh_axes=[..., None, None],
    )


_MODE_LINE = "RingJointSDPA sol mode resolved: "

# Default geometry: small enough to run every parameter in one sitting, large enough that a K chunk holds
# TK = k_chunk / 64 = 8 routing blocks, which is the ratio that makes the sparse path interesting at all.
ROWS, LOGICAL_N, Q_CHUNK, K_CHUNK = 1024, 8192, 256, 512
# Q1 compute unit = the 64-token query block. Q2's staging CBs (~256 KB) only fit L1 at this size: measured
# 2026-10-20, packing at q_chunk=256 overflows by 105 KB (1,680,896 B vs 1,572,864 B). Dropping Sq from 8 to 2
# tiles shrinks every Sq-dependent CB 4x, freeing ~222 tiles. Q2 therefore REQUIRES Q1 -- they ship together.
Q1_CHUNK = 64
H, D = 14, 128
SLOW = pytest.mark.skipif(os.environ.get("SOL_COMPACT_FULL") != "1", reason="set SOL_COMPACT_FULL=1")


@dataclass
class Result:
    dense: torch.Tensor | None  # [H, S*rows, D] from the dense ring op, or None when not requested
    sol: torch.Tensor  # same shape, from the sparse path
    density: float  # fraction of blocks the prepass kept
    modes_logged: set  # what the program factory said it compiled


def _pcc(a, b, logical_n=None):
    """PCC over the LOGICAL rows only.

    Rows at or beyond logical_n are padding: the op leaves them undefined, and dense and sparse leave different
    garbage there. Comparing them made test_ragged_logical_n and test_small_k_chunk read 0.84 when the real
    sequence agreed closely -- at rows=1184 on 8 shards the padded extent is 9472, so a default logical_n of
    8192 makes 13.5 % of the compared rows meaningless.
    """
    if logical_n is not None:
        a, b = a[:, :logical_n], b[:, :logical_n]
    a, b = a.flatten().float(), b.flatten().float()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def _qkv(S, rows, seed=0):
    """Q/K/V with enough structure that attention is not uniform, so a broken mask shows up in the output."""
    g = torch.Generator().manual_seed(seed)
    shape = (S, H, rows, D)
    q = torch.randn(shape, generator=g) * 0.5
    k = torch.randn(shape, generator=g) * 0.5
    v = torch.randn(shape, generator=g)
    pos = torch.linspace(-2.0, 2.0, rows).view(1, 1, rows, 1)
    return q + pos, k + pos, v  # positional ramp -> score mass concentrates near the diagonal


def _drain_mode_log(capfd):
    """Return the set of modes the factory reported, re-emitting captured output so job logs stay complete."""
    out, err = capfd.readouterr()
    with capfd.disabled():
        sys.stdout.write(out)
        sys.stderr.write(err)
    return {l.split(_MODE_LINE, 1)[1].split()[0] for l in (out + err).splitlines() if _MODE_LINE in l}


def _run(
    mesh_device,
    sp_axis,
    num_links,
    topology,
    capfd,
    *,
    mode,
    tau,
    rows=ROWS,
    logical_n=LOGICAL_N,
    q_chunk=Q_CHUNK,
    k_chunk=K_CHUNK,
    want_dense=True,
    seed=0,
    upload_bias=False,
):
    """Prepass + sparse ring SDPA for one (mode, tau), and optionally the dense op on the same inputs.

    This is the only function that touches the device; every test below is a few lines on top of it.
    """
    S = tuple(mesh_device.shape)[sp_axis]
    scale = D**-0.5
    sink_tokens = 300 if rows < 4096 else 849
    geo = SolGeometry(shards=S, rows=rows, logical_n=logical_n, q_chunk=q_chunk, k_chunk=k_chunk)

    ccl = CCLManager(mesh_device=mesh_device, num_links=num_links, topology=topology)
    mesh_axes = [sp_axis, None, None, None]
    q, k, v = _qkv(S, rows, seed)
    tq, tk, tv = (
        from_torch(x, device=mesh_device, dtype=ttnn.bfloat16, mesh_axes=mesh_axes, layout=ttnn.TILE_LAYOUT)
        for x in (q, k, v)
    )

    pre = SolPrepass(mesh_device=mesh_device, ccl_manager=ccl, sp_axis=sp_axis, tau=tau, mode=mode)
    pre.prepare(geo, sink_tokens, scale)
    sol = pre(tq, tk, tv, scale=scale, report_density=True)

    grid = mesh_device.compute_with_storage_grid_size()
    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(grid.x - 1, grid.y),
        q_chunk_size=q_chunk,
        k_chunk_size=k_chunk,
        exp_approx_mode=False,
    )
    ckc = ttnn.init_device_compute_kernel_config(
        mesh_device.arch(), math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=False
    )
    empty_joint = bf16_tensor(torch.zeros((1, H, 0, D)), device=mesh_device)

    def launch(**sol_kwargs):
        out, _, _ = ttnn.transformer.ring_joint_scaled_dot_product_attention(
            tq,
            tk,
            tv,
            empty_joint,
            empty_joint,
            empty_joint,
            persistent_output_buffer_k=ccl.get_ag_ping_pong_buffer(tq.shape, 2, sp_axis, dtype=ttnn.bfloat16),
            persistent_output_buffer_v=ccl.get_ag_ping_pong_buffer(tq.shape, 2, sp_axis, dtype=ttnn.bfloat16),
            joint_strategy="rear",
            logical_n=logical_length_tensor(mesh_device, logical_n),
            program_config=program_config,
            compute_kernel_config=ckc,
            dim=2,
            multi_device_global_semaphore=ccl.get_ag_ping_pong_semaphore(sp_axis),
            num_links=num_links,
            cluster_axis=sp_axis,
            mesh_device=mesh_device,
            topology=topology,
            subdevice_id=ccl.ccl_sub_device_id,
            ccl_core_grid_offset=(grid.x - 1, 0),
            use_column_major_ccl=True,
            **sol_kwargs,
        )
        ttnn.synchronize_device(mesh_device)
        host = ttnn.to_torch(
            out,
            mesh_composer=ttnn.ConcatMesh2dToTensor(
                mesh_device, dims=[1, 2] if sp_axis == 1 else [2, 1], mesh_shape=tuple(mesh_device.shape)
            ),
        )
        return host[0, :H].float()

    dense = launch() if want_dense else None
    capfd.readouterr()  # discard dense-run output so the mode log below is only the sparse launch
    sol_kw = dict(
        sol_block_mask=sol["block_mask"],
        sol_pooled_k=sol["pooled_k"],
        sol_pooled_v=sol["pooled_v"],
        sol_mode=mode,
    )
    if upload_bias and sol.get("bias") is not None:
        # Use the pre-pass's Python-computed bias instead of the op's on-device generator (sol_bias_gen).
        sol_kw["sol_bias"] = sol["bias"]
    sparse = launch(**sol_kw)
    modes = _drain_mode_log(capfd)
    logger.info(f"[sol] mode={mode} tau={tau} density={pre.last_density:.4f} factory_logged={sorted(modes)}")
    return Result(dense=dense, sol=sparse, density=float(pre.last_density), modes_logged=modes)


# --------------------------------------------------------------------------------------------------
# Tests. Each one is a call to _run plus an assertion.
# --------------------------------------------------------------------------------------------------


@GALAXY_RING
@pytest.mark.parametrize("mode", MODES)
def test_mode_reaches_the_kernels(
    mesh_device, sp_axis, tp_axis, num_links, topology, is_fsdp, mode, reset_seeds, capfd
):
    """A requested mode that silently falls back still produces correct output -- this is the only guard against it."""
    r = _run(mesh_device, sp_axis, num_links, topology, capfd, mode=mode, tau=1.0, want_dense=False)
    assert r.modes_logged == {mode}, f"requested {mode!r}, factory compiled {sorted(r.modes_logged)}"


@GALAXY_RING
@pytest.mark.parametrize("mode", MODES)
def test_mask_is_actually_sparse(mesh_device, sp_axis, tp_axis, num_links, topology, is_fsdp, mode, reset_seeds, capfd):
    """Guards every accuracy test below: a mask that keeps everything would pass all of them trivially."""
    r = _run(mesh_device, sp_axis, num_links, topology, capfd, mode=mode, tau=1.0, want_dense=False)
    assert 0.02 < r.density < 0.95, f"density {r.density:.4f} is not a sparse mask"


@GALAXY_RING
@pytest.mark.parametrize("mode", MODES)
def test_dense_equivalence(mesh_device, sp_axis, tp_axis, num_links, topology, is_fsdp, mode, reset_seeds, capfd):
    """tau = -1e4 keeps every block, so the sparse path must reproduce the dense op almost exactly."""
    r = _run(mesh_device, sp_axis, num_links, topology, capfd, mode=mode, tau=-1e4)
    assert r.density > 0.99, f"tau=-1e4 should keep everything, kept {r.density:.4f}"
    assert _pcc(r.dense, r.sol) > 0.9995


@GALAXY_RING
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("tau", [1.0, 1.5])
def test_accuracy_vs_dense(mesh_device, sp_axis, tp_axis, num_links, topology, is_fsdp, mode, tau, reset_seeds, capfd):
    r = _run(mesh_device, sp_axis, num_links, topology, capfd, mode=mode, tau=tau)
    assert _pcc(r.dense, r.sol) > 0.99, f"mode={mode} tau={tau} density={r.density:.4f}"


@GALAXY_RING
def test_modes_agree(mesh_device, sp_axis, tp_axis, num_links, topology, is_fsdp, reset_seeds, capfd):
    """gather/chunk/compact are three implementations of the same math -- they must agree, not merely each be close."""
    outs = {
        m: _run(mesh_device, sp_axis, num_links, topology, capfd, mode=m, tau=1.0, want_dense=False).sol for m in MODES
    }
    for m in MODES[1:]:
        assert _pcc(outs["gather"], outs[m]) > 0.999, f"gather vs {m} disagree"


@GALAXY_RING
@pytest.mark.parametrize("mode", MODES)
def test_determinism(mesh_device, sp_axis, tp_axis, num_links, topology, is_fsdp, mode, reset_seeds, capfd):
    """Same input twice must give the same bits; a race in the chain or the plan shows up here first."""
    kw = dict(mode=mode, tau=1.0, want_dense=False)
    a = _run(mesh_device, sp_axis, num_links, topology, capfd, **kw)
    b = _run(mesh_device, sp_axis, num_links, topology, capfd, **kw)
    assert a.density == b.density
    assert torch.equal(a.sol, b.sol), f"max |diff| {(a.sol - b.sol).abs().max().item():.3e}"


@GALAXY_RING
@pytest.mark.parametrize("mode", MODES)
def test_density_falls_as_tau_rises(
    mesh_device, sp_axis, tp_axis, num_links, topology, is_fsdp, mode, reset_seeds, capfd
):
    """tau is the sparsity knob; if it stops biting, every perf number taken with it is meaningless.

    Non-increasing, not strictly decreasing. Measured 2026-10-20: tau 0.5 and 1.0 give identical density
    (0.5835 gather/compact) because the forcing rules -- band, sink columns, dense-prefix rows, forced block 0 --
    keep a floor of blocks no threshold can drop, and low tau sits on that floor. Chunk mode is flat across the
    whole range (0.8086) because its density is CHUNK occupancy: a chunk survives if any of its 8 blocks does,
    which saturates long before the block mask does. The wide end of the range is where tau has to bite.
    """
    d = [
        _run(mesh_device, sp_axis, num_links, topology, capfd, mode=mode, tau=t, want_dense=False).density
        for t in (0.5, 2.0, 4.0)
    ]
    logger.info(f"[tau] {mode}: densities {d}")
    assert d[0] >= d[1] >= d[2] - 1e-9, f"densities {d} increase with tau"
    if mode != "chunk":
        # Chunk mode is exempt by construction, as the docstring above says: its density is CHUNK occupancy,
        # and a chunk survives if any of its 8 blocks does, so it pins at 0.8086 across the whole tau range.
        # Only the block-form modes expose the knob. Asserting otherwise here was my error, not the kernel's.
        assert d[0] > d[2], f"tau has no effect between 0.5 and 4.0 in {mode} mode: {d}"


PARTIAL_TAIL_XFAIL = pytest.mark.xfail(
    reason=(
        "rows=1184 is not a multiple of the 64-token block, so the last block of every shard is partial and is "
        "pooled rather than computed exactly. Block-form modes (gather, compact) land at PCC ~0.855; chunk mode "
        "passes. Reproduced before any Q1/Q2 change and with packing off, so this is a pre-existing accuracy "
        "question in the sol path at non-block-aligned row counts, not a regression. Left failing deliberately "
        "rather than relaxing the gate -- 2.7 % of keys going through the zeroth-order correction should not "
        "cost 0.14 of PCC, and that discrepancy is worth running down."
    ),
    strict=False,
)


@PARTIAL_TAIL_XFAIL
@GALAXY_RING
@pytest.mark.parametrize("mode", MODES)
def test_ragged_logical_n(mesh_device, sp_axis, tp_axis, num_links, topology, is_fsdp, mode, reset_seeds, capfd):
    """logical_n off a block boundary leaves a partial tail block, which the kernel must pool rather than compute."""
    r = _run(mesh_device, sp_axis, num_links, topology, capfd, mode=mode, tau=1.0, rows=1184, logical_n=9000)
    assert _pcc(r.dense, r.sol, logical_n=9000) > 0.99


@PARTIAL_TAIL_XFAIL
@GALAXY_RING
@pytest.mark.parametrize("mode", MODES)
def test_small_k_chunk(mesh_device, sp_axis, tp_axis, num_links, topology, is_fsdp, mode, reset_seeds, capfd):
    """k_chunk 256 gives TK = 4 blocks per chunk instead of 8, changing every kept-count in the plan.

    logical_n must match the padded extent (8 shards x 1184) or most of the tail is undefined padding.
    """
    r = _run(
        mesh_device, sp_axis, num_links, topology, capfd, mode=mode, tau=1.0, rows=1184, logical_n=9472, k_chunk=256
    )
    assert _pcc(r.dense, r.sol, logical_n=9472) > 0.99


@GALAXY_RING
@pytest.mark.parametrize("mode", ["gather", "compact"])
def test_qblock_granularity(mesh_device, sp_axis, tp_axis, num_links, topology, is_fsdp, mode, reset_seeds, capfd):
    """Q1 without a kernel change: q_chunk = 64 makes the Q chunk BE the query block.

    Sol-Attn (arXiv 2607.24027 Alg. 1) selects per query block, but our route program ORs TQ = q_chunk/64 query
    blocks together because the op's compute unit is a Q chunk, so modes 0-2 compute a strict superset of S_i.
    At q_chunk = 64, TQ = 1 and `qc_pool_t[b, (b*64)//64] = qc_pool_t[b,b]` is the identity -- the OR is over one
    element, block01 == exact, and the emitted mask is already the spec-conformant per-query-block mask. nqc_pad
    collapses onto nb_pad too, so the op's existing `sol_nqc_pad` addressing still works.

    Feasible only on the plain ring-joint path: it spreads B*NH*num_q_chunks over the grid with div_up
    (ring_joint_sdpa_program_factory.cpp:1403). The exp path binds Q chunk x to core column x and could not do this.
    """
    r = _run(mesh_device, sp_axis, num_links, topology, capfd, mode=mode, tau=1.0, q_chunk=64)
    logger.info(f"[q1] q_chunk=64 ({mode}): density {r.density:.4f}, pcc {_pcc(r.dense, r.sol):.5f}")
    assert 0.02 < r.density < 0.95
    assert _pcc(r.dense, r.sol) > 0.99


@GALAXY_RING
def test_qblock_is_sparser_than_qchunk(mesh_device, sp_axis, tp_axis, num_links, topology, is_fsdp, reset_seeds, capfd):
    """The OR's cost, measured: same tau, same inputs, only the Q granularity differs.

    q_chunk=256 ORs 4 query blocks per chunk, so its density is the union of 4 rows and must exceed the
    per-query-block density. The ratio is what Q1 recovers in math -- and it is the number the ~3.4x estimate
    in sol_attn_next_optimizations.md was inferred from rather than measured.
    """
    kw = dict(mode="compact", tau=1.0, want_dense=False)
    fine = _run(mesh_device, sp_axis, num_links, topology, capfd, q_chunk=64, **kw).density
    coarse = _run(mesh_device, sp_axis, num_links, topology, capfd, q_chunk=256, **kw).density
    logger.info(f"[q1] density: q_chunk=64 {fine:.4f}  q_chunk=256 {coarse:.4f}  ratio {coarse / max(fine, 1e-9):.2f}x")
    assert coarse >= fine - 1e-6, "the OR over query blocks cannot reduce density"


def _pack_env(on):
    """TT_SOL_PACK gates Q2 in the program factory; it is read at program creation, so set it before the run."""
    prev = os.environ.get("TT_SOL_PACK")
    os.environ["TT_SOL_PACK"] = "1" if on else "0"
    return prev


def _pack_env_restore(prev):
    if prev is None:
        os.environ.pop("TT_SOL_PACK", None)
    else:
        os.environ["TT_SOL_PACK"] = prev


@GALAXY_RING
def test_q2_packing_matches_unpacked(mesh_device, sp_axis, tp_axis, num_links, topology, is_fsdp, reset_seeds, capfd):
    """Q2's correctness gate: packing regroups blocks into full pages and must not change the math.

    Same inputs, same mask, same tau -- only the delivery unit differs (kept chunks vs packed pages). Any
    disagreement means the page accounting diverged between reader and compute: the two derive page boundaries
    independently from the kept bitmask, so a mismatch shows up as wrong output or a hang rather than a warning.

    This is also the first run that compiles the sol_pack branches of compute_streaming.hpp. Those live behind
    `if constexpr (sol_pack)` with sol_pack a TEMPLATE parameter, so they are not type-checked at all until a
    program is created with packing on -- unlike the reader's, where sol_pack is a plain constexpr bool and the
    discarded branch is still checked.
    """
    kw = dict(mode="compact", tau=1.0, want_dense=False, q_chunk=Q1_CHUNK)
    prev = _pack_env(False)
    try:
        base = _run(mesh_device, sp_axis, num_links, topology, capfd, **kw)
        # TT_SOL_PACK is read in the program factory, which a cache hit never calls: without this the second
        # run silently reuses the pack-off program and the comparison is vacuous. That is exactly what happened
        # on 2026-10-20 -- this test passed while every other packed test OOM'd at the same geometry.
        mesh_device.disable_and_clear_program_cache()
        _pack_env(True)
        packed = _run(mesh_device, sp_axis, num_links, topology, capfd, **kw)
    finally:
        _pack_env_restore(prev)
        mesh_device.disable_and_clear_program_cache()
    logger.info(f"[q2] density {base.density:.4f}; pcc(packed, unpacked) {_pcc(base.sol, packed.sol):.6f}")
    assert base.density == packed.density, "packing changed the mask; it must only change delivery"
    assert _pcc(base.sol, packed.sol) > 0.9999


@GALAXY_RING
@pytest.mark.parametrize("tau", [0.5, 1.0, 2.0])
def test_q2_packing_across_densities(
    mesh_device, sp_axis, tp_axis, num_links, topology, is_fsdp, tau, reset_seeds, capfd
):
    """Sweeps the page-fill pattern. tau sets how many blocks are kept, which sets how chunks straddle page
    boundaries and how big the trailing partial page is -- the two cases most likely to desynchronise the
    reader's `push at sol_blocks_per_page` from compute's `ceil(total / sol_blocks_per_page)`.
    """
    prev = _pack_env(True)
    try:
        mesh_device.disable_and_clear_program_cache()
        r = _run(mesh_device, sp_axis, num_links, topology, capfd, mode="compact", tau=tau, q_chunk=Q1_CHUNK)
    finally:
        _pack_env_restore(prev)
        mesh_device.disable_and_clear_program_cache()
    logger.info(f"[q2] tau={tau} density {r.density:.4f} pcc vs dense {_pcc(r.dense, r.sol):.5f}")
    assert _pcc(r.dense, r.sol) > 0.99


@GALAXY_RING
def test_q2_zero_kept_iteration(mesh_device, sp_axis, tp_axis, num_links, topology, is_fsdp, reset_seeds, capfd):
    """The degenerate page case: a very high tau leaves ring iterations with no kept blocks at all.

    Then no page is ever filled, the partial flush has nothing to push, and compute's page count is 0 for that
    iteration -- it must fall straight through to the pooled chunks. This is the path that deadlocks if the
    reader's trailing un-pushed reserve and compute's loop bound disagree.
    """
    prev = _pack_env(True)
    try:
        mesh_device.disable_and_clear_program_cache()
        r = _run(mesh_device, sp_axis, num_links, topology, capfd, mode="compact", tau=6.0, q_chunk=Q1_CHUNK)
    finally:
        _pack_env_restore(prev)
        mesh_device.disable_and_clear_program_cache()
    logger.info(f"[q2] tau=6 density {r.density:.4f} pcc vs dense {_pcc(r.dense, r.sol):.5f}")
    assert torch.isfinite(r.sol).all(), "non-finite output: the pooled correction did not carry a near-empty mask"


@SLOW
@GALAXY_RING
@pytest.mark.parametrize("mode", MODES)
def test_production_5s_geometry(mesh_device, sp_axis, tp_axis, num_links, topology, is_fsdp, mode, reset_seeds, capfd):
    """The 5 s shape: 4736 rows/chip, q_chunk 320 (TQ = 5 Q blocks), k_chunk 384 (TK = 6 K blocks)."""
    r = _run(
        mesh_device,
        sp_axis,
        num_links,
        topology,
        capfd,
        mode=mode,
        tau=1.0,
        rows=4736,
        logical_n=37800,
        q_chunk=320,
        k_chunk=384,
    )
    assert _pcc(r.dense, r.sol) > 0.99


@GALAXY_RING
@pytest.mark.parametrize("mode", ["chunk", "compact"])
@pytest.mark.parametrize(
    "rows,what",
    [
        (1024, "neither"),  # 16 full blocks, 4 full Q chunks
        (1152, "partial Q chunk"),  # 18 full blocks, 1152/256 = 4.5 -> last Q chunk holds 128 rows
        (1184, "both"),  # 18 full + 32-token block, and a 160-row last Q chunk
    ],
)
def test_ragged_bisect(
    mesh_device, sp_axis, tp_axis, num_links, topology, is_fsdp, mode, rows, what, reset_seeds, capfd
):
    """Which raggedness breaks the block-form path: the partial K block, or the partial Q chunk?

    Device (packing off, pre-existing): rows=1184 gives PCC ~0.855 for gather/compact while chunk mode passes.
    The torch reference says pooling a partial K block costs only 0.007 PCC (0.98134 -> 0.97446), so that is not
    it. But rows=1184 at q_chunk=256 also leaves 160 of 1184 rows in a partial final Q chunk -- 13.5 % of the
    output, which matches a PCC of 0.855 almost exactly.

    1024 isolates neither, 1152 isolates the partial Q chunk alone (all K blocks full), 1184 has both.
      1152 fails  -> the partial Q chunk is the trigger, and production is exposed: 4736/320, 9184/256 and
                     13664/256 all leave a partial final Q chunk.
      1152 passes -> the partial K block is the trigger on device, despite the reference tolerating it.
    """
    r = _run(
        mesh_device,
        sp_axis,
        num_links,
        topology,
        capfd,
        mode=mode,
        tau=1.0,
        rows=rows,
        logical_n=8 * rows,
    )
    pcc = _pcc(r.dense, r.sol, logical_n=8 * rows)
    logger.info(f"[ragged] mode={mode} rows={rows} ({what}) pcc={pcc:.5f}")
    assert pcc > 0.99, f"{mode} rows={rows} ({what}): pcc {pcc:.5f}"


@GALAXY_RING
@pytest.mark.parametrize("upload_bias", [False, True], ids=["gen_bias", "uploaded_bias"])
def test_partial_block_bias_source(
    mesh_device, sp_axis, tp_axis, num_links, topology, is_fsdp, upload_bias, reset_seeds, capfd
):
    """Splits the rows=1184 accuracy bug: is it the on-device bias generator, or something else?

    Block-form modes lose 0.145 PCC when the tail K block is partial (32 of 64 tokens). gather fails too, and
    gather does no compaction, so the fault is in the bias/pooled path the two share -- not in packing. Reading
    sol_bias_gen found nothing: block_len returns 32 correctly, the table holds ln(len)/scale, full_row zeroes
    the mask bit, and the pooled K-bar/V-bar use 1/len.

    Run in GATHER, not compact: ring_joint_sdpa_device_operation.cpp:576 asserts
    `!sol_compact_mode || sol_bias_generated()`, so compact mode cannot accept an uploaded bias at all -- it has
    no escape hatch from sol_bias_gen. Gather fails identically (0.855) and does allow one, so it is the probe.

    gen_bias       -> the op generates the bias on device (compact's only option, and gather's default here).
    uploaded_bias  -> the op consumes the pre-pass's Python bias instead.

    uploaded_bias passes and gen_bias fails -> the defect is in sol_bias_gen.
    both fail                               -> the bias is fine; look at the pooled chunk or the mask instead.
    """
    r = _run(
        mesh_device,
        sp_axis,
        num_links,
        topology,
        capfd,
        mode="gather",
        tau=1.0,
        rows=1184,
        logical_n=8 * 1184,
        upload_bias=upload_bias,
    )
    pcc = _pcc(r.dense, r.sol, logical_n=8 * 1184)
    logger.info(f"[bias] upload_bias={upload_bias} rows=1184 pcc={pcc:.5f}")
    assert pcc > 0.99, f"upload_bias={upload_bias}: pcc {pcc:.5f}"


@GALAXY_RING
@pytest.mark.parametrize(
    "rows,tau,what",
    [
        (1216, 1.0, "19 FULL blocks, no partial"),      # 1216 = 19*64 exactly
        (1184, -1e4, "partial block is the ONLY approximation"),
        (1184, 1.0, "partial block + normal sparsity"),  # the known failure, as control
    ],
)
def test_partial_block_isolate(
    mesh_device, sp_axis, tp_axis, num_links, topology, is_fsdp, rows, tau, what, reset_seeds, capfd
):
    """Narrows the rows=1184 bug after bias was exonerated (both bias sources gave bit-identical 0.85523).

    rows=1216 -- 19 full blocks, so nb is still odd and the padded tile count is still 38, but NOTHING is
      partial. Fails => the trigger is nb/geometry, not partialness. Passes => partialness it is.

    rows=1184 at tau=-1e4 -- every FULL block is exact, so the partial tail block is the only thing the pooled
      correction approximates: 32 of 1184 tokens, 2.7 %. The reference puts that at 0.007 PCC. If this still
      reads 0.855 then the partial block's pooled contribution is badly wrong on its own, which is as narrow as
      this gets without a kernel dump.
    """
    r = _run(
        mesh_device,
        sp_axis,
        num_links,
        topology,
        capfd,
        mode="compact",
        tau=tau,
        rows=rows,
        logical_n=8 * rows,
    )
    pcc = _pcc(r.dense, r.sol, logical_n=8 * rows)
    logger.info(f"[isolate] rows={rows} tau={tau} ({what}) density={r.density:.4f} pcc={pcc:.5f}")
    assert pcc > 0.99, f"rows={rows} tau={tau} ({what}): pcc {pcc:.5f}"


@GALAXY_RING
def test_partial_block_is_dropped_not_pooled(
    mesh_device, sp_axis, tp_axis, num_links, topology, is_fsdp, reset_seeds, capfd
):
    """Is the partial tail block being DROPPED entirely rather than pooled?

    rows=1184 loses 0.145 PCC from one 32-token block per shard, and the loss is identical at tau=-1e4 (where
    that block is the only approximation in the whole op) and tau=1.0. 2.7 % of keys cannot cost that much if
    they are merely approximated -- the reference prices it at 0.007.

    So compare the device output against a dense reference computed WITHOUT those tokens: K/V truncated to the
    1152 full-block tokens per shard, queries unchanged. If sol matches that truncated reference closely, the
    partial block contributes nothing at all -- excluded from the exact path by full_row and never picked up by
    the pooled correction either. That is a dropped-keys bug, not an approximation-quality one.
    """
    rows, full_rows = 1184, 1152  # 18 full 64-token blocks = 1152; the 19th holds 32
    r = _run(
        mesh_device,
        sp_axis,
        num_links,
        topology,
        capfd,
        mode="compact",
        tau=-1e4,
        rows=rows,
        logical_n=8 * rows,
    )
    S = tuple(mesh_device.shape)[sp_axis]
    q, k, v = _qkv(S, rows)
    kt = torch.cat([k[s, :, :full_rows] for s in range(S)], dim=1).float()  # tail tokens removed
    vt = torch.cat([v[s, :, :full_rows] for s in range(S)], dim=1).float()
    trunc = torch.cat(
        [torch.softmax(q[s].float() @ kt.transpose(1, 2) * (D**-0.5), dim=-1) @ vt for s in range(S)], dim=1
    )
    pcc_full = _pcc(r.dense, r.sol, logical_n=8 * rows)
    pcc_trunc = _pcc(trunc, r.sol, logical_n=8 * rows)
    logger.info(f"[drop] sol vs dense={pcc_full:.5f}   sol vs dense-without-tail-tokens={pcc_trunc:.5f}")
    assert pcc_trunc < 0.999, (
        f"sol matches the truncated reference at {pcc_trunc:.5f}: the partial block is being dropped entirely, "
        f"not pooled"
    )
