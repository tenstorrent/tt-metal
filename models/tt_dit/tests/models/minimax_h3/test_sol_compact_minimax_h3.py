# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Sol-Attn sparse ring SDPA: mode coverage and regression guards, on the 4x8 BH mesh.

Complements test_sol_ring_sdpa_minimax_h3.py, which takes its mode from $SOL_MODE (default "gather") and compares
against the out-of-repo sol_ref module. Here mode is a pytest parameter so a plain run covers gather/chunk/compact,
and the reference is the dense ring op on the same inputs, so the file is standalone.

Run everything:      pytest models/tt_dit/tests/models/minimax_h3/test_sol_compact_minimax_h3.py
One mode:            ... -k compact
Slow shapes:         opt-in via SOL_COMPACT_FULL=1.
"""

from __future__ import annotations

import math
import os
import sys
import time
from dataclasses import dataclass

import pytest
import torch
from loguru import logger

import ttnn

from ....models.transformers.minimax_h3.sol_prepass import SolGeometry, SolPrepass
from ....parallel.manager import CCLManager
from ....utils.tensor import bf16_tensor, from_torch
from .common import GALAXY_RING

MODES = ["gather", "chunk", "compact", "resident"]  # resident: chunk semantics, K/V-once loop order


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
# The 64-token query block is the compute unit of the pack and resident modes. Packing's staging CBs (~256 KB) only
# fit L1 at this chunk size: at q_chunk=256 they overflow, and dropping Sq from 8 to 2 tiles shrinks every
# Sq-dependent CB 4x. Pack mode therefore requires the 64-row Q chunk.
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
    garbage there. Including them drags the PCC down even when the real sequence agrees closely: at rows=1184 on
    8 shards the padded extent is 9472, so a logical_n of 8192 makes 13.5 % of the compared rows meaningless.
    """
    if logical_n is not None:
        a, b = a[:, :logical_n], b[:, :logical_n]
    a, b = a.flatten().float(), b.flatten().float()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def _qkv(S, rows, seed=0, ramp=True):
    """Q/K/V with enough structure that attention is not uniform, so a broken mask shows up in the output.

    ramp=False gives pure randn: the routing scores are noise, so the mask is unstructured and its density is
    P(z > tau) -- ~0.16 at tau 1.0, ~0.10 at 1.3. That is the worst case for the Q-OR (no two Q blocks agree)
    and the one the timing test has to clear.
    """
    g = torch.Generator().manual_seed(seed)
    shape = (S, H, rows, D)
    q = torch.randn(shape, generator=g) * 0.5
    k = torch.randn(shape, generator=g) * 0.5
    v = torch.randn(shape, generator=g)
    if not ramp:
        return q, k, v
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
    if mode == "resident" and q_chunk == Q_CHUNK:
        # Resident mode's compute unit is the 64-row query block (SOL_RESIDENT_Q_CHUNK overrides it, e.g. 128), and
        # its L1 FIFO of per-chunk accumulators does not fit next to a 256-row chunk's CBs.
        q_chunk = int(os.environ.get("SOL_RESIDENT_Q_CHUNK", str(Q1_CHUNK)))
    geo = SolGeometry(shards=S, rows=rows, logical_n=logical_n, q_chunk=q_chunk, k_chunk=k_chunk)

    ccl = CCLManager(mesh_device=mesh_device, num_links=num_links, topology=topology)
    mesh_axes = [sp_axis, None, None, None]
    q, k, v = _qkv(S, rows, seed)
    tq, tk, tv = (
        from_torch(x, device=mesh_device, dtype=ttnn.bfloat16, mesh_axes=mesh_axes, layout=ttnn.TILE_LAYOUT)
        for x in (q, k, v)
    )

    # "resident" takes the block-form mask with the bias generated on device, exactly like "compact", plus the
    # static per-row logit bound.
    pre = SolPrepass(
        mesh_device=mesh_device,
        ccl_manager=ccl,
        sp_axis=sp_axis,
        tau=tau,
        mode="compact" if mode == "resident" else mode,
        row_max=False,  # the resident op freezes its max on device; the a-priori bound is unused
    )
    pre.prepare(geo, sink_tokens, scale)
    sol = pre(tq, tk, tv, scale=scale, report_density=True)

    grid = mesh_device.compute_with_storage_grid_size()
    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(grid.x - 1, grid.y),
        q_chunk_size=q_chunk,
        k_chunk_size=k_chunk,
        exp_approx_mode=os.environ.get("SOL_EXP_APPROX", "0") == "1",  # 1: use the approximate SFPU exp
    )
    ckc = ttnn.init_device_compute_kernel_config(
        mesh_device.arch(), math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=False
    )
    empty_joint = bf16_tensor(torch.zeros((1, H, 0, D)), device=mesh_device)

    gathered = {}  # filled by launch(): the K persistent buffer after the op, per device, for the AG slot check

    def launch(**sol_kwargs):
        pk = ccl.get_ag_ping_pong_buffer(tq.shape, 2, sp_axis, dtype=ttnn.bfloat16)
        pv = ccl.get_ag_ping_pong_buffer(tq.shape, 2, sp_axis, dtype=ttnn.bfloat16)
        out, _, _ = ttnn.transformer.ring_joint_scaled_dot_product_attention(
            tq,
            tk,
            tv,
            empty_joint,
            empty_joint,
            empty_joint,
            persistent_output_buffer_k=pk,
            persistent_output_buffer_v=pv,
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
        # Device d on dim 0, its TP heads on dim 1: gathered[d, :H, slot s rows, :] should hold K shard s.
        gathered["k"] = ttnn.to_torch(
            pk,
            mesh_composer=ttnn.ConcatMesh2dToTensor(
                mesh_device, dims=[1, 0] if sp_axis == 1 else [0, 1], mesh_shape=tuple(mesh_device.shape)
            ),
        )[:, :H].float()
        return host[0, :H].float()

    dense = launch() if want_dense else None
    ag_line = None
    if want_dense:
        # All-gather slot check on the dense launch: which torch K shard does each device's slot hold? The
        # local slot (s == d) is never written by the fused all-gather and shows whatever was there.
        gk = gathered["k"]  # [S, H, S*rows, D]
        lines = []
        for d in range(S):
            best = []
            for sl in range(S):
                blk = gk[d, :, sl * rows : (sl + 1) * rows]
                pccs = [_pcc(blk, k[s]) for s in range(S)]
                m = max(range(S), key=lambda i: pccs[i])
                best.append(f"{m if pccs[m] > 0.99 else '?'}")
            lines.append(f"d{d}:[{','.join(best)}]")
        ag_line = "[ag] gathered-K slot -> torch shard per device (expect slot s = shard s, '?' at s = d): " + " ".join(
            lines
        )
    capfd.readouterr()  # discard dense-run output so the mode log below is only the sparse launch
    if ag_line is not None:
        logger.info(ag_line)  # after the discard, or capfd swallows it
    sol_kw = dict(
        sol_block_mask=sol["block_mask"],
        sol_pooled_k=sol["pooled_k"],
        sol_pooled_v=sol["pooled_v"],
        sol_mode=mode,
    )
    if mode == "resident" and sol.get("row_max") is not None:
        sol_kw["sol_row_max"] = sol["row_max"]
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
    # Resident runs at 64-row chunks with the chunk-form mask, a different approximation of the same mask, so it is
    # compared against chunk mode in test_resident_matches_chunk instead.
    modes = [m for m in MODES if m != "resident"]
    outs = {
        m: _run(mesh_device, sp_axis, num_links, topology, capfd, mode=m, tau=1.0, want_dense=False).sol for m in modes
    }
    for m in modes[1:]:
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

    Non-increasing, not strictly decreasing. Low tau values can give identical density because the forcing rules
    -- band, sink columns, dense-prefix rows, forced block 0 -- keep a floor of blocks no threshold can drop, and
    low tau sits on that floor. Chunk mode is flat across the whole range because its density is CHUNK occupancy:
    a chunk survives if any of its 8 blocks does, which saturates long before the block mask does. The wide end
    of the range is where tau has to bite.
    """
    d = [
        _run(mesh_device, sp_axis, num_links, topology, capfd, mode=mode, tau=t, want_dense=False).density
        for t in (0.5, 2.0, 4.0)
    ]
    logger.info(f"[tau] {mode}: densities {d}")
    assert d[0] >= d[1] >= d[2] - 1e-9, f"densities {d} increase with tau"
    if mode != "chunk":
        # Chunk mode is exempt by construction, as the docstring above says: its density is CHUNK occupancy,
        # and a chunk survives if any of its 8 blocks does, so it stays flat across the whole tau range.
        # Only the block-form modes expose the knob.
        assert d[0] > d[2], f"tau has no effect between 0.5 and 4.0 in {mode} mode: {d}"


def _pooled_tail_reference(S, rows, logical_n, seed=0, pool_partial=True):
    """fp32 attention that mirrors the kernel's treatment of a partial tail block.

    Every full 64-token block inside logical_n is attended exactly. Every partial block (0 < len < 64, from a
    shard end or from logical_n) is replaced by the kernel's zeroth-order column: key = mean K, value = mean V,
    logit bias ln(len), so its softmax weight is len * exp(q . kbar). Keys at/after logical_n are dropped.
    Returns [H, S*rows, D]; query rows at/after logical_n are zero and must be excluded from the comparison.
    pool_partial=False attends every valid token exactly (an fp32 dense reference on the same inputs).

    This is what block-form modes compute at tau = -1e4, where nothing else is approximated. On this file's ramp
    data the tail block carries most of the softmax mass for the tail queries, so pooling it alone costs a large
    PCC drop against dense. That is the correction working as designed on adversarial data, not a kernel defect;
    this reference is the right yardstick.

    The device's remaining distance from this reference comes from bf16 storage of QK^T and P and the bf16 bias
    add; with one column carrying most of the mass every rounding on it shows, where in production the pooled
    columns are the low-weight ones. The gate is 0.98; the pinning test logs the device's own bf16 floor next to it.
    """
    q, k, v = _qkv(S, rows, seed)
    Hh = q.shape[1]
    scale = D**-0.5
    qf = q.permute(1, 0, 2, 3).reshape(Hh, S * rows, D).float()
    kf = k.permute(1, 0, 2, 3).reshape(Hh, S * rows, D).float()
    vf = v.permute(1, 0, 2, 3).reshape(Hh, S * rows, D).float()
    block = 64
    nb = -(-rows // block)
    keep = torch.zeros(S * rows, dtype=torch.bool)
    pooled_k, pooled_v, pooled_bias = [], [], []
    for s in range(S):
        for b in range(nb):
            start = s * rows + b * block
            end = min(start + block, s * rows + rows, logical_n)
            length = end - start
            if length <= 0:
                continue
            if length == block or not pool_partial:
                keep[start:end] = True
            else:
                pooled_k.append(kf[:, start:end].mean(1))
                pooled_v.append(vf[:, start:end].mean(1))
                pooled_bias.append(float(torch.log(torch.tensor(float(length)))))
    k_all = kf[:, keep]
    v_all = vf[:, keep]
    bias = torch.zeros(int(keep.sum()))
    if pooled_k:
        k_all = torch.cat([k_all, torch.stack(pooled_k, 1)], 1)
        v_all = torch.cat([v_all, torch.stack(pooled_v, 1)], 1)
        bias = torch.cat([bias, torch.tensor(pooled_bias)])
    out = torch.zeros(Hh, S * rows, D)
    for h in range(Hh):
        logits = (qf[h, :logical_n] @ k_all[h].T) * scale + bias[None, :]
        out[h, :logical_n] = torch.softmax(logits, dim=-1) @ v_all[h]
    return out


def _check_partial_tail(r: Result, mode, S, rows, logical_n):
    """Block-form modes must match the pooled-tail reference; chunk mode computes the tail chunk densely and must
    match dense. Both are logged against both yardsticks so a regression in either direction is visible."""
    ref = _pooled_tail_reference(S, rows, logical_n)
    pcc_ref = _pcc(r.sol, ref, logical_n=logical_n)
    pcc_dense = _pcc(r.dense, r.sol, logical_n=logical_n)
    logger.info(
        f"[tail] mode={mode} rows={rows} logical_n={logical_n}: pcc vs pooled-tail ref {pcc_ref:.5f}, vs dense {pcc_dense:.5f}"
    )
    if mode == "chunk":
        assert pcc_dense > 0.999, f"{mode} mode computes the tail chunk exactly; pcc vs dense {pcc_dense:.5f}"
    else:
        # Block-form exact sets (gather, compact, resident): partial tail blocks are pooled.
        assert pcc_ref > 0.98, f"{mode}: pcc vs pooled-tail reference {pcc_ref:.5f}"


@GALAXY_RING
@pytest.mark.parametrize("mode", MODES)
def test_ragged_logical_n(mesh_device, sp_axis, tp_axis, num_links, topology, is_fsdp, mode, reset_seeds, capfd):
    """logical_n off a block boundary leaves partial tail blocks (32 tokens on shards 0-6, 8 tokens on shard 7),
    which block-form modes pool rather than compute. tau = -1e4 makes that the only approximation."""
    S = tuple(mesh_device.shape)[sp_axis]
    r = _run(mesh_device, sp_axis, num_links, topology, capfd, mode=mode, tau=-1e4, rows=1184, logical_n=9000)
    _check_partial_tail(r, mode, S, 1184, 9000)


@GALAXY_RING
@pytest.mark.parametrize("mode", MODES)
def test_small_k_chunk(mesh_device, sp_axis, tp_axis, num_links, topology, is_fsdp, mode, reset_seeds, capfd):
    """k_chunk 256 gives TK = 4 blocks per chunk instead of 8, changing every kept-count in the plan, with a
    32-token partial tail block on every shard (logical_n = 8 x 1184)."""
    S = tuple(mesh_device.shape)[sp_axis]
    r = _run(
        mesh_device, sp_axis, num_links, topology, capfd, mode=mode, tau=-1e4, rows=1184, logical_n=9472, k_chunk=256
    )
    _check_partial_tail(r, mode, S, 1184, 9472)


@GALAXY_RING
@pytest.mark.parametrize("mode", ["gather", "compact"])
def test_qblock_granularity(mesh_device, sp_axis, tp_axis, num_links, topology, is_fsdp, mode, reset_seeds, capfd):
    """q_chunk = 64 makes the Q chunk BE the query block, with no kernel change.

    Sol-Attn (arXiv 2607.24027 Alg. 1) selects per query block, but the route program ORs TQ = q_chunk/64 query
    blocks together because the op's compute unit is a Q chunk, so modes 0-2 compute a strict superset of S_i.
    At q_chunk = 64, TQ = 1 and `qc_pool_t[b, (b*64)//64] = qc_pool_t[b,b]` is the identity -- the OR is over one
    element, block01 == exact, and the emitted mask is already the spec-conformant per-query-block mask. nqc_pad
    collapses onto nb_pad too, so the op's existing `sol_nqc_pad` addressing still works.

    Feasible only on the plain ring-joint path, which spreads B*NH*num_q_chunks over the grid with div_up
    (ring_joint_sdpa_program_factory.cpp). The exp path binds Q chunk x to core column x and could not do this.
    """
    r = _run(mesh_device, sp_axis, num_links, topology, capfd, mode=mode, tau=1.0, q_chunk=64)
    logger.info(f"[q1] q_chunk=64 ({mode}): density {r.density:.4f}, pcc {_pcc(r.dense, r.sol):.5f}")
    assert 0.02 < r.density < 0.95
    assert _pcc(r.dense, r.sol) > 0.99


@GALAXY_RING
def test_qblock_is_sparser_than_qchunk(mesh_device, sp_axis, tp_axis, num_links, topology, is_fsdp, reset_seeds, capfd):
    """The OR's cost: same tau, same inputs, only the Q granularity differs.

    q_chunk=256 ORs 4 query blocks per chunk, so its density is the union of 4 rows and must exceed the
    per-query-block density. The ratio is the math that the 64-row query block recovers.
    """
    kw = dict(mode="compact", tau=1.0, want_dense=False)
    fine = _run(mesh_device, sp_axis, num_links, topology, capfd, q_chunk=64, **kw).density
    coarse = _run(mesh_device, sp_axis, num_links, topology, capfd, q_chunk=256, **kw).density
    logger.info(f"[q1] density: q_chunk=64 {fine:.4f}  q_chunk=256 {coarse:.4f}  ratio {coarse / max(fine, 1e-9):.2f}x")
    assert coarse >= fine - 1e-6, "the OR over query blocks cannot reduce density"


@GALAXY_RING
def test_q2_packing_matches_unpacked(mesh_device, sp_axis, tp_axis, num_links, topology, is_fsdp, reset_seeds, capfd):
    """Packing's correctness gate: packing regroups blocks into full pages and must not change the math.

    Same inputs, same mask, same tau -- only the delivery unit differs (kept chunks vs packed pages). Packing is
    selected with sol_mode="pack" (compact plus packing), which is part of the program hash, so the two runs
    below compile and cache separate programs without clearing the cache. The pre-pass output for "pack" is
    identical to "compact", so the densities must match exactly.
    """
    kw = dict(tau=1.0, want_dense=False, q_chunk=Q1_CHUNK)
    base = _run(mesh_device, sp_axis, num_links, topology, capfd, mode="compact", **kw)
    packed = _run(mesh_device, sp_axis, num_links, topology, capfd, mode="pack", **kw)
    logger.info(f"[q2] density {base.density:.4f}; pcc(packed, unpacked) {_pcc(base.sol, packed.sol):.6f}")
    assert packed.modes_logged == {"pack"}, f"factory compiled {sorted(packed.modes_logged)}"
    assert base.density == packed.density, "packing changed the mask; it must only change delivery"
    assert _pcc(base.sol, packed.sol) > 0.9999


@GALAXY_RING
@pytest.mark.parametrize("tau", [-1e4, 2.0], ids=["keepall", "tau2"])
@pytest.mark.parametrize(
    "rows,logical_n,k_chunk",
    [(512, 4096, 512), (1024, 8192, 512), (1024, 8192, 256), (1184, 9472, 512), (1184, 9472, 256)],
    ids=["r512k512", "r1024k512", "r1024k256", "r1184k512", "r1184k256"],
)
def test_resident_matches_compact(
    mesh_device, sp_axis, tp_axis, num_links, topology, is_fsdp, rows, logical_n, k_chunk, tau, reset_seeds, capfd
):
    """The correctness gate of resident mode: the K/V-once loop order -- each piece streamed once per core and
    applied to every resident 64-row query chunk from L1 -- with the per-64-token-block exact set of compact mode:
    a chunk multiplies only its kept blocks of a piece, in place.

    Same inputs and the same block-form mask at 64-row query chunks as compact mode (even-up off, so compact's
    exact set is the mask's full blocks plus the forced block 0 of every shard, exactly resident's); only the loop
    order, the block addressing and where the accumulators live differ, so the two outputs must agree to bf16
    noise. tau=-1e4 keeps every block (full-width steps, no state rotation), tau 2 narrows most steps and rotates
    many pieces. rows 1184 leaves a partial last K chunk and partial Q chunks; k_chunk 256 halves the piece
    (Sk_chunk_t 8). At rows=1024 a core holds about two resident chunks and some cores straddle two heads, which
    exercises the interleaved per-head streaming.
    """
    kw = dict(
        tau=tau,
        q_chunk=int(os.environ.get("SOL_RESIDENT_Q_CHUNK", str(Q1_CHUNK))),
        rows=rows,
        logical_n=logical_n,
        k_chunk=k_chunk,
    )
    base = _run(mesh_device, sp_axis, num_links, topology, capfd, mode="compact", **kw)
    res = _run(mesh_device, sp_axis, num_links, topology, capfd, mode="resident", want_dense=False, **kw)
    p_chunk = _pcc(base.sol, res.sol, logical_n)
    p_dense = _pcc(res.sol, base.dense, logical_n)
    p_chunk_dense = _pcc(base.sol, base.dense, logical_n)
    logger.info(
        f"[resident] rows={rows} k_chunk={k_chunk} tau={tau} block density {res.density:.4f}; "
        f"pcc(resident, compact) {p_chunk:.6f}; pcc(resident, dense) {p_dense:.5f}; "
        f"pcc(compact, dense) {p_chunk_dense:.5f}"
    )
    # Where does it go wrong? PCC per (shard, 64-row query chunk) for each pair, worst first: a bad chunk position
    # (first / last per core, the partial tail, one head) points at the loop structure; uniform noise at the step;
    # the same regions in compact-vs-dense point at the shared q_chunk=64 path rather than at resident mode.
    S = tuple(mesh_device.shape)[sp_axis]
    nqc = -(-rows // Q1_CHUNK)
    outs = {"dense": base.dense, "compact": base.sol, "resident": res.sol}
    for name, t in outs.items():
        per_shard = []
        for sh in range(S):
            blk = t[:, sh * rows : min(sh * rows + rows, logical_n)]
            per_shard.append(f"{int((~torch.isfinite(blk)).sum())}nan/{float(blk.float().std()):.3g}sd")
        logger.info(f"[resident] {name}: per-shard nonfinite count / std: {' '.join(per_shard)}")
    # The factory's flat split: chunk-heads [h * nqc + qc] over 120 cores, the first (total % 120) cores one extra.
    # A head's cores form its chain in core order; position = index among them, slot = index within the core.
    # The op's worker grid is the chip's compute grid minus the CCL column (as _run's program_config sets it); on a
    # harvested 11x10 part that is 110 cores, not 120, which moves every chunk-to-core boundary.
    grid = mesh_device.compute_with_storage_grid_size()
    num_cores = (grid.x - 1) * grid.y
    total = H * nqc
    base_per_core, n_extra = divmod(total, num_cores)
    core_start = [c * base_per_core + min(c, n_extra) for c in range(num_cores + 1)]

    def flat_to_core_slot(flat):
        for c in range(num_cores):
            if core_start[c] <= flat < core_start[c + 1]:
                return c, flat - core_start[c], core_start[c + 1] - core_start[c]
        raise AssertionError(flat)

    for name, (x, y) in {
        "res-compact": (res.sol, base.sol),
        "res-dense": (res.sol, base.dense),
        "compact-dense": (base.sol, base.dense),
    }.items():
        worst = []
        by_pos, by_slot = {}, {}
        for sh in range(S):
            for h in range(H):
                chain_cores = []
                for qc in range(nqc):
                    c, slot, n_on_core = flat_to_core_slot(h * nqc + qc)
                    if c not in chain_cores:
                        chain_cores.append(c)
                    r0 = sh * rows + qc * Q1_CHUNK
                    r1 = min(r0 + Q1_CHUNK, sh * rows + rows, logical_n)
                    if r1 <= r0:
                        continue
                    a_, b_ = x[h : h + 1, r0:r1], y[h : h + 1, r0:r1]
                    bad = not (torch.isfinite(a_).all() and torch.isfinite(b_).all())
                    p = float("-inf") if bad else _pcc(a_, b_)
                    worst.append((p, sh, h, qc, bad))
                    pos = chain_cores.index(c)
                    by_pos.setdefault((sh, pos), []).append(p)
                    by_slot.setdefault((slot, n_on_core), []).append(p)
        worst.sort()
        logger.info(
            f"[resident] {name} worst (pcc, shard, head, q_chunk): "
            + ", ".join(f"({p:.4f},{sh},{h},{qc}{',NaN' if bad else ''})" for p, sh, h, qc, bad in worst[:10])
            + f"; below 0.99: {sum(1 for w in worst if w[0] < 0.99)} of {len(worst)}"
        )
        fin = lambda v: [q for q in v if q > float("-inf")]
        mean = lambda v: (sum(fin(v)) / len(fin(v))) if fin(v) else float("nan")
        shards_bad = sorted({sh for (sh, _), v in by_pos.items() if mean(v) < 0.999})
        for sh in shards_bad[:3]:
            logger.info(
                f"[resident] {name} shard {sh} mean pcc by chain position: "
                + " ".join(f"{pos}:{mean(by_pos[(sh, pos)]):.4f}" for (s_, pos) in sorted(by_pos) if s_ == sh)
            )
        logger.info(
            f"[resident] {name} mean pcc by (slot within core, chunks on core): "
            + " ".join(f"{k}:{mean(v):.4f}" for k, v in sorted(by_slot.items()))
        )
    assert res.modes_logged == {"resident"}, f"factory compiled {sorted(res.modes_logged)}"
    assert base.density == res.density, "the mask must be the same block-form one"
    assert p_chunk > 0.999, p_chunk


@GALAXY_RING
def test_sol_row_max_bound(mesh_device, sp_axis, tp_axis, num_links, topology, is_fsdp, reset_seeds):
    """Resident mode's static row max must bound every raw logit of its query row.

    The pre-pass's ||q_row|| * max ||k|| + ln(64)/scale (bf16, read back from the device) is checked on host
    against the row's true max of q.k over ALL keys of the sequence plus the same ln(64)/scale margin, on the
    bf16-rounded inputs the device saw. Also pins the tensor contract: bf16 tile [1, H, rows, 32], column 0 only.
    """
    rows = 512
    S = tuple(mesh_device.shape)[sp_axis]
    scale = D**-0.5
    ccl = CCLManager(mesh_device=mesh_device, num_links=num_links, topology=topology)
    q, k, v = _qkv(S, rows)
    mesh_axes = [sp_axis, None, None, None]
    tq, tk, tv = (
        from_torch(x, device=mesh_device, dtype=ttnn.bfloat16, mesh_axes=mesh_axes, layout=ttnn.TILE_LAYOUT)
        for x in (q, k, v)
    )
    pre = SolPrepass(mesh_device=mesh_device, ccl_manager=ccl, sp_axis=sp_axis, tau=1.0, mode="compact", row_max=True)
    pre.prepare(SolGeometry(shards=S, rows=rows, logical_n=S * rows, q_chunk=Q1_CHUNK, k_chunk=K_CHUNK), 300, scale)
    rm = pre(tq, tk, tv, scale=scale)["row_max"]
    assert tuple(rm.shape) == (1, H, rows, 32), rm.shape
    assert rm.dtype == ttnn.bfloat16 and rm.layout == ttnn.TILE_LAYOUT
    # Device d on dim 0, its TP replicas of the H heads on dim 1 (the composer _run uses for the gathered K).
    host = ttnn.to_torch(
        rm,
        mesh_composer=ttnn.ConcatMesh2dToTensor(
            mesh_device, dims=[1, 0] if sp_axis == 1 else [0, 1], mesh_shape=tuple(mesh_device.shape)
        ),
    )[
        :, :H
    ].float()  # [S, H, rows, 32]
    assert int(torch.count_nonzero(host[..., 1:])) == 0, "columns 1..31 of sol_row_max must be zero"
    bound = host[..., 0]  # [S, H, rows]
    qb, kb = q.bfloat16().float(), k.bfloat16().float()
    margin = math.log(64) / scale
    # Per head: every local query row against every key of the whole sequence (all S shards).
    true_max = torch.stack(
        [(qb[:, h] @ kb[:, h].reshape(S * rows, D).T).amax(dim=-1) for h in range(H)], dim=1
    )  # [S, H, rows]
    true_max = true_max + margin
    slack = bound - true_max
    logger.info(
        f"[row_max] min slack {float(slack.min()):.3f} (bound mean {float(bound.mean()):.2f}, true max mean "
        f"{float(true_max.mean()):.2f}, margin {margin:.2f})"
    )
    assert float(slack.min()) >= 0.0, f"sol_row_max is below the true max raw logit by {-float(slack.min()):.3f}"


@GALAXY_RING
@pytest.mark.parametrize("rows,logical_n", [(1024, 8192), (1184, 9472)], ids=["r1024", "r1184"])
@pytest.mark.parametrize("q_chunk", [64, 128, 256], ids=["q64", "q128", "q256"])
def test_base_determinism(
    mesh_device, sp_axis, tp_axis, num_links, topology, is_fsdp, q_chunk, rows, logical_n, reset_seeds, capfd
):
    """Is the base kernel deterministic and sane at this query chunk, WITHOUT resident mode? The dense ring op and
    chunk mode, each run twice with identical inputs. Every 64-row-chunk mode (pack, resident) sits on the base
    kernel's behaviour at that chunk, so this finds the smallest clean chunk."""
    kw = dict(mode="chunk", tau=-1e4, q_chunk=q_chunk, rows=rows, logical_n=logical_n)
    a = _run(mesh_device, sp_axis, num_links, topology, capfd, **kw)
    b = _run(mesh_device, sp_axis, num_links, topology, capfd, **kw)
    d_dense = (a.dense - b.dense).abs().max().item()
    d_chunk = (a.sol - b.sol).abs().max().item()
    sane = float(a.dense[:, :logical_n].abs().max()) < 1e3 and torch.isfinite(a.dense[:, :logical_n]).all()
    logger.info(
        f"[base] q_chunk={q_chunk} rows={rows}: run-to-run max |diff| dense {d_dense:.3e}, chunk {d_chunk:.3e}; "
        f"dense max |value| {float(a.dense[:, :logical_n].abs().max()):.3g}; pcc(chunk, dense) {_pcc(a.sol, a.dense, logical_n):.6f}"
    )
    # Where do the two dense runs differ? Max |a - b| per (shard, head, q_chunk), worst first, plus how many such
    # cells differ at all: one cell on one device is a per-core fault, a whole shard is a per-device one.
    S = tuple(mesh_device.shape)[sp_axis]
    nqc = -(-rows // q_chunk)
    cells = []
    for sh in range(S):
        for h in range(H):
            for qc in range(nqc):
                r0 = sh * rows + qc * q_chunk
                r1 = min(r0 + q_chunk, sh * rows + rows, logical_n)
                if r1 > r0:
                    cells.append((float((a.dense[h, r0:r1] - b.dense[h, r0:r1]).abs().max()), sh, h, qc))
    cells.sort(reverse=True)
    logger.info(
        f"[base] dense run-to-run worst cells (max|diff|, shard, head, q_chunk): "
        + ", ".join(f"({d:.3g},{sh},{h},{qc})" for d, sh, h, qc in cells[:12])
        + f"; cells differing: {sum(1 for c in cells if c[0] > 0)} of {len(cells)}"
    )
    # Is either dense run even correct? fp32 torch attention on the same inputs (exact: no pooling), per shard.
    ref = _pooled_tail_reference(S, rows, logical_n, pool_partial=False)
    for name, t in (("dense a", a.dense), ("dense b", b.dense), ("chunk a", a.sol)):
        per_shard = " ".join(
            f"{_pcc(t[:, sh * rows : min(sh * rows + rows, logical_n)], ref[:, sh * rows : min(sh * rows + rows, logical_n)]):.4f}"
            for sh in range(S)
        )
        logger.info(f"[base] pcc vs fp32 reference per shard, {name}: {per_shard}")
    assert sane, "dense ring op output is not finite / sane at this geometry"
    assert d_dense == 0.0, f"dense ring op at q_chunk={q_chunk} is not deterministic: {d_dense:.3e}"
    assert d_chunk == 0.0, f"chunk mode at q_chunk={q_chunk} is not deterministic: {d_chunk:.3e}"


@GALAXY_RING
def test_resident_dense_equivalence_q64(
    mesh_device, sp_axis, tp_axis, num_links, topology, is_fsdp, reset_seeds, capfd
):
    """tau = -1e4 keeps every chunk, so with ~2 resident chunks per core every piece steps every chunk and no state
    is ever rotated: isolates the FIFO step path (read front, write back) from the rotation path. Must match dense
    as closely as chunk mode does (test_dense_equivalence gates at 0.9995)."""
    r = _run(mesh_device, sp_axis, num_links, topology, capfd, mode="resident", tau=-1e4, q_chunk=Q1_CHUNK)
    p = _pcc(r.dense, r.sol, LOGICAL_N)
    logger.info(f"[resident] q_chunk=64 tau=-1e4 density {r.density:.4f}; pcc(resident, dense) {p:.6f}")
    assert r.density > 0.99, r.density
    assert p > 0.9995, p


@GALAXY_RING
@pytest.mark.parametrize("tau", [-1e4, 0.5, 1.0, 2.0])
def test_q2_packing_across_densities(
    mesh_device, sp_axis, tp_axis, num_links, topology, is_fsdp, tau, reset_seeds, capfd
):
    """Sweeps the page-fill pattern. tau sets how many blocks are kept, which sets how chunks straddle page
    boundaries and how big the trailing partial page is. tau = -1e4 keeps everything, so every page but the
    last is full and the output must reproduce dense almost exactly.
    """
    r = _run(mesh_device, sp_axis, num_links, topology, capfd, mode="pack", tau=tau, q_chunk=Q1_CHUNK)
    pcc = _pcc(r.dense, r.sol)
    logger.info(f"[q2] tau={tau} density {r.density:.4f} pcc vs dense {pcc:.5f}")
    assert pcc > (0.9995 if tau < 0 else 0.99)


@GALAXY_RING
def test_q2_zero_kept_iteration(mesh_device, sp_axis, tp_axis, num_links, topology, is_fsdp, reset_seeds, capfd):
    """The degenerate page case: a very high tau leaves ring iterations with (almost) no kept blocks.

    Block 0 of every shard is still forced, so each iteration carries one partial page; the reader's trailing
    flush and compute's page count both have to agree on that single page, and the pooled correction carries
    the rest.
    """
    r = _run(mesh_device, sp_axis, num_links, topology, capfd, mode="pack", tau=6.0, q_chunk=Q1_CHUNK)
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
@pytest.mark.parametrize("mode", ["gather", "compact"])
def test_partial_tail_block_is_pooled_not_dropped(
    mesh_device, sp_axis, tp_axis, num_links, topology, is_fsdp, mode, reset_seeds, capfd
):
    """Block-form modes pool the partial tail block rather than drop it.

    This file's positional ramp makes the 32-token tail block of every shard carry most of the softmax mass for
    the tail queries, so the zeroth-order correction on that one block costs a visible PCC drop against dense by
    itself, where the same correction on randn data is nearly free. The kernel does what the algorithm specifies;
    the right yardstick is _pooled_tail_reference.

    Pinned here: sol matches that reference, and sits closer to dense than to the tail-dropped reference, which
    is the signature of a pooled, not dropped, block.
    """
    rows, full_rows = 1184, 1152  # 18 full 64-token blocks = 1152; the 19th holds 32
    S = tuple(mesh_device.shape)[sp_axis]
    r = _run(mesh_device, sp_axis, num_links, topology, capfd, mode=mode, tau=-1e4, rows=rows, logical_n=8 * rows)
    ref = _pooled_tail_reference(S, rows, 8 * rows)
    q, k, v = _qkv(S, rows)
    kt = torch.cat([k[s, :, :full_rows] for s in range(S)], dim=1).float()  # tail tokens removed
    vt = torch.cat([v[s, :, :full_rows] for s in range(S)], dim=1).float()
    trunc = torch.cat(
        [torch.softmax(q[s].float() @ kt.transpose(1, 2) * (D**-0.5), dim=-1) @ vt for s in range(S)], dim=1
    )
    pcc_ref = _pcc(ref, r.sol, logical_n=8 * rows)
    pcc_dense = _pcc(r.dense, r.sol, logical_n=8 * rows)
    pcc_trunc = _pcc(trunc, r.sol, logical_n=8 * rows)
    # The device's own bf16 floor on these inputs: its dense op against fp32 dense. The pooled path should sit
    # near it; the gap between the two numbers is what the pooled column's extra roundings cost.
    dense_fp32 = _pooled_tail_reference(S, rows, 8 * rows, pool_partial=False)
    pcc_floor = _pcc(dense_fp32, r.dense, logical_n=8 * rows)
    logger.info(
        f"[tail] mode={mode}: sol vs pooled-tail ref {pcc_ref:.5f}, vs dense {pcc_dense:.5f}, vs truncated "
        f"{pcc_trunc:.5f}; device dense vs fp32 dense (bf16 floor) {pcc_floor:.5f}"
    )
    assert pcc_ref > 0.98, f"{mode}: sol vs pooled-tail reference {pcc_ref:.5f}"
    assert pcc_trunc < pcc_dense, "sol is closer to the tail-dropped reference than to dense: the tail block is dropped"


TIMING = pytest.mark.skipif(os.environ.get("SOL_TIMING") != "1", reason="set SOL_TIMING=1")
TIMING_REPS = int(os.environ.get("SOL_TIMING_REPS", "3"))
# Per-device rows and the production (q_chunk, k_chunk) for compact; pack uses q_chunk 64 with the same k_chunk.
TIMING_SHAPES = {"5s": (4736, 320, 384), "10s": (9184, 256, 512), "15s": (13664, 256, 512)}
# tau per data generator: randn lands near density 0.16 / 0.10, the ramp saturates (its middle queries have no
# signal and keep about half), so it only shows the correlated-mask behaviour at high density.
TIMING_TAUS = {"randn": (1.0, 1.3), "ramp": (1.0, 3.0)}


@TIMING
@GALAXY_RING
@pytest.mark.parametrize("data", ["randn", "ramp"])
@pytest.mark.parametrize("sol_mode", ["compact", "pack"])
@pytest.mark.parametrize("shape", list(TIMING_SHAPES))
def test_sol_op_timing(
    mesh_device, sp_axis, tp_axis, num_links, topology, is_fsdp, shape, sol_mode, data, reset_seeds, capfd
):
    """Op-only timing of the ring-joint SDPA (fused K/V all-gather included): dense vs sol at a production shape.

    Same method as test_sol_bench_minimax_h3 (which needs the MiniMax-H3 diffusers patch; this file does not):
    host sync, launch, host sync, one warm-up then TIMING_REPS timed launches, median reported. The pre-pass is
    timed separately -- it is the route program's cost and is paid once per layer alongside the op.

    Reads: `[timing] ... dense X ms, sol Y ms, speedup Z x` at the tau whose density is nearest 0.15. The dense op
    is compute-bound and the sparse path data-movement-bound, so a higher power limit shrinks dense more than sol;
    a ratio taken on a lower-power part is optimistic for production.
    """
    import time

    rows, q_prod, k_chunk = TIMING_SHAPES[shape]
    q_chunk = Q1_CHUNK if sol_mode == "pack" else q_prod
    S = tuple(mesh_device.shape)[sp_axis]
    scale = D**-0.5
    sink_tokens = 300 if rows < 4096 else 849
    logical_n = S * rows
    geo = SolGeometry(shards=S, rows=rows, logical_n=logical_n, q_chunk=q_chunk, k_chunk=k_chunk)
    ccl = CCLManager(mesh_device=mesh_device, num_links=num_links, topology=topology)
    mesh_axes = [sp_axis, None, None, None]
    q, k, v = _qkv(S, rows, ramp=(data == "ramp"))
    tq, tk, tv = (
        from_torch(x, device=mesh_device, dtype=ttnn.bfloat16, mesh_axes=mesh_axes, layout=ttnn.TILE_LAYOUT)
        for x in (q, k, v)
    )
    grid = mesh_device.compute_with_storage_grid_size()
    ckc = ttnn.init_device_compute_kernel_config(
        mesh_device.arch(), math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=False
    )
    empty_joint = bf16_tensor(torch.zeros((1, H, 0, D)), device=mesh_device)
    ln_tensor = logical_length_tensor(mesh_device, logical_n)

    def launch(qc, kc, **sol_kwargs):
        program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(grid.x - 1, grid.y),
            q_chunk_size=qc,
            k_chunk_size=kc,
            exp_approx_mode=os.environ.get("SOL_EXP_APPROX", "0") == "1",  # 1: use the approximate SFPU exp
        )
        outs = ttnn.transformer.ring_joint_scaled_dot_product_attention(
            tq,
            tk,
            tv,
            empty_joint,
            empty_joint,
            empty_joint,
            persistent_output_buffer_k=ccl.get_ag_ping_pong_buffer(tq.shape, 2, sp_axis, dtype=ttnn.bfloat16),
            persistent_output_buffer_v=ccl.get_ag_ping_pong_buffer(tq.shape, 2, sp_axis, dtype=ttnn.bfloat16),
            joint_strategy="rear",
            logical_n=ln_tensor,
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
        for t in outs:  # output, joint output (empty), stats: free them so 30 launches at 15 s do not pile up
            try:
                ttnn.deallocate(t)
            except Exception:  # an empty joint output may have nothing to free
                pass

    def timed(fn):
        fn()  # compile / warm-up
        ttnn.synchronize_device(mesh_device)
        ms = []
        for _ in range(TIMING_REPS):
            t0 = time.perf_counter()
            fn()
            ttnn.synchronize_device(mesh_device)
            ms.append(1e3 * (time.perf_counter() - t0))
        ms.sort()
        return ms[len(ms) // 2]

    dense_ms = timed(lambda: launch(q_prod, k_chunk))
    capfd.readouterr()
    pre = SolPrepass(mesh_device=mesh_device, ccl_manager=ccl, sp_axis=sp_axis, tau=TIMING_TAUS[data][0], mode=sol_mode)
    pre.prepare(geo, sink_tokens, scale)
    for tau in TIMING_TAUS[data]:
        pre.tau = tau
        pre_ms = timed(lambda: pre(tq, tk, tv, scale=scale, report_density=False))
        sol = pre(tq, tk, tv, scale=scale, report_density=True)
        sol_kw = dict(
            sol_block_mask=sol["block_mask"],
            sol_pooled_k=sol["pooled_k"],
            sol_pooled_v=sol["pooled_v"],
            sol_mode=sol_mode,
        )
        sol_ms = timed(lambda: launch(q_chunk, k_chunk, **sol_kw))
        modes = _drain_mode_log(capfd)
        logger.info(
            f"[timing] {shape} {data} {sol_mode} q{q_chunk}/k{k_chunk} tau={tau} density={pre.last_density:.3f}: "
            f"dense {dense_ms:.2f} ms, sol {sol_ms:.2f} ms, speedup {dense_ms / sol_ms:.2f}x, prepass {pre_ms:.2f} ms "
            f"(factory {sorted(modes) or 'cached'})"
        )
        assert not modes or modes == {sol_mode}, f"factory compiled {sorted(modes)}"


@SLOW
@GALAXY_RING
def test_production_5s_pack(mesh_device, sp_axis, tp_axis, num_links, topology, is_fsdp, reset_seeds, capfd):
    """Pack mode at the 5 s shape: 64-row Q chunks (TQ = 1, no OR) with kept blocks packed across the 384-token K
    chunks (6 blocks per page). The correctness gate for the production geometry; perf comes from the profiler.
    logical_n is the full padded extent so no shard has a partial tail block and the dense comparison is clean.
    """
    r = _run(
        mesh_device,
        sp_axis,
        num_links,
        topology,
        capfd,
        mode="pack",
        tau=1.0,
        rows=4736,
        logical_n=8 * 4736,
        q_chunk=Q1_CHUNK,
        k_chunk=384,
    )
    pcc = _pcc(r.dense, r.sol)
    logger.info(f"[5s pack] density {r.density:.4f} pcc vs dense {pcc:.5f} factory_logged={sorted(r.modes_logged)}")
    assert r.modes_logged == {"pack"}
    assert pcc > 0.99


def _run_sequence(
    mesh_device, sp_axis, num_links, topology, capfd, *, mode, tau, seeds, rows, logical_n, q_chunk, k_chunk
):
    """`mode` launched once per seed on ONE CCLManager, fresh Q/K/V each call; returns [(output, op ms, density)].

    A call that skipped an all-gather wait would compute on the previous call's K/V and still pass on repeated
    inputs, so every call here gets new inputs and is judged on its own. Prepass + launch per call, nothing else,
    so two sequences on the same seeds see the same device history.
    """
    S = tuple(mesh_device.shape)[sp_axis]
    scale = D**-0.5
    sink_tokens = 300 if rows < 4096 else 849
    if mode == "resident" and q_chunk == Q_CHUNK:
        q_chunk = int(os.environ.get("SOL_RESIDENT_Q_CHUNK", str(Q1_CHUNK)))
    geo = SolGeometry(shards=S, rows=rows, logical_n=logical_n, q_chunk=q_chunk, k_chunk=k_chunk)
    ccl = CCLManager(mesh_device=mesh_device, num_links=num_links, topology=topology)
    mesh_axes = [sp_axis, None, None, None]
    pre = SolPrepass(
        mesh_device=mesh_device,
        ccl_manager=ccl,
        sp_axis=sp_axis,
        tau=tau,
        mode="compact" if mode == "resident" else mode,
        row_max=False,  # the resident op freezes its max on device
    )
    pre.prepare(geo, sink_tokens, scale)
    grid = mesh_device.compute_with_storage_grid_size()
    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(grid.x - 1, grid.y),
        q_chunk_size=q_chunk,
        k_chunk_size=k_chunk,
        exp_approx_mode=os.environ.get("SOL_EXP_APPROX", "0") == "1",  # 1: use the approximate SFPU exp
    )
    ckc = ttnn.init_device_compute_kernel_config(
        mesh_device.arch(), math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=False
    )
    empty_joint = bf16_tensor(torch.zeros((1, H, 0, D)), device=mesh_device)
    logical_n_t = logical_length_tensor(mesh_device, logical_n)
    results = []
    for seed in seeds:
        q, k, v = _qkv(S, rows, seed)
        tq, tk, tv = (
            from_torch(x, device=mesh_device, dtype=ttnn.bfloat16, mesh_axes=mesh_axes, layout=ttnn.TILE_LAYOUT)
            for x in (q, k, v)
        )
        sol = pre(tq, tk, tv, scale=scale, report_density=True)
        sol_kw = dict(
            sol_block_mask=sol["block_mask"], sol_pooled_k=sol["pooled_k"], sol_pooled_v=sol["pooled_v"], sol_mode=mode
        )
        if mode == "resident" and sol.get("row_max") is not None:
            sol_kw["sol_row_max"] = sol["row_max"]
        pk = ccl.get_ag_ping_pong_buffer(tq.shape, 2, sp_axis, dtype=ttnn.bfloat16)
        pv = ccl.get_ag_ping_pong_buffer(tq.shape, 2, sp_axis, dtype=ttnn.bfloat16)
        sems = ccl.get_ag_ping_pong_semaphore(sp_axis)
        ttnn.synchronize_device(mesh_device)
        t0 = time.perf_counter()
        out, _, _ = ttnn.transformer.ring_joint_scaled_dot_product_attention(
            tq,
            tk,
            tv,
            empty_joint,
            empty_joint,
            empty_joint,
            persistent_output_buffer_k=pk,
            persistent_output_buffer_v=pv,
            joint_strategy="rear",
            logical_n=logical_n_t,
            program_config=program_config,
            compute_kernel_config=ckc,
            dim=2,
            multi_device_global_semaphore=sems,
            num_links=num_links,
            cluster_axis=sp_axis,
            mesh_device=mesh_device,
            topology=topology,
            subdevice_id=ccl.ccl_sub_device_id,
            ccl_core_grid_offset=(grid.x - 1, 0),
            use_column_major_ccl=True,
            **sol_kw,
        )
        ttnn.synchronize_device(mesh_device)
        ms = (time.perf_counter() - t0) * 1e3
        host = ttnn.to_torch(
            out,
            mesh_composer=ttnn.ConcatMesh2dToTensor(
                mesh_device, dims=[1, 2] if sp_axis == 1 else [2, 1], mesh_shape=tuple(mesh_device.shape)
            ),
        )
        results.append((host[0, :H].float(), ms, float(pre.last_density)))
    _drain_mode_log(capfd)
    return results


@GALAXY_RING
@pytest.mark.parametrize(
    "rows,logical_n,k_chunk",
    [(1024, 8192, 256), (4736, 37888, 384), (4736, 37888, 256)],
    ids=["r1024k256", "r4736k384", "r4736k256"],
)
def test_resident_call_parity(
    mesh_device, sp_axis, tp_axis, num_links, topology, is_fsdp, rows, logical_n, k_chunk, reset_seeds, capfd
):
    """Four resident calls in a row, fresh inputs each, each checked against compact mode run the same way (the
    same block-form mask and exact set; see test_resident_matches_compact).

    Fresh inputs per call make a skipped all-gather wait visible as a wrong answer (the previous call's K/V), and
    per-call timing ties the parity to it. The 4736-row case is the 5 s production geometry with 384-key pieces.
    """
    kw = dict(tau=2.0, seeds=[11, 12, 13, 14], rows=rows, logical_n=logical_n, q_chunk=Q1_CHUNK, k_chunk=k_chunk)
    res = _run_sequence(mesh_device, sp_axis, num_links, topology, capfd, mode="resident", **kw)
    ref = _run_sequence(mesh_device, sp_axis, num_links, topology, capfd, mode="compact", **kw)
    pccs = []
    for i, (seed, (r, r_ms, dens), (c, c_ms, _)) in enumerate(zip(kw["seeds"], res, ref)):
        pcc = _pcc(r, c, logical_n)
        sane = torch.isfinite(r[:, :logical_n]).all() and float(r[:, :logical_n].abs().max()) < 1e3
        pccs.append(pcc)
        logger.info(
            f"[parity] rows={rows} call {i} seed {seed}: resident {r_ms:.2f} ms, compact {c_ms:.2f} ms, "
            f"pcc(resident, compact) {pcc:.6f}, block density {dens:.4f}, resident sane {bool(sane)}"
        )
    logger.info(f"[parity] rows={rows} resident ms: " + " / ".join(f"{m:.2f}" for _, m, _ in res))
    assert min(pccs) > 0.99, f"a resident call disagrees with compact mode on its own inputs: {pccs}"
