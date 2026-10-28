# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Functional correctness of `ttnn.transformer.sparse_ring_attention` (Sol-Attn) against the dense ring op.

Every test runs the routing pre-pass and the sparse op, and the dense `ring_joint_scaled_dot_product_attention` on
the same inputs. Two yardsticks:

  dense      the dense ring op. At tau = -1e4 every full block is kept, so with 64-aligned shards the sparse op
             must reproduce it; at a finite tau it must stay close.
  reference  `sol_qblock_ref.sol_attention`: fp32 torch Sol-Attn on the device's own mask and pooled K / V (kept
             blocks exact, every other block one pooled entry weighted by its token count). This is what the
             op computes, so it separates kernel errors from the approximation itself.

Run:  pytest models/tt_dit/tests/models/minimax_h3/test_sparse_ring_attention_minimax_h3.py -k 4x8
The 5 s production geometry is opt-in with SPARSE_RING_FULL=1.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import pytest
import torch
from loguru import logger

import ttnn

from ....models.transformers.minimax_h3.sol_prepass import SolGeometry, SolPrepass
from ....models.transformers.minimax_h3.sol_qblock_ref import sol_attention
from ....parallel.manager import CCLManager
from ....utils.tensor import bf16_tensor, from_torch
from .common import GALAXY_RING

H, D = 14, 128
SPARSE_Q_CHUNK = 64  # the sparse op's query unit: one mask row per 64-row block
KEEP_ALL = -1e4
SLOW = pytest.mark.skipif(os.environ.get("SPARSE_RING_FULL") != "1", reason="set SPARSE_RING_FULL=1")


@dataclass
class Result:
    dense: torch.Tensor  # [H, S*rows, D], dense ring op
    sparse: torch.Tensor  # [H, S*rows, D], sparse op
    ref: torch.Tensor | None  # [H, S*rows, D], torch Sol-Attn on the device mask, or None
    density: float  # kept fraction of the full 64x64 blocks


def _pcc(a, b, logical_n=None):
    """PCC over the logical rows only; rows at or past logical_n are padding the ops leave undefined."""
    if logical_n is not None:
        a, b = a[:, :logical_n], b[:, :logical_n]
    a, b = a.flatten().float(), b.flatten().float()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def _qkv(S, rows, seed=0, ramp=True):
    """Q / K / V [S, H, rows, D]. ramp=True adds a positional ramp so score mass concentrates near the diagonal and
    the mask is structured; ramp=False is pure randn, an unstructured mask (density about P(z > tau))."""
    g = torch.Generator().manual_seed(seed)
    shape = (S, H, rows, D)
    q = torch.randn(shape, generator=g) * 0.5
    k = torch.randn(shape, generator=g) * 0.5
    v = torch.randn(shape, generator=g)
    if not ramp:
        return q, k, v
    pos = torch.linspace(-2.0, 2.0, rows).view(1, 1, rows, 1)
    return q + pos, k + pos, v


class Runner:
    """Device state shared by the calls of one test: CCL manager, pre-pass, program configs."""

    def __init__(self, mesh_device, sp_axis, num_links, topology, *, tau, rows, logical_n, k_chunk, dense_chunks):
        self.mesh_device, self.sp_axis, self.num_links, self.topology = mesh_device, sp_axis, num_links, topology
        self.S = tuple(mesh_device.shape)[sp_axis]
        self.rows, self.logical_n = rows, logical_n
        self.scale = D**-0.5
        self.ccl = CCLManager(mesh_device=mesh_device, num_links=num_links, topology=topology)
        self.geo = SolGeometry(shards=self.S, rows=rows, logical_n=logical_n, q_chunk=SPARSE_Q_CHUNK, k_chunk=k_chunk)
        self.pre = SolPrepass(
            mesh_device=mesh_device, ccl_manager=self.ccl, sp_axis=sp_axis, tau=tau, mode="compact", emit_bias=False
        )
        self.pre.prepare(self.geo, 300 if rows < 4096 else 849, self.scale)
        self.grid = mesh_device.compute_with_storage_grid_size()

        def config(q_chunk, kc):
            return ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(self.grid.x - 1, self.grid.y),
                q_chunk_size=q_chunk,
                k_chunk_size=kc,
                exp_approx_mode=False,
            )

        self.sparse_config = config(SPARSE_Q_CHUNK, k_chunk)
        self.dense_config = config(*dense_chunks)
        self.ckc = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(), math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=False
        )
        self.empty_joint = bf16_tensor(torch.zeros((1, H, 0, D)), device=mesh_device)
        self.logical_n_tensor = from_torch(
            torch.tensor([logical_n], dtype=torch.int64).reshape(1, 1, 1, 1),
            device=mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.Layout.ROW_MAJOR,
            mesh_axes=[..., None, None],
        )

    def _ring_kwargs(self, tq):
        return dict(
            persistent_output_buffer_k=self.ccl.get_ag_ping_pong_buffer(tq.shape, 2, self.sp_axis, dtype=ttnn.bfloat16),
            persistent_output_buffer_v=self.ccl.get_ag_ping_pong_buffer(tq.shape, 2, self.sp_axis, dtype=ttnn.bfloat16),
            compute_kernel_config=self.ckc,
            dim=2,
            multi_device_global_semaphore=self.ccl.get_ag_ping_pong_semaphore(self.sp_axis),
            num_links=self.num_links,
            cluster_axis=self.sp_axis,
            mesh_device=self.mesh_device,
            topology=self.topology,
            subdevice_id=self.ccl.ccl_sub_device_id,
            ccl_core_grid_offset=(self.grid.x - 1, 0),
            use_column_major_ccl=True,
        )

    def _to_host(self, t, sp_dim, tp_dim):
        dims = [None, None]
        dims[self.sp_axis], dims[1 - self.sp_axis] = sp_dim, tp_dim
        composer = ttnn.ConcatMesh2dToTensor(self.mesh_device, dims=dims, mesh_shape=tuple(self.mesh_device.shape))
        return ttnn.to_torch(t, mesh_composer=composer)

    def __call__(self, *, seed=0, ramp=True, want_ref=True) -> Result:
        q, k, v = _qkv(self.S, self.rows, seed, ramp)
        mesh_axes = [self.sp_axis, None, None, None]
        tq, tk, tv = (
            from_torch(x, device=self.mesh_device, dtype=ttnn.bfloat16, mesh_axes=mesh_axes, layout=ttnn.TILE_LAYOUT)
            for x in (q, k, v)
        )
        dense, _, _ = ttnn.transformer.ring_joint_scaled_dot_product_attention(
            tq,
            tk,
            tv,
            self.empty_joint,
            self.empty_joint,
            self.empty_joint,
            joint_strategy="rear",
            logical_n=self.logical_n_tensor,
            program_config=self.dense_config,
            **self._ring_kwargs(tq),
        )
        ttnn.synchronize_device(self.mesh_device)
        dense_host = self._to_host(dense, 2, 1)[0, :H].float()  # sequence on dim 2, TP replicas on dim 1
        ttnn.deallocate(dense)

        sol = self.pre(tq, tk, tv, scale=self.scale, report_density=True)
        sparse = ttnn.transformer.sparse_ring_attention(
            tq,
            tk,
            tv,
            sol["block_mask"],
            sol["pooled_k"],
            sol["pooled_v"],
            logical_n=self.logical_n,
            program_config=self.sparse_config,
            **self._ring_kwargs(tq),
        )
        ttnn.synchronize_device(self.mesh_device)
        sparse_host = self._to_host(sparse, 2, 1)[0, :H].float()
        ttnn.deallocate(sparse)

        ref = None
        if want_ref:
            # The device's own mask and pooled K / V (bf16, as the op reads them), on the bf16-rounded inputs.
            mask = self._to_host(sol["block_mask"], 0, 1)[:, :H].float()  # [S, H, nqb_pad, nbg]
            pooled_k = self._to_host(sol["pooled_k"], 0, 1)[0, :H].float()  # [H, nbg, D], same on every device
            pooled_v = self._to_host(sol["pooled_v"], 0, 1)[0, :H].float()
            qb, kb, vb = (x.bfloat16().float() for x in (q, k, v))
            out = sol_attention(qb, kb, vb, mask, pooled_k, pooled_v, self.geo, scale=self.scale)  # [S, H, rows, D]
            ref = out.permute(1, 0, 2, 3).reshape(H, self.S * self.rows, D)
        return Result(dense=dense_host, sparse=sparse_host, ref=ref, density=float(self.pre.last_density))


def _runner(mesh_device, sp_axis, num_links, topology, *, tau, rows, logical_n, k_chunk, dense_chunks=(256, 512)):
    return Runner(
        mesh_device,
        sp_axis,
        num_links,
        topology,
        tau=tau,
        rows=rows,
        logical_n=logical_n,
        k_chunk=k_chunk,
        dense_chunks=dense_chunks,
    )


def _log(tag, r: Result, logical_n):
    p_dense = _pcc(r.dense, r.sparse, logical_n)
    p_ref = _pcc(r.ref, r.sparse, logical_n) if r.ref is not None else float("nan")
    logger.info(f"[{tag}] density {r.density:.4f}; pcc(sparse, dense) {p_dense:.5f}; pcc(sparse, ref) {p_ref:.5f}")
    return p_dense, p_ref


# (rows per device, logical_n, k_chunk) with 64-aligned shards, so tau = -1e4 leaves nothing pooled.
# rows 512: few pieces; 1024: about two Q blocks per core, some cores straddling two heads; k 256 vs 512: 4 vs 8
# blocks per piece; logical_n below the padded length: a short last shard.
ALIGNED = {
    "r512k512": (512, 4096, 512),
    "r1024k512": (1024, 8192, 512),
    "r1024k256": (1024, 8192, 256),
    "r1024k256short": (1024, 7680, 256),
}
# Shards ending in a partial block (32 tokens), pooled by design even at tau = -1e4; judged against the reference.
RAGGED = {
    "r1184k512": (1184, 9472, 512),
    "r1184k256": (1184, 9472, 256),
    "r1184ln9000": (1184, 9000, 256),  # logical_n off a block boundary too
}


@GALAXY_RING
@pytest.mark.parametrize("geom", list(ALIGNED))
def test_keep_all_matches_dense(mesh_device, sp_axis, tp_axis, num_links, topology, is_fsdp, geom, reset_seeds):
    """tau = -1e4 keeps every block: the sparse op is dense attention and must match the dense ring op."""
    rows, logical_n, k_chunk = ALIGNED[geom]
    run = _runner(
        mesh_device, sp_axis, num_links, topology, tau=KEEP_ALL, rows=rows, logical_n=logical_n, k_chunk=k_chunk
    )
    r = run()
    p_dense, p_ref = _log(f"keepall {geom}", r, logical_n)
    assert r.density > 0.99, r.density
    assert p_dense > 0.9995, p_dense
    assert p_ref > 0.9995, p_ref


@GALAXY_RING
@pytest.mark.parametrize("tau", [1.0, 2.0])
@pytest.mark.parametrize("geom", list(ALIGNED))
def test_sparse_vs_dense(mesh_device, sp_axis, tp_axis, num_links, topology, is_fsdp, geom, tau, reset_seeds):
    """Finite tau: the op must compute the Sol-Attn its mask defines, and stay close to dense attention."""
    rows, logical_n, k_chunk = ALIGNED[geom]
    run = _runner(mesh_device, sp_axis, num_links, topology, tau=tau, rows=rows, logical_n=logical_n, k_chunk=k_chunk)
    r = run()
    p_dense, p_ref = _log(f"tau{tau:g} {geom}", r, logical_n)
    assert 0.0 < r.density < 0.95, f"density {r.density:.4f} is not a sparse mask"
    assert p_ref > 0.999, p_ref
    assert p_dense > 0.99, p_dense


@GALAXY_RING
@pytest.mark.parametrize("tau", [KEEP_ALL, 1.0], ids=["keepall", "tau1"])
@pytest.mark.parametrize("geom", list(RAGGED))
def test_ragged_shards(mesh_device, sp_axis, tp_axis, num_links, topology, is_fsdp, geom, tau, reset_seeds):
    """Partial last blocks are pooled, never exact. On the ramp data the tail block carries most of the softmax
    mass of the tail queries, so pooling it alone costs ~0.14 PCC against dense (measured 0.855 on device in the
    old op): the reference is the yardstick here, dense is logged only."""
    rows, logical_n, k_chunk = RAGGED[geom]
    run = _runner(mesh_device, sp_axis, num_links, topology, tau=tau, rows=rows, logical_n=logical_n, k_chunk=k_chunk)
    r = run()
    _, p_ref = _log(f"ragged {geom} tau{tau:g}", r, logical_n)
    assert p_ref > 0.98, p_ref


@GALAXY_RING
@pytest.mark.parametrize("tau", [1.0, 1.3])
def test_unstructured_mask(mesh_device, sp_axis, tp_axis, num_links, topology, is_fsdp, tau, reset_seeds):
    """Pure randn inputs: no mask structure, so the Q blocks of a head group share few kept blocks and the piece
    union is widest relative to each block's own set."""
    rows, logical_n, k_chunk = ALIGNED["r1024k256"]
    run = _runner(mesh_device, sp_axis, num_links, topology, tau=tau, rows=rows, logical_n=logical_n, k_chunk=k_chunk)
    r = run(ramp=False)
    p_dense, p_ref = _log(f"randn tau{tau:g}", r, logical_n)
    assert 0.0 < r.density < 0.5, r.density
    assert p_ref > 0.999, p_ref
    assert p_dense > 0.99, p_dense


@GALAXY_RING
def test_density_falls_as_tau_rises(mesh_device, sp_axis, tp_axis, num_links, topology, is_fsdp, reset_seeds):
    """tau is the sparsity knob; the output must track the reference at every point of the sweep."""
    rows, logical_n, k_chunk = ALIGNED["r1024k256"]
    densities = []
    for tau in (0.5, 2.0, 4.0):
        run = _runner(
            mesh_device, sp_axis, num_links, topology, tau=tau, rows=rows, logical_n=logical_n, k_chunk=k_chunk
        )
        r = run()
        _, p_ref = _log(f"sweep tau{tau:g}", r, logical_n)
        assert p_ref > 0.999, (tau, p_ref)
        densities.append(r.density)
    assert densities[0] >= densities[1] >= densities[2], densities
    assert densities[0] > densities[2], densities


@GALAXY_RING
def test_determinism(mesh_device, sp_axis, tp_axis, num_links, topology, is_fsdp, reset_seeds):
    """Same inputs twice, same bits: a race in the group delivery or the plan shows up here first."""
    rows, logical_n, k_chunk = ALIGNED["r1024k256"]
    run = _runner(mesh_device, sp_axis, num_links, topology, tau=1.0, rows=rows, logical_n=logical_n, k_chunk=k_chunk)
    a = run(want_ref=False)
    b = run(want_ref=False)
    assert torch.equal(a.sparse, b.sparse), f"max |diff| {(a.sparse - b.sparse).abs().max().item():.3e}"


@GALAXY_RING
@pytest.mark.parametrize("geom", ["r1024k256", "r1024k512"])
def test_fresh_inputs_per_call(mesh_device, sp_axis, tp_axis, num_links, topology, is_fsdp, geom, reset_seeds):
    """Four calls in a row with the program cached and new inputs each: a call that skipped an all-gather wait
    would compute on the previous call's K / V and fail its own reference."""
    rows, logical_n, k_chunk = ALIGNED[geom]
    run = _runner(mesh_device, sp_axis, num_links, topology, tau=2.0, rows=rows, logical_n=logical_n, k_chunk=k_chunk)
    for seed in (11, 12, 13, 14):
        r = run(seed=seed)
        p_dense, p_ref = _log(f"call seed {seed} {geom}", r, logical_n)
        assert p_ref > 0.999, (seed, p_ref)
        assert p_dense > 0.99, (seed, p_dense)


@SLOW
@GALAXY_RING
@pytest.mark.parametrize("tau", [KEEP_ALL, 1.0], ids=["keepall", "tau1"])
@pytest.mark.parametrize("k_chunk", [256, 384])
def test_production_5s(mesh_device, sp_axis, tp_axis, num_links, topology, is_fsdp, k_chunk, tau, reset_seeds):
    """The 5 s geometry: 4736 rows per device (74 full blocks), production piece sizes; dense at its measured
    chunks (320, 384). No torch reference (too slow on host at this size)."""
    rows, logical_n = 4736, 8 * 4736
    run = _runner(
        mesh_device,
        sp_axis,
        num_links,
        topology,
        tau=tau,
        rows=rows,
        logical_n=logical_n,
        k_chunk=k_chunk,
        dense_chunks=(320, 384),
    )
    r = run(want_ref=False)
    p_dense, _ = _log(f"5s k{k_chunk} tau{tau:g}", r, logical_n)
    assert p_dense > (0.9995 if tau == KEEP_ALL else 0.99), p_dense
