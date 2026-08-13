# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Residual/collective contracts, measured through the next consuming op.

The decision this probe exists to make: after a **row-parallel** projection every
device holds a full-width partial sum, and the layer has to turn that into the
input of the next sublayer.  Four contracts can do it, they move different bytes,
and three of them change what the *residual stream* is -- so measuring only the
collective (or measuring a reduce-scatter followed by an immediate all-gather back
to the old contract) would answer the wrong question.  Every candidate here runs
the whole boundary chain:

    row-parallel matmul -> collective(s) -> RMSNorm -> residual add -> the next
    column-parallel matmul that consumes the result

and every candidate ends on the same contract it started on, so a winner can be
stacked without a conversion.

Candidates, at ``S = rows * 6656 * 2 B`` and ``P = 4`` devices (ring, so a
collective moves ``(P-1)/P`` of its payload per device and an all-reduce moves
twice that):

=========================  ==========================================  ==========  ====
name                       chain                                        bytes/dev   ops
=========================  ==========================================  ==========  ====
``replicated``             AR -> norm -> add -> matmul                   1.50 S      4
``fractured``              RS -> dist-norm(AG stats) -> add -> AG ->     1.50 S      7
                           matmul
``gather_heads``           AG(attn 4096) -> col matmul -> AG -> norm     1.21 S      5
                           -> add -> matmul
``gather_heads_fractured`` AG(attn 4096) -> col matmul -> dist-norm ->   0.46 S +    7
                           add -> AG -> matmul                          0.75 S
=========================  ==========================================  ==========  ====

Usage::

    python .../bench/topology_probe.py --rows 32 --traced
    python .../bench/topology_probe.py --rows 8192
"""

from __future__ import annotations

import argparse
import time

import torch

import ttnn
from models.autoports.meta_models_muse_glimmer_30b.tt.multichip_decoder import (
    CCL_TOPOLOGY,
    DEFAULT_DECODE_CCL_RS_WORKERS,
    DEFAULT_MESH_SHAPE,
    DEFAULT_PREFILL_CCL_RS_WORKERS,
    close_multichip_mesh,
    open_multichip_mesh,
)

HIDDEN = 6656
ATTN = 4096
LOCAL_INTERMEDIATE = 5120
TILE = 32
EPS = 1e-5


class Chain:
    """One boundary contract, built once and callable in a loop or a trace."""

    def __init__(self, mesh, rows: int, tp: int, dtype=ttnn.bfloat16, tuned: bool = True, rs_workers: int = 1):
        self.mesh = mesh
        self.rows = rows
        self.tp = tp
        self.dtype = dtype
        #: Use the shipped reducer (tuned reduce-scatter + all-gather) on every
        #: arm rather than the op defaults.  ``False`` reproduces the original
        #: capture, which predates the collective tuning.
        self.tuned = tuned
        #: ``num_workers_per_link`` is a **per-payload** knob (see
        #: ``DEFAULT_DECODE_CCL_RS_WORKERS`` / ``DEFAULT_PREFILL_CCL_RS_WORKERS``).
        #: Pinning it to the decode value at a prefill payload costs 2.4x on the
        #: reduce-scatter alone, which inflates every arm that uses one and lets
        #: the ``gather_heads*`` arms -- which use only ``all_gather`` -- escape
        #: it.  Review round 4 caught the 8192-row column doing exactly that.
        self.rs_workers = rs_workers
        self.local_attn = ATTN // tp
        self.local_hidden = HIDDEN // tp
        self.replicate = ttnn.ReplicateTensorToMesh(mesh)
        self.shard = lambda dim: ttnn.ShardTensorToMesh(mesh, dim=dim)

        def weight(k, n, shard_dim):
            torch_w = torch.randn(1, 1, k, n, dtype=torch.bfloat16) * 0.02
            return ttnn.from_torch(
                torch_w,
                device=mesh,
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat8_b,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=self.shard(shard_dim) if shard_dim is not None else self.replicate,
            )

        # o_proj, row parallel: local K, full N, output is a partial sum.
        self.wo_row = weight(ATTN, HIDDEN, -2)
        # o_proj, column parallel: full K (needs the heads gathered), local N.
        self.wo_col = weight(ATTN, HIDDEN, -1)
        # The next sublayer's first matmul: column parallel over the hidden size.
        self.w_next = weight(HIDDEN, LOCAL_INTERMEDIATE * tp, -1)

        # Activations.
        attn_local = torch.randn(1, 1, rows, ATTN, dtype=torch.bfloat16)
        self.gated = ttnn.from_torch(
            attn_local,
            device=mesh,
            layout=ttnn.TILE_LAYOUT,
            dtype=dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self.shard(-1),
        )
        hidden = torch.randn(1, 1, rows, HIDDEN, dtype=torch.bfloat16)
        self.residual_replicated = ttnn.from_torch(
            hidden,
            device=mesh,
            layout=ttnn.TILE_LAYOUT,
            dtype=dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self.replicate,
        )
        self.residual_fractured = ttnn.from_torch(
            hidden,
            device=mesh,
            layout=ttnn.TILE_LAYOUT,
            dtype=dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self.shard(-1),
        )

        # Norm weights: replicated for a local norm over the full hidden size,
        # tile-sharded for the distributed pair.
        gamma = torch.randn(HIDDEN, dtype=torch.bfloat16) * 0.05 + 1.0
        self.gamma_full = ttnn.from_torch(
            gamma.reshape(1, 1, 1, HIDDEN),
            device=mesh,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self.replicate,
        )
        self.gamma_distributed = ttnn.from_torch(
            gamma.reshape(1, 1, HIDDEN // TILE, TILE),
            device=mesh,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self.shard(-2),
        )

    # ---------------------------------------------------------------- pieces

    def _reduce(self, tensor):
        """The **shipped** decode reducer, so both arms pay the same collective.

        The first version of this probe used ``ttnn.all_reduce`` for the
        replicated arm, which was the default at the time; the collective tuning
        that followed (``num_workers_per_link=1`` on the reduce-scatter, and the
        pair instead of the fused op) made that arm 9 us/dispatch cheaper and
        left the comparison confounded.  Both arms now reduce the same way.
        """
        if not self.tuned:
            return ttnn.all_reduce(tensor, memory_config=ttnn.DRAM_MEMORY_CONFIG, topology=CCL_TOPOLOGY)
        scattered = ttnn.reduce_scatter(
            tensor,
            dim=3,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            intermediate_memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=CCL_TOPOLOGY,
            num_workers_per_link=self.rs_workers,
            use_l1_small_for_semaphores=True,
        )
        out = ttnn.all_gather(scattered, dim=3, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(scattered)
        return out

    def _scatter(self, tensor):
        """The reduce-scatter half, tuned the same way."""
        if not self.tuned:
            return ttnn.reduce_scatter(tensor, dim=3, memory_config=ttnn.DRAM_MEMORY_CONFIG, topology=CCL_TOPOLOGY)
        return ttnn.reduce_scatter(
            tensor,
            dim=3,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            intermediate_memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=CCL_TOPOLOGY,
            num_workers_per_link=self.rs_workers,
            use_l1_small_for_semaphores=True,
        )

    def _matmul(self, x, w, out_dtype=None):
        return ttnn.linear(
            x,
            w,
            dtype=out_dtype or self.dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _local_norm(self, x):
        return ttnn.rms_norm(x, epsilon=EPS, weight=self.gamma_full, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def _distributed_norm(self, x):
        """The three-op distributed RMSNorm over a width-fractured activation."""
        stats = ttnn.rms_norm_pre_all_gather(x, dtype=ttnn.bfloat16)
        stats = ttnn.reshape(stats, ttnn.Shape((1, 1, self.rows, TILE)))
        gathered = ttnn.all_gather(stats, dim=3, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        out = ttnn.rms_norm_post_all_gather(
            x, gathered, epsilon=EPS, weight=self.gamma_distributed, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        ttnn.deallocate(stats)
        ttnn.deallocate(gathered)
        return out

    # ------------------------------------------------------------ candidates

    def replicated(self, ccl_dtype=None):
        partial = self._matmul(self.gated, self.wo_row, out_dtype=ccl_dtype)
        reduced = self._reduce(partial)
        ttnn.deallocate(partial)
        normed = self._local_norm(reduced)
        ttnn.deallocate(reduced)
        hidden = ttnn.add(self.residual_replicated, normed, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(normed)
        out = self._matmul(hidden, self.w_next)
        ttnn.deallocate(hidden)
        return out

    def replicated_bfp8(self):
        return self.replicated(ccl_dtype=ttnn.bfloat8_b)

    def fractured(self):
        partial = self._matmul(self.gated, self.wo_row)
        scattered = self._scatter(partial)
        ttnn.deallocate(partial)
        normed = self._distributed_norm(scattered)
        ttnn.deallocate(scattered)
        hidden = ttnn.add(self.residual_fractured, normed, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(normed)
        # The next sublayer's matmul needs the full hidden size, so the fractured
        # residual is gathered here -- this is the cost the contract adds, and it
        # is inside the measured chain rather than outside it.
        gathered = ttnn.all_gather(hidden, dim=3, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(hidden)
        out = self._matmul(gathered, self.w_next)
        ttnn.deallocate(gathered)
        return out

    def gather_heads(self):
        heads = ttnn.all_gather(self.gated, dim=3, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        fractured = self._matmul(heads, self.wo_col)
        ttnn.deallocate(heads)
        gathered = ttnn.all_gather(fractured, dim=3, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(fractured)
        normed = self._local_norm(gathered)
        ttnn.deallocate(gathered)
        hidden = ttnn.add(self.residual_replicated, normed, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(normed)
        out = self._matmul(hidden, self.w_next)
        ttnn.deallocate(hidden)
        return out

    def gather_heads_fractured(self):
        heads = ttnn.all_gather(self.gated, dim=3, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        fractured = self._matmul(heads, self.wo_col)
        ttnn.deallocate(heads)
        normed = self._distributed_norm(fractured)
        ttnn.deallocate(fractured)
        hidden = ttnn.add(self.residual_fractured, normed, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(normed)
        gathered = ttnn.all_gather(hidden, dim=3, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(hidden)
        out = self._matmul(gathered, self.w_next)
        ttnn.deallocate(gathered)
        return out


CANDIDATES = ("replicated", "replicated_bfp8", "fractured", "gather_heads", "gather_heads_fractured")

#: Bytes each candidate moves per device, in units of ``S = rows * 6656 * 2``.
EXPECTED_BYTES = {
    "replicated": 1.5,
    "replicated_bfp8": 0.75,
    "fractured": 1.5,
    "gather_heads": 0.75 * (ATTN / HIDDEN) + 0.75,
    "gather_heads_fractured": 0.75 * (ATTN / HIDDEN) + 0.75,
}


def measure(mesh, chain: Chain, name: str, *, traced: bool, iters: int, rounds: int) -> float:
    fn = getattr(chain, name)
    for _ in range(2):
        ttnn.deallocate(fn())
    ttnn.synchronize_device(mesh)
    if traced:
        out = fn()
        ttnn.synchronize_device(mesh)
        ttnn.deallocate(out)
        trace_id = ttnn.begin_trace_capture(mesh, cq_id=0)
        held = fn()
        ttnn.end_trace_capture(mesh, trace_id, cq_id=0)
        ttnn.synchronize_device(mesh)
        for _ in range(4):
            ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh)
        best = float("inf")
        for _ in range(rounds):
            ttnn.synchronize_device(mesh)
            start = time.perf_counter()
            for _ in range(iters):
                ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)
            best = min(best, (time.perf_counter() - start) / iters)
        ttnn.release_trace(mesh, trace_id)
        ttnn.deallocate(held)
        return best * 1e6
    best = float("inf")
    for _ in range(rounds):
        ttnn.synchronize_device(mesh)
        start = time.perf_counter()
        for _ in range(iters):
            ttnn.deallocate(fn())
        ttnn.synchronize_device(mesh)
        best = min(best, (time.perf_counter() - start) / iters)
    return best * 1e6


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=32)
    parser.add_argument("--traced", action="store_true")
    parser.add_argument("--iters", type=int, default=16)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--candidates", default=",".join(CANDIDATES))
    parser.add_argument("--mesh", default="x".join(str(d) for d in DEFAULT_MESH_SHAPE))
    parser.add_argument("--untuned", action="store_true", help="use the op-default collectives (the original capture)")
    parser.add_argument(
        "--rs-workers",
        type=int,
        default=None,
        help="num_workers_per_link for the reduce-scatter; default follows the payload, "
        "as the shipped layer does (decode rows<=32 -> 1, prefill -> 4)",
    )
    args = parser.parse_args()

    mesh_shape = tuple(int(v) for v in args.mesh.split("x"))
    mesh = open_multichip_mesh(mesh_shape, trace_region_size=90112 * 24 if args.traced else 0)
    tp = mesh.get_num_devices()
    try:
        rs_workers = args.rs_workers
        if rs_workers is None:
            rs_workers = DEFAULT_DECODE_CCL_RS_WORKERS if args.rows <= 32 else DEFAULT_PREFILL_CCL_RS_WORKERS
        chain = Chain(mesh, args.rows, tp, tuned=not args.untuned, rs_workers=rs_workers)
        payload = args.rows * HIDDEN * 2
        print(
            f"TOPO rows={args.rows} tp={tp} traced={args.traced} "
            f"reducer={'op-default' if args.untuned else f'shipped(rs_w{rs_workers}+ag)'} S={payload/1024:.1f} KiB"
        )
        for name in args.candidates.split(","):
            try:
                us = measure(mesh, chain, name, traced=args.traced, iters=args.iters, rounds=args.rounds)
                moved = EXPECTED_BYTES[name] * payload
                print(
                    f"TOPO {name:24s} {us:10.2f} us   expected_ccl_bytes={moved/1024:9.1f} KiB "
                    f"({EXPECTED_BYTES[name]:.2f} S)",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001 - a probe records failures
                msg = " | ".join(line.strip() for line in str(exc).strip().splitlines() if line.strip())
                print(f"TOPO {name:24s} FAILED {msg[:300]}", flush=True)
    finally:
        close_multichip_mesh(mesh)


if __name__ == "__main__":
    main()
