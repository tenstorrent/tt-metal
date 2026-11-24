# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Prototype (phase 1, #65884): prefill's routed experts on the shared ``tp8g4`` expert split, with stock ttnn ops.

The dataflow and the per-hop bytes are in ``reference/hy4_moe_tp8g4.py`` (the host reference computes the same
thing chip by chip). Here every chip of group g (mesh column) holds, as PLAIN ttnn BFP4 tensors (not decode's
per-bank stream container), its 1/8 slice of each of its group's 64 experts: ``gate_projs[j]``, ``up_projs[j]``
``[6144, 256]`` and ``down_projs[j]`` ``[256, 6144]``, local expert j = global ``64 g + j``, slice = mesh row. The
forward, per chip, x the residual ``[S/8, 1536]``:

  1. all_gather over axis 0 (tokens) -> ``[S, 1536]`` (the shared expert's 2D-TP MLP does this same gather)
  2. all_gather over axis 1 (hidden) -> ``[S, 6144]``, untilized to row-major (the gather table)
  3. ``ttnn.embedding`` with the group's row index -> ``[R, 6144]`` BFP8 tiles, expert-major, regions tile-aligned
  4. ``unified_routed_expert_moe``: prefill's EP32 routed-expert kernel, here with 64 local experts of N = 256
     (each slice read once; clamped SwiGLU limit 10), in place on the gathered buffer
  5. the routing weight per row (``ttnn.multiply`` by ``[R, 1]``), untilize
  6. combine: ``sum_k embedding(slot_rows[k], Y)`` -> ``[S, 6144]`` token-major partial (``K`` = most picks one
     token puts in one group)
  7. reduce_scatter over axis 1 (hidden; sums the 4 groups) -> ``[S, 1536]``; [the shared expert's pre-RS partial
     would be added here]; reduce_scatter over axis 0 (tokens; sums the 8 slices) -> ``[S/8, 1536]``.

The plan (rows, counts, offsets, weights, slot rows) comes from the host (``hy4_moe_tp8g4.plan``) for the golden
routing; on device it is routing_setup's job (counts, offsets) plus a slot table. Every mesh tensor has one shape
on all chips, so the per-group plans are padded to the largest group's R and K.

Measured (4K golden, layers 1 / 2 / 4, eager, ``tests/test_hy4_moe_tp8g4_device.py``): routed PCC 0.9949 / 0.9945 /
0.9922 (the host BFP4 emulation predicts 0.9950 / 0.9946 / 0.9924); 6.8-7.0 ms per layer end to end (EP32's whole
MoE, router and shared expert included: 4.5-5.0 ms on layers 2 / 4). By stage (with a synchronize after each):
all-gathers 0.4 + 0.8, row gather 1.8, experts 2.9-3.2 (EP32 1.0-1.3: the same kernel on 64 experts of N = 256 per
chip instead of 8 of N = 2048), combine 1.1, reduce-scatters 0.8 + 0.4.

Missing from stock ttnn for an efficient version: (1) a routed-expert op whose expert matmuls stay wide when each
chip holds a thin slice of many experts (here N = 256 per expert per chip), (2) reading the activation rows by index
inside that op (the separate gather costs 1.8 ms), (3) decode's per-bank stream files as the weight operand: stock
matmuls take a WIDTH_SHARDED DRAM in1 only in the decode-sized DRAM-sharded config (M <= one L1 shard, the whole
tensor as K), with no expert offset (``tests/test_hy4_moe_tp8g4_stream_probe.py``). The combine itself is stock:
``post_combine_reduce`` with ``combine_row_index`` reads the expert rows in place.
"""

from __future__ import annotations

import time
from pathlib import Path

import torch

import ttnn
from models.demos.deepseek_v3_d_p.reference import hy4_moe_tp8g4 as t8
from models.demos.deepseek_v3_d_p.tt.moe.tt_routed_expert import (
    COMPUTE_KERNEL_CONFIG_LOFI,
    routed_expert_weight_memory_config,
)


def _col_mapper(mesh_device):
    """Host ``[4, ...]``: dim 0 over the 4 mesh columns (groups), replicated over the 8 rows."""
    return ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=(None, 0))


def _chip_mapper(mesh_device):
    """Host ``[8, 4, ...]``: dim 0 over the rows (slices), dim 1 over the columns (groups)."""
    return ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=(0, 1))


class TtRoutedTp8g4:
    def __init__(
        self,
        mesh_device,
        gate_up: torch.Tensor | None,
        down: torch.Tensor | None,
        *,
        weights_dtype=ttnn.bfloat4_b,
        cache_path: Path | None = None,
        nd_sharded: bool = True,
        compute_kernel_config=COMPUTE_KERNEL_CONFIG_LOFI,
    ):
        """gate_up ``[256, 4096, 6144]`` / down ``[256, 6144, 2048]`` as stored (None: load ``cache_path``)."""
        rows, cols = tuple(mesh_device.shape)
        assert (rows, cols) == (t8.SLICES, t8.GROUPS), "tp8g4 is the 8x4 mesh: groups = columns, slices = rows"
        self.mesh = mesh_device
        self.compute_kernel_config = compute_kernel_config
        H, I, Is = t8.HIDDEN, t8.INTER, t8.I_SLICE
        if gate_up is not None:
            gu = gate_up.view(cols, t8.EPG, 2, rows, Is, H)  # [g, j, gate|up, s, c, h]
            dn = down.view(cols, t8.EPG, H, rows, Is)  # [g, j, h, s, c]
        mapper = _chip_mapper(mesh_device)

        def place(host, name, n_dim):
            fname = None if cache_path is None else str(Path(cache_path) / name)
            tt = ttnn.as_tensor(
                host, dtype=weights_dtype, layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper, cache_file_name=fname
            )
            tt = ttnn.squeeze(ttnn.squeeze(tt, 0), 0)
            return ttnn.to_device(
                tt,
                mesh_device,
                memory_config=routed_expert_weight_memory_config(mesh_device, n_dim, dram_nd_sharded=nd_sharded),
            )

        self.gate_projs, self.up_projs, self.down_projs = [], [], []
        for j in range(t8.EPG):
            if gate_up is not None:
                g = gu[:, j, 0].permute(1, 0, 3, 2).contiguous()  # [s, g, h, c]
                u = gu[:, j, 1].permute(1, 0, 3, 2).contiguous()
                d = dn[:, j].permute(2, 0, 3, 1).contiguous()  # [s, g, c, h]
            else:
                g = u = torch.empty(rows, cols, H, Is)
                d = torch.empty(rows, cols, Is, H)
            self.gate_projs.append(place(g, f"tp8g4.local_{j}_gate", Is))
            self.up_projs.append(place(u, f"tp8g4.local_{j}_up", Is))
            self.down_projs.append(place(d, f"tp8g4.local_{j}_down", H))
        # local expert j of the chip in column g is global 64 g + j
        table = torch.arange(t8.E, dtype=torch.int32).view(cols, t8.EPG)
        self.idx_table = ttnn.squeeze(
            ttnn.from_torch(
                table,
                mesh_mapper=_col_mapper(mesh_device),
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=mesh_device,
                dtype=ttnn.uint32,
            ),
            0,
        )

    # ------------------------------------------------------------------------------------------------- plan
    def upload_plan(self, plans: list[t8.GroupPlan], S: int) -> dict:
        """The host plans as device tensors (one shape on every chip: R and K padded to the largest group)."""
        R = max(p.R for p in plans) + t8.TILE  # + one zero tile: the rows "no pick" and padding point at
        K = max(p.K for p in plans)
        zero_row = R - 1
        rows = torch.zeros(t8.GROUPS, 1, R, dtype=torch.int32)  # padding rows gather token 0, weight 0
        w = torch.zeros(t8.GROUPS, R, 1)
        counts = torch.zeros(t8.GROUPS, t8.E, dtype=torch.int32)
        offsets = torch.zeros(t8.GROUPS, t8.E, dtype=torch.int32)
        slots = torch.full((t8.GROUPS, K, 1, S), zero_row, dtype=torch.int32)
        k_top = plans[0].pair_rows.shape[1]
        pair_rows = torch.full((t8.GROUPS, 1, 1, S, k_top), R, dtype=torch.int32)  # R: skip_row (never read)
        pair_w = torch.zeros(t8.GROUPS, 1, 1, S, k_top)
        max_m = 0
        for p in plans:
            g = p.group
            valid = p.rows >= 0
            rows[g, 0, : p.R][valid] = p.rows[valid].int()
            w[g, : p.R, 0] = p.weights.float()
            base = g * t8.EPG
            counts[g, base : base + t8.EPG] = p.counts.int()
            offsets[g, base : base + t8.EPG] = p.offsets.int()
            sr = p.slot_rows.clone()
            sr[sr < 0] = zero_row
            slots[g, : p.K, 0] = sr.int()
            has = p.pair_rows >= 0
            pair_rows[g, 0, 0][has] = p.pair_rows[has].int()
            pair_w[g, 0, 0][has] = p.weights[p.pair_rows[has]].float()
            max_m = max(max_m, int(p.counts.max()))
        mesh, cm = self.mesh, _col_mapper(self.mesh)
        up = lambda t, dt, layout=ttnn.ROW_MAJOR_LAYOUT: ttnn.from_torch(
            t, mesh_mapper=cm, layout=layout, device=mesh, dtype=dt
        )
        return dict(
            R=R,
            K=K,
            max_tokens=(max_m + t8.TILE - 1) // t8.TILE * t8.TILE,
            rows=ttnn.squeeze(up(rows, ttnn.uint32), 0),  # [1, R]
            weights=ttnn.squeeze(up(w.to(torch.bfloat16), ttnn.bfloat16, ttnn.TILE_LAYOUT), 0),  # [R, 1]
            counts=ttnn.squeeze(up(counts, ttnn.uint32), 0),  # [256]
            offsets=ttnn.squeeze(up(offsets, ttnn.uint32), 0),
            pair_rows=ttnn.squeeze(up(pair_rows, ttnn.uint32), 0),  # [1, 1, S, 8]
            pair_weights=ttnn.squeeze(up(pair_w.to(torch.bfloat16), ttnn.bfloat16), 0),  # [1, 1, S, 8]
            slots=[ttnn.squeeze(up(slots[:, k].contiguous(), ttnn.uint32), 0) for k in range(K)],  # K x [1, S]
        )

    # ---------------------------------------------------------------------------------------------- forward
    def forward(self, x: ttnn.Tensor, plan: dict, timer=None, combine="pcr"):
        """x: the residual ``[S/8, 1536]`` per chip (TILE bf16). Returns the routed output ``[S/8, 1536]``."""
        tick = timer or (lambda name: None)
        tick(None)
        xs = ttnn.all_gather(x, dim=-2, cluster_axis=0)  # [S, 1536]
        tick("ag_axis0_tokens")
        xf = ttnn.all_gather(xs, dim=-1, cluster_axis=1)  # [S, 6144]
        ttnn.deallocate(xs)
        tick("ag_axis1_hidden")
        xr = ttnn.to_layout(xf, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(xf)
        X = ttnn.embedding(plan["rows"], xr, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b)  # [1, R, 6144]
        ttnn.deallocate(xr)
        X = ttnn.reshape(X, (X.shape[-2], X.shape[-1]))
        tick("gather_rows")
        Y = ttnn.experimental.deepseek_prefill.unified_routed_expert_moe(
            X,
            plan["offsets"],
            plan["counts"],
            self.idx_table,
            self.gate_projs,
            self.up_projs,
            self.down_projs,
            max_dispatched_tokens_per_expert=plan["max_tokens"],
            compute_kernel_config=self.compute_kernel_config,
            activation=ttnn.RoutedExpertActivation.ClampedSiluGlu,
        )
        tick("experts")
        if combine == "pcr":
            # post_combine_reduce reading the expert rows in place (combine_row_index): per token, sum over its 8
            # top-k slots of weight * Y[row]; a slot whose pick is in another group has weight 0 and row R (skipped)
            Yr = ttnn.to_layout(Y, ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16)
            ttnn.deallocate(Y)
            acc = ttnn.experimental.deepseek_prefill.post_combine_reduce(
                Yr,
                plan["pair_weights"],
                combine_row_index=plan["pair_rows"],
                skip_row=plan["R"],
                expert_dim=3,
                output_memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            ttnn.deallocate(Yr)
            acc = ttnn.reshape(acc, (1, 1, acc.shape[-2], acc.shape[-1]))
            tick("combine")
            return self._reduce(acc, tick)
        Yw = ttnn.multiply(Y, plan["weights"], dtype=ttnn.bfloat16)
        ttnn.deallocate(Y)
        Yr = ttnn.to_layout(Yw, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(Yw)
        acc = None
        for idx in plan["slots"]:
            part = ttnn.embedding(idx, Yr, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)  # [1, S, 6144]
            if acc is None:
                acc = part
            else:
                acc = ttnn.add(acc, part, output_tensor=acc)
                ttnn.deallocate(part)
        ttnn.deallocate(Yr)
        acc = ttnn.reshape(acc, (1, 1, acc.shape[-2], acc.shape[-1]))
        tick("combine")
        return self._reduce(acc, tick)

    def _reduce(self, acc, tick):
        h = ttnn.reduce_scatter(acc, dim=3, cluster_axis=1)  # [S, 1536], sums the 4 groups
        ttnn.deallocate(acc)
        tick("rs_axis1_hidden")
        out = ttnn.reduce_scatter(h, dim=2, cluster_axis=0)  # [S/8, 1536], sums the 8 slices
        ttnn.deallocate(h)
        tick("rs_axis0_tokens")
        return ttnn.reshape(out, (out.shape[-2], out.shape[-1]))


class SyncTimer:
    """Wall time per named stage, each stage closed with a device synchronize (eager, not traced)."""

    def __init__(self, mesh_device):
        self.mesh, self.t, self.times = mesh_device, None, {}

    def __call__(self, name):
        ttnn.synchronize_device(self.mesh)
        now = time.perf_counter()
        if name is not None and self.t is not None:
            self.times[name] = self.times.get(name, 0.0) + (now - self.t)
        self.t = now
