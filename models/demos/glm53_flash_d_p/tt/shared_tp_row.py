# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3 shared expert split over each mesh row's chips (TP = the mesh's columns), for the token-split MoE block's
folded shared expert: the block's dispatch already put the row's C S tokens on every chip of the row, this computes
the chip's slice of the clamped SwiGLU MLP on them, and the combine carries each partial row to its token's owner.

    chip (r, c) holds intermediate columns [c I/C, (c + 1) I/C) (the same slice on every row)
    g, u = x @ Wg_c, x @ Wu_c             [C S, I/C] fp32 (one matmul, gate and up side by side)
    h    = silu(min(g, L)) * clamp(u, -L, L)
    o    = h @ Wd_c                        [C S, H] bf16, the partial (summed over the row by the combine)

Weights bf8 (w_dtype), HiFi3 (fidelity; HiFi2 shrank the output ~0.8% vs the golden) + fp32 accumulation, bf16 gate / up and h; the activation is one op
(combine_rf/glu_act.py: the SFPU's clamped SiLU GLU, limit 10). The block runs it before allocating its arena
(the fused op has its own barrier), so l1_budget (circular-buffer bytes per core) can be most of L1."""

from __future__ import annotations

import math

import torch

import ttnn
from models.demos.glm53_flash_d_p.tt.mm_configs import TILE_BYTES, _largest_div


def mm2d_fit(device, M, K, N, in0_dtype, w_dtype, out_dtype, budget):
    """A 2D-multicast config whose circular buffers fit budget bytes per core (mm_configs.mm2d with a budget)."""
    grid = device.compute_with_storage_grid_size()
    Mt, Kt, Nt = math.ceil(M / 32), K // 32, N // 32
    pm = math.ceil(Mt / grid.y)
    gy = math.ceil(Mt / pm)
    pn = math.ceil(Nt / grid.x)
    gx = math.ceil(Nt / pn)
    sw = _largest_div(pn, 4)
    sh = _largest_div(pm, max(1, 4 // sw))
    a, b, o = TILE_BYTES[in0_dtype], TILE_BYTES[w_dtype], TILE_BYTES[out_dtype]
    for bw in (8, 4, 2, 1):
        if Kt % bw == 0 and 2 * pm * bw * a + 2 * bw * pn * b + pm * pn * (o + 4096) <= budget:
            return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
                in0_block_w=bw,
                out_subblock_h=sh,
                out_subblock_w=sw,
                out_block_h=pm,
                out_block_w=pn,
                per_core_M=pm,
                per_core_N=pn,
                transpose_mcast=False,
                fused_activation=None,
                fuse_batch=True,
            )
    raise AssertionError(f"shared expert: no matmul config for {M} x {K} x {N} within {budget} B of L1")


class TtSharedTpRow:
    def __init__(
        self,
        mesh,
        w_gate: torch.Tensor,
        w_up: torch.Tensor,
        w_down: torch.Tensor,
        limit: float = 10.0,
        l1_budget: int = 700 * 1024,
        axis: int = 1,
        w_dtype=ttnn.bfloat8_b,
        fidelity=ttnn.MathFidelity.HiFi3,
    ):
        """HF weights: w_gate, w_up [I, H]; w_down [H, I]. axis: the group is each mesh row (1: slice c on chip (r,
        c)) or each column (0: slice r)."""
        self.mesh = mesh
        self.limit = float(limit)
        self.axis = axis
        self.w_dtype = w_dtype
        R, C = tuple(mesh.shape)
        G = C if axis == 1 else R
        inter, hidden = w_gate.shape
        assert inter % (G * 32) == 0, f"intermediate {inter} not tile aligned over {G} chips"
        self.per = per = inter // G
        gt, ut = w_gate.float().T, w_up.float().T  # [H, I]
        gu = torch.cat(
            [torch.cat([gt[:, c * per : (c + 1) * per], ut[:, c * per : (c + 1) * per]], -1) for c in range(G)], -1
        )
        self.w_gu = self._shard(gu.reshape(1, 1, hidden, 2 * inter), -1)
        self.w_down = self._shard(w_down.float().T.reshape(1, 1, inter, hidden), -2)
        assert float(limit) == 10.0, f"glu_act's SFPU kernel has the limit 10 compiled in, got {limit}"
        self.cfg = ttnn.types.BlackholeComputeKernelConfig(
            math_fidelity=fidelity, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
        )
        self.l1_budget = l1_budget
        from ttnn.bringup.combine_rf.glu_act import glu_act_prepare

        self.act = glu_act_prepare(mesh)

    def _shard(self, t, dim):
        R, C = tuple(self.mesh.shape)
        return ttnn.from_torch(
            t.to(torch.bfloat16),
            dtype=self.w_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                self.mesh, mesh_shape=(R, C), dims=(None, dim) if self.axis == 1 else (dim, None)
            ),
        )

    def _pc(self, x, w, out_dtype, budget):
        return mm2d_fit(x.device(), x.shape[-2], x.shape[-1], w.shape[-1], x.dtype, w.dtype, out_dtype, budget)

    def __call__(self, x: ttnn.Tensor, l1_budget: int | None = None) -> ttnn.Tensor:
        """x: [1, 1, M, H] bf16 TILE (rows of the row's tokens) -> the chip's partial [1, 1, M, H] bf16 TILE.
        l1_budget: the circular-buffer bytes per core the matmuls may take (default: the constructor's).
        (A row-major down output - the 1D config's untilize_out into y_se - writes rows but labels the tensor TILE and
        gives NaNs in an interleaved row-major buffer: not used.)"""
        budget = l1_budget or self.l1_budget
        mc = ttnn.DRAM_MEMORY_CONFIG
        gu = ttnn.linear(
            x,
            self.w_gu,
            dtype=ttnn.bfloat16,
            program_config=self._pc(x, self.w_gu, ttnn.bfloat16, budget),
            compute_kernel_config=self.cfg,
            memory_config=mc,
        )
        h = self.act(gu)  # (silu(min(g, 10)) * clamp(u, +-10): one op)
        ttnn.deallocate(gu)
        o = ttnn.linear(
            h,
            self.w_down,
            dtype=ttnn.bfloat16,
            program_config=self._pc(h, self.w_down, ttnn.bfloat16, budget),
            compute_kernel_config=self.cfg,
            memory_config=mc,
        )
        ttnn.deallocate(h)
        return o


class TtSharedFlat:
    """The same split shared expert on the flat expert op (one expert: this chip's I / G slice, bf8, clamped SiLU,
    HiFi2): it reads the group's rows straight from the row-major dispatch buffer by a fixed token index (the block's,
    own block first) and writes the partial row-major bf16 into the block's y_se, with no tilize / untilize. Runs
    before the block allocates its arena (the op allocates its own)."""

    def __init__(
        self,
        mesh,
        w_gate: torch.Tensor,
        w_up: torch.Tensor,
        w_down: torch.Tensor,
        limit: float = 10.0,
        axis: int = 1,
        max_rows: int = 8192,
        fidelity: int = 2,
        done=None,
        x_bf16: bool = True,
        layout=None,
    ):
        """max_rows: the most rows a call takes (G S); fidelity: the flat op's math_fidelity (2 = HiFi2). done: another
        TtSharedFlat's done words to share (persistent L1, one set for every layer)."""
        from ttnn.bringup.flat_routed_expert_ttnn.flat_expert import FlatRoutedExpert

        assert float(limit) == 10.0, f"the flat op's clamped SiLU has the limit 10 compiled in, got {limit}"
        self.mesh, self.axis, self.fidelity = mesh, axis, fidelity
        R, C = tuple(mesh.shape)
        G = C if axis == 1 else R
        inter, hidden = w_gate.shape
        assert inter % (G * 32) == 0
        self.per = per = inter // G
        pos = [(d % C) if axis == 1 else (d // C) for d in range(R * C)]  # (chip d's slice)

        def weights():
            sl = lambda i: slice(i * per, (i + 1) * per)  # noqa: E731
            return [
                [(w_gate[sl(p)].T.contiguous(), w_up[sl(p)].T.contiguous(), w_down[:, sl(p)].T.contiguous())]
                for p in pos
            ]

        self.flat = FlatRoutedExpert(
            mesh,
            weights,
            m=max_rows,
            H=hidden,
            I=per,
            gids=[[0] for _ in range(R * C)],
            n_global=1,
            wdtype="bf8",
            act="clamped_silu",
            pin=1,
            x_bf16=x_bf16,
            # (I / G <= 1024 is <= 32 tile columns: the plan's 2-3 subgrids are not built any more; one subgrid)
            layout=dict(single_subgrid=True) if layout is None else layout,
        )
        if done is not None:
            ttnn.deallocate(self.flat.done)
            self.flat.done = done
        self._rows = {}

    def _counts(self, n):
        if n not in self._rows:
            row = lambda v: ttnn.from_torch(  # noqa: E731
                torch.tensor([v], dtype=torch.int32),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
            )
            self._rows[n] = (row([n]), row([0]))
        return self._rows[n]

    def partials(self, xg, token_index, out):
        """xg: the dispatch buffer [N S, H] bf16 row-major; token_index [1, G S] uint32 per chip (its rows, in the
        partial's order); out: [G S, H] bf16 row-major (written)."""
        counts, regions = self._counts(int(token_index.shape[-1]))
        return self.flat(
            xg,
            counts,
            regions,
            token_index=token_index,
            y_row_major=True,
            down_fp32=True,
            math_fidelity=self.fidelity,
            output=out,
        )


class TtSharedUnified:
    """The same split shared expert on the bring-up unified routed expert op (ttnn.bringup.unified_routed_expert_moe,
    high_precision: bf16 x / intermediates / output, the requested fidelity with fp32 dest; ClampedSiluGlu fused; x
    row-major, tilized in the op): G "experts" sharing this chip's slice weights, one per owner block of the group,
    each reading its S rows at the block's slot of the dispatch buffer and writing them at the same rows of a TILE bf16
    output (the block untilizes the G blocks into its partial)."""

    def __init__(
        self,
        mesh,
        w_gate: torch.Tensor,
        w_up: torch.Tensor,
        w_down: torch.Tensor,
        limit: float = 10.0,
        axis: int = 1,
        fidelity=ttnn.MathFidelity.HiFi3,
    ):
        assert float(limit) == 10.0, f"ClampedSiluGlu has the limit 10 compiled in, got {limit}"
        self.mesh, self.axis = mesh, axis
        R, C = tuple(mesh.shape)
        self.G = G = C if axis == 1 else R
        inter, hidden = w_gate.shape
        assert inter % (G * 32) == 0
        self.per = per = inter // G
        pos = [(d % C) if axis == 1 else (d // C) for d in range(R * C)]  # (chip d's slice)
        sl = lambda i: slice(i * per, (i + 1) * per)  # noqa: E731

        def put(ts):  # (per device its [K, N] slice: rows of the stacked tensor)
            return ttnn.from_torch(
                torch.cat(ts, 0).to(torch.bfloat16),
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
            )

        self.w_gate = put([w_gate[sl(p)].T.float() for p in pos])  # [H, per] each
        self.w_up = put([w_up[sl(p)].T.float() for p in pos])
        self.w_down = put([w_down[:, sl(p)].T.float() for p in pos])  # [per, H]
        self.cfg = ttnn.types.BlackholeComputeKernelConfig(
            math_fidelity=fidelity, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
        )

    def partials(self, xg, regions, counts, gidx, S):
        """xg: the dispatch buffer [N S, H] bf16 row-major; regions / counts [1, G] uint32 per chip (block j's first row
        and S), gidx [G] uint32 (0 .. G - 1) -> [N S, H] bf16 TILE with the G blocks' partials at their rows."""
        G = self.G
        return ttnn.bringup.unified_routed_expert_moe(
            xg,
            regions,
            counts,
            gidx,
            [self.w_gate] * G,
            [self.w_up] * G,
            [self.w_down] * G,
            max_dispatched_tokens_per_expert=S,
            compute_kernel_config=self.cfg,
            activation=ttnn.bringup.RoutedExpertActivation.ClampedSiluGlu,
            high_precision=True,
        )
