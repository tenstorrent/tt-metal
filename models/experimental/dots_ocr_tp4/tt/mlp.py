# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""TP4 SwiGLU MLP for dots.ocr prefill (replicated-hidden Megatron design).

    gate = x @ gate_proj.T        # column-parallel, output sharded on I
    up   = x @ up_proj.T          # column-parallel, output sharded on I
    act  = silu(gate) * up        # elementwise, per-chip on its I/ndev slice
    out  = act @ down_proj.T      # row-parallel, partial sums per chip
    out  = all_reduce(out)        # -> full replicated hidden

Input  x   : replicated [B, S, H]
Output out : replicated [B, S, H]
"""

import torch
import ttnn

from models.experimental.dots_ocr_tp4.tt.common import (
    all_reduce,
    matmul_m_dim,
    tp_degree,
    prefill_matmul_2d_config,
    shard_to_mesh,
    tp4_lossy_matmul_dtype,
)
from models.experimental.tt_symbiote.core.module import TTNNModule


def _mlp_prefill_pc(mesh_device, m: int, k: int, n: int, fp32_dest: bool):
    """Hardware-swept program configs for the prefill MLP matmuls on the Blackhole
    11x10 grid (both run full-grid vs the adaptive helper's ~64-80 cores):

      * gate_up (K=1536, N=4480): in0_block_w=8, per_core_N=13 -> ~202 us (~15%
        faster than the helper). [in0_block_w=12 was 190 us but its larger in0 CB
        clashes with L1 in the full pipeline.]
      * down    (K=2240, N=1536): in0_block_w=7, per_core_N=5  -> ~76 us  (~17%
        faster). [in0_block_w=14 was 73 us but risks the same L1 clash.]

    Both use ``per_core_M = ceil(M_tiles/10) = 9`` for the 2816 (88 tiles) and the
    OCR 2848 (89 tiles) prefill, and the chosen in0_block_w keeps per-core CBs <=
    the proven helper so they always fit. Falls back to the adaptive helper off
    these shapes / grid."""
    grid = mesh_device.compute_with_storage_grid_size()
    mt = (m + 31) // 32
    if int(grid.x) >= 11 and int(grid.y) >= 10 and 81 <= mt <= 90:
        per_core_m = (mt + 9) // 10  # == 9 across the 2816..2848 prefill range
        tuned = None
        if k == 1536 and n == 4480:
            tuned = (8, 13)  # (in0_block_w, per_core_N)
        elif k == 2240 and n == 1536:
            tuned = (7, 5)
        if tuned is not None:
            in0_bw, per_core_n = tuned
            return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(11, 10),
                in0_block_w=in0_bw,
                out_subblock_h=3,  # per_core_m(9) % 3 == 0; 3*1 <= dst budget
                out_subblock_w=1,
                out_block_h=per_core_m,
                out_block_w=per_core_n,
                per_core_M=per_core_m,
                per_core_N=per_core_n,
                transpose_mcast=False,
                fused_activation=None,
                fuse_batch=True,
            )
    # On Wormhole (8x8) gate_up MUST get an explicit config. The compute kernel is
    # the perf lever (LoFi + no fp32_dest_acc on gate_up_compute), but dropping
    # fp32_dest_acc doubled the dst budget, so ttnn's auto heuristic now widens
    # out_block_w to the full per_core_N=18 -- whose out/interm CBs (~440KB+235KB)
    # overflow L1 in the full 28-layer pipeline (~60KB over; passes standalone in
    # test_mlp_gateup_sweep where no other L1 buffers are resident). We pin the same
    # 8x8/per_core 12x18 geometry but sub-block the output to out_block_w=9, which
    # halves out_CB/interm0_CB/in1_CB (~390KB saved) for the same compute -- far more
    # headroom than the overflow. in0_block_w=6 is the swept gate_up optimum (bw8
    # clashes the in-block CB). Falls back to the auto helper off this shape/grid.
    if int(grid.x) == 8 and int(grid.y) == 8 and k == 1536 and n == 4480 and m % 32 == 0:
        mt, kt, nt = m // 32, k // 32, n // 32
        if 81 <= mt <= 90 and kt % 6 == 0:
            per_core_m = (mt + 7) // 8  # ceil over grid.y=8 -> 12 for the 89-tile prefill
            per_core_n = (nt + 7) // 8  # ceil over grid.x=8 -> 18
            out_block_w = per_core_n // 2  # 9; halves the output/interm CBs to fit L1
            return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
                in0_block_w=6,
                out_subblock_h=2,  # per_core_m(12) % 2 == 0
                out_subblock_w=3,  # out_block_w(9) % 3 == 0; 2*3 <= dst(8)
                out_block_h=per_core_m,
                out_block_w=out_block_w,
                per_core_M=per_core_m,
                per_core_N=per_core_n,
                transpose_mcast=False,
                fused_activation=None,
                fuse_batch=True,
            )
    return prefill_matmul_2d_config(mesh_device, m, k, n, fp32_dest=fp32_dest)


class DotsOCRMLPTP4(TTNNModule):
    def __init__(self, mesh_device, config, weight_dtype=ttnn.bfloat16):
        super().__init__()
        self.mesh_device = mesh_device
        self.config = config
        self.weight_dtype = weight_dtype
        self.tp_degree = max(1, tp_degree(mesh_device))
        self.gate_up_w = None  # fused [H, 2*I/nd] per chip: [gate_shard | up_shard]
        self.down_w = None
        # Precision recipe matched to the production dots.ocr prefill profile:
        #   gate/up : BF16 x BFP4 -> BFP8 @ HiFi2
        #   silu*mul: BFP8 x BFP8 -> BFP8
        #   down    : BFP8 x BFP4 -> BFP8 @ LoFi
        self.gate_up_weight_dtype = tp4_lossy_matmul_dtype()
        self.down_weight_dtype = tp4_lossy_matmul_dtype()
        # gate_up: dropping fp32_dest_acc (711->525 us, the dominant lever; it also
        # halves dst so the tuned bw8 config can't fit with it) + LoFi (525->439 us)
        # gives the lowest device time and still passes the block PCC. See
        # test_mlp_gateup_sweep. Paired with fp32_dest=False in the program config.
        self.gate_up_compute = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
        self.down_compute = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )

    @classmethod
    def from_torch(
        cls,
        mesh_device,
        config,
        torch_mlp,
        weight_dtype=ttnn.bfloat16,
        gate_up_weight_dtype=None,
        down_weight_dtype=None,
    ):
        m = cls(mesh_device, config, weight_dtype=weight_dtype)
        m.set_weight_dtype(gate_up_dtype=gate_up_weight_dtype, down_dtype=down_weight_dtype)
        m.load_weights(torch_mlp)
        m.to_device(mesh_device)
        m._preprocessed_weight = True
        m._weights_on_device = True
        return m

    def set_weight_dtype(self, gate_up_dtype=None, down_dtype=None):
        if gate_up_dtype is not None:
            self.gate_up_weight_dtype = gate_up_dtype
        if down_dtype is not None:
            self.down_weight_dtype = down_dtype
        return self

    def load_weights(self, torch_mlp):
        nd = self.tp_degree
        I = self.config.intermediate_size
        assert I % nd == 0, f"intermediate_size {I} not divisible by {nd}"
        i_shard = I // nd

        # ttnn.linear computes x @ W with W laid out [K, N]; torch weights are
        # [out, in], so transpose to [in, out].
        gate_w = torch_mlp.gate_proj.weight.data.t().contiguous()  # [H, I]
        up_w = torch_mlp.up_proj.weight.data.t().contiguous()  # [H, I]
        down_w = torch_mlp.down_proj.weight.data.t().contiguous()  # [I, H]

        # Fuse gate+up into ONE column-parallel matmul. Interleave per chip so a
        # single shard(dim=-1) lands [gate_shard_c | up_shard_c] on chip c, and a
        # chunk(2) on the output recovers gate/up. Columns ordered:
        #   [chip0_gate | chip0_up | chip1_gate | chip1_up | ...].
        blocks = []
        for c in range(nd):
            blocks.append(gate_w[:, c * i_shard : (c + 1) * i_shard])
            blocks.append(up_w[:, c * i_shard : (c + 1) * i_shard])
        gate_up = torch.cat(blocks, dim=-1).contiguous()  # [H, 2I]
        self.gate_up_w = shard_to_mesh(gate_up, self.mesh_device, dim=-1, dtype=self.gate_up_weight_dtype)
        # Row-parallel down: shard contraction dim (I = K, dim 0) across chips.
        self.down_w = shard_to_mesh(down_w, self.mesh_device, dim=0, dtype=self.down_weight_dtype)

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        # Decode (seq==1) keeps the tiny M=1 activations L1-resident; prefill stays
        # DRAM-interleaved (its large CBs clash with L1-staged in0 -> L1 OOM).
        is_decode = int(x.shape[-2]) == 1
        act_mc = ttnn.L1_MEMORY_CONFIG if is_decode else ttnn.DRAM_MEMORY_CONFIG
        if x.layout != ttnn.TILE_LAYOUT:
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT, memory_config=act_mc)

        m = matmul_m_dim(x)
        K = int(x.shape[-1])
        gate_up_n = int(self.gate_up_w.shape[-1])  # 2 * I/nd
        gate_up_pc = _mlp_prefill_pc(self.mesh_device, m, K, gate_up_n, fp32_dest=False)

        # Single fused gate+up matmul, then split: per chip the output is
        # [gate_shard | up_shard], so chunk(2, dim=-1) recovers them. in0 stays
        # DRAM-interleaved -- L1-staging it clashed with this matmul's large CBs
        # on the 28-layer run (L1 OOM); the tuned in0_block_w is the real win.
        gate_up = ttnn.linear(
            x,
            self.gate_up_w,
            dtype=ttnn.bfloat8_b,
            memory_config=act_mc,
            compute_kernel_config=self.gate_up_compute,
            program_config=gate_up_pc,
        )
        gate, up = ttnn.chunk(gate_up, 2, dim=-1)
        ttnn.deallocate(gate_up)
        act = ttnn.mul(
            gate,
            up,
            input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
            dtype=ttnn.bfloat8_b,
            memory_config=act_mc,
        )
        ttnn.deallocate(gate)
        ttnn.deallocate(up)

        I_shard = int(act.shape[-1])
        H = int(self.down_w.shape[-1])
        down_pc = _mlp_prefill_pc(self.mesh_device, m, I_shard, H, fp32_dest=False)
        out = ttnn.linear(
            act,
            self.down_w,
            dtype=ttnn.bfloat8_b,
            memory_config=act_mc,
            compute_kernel_config=self.down_compute,
            program_config=down_pc,
        )
        ttnn.deallocate(act)

        out = all_reduce(out, self.mesh_device, output_memory_config=act_mc)
        return out
