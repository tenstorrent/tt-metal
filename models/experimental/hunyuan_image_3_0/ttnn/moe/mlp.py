# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# TTNN implementation of the HunyuanImage-3.0 MLP (SwiGLU expert FFN / shared MLP).
# Mirrors ref/moe/mlp.py (hidden_act="silu"):
#     gu = gate_and_up_proj(x)          # [.., 2*I]
#     x1, x2 = gu.chunk(2, dim=-1)
#     out = down_proj(x1 * silu(x2))    # [.., H]
#
# Weight layout in the checkpoint (PyTorch nn.Linear -> [out, in]):
#     gate_and_up_proj.weight : [2*I, H]
#     down_proj.weight        : [H, I]
# ttnn.linear computes x @ W, so we store the transposes: [H, 2*I] and [I, H].

import ttnn
from models.common.lightweightmodule import LightweightModule

from ..cache import cache_file
from ..parallel_utils import decode_mm_program_config, moe_full_seq_mem_config, wide_mm_program_config


class HunyuanTtMLP(LightweightModule):
    """
    Single-device TTNN SwiGLU MLP (one routed expert, or the shared MLP).

    Args:
        device:      TTNN device.
        hidden_size: Model hidden size (e.g. 4096).
        state_dict:  Model state_dict (plain torch tensors).
        prefix:      Module prefix, e.g. ``model.layers.0.mlp.experts.0`` or
                     ``model.layers.0.mlp.shared_mlp``. The weights
                     ``{prefix}.gate_and_up_proj.weight`` and
                     ``{prefix}.down_proj.weight`` are read.
        weight_dtype: TTNN dtype for the linear weights (default bfloat16;
                      use ttnn.bfloat8_b for the BFP8 plan target).
    """

    def __init__(
        self,
        device,
        hidden_size: int,
        state_dict: dict,
        prefix: str,
        weight_dtype=ttnn.bfloat16,
        weight_cache_path=None,
    ):
        super().__init__()
        self.device = device
        self.hidden_size = hidden_size

        w_gu = state_dict[f"{prefix}.gate_and_up_proj.weight"]  # [2I, H]
        w_dn = state_dict[f"{prefix}.down_proj.weight"]  # [H, I]
        self.inter2 = w_gu.shape[0]  # 2I
        self.inter = self.inter2 // 2  # I

        self.w_gate_up = ttnn.as_tensor(
            w_gu.transpose(0, 1).contiguous(),  # [H, 2I]
            dtype=weight_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache_file(weight_cache_path, f"{prefix}.gate_and_up_proj.weight"),
        )
        self.w_down = ttnn.as_tensor(
            w_dn.transpose(0, 1).contiguous(),  # [I, H]
            dtype=weight_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache_file(weight_cache_path, f"{prefix}.down_proj.weight"),
        )

        # bf16/bf8 weights don't need HiFi4's 4 math passes or fp32 dest accumulation;
        # HiFi2 + bf16 accumulation roughly halves compute-bound matmul time. PCC-gated.
        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )

    def deallocate(self):
        """Free this MLP's device weights (used when streaming experts)."""
        ttnn.deallocate(self.w_gate_up)
        ttnn.deallocate(self.w_down)

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """
        Args:
            x: TTNN tensor [.., H] in TILE_LAYOUT.
        Returns:
            [.., H] tensor.
        """
        # The transient SwiGLU multiply follows the measured full-sequence L1->DRAM
        # gate so its full-S buffer never clashes with the ops' static CBs above bound.
        mc = moe_full_seq_mem_config(x.shape[1])
        # gate_up (M x H x 2I): small/mid M (Mt 1-7) wins ~1.3x with the 1D split-N
        # config vs auto; large M (Mt>=8) uses the 2D wide_mm grid.
        Mt_gu = (x.shape[-2] + 31) // 32
        gu_pc = (
            wide_mm_program_config(self.device, x.shape[-2], x.shape[-1], self.w_gate_up.shape[-1])
            if Mt_gu >= 8
            else decode_mm_program_config(self.device, x.shape[-2], x.shape[-1], self.w_gate_up.shape[-1])
        )
        gu = ttnn.linear(
            x,
            self.w_gate_up,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=mc,
            program_config=gu_pc,
        )  # [.., 2I]

        # SwiGLU: split into gate (x1) and up (x2) halves along the last dim.
        x1, x2 = ttnn.chunk(gu, 2, dim=-1)
        ttnn.deallocate(gu)
        # x1 * silu(x2), with SiLU folded into the multiply's LHS activation (one
        # fused BinaryNg instead of a separate silu Unary + multiply). SILU must apply
        # to x2 (the up half); multiply(x2, x1, a_acts=[SILU]) == x1 * silu(x2).
        h = ttnn.multiply(x2, x1, input_tensor_a_activations=[ttnn.UnaryOpType.SILU], memory_config=mc)
        ttnn.deallocate(x1)
        ttnn.deallocate(x2)

        # Down-proj: large M (Mt>=8) needs the 2D rectangular grid (auto mis-schedules
        # onto 110 cores at ~3% FLOP); small/mid M (Mt 1-7) wins ~1.6x with 1D split-N.
        Mt_dn = (h.shape[-2] + 31) // 32
        down_pc = (
            wide_mm_program_config(self.device, h.shape[-2], h.shape[-1], self.w_down.shape[-1])
            if Mt_dn >= 8
            else decode_mm_program_config(self.device, h.shape[-2], h.shape[-1], self.w_down.shape[-1])
        )
        out = ttnn.linear(
            h,
            self.w_down,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=mc,
            program_config=down_pc,
        )  # [.., H]
        ttnn.deallocate(h)
        return out
