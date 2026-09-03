# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

import ttnn

from ....layers.feedforward import ParallelFeedForward
from ....layers.module import Module, ModuleList
from ....layers.normalization import DistributedRMSNorm
from ....parallel.config import DiTParallelConfig
from ....parallel.manager import CCLManager
from ....utils.substate import rename_substate
from .agmm_config import agmm_block_size
from .attention_minimax_h3 import MiniMaxH3Attention


class MiniMaxH3TokenRefinerBlock(Module):
    """Plain pre-norm transformer block over the projected text stream.

    Much simpler than `MiniMaxH3TransformerBlock`: no AdaLN modulation and no rotary embedding.
    The residual updates are unconditional (`x = x + attn(norm1(x))`, `x = x + ff(norm2(x))`).
    The only masking is the optional `cu_window_seqlens` window boundaries, which fence the true
    tokens of a fixed-capacity text buffer off from its pad tail; an exactly-sized text stream
    passes None.

    The text stream is replicated on the SP axis and only fractured on TP. The refiner runs before the
    packed sequence is assembled and fractured, so every SP device holds the whole text stream and
    attention runs locally with plain SDPA rather than ring attention.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        ffn_dim: int,
        norm_eps: float = 1e-5,
        qk_norm_eps: float = 1e-5,
        mesh_device: ttnn.MeshDevice,
        ccl_manager: CCLManager,
        parallel_config: DiTParallelConfig,
        is_fsdp: bool = False,
    ) -> None:
        super().__init__()

        self.mesh_device = mesh_device
        self.ccl_manager = ccl_manager
        self.parallel_config = parallel_config
        self.tp_mesh_axis = parallel_config.tensor_parallel.mesh_axis
        self.tp_factor = parallel_config.tensor_parallel.factor

        fsdp_mesh_axis = parallel_config.sequence_parallel.mesh_axis if is_fsdp else None

        self.norm1 = DistributedRMSNorm(
            embedding_dim=hidden_size,
            norm_eps=norm_eps,
            norm_elementwise_affine=True,
            mesh_axis=self.tp_mesh_axis,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
        )
        self.attn = MiniMaxH3Attention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
            qk_norm_eps=qk_norm_eps,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
            parallel_config=parallel_config,
            is_fsdp=is_fsdp,
            is_sequence_parallel=False,
        )
        self.norm2 = DistributedRMSNorm(
            embedding_dim=hidden_size,
            norm_eps=norm_eps,
            norm_elementwise_affine=True,
            mesh_axis=self.tp_mesh_axis,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
        )
        self.ff = ParallelFeedForward(
            hidden_size,
            inner_dim=ffn_dim,
            activation_fn="swiglu",
            bias=False,
            mesh_device=mesh_device,
            mesh_axis=self.tp_mesh_axis,
            fsdp_mesh_axis=fsdp_mesh_axis,
            ccl_manager=ccl_manager,
        )

        self.mm_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=True,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        self.use_fused_agmm = ccl_manager.topology == ttnn.Topology.Ring and self.tp_factor > 1
        # ff1 packs gate and up together for the fused SwiGLU, so its per-device N is 2 * ffn_dim / tp.
        # M (the sequence length) sets the block's per_core_M and is only known at forward time.
        self._ff1_kn = (hidden_size, 2 * ffn_dim // self.tp_factor)

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        rename_substate(state, "ff.net.0.proj", "ff.ff1")
        rename_substate(state, "ff.net.2", "ff.ff2")

    def forward(self, prompt_1BLP: ttnn.Tensor, cu_window_seqlens: ttnn.Tensor | None = None) -> ttnn.Tensor:
        """prompt_1BLP: replicated on SP, fractured hidden_size on TP. Same on the way out.
        cu_window_seqlens: optional `[0, true_len, L]` window boundaries fencing off the pad tail
        (1-D integer device tensor, see `MiniMaxH3Attention.forward`); None when unpadded."""
        prompt_1BLP = ttnn.add(prompt_1BLP, self.attn(self.norm1(prompt_1BLP), cu_window_seqlens=cu_window_seqlens))

        normed = self.norm2(prompt_1BLP)
        # ff1 folds the TP all-gather into its matmul when parallel_config is passed; ff2 is
        # row-parallel and reduce-scatters back to TP-fractured.
        if not self.use_fused_agmm and self.tp_factor > 1:
            normed = self.ccl_manager.all_gather_persistent_buffer(normed, dim=3, mesh_axis=self.tp_mesh_axis)
        # ff1's block depends on per_core_M (hence M = the sequence length), only known here.
        ff1_block_size = agmm_block_size(*self._ff1_kn, normed.padded_shape[-2])
        ff_out = self.ff(
            normed,
            compute_kernel_config=self.mm_compute_kernel_config,
            parallel_config=self.parallel_config if self.use_fused_agmm else None,
            default_block_size=ff1_block_size,
        )
        return ttnn.add(prompt_1BLP, ff_out)


class MiniMaxH3TokenRefiner(Module):
    """Refines the projected text stream before it is scattered into the packed sequence."""

    def __init__(
        self,
        *,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        ffn_dim: int,
        num_layers: int,
        norm_eps: float = 1e-5,
        qk_norm_eps: float = 1e-5,
        final_norm_eps: float = 1e-5,
        mesh_device: ttnn.MeshDevice,
        ccl_manager: CCLManager,
        parallel_config: DiTParallelConfig,
        is_fsdp: bool = False,
    ) -> None:
        super().__init__()

        self.refiner_blocks = ModuleList(
            [
                MiniMaxH3TokenRefinerBlock(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    ffn_dim=ffn_dim,
                    norm_eps=norm_eps,
                    qk_norm_eps=qk_norm_eps,
                    mesh_device=mesh_device,
                    ccl_manager=ccl_manager,
                    parallel_config=parallel_config,
                    is_fsdp=is_fsdp,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = DistributedRMSNorm(
            embedding_dim=hidden_size,
            norm_eps=final_norm_eps,
            norm_elementwise_affine=True,
            mesh_axis=parallel_config.tensor_parallel.mesh_axis,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
        )

    def forward(self, prompt_1BLP: ttnn.Tensor, cu_window_seqlens: ttnn.Tensor | None = None) -> ttnn.Tensor:
        """prompt_1BLP: replicated on SP, fractured hidden_size on TP. Same on the way out.
        cu_window_seqlens: optional `[0, true_len, L]` window boundaries fencing off the pad tail
        (1-D integer device tensor, see `MiniMaxH3Attention.forward`); None when unpadded."""
        for block in self.refiner_blocks:
            prompt_1BLP = block(prompt_1BLP, cu_window_seqlens=cu_window_seqlens)
        return self.final_norm(prompt_1BLP)
