# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import ttnn

from ....blocks.attention_opt import Attention
from ....layers.linear import AGMMOverride, ColParallelLinear, RowParallelLinear
from ....layers.module import Module
from ....layers.normalization import DistributedLayerNorm
from ....utils.matmul import FusedMMRSConfig, register_fused_mmrs_configs
from ....utils.tensor import prepare_weight_for_concatenated_input

if TYPE_CHECKING:
    from ....parallel.config import DiTParallelConfig
    from ....parallel.manager import CCLManager

HIDDEN = 3072
HEADS = 24
HEAD_DIM = 128
MLP_RATIO = 3
SDPA_Q_CHUNK_MAX = 128


def _agmm(grid: tuple[int, int], blocks: tuple[int, int, int], subblock: tuple[int, int], *, transpose: bool):
    return AGMMOverride(
        ttnn.MinimalMatmulConfig(
            M_block_size=blocks[0],
            K_block_size=blocks[1],
            N_block_size=blocks[2],
            subblock_h=subblock[0],
            subblock_w=subblock[1],
            compute_with_storage_grid_size=ttnn.CoreCoord(*grid),
        ),
        transpose,
    )


# Swept on a BH Galaxy 4x8 (SP8 on axis 1 x TP4 on axis 0, Ring, 2 links) with the production op calls; device
# kernel duration, median over the 32 devices, us. Rows per device: 576 = 1024x1024 with 512 text rows,
# 1024 = 1024x1024 with 4096 text rows, 1536 = the same plus a 1024x1024 reference. Applied only on the 12x10
# Blackhole device grid they were measured on; every other shape keeps the shared resolution.
# QKV (M, K, N/dev) all-gather-matmul, 3 output chunks.
QKV_AGMM_CONFIGS = {
    (576, HIDDEN, 2304): _agmm((11, 9), (2, 8, 7), (2, 1), transpose=False),  # 125.6 -> 77.7
    (1024, HIDDEN, 2304): _agmm((12, 9), (3, 6, 8), (1, 4), transpose=True),  # 137.9 -> 129.2
    (1536, HIDDEN, 2304): _agmm((12, 8), (4, 8, 9), (4, 1), transpose=True),  # 173.9 -> 168.6
}
# proj_mlp (M, K, N/dev) all-gather-matmul with fused SwiGLU (N is the packed [up | gate] width).
MLP_AGMM_CONFIGS = {
    (576, HIDDEN, 4608): _agmm((11, 9), (2, 8, 8), (1, 4), transpose=False),  # 288.7 -> 149.7
    (1024, HIDDEN, 4608): _agmm((11, 8), (4, 4, 14), (2, 2), transpose=False),  # 328.5 -> 203.1
    (1536, HIDDEN, 4608): _agmm((12, 9), (4, 4, 16), (1, 4), transpose=True),  # 308.2 -> 268.0
}
# proj_out (M, K/dev, N) fused matmul + reduce-scatter + addcmul over the concatenated [attn | mlp] input.
# FusedMMRSConfig(mm grid, M_block, K_block, N_block, sub_h, sub_w, buffers, chunk width, workers/link, window)
register_fused_mmrs_configs(
    {
        ttnn.CoreCoord(12, 10): {
            # 215.9 -> 145.9
            (576, HIDDEN, HIDDEN): FusedMMRSConfig(ttnn.CoreCoord(12, 8), 3, 4, 8, 1, 4, None, 1, None, None),
            # 236.7 -> 190.7; the reduce-scatter runs on the 3 rows above the 7-row matmul grid
            (1024, HIDDEN, HIDDEN): FusedMMRSConfig(ttnn.CoreCoord(12, 7), 3, 8, 8, 1, 4, None, 1, 8, 2),
            # 267.4 -> 228.0
            (1536, HIDDEN, HIDDEN): FusedMMRSConfig(ttnn.CoreCoord(12, 8), 3, 8, 8, 1, 4, None, 1, 4, 2),
        }
    }
)


class Flux3KleinTransformerBlock(Module):
    """Single-stream block: one LayerNorm modulated by (shift, scale + 1), QKV + SwiGLU MLP from the same input,
    and one output projection over [attn | mlp] gated back into the residual."""

    def __init__(
        self,
        *,
        device: ttnn.MeshDevice,
        ccl_manager: CCLManager,
        parallel_config: DiTParallelConfig,
        is_fsdp: bool = False,
    ) -> None:
        super().__init__()

        tp_axis = parallel_config.tensor_parallel.mesh_axis
        mlp_hidden_dim = MLP_RATIO * HIDDEN
        fsdp_mesh_axis = parallel_config.sequence_parallel.mesh_axis if is_fsdp else None

        self.attn = Attention(
            query_dim=HIDDEN,
            head_dim=HEAD_DIM,
            heads=HEADS,
            out_dim=HIDDEN,
            added_kv_proj_dim=0,
            pre_only=True,
            proj_bias=False,
            eps=1e-6,
            mesh_device=device,
            ccl_manager=ccl_manager,
            parallel_config=parallel_config,
            padding_config=None,
            use_spatial_weights_for_prompt=True,
            per_head_norm=True,
            is_fsdp=is_fsdp,
            shard_prompt=False,
        )
        self.attn.sdpa_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
        )
        self.attn.ring_sdpa_exp_approx = True
        # fp32 accumulation needs the smaller q chunks to fit L1
        self.attn.ring_sdpa_chunk_size_map = {
            key: {length: (min(q, SDPA_Q_CHUNK_MAX), k) for length, (q, k) in table.items()}
            for key, table in Attention.ring_sdpa_chunk_size_map.items()
        }
        self.attn.default_ring_sdpa_chunk_size = {
            length: (min(q, SDPA_Q_CHUNK_MAX), k) for length, (q, k) in Attention.default_ring_sdpa_chunk_size.items()
        }

        self.norm = DistributedLayerNorm(
            HIDDEN,
            norm_eps=1e-6,
            norm_elementwise_affine=False,
            bias=False,
            mesh_axis=tp_axis,
            mesh_device=device,
            ccl_manager=ccl_manager,
        )
        self.proj_mlp = ColParallelLinear(
            HIDDEN,
            mlp_hidden_dim,
            bias=False,
            activation_fn="swiglu",
            mesh_device=device,
            mesh_axis=tp_axis,
            fsdp_mesh_axis=fsdp_mesh_axis,
            ccl_manager=ccl_manager,
        )
        self.proj_out = RowParallelLinear(
            HIDDEN + mlp_hidden_dim,
            HIDDEN,
            bias=False,
            mesh_device=device,
            mesh_axis=tp_axis,
            ccl_manager=ccl_manager,
            fsdp_mesh_axis=fsdp_mesh_axis,
        )

        grid = device.compute_with_storage_grid_size()
        if (grid.x, grid.y) == (12, 10):
            self.attn.to_qkv.agmm_overrides = QKV_AGMM_CONFIGS
            self.proj_mlp.agmm_overrides = MLP_AGMM_CONFIGS

        self._mlp_hidden_dim = mlp_hidden_dim
        self._tp_factor = parallel_config.tensor_parallel.factor
        self._ccl_manager = ccl_manager
        self._parallel_config = parallel_config
        self._mesh_device = device
        self._rows = None

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        for src, dst in (("q_proj", "attn.to_q"), ("k_proj", "attn.to_k"), ("v_proj", "attn.to_v")):
            if f"{src}.weight" in state:
                state[f"{dst}.weight"] = state.pop(f"{src}.weight")
        for src, dst in (
            ("norm.query_norm.scale", "attn.norm_q.weight"),
            ("norm.key_norm.scale", "attn.norm_k.weight"),
        ):
            if src in state:
                state[dst] = state.pop(src)
        if "mlp_in.weight" in state:
            w = state.pop("mlp_in.weight")
            # the reference packs [gate | up], the fused SwiGLU expects [up | gate]
            state["proj_mlp.weight"] = torch.roll(w, shifts=w.shape[0] // 2, dims=0)
        if "attn_out.weight" in state and "mlp_out.weight" in state:
            state["proj_out.weight"] = prepare_weight_for_concatenated_input(
                torch.cat([state.pop("attn_out.weight"), state.pop("mlp_out.weight")], dim=1),
                [HIDDEN, self._mlp_hidden_dim],
                device_count=self._tp_factor,
            )

    def forward(
        self,
        x: ttnn.Tensor,
        *,
        rope: tuple[ttnn.Tensor, ttnn.Tensor],
        mod: tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor],
        logical_length: int,
    ) -> ttnn.Tensor:
        # Attention keeps the ring SDPA worker grid of its first forward; re-pick it when the row count changes
        if x.shape[1] != self._rows:
            self.attn.ring_sdpa_worker_grid = None
            self._rows = x.shape[1]

        shift, scale_plus_one, gate = mod
        x_norm = ttnn.squeeze(self.norm(ttnn.unsqueeze(x, 0), dynamic_weight=scale_plus_one, dynamic_bias=shift), 0)
        attn, _ = self.attn.forward(sequence_1=x_norm, sequence_1_rope=rope, sequence_1_length=logical_length)
        mlp = self.proj_mlp(
            x_norm,
            parallel_config=self._parallel_config,
            use_heuristic_mmcfg=True,
            core_grid=Attention.get_core_grid(x_norm.shape[-2], self._mesh_device.compute_with_storage_grid_size()),
        )
        if self._ccl_manager.topology == ttnn.Topology.Ring:
            return self.proj_out.forward_fused_addcmul([attn, mlp], x, gate, scalar=1.0)
        return ttnn.addcmul(x, self.proj_out([attn, mlp]), gate)
