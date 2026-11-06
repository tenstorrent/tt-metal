# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

import ttnn

from ....blocks.attention_opt import Attention
from ....layers.linear import ColParallelLinear, RowParallelLinear
from ....layers.module import Module
from ....layers.normalization import DistributedLayerNorm
from ....parallel.config import DiTParallelConfig
from ....parallel.manager import CCLManager
from ....utils.tensor import prepare_weight_for_concatenated_input

HIDDEN = 3072
HEADS = 24
HEAD_DIM = 128
MLP_RATIO = 3
SDPA_Q_CHUNK_MAX = 128
# Ring SDPA (q, k) chunks by the PER-DEVICE row count the kernel tiles (SP8 x TP4). The shared table is keyed by the
# logical length, which includes the caption's valid text rows, so no served Klein length ever matched it. Each
# (head, q chunk) is one unit of work on the 108-core SDPA grid; q is picked so the units fill whole rounds, and k
# divides the shard. Lengths not listed keep the shared rule (q clamped to SDPA_Q_CHUNK_MAX).
RING_SDPA_CHUNKS_BY_LOCAL_ROWS: dict[int, tuple[int, int]] = {
    # 2048x2048: 6 heads x 16 q chunks = 96 units, one round (q 128: 120 units, two rounds); -11.5 % denoise
    2560: (160, 512),
    # 2048x2048 + 1 reference: 6 x 16 = 96 units, one round (q 128: 144 units, two rounds); -11.7 % denoise
    3072: (192, 512),
    # 2048x2048 + 1 reference with a 1024-row text bucket: 6 x 17 = 102 units, one round
    2688: (160, 512),
    # 4096x4096 with a 1024-row text bucket: k 640 divides the shard (13 chunks)
    8320: (128, 640),
    # 2048x2048 + 1 reference with a 2048-row text bucket: 6 x 18 = 108 units, one round; -16.4 % denoise vs today
    2816: (160, 512),
    # 4096x4096 + 1 reference with a 2048-row text bucket: k 640 divides the shard (14 chunks); -1.7 % denoise
    8960: (128, 640),
}


class Flux3KleinAttention(Attention):
    # Ring SDPA on the streaming compute path while keeping fp32 dest accumulation (bf16 score block): 4K model time
    # -13.6 %, 2K + ref -11.4 %; not bit-identical, golden mean 43.57 vs 43.93 dB (perf exp 11). False restores the
    # compute_common.hpp path (fp32 score block).
    ring_sdpa_streaming_fp32_dest = True

    def forward(self, *, sequence_1: ttnn.Tensor, **kwargs):
        self._local_rows = int(sequence_1.shape[-2])
        return super().forward(sequence_1=sequence_1, **kwargs)

    def get_ring_sdpa_program_config(self, per_device_seq_len: int) -> ttnn.SDPAProgramConfig:
        chunks = RING_SDPA_CHUNKS_BY_LOCAL_ROWS.get(getattr(self, "_local_rows", None))
        if chunks is None:
            config = super().get_ring_sdpa_program_config(per_device_seq_len)
        else:
            key = ("local", self._local_rows)
            if key not in self.ring_sdpa_program_config:
                self.ring_sdpa_program_config[key] = ttnn.SDPAProgramConfig(
                    compute_with_storage_grid_size=self.get_ring_sdpa_core_grid(per_device_seq_len),
                    q_chunk_size=chunks[0],
                    k_chunk_size=chunks[1],
                    exp_approx_mode=self.ring_sdpa_exp_approx,
                )
            config = self.ring_sdpa_program_config[key]
        # The cache dict is per instance, so setting the field here touches only Klein's configs.
        config.streaming_fp32_dest = self.ring_sdpa_streaming_fp32_dest
        return config


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

        self.attn = Flux3KleinAttention(
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

        self._mlp_hidden_dim = mlp_hidden_dim
        self._tp_factor = parallel_config.tensor_parallel.factor
        self._ccl_manager = ccl_manager
        self._parallel_config = parallel_config
        self._mesh_device = device

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
        logical_length: int | ttnn.Tensor,
    ) -> ttnn.Tensor:
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
