# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole

from ..layers.linear import ColParallelLinear
from ..layers.module import Module, Parameter, UnregisteredModule
from ..layers.normalization import DistributedRMSNorm
from ..utils.matmul import get_matmul_config, get_matmul_core_grid
from ..utils.mochi import get_rot_transformation_mat
from ..utils.padding import PaddingConfig, pad_weight_tensor
from ..utils.substate import pop_substate

if TYPE_CHECKING:
    pass

    from ..parallel.config import DiTParallelConfig
    from ..parallel.manager import CCLManager


# Best compute core grid per shape from flux2 dual-grid sweep (2026-05-27).
# See flux2_dual_grid_mm_sweep.md "Best grid per shape".
_FLUX2_MATMUL_CORE_GRIDS: dict[tuple[int, int, int], tuple[int, int]] = {
    (1024, 128, 768): ttnn.CoreCoord(11, 10),
    (512, 15360, 768): ttnn.CoreCoord(12, 9),
    (1024, 6144, 768): ttnn.CoreCoord(11, 10),
    (512, 6144, 768): ttnn.CoreCoord(12, 9),
    (1024, 6144, 4608): ttnn.CoreCoord(12, 9),
    (512, 6144, 4608): ttnn.CoreCoord(12, 9),
    (1024, 6144, 2304): ttnn.CoreCoord(12, 9),
    (512, 6144, 2304): ttnn.CoreCoord(12, 9),
}


# adapted from https://github.com/huggingface/diffusers/blob/v0.31.0/src/diffusers/models/attention_processor.py
class Attention(Module):
    # SDPA chunk sizes keyed by (is_blackhole, sp_factor, tp_factor). Resolution priority for
    # non-ring: caller overrides > class map > constructor args > class default.
    sdpa_chunk_size_map: dict[tuple, tuple[int, int]] = {}
    default_sdpa_chunk_size: tuple[int, int] = (128, 512)

    ring_sdpa_chunk_size_map: dict[tuple, tuple[int, int]] = {
        # -1 is the default resolution.
        (False, 2, 4): {-1: (256, 256)},
        (False, 8, 4): {-1: (256, 256)},
        (True, 2, 2): {-1: (128, 512)},
        (True, 4, 8): {-1: (256, 512), 4096: (128, 512), 4096 * 4: (256, 512)},
        (True, 8, 4): {-1: (256, 512)},
    }
    default_ring_sdpa_chunk_size: tuple[int, int] = {-1: (256, 256)}

    def __init__(
        self,
        *,
        query_dim: int,
        head_dim: int,
        heads: int,
        out_dim: int,
        added_kv_proj_dim: int,
        context_pre_only: bool = False,
        pre_only: bool = False,
        use_spatial_weights_for_prompt: bool = False,
        context_head_scaling: bool = False,
        proj_bias: bool = True,
        eps: float,
        mesh_device: ttnn.MeshDevice,
        ccl_manager: CCLManager | None,
        parallel_config: DiTParallelConfig,
        padding_config: PaddingConfig | None,
        k_chunk_size: int | None = None,
        q_chunk_size: int | None = None,
        sdpa_chunk_size_overrides: dict | None = None,
        per_head_norm: bool = False,
        is_fsdp: bool = False,
        shard_prompt: bool = False,
    ) -> None:
        super().__init__()

        self.head_dim = head_dim
        self.pre_only = pre_only
        self.mesh_device = mesh_device
        self.ccl_manager = ccl_manager
        self.parallel_config = parallel_config
        self.shard_prompt = shard_prompt
        self.padding_config = padding_config
        self.use_spatial_weights_for_prompt = use_spatial_weights_for_prompt
        # True = per-head RMS (head_dim divisor, stats stay local); False = global RMS over
        # padded_inner_dim (AG of stats).
        self.per_head_norm = per_head_norm

        self.padded_heads = padding_config.target_heads if padding_config is not None else heads
        self.n_local_heads = self.padded_heads // self.parallel_config.tensor_parallel.factor

        tp_axis = parallel_config.tensor_parallel.mesh_axis
        padded_inner_dim = head_dim * self.padded_heads

        # FSDP: shard weights on sequence parallel axis to reduce memory
        fsdp_mesh_axis = parallel_config.sequence_parallel.mesh_axis if is_fsdp else None

        common_args = dict(mesh_device=mesh_device, ccl_manager=ccl_manager, fsdp_mesh_axis=fsdp_mesh_axis)

        full_grid = self.mesh_device.compute_with_storage_grid_size()

        # Reserve last row for CCL.
        self.sdpa_worker_grid = (full_grid.x, full_grid.y - 1)

        chunk_lookup = {**self.sdpa_chunk_size_map, **(sdpa_chunk_size_overrides or {})}
        resolved_q_chunk, resolved_k_chunk = chunk_lookup.get(
            (
                is_blackhole(),
                parallel_config.sequence_parallel.factor,
                parallel_config.tensor_parallel.factor,
            ),
            (
                q_chunk_size if q_chunk_size is not None else self.default_sdpa_chunk_size[0],
                k_chunk_size if k_chunk_size is not None else self.default_sdpa_chunk_size[1],
            ),
        )

        self.sdpa_program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=self.sdpa_worker_grid,
            q_chunk_size=resolved_q_chunk,
            k_chunk_size=resolved_k_chunk,
            exp_approx_mode=False,  # NOTE: False is more correct
        )

        self.ring_sdpa_worker_grid = (full_grid.x, full_grid.y - 1)
        self.ring_sdpa_program_config = {}
        self.sdpa_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            self.mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,  # NOTE: Set to True if there's a correctness issue
        )
        self.mm_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            self.mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=True,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

        # chunks=3 splits QKV inside minimal_matmul_split (faster than matmul + ttnn.chunk).
        self.to_qkv = ColParallelLinear(
            query_dim, 3 * padded_inner_dim, bias=proj_bias, mesh_axis=tp_axis, chunks=3, **common_args
        )

        # Fuses head-split + RMSNorm + RoPE. Weight is tiled per-head → [padded_inner_dim] in
        # _prepare_torch_state.
        self.norm_q = DistributedRMSNorm(
            embedding_dim=padded_inner_dim,
            norm_eps=eps,
            mesh_axis=tp_axis,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
        )
        self.norm_k = DistributedRMSNorm(
            embedding_dim=padded_inner_dim,
            norm_eps=eps,
            mesh_axis=tp_axis,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
        )

        self.to_out = (
            ColParallelLinear(padded_inner_dim, out_dim, bias=proj_bias, mesh_axis=tp_axis, **common_args)
            if not self.pre_only
            else None
        )

        if use_spatial_weights_for_prompt:
            self.add_qkv_proj = UnregisteredModule(self.to_qkv)
            self.norm_added_q = UnregisteredModule(self.norm_q)
            self.norm_added_k = UnregisteredModule(self.norm_k)
            self.to_add_out = UnregisteredModule(self.to_out) if self.to_out is not None else None
        elif added_kv_proj_dim > 0:
            self.add_qkv_proj = ColParallelLinear(
                added_kv_proj_dim, 3 * padded_inner_dim, bias=proj_bias, mesh_axis=tp_axis, chunks=3, **common_args
            )

            self.norm_added_q = DistributedRMSNorm(
                embedding_dim=padded_inner_dim,
                norm_eps=eps,
                mesh_axis=tp_axis,
                mesh_device=mesh_device,
                ccl_manager=ccl_manager,
            )
            self.norm_added_k = DistributedRMSNorm(
                embedding_dim=padded_inner_dim,
                norm_eps=eps,
                mesh_axis=tp_axis,
                mesh_device=mesh_device,
                ccl_manager=ccl_manager,
            )

            self.to_add_out = (
                ColParallelLinear(padded_inner_dim, out_dim, bias=proj_bias, mesh_axis=tp_axis, **common_args)
                if not context_pre_only
                else None
            )
        else:
            self.add_qkv_proj = None
            self.norm_added_q = None
            self.norm_added_k = None
            self.to_add_out = None

        self.context_head_factors = (
            Parameter(total_shape=[self.padded_heads, 1, 1], device=mesh_device, mesh_axes=[tp_axis, None, None])
            if context_head_scaling and self.add_qkv_proj is not None
            else None
        )

        # Empty joint input for ring SDPA when there is no prompt stream.
        self.dummy_joint_input = ttnn.zeros(
            [1, self.n_local_heads, 0, self.head_dim],
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
        )

        self.trans_mat = ttnn.from_torch(
            get_rot_transformation_mat(),
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )

    def get_ring_sdpa_program_config(self, per_device_seq_len: int) -> ttnn.SDPAProgramConfig:
        if per_device_seq_len not in self.ring_sdpa_program_config:
            device_parallel_key = (
                is_blackhole(),
                self.parallel_config.sequence_parallel.factor,
                self.parallel_config.tensor_parallel.factor,
            )
            ring_chunk_size = self.ring_sdpa_chunk_size_map.get(device_parallel_key, self.default_ring_sdpa_chunk_size)
            if per_device_seq_len in ring_chunk_size:
                ring_chunk_size = ring_chunk_size[per_device_seq_len]
            else:
                logger.warning(
                    f"No ring SDPA chunk size found for resolution {per_device_seq_len}, using default {ring_chunk_size[-1]}"
                )
                ring_chunk_size = ring_chunk_size[-1]

            self.ring_sdpa_program_config[per_device_seq_len] = ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=self.ring_sdpa_worker_grid,
                q_chunk_size=ring_chunk_size[0],
                k_chunk_size=ring_chunk_size[1],
                exp_approx_mode=False,  # NOTE: False is more correct
            )
        return self.ring_sdpa_program_config[per_device_seq_len]

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        weight, bias = self._reshape_and_merge_qkv(
            pop_substate(state, "to_q"),
            pop_substate(state, "to_k"),
            pop_substate(state, "to_v"),
        )
        if weight is not None:
            state["to_qkv.weight"] = weight
        if bias is not None:
            state["to_qkv.bias"] = bias

        weight, bias = self._reshape_and_merge_qkv(
            pop_substate(state, "add_q_proj"),
            pop_substate(state, "add_k_proj"),
            pop_substate(state, "add_v_proj"),
        )
        if weight is not None:
            state["add_qkv_proj.weight"] = weight
        if bias is not None:
            state["add_qkv_proj.bias"] = bias

        if "to_out.0.weight" in state:
            state["to_out.weight"] = state.pop("to_out.0.weight")
        if "to_out.0.bias" in state:
            state["to_out.bias"] = state.pop("to_out.0.bias")

        if self.padding_config is not None:
            if "to_out.weight" in state:
                weight = state["to_out.weight"].T
                weight = pad_weight_tensor(weight, self.padding_config, pad_input_dim=True)
                state["to_out.weight"] = weight.T
            if "to_add_out.weight" in state:
                weight = state["to_add_out.weight"].T
                weight = pad_weight_tensor(weight, self.padding_config, pad_input_dim=True)
                state["to_add_out.weight"] = weight.T

        if "context_head_factors" in state:
            factors = state["context_head_factors"]
            if self.padding_config is not None:
                pad = (0, self.padding_config.head_padding)
                factors = torch.nn.functional.pad(factors, pad)
            state["context_head_factors"] = factors.reshape([-1, 1, 1])

        # HF stores per-head weight [head_dim]; DistributedRMSNorm expects [padded_inner_dim].
        for key in ["norm_q.weight", "norm_k.weight", "norm_added_q.weight", "norm_added_k.weight"]:
            if key in state and state[key].shape[0] == self.head_dim:
                state[key] = state[key].repeat(self.padded_heads)

    def _reshape_and_merge_qkv(
        self,
        q_state: dict[str, torch.Tensor],
        k_state: dict[str, torch.Tensor],
        v_state: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        # Rearrange QKV projections such column-fracturing shards the heads
        def _merge_tensors(q, k, v):
            n_dev = self.parallel_config.tensor_parallel.factor
            q, k, v = q.T, k.T, v.T
            # Pad QKV weights and biases to match the padded heads
            if self.padding_config is not None:
                q = pad_weight_tensor(q, self.padding_config, pad_output_dim=True)
                k = pad_weight_tensor(k, self.padding_config, pad_output_dim=True)
                v = pad_weight_tensor(v, self.padding_config, pad_output_dim=True)
            q = q.reshape(q.shape[0], n_dev, self.n_local_heads, self.head_dim)
            k = k.reshape(k.shape[0], n_dev, self.n_local_heads, self.head_dim)
            v = v.reshape(v.shape[0], n_dev, self.n_local_heads, self.head_dim)
            qkv = torch.cat([q, k, v], dim=2)
            qkv = qkv.reshape(qkv.shape[0], 3 * self.padded_heads * self.head_dim)
            qkv = qkv.T
            return qkv

        if "weight" in q_state and "weight" in k_state and "weight" in v_state:
            weight = _merge_tensors(q_state["weight"], k_state["weight"], v_state["weight"])
        else:
            weight = None

        if "bias" in q_state and "bias" in k_state and "bias" in v_state:
            bias = _merge_tensors(
                q_state["bias"].unsqueeze(-1), k_state["bias"].unsqueeze(-1), v_state["bias"].unsqueeze(-1)
            )
            bias = bias.squeeze(-1)
        else:
            bias = None

        return weight, bias

    def _to_out_fused_addcmul(
        self,
        x: ttnn.Tensor,
        addcmul_residual: ttnn.Tensor | None,
        addcmul_gate: ttnn.Tensor | None,
        to_out_module: ColParallelLinear,
        parallel_config,
        tp_axis: int | None = None,
    ) -> ttnn.Tensor:
        """Fused projection + addcmul: output = residual + proj(x) * gate.

        Ring (parallel_config, tp>1): all_gather_minimal_matmul_async fuses AG+matmul+addcmul.
        Linear (parallel_config=None): gathers x on tp_axis if K-dim mismatches, then
        dit_minimal_matmul_addcmul_fused.
        """
        if (
            to_out_module.fsdp_mesh_axis is not None
            and to_out_module.mesh_device.shape[to_out_module.fsdp_mesh_axis] > 1
        ):
            unsqueezed_weight = ttnn.unsqueeze_to_4D(to_out_module.weight.data)
            weight = self.ccl_manager.all_gather_persistent_buffer(
                unsqueezed_weight, dim=2, mesh_axis=to_out_module.fsdp_mesh_axis
            )
            weight = ttnn.reshape(weight, (weight.shape[-2], weight.shape[-1]))
        else:
            weight = to_out_module.weight.data

        bias = to_out_module.bias.data if to_out_module.bias is not None else None

        if parallel_config is not None and parallel_config.tensor_parallel.factor > 1:
            M = x.padded_shape[-2]
            K = weight.padded_shape[-2]
            N = weight.padded_shape[-1]
            full_grid = self.mesh_device.compute_with_storage_grid_size()
            core_grid = ttnn.CoreCoord(full_grid.x, full_grid.y - 1)
            # core_grid = _FLUX2_MATMUL_CORE_GRIDS.get((M, K, N)) or ttnn.CoreCoord(full_grid.x, full_grid.y - 1)
            # print(f"Using core grid {core_grid} for matmul config (M, K, N) = ({M}, {K}, {N})", flush=True)
            matmul_config = get_matmul_config(
                M, K, N, core_grid, use_heuristic=True, num_k_shards=self.parallel_config.tensor_parallel.factor
            )

            ag_persistent_buffer = self.ccl_manager.get_ag_ping_pong_buffer(
                x.shape, -1, parallel_config.tensor_parallel.mesh_axis, dtype=x.get_dtype()
            )
            ag_global_semaphores = self.ccl_manager.get_ag_ping_pong_semaphore(
                parallel_config.tensor_parallel.mesh_axis
            )
            return ttnn.experimental.all_gather_minimal_matmul_async(
                input_tensor=x,
                weight_tensor=weight,
                bias_tensor=bias,
                config=matmul_config,
                compute_kernel_config=self.mm_compute_kernel_config,
                persistent_output_buffer=ag_persistent_buffer,
                multi_device_global_semaphore=ag_global_semaphores,
                num_links=self.ccl_manager.num_links,
                topology=self.ccl_manager.topology,
                cluster_axis=parallel_config.tensor_parallel.mesh_axis,
                barrier_semaphore=None,
                force_transpose=True,
                num_workers_per_link=full_grid.x // self.ccl_manager.num_links,
                num_buffers_per_channel=48 if not is_blackhole() else 24,
                scalar=1.0,
                addcmul_input_tensor1=addcmul_residual,
                addcmul_input_tensor2=addcmul_gate,
            )[0]
        else:
            if tp_axis is not None and x.padded_shape[-1] != weight.padded_shape[-2]:
                x = self.ccl_manager.all_gather_persistent_buffer(x, dim=-1, mesh_axis=tp_axis, use_hyperparams=True)
            M = x.padded_shape[-2]
            K = x.padded_shape[-1]
            N = weight.padded_shape[-1]
            # core_grid = _FLUX2_MATMUL_CORE_GRIDS.get((M, K, N)) or self.mesh_device.compute_with_storage_grid_size()
            # core_grid = self.mesh_device.compute_with_storage_grid_size()
            core_grid = get_matmul_core_grid(self.mesh_device)
            matmul_config = get_matmul_config(
                M, K, N, core_grid, use_heuristic=True, num_k_shards=self.parallel_config.tensor_parallel.factor
            )
            return ttnn.experimental.dit_minimal_matmul_addcmul_fused(
                x,
                weight,
                1.0,
                addcmul_residual,
                addcmul_gate,
                bias_tensor=bias,
                config=matmul_config,
                compute_kernel_config=self.mm_compute_kernel_config,
            )

    def _project_out(
        self,
        input: ttnn.Tensor,
        proj_op: ColParallelLinear | None,
        addcmul_residual: ttnn.Tensor | None,
        addcmul_gate: ttnn.Tensor | None,
        is_ring: bool,
        tp_axis: int,
    ) -> ttnn.Tensor:
        """Project with optional fused addcmul. Returns input unchanged if proj_op is None."""
        if input is None or proj_op is None:
            return input
        return self._to_out_fused_addcmul(
            input,
            addcmul_residual,
            addcmul_gate,
            proj_op,
            self.parallel_config if is_ring else None,
            None if is_ring else tp_axis,
        )

    def forward(
        self,
        *,
        spatial: ttnn.Tensor,
        prompt: ttnn.Tensor | None = None,
        spatial_sequence_length: int,
        spatial_rope: tuple[ttnn.Tensor, ttnn.Tensor] | None = None,
        prompt_rope: tuple[ttnn.Tensor, ttnn.Tensor] | None = None,
        addcmul_spatial_residual: ttnn.Tensor | None = None,
        addcmul_spatial_gate: ttnn.Tensor | None = None,
        addcmul_prompt_residual: ttnn.Tensor | None = None,
        addcmul_prompt_gate: ttnn.Tensor | None = None,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor | None]:
        """Forward with optional fused to_out + addcmul.

        Ring callers pass spatial/prompt fractured on TP (fused AG+matmul inside);
        Linear callers must all-gather first. addcmul_*_residual+gate fuse the
        out projection's residual+gate when both are provided.
        """
        assert len(spatial.shape) == 3
        if prompt is not None:
            assert len(prompt.shape) == 3
        for t in spatial_rope or ():
            assert len(t.shape) == 2
        for t in prompt_rope or ():
            assert len(t.shape) == 2

        tp_axis = self.parallel_config.tensor_parallel.mesh_axis
        sp_axis = self.parallel_config.sequence_parallel.mesh_axis

        is_ring = self.ccl_manager.topology == ttnn.Topology.Ring

        # When the prompt is sharded across SP, its q/k/v are gathered to the full prompt length
        # for the joint SDPA (which requires the full joint stream), and the prompt attention
        # output is re-sharded back across SP afterwards.
        shard_prompt = self.shard_prompt and self.parallel_config.sequence_parallel.factor > 1

        def _split_heads(x: ttnn.Tensor) -> ttnn.Tensor:
            # V only — Q/K head-split is fused inside DistributedRMSNorm.
            out, _, _ = ttnn.experimental.nlp_create_qkv_heads(
                ttnn.unsqueeze(x, 0), num_heads=self.n_local_heads, num_kv_heads=0, transpose_k_heads=False
            )
            return out

        spatial_cos = spatial_rope[0].reshape([1, 1, *spatial_rope[0].shape]) if spatial_rope is not None else None
        spatial_sin = spatial_rope[1].reshape([1, 1, *spatial_rope[1].shape]) if spatial_rope is not None else None
        trans_mat = self.trans_mat if spatial_rope is not None else None

        q_flat, k_flat, v_flat = self.to_qkv(
            spatial,
            compute_kernel_config=self.mm_compute_kernel_config,
            parallel_config=self.parallel_config,
            use_heuristic_mmcfg=True,
        )
        q = self.norm_q(
            ttnn.unsqueeze(q_flat, 0),
            num_heads_per_device=self.n_local_heads,
            rope_cos=spatial_cos,
            rope_sin=spatial_sin,
            trans_mat=trans_mat,
            per_head_norm=self.per_head_norm,
        )
        k = self.norm_k(
            ttnn.unsqueeze(k_flat, 0),
            num_heads_per_device=self.n_local_heads,
            rope_cos=spatial_cos,
            rope_sin=spatial_sin,
            trans_mat=trans_mat,
            per_head_norm=self.per_head_norm,
        )
        v = _split_heads(v_flat)

        if self.add_qkv_proj is not None and prompt is not None:
            prompt_cos = prompt_rope[0].reshape([1, 1, *prompt_rope[0].shape]) if prompt_rope is not None else None
            prompt_sin = prompt_rope[1].reshape([1, 1, *prompt_rope[1].shape]) if prompt_rope is not None else None
            prompt_trans_mat = self.trans_mat if prompt_rope is not None else None

            add_q_flat, add_k_flat, add_v_flat = self.add_qkv_proj(
                prompt,
                compute_kernel_config=self.mm_compute_kernel_config,
                parallel_config=self.parallel_config,
                use_heuristic_mmcfg=True,
            )
            add_q = self.norm_added_q(
                ttnn.unsqueeze(add_q_flat, 0),
                num_heads_per_device=self.n_local_heads,
                rope_cos=prompt_cos,
                rope_sin=prompt_sin,
                trans_mat=prompt_trans_mat,
                per_head_norm=self.per_head_norm,
            )
            add_k = self.norm_added_k(
                ttnn.unsqueeze(add_k_flat, 0),
                num_heads_per_device=self.n_local_heads,
                rope_cos=prompt_cos,
                rope_sin=prompt_sin,
                trans_mat=prompt_trans_mat,
                per_head_norm=self.per_head_norm,
            )
            add_v = _split_heads(add_v_flat)

            if self.context_head_factors is not None:
                add_q = add_q * self.context_head_factors.data

            if shard_prompt:
                # Prompt q/k/v are sharded on the sequence dim (dim=2) across SP. The joint SDPA
                # needs the full prompt stream, so gather them to full prompt length. RoPE has
                # already been applied per-rank with the matching sharded positions.
                add_q = self.ccl_manager.all_gather(add_q, dim=2, mesh_axis=sp_axis, use_hyperparams=True)
                add_k = self.ccl_manager.all_gather(add_k, dim=2, mesh_axis=sp_axis, use_hyperparams=True)
                add_v = self.ccl_manager.all_gather(add_v, dim=2, mesh_axis=sp_axis, use_hyperparams=True)
        else:
            add_q = add_k = add_v = self.dummy_joint_input

        if self.parallel_config.sequence_parallel.factor > 1:
            spatial, prompt, _lse = ttnn.transformer.ring_joint_scaled_dot_product_attention(
                q,
                k,
                v,
                add_q,
                add_k,
                add_v,
                persistent_output_buffer_k=self.ccl_manager.get_ag_ping_pong_buffer(
                    k.shape, 2, self.parallel_config.sequence_parallel.mesh_axis
                ),
                persistent_output_buffer_v=self.ccl_manager.get_ag_ping_pong_buffer(
                    v.shape, 2, self.parallel_config.sequence_parallel.mesh_axis
                ),
                joint_strategy="rear",
                logical_n=spatial_sequence_length,
                program_config=self.get_ring_sdpa_program_config(spatial_sequence_length),
                compute_kernel_config=self.sdpa_compute_kernel_config,
                dim=2,
                multi_device_global_semaphore=self.ccl_manager.get_ag_ping_pong_semaphore(
                    self.parallel_config.sequence_parallel.mesh_axis
                ),
                num_links=self.ccl_manager.num_links,
                cluster_axis=self.parallel_config.sequence_parallel.mesh_axis,
                mesh_device=self.mesh_device,
                topology=self.ccl_manager.topology,
                subdevice_id=self.ccl_manager.ccl_sub_device_id,
                ccl_core_grid_offset=(0, self.ring_sdpa_worker_grid[1]),
            )
        else:
            assert (
                spatial_sequence_length == spatial.shape[1]
            ), "spatial sequence must not be padded without sequence parallelism"

            spatial, prompt = ttnn.transformer.joint_scaled_dot_product_attention(
                q,
                k,
                v,
                add_q,
                add_k,
                add_v,
                joint_strategy="rear",
                program_config=self.sdpa_program_config,
                compute_kernel_config=self.sdpa_compute_kernel_config,
            )

        spatial = ttnn.transformer.concatenate_heads(spatial)
        if prompt is not None:
            prompt = ttnn.transformer.concatenate_heads(prompt)

            if shard_prompt:
                # The joint SDPA produces the full prompt output replicated across SP ranks.
                # Re-shard it on the sequence dim (dim=1) so the prompt residual stream stays
                # sharded for the downstream projection and the next block.
                prompt = ttnn.mesh_partition(prompt, dim=1, cluster_axis=sp_axis, memory_config=prompt.memory_config())

        spatial = self._project_out(
            spatial, self.to_out, addcmul_spatial_residual, addcmul_spatial_gate, is_ring, tp_axis
        )
        prompt = self._project_out(
            prompt, self.to_add_out, addcmul_prompt_residual, addcmul_prompt_gate, is_ring, tp_axis
        )
        return spatial, prompt

    @classmethod
    def spatial_sequence_padding_length(cls, *, length: int, sp_factor: int, k_chunk_size: int = 512) -> int:
        if sp_factor == 1:
            return 0

        divisor = k_chunk_size * sp_factor
        return -length % divisor

    @classmethod
    def pad_spatial_sequence(cls, x: torch.Tensor, /, *, sp_factor: int, k_chunk_size: int = 512) -> torch.Tensor:
        padding_len = cls.spatial_sequence_padding_length(
            length=x.shape[-2], sp_factor=sp_factor, k_chunk_size=k_chunk_size
        )
        return torch.nn.functional.pad(x, (0, 0, 0, padding_len))
