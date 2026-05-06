# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import ttnn
from models.common.utility_functions import is_blackhole

from ...layers.linear import ColParallelLinear
from ...layers.module import Module, Parameter, UnregisteredModule
from ...layers.normalization import RMSNorm
from ...utils.matmul import get_matmul_config
from ...utils.padding import PaddingConfig, pad_weight_tensor
from ...utils.substate import pop_substate

if TYPE_CHECKING:
    from ...parallel.config import DiTParallelConfig
    from ...parallel.manager import CCLManager


class Flux2Attention(Module):
    """Optimized FLUX.2 attention with WAN-style kernel fusion.

    Key improvements over the baseline Attention class:
    - Fused all-gather + QKV matmul via ColParallelLinear parallel_config (Ring topology).
    - Fused all-gather + to_out matmul + addcmul (residual gate): eliminates a separate
      all-gather + gate multiply + add when addcmul params are provided by the caller.
    - Linear topology fallback: explicit all-gather before each matmul for correctness.
    - norm_q/norm_k use plain RMSNorm (per head, after head split) matching original semantics.
    """

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
        k_chunk_size: int = 512,
        q_chunk_size: int = 128,
        is_fsdp: bool = False,
    ) -> None:
        super().__init__()

        self.head_dim = head_dim
        self.pre_only = pre_only
        self.mesh_device = mesh_device
        self.ccl_manager = ccl_manager
        self.parallel_config = parallel_config
        self.padding_config = padding_config
        self.use_spatial_weights_for_prompt = use_spatial_weights_for_prompt

        self.padded_heads = padding_config.target_heads if padding_config is not None else heads
        self.n_local_heads = self.padded_heads // self.parallel_config.tensor_parallel.factor

        tp_axis = parallel_config.tensor_parallel.mesh_axis
        padded_inner_dim = head_dim * self.padded_heads

        fsdp_mesh_axis = parallel_config.sequence_parallel.mesh_axis if is_fsdp else None

        common_args = dict(mesh_device=mesh_device, ccl_manager=ccl_manager, fsdp_mesh_axis=fsdp_mesh_axis)

        # Per-head RMSNorm (applied after head split, matching original Attention semantics).
        # Each device holds complete heads; no cross-device stats communication needed.
        self.norm_q = RMSNorm(embedding_dim=head_dim, norm_eps=eps, bias=False, mesh_device=mesh_device)
        self.norm_k = RMSNorm(embedding_dim=head_dim, norm_eps=eps, bias=False, mesh_device=mesh_device)

        # chunks=3: to_qkv returns [q, k, v] as separate tensors [B, N, n_local_heads*head_dim].
        # Each is then split into heads and normalized independently.
        self.to_qkv = ColParallelLinear(
            query_dim, 3 * padded_inner_dim, bias=proj_bias, mesh_axis=tp_axis, chunks=3, **common_args
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
            self.norm_added_q = RMSNorm(embedding_dim=head_dim, norm_eps=eps, bias=False, mesh_device=mesh_device)
            self.norm_added_k = RMSNorm(embedding_dim=head_dim, norm_eps=eps, bias=False, mesh_device=mesh_device)
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

        full_grid = self.mesh_device.compute_with_storage_grid_size()

        self.sdpa_worker_grid = (full_grid.x, full_grid.y - 1)
        self.sdpa_program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=self.sdpa_worker_grid,
            q_chunk_size=q_chunk_size,
            k_chunk_size=k_chunk_size,
            exp_approx_mode=False,
        )
        self.sdpa_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
        )
        self.mm_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=True,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

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
                factors = torch.nn.functional.pad(factors, (0, self.padding_config.head_padding))
            state["context_head_factors"] = factors.reshape([-1, 1, 1])

    def _reshape_and_merge_qkv(
        self,
        q_state: dict[str, torch.Tensor],
        k_state: dict[str, torch.Tensor],
        v_state: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        def _merge_tensors(q, k, v):
            n_dev = self.parallel_config.tensor_parallel.factor
            q, k, v = q.T, k.T, v.T
            if self.padding_config is not None:
                q = pad_weight_tensor(q, self.padding_config, pad_output_dim=True)
                k = pad_weight_tensor(k, self.padding_config, pad_output_dim=True)
                v = pad_weight_tensor(v, self.padding_config, pad_output_dim=True)
            q = q.reshape(q.shape[0], n_dev, self.n_local_heads, self.head_dim)
            k = k.reshape(k.shape[0], n_dev, self.n_local_heads, self.head_dim)
            v = v.reshape(v.shape[0], n_dev, self.n_local_heads, self.head_dim)
            qkv = torch.cat([q, k, v], dim=2)
            qkv = qkv.reshape(qkv.shape[0], 3 * self.padded_heads * self.head_dim)
            return qkv.T

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
        addcmul_residual: ttnn.Tensor,
        addcmul_gate: ttnn.Tensor,
        to_out_module: ColParallelLinear,
        parallel_config,
    ) -> ttnn.Tensor:
        """Fused projection + addcmul: output = residual + proj(x) * gate.

        For Ring topology (parallel_config provided with tp>1):
            Uses all_gather_minimal_matmul_async to fuse all-gather(x) + matmul + addcmul.
            x must be fractured [B, N, H_local*head_dim] from concatenate_heads.

        For Linear topology or tp=1 (parallel_config=None or tp=1):
            x must already be all-gathered [B, N, padded_inner_dim].
            Uses dit_minimal_matmul_addcmul_fused for matmul + addcmul fusion.
        """
        # Handle FSDP weight gathering (mirrors ColParallelLinear.forward)
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
            # Ring: fused all-gather(x) + matmul + addcmul in one kernel
            M = x.padded_shape[-2]
            K = weight.padded_shape[-2]
            N = weight.padded_shape[-1]
            full_grid = self.mesh_device.compute_with_storage_grid_size()
            core_grid = ttnn.CoreCoord(full_grid.x, full_grid.y - 1)
            matmul_config = get_matmul_config(M, K, N, core_grid)

            ag_persistent_buffer = self.ccl_manager.get_ag_ping_pong_buffer(
                x.shape, 3, parallel_config.tensor_parallel.mesh_axis, dtype=x.get_dtype()
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
            # tp=1 or Linear (x already gathered): simple matmul + addcmul
            M = x.padded_shape[-2]
            K = x.padded_shape[-1]
            N = weight.padded_shape[-1]
            core_grid = self.mesh_device.compute_with_storage_grid_size()
            matmul_config = get_matmul_config(M, K, N, core_grid)
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
        """Forward pass with optional fused to_out + addcmul (residual gate).

        Args:
            spatial: [B, N/sp, D/tp] - fractured on both SP and TP. For Ring topology this is
                still fractured; for Linear topology the caller must all-gather before calling.
            prompt: [B, L, D/tp] - fractured on TP only. Optional. Same gather contract as spatial.
            spatial_sequence_length: logical spatial sequence length (for ring SDPA).
            spatial_rope / prompt_rope: RoPE cos/sin tuples.
            addcmul_spatial_residual: if provided with gate, fuses to_out + gate * result + residual.
            addcmul_spatial_gate: gate tensor [B, 1, D/tp] for spatial fused addcmul.
            addcmul_prompt_residual / addcmul_prompt_gate: same for prompt stream.

        Returns:
            (spatial_out, prompt_out) - both [B, N, D/tp] fractured on TP.
            If fused addcmul: spatial_out = residual + to_out(attn) * gate (already applied).
            If no addcmul: spatial_out is the raw to_out projection result.
        """
        assert len(spatial.shape) == 3
        if prompt is not None:
            assert len(prompt.shape) == 3

        tp_axis = self.parallel_config.tensor_parallel.mesh_axis

        # For Ring: ColParallelLinear fuses the all-gather with the matmul via parallel_config.
        # For Linear (or tp=1): the caller is responsible for gathering inputs before calling.
        is_ring = self.ccl_manager.topology == ttnn.Topology.Ring
        qkv_parallel_config = self.parallel_config if is_ring else None

        # === SPATIAL QKV PROJECTIONS ===
        # Returns [q, k, v] each [B, N, n_local_heads*head_dim] - fractured on TP.
        q, k, v = self.to_qkv(
            spatial, compute_kernel_config=self.mm_compute_kernel_config, parallel_config=qkv_parallel_config
        )

        # Split into heads: [B, N, H_local*head_dim] → [B, H_local, N, head_dim]
        q, _, _ = ttnn.experimental.nlp_create_qkv_heads(
            ttnn.unsqueeze(q, 0), num_heads=self.n_local_heads, num_kv_heads=0, transpose_k_heads=False
        )
        k, _, _ = ttnn.experimental.nlp_create_qkv_heads(
            ttnn.unsqueeze(k, 0), num_heads=self.n_local_heads, num_kv_heads=0, transpose_k_heads=False
        )
        v, _, _ = ttnn.experimental.nlp_create_qkv_heads(
            ttnn.unsqueeze(v, 0), num_heads=self.n_local_heads, num_kv_heads=0, transpose_k_heads=False
        )

        # Per-head RMSNorm on [B, H_local, N, head_dim] — matches original Attention.
        q = self.norm_q(q)
        k = self.norm_k(k)

        if spatial_rope is not None:
            q = _apply_rope(q, spatial_rope)
            k = _apply_rope(k, spatial_rope)

        # === PROMPT QKV PROJECTIONS ===
        if self.add_qkv_proj is not None and prompt is not None:
            add_q, add_k, add_v = self.add_qkv_proj(
                prompt, compute_kernel_config=self.mm_compute_kernel_config, parallel_config=qkv_parallel_config
            )

            add_q, _, _ = ttnn.experimental.nlp_create_qkv_heads(
                ttnn.unsqueeze(add_q, 0), num_heads=self.n_local_heads, num_kv_heads=0, transpose_k_heads=False
            )
            add_k, _, _ = ttnn.experimental.nlp_create_qkv_heads(
                ttnn.unsqueeze(add_k, 0), num_heads=self.n_local_heads, num_kv_heads=0, transpose_k_heads=False
            )
            add_v, _, _ = ttnn.experimental.nlp_create_qkv_heads(
                ttnn.unsqueeze(add_v, 0), num_heads=self.n_local_heads, num_kv_heads=0, transpose_k_heads=False
            )

            add_q = self.norm_added_q(add_q)
            add_k = self.norm_added_k(add_k)

            if prompt_rope is not None:
                add_q = _apply_rope(add_q, prompt_rope)
                add_k = _apply_rope(add_k, prompt_rope)

            if self.context_head_factors is not None:
                add_q = add_q * self.context_head_factors.data
        else:
            add_q = add_k = add_v = ttnn.zeros(
                [1, self.n_local_heads, 0, self.head_dim],
                device=self.mesh_device,
                layout=q.layout,
                dtype=q.dtype,
            )

        # === SDPA ===
        if self.parallel_config.sequence_parallel.factor > 1:
            spatial_out, prompt_out, _lse = ttnn.transformer.ring_joint_scaled_dot_product_attention(
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
                program_config=self.sdpa_program_config,
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
                ccl_core_grid_offset=(0, self.sdpa_worker_grid[1]),
            )
        else:
            assert (
                spatial_sequence_length == spatial.shape[1]
            ), "spatial sequence must not be padded without sequence parallelism"
            spatial_out, prompt_out = ttnn.transformer.joint_scaled_dot_product_attention(
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

        # concatenate_heads: [B, H, N, E] → [B, N, H*E] (fractured on TP)
        spatial_out = ttnn.transformer.concatenate_heads(spatial_out)
        if prompt_out is not None:
            prompt_out = ttnn.transformer.concatenate_heads(prompt_out)

        # Gather SDPA outputs before to_out if K doesn't match K_w AND we are not using the
        # Ring fused AG+MM path (which handles the gather internally inside _to_out_fused_addcmul).
        if self.to_out is not None and not is_ring:
            if spatial_out.padded_shape[-1] != self.to_out.weight.data.padded_shape[-2]:
                spatial_out = self.ccl_manager.all_gather_persistent_buffer(spatial_out, dim=-1, mesh_axis=tp_axis)
        if self.to_add_out is not None and prompt_out is not None and not is_ring:
            if prompt_out.padded_shape[-1] != self.to_add_out.weight.data.padded_shape[-2]:
                prompt_out = self.ccl_manager.all_gather_persistent_buffer(prompt_out, dim=-1, mesh_axis=tp_axis)

        # parallel_config passed to _to_out_fused_addcmul: Ring → fused AG+MM+addcmul,
        # None (Linear/tp=1) → input already gathered, just matmul+addcmul.
        to_out_parallel_config = self.parallel_config if is_ring else None

        # === SPATIAL TO_OUT ===
        if self.to_out is not None:
            if addcmul_spatial_residual is not None and addcmul_spatial_gate is not None:
                spatial_out = self._to_out_fused_addcmul(
                    spatial_out,
                    addcmul_spatial_residual,
                    addcmul_spatial_gate,
                    self.to_out,
                    to_out_parallel_config,
                )
            else:
                if spatial_out.padded_shape[-1] != self.to_out.weight.data.padded_shape[-2]:
                    spatial_out = self.ccl_manager.all_gather_persistent_buffer(
                        spatial_out, dim=-1, mesh_axis=tp_axis, use_hyperparams=True
                    )
                spatial_out = self.to_out(spatial_out)

        # === PROMPT TO_OUT ===
        prompt_result: ttnn.Tensor | None = None
        if self.to_add_out is not None and prompt_out is not None:
            if addcmul_prompt_residual is not None and addcmul_prompt_gate is not None:
                prompt_result = self._to_out_fused_addcmul(
                    prompt_out,
                    addcmul_prompt_residual,
                    addcmul_prompt_gate,
                    self.to_add_out,
                    to_out_parallel_config,
                )
            else:
                if prompt_out.padded_shape[-1] != self.to_add_out.weight.data.padded_shape[-2]:
                    prompt_out = self.ccl_manager.all_gather_persistent_buffer(
                        prompt_out, dim=-1, mesh_axis=tp_axis, use_hyperparams=True
                    )
                prompt_result = self.to_add_out(prompt_out)
        elif self.pre_only and prompt_out is not None:
            prompt_result = prompt_out

        return spatial_out, prompt_result

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


def _apply_rope(x: ttnn.Tensor, freqs_cis: tuple[ttnn.Tensor, ttnn.Tensor]) -> ttnn.Tensor:
    cos, sin = freqs_cis
    cos = cos.reshape([1, 1, *cos.shape])
    sin = sin.reshape([1, 1, *sin.shape])
    return x * cos + ttnn.alt_complex_rotate90(x) * sin
