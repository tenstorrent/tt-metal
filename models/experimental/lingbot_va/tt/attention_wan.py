# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
WanAttention for Lingbot-VA TT model.

Copied from tt_dit/models/transformers/wan2_2/attention_wan.py so that lingbot_va
has no dependency on wan2_2. Imports use models.tt_dit.* (layers, utils, parallel).
"""

from __future__ import annotations

import torch

import ttnn
from models.common.utility_functions import is_blackhole

from models.tt_dit.layers.linear import ColParallelLinear
from models.tt_dit.layers.module import Module
from models.tt_dit.layers.normalization import DistributedRMSNorm
from models.tt_dit.parallel.config import DiTParallelConfig
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.utils.matmul import get_matmul_config
from models.tt_dit.utils.substate import pop_substate, rename_substate
from models.tt_dit.utils.tensor import bf16_tensor


class WanAttention(Module):
    # Map from (is_blackhole, sp_factor, tp_factor) -> (q_chunk_size, k_chunk_size)
    sdpa_chunk_size_map = {
        (False, 2, 4): (256, 256),
        (False, 8, 4): (256, 256),
        (True, 2, 2): (128, 512),
        (True, 8, 4): (128, 512),
    }
    default_sdpa_chunk_size = (256, 256)

    def __init__(
        self,
        *,
        dim: int,
        num_heads: int,
        qk_norm: bool = True,
        eps: float = 1e-5,
        mesh_device: ttnn.MeshDevice,
        ccl_manager: "CCLManager"
        | None = None,  # CCLManager handles collective communication operations for model/gradient parallelism across chips or devices.
        parallel_config: DiTParallelConfig,
        is_fsdp: bool = False,
        is_self: bool = True,
    ) -> None:
        super().__init__()

        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qk_norm = qk_norm
        self.eps = eps

        self.mesh_device = mesh_device
        self.ccl_manager = ccl_manager
        self.parallel_config = parallel_config
        self.is_self = is_self

        self.n_local_heads = self.num_heads // self.parallel_config.tensor_parallel.factor

        fsdp_mesh_axis = self.parallel_config.sequence_parallel.mesh_axis if is_fsdp else None

        rms_kwargs = {
            "embedding_dim": dim,
            "norm_eps": eps,
            "norm_elementwise_affine": True,
            "bias": False,
            "mesh_device": mesh_device,
            "mesh_axis": parallel_config.tensor_parallel.mesh_axis,
            "ccl_manager": ccl_manager,
        }

        self.norm_q = DistributedRMSNorm(**rms_kwargs)
        self.norm_k = DistributedRMSNorm(**rms_kwargs)

        col_parallel_kwargs = {
            "bias": True,
            "mesh_device": mesh_device,
            "mesh_axis": parallel_config.tensor_parallel.mesh_axis,
            "fsdp_mesh_axis": fsdp_mesh_axis,
            "ccl_manager": ccl_manager,
        }

        if is_self:
            # Fused QKV for self-attention: single matmul split into 3 outputs
            self.to_qkv = ColParallelLinear(
                dim,
                3 * dim,
                chunks=3,
                **col_parallel_kwargs,
            )
        else:
            # Cross-attention: Q from spatial, K/V from prompt
            self.to_q = ColParallelLinear(dim, dim, **col_parallel_kwargs)
            # Fused KV: single matmul split into 2 outputs
            self.to_kv = ColParallelLinear(
                dim,
                2 * dim,
                chunks=2,
                **col_parallel_kwargs,
            )

        self.to_out = ColParallelLinear(
            dim,
            dim,
            bias=True,
            mesh_device=mesh_device,
            mesh_axis=parallel_config.tensor_parallel.mesh_axis,
            fsdp_mesh_axis=fsdp_mesh_axis,
            ccl_manager=ccl_manager,
        )

        self.dummy_joint_input = bf16_tensor(torch.zeros((1, self.n_local_heads, 0, self.head_dim)), device=mesh_device)

        full_grid = self.mesh_device.compute_with_storage_grid_size()
        # Single-device (sp=1, tp=1): use smaller chunks to avoid SDPA hang on seq_len 1152.
        sp_factor = self.parallel_config.sequence_parallel.factor
        tp_factor = self.parallel_config.tensor_parallel.factor
        if sp_factor == 1 and tp_factor == 1:
            q_chunk, k_chunk = 128, 128
        else:
            q_chunk, k_chunk = 256, 256
        self.sdpa_program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=full_grid,
            q_chunk_size=q_chunk,
            k_chunk_size=k_chunk,
            exp_approx_mode=False,  # NOTE: False is more correct
        )

        self.sdpa_worker_grid = (full_grid.x, full_grid.y - 1)
        ring_sdpa_chunk_size = self.sdpa_chunk_size_map.get(
            (
                is_blackhole(),
                self.parallel_config.sequence_parallel.factor,
                self.parallel_config.tensor_parallel.factor,
            ),
            self.default_sdpa_chunk_size,
        )

        self.ring_sdpa_program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=self.sdpa_worker_grid,
            q_chunk_size=ring_sdpa_chunk_size[0],
            k_chunk_size=ring_sdpa_chunk_size[1],
            exp_approx_mode=False,  # NOTE: False is more correct
        )

        self.sdpa_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            self.mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,  # Reduces RMSE vs PyTorch (attention accumulation in fp32)
        )

        self.rope_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            self.mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

        device_grid = self.mesh_device.compute_with_storage_grid_size()
        self.core_grid = ttnn.CoreGrid(x=device_grid.x, y=device_grid.y)

        # HiFi4 + no approx on QKV / to_out matmuls (native substitute for torch SDP + Linear precision).
        self.mm_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            self.mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    def clear_pred_cache(self, cache_name: str) -> None:
        """Reference API; Lingbot KV lives on WanTransformer3DModel._attn_caches."""

    def clear_cache(self, cache_name: str) -> None:
        """Reference API; Lingbot KV lives on WanTransformer3DModel._attn_caches."""

    def init_kv_cache(self, *args, **kwargs) -> None:
        """Reference API; parent create_empty_cache fills _attn_caches — no buffers on TT attn."""

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        rename_substate(state, "to_out.0", "to_out")

        def _interleave_heads(tensors: list[torch.Tensor]):
            """Interleave weight/bias tensors so column-parallel fracturing assigns correct heads per device.

            Each tensor has out-dim = num_heads * head_dim.  We interleave them on the heads
            axis so that after TP column-sharding each device gets the right heads from every tensor.

            Args:
                tensors: list of tensors in [out, in] PyTorch convention (or [out, 1] for bias).
            Returns:
                Merged tensor in [out, in] PyTorch convention.
            """
            n_dev = self.parallel_config.tensor_parallel.factor
            # Transpose to [in, out]
            tensors = [t.T for t in tensors]
            # Reshape out dim to [in, n_dev, n_local_heads, head_dim]
            tensors = [t.reshape(t.shape[0], n_dev, self.n_local_heads, self.head_dim) for t in tensors]
            # Concatenate on the heads dim so each device shard gets its own heads from every tensor
            merged = torch.cat(tensors, dim=2)  # [in, n_dev, len(tensors)*n_local_heads, head_dim]
            merged = merged.reshape(merged.shape[0], len(tensors) * self.num_heads * self.head_dim)
            # Transpose back to [out, in] PyTorch convention
            return merged.T

        if self.is_self:
            # Merge separate to_q, to_k, to_v weights into fused to_qkv
            q_state = pop_substate(state, "to_q")
            k_state = pop_substate(state, "to_k")
            v_state = pop_substate(state, "to_v")

            state["to_qkv.weight"] = _interleave_heads([q_state["weight"], k_state["weight"], v_state["weight"]])
            if "bias" in q_state:
                bias = _interleave_heads(
                    [q_state["bias"].unsqueeze(-1), k_state["bias"].unsqueeze(-1), v_state["bias"].unsqueeze(-1)]
                )
                state["to_qkv.bias"] = bias.squeeze(-1)
        else:
            # Merge separate to_k, to_v weights into fused to_kv
            k_state = pop_substate(state, "to_k")
            v_state = pop_substate(state, "to_v")

            state["to_kv.weight"] = _interleave_heads([k_state["weight"], v_state["weight"]])
            if "bias" in k_state:
                bias = _interleave_heads([k_state["bias"].unsqueeze(-1), v_state["bias"].unsqueeze(-1)])
                state["to_kv.bias"] = bias.squeeze(-1)

    def _to_out_weight_for_mm(self) -> ttnn.Tensor:
        """Gathered weight [K, N_local] for matmul (matches ColParallelLinear.forward)."""
        to_out = self.to_out
        if to_out.fsdp_mesh_axis is not None and to_out.mesh_device.shape[to_out.fsdp_mesh_axis] > 1:
            unsqueezed_weight = ttnn.unsqueeze_to_4D(to_out.weight.data)
            weight = self.ccl_manager.all_gather_persistent_buffer(
                unsqueezed_weight, dim=2, mesh_axis=to_out.fsdp_mesh_axis
            )
            return ttnn.reshape(weight, (weight.shape[-2], weight.shape[-1]))
        return to_out.weight.data

    @staticmethod
    def _addcmul_gate_fits_fused_kernel(addcmul_gate: ttnn.Tensor, n_out: int) -> bool:
        # dit_minimal_matmul_addcmul_fused requires ternary_b logical shape [1, N] (broadcast on M).
        sh = addcmul_gate.shape
        return len(sh) >= 2 and sh[-2] == 1 and sh[-1] == n_out

    def _to_out_fused_addcmul(
        self,
        x: ttnn.Tensor,
        addcmul_residual: ttnn.Tensor,
        addcmul_gate: ttnn.Tensor,
        compute_kernel_config=None,
        *,
        weight: ttnn.Tensor | None = None,
    ) -> ttnn.Tensor:
        """Fused to_out projection + addcmul: output = residual + (matmul(x, W) + bias) * gate."""
        to_out = self.to_out
        if weight is None:
            weight = self._to_out_weight_for_mm()
        M, K, N_out = x.padded_shape[-2], x.padded_shape[-1], weight.padded_shape[-1]
        core_grid = self.mesh_device.compute_with_storage_grid_size()
        matmul_config = get_matmul_config(M, K, N_out, core_grid)

        output = ttnn.experimental.dit_minimal_matmul_addcmul_fused(
            x,
            weight,
            1.0,  # scalar
            addcmul_residual,
            addcmul_gate,
            bias_tensor=to_out.bias.data if to_out.bias is not None else None,
            config=matmul_config,
            compute_kernel_config=compute_kernel_config or to_out.compute_config,
        )
        return output

    def _to_out_unfused_addcmul(
        self,
        x: ttnn.Tensor,
        addcmul_residual: ttnn.Tensor,
        addcmul_gate: ttnn.Tensor,
        compute_kernel_config=None,
    ) -> ttnn.Tensor:
        """Same math as fused op when gate is full [M, N] / per-token (fused kernel only supports gate [1, N])."""
        # Match fused path blocking (see grid_88 (256, 3072, 3072)); avoid default 8x8x8 L1 overflow on WH.
        proj = self.to_out(
            x,
            compute_kernel_config=compute_kernel_config or self.mm_compute_kernel_config,
            default_block_size=(2, 4, 16),
        )
        return ttnn.add(addcmul_residual, ttnn.mul(proj, addcmul_gate))

    def forward(
        self,
        spatial_1BND: ttnn.Tensor,
        N: int,
        prompt_1BLP: ttnn.Tensor | None = None,
        rope_cos: ttnn.Tensor | None = None,
        rope_sin: ttnn.Tensor | None = None,
        trans_mat: ttnn.Tensor | None = None,
        addcmul_residual: ttnn.Tensor | None = None,
        addcmul_gate: ttnn.Tensor | None = None,
        cached_k: ttnn.Tensor | None = None,
        cached_v: ttnn.Tensor | None = None,
        return_kv: bool = False,
    ) -> ttnn.Tensor | tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
        """
        spatial_1BND: fractured N on SP, fracturd D on TP
        prompt_1BLP: replicated on SP, replicated D on TP (optional)
        rope_cos: fractured on SP, TP
        rope_sin: fractured on SP, TP
        trans_mat: replicated
        addcmul_residual: (optional) residual tensor for fused matmul+addcmul (self-attn only)
        addcmul_gate: (optional) gate tensor for fused matmul+addcmul (self-attn only)
        cached_k, cached_v: (optional) extended KV context for self-attn; concat with current k,v on dim=2.
        return_kv: if True (self-attn only), return (spatial_1BND, k_cur, v_cur) for cache update.

        If prompt_1BLP is not provided, run self-attention.
        Otherwise, run cross-attention on prompt.

        When addcmul_residual and addcmul_gate are both provided (self-attention only),
        uses dit_minimal_matmul_addcmul_fused when gate broadcasts as [1, N]; otherwise
        to_out + mul + add (per-token gate, e.g. robotwin / [1,B,L,C] temb).
            output = addcmul_residual + to_out(attn_output) * addcmul_gate

        Outputs:
        spatial_1BND: fractured N on SP, fractured D on TP
        Or (spatial_1BND, k_cur, v_cur) when return_kv=True (self-attn).
        """

        if rope_cos is not None:
            # If ROPE is given, this is self-attention
            assert rope_sin is not None
            assert trans_mat is not None
            assert prompt_1BLP is None

        if self.parallel_config.tensor_parallel.factor > 1:
            spatial_1BND = self.ccl_manager.all_gather_persistent_buffer(
                spatial_1BND, dim=3, mesh_axis=self.parallel_config.tensor_parallel.mesh_axis
            )

        if self.is_self:
            # Fused QKV matmul with split output for self-attention
            q_1BNF, k_1BNF, v_1BNF = self.to_qkv(spatial_1BND, compute_kernel_config=self.mm_compute_kernel_config)
        else:
            # Cross-attention: Q from spatial, fused KV from prompt
            kv_input = prompt_1BLP if prompt_1BLP is not None else spatial_1BND
            q_1BNF = self.to_q(spatial_1BND, compute_kernel_config=self.mm_compute_kernel_config)
            k_1BNF, v_1BNF = self.to_kv(kv_input, compute_kernel_config=self.mm_compute_kernel_config)
            # wan_fused_rmsnorm_post_allgather expects 4D [1, B, L, D]; ensure K/V are 4D (prompt may be 3D).
            if len(k_1BNF.shape) == 3:
                k_1BNF = ttnn.unsqueeze(k_1BNF, 0)
                v_1BNF = ttnn.unsqueeze(v_1BNF, 0)

        # Norm spatial before splitting heads
        q_BHNE = self.norm_q(
            q_1BNF, num_heads_per_device=self.n_local_heads, rope_cos=rope_cos, rope_sin=rope_sin, trans_mat=trans_mat
        )
        k_BHNE = self.norm_k(
            k_1BNF, num_heads_per_device=self.n_local_heads, rope_cos=rope_cos, rope_sin=rope_sin, trans_mat=trans_mat
        )

        def create_heads(inp):
            out, _, _ = ttnn.experimental.nlp_create_qkv_heads(
                inp,
                num_heads=self.n_local_heads,
                num_kv_heads=0,
                transpose_k_heads=False,
            )
            return out

        # nlp_create_qkv_heads requires shape [batch, 1, seq, hidden]; Wan linear uses [1, B, L, hidden].
        v_in = (
            ttnn.permute(v_1BNF, (1, 0, 2, 3))
            if len(v_1BNF.shape) == 4 and int(v_1BNF.shape[0]) == 1 and int(v_1BNF.shape[1]) > 1
            else v_1BNF
        )
        v_BHNE = create_heads(v_in)

        if self.is_self and return_kv:
            k_cur, v_cur = k_BHNE, v_BHNE
        if self.is_self and cached_k is not None and cached_v is not None:
            # ttnn.concat expects TILE_LAYOUT
            cached_k = ttnn.to_layout(cached_k, ttnn.TILE_LAYOUT)
            cached_v = ttnn.to_layout(cached_v, ttnn.TILE_LAYOUT)
            k_BHNE = ttnn.concat([cached_k, k_BHNE], dim=2)
            v_BHNE = ttnn.concat([cached_v, v_BHNE], dim=2)

        # Rope

        if prompt_1BLP is None:
            # Self attention (use standard SDPA when KV cache is used; ring does not support it)
            # Ring path assumes SDPA batch dim 1; batched CFG uses [B, H, L, D] from norms + create_heads.
            sdpa_batch = int(q_BHNE.shape[0])
            if (
                self.parallel_config.sequence_parallel.factor > 1
                and cached_k is None
                and not return_kv
                and sdpa_batch == 1
            ):
                # HACK: pass null joint inputs to take advantage of ring attention, even though this is self-attention.
                spatial_BHNE, prompt_BHLE, _lse = ttnn.transformer.ring_joint_scaled_dot_product_attention(
                    q_BHNE,
                    k_BHNE,
                    v_BHNE,
                    self.dummy_joint_input,
                    self.dummy_joint_input,
                    self.dummy_joint_input,
                    persistent_output_buffer_k=self.ccl_manager.get_ag_ping_pong_buffer(
                        k_BHNE.shape, 2, self.parallel_config.sequence_parallel.mesh_axis
                    ),
                    persistent_output_buffer_v=self.ccl_manager.get_ag_ping_pong_buffer(
                        v_BHNE.shape, 2, self.parallel_config.sequence_parallel.mesh_axis
                    ),
                    joint_strategy="rear",
                    logical_n=N,
                    program_config=self.ring_sdpa_program_config,
                    compute_kernel_config=self.sdpa_compute_kernel_config,
                    dim=2,
                    multi_device_global_semaphore=self.ccl_manager.get_ag_ping_pong_semaphore(
                        self.parallel_config.sequence_parallel.mesh_axis
                    ),
                    num_links=self.ccl_manager.num_links,
                    cluster_axis=self.parallel_config.sequence_parallel.mesh_axis,
                    mesh_device=self.mesh_device,
                    topology=ttnn.Topology.Linear,  # RJA always uses Linear topology
                    subdevice_id=self.ccl_manager.ccl_sub_device_id,
                    ccl_core_grid_offset=(0, self.sdpa_worker_grid[1]),
                )
            else:
                spatial_BHNE = ttnn.transformer.scaled_dot_product_attention(
                    q_BHNE,
                    k_BHNE,
                    v_BHNE,
                    is_causal=False,
                    program_config=self.sdpa_program_config,
                    compute_kernel_config=self.sdpa_compute_kernel_config,
                )
        else:
            # Cross attention
            spatial_BHNE = ttnn.transformer.scaled_dot_product_attention(
                q_BHNE,
                k_BHNE,
                v_BHNE,
                is_causal=False,
                program_config=self.sdpa_program_config,
                compute_kernel_config=self.sdpa_compute_kernel_config,
            )

        spatial_1BND = ttnn.transformer.concatenate_heads(spatial_BHNE)
        spatial_1BND = ttnn.unsqueeze(spatial_1BND, 0)

        if self.parallel_config.tensor_parallel.factor > 1:
            # Gather spatial on TP axis before projection
            spatial_1BND = self.ccl_manager.all_gather_persistent_buffer(
                spatial_1BND, dim=3, mesh_axis=self.parallel_config.tensor_parallel.mesh_axis
            )

        if addcmul_residual is not None and addcmul_gate is not None:
            to_out_weight = self._to_out_weight_for_mm()
            n_out = to_out_weight.padded_shape[-1]
            if self._addcmul_gate_fits_fused_kernel(addcmul_gate, n_out):
                spatial_1BND = self._to_out_fused_addcmul(
                    spatial_1BND,
                    addcmul_residual,
                    addcmul_gate,
                    compute_kernel_config=self.mm_compute_kernel_config,
                    weight=to_out_weight,
                )
            else:
                spatial_1BND = self._to_out_unfused_addcmul(
                    spatial_1BND, addcmul_residual, addcmul_gate, compute_kernel_config=self.mm_compute_kernel_config
                )
        else:
            spatial_1BND = self.to_out(spatial_1BND, compute_kernel_config=self.mm_compute_kernel_config)

        if self.is_self and return_kv:
            return (spatial_1BND, k_cur, v_cur)
        return spatial_1BND
