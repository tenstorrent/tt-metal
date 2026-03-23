# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
LTX-2 Attention module for tt_dit.

Mirrors WanAttention with LTX-2 defaults (dim=4096, 32 heads, 128 head_dim).
Supports self-attention (fused QKV) and cross-attention (Q + fused KV).

Reference: LTX-2 attention.py + Wan attention_wan.py
"""

from __future__ import annotations

import torch

import ttnn
from models.common.utility_functions import is_blackhole

from ....layers.linear import ColParallelLinear
from ....layers.module import Module
from ....layers.normalization import DistributedRMSNorm
from ....parallel.config import DiTParallelConfig
from ....parallel.manager import CCLManager
from ....utils.matmul import get_matmul_config
from ....utils.substate import pop_substate, rename_substate
from ....utils.tensor import bf16_tensor


class LTXAttention(Module):
    """
    LTX-2 attention block. Structurally identical to WanAttention.

    Supports:
    - Self-attention with fused QKV, RMSNorm on Q/K, RoPE
    - Cross-attention with Q + fused KV, RMSNorm on Q/K
    - Ring attention for sequence parallelism
    - Fused to_out + addcmul for gated residual connections
    - Per-head gating: 2 * sigmoid(linear(x)) applied to SDPA output
    """

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
        eps: float = 1e-6,
        mesh_device: ttnn.MeshDevice,
        ccl_manager: CCLManager | None = None,
        parallel_config: DiTParallelConfig,
        is_fsdp: bool = False,
        is_self: bool = True,
        context_dim: int | None = None,
        query_input_dim: int | None = None,
        output_dim: int | None = None,
    ) -> None:
        super().__init__()

        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qk_norm = qk_norm
        self.eps = eps
        self.is_self = is_self
        self.query_input_dim = query_input_dim or dim
        self.output_dim = output_dim or dim

        self.mesh_device = mesh_device
        self.ccl_manager = ccl_manager
        self.parallel_config = parallel_config

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

        kv_input_dim = context_dim if (context_dim is not None and not is_self) else dim

        if is_self:
            self.to_qkv = ColParallelLinear(dim, 3 * dim, chunks=3, **col_parallel_kwargs)
        else:
            self.to_q = ColParallelLinear(self.query_input_dim, dim, **col_parallel_kwargs)
            self.to_kv = ColParallelLinear(kv_input_dim, 2 * dim, chunks=2, **col_parallel_kwargs)

        self.to_out = ColParallelLinear(
            dim,
            self.output_dim,
            bias=True,
            mesh_device=mesh_device,
            mesh_axis=parallel_config.tensor_parallel.mesh_axis,
            fsdp_mesh_axis=fsdp_mesh_axis,
            ccl_manager=ccl_manager,
        )

        # Per-head gate weights stored on host for exact fp32 gate computation.
        # Gate logits are small (32 outputs), so host F.linear is fast and avoids
        # TT bf16 matmul precision issues on the K=4096 reduction.
        self._gate_weight_host = None  # (num_heads, query_input_dim)
        self._gate_bias_host = None  # (num_heads,)

        self.dummy_joint_input = bf16_tensor(torch.zeros((1, self.n_local_heads, 0, self.head_dim)), device=mesh_device)

        full_grid = self.mesh_device.compute_with_storage_grid_size()
        self.sdpa_program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=full_grid,
            q_chunk_size=256,
            k_chunk_size=256,
            exp_approx_mode=False,
        )

        self.sdpa_worker_grid = (full_grid.x - 1, full_grid.y)
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
            exp_approx_mode=False,
        )

        self.sdpa_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            self.mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
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

        self.mm_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            self.mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=True,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        rename_substate(state, "to_out.0", "to_out")
        rename_substate(state, "q_norm", "norm_q")
        rename_substate(state, "k_norm", "norm_k")

        # Extract gate weights to host and remove from device loading.
        # Gate logits are computed on host (tiny matmul: N×32 vs N×4096 for QKV)
        # to get exact fp32 precision. The gate values are then pushed to device
        # as bf16 for the per-head multiply on the SDPA output.
        gate_state = pop_substate(state, "to_gate_logits")
        if gate_state:
            self._gate_weight_host = gate_state["weight"].float()  # (num_heads, query_input_dim)
            self._gate_bias_host = gate_state.get("bias", torch.zeros(self.num_heads)).float()

        def _interleave_heads(tensors: list[torch.Tensor]):
            n_dev = self.parallel_config.tensor_parallel.factor
            tensors = [t.T for t in tensors]
            tensors = [t.reshape(t.shape[0], n_dev, self.n_local_heads, self.head_dim) for t in tensors]
            merged = torch.cat(tensors, dim=2)
            merged = merged.reshape(merged.shape[0], len(tensors) * self.num_heads * self.head_dim)
            return merged.T

        if self.is_self:
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
            k_state = pop_substate(state, "to_k")
            v_state = pop_substate(state, "to_v")

            state["to_kv.weight"] = _interleave_heads([k_state["weight"], v_state["weight"]])
            if "bias" in k_state:
                bias = _interleave_heads([k_state["bias"].unsqueeze(-1), v_state["bias"].unsqueeze(-1)])
                state["to_kv.bias"] = bias.squeeze(-1)

    def _to_out_fused_addcmul(
        self,
        x: ttnn.Tensor,
        addcmul_residual: ttnn.Tensor,
        addcmul_gate: ttnn.Tensor,
        compute_kernel_config=None,
    ) -> ttnn.Tensor:
        """Fused to_out projection + addcmul: output = residual + (matmul(x, W) + bias) * gate."""
        to_out = self.to_out

        if to_out.fsdp_mesh_axis is not None and to_out.mesh_device.shape[to_out.fsdp_mesh_axis] > 1:
            unsqueezed_weight = ttnn.unsqueeze_to_4D(to_out.weight.data)
            weight = self.ccl_manager.all_gather_persistent_buffer(
                unsqueezed_weight, dim=2, mesh_axis=to_out.fsdp_mesh_axis
            )
            weight = ttnn.reshape(weight, (weight.shape[-2], weight.shape[-1]))
        else:
            weight = to_out.weight.data

        M, K, N_out = x.padded_shape[-2], x.padded_shape[-1], weight.padded_shape[-1]
        core_grid = self.mesh_device.compute_with_storage_grid_size()
        matmul_config = get_matmul_config(M, K, N_out, core_grid)

        output = ttnn.experimental.dit_minimal_matmul_addcmul_fused(
            x,
            weight,
            1.0,
            addcmul_residual,
            addcmul_gate,
            bias_tensor=to_out.bias.data if to_out.bias is not None else None,
            config=matmul_config,
            compute_kernel_config=compute_kernel_config or to_out.compute_config,
        )
        return output

    def _compute_gate_on_host(self, spatial_1BND: ttnn.Tensor) -> ttnn.Tensor | None:
        """Compute per-head gate on host for exact fp32 precision.

        Gate logits = linear(x, W_gate, b_gate) → 2*sigmoid(logits)
        Returns gate tensor on device with shape (B, H_local, N, 1) for BHNE broadcast multiply.
        Returns None if no gate weights are loaded.
        """
        if self._gate_weight_host is None:
            return None

        # Read spatial from device for host-side gate computation
        spatial_host = ttnn.to_torch(ttnn.get_device_tensors(spatial_1BND)[0]).float()
        gate_logits = torch.nn.functional.linear(spatial_host, self._gate_weight_host, self._gate_bias_host)
        gate_values = 2.0 * torch.sigmoid(gate_logits)  # (1, B, N, H_total) in fp32

        # Permute to BHNE format: (B, H_total, N, 1)
        gate_bhne = gate_values.permute(1, 3, 2, 0).contiguous().bfloat16()

        # Shard across TP devices if needed
        tp_factor = self.parallel_config.tensor_parallel.factor
        if tp_factor > 1:
            tp_axis = self.parallel_config.tensor_parallel.mesh_axis
            mapper = ttnn.ShardTensor2dMesh(
                self.mesh_device,
                mesh_shape=tuple(self.mesh_device.shape),
                dims=[None if i != tp_axis else 1 for i in range(2)],
            )
            return ttnn.from_torch(
                gate_bhne,
                device=self.mesh_device,
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat16,
                mesh_mapper=mapper,
            )
        else:
            return ttnn.from_torch(
                gate_bhne,
                device=self.mesh_device,
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat16,
            )

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
    ) -> ttnn.Tensor:
        """
        Same interface as WanAttention.forward().
        """
        if rope_cos is not None:
            assert rope_sin is not None
            assert trans_mat is not None
            assert prompt_1BLP is None

        if self.parallel_config.tensor_parallel.factor > 1:
            spatial_1BND = self.ccl_manager.all_gather_persistent_buffer(
                spatial_1BND, dim=3, mesh_axis=self.parallel_config.tensor_parallel.mesh_axis
            )

        # Compute per-head gate on host (exact fp32) before QKV projections modify spatial_1BND.
        # Reference: gate_logits = to_gate_logits(x), then applied after SDPA as:
        #   out = out.view(B,T,H,D) * (2*sigmoid(gate_logits)).unsqueeze(-1)
        gate_bhne = self._compute_gate_on_host(spatial_1BND)

        if self.is_self:
            q_1BNF, k_1BNF, v_1BNF = self.to_qkv(spatial_1BND, compute_kernel_config=self.mm_compute_kernel_config)
        else:
            kv_input = prompt_1BLP if prompt_1BLP is not None else spatial_1BND
            q_1BNF = self.to_q(spatial_1BND, compute_kernel_config=self.mm_compute_kernel_config)
            k_1BNF, v_1BNF = self.to_kv(kv_input, compute_kernel_config=self.mm_compute_kernel_config)

        # RMSNorm on Q/K (RoPE applied per-head after head split)
        q_normed = self.norm_q(q_1BNF)
        k_normed = self.norm_k(k_1BNF)

        def create_heads(inp):
            out, _, _ = ttnn.experimental.nlp_create_qkv_heads(
                inp, num_heads=self.n_local_heads, num_kv_heads=0, transpose_k_heads=False
            )
            return out

        q_BHNE = create_heads(q_normed)
        k_BHNE = create_heads(k_normed)
        v_BHNE = create_heads(v_1BNF)

        if rope_cos is not None:
            # LTX-2 uses INTERLEAVED RoPE; cos/sin must be pre-converted to SPLIT layout.
            q_BHNE = ttnn.experimental.rotary_embedding_llama(
                q_BHNE, rope_cos, rope_sin, trans_mat, compute_kernel_config=self.rope_compute_kernel_config
            )
            k_BHNE = ttnn.experimental.rotary_embedding_llama(
                k_BHNE, rope_cos, rope_sin, trans_mat, compute_kernel_config=self.rope_compute_kernel_config
            )

        if prompt_1BLP is None:
            if self.parallel_config.sequence_parallel.factor > 1:
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
                    topology=ttnn.Topology.Linear,
                    subdevice_id=self.ccl_manager.ccl_sub_device_id,
                    ccl_core_grid_offset=(self.sdpa_worker_grid[0], 0),
                    use_column_major_ccl=True,
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
            spatial_BHNE = ttnn.transformer.scaled_dot_product_attention(
                q_BHNE,
                k_BHNE,
                v_BHNE,
                is_causal=False,
                program_config=self.sdpa_program_config,
                compute_kernel_config=self.sdpa_compute_kernel_config,
            )

        # Apply per-head gate in BHNE space (before concatenate_heads).
        # Mathematically equivalent to the reference which applies gate after concat_heads
        # in (B,T,H,D) format — both multiply each head's output by its scalar gate.
        if gate_bhne is not None:
            spatial_BHNE = ttnn.multiply(spatial_BHNE, gate_bhne)

        spatial_1BND = ttnn.transformer.concatenate_heads(spatial_BHNE)
        spatial_1BND = ttnn.unsqueeze(spatial_1BND, 0)

        if self.parallel_config.tensor_parallel.factor > 1:
            spatial_1BND = self.ccl_manager.all_gather_persistent_buffer(
                spatial_1BND, dim=3, mesh_axis=self.parallel_config.tensor_parallel.mesh_axis
            )

        if addcmul_residual is not None and addcmul_gate is not None:
            spatial_1BND = self._to_out_fused_addcmul(
                spatial_1BND, addcmul_residual, addcmul_gate, compute_kernel_config=self.mm_compute_kernel_config
            )
        else:
            spatial_1BND = self.to_out(spatial_1BND, compute_kernel_config=self.mm_compute_kernel_config)

        return spatial_1BND
