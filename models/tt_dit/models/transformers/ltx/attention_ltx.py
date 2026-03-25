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

from ....layers.linear import ColParallelLinear, Linear
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
        apply_gated_attention: bool = False,
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

        # Per-head gate: linear(x) → 2*sigmoid → per-head scaling of SDPA output.
        # Runs on device with HiFi4 for precision.
        self.to_gate_logits = (
            Linear(query_input_dim or dim, num_heads, bias=True, mesh_device=mesh_device)
            if apply_gated_attention
            else None
        )

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
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        rename_substate(state, "to_out.0", "to_out")
        rename_substate(state, "q_norm", "norm_q")
        rename_substate(state, "k_norm", "norm_k")

        # Gate logits weights stay in state dict for device-side Linear loading.
        # If no gate weights in checkpoint, to_gate_logits module is unused.

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

    def _compute_gate(self, spatial_1BND: ttnn.Tensor) -> ttnn.Tensor | None:
        """Compute per-head gate on device: 2 * sigmoid(linear(x)).

        Returns gate tensor with shape (1, B, N, H) → reshaped to (B, H_local, N, 1) for BHNE multiply.
        Returns None if no gate module.
        """
        if self.to_gate_logits is None:
            return None

        # spatial_1BND is already TP-gathered at this point (by forward() line 403-406)
        gate_logits = self.to_gate_logits(spatial_1BND, compute_kernel_config=self.mm_compute_kernel_config)
        # gate_logits: (1, B, N, H_total). Apply 2*sigmoid.
        gate = ttnn.multiply(ttnn.sigmoid(gate_logits), 2.0)

        # Reshape (1, B, N, H) → (B, H, N, 1) for BHNE broadcast
        gate = ttnn.permute(gate, (1, 3, 2, 0))

        # TP-shard on head dim if needed
        if self.parallel_config.tensor_parallel.factor > 1:
            gate = ttnn.mesh_partition(gate, dim=1, cluster_axis=self.parallel_config.tensor_parallel.mesh_axis)

        return gate

    @staticmethod
    def _apply_split_rope(x: ttnn.Tensor, cos: ttnn.Tensor, sin: ttnn.Tensor, d_half: int) -> ttnn.Tensor:
        """Apply split-style rotary embedding: pairs (x[i], x[i+D/2]) are rotated."""
        x1 = x[:, :, :, :d_half]
        x2 = x[:, :, :, d_half:]
        out1 = ttnn.subtract(ttnn.multiply(x1, cos), ttnn.multiply(x2, sin))
        out2 = ttnn.add(ttnn.multiply(x2, cos), ttnn.multiply(x1, sin))
        return ttnn.concat([out1, out2], dim=-1)

    def _apply_split_rope_host(self, x: ttnn.Tensor, cos: ttnn.Tensor, sin: ttnn.Tensor, d_half: int) -> ttnn.Tensor:
        """Apply split rope on host — fallback for D_half sizes that cause TTNN subtile issues.

        Reads each device shard, applies rotation on host, reassembles the full
        tensor, and pushes back with the same sharding as x.
        """
        import torch

        device_x = ttnn.get_device_tensors(x)
        device_cos = ttnn.get_device_tensors(cos)
        device_sin = ttnn.get_device_tensors(sin)

        # Process each shard on host
        host_results = []
        for i, (dx, dc, ds) in enumerate(zip(device_x, device_cos, device_sin)):
            xh = ttnn.to_torch(dx).float()
            ch = ttnn.to_torch(dc).float()
            sh = ttnn.to_torch(ds).float()
            x1, x2 = xh[..., :d_half], xh[..., d_half:]
            out = torch.cat([x1 * ch - x2 * sh, x2 * ch + x1 * sh], dim=-1).bfloat16()
            host_results.append(out)

        if len(host_results) == 1:
            return ttnn.from_torch(
                host_results[0],
                device=self.mesh_device,
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat16,
            )

        # Multi-device: reassemble full tensor then re-shard.
        # x is sharded on heads (TP, dim 1) and seq (SP, dim 2).
        # Detect shard dims by comparing shard shape to expected full shape.
        shard_shape = host_results[0].shape
        n_devices = len(host_results)
        mesh_shape = tuple(self.mesh_device.shape)

        # Reconstruct by concatenating along sharded dims
        # For 2D mesh: dim 0 of mesh = SP (shards seq, dim 2), dim 1 = TP (shards heads, dim 1)
        sp_factor = self.parallel_config.sequence_parallel.factor
        tp_factor = self.parallel_config.tensor_parallel.factor
        sp_axis = self.parallel_config.sequence_parallel.mesh_axis
        tp_axis = self.parallel_config.tensor_parallel.mesh_axis

        # Reconstruct full tensor from shards
        # Shards are ordered row-major by (mesh_dim0, mesh_dim1)
        shards_2d = []
        for i in range(mesh_shape[0]):
            row = []
            for j in range(mesh_shape[1]):
                idx = i * mesh_shape[1] + j
                row.append(host_results[idx])
            # Concat along TP dim (heads, dim 1) within each row
            if tp_axis == 1 and tp_factor > 1:
                shards_2d.append(torch.cat(row, dim=1))
            else:
                shards_2d.append(row[0] if len(row) == 1 else torch.cat(row, dim=2))

        # Concat along SP dim (seq, dim 2) across rows
        if sp_axis == 0 and sp_factor > 1:
            full = torch.cat(shards_2d, dim=2)
        else:
            full = shards_2d[0] if len(shards_2d) == 1 else torch.cat(shards_2d, dim=1)

        # Re-shard with same mapping as x
        mapper = ttnn.ShardTensor2dMesh(
            self.mesh_device,
            mesh_shape=mesh_shape,
            dims=[
                2 if i == sp_axis and sp_factor > 1 else (1 if i == tp_axis and tp_factor > 1 else None)
                for i in range(2)
            ],
        )
        return ttnn.from_torch(
            full,
            device=self.mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            mesh_mapper=mapper,
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
        k_rope_cos: ttnn.Tensor | None = None,
        k_rope_sin: ttnn.Tensor | None = None,
        attn_mask: ttnn.Tensor | None = None,
        skip_qk: bool = False,
    ) -> ttnn.Tensor:
        """
        Same interface as WanAttention.forward().

        For cross-attention with positional embeddings (e.g. A↔V cross-attention),
        pass rope_cos/sin for Q and k_rope_cos/sin for K.
        """
        if rope_cos is not None:
            assert rope_sin is not None

        if self.parallel_config.tensor_parallel.factor > 1:
            spatial_1BND = self.ccl_manager.all_gather_persistent_buffer(
                spatial_1BND, dim=3, mesh_axis=self.parallel_config.tensor_parallel.mesh_axis
            )

        # Compute per-head gate on host (exact fp32) before QKV projections modify spatial_1BND.
        # Reference: gate_logits = to_gate_logits(x), then applied after SDPA as:
        #   out = out.view(B,T,H,D) * (2*sigmoid(gate_logits)).unsqueeze(-1)
        gate_bhne = self._compute_gate(spatial_1BND)

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
            # Determine K RoPE: use separate k_rope if provided (cross-attention),
            # otherwise use same rope as Q (self-attention).
            _k_cos = k_rope_cos if k_rope_cos is not None else rope_cos
            _k_sin = k_rope_sin if k_rope_sin is not None else rope_sin

            if trans_mat is not None:
                # Interleaved RoPE: rotary_embedding_llama with trans_mat
                q_BHNE = ttnn.experimental.rotary_embedding_llama(
                    q_BHNE, rope_cos, rope_sin, trans_mat, compute_kernel_config=self.rope_compute_kernel_config
                )
                k_BHNE = ttnn.experimental.rotary_embedding_llama(
                    k_BHNE, _k_cos, _k_sin, trans_mat, compute_kernel_config=self.rope_compute_kernel_config
                )
            else:
                # Split RoPE: manual elementwise rotation
                # out[:D/2] = x[:D/2]*cos - x[D/2:]*sin
                # out[D/2:] = x[D/2:]*cos + x[:D/2]*sin
                # cos/sin shape: (B, H, N, D/2) — half head_dim
                D_half = self.head_dim // 2
                # For cross-attention PE, D_half may be small (e.g. 32 for
                # audio_cross_attention_dim=2048, head_dim=64). TTNN elementwise
                # ops fail with "Invalid subtile broadcast type" when slicing
                # to 32 on the last dim. Fall back to host-side split rope
                # computation when D_half < 64 (minimum reliable tile width).
                q_d_half = rope_cos.shape[-1]
                if q_d_half < 64:
                    q_BHNE = self._apply_split_rope_host(q_BHNE, rope_cos, rope_sin, q_d_half)
                else:
                    q_BHNE = self._apply_split_rope(q_BHNE, rope_cos, rope_sin, D_half)

                k_d_half = _k_cos.shape[-1]
                if k_d_half < 64:
                    k_BHNE = self._apply_split_rope_host(k_BHNE, _k_cos, _k_sin, k_d_half)
                else:
                    k_BHNE = self._apply_split_rope(k_BHNE, _k_cos, _k_sin, D_half)

        if skip_qk:
            # STG perturbation: skip Q/K attention, use V passthrough.
            # Reference: out = to_v(context) when all_perturbed=True.
            spatial_BHNE = v_BHNE
        elif prompt_1BLP is None:
            if self.parallel_config.sequence_parallel.factor > 1 and attn_mask is None:
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
            elif self.parallel_config.sequence_parallel.factor > 1 and attn_mask is not None:
                # Ring attention does not support attn_mask. Gather K/V across SP
                # devices and use standard SDPA with the mask instead.
                # attn_mask shape is (1, 1, N_local, N_full) — already covers the
                # full K sequence, so no mask gathering is needed.
                sp_axis = self.parallel_config.sequence_parallel.mesh_axis
                k_full = self.ccl_manager.all_gather_persistent_buffer(k_BHNE, dim=2, mesh_axis=sp_axis)
                v_full = self.ccl_manager.all_gather_persistent_buffer(v_BHNE, dim=2, mesh_axis=sp_axis)
                spatial_BHNE = ttnn.transformer.scaled_dot_product_attention(
                    q_BHNE,
                    k_full,
                    v_full,
                    attn_mask=attn_mask,
                    is_causal=False,
                    program_config=self.sdpa_program_config,
                    compute_kernel_config=self.sdpa_compute_kernel_config,
                )
            else:
                spatial_BHNE = ttnn.transformer.scaled_dot_product_attention(
                    q_BHNE,
                    k_BHNE,
                    v_BHNE,
                    attn_mask=attn_mask,
                    is_causal=False,
                    program_config=self.sdpa_program_config,
                    compute_kernel_config=self.sdpa_compute_kernel_config,
                )
        else:
            spatial_BHNE = ttnn.transformer.scaled_dot_product_attention(
                q_BHNE,
                k_BHNE,
                v_BHNE,
                attn_mask=attn_mask,
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
