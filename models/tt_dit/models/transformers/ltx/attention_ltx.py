# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

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
    # Map from (is_blackhole, sp_factor, tp_factor) -> (q_chunk_size, k_chunk_size)
    sdpa_chunk_size_map = {
        (False, 2, 4): (256, 256),
        (False, 8, 4): (256, 256),
        (True, 2, 2): (128, 512),
        (True, 8, 4): (128, 512),
    }
    default_sdpa_chunk_size = (256, 256)

    # Per-stage ring-SDPA chunk, keyed by (is_blackhole, sp, tp, N); N is the SP-padded
    # sequence length passed to the op. Misses fall back to sdpa_chunk_size_map.
    ring_sdpa_chunk_by_n = {
        (True, 8, 4, 9728): (96, 256),
        (True, 8, 4, 38912): (192, 512),
    }

    # Per-shape cross-attn SDPA chunk, keyed by (is_blackhole, q_seq, kv_seq); seqs are
    # the per-device Q shard and full K. Misses fall back to sdpa_program_config.
    sdpa_chunk_by_shape = {
        (True, 1216, 32): (128, 128),  # video text cross-attn, stage 1
        (True, 4864, 32): (192, 128),  # video text cross-attn, stage 2
        (True, 1216, 256): (128, 128),  # audio->video cross-attn, stage 1
        (True, 4864, 256): (192, 256),  # audio->video cross-attn, stage 2
    }

    # V2A cross ring-SDPA q_chunk (per-device audio Q = audio_N / sp_factor), keyed by
    # (is_blackhole, sp, tp); assumes audio_N=256. k_chunk reuses the self-attn ring value; misses
    # fall back to the self-attn ring q_chunk.
    # TODO: audio_N depends on video duration (ceil(round((num_frames/fps)*25), 32*sp)); derive
    # q_chunk from the actual Q shard (q_BHNE.shape[2]) instead of hardcoding per mesh.
    cross_ring_sdpa_q_chunk_map = {
        (True, 4, 2): 64,  # BH 2x4
        (True, 8, 4): 32,  # BH 4x8
    }

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

        self.kv_input_dim = context_dim if (context_dim is not None and not is_self) else dim

        if is_self:
            self.to_qkv = ColParallelLinear(dim, 3 * dim, chunks=3, **col_parallel_kwargs)
        else:
            self.to_q = ColParallelLinear(self.query_input_dim, dim, **col_parallel_kwargs)
            self.to_kv = ColParallelLinear(self.kv_input_dim, 2 * dim, chunks=2, **col_parallel_kwargs)

        self.to_out = ColParallelLinear(
            dim,
            self.output_dim,
            bias=True,
            mesh_device=mesh_device,
            mesh_axis=parallel_config.tensor_parallel.mesh_axis,
            fsdp_mesh_axis=fsdp_mesh_axis,
            ccl_manager=ccl_manager,
        )

        # Per-head gate, sharded on num_heads to match the SDPA-output head layout. bf16 matches the
        # reference (which runs the gate in the model's working dtype). Sigmoid is a standalone op:
        # fusing it into the matmul trips minimal_matmul's sigmoid VecMode assert (needs C/RC). The
        # ×2 stays a separate multiply (2·sigmoid is nonlinear, can't fold).
        self.apply_gated_attention = apply_gated_attention
        if apply_gated_attention:
            self.to_gate_logits = ColParallelLinear(
                in_features=self.query_input_dim,
                out_features=self.num_heads,
                bias=True,
                dtype=ttnn.bfloat16,
                mesh_device=mesh_device,
                mesh_axis=parallel_config.tensor_parallel.mesh_axis,
                ccl_manager=ccl_manager,
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
        mesh_key = (
            is_blackhole(),
            self.parallel_config.sequence_parallel.factor,
            self.parallel_config.tensor_parallel.factor,
        )
        ring_sdpa_chunk_size = self.sdpa_chunk_size_map.get(mesh_key, self.default_sdpa_chunk_size)
        self.ring_sdpa_program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=self.sdpa_worker_grid,
            q_chunk_size=ring_sdpa_chunk_size[0],
            k_chunk_size=ring_sdpa_chunk_size[1],
            exp_approx_mode=False,
        )
        self._ring_pc_by_n = {
            n: ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=self.sdpa_worker_grid,
                q_chunk_size=chunk[0],
                k_chunk_size=chunk[1],
                exp_approx_mode=False,
            )
            for (b, sp, tp, n), chunk in self.ring_sdpa_chunk_by_n.items()
            if (b, sp, tp) == mesh_key
        }
        self._sdpa_pc_by_shape = {
            (q, kv): ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=full_grid,
                q_chunk_size=chunk[0],
                k_chunk_size=chunk[1],
                exp_approx_mode=False,
            )
            for (b, q, kv), chunk in self.sdpa_chunk_by_shape.items()
            if b == mesh_key[0]
        }

        # V2A cross ring SDPA: q_chunk matched to the per-device audio Q; k_chunk reuses the
        # self-attn ring value.
        cross_ring_q_chunk = self.cross_ring_sdpa_q_chunk_map.get(mesh_key, ring_sdpa_chunk_size[0])
        self.cross_ring_sdpa_program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=self.sdpa_worker_grid,
            q_chunk_size=cross_ring_q_chunk,
            k_chunk_size=ring_sdpa_chunk_size[1],
            exp_approx_mode=False,
        )

        # All SDPA (ring + cross) runs HiFi2, matching the Wan attention config.
        self.sdpa_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            self.mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
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

        # Attention QKV/out matmuls run HiFi2, matching the Wan attention config.
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

        # Permute Q/K (and norm_q/norm_k) channels per head from checkpoint SPLIT rotation to
        # the INTERLEAVED layout rotary_embedding_llama expects: even output lanes take the
        # first half of each head's channels, odd lanes the second half.
        D = self.head_dim
        D_half = D // 2
        perm = torch.empty(D, dtype=torch.long)
        perm[0::2] = torch.arange(D_half)
        perm[1::2] = torch.arange(D_half, D)

        def _permute_qk(t: torch.Tensor) -> torch.Tensor:
            rest = t.shape[1:]
            t = t.reshape(self.num_heads, D, *rest).index_select(1, perm)
            return t.reshape(self.num_heads * D, *rest)

        for nk in ("norm_q.weight", "norm_k.weight"):
            if nk in state:
                state[nk] = _permute_qk(state[nk])

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

            q_state["weight"] = _permute_qk(q_state["weight"])
            k_state["weight"] = _permute_qk(k_state["weight"])
            if "bias" in q_state:
                q_state["bias"] = _permute_qk(q_state["bias"])
            if "bias" in k_state:
                k_state["bias"] = _permute_qk(k_state["bias"])

            state["to_qkv.weight"] = _interleave_heads([q_state["weight"], k_state["weight"], v_state["weight"]])
            if "bias" in q_state:
                bias = _interleave_heads(
                    [q_state["bias"].unsqueeze(-1), k_state["bias"].unsqueeze(-1), v_state["bias"].unsqueeze(-1)]
                )
                state["to_qkv.bias"] = bias.squeeze(-1)
        else:
            k_state = pop_substate(state, "to_k")
            v_state = pop_substate(state, "to_v")

            k_state["weight"] = _permute_qk(k_state["weight"])
            if "bias" in k_state:
                k_state["bias"] = _permute_qk(k_state["bias"])

            if "to_q.weight" in state:
                state["to_q.weight"] = _permute_qk(state["to_q.weight"])
            if "to_q.bias" in state:
                state["to_q.bias"] = _permute_qk(state["to_q.bias"])

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
        parallel_config: DiTParallelConfig | None = None,
        dtype=None,
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

        if parallel_config is not None and parallel_config.tensor_parallel.factor > 1:
            M, K, N_out = x.padded_shape[-2], weight.padded_shape[-2], weight.padded_shape[-1]
            full_grid = self.mesh_device.compute_with_storage_grid_size()
            core_grid = ttnn.CoreCoord(full_grid.x, full_grid.y - 1)
            matmul_config = get_matmul_config(M, K, N_out, core_grid)

            ag_persistent_buffer = self.ccl_manager.get_ag_ping_pong_buffer(
                x.shape, 3, parallel_config.tensor_parallel.mesh_axis, dtype=x.get_dtype()
            )
            ag_global_semaphores = self.ccl_manager.get_ag_ping_pong_semaphore(
                parallel_config.tensor_parallel.mesh_axis
            )
            output = ttnn.experimental.all_gather_minimal_matmul_async(
                input_tensor=x,
                weight_tensor=weight,
                bias_tensor=to_out.bias.data if to_out.bias is not None else None,
                config=matmul_config,
                compute_kernel_config=compute_kernel_config or to_out.compute_config,
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
                dtype=dtype,
            )[0]
        else:
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
                dtype=dtype,
            )
        return output

    def _compute_gate(
        self, spatial_1BND: ttnn.Tensor, qkv_parallel_config: DiTParallelConfig | None
    ) -> ttnn.Tensor | None:
        """Per-head gate 2 * sigmoid(to_gate_logits(x)); returns (B, H_local, N, 1) or None."""
        if not self.apply_gated_attention or self.to_gate_logits.weight._data is None:
            return None

        gate_logits = self.to_gate_logits(spatial_1BND, parallel_config=qkv_parallel_config)
        gate = ttnn.multiply(ttnn.sigmoid(gate_logits), 2.0)

        # (1, B, N, H_local) -> (B, H_local, N, 1) in one pass: the N<->H_local swap is the real
        # data movement, and the leading unit axis is parked into the trailing broadcast slot (over
        # E). Replaces squeeze+transpose+unsqueeze (the unsqueeze was a retile, not a free view).
        gate = ttnn.permute(gate, (1, 3, 2, 0))
        return gate

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
        kv_replicated: bool = False,
        kv_logical_n: int | None = None,
    ) -> ttnn.Tensor:
        """Same interface as WanAttention.forward(); pass k_rope_cos/sin for separate K RoPE
        in A2V/V2A cross-attention."""
        if rope_cos is not None:
            assert rope_sin is not None
            assert trans_mat is not None, "INTERLEAVED RoPE requires trans_mat (load-time Q/K permute assumes it)"

        use_nonfused_agmm = (self.ccl_manager.topology == ttnn.Topology.Linear) and (
            self.parallel_config.tensor_parallel.factor > 1
        )
        if use_nonfused_agmm:
            spatial_1BND = self.ccl_manager.all_gather_persistent_buffer(
                spatial_1BND, dim=3, mesh_axis=self.parallel_config.tensor_parallel.mesh_axis
            )

        qkv_parallel_config = None if use_nonfused_agmm else self.parallel_config

        # Per-head gate, computed before QKV consumes spatial_1BND.
        gate_bhne = self._compute_gate(spatial_1BND, qkv_parallel_config)

        if self.is_self:
            q_1BNF, k_1BNF, v_1BNF = self.to_qkv(
                spatial_1BND,
                compute_kernel_config=self.mm_compute_kernel_config,
                parallel_config=qkv_parallel_config,
            )
        else:
            kv_input = prompt_1BLP if prompt_1BLP is not None else spatial_1BND
            # Cross K/V: gather TP-sharded context for to_kv (replicated text prompt is already full).
            kv_parallel_config = None
            if prompt_1BLP is not None and self.parallel_config.tensor_parallel.factor > 1:
                local_k = kv_input.shape[-1]
                kv_is_tp_sharded = local_k * self.parallel_config.tensor_parallel.factor == self.kv_input_dim
                if kv_is_tp_sharded:
                    if use_nonfused_agmm:
                        kv_input = self.ccl_manager.all_gather_persistent_buffer(
                            kv_input, dim=3, mesh_axis=self.parallel_config.tensor_parallel.mesh_axis
                        )
                    else:
                        kv_parallel_config = self.parallel_config
            q_1BNF = self.to_q(
                spatial_1BND,
                compute_kernel_config=self.mm_compute_kernel_config,
                parallel_config=qkv_parallel_config,
            )
            k_1BNF, v_1BNF = self.to_kv(
                kv_input,
                compute_kernel_config=self.mm_compute_kernel_config,
                parallel_config=kv_parallel_config,
            )

        # RMSNorm on Q/K fused with the head split (emits BHNE via num_heads_per_device).
        q_BHNE = self.norm_q(q_1BNF, num_heads_per_device=self.n_local_heads)
        k_BHNE = self.norm_k(k_1BNF, num_heads_per_device=self.n_local_heads)

        def create_heads(inp):
            out, _, _ = ttnn.experimental.nlp_create_qkv_heads(
                inp, num_heads=self.n_local_heads, num_kv_heads=0, transpose_k_heads=False
            )
            return out

        # V still goes through the explicit reshape (no norm to fuse with).
        v_BHNE = create_heads(v_1BNF)

        # Cross-attn K/V must be full-seq for SDPA; gather across SP only when genuinely sharded.
        is_cross = prompt_1BLP is not None
        sp_factor = self.parallel_config.sequence_parallel.factor
        _k_cos_pe = k_rope_cos if k_rope_cos is not None else rope_cos
        # V2A cross: K/V stay SP-sharded (caller passes the sharded K-rope and kv_logical_n) so the
        # ring SDPA fuses the gather instead of an explicit K/V all-gather + local SDPA.
        use_ring_cross = is_cross and sp_factor > 1 and not kv_replicated and kv_logical_n is not None
        if is_cross and sp_factor > 1 and not use_ring_cross:
            sp_axis = self.parallel_config.sequence_parallel.mesh_axis
            if kv_replicated:
                need_gather = False
            elif _k_cos_pe is not None:
                need_gather = k_BHNE.shape[2] < _k_cos_pe.shape[2]
            else:
                # Unknown sharded context with no rope reference: gather conservatively.
                need_gather = True
            if need_gather:
                k_BHNE = self.ccl_manager.all_gather_persistent_buffer(k_BHNE, dim=2, mesh_axis=sp_axis)
                v_BHNE = self.ccl_manager.all_gather_persistent_buffer(v_BHNE, dim=2, mesh_axis=sp_axis)

        if rope_cos is not None:
            _k_cos = _k_cos_pe
            _k_sin = k_rope_sin if k_rope_sin is not None else rope_sin
            q_BHNE = ttnn.experimental.rotary_embedding_llama(
                q_BHNE, rope_cos, rope_sin, trans_mat, compute_kernel_config=self.rope_compute_kernel_config
            )
            k_BHNE = ttnn.experimental.rotary_embedding_llama(
                k_BHNE, _k_cos, _k_sin, trans_mat, compute_kernel_config=self.rope_compute_kernel_config
            )

        if skip_qk:
            # STG perturbation: skip Q/K attention, use V passthrough.
            spatial_BHNE = v_BHNE
        elif prompt_1BLP is None:
            if sp_factor > 1 and attn_mask is None:
                spatial_BHNE, _prompt_BHLE, _lse = ttnn.transformer.ring_joint_scaled_dot_product_attention(
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
                    program_config=self._ring_pc_by_n.get(N, self.ring_sdpa_program_config),
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
                    ccl_core_grid_offset=(self.sdpa_worker_grid[0], 0),
                    use_column_major_ccl=True,
                )
            elif sp_factor > 1:
                # Masked audio self-attn: gather K/V, keep Q sharded; gather+local SDPA beats ring-joint here.
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
        elif use_ring_cross:
            # Short audio Q attends non-causally to the SP-sharded video K/V; is_cross fuses the
            # K/V gather into the ring SDPA. Output is the per-device Q shard (same as local SDPA).
            sp_mesh_axis = self.parallel_config.sequence_parallel.mesh_axis
            spatial_BHNE, _prompt_BHLE, _lse = ttnn.transformer.ring_joint_scaled_dot_product_attention(
                q_BHNE,
                k_BHNE,
                v_BHNE,
                self.dummy_joint_input,
                self.dummy_joint_input,
                self.dummy_joint_input,
                persistent_output_buffer_k=self.ccl_manager.get_ag_ping_pong_buffer(k_BHNE.shape, 2, sp_mesh_axis),
                persistent_output_buffer_v=self.ccl_manager.get_ag_ping_pong_buffer(v_BHNE.shape, 2, sp_mesh_axis),
                joint_strategy="rear",
                logical_n=kv_logical_n,
                is_cross=True,
                program_config=self.cross_ring_sdpa_program_config,
                compute_kernel_config=self.sdpa_compute_kernel_config,
                dim=2,
                multi_device_global_semaphore=self.ccl_manager.get_ag_ping_pong_semaphore(sp_mesh_axis),
                num_links=self.ccl_manager.num_links,
                cluster_axis=sp_mesh_axis,
                mesh_device=self.mesh_device,
                topology=self.ccl_manager.topology,
                subdevice_id=self.ccl_manager.ccl_sub_device_id,
                ccl_core_grid_offset=(self.sdpa_worker_grid[0], 0),
                use_column_major_ccl=True,
            )
        else:
            # Cross-attention: K/V full-seq, Q SP-sharded so local SDPA returns the local shard.
            spatial_BHNE = ttnn.transformer.scaled_dot_product_attention(
                q_BHNE,
                k_BHNE,
                v_BHNE,
                attn_mask=attn_mask,
                is_causal=False,
                program_config=self._sdpa_pc_by_shape.get((q_BHNE.shape[2], k_BHNE.shape[2]), self.sdpa_program_config),
                compute_kernel_config=self.sdpa_compute_kernel_config,
            )

        # Apply per-head gate in BHNE space.
        if gate_bhne is not None:
            spatial_BHNE = ttnn.multiply(spatial_BHNE, gate_bhne)

        spatial_1BND = ttnn.transformer.concatenate_heads(spatial_BHNE)
        spatial_1BND = ttnn.unsqueeze(spatial_1BND, 0)

        # Ring fuses the TP all-gather into the to_out matmul; only Linear needs explicit AG.
        addcmul_fused = addcmul_residual is not None and addcmul_gate is not None
        to_out_explicit_ag = self.parallel_config.tensor_parallel.factor > 1 and use_nonfused_agmm
        if to_out_explicit_ag:
            spatial_1BND = self.ccl_manager.all_gather_persistent_buffer(
                spatial_1BND, dim=3, mesh_axis=self.parallel_config.tensor_parallel.mesh_axis
            )

        if addcmul_fused:
            spatial_1BND = self._to_out_fused_addcmul(
                spatial_1BND,
                addcmul_residual,
                addcmul_gate,
                compute_kernel_config=self.mm_compute_kernel_config,
                parallel_config=None if use_nonfused_agmm else self.parallel_config,
            )
        else:
            spatial_1BND = self.to_out(
                spatial_1BND,
                compute_kernel_config=self.mm_compute_kernel_config,
                parallel_config=None if to_out_explicit_ag else self.parallel_config,
            )

        return spatial_1BND
