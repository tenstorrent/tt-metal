# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

import ttnn
from models.common.utility_functions import is_blackhole

from ....layers.linear import ColParallelLinear, LoRAColParallelLinear
from ....layers.module import Module
from ....layers.normalization import DistributedRMSNorm
from ....parallel.config import DiTParallelConfig
from ....parallel.manager import CCLManager
from ....utils.matmul import get_matmul_config, register_matmul_configs
from ....utils.substate import pop_substate, rename_substate
from ....utils.tensor import bf16_tensor

# Attention projection blockings for the CFG-batched rows (WanTransformer3DModel.inner_step_cfg) on WH Galaxy 4x8:
# 480p has 8192 rows per device at SP=8, TP=4. Swept with sweep_mm_block_sizes.py on the 8x8 AGMM worker grid.
register_matmul_configs(
    {
        "8x8": {
            (8192, 5120, 3840): (8, 10, 10, (2, 2)),  # to_qkv 4398.0 us
            (8192, 5120, 1280): (12, 8, 5, (4, 1)),  # to_out + addcmul 2305.0 us; cross to_q, to_out 1873.7 us
        }
    }
)


def _fold_heads(t: ttnn.Tensor) -> ttnn.Tensor:
    """[1, H, 2m, E] -> [1, 2H, m, E] (a view). With the rows of the two CFG passes stacked pass-major within each
    head, as in the head-split norm output of the stacked rows, head 2h + b becomes pass b of head h."""
    return ttnn.reshape(t, (1, 2 * t.shape[1], t.shape[2] // 2, t.shape[3]))


class WanAttention(Module):
    # Map from (is_blackhole, sp_factor, tp_factor) -> (q_chunk_size, k_chunk_size)
    sdpa_chunk_size_map = {
        (False, 2, 4): (256, 256),
        (False, 8, 4): (256, 512),  # WH Galaxy 4x8 480p sweep: 13.06 ms per block ring SDPA, (256, 256) 14.69 ms
        (True, 2, 2): (128, 512),
        (True, 8, 4): (288, 512),
        (True, 32, 4): (224, 512),
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
        ccl_manager: CCLManager | None = None,
        parallel_config: DiTParallelConfig,
        is_fsdp: bool = False,
        is_self: bool = True,
        sdpa_chunk_size_overrides: dict | None = None,
        lora_enabled: bool = False,
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
        ColCls = LoRAColParallelLinear if lora_enabled else ColParallelLinear

        if is_self:
            # Fused QKV for self-attention: single matmul split into 3 outputs
            self.to_qkv = ColCls(
                dim,
                3 * dim,
                chunks=3,
                **col_parallel_kwargs,
            )
        else:
            # Cross-attention: Q from spatial, K/V from prompt
            self.to_q = ColCls(dim, dim, **col_parallel_kwargs)
            # Fused KV: single matmul split into 2 outputs
            self.to_kv = ColCls(
                dim,
                2 * dim,
                chunks=2,
                **col_parallel_kwargs,
            )

        self.to_out = ColCls(
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
        self.sdpa_program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=full_grid,
            q_chunk_size=256,
            k_chunk_size=256,
            exp_approx_mode=False,  # NOTE: False is more correct
        )

        self.sdpa_worker_grid = (full_grid.x - 1, full_grid.y)  # Reserve last column for CCL
        chunk_lookup = {**self.sdpa_chunk_size_map, **(sdpa_chunk_size_overrides or {})}
        ring_sdpa_chunk_size = chunk_lookup.get(
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

        # With CFG batching, self-attention folds the cond and uncond passes into the head dim and runs one
        # ring SDPA with twice the heads. Its workers take the full width and leave the last row to the CCL:
        # on WH 4x8 480p the 2 x 10 heads x 16 Q chunks split evenly over 8x8 workers (5 per core), while
        # 7x9 leaves some cores one chunk more (22.39 vs 24.08 ms per block).
        self.fold_sdpa_worker_grid = (full_grid.x, full_grid.y - 1)
        self.fold_ring_sdpa_program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=self.fold_sdpa_worker_grid,
            q_chunk_size=ring_sdpa_chunk_size[0],
            k_chunk_size=ring_sdpa_chunk_size[1],
            exp_approx_mode=False,  # NOTE: False is more correct
        )
        self.fold_dummy_joint_input = bf16_tensor(
            torch.zeros((1, 2 * self.n_local_heads, 0, self.head_dim)), device=mesh_device
        )

        self.use_exp_ring_sdpa = (
            self.parallel_config.tensor_parallel.factor == 4 and self.parallel_config.sequence_parallel.factor == 32
        )

        if self.use_exp_ring_sdpa:
            self.exp_ring_sdpa_program_config = ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=full_grid,
                q_chunk_size=ring_sdpa_chunk_size[0],
                k_chunk_size=ring_sdpa_chunk_size[1],
                exp_approx_mode=False,
            )

        self.sdpa_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            self.mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,  # NOTE: Set to True if there's a correctness issue
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

    def _create_heads(self, x: ttnn.Tensor) -> ttnn.Tensor:
        out, _, _ = ttnn.experimental.nlp_create_qkv_heads(
            x,
            num_heads=self.n_local_heads,
            num_kv_heads=0,
            transpose_k_heads=False,
        )
        return out

    def _cross_attention_kv(self, prompt_1BLP: ttnn.Tensor | tuple[ttnn.Tensor, ttnn.Tensor]):
        """Head-split cross-attention K and V of the prompt: [1, H, L, E] each.

        For a (cond, uncond) prompt pair (CFG batching) the K / V of the two prompts are joined per head and folded
        like the query: [1, 2H, L, E], head 2h + b holds prompt b of head h.
        """
        prompts = prompt_1BLP if isinstance(prompt_1BLP, tuple) else (prompt_1BLP,)
        ks, vs = [], []
        for prompt in prompts:
            k_1BLF, v_1BLF = self.to_kv(prompt, compute_kernel_config=self.mm_compute_kernel_config)
            ks.append(self.norm_k(k_1BLF, num_heads_per_device=self.n_local_heads))
            vs.append(self._create_heads(v_1BLF))
        if len(prompts) == 1:
            return ks[0], vs[0]
        return _fold_heads(ttnn.concat(ks, dim=2)), _fold_heads(ttnn.concat(vs, dim=2))

    def _to_out_fused_addcmul(
        self,
        x: ttnn.Tensor,
        addcmul_residual: ttnn.Tensor,
        addcmul_gate: ttnn.Tensor,
        compute_kernel_config=None,
        parallel_config=None,
        dtype=None,
    ) -> ttnn.Tensor:
        """Fused to_out projection + addcmul: output = residual + (matmul(x, W) + bias) * gate."""
        to_out = self.to_out

        # Reads to_out.weight.data directly, so runtime-mode LoRA (which keeps
        # the delta in self._runtime_A/B) would silently no-op. Fuse mode is
        # fine — the delta lives in weight.data.
        if getattr(to_out, "lora_mode", None) == "runtime" and getattr(to_out, "is_lora_active", False):
            raise RuntimeError(
                "runtime LoRA mode is incompatible with WanAttention._to_out_fused_addcmul; "
                "construct to_out with lora_mode='fuse'"
            )

        # Handle FSDP weight gathering (mirrors ColParallelLinear.forward)
        if to_out.fsdp_mesh_axis is not None and to_out.mesh_device.shape[to_out.fsdp_mesh_axis] > 1:
            unsqueezed_weight = ttnn.unsqueeze_to_4D(to_out.weight.data)
            weight = self.ccl_manager.all_gather_persistent_buffer(
                unsqueezed_weight, dim=2, mesh_axis=to_out.fsdp_mesh_axis
            )
            weight = ttnn.reshape(weight, (weight.shape[-2], weight.shape[-1]))
        else:
            weight = to_out.weight.data

        if parallel_config is not None and parallel_config.tensor_parallel.factor > 1:
            M, K, N = x.padded_shape[-2], weight.padded_shape[-2], weight.padded_shape[-1]
            full_grid = self.mesh_device.compute_with_storage_grid_size()
            core_grid = ttnn.CoreCoord(full_grid.x, full_grid.y - 1)
            matmul_config = get_matmul_config(M, K, N, core_grid)

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
                1.0,  # scalar
                addcmul_residual,
                addcmul_gate,
                bias_tensor=to_out.bias.data if to_out.bias is not None else None,
                config=matmul_config,
                compute_kernel_config=compute_kernel_config or to_out.compute_config,
                dtype=dtype,
            )
        return output

    def forward(
        self,
        spatial_1BND: ttnn.Tensor,
        N: int,
        prompt_1BLP: ttnn.Tensor | None = None,
        cross_attn_mask: ttnn.Tensor | None = None,
        rope_cos: ttnn.Tensor | None = None,
        rope_sin: ttnn.Tensor | None = None,
        trans_mat: ttnn.Tensor | None = None,
        addcmul_residual: ttnn.Tensor | None = None,
        addcmul_gate: ttnn.Tensor | None = None,
        cfg_batch: int = 1,
    ) -> ttnn.Tensor:
        """
        spatial_1BND: fractured N on SP, fracturd D on TP
        prompt_1BLP: replicated on SP, replicated D on TP (optional); a (cond, uncond) pair when cfg_batch == 2
        cross_attn_mask: (optional, cross-attn only) mask for SDPA, shape [B|1, nqh|1, Sq, Sk].
            Must be TILE layout, bf16/bfp8/bfp4 dtype, on device in DRAM.
        rope_cos: fractured on SP, TP
        rope_sin: fractured on SP, TP
        trans_mat: replicated
        addcmul_residual: (optional) residual tensor for fused matmul+addcmul (self-attn only)
        addcmul_gate: (optional) gate tensor for fused matmul+addcmul (self-attn only)
        cfg_batch: 2 when spatial_1BND holds the rows of both classifier-free guidance passes, stacked per
            device ([1, 1, 2n, D]: cond rows, then uncond rows). Row-wise ops run once on the 2n rows, and
            attention folds the passes into the head dim ([1, 2H, n, E], head 2h + b is pass b of head h):
            self-attention runs one ring SDPA, and cross-attention gives each folded head the K/V of its
            own prompt.

        If prompt_1BLP is not provided, run self-attention.
        Otherwise, run cross-attention on prompt.

        When addcmul_residual and addcmul_gate are both provided (self-attention only),
        the to_out projection and residual addcmul are fused into a single op:
            output = addcmul_residual + to_out(attn_output) * addcmul_gate

        Outputs:
        spatial_1BND: fractured N on SP, fractured D on TP
        """

        if rope_cos is not None:
            # If ROPE is given, this is self-attention
            assert rope_sin is not None
            assert trans_mat is not None
            assert prompt_1BLP is None

        assert cfg_batch in (1, 2)
        if cfg_batch == 2:
            assert not self.use_exp_ring_sdpa, "CFG batching is not supported with the experimental ring SDPA"
            if not self.is_self:
                assert isinstance(prompt_1BLP, tuple) and len(prompt_1BLP) == 2, "expected a (cond, uncond) prompt pair"
                assert cross_attn_mask is None, "CFG batching does not support a cross-attention mask"

        def split_passes(t):
            # [1, 1, 2n, F] -> [1, 2, n, F], the batched layout the fused distributed RMSNorm accepts (a view)
            return ttnn.reshape(t, (1, 2, t.shape[2] // 2, t.shape[3])) if cfg_batch == 2 else t

        def fold_passes(t):
            return _fold_heads(t) if cfg_batch == 2 else t

        use_nonfused_agmm = (self.ccl_manager.topology == ttnn.Topology.Linear) and (
            self.parallel_config.tensor_parallel.factor > 1
        )
        if use_nonfused_agmm:
            spatial_1BND = self.ccl_manager.all_gather_persistent_buffer(
                spatial_1BND, dim=3, mesh_axis=self.parallel_config.tensor_parallel.mesh_axis
            )

        # Set norm output dtype to the input dtype required for ring self-attn.
        sdpa_input_dtype = getattr(self, "_sdpa_input_dtype", None)
        use_ring_sdpa = self.parallel_config.sequence_parallel.factor > 1
        norm_output_dtype = sdpa_input_dtype if (use_ring_sdpa and prompt_1BLP is None) else None

        if self.is_self:
            # Fused QKV matmul with split output for self-attention
            q_1BNF, k_1BNF, v_1BNF = self.to_qkv(
                spatial_1BND,
                compute_kernel_config=self.mm_compute_kernel_config,
                parallel_config=None if use_nonfused_agmm else self.parallel_config,
            )
        else:
            # Cross-attention: Q from spatial, fused KV from prompt
            assert prompt_1BLP is not None
            q_1BNF = self.to_q(
                spatial_1BND,
                compute_kernel_config=self.mm_compute_kernel_config,
                parallel_config=None if use_nonfused_agmm else self.parallel_config,
            )

        # Norm spatial before splitting heads
        q_BHNE = fold_passes(
            self.norm_q(
                split_passes(q_1BNF),
                num_heads_per_device=self.n_local_heads,
                rope_cos=rope_cos,
                rope_sin=rope_sin,
                trans_mat=trans_mat,
                dtype=norm_output_dtype,
            )
        )
        if self.is_self:
            k_BHNE = fold_passes(
                self.norm_k(
                    split_passes(k_1BNF),
                    num_heads_per_device=self.n_local_heads,
                    rope_cos=rope_cos,
                    rope_sin=rope_sin,
                    trans_mat=trans_mat,
                    dtype=norm_output_dtype,
                )
            )
            # V of the stacked passes splits into [1, H, 2n, E] like the norm output
            v_BHNE = fold_passes(self._create_heads(v_1BNF))
        else:
            k_BHNE, v_BHNE = self._cross_attention_kv(prompt_1BLP)

        if prompt_1BLP is None:
            # Self attention
            if self.parallel_config.sequence_parallel.factor > 1:
                # Q and K already cast by norm kernel; cast V and dummy joint inputs to match
                dummy_joint = self.fold_dummy_joint_input if cfg_batch == 2 else self.dummy_joint_input
                if sdpa_input_dtype is not None:
                    if v_BHNE.dtype != sdpa_input_dtype:
                        v_BHNE = ttnn.typecast(v_BHNE, sdpa_input_dtype)
                    if dummy_joint.dtype != sdpa_input_dtype:
                        dummy_joint = ttnn.typecast(dummy_joint, sdpa_input_dtype)

                # HACK: pass null joint inputs to take advantage of ring attention, even though this is self-attention.
                if self.use_exp_ring_sdpa:
                    spatial_BHNE, prompt_BHLE, _lse = ttnn.transformer.exp_ring_joint_scaled_dot_product_attention(
                        q_BHNE,
                        k_BHNE,
                        v_BHNE,
                        dummy_joint,
                        dummy_joint,
                        dummy_joint,
                        persistent_output_buffer_k=self.ccl_manager.get_ag_ping_pong_buffer(
                            k_BHNE.shape, 2, self.parallel_config.sequence_parallel.mesh_axis, dtype=k_BHNE.dtype
                        ),
                        persistent_output_buffer_v=self.ccl_manager.get_ag_ping_pong_buffer(
                            v_BHNE.shape, 2, self.parallel_config.sequence_parallel.mesh_axis, dtype=v_BHNE.dtype
                        ),
                        joint_strategy="rear",
                        logical_n=N,
                        program_config=self.exp_ring_sdpa_program_config,
                        compute_kernel_config=self.sdpa_compute_kernel_config,
                        dim=2,
                        multi_device_global_semaphore=self.ccl_manager.get_exp_ring_ping_pong_semaphore(
                            self.parallel_config.sequence_parallel.mesh_axis
                        ),
                        num_links=self.ccl_manager.num_links,
                        cluster_axis=self.parallel_config.sequence_parallel.mesh_axis,
                        mesh_device=self.mesh_device,
                        topology=self.ccl_manager.topology,
                        subdevice_id=self.ccl_manager.ccl_sub_device_id,
                        num_workers_per_link=5,
                        num_buffers_per_channel=32,
                    )
                else:
                    if cfg_batch == 2:
                        # folded heads: workers on the first rows, CCL on the last row
                        program_config = self.fold_ring_sdpa_program_config
                        ccl_core_grid_offset = (0, self.fold_sdpa_worker_grid[1])
                        use_column_major_ccl = False
                    else:
                        program_config = self.ring_sdpa_program_config
                        ccl_core_grid_offset = (self.sdpa_worker_grid[0], 0)
                        use_column_major_ccl = True
                    spatial_BHNE, prompt_BHLE, _lse = ttnn.transformer.ring_joint_scaled_dot_product_attention(
                        q_BHNE,
                        k_BHNE,
                        v_BHNE,
                        dummy_joint,
                        dummy_joint,
                        dummy_joint,
                        persistent_output_buffer_k=self.ccl_manager.get_ag_ping_pong_buffer(
                            k_BHNE.shape, 2, self.parallel_config.sequence_parallel.mesh_axis, dtype=k_BHNE.dtype
                        ),
                        persistent_output_buffer_v=self.ccl_manager.get_ag_ping_pong_buffer(
                            v_BHNE.shape, 2, self.parallel_config.sequence_parallel.mesh_axis, dtype=v_BHNE.dtype
                        ),
                        joint_strategy="rear",
                        logical_n=N,
                        program_config=program_config,
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
                        ccl_core_grid_offset=ccl_core_grid_offset,
                        use_column_major_ccl=use_column_major_ccl,
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
                attn_mask=cross_attn_mask,
                program_config=self.sdpa_program_config,
                compute_kernel_config=self.sdpa_compute_kernel_config,
            )

        if cfg_batch == 2:
            # [1, 2H, n, E] -> [1, H, 2n, E]: per head, the cond rows and then the uncond rows, like the input
            _, h2, n, e = spatial_BHNE.shape
            spatial_BHNE = ttnn.reshape(spatial_BHNE, (1, h2 // 2, 2 * n, e))
        spatial_1BND = ttnn.transformer.concatenate_heads(spatial_BHNE)
        spatial_1BND = ttnn.unsqueeze(spatial_1BND, 0)

        if use_nonfused_agmm:
            # Gather spatial on TP axis before projection
            spatial_1BND = self.ccl_manager.all_gather_persistent_buffer(
                spatial_1BND, dim=3, mesh_axis=self.parallel_config.tensor_parallel.mesh_axis
            )

        if addcmul_residual is not None and addcmul_gate is not None:
            # Fused to_out projection + addcmul (self-attention only)
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
                parallel_config=None if use_nonfused_agmm else self.parallel_config,
            )

        return spatial_1BND
