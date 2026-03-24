# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""TTNN WanAttention for Lingbot-VA.

Aligned with ``reference.transformer_wan.WanAttention`` (PyTorch):

1. **Project** inputs: ``query = to_q(q)``, ``key = to_k(k)``, ``value = to_v(v)``.
   On device we use fused ``ColParallelLinear`` (``to_qkv`` for self, ``to_q`` + ``to_kv`` for cross)
   with the same fused weight layout as ``_prepare_torch_state``.
2. **RMSNorm** on Q and K over full head dimension (``DistributedRMSNorm``); V is split to heads only.
3. **RoPE** on Q and K when ``rope_cos`` / ``rope_sin`` / ``trans_mat`` are set (self-attention).
   Cross-attention passes ``rope_* = None``, matching reference ``rotary_emb=None``.
4. **Optional prefix KV** for chunked inference: ``cached_k`` / ``cached_v`` concatenated on seq (dim 2),
   then ``ttnn.transformer.scaled_dot_product_attention``.
5. **Output projection** ``to_out`` (reference ``to_out[0]``; dropout ``to_out[1]`` is omitted on device).

Sequence-parallel ``factor > 1`` uses ring joint SDPA when there is no prefix cache and ``return_kv`` is false;
otherwise standard SDPA (matches constraints of ring kernels).
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
    """Multi-head attention matching ``reference.transformer_wan.WanAttention`` numerics (no host KV dict)."""

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
        ccl_manager: CCLManager | None,
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
        self.eps_float = eps
        self.mesh_device = mesh_device
        self.ccl_manager = ccl_manager
        self.parallel_config = parallel_config
        self.is_self = is_self
        self.n_local_heads = num_heads // parallel_config.tensor_parallel.factor
        fsdp_mesh_axis = parallel_config.sequence_parallel.mesh_axis if is_fsdp else None

        rms_kw = {
            "embedding_dim": dim,
            "norm_eps": eps,
            "norm_elementwise_affine": True,
            "bias": False,
            "mesh_device": mesh_device,
            "mesh_axis": parallel_config.tensor_parallel.mesh_axis,
            "ccl_manager": ccl_manager,
        }
        self.norm_q = DistributedRMSNorm(**rms_kw)
        self.norm_k = DistributedRMSNorm(**rms_kw)

        col_kw = {
            "bias": True,
            "mesh_device": mesh_device,
            "mesh_axis": parallel_config.tensor_parallel.mesh_axis,
            "fsdp_mesh_axis": fsdp_mesh_axis,
            "ccl_manager": ccl_manager,
        }
        if is_self:
            self.to_qkv = ColParallelLinear(dim, 3 * dim, chunks=3, **col_kw)
            self.to_q = None
            self.to_kv = None
        else:
            self.to_q = ColParallelLinear(dim, dim, **col_kw)
            self.to_kv = ColParallelLinear(dim, 2 * dim, chunks=2, **col_kw)
            self.to_qkv = None

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

        grid = mesh_device.compute_with_storage_grid_size()
        sp, tp = parallel_config.sequence_parallel.factor, parallel_config.tensor_parallel.factor
        q_chunk, k_chunk = (128, 128) if sp == 1 and tp == 1 else (256, 256)
        self.sdpa_program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=grid,
            q_chunk_size=q_chunk,
            k_chunk_size=k_chunk,
            exp_approx_mode=False,
        )
        self.sdpa_worker_grid = (grid.x, grid.y - 1)
        ring_chunks = self.sdpa_chunk_size_map.get(
            (is_blackhole(), sp, parallel_config.tensor_parallel.factor),
            self.default_sdpa_chunk_size,
        )
        self.ring_sdpa_program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=self.sdpa_worker_grid,
            q_chunk_size=ring_chunks[0],
            k_chunk_size=ring_chunks[1],
            exp_approx_mode=False,
        )
        self.sdpa_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
        )
        self.rope_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        self.mm_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    # --- API parity with reference WanAttention (CPU KV lives on WanTransformer3DModel._attn_caches) ---
    def clear_pred_cache(self, cache_name: str) -> None:
        return

    def clear_cache(self, cache_name: str) -> None:
        return

    def init_kv_cache(self, *args, **kwargs) -> None:
        return

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        rename_substate(state, "to_out.0", "to_out")

        def _interleave_heads(tensors: list[torch.Tensor]) -> torch.Tensor:
            """Merge separate Q/K/V Linear weights for column-parallel sharded heads."""
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
                    [
                        q_state["bias"].unsqueeze(-1),
                        k_state["bias"].unsqueeze(-1),
                        v_state["bias"].unsqueeze(-1),
                    ]
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
        to_out = self.to_out
        if to_out.fsdp_mesh_axis is not None and to_out.mesh_device.shape[to_out.fsdp_mesh_axis] > 1:
            w = ttnn.unsqueeze_to_4D(to_out.weight.data)
            weight = self.ccl_manager.all_gather_persistent_buffer(w, dim=2, mesh_axis=to_out.fsdp_mesh_axis)
            weight = ttnn.reshape(weight, (weight.shape[-2], weight.shape[-1]))
        else:
            weight = to_out.weight.data

        m, k, n = x.padded_shape[-2], x.padded_shape[-1], weight.padded_shape[-1]
        matmul_config = get_matmul_config(m, k, n, self.mesh_device.compute_with_storage_grid_size())
        return ttnn.experimental.dit_minimal_matmul_addcmul_fused(
            x,
            weight,
            1.0,
            addcmul_residual,
            addcmul_gate,
            bias_tensor=to_out.bias.data if to_out.bias is not None else None,
            config=matmul_config,
            compute_kernel_config=compute_kernel_config or to_out.compute_config,
        )

    def _maybe_all_gather_spatial(self, spatial_1BND: ttnn.Tensor) -> ttnn.Tensor:
        if self.parallel_config.tensor_parallel.factor <= 1:
            return spatial_1BND
        return self.ccl_manager.all_gather_persistent_buffer(
            spatial_1BND, dim=3, mesh_axis=self.parallel_config.tensor_parallel.mesh_axis
        )

    def _maybe_all_gather_out(self, spatial_1BND: ttnn.Tensor) -> ttnn.Tensor:
        if self.parallel_config.tensor_parallel.factor <= 1:
            return spatial_1BND
        return self.ccl_manager.all_gather_persistent_buffer(
            spatial_1BND, dim=3, mesh_axis=self.parallel_config.tensor_parallel.mesh_axis
        )

    def _split_v_heads(self, v_1BNF: ttnn.Tensor) -> ttnn.Tensor:
        out, _, _ = ttnn.experimental.nlp_create_qkv_heads(
            v_1BNF,
            num_heads=self.n_local_heads,
            num_kv_heads=0,
            transpose_k_heads=False,
        )
        return out

    def _run_sdpa(
        self,
        q_BHNE: ttnn.Tensor,
        k_BHNE: ttnn.Tensor,
        v_BHNE: ttnn.Tensor,
        N: int,
        *,
        use_ring: bool,
    ) -> ttnn.Tensor:
        if use_ring:
            out, _, _ = ttnn.transformer.ring_joint_scaled_dot_product_attention(
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
                ccl_core_grid_offset=(0, self.sdpa_worker_grid[1]),
            )
            return out
        return ttnn.transformer.scaled_dot_product_attention(
            q_BHNE,
            k_BHNE,
            v_BHNE,
            is_causal=False,
            program_config=self.sdpa_program_config,
            compute_kernel_config=self.sdpa_compute_kernel_config,
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
        cached_k: ttnn.Tensor | None = None,
        cached_v: ttnn.Tensor | None = None,
        return_kv: bool = False,
    ) -> ttnn.Tensor | tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
        if rope_cos is not None:
            assert rope_sin is not None and trans_mat is not None and prompt_1BLP is None

        x = self._maybe_all_gather_spatial(spatial_1BND)
        mm = self.mm_compute_kernel_config

        if self.is_self:
            q_1BNF, k_1BNF, v_1BNF = self.to_qkv(x, compute_kernel_config=mm)
        else:
            assert prompt_1BLP is not None
            q_1BNF = self.to_q(x, compute_kernel_config=mm)
            k_1BNF, v_1BNF = self.to_kv(prompt_1BLP, compute_kernel_config=mm)
            if len(k_1BNF.shape) == 3:
                k_1BNF = ttnn.unsqueeze(k_1BNF, 0)
                v_1BNF = ttnn.unsqueeze(v_1BNF, 0)

        q_BHNE = self.norm_q(
            q_1BNF,
            num_heads_per_device=self.n_local_heads,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
            trans_mat=trans_mat,
        )
        k_BHNE = self.norm_k(
            k_1BNF,
            num_heads_per_device=self.n_local_heads,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
            trans_mat=trans_mat,
        )
        v_BHNE = self._split_v_heads(v_1BNF)

        k_cur = v_cur = None
        if self.is_self and return_kv:
            k_cur, v_cur = k_BHNE, v_BHNE

        if self.is_self and cached_k is not None and cached_v is not None:
            cached_k = ttnn.to_layout(cached_k, ttnn.TILE_LAYOUT)
            cached_v = ttnn.to_layout(cached_v, ttnn.TILE_LAYOUT)
            k_BHNE = ttnn.concat([cached_k, k_BHNE], dim=2)
            v_BHNE = ttnn.concat([cached_v, v_BHNE], dim=2)

        sp = self.parallel_config.sequence_parallel.factor
        use_ring = (
            prompt_1BLP is None and sp > 1 and cached_k is None and not return_kv and self.ccl_manager is not None
        )
        out_BHNE = self._run_sdpa(q_BHNE, k_BHNE, v_BHNE, N, use_ring=use_ring)

        out_1BND = ttnn.transformer.concatenate_heads(out_BHNE)
        out_1BND = ttnn.unsqueeze(out_1BND, 0)
        out_1BND = self._maybe_all_gather_out(out_1BND)

        if addcmul_residual is not None and addcmul_gate is not None:
            out_1BND = self._to_out_fused_addcmul(out_1BND, addcmul_residual, addcmul_gate, compute_kernel_config=mm)
        else:
            out_1BND = self.to_out(out_1BND, compute_kernel_config=mm)

        if self.is_self and return_kv:
            return (out_1BND, k_cur, v_cur)
        return out_1BND
