# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Full-attention (GQA) token mixer for the 16 ``full_attention`` layers.

    fused QKV+gate proj -> head split -> per-head QK-norm -> partial RoPE -> KV-cache write
    -> ring SDPA over the SP axis -> concat heads -> sigmoid output gate -> o_proj -> TP all-reduce

Two shapes differ from every sibling GQA package and both are load-bearing:

* **head_dim 256**, twice the usual, which halves how many k tokens fit a ring-SDPA chunk. The
  program config below is re-derived for it rather than inherited (recipe section 2.3).
* **an output gate**: ``q_proj`` emits ``2 * head_dim`` per head and the second half becomes
  ``sigmoid(gate)`` applied to the attention output *before* ``o_proj``. Dropping it leaves a
  perfectly well-formed attention block that is simply wrong.

Sequence parallelism is the ring SDPA's: each device's query shard attends the whole sequence
reconstructed across the SP ring by online softmax, with no explicit all-gather here.
"""

from __future__ import annotations

from typing import Optional

import ttnn
from models.common.lightweightmodule import LightweightModule

from ...config import MeshConfig
from ...reference.config import Qwen35TextConfig
from ..caches import PrefillCaches, write_kv_chunk
from ..compute import matmul_compute_config
from ..context import ChunkContext
from ..rope import apply_partial_rope
from .weights import AttentionWeights, load_attention_weights


def sdpa_program_config(mesh_device, head_dim: int) -> ttnn.SDPAProgramConfig:
    """Re-derived for head_dim 256, not inherited from a 128-wide donor.

    The ring op needs the CCL column carved out of the compute grid (they must not overlap), and
    ``k_chunk_size`` is what a core's L1 has to hold as ``k_chunk x head_dim``. The donor's
    ``k_chunk=512`` was measured at head_dim 128; at 256 the same L1 footprint is ``k_chunk=256``,
    so that is the starting point here. ``q_chunk`` stays 128 — it bounds the score block, which
    is ``q_chunk x k_chunk`` and independent of head_dim.
    """
    grid = mesh_device.compute_with_storage_grid_size()
    k_chunk = max(128, 512 * 128 // head_dim)
    return ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(grid.x - 1, grid.y),
        q_chunk_size=128,
        k_chunk_size=k_chunk,
        exp_approx_mode=False,
    )


def sdpa_compute_config(mesh_device, fp32_dest_acc: bool = True) -> ttnn.DeviceComputeKernelConfig:
    """Bring-up default: HiFi4 with ``fp32_dest_acc_en=True`` (recipe section 2.3).

    A narrower setting is a measurement, not an inheritance. ``fp32_dest_acc=False`` is passed
    explicitly by the cache-read call site if it turns out to be required there — that constraint
    belongs to the ring cache-read op, not to SDPA in general.
    """
    return ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=fp32_dest_acc,
        packer_l1_acc=False,
    )


class Attention(LightweightModule):
    def __init__(
        self,
        mesh_device,
        cfg: Qwen35TextConfig,
        state_dict: dict,
        *,
        mesh_config: MeshConfig,
        ccl_manager,
        layer_idx: int,
        weight_dtype=ttnn.bfloat8_b,
        activation_dtype=ttnn.bfloat16,
        cache_dtype=ttnn.bfloat8_b,
        tensor_cache_path: Optional[str] = None,
        weights: Optional[AttentionWeights] = None,
    ) -> None:
        self.mesh_device = mesh_device
        self.cfg = cfg
        self.mesh_config = mesh_config
        self.ccl_manager = ccl_manager
        self.layer_idx = layer_idx
        self.kv_slot = cfg.kv_slot(layer_idx)
        self.activation_dtype = activation_dtype
        self.cache_dtype = cache_dtype

        tp = mesh_config.tp
        self.n_q_local = cfg.num_attention_heads // tp
        self.n_kv_local = cfg.num_key_value_heads // tp
        self.head_dim = cfg.head_dim
        self.qkv_width = (self.n_q_local + 2 * self.n_kv_local) * cfg.head_dim
        self.gate_width = self.n_q_local * cfg.head_dim
        self.scale = cfg.head_dim**-0.5

        self.weights = weights or load_attention_weights(
            mesh_device,
            cfg,
            state_dict,
            mesh_config=mesh_config,
            weight_dtype=weight_dtype,
            tensor_cache_path=tensor_cache_path,
        )
        self.matmul_config = matmul_compute_config(mesh_device)
        self.program_config = sdpa_program_config(mesh_device, cfg.head_dim)
        self.compute_config = sdpa_compute_config(mesh_device)
        # The cache-read path only: ring_joint asserts
        #   "kv_actual_isl requires the ring-joint streaming compute path; the compute_common.hpp
        #    path selected by fp32_dest_acc_en=true is not supported"
        # so the cache-read call has to give up fp32 dest accumulation. Measured, not inherited:
        # it is the narrowing the op forces, and it is LOCAL to this one call — the live-K/V path
        # above keeps fp32_dest_acc_en=True. See the PCC table in README.md for what it costs.
        self.cache_read_compute_config = sdpa_compute_config(mesh_device, fp32_dest_acc=False)

    # --- pieces, each with its own PCC test ------------------------------------------------
    def project(self, x: ttnn.Tensor) -> tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
        """-> (q, k, v, gate). q/k/v are ``[1, n_local, S, head_dim]``; gate is ``[1, 1, S, n_q_local*head_dim]``."""
        fused = ttnn.linear(
            x, self.weights.wqkvg, dtype=self.activation_dtype, compute_kernel_config=self.matmul_config
        )
        shape = list(fused.shape)
        qkv = ttnn.slice(fused, [0, 0, 0, 0], shape[:-1] + [self.qkv_width])
        gate = ttnn.slice(fused, [0, 0, 0, self.qkv_width], shape)
        fused.deallocate(True)
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            qkv,
            num_heads=self.n_q_local,
            num_kv_heads=self.n_kv_local,
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        qkv.deallocate(True)
        return q, k, v, gate

    def qk_norm(self, q: ttnn.Tensor, k: ttnn.Tensor) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """Per-head RMSNorm over head_dim, applied BEFORE RoPE, on Q and K only.

        head_dim is not TP-sharded, so this is local to each chip — no cross-TP reduction. The
        Gemma ``(1 + w)`` fold is already baked into the gain at load.
        """
        eps = self.cfg.rms_norm_eps
        cfg = self.matmul_config
        return (
            ttnn.rms_norm(q, weight=self.weights.q_norm, epsilon=eps, compute_kernel_config=cfg),
            ttnn.rms_norm(k, weight=self.weights.k_norm, epsilon=eps, compute_kernel_config=cfg),
        )

    def sdpa(
        self,
        q: ttnn.Tensor,
        k: ttnn.Tensor,
        v: ttnn.Tensor,
        *,
        caches: Optional[PrefillCaches],
        cached_len: int,
    ) -> ttnn.Tensor:
        """Ring SDPA across the SP axis: live K/V on the first chunk, the packed cache after."""
        sp = self.mesh_config.sp
        s_local = q.shape[-2]
        common = dict(
            joint_strategy="rear",
            program_config=self.program_config,
            dim=2,
            multi_device_global_semaphore=self.ccl_manager.ring_attention_ccl_semaphore_handles,
            num_links=self.ccl_manager.num_links,
            cluster_axis=self.mesh_config.sp_axis,
            mesh_device=self.mesh_device,
            topology=ttnn.Topology.Linear,
            ccl_core_grid_offset=self.ccl_manager.ring_attention_ccl_core_grid_offset,
            use_column_major_ccl=True,
            is_causal=True,
            scale=self.scale,
            is_balanced=False,
        )
        n_kv_global = self.cfg.num_key_value_heads

        if cached_len == 0 or caches is None:
            out, _, _ = ttnn.transformer.ring_joint_scaled_dot_product_attention(
                q,
                k,
                v,
                None,
                None,
                None,
                persistent_output_buffer_k=self.ccl_manager.get_ring_gather_buffer(
                    "live_k", n_kv_global, s_local * sp, self.head_dim, self.activation_dtype
                ),
                persistent_output_buffer_v=self.ccl_manager.get_ring_gather_buffer(
                    "live_v", n_kv_global, s_local * sp, self.head_dim, self.activation_dtype
                ),
                logical_n=s_local * sp,
                compute_kernel_config=self.compute_config,
                **common,
            )
            return out

        kv = caches.kv
        out, _, _ = ttnn.transformer.ring_joint_scaled_dot_product_attention(
            q,
            kv.k,
            kv.v,
            None,
            None,
            None,
            # The gather buffer must span the FULL cache capacity: the op gathers the entire
            # per-device cache shard (seq_local rows x sp = max_seq_len) regardless of how much of
            # it is valid — logical_n / kv_actual_isl drive the causal masking of the unwritten tail.
            persistent_output_buffer_k=self.ccl_manager.get_ring_gather_buffer(
                "cache_k", n_kv_global, kv.max_seq_len, self.head_dim, self.cache_dtype
            ),
            persistent_output_buffer_v=self.ccl_manager.get_ring_gather_buffer(
                "cache_v", n_kv_global, kv.max_seq_len, self.head_dim, self.cache_dtype
            ),
            logical_n=cached_len + s_local * sp,
            # Fold the layer into the cache batch index, matching update_padded_kv_cache's write.
            # Passing the user slot alone makes every layer read layer 0's cache.
            kv_cache_batch_idx=kv.slot(0, self.kv_slot),
            kv_actual_isl=cached_len,
            compute_kernel_config=self.cache_read_compute_config,
            **common,
        )
        return out

    def forward(
        self,
        x: ttnn.Tensor,
        *,
        cos: ttnn.Tensor,
        sin: ttnn.Tensor,
        caches: Optional[PrefillCaches] = None,
        user_id: int = 0,
        cached_len: int = 0,
    ) -> ttnn.Tensor:
        q, k, v, gate = self.project(x)
        q_pre, k_pre = q, k
        q, k = self.qk_norm(q, k)
        q_pre.deallocate(True)
        k_pre.deallocate(True)

        q_pre, k_pre = q, k
        q = apply_partial_rope(q, cos, sin)
        k = apply_partial_rope(k, cos, sin)
        q_pre.deallocate(True)
        k_pre.deallocate(True)

        if caches is not None:
            write_kv_chunk(
                caches.kv,
                k,
                v,
                user_id=user_id,
                kv_slot=self.kv_slot,
                cached_len=cached_len,
                sp_axis=self.mesh_config.sp_axis,
            )

        attn = self.sdpa(q, k, v, caches=caches, cached_len=cached_len)
        q.deallocate(True)
        k.deallocate(True)
        v.deallocate(True)

        concat = ttnn.experimental.nlp_concat_heads(attn, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        attn.deallocate(True)

        # The output gate. Applied to the concatenated heads, before o_proj.
        gated = ttnn.multiply(concat, ttnn.sigmoid(gate))
        concat.deallocate(True)
        gate.deallocate(True)

        out = ttnn.linear(
            gated, self.weights.o_proj, dtype=self.activation_dtype, compute_kernel_config=self.matmul_config
        )
        gated.deallocate(True)
        if self.mesh_config.tp > 1:
            out = self.mesh_config.allreduce(out, self.ccl_manager, axis=self.mesh_config.tp_axis)
        return out

    def mix(self, x: ttnn.Tensor, ctx: ChunkContext) -> ttnn.Tensor:
        """The uniform token-mixer entry point ``DecoderLayer`` calls (see ``tt/context.py``)."""
        return self.forward(
            x, cos=ctx.cos, sin=ctx.sin, caches=ctx.caches, user_id=ctx.user_id, cached_len=ctx.cached_len
        )
