"""Narrow head-helper boundaries with L1 interleaved normalization."""

import ttnn
from models.autoports.google_gemma_4_31b_it.tests.optimized_prefill_candidate import PrefillCandidate


class TopologyCandidate(PrefillCandidate):
    @classmethod
    def from_state_dict(cls, state_dict, **kwargs):
        obj = super().from_state_dict(state_dict, **kwargs)
        if obj.geometry.get("separate_full") and not obj.sliding_window:
            packed = ttnn.to_memory_config(obj.weights.pop("self_attn.qk_proj.weight"), ttnn.DRAM_MEMORY_CONFIG)
            width = obj.nq * obj.head_dim
            import math

            grid = obj.mesh_device.dram_grid_size()
            cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
            for name, start, end in [("q", 0, width), ("k", width, packed.shape[-1])]:
                weight = ttnn.slice(packed, (0, start), (packed.shape[0], end))
                mem = ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                    ttnn.BufferType.DRAM,
                    ttnn.ShardSpec(
                        cores,
                        (weight.shape[0], math.ceil((end - start) / (32 * obj.banks)) * 32),
                        ttnn.ShardOrientation.ROW_MAJOR,
                    ),
                )
                obj.weights[f"self_attn.{name}_proj.weight"] = ttnn.to_memory_config(weight, mem)
        return obj

    def _qkv(self, x, cos, sin, *, decode):
        if decode:
            mem = self._decode_memory(x.shape[2])
            cos, sin = ttnn.to_memory_config(cos, mem), ttnn.to_memory_config(sin, mem)
        z = self._norm(x, "input_layernorm")
        if decode and x.shape[-2] > 32:
            z = ttnn.to_memory_config(z, ttnn.L1_MEMORY_CONFIG)
        if self.geometry.get("attn_act8"):
            z = ttnn.typecast(z, ttnn.bfloat8_b)
        if self.geometry.get("direct_qkv") and self.pack_attention and (not decode or x.shape[2] <= 32):
            projected = self._linear(z, "self_attn.qkv_proj")
            if decode:
                q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
                    projected,
                    num_heads=self.nq,
                    num_kv_heads=self.nkv,
                    memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
                    overlap_qk_coregrid=True,
                )
                q, k, v = [
                    ttnn.to_memory_config(
                        t, ttnn.L1_MEMORY_CONFIG if self.geometry.get("head_l1") else ttnn.DRAM_MEMORY_CONFIG
                    )
                    for t in (q, k, v)
                ]
            else:
                q, k, v = ttnn.experimental.nlp_create_qkv_heads(
                    projected, num_heads=self.nq, num_kv_heads=self.nkv, transpose_k_heads=False
                )
        elif self.sliding_window and not decode and (x.shape[0] >= 32 or self.geometry.get("pack_prefill_only")):
            q, k, v = ttnn.experimental.nlp_create_qkv_heads(
                self._linear(z, "self_attn.prefill_qkv"),
                num_heads=self.nq,
                num_kv_heads=self.nkv,
                transpose_k_heads=False,
            )
        elif self.sliding_window:
            qp = self._linear(z, "self_attn.q_proj")
            kp = self._linear(z, "self_attn.k_proj")
            vp = self._linear(z, "self_attn.v_proj")
            if decode and 16 <= x.shape[2] <= 32:
                q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
                    ttnn.concat([qp, kp, vp], dim=-1),
                    num_heads=self.nq,
                    num_kv_heads=self.nkv,
                    memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
                    overlap_qk_coregrid=True,
                )
                # RMSNorm cannot consume height-sharded heads in this runtime.
                q, k, v = [
                    ttnn.to_memory_config(
                        t, ttnn.L1_MEMORY_CONFIG if self.geometry.get("head_l1") else ttnn.DRAM_MEMORY_CONFIG
                    )
                    for t in (q, k, v)
                ]
            elif decode:
                q = ttnn.reshape(qp, (1, x.shape[2], self.nq, self.head_dim))
                k = ttnn.reshape(kp, (1, x.shape[2], self.nkv, self.head_dim))
                v = ttnn.reshape(vp, (1, x.shape[2], self.nkv, self.head_dim))
            else:
                q, k, v = ttnn.experimental.nlp_create_qkv_heads(
                    qp,
                    ttnn.concat([kp, vp], dim=-1),
                    num_heads=self.nq,
                    num_kv_heads=self.nkv,
                    transpose_k_heads=False,
                )
        else:
            qk = (
                ttnn.concat([self._linear(z, "self_attn.q_proj"), self._linear(z, "self_attn.k_proj")], dim=-1)
                if self.geometry.get("separate_full")
                else self._linear(z, "self_attn.qk_proj")
            )
            if decode and 16 <= x.shape[2] <= 32:
                width = self.nq * self.head_dim
                kp = ttnn.slice(qk, (0, 0, 0, width), tuple(qk.shape))
                q, unused_k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
                    ttnn.concat([qk, kp], dim=-1),
                    num_heads=self.nq,
                    num_kv_heads=self.nkv,
                    memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
                    overlap_qk_coregrid=True,
                )
                # The decode splitter has no tied-KV mode. V carries the shared
                # raw projection; only Q and V need interleaved norm inputs.
                q, v = [
                    ttnn.to_memory_config(
                        t, ttnn.L1_MEMORY_CONFIG if self.geometry.get("head_l1") else ttnn.DRAM_MEMORY_CONFIG
                    )
                    for t in (q, v)
                ]
                k = v
                del unused_k
            elif decode:
                width = self.nq * self.head_dim
                qp = ttnn.slice(qk, (0, 0, 0, 0), (*tuple(qk.shape)[:-1], width))
                kp = ttnn.slice(qk, (0, 0, 0, width), tuple(qk.shape))
                q = ttnn.reshape(qp, (1, x.shape[2], self.nq, self.head_dim))
                k = v = ttnn.reshape(kp, (1, x.shape[2], self.nkv, self.head_dim))
            else:
                q, k, v = ttnn.experimental.nlp_create_qkv_heads(
                    qk,
                    num_heads=self.nq,
                    num_kv_heads=self.nkv,
                    transpose_k_heads=False,
                    kv_tied=True,
                )
        q = self._rope(self._norm(q, "self_attn.q_norm"), cos, sin, decode=decode)
        v = self._norm(v)
        # Full attention normalizes the same raw K/V vector. Reuse its reduction;
        # the extra BF16 rounding is covered by real-weight and context PCC tests.
        k = (
            self._norm(k, "self_attn.k_norm")
            if self.sliding_window
            else ttnn.multiply(v, self.weights["self_attn.k_norm.weight"])
        )
        k = self._rope(k, cos, sin, decode=decode)
        if decode and not self.sliding_window and not self.geometry.get("keep_full_q"):
            # Long-context SDPA's static buffers collide with resident sharded Q
            # at D512, even after cache-update temporaries are released.
            q = ttnn.to_memory_config(q, ttnn.DRAM_MEMORY_CONFIG)
        if not decode and self.cache_dtype != ttnn.bfloat16:
            k, v = ttnn.typecast(k, self.cache_dtype), ttnn.typecast(v, self.cache_dtype)
        return q, k, v
