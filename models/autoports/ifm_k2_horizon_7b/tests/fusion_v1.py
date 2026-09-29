"""Fused single-chip IFM/K2-Horizon-7B dense decoder.

Setup owns HF conversion. Forward methods consume/return device TTNN tensors.
Prefill: x [1,B,S,4096], rope pair [B,1,S,128], paged cache pair
[physical_pages,8,32,128], INT32 ROW_MAJOR page_table [B,logical_pages].
Decode: x [1,1,B,4096], rope pair [1,B,32,128] (identical head rows),
INT32 ROW_MAJOR current_pos [B]; output has the input shape.
Positions are absolute, zero based. The caller owns cache allocation and page
ownership; rows must have disjoint physical pages. Prefill extends a prefix of
``start_pos`` tokens and returns every logical output. Fresh requests start at 0.
RoPE is supplied at the input boundary and must match the absolute positions.
No runtime tensor readback or host tensor conversion occurs in this module.
"""

from dataclasses import dataclass

import ttnn
from models.common.lightweightmodule import LightweightModule


@dataclass
class PrefillPlan:
    """Setup-time metadata for unaligned prefix continuation."""

    start_pos: int
    seq_len: int
    leading_positions: tuple


class InitialFusion(LightweightModule):
    page_size = 32
    chunk_size = 256

    @classmethod
    def from_state_dict(cls, state_dict, *, hf_config, layer_idx, mesh_device):
        import torch

        from ..tt.accurate_attention import _load, accurate_attention

        _load()  # Host extension/kernel-path loading is an explicit setup boundary.
        c = hf_config
        if (
            c.hidden_size,
            c.intermediate_size,
            c.num_attention_heads,
            c.num_key_value_heads,
            c.head_dim,
            c.layernorm_num_groups,
        ) != (4096, 12288, 32, 8, 128, 4):
            raise ValueError("Expected the real K2-Horizon-7B dimensions")
        if c.num_experts or c.query_key_norm or c.attention_bias or c.attention_gate_func or c.use_sliding_window:
            raise ValueError("Only the target dense, ungated, full-attention configuration is supported")
        if c.rope_parameters != {"rope_theta": 10000000.0, "rope_type": "default"}:
            raise ValueError("Unexpected target RoPE configuration")
        if not 0 <= layer_idx < c.num_hidden_layers:
            raise ValueError("Invalid layer index")
        self = cls()
        self.mesh_device = mesh_device
        self._attention = accurate_attention
        self.eps = c.rms_norm_eps
        self.context = c.max_position_embeddings
        self.compute = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        prefix = f"model.layers.{layer_idx}."

        def weight(name):
            return state_dict[prefix + name].detach().to(torch.bfloat16)

        def device(tensor):
            return ttnn.from_torch(
                tensor.contiguous(),
                device=mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        self.wqkv = device(torch.cat([weight(f"self_attn.{p}_proj.weight").T for p in "qkv"], dim=-1))
        self.wo = device(weight("self_attn.o_proj.weight").T)
        self.wgate = device(weight("mlp.gate_proj.weight").T)
        self.wup = device(weight("mlp.up_proj.weight").T)
        self.wdown = device(weight("mlp.down_proj.weight").T)
        self.norm1 = device(weight("input_layernorm.weight").reshape(1, 1, 1, 4096))
        self.norm2 = device(weight("post_attention_layernorm.weight").reshape(1, 1, 1, 4096))
        self.norm_groups = {
            id(w): tuple(w[..., i : i + 1024] for i in range(0, 4096, 1024)) for w in (self.norm1, self.norm2)
        }
        return self

    def prepare_prefill(self, *, seq_len, start_pos=0):
        """Call outside measurement/capture; produces constant position inputs."""
        import torch

        if seq_len < 1 or start_pos < 0 or start_pos + seq_len > self.context:
            raise ValueError("Prefill must fit the advertised context")
        leading = min(seq_len, (-start_pos) % self.page_size)
        positions = tuple(
            ttnn.from_torch(
                torch.tensor([p], dtype=torch.int32),
                device=self.mesh_device,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
            for p in range(start_pos, start_pos + leading)
        )
        return PrefillPlan(start_pos, seq_len, positions)

    def _linear(self, x, w):
        return ttnn.linear(
            x, w, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG, compute_kernel_config=self.compute
        )

    def _norm(self, x, weight):
        # HF computes independent RMS statistics in four contiguous groups.
        groups = [
            ttnn.rms_norm(x[..., i : i + 1024], epsilon=self.eps, weight=w, compute_kernel_config=self.compute)
            for i, w in zip(range(0, 4096, 1024), self.norm_groups[id(weight)])
        ]
        return ttnn.concat(groups, dim=-1)

    @staticmethod
    def _rope(x, rope):
        return ttnn.experimental.rotary_embedding(x, rope[0], rope[1])

    def _decode_rope(self, x, rope):
        # The dedicated kernel broadcasts one request's table over heads.
        # Keep request-specific positions by invoking it per request.
        rows = [
            ttnn.experimental.rotary_embedding(
                x[:, b : b + 1, :, :], rope[0][:, b : b + 1, :, :], rope[1][:, b : b + 1, :, :], token_index=0
            )
            for b in range(x.shape[1])
        ]
        return ttnn.concat(rows, dim=1) if len(rows) > 1 else rows[0]

    def _decode_qk(self, q, k, rope):
        return (
            self._decode_rope(ttnn.to_memory_config(q, ttnn.DRAM_MEMORY_CONFIG), rope),
            self._decode_rope(ttnn.to_memory_config(k, ttnn.DRAM_MEMORY_CONFIG), rope),
        )

    def _update_cache(self, kv_cache, k, v, current_pos, page_table):
        for cache, update in zip(kv_cache, (k, v)):
            ttnn.experimental.paged_update_cache(cache, update, update_idxs_tensor=current_pos, page_table=page_table)

    def _finish(self, x, attention):
        residual = ttnn.add(x, self._linear(attention, self.wo))
        normed = self._norm(residual, self.norm2)
        mlp = ttnn.multiply(
            self._linear(normed, self.wgate),
            self._linear(normed, self.wup),
            input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
        )
        return ttnn.add(residual, self._linear(mlp, self.wdown))

    def decode_forward(self, x, *, rope, kv_cache, page_table, current_pos):
        """One device-only token pass; tensor positions/page tables can change on replay."""
        batch = x.shape[2]
        if tuple(x.shape)[:2] != (1, 1) or not 1 <= batch <= 32:
            raise ValueError("Decode expects [1,1,B,4096], 1 <= B <= 32")
        qkv = self._linear(self._norm(x, self.norm1), self.wqkv)
        q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
            qkv, num_heads=32, num_kv_heads=8, memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG
        )
        q, k = self._decode_qk(q, k, rope)
        grid = ttnn.num_cores_to_corerangeset(batch, ttnn.CoreCoord(8, 8), row_wise=True)
        shard = ttnn.create_sharded_memory_config(
            (32, 128),
            core_grid=grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        k = ttnn.to_memory_config(k, shard)
        v = ttnn.to_memory_config(v, shard)
        self._update_cache(kv_cache, k, v, current_pos, page_table)
        # Preserve FP32 long-context recurrence for decode as well as prefill.
        # All position arithmetic and row selection stays on device. Repeated Q
        # rows cover the current 32-token page; only the current row is returned.
        per_request = []
        for b in range(batch):
            pos = ttnn.to_layout(ttnn.reshape(current_pos[b : b + 1], (1, 1, 1, 1)), ttnn.TILE_LAYOUT)
            offset = ttnn.reshape(ttnn.to_layout(ttnn.bitwise_and(pos, -32), ttnn.ROW_MAJOR_LAYOUT), (1,))
            index = ttnn.repeat(ttnn.typecast(ttnn.bitwise_and(pos, 31), ttnn.uint32), (1, 32, 1, 128))
            query = ttnn.repeat(ttnn.permute(q[:, b : b + 1, :, :], (0, 2, 1, 3)), (1, 1, 32, 1))
            table = page_table[b : b + 1, :]
            # Flexible SDPA requires a 32-byte table row. Unused entries point
            # to this request's final page and are hidden by the causal mask.
            padding = (-table.shape[1]) % 8
            if padding:
                table = ttnn.concat([table, ttnn.repeat(table[:, -1:], (1, padding))], dim=1)
            attended = self._attention(
                query, kv_cache[0], kv_cache[1], table, chunk_start_idx_tensor=offset, q_chunk_size=32, k_chunk_size=128
            )
            per_request.append(ttnn.permute(ttnn.gather(attended, 2, index), (0, 2, 1, 3)))
        attention = ttnn.concat(per_request, dim=1) if batch > 1 else per_request[0]
        attention = ttnn.experimental.nlp_concat_heads_decode(ttnn.to_memory_config(attention, shard), num_heads=32)
        attention = ttnn.to_memory_config(attention, ttnn.DRAM_MEMORY_CONFIG)
        attention = attention[:, :, :batch, :]
        return self._finish(x, attention)

    def _prefill_chunk(self, x, *, rope, kv_cache, page_table, start_pos):
        logical = x.shape[2]
        physical = (logical + 31) // 32 * 32
        if physical != logical:
            x = ttnn.pad(x, [(0, 0), (0, 0), (0, physical - logical), (0, 0)], 0)
            rope = tuple(ttnn.pad(r, [(0, 0), (0, 0), (0, physical - logical), (0, 0)], 0) for r in rope)
        qkv = self._linear(self._norm(x, self.norm1), self.wqkv)
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            qkv, num_heads=32, num_kv_heads=8, transpose_k_heads=False, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        q, k = self._rope(q, rope), self._rope(k, rope)
        chunk_table = page_table[:, start_pos // 32 : (start_pos + physical) // 32]
        for cache, update in zip(kv_cache, (k, v)):
            ttnn.experimental.paged_fill_cache(cache, update, chunk_table, batch_idx=0)
        sdpa_chunk = 128 if physical % 128 == 0 and start_pos % 128 == 0 else 32
        attention = self._attention(
            q,
            kv_cache[0],
            kv_cache[1],
            page_table,
            chunk_start_idx=start_pos,
            q_chunk_size=sdpa_chunk,
            k_chunk_size=sdpa_chunk,
        )
        attention = ttnn.experimental.nlp_concat_heads(attention, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return self._finish(x, attention)[:, :, :logical, :]

    def prefill_forward(self, x, *, rope, kv_cache, page_table, plan):
        """Chunked paged prefill, arbitrary logical S and absolute prefix continuation.

        Padding occupies only unused positions beyond the logical request end;
        causal attention prevents it from affecting valid rows. Subsequent
        continuation overwrites those unused positions before reading them.
        """
        if x.shape[2] != plan.seq_len or x.shape[1] != page_table.shape[0]:
            raise ValueError("Input and plan/page-table shapes disagree")
        rows = []
        for b in range(x.shape[1]):
            table = page_table[b : b + 1, :]
            outputs = []
            offset = 0
            for position in plan.leading_positions:
                one_rope = tuple(ttnn.repeat(r[b : b + 1, :, offset : offset + 1, :], (1, 1, 32, 1)) for r in rope)
                outputs.append(
                    self.decode_forward(
                        x[:, b : b + 1, offset : offset + 1, :],
                        rope=one_rope,
                        kv_cache=kv_cache,
                        page_table=table,
                        current_pos=position,
                    )
                )
                offset += 1
            while offset < plan.seq_len:
                end = min(offset + self.chunk_size, plan.seq_len)
                outputs.append(
                    self._prefill_chunk(
                        x[:, b : b + 1, offset:end, :],
                        rope=tuple(r[b : b + 1, :, offset:end, :] for r in rope),
                        kv_cache=kv_cache,
                        page_table=table,
                        start_pos=plan.start_pos + offset,
                    )
                )
                offset = end
            rows.append(ttnn.concat(outputs, dim=2) if len(outputs) > 1 else outputs[0])
        return ttnn.concat(rows, dim=1) if len(rows) > 1 else rows[0]
