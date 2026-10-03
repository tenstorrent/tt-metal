# SPDX-License-Identifier: Apache-2.0
"""Single-chip Granite decoder. Setup owns weights; callers own paged KV storage.

Prefill consumes [1,1,S,4096] per request, with explicit absolute start and slot.
Decode consumes [1,1,B,4096], device positions [B], page table [B,pages],
and device HF cos/sin [1,B,32,128] (first row used), in height-sharded L1.
The decode method is entirely device-side and may be captured by TTNN.
"""

from dataclasses import dataclass, replace

import ttnn
from models.common.lightweightmodule import LightweightModule


@dataclass
class PrefillEntry:
    """Stable device inputs plus logical output length. Construct outside forward."""

    x: ttnn.Tensor
    cos: ttnn.Tensor
    sin: ttnn.Tensor
    position: ttnn.Tensor
    page_table: ttnn.Tensor
    write_page_table: ttnn.Tensor | None
    valid_tokens: int


class FunctionalDecoder(LightweightModule):
    @classmethod
    def from_state_dict(cls, state_dict, *, hf_config, layer_idx, mesh_device, chunk_size=1024):
        import torch

        c = hf_config
        if (c.hidden_size, c.intermediate_size, c.num_attention_heads, c.num_key_value_heads) != (4096, 32768, 32, 8):
            raise ValueError("Requires the pinned Granite 4.2 30B config")
        if c.attention_bias or c.mlp_bias or c.hidden_act != "silu":
            raise ValueError("Unsupported bias or activation")
        if chunk_size != 1024:
            raise ValueError("The prepared prefill physical chunk is 1024 tokens")
        obj = cls()
        obj.device = mesh_device
        obj.config = c
        obj.chunk_size = chunk_size
        obj.compute = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
        )
        prefix = f"model.layers.{layer_idx}."

        def get(key):
            return state_dict[prefix + key] if prefix + key in state_dict else state_dict[key]

        def upload(w):
            return ttnn.from_torch(
                w.contiguous(),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        obj.qkv = upload(torch.cat([get(f"self_attn.{k}_proj.weight") for k in ("q", "k", "v")]).T)
        obj.o = upload(get("self_attn.o_proj.weight").T)
        obj.gate = upload(get("mlp.gate_proj.weight").T)
        obj.up = upload(get("mlp.up_proj.weight").T)
        obj.down = upload(get("mlp.down_proj.weight").T)
        obj.norm1 = upload(get("input_layernorm.weight").reshape(1, 1, 1, -1))
        obj.norm2 = upload(get("post_attention_layernorm.weight").reshape(1, 1, 1, -1))
        grid = mesh_device.compute_with_storage_grid_size()
        obj.decode_grid = (grid.x, 8)
        obj.worker_grid = ttnn.num_cores_to_corerangeset(grid.x * grid.y, grid, row_wise=True)
        obj.single_rope_memory = ttnn.create_sharded_memory_config(
            (32, 128), ttnn.CoreGrid(y=1, x=1), ttnn.ShardStrategy.HEIGHT, use_height_and_width_as_shard_shape=True
        )
        obj.head_memory = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1)
        return obj

    def linear(self, x, w):
        return ttnn.linear(
            x, w, compute_kernel_config=self.compute, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16
        )

    def norm(self, x, weight):
        # HF rounds the normalized value to the input dtype before weight multiply.
        n = ttnn.rms_norm(x, epsilon=self.config.rms_norm_eps, compute_kernel_config=self.compute)
        return ttnn.multiply(n, weight)

    def finish(self, x, attention):
        x = ttnn.add(x, ttnn.multiply(self.linear(attention, self.o), self.config.residual_multiplier))
        n = self.norm(x, self.norm2)
        m = ttnn.multiply(ttnn.silu(self.linear(n, self.gate)), self.linear(n, self.up))
        return ttnn.add(x, ttnn.multiply(self.linear(m, self.down), self.config.residual_multiplier))

    def decode_forward(self, x, *, current_pos, page_table, kv_cache, cos, sin):
        """Device-only token pass; positions and page table are mutable trace inputs."""
        if x.shape[2] not in (1, 8, 16, 32):
            raise ValueError("Decode uses physical batches 1/8/16/32; mask inactive positions with -1")
        n = self.norm(x, self.norm1)
        fused = ttnn.to_memory_config(self.linear(n, self.qkv), ttnn.L1_MEMORY_CONFIG)
        q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
            fused, num_heads=32, num_kv_heads=8, memory_config=self.head_memory
        )
        q = ttnn.experimental.rotary_embedding_hf(q, cos, sin, is_decode_mode=True)
        k = ttnn.experimental.rotary_embedding_hf(k, cos, sin, is_decode_mode=True)
        ttnn.experimental.paged_update_cache(kv_cache[0], k, update_idxs_tensor=current_pos, page_table=page_table)
        ttnn.experimental.paged_update_cache(kv_cache[1], v, update_idxs_tensor=current_pos, page_table=page_table)
        a = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            q,
            *kv_cache,
            page_table_tensor=page_table,
            cur_pos_tensor=current_pos,
            scale=self.config.attention_multiplier,
            program_config=ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=self.decode_grid,
                q_chunk_size=32,
                k_chunk_size=512,
                exp_approx_mode=False,
            ),
            compute_kernel_config=self.compute,
        )
        a = ttnn.to_memory_config(a, q.memory_config())
        a = ttnn.experimental.nlp_concat_heads_decode(a, num_heads=32, sub_core_grids=self.worker_grid)
        a = ttnn.to_memory_config(a, ttnn.DRAM_MEMORY_CONFIG)
        a = a[:, :, : x.shape[2], :]
        return self.finish(x, a)

    def prefill_chunk(self, x, *, page_table, chunk_page_table, kv_cache, cos, sin, start_pos, slot=0):
        """Physical tile-aligned chunk; page table for attention contains one request."""
        n = self.norm(x, self.norm1)
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            self.linear(n, self.qkv),
            num_heads=32,
            num_kv_heads=8,
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        q = ttnn.experimental.rotary_embedding_hf(q, cos, sin)
        k = ttnn.experimental.rotary_embedding_hf(k, cos, sin)
        ttnn.experimental.paged_fill_cache(kv_cache[0], k, chunk_page_table, batch_idx=slot)
        ttnn.experimental.paged_fill_cache(kv_cache[1], v, chunk_page_table, batch_idx=slot)
        a = ttnn.transformer.chunked_scaled_dot_product_attention(
            q,
            *kv_cache,
            page_table,
            chunk_start_idx=None,
            chunk_start_idx_tensor=start_pos,
            scale=self.config.attention_multiplier,
            program_config=ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=(8, 8), q_chunk_size=32, k_chunk_size=512, exp_approx_mode=False
            ),
            compute_kernel_config=self.compute,
        )
        return self.finish(x, ttnn.experimental.nlp_concat_heads(a))

    def prepare_prefill(self, x, *, page_table, start_pos=0, slot=0):
        """SETUP boundary: host x [1,S,H] and INT32 page table -> device entries.

        Allocate plans before capture. To reuse their storage for changed requests,
        use refresh_prefill_entry outside forward. The caller owns a pool of plans
        sized for its request capacities, and keeps all inputs alive for traces.
        Nonaligned prefixes use token entries until the next 1024-token boundary.
        """
        length = x.shape[1]
        if start_pos < 0 or length < 1 or start_pos + length > self.config.max_position_embeddings:
            raise ValueError("Invalid context length")
        entries = []
        fringe = min(length, (-start_pos) % self.chunk_size)
        for offset in range(fringe):
            entries.append(
                self._prepare_entry(
                    x[:, offset : offset + 1], page_table[slot : slot + 1], start_pos + offset, token=True
                )
            )
        for offset in range(fringe, length, self.chunk_size):
            entries.append(
                self._prepare_entry(
                    x[:, offset : offset + self.chunk_size],
                    page_table[slot : slot + 1],
                    start_pos + offset,
                    token=False,
                )
            )
        return entries

    def _entry_host_values(self, x, table, start, *, token):
        import torch

        length = x.shape[1]
        physical = 1 if token else self.chunk_size
        if not 1 <= length <= physical or start < 0 or start + length > self.config.max_position_embeddings:
            raise ValueError("Invalid prepared entry")
        if not token and start % self.chunk_size:
            raise ValueError("Bulk entry starts must be 1024 aligned")
        padded = torch.nn.functional.pad(x, (0, 0, 0, physical - length)).unsqueeze(1).bfloat16()
        positions = torch.arange(start, start + physical, dtype=torch.float32)
        inv_freq = 1.0 / (
            self.config.rope_parameters["rope_theta"] ** (torch.arange(0, 128, 2, dtype=torch.float32) / 128)
        )
        angles = positions[:, None] * inv_freq[None, :]
        angles = torch.cat((angles, angles), dim=-1)
        cos, sin = (
            angles.cos().reshape(1, 1, physical, 128).bfloat16(),
            angles.sin().reshape(1, 1, physical, 128).bfloat16(),
        )
        if token:
            cos = torch.nn.functional.pad(cos, (0, 0, 0, 31))
            sin = torch.nn.functional.pad(sin, (0, 0, 0, 31))
        values = {
            "x": padded,
            "cos": cos,
            "sin": sin,
            "position": torch.tensor([start], dtype=torch.int32),
            "page_table": table.to(torch.int32),
        }
        if not token:
            pages = table[:, start // 32 : (start + self.chunk_size) // 32]
            if pages.shape[1] != self.chunk_size // 32:
                raise ValueError("Page table capacity must cover the physical final chunk")
            values["write_page_table"] = pages.to(torch.int32)
        return values

    def _prepare_entry(self, x, table, start, *, token):
        values = self._entry_host_values(x, table, start, token=token)
        device_values = {}
        for name, value in values.items():
            integer = name in ("position", "page_table", "write_page_table")
            device_values[name] = ttnn.from_torch(
                value.contiguous(),
                dtype=ttnn.int32 if integer else ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT if integer else ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=self.single_rope_memory if token and name in ("cos", "sin") else ttnn.DRAM_MEMORY_CONFIG,
            )
        return PrefillEntry(**device_values, **({"write_page_table": None} if token else {}), valid_tokens=x.shape[1])

    def refresh_prefill_entry(self, entry, x, *, page_table, start_pos, slot=0):
        """SETUP boundary: refresh stable storage; logical length/offset may change."""
        values = self._entry_host_values(
            x, page_table[slot : slot + 1], start_pos, token=entry.write_page_table is None
        )
        for name, value in values.items():
            integer = name in ("position", "page_table", "write_page_table")
            host = ttnn.from_torch(
                value.contiguous(),
                dtype=ttnn.int32 if integer else ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT if integer else ttnn.TILE_LAYOUT,
            )
            ttnn.copy_host_to_device_tensor(host, getattr(entry, name))
        return replace(entry, valid_tokens=x.shape[1])

    def prefill_forward(self, entries, *, kv_cache):
        """Device-only full logical prefill; returns one logical TT tensor per entry.

        Concatenating these tensors in list order gives [1,1,S,4096]. Output
        logical shapes exclude padding; physical storage remains 1024 per bulk
        chunk. The caller may consume chunks without a large concatenate kernel.
        Every entry's position, RoPE and page mappings are device inputs. Prepare
        bulk and token paths before the first trace; refresh existing buffers at
        request boundaries. No per-request scalar offsets enter kernel keys.
        """
        outputs = []
        for entry in entries:
            if entry.write_page_table is None:
                y = self.decode_forward(
                    entry.x,
                    current_pos=entry.position,
                    page_table=entry.page_table,
                    kv_cache=kv_cache,
                    cos=entry.cos,
                    sin=entry.sin,
                )
            else:
                y = self.prefill_chunk(
                    entry.x,
                    page_table=entry.page_table,
                    chunk_page_table=entry.write_page_table,
                    kv_cache=kv_cache,
                    cos=entry.cos,
                    sin=entry.sin,
                    start_pos=entry.position,
                )
            outputs.append(ttnn.reshape(y, [1, 1, entry.valid_tokens, 4096], list(y.padded_shape)))
        return outputs
