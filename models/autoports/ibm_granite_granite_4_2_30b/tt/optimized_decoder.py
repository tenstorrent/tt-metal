# SPDX-License-Identifier: Apache-2.0
"""Optimized single-chip Granite decoder. Setup owns weights; callers own paged KV storage.

Prefill consumes [1,1,S,4096] per request, with explicit absolute start and slot.
Decode consumes [1,1,B,4096], device positions [B], page table [B,pages],
and device HF cos/sin [1,B,32,128] (first row used), in height-sharded L1.
The decode method is entirely device-side and may be captured by TTNN.
"""

from dataclasses import dataclass, replace

import ttnn
from models.common.lightweightmodule import LightweightModule


def default_policy():
    """Measured p300c policy: BFP4 projections except BFP8 down; BF16 activations, BFP8 KV."""
    policy = dict(
        dram=True,
        cores=64,
        cache_dtype="bfloat8_b",
        directqkv=True,
        sdpa_chunk=0,
        prefill_program="2d",
        prefill_block=16,
        prefill_buckets=True,
        prefill_qchunk=256,
        prefill_kchunk=512,
        sdpa_fidelity="LoFi",
        sdpa_fp32=False,
    )
    for role in ("qkv", "o", "gate", "up", "down"):
        policy.update(
            {
                role + "_dtype": "bfloat4_b",
                role + "_fidelity": "LoFi",
                role + "_fp32": False,
                role + "_cores": 64,
                role + "_readers": 2,
                role + "_block": 2 if role in ("gate", "up") else 8,
            }
        )
    policy["down_dtype"] = "bfloat8_b"
    policy["down_block"] = 4
    return policy


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


class OptimizedDecoder(LightweightModule):
    @classmethod
    def from_state_dict(cls, state_dict, *, hf_config, layer_idx, mesh_device, chunk_size=1024, policy=None):
        import torch

        c = hf_config
        if (c.hidden_size, c.intermediate_size, c.num_attention_heads, c.num_key_value_heads) != (4096, 32768, 32, 8):
            raise ValueError("Requires the pinned Granite 4.2 30B config")
        if c.residual_multiplier != 1:
            raise ValueError("Pinned Granite residual multiplier must be one")
        if c.attention_bias or c.mlp_bias or c.hidden_act != "silu":
            raise ValueError("Unsupported bias or activation")
        if chunk_size != 1024:
            raise ValueError("The prepared prefill physical chunk is 1024 tokens")
        policy = dict(policy) if policy else default_policy()
        obj = cls()
        obj.policy = policy
        obj.device = mesh_device
        obj.config = c
        obj.chunk_size = chunk_size
        obj.compute = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
        )
        obj.attention_compute = ttnn.WormholeComputeKernelConfig(
            math_fidelity=getattr(ttnn.MathFidelity, policy.get("sdpa_fidelity", "HiFi4")),
            math_approx_mode=False,
            fp32_dest_acc_en=policy.get("sdpa_fp32", True),
            packer_l1_acc=True,
        )
        prefix = f"model.layers.{layer_idx}."

        def get(key):
            return state_dict[prefix + key] if prefix + key in state_dict else state_dict[key]

        obj.decode_weights = {}

        def upload(w, role="other"):
            dtype = getattr(ttnn, policy.get(role + "_dtype", "bfloat16"))
            if role != "other" and policy.get("dram", False):
                dg = mesh_device.dram_grid_size()
                readers = policy.get(role + "_readers", policy.get("readers", 1))
                width = ((w.shape[-1] + 32 * dg.x * readers - 1) // (32 * dg.x * readers)) * 32 * readers
                dm = ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                    ttnn.BufferType.DRAM,
                    ttnn.ShardSpec(
                        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dg.x - 1, dg.y - 1))}),
                        [w.shape[-2], width],
                        ttnn.ShardOrientation.ROW_MAJOR,
                    ),
                )
                obj.decode_weights[role] = ttnn.from_torch(
                    w.contiguous(), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=mesh_device, memory_config=dm
                )
            if policy.get("residual_fusion", False) and role in ("o", "down"):
                dtype = ttnn.bfloat16
            return ttnn.from_torch(
                w.contiguous(),
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        # Fold each norm gamma into its consuming projection at setup. BF16
        # rounding differs slightly; real-weight equivalence is measured.
        gamma1 = get("input_layernorm.weight").float()
        gamma2 = get("post_attention_layernorm.weight").float()
        obj.qkv = upload(
            (torch.cat([get(f"self_attn.{k}_proj.weight") for k in ("q", "k", "v")]).float() * gamma1).T.bfloat16(),
            "qkv",
        )
        if policy.get("splitqkv", False):
            for role in ("q", "k", "v"):
                setattr(obj, role, upload((get(f"self_attn.{role}_proj.weight").float() * gamma1).T.bfloat16(), role))
        obj.o = upload(get("self_attn.o_proj.weight").T, "o")
        obj.gate = upload((get("mlp.gate_proj.weight").float() * gamma2).T.bfloat16(), "gate")
        obj.up = upload((get("mlp.up_proj.weight").float() * gamma2).T.bfloat16(), "up")
        obj.down = upload(get("mlp.down_proj.weight").T, "down")
        if policy.get("packed", False):
            obj.gate_up = upload(
                (torch.cat([get("mlp.gate_proj.weight"), get("mlp.up_proj.weight")]).float() * gamma2).T.bfloat16(),
                "gate_up",
            )
        obj.prefill_ones = (
            upload(torch.ones(1, 1, chunk_size, 4096, dtype=torch.bfloat16))
            if policy.get("residual_fusion", False)
            else None
        )
        grid = mesh_device.compute_with_storage_grid_size()
        obj.decode_grid = policy.get("sdpa_grid", (grid.x, 8))
        obj.worker_grid = ttnn.num_cores_to_corerangeset(grid.x * grid.y, grid, row_wise=True)
        obj.single_rope_memory = ttnn.create_sharded_memory_config(
            (32, 128), ttnn.CoreGrid(y=1, x=1), ttnn.ShardStrategy.HEIGHT, use_height_and_width_as_shard_shape=True
        )
        qkv_cores = ttnn.num_cores_to_corerangeset(96, grid, row_wise=True)
        obj.qkv_shards = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(qkv_cores, [32, 64], ttnn.ShardOrientation.ROW_MAJOR),
        )
        obj.split_head_memory = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(obj.worker_grid, [32, 128], ttnn.ShardOrientation.ROW_MAJOR),
        )
        obj.kernels = {}
        for role in (
            ("qkv", "o", "gate", "up", "down", "gate_up")
            if policy.get("packed", False)
            else ("qkv", "o", "gate", "up", "down")
        ):
            obj.kernels[role] = ttnn.WormholeComputeKernelConfig(
                math_fidelity=getattr(ttnn.MathFidelity, policy.get(role + "_fidelity", "HiFi4")),
                math_approx_mode=False,
                fp32_dest_acc_en=policy.get(role + "_fp32", True),
                packer_l1_acc=True,
            )
        if policy.get("splitqkv", False):
            for role in ("q", "k", "v"):
                obj.kernels[role] = obj.kernels["qkv"]
        return obj

    def role(self, w):
        return next(r for r in self.kernels if getattr(self, r) is w)

    def shard(self, width, cores=None):
        cores = cores or self.policy.get("cores", 32)
        grid = self.device.compute_with_storage_grid_size()
        return ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.num_cores_to_corerangeset(cores, grid, row_wise=True),
                [32, ((width + cores * 32 - 1) // (cores * 32)) * 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )

    def kernel(self, w):
        return next(self.kernels[r] for r in self.kernels if getattr(self, r) is w)

    def linear(self, x, w):
        if x.shape[2] <= 32 and self.policy.get("dram", False):
            role = self.role(w)
            cores = self.policy.get(role + "_cores", self.policy.get("cores", 32))
            x = ttnn.to_memory_config(x, self.shard(x.shape[-1], cores))
            output_mem = (
                self.qkv_shards if role == "qkv" and self.policy.get("directqkv", False) else self.shard(w.shape[-1])
            )
            if (
                self.policy.get("attention_act8", False)
                and role in ("qkv", "o", "q", "k", "v")
                or self.policy.get("mlp_act8", False)
                and role in ("gate", "up", "down", "gate_up")
            ):
                x = ttnn.typecast(x, ttnn.bfloat8_b)
            ktiles = x.shape[-1] // 32 // cores
            block = self.policy.get(role + "_block", self.policy.get("block", ktiles))
            return ttnn.linear(
                x,
                self.decode_weights[role],
                program_config=ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                    in0_block_w=block,
                    per_core_M=1,
                    per_core_N=output_mem.shard_spec.shape[1] // 32,
                    num_workers_per_dram_bank=self.policy.get(role + "_readers", self.policy.get("readers", 1)),
                    fused_activation=(
                        ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU)
                        if role == "gate" and self.policy.get("gate_activation", False)
                        else None
                    ),
                ),
                compute_kernel_config=self.kernel(w),
                dtype=getattr(ttnn, self.policy.get("activation_dtype", "bfloat16")),
                memory_config=output_mem,
            )
        if x.shape[2] > 32 and self.policy.get("prefill_program"):
            import math

            mode = self.policy["prefill_program"]
            role = self.role(w)
            kernel = self.kernel(w)
            if self.policy.get("prefill_hifi", False):
                kernel = ttnn.WormholeComputeKernelConfig(
                    math_fidelity=ttnn.MathFidelity.HiFi2,
                    math_approx_mode=False,
                    fp32_dest_acc_en=False,
                    packer_l1_acc=True,
                )
            if self.policy.get("prefill_l1", False):
                x = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)
            if mode == "minimal":
                from models.tt_dit.utils.matmul import get_matmul_config

                cfg = get_matmul_config(x.shape[2], x.shape[3], w.shape[-1], ttnn.CoreCoord(11, 10), use_heuristic=True)
                return ttnn.experimental.minimal_matmul(x, w, config=cfg, compute_kernel_config=kernel)
            gx, gy = self.policy.get("prefill_grid", (8, 8))
            pm, pn = math.ceil(x.shape[2] / 32 / gy), math.ceil(w.shape[-1] / 32 / gx)
            bw = self.policy.get("prefill_block", 4)
            sub = self.policy.get("prefill_subblock", 4)
            while pn % sub:
                sub -= 1
            cfg = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=(gx, gy),
                in0_block_w=bw,
                per_core_M=pm,
                per_core_N=pn,
                out_subblock_h=1,
                out_subblock_w=sub,
                out_block_h=pm,
                out_block_w=next(
                    v
                    for v in range(min(self.policy.get("prefill_outblock", 32), pn), 0, -1)
                    if pn % v == 0 and v % sub == 0
                ),
                transpose_mcast=False,
                fuse_batch=False,
                fused_activation=(
                    ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU)
                    if role == "gate" and self.policy.get("prefill_gate_activation", False)
                    else None
                ),
            )
            return ttnn.linear(
                x,
                w,
                program_config=cfg,
                compute_kernel_config=kernel,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                dtype=getattr(ttnn, self.policy.get("activation_dtype", "bfloat16")),
            )
        return ttnn.linear(
            x,
            w,
            compute_kernel_config=self.kernel(w),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=getattr(ttnn, self.policy.get("activation_dtype", "bfloat16")),
        )

    def norm(self, x):
        norm_dtype = getattr(ttnn, self.policy.get("norm_dtype", "bfloat16"))
        if x.dtype != norm_dtype:
            x = ttnn.typecast(x, norm_dtype)
        if x.shape[2] <= 32 and self.policy.get("dram", False):
            x = ttnn.to_memory_config(x, self.shard(4096))
            block = 128 // self.policy.get("cores", 32)
            grid = self.device.compute_with_storage_grid_size()
            return ttnn.rms_norm(
                x,
                epsilon=self.config.rms_norm_eps,
                compute_kernel_config=self.compute,
                memory_config=x.memory_config(),
                program_config=ttnn.LayerNormShardedMultiCoreProgramConfig(
                    compute_with_storage_grid_size=[grid.x, grid.y],
                    subblock_w=min(4, block),
                    block_h=1,
                    block_w=block,
                    inplace=False,
                ),
            )
        return ttnn.rms_norm(x, epsilon=self.config.rms_norm_eps, compute_kernel_config=self.compute)

    def finish(self, x, attention):
        if x.shape[2] <= 32 and self.policy.get("dram", False):
            x = ttnn.to_memory_config(x, self.shard(4096))
        bulk = x.shape[2] == self.chunk_size and self.policy.get("residual_fusion", False)
        if bulk:
            x = ttnn.experimental.dit_minimal_matmul_addcmul_fused(
                attention, self.o, 1.0, x, self.prefill_ones, compute_kernel_config=self.kernel(self.o)
            )
        else:
            x = ttnn.add(x, self.linear(attention, self.o))
        n = self.norm(x)
        if x.shape[2] <= 32 and self.policy.get("dram", False):
            n = ttnn.to_memory_config(n, self.shard(4096, self.policy.get("gate_cores", self.policy.get("cores", 32))))
        if self.policy.get("packed", False):
            both = self.linear(n, self.gate_up)
            gate = ttnn.slice(both, [0, 0, 0, 0], [1, 1, n.shape[2], 32768])
            up = ttnn.slice(both, [0, 0, 0, 32768], [1, 1, n.shape[2], 65536])
        else:
            gate, up = self.linear(n, self.gate), self.linear(n, self.up)
        m = ttnn.multiply(
            gate,
            up,
            input_tensor_a_activations=(
                []
                if (x.shape[2] <= 32 and self.policy.get("gate_activation", False))
                or (x.shape[2] > 32 and self.policy.get("prefill_gate_activation", False))
                else [ttnn.UnaryOpType.SILU]
            ),
        )
        if bulk:
            return ttnn.experimental.dit_minimal_matmul_addcmul_fused(
                m, self.down, 1.0, x, self.prefill_ones, compute_kernel_config=self.kernel(self.down)
            )
        return ttnn.add(x, self.linear(m, self.down))

    def decode_forward(self, x, *, current_pos, page_table, kv_cache, cos, sin):
        """Device-only token pass; positions and page table are mutable trace inputs."""
        if x.shape[2] not in (1, 8, 16, 32):
            raise ValueError("Decode uses physical batches 1/8/16/32; mask inactive positions with -1")
        if self.policy.get("dram", False):
            x = ttnn.to_memory_config(x, self.shard(4096))
        n = self.norm(x)
        fused = (
            None
            if self.policy.get("splitqkv", False)
            else (
                self.linear(n, self.qkv)
                if self.policy.get("dram", False)
                else ttnn.linear(
                    n,
                    self.qkv,
                    compute_kernel_config=self.kernel(self.qkv),
                    memory_config=self.qkv_shards,
                    dtype=ttnn.bfloat16,
                )
            )
        )
        if self.policy.get("splitqkv", False):
            parts = [
                ttnn.to_memory_config(self.linear(n, getattr(self, r)), ttnn.L1_MEMORY_CONFIG) for r in ("q", "k", "v")
            ]
            fused = ttnn.concat(parts, dim=3, memory_config=ttnn.L1_MEMORY_CONFIG)
        fused = ttnn.to_memory_config(fused, self.qkv_shards)
        q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
            fused, num_heads=32, num_kv_heads=8, memory_config=self.split_head_memory, overlap_qk_coregrid=False
        )
        q = ttnn.experimental.rotary_embedding_hf(q, cos, sin, is_decode_mode=True)
        k = ttnn.experimental.rotary_embedding_hf(
            k,
            ttnn.to_memory_config(cos, k.memory_config()),
            ttnn.to_memory_config(sin, k.memory_config()),
            is_decode_mode=True,
        )
        # Q/V and K are emitted on disjoint grids, as fused cache update requires.
        ttnn.experimental.paged_fused_update_cache(
            kv_cache[0], k, kv_cache[1], v, update_idxs_tensor=current_pos, page_table=page_table
        )
        a = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            q,
            *kv_cache,
            page_table_tensor=page_table,
            cur_pos_tensor=current_pos,
            scale=self.config.attention_multiplier,
            program_config=ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=self.decode_grid,
                q_chunk_size=32,
                k_chunk_size=self.policy.get("sdpa_chunk", 512),
                exp_approx_mode=False,
            ),
            compute_kernel_config=self.attention_compute,
        )
        a = ttnn.to_memory_config(a, q.memory_config())
        a = ttnn.experimental.nlp_concat_heads_decode(a, num_heads=32, sub_core_grids=self.worker_grid)
        a = ttnn.reshape(a, [1, 1, x.shape[2], 4096], list(a.padded_shape))
        return self.finish(x, a)

    def prefill_chunk(self, x, *, page_table, chunk_page_table, kv_cache, cos, sin, start_pos, slot=0):
        """Physical tile-aligned chunk; page table for attention contains one request."""
        n = self.norm(x)
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            self.linear(n, self.qkv),
            num_heads=32,
            num_kv_heads=8,
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        q = ttnn.experimental.rotary_embedding_hf(q, cos, sin)
        k = ttnn.experimental.rotary_embedding_hf(k, cos, sin)
        ttnn.experimental.paged_fill_cache(
            kv_cache[0], ttnn.typecast(k, kv_cache[0].dtype), chunk_page_table, batch_idx=slot
        )
        ttnn.experimental.paged_fill_cache(
            kv_cache[1], ttnn.typecast(v, kv_cache[1].dtype), chunk_page_table, batch_idx=slot
        )
        a = ttnn.transformer.chunked_scaled_dot_product_attention(
            q,
            *kv_cache,
            page_table,
            chunk_start_idx=None,
            chunk_start_idx_tensor=start_pos,
            scale=self.config.attention_multiplier,
            program_config=ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=self.policy.get("prefill_sdpa_grid", (8, 8)),
                q_chunk_size=min(x.shape[2], self.policy.get("prefill_qchunk", 32)),
                k_chunk_size=min(x.shape[2], self.policy.get("prefill_kchunk", 512)),
                exp_approx_mode=False,
            ),
            compute_kernel_config=self.attention_compute,
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

    def _entry_host_values(self, x, table, start, *, token, physical=None, logical_length=None):
        import torch

        length = x.shape[1] if x is not None else logical_length
        if physical is None:
            physical = (
                1
                if token
                else (
                    next(b for b in (128, 256, 512, 1024) if b >= length)
                    if self.policy.get("prefill_buckets", False)
                    else self.chunk_size
                )
            )
        if not 1 <= length <= physical or start < 0 or start + length > self.config.max_position_embeddings:
            raise ValueError("Invalid prepared entry")
        if not token and start % physical:
            raise ValueError("Bulk entry start must align to its prepared physical bucket")
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
        values = (
            {} if x is None else {"x": torch.nn.functional.pad(x, (0, 0, 0, physical - length)).unsqueeze(1).bfloat16()}
        )
        values.update(
            {
                "cos": cos,
                "sin": sin,
                "position": torch.tensor([start], dtype=torch.int32),
                "page_table": table.to(torch.int32),
            }
        )
        if not token:
            pages = table[:, start // 32 : (start + physical) // 32]
            if pages.shape[1] != physical // 32:
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
            x, page_table[slot : slot + 1], start_pos, token=entry.write_page_table is None, physical=entry.x.shape[2]
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
        logical shapes exclude padding; bulk storage uses the prepared physical
        bucket (128, 256, 512, or 1024 tokens). The caller may consume chunks without a large concatenate kernel.
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
