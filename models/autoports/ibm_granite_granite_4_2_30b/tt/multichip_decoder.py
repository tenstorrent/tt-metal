# SPDX-License-Identifier: Apache-2.0
"""Granite TP4 decoder on a 1x4 Blackhole ring.

OptimizedDecoder is the immutable single-chip baseline and supplies tuned local
matmul and prepared logical-length handling. Each rank owns eight Q and two KV
heads and 8192 MLP channels. Page tables/positions are replicated; caches are
local-head shards. Inputs/outputs share the selected residual contract.
"""

from dataclasses import dataclass, replace

import ttnn
from models.common.modules.tt_ccl import TT_CCL

from .optimized_decoder import OptimizedDecoder, PrefillEntry, default_policy


def multichip_policy():
    p = default_policy()
    p.update(residual="replicated", ccl_dtype="bfloat16")
    p.update(
        collective="sharded_ar",
        ar_links=2,
        qkv_cores=64,
        qkv_block=32,
        o_cores=8,
        o_block=8,
        gate_block=4,
        up_block=4,
        down_cores=64,
        down_readers=2,
        down_block=4,
    )
    p.update(
        persistent=True,
        persistent_contiguous=True,
        prefill_links=2,
        prefill_grid=(11, 8),
        prefill_qchunk=128,
        prefill_subblock=8,
    )
    return p


@dataclass(frozen=True)
class CollectiveResources:
    """Setup-owned pool for a sequential layer stack on one command queue.

    Share this object across layers; keep it alive until every trace is released.
    Each sublayer consumes its collective output before the next reuses a slot.
    Concurrent forwards on the same pool are not supported.
    """

    device: object
    policy_key: tuple
    ccl: object
    rs_buffers: object
    ag_buffers: object
    ar_buffers: object = None


class MultichipDecoder(OptimizedDecoder):
    @classmethod
    def from_state_dict(
        cls, state_dict, *, hf_config, layer_idx, mesh_device, chunk_size=1024, policy=None, collective_resources=None
    ):
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
        if mesh_device.get_num_devices() != 4 or list(mesh_device.shape) != [1, 4]:
            raise ValueError("Requires the four-device 1x4 ring")
        policy = dict(policy) if policy else multichip_policy()
        obj = cls()
        obj.policy = policy
        obj.device = mesh_device
        obj.config = c
        resource_key = (
            policy.get("persistent", False),
            policy.get("persistent_contiguous", False),
            policy["ccl_dtype"],
            policy.get("collective", "rs_ag"),
            policy.get("cores", 64),
            policy.get("residual", "replicated"),
        )
        if collective_resources is not None:
            if collective_resources.device is not mesh_device or collective_resources.policy_key != resource_key:
                raise ValueError("Collective pool must match this mesh and CCL storage policy")
        obj.ccl = collective_resources.ccl if collective_resources is not None else TT_CCL(mesh_device)
        obj.residual_sharded = policy.get("residual", "replicated") != "replicated"
        obj.chunk_size = chunk_size
        obj.compute = ttnn.WormholeComputeKernelConfig(
            math_fidelity=getattr(ttnn.MathFidelity, policy.get("norm_fidelity", "HiFi4")),
            math_approx_mode=False,
            fp32_dest_acc_en=policy.get("norm_fp32", True),
            packer_l1_acc=True,
        )
        obj.attention_compute = ttnn.WormholeComputeKernelConfig(
            math_fidelity=getattr(ttnn.MathFidelity, policy.get("sdpa_fidelity", "HiFi4")),
            math_approx_mode=False,
            fp32_dest_acc_en=policy.get("sdpa_fp32", True),
            packer_l1_acc=True,
        )
        obj.prefill_attention_compute = ttnn.WormholeComputeKernelConfig(
            math_fidelity=getattr(
                ttnn.MathFidelity, policy.get("sdpa_prefill_fidelity", policy.get("sdpa_fidelity", "HiFi4"))
            ),
            math_approx_mode=False,
            fp32_dest_acc_en=policy.get("sdpa_prefill_fp32", policy.get("sdpa_fp32", True)),
            packer_l1_acc=True,
        )
        prefix = f"model.layers.{layer_idx}."

        def get(key):
            return state_dict[prefix + key] if prefix + key in state_dict else state_dict[key]

        obj.decode_weights = {}

        def upload(w, role="other"):
            dtype = getattr(ttnn, policy.get(role + "_dtype", "bfloat16"))
            axis = 0 if role in ("o", "down") else 1
            mapper = ttnn.ShardTensorToMesh(mesh_device, dim=axis)
            local_k, local_n = w.shape
            if axis == 0:
                local_k //= 4
            else:
                local_n //= 4
            if role != "other" and policy.get("dram", False):
                dg = mesh_device.dram_grid_size()
                readers = policy.get(role + "_readers", policy.get("readers", 1))
                width = ((local_n + 32 * dg.x * readers - 1) // (32 * dg.x * readers)) * 32 * readers
                dm = ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                    ttnn.BufferType.DRAM,
                    ttnn.ShardSpec(
                        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dg.x - 1, dg.y - 1))}),
                        [local_k, width],
                        ttnn.ShardOrientation.ROW_MAJOR,
                    ),
                )
                obj.decode_weights[role] = ttnn.from_torch(
                    w.contiguous(),
                    dtype=dtype,
                    layout=ttnn.TILE_LAYOUT,
                    device=mesh_device,
                    memory_config=dm,
                    mesh_mapper=mapper,
                )
            return ttnn.from_torch(
                w.contiguous(),
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=mapper,
            )

        # Fold each norm gamma into its consuming projection at setup. BF16
        # rounding differs slightly; real-weight equivalence is measured.
        gamma1 = get("input_layernorm.weight").float()
        gamma2 = get("post_attention_layernorm.weight").float()
        qkv = torch.cat(
            [
                torch.cat([get(f"self_attn.{k}_proj.weight").chunk(4, dim=0)[rank] for k in ("q", "k", "v")])
                for rank in range(4)
            ]
        )
        obj.qkv = upload(
            (qkv.float() * gamma1).T.bfloat16(),
            "qkv",
        )
        obj.o = upload(get("self_attn.o_proj.weight").T, "o")
        obj.gate = upload((get("mlp.gate_proj.weight").float() * gamma2).T.bfloat16(), "gate")
        obj.up = upload((get("mlp.up_proj.weight").float() * gamma2).T.bfloat16(), "up")
        obj.down = upload(get("mlp.down_proj.weight").T, "down")
        grid = mesh_device.compute_with_storage_grid_size()
        obj.decode_grid = policy.get("sdpa_grid", (grid.x, 8))
        obj.worker_grid = ttnn.num_cores_to_corerangeset(grid.x * grid.y, grid, row_wise=True)
        obj.single_rope_memory = ttnn.create_sharded_memory_config(
            (32, 128), ttnn.CoreGrid(y=1, x=1), ttnn.ShardStrategy.HEIGHT, use_height_and_width_as_shard_shape=True
        )
        qkv_cores = ttnn.num_cores_to_corerangeset(48, grid, row_wise=True)
        obj.qkv_shards = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(qkv_cores, [32, 32], ttnn.ShardOrientation.ROW_MAJOR),
        )
        obj.split_head_memory = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(obj.worker_grid, [32, 128], ttnn.ShardOrientation.ROW_MAJOR),
        )
        obj.kernels = {}
        for role in ("qkv", "o", "gate", "up", "down"):
            obj.kernels[role] = ttnn.WormholeComputeKernelConfig(
                math_fidelity=getattr(ttnn.MathFidelity, policy.get(role + "_fidelity", "HiFi4")),
                math_approx_mode=False,
                fp32_dest_acc_en=policy.get(role + "_fp32", True),
                packer_l1_acc=True,
            )
        if policy.get("persistent", False):
            if collective_resources is None:
                obj._prepare_ccl_buffers()
            else:
                obj.rs_buffers = collective_resources.rs_buffers
                obj.ag_buffers = collective_resources.ag_buffers
                obj.ar_buffers = collective_resources.ar_buffers
                obj.ar_index = 0
                obj.rs_index = obj.ag_index = 0
        obj.collective_resources = CollectiveResources(
            mesh_device,
            resource_key,
            obj.ccl,
            getattr(obj, "rs_buffers", None),
            getattr(obj, "ag_buffers", None),
            getattr(obj, "ar_buffers", None),
        )
        return obj

    def gather(self, x, dim=3):
        if (
            self.policy.get("persistent", False)
            and x.shape[-2] <= 32
            and x.shape[-1] == 1024
            and not self.residual_sharded
        ):
            return self._persistent_gather(x, dim)
        mem = ttnn.L1_MEMORY_CONFIG if x.shape[-2] <= 32 else ttnn.DRAM_MEMORY_CONFIG
        x = ttnn.to_memory_config(x, mem)
        return ttnn.experimental.all_gather_async(
            x,
            dim=dim,
            persistent_output_buffer=None,
            multi_device_global_semaphore=self.ccl.get_and_cycle_ag_semaphore_handles(),
            barrier_semaphore=self.ccl.get_and_cycle_barrier_semaphore_handle(),
            num_links=self.policy.get("prefill_links", 2) if x.shape[-2] > 32 else self.policy.get("num_links", 1),
            topology=ttnn.Topology.Ring,
            memory_config=mem,
            chunks_per_sync=self.policy.get("ccl_chunks", 10),
            num_workers_per_link=self.policy.get("ccl_workers", 2) if x.shape[-2] <= 32 else 2,
            num_buffers_per_channel=2,
        )

    def reduce(self, x):
        if self.policy.get("collective") == "sharded_ar" and not self.residual_sharded and x.shape[-2] <= 32:
            return self._sharded_all_reduce(x)
        if self.policy.get("persistent", False) and x.shape[-2] <= 32:
            return self._persistent_reduce(x)
        mem = ttnn.L1_MEMORY_CONFIG if x.shape[-2] <= 32 else ttnn.DRAM_MEMORY_CONFIG
        x = ttnn.to_memory_config(x, mem)
        dtype = getattr(ttnn, self.policy.get("ccl_dtype", "bfloat16"))
        if x.dtype != dtype:
            x = ttnn.typecast(x, dtype)
        y = ttnn.experimental.reduce_scatter_minimal_async(
            x,
            dim=3,
            persistent_output_buffers=None,
            multi_device_global_semaphore=self.ccl.get_and_cycle_rs_semaphore_handles(),
            barrier_semaphore=self.ccl.get_and_cycle_barrier_semaphore_handle(),
            num_links=self.policy.get("prefill_links", 2) if x.shape[-2] > 32 else self.policy.get("num_links", 1),
            topology=ttnn.Topology.Ring,
            memory_config=mem,
            intermediate_memory_config=mem,
            chunks_per_sync=self.policy.get("ccl_chunks", 10),
            num_workers_per_link=self.policy.get("ccl_workers", 2) if x.shape[-2] <= 32 else 2,
            num_buffers_per_channel=2,
        )
        if not self.residual_sharded:
            y = self.gather(y)
        return ttnn.typecast(y, ttnn.bfloat16) if y.dtype != ttnn.bfloat16 else y

    def projection_input(self, x):
        if not self.residual_sharded:
            return self.norm(x)
        if self.policy["residual"] == "gather_norm":
            return self.norm(self.gather(x))
        mem = ttnn.L1_MEMORY_CONFIG if x.shape[-2] <= 32 else ttnn.DRAM_MEMORY_CONFIG
        x = ttnn.to_memory_config(x, mem)
        stats = ttnn.rms_norm_pre_all_gather(x, compute_kernel_config=self.compute, dtype=ttnn.bfloat16)
        stats = self.gather(stats)
        n = ttnn.rms_norm_post_all_gather(
            x, stats, epsilon=self.config.rms_norm_eps, compute_kernel_config=self.compute
        )
        return self.gather(n)

    def finish(self, x, attention):
        residual_dtype = getattr(ttnn, self.policy.get("residual_dtype", "bfloat16"))
        if x.dtype != residual_dtype:
            x = ttnn.typecast(x, residual_dtype)
        mem = self.shard(1024, 32) if self.residual_sharded else self.shard(4096)
        if x.shape[2] > 32:
            mem = ttnn.DRAM_MEMORY_CONFIG
        x = ttnn.add(
            ttnn.to_memory_config(x, mem), ttnn.to_memory_config(self.reduce(self.linear(attention, self.o)), mem)
        )
        n = self.projection_input(x)
        gate, up = self.linear(n, self.gate), self.linear(n, self.up)
        m = ttnn.multiply(
            gate,
            up,
            input_tensor_a_activations=(
                []
                if self.policy.get("gate_activation" if x.shape[2] <= 32 else "prefill_gate_activation", False)
                else [ttnn.UnaryOpType.SILU]
            ),
        )
        return ttnn.add(x, ttnn.to_memory_config(self.reduce(self.linear(m, self.down)), mem))

    def _prepare_entry(self, x, table, start, *, token):
        values = self._entry_host_values(x, table, start, token=token)
        device_values = {}
        for name, value in values.items():
            integer = name in ("position", "page_table", "write_page_table")
            mapper = (
                ttnn.ShardTensorToMesh(self.device, dim=3)
                if name == "x" and self.residual_sharded
                else ttnn.ReplicateTensorToMesh(self.device)
            )
            device_values[name] = ttnn.from_torch(
                value.contiguous(),
                dtype=ttnn.int32 if integer else ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT if integer else ttnn.TILE_LAYOUT,
                device=self.device,
                mesh_mapper=mapper,
                memory_config=self.single_rope_memory if token and name in ("cos", "sin") else ttnn.DRAM_MEMORY_CONFIG,
            )
        return PrefillEntry(**device_values, **({"write_page_table": None} if token else {}), valid_tokens=x.shape[1])

    def refresh_prefill_entry(self, entry, x=None, *, page_table, start_pos, slot=0, logical_length=None):
        # Full-model embedding supplies hidden states on device. Its caller only
        # refreshes metadata; decoder-only callers still supply and copy x.
        values = self._entry_host_values(
            x,
            page_table[slot : slot + 1],
            start_pos,
            token=entry.write_page_table is None,
            physical=entry.x.shape[2],
            logical_length=logical_length,
        )
        for name, value in values.items():
            integer = name in ("position", "page_table", "write_page_table")
            mapper = (
                ttnn.ShardTensorToMesh(self.device, dim=3)
                if name == "x" and self.residual_sharded
                else ttnn.ReplicateTensorToMesh(self.device)
            )
            host = ttnn.from_torch(
                value.contiguous(),
                dtype=ttnn.int32 if integer else ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT if integer else ttnn.TILE_LAYOUT,
                mesh_mapper=mapper,
            )
            ttnn.copy_host_to_device_tensor(host, getattr(entry, name))
        return replace(entry, valid_tokens=x.shape[1] if x is not None else logical_length)

    def prefill_forward(self, entries, *, kv_cache):
        outputs = []
        for e in entries:
            if e.write_page_table is None:
                y = self.decode_forward(
                    e.x, current_pos=e.position, page_table=e.page_table, kv_cache=kv_cache, cos=e.cos, sin=e.sin
                )
            else:
                y = self.prefill_chunk(
                    e.x,
                    page_table=e.page_table,
                    chunk_page_table=e.write_page_table,
                    kv_cache=kv_cache,
                    cos=e.cos,
                    sin=e.sin,
                    start_pos=e.position,
                )
            width = 1024 if self.residual_sharded else 4096
            outputs.append(ttnn.reshape(y, [1, 1, e.valid_tokens, width], list(y.padded_shape)))
        return outputs

    def decode_forward(self, x, *, current_pos, page_table, kv_cache, cos, sin):
        """Device-only token pass; positions and page table are mutable trace inputs."""
        if x.shape[2] not in (1, 8, 16, 32):
            raise ValueError("Decode uses physical batches 1/8/16/32; mask inactive positions with -1")
        if self.policy.get("dram", False):
            x = ttnn.to_memory_config(x, self.shard(1024, 32) if self.residual_sharded else self.shard(4096))
        n = self.projection_input(x)
        fused = self.linear(n, self.qkv)
        fused = ttnn.to_memory_config(fused, self.qkv_shards)
        q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
            fused, num_heads=8, num_kv_heads=2, memory_config=self.split_head_memory, overlap_qk_coregrid=False
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
        a = ttnn.experimental.nlp_concat_heads_decode(a, num_heads=8, sub_core_grids=self.worker_grid)
        a = ttnn.reshape(a, [1, 1, x.shape[2], 1024], list(a.padded_shape))
        return self.finish(x, a)

    def prefill_chunk(
        self, x, *, page_table, chunk_page_table, kv_cache, cos, sin, start_pos, slot=0, valid_seq_len=None
    ):
        """Prepared 128/256/512/1024 bucket; logical lengths use prepare_prefill."""
        n = self.projection_input(x)
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            self.linear(n, self.qkv),
            num_heads=8,
            num_kv_heads=2,
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        q = ttnn.experimental.rotary_embedding_hf(q, cos, sin)
        k = ttnn.experimental.rotary_embedding_hf(k, cos, sin)
        fill_kwargs = (
            {}
            if valid_seq_len is None
            else {"cache_position_modulo": x.shape[2], "valid_seq_len_tensor": valid_seq_len}
        )
        ttnn.experimental.paged_fill_cache(
            kv_cache[0], ttnn.typecast(k, kv_cache[0].dtype), chunk_page_table, batch_idx=slot, **fill_kwargs
        )
        ttnn.experimental.paged_fill_cache(
            kv_cache[1], ttnn.typecast(v, kv_cache[1].dtype), chunk_page_table, batch_idx=slot, **fill_kwargs
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
            compute_kernel_config=self.prefill_attention_compute,
        )
        return self.finish(x, ttnn.experimental.nlp_concat_heads(a))

    def _prepare_ccl_buffers(self):
        dtype = getattr(ttnn, self.policy["ccl_dtype"])

        if self.policy.get("collective") == "sharded_ar" and not self.residual_sharded:
            self.ar_buffers = [
                ttnn.zeros(
                    [1, 1, 32, 16384],
                    dtype=dtype,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.device,
                    memory_config=self.shard(16384),
                )
                for _ in range(2)
            ]
            self.ar_index = 0
            self.rs_buffers = self.ag_buffers = None
            return

        def alloc(width):
            return ttnn.zeros(
                [1, 1, 32, width],
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )

        self.rs_buffers = [[alloc(4096), alloc(1024)] for _ in range(2)]
        self.ag_buffers = [alloc(4096) for _ in range(2)]
        if self.policy.get("persistent_contiguous", False):
            for i in range(2):
                staging = ttnn.experimental.reduce_scatter_minimal_async_create_intermediate_buffer(
                    self.rs_buffers[i][0], dim=3, topology=ttnn.Topology.Ring
                )
                self.rs_buffers[i] = [staging[0], self.rs_buffers[i][1], staging[1]]
        self.rs_index = self.ag_index = 0

    def _sharded_all_reduce(self, x):
        """Preserve the replicated width-sharded residual without interleaving.

        The shared pool has two slots, consumed sequentially by O and down.
        Each slot holds four ranks' partial values per output core. It must
        remain alive until all traces using the pool are released.
        """
        batch = x.shape[-2]
        memory = self.shard(4096)
        x = ttnn.to_memory_config(x, memory)
        x = ttnn.reshape(x, [1, 1, 32, 4096], [1, 1, 32, 4096])
        dtype = getattr(ttnn, self.policy["ccl_dtype"])
        if x.dtype != dtype:
            x = ttnn.typecast(x, dtype)
        if self.policy.get("persistent", False):
            buffer = self.ar_buffers[self.ar_index]
            self.ar_index = 1 - self.ar_index
        else:
            buffer = ttnn.empty(
                [1, 1, 32, 16384],
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=self.shard(16384),
            )
        y = ttnn.experimental.all_reduce_async(
            x,
            buffer,
            cluster_axis=1,
            mesh_device=self.device,
            multi_device_global_semaphore=self.ccl.get_and_cycle_barrier_semaphore_handle(),
            memory_config=memory,
            dtype=ttnn.bfloat16,
            topology=ttnn.Topology.Ring,
            num_links=self.policy.get("ar_links", 2),
        )
        return ttnn.reshape(y, [1, 1, batch, 4096], list(y.padded_shape))

    def _persistent_gather(self, x, dim=3):
        if x.shape[-2] > 32 or x.shape[-1] != 1024 or self.policy["residual"] != "replicated":
            return self.gather(x, dim)
        batch = x.shape[-2]
        x = ttnn.reshape(ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG), [1, 1, 32, 1024], [1, 1, 32, 1024])
        out = self.ag_buffers[self.ag_index]
        self.ag_index = 1 - self.ag_index
        y = ttnn.experimental.all_gather_async(
            x,
            dim=3,
            persistent_output_buffer=out,
            multi_device_global_semaphore=self.ccl.get_and_cycle_ag_semaphore_handles(),
            barrier_semaphore=self.ccl.get_and_cycle_barrier_semaphore_handle(),
            num_links=1,
            topology=ttnn.Topology.Ring,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            chunks_per_sync=self.policy.get("ccl_chunks", 10),
            num_workers_per_link=self.policy.get("ccl_workers", 2) if x.shape[-2] <= 32 else 2,
            num_buffers_per_channel=2,
        )
        return ttnn.reshape(y, [1, 1, batch, 4096], list(y.padded_shape))

    def _persistent_reduce(self, x):
        if x.shape[-2] > 32:
            return self.reduce(x)
        batch = x.shape[-2]
        x = ttnn.reshape(ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG), [1, 1, 32, 4096], [1, 1, 32, 4096])
        dtype = getattr(ttnn, self.policy["ccl_dtype"])
        if x.dtype != dtype:
            x = ttnn.typecast(x, dtype)
        buffers = self.rs_buffers[self.rs_index]
        self.rs_index = 1 - self.rs_index
        y = ttnn.experimental.reduce_scatter_minimal_async(
            x,
            dim=3,
            persistent_output_buffers=buffers,
            multi_device_global_semaphore=self.ccl.get_and_cycle_rs_semaphore_handles(),
            barrier_semaphore=self.ccl.get_and_cycle_barrier_semaphore_handle(),
            num_links=1,
            topology=ttnn.Topology.Ring,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            intermediate_memory_config=ttnn.L1_MEMORY_CONFIG,
            chunks_per_sync=self.policy.get("ccl_chunks", 10),
            num_workers_per_link=self.policy.get("ccl_workers", 2) if x.shape[-2] <= 32 else 2,
            num_buffers_per_channel=2,
        )
        if not self.residual_sharded:
            y = self.gather(y)
        if y.dtype != ttnn.bfloat16:
            y = ttnn.typecast(y, ttnn.bfloat16)
        return ttnn.reshape(y, [1, 1, batch, 1024 if self.residual_sharded else 4096], list(y.padded_shape))
