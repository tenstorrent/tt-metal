"""Stage05 experiments against the completed TP4 default."""
import math

import ttnn
from models.autoports.google_gemma_4_31b_it.tt.multichip_decoder import MultichipDecoder


class OptimizedMultichipCandidate(MultichipDecoder):
    @classmethod
    def from_state_dict(cls, state_dict, **kwargs):
        candidate = kwargs.pop("candidate")
        for key in ("precision", "attention_precision", "down_precision"):
            if key in candidate:
                kwargs[key] = candidate[key]
        obj = super().from_state_dict(state_dict, **kwargs)
        if candidate.get("output_ag"):
            import torch

            obj.output_weights = {}
            for name in ("self_attn.o_proj", "mlp.down_proj"):
                w = state_dict[name + ".weight"]
                if name == "mlp.down_proj":
                    w = torch.nn.functional.pad(w.reshape(5376, 4, 5376), (0, 256)).reshape(5376, 22528)
                w = (
                    torch.nn.functional.pad(w.reshape(4, 1344, w.shape[-1]), (0, 0, 0, 192))
                    .reshape(6144, w.shape[-1])
                    .T.contiguous()
                )
                host = ttnn.from_torch(
                    w,
                    dtype=ttnn.bfloat4_b,
                    layout=ttnn.TILE_LAYOUT,
                    mesh_mapper=ttnn.ShardTensorToMesh(obj.mesh_device, dim=1),
                )
                obj.output_weights[name] = ttnn.to_device(host, obj.mesh_device)

        if candidate.get("decode_ccl8"):
            for (batch, name), (_, full) in list(obj.persistent_ccl.items()):
                full = ttnn.typecast(full, ttnn.bfloat8_b)
                local = ttnn.zeros(
                    (1, 1, batch, 1344),
                    device=obj.mesh_device,
                    dtype=ttnn.bfloat8_b,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                staging = ttnn.experimental.reduce_scatter_minimal_async_create_intermediate_buffer(
                    full, 3, topology=obj.topology
                )
                obj.persistent_ccl[(batch, name)] = ([staging[0], local, staging[1]], full)
        if candidate.get("ccl_l1"):
            pool = {}
            for (batch, name), (staging, full) in list(obj.persistent_ccl.items()):
                if candidate.get("pool_ccl"):
                    if name not in pool:
                        pool[name] = ttnn.to_memory_config(full, obj._working_memory(full, 8))
                    output = obj._sequence_view(pool[name], batch)
                else:
                    output = ttnn.to_memory_config(full, obj._working_memory(full, 8))
                obj.persistent_ccl[(batch, name)] = (staging, output)
        obj.candidate = candidate
        obj.geometry.update(candidate.get("geometry", {}))
        return obj

    def _gather(self, x, *, group=None):
        if self.decoding and self.candidate.get("ccl_l1"):
            output = self.persistent_ccl[(x.shape[-2], group)][1]
            return ttnn.experimental.all_gather_async(
                x,
                dim=3,
                persistent_output_buffer=output,
                multi_device_global_semaphore=self.ccl.get_and_cycle_ag_semaphore_handles(),
                barrier_semaphore=self.ccl.get_and_cycle_barrier_semaphore_handle(),
                topology=self.topology,
                num_links=2,
                num_workers_per_link=2,
                num_buffers_per_channel=1,
                memory_config=output.memory_config(),
            )
        # Suppress an identity BFP8 typecast without changing the payload.
        if self.candidate.get("no_identity_cast") and x.dtype == ttnn.bfloat8_b:
            saved = self.prefill_ccl_bfp8
            self.prefill_ccl_bfp8 = False
            try:
                out = super()._gather(x, group=group)
            finally:
                self.prefill_ccl_bfp8 = saved
            return ttnn.typecast(out, ttnn.bfloat16)
        return super()._gather(x, group=group)

    def _norm(self, x, name=None):
        if (
            not self.decoding
            and name == "pre_feedforward_layernorm"
            and self.candidate.get("prefill_shared_mlp_input_l1")
        ):
            return ttnn.to_memory_config(super()._norm(x, name), ttnn.L1_MEMORY_CONFIG)
        if (
            self.candidate.get("prefill_norm")
            and not self.decoding
            and name
            and "layernorm" in name
            and x.shape[-1] == 5376
        ):
            rows = math.prod(tuple(x.padded_shape)[:-1])
            if rows <= 1024:
                height = math.ceil(rows / (8 * 32)) * 32
                gy = math.ceil(rows / height)
                grid = ttnn.CoreGrid(x=8, y=gy)
                memory = ttnn.create_sharded_memory_config(
                    (height, 672),
                    grid,
                    ttnn.ShardStrategy.BLOCK,
                    ttnn.ShardOrientation.ROW_MAJOR,
                    use_height_and_width_as_shard_shape=True,
                )
                sharded = ttnn.to_memory_config(x, memory)
                out = ttnn.rms_norm(
                    sharded,
                    epsilon=self.config.rms_norm_eps,
                    weight=self.weights[name + ".weight"],
                    compute_kernel_config=self.kernel,
                    program_config=ttnn.LayerNormShardedMultiCoreProgramConfig(
                        compute_with_storage_grid_size=(8, gy),
                        subblock_w=3,
                        block_h=height // 32,
                        block_w=21,
                        inplace=True,
                    ),
                    memory_config=memory,
                )
                return ttnn.to_memory_config(out, x.memory_config())
        return super()._norm(x, name)

    def _working_memory(self, x, cores):
        return super()._working_memory(x, cores)

    def _linear(self, x, name):
        if not self.decoding and name in self.candidate.get("prefill_input_l1", []):
            seen = getattr(self, "input_l1_seen", set())
            if name not in seen:
                view = ttnn.get_memory_view(self.mesh_device, ttnn.BufferType.L1)
                print(
                    "INPUT_L1_CAPACITY",
                    name,
                    tuple(x.padded_shape),
                    str(x.dtype),
                    {
                        key: getattr(view, key)
                        for key in (
                            "num_banks",
                            "total_bytes_per_bank",
                            "total_bytes_allocated_per_bank",
                            "largest_contiguous_bytes_free_per_bank",
                        )
                    },
                    flush=True,
                )
                seen.add(name)
                self.input_l1_seen = seen
            x = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)
            kernel_group = (
                "attention" if name.startswith("self_attn.") else "down" if name == "mlp.down_proj" else "mlp"
            )
            out = ttnn.experimental.minimal_matmul(
                input_tensor=x,
                weight_tensor=self.weights[name + ".weight"],
                config=ttnn.MinimalMatmulConfig(
                    M_block_size=self.candidate.get("prefill_mblock", 4),
                    K_block_size=self.candidate.get("prefill_kblock", 8),
                    N_block_size=self.candidate.get("prefill_nblock", 16),
                    subblock_h=1,
                    subblock_w=min(8, self.candidate.get("prefill_nblock", 16)),
                    compute_with_storage_grid_size=self.mesh_device.compute_with_storage_grid_size(),
                ),
                compute_kernel_config=self.linear_kernels[kernel_group],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                dtype=ttnn.bfloat8_b if name in ("mlp.gate_proj", "mlp.up_proj") else ttnn.bfloat16,
            )
            if name in self.output_padding:
                out = ttnn.slice(out, [0] * len(out.shape), [*tuple(out.shape)[:-1], self.hidden])
                out = ttnn.typecast(out, ttnn.bfloat8_b)
                out = ttnn.experimental.reduce_scatter_minimal_async(
                    out,
                    dim=3,
                    multi_device_global_semaphore=self.ccl.get_and_cycle_rs_semaphore_handles(),
                    barrier_semaphore=self.ccl.get_and_cycle_barrier_semaphore_handle(),
                    topology=self.topology,
                    num_links=2,
                    num_buffers_per_channel=1,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                out = self._gather(out, group=name)
            return out
        group = "attention" if name.startswith("self_attn.") else "mlp"
        if self.decoding and self.candidate.get("activation") == group:
            x = ttnn.typecast(x, ttnn.bfloat8_b)
        if name in self.output_padding and self.candidate.get("output_ag"):
            x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
            if self.candidate.get("ag_bf16_input") and x.dtype != ttnn.bfloat16:
                x = ttnn.typecast(x, ttnn.bfloat16)
            kblock = self.candidate.get("ag_kblock", 8)
            # Each input shard's K tiles must divide the fused transfer block.
            kblock = max(v for v in range(1, kblock + 1) if (x.shape[-1] // 32) % v == 0)
            out = ttnn.experimental.all_gather_minimal_matmul_async(
                x,
                self.output_weights[name],
                config=ttnn.MinimalMatmulConfig(
                    M_block_size=1 if self.decoding else 4,
                    K_block_size=kblock,
                    N_block_size=self.candidate.get("ag_nblock", 8),
                    subblock_h=1,
                    subblock_w=4,
                    compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
                ),
                dtype=ttnn.bfloat16,
                compute_kernel_config=self.linear_kernels["attention" if name.startswith("self_attn.") else "down"],
                multi_device_global_semaphore=self.ccl.get_and_cycle_ag_semaphore_handles(),
                barrier_semaphore=self.ccl.get_and_cycle_barrier_semaphore_handle(),
                num_links=2,
                num_workers_per_link=4,
                num_buffers_per_channel=1,
                topology=self.topology,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )[0]
            out = ttnn.slice(out, [0] * len(out.shape), [*tuple(out.shape)[:-1], 1344])
            if not self.residual_sharded:
                out = self._gather(out, group=name)
                if self.decoding:
                    out = ttnn.to_memory_config(out, self._working_memory(out, 8))
            return out
        if not (self.decoding and name in self.output_padding and self.candidate.get("row_direct")):
            return super()._linear(x, name)
        role = self.geometry[name]
        weight = self.weights[name + ".weight"]
        x = ttnn.to_memory_config(x, self._working_memory(x, role["cores"]))
        out = ttnn.linear(
            x,
            weight,
            program_config=ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                in0_block_w=role["block"],
                per_core_M=1,
                per_core_N=math.ceil(weight.shape[-1] / 32 / role["cores"]),
                num_workers_per_dram_bank=role["readers"],
            ),
            compute_kernel_config=self.linear_kernels["attention" if name.startswith("self_attn.") else "down"],
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
        )
        out = ttnn.to_memory_config(
            out, ttnn.L1_MEMORY_CONFIG if self.candidate.get("row_l1_slice") else ttnn.DRAM_MEMORY_CONFIG
        )
        out = ttnn.slice(out, [0] * len(out.shape), [*tuple(out.shape)[:-1], self.hidden])
        out = ttnn.to_memory_config(out, ttnn.DRAM_MEMORY_CONFIG)
        if self.candidate.get("decode_ccl8"):
            out = ttnn.typecast(out, ttnn.bfloat8_b)
        out = ttnn.experimental.reduce_scatter_minimal_async(
            out,
            dim=3,
            persistent_output_buffers=self.persistent_ccl[(out.shape[-2], name)][0],
            multi_device_global_semaphore=self.ccl.get_and_cycle_rs_semaphore_handles(),
            barrier_semaphore=self.ccl.get_and_cycle_barrier_semaphore_handle(),
            topology=self.topology,
            num_links=2,
            num_workers_per_link=2,
            num_buffers_per_channel=1,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        out = self._gather(out, group=name)
        if out.dtype != ttnn.bfloat16:
            out = ttnn.typecast(out, ttnn.bfloat16)
        return ttnn.to_memory_config(out, self._working_memory(out, 8))

    def prefill_forward(self, x, *, cos, sin, kv_cache, page_table, chunk_start=0):
        if not self.candidate.get("batch_l1_chunks"):
            return super().prefill_forward(
                x, cos=cos, sin=sin, kv_cache=kv_cache, page_table=page_table, chunk_start=chunk_start
            )
        chunk_size = max(32, self.candidate.get("prefill_chunk_rows", 1024) // x.shape[0])
        outputs = []
        for offset in range(0, x.shape[2], chunk_size):
            length = min(chunk_size, x.shape[2] - offset)

            def part(t):
                return ttnn.slice(t, (0, 0, offset, 0), (t.shape[0], 1, offset + length, t.shape[3]))

            outputs.append(
                super().prefill_forward(
                    part(x),
                    cos=part(cos),
                    sin=part(sin),
                    kv_cache=kv_cache,
                    page_table=page_table,
                    chunk_start=chunk_start + offset,
                )
            )
        return ttnn.concat(outputs, dim=2)
