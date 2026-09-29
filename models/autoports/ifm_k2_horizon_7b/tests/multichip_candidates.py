"""Shape-faithful topology experiments; never selected implicitly at runtime."""

import dataclasses
import math
import types

import ttnn

from ..tt.optimized_decoder import MatmulGeometry, PrecisionPolicy


def policy_for(name, policy=None):
    policy = policy or PrecisionPolicy(
        qkv_geometry=MatmulGeometry(64, 8, 2, False),
        o_geometry=MatmulGeometry(32, 4, 2, False),
        mlp_geometry=MatmulGeometry(64, 8, 2, False),
        down_geometry=MatmulGeometry(32, 12, 2, False),
    )
    if not name:
        return policy
    if "+" in name:
        for part in name.split("+"):
            policy = policy_for(part, policy)
        return policy
    if name == "selected":
        return dataclasses.replace(
            policy,
            qkv_geometry=MatmulGeometry(16, 32, 3, False),
            o_geometry=MatmulGeometry(32, 4, 2, False),
            mlp_geometry=MatmulGeometry(32, 8, 2, False),
            down_geometry=MatmulGeometry(32, 12, 2, False),
        )
    if name.startswith("geometry:"):
        _, role, cores, block, readers = name.split(":")
        return dataclasses.replace(
            policy, **{role + "_geometry": MatmulGeometry(int(cores), int(block), int(readers), False)}
        )
    if name == "fused_gate":
        return dataclasses.replace(policy, fused_gate=True)
    if name == "packed_mlp":
        return dataclasses.replace(policy, packed_mlp=True)
    if name.startswith("fidelity:"):
        _, role, value = name.split(":")
        return dataclasses.replace(policy, **{role + "_fidelity": value})
    if name.startswith("dtype:"):
        _, role, value = name.split(":")
        return dataclasses.replace(policy, **{role: value})
    raise ValueError(name)


def configure(layer, name, state):
    """Bind an experimental complete residual/consumer contract before warmup."""
    if name.startswith("stage5:"):
        from .optimized_multichip_candidates import configure as configure_stage5

        return configure_stage5(layer, name[7:], state)
    layer.prefill_agmm = False
    if "+" in name:
        for part in name.split("+"):
            configure(layer, part, state)
        return
    layer.candidate_buffers = {}
    if name.startswith("fused_prefill_agmm") or name in ("fused_all_agmm", "fused_all_agmm8"):
        parts = name.split(":")
        family = parts[0]
        blocks = tuple(map(int, parts[1:4])) if len(parts) > 1 else (8, 8, 8)
        auto_orientation = len(parts) > 4 and parts[4] == "auto"
        original_norm, original_linear, original_swiglu = layer._norm, layer._linear, layer._prefill_swiglu
        original_decode_linear, original_mlp = layer._decode_linear, layer._mlp
        all_rows = name in ("fused_all_agmm", "fused_all_agmm8")

        def norm(x):
            if x.shape[2] < 256 and not all_rows:
                return original_norm(x)
            return ttnn.rms_norm(
                ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG if x.shape[2] <= 32 else ttnn.DRAM_MEMORY_CONFIG),
                epsilon=layer.eps,
                compute_kernel_config=layer.compute,
            )

        def fused(x, w, swiglu=False):
            memory = ttnn.L1_MEMORY_CONFIG if x.shape[2] <= 32 else ttnn.DRAM_MEMORY_CONFIG
            if family.endswith("8"):
                x = ttnn.typecast(x, ttnn.bfloat8_b)
            key = ("agmm", tuple(x.shape))
            if key not in layer.candidate_buffers:
                layer.candidate_buffers[key] = ttnn.empty(
                    list(x.shape)[:-1] + [4096],
                    dtype=x.dtype,
                    layout=ttnn.TILE_LAYOUT,
                    device=layer.mesh_device,
                    memory_config=memory,
                )
            return ttnn.experimental.all_gather_minimal_matmul_async(
                x,
                w,
                config=ttnn.MinimalMatmulConfig(
                    M_block_size=2 if x.shape[2] <= 32 else blocks[0],
                    K_block_size=blocks[1],
                    N_block_size=blocks[2],
                    subblock_h=2,
                    subblock_w=4,
                    compute_with_storage_grid_size=(
                        ttnn.CoreCoord(10, 10) if auto_orientation and swiglu else ttnn.CoreCoord(11, 9)
                    ),
                ),
                multi_device_global_semaphore=layer.ccl.get_and_cycle_ag_semaphore_handles(),
                topology=ttnn.Topology.Ring,
                memory_config=memory,
                dtype=ttnn.bfloat16,
                compute_kernel_config=layer.role_compute["mlp" if swiglu else "attention"],
                persistent_output_buffer=layer.candidate_buffers[key],
                num_links=layer.num_links,
                barrier_semaphore=layer.ccl.get_and_cycle_barrier_semaphore_handle(),
                force_transpose=not (auto_orientation and swiglu),
                num_workers_per_link=5 if auto_orientation and swiglu else 6,
                num_buffers_per_channel=24,
                fuse_swiglu=swiglu,
            )[0]

        def linear(x, w):
            return fused(x, w) if w is layer.wqkv and x.shape[2] >= 256 else original_linear(x, w)

        layer._norm, layer._linear, layer._prefill_swiglu = norm, linear, lambda x: fused(x, layer.wswiglu, True)
        if all_rows:
            layer._decode_linear = lambda x, w: fused(x, w) if w is layer.wqkv else original_decode_linear(x, w)
            layer._linear = lambda x, w: fused(x, w) if w is layer.wqkv else original_linear(x, w)
            layer._mlp = lambda x, rows, decode: fused(x, layer.wswiglu, True)
        return
    if name.startswith("prefill_mm:"):
        _, m, k, n = name.split(":")

        def config(self):
            return ttnn.MinimalMatmulConfig(
                M_block_size=int(m),
                K_block_size=int(k),
                N_block_size=int(n),
                subblock_h=2,
                subblock_w=4,
                compute_with_storage_grid_size=ttnn.CoreCoord(11, 10),
            )

        layer._prefill_matmul_config = types.MethodType(config, layer)
        return
    if name.startswith("prefill_sdpa:"):
        _, q, k = name.split(":")
        layer.prefill_q_chunk, layer.prefill_k_chunk = int(q), int(k)
        return
    if name == "prefill_l1":
        original_linear, original_norm = layer._linear, layer._norm

        def linear(x, w):
            return original_linear(ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG) if x.shape[2] >= 256 else x, w)

        def norm(x):
            y = original_norm(x)
            return ttnn.to_memory_config(y, ttnn.L1_MEMORY_CONFIG) if x.shape[2] >= 256 else y

        layer._linear, layer._norm = linear, norm
        return
    if name in ("ccl_attn8", "ccl_mlp8", "ccl_both8"):
        for method in ("_gather", "_reduce"):
            original = getattr(layer, method)

            def wrap(original):
                count = 0

                def call(x):
                    nonlocal count
                    use8 = name == "ccl_both8" or (count % 2 == 0 if name == "ccl_attn8" else count % 2 == 1)
                    count += 1
                    prior = layer.ccl_dtype
                    layer.ccl_dtype = ttnn.bfloat8_b if use8 else ttnn.bfloat16
                    try:
                        return original(x)
                    finally:
                        layer.ccl_dtype = prior

                return call

            setattr(layer, method, wrap(original))
        return
    if name.startswith("norm"):
        n = int(name[4:])
        original_norm = layer._norm

        def norm(self, x):
            if x.shape[2] > 32:
                return original_norm(x)
            grid = ttnn.CoreGrid(x=n, y=1)
            mem = ttnn.create_sharded_memory_config(
                (32, 1024 // n),
                core_grid=grid,
                strategy=ttnn.ShardStrategy.WIDTH,
                use_height_and_width_as_shard_shape=True,
            )
            y = ttnn.rms_norm(
                ttnn.to_memory_config(x, mem),
                epsilon=self.eps,
                program_config=ttnn.LayerNormShardedMultiCoreProgramConfig(
                    compute_with_storage_grid_size=(n, 1), subblock_w=1, block_h=1, block_w=32 // n, inplace=False
                ),
                compute_kernel_config=self.compute,
            )
            return self._gather(y)

        layer._norm = types.MethodType(norm, layer)
        return
    if name in ("persistent", "persistent_decode"):
        original_gather, original_reduce = layer._gather, layer._reduce

        def gather(self, x):
            if name == "persistent_decode" and x.shape[2] > 32:
                return original_gather(x)
            mem = ttnn.L1_MEMORY_CONFIG if x.shape[2] <= 32 else ttnn.DRAM_MEMORY_CONFIG
            x = ttnn.to_memory_config(x, mem)
            if x.dtype != self.ccl_dtype:
                x = ttnn.typecast(x, self.ccl_dtype)
            key = ("ag", tuple(x.shape), str(x.dtype))
            if key not in self.candidate_buffers:
                self.candidate_buffers[key] = ttnn.empty(
                    list(x.shape)[:-1] + [x.shape[-1] * 4],
                    dtype=x.dtype,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.mesh_device,
                    memory_config=mem,
                )
            out = ttnn.experimental.all_gather_async(
                x,
                dim=3,
                persistent_output_buffer=self.candidate_buffers[key],
                multi_device_global_semaphore=self.ccl.get_and_cycle_ag_semaphore_handles(),
                barrier_semaphore=self.ccl.get_and_cycle_barrier_semaphore_handle(),
                num_links=self.num_links,
                topology=ttnn.Topology.Ring,
                memory_config=mem,
                chunks_per_sync=10,
                num_workers_per_link=2,
                num_buffers_per_channel=2,
            )
            return ttnn.typecast(out, ttnn.bfloat16) if out.dtype != ttnn.bfloat16 else out

        def reduce(self, x):
            if name == "persistent_decode" and x.shape[2] > 32:
                return original_reduce(x)
            mem = ttnn.L1_MEMORY_CONFIG if x.shape[2] <= 32 else ttnn.DRAM_MEMORY_CONFIG
            x = ttnn.to_memory_config(x, mem)
            if x.dtype != self.ccl_dtype:
                x = ttnn.typecast(x, self.ccl_dtype)
            key = ("rs", tuple(x.shape), str(x.dtype))
            if key not in self.candidate_buffers:
                middle, penult = ttnn.experimental.reduce_scatter_minimal_async_create_intermediate_buffer(
                    x, dim=3, topology=ttnn.Topology.Ring
                )
                out = ttnn.empty(
                    list(x.shape)[:-1] + [x.shape[-1] // 4],
                    dtype=x.dtype,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.mesh_device,
                    memory_config=mem,
                )
                self.candidate_buffers[key] = [middle, out, penult]
            out = ttnn.experimental.reduce_scatter_minimal_async(
                x,
                dim=3,
                persistent_output_buffers=self.candidate_buffers[key],
                multi_device_global_semaphore=self.ccl.get_and_cycle_rs_semaphore_handles(),
                barrier_semaphore=self.ccl.get_and_cycle_barrier_semaphore_handle(),
                num_links=self.num_links,
                topology=ttnn.Topology.Ring,
                memory_config=mem,
                intermediate_memory_config=ttnn.DRAM_MEMORY_CONFIG,
                chunks_per_sync=10,
                num_workers_per_link=2,
                num_buffers_per_channel=2,
            )
            if out.dtype != ttnn.bfloat16:
                out = ttnn.typecast(out, ttnn.bfloat16)
            return out if self.residual_sharded else self._gather(out)

        layer._gather = types.MethodType(gather, layer)
        layer._reduce = types.MethodType(reduce, layer)
        return
    if name.startswith("gather") or name.startswith("fused_ag"):
        layer.column_weights = {}
        for w, key in ((layer.wo, "self_attn.o_proj.weight"), (layer.wdown, "mlp.down_proj.weight")):
            tensor = state["model.layers.0." + key].bfloat16().T[None, None].contiguous()
            layer.column_weights[id(w)] = ttnn.from_torch(
                tensor,
                device=layer.mesh_device,
                dtype=ttnn.bfloat4_b,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=ttnn.ShardTensorToMesh(layer.mesh_device, dim=-1),
            )

    def project(self, x, w, decode):
        role = "o" if w is self.wo else "down"
        active = name.endswith("both") or name.endswith(role) or name in ("fused_rs", "persistent")
        if not active:
            return self._reduce((self._decode_linear if decode else self._linear)(x, w))
        mem = ttnn.L1_MEMORY_CONFIG if decode else ttnn.DRAM_MEMORY_CONFIG
        if name.startswith("gather"):
            gathered = self._gather(x)
            out = ttnn.linear(
                gathered,
                self.column_weights[id(w)],
                dtype=ttnn.bfloat16,
                memory_config=mem,
                compute_kernel_config=self.role_compute["attention" if role == "o" else "down"],
            )
            return out
        if name.startswith("fused_ag"):
            if not decode:
                return self._reduce(self._linear(x, w))
            cores = ttnn.CoreGrid(x=8, y=2)
            k = x.shape[-1] * 4
            local_mem = ttnn.L1_MEMORY_CONFIG
            ag_mem = ttnn.L1_MEMORY_CONFIG
            out_mem = ttnn.create_sharded_memory_config(
                (32, 64), core_grid=cores, strategy=ttnn.ShardStrategy.WIDTH, use_height_and_width_as_shard_shape=True
            )
            config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=(8, 2),
                in0_block_w=4 if role == "o" else 12,
                out_subblock_h=1,
                out_subblock_w=2,
                per_core_M=1,
                per_core_N=2,
                fuse_batch=True,
                mcast_in0=True,
            )
            _, out = ttnn.experimental.all_gather_matmul_async(
                ttnn.to_memory_config(x, local_mem),
                self.column_weights[id(w)],
                persistent_output_buffer=None,
                dim=3,
                multi_device_global_semaphore=self.ccl.get_and_cycle_ag_semaphore_handles(),
                all_gather_core_grid_offset=(0, 4),
                barrier_semaphore=self.ccl.get_and_cycle_barrier_semaphore_handle(),
                num_links=self.num_links,
                memory_config_ag=ag_mem,
                memory_config_mm=out_mem,
                program_config=config,
                compute_kernel_config=self.role_compute["attention" if role == "o" else "down"],
                chunks_per_sync=10,
                num_workers_per_link=1,
                num_buffers_per_channel=2,
            )
            return out
        if name == "fused_rs":
            shape = list(x.shape)
            shape[-1] = 4096
            key = (role, tuple(shape), decode)
            if key not in self.candidate_buffers:
                out_shape = shape[:-1] + [1024]
                self.candidate_buffers[key] = [
                    ttnn.empty(
                        shape,
                        dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT,
                        device=self.mesh_device,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    ),
                    ttnn.empty(
                        out_shape,
                        dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT,
                        device=self.mesh_device,
                        memory_config=mem,
                    ),
                ]
            middle, out = self.candidate_buffers[key]
            grid = (8, 1) if decode else (8, 6)
            config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=grid,
                in0_block_w=4 if decode else 8,
                out_subblock_h=1,
                out_subblock_w=4,
                per_core_M=max(1, math.ceil(x.shape[2] / 32 / grid[1])),
                per_core_N=16,
                out_block_w=8,
                transpose_mcast=False,
                fuse_batch=False,
            )
            _, out = ttnn.experimental.matmul_reduce_scatter_async(
                ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG),
                w,
                persistent_intermediate_buffer=middle,
                persistent_output_buffer=out,
                dim=3,
                multi_device_global_semaphore=self.ccl.get_and_cycle_rs_semaphore_handles(),
                reduce_scatter_core_grid_offset=(0, 6),
                barrier_semaphore=self.ccl.get_and_cycle_barrier_semaphore_handle(),
                num_links=self.num_links,
                memory_config_rs=mem,
                intermediate_memory_config_rs=ttnn.DRAM_MEMORY_CONFIG,
                topology=ttnn.Topology.Ring,
                memory_config_mm=ttnn.DRAM_MEMORY_CONFIG,
                program_config=config,
                dtype=ttnn.bfloat16,
                compute_kernel_config=self.role_compute["attention" if role == "o" else "down"],
            )
            return out
        return self._reduce((self._decode_linear if decode else self._linear)(x, w))

    layer._project_reduce = types.MethodType(project, layer)
