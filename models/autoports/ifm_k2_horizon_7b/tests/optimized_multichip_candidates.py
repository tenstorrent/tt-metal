"""Stage05 candidates preserve the cumulative selected prefill and TP4 contract."""

import types

import ttnn


def configure(layer, name, state):
    if "+" in name:
        for part in name.split("+"):
            configure(layer, part, state)
        return
    parts = name.split(":")
    if parts[0] == "prefill_ccl":
        original = layer._project_reduce
        selected_role = parts[1]

        def project(x, w, decode):
            active = not decode and (selected_role == "both" or (w is layer.wo) == (selected_role == "attention"))
            saved = layer.ccl_dtype
            if active:
                layer.ccl_dtype = ttnn.bfloat8_b
            try:
                return original(x, w, decode)
            finally:
                layer.ccl_dtype = saved

        layer._project_reduce = project
        return
    if parts[0] == "entry":
        original = layer.decode_forward

        def decode(x, **kwargs):
            return original(ttnn.to_memory_config(x, layer._stage5_residual_mem), **kwargs)

        layer.decode_forward = decode
        return
    if parts[0] == "attention_math":
        compute = ttnn.init_device_compute_kernel_config(
            layer.mesh_device.arch(),
            math_fidelity=getattr(ttnn.MathFidelity, parts[1]),
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        for method in ("chunked_scaled_dot_product_attention", "paged_scaled_dot_product_attention_decode"):
            original = getattr(ttnn.transformer, method)

            def wrap(original):
                def attention(*args, **kwargs):
                    kwargs["compute_kernel_config"] = compute
                    return original(*args, **kwargs)

                return attention

            setattr(ttnn.transformer, method, wrap(original))
        return
    if parts[0] == "split_qkv":
        pass

        original = layer._decode_linear
        gamma = state["model.layers.0.input_layernorm.weight"].bfloat16().float()
        projections = []
        banks = layer.mesh_device.dram_grid_size()
        bank_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(banks.x - 1, banks.y - 1))})
        for role, cores, readers in (("q", 16, 2), ("k", 8, 1), ("v", 8, 1)):
            weight = (
                state[f"model.layers.0.self_attn.{role}_proj.weight"].bfloat16().float().T * gamma[:, None]
            ).bfloat16()
            n = weight.shape[-1] // 4
            memory = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.DRAM,
                ttnn.ShardSpec(bank_grid, (4096, n // banks.x), ttnn.ShardOrientation.ROW_MAJOR),
            )
            w = ttnn.from_torch(
                weight.contiguous(),
                device=layer.mesh_device,
                dtype=ttnn.bfloat4_b,
                layout=ttnn.TILE_LAYOUT,
                memory_config=memory,
                mesh_mapper=ttnn.ShardTensorToMesh(layer.mesh_device, dim=-1),
            )
            projections.append(w)
            layer.decode_weights[id(w)] = w
            grid = ttnn.num_cores_to_corerangeset(
                cores, layer.mesh_device.compute_with_storage_grid_size(), row_wise=True
            )
            layer.decode_inputs[id(w)] = ttnn.create_sharded_memory_config(
                (32, 4096 // cores),
                core_grid=grid,
                strategy=ttnn.ShardStrategy.WIDTH,
                use_height_and_width_as_shard_shape=True,
            )
            layer.decode_programs[id(w)] = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                in0_block_w=32, per_core_M=1, per_core_N=n // (32 * cores), num_workers_per_dram_bank=readers
            )
            layer.decode_computes[id(w)] = layer.decode_computes[id(layer.wqkv)]
        output_memory = ttnn.create_sharded_memory_config(
            (32, 64),
            core_grid=ttnn.CoreGrid(x=8, y=3),
            strategy=ttnn.ShardStrategy.WIDTH,
            use_height_and_width_as_shard_shape=True,
        )

        def linear(x, w):
            if w is not layer.wqkv:
                return original(x, w)
            values = [ttnn.to_memory_config(original(x, p), ttnn.L1_MEMORY_CONFIG) for p in projections]
            return ttnn.to_memory_config(
                ttnn.concat(values, dim=-1, memory_config=ttnn.L1_MEMORY_CONFIG), output_memory
            )

        layer._decode_linear = linear
        return
    if parts[0] == "sdpa":
        gx, gy, chunk = map(int, parts[1:4])
        max_cores = int(parts[4]) if len(parts) > 4 else 16
        original = ttnn.transformer.paged_scaled_dot_product_attention_decode

        def attention(*args, **kwargs):
            table = kwargs["page_table_tensor"]
            padding = (-table.shape[1]) % (chunk // 32)
            if padding:
                kwargs["page_table_tensor"] = ttnn.concat([table, ttnn.repeat(table[:, -1:], (1, padding))], dim=1)
            kwargs["program_config"] = ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=(gx, gy),
                q_chunk_size=0,
                k_chunk_size=chunk,
                exp_approx_mode=False,
                max_cores_per_head_batch=max_cores,
            )
            return original(*args, **kwargs)

        ttnn.transformer.paged_scaled_dot_product_attention_decode = attention
        return
    if parts[0] == "legacy":
        from .multichip_candidates import configure as prior

        prior(layer, ":".join(parts[1:]), state)
        # Restore the selected fused prefill contract after decode-only hooks.
        layer.prefill_agmm = True
        return
    if parts[0] not in ("residual", "ccl"):
        raise ValueError(name)
    residual_cores = int(parts[1]) if parts[0] == "residual" else 0
    workers = int(parts[2]) if len(parts) > 2 else 2
    intermediate_l1 = len(parts) > 3 and parts[3] == "l1"
    links = int(parts[4]) if len(parts) > 4 else layer.num_links
    ccl_dtype = getattr(ttnn, parts[5]) if len(parts) > 5 else ttnn.bfloat16
    persistence = parts[6] if len(parts) > 6 else "none"
    buffers = {}
    original_gather, original_reduce, original_norm = layer._gather, layer._reduce, layer._norm
    if residual_cores:
        residual_mem = ttnn.create_sharded_memory_config(
            (32, 1024 // residual_cores),
            core_grid=ttnn.CoreGrid(x=residual_cores, y=1),
            strategy=ttnn.ShardStrategy.WIDTH,
            use_height_and_width_as_shard_shape=True,
        )
        layer._stage5_residual_mem = residual_mem
        norm_program = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=(residual_cores, 1),
            subblock_w=1,
            block_h=1,
            block_w=32 // residual_cores,
            inplace=False,
        )

        def norm(self, x):
            if x.shape[2] > 32:
                return original_norm(x)
            normalized = ttnn.rms_norm(
                ttnn.to_memory_config(x, residual_mem),
                epsilon=self.eps,
                program_config=norm_program,
                compute_kernel_config=self.compute,
            )
            return self._gather(normalized)

        layer._norm = types.MethodType(norm, layer)

    def gather(self, x):
        if x.shape[2] > 32:
            return original_gather(x)
        # Direct sharded input/output avoids both norm→AG and AG→QKV conversions.
        if x.dtype != ccl_dtype:
            x = ttnn.typecast(x, ccl_dtype)
        memory = self.decode_inputs[id(self.wqkv)] if residual_cores else ttnn.L1_MEMORY_CONFIG
        if persistence != "none" and "ag" not in buffers:
            buffers["ag"] = ttnn.empty(
                list(x.shape)[:-1] + [4096],
                dtype=x.dtype,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
                memory_config=memory,
            )
        result = ttnn.experimental.all_gather_async(
            x,
            dim=3,
            persistent_output_buffer=buffers.get("ag"),
            multi_device_global_semaphore=self.ccl.get_and_cycle_ag_semaphore_handles(),
            barrier_semaphore=self.ccl.get_and_cycle_barrier_semaphore_handle(),
            num_links=links,
            topology=ttnn.Topology.Ring,
            memory_config=memory,
            chunks_per_sync=10,
            num_workers_per_link=workers,
            num_buffers_per_channel=2,
        )
        return ttnn.typecast(result, ttnn.bfloat16) if result.dtype != ttnn.bfloat16 else result

    def reduce(self, x):
        if x.shape[2] > 32:
            return original_reduce(x)
        if x.dtype != ccl_dtype:
            x = ttnn.typecast(x, ccl_dtype)
        memory = residual_mem if residual_cores else ttnn.L1_MEMORY_CONFIG
        if persistence != "none" and "rs" not in buffers:
            out = ttnn.empty(
                list(x.shape)[:-1] + [1024],
                dtype=x.dtype,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
                memory_config=memory,
            )
            if persistence == "tile":
                middle = ttnn.empty(
                    x.shape,
                    dtype=x.dtype,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.mesh_device,
                    memory_config=ttnn.L1_MEMORY_CONFIG if intermediate_l1 else ttnn.DRAM_MEMORY_CONFIG,
                )
                buffers["rs"] = [middle, out]
            else:
                middle, penult = ttnn.experimental.reduce_scatter_minimal_async_create_intermediate_buffer(
                    x, dim=3, topology=ttnn.Topology.Ring
                )
                buffers["rs"] = [middle, out, penult]
        result = ttnn.experimental.reduce_scatter_minimal_async(
            x,
            dim=3,
            persistent_output_buffers=buffers.get("rs"),
            multi_device_global_semaphore=self.ccl.get_and_cycle_rs_semaphore_handles(),
            barrier_semaphore=self.ccl.get_and_cycle_barrier_semaphore_handle(),
            num_links=links,
            topology=ttnn.Topology.Ring,
            memory_config=memory,
            intermediate_memory_config=ttnn.L1_MEMORY_CONFIG if intermediate_l1 else ttnn.DRAM_MEMORY_CONFIG,
            chunks_per_sync=10,
            num_workers_per_link=workers,
            num_buffers_per_channel=2,
        )
        return ttnn.typecast(result, ttnn.bfloat16) if result.dtype != ttnn.bfloat16 else result

    layer._gather = types.MethodType(gather, layer)
    layer._reduce = types.MethodType(reduce, layer)
