"""TP4 decoder for IFM/K2-Horizon-7B; OptimizedDecoder is the single-chip baseline.

One rank owns8 Q heads,2 KV heads,3072 intermediate channels and optionally one
1024-channel grouped-RMS residual. Page maps, absolute positions and RoPE are
replicated. Public prefill preserves arbitrary logical lengths and continuation.
Setup alone performs host conversion. Inherited helpers preserve optimized RoPE,
paged update, decode-assisted continuation and long-context accurate attention.
"""

import math

import ttnn
from models.common.modules.tt_ccl import TT_CCL

from .collective_buffers import DecodeCollectiveBuffers
from .optimized_decoder import MatmulGeometry, OptimizedDecoder, PrecisionPolicy


class MultichipDecoder(OptimizedDecoder):
    prefill_q_chunk = 128
    prefill_k_chunk = 256
    accurate_attention_batch_size = 32
    accurate_attention_transport = "packed_gqa_rows_v1"
    # Beyond the stock decode bound: split-K flash decode with FP32 recurrence
    # (accurate_flash_decode). "chunked" restores the chunked-prefill fallback.
    accurate_decode_kernel = "flash"
    flash_decode_k_chunk = 256
    flash_decode_max_cores_per_head = 64  # tree reduction limit; B1 uses all 55 per KV head

    @classmethod
    def from_state_dict(
        cls,
        state_dict,
        *,
        hf_config,
        layer_idx,
        mesh_device,
        policy=None,
        residual_sharded=True,
        ccl_dtype="bfloat16",
        num_links=2,
        prefill_agmm=True,
        collective_buffers=None,
        residual_dtype="bfloat16",
        norm_fidelity="HiFi4",
        matmul_output_dtype="bfloat16",
        math_approx_mode=False,
        packer_l1_acc=True,
    ):
        import torch

        from .accurate_attention import _load, accurate_attention, accurate_flash_decode

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
        if mesh_device.get_num_devices() != 4:
            raise ValueError("This implementation targets exactly four Blackhole chips")
        self.policy = policy or PrecisionPolicy(
            qkv_geometry=MatmulGeometry(16, 32, 1, False),
            o_geometry=MatmulGeometry(32, 4, 2, False),
            mlp_geometry=MatmulGeometry(32, 8, 3, False),
            down_geometry=MatmulGeometry(32, 12, 1, False),
        )
        self.residual_sharded = residual_sharded
        self.prefill_agmm = prefill_agmm and residual_sharded and self.policy.prefill_fused
        self._prefill_gather_buffer = None
        self._prefill_gather_shape = None
        self.prefill_qkv_dtype = getattr(ttnn, self.policy.prefill_qkv_activation)
        self.prefill_mlp_dtype = getattr(ttnn, self.policy.prefill_mlp_activation)
        self.ccl_dtype = getattr(ttnn, ccl_dtype)
        # A shared QKV/MLP BFP8 input policy can consume the gather directly.
        # Avoid a lossless BFP8 -> BF16 -> BFP8 round trip at both norms.
        self.gather_output_dtype = (
            self.ccl_dtype
            if self.policy.attention_activation == self.policy.mlp_activation == "bfloat8_b"
            else ttnn.bfloat16
        )
        self.residual_dtype = getattr(ttnn, residual_dtype)
        self.matmul_output_dtype = getattr(ttnn, matmul_output_dtype)
        self.num_links = num_links
        self.ccl = TT_CCL(mesh_device)
        self.collective_buffers = collective_buffers or DecodeCollectiveBuffers(mesh_device)
        if self.collective_buffers.mesh_device is not mesh_device:
            raise ValueError("Collective buffers must belong to this mesh")
        # Each rank owns a complete1024-channel RMS group. Keep that residual
        # width-sharded on two cores through both adds and the next layer.
        self.decode_residual_memory = ttnn.create_sharded_memory_config(
            (32, 512),
            core_grid=ttnn.CoreGrid(x=2, y=1),
            strategy=ttnn.ShardStrategy.WIDTH,
            use_height_and_width_as_shard_shape=True,
        )
        self.decode_norm_program = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=(2, 1),
            subblock_w=1,
            block_h=1,
            block_w=16,
            inplace=False,
        )
        self.local_q_heads, self.local_kv_heads = 8, 2
        self.weight_allocations = []
        self.kv_dtype = getattr(ttnn, self.policy.kv)
        self._attention = accurate_attention
        self._flash_decode = accurate_flash_decode
        self.eps = c.rms_norm_eps
        self.context = c.max_position_embeddings
        self.compute = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=getattr(ttnn.MathFidelity, norm_fidelity),
            math_approx_mode=math_approx_mode,
            fp32_dest_acc_en=True,
            packer_l1_acc=packer_l1_acc,
        )
        self.attention_compute = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=getattr(ttnn.MathFidelity, self.policy.sdpa_fidelity),
            math_approx_mode=math_approx_mode,
            fp32_dest_acc_en=True,
            packer_l1_acc=packer_l1_acc,
        )
        prefix = f"model.layers.{layer_idx}."

        def weight(name):
            return state_dict[prefix + name].detach().to(torch.bfloat16)

        self.decode_weights = {}
        self.decode_programs = {}
        self.decode_inputs = {}
        self.decode_computes = {}

        def device(tensor, dtype, role):
            # Host tensor [4,Klocal,Nlocal]; each rank's independently packed
            # weights are concatenated only for mesh upload, not for execution.
            def upload(parts, memory_config):
                return ttnn.from_torch(
                    torch.cat(list(parts), dim=-1).contiguous(),
                    device=mesh_device,
                    dtype=getattr(ttnn, dtype),
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=memory_config,
                    mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=-1),
                )

            result = upload(tensor, ttnn.DRAM_MEMORY_CONFIG)
            self.weight_allocations.append((role, "prefill", tuple(tensor.shape[-2:]), dtype))
            if self.policy.dram:
                g = getattr(self.policy, role + "_geometry")
                k, n = tensor.shape[-2:]
                banks = mesh_device.dram_grid_size()
                bank_grid = ttnn.CoreRangeSet(
                    {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(banks.x - 1, banks.y - 1))}
                )
                nphysical = math.ceil(n / math.lcm(32 * g.cores, 32 * banks.x * g.readers)) * math.lcm(
                    32 * g.cores, 32 * banks.x * g.readers
                )
                bank_width = nphysical // banks.x
                tensor = torch.nn.functional.pad(tensor, (0, nphysical - n))
                wm = ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                    ttnn.BufferType.DRAM,
                    ttnn.ShardSpec(bank_grid, (k, bank_width), ttnn.ShardOrientation.ROW_MAJOR),
                )
                self.decode_weights[id(result)] = upload(tensor, wm)
                self.weight_allocations.append((role, "decode", tuple(tensor.shape[-2:]), dtype))
                grid = ttnn.num_cores_to_corerangeset(
                    g.cores, mesh_device.compute_with_storage_grid_size(), row_wise=True
                )
                self.decode_inputs[id(result)] = ttnn.create_sharded_memory_config(
                    (32, k // g.cores),
                    core_grid=grid,
                    strategy=ttnn.ShardStrategy.WIDTH,
                    orientation=ttnn.ShardOrientation.ROW_MAJOR,
                    use_height_and_width_as_shard_shape=True,
                )
                self.decode_programs[id(result)] = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                    in0_block_w=g.block_w,
                    per_core_M=1,
                    per_core_N=nphysical // (32 * g.cores),
                    num_workers_per_dram_bank=g.readers,
                )
                group = "attention" if role in ("qkv", "o") else role
                fidelity = getattr(self.policy, role + "_fidelity", None) or getattr(self.policy, group + "_fidelity")
                self.decode_computes[id(result)] = ttnn.init_device_compute_kernel_config(
                    mesh_device.arch(),
                    math_fidelity=getattr(ttnn.MathFidelity, fidelity),
                    math_approx_mode=math_approx_mode,
                    fp32_dest_acc_en=g.fp32,
                    packer_l1_acc=packer_l1_acc,
                )
            return result

        # Fold the norm affine into each consumer's input channels once at setup.
        # Multiply BF16 checkpoint values in FP32, then store BF16 fused weights.
        gamma1 = weight("input_layernorm.weight").float()
        gamma2 = weight("post_attention_layernorm.weight").float()

        def folded(name, gamma):
            return (weight(name).float().T * gamma[:, None]).bfloat16()

        self.wqkv = device(
            torch.stack(
                [
                    torch.cat([folded(f"self_attn.{p}_proj.weight", gamma1).chunk(4, -1)[rank] for p in "qkv"], -1)
                    for rank in range(4)
                ]
            ),
            self.policy.qkv_dtype or self.policy.attention,
            "qkv",
        )
        self.wo = device(
            torch.stack(weight("self_attn.o_proj.weight").T.chunk(4, -2)),
            self.policy.o_dtype or self.policy.attention,
            "o",
        )
        gate, up = folded("mlp.gate_proj.weight", gamma2), folded("mlp.up_proj.weight", gamma2)
        gate, up = [torch.stack(w.chunk(4, -1)) for w in (gate, up)]
        self.wgate = device(gate, self.policy.mlp, "mlp")
        self.wup = device(up, self.policy.mlp, "mlp")
        self.wdown = device(torch.stack(weight("mlp.down_proj.weight").T.chunk(4, -2)), self.policy.down, "down")
        # Packed and separate are compared as complete paths. The selected separate
        # path avoids split/layout costs, so it does not retain unused packed weights.
        self.wgateup = device(torch.cat([gate, up], dim=-1), self.policy.mlp, "mlp") if self.policy.packed_mlp else None
        if self.policy.prefill_fused:
            packed = torch.stack([w.reshape(4, 4096, 96, 32) for w in (gate, up)], dim=3).reshape(4, 4096, 6144)
            self.wswiglu = ttnn.from_torch(
                torch.cat(list(packed), dim=-1).contiguous(),
                mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=-1),
                device=mesh_device,
                dtype=getattr(ttnn, self.policy.mlp),
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            self.weight_allocations.append(("swiglu", "prefill", (4096, 6144), self.policy.mlp))
        self.role_compute = {}
        for role in ("attention", "o", "mlp", "down"):
            fidelity_name = (
                self.policy.qkv_fidelity or self.policy.attention_fidelity
                if role == "attention"
                else (
                    self.policy.o_fidelity or self.policy.attention_fidelity
                    if role == "o"
                    else getattr(self.policy, role + "_fidelity")
                )
            )
            fidelity = getattr(ttnn.MathFidelity, fidelity_name)
            self.role_compute[role] = ttnn.init_device_compute_kernel_config(
                mesh_device.arch(),
                math_fidelity=fidelity,
                math_approx_mode=math_approx_mode,
                fp32_dest_acc_en=self.policy.prefill_fp32,
                packer_l1_acc=packer_l1_acc,
            )
        if self.policy.dram and self.policy.fused_gate:
            self.decode_programs[id(self.wgate)].fused_activation = ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU)
        return self

    def _weight_compute(self, w):
        if w is self.wo:
            return self.role_compute["o"]
        return super()._weight_compute(w)

    def _gather(self, x):
        decode = self.residual_sharded and self.policy.dram and x.shape[2] <= 32
        memory = ttnn.L1_MEMORY_CONFIG if x.shape[2] <= 32 else ttnn.DRAM_MEMORY_CONFIG
        if not decode:
            x = ttnn.to_memory_config(x, memory)
        if x.dtype != self.ccl_dtype:
            x = ttnn.typecast(x, self.ccl_dtype)
        output_memory = self.decode_inputs[id(self.wqkv)] if decode else memory
        result = ttnn.experimental.all_gather_async(
            x,
            dim=3,
            persistent_output_buffer=self.collective_buffers.acquire("gather", x, output_memory) if decode else None,
            multi_device_global_semaphore=self.ccl.get_and_cycle_ag_semaphore_handles(),
            barrier_semaphore=self.ccl.get_and_cycle_barrier_semaphore_handle(),
            num_links=self.num_links,
            topology=ttnn.Topology.Ring,
            memory_config=output_memory,
            chunks_per_sync=10,
            num_workers_per_link=1 if decode else 2,
            num_buffers_per_channel=2,
        )
        return ttnn.typecast(result, self.gather_output_dtype) if result.dtype != self.gather_output_dtype else result

    def _reduce(self, x):
        decode = self.residual_sharded and self.policy.dram and x.shape[2] <= 32
        memory = ttnn.L1_MEMORY_CONFIG if x.shape[2] <= 32 else ttnn.DRAM_MEMORY_CONFIG
        if not decode:
            x = ttnn.to_memory_config(x, memory)
        if x.dtype != self.ccl_dtype:
            x = ttnn.typecast(x, self.ccl_dtype)
        result = ttnn.experimental.reduce_scatter_minimal_async(
            x,
            dim=3,
            persistent_output_buffers=(
                self.collective_buffers.acquire("reduce", x, self.decode_residual_memory) if decode else None
            ),
            multi_device_global_semaphore=self.ccl.get_and_cycle_rs_semaphore_handles(),
            barrier_semaphore=self.ccl.get_and_cycle_barrier_semaphore_handle(),
            num_links=self.num_links,
            topology=ttnn.Topology.Ring,
            memory_config=self.decode_residual_memory if decode else memory,
            intermediate_memory_config=ttnn.DRAM_MEMORY_CONFIG,
            chunks_per_sync=10,
            num_workers_per_link=1 if decode else 2,
            num_buffers_per_channel=2,
        )
        if result.dtype != ttnn.bfloat16:
            result = ttnn.typecast(result, ttnn.bfloat16)
        return result if self.residual_sharded else self._gather(result)

    def _norm(self, x):
        if not self.residual_sharded:
            return super()._norm(x)
        if x.shape[2] <= 32:
            normalized = ttnn.rms_norm(
                ttnn.to_memory_config(x, self.decode_residual_memory),
                epsilon=self.eps,
                program_config=self.decode_norm_program,
                compute_kernel_config=self.compute,
            )
            return self._gather(normalized)
        # K2 uses four independent contiguous RMS groups, NOT a single global
        # RMS. Each TP rank owns exactly one full group: stats stay local.
        x = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG if x.shape[2] <= 32 else ttnn.DRAM_MEMORY_CONFIG)
        normalized = ttnn.rms_norm(x, epsilon=self.eps, compute_kernel_config=self.compute)
        return normalized if self.prefill_agmm and x.shape[2] >= 256 else self._gather(normalized)

    def _prefill_fused_matmul_config(self, x, *, swiglu):
        return ttnn.MinimalMatmulConfig(
            M_block_size=8,
            K_block_size=8,
            N_block_size=8,
            subblock_h=2,
            subblock_w=2 if self.policy.prefill_fp32 else 4,
            compute_with_storage_grid_size=ttnn.CoreCoord(11, 9),
        )

    def _fused_prefill_projection(self, x, w, *, swiglu=False):
        # The local norm group feeds a fused ring gather + column projection.
        # The persistent gather and local input must have the same format;
        # native AGMM reads both through the gather tensor's tile format.
        # Keep one shape/dtype at a time, bounded by the maximum chunk size.
        # Residual/output stay BF16; decode retains separate DRAM matmuls.
        dtype = self.prefill_mlp_dtype if swiglu else self.prefill_qkv_dtype
        shape = list(x.shape)
        shape[-1] = 4096
        if self._prefill_gather_shape != (tuple(shape), dtype):
            self._prefill_gather_buffer = ttnn.empty(
                shape,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            self._prefill_gather_shape = (tuple(shape), dtype)
        return ttnn.experimental.all_gather_minimal_matmul_async(
            ttnn.typecast(x, dtype),
            w,
            config=self._prefill_fused_matmul_config(x, swiglu=swiglu),
            multi_device_global_semaphore=self.ccl.get_and_cycle_ag_semaphore_handles(),
            topology=ttnn.Topology.Ring,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=getattr(self, "matmul_output_dtype", ttnn.bfloat16),
            compute_kernel_config=self.role_compute["mlp" if swiglu else "attention"],
            persistent_output_buffer=self._prefill_gather_buffer,
            num_links=self.num_links,
            barrier_semaphore=self.ccl.get_and_cycle_barrier_semaphore_handle(),
            force_transpose=True,
            num_workers_per_link=math.ceil(11 / self.num_links),
            num_buffers_per_channel=24,
            fuse_swiglu=swiglu,
        )[0]

    def _prefill_matmul_config(self):
        return ttnn.MinimalMatmulConfig(
            M_block_size=self.policy.prefill_m_block,
            K_block_size=8,
            N_block_size=16,
            subblock_h=2,
            subblock_w=2 if self.policy.prefill_fp32 else 4,
            compute_with_storage_grid_size=ttnn.CoreCoord(11, 10),
        )

    def _linear(self, x, w):
        if self.prefill_agmm and x.shape[2] >= 256 and w is self.wqkv:
            return self._fused_prefill_projection(x, w)
        return super()._linear(x, w)

    def _project_reduce(self, x, w, decode):
        return self._reduce((self._decode_linear if decode else self._linear)(x, w))

    def _prefill_swiglu(self, normed):
        if self.prefill_agmm:
            return self._fused_prefill_projection(normed, self.wswiglu, swiglu=True)
        return ttnn.experimental.minimal_matmul(
            normed,
            self.wswiglu,
            config=self._prefill_matmul_config(),
            compute_kernel_config=self.role_compute["mlp"],
            dtype=getattr(self, "matmul_output_dtype", ttnn.bfloat16),
            fuse_swiglu=True,
        )

    def _mlp(self, normed, rows, decode):
        linear = self._decode_linear if decode else self._linear
        if rows >= 256 and self.policy.prefill_fused:
            mlp = self._prefill_swiglu(normed)
        else:
            if self.policy.packed_mlp:
                packed = ttnn.to_memory_config(
                    linear(normed, self.wgateup), ttnn.L1_MEMORY_CONFIG if decode else ttnn.DRAM_MEMORY_CONFIG
                )
                gate, up = packed[..., :3072], packed[..., 3072:]
            else:
                if decode:
                    normed = ttnn.to_memory_config(normed, self.decode_inputs[id(self.wgate)])
                gate, up = linear(normed, self.wgate), linear(normed, self.wup)
            mlp = ttnn.multiply(
                gate,
                up,
                input_tensor_a_activations=(
                    [] if decode and self.policy.fused_gate and not self.policy.packed_mlp else [ttnn.UnaryOpType.SILU]
                ),
                memory_config=gate.memory_config(),
            )
        return mlp

    def _finish(self, x, attention):
        decode = self.policy.dram and x.shape[2] <= 32
        projected = self._project_reduce(attention, self.wo, decode)
        residual = ttnn.add(ttnn.to_memory_config(x, projected.memory_config()), projected, dtype=self.residual_dtype)
        mlp = self._mlp(self._norm(residual), x.shape[2], decode)
        projected = self._project_reduce(mlp, self.wdown, decode)
        return ttnn.add(
            ttnn.to_memory_config(residual, projected.memory_config()), projected, dtype=self.residual_dtype
        )

    def decode_forward(self, x, *, rope, kv_cache, page_table, current_pos):
        """One device-only token pass; tensor positions/page tables can change on replay."""
        batch = x.shape[2]
        width = 1024 if self.residual_sharded else 4096
        if tuple(x.shape)[:2] != (1, 1) or not 1 <= batch <= 32 or x.shape[-1] != width:
            raise ValueError(f"Decode expects local [1,1,B,{width}], 1 <= B <= 32")
        if self.residual_sharded:
            x = ttnn.to_memory_config(x, self.decode_residual_memory)
        normed = self._norm(x)
        if self.policy.dram:
            qkv = self._decode_linear(normed, self.wqkv)
        elif batch == 1:
            # Produce the head splitter's input directly in L1. 96 x 64-wide
            # shards divide head_dim128, and overlapping heads preserve the
            # rotary kernel's origin-prefix shard-grid contract.
            qkv = ttnn.linear(
                normed,
                self.wqkv,
                dtype=getattr(self, "matmul_output_dtype", ttnn.bfloat16),
                memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                compute_kernel_config=self.role_compute["attention"],
                program_config=ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                    compute_with_storage_grid_size=(11, 9),
                    in0_block_w=4,
                    out_subblock_h=1,
                    out_subblock_w=2,
                    per_core_M=1,
                    per_core_N=2,
                    fuse_batch=True,
                    mcast_in0=True,
                ),
            )
        else:
            qkv = self._linear(normed, self.wqkv)
        q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
            qkv,
            num_heads=8,
            num_kv_heads=2,
            overlap_qk_coregrid=True,
            memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
        )
        q, k = self._decode_qk(q, k, rope)
        # Decode RoPE promotes the padded head tile to a logical32-row tensor.
        # SDPA derives GQA grouping from logical Q heads, so restore8 before it.
        q = q[:, :, : self.local_q_heads, :]
        grid = ttnn.num_cores_to_corerangeset(batch, ttnn.CoreCoord(8, 8), row_wise=True)
        shard = ttnn.create_sharded_memory_config(
            (32, 128),
            core_grid=grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        k = ttnn.to_memory_config(k, shard)
        self._update_cache(kv_cache, k, v, current_pos, page_table)
        # SDPA caps each head/batch group at16 workers by default. Keep the
        # established <=4096-token recurrence bound after TP reduces KV heads.
        cores_per_kv_head = min(16, max(1, (64 // batch) // 2))
        if page_table.shape[1] * self.page_size <= 4096 * cores_per_kv_head:
            # Bound each worker's reduction, including batch-dependent split-K.
            # Capacity is fixed across mutable positions in a captured trace.
            # The K128 reader fetches whole four-page chunks before causal
            # masking. Give every extra lookup a valid request-owned page;
            # allocator padding is not initialized page-table data. This stays
            # on device so mutable tables and different positions replay safely.
            padding = (-page_table.shape[1]) % 4
            attention_table = (
                ttnn.concat([page_table, ttnn.repeat(page_table[:, -1:], (1, padding))], dim=1)
                if padding
                else page_table
            )
            attention = ttnn.transformer.paged_scaled_dot_product_attention_decode(
                q,
                kv_cache[0],
                kv_cache[1],
                page_table_tensor=attention_table,
                cur_pos_tensor=current_pos,
                compute_kernel_config=self.attention_compute,
                program_config=ttnn.SDPAProgramConfig(
                    compute_with_storage_grid_size=(11, 10), q_chunk_size=0, k_chunk_size=128, exp_approx_mode=False
                ),
            )
        elif self.accurate_decode_kernel == "flash":
            attention = self._flash_decode_attention(q, kv_cache, page_table, current_pos)
        else:
            attention = self._accurate_decode_attention(q, kv_cache, page_table, current_pos)
        # [1,B,H,D] is already in concatenated logical order. Tile reshape
        # avoids the dedicated concat op's compulsory L1 sharding roundtrip.
        attention = ttnn.reshape(attention, (1, 1, batch, 1024))
        return self._finish(x, attention)

    def _flash_decode_attention(self, q, kv_cache, page_table, current_pos):
        """Stock split-K decode contract with FP32 scores/recurrence and accurate exp.

        All 110 workers split each KV head's history and merge through the stock tree
        (B1: 55 per KV head). The chunked-prefill fallback ran one query head per core,
        8 of 64 cores at B1, serially over the whole context.
        """
        # The reader fetches whole K chunks before causal masking; pad with a valid
        # request-owned page, as the stock branch does for K128.
        padding = (-page_table.shape[1]) % (self.flash_decode_k_chunk // self.page_size)
        table = (
            ttnn.concat([page_table, ttnn.repeat(page_table[:, -1:], (1, padding))], dim=1) if padding else page_table
        )
        return self._flash_decode(
            q,
            kv_cache[0],
            kv_cache[1],
            table,
            current_pos,
            max_cores_per_head=self.flash_decode_max_cores_per_head,
            k_chunk_size=self.flash_decode_k_chunk,
        )

    def _accurate_decode_attention(self, q, kv_cache, page_table, current_pos):
        """Pack the four query heads sharing each KV head into independent rows."""
        q = ttnn.to_memory_config(q, ttnn.DRAM_MEMORY_CONFIG)
        batch = q.shape[1]
        per_group = []
        start = 0
        while start < batch:
            count = 1 << (min(self.accurate_attention_batch_size, batch - start).bit_length() - 1)
            if count == 1:
                # Retain the original scalar path and its eight separate Q heads.
                pos = ttnn.to_layout(ttnn.reshape(current_pos[start : start + 1], (1, 1, 1, 1)), ttnn.TILE_LAYOUT)
                pos = ttnn.maximum(pos, 0)
                offset = ttnn.reshape(ttnn.to_layout(ttnn.bitwise_and(pos, -32), ttnn.ROW_MAJOR_LAYOUT), (1,))
                index = ttnn.repeat(ttnn.typecast(ttnn.bitwise_and(pos, 31), ttnn.uint32), (1, 8, 1, 128))
                query = ttnn.repeat(ttnn.permute(q[:, start : start + 1, :, :], (0, 2, 1, 3)), (1, 1, 32, 1))
                offset_args = {"chunk_start_idx_tensor": offset}
            else:
                if q.dtype != ttnn.bfloat16:
                    raise ValueError("Packed accurate attention requires the selected BF16 queries without a cast")
                # The unchanged reader/compute use floor(position / 32) for the K
                # extent. The private writer also needs position % 32, so retain
                # each full clamped position in its distinct scalar buffer owner.
                pos = ttnn.to_layout(
                    ttnn.reshape(current_pos[start : start + count], (1, 1, 1, count)), ttnn.TILE_LAYOUT
                )
                pos = ttnn.maximum(pos, 0)
                positions = ttnn.reshape(ttnn.to_layout(pos, ttnn.ROW_MAJOR_LAYOUT), (count,))
                offsets = [positions[row : row + 1] for row in range(count)]
                offset_args = {"chunk_start_idx_tensors": offsets, "packed_gqa": True}
                # Local heads 0..3 share KV head 0 and heads 4..7 share KV head 1.
                # Explicit row-major reshapes and padding preserve every BF16 bit.
                query = ttnn.to_layout(q[:, start : start + count, :, :], ttnn.ROW_MAJOR_LAYOUT)
                query = ttnn.reshape(query, (count, 2, 4, 128))
                query = ttnn.pad(query, [(0, 0), (0, 0), (0, 28), (0, 0)], 0)
                query = ttnn.to_layout(query, ttnn.TILE_LAYOUT)
            table = page_table[start : start + count, :]
            padding = (-table.shape[1]) % 8
            if padding:
                table = ttnn.concat([table, ttnn.repeat(table[:, -1:], (1, padding))], dim=1)
            attended = self._attention(
                query,
                kv_cache[0],
                kv_cache[1],
                table,
                q_chunk_size=32,
                k_chunk_size=128,
                **offset_args,
            )
            if count == 1:
                output = ttnn.permute(ttnn.gather(attended, 2, index), (0, 2, 1, 3))
            else:
                # Discard zero-query rows and restore KV-major, query-head-major
                # order before the existing output projection sees its eight heads.
                output = ttnn.to_layout(attended[:, :, :4, :], ttnn.ROW_MAJOR_LAYOUT)
                output = ttnn.reshape(output, (1, count, 8, 128))
                output = ttnn.to_layout(output, ttnn.TILE_LAYOUT)
            per_group.append(output)
            start += count
        return ttnn.concat(per_group, dim=1) if len(per_group) > 1 else per_group[0]

    def _prefill_chunk(self, x, *, rope, kv_cache, page_table, start_pos):
        logical = x.shape[2]
        physical = (logical + 31) // 32 * 32
        if physical != logical:
            x = ttnn.pad(x, [(0, 0), (0, 0), (0, physical - logical), (0, 0)], 0)
            rope = tuple(ttnn.pad(r, [(0, 0), (0, 0), (0, physical - logical), (0, 0)], 0) for r in rope)
        qkv = self._linear(self._norm(x), self.wqkv)
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            qkv, num_heads=8, num_kv_heads=2, transpose_k_heads=False, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        q, k = self._rope(q, rope), self._rope(k, rope)
        chunk_table = page_table[:, start_pos // 32 : (start_pos + physical) // 32]
        for cache, update in zip(kv_cache, (k, v)):
            ttnn.experimental.paged_fill_cache(
                cache,
                ttnn.typecast(update, cache.dtype) if update.dtype != cache.dtype else update,
                chunk_table,
                batch_idx=0,
            )
        sdpa_chunk = (
            self.prefill_q_chunk
            if physical % self.prefill_q_chunk == 0 and start_pos % self.prefill_q_chunk == 0
            else 32
        )
        fast_k_chunk = (
            self.prefill_k_chunk
            if physical >= self.prefill_k_chunk and start_pos % self.prefill_k_chunk == 0
            else sdpa_chunk
        )
        if self.policy.fast_prefill and start_pos + physical <= fast_k_chunk * 256:
            # Bound the stock numerator recurrence to 256 K blocks. Real-weight
            # controls cover K32/K128/K256 at 8192/32768/65536 tokens respectively.
            attention = ttnn.transformer.chunked_scaled_dot_product_attention(
                q,
                kv_cache[0],
                kv_cache[1],
                page_table,
                chunk_start_idx=start_pos,
                compute_kernel_config=self.attention_compute,
                program_config=ttnn.SDPAProgramConfig(
                    compute_with_storage_grid_size=(11, 10),
                    q_chunk_size=sdpa_chunk,
                    k_chunk_size=fast_k_chunk,
                    exp_approx_mode=False,
                ),
            )
        else:
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
        # Prefill assembles chunks and decode-assisted prefix tokens in one public
        # interleaved layout, including a short tail that uses sharded matmuls.
        return ttnn.to_memory_config(self._finish(x, attention), ttnn.DRAM_MEMORY_CONFIG)[:, :, :logical, :]
