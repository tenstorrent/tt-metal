"""Optimized single-chip IFM/K2-Horizon-7B dense decoder.

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

import math
from dataclasses import dataclass

import ttnn
from models.common.lightweightmodule import LightweightModule


@dataclass
class PrefillPlan:
    """Setup-time metadata for unaligned prefix continuation."""

    start_pos: int
    seq_len: int
    leading_positions: tuple


@dataclass(frozen=True)
class MatmulGeometry:
    cores: int = 16
    block_w: int = 8
    readers: int = 1
    fp32: bool = True


@dataclass(frozen=True)
class PrecisionPolicy:
    """Setup-time tensor-group policy, independently measured by operation role."""

    attention: str = "bfloat4_b"
    mlp: str = "bfloat4_b"
    down: str = "bfloat4_b"
    attention_fidelity: str = "LoFi"
    mlp_fidelity: str = "LoFi"
    down_fidelity: str = "LoFi"
    kv: str = "bfloat8_b"
    packed_mlp: bool = False
    dram: bool = True
    qkv_geometry: MatmulGeometry = MatmulGeometry(64, 8, 2, False)
    o_geometry: MatmulGeometry = MatmulGeometry(64, 8, 2, False)
    mlp_geometry: MatmulGeometry = MatmulGeometry(64, 4, 2, False)
    down_geometry: MatmulGeometry = MatmulGeometry(64, 12, 2, False)
    fused_gate: bool = False
    norm_layout: str = "l1"
    prefill_fp32: bool = False
    prefill_fused: bool = True
    fast_prefill: bool = True
    attention_activation: str = "bfloat16"
    mlp_activation: str = "bfloat16"
    # Optional TP4 overrides let full-model accuracy repairs distinguish the
    # column-parallel QKV operand from the row-parallel output projection.
    qkv_dtype: str | None = None
    o_dtype: str | None = None
    qkv_fidelity: str | None = None
    o_fidelity: str | None = None
    sdpa_fidelity: str = "LoFi"
    prefill_qkv_activation: str = "bfloat8_b"
    prefill_mlp_activation: str = "bfloat8_b"
    prefill_m_block: int = 8


class OptimizedDecoder(LightweightModule):
    page_size = 32
    chunk_size = 4096

    @classmethod
    def from_state_dict(cls, state_dict, *, hf_config, layer_idx, mesh_device, policy=None):
        import torch

        from .accurate_attention import _load, accurate_attention

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
        self.policy = policy or PrecisionPolicy()
        self.kv_dtype = getattr(ttnn, self.policy.kv)
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

        self.decode_weights = {}
        self.decode_programs = {}
        self.decode_inputs = {}
        self.decode_computes = {}

        def device(tensor, dtype, role):
            result = ttnn.from_torch(
                tensor.contiguous(),
                device=mesh_device,
                dtype=getattr(ttnn, dtype),
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
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
                self.decode_weights[id(result)] = ttnn.from_torch(
                    tensor.contiguous(),
                    device=mesh_device,
                    dtype=getattr(ttnn, dtype),
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=wm,
                )
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
                self.decode_computes[id(result)] = ttnn.init_device_compute_kernel_config(
                    mesh_device.arch(),
                    math_fidelity=getattr(ttnn.MathFidelity, getattr(self.policy, group + "_fidelity")),
                    math_approx_mode=False,
                    fp32_dest_acc_en=g.fp32,
                    packer_l1_acc=True,
                )
            return result

        # Fold the norm affine into each consumer's input channels once at setup.
        # Multiply BF16 checkpoint values in FP32, then store BF16 fused weights.
        gamma1 = weight("input_layernorm.weight").float()
        gamma2 = weight("post_attention_layernorm.weight").float()

        def folded(name, gamma):
            return (weight(name).float().T * gamma[:, None]).bfloat16()

        self.wqkv = device(
            torch.cat([folded(f"self_attn.{p}_proj.weight", gamma1) for p in "qkv"], dim=-1),
            self.policy.attention,
            "qkv",
        )
        self.wo = device(weight("self_attn.o_proj.weight").T, self.policy.attention, "o")
        gate, up = folded("mlp.gate_proj.weight", gamma2), folded("mlp.up_proj.weight", gamma2)
        self.wgate = device(gate, self.policy.mlp, "mlp")
        self.wup = device(up, self.policy.mlp, "mlp")
        self.wdown = device(weight("mlp.down_proj.weight").T, self.policy.down, "down")
        # Packed and separate are compared as complete paths. The selected separate
        # path avoids split/layout costs, so it does not retain unused packed weights.
        self.wgateup = device(torch.cat([gate, up], dim=-1), self.policy.mlp, "mlp") if self.policy.packed_mlp else None
        if self.policy.prefill_fused:
            packed = torch.stack([w.reshape(4096, 384, 32) for w in (gate, up)], dim=2).reshape(4096, 24576)
            self.wswiglu = ttnn.from_torch(
                packed,
                device=mesh_device,
                dtype=getattr(ttnn, self.policy.mlp),
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        self.role_compute = {}
        for role in ("attention", "mlp", "down"):
            fidelity = getattr(ttnn.MathFidelity, getattr(self.policy, role + "_fidelity"))
            self.role_compute[role] = ttnn.init_device_compute_kernel_config(
                mesh_device.arch(),
                math_fidelity=fidelity,
                math_approx_mode=False,
                fp32_dest_acc_en=self.policy.prefill_fp32,
                packer_l1_acc=True,
            )
        if self.policy.dram and self.policy.fused_gate:
            self.decode_programs[id(self.wgate)].fused_activation = ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU)
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

    def _weight_compute(self, w):
        role = "attention" if w is self.wqkv or w is self.wo else "down" if w is self.wdown else "mlp"
        return self.role_compute[role]

    def _prefill_matmul_config(self):
        return ttnn.MinimalMatmulConfig(
            M_block_size=8,
            K_block_size=8,
            N_block_size=16,
            subblock_h=2,
            subblock_w=2 if self.policy.prefill_fp32 else 4,
            compute_with_storage_grid_size=ttnn.CoreCoord(11, 10),
        )

    def _linear(self, x, w):
        if x.shape[2] >= 256:
            return ttnn.experimental.minimal_matmul(
                x,
                w,
                config=self._prefill_matmul_config(),
                compute_kernel_config=self._weight_compute(w),
                dtype=getattr(self, "matmul_output_dtype", ttnn.bfloat16),
            )
        return ttnn.linear(
            x,
            w,
            dtype=getattr(self, "matmul_output_dtype", ttnn.bfloat16),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self._weight_compute(w),
        )

    def _decode_linear(self, x, w):
        activation = self.policy.attention_activation if w is self.wqkv or w is self.wo else self.policy.mlp_activation
        if x.dtype != getattr(ttnn, activation):
            x = ttnn.typecast(x, getattr(ttnn, activation))
        x = ttnn.to_memory_config(x, self.decode_inputs[id(w)])
        result = ttnn.linear(
            x,
            self.decode_weights[id(w)],
            dtype=getattr(self, "matmul_output_dtype", ttnn.bfloat16),
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            program_config=self.decode_programs[id(w)],
            compute_kernel_config=self.decode_computes[id(w)],
        )
        if result.shape[-1] != w.shape[-1]:
            result = ttnn.to_memory_config(result, ttnn.L1_MEMORY_CONFIG)[..., : w.shape[-1]]
        return result

    def _norm(self, x):
        if x.is_sharded():
            x = ttnn.to_memory_config(
                x, ttnn.L1_MEMORY_CONFIG if self.policy.norm_layout != "dram" else ttnn.DRAM_MEMORY_CONFIG
            )
        if x.shape[2] <= 32:
            # Preserve contiguous groups as independent rows. Logical volume is
            # essential for padded decode batches, especially B1.
            shape = x.shape
            grouped = ttnn.reshape(
                x,
                (1, 1, math.prod(shape) // 1024, 1024),
                memory_config=ttnn.L1_MEMORY_CONFIG if self.policy.norm_layout == "l1" else ttnn.DRAM_MEMORY_CONFIG,
            )
            grouped = ttnn.rms_norm(grouped, epsilon=self.eps, compute_kernel_config=self.compute)
            return ttnn.reshape(grouped, shape)
        groups = [
            ttnn.rms_norm(x[..., i : i + 1024], epsilon=self.eps, compute_kernel_config=self.compute)
            for i in range(0, 4096, 1024)
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
        if q.shape[1] == 1:
            q = ttnn.experimental.rotary_embedding(q, rope[0], rope[1], token_index=0, memory_config=q.memory_config())
            k = ttnn.experimental.rotary_embedding(k, rope[0], rope[1], token_index=0, memory_config=k.memory_config())
            return q, k
        if q.shape[1] >= 4:
            # Use request index as the rotary sequence axis. Each request keeps
            # its own angle, broadcast over heads, in two rotary launches.
            batch = q.shape[1]
            rr = tuple(ttnn.reshape(r[:, :, :1, :], (1, 1, batch, 128)) for r in rope)
            outputs = []
            for t in (q, k):
                t = ttnn.permute(ttnn.to_memory_config(t, ttnn.DRAM_MEMORY_CONFIG), (0, 2, 1, 3))
                t = ttnn.experimental.rotary_embedding(t, rr[0], rr[1])[:, :, :batch, :]
                outputs.append(ttnn.permute(t, (0, 2, 1, 3)))
            return tuple(outputs)
        return (
            self._decode_rope(ttnn.to_memory_config(q, ttnn.DRAM_MEMORY_CONFIG), rope),
            self._decode_rope(ttnn.to_memory_config(k, ttnn.DRAM_MEMORY_CONFIG), rope),
        )

    def _update_cache(self, kv_cache, k, v, current_pos, page_table):
        batch = k.shape[1]
        grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))})
        available = grid.subtract(k.memory_config().shard_spec.grid)
        vg = ttnn.num_cores_to_corerangeset_in_subcoregrids(available.ranges()[0].start, batch, available, True)
        vs = ttnn.create_sharded_memory_config(
            (32, 128),
            core_grid=vg,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        v = ttnn.to_memory_config(v, vs)
        ttnn.experimental.paged_fused_update_cache(
            kv_cache[0], k, kv_cache[1], v, update_idxs_tensor=current_pos, page_table=page_table
        )

    def _finish(self, x, attention):
        if self.policy.dram and x.shape[2] <= 32:
            projected = self._decode_linear(attention, self.wo)
            residual = ttnn.add(
                ttnn.to_memory_config(x, projected.memory_config()), projected, memory_config=projected.memory_config()
            )
            normed = self._norm(residual)
            if self.policy.packed_mlp:
                packed = self._decode_linear(normed, self.wgateup)
                packed = ttnn.to_memory_config(packed, ttnn.L1_MEMORY_CONFIG)
                gate, up = packed[..., :12288], packed[..., 12288:]
            else:
                # One shared working shard feeds both projections. Each call's
                # to_memory_config is then a no-op for the selected equal geometry.
                normed = ttnn.to_memory_config(normed, self.decode_inputs[id(self.wgate)])
                gate, up = self._decode_linear(normed, self.wgate), self._decode_linear(normed, self.wup)
            mlp = ttnn.multiply(
                gate,
                up,
                input_tensor_a_activations=(
                    [] if self.policy.fused_gate and not self.policy.packed_mlp else [ttnn.UnaryOpType.SILU]
                ),
                memory_config=gate.memory_config(),
            )
            projected = self._decode_linear(mlp, self.wdown)
            return ttnn.add(
                ttnn.to_memory_config(residual, projected.memory_config()),
                projected,
                memory_config=projected.memory_config(),
            )
        # A single logical row is a legal dynamic linear bias. Multiple rows
        # need elementwise residual addition rather than broadcast bias.
        single = x.shape[2] == 1
        residual = (
            ttnn.linear(
                attention,
                self.wo,
                bias=x,
                dtype=getattr(self, "matmul_output_dtype", ttnn.bfloat16),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=self.role_compute["attention"],
            )
            if single
            else ttnn.add(x, self._linear(attention, self.wo))
        )
        normed = self._norm(residual)
        if x.shape[2] >= 256 and self.policy.prefill_fused:
            mlp = ttnn.experimental.minimal_matmul(
                normed,
                self.wswiglu,
                config=self._prefill_matmul_config(),
                compute_kernel_config=self.role_compute["mlp"],
                dtype=getattr(self, "matmul_output_dtype", ttnn.bfloat16),
                fuse_swiglu=True,
            )
            return ttnn.add(residual, self._linear(mlp, self.wdown))
        if x.shape[2] <= 32 and self.policy.packed_mlp:
            packed = self._linear(normed, self.wgateup)
            gate, up = packed[..., :12288], packed[..., 12288:]
        else:
            gate, up = self._linear(normed, self.wgate), self._linear(normed, self.wup)
        mlp = ttnn.multiply(gate, up, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
        if single:
            return ttnn.linear(
                mlp,
                self.wdown,
                bias=residual,
                dtype=getattr(self, "matmul_output_dtype", ttnn.bfloat16),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=self.role_compute["down"],
            )
        return ttnn.add(residual, self._linear(mlp, self.wdown))

    def decode_forward(self, x, *, rope, kv_cache, page_table, current_pos):
        """One device-only token pass; tensor positions/page tables can change on replay."""
        batch = x.shape[2]
        if tuple(x.shape)[:2] != (1, 1) or not 1 <= batch <= 32:
            raise ValueError("Decode expects [1,1,B,4096], 1 <= B <= 32")
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
            num_heads=32,
            num_kv_heads=8,
            overlap_qk_coregrid=True,
            memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
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
        self._update_cache(kv_cache, k, v, current_pos, page_table)
        cores_per_kv_head = max(1, (64 // batch) // 8)
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
                compute_kernel_config=self.compute,
                program_config=ttnn.SDPAProgramConfig(
                    compute_with_storage_grid_size=(11, 10), q_chunk_size=0, k_chunk_size=128, exp_approx_mode=False
                ),
            )
        else:
            q = ttnn.to_memory_config(q, ttnn.DRAM_MEMORY_CONFIG)
            # Preserve FP32 long-context recurrence for decode as well as prefill.
            # All position arithmetic and row selection stays on device. Repeated Q
            # rows cover the current 32-token page; only the current row is returned.
            per_request = []
            for b in range(batch):
                pos = ttnn.to_layout(ttnn.reshape(current_pos[b : b + 1], (1, 1, 1, 1)), ttnn.TILE_LAYOUT)
                # Cache writes skip inactive (-1) slots; the attention reader's
                # uint32 offset must still be valid for their discarded output.
                pos = ttnn.maximum(pos, 0)
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
                    query,
                    kv_cache[0],
                    kv_cache[1],
                    table,
                    chunk_start_idx_tensor=offset,
                    q_chunk_size=32,
                    k_chunk_size=128,
                )
                per_request.append(ttnn.permute(ttnn.gather(attended, 2, index), (0, 2, 1, 3)))
            attention = ttnn.concat(per_request, dim=1) if batch > 1 else per_request[0]
        # [1,B,H,D] is already in concatenated logical order. Tile reshape
        # avoids the dedicated concat op's compulsory L1 sharding roundtrip.
        attention = ttnn.reshape(attention, (1, 1, batch, 4096))
        return self._finish(x, attention)

    def _prefill_chunk(self, x, *, rope, kv_cache, page_table, start_pos):
        logical = x.shape[2]
        physical = (logical + 31) // 32 * 32
        if physical != logical:
            x = ttnn.pad(x, [(0, 0), (0, 0), (0, physical - logical), (0, 0)], 0)
            rope = tuple(ttnn.pad(r, [(0, 0), (0, 0), (0, physical - logical), (0, 0)], 0) for r in rope)
        qkv = self._linear(self._norm(x), self.wqkv)
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            qkv, num_heads=32, num_kv_heads=8, transpose_k_heads=False, memory_config=ttnn.DRAM_MEMORY_CONFIG
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
        sdpa_chunk = 128 if physical % 128 == 0 and start_pos % 128 == 0 else 32
        fast_k_chunk = 256 if physical >= 256 and start_pos % 256 == 0 else sdpa_chunk
        if self.policy.fast_prefill and start_pos + physical <= fast_k_chunk * 256:
            # Bound the stock numerator recurrence to 256 K blocks. Real-weight
            # controls cover K32/K128/K256 at 8192/32768/65536 tokens respectively.
            attention = ttnn.transformer.chunked_scaled_dot_product_attention(
                q,
                kv_cache[0],
                kv_cache[1],
                page_table,
                chunk_start_idx=start_pos,
                compute_kernel_config=self.compute,
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
                    ttnn.to_memory_config(
                        self.decode_forward(
                            x[:, b : b + 1, offset : offset + 1, :],
                            rope=one_rope,
                            kv_cache=kv_cache,
                            page_table=table,
                            current_pos=position,
                        ),
                        ttnn.DRAM_MEMORY_CONFIG,
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
