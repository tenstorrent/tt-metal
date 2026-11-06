# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Muse-Glimmer text decoder, single-chip optimized implementation.

Setup loads an HF layer-local state dict (or the canonical checkpoint prefix).
Prefill takes [batch, 1, logical_sequence, 6656] tiled device activations and
returns the same shape. Each batch row names a page-table slot via ``slots``.
``start_pos`` is the absolute cached prefix length (append-only); fresh requests
use zero. Arbitrary logical lengths are padded internally and processed in
chunks. The caller owns page allocation; pages must be disjoint across requests.

Decode takes [1, 1, batch, 6656], a device int32 current-position vector [batch],
a page table [batch, context/32], and optional RoPE tensors [1,batch,1,128].
Positions are the slots to WRITE, and attention includes those slots. The entire
method is device-only and can be captured/replayed. Refresh these persistent
inputs before replay. The caller prepares all eager/trace variants before capture.

Caches are BFP8 [physical_pages,2,32,128] for both layer kinds. Sliding attention
masks to the HF window without reducing the advertised absolute context.
"""

import math

import ttnn
from models.common.lightweightmodule import LightweightModule


class OptimizedDecoder(LightweightModule):
    # Measured single-chip Blackhole policy; see doc/optimized_decoder.
    POLICY = {
        "sharded_norm": True,
        "dram_sharded": True,
        "block": 2,
        "fp32": False,
        "mlp_dtype": "bfloat4_b",
        "mlp_fidelity": "LoFi",
        "attention_dtype": "bfloat4_b",
        "attention_fidelity": "LoFi",
        "kv_dtype": "bfloat8_b",
        "grid": [8, 8],
        "readers": 2,
        "qkv.block": 4,
        "self_attn.gate_proj.block": 8,
        "self_attn.o_proj.block": 8,
        "mlp.down_proj.block": 6,
        "fuse_silu": False,
        "prefill_interleaved": True,
        "pack_mlp": True,
        "pack_attention_gate": True,
        "sdpa_k": 128,
        "mlp.gate_up.block": 2,
        "prefill_kernel": "minimal",
        "minimal_mlp": "prefill",
        "keep_residual": True,
        "keep_attn_l1": True,
        "prefill_sdpa_q": 256,
        "prefill_sdpa_k": 256,
        "sharded_heads": True,
        "aligned_heads": True,
        "mlp_act_dtype": "bfloat8_b",
        "minimal_n": 32,
        "sdpa_fidelity": "HiFi2",
        "chunk_size": 4096,
        "qk_norm_l1": True,
        "release_decode": True,
        "short_prefill_dram": True,
    }
    PAGE_SIZE = 32
    CHUNK_SIZE = 1024

    @classmethod
    def from_state_dict(cls, state_dict, *, hf_config, layer_idx, mesh_device):
        import torch

        self = cls()
        self.policy = dict(cls.POLICY)
        self.CHUNK_SIZE = self.policy.get("chunk_size", cls.CHUNK_SIZE)
        self.kv_dtype = getattr(ttnn, self.policy.get("kv_dtype", "bfloat16"))
        c = getattr(hf_config, "text_config", hf_config)
        if (c.hidden_size, c.intermediate_size, c.num_attention_heads, c.num_key_value_heads, c.head_dim) != (
            6656,
            19968,
            32,
            2,
            128,
        ):
            raise ValueError("This implementation requires the real Muse-Glimmer-30B text config")
        if c.attention_bias or c.hidden_activation != "silu":
            raise ValueError("Unsupported attention bias / MLP activation")
        self.config, self.device, self.layer_idx = (c, mesh_device, layer_idx)
        # noconvert binding requires a Python float exactly representable in float32.
        self.attention_scale = torch.tensor(128 ** (-0.5) * c.qk_scale_factor, dtype=torch.float32).item()
        self.kind = c.layer_types[layer_idx]
        self.window = c.sliding_window if self.kind == "sliding_attention" else None
        self.use_rope = c.layer_rope_theta[layer_idx] != 0
        self.ck = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        self.attention_ck = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=getattr(ttnn.MathFidelity, self.policy.get("sdpa_fidelity", "HiFi4")),
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        self.sdpa = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(*self.policy.get("prefill_sdpa_grid", (8, 8))),
            q_chunk_size=self.policy.get("prefill_sdpa_q", 32),
            k_chunk_size=self.policy.get("prefill_sdpa_k", 32),
            exp_approx_mode=False,
        )
        self.weights = {}
        self.mm_ck = {}
        self.norm_weights = {}
        self.decode_mem = {}
        self.decode_pc = {}
        self.dram_sharded = self.policy.get("dram_sharded", False)
        self.sharded_norm = self.policy.get("sharded_norm", False)
        self.grid = tuple(self.policy.get("grid", (8, 2)))
        self.core_grid = ttnn.CoreGrid(x=self.grid[0], y=self.grid[1])
        self.residual_mem = self._width_mem(6656, exact=True)
        self.sdpa_decode = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(*self.policy.get("sdpa_grid", (8, 8))),
            q_chunk_size=0,
            k_chunk_size=self.policy.get("sdpa_k", 32),
            exp_approx_mode=False,
        )
        prefix = f"model.language_model.layers.{layer_idx}."
        for name in (
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
            "self_attn.o_proj",
            "self_attn.gate_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        ):
            key = name + ".weight"
            w = state_dict[key] if key in state_dict else state_dict[prefix + key]
            if w.ndim == 1:
                w = (1 + w.float()).reshape(1, 1, 1, -1)
            else:
                w = w.t().contiguous()
            self.weights[name] = ttnn.from_torch(
                w,
                dtype=self._weight_dtype(name),
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=(
                    self._weight_mem(w.shape[-2], w.shape[-1]) if w.shape[-2] > 1 else ttnn.DRAM_MEMORY_CONFIG
                ),
            )
            if "layernorm" in name:
                self.norm_weights[name] = ttnn.from_torch(
                    w.reshape(1, 1, -1, 32),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    device=mesh_device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
        names = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]
        if self.policy.get("pack_attention_gate"):
            names.append("self_attn.gate_proj")
            del self.weights["self_attn.gate_proj"]
        packed = (
            torch.cat(
                [
                    state_dict[n + ".weight"] if n + ".weight" in state_dict else state_dict[prefix + n + ".weight"]
                    for n in names
                ],
                dim=0,
            )
            .t()
            .contiguous()
        )
        self.weights["qkv"] = ttnn.from_torch(
            packed,
            dtype=self._weight_dtype("qkv"),
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=self._weight_mem(6656, packed.shape[-1]),
        )
        if self.policy.get("pack_mlp"):
            names = ["mlp.gate_proj", "mlp.up_proj"]
            packed = (
                torch.cat(
                    [
                        (
                            state_dict[n + ".weight"]
                            if n + ".weight" in state_dict
                            else state_dict[prefix + n + ".weight"]
                        )
                        for n in names
                    ],
                    dim=0,
                )
                .t()
                .contiguous()
            )
            for n in names:
                del self.weights[n]
            self.weights["mlp.gate_up"] = ttnn.from_torch(
                packed,
                dtype=self._weight_dtype("mlp.gate_up"),
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=self._weight_mem(6656, packed.shape[-1]),
            )
        for name, weight in self.weights.items():
            if "layernorm" in name:
                continue
            group = "mlp" if name.startswith("mlp") else "attention"
            fidelity = self.policy.get(name + ".fidelity", self.policy.get(group + "_fidelity", "HiFi2"))
            self.mm_ck[name] = ttnn.init_device_compute_kernel_config(
                mesh_device.arch(),
                math_fidelity=getattr(ttnn.MathFidelity, fidelity),
                math_approx_mode=False,
                fp32_dest_acc_en=self.policy.get("fp32", True),
                packer_l1_acc=True,
            )
            k, n = weight.shape[-2], weight.shape[-1]
            grid = tuple(self.policy.get(name + ".grid", self.grid))
            self.decode_mem[name] = self._width_mem(k, grid, exact=True)
            shard_tiles = self.decode_mem[name].shard_spec.shape[1] // 32
            block = self.policy.get(name + ".block", self.policy.get("block", 1))
            if block == "max":
                block = shard_tiles
            self.decode_pc[name] = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                in0_block_w=block,
                per_core_M=1,
                per_core_N=math.ceil(n / (32 * self.decode_mem[name].shard_spec.grid.num_cores())),
                num_workers_per_dram_bank=self.policy.get(name + ".readers", self.policy.get("readers", 1)),
                fused_activation=(
                    ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU)
                    if self.policy.get("fuse_silu") and name == "mlp.gate_proj"
                    else None
                ),
            )
        self.prefill_weights = {
            name: (
                ttnn.to_memory_config(weight, ttnn.DRAM_MEMORY_CONFIG)
                if self.policy.get("prefill_interleaved")
                else weight
            )
            for name, weight in self.weights.items()
            if not (name == "mlp.gate_up" and self.policy.get("minimal_mlp") in ("prefill", "both"))
        }
        if self.policy.get("minimal_mlp"):
            gate = (
                state_dict.get("mlp.gate_proj.weight", state_dict.get(prefix + "mlp.gate_proj.weight")).t().contiguous()
            )
            up = state_dict.get("mlp.up_proj.weight", state_dict.get(prefix + "mlp.up_proj.weight")).t().contiguous()
            paired = (
                torch.stack((gate.reshape(6656, 624, 32), up.reshape(6656, 624, 32)), dim=2)
                .reshape(6656, 39936)
                .contiguous()
            )
            self.minimal_gate_up = ttnn.from_torch(
                paired,
                dtype=self._weight_dtype("mlp.gate_up"),
                device=mesh_device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        self.positions = ttnn.from_torch(
            torch.arange(c.max_position_embeddings, dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
        )
        if self.use_rope:
            inv = 1 / c.layer_rope_theta[layer_idx] ** (torch.arange(0, 128, 2).float() / 128)
            freq = torch.arange(c.max_position_embeddings).float()[:, None] * inv
            freq = torch.cat((freq, freq), dim=-1)[None, None]
            self.cos = ttnn.from_torch(freq.cos(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device)
            self.sin = ttnn.from_torch(freq.sin(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device)
        return self

    def allocate_cache(self, *, batch_size=1):
        """Setup only: full-context, disjoint pages for up to batch_size requests."""
        shape = (batch_size * self.config.max_position_embeddings // self.PAGE_SIZE, 2, self.PAGE_SIZE, 128)
        return tuple(
            (
                ttnn.zeros(
                    shape,
                    dtype=self.kv_dtype,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                for _ in range(2)
            )
        )

    def _width_mem(self, width, grid=None, *, exact=False, height=32):
        grid = self.grid if grid is None else grid
        if exact:
            tiles = width // 32
            count = max(c for c in range(1, math.prod(grid) + 1) if tiles % c == 0)
            cores = ttnn.num_cores_to_corerangeset(count, ttnn.CoreCoord(*grid), row_wise=True)
            return ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(cores, [height, width // count], ttnn.ShardOrientation.ROW_MAJOR),
            )
        return ttnn.create_sharded_memory_config(
            (height, ((width + 31) // 32 + grid[0] * grid[1] - 1) // (grid[0] * grid[1]) * 32),
            ttnn.CoreGrid(x=grid[0], y=grid[1]),
            ttnn.ShardStrategy.WIDTH,
            ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

    def _weight_dtype(self, name):
        if "layernorm" in name:
            return ttnn.bfloat16
        group = "mlp" if name.startswith("mlp") else "attention"
        return getattr(ttnn, self.policy.get(name + ".dtype", self.policy.get(group + "_dtype", "bfloat8_b")))

    def _weight_mem(self, k, n):
        if not self.dram_sharded:
            return ttnn.DRAM_MEMORY_CONFIG
        dg = self.device.dram_grid_size()
        banks = dg.x * dg.y
        multiple = self.policy.get("weight_bank_multiple", 1)
        width = ((n // 32 + banks * multiple - 1) // (banks * multiple)) * multiple * 32
        return ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.DRAM,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dg.x - 1, dg.y - 1))}),
                [k, width],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )

    def _linear(self, x, name):
        short_prefill = not self._decode and self.policy.get("short_prefill_dram") and x.padded_shape[-2] == 32
        if (self._decode or short_prefill) and self.dram_sharded:
            x = ttnn.to_memory_config(x, self.decode_mem[name])
            pc = self.decode_pc[name]
            if short_prefill and self.policy.get("fuse_silu") and name == "mlp.gate_proj":
                pc = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                    in0_block_w=pc.in0_block_w,
                    per_core_M=1,
                    per_core_N=pc.per_core_N,
                    num_workers_per_dram_bank=pc.num_workers_per_dram_bank,
                )
            out = ttnn.linear(
                x,
                self.weights[name],
                compute_kernel_config=self.mm_ck[name],
                program_config=pc,
                memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                dtype=(
                    ttnn.bfloat16
                    if short_prefill
                    else getattr(
                        ttnn,
                        self.policy.get(("mlp" if name.startswith("mlp") else "attention") + "_act_dtype", "bfloat16"),
                    )
                ),
            )
            # The prefill head-split API consumes interleaved packed heads.
            return ttnn.to_memory_config(out, ttnn.DRAM_MEMORY_CONFIG) if short_prefill and name == "qkv" else out
        preserve = self._sharded_prefill_norm(x) and x.is_sharded() and not self.policy.get("prefill_norm_to_dram")
        if not preserve:
            mem = ttnn.L1_MEMORY_CONFIG if self._prefill_input_l1(x) else ttnn.DRAM_MEMORY_CONFIG
            x = ttnn.to_memory_config(x, mem)
        if not self._decode and self.policy.get("prefill_kernel") == "minimal":
            return ttnn.experimental.minimal_matmul(
                x,
                self.prefill_weights[name],
                config=self._minimal_config(x),
                compute_kernel_config=self.mm_ck[name],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                dtype=getattr(
                    ttnn,
                    self.policy.get(
                        "prefill_" + ("mlp" if name.startswith("mlp") else "attention") + "_act_dtype", "bfloat16"
                    ),
                ),
            )
        pc = None
        if self.dram_sharded or self.policy.get("prefill_2d"):
            m = x.padded_shape[-2] // 32
            n = self.weights[name].padded_shape[-1] // 32
            gx, gy = self.policy.get("prefill_grid", (8, 8))
            gy = min(gy, m)
            pm, pn = math.ceil(m / gy), math.ceil(n / gx)
            bn = self.policy.get("prefill_nblock", 24)
            bn = next(b for b in range(min(bn, pn), 0, -1) if pn % b == 0)
            # DRAM-width-sharded 2D readers describe the full per-core N extent.
            # Internal N subdivision overruns their reserved input-B buffers.
            if self.prefill_weights[name].is_sharded():
                bn = pn
            bm = self.policy.get("prefill_mblock", pm)
            bm = next(b for b in range(min(bm, pm), 0, -1) if pm % b == 0)
            subh, sub = next(
                (h, w)
                for h, w in ((4, 1), (2, 2), (1, 4), (1, 3), (2, 1), (1, 2), (1, 1))
                if bm % h == 0 and bn % w == 0
            )
            pc = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=(gx, gy),
                in0_block_w=self.policy.get("prefill_block", 4),
                out_subblock_h=subh,
                out_subblock_w=sub,
                out_block_h=bm,
                out_block_w=bn,
                per_core_M=pm,
                per_core_N=pn,
                transpose_mcast=False,
                fused_activation=None,
                fuse_batch=False,
            )
        return ttnn.linear(
            x,
            self.prefill_weights[name],
            compute_kernel_config=self.mm_ck[name],
            program_config=pc,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
        )

    def _prefill_input_l1(self, x):
        limit = self.policy.get("prefill_input_l1_max_rows", 0)
        return (
            not self._decode
            and self.policy.get("prefill_input_l1", False)
            and (not limit or x.padded_shape[-2] <= limit)
        )

    def _sharded_prefill_norm(self, x):
        limit = self.policy.get("prefill_norm_max_rows", 0)
        return (
            not self._decode
            and self.policy.get("prefill_norm_sharded", False)
            and x.padded_shape[-2] >= self.policy.get("prefill_norm_min_rows", 0)
            and (not limit or x.padded_shape[-2] <= limit)
        )

    def _norm(self, x, name=None, *, eps=None):
        phase = "decode" if self._decode else "prefill"
        ck = getattr(self, "norm_configs", {}).get(phase, {}).get(name, self.ck)
        if x.dtype != ttnn.bfloat16:
            x = ttnn.typecast(x, ttnn.bfloat16)
        if name and ((self._decode and self.sharded_norm) or self._sharded_prefill_norm(x)):
            height = 32 if self._decode else x.padded_shape[-2]
            norm_grid = self.grid if self._decode else tuple(self.policy.get("prefill_norm_grid", (11, 10)))
            mem = self.residual_mem if self._decode else self._width_mem(6656, norm_grid, exact=True, height=height)
            if not self._decode and self.policy.get("prefill_norm_block"):
                gy = min(8, height // 32)
                while (height // 32) % gy:
                    gy -= 1
                # Irregular tile counts can collapse gy to 1, leaving a
                # nearly 1024-row BF16 shard on one core row. Preserve the
                # bounded BLOCK geometry of prepared buckets; use the
                # already-supported WIDTH layout for taller eager shards.
                if height // gy <= 128:
                    norm_grid = (8, gy)
                    mem = ttnn.create_sharded_memory_config(
                        (height // gy, 832),
                        ttnn.CoreGrid(x=8, y=gy),
                        ttnn.ShardStrategy.BLOCK,
                        ttnn.ShardOrientation.ROW_MAJOR,
                        use_height_and_width_as_shard_shape=True,
                    )
            x = ttnn.to_memory_config(x, mem)
            width = mem.shard_spec.shape[1] // 32
            norm_height = mem.shard_spec.shape[0] // 32
            pc = ttnn.LayerNormShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=norm_grid,
                subblock_w=next(b for b in (4, 3, 2, 1) if width % b == 0),
                block_h=norm_height,
                block_w=width,
                inplace=not self._decode and self.policy.get("prefill_norm_inplace", False),
            )
            return ttnn.rms_norm(
                x,
                epsilon=eps or self.config.rms_norm_eps,
                weight=self.norm_weights[name],
                compute_kernel_config=ck,
                program_config=pc,
                memory_config=mem,
            )
        mem = (
            ttnn.L1_MEMORY_CONFIG
            if name is None and self._decode and self.policy.get("qk_norm_l1")
            else ttnn.DRAM_MEMORY_CONFIG
        )
        return ttnn.rms_norm(
            ttnn.to_memory_config(x, mem),
            epsilon=eps or self.config.rms_norm_eps,
            weight=self.weights[name] if name else None,
            compute_kernel_config=ck,
            memory_config=mem,
        )

    def _add(self, x, y):
        mem = self.residual_mem if self._decode and self.sharded_norm else ttnn.DRAM_MEMORY_CONFIG
        return ttnn.add(ttnn.to_memory_config(x, mem), ttnn.to_memory_config(y, mem), memory_config=mem)

    def _project_attention(self, normalized):
        packed = self._linear(normalized, "qkv")
        if not self.policy.get("pack_attention_gate"):
            return packed, None
        shape = list(packed.shape)
        gate = ttnn.slice(packed, [0, 0, 0, 4608], shape)
        packed = ttnn.slice(packed, [0, 0, 0, 0], shape[:-1] + [4608])
        return packed, gate

    def _rope(self, x, cos, sin):
        return ttnn.experimental.rotary_embedding_hf(x, cos, sin, compute_kernel_config=self.ck)

    def _finish(self, residual, normalized, attn, gate=None):
        gate = self._linear(normalized, "self_attn.gate_proj") if gate is None else gate
        gated = ttnn.multiply(
            gate,
            ttnn.to_memory_config(attn, gate.memory_config()),
            input_tensor_a_activations=[ttnn.UnaryOpType.SIGMOID],
        )
        attn = self._linear(gated, "self_attn.o_proj")
        x = self._add(residual, self._norm(attn, "post_attention_layernorm", eps=self.config.post_norm_eps))
        normalized = self._norm(x, "pre_feedforward_layernorm")
        mlp = self._mlp(normalized)
        return self._add(x, self._norm(mlp, "post_feedforward_layernorm", eps=self.config.post_norm_eps))

    def _minimal_config(self, x):
        bm = min(self.policy.get("minimal_m", 4), x.padded_shape[-2] // 32)
        return ttnn.MinimalMatmulConfig(
            M_block_size=bm,
            K_block_size=self.policy.get("minimal_k", 8),
            N_block_size=self.policy.get("minimal_n", 16),
            subblock_h=2 if bm % 2 == 0 else 1,
            subblock_w=4,
            compute_with_storage_grid_size=ttnn.CoreCoord(*self.policy.get("minimal_grid", (11, 10))),
        )

    def _mlp(self, normalized):
        minimal = self.policy.get("minimal_mlp")
        if not self._decode and self.policy.get("short_prefill_dram") and normalized.padded_shape[-2] == 32:
            minimal = False
        if minimal and (minimal == "both" or (minimal == "decode") == self._decode):
            product = ttnn.experimental.minimal_matmul(
                (
                    normalized
                    if (
                        self._sharded_prefill_norm(normalized)
                        and normalized.is_sharded()
                        and not self.policy.get("prefill_norm_to_dram")
                    )
                    else ttnn.to_memory_config(
                        normalized,
                        (ttnn.L1_MEMORY_CONFIG if self._prefill_input_l1(normalized) else ttnn.DRAM_MEMORY_CONFIG),
                    )
                ),
                self.minimal_gate_up,
                config=self._minimal_config(normalized),
                fuse_swiglu=True,
                compute_kernel_config=self.mm_ck["mlp.gate_up"],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                dtype=ttnn.bfloat16,
            )
            return self._linear(product, "mlp.down_proj")
        if self.policy.get("pack_mlp"):
            packed = self._linear(normalized, "mlp.gate_up")
            shape = list(packed.shape)
            gate = ttnn.slice(packed, [0, 0, 0, 0], shape[:-1] + [19968])
            up = ttnn.slice(packed, [0, 0, 0, 19968], shape)
        else:
            gate = self._linear(normalized, "mlp.gate_proj")
            up = self._linear(normalized, "mlp.up_proj")
        product = ttnn.multiply(
            gate,
            up,
            input_tensor_a_activations=(
                [] if self._decode and self.policy.get("fuse_silu") else [ttnn.UnaryOpType.SILU]
            ),
        )
        return self._linear(product, "mlp.down_proj")

    def decode_forward(self, x, *, kv_cache, page_table, current_pos, rope_cos=None, rope_sin=None):
        """One on-device step; position/page/RoPE inputs can change between replays."""
        self._decode = True
        if self.policy.get("keep_residual"):
            x = ttnn.to_memory_config(x, self.residual_mem)
        batch = x.shape[2]
        partial_row = batch > 8 and batch % 8 != 0
        normalized = self._norm(x, "input_layernorm")
        packed, attention_gate = self._project_attention(normalized)
        heads_mem = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))}),
                [32, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )
        if self.policy.get("aligned_heads"):
            cores = ttnn.num_cores_to_corerangeset(36, ttnn.CoreCoord(8, 8), row_wise=True)
            packed = ttnn.to_memory_config(
                packed,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                    ttnn.BufferType.L1,
                    ttnn.ShardSpec(cores, [32, 128], ttnn.ShardOrientation.ROW_MAJOR),
                ),
            )
        if packed.dtype != ttnn.bfloat16:
            packed = ttnn.typecast(packed, ttnn.bfloat16)
        q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
            packed if self.policy.get("sharded_heads") else ttnn.to_memory_config(packed, ttnn.DRAM_MEMORY_CONFIG),
            num_heads=32,
            num_kv_heads=2,
            memory_config=heads_mem,
        )
        # Native RMSNorm cannot consume height shards; V stays in its cache layout.
        q, k = (self._norm(q), self._norm(k))
        grid = ttnn.num_cores_to_corerangeset(batch, ttnn.CoreCoord(8, 8), row_wise=True)
        mem = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(grid, [32, 128], ttnn.ShardOrientation.ROW_MAJOR),
        )
        if self.use_rope:
            if rope_cos is None or rope_sin is None:
                raise ValueError("RoPE device tensors are required for sliding layers")
            if partial_row:
                # Native decode RoPE launches on a bounding rectangle. For an
                # incomplete core row, make users its sequence axis instead.
                cos, sin = (
                    ttnn.reshape(ttnn.to_memory_config(t, ttnn.DRAM_MEMORY_CONFIG), (1, 1, batch, 128))
                    for t in (rope_cos, rope_sin)
                )
                q, k = (ttnn.transpose(self._rope(ttnn.transpose(t, 1, 2), cos, sin), 1, 2) for t in (q, k))
            else:
                cos, sin = (ttnn.to_memory_config(t, mem) for t in (rope_cos, rope_sin))
                q, k = (
                    ttnn.experimental.rotary_embedding_hf(
                        ttnn.to_memory_config(t, mem), cos, sin, is_decode_mode=True, compute_kernel_config=self.ck
                    )
                    for t in (q, k)
                )
        # Fused cache update needs disjoint K/V cores, including at batch32.
        kgrid = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(i % 8, 4 + i // 8), ttnn.CoreCoord(i % 8, 4 + i // 8)) for i in range(batch)}
        )
        kmem = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(kgrid, [32, 128], ttnn.ShardOrientation.ROW_MAJOR),
        )
        k = ttnn.to_memory_config(k, kmem)
        # V already uses these cores; head creation writes the cache layout directly.
        ttnn.experimental.paged_fused_update_cache(
            kv_cache[0], k, kv_cache[1], v, update_idxs_tensor=current_pos, page_table=page_table
        )
        # Both NoPE and the partial-core-row RoPE path produce interleaved Q.
        # SDPA accepts L1 only for a sharded Q tensor.
        if not q.is_sharded():
            q = ttnn.to_memory_config(q, ttnn.DRAM_MEMORY_CONFIG)
        attn = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            q,
            *kv_cache,
            page_table,
            cur_pos_tensor=current_pos,
            sliding_window_size=self.window,
            scale=self.attention_scale,
            program_config=self.sdpa_decode,
            compute_kernel_config=self.attention_ck,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        # GQA SDPA requires DRAM output; concat-heads requires L1 height shards.
        attn = ttnn.to_memory_config(attn, mem)
        # A partial input row selects the subgrid kernel, which requires an
        # explicit 32-core destination grid before output allocation.
        attn = ttnn.experimental.nlp_concat_heads_decode(
            attn, num_heads=32, sub_core_grids=heads_mem.shard_spec.grid if partial_row else None
        )
        if not self.policy.get("keep_attn_l1"):
            attn = ttnn.to_memory_config(attn, ttnn.DRAM_MEMORY_CONFIG)
        attn = ttnn.reshape(attn, ttnn.Shape([1, 1, batch, 4096]), ttnn.Shape([1, 1, 32, 4096]))
        if self.policy.get("release_decode"):
            del q, k, v, packed
            if attention_gate is not None:
                del normalized
                normalized = None
        return self._finish(x, normalized, attn, attention_gate)

    def _prefill_chunk(self, x, *, start_pos, kv_cache, page_table, slot):
        self._decode = False
        length = x.shape[2]
        normalized = self._norm(x, "input_layernorm")

        packed, attention_gate = self._project_attention(normalized)
        if attention_gate is not None and self.policy.get("release_normalized"):
            del normalized
            normalized = None
        q, k, v = ttnn.transformer.split_query_key_value_and_split_heads(
            ttnn.reshape(packed, (1, length, 4608)), num_heads=32, num_kv_heads=2, transpose_key=False
        )
        q, k = (self._norm(q), self._norm(k))
        if self.use_rope:
            cos = ttnn.slice(self.cos, [0, 0, start_pos, 0], [1, 1, start_pos + length, 128])
            sin = ttnn.slice(self.sin, [0, 0, start_pos, 0], [1, 1, start_pos + length, 128])
            q, k = (self._rope(q, cos, sin), self._rope(k, cos, sin))
        pages = ttnn.slice(page_table, [slot, start_pos // 32], [slot + 1, (start_pos + length) // 32])
        for cache, new in zip(kv_cache, (k, v)):
            ttnn.experimental.paged_fill_cache(cache, ttnn.typecast(new, self.kv_dtype), pages, batch_idx=0)
        row = ttnn.slice(page_table, [slot, 0], [slot + 1, page_table.shape[1]])
        sdpa = self.sdpa
        q_chunk = self.policy.get("prefill_sdpa_q", 32)
        k_chunk = self.policy.get("prefill_sdpa_k", 32)
        if start_pos % q_chunk or start_pos % k_chunk:
            # Chunked SDPA indexes blocks from the absolute cache position.
            # Append prefixes are page-aligned, not necessarily 256-aligned.
            while start_pos % q_chunk:
                q_chunk //= 2
            while start_pos % k_chunk:
                k_chunk //= 2
            sdpa = ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(*self.policy.get("prefill_sdpa_grid", (8, 8))),
                q_chunk_size=q_chunk,
                k_chunk_size=k_chunk,
                exp_approx_mode=False,
            )
        attn = ttnn.transformer.chunked_scaled_dot_product_attention(
            q,
            *kv_cache,
            row,
            chunk_start_idx=start_pos,
            scale=self.attention_scale,
            sliding_window_size=self.window,
            program_config=sdpa,
            compute_kernel_config=self.attention_ck,
        )
        attn = ttnn.reshape(ttnn.transformer.concatenate_heads(attn), (1, 1, length, 4096))
        return self._finish(x, normalized, attn, attention_gate)

    def prefill_forward(self, x, *, kv_cache, page_table, start_pos=0, slots=None):
        """Chunked, append-only prefill; logical S may be any positive length.

        Initial unaligned prefix positions use single-row cache updates until the
        next page boundary. Padded final rows occupy only unused request-owned
        cache slots, are causally invisible, and are replaced on continuation.
        """
        batch, _, length, hidden = x.shape
        if not (
            0 <= start_pos < self.config.max_position_embeddings
            and 0 < length <= self.config.max_position_embeddings - start_pos
        ):
            raise ValueError("Prefill exceeds the advertised context or is empty")
        slots = tuple(range(batch)) if slots is None else tuple(slots)
        if len(slots) != batch or len(set(slots)) != batch:
            raise ValueError("Each batch row needs a distinct cache slot")
        results = []
        for b, slot in enumerate(slots):
            pieces, offset = ([], 0)
            while offset < length:
                absolute = start_pos + offset
                if absolute % 32:
                    token = ttnn.slice(x, [b, 0, offset, 0], [b + 1, 1, offset + 1, hidden])
                    row = ttnn.slice(page_table, [slot, 0], [slot + 1, page_table.shape[1]])
                    pos = ttnn.slice(self.positions, [absolute], [absolute + 1])
                    cos = (
                        ttnn.slice(self.cos, [0, 0, absolute, 0], [1, 1, absolute + 1, 128]) if self.use_rope else None
                    )
                    sin = (
                        ttnn.slice(self.sin, [0, 0, absolute, 0], [1, 1, absolute + 1, 128]) if self.use_rope else None
                    )
                    pieces.append(
                        # Prefix rows must not retain decode's L1 residual
                        # buffers across the following large prefill kernel.
                        ttnn.to_memory_config(
                            self.decode_forward(
                                token, kv_cache=kv_cache, page_table=row, current_pos=pos, rope_cos=cos, rope_sin=sin
                            ),
                            ttnn.DRAM_MEMORY_CONFIG,
                        )
                    )
                    offset += 1
                    continue
                valid = min(self.CHUNK_SIZE, length - offset)
                if self.policy.get("short_prefill_dram") and valid <= 64:
                    # The fast DRAM-sharded kernel accepts one physical M tile.
                    # Two short chunks beat the minimal kernel for M=64 too.
                    valid = min(32, valid)
                physical = (valid + 31) // 32 * 32
                chunk = ttnn.slice(x, [b, 0, offset, 0], [b + 1, 1, offset + valid, hidden])
                if physical != valid:
                    chunk = ttnn.pad(chunk, [(0, 0), (0, 0), (0, physical - valid), (0, 0)], 0)
                out = self._prefill_chunk(
                    chunk, start_pos=absolute, kv_cache=kv_cache, page_table=page_table, slot=slot
                )
                pieces.append(ttnn.slice(out, [0, 0, 0, 0], [1, 1, valid, hidden]))
                offset += valid
            if len(pieces) > 1 and start_pos % self.PAGE_SIZE == 0:
                # Only the last piece has padding. Joining physical tile views
                # does not insert padding between logical tokens.
                physical_pieces = [ttnn.reshape(p, p.padded_shape, p.padded_shape) for p in pieces]
                joined = ttnn.concat(physical_pieces, dim=2)
                results.append(
                    ttnn.reshape(
                        joined,
                        ttnn.Shape([1, 1, length, hidden]),
                        ttnn.Shape([1, 1, (length + 31) // 32 * 32, hidden]),
                    )
                )
            else:
                pieces = [ttnn.to_memory_config(p, ttnn.DRAM_MEMORY_CONFIG) for p in pieces]
                results.append(ttnn.concat(pieces, dim=2) if len(pieces) > 1 else pieces[0])
        return ttnn.concat(results, dim=0) if batch > 1 else results[0]
