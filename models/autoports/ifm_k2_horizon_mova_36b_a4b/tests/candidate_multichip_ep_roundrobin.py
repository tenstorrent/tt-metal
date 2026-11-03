# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""TP4 Blackhole decoder; optimized single-chip arithmetic is the baseline.

Setup shards checkpoint weights. Runtime tensors live on a 1x4 ring mesh.
Local attention has eight Q and two KV heads; caches are head-sharded.
Decode uses TP active experts; larger prefill uses EP active-expert banks.
"""

import math

import ttnn
from models.autoports.ifm_k2_horizon_mova_36b_a4b.tt.optimized_decoder import DENSE_POLICY, MOE_POLICY, OptimizedDecoder
from models.common.tt_ccl import get_tt_ccl


class MultichipDecoder(OptimizedDecoder):
    TP = 4
    Q_WIDTH = 1024
    KV_WIDTH = 256
    EXPERT_WIDTH = 192
    SHARED_WIDTH = 256
    DENSE_WIDTH = 1536

    @classmethod
    def from_state_dict(cls, state_dict, *, hf_config, layer_idx, mesh_device, chunk_size=2048, options=None, ccl=None):
        import torch

        c = hf_config
        if tuple(mesh_device.shape) != (1, 4):
            raise ValueError("MultichipDecoder requires the target 1x4 Blackhole ring")
        if (c.hidden_size, c.num_attention_heads, c.num_key_value_heads, c.head_dim) != (2560, 32, 8, 128):
            raise ValueError("Expected IFM/K2-Horizon-MoVA-36B-A4B")
        if c.query_key_norm or c.sliding_window is not None or c.rope_head_dim != 128:
            raise ValueError("Unsupported target configuration")
        if not 0 <= layer_idx < c.num_hidden_layers or chunk_size <= 0 or chunk_size % 32:
            raise ValueError("Invalid layer index or physical chunk size")
        if (
            c.mova_num_experts,
            c.mova_num_experts_per_tok,
            c.num_experts,
            c.num_experts_per_tok,
            c.moe_intermediate_size,
            c.num_shared_experts,
            c.layernorm_num_groups,
            c.intermediate_size,
        ) != (64, 4, 100, 8, 768, 1, 2, 6144):
            raise ValueError("Expert and normalization shapes must match the target checkpoint")
        if (
            c.attention_bias
            or c.attention_gate_func != "softplus"
            or c.hidden_act != "silu"
            or c.router_score_func != "sigmoid"
            or c.router_scaling_factor != 2.5
            or not c.norm_topk_prob
            or not c.moe_gate_bias
            or c.decoder_sparse_step != 1
        ):
            raise ValueError("Unsupported attention or router semantics")
        self = cls()
        self.config, self.mesh_device, self.layer_idx, self.chunk_size = c, mesh_device, layer_idx, chunk_size
        self.sparse = layer_idx not in c.mlp_only_layers
        self.options = {
            **(MOE_POLICY if self.sparse else DENSE_POLICY),
            "dram_output_block": 4,
            "dram_down_block": 1 if self.sparse else 6,
            "gate_up_grid": [6, 2],
            "gate_up_subblock": 1,
            "value_grid": [8, 1],
            "value_subblock": 1,
            "down_block": 6,
            "prefill_down_block": 6,
            "prefill_gate_up_grid": [6, 2],
            "prefill_gate_up_subblock": 1,
            "prefill_value_grid": [8, 1],
            "prefill_value_subblock": 1,
            "residual_layout": "replicated",
            "num_links": 2,
            **(
                {
                    "gate_up_block": 80,
                    "down_grid": [8, 10],
                    "down_subblock": 1,
                    "value_block": 80,
                    "dram_down_block": 8,
                    "dram_output_block": 8,
                    "down_readers": 1,
                    "up_readers": 1,
                }
                if self.sparse
                else {"output_readers": 1, "down_readers": 1}
            ),
            **(options or {}),
        }
        self.residual_sharded = self.options["residual_layout"] == "sharded"
        self.kv_cache_dtype = getattr(ttnn, self.options["cache_dtype"])
        self.ccl = ccl or get_tt_ccl(mesh_device)
        self.compute = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False
        )
        self.matmul_compute = {
            group: ttnn.WormholeComputeKernelConfig(
                math_fidelity=getattr(ttnn.MathFidelity, self.options.get(group + "_fidelity", "LoFi")),
                math_approx_mode=False,
                fp32_dest_acc_en=False,
                packer_l1_acc=True,
            )
            for group in ("attention", "mlp", "experts", "value")
        }
        self.sdpa_compute = {
            phase: ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.LoFi if phase == "short" else ttnn.MathFidelity.HiFi2,
                math_approx_mode=False,
                fp32_dest_acc_en=phase == "long",
                packer_l1_acc=False,
            )
            for phase in ("prefill", "short", "middle", "long")
        }
        prefix = f"model.layers.{layer_idx}."
        sd = {k.removeprefix(prefix): v for k, v in state_dict.items()}
        self.weights = {}
        self.weight_manifest = {}

        def put(name, values, dtype=ttnn.bfloat16):
            sharded = isinstance(values, list)
            value = torch.cat(values, dim=0) if sharded else values
            mapper = ttnn.ShardTensorToMesh(mesh_device, dim=0) if sharded else ttnn.ReplicateTensorToMesh(mesh_device)
            self.weights[name] = ttnn.from_torch(
                value.contiguous(),
                device=mesh_device,
                mesh_mapper=mapper,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            self.weight_manifest[name] = {
                "local_shape": list(values[0].shape if sharded else value.shape),
                "dtype": str(dtype),
                "mapping": "TP4" if sharded else "replicated",
            }

        def col(name, rank):
            return sd[name + ".weight"].T.chunk(4, dim=-1)[rank]

        def row(name, rank):
            return sd[name + ".weight"].T.chunk(4, dim=-2)[rank]

        for name in ("input_layernorm", "post_attention_layernorm"):
            put(name + ".weight", sd[name + ".weight"].reshape(1, 1, 1, -1))
        packed = []
        for r in range(4):
            parts = [col("self_attn.q_proj", r), col("self_attn.k_proj", r)]
            if not self.sparse:
                parts.append(col("self_attn.v_proj", r))
            parts.append(col("self_attn.gate_proj", r))
            if self.sparse:
                parts.append(sd["self_attn.v_router.weight"].T)
            packed.append(torch.cat(parts, dim=-1))
        put("attention_packed.weight", packed, getattr(ttnn, self.options["attention_dtype"]))
        put("self_attn.o_proj.weight", [row("self_attn.o_proj", r) for r in range(4)], ttnn.bfloat4_b)
        mlp = "mlp.shared_experts" if self.sparse else "mlp"
        gate_up, swiglu, down = [], [], []
        for r in range(4):
            gate, up, dw = col(mlp + ".gate_proj", r), col(mlp + ".up_proj", r), row(mlp + ".down_proj", r)
            if self.sparse:
                gate, up = [torch.nn.functional.pad(v, (0, 64)) for v in (gate, up)]
                dw = torch.nn.functional.pad(dw, (0, 0, 0, 64))
                router = torch.nn.functional.pad(sd["mlp.gate.weight"].T, (0, 28))
                gate_up.append(torch.cat([gate, up, router], dim=-1))
            else:
                gate_up.append(torch.cat([gate, up], dim=-1))
                swiglu.append(
                    torch.stack([gate.reshape(2560, -1, 32), up.reshape(2560, -1, 32)], dim=2).reshape(2560, -1)
                )
            down.append(dw)
        put("mlp.shared_router.weight" if self.sparse else "mlp.gate_up.weight", gate_up, ttnn.bfloat4_b)
        put(mlp + ".down_proj.weight", down, ttnn.bfloat4_b)
        if not self.sparse:
            put("mlp.swiglu.weight", swiglu, ttnn.bfloat4_b)
        if self.sparse:
            put(
                "value_experts",
                [torch.stack([col(f"self_attn.v_experts.{e}", r) for e in range(64)]).unsqueeze(0) for r in range(4)],
                ttnn.bfloat4_b,
            )
            put(
                "experts.gate_up",
                [
                    torch.stack(
                        [
                            torch.cat(
                                [col(f"mlp.experts.{e}.gate_proj", r), col(f"mlp.experts.{e}.up_proj", r)], dim=-1
                            )
                            for e in range(100)
                        ]
                    ).unsqueeze(0)
                    for r in range(4)
                ],
                ttnn.bfloat4_b,
            )
            put(
                "experts.down_proj",
                [
                    torch.stack([row(f"mlp.experts.{e}.down_proj", r) for e in range(100)]).unsqueeze(0)
                    for r in range(4)
                ],
                ttnn.bfloat4_b,
            )
        # A second layout is worthwhile for prefill: each chip owns25 full-width
        # experts. Padding to32 enables tiled score projection and sparse kernels.
        # Keep TP banks for indexed decode. Both layouts fit the full-stack plan.
        if self.sparse:
            ep_gu, ep_down, selectors = [], [], []
            for rank in range(4):
                gu = torch.zeros(1, 32, 2560, 1536, dtype=torch.bfloat16)
                dw = torch.zeros(1, 32, 768, 2560, dtype=torch.bfloat16)
                selector = torch.zeros(1, 1, 128, 32, dtype=torch.bfloat16)
                for local in range(25):
                    expert = local * 4 + rank
                    gu[0, local] = torch.cat(
                        [sd[f"mlp.experts.{expert}.{part}_proj.weight"].T for part in ("gate", "up")], dim=-1
                    )
                    dw[0, local] = sd[f"mlp.experts.{expert}.down_proj.weight"].T
                    selector[0, 0, expert, local] = 1
                ep_gu.append(gu)
                ep_down.append(dw)
                selectors.append(selector)
            put("experts.ep_gate_up", ep_gu, ttnn.bfloat4_b)
            put("experts.ep_down", ep_down, ttnn.bfloat4_b)
            put("experts.ep_selector", selectors, ttnn.bfloat16)
            for name in ("experts.ep_gate_up", "experts.ep_down", "experts.ep_selector"):
                self.weight_manifest[name]["mapping"] = "EP4:25 real +7 inert experts"

        def replicated(value, dtype, layout=ttnn.TILE_LAYOUT):
            return ttnn.from_torch(
                value.contiguous(),
                device=mesh_device,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
                dtype=dtype,
                layout=layout,
            )

        self.positions = replicated(
            torch.arange(c.max_position_embeddings, dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT
        )
        if self.sparse:
            self.route_bias, self.placeholders = {}, {}
            for name, e in (("self_attn.v_router", 64), ("mlp.gate", 100)):
                bias = torch.full((1, 1, 1, math.ceil(e / 32) * 32), -float("inf"))
                bias[..., :e] = sd[name + ".bias"].float()
                put(name + ".bias", bias, ttnn.float32)
                self.route_bias[name] = replicated(bias.expand(1, 1, max(chunk_size, 32), -1), ttnn.float32)
                self.placeholders[e] = replicated(torch.zeros(1, 1, 1, e), ttnn.float32, ttnn.ROW_MAJOR_LAYOUT)
            self.reduce_ids = replicated(
                torch.arange(100).int().reshape(1, 1, 1, 100).expand(32, -1, -1, -1), ttnn.uint16, ttnn.ROW_MAJOR_LAYOUT
            )
            self.reduce_map = replicated(torch.zeros(1, 100).int(), ttnn.uint16, ttnn.ROW_MAJOR_LAYOUT)
        self._prepare_tp_decode_weights()
        self._prepare_all_reduce()
        return self

    def _prepare_all_reduce(self):
        # The caller may pass a shared TT_CCL; the default is the common per-mesh
        # manager. Scratch belongs to that manager, not each of48 layers.
        # One ordered command queue serializes layer uses. WO and MLP have distinct
        # slots, and neither scratch aliases a residual/output or changes address.
        self.ar_dtype = ttnn.bfloat8_b
        grid = ttnn.num_cores_to_corerangeset(8, self.mesh_device.compute_with_storage_grid_size(), True)

        def mem(width):
            return ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(grid, [32, width // 8], ttnn.ShardOrientation.ROW_MAJOR),
            )

        self.ar_mem = mem(2560)
        if not hasattr(self.ccl, "k2_all_reduce_slots"):
            self.ccl.k2_all_reduce_slots = [
                (
                    ttnn.empty(
                        (1, 1, 32, 10240),
                        dtype=self.ar_dtype,
                        layout=ttnn.TILE_LAYOUT,
                        device=self.mesh_device,
                        memory_config=mem(10240),
                    ),
                    ttnn.create_global_semaphore(self.mesh_device, grid, 0),
                )
                for _ in range(2)
            ]
        self.ar_slots = self.ccl.k2_all_reduce_slots

    def _prepare_tp_decode_weights(self):
        grid = self.mesh_device.dram_grid_size()
        self.banks = grid.x * grid.y
        dram_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
        self.decode_weights = {}
        for name, w in self.weights.items():
            if not name.endswith(".weight") or w.shape[-2] <= 1 or "swiglu" in name:
                continue
            k, n = tuple(w.shape)[-2:]
            role = self._role(name.removesuffix(".weight"))
            readers = self.options.get(role + "_readers", self.options.get("readers", 1))
            shard_n = math.ceil(n / (32 * self.banks * readers)) * 32 * readers
            mem = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.DRAM,
                ttnn.ShardSpec(dram_grid, [k, shard_n], ttnn.ShardOrientation.ROW_MAJOR),
            )
            if shard_n * self.banks != n:
                dtype = w.dtype
                if dtype == ttnn.bfloat4_b:
                    w = ttnn.typecast(w, ttnn.bfloat16)
                w = ttnn.pad(w, [(0, 0)] * (len(w.shape) - 1) + [(0, shard_n * self.banks - n)], 0)
                if w.dtype != dtype:
                    w = ttnn.typecast(w, dtype)
            self.decode_weights[name] = ttnn.to_memory_config(w, mem)

    def _gather(self, x):
        return ttnn.experimental.all_gather_async(
            x,
            dim=3,
            cluster_axis=1,
            mesh_device=self.mesh_device,
            topology=ttnn.Topology.Ring,
            num_links=self.options["num_links"],
            multi_device_global_semaphore=self.ccl.get_and_cycle_ag_semaphore_handles(1),
        )

    def _reduce(self, x, slot=0):
        if x.shape[-2] == 32 and not self.residual_sharded:
            partial = ttnn.to_memory_config(ttnn.typecast(x, self.ar_dtype), self.ar_mem)
            scratch, semaphore = self.ar_slots[slot]
            return ttnn.experimental.all_reduce_async(
                partial,
                scratch,
                cluster_axis=1,
                mesh_device=self.mesh_device,
                multi_device_global_semaphore=semaphore,
                memory_config=self.ar_mem,
                dtype=ttnn.bfloat16,
                topology=ttnn.Topology.Ring,
                num_links=self.options["num_links"],
            )
        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
        reduced = ttnn.experimental.reduce_scatter_minimal_async(
            x,
            dim=3,
            cluster_axis=1,
            topology=ttnn.Topology.Ring,
            num_links=self.options["num_links"],
            multi_device_global_semaphore=self.ccl.get_and_cycle_rs_semaphore_handles(1),
            barrier_semaphore=self.ccl.get_and_cycle_barrier_semaphore_handle(1),
        )
        return reduced if self.residual_sharded else self._gather(reduced)

    def _mlp(self, x, *, valid_tokens=None):
        if self.sparse:
            gu = self._linear(x, "mlp.shared_router")
            hidden = ttnn.mul(gu[..., :256], gu[..., 256:512], input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
            local = ttnn.add(
                self._linear(hidden, "mlp.shared_experts.down_proj"),
                self._experts(x, logits=gu[..., 512:], valid_tokens=valid_tokens),
            )
        elif x.shape[-2] > 32:
            hidden = ttnn.experimental.minimal_matmul(
                x,
                self.weights["mlp.swiglu.weight"],
                fuse_swiglu=True,
                config=ttnn.MinimalMatmulConfig(
                    M_block_size=4,
                    K_block_size=16,
                    N_block_size=8,
                    subblock_h=2,
                    subblock_w=4,
                    compute_with_storage_grid_size=self.mesh_device.compute_with_storage_grid_size(),
                ),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                dtype=ttnn.bfloat8_b,
                compute_kernel_config=self.matmul_compute["mlp"],
            )
            local = self._linear(hidden, "mlp.down_proj")
        else:
            gu = self._linear(x, "mlp.gate_up")
            hidden = ttnn.mul(gu[..., :1536], gu[..., 1536:], input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
            local = self._linear(hidden, "mlp.down_proj")
        return self._reduce(local, slot=1)

    def _finish(self, residual, gate_logits, attention, *, valid_tokens=None):
        gated = ttnn.mul(
            attention,
            gate_logits,
            input_tensor_b_activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.SOFTPLUS, math.log(2), 20.0)],
        )
        projected = self._reduce(self._linear(gated, "self_attn.o_proj"))
        x = ttnn.add(residual, projected)
        normed = self._norm(self._gather(x) if self.residual_sharded else x, "post_attention_layernorm")
        return ttnn.add(x, self._mlp(normed, valid_tokens=valid_tokens))

    def _sparse_mm(self, x, w, sparsity, *, sparse_input=False):
        n = w.shape[-1]
        role = "down" if sparse_input else ("value" if n == 256 else "gate_up")
        groups = x.shape[0] if sparse_input else x.shape[1]

        def option(key, default):
            return self.options.get(
                ("prefill_" if groups > 1 else "") + role + "_" + key, self.options.get(role + "_" + key, default)
            )

        grid = option("grid", [8, 1])
        program = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=tuple(grid),
            in0_block_w=option("block", 2),
            out_subblock_h=1,
            out_subblock_w=option("subblock", 1),
            out_block_h=1,
            out_block_w=option("subblock", 1),
            per_core_M=x.shape[-2] // 32,
            per_core_N=math.ceil(n / (32 * math.prod(grid))),
            fuse_batch=False,
            fused_activation=None,
            mcast_in0=True,
        )
        return ttnn.sparse_matmul(
            x,
            w,
            sparsity=sparsity,
            nnz=None,
            is_input_a_sparse=sparse_input,
            is_input_b_sparse=not sparse_input,
            program_config=program,
            compute_kernel_config=self.matmul_compute["value" if role == "value" else "experts"],
            dtype=(
                (getattr(ttnn, self.options["prefill_value_activation"]) if groups > 1 else ttnn.bfloat16)
                if role == "value"
                else getattr(ttnn, self.options["expanded_expert_activation"])
            ),
            memory_config=(
                ttnn.L1_MEMORY_CONFIG
                if self.options.get("sparse_l1") and groups <= self.options.get("l1_groups", 1)
                else ttnn.DRAM_MEMORY_CONFIG
            ),
        )

    def _prefill_chunk(self, x, *, cos, sin, page_table, kv_cache, start_pos):
        logical = x.shape[2]
        physical = math.ceil(logical / 32) * 32
        x = ttnn.pad(x, [(0, 0), (0, 0), (0, physical - logical), (0, 0)], 0)
        cos = ttnn.pad(cos, [(0, 0), (0, 0), (0, physical - logical), (0, 0)], 0)
        sin = ttnn.pad(sin, [(0, 0), (0, 0), (0, physical - logical), (0, 0)], 0)
        normed = self._norm(self._gather(x) if self.residual_sharded else x, "input_layernorm")
        packed = self._linear(normed, "attention_packed")
        qkv_width = 1280 if self.sparse else 1536
        gate_logits = packed[..., qkv_width : qkv_width + 1024]
        qkv = packed[..., :qkv_width]
        if self.sparse:
            qkv = ttnn.concat([qkv, self._experts(normed, value=True, logits=packed[..., qkv_width + 1024 :])], dim=-1)
        q, k, v = ttnn.transformer.split_query_key_value_and_split_heads(
            ttnn.reshape(qkv, (1, physical, 1536)), num_heads=8, num_kv_heads=2, transpose_key=False
        )
        q, k = self._rope(q, cos, sin), self._rope(k, cos, sin)
        first_page = start_pos // kv_cache[0].shape[2]
        pages = page_table[:, first_page : first_page + math.ceil(physical / kv_cache[0].shape[2])]
        full_tiles = logical // 32 * 32
        if full_tiles:
            ttnn.experimental.paged_fill_cache(
                kv_cache[0], ttnn.typecast(k[:, :, :full_tiles, :], kv_cache[0].dtype), pages, batch_idx=0
            )
            ttnn.experimental.paged_fill_cache(
                kv_cache[1], ttnn.typecast(v[:, :, :full_tiles, :], kv_cache[1].dtype), pages, batch_idx=0
            )
        # Fill writes whole tiles. Tail updates touch only valid rows, preserving
        # all other cache entries even when the physical compute input is padded.
        for token in range(full_tiles, logical):
            position = self.positions[start_pos + token : start_pos + token + 1]
            rows = [ttnn.permute(heads[:, :, token : token + 1, :], (0, 2, 1, 3)) for heads in (k, v)]
            self._update_cache(*rows, position, page_table, kv_cache)
        # Bound BF16 online accumulation to at most 512 K-block merges at
        # the advertised 524288 context. Smaller blocks fail long-context PCC.
        full_grid = self.mesh_device.compute_with_storage_grid_size()
        attention_grid = (full_grid.x, full_grid.y) if start_pos + physical >= 65536 else (8, 8)
        out = ttnn.transformer.chunked_scaled_dot_product_attention(
            q,
            kv_cache[0],
            kv_cache[1],
            self._attention_pages(page_table, kv_cache[0].shape[2]),
            chunk_start_idx=None,
            chunk_start_idx_tensor=self.positions[start_pos : start_pos + 1],
            compute_kernel_config=self.sdpa_compute["prefill"],
            program_config=ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=attention_grid, q_chunk_size=32, k_chunk_size=1024, exp_approx_mode=False
            ),
        )
        out = ttnn.reshape(ttnn.transformer.concatenate_heads(out), (1, 1, physical, 1024))
        return self._finish(x, gate_logits, out)[:, :, :logical, :]

    def decode_forward(self, hidden_states, *, cos, sin, page_table, current_pos, kv_cache):
        """Device-only traceable decode; current_pos and page_table are mutable inputs.

        cos/sin: [1,B,1,128]. No host reads, conversions, cache construction or
        state reset occurs in this call. The output is [1,1,B,2560].
        """
        b = hidden_states.shape[2]
        if tuple(hidden_states.shape) != (1, 1, b, 640 if self.residual_sharded else 2560) or not 1 <= b <= 32:
            raise ValueError("Decode expects [1,1,B,2560], 1<=B<=32")
        padded = math.ceil(b / 32) * 32
        x = ttnn.pad(hidden_states, [(0, 0), (0, 0), (0, padded - b), (0, 0)], 0)
        normed = self._norm(self._gather(x) if self.residual_sharded else x, "input_layernorm")
        packed = self._linear(normed, "attention_packed")
        qkv_width = 1280 if self.sparse else 1536
        gate_logits = packed[..., qkv_width : qkv_width + 1024]
        qkv = packed[..., :qkv_width]
        if self.sparse:
            qkv = ttnn.concat(
                [qkv, self._experts(normed, value=True, valid_tokens=b, logits=packed[..., qkv_width + 1024 :])], dim=-1
            )
        # SDPA's sharded reader uses its 8-wide reducer grid, not the physical
        # device width. Producer and consumer must agree on each user's core.
        grid = ttnn.num_cores_to_corerangeset(b, ttnn.CoreCoord(8, 8), True)
        mem = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(grid, [32, 128], ttnn.ShardOrientation.ROW_MAJOR),
        )
        q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
            qkv[:, :, :b, :], num_heads=8, num_kv_heads=2, memory_config=mem
        )
        if b > 8 and b % 8:
            # Native rotary launches the grid's bounding rectangle. Back every
            # launched core, including the inactive final-row slots.
            rectangle = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, (b - 1) // 8))})
            mem = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(rectangle, [32, 128], ttnn.ShardOrientation.ROW_MAJOR),
            )
            q, k = ttnn.to_memory_config(q, mem), ttnn.to_memory_config(k, mem)
        cos, sin = ttnn.to_memory_config(cos, mem), ttnn.to_memory_config(sin, mem)
        q, k = self._rope(q, cos, sin, decode=True), self._rope(k, cos, sin, decode=True)
        self._update_cache(k, v, current_pos, page_table, kv_cache)
        del k, v, cos, sin
        if page_table.shape[1] * kv_cache[0].shape[2] >= 262144:
            # The long page-table CB and k_chunk1024 nearly fill L1. Release
            # the transient query shard before SDPA; capacity stays unchanged.
            q = ttnn.to_memory_config(q, ttnn.DRAM_MEMORY_CONFIG)
        attention_compute, attention_program = self._decode_attention_config(b, page_table, kv_cache[0])
        out = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            q,
            kv_cache[0],
            kv_cache[1],
            self._attention_pages(page_table, kv_cache[0].shape[2]),
            cur_pos_tensor=current_pos,
            compute_kernel_config=attention_compute,
            program_config=attention_program,
        )
        out = ttnn.reshape(out, (1, 1, b, 1024))
        out = ttnn.pad(out, [(0, 0), (0, 0), (0, padded - b), (0, 0)], 0)
        return self._finish(x, gate_logits, out, valid_tokens=b)[:, :, :b, :]

    def _experts_expanded(self, x, *, logits, value=False, valid_tokens=None):
        # Tiles form independent sparse batches. An expert is active for a tile
        # iff at least one token selects it; per-token weights mask the result.
        tokens = x.shape[-2]
        e = self.config.mova_num_experts if value else self.config.num_experts
        k = self.config.mova_num_experts_per_tok if value else self.config.num_experts_per_tok
        name = "self_attn.v_router" if value else "mlp.gate"
        routes = self._route(logits, name, k, e)
        if valid_tokens is not None and valid_tokens < tokens:
            routes = ttnn.pad(routes[:, :, :valid_tokens, :], [(0, 0), (0, 0), (0, tokens - valid_tokens), (0, 0)], 0)
        groups = tokens // 32
        routes = ttnn.reshape(routes, (1, groups, 32, e))
        sparsity = ttnn.to_layout(ttnn.sum(routes, dim=2, keepdim=True), ttnn.ROW_MAJOR_LAYOUT)
        x = ttnn.reshape(x, (1, groups, 32, 2560))
        if value:
            out = self._sparse_mm(x, self.weights["value_experts"], sparsity)
            out = ttnn.reshape(out, (groups, e, 32, 256))
        else:
            gate_up = self._sparse_mm(x, self.weights["experts.gate_up"], sparsity)
            gate, up = gate_up[..., :192], gate_up[..., 192:]
            product = ttnn.reshape(
                ttnn.mul(gate, up, input_tensor_a_activations=[ttnn.UnaryOpType.SILU]), (groups, e, 32, 192)
            )
            down_sparsity = ttnn.reshape(sparsity, (1, 1, groups, e))
            out = self._sparse_mm(product, self.weights["experts.down_proj"], down_sparsity, sparse_input=True)
            out = ttnn.reshape(out, (groups, e, 32, 2560))
        if not value and tokens == 32:
            activ = ttnn.reshape(out, (e, 1, 32, 2560))
            scores = ttnn.to_layout(ttnn.reshape(routes, (32, 1, 1, e)), ttnn.ROW_MAJOR_LAYOUT)
            return ttnn.experimental.deepseek_moe_fast_reduce_nc_fused(
                activ,
                self.reduce_ids,
                self.reduce_map,
                reduce_dim=0,
                split_size=2560,
                # This primitive filters by EP ownership orthogonal to the axis.
                # On1x4 TP, axis1 retains zero-map scores on every column in row0.
                # Axis0 would silently keep only column0's partial expert output.
                cluster_axis=1,
                scores_tensor=scores,
                compute_kernel_config=self.compute,
            )[0]
        routes = ttnn.reshape(ttnn.permute(routes, (0, 1, 3, 2)), (groups, e, 32, 1))
        out = ttnn.experimental.fast_reduce_nc(
            ttnn.mul(out, routes, input_tensor_a_activations=[ttnn.UnaryOpType.SILU] if value else []), dims=[1]
        )
        out = ttnn.reshape(out, (1, 1, tokens, 256 if value else 2560))
        return ttnn.typecast(out, ttnn.bfloat16) if value and out.dtype != ttnn.bfloat16 else out

    def _experts_indexed(self, x, *, logits, value=False, valid_tokens=None):
        if valid_tokens != 1:
            return self._experts_expanded(x, logits=logits, value=value, valid_tokens=valid_tokens)
        e = self.config.mova_num_experts if value else self.config.num_experts
        k = self.config.mova_num_experts_per_tok if value else self.config.num_experts_per_tok
        name = "self_attn.v_router" if value else "mlp.gate"
        logits = logits[:, :, :1, :]
        weights, indices = ttnn.experimental.deepseek_prefill.moe_grouped_topk(
            logits,
            self.weights[name + ".bias"],
            n_groups=1,
            topk_groups=1,
            n_activated_experts=k,
            route_scale=2.5,
            epsilon=0.0,
            score_func="sigmoid",
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        ids = ttnn.to_layout(indices, ttnn.ROW_MAJOR_LAYOUT)
        placeholder = self.placeholders[e]
        if value:
            out = self._indexed_mm(x, self.weights["value_experts"], ids, placeholder, value=True)
        else:
            gu = self._indexed_mm(x, self.weights["experts.gate_up"], ids, placeholder)
            product = ttnn.mul(gu[..., :192], gu[..., 192:], input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
            out = self._indexed_mm(product, self.weights["experts.down_proj"], ids, placeholder, down=True)
        weights = ttnn.reshape(weights, (1, k, 1, 1))
        weighted = ttnn.mul(out, weights, input_tensor_a_activations=[ttnn.UnaryOpType.SILU] if value else [])
        result = ttnn.experimental.fast_reduce_nc(weighted, dims=[1])
        return ttnn.typecast(result, ttnn.bfloat16) if value and result.dtype != ttnn.bfloat16 else result

    def prefill_forward(self, hidden_states, *, cos, sin, page_table, kv_cache, start_pos=0):
        """Paged, chunked prefill. Logical sequence length is returned unchanged.

        ``start_pos`` is an absolute cached-prefix offset. ``cos/sin`` describe
        the new input only, [B,1,S,128]. Request pages must be exclusively owned.
        """
        b, s = hidden_states.shape[1], hidden_states.shape[2]
        if (
            hidden_states.shape[0] != 1
            or hidden_states.shape[3] != (640 if self.residual_sharded else 2560)
            or not 1 <= b <= 32
        ):
            raise ValueError("Prefill expects [1,B,S,2560], 1<=B<=32")
        if s < 1 or start_pos < 0 or start_pos + s > self.config.max_position_embeddings:
            raise ValueError("Prefill exceeds the advertised context or has an invalid length")
        page_size = kv_cache[0].shape[2]
        if page_size % 32 or self.chunk_size % page_size:
            raise ValueError("Page size must be tile-aligned and divide the physical chunk size")
        if page_table.shape[0] != b or page_table.shape[1] * page_size < start_pos + s:
            raise ValueError("Page table does not cover the requested batch and context")
        results = []
        for user in range(b):
            chunks = []
            offset = 0
            # Preserve partial prefix pages with individual paged updates until
            # a page-aligned fill is legal. Position slicing stays on device.
            while offset < s and (start_pos + offset) % kv_cache[0].shape[2]:
                chunks.append(
                    self.decode_forward(
                        hidden_states[:, user : user + 1, offset : offset + 1, :],
                        cos=cos[user : user + 1, :, offset : offset + 1, :],
                        sin=sin[user : user + 1, :, offset : offset + 1, :],
                        page_table=page_table[user : user + 1, :],
                        kv_cache=kv_cache,
                        current_pos=self.positions[start_pos + offset : start_pos + offset + 1],
                    )
                )
                # Do not retain one padded L1 tensor per continuation token.
                chunks[-1] = ttnn.to_memory_config(chunks[-1], ttnn.DRAM_MEMORY_CONFIG)
                offset += 1
            for offset in range(offset, s, self.chunk_size):
                count = min(s - offset, self.chunk_size)
                chunks.append(
                    self._prefill_chunk(
                        hidden_states[:, user : user + 1, offset : offset + count, :],
                        cos=cos[user : user + 1, :, offset : offset + count, :],
                        sin=sin[user : user + 1, :, offset : offset + count, :],
                        page_table=page_table[user : user + 1, :],
                        kv_cache=kv_cache,
                        start_pos=start_pos + offset,
                    )
                )
            # Short tails and continuation decode use the sharded decode residual.
            # Public prefill joins them with DRAM chunks and across users.
            chunks = [ttnn.to_memory_config(chunk, ttnn.DRAM_MEMORY_CONFIG) for chunk in chunks]
            results.append(ttnn.concat(chunks, dim=2) if len(chunks) > 1 else chunks[0])
        return ttnn.concat(results, dim=1) if b > 1 else results[0]

    def _experts(self, x, *, logits, value=False, valid_tokens=None):
        if value or x.shape[-2] == 32 or valid_tokens is not None:
            return super()._experts(x, logits=logits, value=value, valid_tokens=valid_tokens)
        return self._experts_ep(x, logits)

    def _experts_ep(self, x, logits):
        tokens = x.shape[-2]
        groups = tokens // 32
        routes = self._route(logits, "mlp.gate", 8, 100)
        routes = ttnn.pad(routes, [(0, 0), (0, 0), (0, 0), (0, 28)], 0)
        local_routes = ttnn.linear(
            routes, self.weights["experts.ep_selector"], dtype=ttnn.bfloat16, compute_kernel_config=self.compute
        )
        local_routes = ttnn.reshape(local_routes, (1, groups, 32, 32))
        sparsity = ttnn.to_layout(ttnn.sum(local_routes, dim=2, keepdim=True), ttnn.ROW_MAJOR_LAYOUT)

        def mm(inp, w, sparse, down=False):
            grid = self.options.get("ep_down_grid" if down else "ep_up_grid", [8, 5] if down else [8, 3])
            n = w.shape[-1]
            sub = self.options.get("ep_down_subblock" if down else "ep_up_subblock", 2)
            prog = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=tuple(grid),
                in0_block_w=self.options.get("ep_down_block" if down else "ep_up_block", 24 if down else 40),
                out_subblock_h=1,
                out_subblock_w=sub,
                out_block_h=1,
                out_block_w=sub,
                per_core_M=1,
                per_core_N=math.ceil(n / 32 / math.prod(grid)),
                fuse_batch=False,
                fused_activation=None,
                mcast_in0=True,
            )
            return ttnn.sparse_matmul(
                inp,
                w,
                sparsity=sparse,
                nnz=None,
                is_input_a_sparse=down,
                is_input_b_sparse=not down,
                program_config=prog,
                compute_kernel_config=self.matmul_compute["experts"],
                dtype=ttnn.bfloat8_b,
                memory_config=ttnn.L1_MEMORY_CONFIG if groups == 1 else ttnn.DRAM_MEMORY_CONFIG,
            )

        x = ttnn.reshape(x, (1, groups, 32, 2560))
        gu = mm(x, self.weights["experts.ep_gate_up"], sparsity)
        product = ttnn.reshape(
            ttnn.mul(gu[..., :768], gu[..., 768:], input_tensor_a_activations=[ttnn.UnaryOpType.SILU]),
            (groups, 32, 32, 768),
        )
        down = mm(product, self.weights["experts.ep_down"], ttnn.reshape(sparsity, (1, 1, groups, 32)), True)
        down = ttnn.reshape(down, (groups, 32, 32, 2560))
        scores = ttnn.reshape(ttnn.permute(local_routes, (0, 1, 3, 2)), (groups, 32, 32, 1))
        result = ttnn.experimental.fast_reduce_nc(ttnn.mul(down, scores), dims=[1])
        return ttnn.reshape(result, (1, 1, tokens, 2560))
