# SPDX-License-Identifier: Apache-2.0
"""Single-device Gemma 4 31B decoder with explicit phase-specific TTNN policies.

Weight conversion and persistent metadata are setup-only. Both forwards consume
TTNN tensors; logical prompt tails and paged-cache ownership stay on device.
"""

import math

from ttnn.types import BlackholeComputeKernelConfig

import ttnn
from models.common.lightweightmodule import LightweightModule

HF_MODEL = "google/gemma-4-31B-it"
HF_REVISION = "842da3794eaa0b77d5f08bae87a17459d91ff475"


class OptimizedDecoder(LightweightModule):
    @classmethod
    def from_state_dict(cls, state_dict, *, hf_config, layer_idx, mesh_device, **kwargs):
        obj = cls()
        c = hf_config.text_config if hasattr(hf_config, "text_config") else hf_config
        policy = kwargs.pop("precision", "bfp4")
        policies = {
            "bf16": (ttnn.bfloat16, ttnn.MathFidelity.HiFi4),
            "bfp8": (ttnn.bfloat8_b, ttnn.MathFidelity.HiFi2),
            "bfp8_lofi": (ttnn.bfloat8_b, ttnn.MathFidelity.LoFi),
            "bfp4": (ttnn.bfloat4_b, ttnn.MathFidelity.LoFi),
        }
        attention_policy = kwargs.pop("attention_precision", "bfp4")
        down_policy = kwargs.pop("down_precision", policy)
        obj.precision = (attention_policy, policy, down_policy)
        obj.linear_kernels = {}
        for group, selected in zip(("attention", "mlp", "down"), obj.precision):
            _, fidelity = policies[selected]
            obj.linear_kernels[group] = BlackholeComputeKernelConfig(
                math_fidelity=fidelity,
                math_approx_mode=False,
                fp32_dest_acc_en=fidelity == ttnn.MathFidelity.HiFi4,
                packer_l1_acc=True,
            )
        if kwargs:
            raise TypeError(f"Unsupported decoder options: {tuple(kwargs)}")
        if (
            c.hidden_size,
            c.intermediate_size,
            c.num_attention_heads,
            c.head_dim,
            c.global_head_dim,
            c.num_key_value_heads,
            c.num_global_key_value_heads,
        ) != (5376, 21504, 32, 256, 512, 16, 4):
            raise ValueError("Expected pinned Gemma 4 31B text geometry")
        if c.enable_moe_block or c.num_kv_shared_layers or c.hidden_size_per_layer_input:
            raise ValueError("Pinned target has no MoE, KV sharing or per-layer input")
        if c.max_position_embeddings != 262144 or not c.attention_k_eq_v or c.attention_bias:
            raise ValueError("Expected pinned context and attention projection contract")
        if c.hidden_activation != "gelu_pytorch_tanh" or c.sliding_window != 1024:
            raise ValueError("Expected pinned activation and sliding-window contract")
        obj.config, obj.mesh_device = c, mesh_device
        obj.layer_type = c.layer_types[layer_idx]
        if obj.layer_type not in ("sliding_attention", "full_attention"):
            raise ValueError("Unknown target layer kind")
        obj.sliding_window = c.sliding_window if obj.layer_type == "sliding_attention" else None
        obj.head_dim = c.head_dim if obj.sliding_window else c.global_head_dim
        obj.nkv = c.num_key_value_heads if obj.sliding_window else c.num_global_key_value_heads
        obj.nq, obj.hidden = c.num_attention_heads, c.hidden_size
        obj.kernel = BlackholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
        )
        obj.output_padding = {"self_attn.o_proj", "mlp.down_proj"}
        obj.weights = {}
        prefix = f"model.language_model.layers.{layer_idx}."
        if any(name.startswith(prefix) for name in state_dict):
            state_dict = {
                name.removeprefix(prefix): tensor for name, tensor in state_dict.items() if name.startswith(prefix)
            }
        expected = {
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "pre_feedforward_layernorm.weight",
            "post_feedforward_layernorm.weight",
            "self_attn.q_norm.weight",
            "self_attn.k_norm.weight",
            "layer_scalar",
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.o_proj.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
        }
        if obj.sliding_window:
            expected.add("self_attn.v_proj.weight")
        if set(state_dict) != expected:
            raise ValueError(
                f"Layer weight keys mismatch: missing={expected-set(state_dict)}, extra={set(state_dict)-expected}"
            )
        for name, tensor in state_dict.items():
            if name.endswith("proj.weight"):
                if name.removesuffix(".weight") in obj.output_padding:
                    # Two DRAM readers need an even bank-shard width. Extra
                    # output channels are zero and are removed before norms.
                    import torch

                    tensor = torch.nn.functional.pad(tensor, (0, 0, 0, (-tensor.shape[0]) % 512))
                tensor = tensor.T.contiguous()
            elif name == "layer_scalar":
                obj.scalar = float(tensor.item())
                continue
            else:
                tensor = tensor.reshape(1, 1, 1, -1)
            obj.weights[name] = ttnn.from_torch(
                tensor,
                dtype=(
                    policies[
                        (
                            attention_policy
                            if name.startswith("self_attn.")
                            else down_policy
                            if name == "mlp.down_proj.weight"
                            else policy
                        )
                    ][0]
                    if name.endswith("proj.weight")
                    else ttnn.bfloat16
                ),
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        if obj.sliding_window:
            # The same packed projection serves both phases; keep one copy.
            obj.weights["self_attn.qkv_proj.weight"] = ttnn.concat(
                [obj.weights["self_attn." + name + "_proj.weight"] for name in ("q", "k", "v")], dim=-1
            )
            for name in ("q", "k", "v"):
                del obj.weights["self_attn." + name + "_proj.weight"]
        else:
            # Full attention shares raw K/V. Pack only Q and K; discard the
            # individual device weights so persistent weight bytes do not grow.
            obj.weights["self_attn.qk_proj.weight"] = ttnn.concat(
                [obj.weights["self_attn.q_proj.weight"], obj.weights["self_attn.k_proj.weight"]], dim=-1
            )
            del obj.weights["self_attn.q_proj.weight"], obj.weights["self_attn.k_proj.weight"]
        grid = mesh_device.compute_with_storage_grid_size()
        # Prepare row identities once. Prefill takes a device slice for its
        # logical batch; no host index construction occurs in either forward.
        obj.prefill_batch_indices = ttnn.arange(
            0, grid.x * grid.y, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device
        )
        obj.decoding = False
        obj.carry = True
        obj.cache_dtype = ttnn.bfloat8_b
        obj.activation_precision = {
            "prefill_gate_up_and_intermediate": "bfloat8_b",
            "residual_and_norm": "bfloat16",
            "decode": "bfloat16",
        }
        obj.geometry = {
            "residual_cores": 8,
            "self_attn.qkv_proj": {"cores": 28, "block": 6, "readers": 2},
            "self_attn.qk_proj": {"cores": 8, "block": 3, "readers": 2},
            "self_attn.o_proj": {"cores": 8, "block": 16, "readers": 2},
            "mlp.gate_proj": {"cores": 56, "block": 6, "readers": 2},
            "mlp.up_proj": {"cores": 56, "block": 6, "readers": 2},
            "mlp.down_proj": {"cores": 8, "block": 14, "readers": 2},
            "prefill_q": 128,
            "prefill_k": 128,
        }
        dram_grid = mesh_device.dram_grid_size()
        obj.banks = dram_grid.x * dram_grid.y
        dram_cores = ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram_grid.x - 1, dram_grid.y - 1))]
        )
        for name, weight in list(obj.weights.items()):
            if name.endswith("proj.weight") or name.endswith("prefill_qkv.weight"):
                k, n = tuple(weight.shape)[-2:]
                memory = ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                    ttnn.BufferType.DRAM,
                    ttnn.ShardSpec(
                        dram_cores, (k, math.ceil(n / (32 * obj.banks)) * 32), ttnn.ShardOrientation.ROW_MAJOR
                    ),
                )
                obj.weights[name] = ttnn.to_memory_config(weight, memory)
        obj.sdpa_kernel = BlackholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
        )

        obj.prefill_sdpa_kernel = BlackholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=True
        )

        def decode_program(size, chunk):
            return ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=size,
                q_chunk_size=0,
                k_chunk_size=chunk,
                max_cores_per_head_batch=16,
                exp_approx_mode=False,
            )

        obj.decode_programs = {
            (large, chunk): decode_program((grid.x, grid.y) if large else (8, 8), chunk)
            for large in (False, True)
            for chunk in (32, 64, 128)
        }
        return obj

    def allocate_cache(self, *, num_blocks, page_size=32):
        if page_size != 32 or num_blocks < 1:
            raise ValueError("Expected positive 32-token paged capacity")
        return tuple(
            ttnn.zeros(
                (num_blocks, self.nkv, page_size, self.head_dim),
                dtype=self.cache_dtype,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            for _ in range(2)
        )

    def _norm(self, x, name=None):
        if self.decoding and self.carry and x.memory_config().memory_layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED:
            shard = x.memory_config().shard_spec
            box = shard.grid.bounding_box().end
            width = shard.shape[1] // 32
            sub = max(v for v in (1, 2, 4, 8) if width % v == 0)
            pc = ttnn.LayerNormShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=(box.x + 1, box.y + 1),
                subblock_w=sub,
                block_h=shard.shape[0] // 32,
                block_w=width,
                inplace=False,
            )
            return ttnn.rms_norm(
                x,
                epsilon=self.config.rms_norm_eps,
                weight=self.weights[name + ".weight"] if name else None,
                compute_kernel_config=self.kernel,
                program_config=pc,
                memory_config=x.memory_config(),
            )
        return ttnn.rms_norm(
            x,
            epsilon=self.config.rms_norm_eps,
            weight=self.weights[name + ".weight"] if name else None,
            compute_kernel_config=self.kernel,
        )

    def _linear(self, x, name):
        weight = self.weights[name + ".weight"]
        k, n = tuple(weight.shape)[-2:]
        group = "attention" if name.startswith("self_attn.") else "down" if name == "mlp.down_proj" else "mlp"
        kernel = self.linear_kernels[group]
        if not self.decoding:
            if self.prefill_l1:
                x = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)
            grid = self.mesh_device.compute_with_storage_grid_size()
            config = ttnn.MinimalMatmulConfig(
                M_block_size=4,
                K_block_size=8,
                N_block_size=16,
                subblock_h=1,
                subblock_w=8,
                compute_with_storage_grid_size=ttnn.CoreCoord(grid.x, grid.y),
            )
            out = ttnn.experimental.minimal_matmul(
                input_tensor=x,
                weight_tensor=weight,
                config=config,
                compute_kernel_config=kernel,
                memory_config=(
                    ttnn.L1_MEMORY_CONFIG
                    if self.prefill_l1 and (name.startswith("mlp.") or name == "self_attn.o_proj")
                    else ttnn.DRAM_MEMORY_CONFIG
                ),
                dtype=(
                    getattr(self, "prefill_activation_dtype", ttnn.bfloat8_b)
                    if name in ("mlp.gate_proj", "mlp.up_proj")
                    else getattr(self, "residual_dtype", ttnn.bfloat16)
                ),
            )
            if name in self.output_padding:
                out = ttnn.slice(out, [0] * len(out.shape), [*tuple(out.shape)[:-1], self.hidden])
            return out
        if x.shape[-2] > 32:
            # The DRAM-sharded decode factory accepts one physical M tile.
            # The explicit 2D path preserves larger logical batches and policy.
            x = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)
            gy = min(10, math.ceil(x.shape[-2] / 32))
            pn = math.ceil(n / 32 / self.banks)
            sub = max(v for v in (1, 2, 4, 8) if pn % v == 0)
            config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=(self.banks, gy),
                in0_block_w=4,
                out_subblock_h=1,
                out_subblock_w=sub,
                per_core_M=math.ceil(x.shape[-2] / 32 / gy),
                per_core_N=pn,
                transpose_mcast=False,
                fuse_batch=False,
            )
            out = ttnn.linear(
                x,
                weight,
                program_config=config,
                compute_kernel_config=kernel,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                dtype=getattr(self, "residual_dtype", ttnn.bfloat16),
            )
        else:
            role = self.geometry[name]
            x = ttnn.to_memory_config(x, self._working_memory(x, role["cores"]))
            config = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                in0_block_w=role["block"],
                per_core_M=1,
                per_core_N=math.ceil(n / 32 / role["cores"]),
                num_workers_per_dram_bank=role["readers"],
            )
            out = ttnn.linear(
                x,
                weight,
                program_config=config,
                compute_kernel_config=kernel,
                memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                dtype=getattr(self, "residual_dtype", ttnn.bfloat16),
            )
        if name in self.output_padding:
            out = ttnn.to_memory_config(out, ttnn.L1_MEMORY_CONFIG)
            out = ttnn.slice(out, [0] * len(out.shape), [*tuple(out.shape)[:-1], self.hidden])
            return ttnn.to_memory_config(out, self._working_memory(out, 8))
        if name.startswith("mlp."):
            return out
        return ttnn.to_memory_config(out, ttnn.L1_MEMORY_CONFIG)

    def _decode_memory(self, batch, *, offset=0):
        grid = self.mesh_device.compute_with_storage_grid_size()
        cores = ttnn.CoreRangeSet(
            [
                ttnn.CoreRange(ttnn.CoreCoord(i % grid.x, i // grid.x), ttnn.CoreCoord(i % grid.x, i // grid.x))
                for i in range(offset, offset + batch)
            ]
        )
        return ttnn.create_sharded_memory_config(
            (32, self.head_dim),
            cores,
            ttnn.ShardStrategy.HEIGHT,
            ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

    def _rope(self, x, cos, sin, *, decode):
        if decode:
            return ttnn.experimental.rotary_embedding_hf(
                ttnn.to_memory_config(x, self._decode_memory(x.shape[1])),
                cos,
                sin,
                is_decode_mode=True,
                compute_kernel_config=self.kernel,
            )
        batch, heads, length, dim = tuple(x.shape)
        # RoPE's shared-cache kernel requires [1,1,S,D]. Flatten batch into
        # sequence, preserving each request's own absolute-position cos/sin.
        flat = ttnn.reshape(ttnn.permute(x, (1, 0, 2, 3)), (1, heads, batch * length, dim)) if batch > 1 else x
        out = ttnn.experimental.rotary_embedding(
            flat,
            ttnn.reshape(cos, (1, 1, batch * length, dim)),
            ttnn.reshape(sin, (1, 1, batch * length, dim)),
            compute_kernel_config=self.kernel,
        )
        return ttnn.permute(ttnn.reshape(out, (heads, batch, length, dim)), (1, 0, 2, 3)) if batch > 1 else out

    def _qkv(self, x, cos, sin, *, decode):
        if decode:
            mem = self._decode_memory(x.shape[2])
            cos, sin = ttnn.to_memory_config(cos, mem), ttnn.to_memory_config(sin, mem)
        z = self._norm(x, "input_layernorm")
        if decode and x.shape[-2] > 32:
            z = ttnn.to_memory_config(z, ttnn.L1_MEMORY_CONFIG)
        if self.sliding_window and not decode:
            q, k, v = ttnn.experimental.nlp_create_qkv_heads(
                self._linear(z, "self_attn.qkv_proj"),
                num_heads=self.nq,
                num_kv_heads=self.nkv,
                transpose_k_heads=False,
            )
        elif self.sliding_window:
            projected = self._linear(z, "self_attn.qkv_proj")
            if x.shape[2] <= 32:
                q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
                    projected,
                    num_heads=self.nq,
                    num_kv_heads=self.nkv,
                    memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
                    overlap_qk_coregrid=True,
                )
                # Head RMSNorm requires interleaved input in this runtime.
                q, k, v = [ttnn.to_memory_config(t, ttnn.L1_MEMORY_CONFIG) for t in (q, k, v)]
            else:
                # The dedicated decode splitter supports at most one batch tile.
                parts = []
                offset = 0
                for heads in (self.nq, self.nkv, self.nkv):
                    width = heads * self.head_dim
                    part = ttnn.slice(projected, (0, 0, 0, offset), (*tuple(projected.shape)[:-1], offset + width))
                    parts.append(ttnn.reshape(part, (1, x.shape[2], heads, self.head_dim)))
                    offset += width
                q, k, v = parts
        else:
            qk = self._linear(z, "self_attn.qk_proj")
            if decode and 16 <= x.shape[2] <= 32:
                width = self.nq * self.head_dim
                kp = ttnn.slice(qk, (0, 0, 0, width), tuple(qk.shape))
                q, unused_k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
                    ttnn.concat([qk, kp], dim=-1),
                    num_heads=self.nq,
                    num_kv_heads=self.nkv,
                    memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
                    overlap_qk_coregrid=True,
                )
                # The decode splitter has no tied-KV mode. V carries the shared
                # raw projection; only Q and V need interleaved norm inputs.
                q, v = [ttnn.to_memory_config(t, ttnn.L1_MEMORY_CONFIG) for t in (q, v)]
                k = v
                del unused_k
            elif decode:
                width = self.nq * self.head_dim
                qp = ttnn.slice(qk, (0, 0, 0, 0), (*tuple(qk.shape)[:-1], width))
                kp = ttnn.slice(qk, (0, 0, 0, width), tuple(qk.shape))
                q = ttnn.reshape(qp, (1, x.shape[2], self.nq, self.head_dim))
                k = v = ttnn.reshape(kp, (1, x.shape[2], self.nkv, self.head_dim))
            else:
                q, k, v = ttnn.experimental.nlp_create_qkv_heads(
                    qk,
                    num_heads=self.nq,
                    num_kv_heads=self.nkv,
                    transpose_k_heads=False,
                    kv_tied=True,
                )
        q = self._rope(self._norm(q, "self_attn.q_norm"), cos, sin, decode=decode)
        v = self._norm(v)
        # Full attention normalizes the same raw K/V vector. Reuse its reduction;
        # the extra BF16 rounding is covered by real-weight and context PCC tests.
        k = (
            self._norm(k, "self_attn.k_norm")
            if self.sliding_window
            else ttnn.multiply(v, self.weights["self_attn.k_norm.weight"])
        )
        k = self._rope(k, cos, sin, decode=decode)
        if decode and not self.sliding_window:
            # The measured DRAM-Q path is faster and decouples SDPA reducers
            # from the RoPE shard grid.
            q = ttnn.to_memory_config(q, ttnn.DRAM_MEMORY_CONFIG)
        if not decode:
            k, v = ttnn.typecast(k, self.cache_dtype), ttnn.typecast(v, self.cache_dtype)
        return q, k, v

    def _finish(self, x, attention, *, decode):
        if not decode:
            if self.head_dim == 512:
                # A single 32-head concatenation needs >2MiB L1 at D512.
                # Two 16-head groups stay within the hardware's per-core L1.
                attention = ttnn.concat(
                    [
                        ttnn.transformer.concatenate_heads(
                            ttnn.slice(
                                attention,
                                (0, start, 0, 0),
                                (attention.shape[0], start + 16, attention.shape[2], self.head_dim),
                            )
                        )
                        for start in (0, 16)
                    ],
                    dim=-1,
                )
            else:
                attention = ttnn.transformer.concatenate_heads(attention)
        attention = ttnn.reshape(attention, (*tuple(x.shape)[:-1], self.nq * self.head_dim))
        add = ttnn.add_ if not decode and self.prefill_l1 else ttnn.add
        x = add(x, self._norm(self._linear(attention, "self_attn.o_proj"), "post_attention_layernorm"))
        z = self._norm(x, "pre_feedforward_layernorm")
        if self.decoding and x.shape[-2] > 32:
            z = ttnn.to_memory_config(z, ttnn.L1_MEMORY_CONFIG)
        elif self.decoding:
            role = "mlp.gate_proj"
            count = self.geometry.get(role, self.geometry.get("default", {})).get("cores", 8)
            z = ttnn.to_memory_config(z, self._working_memory(z, count))
        gate, up = self._linear(z, "mlp.gate_proj"), self._linear(z, "mlp.up_proj")
        multiply = ttnn.multiply_ if not decode and self.prefill_l1 else ttnn.multiply
        intermediate = multiply(gate, up, input_tensor_a_activations=[ttnn.UnaryOpType.GELU_TANH])
        del gate, up, z
        mlp = self._linear(intermediate, "mlp.down_proj")
        return ttnn.add(
            x,
            self._norm(mlp, "post_feedforward_layernorm"),
            activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.MUL_UNARY_SFPU, self.scalar)],
        )

    @staticmethod
    def _sequence_view(x, length):
        # Only the logical row count changes; backing storage and tile geometry
        # stay fixed. No FillPad program or new persistent buffer is needed when
        # a request changes logical length inside its 32-row physical bucket.
        logical = list(x.shape)
        padded = list(x.padded_shape)
        if (length + 31) // 32 * 32 != padded[-2]:
            raise ValueError("Sequence view must preserve physical tile storage")
        logical[-2] = length
        return ttnn.reshape(x, ttnn.Shape(logical), ttnn.Shape(padded), skip_padding_fill=True)

    def _prefill_indices(self, batch):
        if batch == 1 or batch > self.prefill_batch_indices.shape[0]:
            return None
        return ttnn.slice(self.prefill_batch_indices, (0,), (batch,))

    def prefill_forward(self, x, *, cos, sin, kv_cache, page_table, chunk_start=0):
        """Prefill arbitrary logical [B,1,S,H] in bounded 1024-token chunks.

        RoPE inputs are [B,1,S,D], already evaluated at absolute positions.
        Explicit continuation starts at a page boundary and retains prefix pages.
        All chunk planning uses shape metadata; tensor work stays on device.
        """
        b, _, length, _ = tuple(x.shape)
        page_size = kv_cache[0].shape[2]
        if length < 1 or chunk_start % page_size:
            raise ValueError("Empty prefill or continuation outside a page boundary")
        if chunk_start + length > self.config.max_position_embeddings:
            raise ValueError("Context exceeds pinned HF contract")
        batch_idx_tensor = self._prefill_indices(b)
        outputs = []
        for offset in range(0, length, 1024):
            valid = min(1024, length - offset)
            physical = (valid + 31) // 32 * 32

            def chunk(t):
                physical_input = self._sequence_view(t, t.padded_shape[2])
                return ttnn.slice(physical_input, (0, 0, offset, 0), (b, 1, offset + physical, t.shape[3]))

            first_page = (chunk_start + offset) // page_size
            table = ttnn.slice(page_table, (0, first_page), (b, first_page + (physical + page_size - 1) // page_size))
            outputs.append(
                self.prefill_chunk_forward(
                    chunk(x),
                    cos=chunk(cos),
                    sin=chunk(sin),
                    kv_cache=kv_cache,
                    page_table=page_table,
                    chunk_page_table=table,
                    chunk_start=chunk_start + offset,
                    valid_tokens=valid,
                    batch_idx_tensor=batch_idx_tensor,
                )
            )
        physical_outputs = [self._sequence_view(y, y.padded_shape[2]) for y in outputs]
        combined = physical_outputs[0] if len(physical_outputs) == 1 else ttnn.concat(physical_outputs, dim=2)
        return self._sequence_view(combined, length)

    def prefill_chunk_forward(
        self,
        x,
        *,
        cos,
        sin,
        kv_cache,
        page_table,
        chunk_page_table,
        chunk_start,
        valid_tokens,
        batch_idx_tensor=None,
        attention_page_table=None,
        chunk_start_idx_tensor=None,
    ):
        """x [B,1,S,H]; RoPE [B,1,S,D]; returns [B,1,valid_tokens,H].

        S is the padded physical length; 0 < valid_tokens <= S. Each request
        owns distinct cache pages. Resume retains all prefix pages and starts at
        a page boundary. The final partial page is private to this request.

        A prepared attention_page_table bypasses request-dependent table slices.
        For sliding attention it maps the preceding window plus the current
        chunk, rebased at max(0, chunk_start - sliding_window). The optional
        device INT32 [1] chunk_start_idx_tensor must contain that view's offset
        (min(chunk_start, sliding_window), or chunk_start for full attention).
        Callers refresh both persistent inputs before execution; chunk_start
        remains the matching host value for admission and kernel geometry.
        """
        if not 0 < valid_tokens <= x.shape[2] or chunk_start % kv_cache[0].shape[2]:
            raise ValueError("Invalid logical length or non-page-aligned chunk start")
        if chunk_start + valid_tokens > self.config.max_position_embeddings:
            raise ValueError("Context exceeds pinned HF contract")
        if chunk_start_idx_tensor is not None and attention_page_table is None:
            raise ValueError("Tensor-offset prefill requires a prepared attention page table")
        if attention_page_table is not None:
            attention_start = min(chunk_start, self.sliding_window) if self.sliding_window else chunk_start
            if attention_start + x.shape[2] > attention_page_table.shape[-1] * kv_cache[0].shape[2]:
                raise ValueError("Prepared attention page table does not cover the physical chunk")
        # Bound live interleaved L1 tensors by total physical token rows.
        # Own the residual buffer before in-place operations; callers retain x.
        self.prefill_l1 = x.shape[0] * x.padded_shape[2] <= 1024
        if self.prefill_l1:
            x = ttnn.clone(x, memory_config=ttnn.L1_MEMORY_CONFIG)
        q, k, v = self._qkv(x, cos, sin, decode=False)
        if batch_idx_tensor is None:
            batch_idx_tensor = self._prefill_indices(x.shape[0])
        if batch_idx_tensor is not None:
            ttnn.experimental.paged_fill_cache(kv_cache[0], k, chunk_page_table, batch_idx_tensor=batch_idx_tensor)
            ttnn.experimental.paged_fill_cache(kv_cache[1], v, chunk_page_table, batch_idx_tensor=batch_idx_tensor)
        else:
            for b in range(x.shape[0]):
                kb = ttnn.slice(k, (b, 0, 0, 0), (b + 1, self.nkv, x.shape[2], self.head_dim))
                vb = ttnn.slice(v, (b, 0, 0, 0), (b + 1, self.nkv, x.shape[2], self.head_dim))
                ttnn.experimental.paged_fill_cache(kv_cache[0], kb, chunk_page_table, batch_idx=b)
                ttnn.experimental.paged_fill_cache(kv_cache[1], vb, chunk_page_table, batch_idx=b)
        if attention_page_table is None:
            attention_page_table, attention_start = self._prefill_attention_view(x, kv_cache, page_table, chunk_start)
        attention_offset = (
            {"chunk_start_idx": attention_start}
            if chunk_start_idx_tensor is None
            else {"chunk_start_idx_tensor": chunk_start_idx_tensor}
        )
        a = ttnn.transformer.chunked_scaled_dot_product_attention(
            q,
            *kv_cache,
            attention_page_table,
            **attention_offset,
            scale=1.0,
            sliding_window_size=self.sliding_window,
            program_config=ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=(8, 8),
                q_chunk_size=32 if attention_start % 128 else self.geometry["prefill_q"],
                k_chunk_size=32 if attention_start % 128 else self.geometry["prefill_k"],
            ),
            compute_kernel_config=self.prefill_sdpa_kernel,
        )
        out = self._finish(x, a, decode=False)
        out = ttnn.to_memory_config(out, ttnn.DRAM_MEMORY_CONFIG)
        return self._sequence_view(out, valid_tokens)

    def _prefill_attention_view(self, x, kv_cache, page_table, chunk_start):
        attention_page_table = page_table
        attention_start = chunk_start
        if self.sliding_window and chunk_start >= self.sliding_window:
            # Rebase only SDPA's page view. RoPE and cache writes stay absolute.
            # Retain the preceding window plus this chunk, bounding the attention page shape and keeping
            # masked historical prefix work out of either SDPA implementation.
            base = chunk_start - self.sliding_window
            attention_start = self.sliding_window
            attention_page_table = ttnn.slice(
                page_table,
                (0, base // kv_cache[0].shape[2]),
                (x.shape[0], (chunk_start + x.shape[2]) // kv_cache[0].shape[2]),
            )
        return attention_page_table, attention_start

    def decode_forward(self, x, **kwargs):
        self.decoding = True
        if self.carry:
            x = ttnn.to_memory_config(x, self._working_memory(x, self.geometry.get("residual_cores", 8)))
        try:
            return self._decode_attention_forward(x, **kwargs)
        finally:
            self.decoding = False

    def _working_memory(self, x, count):
        grid = self.mesh_device.compute_with_storage_grid_size()
        gx = max(v for v in range(1, min(grid.x, count) + 1) if count % v == 0 and count // v <= grid.y)
        cores = ttnn.CoreGrid(x=gx, y=count // gx)
        return ttnn.create_sharded_memory_config(
            (math.ceil(x.shape[-2] / 32) * 32, x.shape[-1] // count),
            cores,
            ttnn.ShardStrategy.WIDTH,
            ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

    def _decode_attention_forward(self, x, *, cos, sin, kv_cache, page_table, current_pos):
        """x [1,1,B,H], RoPE [1,B,1,D], int32 positions [B], returns x shape.

        Positions are absolute token indices. Updates include the current token;
        the sliding mask is applied in absolute position space. No host reads.
        """
        q, k, v = self._qkv(x, cos, sin, decode=True)
        batch = x.shape[2]
        # Fused cache update requires disjoint core sets. K already occupies
        # the first batch cores after RoPE; V uses the following batch cores.
        grid = self.mesh_device.compute_with_storage_grid_size()
        if 2 * batch <= grid.x * grid.y:
            v_sharded = ttnn.to_memory_config(v, self._decode_memory(batch, offset=batch))
            ttnn.experimental.paged_fused_update_cache(
                kv_cache[0],
                k,
                kv_cache[1],
                v_sharded,
                update_idxs_tensor=current_pos,
                page_table=page_table,
            )
        else:
            # Preserve the functional interface beyond the tuned batch set when
            # two disjoint cache-update grids cannot fit on this device.
            ttnn.experimental.paged_update_cache(kv_cache[0], k, update_idxs_tensor=current_pos, page_table=page_table)
            v_sharded = ttnn.to_memory_config(v, self._decode_memory(batch))
            ttnn.experimental.paged_update_cache(
                kv_cache[1], v_sharded, update_idxs_tensor=current_pos, page_table=page_table
            )
        del k, v, v_sharded
        # The reader fetches every tile in a K chunk, even masked future tiles.
        # Choose a chunk dividing owned page-table capacity so the final chunk
        # cannot index unowned page entries. Logical positions remain arbitrary.
        capacity = page_table.shape[1] * kv_cache[0].shape[2]
        preferred_chunk = 64 if self.sliding_window else 32
        chunk = max(size for size in (32, 64, 128) if size <= preferred_chunk and capacity % size == 0)
        large_grid = batch > (8 if self.sliding_window else 64)
        a = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            q,
            *kv_cache,
            page_table,
            cur_pos_tensor=current_pos,
            program_config=self.decode_programs[(large_grid, chunk)],
            scale=1.0,
            sliding_window_size=self.sliding_window,
            compute_kernel_config=self.sdpa_kernel,
        )
        return self._finish(x, a, decode=True)
