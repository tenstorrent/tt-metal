# SPDX-License-Identifier: Apache-2.0
"""TP4 decoder using the completed optimized decoder's phase and cache baseline.

The selected path keeps replicated BF16 residuals between layers. Weight conversion is setup
only; each device owns local query/KV heads and intermediate MLP channels.
"""

import math

from ttnn.types import BlackholeComputeKernelConfig

import ttnn
from models.common.modules.tt_ccl import TT_CCL

from .optimized_decoder import OptimizedDecoder


class MultichipDecoder(OptimizedDecoder):
    @classmethod
    def from_state_dict(cls, state_dict, *, hf_config, layer_idx, mesh_device, **kwargs):
        obj = cls()
        if mesh_device.get_num_devices() != 4:
            raise ValueError("Multichip decoder requires this run's TP4 mesh")
        candidate_baseline = kwargs.pop("_candidate_baseline", False)
        obj.decode_gather_buffers = kwargs.pop("decode_gather_buffers", {})
        obj.residual_dtype = kwargs.pop("residual_dtype", ttnn.bfloat16)
        obj.decode_ccl_dtype = kwargs.pop("decode_ccl_dtype", ttnn.bfloat16)
        norm_dtype = kwargs.pop("norm_dtype", ttnn.bfloat16)
        obj.topology = ttnn.Topology.Linear if candidate_baseline else ttnn.Topology.Ring
        obj.residual_sharded = kwargs.pop("residual_sharded", False)
        obj.prefill_ccl_bfp8 = kwargs.pop("prefill_ccl_bfp8", not candidate_baseline)
        cache_dtype = kwargs.pop("cache_dtype", ttnn.bfloat8_b)
        activation_dtype = kwargs.pop("activation_dtype", ttnn.bfloat8_b)
        obj.carry_reduced = not candidate_baseline
        obj.cache_position_modulo = kwargs.pop("sliding_cache_tokens", None)
        if obj.cache_position_modulo not in (None, 2048):
            raise ValueError("Sliding cyclic cache supports a 2048-token physical pool")
        obj.ccl = TT_CCL(mesh_device)
        c = hf_config.text_config if hasattr(hf_config, "text_config") else hf_config
        policy = kwargs.pop("precision", "bfp4")
        policies = {
            "bf16": (ttnn.bfloat16, ttnn.MathFidelity.HiFi4),
            "bfp8": (ttnn.bfloat8_b, ttnn.MathFidelity.HiFi2),
            "bfp8_lofi": (ttnn.bfloat8_b, ttnn.MathFidelity.LoFi),
            "bfp4": (ttnn.bfloat4_b, ttnn.MathFidelity.LoFi),
            "bfp4_hifi2": (ttnn.bfloat4_b, ttnn.MathFidelity.HiFi2),
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
        obj.nq, obj.hidden = c.num_attention_heads // 4, c.hidden_size
        obj.nkv //= 4
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
        if not candidate_baseline:
            import torch

            # Pad each TP shard separately, retaining channel ownership. This
            # changes 21 tiles/DRAM bank to22, enabling two decode readers.
            state_dict = dict(state_dict)
            for name in ("mlp.gate_proj.weight", "mlp.up_proj.weight"):
                tensor = state_dict[name].reshape(4, 5376, 5376)
                state_dict[name] = torch.nn.functional.pad(tensor, (0, 0, 0, 256)).reshape(22528, 5376)
            tensor = state_dict["mlp.down_proj.weight"].reshape(5376, 4, 5376)
            state_dict["mlp.down_proj.weight"] = torch.nn.functional.pad(tensor, (0, 256)).reshape(5376, 22528)
        for name, tensor in state_dict.items():
            mapper = ttnn.ReplicateTensorToMesh(mesh_device)
            if name.endswith("proj.weight"):
                mapper = ttnn.ShardTensorToMesh(
                    mesh_device, dim=0 if name in ("self_attn.o_proj.weight", "mlp.down_proj.weight") else 1
                )
            elif obj.residual_sharded and "layernorm" in name:
                mapper = ttnn.ShardTensorToMesh(mesh_device, dim=3)
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
                    else norm_dtype
                ),
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                mesh_mapper=mapper,
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
        obj.cache_dtype = cache_dtype
        obj.prefill_activation_dtype = activation_dtype
        obj.activation_precision = {
            "prefill_gate_up_and_intermediate": str(activation_dtype),
            "residual_and_norm": "bfloat16",
            "decode": "bfloat16",
        }
        obj.geometry = {
            "residual_cores": 7 if obj.residual_sharded else 8,
            "self_attn.qkv_proj": {"cores": 28, "block": 6, "readers": 2},
            "self_attn.qk_proj": {"cores": 8, "block": 3, "readers": 2},
            "self_attn.o_proj": {"cores": 8, "block": 8 if obj.sliding_window else 16, "readers": 2},
            "mlp.gate_proj": {"cores": 28, "block": 6, "readers": 1},
            "mlp.up_proj": {"cores": 28, "block": 6, "readers": 1},
            "mlp.down_proj": {"cores": 8, "block": 7, "readers": 2},
            "prefill_q": 64 if cache_dtype == ttnn.bfloat16 and not obj.sliding_window else 128,
            "prefill_k": 128,
        }
        if not candidate_baseline:
            obj.geometry["self_attn.qk_proj"]["block"] = 21
            obj.geometry["mlp.gate_proj"]["readers"] = 2
            obj.geometry["mlp.up_proj"]["readers"] = 2
            obj.geometry["mlp.down_proj"]["block"] = 11
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
        obj.persistent_ccl = {}
        if not candidate_baseline and not obj.residual_sharded:
            # RS staging is per projection/signature. AG output storage is
            # shared across batch views and may be shared by sequential layers.
            # Every replay overwrites it before the consuming norm reads it.
            for batch in (1, 8, 16, 32):
                for name in ("self_attn.o_proj", "mlp.down_proj"):
                    full = ttnn.zeros(
                        (1, 1, batch, 5376),
                        device=mesh_device,
                        dtype=obj.decode_ccl_dtype,
                        layout=ttnn.TILE_LAYOUT,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    )
                    local = ttnn.zeros(
                        (1, 1, batch, 1344),
                        device=mesh_device,
                        dtype=obj.decode_ccl_dtype,
                        layout=ttnn.TILE_LAYOUT,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    )
                    staging = ttnn.experimental.reduce_scatter_minimal_async_create_intermediate_buffer(
                        full, 3, topology=obj.topology
                    )
                    if name not in obj.decode_gather_buffers:
                        obj.decode_gather_buffers[name] = ttnn.zeros(
                            (1, 1, 32, 5376),
                            device=mesh_device,
                            dtype=obj.decode_ccl_dtype,
                            layout=ttnn.TILE_LAYOUT,
                            memory_config=obj._working_memory(full, 8),
                        )
                    backing = obj.decode_gather_buffers[name]
                    if tuple(backing.shape) != (1, 1, 32, 5376) or backing.dtype != obj.decode_ccl_dtype:
                        raise ValueError("Decode gather workspace must match the requested CCL dtype and 32-row shape")
                    # All supported logical batches occupy one physical tile row.
                    # The next norm consumes this storage before another gather
                    # can overwrite it; returned residuals never alias it.
                    output = obj._sequence_view(backing, batch)
                    obj.persistent_ccl[(batch, name)] = ([staging[0], local, staging[1]], output)
        return obj

    def _gather(self, x, *, group=None):
        output = self.persistent_ccl.get((x.shape[-2], group), (None, None))[1] if self.decoding else None
        compressed = self.prefill_ccl_bfp8 and not self.decoding and x.shape[-1] > 128
        if compressed and x.dtype != ttnn.bfloat8_b:
            x = ttnn.typecast(x, ttnn.bfloat8_b)
        out = ttnn.experimental.all_gather_async(
            x,
            dim=3,
            persistent_output_buffer=output,
            multi_device_global_semaphore=self.ccl.get_and_cycle_ag_semaphore_handles(),
            barrier_semaphore=self.ccl.get_and_cycle_barrier_semaphore_handle(),
            topology=self.topology,
            num_links=2,
            num_workers_per_link=2 if self.decoding else None,
            num_buffers_per_channel=1,
            memory_config=output.memory_config() if output is not None else ttnn.DRAM_MEMORY_CONFIG,
        )

        return ttnn.typecast(out, self.residual_dtype) if compressed else out

    def _norm(self, x, name=None):
        rows = math.prod(tuple(x.padded_shape)[:-1])
        if not self.decoding and name and "layernorm" in name and x.shape[-1] == 5376 and rows <= 1024:
            height = math.ceil(rows / 256) * 32
            gy = math.ceil(rows / height)
            memory = ttnn.create_sharded_memory_config(
                (height, 672),
                ttnn.CoreGrid(x=8, y=gy),
                ttnn.ShardStrategy.BLOCK,
                ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            # Conversion owns a private buffer; the residual input stays intact.
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
        if not (self.residual_sharded and name and "layernorm" in name):
            return super()._norm(x, name)
        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
        stats = ttnn.rms_norm_pre_all_gather(x, compute_kernel_config=self.kernel, dtype=ttnn.bfloat16)
        stats = self._gather(stats)
        out = ttnn.rms_norm_post_all_gather(
            x,
            stats,
            epsilon=self.config.rms_norm_eps,
            weight=self.weights[name + ".weight"],
            compute_kernel_config=self.kernel,
        )
        if name in ("input_layernorm", "pre_feedforward_layernorm"):
            return self._gather(out)
        return ttnn.to_memory_config(out, self._working_memory(out, 7)) if self.decoding else out

    def _linear(self, x, name):
        if self.decoding and name in self.output_padding and x.shape[-2] <= 32:
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
                dtype=self.residual_dtype,
            )
            out = ttnn.to_memory_config(out, ttnn.L1_MEMORY_CONFIG)
            out = ttnn.slice(out, [0] * len(out.shape), [*tuple(out.shape)[:-1], self.hidden])
        else:
            out = super()._linear(x, name)
        if name not in self.output_padding:
            return out
        out = ttnn.to_memory_config(out, ttnn.DRAM_MEMORY_CONFIG)
        if self.decoding and out.dtype != self.decode_ccl_dtype:
            out = ttnn.typecast(out, self.decode_ccl_dtype)
        elif self.prefill_ccl_bfp8 and not self.decoding:
            out = ttnn.typecast(out, ttnn.bfloat8_b)
        out = ttnn.experimental.reduce_scatter_minimal_async(
            out,
            dim=3,
            persistent_output_buffers=(
                self.persistent_ccl[(out.shape[-2], name)][0]
                if self.decoding and (out.shape[-2], name) in self.persistent_ccl
                else None
            ),
            multi_device_global_semaphore=self.ccl.get_and_cycle_rs_semaphore_handles(),
            barrier_semaphore=self.ccl.get_and_cycle_barrier_semaphore_handle(),
            topology=self.topology,
            num_links=2,
            num_workers_per_link=2 if self.decoding else None,
            num_buffers_per_channel=1,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        if not self.residual_sharded:
            out = self._gather(out, group=name)
        if out.dtype != self.residual_dtype:
            out = ttnn.typecast(out, self.residual_dtype)
        if self.decoding and not self.residual_sharded and self.carry_reduced:
            out = ttnn.to_memory_config(out, self._working_memory(out, 8))
        return out

    def _finish(self, x, attention, *, decode):
        if not decode:
            if self.nq > 16 and self.head_dim == 512:
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

    def _decode_attention_forward(self, x, *, cos, sin, kv_cache, page_table, current_pos):
        if not self.sliding_window or self.cache_position_modulo is None:
            return super()._decode_attention_forward(
                x, cos=cos, sin=sin, kv_cache=kv_cache, page_table=page_table, current_pos=current_pos
            )
        # The fused update op requires logical page count <= physical page count.
        # Generic modulo-aware updates support the bounded sliding pool while
        # RoPE and the attention mask continue to use absolute positions.
        q, k, v = self._qkv(x, cos, sin, decode=True)
        batch = x.shape[2]
        for cache, value in zip(kv_cache, (k, v)):
            value = ttnn.to_memory_config(value, self._decode_memory(batch))
            ttnn.experimental.paged_update_cache(
                cache,
                value,
                update_idxs_tensor=current_pos,
                page_table=page_table,
                cache_position_modulo=self.cache_position_modulo,
            )
        attention = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            q,
            *kv_cache,
            page_table,
            cur_pos_tensor=current_pos,
            program_config=self.decode_programs[(batch > 8, 64)],
            scale=1.0,
            sliding_window_size=self.sliding_window,
            cache_position_modulo=self.cache_position_modulo,
            compute_kernel_config=self.sdpa_kernel,
        )
        return self._finish(x, attention, decode=True)
