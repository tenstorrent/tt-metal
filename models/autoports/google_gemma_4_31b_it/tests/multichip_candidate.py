"""Same-contract layout, CCL dtype and geometry candidates for stage04."""

import ttnn
from models.autoports.google_gemma_4_31b_it.tt.multichip_decoder import MultichipDecoder
from models.autoports.google_gemma_4_31b_it.tt.optimized_decoder import OptimizedDecoder


class MultichipCandidate(MultichipDecoder):
    @classmethod
    def from_state_dict(cls, state_dict, **kwargs):
        candidate = kwargs.pop("candidate", {})
        if candidate.get("pad_mlp"):
            import torch

            state_dict = dict(state_dict)
            width = candidate.get("mlp_width", 5632)
            for name in ("mlp.gate_proj.weight", "mlp.up_proj.weight"):
                w = state_dict[name]
                state_dict[name] = torch.nn.functional.pad(w.reshape(4, 5376, 5376), (0, 0, 0, width - 5376)).reshape(
                    4 * width, 5376
                )
            w = state_dict["mlp.down_proj.weight"]
            state_dict["mlp.down_proj.weight"] = torch.nn.functional.pad(
                w.reshape(5376, 4, 5376), (0, width - 5376)
            ).reshape(5376, 4 * width)
        obj = super().from_state_dict(state_dict, _candidate_baseline=True, **kwargs)
        if candidate.get("pad_mlp"):
            for name in ("mlp.gate_proj", "mlp.up_proj"):
                obj.geometry[name] = {
                    "cores": 28,
                    "block": 6,
                    "readers": 3 if candidate.get("mlp_width") == 6144 else 2,
                }
            obj.geometry["mlp.down_proj"] = {
                "cores": 8,
                "block": 24 if candidate.get("mlp_width") == 6144 else 11,
                "readers": 2,
            }
        obj.candidate = candidate
        if candidate.get("fused_ag"):
            obj.ag_weights = {
                n: ttnn.to_memory_config(w, ttnn.DRAM_MEMORY_CONFIG)
                for n, w in obj.weights.items()
                if n.endswith("proj.weight") and n.removesuffix(".weight") not in obj.output_padding
            }
        if candidate.get("ring_all"):
            obj.topology = ttnn.Topology.Ring
        if candidate.get("fused_rows"):
            import torch

            obj.fused_weights = {}
            obj.fused_buffers = {}
            batch = candidate.get("batch", 1)
            for name in ("self_attn.o_proj", "mlp.down_proj"):
                w = state_dict[name + ".weight"]
                packed = (
                    torch.nn.functional.pad(w.reshape(4, 1344, w.shape[1]), (0, 0, 0, 192))
                    .reshape(6144, w.shape[1])
                    .T.contiguous()
                )
                host = ttnn.from_torch(
                    packed,
                    dtype=ttnn.bfloat4_b,
                    layout=ttnn.TILE_LAYOUT,
                    mesh_mapper=ttnn.ShardTensorToMesh(obj.mesh_device, dim=0),
                )
                obj.fused_weights[name] = ttnn.to_device(host, obj.mesh_device)
                obj.fused_buffers[name] = [
                    ttnn.zeros(
                        (2 if n == 6144 and not candidate.get("ring_fused") else 1, 1, 32 if n == 6144 else batch, n),
                        device=obj.mesh_device,
                        dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    )
                    for n in (6144, 1536)
                ]
        obj.geometry.update(candidate.get("geometry", {}))
        if candidate.get("fused_ag"):
            obj.geometry["mlp.gate_proj"]["cores"] = 7
        if candidate.get("gather_before_norm") and obj.residual_sharded:
            for name in ("input_layernorm.weight", "pre_feedforward_layernorm.weight"):
                tensor = state_dict[name].reshape(1, 1, 1, -1)
                obj.weights[name] = ttnn.to_device(
                    ttnn.from_torch(
                        tensor,
                        dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT,
                        mesh_mapper=ttnn.ReplicateTensorToMesh(obj.mesh_device),
                    ),
                    obj.mesh_device,
                )
        if candidate.get("packed_mlp"):
            pair = [
                ttnn.to_memory_config(obj.weights["mlp." + n + "_proj.weight"], ttnn.DRAM_MEMORY_CONFIG)
                for n in ("gate", "up")
            ]
            packed = ttnn.concat(pair, dim=-1)
            memory = obj.weights["mlp.gate_proj.weight"].memory_config()
            shard = memory.shard_spec
            spec = ttnn.ShardSpec(
                shard.grid, (packed.shape[-2], packed.shape[-1] // 8), ttnn.ShardOrientation.ROW_MAJOR
            )
            obj.weights["mlp.gate_up_proj.weight"] = ttnn.to_memory_config(
                packed, ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, spec)
            )
            obj.geometry["mlp.gate_up_proj"] = dict(obj.geometry["mlp.gate_proj"])
            del obj.weights["mlp.gate_proj.weight"], obj.weights["mlp.up_proj.weight"]
        if candidate.get("persistent_ccl"):
            obj.persistent_ccl = {}
            for batch in (1, 8, 16, 32):
                for group in ("attention", "mlp"):
                    full = ttnn.zeros(
                        (1, 1, batch, 5376),
                        device=obj.mesh_device,
                        dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    )
                    local = ttnn.zeros(
                        (1, 1, batch, 1344),
                        device=obj.mesh_device,
                        dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    )
                    intermediate = ttnn.experimental.reduce_scatter_minimal_async_create_intermediate_buffer(
                        full, 3, topology=obj.topology
                    )
                    obj.persistent_ccl[(batch, group)] = ([intermediate[0], local, intermediate[1]], full)
        return obj

    def _check(self, x, label):
        if self.candidate.get("diagnose") and not self.decoding:
            import torch

            host = ttnn.to_torch(x, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=0))
            ok = bool(torch.isfinite(host).all())
            print("COMPONENT_FINITE", label, tuple(x.shape), ok, flush=True)
            if not ok:
                raise RuntimeError("Nonfinite at " + label)
        return x

    def _use_ccl8(self):
        if self.decoding and self.candidate.get("ccl8_prefill_only"):
            return False
        mode = self.candidate.get("ccl8", False)
        return mode is True or mode == getattr(self, "ccl_group", None)

    def _gather(self, x):
        use8 = self._use_ccl8() and x.shape[-1] > 128
        if use8:
            x = ttnn.typecast(x, ttnn.bfloat8_b)
        self._check(x, "gather_input")
        if self.decoding and self.candidate.get("persistent_ccl"):
            out = ttnn.experimental.all_gather_async(
                x,
                dim=3,
                persistent_output_buffer=self.persistent_ccl[(x.shape[-2], self.ccl_group)][1],
                multi_device_global_semaphore=self.ccl.get_and_cycle_ag_semaphore_handles(),
                barrier_semaphore=self.ccl.get_and_cycle_barrier_semaphore_handle(),
                num_links=2,
                topology=self.topology,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        else:
            out = super()._gather(x)
        self._check(out, "gather_output")
        return ttnn.typecast(out, ttnn.bfloat16) if use8 else out

    def _norm(self, x, name=None):
        if name and "layernorm" in name:
            self.ccl_group = "attention" if name in ("input_layernorm", "post_attention_layernorm") else "mlp"
        if (
            self.residual_sharded
            and self.candidate.get("fused_ag")
            and name in ("input_layernorm", "pre_feedforward_layernorm")
        ):
            x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
            stats = ttnn.rms_norm_pre_all_gather(x, compute_kernel_config=self.kernel, dtype=ttnn.bfloat16)
            stats = self._gather(stats)
            return ttnn.rms_norm_post_all_gather(
                x,
                stats,
                epsilon=self.config.rms_norm_eps,
                weight=self.weights[name + ".weight"],
                compute_kernel_config=self.kernel,
            )
        if (
            self.residual_sharded
            and self.candidate.get("gather_before_norm")
            and name in ("input_layernorm", "pre_feedforward_layernorm")
        ):
            return OptimizedDecoder._norm(self, self._gather(ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)), name)
        return super()._norm(x, name)

    def _linear(self, x, name):
        self.ccl_group = "attention" if name.startswith("self_attn.") else "mlp"
        if self.candidate.get("packed_mlp") and name in ("mlp.gate_proj", "mlp.up_proj"):
            if name == "mlp.up_proj":
                out = self._packed_up
                self._packed_up = None
                return out
            if not self.decoding and self.candidate.get("packed_direct_bfp8"):
                grid = self.mesh_device.compute_with_storage_grid_size()
                packed = ttnn.experimental.minimal_matmul(
                    input_tensor=ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG) if self.prefill_l1 else x,
                    weight_tensor=self.weights["mlp.gate_up_proj.weight"],
                    config=ttnn.MinimalMatmulConfig(
                        M_block_size=4,
                        K_block_size=8,
                        N_block_size=16,
                        subblock_h=1,
                        subblock_w=8,
                        compute_with_storage_grid_size=ttnn.CoreCoord(grid.x, grid.y),
                    ),
                    compute_kernel_config=self.linear_kernels["mlp"],
                    memory_config=ttnn.L1_MEMORY_CONFIG if self.prefill_l1 else ttnn.DRAM_MEMORY_CONFIG,
                    dtype=ttnn.bfloat8_b,
                )
            else:
                packed = OptimizedDecoder._linear(self, x, "mlp.gate_up_proj")
            packed = ttnn.to_memory_config(
                packed, ttnn.L1_MEMORY_CONFIG if self.decoding or self.prefill_l1 else ttnn.DRAM_MEMORY_CONFIG
            )
            if not self.decoding and not self.candidate.get("packed_direct_bfp8"):
                packed = ttnn.typecast(packed, ttnn.bfloat8_b)
            width = packed.shape[-1] // 2
            gate = ttnn.slice(packed, (0, 0, 0, 0), (*tuple(packed.shape)[:-1], width))
            self._packed_up = ttnn.slice(packed, (0, 0, 0, width), tuple(packed.shape))
            return gate
        if self.candidate.get("fused_ag") and name in (
            "self_attn.qkv_proj",
            "self_attn.qk_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
        ):
            x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
            cfg = ttnn.MinimalMatmulConfig(
                M_block_size=1 if self.decoding else 4,
                K_block_size=7,
                N_block_size=8,
                subblock_h=1,
                subblock_w=8,
                compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
            )
            out = ttnn.experimental.all_gather_minimal_matmul_async(
                x,
                self.ag_weights[name + ".weight"],
                config=cfg,
                dtype=(
                    ttnn.bfloat8_b
                    if not self.decoding and name.startswith("mlp.") and not self.candidate.get("ag_bf16_output")
                    else ttnn.bfloat16
                ),
                compute_kernel_config=self.linear_kernels[self.ccl_group],
                multi_device_global_semaphore=self.ccl.get_and_cycle_ag_semaphore_handles(),
                barrier_semaphore=self.ccl.get_and_cycle_barrier_semaphore_handle(),
                num_links=2,
                num_workers_per_link=4,
                topology=self.topology,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            return out[0]
        if not self.decoding and self.candidate.get("dram_prefill"):
            self.prefill_l1 = False
        if self.decoding and self.candidate.get("fused_rows") and name in self.output_padding:
            x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
            k = x.shape[-1] // 32
            block = max(v for v in (4, 7, 8, 11, 14, 16, 21, 22, 28, 32) if k % v == 0)
            pc = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=(8, 1),
                in0_block_w=block,
                out_subblock_h=1,
                out_subblock_w=4,
                per_core_M=1,
                per_core_N=24,
                out_block_w=12,
                transpose_mcast=False,
                fuse_batch=False,
            )
            partial, out = ttnn.experimental.matmul_reduce_scatter_async(
                x,
                self.fused_weights[name],
                dim=3,
                persistent_intermediate_buffer=self.fused_buffers[name][0],
                persistent_output_buffer=self.fused_buffers[name][1],
                multi_device_global_semaphore=self.ccl.get_and_cycle_rs_semaphore_handles(),
                reduce_scatter_core_grid_offset=(0, 1),
                num_links=2,
                topology=ttnn.Topology.Ring if self.candidate.get("ring_fused") else ttnn.Topology.Linear,
                memory_config_mm=ttnn.DRAM_MEMORY_CONFIG,
                memory_config_rs=ttnn.DRAM_MEMORY_CONFIG,
                program_config=pc,
                compute_kernel_config=self.linear_kernels["attention" if name.startswith("self_attn.") else "down"],
            )
            out = ttnn.slice(out, (0, 0, 0, 0), (*tuple(out.shape)[:-1], 1344))
            if not self.residual_sharded:
                out = self._gather(out)
                if self.candidate.get("carry_reduced"):
                    out = ttnn.to_memory_config(out, self._working_memory(out, 8))
            return out
        if not self.decoding and self.candidate.get("prefill_kblock"):
            weight = self.weights[name + ".weight"]
            k = weight.shape[-2] // 32
            preferred = self.candidate["prefill_kblock"]
            block = max(v for v in range(1, preferred + 1) if k % v == 0)
            grid = self.candidate.get("prefill_grid", [11, 10])
            cfg = ttnn.MinimalMatmulConfig(
                M_block_size=4,
                K_block_size=block,
                N_block_size=self.candidate.get("prefill_nblock", 16),
                subblock_h=1,
                subblock_w=8,
                compute_with_storage_grid_size=ttnn.CoreCoord(*grid),
            )
            if self.prefill_l1:
                x = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)
            out = ttnn.experimental.minimal_matmul(
                input_tensor=x,
                weight_tensor=weight,
                config=cfg,
                compute_kernel_config=self.linear_kernels[
                    "attention" if name.startswith("self_attn.") else "down" if name == "mlp.down_proj" else "mlp"
                ],
                memory_config=(
                    ttnn.L1_MEMORY_CONFIG
                    if self.prefill_l1 and (name.startswith("mlp.") or name == "self_attn.o_proj")
                    else ttnn.DRAM_MEMORY_CONFIG
                ),
                dtype=ttnn.bfloat8_b if name in ("mlp.gate_proj", "mlp.up_proj") else ttnn.bfloat16,
            )
            if name in self.output_padding:
                out = ttnn.slice(out, [0] * len(out.shape), [*tuple(out.shape)[:-1], self.hidden])
        else:
            out = OptimizedDecoder._linear(self, x, name)
        self._check(out, name + "_local_matmul")
        if name not in self.output_padding:
            return out
        out = ttnn.to_memory_config(out, ttnn.DRAM_MEMORY_CONFIG)
        if self._use_ccl8():
            out = ttnn.typecast(out, ttnn.bfloat8_b)
        out = ttnn.experimental.reduce_scatter_minimal_async(
            out,
            dim=3,
            persistent_output_buffers=(
                self.persistent_ccl[(out.shape[-2], self.ccl_group)][0]
                if self.decoding and self.candidate.get("persistent_ccl")
                else None
            ),
            multi_device_global_semaphore=self.ccl.get_and_cycle_rs_semaphore_handles(),
            barrier_semaphore=(
                self.ccl.get_and_cycle_barrier_semaphore_handle() if self.candidate.get("rs_barrier", True) else None
            ),
            topology=self.topology,
            num_links=2,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self._check(out, name + "_reduce_scatter")
        if not self.residual_sharded:
            out = self._gather(out)
        out = ttnn.typecast(out, ttnn.bfloat16) if out.dtype != ttnn.bfloat16 else out
        if self.decoding and not self.residual_sharded and self.candidate.get("carry_reduced"):
            out = ttnn.to_memory_config(out, self._working_memory(out, 8))
        return out
