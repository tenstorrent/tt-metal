# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage-five coherent TP4 optimization candidates, selected only by evidence."""

import math

import ttnn
from models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_stage4_reference import MultichipDecoder


class Candidate(MultichipDecoder):
    prefill_ar_rows = 0
    packed_decode = False
    minimal_decode = False
    input_l1 = False
    release_input = False

    @classmethod
    def from_state_dict(cls, sd, **kwargs):
        self = super().from_state_dict(sd, **kwargs)
        if self.prefill_ar_rows:
            rows = self.prefill_ar_rows
            self.prefill_ar_mem = ttnn.create_sharded_memory_config(
                (rows, 64),
                ttnn.CoreGrid(x=8, y=8),
                ttnn.ShardStrategy.WIDTH,
                use_height_and_width_as_shard_shape=True,
            )
            mem = ttnn.create_sharded_memory_config(
                (rows, 256),
                ttnn.CoreGrid(x=8, y=8),
                ttnn.ShardStrategy.WIDTH,
                use_height_and_width_as_shard_shape=True,
            )
            self.prefill_ar_buffers = [
                ttnn.empty(
                    (1, 1, rows, 16384),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.device,
                    memory_config=mem,
                )
                for _ in range(2)
            ]
        if self.packed_decode:
            import torch

            prefix = f'model.language_model.layers.{kwargs["layer_idx"]}.'
            sd = {k.removeprefix(prefix): v for k, v in sd.items()}
            pieces = [
                torch.cat(
                    [
                        sd["mlp.gate_proj.weight"][r * 3072 : (r + 1) * 3072].T,
                        sd["mlp.up_proj.weight"][r * 3072 : (r + 1) * 3072].T,
                    ],
                    dim=1,
                )
                for r in range(4)
            ]
            banks = self.device.dram_grid_size().x
            grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(banks - 1, 0))})
            mc = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.DRAM,
                ttnn.ShardSpec(grid, (4096, math.ceil(6144 / (32 * banks)) * 32), ttnn.ShardOrientation.ROW_MAJOR),
            )
            self.dram_weights["gate_up"] = ttnn.from_torch(
                torch.cat(pieces, dim=0).contiguous(),
                device=self.device,
                dtype=self.mlp_dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=mc,
                mesh_mapper=ttnn.ShardTensorToMesh(self.device, dim=0),
            )
            im = self.dram_layouts["mlp.gate_proj"][0]
            om = ttnn.create_sharded_memory_config(
                (32, 96), ttnn.CoreGrid(x=8, y=8), ttnn.ShardStrategy.WIDTH, use_height_and_width_as_shard_shape=True
            )
            self.dram_layouts["gate_up"] = (
                im,
                om,
                ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                    in0_block_w=2, per_core_M=1, per_core_N=3, num_workers_per_dram_bank=2, use_multicore_pipeline=True
                ),
            )
        return self

    def linear(self, x, name):
        if self.input_l1 and x.shape[2] == 1 and name == "packed_input":
            im, om, cfg = self.dram_layouts[name]
            out = ttnn.linear(
                ttnn.to_memory_config(x, im),
                self.dram_weights[name],
                program_config=cfg,
                compute_kernel_config=self.matmul_compute(name),
                memory_config=om,
            )
            return ttnn.to_memory_config(out, ttnn.L1_MEMORY_CONFIG)
        out = super().linear(x, name)
        if self.release_input and name == "packed_input" and x.shape[2] > 1:
            ttnn.deallocate(x)
        return out

    def reduce(self, x):
        if self.prefill_ar_rows and x.shape[2] > 1:
            n, rows = x.shape[2], self.prefill_ar_rows
            parts = []
            for start in range(0, n, rows):
                part = x[:, :, start : min(start + rows, n), :]
                logical = part.shape[2]
                if logical < rows:
                    part = ttnn.pad(part, [(0, 0), (0, 0), (0, rows - logical), (0, 0)], 0.0)
                i = self.ar_index
                self.ar_index = 1 - i
                out = ttnn.experimental.all_reduce_async(
                    ttnn.to_memory_config(part, self.prefill_ar_mem),
                    self.prefill_ar_buffers[i],
                    cluster_axis=1,
                    mesh_device=self.device,
                    multi_device_global_semaphore=self.ar_semaphores[i],
                    memory_config=self.prefill_ar_mem,
                    dtype=ttnn.bfloat16,
                    topology=self.ccl.topology,
                    num_links=self.ccl.num_links,
                )
                parts.append(ttnn.to_memory_config(out, ttnn.DRAM_MEMORY_CONFIG)[:, :, :logical, :])
            return ttnn.concat(parts, dim=2) if len(parts) > 1 else parts[0]
        return super().reduce(x)

    def finish(self, x, mixed):
        if x.shape[2] != 1 or not (self.packed_decode or self.minimal_decode):
            return super().finish(x, mixed)
        if self.native_local:
            x = ttnn.to_memory_config(x, self.ar_mem)
        residual = ttnn.add(x, mixed)
        h = self.norm(residual, "post_attention_layernorm")
        if self.minimal_decode:
            h = ttnn.experimental.minimal_matmul(
                ttnn.to_memory_config(h, ttnn.L1_MEMORY_CONFIG),
                self.weights["packed_mlp.weight"],
                fuse_swiglu=True,
                config=ttnn.MinimalMatmulConfig(
                    M_block_size=1,
                    K_block_size=16,
                    N_block_size=8,
                    subblock_h=1,
                    subblock_w=2,
                    compute_with_storage_grid_size=(11, 10),
                ),
                compute_kernel_config=self.mlp_compute,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        else:
            packed = ttnn.to_memory_config(self.linear(h, "gate_up"), ttnn.L1_MEMORY_CONFIG)
            gate, up = packed[:, :, :, :3072], packed[:, :, :, 3072:]
            h = ttnn.multiply(gate, up, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
        return ttnn.add(residual, self.reduce(self.linear(h, "mlp.down_proj")))


class FamilyCandidate(Candidate):
    fused_gather_enabled = False
    fused_scatter_enabled = False
    communication_dtype = ttnn.bfloat16
    persistent = False
    norm_communication_dtype = None

    @classmethod
    def from_state_dict(cls, sd, **kwargs):
        kwargs["collective"] = "separate"
        self = super().from_state_dict(sd, **kwargs)
        self.prefill_headroom = ttnn.get_memory_view(self.device, ttnn.BufferType.L1).total_bytes_per_bank - 64 * 1024
        self.rs_index = 0
        self.persistent_rs, self.persistent_ag = {}, {}
        if self.persistent:
            for n in (1, 1024):
                mem = ttnn.L1_MEMORY_CONFIG if n == 1 else ttnn.DRAM_MEMORY_CONFIG

                def empty(shape):
                    return ttnn.empty(
                        shape,
                        dtype=self.communication_dtype,
                        layout=ttnn.TILE_LAYOUT,
                        device=self.device,
                        memory_config=mem,
                    )

                self.persistent_rs[n] = [[empty((2, 1, max(32, n), 4096)), empty((1, 1, n, 1024))] for _ in range(2)]
                self.persistent_ag[n] = [empty((1, 1, n, 4096)) for _ in range(2)]
        return self

    def reduce(self, x):
        if self.fused_scatter_enabled:
            return x if self.residual_layout == "sharded" else self.gather(x)
        n = x.shape[2]
        mem = ttnn.L1_MEMORY_CONFIG if n <= 32 else ttnn.DRAM_MEMORY_CONFIG
        x = ttnn.typecast(ttnn.to_memory_config(x, mem), self.communication_dtype)
        i = self.rs_index
        self.rs_index = 1 - i
        y = ttnn.experimental.reduce_scatter_minimal_async(
            x,
            persistent_output_buffers=self.persistent_rs[n][i] if self.persistent else None,
            dim=3,
            cluster_axis=1,
            topology=self.ccl.topology,
            num_links=self.ccl.num_links,
            multi_device_global_semaphore=self.ccl.get_rs_ping_pong_semaphore(),
            barrier_semaphore=self.ccl.get_barrier_semaphore(),
            memory_config=mem,
        )
        if self.residual_layout == "replicated":
            y = ttnn.experimental.all_gather_async(
                y,
                persistent_output_tensor=self.persistent_ag[n][i] if self.persistent else None,
                dim=3,
                cluster_axis=1,
                mesh_device=self.device,
                topology=self.ccl.topology,
                num_links=self.ccl.num_links,
                multi_device_global_semaphore=self.ccl.get_ag_ping_pong_semaphore(),
                barrier_semaphore=self.ccl.get_barrier_semaphore(),
                memory_config=mem,
            )
        return ttnn.typecast(y, ttnn.bfloat16)

    def norm(self, x, name):
        if self.residual_layout != "sharded" or name not in ("input_layernorm", "post_attention_layernorm"):
            return super().norm(x, name)
        mem = ttnn.L1_MEMORY_CONFIG if x.shape[2] <= 32 else ttnn.DRAM_MEMORY_CONFIG
        x = ttnn.to_memory_config(x, mem)
        stats = self.gather(ttnn.rms_norm_pre_all_gather(x, compute_kernel_config=self.compute, dtype=ttnn.bfloat16))
        out = ttnn.rms_norm_post_all_gather(
            x,
            stats,
            epsilon=self.config.rms_norm_eps,
            weight=self.weights[name + ".weight"],
            compute_kernel_config=self.compute,
            memory_config=mem,
        )
        out = ttnn.typecast(out, self.norm_communication_dtype or self.communication_dtype)
        return out if self.fused_gather_enabled else self.gather(out)

    def fused_gather(self, x, name, swiglu=False):
        n = x.shape[2]
        if n % 32:
            x = ttnn.pad(x, [(0, 0), (0, 0), (0, 32 - n % 32), (0, 0)], 0.0)
        return ttnn.experimental.all_gather_minimal_matmul_async(
            x,
            ttnn.reshape(self.weights[name + ".weight"], (1, 1, *self.weights[name + ".weight"].shape)),
            fuse_swiglu=swiglu,
            config=ttnn.MinimalMatmulConfig(
                M_block_size=1 if n <= 32 else 4,
                K_block_size=16,
                N_block_size=4,
                subblock_h=1,
                subblock_w=2,
                compute_with_storage_grid_size=(8, 4),
            ),
            multi_device_global_semaphore=self.ccl.get_ag_ping_pong_semaphore(),
            barrier_semaphore=self.ccl.get_barrier_semaphore(),
            topology=self.ccl.topology,
            cluster_axis=1,
            num_links=2,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.matmul_compute(name),
            force_transpose=True,
            num_workers_per_link=4,
        )[0][:, :, :n, :]

    def linear(self, x, name):
        if self.fused_gather_enabled and name == "packed_input":
            return self.fused_gather(x, name)
        if self.fused_scatter_enabled and name in ("linear_attn.out_proj", "self_attn.o_proj", "mlp.down_proj"):
            n = x.shape[2]
            x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
            if n % 32:
                x = ttnn.pad(x, [(0, 0), (0, 0), (0, 32 - n % 32), (0, 0)], 0.0)
            return ttnn.experimental.matmul_reduce_scatter_prefill(
                x,
                self.weights[name + ".weight"],
                cluster_axis=1,
                scatter_dim=-1,
                topology=self.ccl.topology,
                num_links=2,
                compute_kernel_config=ttnn.ComputeConfigDescriptor(
                    math_fidelity=self.mlp_fidelity if name.startswith("mlp.") else self.attention_fidelity
                ),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                l1_headroom=self.prefill_headroom,
            )[:, :, :n, :]
        if x.dtype != ttnn.bfloat16:
            if x.shape[2] == 1:
                im, om, cfg = self.dram_layouts[name]
                result = ttnn.linear(
                    ttnn.to_memory_config(x, im),
                    self.dram_weights[name],
                    program_config=cfg,
                    compute_kernel_config=self.matmul_compute(name),
                    memory_config=om,
                    dtype=ttnn.bfloat16,
                )
                return ttnn.to_memory_config(result, ttnn.DRAM_MEMORY_CONFIG)
            return ttnn.experimental.minimal_matmul(
                ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG),
                self.weights[name + ".weight"],
                config=self.prefill_program,
                compute_kernel_config=self.matmul_compute(name),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                dtype=ttnn.bfloat16,
            )
        return super().linear(x, name)

    def finish(self, x, mixed):
        if not self.fused_gather_enabled:
            return super().finish(x, mixed)
        residual = ttnn.add(x, mixed)
        h = self.fused_gather(self.norm(residual, "post_attention_layernorm"), "packed_mlp", True)
        return ttnn.add(residual, self.reduce(self.linear(h, "mlp.down_proj")))


def candidate(name):
    if name == "stage4":
        return MultichipDecoder
    if name == "prefill-kernel":
        from models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_prefill_kernel_candidate import PrefillKernelCandidate
        return PrefillKernelCandidate
    if name == "decode-mask":
        from models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_decode_mask_candidate import DecodeMaskCandidate
        return DecodeMaskCandidate
    policies = {
        "control": {},
        "ar64": {"prefill_ar_rows": 64},
        "ar128": {"prefill_ar_rows": 128},
        "ar256": {"prefill_ar_rows": 256},
        "ar256_release": {"prefill_ar_rows": 256, "release_input": True},
        "ar224": {"prefill_ar_rows": 224},
        "packed_mlp": {"packed_decode": True},
        "minimal_mlp": {"minimal_decode": True},
        "input_l1": {"input_l1": True},
        "attention_lofi": {"attention_fidelity": ttnn.MathFidelity.LoFi},
        "mlp_hifi2": {"mlp_fidelity": ttnn.MathFidelity.HiFi2},
        "attention_bfp4": {"attention_dtype": ttnn.bfloat4_b, "attention_fidelity": ttnn.MathFidelity.LoFi},
        "mlp_bfp4": {"mlp_dtype": ttnn.bfloat4_b},
        "kv_bfp4": {"kv_dtype": ttnn.bfloat4_b},
    }
    families = {
        "family_separate": {},
        "family_persistent": {"persistent": True},
        "family_bfp8": {"communication_dtype": ttnn.bfloat8_b},
        "family_ccl_bfp8": {"communication_dtype": ttnn.bfloat8_b, "norm_communication_dtype": ttnn.bfloat16},
        "family_bfp8_persistent": {"communication_dtype": ttnn.bfloat8_b, "persistent": True},
        "family_gather": {"fused_gather_enabled": True},
        "family_scatter": {"fused_scatter_enabled": True},
        "family_fused": {"fused_gather_enabled": True, "fused_scatter_enabled": True},
        "family_gather_bfp8": {"fused_gather_enabled": True, "communication_dtype": ttnn.bfloat8_b},
        "family_fused_bfp8": {
            "fused_gather_enabled": True,
            "fused_scatter_enabled": True,
            "communication_dtype": ttnn.bfloat8_b,
        },
        "family_packed": {"packed_decode": True},
        "family_packed_bfp8": {"packed_decode": True, "communication_dtype": ttnn.bfloat8_b},
    }
    if name not in families and name not in policies:
        from models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_extra_candidates import extra_candidate

        return extra_candidate(name)
    return type(
        "Optimization_" + name,
        (FamilyCandidate if name in families else Candidate,),
        families[name] if name in families else policies[name],
    )
