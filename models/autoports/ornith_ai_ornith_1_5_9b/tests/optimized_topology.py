# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Whole-layer topology experiments on the production optimized decoder."""
import math

import ttnn
from models.autoports.ornith_ai_ornith_1_5_9b.tests.optimized_reference import OptimizedDecoder


class TopologyCandidate(OptimizedDecoder):
    topology = "packed"
    residual_cores = 32

    @classmethod
    def from_state_dict(cls, sd, **kw):
        self = super().from_state_dict(sd, **kw)
        if "sharded" in self.topology:
            cg = ttnn.CoreGrid(x=8, y=self.residual_cores // 8)
            self.residual_mem = ttnn.create_sharded_memory_config(
                (32, 4096 // self.residual_cores),
                cg,
                ttnn.ShardStrategy.WIDTH,
                use_height_and_width_as_shard_shape=True,
            )
            self.norm_program = ttnn.LayerNormShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=(8, self.residual_cores // 8),
                subblock_w=min(4, 128 // self.residual_cores),
                block_h=1,
                block_w=128 // self.residual_cores,
                inplace=False,
            )
        if "split" not in self.topology:
            return self
        prefix = f'model.language_model.layers.{kw["layer_idx"]}.'
        sd = {k.removeprefix(prefix): v for k, v in sd.items()}
        names = (
            ["mlp.gate_proj", "mlp.up_proj"]
            if "split_mlp" in self.topology
            else (
                ["linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.in_proj_b", "linear_attn.in_proj_a"]
                if self.kind == "linear_attention"
                else ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]
            )
        )
        self.split_names = names
        banks = self.device.dram_grid_size().x
        grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(banks - 1, 0))})
        for name in names:
            w = sd[name + ".weight"]
            if name == "self_attn.q_proj":
                w = w.reshape(16, 2, 256, 4096).permute(1, 0, 2, 3).reshape(8192, 4096)
            w = w.T.contiguous()
            k, n = w.shape
            role = self.role_configs.get(name, {})
            cores = role.get("cores", self.decode_cores)
            readers = role.get("readers", self.decode_readers)
            block = role.get("block", self.decode_block)
            mc = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.DRAM,
                ttnn.ShardSpec(grid, (k, math.ceil(n / (32 * banks)) * 32), ttnn.ShardOrientation.ROW_MAJOR),
            )
            self.dram_weights[name] = ttnn.from_torch(
                w, device=self.device, dtype=self.weight_dtype(name), layout=ttnn.TILE_LAYOUT, memory_config=mc
            )

            def layout(width):
                sw = math.ceil(width / (32 * cores))
                nc = math.ceil(width / (32 * sw))
                cg = ttnn.num_cores_to_corerangeset(nc, self.device.compute_with_storage_grid_size(), True)
                return (
                    ttnn.MemoryConfig(
                        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                        ttnn.BufferType.L1,
                        ttnn.ShardSpec(cg, (32, sw * 32), ttnn.ShardOrientation.ROW_MAJOR),
                    ),
                    sw,
                )

            im, _ = layout(k)
            om, ow = layout(n)
            self.dram_layouts[name] = (
                im,
                om,
                ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                    in0_block_w=block,
                    per_core_M=1,
                    per_core_N=ow,
                    num_workers_per_dram_bank=readers,
                    use_multicore_pipeline=readers > 1,
                ),
            )
        return self

    def linear(self, x, name):
        if "split_input" in self.topology and name == "packed_input" and x.shape[2] == 1:
            return ttnn.concat([super(TopologyCandidate, self).linear(x, n) for n in self.split_names], dim=-1)
        if "sharded" in self.topology and x.shape[2] == 1 and name in self.dram_weights:
            im, om, cfg = self.dram_layouts[name]
            out = ttnn.linear(
                ttnn.to_memory_config(x, im),
                self.dram_weights[name],
                program_config=cfg,
                compute_kernel_config=self.matmul_compute(name),
                memory_config=om,
            )
            target = (
                self.residual_mem
                if name in ("linear_attn.out_proj", "self_attn.o_proj", "mlp.down_proj")
                else ttnn.DRAM_MEMORY_CONFIG
            )
            return ttnn.to_memory_config(out, target)
        return super().linear(x, name)

    def finish(self, x, mixed):
        if "split_mlp" not in self.topology or x.shape[2] != 1:
            return super().finish(x, mixed)
        residual = ttnn.add(x, mixed)
        h = self.norm(residual, "post_attention_layernorm")
        gate = self.linear(h, "mlp.gate_proj")
        up = self.linear(h, "mlp.up_proj")
        h = ttnn.multiply(gate, up, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
        return ttnn.add(residual, self.linear(h, "mlp.down_proj"))

    def norm(self, x, name):
        if "sharded" in self.topology and x.shape[2] == 1 and name in ("input_layernorm", "post_attention_layernorm"):
            return ttnn.rms_norm(
                ttnn.to_memory_config(x, self.residual_mem),
                weight=self.weights[name + ".weight"],
                epsilon=self.config.rms_norm_eps,
                program_config=self.norm_program,
                compute_kernel_config=self.compute,
                memory_config=self.residual_mem,
            )
        return super().norm(x, name)

    def decode_forward(self, x, **kwargs):
        if "sharded" not in self.topology:
            return super().decode_forward(x, **kwargs)
        out = super().decode_forward(ttnn.to_memory_config(x, self.residual_mem), **kwargs)
        return ttnn.to_memory_config(out, ttnn.DRAM_MEMORY_CONFIG)
