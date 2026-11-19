# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Shape-faithful alternatives through residual/norm consumers, for measured selection."""

import ttnn
from models.autoports.ornith_ai_ornith_1_5_9b.tt.multichip_decoder import MultichipDecoder


class FusedGatherDecoder(MultichipDecoder):
    def norm(self, x, name):
        if name not in ("input_layernorm", "post_attention_layernorm"):
            return super().norm(x, name)
        assert self.residual_layout == "sharded"
        x = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG if x.shape[2] <= 32 else ttnn.DRAM_MEMORY_CONFIG)
        stats = self.gather(ttnn.rms_norm_pre_all_gather(x, compute_kernel_config=self.compute, dtype=ttnn.bfloat16))
        return ttnn.rms_norm_post_all_gather(
            x,
            stats,
            epsilon=self.config.rms_norm_eps,
            weight=self.weights[name + ".weight"],
            compute_kernel_config=self.compute,
            memory_config=x.memory_config(),
        )

    def fused_gather(self, x, name, swiglu=False):
        n = x.shape[2]
        if n % 32:
            x = ttnn.pad(x, [(0, 0), (0, 0), (0, 32 - n % 32), (0, 0)], 0.0)
        out = ttnn.experimental.all_gather_minimal_matmul_async(
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
            num_links=1,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.matmul_compute(name),
            force_transpose=True,
            num_workers_per_link=8,
        )[0]
        return out[:, :, :n, :]

    def linear(self, x, name):
        if name == "packed_input":
            return self.fused_gather(x, name)
        return super().linear(x, name)

    def finish(self, x, mixed):
        residual = ttnn.add(x, mixed)
        h = self.norm(residual, "post_attention_layernorm")
        h = self.fused_gather(h, "packed_mlp", True)
        return ttnn.add(residual, self.reduce(self.linear(h, "mlp.down_proj")))


class FusedScatterDecoder(MultichipDecoder):
    def linear(self, x, name):
        if name not in ("linear_attn.out_proj", "self_attn.o_proj", "mlp.down_proj"):
            return super().linear(x, name)
        n = x.shape[2]
        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
        if n % 32:
            x = ttnn.pad(x, [(0, 0), (0, 0), (0, 32 - n % 32), (0, 0)], 0.0)
        out = ttnn.experimental.matmul_reduce_scatter_prefill(
            x,
            self.weights[name + ".weight"],
            cluster_axis=1,
            scatter_dim=-1,
            topology=self.ccl.topology,
            num_links=1,
            compute_kernel_config=ttnn.ComputeConfigDescriptor(
                math_fidelity=self.mlp_fidelity if name.startswith("mlp.") else self.attention_fidelity
            ),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return out[:, :, :n, :] if self.residual_layout == "sharded" else self.gather(out)[:, :, :n, :]

    def reduce(self, x):
        return x


class FusedBothDecoder(FusedGatherDecoder, FusedScatterDecoder):
    pass


class DirectReduceDecoder(MultichipDecoder):
    @classmethod
    def from_state_dict(cls, *args, **kwargs):
        self = super().from_state_dict(*args, **kwargs)
        assert self.residual_layout == "replicated"
        self.ar_mem = ttnn.create_sharded_memory_config(
            (32, 128), ttnn.CoreGrid(x=8, y=4), ttnn.ShardStrategy.WIDTH, use_height_and_width_as_shard_shape=True
        )
        buffer_mem = ttnn.create_sharded_memory_config(
            (32, 512), ttnn.CoreGrid(x=8, y=4), ttnn.ShardStrategy.WIDTH, use_height_and_width_as_shard_shape=True
        )
        self.ar_buffers = [
            ttnn.empty(
                (1, 1, 32, 16384),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=buffer_mem,
            )
            for _ in range(2)
        ]
        self.ar_semaphores = [ttnn.create_global_semaphore(self.device, self.ccl.ccl_cores, 0) for _ in range(2)]
        self.ar_index = 0
        return self

    def reduce(self, x):
        if x.shape[2] != 1:
            return super().reduce(x)
        i = self.ar_index
        self.ar_index = 1 - i
        out = ttnn.experimental.all_reduce_async(
            ttnn.to_memory_config(x, self.ar_mem),
            self.ar_buffers[i],
            cluster_axis=1,
            mesh_device=self.device,
            multi_device_global_semaphore=self.ar_semaphores[i],
            memory_config=self.ar_mem,
            dtype=ttnn.bfloat16,
            topology=self.ccl.topology,
            num_links=1,
        )
        return ttnn.to_memory_config(out, ttnn.DRAM_MEMORY_CONFIG)


class HybridDecoder(DirectReduceDecoder):
    def linear(self, x, name):
        if x.shape[2] == 1 or name not in ("linear_attn.out_proj", "self_attn.o_proj", "mlp.down_proj"):
            return super().linear(x, name)
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
            num_links=1,
            compute_kernel_config=ttnn.ComputeConfigDescriptor(
                math_fidelity=self.mlp_fidelity if name.startswith("mlp.") else self.attention_fidelity
            ),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )[:, :, :n, :]

    def reduce(self, x):
        if x.shape[-1] == 1024:
            return self.gather(x)
        return super().reduce(x)


class NativeResidualDecoder(MultichipDecoder):
    """Retain the optimized baseline's local L1 residual/norm/MLP contract."""

    @classmethod
    def from_state_dict(cls, *args, **kwargs):
        kwargs.update(collective="direct", fused_prefill=True, native_local=True)
        self = super().from_state_dict(*args, **kwargs)
        self.norm_program = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=(8, 4), subblock_w=2, block_h=1, block_w=4, inplace=False
        )
        return self

    def linear(self, x, name):
        if x.shape[2] != 1:
            return super().linear(x, name)
        im, om, cfg = self.dram_layouts[name]
        out = ttnn.linear(
            ttnn.to_memory_config(x, im),
            self.dram_weights[name],
            program_config=cfg,
            compute_kernel_config=self.matmul_compute(name),
            memory_config=om,
        )
        return ttnn.to_memory_config(out, ttnn.DRAM_MEMORY_CONFIG) if name == "packed_input" else out

    def reduce(self, x):
        if x.shape[2] != 1 or x.shape[-1] != 4096:
            return super().reduce(x)
        i = self.ar_index
        self.ar_index = 1 - i
        return ttnn.experimental.all_reduce_async(
            ttnn.to_memory_config(x, self.ar_mem),
            self.ar_buffers[i],
            cluster_axis=1,
            mesh_device=self.device,
            multi_device_global_semaphore=self.ar_semaphores[i],
            memory_config=self.ar_mem,
            dtype=ttnn.bfloat16,
            topology=self.ccl.topology,
            num_links=1,
        )

    def norm(self, x, name):
        if x.shape[2] == 1 and name in ("input_layernorm", "post_attention_layernorm"):
            return ttnn.rms_norm(
                ttnn.to_memory_config(x, self.ar_mem),
                weight=self.weights[name + ".weight"],
                epsilon=self.config.rms_norm_eps,
                program_config=self.norm_program,
                compute_kernel_config=self.compute,
                memory_config=self.ar_mem,
            )
        return super().norm(x, name)

    def finish(self, x, mixed):
        if x.shape[2] == 1:
            x = ttnn.to_memory_config(x, self.ar_mem)
        return super().finish(x, mixed)

    def decode_forward(self, x, *, state, page_table=None, current_position=None):
        x = ttnn.to_memory_config(x, self.ar_mem)
        h = self.norm(x, "input_layernorm")
        mixed = (
            self.linear_attention(h, state, decode=True)
            if self.kind == "linear_attention"
            else self.full_attention(h, state, page_table=page_table, position=current_position, decode=True)
        )
        return ttnn.to_memory_config(self.finish(x, mixed), ttnn.DRAM_MEMORY_CONFIG)


class BudgetPrefillDecoder(NativeResidualDecoder):
    """Constrain the fused planner to an enumerated NoC0-only family."""

    budget_kib = 96

    @classmethod
    def from_state_dict(cls, *args, **kwargs):
        self = super().from_state_dict(*args, **kwargs)
        bank = ttnn.get_memory_view(self.device, ttnn.BufferType.L1).total_bytes_per_bank
        self.prefill_headroom = bank - self.budget_kib * 1024
        print("PREFILL_L1_BANK_HEADROOM", bank, self.prefill_headroom, flush=True)
        return self

    def linear(self, x, name):
        if x.shape[2] > 1 and name in ("linear_attn.out_proj", "self_attn.o_proj", "mlp.down_proj"):
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
                num_links=1,
                compute_kernel_config=ttnn.ComputeConfigDescriptor(
                    math_fidelity=self.mlp_fidelity if name.startswith("mlp.") else self.attention_fidelity
                ),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                l1_headroom=self.prefill_headroom,
            )[:, :, :n, :]
        return super().linear(x, name)


class NativeUnfusedDecoder(NativeResidualDecoder):
    @classmethod
    def from_state_dict(cls, *args, **kwargs):
        self = super().from_state_dict(*args, **kwargs)
        self.fused_prefill = False
        return self


class Budget64PrefillDecoder(BudgetPrefillDecoder):
    budget_kib = 64


class L1PrefillDecoder(NativeUnfusedDecoder):
    """Apply profiler's L1 input advice to both minimal-matmul roles."""

    def norm(self, x, name):
        out = super().norm(x, name)
        if x.shape[2] > 1 and name in ("input_layernorm", "post_attention_layernorm"):
            out = ttnn.to_memory_config(out, ttnn.L1_MEMORY_CONFIG)
        return out

    def linear(self, x, name):
        if x.shape[2] > 1:
            x = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)
        return super().linear(x, name)


class AvailableLinksDecoder(L1PrefillDecoder):
    @classmethod
    def from_state_dict(cls, *args, **kwargs):
        self = super().from_state_dict(*args, **kwargs)
        self.ccl.num_links = ttnn.get_num_links(self.device, 1)
        print("AVAILABLE_CCL_LINKS", self.ccl.num_links, flush=True)
        return self


class TwoLinkReduceDecoder(AvailableLinksDecoder):
    def reduce(self, x):
        if x.shape[2] != 1 or x.shape[-1] != 4096:
            return super().reduce(x)
        i = self.ar_index
        self.ar_index = 1 - i
        return ttnn.experimental.all_reduce_async(
            ttnn.to_memory_config(x, self.ar_mem),
            self.ar_buffers[i],
            cluster_axis=1,
            mesh_device=self.device,
            multi_device_global_semaphore=self.ar_semaphores[i],
            memory_config=self.ar_mem,
            dtype=ttnn.bfloat16,
            topology=self.ccl.topology,
            num_links=2,
        )
