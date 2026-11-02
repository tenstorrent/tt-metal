# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Gemma4 TP4 decoder candidate using the optimized single-chip contract.

Inputs and outputs are replicated BF16 [1,1,S,2816]. Weights and cache heads
are tensor parallel; collectives reduce local projections before RMSNorm.
The caller opens a FABRIC_1D 1x4 mesh and owns paged cache and position tensors.
"""

import os
from copy import copy
from dataclasses import replace
from types import SimpleNamespace

import ttnn
from models.demos.gemma4.config import MeshConfig, ModeConfig
from models.demos.gemma4.tt.experts.weights import ExpertWeights
from models.demos.gemma4.tt.layer import Gemma4DecoderLayer
from models.demos.gemma4.tt.model_config import Gemma4ModelArgs
from models.demos.gemma4_26b_a4b_qb2.tt.fused_decoder import BroadcastRouter, FusedAttention, PackedExperts
from models.demos.gemma4_26b_a4b_qb2.tt.optimized_decoder import (
    GeneralizedRouter,
    MinimalPrefillProjection,
    OptimizedAttention,
    OptimizedDecoder,
    OptimizedExperts,
)
from models.demos.gemma4_26b_a4b_qb2.tt.precision_ops import norm_weight
from models.demos.gemma4_26b_a4b_qb2.tt.precision_policy import assert_precision_matches
from models.demos.gemma4_26b_a4b_qb2.tt.precision_policy import dtype as policy_dtype
from models.demos.gemma4_26b_a4b_qb2.tt.precision_policy import dtype_name
from models.demos.gemma4_26b_a4b_qb2.tt.precision_policy import fidelity as policy_fidelity
from models.demos.gemma4_26b_a4b_qb2.tt.precision_policy import fidelity_name
from models.demos.gemma4_26b_a4b_qb2.tt.routing_precision import Router
from models.demos.gpt_oss.tt.ccl import CCLManager


class _MeshCCLManager(CCLManager):
    """Cover every worker that native CCL may choose on this mesh."""

    def _init_subdevice(self):
        grid = self.mesh_device.compute_with_storage_grid_size()
        self.ccl_cores = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))}
        )
        self.ccl_sub_device_id = ttnn.SubDeviceId(0)


class CollectiveBufferPool:
    """Caller-owned buffers for serial layers on one mesh/command queue.

    Roles keep separate storage to prevent accidental cross-role aliases.
    With Linear TP4 and separate RS/AG workloads on one CQ, the next RS needs
    every rank's contribution before the following AG can overwrite a gathered
    output. Per-rank dispatch orders its prior consumer before that RS; an
    intervening opposite-role allreduce is not required. This contract excludes
    concurrent callers, multiple CQs and fused/streaming collectives.
    Layer outputs own fresh storage.
    """

    def __init__(self, mesh_device):
        self.mesh_device = mesh_device
        self.buffers = {}


class _DramAttentionProjection:
    """Decode-only bank-sharded copy; compute policy is supplied at each call."""

    def __init__(self, weight, mesh, block, readers=1, storage_cores=8):
        k, n = weight.shape[-2], weight.shape[-1]
        self.logical_output_width = n
        padded_n = ((n + 256 * readers - 1) // (256 * readers)) * 256 * readers
        if padded_n != n:
            padding = [(0, 0)] * len(weight.shape)
            padding[-1] = (0, padded_n - n)
            dtype = weight.dtype
            if dtype == ttnn.bfloat4_b:
                weight = ttnn.typecast(weight, ttnn.bfloat16)
            weight = ttnn.pad(weight, padding, 0)
            if weight.dtype != dtype:
                weight = ttnn.typecast(weight, dtype)
            n = padded_n
        if mesh.dram_grid_size().x != 8 or k % 256 or n % 256 or weight.dtype not in (ttnn.bfloat8_b, ttnn.bfloat4_b):
            raise ValueError("Attention DRAM projection requires eight banks and tile-aligned BFP8/BFP4 weights")
        if (k // (32 * storage_cores)) % block:
            raise ValueError("Attention DRAM input shard must divide the K block")
        grid = ttnn.CoreGrid(x=storage_cores, y=1)
        bank_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0))})
        weight_memory = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.DRAM,
            ttnn.ShardSpec(bank_grid, (k, n // 8), ttnn.ShardOrientation.ROW_MAJOR),
        )
        # Preserve existing quantization and mesh ownership; original weights serve prefill.
        self.weight = ttnn.to_memory_config(weight, weight_memory)
        self.input_memory, self.output_memory = (
            ttnn.create_sharded_memory_config(
                (32, width // storage_cores),
                grid,
                ttnn.ShardStrategy.WIDTH,
                ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            for width in (k, n)
        )
        self.program = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
            in0_block_w=block,
            per_core_M=1,
            per_core_N=n // (32 * storage_cores),
            num_workers_per_dram_bank=readers,
        )
        self.extra_weight_bytes = (k // 32) * (n // 32) * (1088 if weight.dtype == ttnn.bfloat8_b else 576)

    def __call__(self, value, *, compute, memory_config):
        output = ttnn.linear(
            ttnn.to_memory_config(value, self.input_memory),
            self.weight,
            dtype=ttnn.float32,
            memory_config=self.output_memory,
            program_config=self.program,
            compute_kernel_config=compute,
        )
        output = ttnn.to_memory_config(output, memory_config)
        return output if output.shape[-1] == self.logical_output_width else output[..., : self.logical_output_width]


class _Projection:
    def __init__(self, weight, compute, mesh, sliding, decode_fidelity=ttnn.MathFidelity.LoFi):
        self.weight, self.compute = weight, compute
        self.decode_dram = None
        self.decode_compute = ttnn.init_device_compute_kernel_config(
            mesh.arch(),
            math_fidelity=decode_fidelity,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        self.program = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(8, 4) if sliding else (8, 6),
            in0_block_w=22,
            out_subblock_h=1,
            out_subblock_w=2,
            per_core_M=1,
            per_core_N=2,
            fuse_batch=True,
            mcast_in0=True,
        )
        self.prefill = MinimalPrefillProjection(
            weight, compute, mesh, block_w=8 if sliding else 16, block_h=4 if sliding else 2
        )

    def __call__(self, x):
        if x.shape[-2] > 1:
            return self.prefill(x, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if getattr(self, "input_bfp8", False):
            x = ttnn.typecast(x, ttnn.bfloat8_b)
        if self.decode_dram is not None:
            return self.decode_dram(x, compute=self.decode_compute, memory_config=ttnn.L1_MEMORY_CONFIG)
        return ttnn.linear(
            ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG),
            self.weight,
            dtype=ttnn.float32,
            compute_kernel_config=self.decode_compute,
            program_config=self.program,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )


class _GatherProjection:
    """Decode-only gather fusion; prefill retains the optimized projection."""

    def __init__(self, projection, mesh):
        import torch

        self.projection = projection
        self.weight = projection.weight
        self.hidden = self.weight.shape[-2]
        self.local_hidden = self.hidden // 4
        self.local_output = self.weight.shape[-1]
        block = projection.program.in0_block_w
        if block <= 0 or (self.local_hidden // 32) % block:
            raise ValueError("Fused gather-QKV K block must divide each local ready slice")
        grid = mesh.compute_with_storage_grid_size()
        cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
        self.semaphores = [ttnn.create_global_semaphore(mesh, cores, 0) for _ in range(2)]
        self.barrier = ttnn.create_global_semaphore(mesh, cores, 0)
        # Logical row1 with tile-padded physical row32; buffer belongs to this layer.
        self.gather_buffer = ttnn.from_torch(
            torch.zeros(1, 1, 1, self.hidden),
            device=mesh,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.program = projection.program

    def __call__(self, x):
        if x.shape[-1] == self.hidden:
            return self.projection(x)
        if tuple(x.shape) != (1, 1, 1, self.local_hidden):
            raise ValueError("Fused gather-QKV expects one logical decode row with hidden width704")
        _, projected = ttnn.experimental.all_gather_matmul_async(
            ttnn.to_memory_config(ttnn.typecast(x, ttnn.float32), ttnn.DRAM_MEMORY_CONFIG),
            self.weight,
            persistent_output_buffer=self.gather_buffer,
            dim=3,
            multi_device_global_semaphore=self.semaphores,
            all_gather_core_grid_offset=(0, 8),
            barrier_semaphore=self.barrier,
            num_links=1,
            topology=ttnn.Topology.Ring,
            memory_config_ag=ttnn.DRAM_MEMORY_CONFIG,
            memory_config_mm=ttnn.L1_MEMORY_CONFIG,
            program_config=self.program,
            compute_kernel_config=self.projection.decode_compute,
            dtype=ttnn.float32,
        )
        # The downstream head splitter sees logical S1 even if a kernel exposes padding.
        if projected.shape[-2] != 1:
            projected = projected[:, :, :1, :]
        return projected


class _LocalAttention(OptimizedAttention):
    def rotary(self, value, cos, sin, *, decode=False):
        if not decode or not getattr(self, "sharded_decode_rope", False):
            return super().rotary(value, cos, sin, decode=decode)
        width = value.shape[-1]
        if width not in (256, 512):
            raise ValueError("Sharded decode RoPE supports head dimensions 256 and 512")
        # Native sharded RoPE requires uniform BF16 formats and at most eight DST tiles.
        compute = ttnn.init_device_compute_kernel_config(
            value.device().arch(),
            math_fidelity=self.compute.math_fidelity,
            math_approx_mode=self.compute.math_approx_mode,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )
        memory = ttnn.create_sharded_memory_config(
            (32, 256),
            ttnn.CoreGrid(x=1, y=1),
            ttnn.ShardStrategy.HEIGHT,
            ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        operands = tuple(ttnn.typecast(tensor, ttnn.bfloat16) for tensor in (value, cos, sin))
        outputs = []
        half = width // 2
        for start in range(0, half, 128):
            # Pair matching segments from the two global rotate-half operands.
            paired = (
                operands
                if width == 256
                else tuple(
                    ttnn.concat(
                        (tensor[..., start : start + 128], tensor[..., half + start : half + start + 128]), dim=-1
                    )
                    for tensor in operands
                )
            )
            result = ttnn.experimental.rotary_embedding_hf(
                *(ttnn.to_memory_config(tensor, memory) for tensor in paired),
                is_decode_mode=True,
                compute_kernel_config=compute,
            )
            outputs.append(ttnn.to_memory_config(result, value.memory_config()))
        result = (
            outputs[0]
            if width == 256
            else ttnn.concat(
                tuple(output[..., :128] for output in outputs) + tuple(output[..., 128:] for output in outputs),
                dim=-1,
                memory_config=value.memory_config(),
            )
        )
        return ttnn.typecast(result, value.dtype)

    def project(self, attention, decode):
        input_bfp8 = getattr(self, "output_input_bfp8", False)
        fused_output = getattr(self, "decode_output_fused", None)
        if not decode or (
            getattr(self, "decode_output_dram", None) is None and fused_output is None and not input_bfp8
        ):
            return self.reduce(super().project(attention, decode))
        cfg = self.source.config
        memory = ttnn.create_sharded_memory_config(
            (32, cfg.head_dim),
            ttnn.CoreGrid(x=1, y=1),
            ttnn.ShardStrategy.HEIGHT,
            ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        combined = ttnn.experimental.nlp_concat_heads_decode(
            ttnn.to_memory_config(attention, memory), num_heads=cfg.num_attention_heads
        )
        width = cfg.num_attention_heads * cfg.head_dim
        combined = ttnn.reshape(combined, (1, 1, 1, width), (1, 1, 32, width))
        if input_bfp8:
            combined = ttnn.typecast(ttnn.to_memory_config(combined, ttnn.L1_MEMORY_CONFIG), ttnn.bfloat8_b)
        if fused_output is not None:
            return fused_output(combined)
        if getattr(self, "decode_output_dram", None) is None:
            return self.reduce(
                ttnn.linear(
                    combined,
                    self.source.weights.o_proj,
                    dtype=ttnn.float32,
                    program_config=self.output_program,
                    compute_kernel_config=self.output_compute,
                    memory_config=self.output_memory,
                )
            )
        projected = self.decode_output_dram(combined, compute=self.output_compute, memory_config=self.output_memory)
        return self.reduce(projected)


class _SharedMLP:
    def __init__(self, source, compute, reduce):
        self.gate_up = source.gate_up_proj
        self.down = source.down_proj
        self.width = source._inter_per_device
        self.compute = compute
        self.reduce = reduce
        self.decode_weights = None

    def configure_decode(self, state, mesh, sliding, geometry=0, precision=None):
        import torch

        gate = state["mlp.gate_proj.weight"].transpose(-2, -1)
        up = state["mlp.up_proj.weight"].transpose(-2, -1)
        down = state["mlp.down_proj.weight"].transpose(-2, -1)
        padding = 4 * self.width - gate.shape[-1]
        gate, up = (torch.nn.functional.pad(t, (0, padding)) for t in (gate, up))
        down = torch.nn.functional.pad(down, (0, 0, 0, padding))
        gate_parts, up_parts = gate.chunk(4, dim=-1), up.chunk(4, dim=-1)
        packed = torch.cat([torch.cat((u, g), dim=-1) for u, g in zip(up_parts, gate_parts)], dim=-1)
        down_dtype = ttnn.bfloat8_b if sliding else ttnn.bfloat4_b
        gate_dtype = ttnn.bfloat4_b
        fidelity = ttnn.MathFidelity.LoFi
        if precision is not None:
            gate_dtype = policy_dtype(precision["shared_gate_dtype"])
            down_dtype = policy_dtype(precision["shared_down_dtype"])
            fidelity = policy_fidelity(precision["shared_fidelity"])
        self.decode_weights = tuple(
            ttnn.from_torch(
                t[None, None],
                device=mesh,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=axis),
            )
            for t, axis, dtype in ((packed, -1, gate_dtype), (down, -2, down_dtype))
        )
        self.decode_compute = ttnn.init_device_compute_kernel_config(
            mesh.arch(),
            math_fidelity=fidelity,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )

        self.decode_programs = (None, None)
        if geometry:
            gate_grid, gate_n, gate_k = ((11, 4), 1, 44) if geometry == 1 else ((9, 2), 2, 88)

            def program(grid, per_n, block_k):
                return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                    compute_with_storage_grid_size=grid,
                    in0_block_w=block_k,
                    out_subblock_h=1,
                    out_subblock_w=per_n,
                    out_block_h=1,
                    out_block_w=per_n,
                    per_core_M=1,
                    per_core_N=per_n,
                    fuse_batch=True,
                    mcast_in0=True,
                )

            self.decode_programs = (program(gate_grid, gate_n, gate_k), program((11, 4), 2, 17))

    def __call__(self, x, *, reduce_output=True):
        # Imported Gemma4 loader packs [up_i, gate_i] on each rank.
        decode = self.decode_weights is not None and x.shape[-2] == 1
        return self._forward(x, decode=decode, reduce_output=reduce_output)

    def decode_batch(self, x, *, reduce_output=True):
        """Use the selected decode precision for up to one logical token tile."""
        if self.decode_weights is None or not 1 <= x.shape[-2] <= 32:
            raise ValueError("Batched shared decode requires selected weights and at most 32 rows")
        return self._forward(x, decode=True, reduce_output=reduce_output)

    def _forward(self, x, *, decode, reduce_output):
        if decode and getattr(self, "input_bfp8", False):
            x = ttnn.typecast(x, ttnn.bfloat8_b)
        gu = (
            ttnn.linear(
                x,
                self.decode_weights[0],
                dtype=ttnn.bfloat16,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                compute_kernel_config=self.decode_compute,
                program_config=self.decode_programs[0],
            )
            if decode
            else self.gate_up(x)
        )
        up, gate = gu[..., : self.width], gu[..., self.width :]
        hidden = ttnn.mul(gate, up, input_tensor_a_activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.GELU, 0.0)])
        if decode and getattr(self, "input_bfp8", False):
            hidden = ttnn.typecast(hidden, ttnn.bfloat8_b)
        output = (
            ttnn.linear(
                hidden,
                self.decode_weights[1],
                dtype=ttnn.bfloat16,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                compute_kernel_config=self.decode_compute,
                program_config=self.decode_programs[1],
            )
            if decode
            else self.down(hidden)
        )
        return self.reduce(output) if reduce_output else output


class _DramSharedMLP(_SharedMLP):
    """Bank-sharded shared decode; the inherited BF16 prefill stays unchanged."""

    def configure_decode(self, state, mesh, sliding, geometry=0, readers=1):
        import torch

        banks = mesh.dram_grid_size().x
        if banks != 8 or self.width != 544 or state["mlp.gate_proj.weight"].shape[-1] != 2816:
            raise ValueError("Shared DRAM decode requires eight banks and local shared width544/hidden2816")
        gate = state["mlp.gate_proj.weight"].transpose(-2, -1)
        up = state["mlp.up_proj.weight"].transpose(-2, -1)
        down = state["mlp.down_proj.weight"].transpose(-2, -1)
        padding = 4 * self.width - gate.shape[-1]
        gate, up = (torch.nn.functional.pad(t, (0, padding)) for t in (gate, up))
        down = torch.nn.functional.pad(down, (0, 0, 0, padding))
        local_n = 2 * self.width
        alignment = banks * 32 * readers
        physical_n = ((local_n + alignment - 1) // alignment) * alignment
        down_n = ((2816 + alignment - 1) // alignment) * alignment
        down = torch.nn.functional.pad(down, (0, down_n - 2816))
        # Bank padding belongs inside each mesh rank, after its [up, gate] pair.
        packed = torch.cat(
            [
                torch.nn.functional.pad(torch.cat((u, g), dim=-1), (0, physical_n - local_n))
                for u, g in zip(up.chunk(4, dim=-1), gate.chunk(4, dim=-1))
            ],
            dim=-1,
        )
        down_dtype = ttnn.bfloat8_b if sliding else ttnn.bfloat4_b
        bank_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(banks - 1, 0))})
        self.decode_weights = []
        self.decode_inputs = []
        self.decode_outputs = []
        self.decode_programs = []
        for matrix, axis, k, n, cores, block, dtype in (
            (packed, -1, 2816, physical_n, 8, 11, ttnn.bfloat4_b),
            (down, -2, self.width, down_n, 1, 17, down_dtype),
        ):
            weight_memory = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.DRAM,
                ttnn.ShardSpec(bank_grid, (k, n // banks), ttnn.ShardOrientation.ROW_MAJOR),
            )
            self.decode_weights.append(
                ttnn.from_torch(
                    matrix[None, None],
                    device=mesh,
                    dtype=dtype,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=weight_memory,
                    mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=axis),
                )
            )
            storage_grid = ttnn.CoreGrid(x=cores, y=1)
            for memories, width in ((self.decode_inputs, k), (self.decode_outputs, n)):
                memories.append(
                    ttnn.create_sharded_memory_config(
                        (32, width // cores),
                        storage_grid,
                        ttnn.ShardStrategy.WIDTH,
                        ttnn.ShardOrientation.ROW_MAJOR,
                        use_height_and_width_as_shard_shape=True,
                    )
                )
            self.decode_programs.append(
                ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                    in0_block_w=block,
                    per_core_M=1,
                    per_core_N=n // 32 // cores,
                    num_workers_per_dram_bank=readers,
                )
            )
        self.decode_compute = ttnn.init_device_compute_kernel_config(
            mesh.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
        self.extra_decode_weight_tiles = (2816 // 32) * ((physical_n - local_n) // 32)

    def _project(self, value, index):
        output = ttnn.linear(
            ttnn.to_memory_config(value, self.decode_inputs[index]),
            self.decode_weights[index],
            dtype=ttnn.bfloat16,
            memory_config=self.decode_outputs[index],
            program_config=self.decode_programs[index],
            compute_kernel_config=self.decode_compute,
        )
        return ttnn.to_memory_config(output, ttnn.L1_MEMORY_CONFIG)

    def __call__(self, x, *, reduce_output=True):
        if x.shape[-2] != 1:
            return super().__call__(x, reduce_output=reduce_output)
        gu = self._project(x, 0)[..., : 2 * self.width]
        up, gate = gu[..., : self.width], gu[..., self.width :]
        hidden = ttnn.mul(gate, up, input_tensor_a_activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.GELU, 0.0)])
        output = self._project(hidden, 1)[..., :2816]
        return self.reduce(output) if reduce_output else output


class _ExpertParallelExperts(OptimizedExperts):
    """Own 32 complete experts per rank and scan only their routing mask."""

    def __init__(self, state_dict, config, mesh_device, sliding):
        import torch

        if config.num_experts != 128 or config.moe_intermediate_size != 704:
            raise ValueError("EP4 requires 128 experts with intermediate width 704")
        width = config.moe_intermediate_size
        mapper = ttnn.ShardTensorToMesh(mesh_device, dim=1)

        def weight(value):
            return ttnn.from_torch(
                value,
                device=mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=mapper,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        fused = state_dict["experts.gate_up_proj"]
        weights = ExpertWeights(
            gate_proj=weight(fused[:, :width, :].transpose(-2, -1).unsqueeze(0)),
            up_proj=weight(fused[:, width:, :].transpose(-2, -1).unsqueeze(0)),
            down_proj=weight(state_dict["experts.down_proj"].transpose(-2, -1).unsqueeze(0)),
            intermediate_size_per_device=width,
        )
        local_config = SimpleNamespace(
            hidden_size=config.hidden_size, num_experts=32, top_k=config.top_k_experts, moe_intermediate_size=width
        )
        source = SimpleNamespace(config=local_config, mesh_device=mesh_device, weights=weights, prefill_sparsity=None)
        packed = PackedExperts(source, fused_gelu=True, prefill_batch_tokens=32, matmul_mix=True, mix_fp32=sliding)
        super().__init__(
            packed,
            gate_dtype=ttnn.bfloat8_b if sliding else ttnn.bfloat4_b,
            down_dtype=ttnn.bfloat4_b,
            block_w=22,
            gate_block_w=44,
            fidelity=ttnn.MathFidelity.LoFi,
            mesh_device=mesh_device,
            expert_grid=(11, 4),
            down_grid=(11, 8),
            active_prefill=True,
            prefill_tokens=32,
            prefill_dtype=ttnn.bfloat8_b if sliding else ttnn.bfloat4_b,
            prefill_down_dtype=ttnn.bfloat4_b,
            prefill_fidelity=ttnn.MathFidelity.LoFi,
            expert_fused_gelu=True,
            activation_dtype=None if sliding else ttnn.bfloat8_b,
        )

        self.short_prefill_batch_tokens = int(os.environ.get("GEMMA4_PREFILL_EXPERT_BATCH", "32"))
        if self.short_prefill_batch_tokens not in (32, 64, 128):
            raise ValueError("GEMMA4_PREFILL_EXPERT_BATCH must be 32, 64 or 128")
        prefill_gate_k = int(os.environ.get("GEMMA4_PREFILL_GATE_K", "22"))
        if prefill_gate_k not in (11, 22, 44, 88):
            raise ValueError("GEMMA4_PREFILL_GATE_K must be 11, 22, 44 or 88")
        prefill_down_cores = int(os.environ.get("GEMMA4_PREFILL_DOWN_CORES", "44"))
        if prefill_down_cores not in (44, 88):
            raise ValueError("GEMMA4_PREFILL_DOWN_CORES must be 44 or 88")

        def prefill_program(grid, block, rows, columns=1):
            return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=grid,
                in0_block_w=block,
                out_subblock_h=1,
                out_subblock_w=columns,
                out_block_h=1,
                out_block_w=columns,
                per_core_M=rows // 32,
                per_core_N=columns,
                fuse_batch=False,
                mcast_in0=True,
            )

        self.prefill_configs = {
            rows: (
                prefill_program((11, 4), prefill_gate_k if rows == 32 else (11 if sliding else 22), rows),
                (
                    prefill_program((11, 4), 22, rows, 2)
                    if rows == 32 and prefill_down_cores == 44
                    else prefill_program((11, 8), 22, rows)
                ),
            )
            for rows in range(32, self.short_prefill_batch_tokens + 1, 32)
        }
        self.mix_memory = ttnn.L1_MEMORY_CONFIG
        self.mix_program = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(11, 8),
            in0_block_w=1,
            out_subblock_h=1,
            out_subblock_w=1,
            per_core_M=1,
            per_core_N=1,
            fuse_batch=True,
            mcast_in0=True,
        )
        ownership = torch.arange(128, dtype=torch.int32).reshape(1, 1, 1, 128)
        self.route_indices = {
            rows: ttnn.from_torch(
                ownership.repeat(1, 1, rows, 1),
                device=mesh_device,
                dtype=ttnn.uint16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=-1),
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            for rows in (1, *range(32, self.short_prefill_batch_tokens + 1, 32))
        }

    def __call__(self, x, routing):
        if x.shape[-2] == 1:
            return self._chunk(x, routing, True)
        width = self.prefill_batch_tokens
        # S128-S256 uses at least the short-prefill width to amortize routing.
        # Tiny inputs and longer chunks retain their configured width.
        if 128 <= x.shape[-2] <= 256:
            width = max(width, self.short_prefill_batch_tokens)
        outputs = [
            self._chunk(
                x[:, :, i : min(i + width, x.shape[-2]), :], routing[:, :, i : min(i + width, x.shape[-2]), :], False
            )
            for i in range(0, x.shape[-2], width)
        ]
        return outputs[0] if len(outputs) == 1 else ttnn.concat(outputs, dim=2)

    def _chunk(self, x, routing, decode):
        # Ownership IDs address the replicated router output, not local weights.
        local = ttnn.gather(
            ttnn.to_layout(routing, ttnn.ROW_MAJOR_LAYOUT),
            dim=-1,
            index=self.route_indices[x.shape[-2]],
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        routing = ttnn.to_layout(local, ttnn.TILE_LAYOUT)
        if not decode:
            return self._active_prefill(x, routing)
        if self.decode_activation_dtype is not None:
            x = ttnn.typecast(x, self.decode_activation_dtype)
        # Counts vary from zero to eight by rank and replay. Omit nnz/indices so
        # kernels exchange validity for all slots and skip inactive projections.
        common = dict(
            sparsity=local,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            output_tile=ttnn.Tile([32, 32]),
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.decode_compute,
        )
        gu = ttnn.sparse_matmul(x, self.gate_up, program_config=self.gate_config, **common)
        gu = ttnn.reshape(gu, (1, 32, 1, 2 * self.width))
        gate, up = gu[..., : self.width], gu[..., self.width :]
        hidden = ttnn.mul(gate, up, input_tensor_a_activations=self.decode_gelu_activations)
        down = ttnn.sparse_matmul(hidden, self.down, program_config=self.down_config, is_input_a_sparse=True, **common)
        down = ttnn.reshape(down, (1, 32, 1, self.config.hidden_size))
        return ttnn.matmul(
            routing,
            ttnn.permute(down, (0, 2, 1, 3)),
            dtype=ttnn.bfloat16,
            memory_config=self.mix_memory,
            program_config=self.mix_program,
            compute_kernel_config=self.mix_compute,
        )


class _HybridExperts:
    """EP prefill and indexed TP decode with setup-owned weight layouts."""

    def __init__(self, prefill, decode):
        self.prefill, self.decode = prefill, decode

    def __call__(self, x, routing):
        return (self.decode if x.shape[-2] == 1 else self.prefill)(x, routing)


class MultichipDecoder(OptimizedDecoder):
    """TP4 implementation; optimized chunking and tensor-owned positions are inherited."""

    baseline_class = OptimizedDecoder
    tp = 4

    @classmethod
    def from_state_dict(
        cls,
        state_dict,
        *,
        hf_config,
        layer_idx,
        mesh_device,
        chunk_size=1024,
        sharded_residual=False,
        expert_parallel=False,
        fused_tail=True,
        hybrid_experts=True,
        grouped_moe_reduce=True,
        fused_agmm=False,
        topology=ttnn.Topology.Linear,
        optimized_shared=True,
        shared_dram=False,
        shared_geometry=None,
        qkv_fidelity=ttnn.MathFidelity.LoFi,
        output_fidelity=ttnn.MathFidelity.LoFi,
        attention_ccl_dtype=ttnn.bfloat16,
        full_attention_ccl_dtype=ttnn.bfloat8_b,
        attention_dram=None,
        sharded_decode_rope=None,
        moe_ccl_bfp8=None,
        persistent_ccl=None,
        optimized_decode=True,
        attention_precision=None,
        expert_gate_dtype=None,
        collective_buffer_pool=None,
        precision_config=None,
    ):
        import torch

        if tuple(mesh_device.shape) != (1, 4):
            raise ValueError("This decoder targets the four-chip 1x4 Blackhole mesh")
        workers = mesh_device.compute_with_storage_grid_size()
        if workers.x < 11 or workers.y < 10:
            raise ValueError("The router placement requires an 11x10 Blackhole worker grid")
        if attention_dram not in (None, "qkv", "output"):
            raise ValueError("attention_dram must be None, 'qkv' or 'output'")
        if attention_dram == "qkv" and fused_agmm:
            raise ValueError("QKV DRAM sharding and fused AGMM are separate decode backends")
        for dtype in (attention_ccl_dtype, full_attention_ccl_dtype):
            if dtype is not None and dtype not in (ttnn.float32, ttnn.bfloat16, ttnn.bfloat8_b):
                raise ValueError("Attention CCL dtype must be FP32, BF16 or BFP8")
        if attention_ccl_dtype is None:
            raise ValueError("attention_ccl_dtype must be explicit")
        if shared_dram and not optimized_shared:
            raise ValueError("Shared DRAM decode requires optimized_shared=True")
        if shared_dram and shared_geometry:
            raise ValueError("Shared DRAM decode requires shared_geometry=0")
        config = getattr(hf_config, "text_config", hf_config)
        sliding = config.layer_types[layer_idx] == "sliding_attention"
        if precision_config is not None:
            if not (optimized_decode and optimized_shared and hybrid_experts and grouped_moe_reduce and fused_tail):
                raise ValueError("Precision policy requires the accepted optimized hybrid TP4/EP4 path")
            if expert_parallel or sharded_residual or shared_dram or attention_dram or fused_agmm:
                raise ValueError("Precision policy does not support alternate projection or parallel backends")
            if topology != ttnn.Topology.Linear:
                raise ValueError("Precision policy requires Linear collectives")
            qkv_fidelity = policy_fidelity(precision_config["qkv_fidelity"])
            output_fidelity = policy_fidelity(precision_config["output_fidelity"])
            expert_gate_dtype = policy_dtype(precision_config["expert_gate_dtype"])
            attention_ccl_dtype = policy_dtype(precision_config["attention_ccl_dtype"])
            full_attention_ccl_dtype = attention_ccl_dtype
            moe_ccl_bfp8 = precision_config["moe_ccl_dtype"] == "bfloat8_b"
        if attention_precision is None:
            attention_precision = "qkv" if optimized_decode and not sliding else "baseline"
        if attention_precision not in ("baseline", "qkv", "output", "both"):
            raise ValueError("Unknown attention precision policy")
        if expert_gate_dtype is None:
            expert_gate_dtype = ttnn.bfloat8_b if optimized_decode and sliding else ttnn.bfloat4_b
        if expert_gate_dtype not in (ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.bfloat16):
            raise ValueError("Expert gate weights must use BFP4, BFP8 or BF16")
        if shared_geometry is None:
            shared_geometry = 2 if sliding else 1
        if chunk_size != 1024:
            raise ValueError("Internal prefill chunk size must be 1024; logical lengths are unrestricted")
        if shared_geometry not in (0, 1, 2):
            raise ValueError("shared_geometry must be 0 (automatic), 1 or 2")
        if shared_geometry and not optimized_shared:
            raise ValueError("Explicit shared geometry requires optimized_shared=True")
        self = cls()
        self.config, self.layer_idx, self.chunk_size = config, layer_idx, chunk_size
        self.mesh_device = mesh_device
        self.selected_precision = precision_config
        self.attention_dram_extra_weight_bytes = 0
        self.mesh_config = MeshConfig(mesh_device.shape, decode=ModeConfig(tp=4))
        if fused_agmm and (not sharded_residual or topology != ttnn.Topology.Ring):
            raise ValueError("Fused gather-QKV requires sharded residuals and Ring fabric/topology")
        self.topology = topology
        self.optimized_decode = optimized_decode
        self.attention_precision = attention_precision
        self.ccl_tuning = {"num_workers_per_link": 1 if sliding else 2} if optimized_decode else {}
        self.persistent_ccl = optimized_decode if persistent_ccl is None else persistent_ccl
        self.collective_memory = ttnn.L1_MEMORY_CONFIG if optimized_decode else ttnn.DRAM_MEMORY_CONFIG
        if collective_buffer_pool is not None:
            if collective_buffer_pool.mesh_device is not mesh_device:
                raise ValueError("Collective buffers belong to a different mesh")
            if topology != ttnn.Topology.Linear or sharded_residual or not grouped_moe_reduce:
                raise ValueError("Shared CCL buffers require Linear replicated residuals and grouped MoE")
            self._collective_buffers = collective_buffer_pool.buffers
        else:
            self._collective_buffers = {}
        self.fused_agmm = fused_agmm
        self.ccl = _MeshCCLManager(mesh_device, 1, topology)
        if grouped_moe_reduce and sharded_residual:
            raise ValueError("Grouped MoE reduction requires replicated residuals")
        self.grouped_moe_reduce = grouped_moe_reduce
        self.sharded_residual = sharded_residual
        self.fused_tail = fused_tail
        self.shared_geometry = shared_geometry
        self.batched_shared_decode = os.environ.get("GEMMA4_BATCHED_SHARED_DECODE", "1") != "0"
        self.moe_ccl_bfp8 = sliding if moe_ccl_bfp8 is None else bool(moe_ccl_bfp8)
        if sharded_residual:
            self.allreduce = self.reduce_scatter
        self.shard_norm_weights = {}
        if sharded_residual:
            for name in (
                "input_layernorm",
                "post_attention_layernorm",
                "post_feedforward_layernorm_1",
                "post_feedforward_layernorm_2",
                "post_feedforward_layernorm",
            ):
                self.shard_norm_weights[name] = ttnn.from_torch(
                    state_dict[name + ".weight"].reshape(1, 1, 1, -1),
                    device=mesh_device,
                    dtype=ttnn.float32,
                    layout=ttnn.TILE_LAYOUT,
                    mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=-1),
                )
        self.layer = Gemma4DecoderLayer(
            mesh_device=mesh_device,
            hf_config=Gemma4ModelArgs.from_hf_config(config),
            state_dict={f"model.layers.{layer_idx}.{k}": v for k, v in state_dict.items()},
            layer_idx=layer_idx,
            ccl_manager=self.ccl,
            dtype=ttnn.bfloat16,
            tensor_cache_path=None,
            mesh_config=self.mesh_config,
            max_seq_len=config.max_position_embeddings,
            max_local_batch_size=32,
        )
        self.attention_ccl_dtype = (
            full_attention_ccl_dtype if not sliding and full_attention_ccl_dtype is not None else attention_ccl_dtype
        )
        self.fuse_norm, self.fuse_rsqrt, self.precise_heads = True, sliding, sliding
        self.use_sharded_norms = True
        self.sharded_norm_site = "all"
        self.prefill_qkv_input_l1 = False
        self.kv_cache_dtype = (
            ttnn.bfloat8_b if precision_config is None else policy_dtype(precision_config["kv_cache_dtype"])
        )
        for name in ("post_feedforward_layernorm_1", "post_feedforward_layernorm_2", "post_feedforward_layernorm"):
            norm = getattr(self.layer, name)
            norm._sharded_cfg = norm._build_sharded_cfg(config.hidden_size)
            norm._sharded_dim = config.hidden_size
        self.input_norm_weight = norm_weight(self.layer.input_layernorm.tt_weight, config.hidden_size)
        self.post_attention_norm_weight = norm_weight(self.layer.post_attention_layernorm.tt_weight, config.hidden_size)
        self.shared_norm_weight = norm_weight(self.layer.pre_feedforward_layernorm.tt_weight, config.hidden_size)
        self.expert_norm_weight = norm_weight(self.layer.pre_feedforward_layernorm_2.tt_weight, config.hidden_size)
        self.tail_weights = tuple(
            norm_weight(getattr(self.layer, name).tt_weight, config.hidden_size)
            for name in ("post_feedforward_layernorm_1", "post_feedforward_layernorm_2", "post_feedforward_layernorm")
        )
        compute = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        source = self.layer.self_attn
        source.config = copy(source.config)
        source.config.num_attention_heads //= 4
        source.config.num_key_value_heads = max(1, source.config.num_key_value_heads // 4)
        source.config.num_key_value_groups = source.config.num_attention_heads // source.config.num_key_value_heads
        projection = _Projection(
            ttnn.typecast(source.weights.wqkv, ttnn.bfloat8_b), compute, mesh_device, sliding, qkv_fidelity
        )
        if precision_config is not None:
            qkv_dtype = policy_dtype(precision_config["qkv_weight_dtype"])
            # Keep the baseline BFP8 -> BFP4 quantization path. BF16 recovery
            # must use the original checkpoint upload, never widened BFP4.
            if qkv_dtype == ttnn.bfloat16:
                projection.weight = source.weights.wqkv
            elif projection.weight.dtype != qkv_dtype:
                projection.weight = ttnn.typecast(projection.weight, qkv_dtype)
            projection.input_bfp8 = precision_config["qkv_input_dtype"] == "bfloat8_b"
        elif attention_precision in ("qkv", "both"):
            projection.weight = ttnn.typecast(projection.weight, ttnn.bfloat4_b)
        if optimized_decode and not fused_agmm:
            projection.program.in0_block_w = 44
        if attention_dram == "qkv":
            projection.decode_dram = _DramAttentionProjection(projection.weight, mesh_device, 11)
            self.attention_dram_extra_weight_bytes = projection.decode_dram.extra_weight_bytes
        if fused_agmm:
            projection = _GatherProjection(projection, mesh_device)
        original_output_weight = source.weights.o_proj
        source.weights = replace(
            source.weights,
            wqkv=projection,
            o_proj=ttnn.typecast(source.weights.o_proj, ttnn.bfloat8_b),
        )
        base = SimpleNamespace(
            source=source,
            compute=compute,
            q_weight=norm_weight(source.weights.q_norm_weight, source.config.head_dim),
            k_weight=norm_weight(source.weights.k_norm_weight, source.config.head_dim),
            decode_sdpa=None,
        )
        fused = FusedAttention(base, self.normalize, "decode" if sliding else True)
        fused.fuse_cache, fused.common_kv = True, True
        fused.pack_heads = fused.cast_shard = fused.i2s_cast = False
        fused.unary_cast_shard = True
        fused.project_sharded = True
        attention = _LocalAttention.from_existing(
            fused,
            mesh_device=mesh_device,
            native_sdpa=True,
            native_sdpa_fidelity=ttnn.MathFidelity.HiFi4 if sliding else ttnn.MathFidelity.LoFi,
            prefill_attention_fidelity=ttnn.MathFidelity.LoFi if sliding else ttnn.MathFidelity.HiFi2,
            output_grid=(11, 8),
            output_block_w=32 if sliding else 8,
            output_fidelity=output_fidelity,
            output_l1=True,
        )
        attention.configure_prefill_output(
            mesh_device, 1024, minimal=True, minimal_block_w=8, fidelity=ttnn.MathFidelity.LoFi
        )
        if precision_config is not None:
            output_dtype = policy_dtype(precision_config["output_weight_dtype"])
            output_weight = attention.source.weights.o_proj
            if output_dtype == ttnn.bfloat16:
                output_weight = original_output_weight
            elif output_weight.dtype != output_dtype:
                output_weight = ttnn.typecast(output_weight, output_dtype)
            attention.source.weights = replace(attention.source.weights, o_proj=output_weight)
            attention.output_input_bfp8 = precision_config["output_input_dtype"] == "bfloat8_b"
        elif attention_precision in ("output", "both"):
            # Prefill owns its original BFP8 weight through configure_prefill_output.
            attention.source.weights = replace(
                attention.source.weights,
                o_proj=ttnn.typecast(attention.source.weights.o_proj, ttnn.bfloat4_b),
            )
        if attention_dram == "output":
            attention.decode_output_dram = _DramAttentionProjection(
                attention.source.weights.o_proj, mesh_device, 4 if sliding else 8
            )
            self.attention_dram_extra_weight_bytes = attention.decode_output_dram.extra_weight_bytes
        attention.sharded_decode_rope = sliding if sharded_decode_rope is None else bool(sharded_decode_rope)
        attention.reduce = self._reduce_attention
        self.layer.self_attn = attention
        router = BroadcastRouter(Router(self.layer.moe.router, config.rms_norm_eps), self.normalize)
        self.layer.moe.router = GeneralizedRouter(
            router,
            mesh_device,
            center_logits=True,
            direct_projection=True,
            projection_block_w=22 if sliding else 44,
            projection_fidelity=ttnn.MathFidelity.HiFi4 if sliding else ttnn.MathFidelity.LoFi,
        )
        # Isolate the single-core gate from expert and normalization compute workers.
        router = self.layer.moe.router
        router_core = ttnn.CoreCoord(10, 9)
        router.memory = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet({ttnn.CoreRange(router_core, router_core)}),
                (32, 32),
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )
        for name in ("bias", "indices", "output", "output_indices"):
            setattr(router, name, ttnn.to_memory_config(getattr(router, name), router.memory))
        packed = PackedExperts(
            self.layer.moe.experts, fused_gelu=True, prefill_batch_tokens=32, matmul_mix=True, mix_fp32=sliding
        )
        experts = OptimizedExperts(
            packed,
            gate_dtype=expert_gate_dtype,
            down_dtype=(
                ttnn.bfloat4_b if precision_config is None else policy_dtype(precision_config["expert_down_dtype"])
            ),
            block_w=6,
            gate_block_w=44,
            fidelity=(
                ttnn.MathFidelity.LoFi
                if precision_config is None
                else policy_fidelity(precision_config["expert_fidelity"])
            ),
            mesh_device=mesh_device,
            active_prefill=True,
            prefill_tokens=32,
            prefill_dtype=ttnn.bfloat8_b if sliding and not hybrid_experts else ttnn.bfloat4_b,
            prefill_down_dtype=ttnn.bfloat4_b,
            prefill_fidelity=ttnn.MathFidelity.LoFi,
            expert_fused_gelu=True,
            activation_dtype=(
                ttnn.bfloat8_b if precision_config is None else policy_dtype(precision_config["expert_input_dtype"])
            ),
        )

        if sliding:
            # Pack directly from checkpoint values; widening a BFP4 tensor
            # cannot recover the BF8 policy used by consecutive sliding layers.
            gate, up = state_dict["experts.gate_up_proj"].chunk(2, dim=-2)
            gate, up = (torch.nn.functional.pad(t.transpose(-2, -1), (0, 64)) for t in (gate, up))
            packed_gate = torch.cat(
                [torch.cat((g, u), dim=-1) for g, u in zip(gate.chunk(4, dim=-1), up.chunk(4, dim=-1))],
                dim=-1,
            )
            experts.gate_up = ttnn.from_torch(
                packed_gate.unsqueeze(0),
                device=mesh_device,
                dtype=expert_gate_dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=-1),
            )
            if hybrid_experts:
                # This object is decode-only; EP owns the separate prefill weights.
                experts.prefill_gate = experts.gate_up

        # Output tile counts are12 for gate/up and88 for down. Blackhole's
        # 11-column grid avoids the8x8 helper's eight-core down projection.
        def sparse_program(rows, grid, block):
            return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=grid,
                in0_block_w=block,
                out_subblock_h=1,
                out_subblock_w=1,
                out_block_h=1,
                out_block_w=1,
                per_core_M=rows // 32,
                per_core_N=1,
                fuse_batch=False,
                mcast_in0=True,
            )

        experts.gate_config = sparse_program(32, (6, 2), 44)
        experts.down_config = sparse_program(32, (11, 8), 6)
        experts.prefill_configs = {32: (sparse_program(32, (6, 2), 11), sparse_program(32, (11, 8), 6))}
        experts.mix_program = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(11, 8),
            in0_block_w=4,
            out_subblock_h=1,
            out_subblock_w=1,
            per_core_M=1,
            per_core_N=1,
            fuse_batch=True,
            mcast_in0=True,
        )
        experts.mix_memory = ttnn.L1_MEMORY_CONFIG
        experts.enable_indexed_decode(self.layer.moe.router)
        self.layer.moe.experts = (
            _ExpertParallelExperts(state_dict, config, mesh_device, sliding) if expert_parallel else experts
        )
        if hybrid_experts:
            if expert_parallel:
                raise ValueError("Select EP-only or hybrid experts, not both")
            self.layer.moe.experts = _HybridExperts(
                _ExpertParallelExperts(state_dict, config, mesh_device, sliding), experts
            )
        self.hybrid_experts = hybrid_experts
        self.expert_parallel = expert_parallel
        shared_class = _DramSharedMLP if shared_dram else _SharedMLP
        self.layer.shared_mlp = shared_class(self.layer.shared_mlp, compute, self.allreduce)
        if optimized_shared:
            shared_kwargs = {} if precision_config is None else {"precision": precision_config}
            self.layer.shared_mlp.configure_decode(state_dict, mesh_device, sliding, shared_geometry, **shared_kwargs)
        if precision_config is not None:
            self.layer.shared_mlp.input_bfp8 = precision_config["shared_input_dtype"] == "bfloat8_b"
        positions = torch.arange(config.max_position_embeddings, dtype=torch.int32)[None]
        mapper = ttnn.ReplicateTensorToMesh(mesh_device)
        self.positions_u32 = ttnn.from_torch(
            positions, device=mesh_device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=mapper
        )
        self.positions_i32 = ttnn.from_torch(
            positions, device=mesh_device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=mapper
        )
        return self

    def _validate_kv_cache(self, kv_cache):
        super()._validate_kv_cache(kv_cache)
        if self.selected_precision is not None and any(cache.dtype != self.kv_cache_dtype for cache in kv_cache):
            raise ValueError("KV cache dtype does not match the constructed precision policy")

    def precision_summary(self):
        """Read the tensors and compute configs actually bound to each mode."""
        if self.selected_precision is None:
            raise ValueError("A resolved precision policy is required for the construction audit")
        attention = self.layer.self_attn
        qkv = attention.source.weights.wqkv
        experts = self.layer.moe.experts
        decode, prefill = experts.decode, experts.prefill
        shared = self.layer.shared_mlp
        router = self.layer.moe.router

        def closure_weight(project):
            if hasattr(project, "weight"):
                return project.weight
            weights = [
                cell.cell_contents for cell in project.__closure__ if isinstance(cell.cell_contents, ttnn.Tensor)
            ]
            if len(weights) != 1:
                raise ValueError("Cannot audit shared prefill projection weight")
            return weights[0]

        shared_prefill = [dtype_name(closure_weight(project).dtype) for project in (shared.gate_up, shared.down)]
        if shared_prefill[0] != shared_prefill[1]:
            raise ValueError("Shared prefill projection dtypes disagree")
        compute_groups = {
            "qkv": qkv.decode_compute,
            "output": attention.output_compute,
            "expert": decode.decode_compute,
            "shared": shared.decode_compute,
        }
        fixed = {
            "prefill_qkv_weight_dtype": dtype_name(qkv.prefill.weight.dtype),
            "prefill_qkv_fidelity": fidelity_name(qkv.prefill.compute.math_fidelity),
            "prefill_output_weight_dtype": dtype_name(attention.prefill_minimal_output.weight.dtype),
            "prefill_output_fidelity": fidelity_name(attention.prefill_output_compute.math_fidelity),
            "prefill_expert_gate_dtype": dtype_name(prefill.prefill_gate.dtype),
            "prefill_expert_down_dtype": dtype_name(prefill.prefill_down.dtype),
            "prefill_expert_fidelity": fidelity_name(prefill.prefill_compute.math_fidelity),
            "prefill_shared_weight_dtype": shared_prefill[0],
            "prefill_shared_fidelity": "library_default",
            "router_weight_dtype": dtype_name(router.projection_weight.dtype),
            "router_fidelity": fidelity_name(router.projection_compute.math_fidelity),
            "norm_weight_dtype": dtype_name(self.input_norm_weight.dtype),
            "norm_fidelity": fidelity_name(attention.compute.math_fidelity),
            "decode_sdpa_fidelity": fidelity_name(attention.decode_sdpa.compute.math_fidelity),
            "prefill_sdpa_fidelity": fidelity_name(attention.prefill_attention_compute.math_fidelity),
            "expert_mix_fidelity": fidelity_name(decode.mix_compute.math_fidelity),
            "expert_mix_fp32_dest_acc_en": decode.mix_compute.fp32_dest_acc_en,
            **{f"{key}_fp32_dest_acc_en": cfg.fp32_dest_acc_en for key, cfg in compute_groups.items()},
            **{f"{key}_packer_l1_acc": cfg.packer_l1_acc for key, cfg in compute_groups.items()},
            "math_approx_mode": any(cfg.math_approx_mode for cfg in compute_groups.values()),
        }
        summary = {
            "qkv_weight_dtype": dtype_name(qkv.weight.dtype),
            "qkv_fidelity": fidelity_name(qkv.decode_compute.math_fidelity),
            "output_weight_dtype": dtype_name(attention.source.weights.o_proj.dtype),
            "output_fidelity": fidelity_name(attention.output_compute.math_fidelity),
            "expert_gate_dtype": dtype_name(decode.gate_up.dtype),
            "expert_down_dtype": dtype_name(decode.down.dtype),
            "expert_fidelity": fidelity_name(decode.decode_compute.math_fidelity),
            "shared_gate_dtype": dtype_name(shared.decode_weights[0].dtype),
            "shared_down_dtype": dtype_name(shared.decode_weights[1].dtype),
            "shared_fidelity": fidelity_name(shared.decode_compute.math_fidelity),
            "qkv_input_dtype": "bfloat8_b" if qkv.input_bfp8 else "float32",
            "output_input_dtype": "bfloat8_b" if attention.output_input_bfp8 else "bfloat16",
            "expert_input_dtype": dtype_name(decode.decode_activation_dtype),
            "shared_input_dtype": "bfloat8_b" if shared.input_bfp8 else "bfloat16",
            "attention_ccl_dtype": dtype_name(self.attention_ccl_dtype),
            "moe_ccl_dtype": "bfloat8_b" if self.moe_ccl_bfp8 else "bfloat16",
            "kv_cache_dtype": dtype_name(self.kv_cache_dtype),
            "fixed": fixed,
        }
        assert_precision_matches(summary, self.selected_precision, f"layer.{self.layer_idx}")
        return summary

    def _prefill_continuation(self, hidden_states, *, rope_mats, page_table, kv_cache, user_id, start_pos):
        """Preserve per-token cache updates while bounding retained output tiles."""
        length = hidden_states.shape[-2]
        self._validate_kv_cache(kv_cache)
        if length <= 0 or start_pos < 0 or start_pos + length > self.config.max_position_embeddings:
            raise ValueError("Sequence outside HF context contract")
        rope_2d = tuple(ttnn.reshape(r, (r.shape[-2], r.shape[-1])) for r in rope_mats)
        request_table = page_table[user_id : user_id + 1, :]
        rows, tiles, chunks, groups = [], [], [], []

        def merge(parts):
            result = parts[0] if len(parts) == 1 else ttnn.concat(parts, dim=2)
            parts.clear()
            return result

        for offset in range(length):
            position = start_pos + offset
            rows.append(
                self.decode_forward(
                    hidden_states[:, :, offset : offset + 1, :],
                    rope_mats=rope_2d,
                    current_pos=self.positions_u32[:, position : position + 1],
                    cache_pos=ttnn.reshape(self.positions_i32[:, position : position + 1], (1,)),
                    page_table=request_table,
                    kv_cache=kv_cache,
                )
            )
            if len(rows) == 32:
                # Only these one-token tensors need concat's bounded RM path.
                tiles.append(merge(rows))
                if len(tiles) == 32:
                    chunks.append(merge(tiles))
                    if len(chunks) == 32:
                        groups.append(merge(chunks))
        if length == 1:
            return rows[0]
        if rows:
            # Padding participates only in assembly and is sliced away below.
            # Reusing a row creates no new cache update or tensor allocation.
            rows.extend([rows[-1]] * (32 - len(rows)))
            tiles.append(merge(rows))
        if tiles:
            chunks.append(merge(tiles))
        if chunks:
            groups.append(merge(chunks))
        # At most eight groups at 262144 tokens. Clearing references also works
        # when a single-input merge aliases its input; do not force deallocate.
        result = merge(groups)
        return result if result.shape[-2] == length else result[:, :, :length, :]

    def prefill_forward(self, hidden_states, *, rope_mats, page_table, kv_cache, user_id=0, start_pos=0):
        """Assemble long replicated prefill outputs in bounded tiled groups."""
        length = hidden_states.shape[-2]
        if start_pos > 0:
            return self._prefill_continuation(
                hidden_states,
                rope_mats=rope_mats,
                page_table=page_table,
                kv_cache=kv_cache,
                user_id=user_id,
                start_pos=start_pos,
            )
        if start_pos < 0 or length <= self.chunk_size or self.sharded_residual:
            return super().prefill_forward(
                hidden_states,
                rope_mats=rope_mats,
                page_table=page_table,
                kv_cache=kv_cache,
                user_id=user_id,
                start_pos=start_pos,
            )
        self._validate_kv_cache(kv_cache)
        if length > self.config.max_position_embeddings:
            raise ValueError("Sequence outside HF context contract")
        block = kv_cache[0].shape[-2]
        if self.chunk_size % block:
            raise ValueError("Chunk size must be a multiple of the cache page size")
        attention = self.layer.self_attn
        attention._release_sliding_prefill_tail(clear_persistent=True)
        groups, chunks = [], []
        for start in range(0, length, self.chunk_size):
            valid = min(self.chunk_size, length - start)
            tiled_rows = (valid + 31) // 32 * 32
            short_sliding_tail = attention.config.is_sliding and start > 0 and valid < self.config.sliding_window
            physical = self.config.sliding_window if short_sliding_tail else tiled_rows
            x = hidden_states[:, :, start : start + valid, :]
            if physical != valid:
                x = ttnn.pad(x, [(0, 0), (0, 0), (0, physical - valid), (0, 0)], 0.0)
            rope = tuple(r[:, :, start : start + physical, :] for r in rope_mats)
            chunk_table = page_table[:, start // block : (start + valid + block - 1) // block]
            chunk_start = self.positions_i32[:, start : start + 1] if short_sliding_tail else start
            out = self._forward(
                x,
                rope_mats=rope,
                page_table=page_table,
                kv_cache=kv_cache,
                position_idx=None,
                is_decode=False,
                user_id=user_id,
                valid_seq_len=valid,
                chunk_start_idx=chunk_start,
                chunk_page_table=chunk_table,
                retain_prefill_tail=start + valid < length,
            )
            # Keep the final tile physical until assembly is complete. A logical
            # partial tile here makes concat untilize every retained chunk.
            chunks.append(out if out.shape[-2] == tiled_rows else out[:, :, :tiled_rows, :])
            del out
            if len(chunks) == 32:
                groups.append(ttnn.concat(chunks, dim=2))
                chunks.clear()
        if chunks:
            groups.append(chunks[0] if len(chunks) == 1 else ttnn.concat(chunks, dim=2))
            chunks.clear()
        # At the supported context limit there are at most eight groups, below
        # concat's 47-input batching threshold. Release them before final unpad.
        result = groups[0] if len(groups) == 1 else ttnn.concat(groups, dim=2)
        groups.clear()
        return result if result.shape[-2] == length else result[:, :, :length, :]

    def _reduce_moe_pair(self, shared, routed):
        if shared.dtype != ttnn.bfloat16 or routed.dtype != ttnn.bfloat16:
            raise ValueError("Grouped MoE reduction requires BF16 local outputs")
        if tuple(shared.shape) != tuple(routed.shape) or tuple(shared.shape)[:2] != (1, 1):
            raise ValueError("Grouped MoE outputs must share shape [1,1,S,H]")
        memory = (
            getattr(self, "collective_memory", ttnn.DRAM_MEMORY_CONFIG)
            if shared.shape[-2] == 1
            else ttnn.DRAM_MEMORY_CONFIG
        )
        paired = ttnn.concat((shared, routed), dim=1, memory_config=memory)
        if getattr(self, "moe_ccl_bfp8", False):
            paired = ttnn.typecast(paired, ttnn.bfloat8_b)
        reduced = self.allreduce(paired, role="moe_pair")
        if reduced.dtype != ttnn.bfloat16:
            reduced = ttnn.typecast(reduced, ttnn.bfloat16)
        return reduced[:, :1, :, :], reduced[:, 1:2, :, :]

    def _reduce_attention(self, value):
        return self.allreduce(ttnn.typecast(value, self.attention_ccl_dtype), role="attention")

    def allreduce(self, value, *, role="shared"):
        memory = (
            getattr(self, "collective_memory", ttnn.DRAM_MEMORY_CONFIG)
            if value.shape[-2] == 1
            else ttnn.DRAM_MEMORY_CONFIG
        )
        value = ttnn.to_memory_config(value, memory)
        if self.persistent_ccl and value.shape[-2] == 1:
            key = (role, tuple(value.shape), tuple(value.padded_shape), value.dtype, str(memory))
            if key not in self._collective_buffers:
                output_shape = list(value.shape)
                output_shape[-1] //= 4

                def allocate(shape):
                    return ttnn.empty(
                        shape,
                        dtype=value.dtype,
                        layout=ttnn.TILE_LAYOUT,
                        device=self.mesh_device,
                        memory_config=memory,
                    )

                scattered = allocate(output_shape)
                gathered = allocate(value.shape)
                if self.topology == ttnn.Topology.Ring:
                    staging = ttnn.experimental.reduce_scatter_minimal_async_create_intermediate_buffer(
                        value,
                        dim=3,
                        topology=self.topology,
                        cluster_axis=1,
                    )
                    buffers = [staging[0], scattered, staging[1]]
                else:
                    intermediate_shape = list(value.padded_shape)
                    intermediate_shape[0] *= 2
                    buffers = [allocate(intermediate_shape), scattered]
                self._collective_buffers[key] = (buffers, gathered)
            buffers, gathered = self._collective_buffers[key]
            scattered = ttnn.experimental.reduce_scatter_minimal_async(
                value,
                persistent_output_buffers=buffers,
                dim=3,
                multi_device_global_semaphore=self.ccl.get_rs_ping_pong_semaphore(),
                barrier_semaphore=self.ccl.get_barrier_semaphore(),
                num_links=1,
                memory_config=memory,
                topology=self.topology,
                cluster_axis=1,
                **getattr(self, "ccl_tuning", {}),
            )
            tuning = getattr(self, "ccl_tuning", {})
            output_args = (
                {"persistent_output_buffer": gathered}
                if "chunks_per_sync" in tuning
                else {"persistent_output_tensor": gathered, "mesh_device": self.mesh_device}
            )
            return ttnn.experimental.all_gather_async(
                scattered,
                **output_args,
                dim=3,
                cluster_axis=1,
                topology=self.topology,
                multi_device_global_semaphore=self.ccl.get_ag_ping_pong_semaphore(),
                barrier_semaphore=self.ccl.get_barrier_semaphore(),
                num_links=1,
                memory_config=memory,
                **getattr(self, "ccl_tuning", {}),
            )
        return self.mesh_config.allreduce(value, self.ccl, memory_config=memory, axis=1)

    def reduce_scatter(self, value, *, role="shared"):
        value = ttnn.to_memory_config(value, ttnn.DRAM_MEMORY_CONFIG)
        buffers = None
        if self.persistent_ccl and value.shape[-2] == 1:
            key = ("scatter", role, tuple(value.shape), value.dtype)
            if key not in self._collective_buffers:
                shape = list(value.shape)
                shape[-1] //= 4
                scattered = ttnn.empty(
                    shape,
                    dtype=value.dtype,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.mesh_device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                if self.topology == ttnn.Topology.Ring:
                    staging = ttnn.experimental.reduce_scatter_minimal_async_create_intermediate_buffer(
                        value,
                        dim=3,
                        topology=self.topology,
                        cluster_axis=1,
                    )
                    buffers = [staging[0], scattered, staging[1]]
                else:
                    intermediate_shape = list(value.padded_shape)
                    intermediate_shape[0] *= 2
                    intermediate = ttnn.empty(
                        intermediate_shape,
                        dtype=value.dtype,
                        layout=ttnn.TILE_LAYOUT,
                        device=self.mesh_device,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    )
                    buffers = [intermediate, scattered]
                self._collective_buffers[key] = buffers
            buffers = self._collective_buffers[key]
        return ttnn.experimental.reduce_scatter_minimal_async(
            value,
            persistent_output_buffers=buffers,
            dim=3,
            multi_device_global_semaphore=self.ccl.get_rs_ping_pong_semaphore(),
            num_links=1,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=self.topology,
            cluster_axis=1,
            barrier_semaphore=self.ccl.get_barrier_semaphore(),
        )

    def gather(self, value):
        if self.persistent_ccl and value.shape[-2] == 1:
            key = ("gather", tuple(value.shape), value.dtype)
            if key not in self._collective_buffers:
                shape = list(value.shape)
                shape[-1] *= 4
                self._collective_buffers[key] = ttnn.empty(
                    shape,
                    dtype=value.dtype,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.mesh_device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
            return ttnn.experimental.all_gather_async(
                value,
                persistent_output_tensor=self._collective_buffers[key],
                dim=3,
                cluster_axis=1,
                mesh_device=self.mesh_device,
                topology=self.topology,
                multi_device_global_semaphore=self.ccl.get_ag_ping_pong_semaphore(),
                barrier_semaphore=self.ccl.get_barrier_semaphore(),
                num_links=1,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        return self.mesh_config.allgather(value, self.ccl, axis=1)

    def distributed_norm(self, value, name=None):
        value = ttnn.to_memory_config(value, ttnn.DRAM_MEMORY_CONFIG)
        stats = ttnn.rms_norm_pre_all_gather(
            value, dtype=ttnn.float32, compute_kernel_config=self.layer.self_attn.compute
        )
        stats = self.gather(stats)
        result = ttnn.rms_norm_post_all_gather(
            value, stats, epsilon=self.config.rms_norm_eps, compute_kernel_config=self.layer.self_attn.compute
        )
        return result if name is None else ttnn.mul(result, self.shard_norm_weights[name])

    def _sharded_forward(self, x, **attention_kwargs):
        normed = self.distributed_norm(x, "input_layernorm")
        if not (self.fused_agmm and attention_kwargs.get("is_decode", True)):
            normed = self.gather(normed)
        attention = self.layer.self_attn(normed, **attention_kwargs)
        residual = ttnn.add(
            ttnn.typecast(x, ttnn.float32), self.distributed_norm(attention, "post_attention_layernorm")
        )
        normalized = self.gather(self.distributed_norm(residual))
        routes = self.layer.moe.router(residual, normalized=normalized)
        expert_input = ttnn.mul(normalized, self.expert_norm_weight, dtype=ttnn.bfloat16)
        routed = self.layer.moe.experts(expert_input, routes)
        if getattr(self, "sharded_moe_bfp8", False):
            routed = ttnn.typecast(routed, ttnn.bfloat8_b)
        routed = self.reduce_scatter(routed, role="routed")
        shared_input = ttnn.mul(normalized, self.shared_norm_weight, dtype=ttnn.bfloat16)
        shared = self.layer.shared_mlp(shared_input)
        combined = ttnn.add(
            self.distributed_norm(shared, "post_feedforward_layernorm_1"),
            self.distributed_norm(routed, "post_feedforward_layernorm_2"),
        )
        combined = self.distributed_norm(combined, "post_feedforward_layernorm")
        return ttnn.typecast(ttnn.mul(ttnn.add(residual, combined), self.layer.layer_scalar), ttnn.bfloat16)

    def decode_forward(self, hidden_states, *, rope_mats, current_pos, cache_pos, page_table, kv_cache):
        shared = self.layer.shared_mlp
        if (
            getattr(self, "batched_shared_decode", False)
            and 8 <= hidden_states.shape[-2] <= 32
            and self.topology == ttnn.Topology.Linear
            and not self.sharded_residual
            and self.grouped_moe_reduce
            and self.fused_tail
            and type(shared) is _SharedMLP
            and shared.decode_weights is not None
        ):
            return self._decode_shared_batch(
                hidden_states,
                rope_mats=rope_mats,
                current_pos=current_pos,
                cache_pos=cache_pos,
                page_table=page_table,
                kv_cache=kv_cache,
            )
        return super().decode_forward(
            hidden_states,
            rope_mats=rope_mats,
            current_pos=current_pos,
            cache_pos=cache_pos,
            page_table=page_table,
            kv_cache=kv_cache,
        )

    def _decode_shared_batch(self, hidden_states, *, rope_mats, current_pos, cache_pos, page_table, kv_cache):
        """Keep per-slot attention/routing; amortize shared MLP, MoE CCL and tail."""
        self._validate_kv_cache(kv_cache)
        eps = self.config.rms_norm_eps
        residuals, routed_rows, shared_inputs = [], [], []
        for slot in range(hidden_states.shape[-2]):
            x = hidden_states[:, :, slot : slot + 1]
            memory = getattr(self, "decode_residual_memory", None)
            if memory is not None:
                x = ttnn.to_memory_config(x, memory)
            normed = self.normalize(x, eps, self.input_norm_weight)
            attention = self.layer.self_attn(
                normed,
                rope_mats=rope_mats,
                position_idx=current_pos[:, slot : slot + 1],
                position_idx_cache=cache_pos[slot : slot + 1],
                page_table=page_table[slot : slot + 1],
                kv_cache=kv_cache,
                is_decode=True,
            )
            # Consume the persistent attention collective result before the
            # next slot reuses its storage; residuals own independent buffers.
            residual = ttnn.add(
                ttnn.typecast(x, ttnn.float32), self.normalize(attention, eps, self.post_attention_norm_weight)
            )
            normalized = self.normalize(residual, eps)
            routes = self.layer.moe.router(residual, normalized=normalized)
            expert_input = ttnn.mul(normalized, self.expert_norm_weight, dtype=ttnn.bfloat16)
            if memory is not None:
                expert_input = ttnn.to_memory_config(expert_input, ttnn.L1_MEMORY_CONFIG)
            routed_rows.append(self.layer.moe.experts(expert_input, routes))
            shared_inputs.append(ttnn.mul(normalized, self.shared_norm_weight, dtype=ttnn.bfloat16))
            residuals.append(residual)
        shared = self.layer.shared_mlp.decode_batch(
            ttnn.concat(shared_inputs, dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG), reduce_output=False
        )
        shared, routed = self._reduce_moe_pair(
            shared, ttnn.concat(routed_rows, dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        )
        residual = ttnn.concat(residuals, dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return self._fused_tail(residual, shared, routed, True)

    def _forward(self, x, **attention_kwargs):
        if self.sharded_residual:
            return self._sharded_forward(x, **attention_kwargs)
        memory = getattr(self, "decode_residual_memory", None) if x.shape[-2] == 1 else None
        if memory is not None:
            x = ttnn.to_memory_config(x, memory)
        eps = self.config.rms_norm_eps
        normed = self.normalize(x, eps, self.input_norm_weight)
        attention = self.layer.self_attn(normed, **attention_kwargs)
        residual = ttnn.add(
            ttnn.typecast(x, ttnn.float32), self.normalize(attention, eps, self.post_attention_norm_weight)
        )
        normalized = self.normalize(residual, eps)
        routes = self.layer.moe.router(residual, normalized=normalized)
        expert_input = ttnn.mul(normalized, self.expert_norm_weight, dtype=ttnn.bfloat16)
        if memory is not None:
            expert_input = ttnn.to_memory_config(expert_input, ttnn.L1_MEMORY_CONFIG)
        routed = self.layer.moe.experts(expert_input, routes)
        if not self.grouped_moe_reduce:
            routed = self.allreduce(routed, role="routed")
        shared_input = ttnn.mul(normalized, self.shared_norm_weight, dtype=ttnn.bfloat16)
        if memory is not None:
            shared_input = ttnn.to_memory_config(shared_input, ttnn.L1_MEMORY_CONFIG)
        shared = self.layer.shared_mlp(shared_input, reduce_output=not self.grouped_moe_reduce)
        if self.grouped_moe_reduce:
            shared, routed = self._reduce_moe_pair(shared, routed)
        if self.fused_tail:
            return self._fused_tail(residual, shared, routed, x.shape[-2] <= 32)
        combined = ttnn.add(
            self.normalize(shared, eps, self.tail_weights[0]), self.normalize(routed, eps, self.tail_weights[1])
        )
        combined = self.normalize(combined, eps, self.tail_weights[2])
        return ttnn.typecast(ttnn.mul(ttnn.add(residual, combined), self.layer.layer_scalar), ttnn.bfloat16)

    def _fused_tail(self, residual, shared, routed, decode):
        layer = self.layer
        norm = layer.post_feedforward_layernorm
        debug = getattr(self, "debug_residual", False) and residual.shape[-2] == 1
        if debug:
            self.debug_tensors = {
                name: ttnn.to_memory_config(value, ttnn.DRAM_MEMORY_CONFIG)
                for name, value in (("residual", residual), ("shared", shared), ("routed", routed))
            }
        if decode:

            def normalize(value, norm):
                memory, program = norm._sharded_cfg
                return ttnn.rms_norm(
                    ttnn.to_memory_config(value, memory),
                    weight=norm.tt_weight,
                    epsilon=norm.eps,
                    program_config=program,
                )

            shared = normalize(shared, layer.post_feedforward_layernorm_1)
            routed = normalize(routed, layer.post_feedforward_layernorm_2)
            combined = ttnn.rms_norm(
                shared,
                residual_input_tensor=routed,
                weight=norm.tt_weight,
                epsilon=norm.eps,
                program_config=norm._sharded_cfg[1],
            )
        else:
            shared = layer.post_feedforward_layernorm_1.forward(shared)
            routed = layer.post_feedforward_layernorm_2.forward(routed)
            combined = ttnn.rms_norm(shared, residual_input_tensor=routed, weight=norm.tt_weight, epsilon=norm.eps)
        memory = getattr(self, "decode_residual_memory", None) if residual.shape[-2] == 1 else None
        if debug:
            self.debug_tensors["combined_before_reshard"] = ttnn.to_memory_config(combined, ttnn.DRAM_MEMORY_CONFIG)
        if memory is not None:
            combined = ttnn.to_memory_config(combined, memory)
        result = ttnn.add(
            residual,
            combined,
            dtype=ttnn.bfloat16,
            memory_config=memory or ttnn.DRAM_MEMORY_CONFIG,
            fast_and_approximate_mode=memory is None,
            activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.MUL_UNARY_SFPU, layer.layer_scalar)],
        )
        if debug:
            self.debug_tensors.update(
                combined_after_reshard=ttnn.to_memory_config(combined, ttnn.DRAM_MEMORY_CONFIG),
                final=ttnn.to_memory_config(result, ttnn.DRAM_MEMORY_CONFIG),
            )
        return result
