# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Per-device optimized decoder. Candidate policies are validated by stage probes.

Request state, logical tails and exact sensitive projections inherit the fused
contract. Host weight conversion is confined to construction.
"""
import math
from copy import copy

import ttnn

from .decoder_components import linear, rms
from .fused_decoder import (
    _PACKED_SILU_COMPUTE,
    _PACKED_SILU_READER,
    _PACKED_SILU_WRITER,
    _RESIDUAL_WRITE_COMPUTE,
    _RESIDUAL_WRITE_READER,
    _RESIDUAL_WRITE_WRITER,
    FusedDecoder,
    FusedResidualWriter,
    FusedSparseMoe,
    fused_compact_routes,
    fused_compact_union_routes,
)
from .precise_gdn_projection import PreciseGDNProjection
from .router_topk import aten_router_topk


def optimized_packed_silu(x, output_tensor=None):
    if (
        x.dtype != ttnn.bfloat16
        or x.layout != ttnn.TILE_LAYOUT
        or x.memory_config() not in (ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG)
        or len(x.shape) < 2
        or x.shape[-1] % 64
    ):
        raise ValueError("Packed SiLU requires interleaved BF16 tiles with tile-aligned half width")
    device = x.device()
    shape = list(x.shape)
    shape[-1] //= 2
    out = output_tensor
    if out is None:
        out = ttnn.empty(
            shape, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=x.memory_config()
        )
    elif (
        list(out.shape) != shape
        or out.dtype != ttnn.bfloat16
        or out.layout != ttnn.TILE_LAYOUT
        or out.device() != device
        or out.memory_config() not in (ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG)
    ):
        raise ValueError("Packed activation destination must match the output shape, dtype, layout and device")
    tiles = math.prod(shape[:-2]) * ((shape[-2] + 31) // 32) * (shape[-1] // 32)
    half_tiles = shape[-1] // 32
    grid = device.compute_with_storage_grid_size()
    n = min(grid.x * grid.y, tiles)
    cores = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(n)]
    ranges = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])
    reader, compute, writer = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    start = 0
    for i, c in enumerate(cores):
        count = tiles // n + int(i < tiles % n)
        reader[c.x][c.y] = [x.buffer_address(), start, count, half_tiles]
        compute[c.x][c.y] = [count]
        writer[c.x][c.y] = [out.buffer_address(), start, count]
        start += count
    config = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=False, dst_full_sync_en=False
    )
    config.unpack_to_dest_mode = [ttnn.UnpackToDestMode.UnpackToDestFp32] * 64
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=source,
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=ranges,
            compile_time_args=ct,
            runtime_args=rt,
            config=config,
        )
        for source, ct, rt, config in [
            (
                _PACKED_SILU_READER,
                ttnn.TensorAccessorArgs(x).get_compile_time_args(),
                reader,
                ttnn.ReaderConfigDescriptor(),
            ),
            (_PACKED_SILU_COMPUTE, [], compute, config),
            (
                _PACKED_SILU_WRITER,
                ttnn.TensorAccessorArgs(out).get_compile_time_args(),
                writer,
                ttnn.WriterConfigDescriptor(),
            ),
        ]
    ]
    cbs = [
        ttnn.CBDescriptor(
            total_size=4096,
            core_ranges=ranges,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.bfloat16, page_size=2048)],
        )
        for i in range(4)
    ]
    ttnn.generic_op([x, out], ttnn.ProgramDescriptor(kernels=kernels, cbs=cbs, semaphores=[]))
    return out


def optimized_residual_write(residual, block, injection):
    tokens = residual.shape[-2]
    if (
        tuple(residual.shape) != (1, 1, tokens, 10240)
        or tuple(block.shape) != (1, 1, tokens, 2560)
        or tuple(injection.shape) != (1, 4, tokens, 1)
        or tokens < 1
        or any(t.dtype != ttnn.bfloat16 or t.layout != ttnn.TILE_LAYOUT for t in (residual, block, injection))
        or any(t.device() != residual.device() for t in (block, injection))
    ):
        raise ValueError("Residual write requires matching BF16 TILE four-stream inputs")
    padded_tokens = (tokens + 31) // 32 * 32
    for tensor, expected in (
        (residual, (1, 1, padded_tokens, 10240)),
        (block, (1, 1, padded_tokens, 2560)),
        (injection, (1, 4, padded_tokens, 32)),
    ):
        tile = tensor.get_tile()
        if tuple(tile.tile_shape) != (32, 32) or tile.transpose_within_face or tile.transpose_of_faces:
            raise ValueError("Residual write requires standard 32x32 tiles")
        if tuple(tensor.padded_shape) != expected:
            raise ValueError("Residual write received unexpected physical padding")
    mesh = residual.device()
    output = ttnn.empty_like(residual)
    mt = (tokens + 31) // 32
    total = mt * 320
    grid = mesh.compute_with_storage_grid_size()
    cores = min(total, grid.x * grid.y)
    ranges = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(i % grid.x, i // grid.x), ttnn.CoreCoord(i % grid.x, i // grid.x))
            for i in range(cores)
        ]
    )
    reader, compute, writer = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for i in range(cores):
        x, y = i % grid.x, i // grid.x
        start, end = total * i // cores, total * (i + 1) // cores
        reader[x][y] = [
            block.buffer_address(),
            injection.buffer_address(),
            residual.buffer_address(),
            start,
            end - start,
            mt,
        ]
        compute[x][y] = [end - start]
        writer[x][y] = [output.buffer_address(), start, end - start]
    ct = []
    for t in (block, injection, residual):
        ct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=source,
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=ranges,
            compile_time_args=args,
            runtime_args=rt,
            config=config,
        )
        for source, args, rt, config in [
            (_RESIDUAL_WRITE_READER, ct, reader, ttnn.ReaderConfigDescriptor()),
            (
                _RESIDUAL_WRITE_COMPUTE,
                [],
                compute,
                ttnn.ComputeConfigDescriptor(
                    math_fidelity=ttnn.MathFidelity.HiFi4,
                    fp32_dest_acc_en=True,
                    math_approx_mode=False,
                    dst_full_sync_en=False,
                ),
            ),
            (
                _RESIDUAL_WRITE_WRITER,
                ttnn.TensorAccessorArgs(output).get_compile_time_args(),
                writer,
                ttnn.WriterConfigDescriptor(),
            ),
        ]
    ]
    cbs = [
        ttnn.CBDescriptor(
            total_size=4096,
            core_ranges=ranges,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.bfloat16, page_size=2048)],
        )
        for i in range(4)
    ]
    ttnn.generic_op(
        [block, injection, residual, output], ttnn.ProgramDescriptor(kernels=kernels, cbs=cbs, semaphores=[])
    )
    return output


def optimized_residual_read(component, residual):
    """Same exact arithmetic as GatedResidual.read, with explicit boundaries.

    Keep the stored residual untouched. Interleave only the reader's view. The
    expanded SFPU projection products can be tens of MiB at 32 rows, so their
    input moves to DRAM once and is shared by both same-input projections.
    """

    def convert(value, target, reason):
        if value.memory_config() == target:
            return value
        return ttnn.to_memory_config(value, target)

    work = ttnn.L1_MEMORY_CONFIG
    x = convert(residual, work, "stored_residual_to_read_consumer")
    tokens = x.shape[-2]
    grouped = ttnn.permute(ttnn.reshape(x, (1, tokens, component.streams, component.width)), (0, 2, 1, 3))
    normalized = rms(grouped, component.weight, component.eps)
    flattened = ttnn.reshape(
        ttnn.permute(normalized, (0, 2, 1, 3)), (1, 1, tokens, component.streams * component.width)
    )
    flattened = convert(flattened, ttnn.DRAM_MEMORY_CONFIG, "normalized_to_dram_projection_workspace")
    scaled = ttnn.multiply(component.precise_down(flattened), 1 / component.streams)
    mixing = ttnn.typecast(ttnn.silu(ttnn.typecast(scaled, ttnn.float32)), ttnn.bfloat16)
    mixing = ttnn.typecast(ttnn.sigmoid(ttnn.typecast(component.precise_up(mixing), ttnn.float32)), ttnn.bfloat16)
    mixing = ttnn.permute(ttnn.reshape(mixing, (1, tokens, component.streams, component.width)), (0, 2, 1, 3))
    mixed = ttnn.typecast(
        ttnn.mean(ttnn.typecast(ttnn.multiply(mixing, normalized), ttnn.float32), dim=1, keepdim=True),
        ttnn.bfloat16,
    )
    scaled_injection = ttnn.multiply(linear(flattened, component.inject, component.compute), 1 / component.streams)
    injection = ttnn.multiply(
        ttnn.typecast(ttnn.sigmoid(ttnn.typecast(scaled_injection, ttnn.float32)), ttnn.bfloat16), 2
    )
    injection = ttnn.permute(ttnn.reshape(injection, (1, 1, tokens, component.streams)), (0, 3, 2, 1))
    return mixed, injection


class L1Residual(FusedResidualWriter):
    def read(self, x):
        return optimized_residual_read(self, x)

    def write(self, residual, block, injection):
        if residual.memory_config() != self.residual_memory:
            residual = ttnn.to_memory_config(residual, self.residual_memory)
        return optimized_residual_write(residual, block, injection)


class DramLinear(PreciseGDNProjection):
    """One DRAM-sharded weight copy for physical chunks of at most32 rows.

    The callable protocol is shared with exact projection objects. The public
    decoder still handles arbitrary logical lengths through its existing chunks.
    """

    def __init__(self, weight, device, *, dtype, cores=4, readers=2, block=2, fidelity="HiFi2"):
        k, n = weight.shape
        if k % (32 * cores) or (k // (32 * cores)) % block:
            raise ValueError("Input shard width must divide K and the selected K block")
        self.shape = (1, 1, k, n)
        self.cores, self.readers, self.block = cores, readers, block
        self.grid = ttnn.num_cores_to_corerangeset(cores, device.compute_with_storage_grid_size(), row_wise=True)
        banks = device.dram_grid_size().x
        bank_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(banks - 1, 0))})
        memory = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.DRAM,
            ttnn.ShardSpec(bank_grid, (k, math.ceil(n / (32 * banks)) * 32), ttnn.ShardOrientation.ROW_MAJOR),
        )
        self.weight = ttnn.from_torch(
            weight.reshape(self.shape).contiguous(),
            device=device,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=memory,
        )
        self.compute = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=getattr(ttnn.MathFidelity, fidelity),
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    def __call__(self, value):
        if not 1 <= value.shape[-2] <= 32:
            raise ValueError("DRAM projection requires a physical one-tile-row chunk")
        k, n = self.shape[-2:]

        def memory(width):
            return ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(self.grid, (32, width), ttnn.ShardOrientation.ROW_MAJOR),
            )

        activation = ttnn.to_memory_config(value, memory(k // self.cores))
        width = math.ceil(n / (32 * self.cores)) * 32
        program = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
            in0_block_w=self.block,
            per_core_M=1,
            per_core_N=width // 32,
            fused_activation=None,
            num_workers_per_dram_bank=self.readers,
            use_multicore_pipeline=self.readers >= 2,
        )
        output = ttnn.linear(
            activation,
            self.weight,
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.compute,
            program_config=program,
            memory_config=memory(width),
        )
        return ttnn.to_memory_config(output, ttnn.DRAM_MEMORY_CONFIG)


class OptimizedSparseMoe(FusedSparseMoe):
    def __call__(self, x):
        tokens = x.shape[-2]
        if tokens > 32:
            raise ValueError("SparseMoe physical chunk is at most one tile; caller must chunk")
        logits = self.router_logits(x)
        scores, indices = aten_router_topk(logits)
        scores = ttnn.softmax(scores, dim=-1, numeric_stable=True, compute_kernel_config=self.compute)
        compact = getattr(self, "compact_experts", False) and tokens <= getattr(self, "compact_prefill_limit", 1)
        if compact:
            if tokens == 1:
                compact_ids, compact_scores = fused_compact_routes(indices, ttnn.typecast(scores, ttnn.bfloat16))
                compact_reduce_ids = compact_ids
            else:
                compact_ids, compact_scores, compact_reduce_ids = fused_compact_union_routes(
                    indices, ttnn.typecast(scores, ttnn.bfloat16)
                )
            active = self.compact_sparsity
        else:
            if getattr(self, "fuse_moe_cleanup", False):
                # zeros_like with a dtype override may upload small tensors.
                # Explicit device storage/fill is legal inside trace capture.
                zero_scores = ttnn.empty_like(logits, dtype=ttnn.bfloat16)
                zero_scores = ttnn.fill(zero_scores, 0.0, output_tensor=zero_scores)
            else:
                zero_scores = ttnn.zeros_like(ttnn.typecast(logits, ttnn.bfloat16))
            dense = ttnn.scatter(
                zero_scores,
                dim=3,
                index=indices,
                src=ttnn.typecast(scores, ttnn.bfloat16),
            )
            del zero_scores
            # Sparse matmul chooses experts per token tile. Union the selected experts
            # across rows, retaining the independent per-token scores for reduction.
            active = ttnn.to_layout(ttnn.max(dense, dim=2, keepdim=True), ttnn.ROW_MAJOR_LAYOUT)
        kwargs = dict(
            sparsity=active,
            nnz=None,
            memory_config=ttnn.L1_MEMORY_CONFIG if compact and self.expert_l1_output else ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.expert_compute,
        )
        if self.expert_l1_input:
            x = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)
        if compact:
            kwargs["indices"] = compact_ids
        if hasattr(self, "expert_packed"):
            activation_destination = None
            if getattr(self, "early_packed_activation", False):
                # Keep the surviving activation in the first allocation, before
                # the wider temporary. This changes placement, not live bytes.
                shape = [*list(x.shape)[:-2], *list(self.expert_packed.shape)[:-2], tokens, self.intermediate]
                if compact:
                    shape[-3] = compact_ids.shape[-1]
                activation_destination = ttnn.empty(
                    shape,
                    device=x.device(),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=kwargs["memory_config"],
                )
            program = self.program.get_decode_gate_up_config(tokens, 2 * self.intermediate, k=self.width)
            packed = ttnn.sparse_matmul(x, self.expert_packed, program_config=program, **kwargs)
            if getattr(self, "fuse_packed_activation", False):
                activated = optimized_packed_silu(packed, output_tensor=activation_destination)
                del activation_destination
            else:
                gate, up = packed[..., : self.intermediate], packed[..., self.intermediate :]
            del packed
        else:
            program = self.program.get_decode_gate_up_config(tokens, self.intermediate, k=self.width)
            if getattr(self, "matmul_expert_activation", False):
                program.fused_activation = ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU)
            gate = ttnn.sparse_matmul(x, self.gate, program_config=program, **kwargs)
            program = self.program.get_decode_gate_up_config(tokens, self.intermediate, k=self.width)
            up = ttnn.sparse_matmul(x, self.up, program_config=program, **kwargs)
        if not (hasattr(self, "expert_packed") and getattr(self, "fuse_packed_activation", False)):
            if getattr(self, "matmul_expert_activation", False):
                activated = ttnn.multiply(gate, up, output_tensor=gate if self.reuse_private_buffers else None)
            elif self.fuse_expert_activation:
                activated = ttnn.multiply(
                    gate,
                    up,
                    input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
                    output_tensor=gate if self.reuse_private_buffers else None,
                )
            elif self.reuse_private_buffers:
                # The expert gate has no other consumer. Preserve both BF16
                # boundaries while avoiding two private 20 MiB temporaries.
                activated = ttnn.multiply(ttnn.silu(gate, output_tensor=gate), up, output_tensor=gate)
            else:
                activated = ttnn.multiply(ttnn.silu(gate), up)
            # These private intermediates have no further consumers. Release their
            # references before the 80 MiB per-expert output allocation on PLE.
            del gate, up
        down = ttnn.sparse_matmul(
            activated,
            self.down,
            is_input_a_sparse=True,
            program_config=self.program.get_decode_down_config(tokens, self.width, k=self.intermediate),
            **kwargs,
        )
        del activated
        if hasattr(self, "reduce_indices"):
            scores_rm = (
                compact_scores
                if compact
                else (
                    ttnn.reshape(ttnn.to_layout(dense, ttnn.ROW_MAJOR_LAYOUT), (tokens, 1, 1, self.experts))
                    if getattr(self, "fuse_moe_cleanup", False)
                    else ttnn.to_layout(ttnn.permute(dense, (2, 0, 1, 3)), ttnn.ROW_MAJOR_LAYOUT)
                )
            )
            reduction_ids = compact_reduce_ids if compact else self.reduce_indices[tokens]
            routed = ttnn.experimental.deepseek_moe_fast_reduce_nc_fused(
                down,
                reduction_ids,
                self.reduce_mapping,
                reduce_dim=1,
                split_size=self.width,
                cluster_axis=0,
                output_memory_config=ttnn.DRAM_MEMORY_CONFIG,
                scores_tensor=scores_rm,
                compute_kernel_config=self.compute,
            )[0]
        else:
            weights = ttnn.permute(ttnn.typecast(dense, ttnn.bfloat16), (0, 3, 2, 1))
            # The private per-expert output has no other consumer. Weight it in
            # place so the resident PLE table need not coexist with an 80 MiB copy.
            ttnn.multiply(down, weights, output_tensor=down)
            if self.reuse_private_buffers and tokens < 32:
                # Generic sum clones a padded input before zeroing its padding.
                # This weighted down buffer is private, so zero it in place and
                # expose its existing physical height to the same reduction.
                # Explicit padded shapes make both reshapes metadata-only views.
                ttnn.fill_implicit_tile_padding(down, 0.0)
                down = ttnn.reshape(down, (1, self.experts, 32, self.width), (1, self.experts, 32, self.width))
                routed = ttnn.sum(down, dim=1, keepdim=True)
                routed = ttnn.reshape(routed, (1, 1, tokens, self.width), (1, 1, 32, self.width))
            else:
                routed = ttnn.sum(down, dim=1, keepdim=True)
        del down
        if hasattr(self, "shared_packed"):
            packed = linear(x, self.shared_packed, self.compute)
            gate = packed[..., : self.shared_width]
            up = packed[..., self.shared_width : 2 * self.shared_width]
            route = packed[..., 2 * self.shared_width :]
            del packed
        else:
            gate = (
                ttnn.linear(
                    x,
                    self.shared_gate,
                    activation="silu",
                    compute_kernel_config=self.compute,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                if getattr(self, "matmul_shared_activation", False)
                else linear(x, self.shared_gate, self.compute)
            )
            up = linear(x, self.shared_up, self.compute)
            route = linear(x, self.shared_route, self.compute)
        shared = (
            ttnn.multiply(gate, up)
            if getattr(self, "matmul_shared_activation", False)
            else (
                ttnn.multiply(gate, up, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
                if self.fuse_shared_activation
                else ttnn.multiply(ttnn.silu(gate), up)
            )
        )
        shared = linear(shared, self.shared_down, self.compute)
        shared = (
            ttnn.multiply(shared, route, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])
            if getattr(self, "fuse_moe_cleanup", False)
            else ttnn.multiply(shared, ttnn.sigmoid(route))
        )
        return ttnn.add(routed, shared)


class OptimizedDecoder(FusedDecoder):
    @classmethod
    def from_state_dict(
        cls,
        *args,
        expert_dtype=ttnn.bfloat8_b,
        expert_gate_dtype=None,
        expert_down_dtype=None,
        shared_dtype="bfloat8_b",
        attention_dtype=None,
        ple_dtype="bfloat8_b",
        dense_programs=None,
        residual_storage="l1",
        sdpa_k_chunk=128,
        sdpa_grid=(8, 8),
        expert_fidelity="LoFi",
        gate_cores=(10, 4),
        down_cores=(10, 4),
        gate_block=20,
        down_block=10,
        gate_subblock=1,
        down_subblock=2,
        expert_l1_input=True,
        expert_l1_output=True,
        **kwargs,
    ):
        model = super().from_state_dict(*args, expert_dtype=expert_dtype, **kwargs)
        # GDN z/a/b feeds recurrent gates and subsequently discrete MoE routing.
        # Real batch/continuation controls require both BF16 weights and HiFi4;
        # QSA's ordinary projections retain the independently validated BFP8 policy.
        if attention_dtype is None:
            attention_dtype = "bfloat16" if model.kind == "linear_attention" else "bfloat8_b"
        # Group precision candidates reload original checkpoint tensors, never
        # quantize an already quantized expert as a proxy for the checkpoint.
        state_dict = args[0] if args else kwargs.get("state_dict")
        prefix = f"model.language_model.layers.{model.layer_idx}."
        checkpoint = {k.removeprefix(prefix): v for k, v in state_dict.items() if k.startswith(prefix)} or state_dict
        if expert_gate_dtype is not None:
            dtype = getattr(ttnn, expert_gate_dtype)
            packed = checkpoint["mlp.experts.gate_up_proj"]
            if hasattr(model.mlp, "expert_packed"):
                del model.mlp.expert_packed
                model.mlp.expert_packed = ttnn.from_torch(
                    packed.transpose(-1, -2).unsqueeze(0).contiguous(),
                    device=model.mesh_device,
                    dtype=dtype,
                    layout=ttnn.TILE_LAYOUT,
                )
            else:
                del model.mlp.gate, model.mlp.up
                gate, up = packed.chunk(2, dim=-2)
                model.mlp.gate = ttnn.from_torch(
                    gate.transpose(-1, -2).unsqueeze(0).contiguous(),
                    device=model.mesh_device,
                    dtype=dtype,
                    layout=ttnn.TILE_LAYOUT,
                )
                model.mlp.up = ttnn.from_torch(
                    up.transpose(-1, -2).unsqueeze(0).contiguous(),
                    device=model.mesh_device,
                    dtype=dtype,
                    layout=ttnn.TILE_LAYOUT,
                )
        if expert_down_dtype is not None:
            del model.mlp.down
            model.mlp.down = ttnn.from_torch(
                checkpoint["mlp.experts.down_proj"].transpose(-1, -2).unsqueeze(0).contiguous(),
                device=model.mesh_device,
                dtype=getattr(ttnn, expert_down_dtype),
                layout=ttnn.TILE_LAYOUT,
            )

        def convert(owner, names, dtype):
            if dtype is not None:
                for name in names:
                    if hasattr(owner, name):
                        setattr(owner, name, ttnn.typecast(getattr(owner, name), getattr(ttnn, dtype)))

        convert(model.mlp, ("shared_packed", "shared_gate", "shared_up", "shared_route", "shared_down"), shared_dtype)
        if model.kind == "linear_attention":
            convert(model.attn, ("zab",), attention_dtype)
        else:
            convert(model.chunked_attn, ("packed_qkv",), attention_dtype)
            convert(model.attn, ("q", "k", "v", "out"), attention_dtype)
        if model.ple is not None:
            convert(model.ple, ("key", "value"), ple_dtype)
        if dense_programs is None:
            dense_programs = {
                role: dict(cores=4, readers=2, block=2, fidelity="HiFi4" if role == "gdn" else "HiFi2")
                for role in ("gdn", "shared", "qkv", "attention_out", "ple_key", "ple_value")
            }
        if dense_programs:
            import torch

            roles = {
                "shared": (
                    model.mlp,
                    "shared_packed",
                    (
                        "mlp.shared_expert.gate_proj.weight",
                        "mlp.shared_expert.up_proj.weight",
                        "mlp.shared_expert_gate.weight",
                    ),
                    shared_dtype,
                ),
                "shared_down": (model.mlp, "shared_down", ("mlp.shared_expert.down_proj.weight",), shared_dtype),
            }
            if model.kind == "linear_attention":
                roles["gdn"] = (
                    model.attn,
                    "zab",
                    ("linear_attn.in_proj_z.weight", "linear_attn.in_proj_a.weight", "linear_attn.in_proj_b.weight"),
                    attention_dtype,
                )
            else:
                roles["qkv"] = (
                    model.chunked_attn,
                    "packed_qkv",
                    ("self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight"),
                    attention_dtype,
                )
                roles["attention_out"] = (model.attn, "out", ("self_attn.o_proj.weight",), attention_dtype)
            if model.ple is not None:
                roles["ple_key"] = (model.ple, "key", ("ple.key_proj.weight",), ple_dtype)
                roles["ple_value"] = (model.ple, "value", ("ple.value_proj.weight",), ple_dtype)
            for role, config in dense_programs.items():
                if role not in roles:
                    continue
                owner, field, names, dtype = roles[role]
                matrices = [checkpoint[name] for name in names]
                host = (torch.cat(matrices, dim=0) if len(matrices) > 1 else matrices[0]).T
                delattr(owner, field)
                setattr(
                    owner,
                    field,
                    DramLinear(host, model.mesh_device, dtype=getattr(ttnn, dtype or "bfloat16"), **config),
                )
        component = OptimizedSparseMoe.__new__(OptimizedSparseMoe)
        component.__dict__.update(model.mlp.__dict__)
        model.mlp = component
        if residual_storage not in ("dram", "l1", "width_sharded"):
            raise ValueError("Unknown residual storage")
        model.residual_storage = residual_storage
        if residual_storage != "dram":
            for name in ("attn_mix", "mlp_mix"):
                original = getattr(model, name)
                residual = L1Residual.__new__(L1Residual)
                residual.__dict__.update(original.__dict__)
                residual.residual_memory = ttnn.L1_MEMORY_CONFIG
                if residual_storage == "width_sharded":
                    grid = ttnn.num_cores_to_corerangeset(
                        32, model.mesh_device.compute_with_storage_grid_size(), row_wise=True
                    )
                    residual.residual_memory = ttnn.MemoryConfig(
                        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                        ttnn.BufferType.L1,
                        ttnn.ShardSpec(grid, (32, 320), ttnn.ShardOrientation.ROW_MAJOR),
                    )
                setattr(model, name, residual)
        model.configure_experts(
            sdpa_k_chunk=sdpa_k_chunk,
            sdpa_grid=sdpa_grid,
            expert_fidelity=expert_fidelity,
            gate_cores=gate_cores,
            down_cores=down_cores,
            gate_block=gate_block,
            down_block=down_block,
            gate_subblock=gate_subblock,
            down_subblock=down_subblock,
            expert_l1_input=expert_l1_input,
            expert_l1_output=expert_l1_output,
        )
        return model

    def configure_experts(
        self,
        *,
        expert_fidelity="HiFi4",
        sdpa_k_chunk=32,
        sdpa_grid=None,
        gate_cores=(10, 4),
        down_cores=(8, 1),
        gate_block=1,
        down_block=1,
        gate_subblock=1,
        down_subblock=1,
        expert_l1_input=False,
        expert_l1_output=False,
    ):
        """Setup only; all traces must be released before changing a policy."""
        if expert_fidelity not in ("LoFi", "HiFi2", "HiFi4"):
            raise ValueError("Unsupported expert fidelity")
        if self.chunked_attn is not None:
            grid = (
                self.mesh_device.compute_with_storage_grid_size() if sdpa_grid is None else ttnn.CoreCoord(*sdpa_grid)
            )
            self.chunked_attn.sdpa_program = ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=grid, q_chunk_size=32, k_chunk_size=sdpa_k_chunk, exp_approx_mode=False
            )
        self.expert_policy = dict(
            expert_fidelity=expert_fidelity,
            gate_cores=gate_cores,
            down_cores=down_cores,
            gate_block=gate_block,
            down_block=down_block,
            gate_subblock=gate_subblock,
            down_subblock=down_subblock,
            expert_l1_input=expert_l1_input,
            expert_l1_output=expert_l1_output,
            sdpa_k_chunk=sdpa_k_chunk,
            sdpa_grid=sdpa_grid,
        )
        self.mlp.program = copy(self.mlp.program)
        p = self.mlp.program
        p.decode_gate_up_cores = tuple(gate_cores)
        p.decode_down_cores = tuple(down_cores)
        p.decode_gate_up_in0_block_w = gate_block
        p.decode_down_in0_block_w = down_block
        p.decode_gate_up_subblock_w = gate_subblock
        p.decode_down_subblock_w = down_subblock
        self.mlp.expert_compute = ttnn.init_device_compute_kernel_config(
            self.mesh_device.arch(),
            math_fidelity=getattr(ttnn.MathFidelity, expert_fidelity),
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        self.mlp.expert_l1_input = expert_l1_input
        self.mlp.expert_l1_output = expert_l1_output

    def _chunk(self, hidden_states, state, token_ids=None):
        if self.residual_storage != "dram":
            hidden_states = ttnn.to_memory_config(hidden_states, self.attn_mix.residual_memory)
        output = super()._chunk(hidden_states, state, token_ids)
        # Public chunk assembly uses the inherited interleaved DRAM contract.
        if output.memory_config() != ttnn.DRAM_MEMORY_CONFIG:
            output = ttnn.to_memory_config(output, ttnn.DRAM_MEMORY_CONFIG)
        return output
