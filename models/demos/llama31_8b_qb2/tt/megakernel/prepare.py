# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Batch-one pre-attention stages with original precision and paged KV ownership."""

from pathlib import Path
import ttnn
from .mlp import _grid, _width_memory
from .norm import FusedNorm


class FusedPreparation:
    def __init__(self, body):
        self.body, self.mesh = body, body.mesh
        layer = body.layers[0]
        if body.gu_workers != 8 or self.mesh.compute_with_storage_grid_size().x < 11:
            raise ValueError("Complete layer composition requires GU8 and eleven worker columns")
        self.projection_cores = [ttnn.CoreCoord(x, 2) for x in range(8)]
        self.norm_cores = body.placement.map([ttnn.CoreCoord(x, 4) for x in range(2, 10)])
        self.rope_cores = [ttnn.CoreCoord(8, 2), ttnn.CoreCoord(9, 2)]
        self.cache_cores = [ttnn.CoreCoord(9, 3), ttnn.CoreCoord(10, 3)]
        self.cores = self.projection_cores + self.norm_cores + self.rope_cores + self.cache_cores
        self.normalizer = FusedNorm(
            self.mesh,
            layer.decode_inputs["qkv"],
            layer.eps,
            cores=self.norm_cores,
            output=body.normalizer.output,
        )
        self.packed = ttnn.empty(
            (1, 1, 1, 1536),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh,
            memory_config=_width_memory(self.projection_cores, 1536),
        )
        self.heads = []
        for core, heads in zip(self.rope_cores, (8, 2)):
            memory = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(_grid([core]), [32, 128], ttnn.ShardOrientation.ROW_MAJOR),
            )
            self.heads.append(
                ttnn.empty(
                    (1, 1, heads, 128),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.mesh,
                    memory_config=memory,
                )
            )
        self.cos, self.sin = layer.cos_embedding, layer.sin_embedding

    def append(self, program, layer_index, cache_inputs):
        k_cache, v_cache, position, page_table, rotary_position = cache_inputs
        for cache in (k_cache, v_cache):
            if (
                tuple(cache.shape)[1:] != (2, 128, 128)
                or cache.dtype != ttnn.bfloat8_b
                or cache.layout != ttnn.TILE_LAYOUT
                or cache.is_sharded()
            ):
                raise ValueError("Expected native interleaved BFP8 paged cache [pages,2,128,128]")
        for tensor in (position, rotary_position):
            if (
                tuple(tensor.shape) != (1,)
                or tensor.dtype != ttnn.int32
                or tensor.layout != ttnn.ROW_MAJOR_LAYOUT
                or tensor.is_sharded()
            ):
                raise ValueError("Expected one interleaved row-major int32 position")
        if (
            page_table.is_sharded()
            or page_table.dtype != ttnn.int32
            or page_table.layout != ttnn.ROW_MAJOR_LAYOUT
            or len(page_table.shape) != 2
            or page_table.shape[0] != 1
            or page_table.shape[1] < 1
        ):
            raise ValueError("Expected the native batch-one row-major int32 page table")
        program = self.normalizer.append(program, self.body.gather_output, self.projection_cores, wait_for_gather=True)
        program.semaphores = [
            *program.semaphores,
            *[ttnn.SemaphoreDescriptor(id=i, core_ranges=_grid(self.cores), initial_value=0) for i in range(13)],
        ]
        kernels, cbs = list(program.kernels), list(program.cbs)
        source = Path(__file__).with_name("kernels")
        accessor = lambda tensor: ttnn.TensorAccessorArgs(tensor).get_compile_time_args()
        physical = lambda cores: [
            v for core in cores for c in [self.mesh.worker_core_from_logical_core(core)] for v in (c.x, c.y)
        ]

        def cb(indices, count, dtype, cores):
            size = ttnn.Tile([32, 32]).get_tile_size(dtype)
            cbs.append(
                ttnn.CBDescriptor(
                    total_size=count * size,
                    core_ranges=_grid(cores),
                    format_descriptors=[
                        ttnn.CBFormatDescriptor(buffer_index=i, data_format=dtype, page_size=size) for i in indices
                    ],
                )
            )

        def kernel(file, cores, config, ct, rt, defines=()):
            kernels.append(
                ttnn.KernelDescriptor(
                    kernel_source=str(source / file),
                    core_ranges=_grid(cores),
                    compile_time_args=ct,
                    runtime_args=rt,
                    defines=[*defines, *self.body.tuning.defines],
                    config=config,
                )
            )

        rt = ttnn.RuntimeArgs()
        coords = physical(self.projection_cores + self.rope_cores + self.cache_cores[1:])
        for rank, core in enumerate(self.projection_cores):
            rt[core.x][core.y] = [
                rank,
                self.normalizer.output.buffer_address(),
                self.body.address_table.buffer_address(),
                layer_index,
                *coords,
            ]
        ct = (
            accessor(self.normalizer.output)
            + accessor(self.body.layers[0].decode_weights["qkv"])
            + accessor(self.body.address_table)
        )
        for role, config in (
            ("READER", ttnn.ReaderConfigDescriptor()),
            ("WRITER", ttnn.WriterConfigDescriptor()),
            (
                "COMPUTE",
                ttnn.ComputeConfigDescriptor(
                    math_fidelity=ttnn.MathFidelity.LoFi, math_approx_mode=False, fp32_dest_acc_en=False
                ),
            ),
        ):
            kernel("qkv.cpp", self.projection_cores, config, ct, rt, [(role, "1")])
        for index, count, dtype in (
            (0, 16 * self.body.tuning.buffers, ttnn.bfloat16),
            (1, 96 * self.body.tuning.buffers, ttnn.bfloat8_b),
            (24, 6, ttnn.bfloat16),
            (31, 1, ttnn.uint32),
        ):
            cb([index], count, dtype, self.projection_cores)
        cbs.append(ttnn.cb_descriptor_from_sharded_tensor(16, self.packed))

        for role, core, head_tensor in zip(("QUERY", "KEY"), self.rope_cores, self.heads):
            rt = ttnn.RuntimeArgs()
            destination = self.rope_cores[0] if role == "QUERY" else self.cache_cores[0]
            rt[core.x][core.y] = [
                self.packed.buffer_address(),
                self.cos.buffer_address(),
                self.sin.buffer_address(),
                rotary_position.buffer_address(),
                *physical([destination]),
                *physical(self.body.attention_stage.cores),
            ]
            ct = accessor(self.packed) + accessor(self.cos) + accessor(self.sin) + accessor(rotary_position)
            for name, config in (("READER", ttnn.ReaderConfigDescriptor()), ("WRITER", ttnn.WriterConfigDescriptor())):
                kernel("rope_dataflow.cpp", [core], config, ct, rt, [(role, "1"), (name, "1")])
            kernel(
                "rope_compute.cpp",
                [core],
                ttnn.ComputeConfigDescriptor(
                    math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True
                ),
                [0, 1, 2, 3, 24, 25, 26, 16, 4, 1, 1, 1],
                rt,
            )
            for index, count in ((0, 4), (1, 4), (2, 4), (3, 1), (24, 4), (25, 4), (26, 4)):
                cb([index], count, ttnn.bfloat16, [core])
            cb([31], 1, ttnn.uint32, [core])
            cbs.append(ttnn.cb_descriptor_from_sharded_tensor(16, head_tensor))

        for role, core, input_tensor, cache in zip(
            ("KEY", "VALUE"), self.cache_cores, (self.heads[1], self.packed), (k_cache, v_cache)
        ):
            rt = ttnn.RuntimeArgs()
            rt[core.x][core.y] = [
                input_tensor.buffer_address(),
                cache.buffer_address(),
                position.buffer_address(),
                page_table.buffer_address(),
                *physical(self.rope_cores[:1]),
            ]
            ct = accessor(input_tensor) + accessor(cache) + accessor(position) + accessor(page_table)
            for name, config in (("READER", ttnn.ReaderConfigDescriptor()), ("WRITER", ttnn.WriterConfigDescriptor())):
                kernel(
                    "cache_dataflow.cpp",
                    [core],
                    config,
                    ct,
                    rt,
                    [(role, "1"), (name, "1"), ("PAGE_BYTES", str(page_table.padded_shape[-1] * 4))],
                )
            kernel("cache_compute.cpp", [core], ttnn.ComputeConfigDescriptor(fp32_dest_acc_en=False), [], rt)
            for indices, count, dtype in (
                ([0], 8, ttnn.bfloat8_b),
                ([1], 4, ttnn.bfloat16),
                ([8], 4, ttnn.bfloat8_b),
                ([16], 4, ttnn.bfloat16),
                ([24, 25], 4, ttnn.bfloat16),
                ([30], 1, ttnn.uint32),
                ([31], 1, ttnn.uint32),
            ):
                cb(indices, count, dtype, [core])
        program.kernels, program.cbs = kernels, cbs
        return program

    def tensors(self, cache_inputs):
        return [
            self.packed,
            *self.heads,
            self.cos,
            self.sin,
            *cache_inputs,
            *[layer.decode_weights["qkv"] for layer in self.body.layers],
        ]
