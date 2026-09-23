# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native final RMSNorm arithmetic and selected HiFi2/BFP8 vocabulary projection."""

from pathlib import Path
import ttnn
from .mlp import _grid
from .norm import FusedNorm
from .tuning import ProjectionTuning


class FusedHead:
    def __init__(self, model, *, cores=None, norm_cores=None, norm_output=None, tuning=None):
        self.tuning = tuning or ProjectionTuning()
        # The terminal vocabulary reader is independently capped by its L1
        # weight footprint. Layer projections may use deeper alias buffers.
        self.buffers = min(3, self.tuning.buffers)
        self.mesh = model.mesh_device
        if model.max_batch_size != 1 or model.padded_vocab_size != 131072:
            raise ValueError("Fused head requires the batch-one Llama3.1 vocabulary")
        policy = model.precision_policy
        if (
            policy["compute_fidelities"]["decode"]["lm_head"] != "HiFi2"
            or policy["accumulation"]["matmul_fp32"]
            or policy["accumulation"]["math_approx_mode"]
            or policy["logits_dtype"] != "bfloat16"
        ):
            raise ValueError("Fused head requires the original HiFi2/BF16 accumulation policy")
        if any(
            policy["lm_head_geometry"][key] != value
            for key, value in {"cores": 8, "block": 4, "readers": 2, "splits": 1}.items()
        ):
            raise ValueError("Fused head requires the selected native block4/two-reader geometry")
        model.lm_head.load_device_weights()
        if len(model.lm_head.output_weights) != 1:
            raise ValueError("Fused head requires one native weight split")
        self.weight = model.lm_head.output_weights[0]
        memory = self.weight.memory_config()
        if (
            tuple(self.weight.shape) != (4096, 32768)
            or self.weight.dtype != ttnn.bfloat8_b
            or memory.memory_layout != ttnn.TensorMemoryLayout.WIDTH_SHARDED
            or memory.buffer_type != ttnn.BufferType.DRAM
            or memory.shard_spec.shape != [4096, 4096]
        ):
            raise ValueError("Expected the original eight-bank BFP8 head weight")
        self.cores = list(cores) if cores is not None else [ttnn.CoreCoord(x, y) for y in (2, 3) for x in range(8)]
        if len(self.cores) != 16:
            raise ValueError("The native head partition requires sixteen workers")
        self.grid = _grid(self.cores)
        self.norm = FusedNorm(
            self.mesh, model.lm_head.config.input_memcfg, model.layers[0].eps, cores=norm_cores, output=norm_output,
            compact_output=self.tuning.compact_activations != "off", tile_height=self.tuning.norm_tile_height, full_dst=self.tuning.norm_full_dst
        )
        self.output = ttnn.empty(
            (1, 1, 1, 32768),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def append(self, program, gathered, *, wait_for_gather=False):
        if self.tuning.head_early_blocks and not hasattr(self, "weight_ready"):
            raise ValueError("Head own prefix requires DecoderLoop to bind its trigger")
        rt = ttnn.RuntimeArgs()
        physical = [self.mesh.worker_core_from_logical_core(c) for c in self.cores]
        coords = [v for c in physical for v in (c.x, c.y)]
        for index, core in enumerate(self.cores):
            rt[core.x][core.y] = [
                index,
                self.norm.output.buffer_address(),
                self.weight.buffer_address(),
                self.output.buffer_address(),
                *coords,
                *([ttnn.get_global_semaphore_address(self.weight_ready)] if self.tuning.head_early_blocks else []),
            ]
        ct = [
            v
            for t in (self.norm.output, self.weight, self.output)
            for v in ttnn.TensorAccessorArgs(t).get_compile_time_args()
        ]
        source = str(Path(__file__).with_name("kernels") / "head.cpp")
        kernels = []
        for role, config in (
            ("READER", ttnn.ReaderConfigDescriptor()),
            ("WRITER", ttnn.WriterConfigDescriptor()),
            (
                "COMPUTE",
                ttnn.ComputeConfigDescriptor(
                    math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=False,
                    dst_full_sync_en=self.tuning.projection_full_dst in ("head", "all")
                ),
            ),
        ):
            kernels.append(
                ttnn.KernelDescriptor(
                    kernel_source=source,
                    core_ranges=self.grid,
                    compile_time_args=ct,
                    runtime_args=rt,
                    defines=[(role, "1"), *[(name,
                        str(self.tuning.head_early_blocks) if name == "EARLY_WEIGHT_BLOCKS" and self.tuning.head_early_blocks else
                        str(self.buffers) if name == "PROJECTION_BUFFERS" else
                        str(min(self.buffers, self.tuning.lookahead)) if name == "PROJECTION_LOOKAHEAD" else value)
                        for name, value in self.tuning.defines]],
                    config=config,
                )
            )
        cbs = []
        for index, count, dtype in (
            (0, 4 * self.buffers, ttnn.bfloat16),
            (1, 256 * self.buffers, ttnn.bfloat8_b),
            (16, 64, ttnn.bfloat16),
            (24, 64, ttnn.bfloat16),
        ):
            page = ttnn.Tile([32, 32]).get_tile_size(dtype)
            cbs.append(
                ttnn.CBDescriptor(
                    total_size=count * page,
                    core_ranges=self.grid,
                    format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page)],
                )
            )
        if self.tuning.compact_activations != "off":
            cbs.append(ttnn.CBDescriptor(total_size=4096, core_ranges=self.grid,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=31, data_format=ttnn.uint32, page_size=4096)]))
        if self.tuning.alias_projection_cbs:
            # Final-block reload consumes each partial subblock before the
            # output pack overwrites that same subblock. Both rings are64 tiles.
            partial = next(item for item in cbs if item.format_descriptors[0].buffer_index == 24)
            output = next(item for item in cbs if item.format_descriptors[0].buffer_index == 16)
            partial.format_descriptors = [*partial.format_descriptors, *output.format_descriptors]
            cbs.remove(output)
        if self.tuning.projection_tile_height == 16:
            for item in cbs:
                formats = list(item.format_descriptors)
                for fmt in formats:
                    if fmt.buffer_index in (0, 16, 24):
                        fmt.tile = ttnn.TileDescriptor(16, 32)
                item.format_descriptors = formats
        program.kernels = [*program.kernels, *kernels]
        program.cbs = [*program.cbs, *cbs]
        program.semaphores = [
            *program.semaphores,
            *[
                ttnn.SemaphoreDescriptor(id=i, core_ranges=_grid(self.cores + self.norm.cores), initial_value=0)
                for i in range(13 if wait_for_gather else 10)
            ],
        ]
        return self.norm.append(program, gathered, self.cores, wait_for_gather=wait_for_gather)

    def tensors(self):
        return [self.norm.output, self.weight, self.output]

    def __call__(self, gathered):
        return ttnn.generic_op([gathered, *self.tensors()], self.append(ttnn.ProgramDescriptor(), gathered))
