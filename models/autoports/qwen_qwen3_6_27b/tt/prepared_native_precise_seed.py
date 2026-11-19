# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Isolated seed-preservation discriminator; no production registration.

Native FPU decay/prediction/delta/outer product remain unchanged. Only the
rank-one DST seed copy reads the unchanged decayed FP32 tile directly.
"""

from .prepared_native_precise_readout import PreparedNativePreciseReadout
from .prepared_native_recurrent_candidate import (
    PREP_READER,
    PREP_WRITER,
    PREP_COMPUTE,
    REC_WRITER,
    head_partition,
)


def seed_edits():
    return (
        (
            "    cb_o.reserve_back(n);\n    mul_bcast_scalar_init(a, scal);",
            "    cb_o.reserve_back(n);\n    CircularBuffer seed(17); seed.reserve_back(n);\n    mul_bcast_scalar_init(a, scal);",
        ),
        (
            "        pack_tile(0, o, i);\n        tile_regs_release();\n    }\n    cb_o.push_back(n);\n}\n\n// out[Kt,1]",
            "        pack_tile(0, o, i);\n        pack_tile(0, 17, i);\n        tile_regs_release();\n    }\n    cb_o.push_back(n); seed.push_back(n);\n}\n\n// out[Kt,1]",
        ),
        (
            "            copy_init(sd);\n            for (uint32_t j = 0; j < nn; j++) {\n                copy_tile(sd, mi * Vt + n0 + j, j);\n            }\n            matmul_init(kcol, u, 0);",
            "            reconfig_data_format_srca(17); copy_init(17);\n            for (uint32_t j = 0; j < nn; j++) {\n                copy_tile(17, mi * Vt + n0 + j, j);\n            }\n            reconfig_data_format(kcol, u); matmul_init(kcol, u, 0);",
        ),
        (
            "        rank1_update<4,4>(cb_sd,cb_kcol,cb_u,cb_s2,!BF16);",
            "        CircularBuffer seed(17); seed.wait_front(16);\n        rank1_update<4,4>(cb_sd,cb_kcol,cb_u,cb_s2,!BF16); seed.pop_front(16);",
        ),
    )


def seed_source(source):
    # Helper definitions precede the precise-readout includes, so this include
    # must be at the start, before rank1_update calls reconfig_data_format.
    result = '#include "api/compute/reconfig_data_format.h"\n' + source
    for before, after in seed_edits():
        if result.count(before) != 1:
            raise ValueError(f"Seed source anchor drift: {before[:100]!r}")
        result = result.replace(before, after, 1)
    return result


class PreparedNativePreciseSeed(PreparedNativePreciseReadout):
    def __init__(self, initial_state):
        super().__init__(initial_state)
        self.source = seed_source(self.source)

    def step(self, packed, g, beta):
        import ttnn

        assert tuple(packed.shape) == tuple(packed.padded_shape) == (self.batch, 1, 2560)
        assert packed.dtype == ttnn.bfloat16 and packed.layout == ttnn.ROW_MAJOR_LAYOUT
        for gate in (g, beta):
            assert tuple(gate.shape) == (self.batch, 1, 12)
            assert tuple(gate.padded_shape) == (self.batch, 32, 32)
            assert gate.dtype in (ttnn.float32, ttnn.bfloat16) and gate.layout == ttnn.TILE_LAYOUT
        for tensor in (packed, g, beta, self.state, self.prepared, self.output):
            assert tensor.device() == self.mesh and tensor.memory_config() == ttnn.DRAM_MEMORY_CONFIG
        assert tuple(t.buffer_address() for t in (self.state, self.prepared, self.output)) == self.addresses
        grid = self.mesh.compute_with_storage_grid_size()
        assert grid.x * grid.y >= self.batch

        def launch(coords, definitions, cb_specs, tensors):
            ranges = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in coords])
            kernels = []
            for source, config, ct, per_core in definitions:
                runtime = ttnn.RuntimeArgs()
                for c, values in zip(coords, per_core):
                    runtime[c.x][c.y] = values
                kernels.append(
                    ttnn.KernelDescriptor(
                        kernel_source=source,
                        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                        core_ranges=ranges,
                        compile_time_args=ct,
                        runtime_args=runtime,
                        config=config,
                    )
                )
            cbs = [
                ttnn.CBDescriptor(
                    total_size=n * size,
                    core_ranges=ranges,
                    format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=dtype, page_size=size)],
                )
                for i, n, size, dtype in cb_specs
            ]
            ttnn.generic_op(tensors, ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))

        def accessors(tensors):
            result = []
            for tensor in tensors:
                result.extend(ttnn.TensorAccessorArgs(tensor).get_compile_time_args())
            return result

        def compute_config(direct=()):
            cfg = ttnn.ComputeConfigDescriptor(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                fp32_dest_acc_en=True,
                dst_full_sync_en=False,
                math_approx_mode=False,
            )
            modes = [ttnn.UnpackToDestMode.Default] * 64
            for index in direct:
                modes[index] = ttnn.UnpackToDestMode.UnpackToDestFp32
            cfg.unpack_to_dest_mode = modes
            return cfg

        coords = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(self.batch)]
        gf, bf = int(g.dtype == ttnn.float32), int(beta.dtype == ttnn.float32)
        launch(
            coords,
            [
                (
                    PREP_READER,
                    ttnn.ReaderConfigDescriptor(),
                    [gf, bf] + accessors([packed, g, beta]),
                    [
                        [packed.buffer_address(), g.buffer_address(), beta.buffer_address(), u]
                        for u in range(self.batch)
                    ],
                ),
                (
                    PREP_WRITER,
                    ttnn.WriterConfigDescriptor(),
                    [bf] + accessors([self.prepared]),
                    [[self.prepared.buffer_address(), u] for u in range(self.batch)],
                ),
                (PREP_COMPUTE, compute_config((0, 1, 4)), [], [[] for _ in coords]),
            ],
            [(i, n, 4096, ttnn.float32) for i, n in {0: 4, 1: 1, 2: 4, 3: 1, 4: 1, 15: 2, 16: 3}.items()],
            [packed, g, beta, self.prepared],
        )

        partition = head_partition(self.batch, grid.x * grid.y)
        coords = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(len(partition))]
        sb = 2048 if self.bf16 else 4096
        sizes = {
            0: 4,
            1: 4,
            2: 4,
            3: 1,
            4: 1,
            5: 16,
            6: 8,
            7: 32,
            8: 16,
            9: 16,
            10: 16,
            11: 4,
            12: 4,
            13: 4,
            15: 1,
            16: 1,
            17: 16,
        }
        cb_specs = [
            (i, n, 2048 if self.bf16 and i == 7 else 4096, ttnn.bfloat16 if self.bf16 and i == 7 else ttnn.float32)
            for i, n in sizes.items()
        ]
        if self.bf16:
            cb_specs.append((14, 16, 2048, ttnn.bfloat16))
        launch(
            coords,
            [
                (
                    self.reader,
                    ttnn.ReaderConfigDescriptor(),
                    [sb] + accessors([self.prepared, self.state]),
                    [[self.prepared.buffer_address(), self.state.buffer_address(), s, n] for s, n in partition],
                ),
                (
                    REC_WRITER,
                    ttnn.WriterConfigDescriptor(),
                    [sb, self.batch * 12] + accessors([self.state, self.output]),
                    [[self.state.buffer_address(), self.output.buffer_address(), s, n] for s, n in partition],
                ),
                (self.source, compute_config((0, 8, 17)), [int(self.bf16)], [[n] for _, n in partition]),
            ],
            cb_specs,
            [self.prepared, self.state, self.output],
        )
        return self.output
