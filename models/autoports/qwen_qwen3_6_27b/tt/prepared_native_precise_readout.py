# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Opt-in T1 prepared native recurrence with only its readout made SFPU FP32.

No TTNN import until constructing/running the owner. Native preparation,
prediction, delta, update and persistence arithmetic are reused unchanged.
This is a diagnostic hypothesis, not a validated quality/performance result.
"""

from .prepared_native_recurrent_candidate import (
    PreparedNativeRecurrentCandidate,
    PREP_READER,
    PREP_WRITER,
    PREP_COMPUTE,
    REC_READER,
    REC_WRITER,
    head_partition,
)

READOUT_CALL = "q.wait_front(4); mm(cb_q,cb_s2,cb_out,1,4,4,false); q.pop_front(4);"
PRECISE_CALL = "q.wait_front(4); precise_readout(cb_s2,cb_q,cb_out); q.pop_front(4);"
ROW_MAPPING = "dst[(i/32)*1024+((i%32)/16)*256+i%16]=src[i];"
COLUMN_MAPPING = "dst[(i/32)*1024+(id==0?((i%32)/16)*512+(i%16)*16:((i%32)/16)*256+i%16)]=src[i];"

READOUT = r"""
#include "api/compute/compute_kernel_api.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/reduce.h"
// CB0 is now a prepared Q column; CB8 is live FP32 updated state. Both are
// direct-to-Dest, never FPU matmul operands. Three of four half-sync DST slots.
void precise_readout(uint32_t matrix,uint32_t column,uint32_t result) {
    CircularBuffer out(result); out.reserve_back(4);
    PACK((llk_pack_reduce_mask_config<ReduceDim::REDUCE_COL,PackMode::Default>(result)));
    for(uint32_t n=0;n<4;n++) {
        tile_regs_acquire();
        for(uint32_t m=0;m<4;m++) {
            const uint32_t work=m==0 ? 0 : 2;
            reconfig_data_format_srca(matrix); copy_init(matrix);
            copy_tile(matrix,m*4+n,work);
            reconfig_data_format_srca(column); copy_init(column);
            copy_tile(column,m,1);
            sfpu_mul_bcast_col_init(); sfpu_mul_bcast_col(work,1);
            if(m) { add_binary_tile_init(); add_binary_tile(0,2,0); }
        }
        sfpu_reduce_init<PoolType::SUM,DataFormat::Float32>();
        sfpu_reduce<PoolType::SUM,DataFormat::Float32,ReduceDim::REDUCE_COL>(0,1,1);
        tile_regs_commit(); tile_regs_wait(); pack_tile<true>(0,result,n); tile_regs_release();
    }
    PACK((llk_pack_reduce_mask_clear())); out.push_back(4);
}

"""


def _once(source, before, after):
    if source.count(before) != 1:
        raise ValueError(f"Prepared readout source drift: {before[:100]!r}")
    return source.replace(before, after, 1)


def precise_readout_sources(reader, compute):
    """Pure source adaptation of prepared REC_READER and recurrence_source().

    Caller must set recurrence direct unpack to CB0+CB8 for both state dtypes.
    No tensor allocation or source-file/AST access occurs in this transform.
    """
    reader = _once(reader, ROW_MAPPING, COLUMN_MAPPING)
    compute = _once(compute, "void kernel_main() {", READOUT + "void kernel_main() {")
    compute = _once(compute, READOUT_CALL, PRECISE_CALL)
    compute = _once(
        compute, "compute_kernel_hw_startup(cb_q,cb_v,cb_out);", "compute_kernel_hw_startup(cb_S,cb_decay,cb_out);"
    )
    compute = _once(
        compute,
        "for(uint32_t head=0;head<count;head++) {",
        "for(uint32_t head=0;head<count;head++) {\n"
        "        // Restore both ordinary FPU source formats before every head.\n"
        "        reconfig_data_format(cb_S,cb_decay);",
    )
    compute = _once(compute, "reconfig_data_format_srca(cb_q);", "reconfig_data_format_srca(cb_S);")
    compute = _once(
        compute, "            copy_init(cb_s2);", "            reconfig_data_format_srca(cb_s2); copy_init(cb_s2);"
    )
    return reader, compute


class PreparedNativePreciseReadout(PreparedNativeRecurrentCandidate):
    """Independent retained buffers/programs; no monkeypatch of shared helpers."""

    def __init__(self, initial_state):
        super().__init__(initial_state)
        self.reader, self.source = precise_readout_sources(REC_READER, self.source)

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
                (self.source, compute_config((0, 8)), [int(self.bf16)], [[n] for _, n in partition]),
            ],
            cb_specs,
            [self.prepared, self.state, self.output],
        )
        return self.output
