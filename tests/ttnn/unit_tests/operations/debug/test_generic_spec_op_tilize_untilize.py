# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""tilize_block and pack_untilize_block as tt-blaze's KV-cache updates call them, through ttnn.experimental.generic_op.

blaze/ops/kv_cache_update (MLA) untilizes and re-tilizes one 32x32 tile at a time: pack_untilize_block<1, 1> and
tilize_block with block 1. blaze/ops/gqa_kv_cache_update does a row of DHt_per_core tiles at once,
pack_untilize_block<DHt, DHt> and tilize_block with block DHt; its callers use head_dim 64, 128 and 256 on one core per
head, so DHt is 2, 4 or 8. Both use full 32x32 tiles only, and neither uses pack_untilize_dest. Blaze's cache is
bfloat8_b, which Quasar does not have; this test is bf16, so every copy is exact and is checked element by element.

Each test runs NUM_ITERS row blocks of [32, 32 * DHt]:
- tilize: row-major in (32 rows per block) -> tilize_block -> tiled out.
- untilize: tiled in -> pack_untilize_block -> row-major out.
- round_trip, blaze's sequence fused with another op in one compute kernel, every iteration: sum = x + x
  (binary_tiles_init + pack_init), then pack_untilize x into the row-major DFB rm, then tilize rm into out.
  rm is a compute self-loop.

Inits on Quasar (tilize.h, pack_untilize.h): tilize_init re-points the packer at its output (llk_pack_init), and
pack_untilize_init programs unpack, math and pack. On Quasar both uninits are empty: pack_untilize_uninit compiles to
nothing, and tilize_uninit calls llk_unpack_tilize_uninit, an empty function. So what the next op needs is its own init
and pack_init, which the add re-runs every iteration. On Blackhole both uninits restore state, so the kernel keeps them.
NO_TILIZE_UNINIT (a mutant, not part of the test) drops tilize_uninit before the next iteration's add; on Quasar
craq-sim, with the watcher and LLK asserts on, out and sum stay exact at 1 and 8 tiles.
"""

import pytest
import torch
import ttnn

E = ttnn.experimental

NUM_ITERS = 3
TILE_BYTES = 32 * 32 * 2

DM_SOURCE = """
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

// A block is d tiles: d tile pages of a tiled tensor, or 32 row pages of a row-major one.
template <bool row_major, typename Q, typename T>
void load(Noc& noc, Q& q, const T& tensor, uint32_t d, uint32_t block) {
    const uint32_t row_bytes = d * 32 * 2;
    q.reserve_back(d);
    if constexpr (row_major) {
        for (uint32_t r = 0; r < 32; ++r) {
            noc.async_read(tensor, q, row_bytes, {.page_id = block * 32 + r}, {.offset_bytes = r * row_bytes});
        }
    } else {
        for (uint32_t t = 0; t < d; ++t) {
            noc.async_read(
                tensor, q, q.get_entry_size(), {.page_id = block * d + t}, {.offset_bytes = t * q.get_entry_size()});
        }
    }
    noc.async_read_barrier();
    q.push_back(d);
}

template <bool row_major, typename Q, typename T>
void store(Noc& noc, Q& q, const T& tensor, uint32_t d, uint32_t block) {
    const uint32_t row_bytes = d * 32 * 2;
    q.wait_front(d);
    if constexpr (row_major) {
        for (uint32_t r = 0; r < 32; ++r) {
            noc.async_write(q, tensor, row_bytes, {.offset_bytes = r * row_bytes}, {.page_id = block * 32 + r});
        }
    } else {
        for (uint32_t t = 0; t < d; ++t) {
            noc.async_write(
                q, tensor, q.get_entry_size(), {.offset_bytes = t * q.get_entry_size()}, {.page_id = block * d + t});
        }
    }
    noc.async_write_barrier();
    q.pop_front(d);
}

void kernel_main() {
    constexpr uint32_t d = get_arg(args::d);
    constexpr uint32_t num_iters = get_arg(args::num_iters);
    constexpr bool in_row_major = get_arg(args::in_row_major);
    constexpr bool out_row_major = get_arg(args::out_row_major);
    Noc noc;
    DataflowBuffer in(dfb::in);
    DataflowBuffer out(dfb::out);
    const auto in_tensor = TensorAccessor(tensor::in);
    const auto out_tensor = TensorAccessor(tensor::out);
#ifdef ROUND_TRIP
    DataflowBuffer sum(dfb::sum);
    const auto sum_tensor = TensorAccessor(tensor::sum);
#endif
    for (uint32_t iter = 0; iter < num_iters; ++iter) {
        load<in_row_major>(noc, in, in_tensor, d, iter);
#ifdef ROUND_TRIP
        store<false>(noc, sum, sum_tensor, d, iter);
#endif
        store<out_row_major>(noc, out, out_tensor, d, iter);
    }
}
"""

COMPUTE_SOURCE = """
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/pack.h"
#include "api/compute/pack_untilize.h"
#include "api/compute/tilize.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t d = get_arg(args::d);
    constexpr uint32_t num_iters = get_arg(args::num_iters);
    DataflowBuffer in(dfb::in);
    DataflowBuffer out(dfb::out);
#if defined(MODE_TILIZE)
    compute_kernel_hw_startup(dfb::in, dfb::in, dfb::out);
    tilize_init(dfb::in, d, dfb::out);
    for (uint32_t iter = 0; iter < num_iters; ++iter) {
        in.wait_front(d);
        out.reserve_back(d);
        tilize_block(dfb::in, d, dfb::out);
        out.push_back(d);
        in.pop_front(d);
    }
    tilize_uninit(dfb::in, dfb::out);
#elif defined(MODE_UNTILIZE)
    compute_kernel_hw_startup(dfb::in, dfb::in, dfb::out);
    pack_untilize_init<d, d>(dfb::in, dfb::out);
    for (uint32_t iter = 0; iter < num_iters; ++iter) {
        in.wait_front(d);
        out.reserve_back(d);
        pack_untilize_block<d, d>(dfb::in, 1, dfb::out, 0);
        out.push_back(d);
        in.pop_front(d);
    }
    pack_untilize_uninit(dfb::out);
#else  // ROUND_TRIP
    DataflowBuffer rm(dfb::rm);
    DataflowBuffer sum(dfb::sum);
    compute_kernel_hw_startup(dfb::in, dfb::in, dfb::sum);
    for (uint32_t iter = 0; iter < num_iters; ++iter) {
        in.wait_front(d);

        // Another op: sum = in + in.
        binary_tiles_init<true, EltwiseBinaryType::ELWADD>(dfb::in, dfb::in);
        pack_init(dfb::sum);
        tile_regs_acquire();
        for (uint32_t t = 0; t < d; ++t) {
            add_tiles(dfb::in, dfb::in, t, t, t);
        }
        tile_regs_commit();
        sum.reserve_back(d);
        tile_regs_wait();
        for (uint32_t t = 0; t < d; ++t) {
            pack_tile(t, dfb::sum);
        }
        tile_regs_release();
        sum.push_back(d);

        // Untilize in -> rm, as blaze's kv-cache update does.
        pack_untilize_init<d, d>(dfb::in, dfb::rm);
        rm.reserve_back(d);
        pack_untilize_block<d, d>(dfb::in, 1, dfb::rm, 0);
        rm.push_back(d);
        pack_untilize_uninit(dfb::rm);
        in.pop_front(d);

        // Tilize rm -> out.
        tilize_init(dfb::rm, d, dfb::out);
        rm.wait_front(d);
        out.reserve_back(d);
        tilize_block(dfb::rm, d, dfb::out);
        out.push_back(d);
        rm.pop_front(d);
#ifndef NO_TILIZE_UNINIT
        tilize_uninit(dfb::rm, dfb::out);
#endif
    }
#endif
}
"""

MODES = {
    # mode: (input row-major, output row-major)
    "tilize": (True, False),
    "untilize": (False, True),
    "round_trip": (False, False),
}


def run(device, mode, d, defines=None):
    """Returns (x, {output name: torch result}) for NUM_ITERS blocks of [32, 32 * d] bf16."""
    in_rm, out_rm = MODES[mode]
    torch.manual_seed(d)
    # Distinct bf16 values: a misplaced element cannot match by accident.
    x = torch.randn(NUM_ITERS * 32, 32 * d).bfloat16()

    def dram(t, row_major):
        return ttnn.from_torch(
            t, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT if row_major else ttnn.TILE_LAYOUT, device=device
        )

    zeros = torch.zeros_like(x)
    tensors = {"in": dram(x, in_rm), "out": dram(zeros, out_rm)}
    if mode == "round_trip":
        tensors["sum"] = dram(zeros, False)

    def dfb(name):
        return E.DataflowBufferSpec(
            unique_id=name, entry_size=TILE_BYTES, num_entries=d, data_format_metadata=ttnn.bfloat16
        )

    dfb_names = list(tensors) + (["rm"] if mode == "round_trip" else [])
    compute_dfbs = [E.consumer_of("in", "in"), E.producer_of("out", "out")]
    dm_dfbs = [E.producer_of("in", "in"), E.consumer_of("out", "out")]
    if mode == "round_trip":
        compute_dfbs += [E.producer_of("sum", "sum"), E.producer_of("rm", "rm"), E.consumer_of("rm", "rm")]
        dm_dfbs.append(E.consumer_of("sum", "sum"))
    common = {"d": d, "num_iters": NUM_ITERS}
    mode_defines = {"tilize": "MODE_TILIZE", "untilize": "MODE_UNTILIZE", "round_trip": "ROUND_TRIP"}
    spec = E.ProgramSpec(
        name=f"tilize_untilize_{mode}_{d}",
        kernels=[
            E.KernelSpec(
                unique_id="dm",
                source=E.KernelSpec.SourceCode(code=DM_SOURCE),
                hw_config=E.create_reader_datamovement_config(disable_dfb_implicit_sync_for_all=True),
                compiler_options=E.KernelSpec.CompilerOptions(defines={mode_defines[mode]: "1"}),
                dfb_bindings=dm_dfbs,
                tensor_bindings=[E.TensorBinding(tensor_parameter_name=t, accessor_name=t) for t in tensors],
                compile_time_args={**common, "in_row_major": int(in_rm), "out_row_major": int(out_rm)},
            ),
            E.KernelSpec(
                unique_id="compute",
                source=E.KernelSpec.SourceCode(code=COMPUTE_SOURCE),
                hw_config=E.ComputeHardwareConfig(),
                compiler_options=E.KernelSpec.CompilerOptions(defines={mode_defines[mode]: "1", **(defines or {})}),
                dfb_bindings=compute_dfbs,
                compile_time_args=common,
            ),
        ],
        dataflow_buffers=[dfb(n) for n in dfb_names],
        tensor_parameters=[E.TensorParameter(unique_id=n, spec=t.spec) for n, t in tensors.items()],
        work_units=[E.WorkUnitSpec(kernels=["dm", "compute"], target_nodes=ttnn.CoreCoord(0, 0))],
    )
    E.generic_op(list(tensors.values()), spec, E.ProgramRunArgs(tensor_args=tensors))
    return x, {n: ttnn.to_torch(t) for n, t in tensors.items() if n != "in"}


def assert_exact(expected, actual, what):
    wrong = (expected != actual).nonzero()
    assert wrong.numel() == 0, (
        f"{what}: {len(wrong)} of {expected.numel()} elements wrong; first at {wrong[0].tolist()}: "
        f"expected {expected[tuple(wrong[0])].item()}, got {actual[tuple(wrong[0])].item()}"
    )


@pytest.mark.parametrize("d", [1, 2, 4, 8], ids=lambda d: f"{d}tiles")
@pytest.mark.parametrize("mode", list(MODES))
def test_generic_spec_op_tilize_untilize(device, mode, d):
    x, outputs = run(device, mode, d)
    assert_exact(x, outputs["out"], f"{mode} out")
    if mode == "round_trip":
        assert_exact(x * 2, outputs["sum"], "round_trip sum")
