# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""DM CPU stores across a multi-entry DataflowBuffer::scoped_write_lock(n), through generic_op.

A DM producer fills a DFB with CPU stores through the lock's get_ptr(), and a DM consumer writes the entries to DRAM
over the NoC.
- "lock_n": one scoped_write_lock(4) per iteration, with stores across all 4 entries.
- "one_at_a_time": 4 locks of one entry each.

The DFB has 8 entries of 2 KiB with one producer and one consumer, so its stride is 1 and its entries are contiguous.
Over 4 iterations the ring wraps only between locks.

Background:
- get_ptr() addresses only the first locked entry (DfbScopedLock::get_ptr, api/dataflow/dataflow_buffer.h).
- lock_acquire_impl / lock_release_impl (internal/tt-2xx/dataflow_buffer.inl) walk entry k at
  start + k * entry_size * stride and wrap at the ring limit. So linear stores from get_ptr() stay inside the lock
  only when the stride is 1 and the lock doesn't straddle the ring end.
- Nothing checks CPU stores against the lock extent.

Observed: both modes are exact on Quasar craq-sim (with and without the watcher) and on BH ttsim. A suspected NoC wedge
from lock(4) plus CPU stores did not reproduce here; the blaze hang it was blamed for was watcher clock starvation on
craq-sim.
"""

import pytest
import torch
import ttnn

E = ttnn.experimental

CORE = ttnn.CoreCoord(0, 0)
ENTRY_WORDS = 512  # 2 KiB entries
NUM_ENTRIES = 8
ITERS = 4
LOCK_N = 4

PRODUCER_SOURCE = """
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t iters = get_arg(args::iters);
    constexpr uint32_t n = get_arg(args::lock_n);
    constexpr uint32_t words = get_arg(args::entry_words);
    DataflowBuffer d(dfb::d);
    for (uint32_t it = 0; it < iters; ++it) {
        d.reserve_back(n);
#ifdef ONE_AT_A_TIME
        for (uint32_t e = 0; e < n; ++e) {
            auto lock = d.scoped_write_lock(1);
            auto p = lock.get_ptr<uint32_t>();
            for (uint32_t w = 0; w < words; ++w) {
                p[w] = (it * n + e) * words + w;
            }
            d.push_back(1);
        }
#else
        {
            auto lock = d.scoped_write_lock(n);
            auto p = lock.get_ptr<uint32_t>();
            for (uint32_t w = 0; w < n * words; ++w) {
                p[w] = it * n * words + w;
            }
        }
        d.push_back(n);
#endif
    }
}
"""

CONSUMER_SOURCE = """
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t iters = get_arg(args::iters);
    constexpr uint32_t n = get_arg(args::lock_n);
    Noc noc;
    DataflowBuffer d(dfb::d);
    const auto out = TensorAccessor(tensor::out);
    for (uint32_t it = 0; it < iters; ++it) {
        d.wait_front(n);
        for (uint32_t e = 0; e < n; ++e) {
            noc.async_write(d, out, d.get_entry_size(), {.offset_bytes = e * d.get_entry_size()}, {.page_id = it * n + e});
        }
        noc.async_write_barrier();
        d.pop_front(n);
    }
}
"""


@pytest.mark.parametrize("mode", ["one_at_a_time", "lock_n"])
def test_generic_spec_op_dfb_multi_entry_lock(device, mode):
    rows = ITERS * LOCK_N
    out = ttnn.from_torch(
        torch.zeros(rows, ENTRY_WORDS, dtype=torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )
    ct_args = {"iters": ITERS, "lock_n": LOCK_N, "entry_words": ENTRY_WORDS}
    spec = E.ProgramSpec(
        name="dfb_multi_entry_lock",
        kernels=[
            E.KernelSpec(
                unique_id="producer",
                source=E.KernelSpec.SourceCode(code=PRODUCER_SOURCE),
                hw_config=E.create_reader_datamovement_config(disable_dfb_implicit_sync_for_all=True),
                compiler_options=E.KernelSpec.CompilerOptions(
                    defines={"ONE_AT_A_TIME": "1"} if mode == "one_at_a_time" else {}
                ),
                dfb_bindings=[E.producer_of("d", "d")],
                compile_time_args=ct_args,
            ),
            E.KernelSpec(
                unique_id="consumer",
                source=E.KernelSpec.SourceCode(code=CONSUMER_SOURCE),
                hw_config=E.create_writer_datamovement_config(disable_dfb_implicit_sync_for_all=True),
                dfb_bindings=[E.consumer_of("d", "d")],
                tensor_bindings=[E.TensorBinding(tensor_parameter_name="out", accessor_name="out")],
                compile_time_args=ct_args,
            ),
        ],
        dataflow_buffers=[E.DataflowBufferSpec(unique_id="d", entry_size=ENTRY_WORDS * 4, num_entries=NUM_ENTRIES)],
        tensor_parameters=[E.TensorParameter(unique_id="out", spec=out.spec)],
        work_units=[E.WorkUnitSpec(kernels=["producer", "consumer"], target_nodes=CORE)],
    )
    E.generic_op([out], spec, E.ProgramRunArgs(tensor_args={"out": out}))
    result = ttnn.to_torch(out).to(torch.int64).flatten()
    assert torch.equal(result, torch.arange(rows * ENTRY_WORDS, dtype=torch.int64))
