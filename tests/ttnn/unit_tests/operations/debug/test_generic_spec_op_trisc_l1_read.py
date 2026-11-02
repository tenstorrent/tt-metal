# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Does a compute (TRISC) kernel's plain load of L1 see a value a DM core stored? A check through generic_op.

On Quasar a DM core's CPU loads go through write-back caches that don't see NoC writes (porting guide rule 12). Is
TRISC the same? invalidate_l1_cache() for Quasar non-DM cores is a no-op "no cache management needed"
(internal/tt-2xx/risc_common.h), and DFB init reads "used by TRISC (no private L2 cache)"
(dataflow_buffer/dataflow_buffer_init.h). But TRISC firmware still reads its runtime args through
MEM_L1_UNCACHED_BASE, with a TODO tied to PR #38124 (firmware/src/tt-2xx/trisc.cc).

The DM kernel spins for a while, then stores 0xC0FFEE into word 0 of an L1 tensor under scoped_lock, which flushes it.
On the UNPACK thread, the compute kernel reads the word once (expected 0, so the line is primed), then polls it with
either a plain load through LocalTensorAccessor operator[] or the uncached alias (Quasar; the same pointer elsewhere).
It records what it saw in words 1..5 through the uncached alias.

Observed on Quasar craq-sim, watcher + LLK asserts: plain and uncached polls both saw the store, after a primed 0, at
the same poll count. So TRISC has no stale L1 data cache on craq-sim. Silicon may differ (see the firmware TODO above).
"""

import pytest
import torch
import ttnn
from tt_py_test_utils_common.utility_functions import is_quasar

E = ttnn.experimental

CORE = ttnn.CoreCoord(0, 0)
WORDS = 32
FLAG = 0xC0FFEE
MAX_SPINS = 20_000

DM_SOURCE = """
#include "api/dataflow/dataflow_api.h"
#include "api/tensor/local_tensor_accessor.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t delay = get_arg(args::delay);
    const LocalTensorAccessor<uint32_t> w(tensor::w);
    volatile uint32_t spin_sink = 0;
    for (uint32_t i = 0; i < delay; ++i) {
        spin_sink = spin_sink + 1;
    }
    {
        auto lock = w.scoped_lock(0, 1);  // flushes the store to L1 on release
        w[0] = 0xC0FFEEu;
    }
}
"""

COMPUTE_SOURCE = """
#include "api/compute/common.h"
#include "api/tensor/local_tensor_accessor.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    UNPACK(({
        constexpr uint32_t max_spins = get_arg(args::max_spins);
        const LocalTensorAccessor<uint32_t> w(tensor::w);
        volatile uint32_t* plain = &w[0];
#ifdef ARCH_QUASAR
        volatile uint32_t* uncached =
            reinterpret_cast<volatile uint32_t*>(reinterpret_cast<uintptr_t>(plain) + MEM_L1_UNCACHED_BASE);
#else
        volatile uint32_t* uncached = plain;
#endif
        const uint32_t primed = *POLL_PTR;
        uint32_t spins = 0, seen = 0;
        for (; spins < max_spins; ++spins) {
            if (*POLL_PTR == 0xC0FFEEu) {
                seen = 1;
                break;
            }
        }
        const uint32_t uncached_now = *uncached;
        const uint32_t plain_now = *plain;
        uncached[1] = seen;
        uncached[2] = spins;
        uncached[3] = primed;
        uncached[4] = uncached_now;
        uncached[5] = plain_now;
    }));
}
"""


@pytest.mark.parametrize("poll", ["plain", "uncached"])
def test_generic_spec_op_trisc_l1_read(device, poll):
    cfg = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(ttnn.CoreRangeSet({ttnn.CoreRange(CORE, CORE)}), [1, WORDS], ttnn.ShardOrientation.ROW_MAJOR),
    )
    w = ttnn.from_torch(
        torch.zeros(1, WORDS, dtype=torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=cfg,
    )
    binding = [E.TensorBinding(tensor_parameter_name="w", accessor_name="w")]
    spec = E.ProgramSpec(
        name="trisc_l1_read",
        kernels=[
            E.KernelSpec(
                unique_id="dm",
                source=E.KernelSpec.SourceCode(code=DM_SOURCE),
                hw_config=E.create_reader_datamovement_config(disable_dfb_implicit_sync_for_all=True),
                tensor_bindings=binding,
                compile_time_args={"delay": 2000},
            ),
            E.KernelSpec(
                unique_id="compute",
                source=E.KernelSpec.SourceCode(code=COMPUTE_SOURCE.replace("POLL_PTR", poll)),
                hw_config=E.ComputeHardwareConfig(),
                tensor_bindings=binding,
                compile_time_args={"max_spins": MAX_SPINS},
            ),
        ],
        tensor_parameters=[E.TensorParameter(unique_id="w", spec=w.spec)],
        work_units=[E.WorkUnitSpec(kernels=["dm", "compute"], target_nodes=CORE)],
    )
    E.generic_op([w], spec, E.ProgramRunArgs(tensor_args={"w": w}))
    word0, seen, spins, primed, uncached_now, plain_now = ttnn.to_torch(w).to(torch.int64).flatten().tolist()[:6]
    print(f"TRISC_L1_READ poll={poll} seen={seen} spins={spins} primed={primed:#x} plain_now={plain_now:#x}")
    assert word0 == FLAG
    assert seen == 1 and spins < MAX_SPINS
    assert uncached_now == FLAG and plain_now == FLAG
    if is_quasar():
        # The read before the store primed any cache line; a stale TRISC cache would keep returning 0.
        assert primed == 0
