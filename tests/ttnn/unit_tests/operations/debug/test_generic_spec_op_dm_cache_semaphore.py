# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Does a raw semaphore pointer see a NoC increment on a data movement core? A bounded check through generic_op.

Quasar's data movement RISC-Vs reach L1 through write-back caches that do not see NoC writes. Legacy kernels poll a
semaphore with noc_semaphore_wait on the get_semaphore() pointer, and invalidate_l1_cache() inside that loop is empty on
Quasar. The Semaphore class instead reads the word through the uncached view of L1.

Core A reads the "flag" word once through the raw get_semaphore() pointer, which brings it into its cache. A then gives
core B a "go" credit, and B increments flag on A over the NoC. A polls flag for up to MAX_SPINS reads, either raw (the
noc_semaphore_wait loop body) or with Semaphore::value(). After that A waits on flag with the class, which always
returns because the increment is in L1, and reads the raw pointer again. A stores what it saw in a result tensor.

Expected: the class always sees the increment. The raw pointer sees it on Blackhole and does not see it on Quasar, even
after the class has observed the new value. Observed (seen, reads, raw primed, raw after the class wait):
- Quasar craq-sim, watcher + LLK asserts: class (1, 31, 0, 0); raw (0, 20000, 0, 0).
- Blackhole ttsim: class and raw both (1, 5, 0, 1).
"""

import pytest
import torch
import ttnn
from tt_py_test_utils_common.utility_functions import is_quasar

E = ttnn.experimental

CORE_A = ttnn.CoreCoord(0, 0)
CORE_B = ttnn.CoreCoord(1, 0)
# The class sees the increment within ~30 reads; 20k raw reads take ~6 min on craq-sim with the watcher.
MAX_SPINS = 20_000
RESULT_WORDS = 32
# Result words written by core A.
SEEN, SPINS, RAW_PRIMED, RAW_AFTER_CLASS_WAIT = range(4)

KERNEL_INCLUDES = """
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/tensor/local_tensor_accessor.h"
#include "experimental/kernel_args.h"
"""

POLL_SOURCE = (
    KERNEL_INCLUDES
    + """
void kernel_main() {
    constexpr uint32_t max_spins = get_arg(args::max_spins);
    const uint32_t b_x = get_arg(args::b_x);
    const uint32_t b_y = get_arg(args::b_y);
    Noc noc;
    Semaphore<> flag(sem::flag);
    Semaphore<> go(sem::go);
    // The legacy view of the same word: what noc_semaphore_wait(get_semaphore(id), v) polls.
    volatile tt_l1_ptr uint32_t* raw = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem::flag.id));
    const uint32_t raw_primed = *raw;  // Brings the word's line into this core's cache.
    go.up(noc, b_x, b_y, 1);
    uint32_t seen = 0;
    uint32_t spins = 0;
    for (; spins < max_spins; ++spins) {
#ifdef POLL_RAW
        invalidate_l1_cache();
        const uint32_t v = *raw;
#else
        const uint32_t v = flag.value();
#endif
        if (v >= 1) {
            seen = 1;
            break;
        }
    }
    flag.wait(1);  // Always returns: B's increment is in L1.
    const uint32_t raw_after = *raw;
    flag.down(1);
    const LocalTensorAccessor<uint32_t> result(tensor::result);
    {
        auto lock = result.scoped_lock(0, 4);
        result[0] = seen;
        result[1] = spins;
        result[2] = raw_primed;
        result[3] = raw_after;
    }
}
"""
)

BUMP_SOURCE = (
    KERNEL_INCLUDES
    + """
void kernel_main() {
    const uint32_t a_x = get_arg(args::a_x);
    const uint32_t a_y = get_arg(args::a_y);
    Noc noc;
    Semaphore<> flag(sem::flag);
    Semaphore<> go(sem::go);
    go.down(1);  // A has primed its cache.
    flag.up(noc, a_x, a_y, 1);
}
"""
)


def run(device, poll_raw):
    a_noc = device.worker_core_from_logical_core(CORE_A)
    b_noc = device.worker_core_from_logical_core(CORE_B)
    result_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            ttnn.CoreRangeSet({ttnn.CoreRange(CORE_A, CORE_A)}), [1, RESULT_WORDS], ttnn.ShardOrientation.ROW_MAJOR
        ),
    )
    result = ttnn.from_torch(
        torch.full((1, RESULT_WORDS), 0x7EADBEEF, dtype=torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=result_config,
    )
    semaphore_bindings = [E.SemaphoreBinding(semaphore_spec_name=name, accessor_name=name) for name in ("flag", "go")]
    defines = {"POLL_RAW": "1"} if poll_raw else {}
    spec = E.ProgramSpec(
        name="dm_cache_semaphore",
        kernels=[
            E.KernelSpec(
                unique_id="poll",
                source=E.KernelSpec.SourceCode(code=POLL_SOURCE),
                hw_config=E.create_reader_datamovement_config(disable_dfb_implicit_sync_for_all=True),
                compiler_options=E.KernelSpec.CompilerOptions(defines=defines),
                semaphore_bindings=semaphore_bindings,
                tensor_bindings=[E.TensorBinding(tensor_parameter_name="result", accessor_name="result")],
                compile_time_args={"max_spins": MAX_SPINS},
                runtime_arg_schema=E.KernelSpec.RuntimeArgSchema(runtime_arg_names=["b_x", "b_y"]),
            ),
            E.KernelSpec(
                unique_id="bump",
                source=E.KernelSpec.SourceCode(code=BUMP_SOURCE),
                hw_config=E.create_reader_datamovement_config(disable_dfb_implicit_sync_for_all=True),
                semaphore_bindings=semaphore_bindings,
                runtime_arg_schema=E.KernelSpec.RuntimeArgSchema(runtime_arg_names=["a_x", "a_y"]),
            ),
        ],
        semaphores=[
            E.SemaphoreSpec(unique_id=name, target_nodes=ttnn.CoreRange(CORE_A, CORE_B)) for name in ("flag", "go")
        ],
        tensor_parameters=[E.TensorParameter(unique_id="result", spec=result.spec)],
        work_units=[
            E.WorkUnitSpec(name="a", kernels=["poll"], target_nodes=CORE_A),
            E.WorkUnitSpec(name="b", kernels=["bump"], target_nodes=CORE_B),
        ],
    )
    run_args = E.ProgramRunArgs(
        kernel_run_args=[
            E.ProgramRunArgs.KernelRunArgs(
                kernel="poll", runtime_arg_values={"b_x": {CORE_A: b_noc.x}, "b_y": {CORE_A: b_noc.y}}
            ),
            E.ProgramRunArgs.KernelRunArgs(
                kernel="bump", runtime_arg_values={"a_x": {CORE_B: a_noc.x}, "a_y": {CORE_B: a_noc.y}}
            ),
        ],
        tensor_args={"result": result},
    )
    E.generic_op([result], spec, run_args)
    return ttnn.to_torch(result).to(torch.int64).flatten().tolist()


@pytest.mark.parametrize("poll_raw", [False, True], ids=["semaphore_class", "raw_pointer"])
def test_generic_spec_op_dm_cache_semaphore(device, poll_raw):
    if device.compute_with_storage_grid_size().x < 2:
        pytest.skip("Needs two cores in a row")
    words = run(device, poll_raw)
    print(f"DM_CACHE_SEMAPHORE poll_raw={poll_raw} result={words[:4]}")
    assert words[RAW_PRIMED] == 0, words
    expect_stale = poll_raw and is_quasar()
    if expect_stale:
        assert words[SEEN] == 0 and words[SPINS] == MAX_SPINS, words
        # Still stale after the class has observed the increment in L1.
        assert words[RAW_AFTER_CLASS_WAIT] == 0, words
    else:
        assert words[SEEN] == 1 and words[SPINS] < MAX_SPINS, words
