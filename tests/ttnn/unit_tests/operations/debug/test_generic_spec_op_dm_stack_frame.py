# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""How much stack a Quasar DM kernel gets, measured with a growing local array, through generic_op.

The DM kernel fills a volatile local array of FRAME_BYTES, then reads one runtime arg and writes it and a checksum of
the array to an L1 tensor. Each Quasar DM hart has 8 KiB of local memory (MEM_DM_LOCAL_SIZE) shared by the firmware
TLS block and the kernel's stack.

TLS takes 6232 B:
- g_dfb_interface: 4096 B;
- cb_interface: 2048 B, the legacy CB array, unused on Quasar DM;
- a few words at its top: g_dfb_config_base_addr, crta_count, rta_count, rta_l1_base, hw_thread_idx.

The stack grows down into those words. An overflow therefore surfaces as a misleading "runtime arg index out of
bounds" or a hw_thread_idx assert (internal/hw_thread.h get_hw_thread_idx), or as a simulator abort at a garbage
address built from the kernel's own stack data.

Observed on Quasar craq-sim, watcher + LLK asserts, before the fix:
- FRAME_BYTES 512 and 1536 pass.
- 1800 and 2048 abort with `t_tile_mmio_rd32 ... addr=0x7776757473727170`: the array's fill bytes, written over a TLS
  pointer.
So the usable DM kernel stack is about 1.6-1.7 KB, and a fused blaze DM kernel_main frame of about 1.3 KB plus callees
and watcher frames crosses it.

After the fix (dm.cc no longer defines the unused cb_interface; get_arg_addr / get_common_arg_addr assert that the
stack is intact before trusting the counts):
- 1800, 2560, 3072 and 3584 pass. The usable stack is about 3.6-3.9 KB.
- 4096 trips "DM2 tripped an assert on line N", where N is the DM_ASSERT_STACK_INTACT() in get_arg_addr
  (api/dataflow/dataflow_api.h), instead of a misleading RTA error or a simulator abort.

Only sizes that must pass are parametrized, since an overflow kills the process instead of failing an assert.
"""

import pytest
import torch
import ttnn
from tt_py_test_utils_common.utility_functions import is_quasar

E = ttnn.experimental

# The sizes are about Quasar DM's local memory and TLS layout; other architectures size their DM stacks differently.
pytestmark = pytest.mark.skipif(not is_quasar(), reason="Measures the Quasar DM stack budget")
CORE = ttnn.CoreCoord(0, 0)
VALUE = 0x1234

SOURCE = """
#include "api/dataflow/dataflow_api.h"
#include "api/tensor/local_tensor_accessor.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    volatile uint8_t frame[FRAME_BYTES];
    for (uint32_t i = 0; i < FRAME_BYTES; ++i) {
        frame[i] = static_cast<uint8_t>(i);
    }
    uint32_t sum = 0;
    for (uint32_t i = 0; i < FRAME_BYTES; ++i) {
        sum += frame[i];
    }
    const uint32_t value = get_arg(args::value);
    const LocalTensorAccessor<uint32_t> out(tensor::out);
    auto lock = out.scoped_lock(0, 2);
    out[0] = value;
    out[1] = sum;
}
"""


@pytest.mark.parametrize("frame_bytes", [512, 1024, 1536, 1800, 2560, 3072, 3584])
def test_generic_spec_op_dm_stack_frame(device, frame_bytes):
    cfg = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(ttnn.CoreRangeSet({ttnn.CoreRange(CORE, CORE)}), [1, 32], ttnn.ShardOrientation.ROW_MAJOR),
    )
    out = ttnn.from_torch(
        torch.zeros(1, 32, dtype=torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=cfg,
    )
    spec = E.ProgramSpec(
        name="dm_stack_frame",
        kernels=[
            E.KernelSpec(
                unique_id="dm",
                source=E.KernelSpec.SourceCode(code=SOURCE),
                hw_config=E.create_reader_datamovement_config(disable_dfb_implicit_sync_for_all=True),
                compiler_options=E.KernelSpec.CompilerOptions(defines={"FRAME_BYTES": str(frame_bytes)}),
                tensor_bindings=[E.TensorBinding(tensor_parameter_name="out", accessor_name="out")],
                runtime_arg_schema=E.KernelSpec.RuntimeArgSchema(runtime_arg_names=["value"]),
            )
        ],
        tensor_parameters=[E.TensorParameter(unique_id="out", spec=out.spec)],
        work_units=[E.WorkUnitSpec(kernels=["dm"], target_nodes=CORE)],
    )
    run_args = E.ProgramRunArgs(
        kernel_run_args=[E.ProgramRunArgs.KernelRunArgs(kernel="dm", runtime_arg_values={"value": {CORE: VALUE}})],
        tensor_args={"out": out},
    )
    E.generic_op([out], spec, run_args)
    value, checksum = ttnn.to_torch(out).to(torch.int64).flatten().tolist()[:2]
    assert value == VALUE
    assert checksum == sum(i % 256 for i in range(frame_bytes))
