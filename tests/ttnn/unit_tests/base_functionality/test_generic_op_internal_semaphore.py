# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Test: an op-internal GlobalSemaphore parked in MeshProgramDescriptor.semaphores
survives a generic_op program-cache hit.

This is the single-device unit test for the B1 fix (MeshProgramDescriptor.semaphores —
an optional slot the generic_op adapter copies into the cached workload's shared_variables
at cache-miss, keeping the GlobalSemaphore's L1 allocation alive for the cached workload's
lifetime). The cross-device transfer that motivates the slot is multi-device, but the
cache/lifetime contract the slot provides is observable on a single device.

How it proves the contract deterministically:
  * The op-internal GlobalSemaphore is created ONCE (initial value 0) and parked via
    `mesh_pd.semaphores = [sem]` on the cache-MISS call only.
  * The kernel ACCUMULATES into the semaphore's L1 (`*sem += 5`) and writes the running
    value to the output. So the miss call yields 5.
  * Before the second (cache-HIT) call we DROP the only Python reference to the semaphore
    (`del sem; gc.collect()`) and pass `mesh_pd.semaphores = []`. The descriptor still bakes
    the same semaphore address, and the program hash excludes `.semaphores`, so the second
    call hits the cache.
  * If the framework kept the parked semaphore's L1 alive, its value persisted (5), so the
    hit call accumulates to 10. If the L1 had been freed when Python released the handle,
    the hit call could not read back 5 → it would not reach 10. The value 10 is therefore a
    deterministic witness that the *framework* (not the Python caller) owns the lifetime.
"""

import gc

import pytest
import torch
from loguru import logger

import ttnn

# A single dataflow kernel: accumulate `increment` into the op-internal semaphore's L1,
# then publish the running value to the (tensor-backed) output CB.
KERNEL_SOURCE = """
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t sem_addr   = get_arg_val<uint32_t>(0);
    uint32_t increment  = get_arg_val<uint32_t>(1);
    constexpr uint32_t out_cb = get_compile_time_arg_val(0);

    volatile tt_l1_ptr uint32_t* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_addr);
    *sem += increment;                 // accumulate into the persistent GlobalSemaphore L1
    uint32_t running = *sem;

    cb_reserve_back(out_cb, 1);
    uint32_t dst = get_write_ptr(out_cb);
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst) = running;
    cb_push_back(out_cb, 1);
}
"""

INCREMENT = 5


@pytest.mark.parametrize("device", [(1,)], indirect=True)
def test_op_internal_semaphore_survives_cache_hit(device):
    """A GlobalSemaphore parked in MeshProgramDescriptor.semaphores stays alive across a
    generic_op cache hit even after the Python handle is dropped."""
    core = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])

    # A dummy input (generic_op requires >= 2 io_tensors) and a 1-uint32 output, both L1-sharded.
    def l1_uint32_tensor(value):
        return ttnn.from_torch(
            torch.full((1, 1), value, dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(core, (1, 1), ttnn.ShardOrientation.ROW_MAJOR),
            ),
        )

    dummy_in = l1_uint32_tensor(0)
    out_tensor = l1_uint32_tensor(0)

    # Op-internal semaphore: created ONCE, value 0, over the single worker core.
    sem = ttnn.create_global_semaphore(device, core, 0)
    sem_addr = ttnn.get_global_semaphore_address(sem)
    logger.info(f"op-internal GlobalSemaphore @ 0x{sem_addr:x}")

    def build_mesh_pd(park_handle):
        out_cb = ttnn.cb_descriptor_from_sharded_tensor(0, out_tensor, total_size=4, core_ranges=core)
        out_cb.format_descriptors = [ttnn.CBFormatDescriptor(0, ttnn.uint32, 4)]
        rt = ttnn.RuntimeArgs()
        rt[0][0] = [sem_addr, INCREMENT]
        kernel = ttnn.KernelDescriptor(
            kernel_source=KERNEL_SOURCE,
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=core,
            compile_time_args=[0],
            runtime_args=rt,
            config=ttnn.DataMovementConfigDescriptor(
                processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.RISCV_0_default
            ),
        )
        pd = ttnn.ProgramDescriptor(cbs=[out_cb], kernels=[kernel])
        mesh_pd = ttnn.MeshProgramDescriptor()
        coord = ttnn.MeshCoordinate(0, 0)
        mesh_pd[ttnn.MeshCoordinateRange(coord, coord)] = pd
        # Park the op-internal semaphore so the framework keeps its L1 alive for the cached
        # workload's lifetime (only needed on the cache-miss call that builds the workload).
        mesh_pd.semaphores = [sem] if park_handle else []
        return mesh_pd

    def run_and_read():
        ttnn.generic_op([dummy_in, out_tensor], build_mesh_pd.current)
        ttnn.synchronize_device(device)
        return int(ttnn.to_torch(out_tensor).flatten()[0].item())

    entries_before = device.num_program_cache_entries()

    # Cache MISS: park the semaphore; first accumulation -> 5.
    build_mesh_pd.current = build_mesh_pd(park_handle=True)
    val_miss = run_and_read()
    logger.info(f"miss call: out={val_miss} (expect {INCREMENT})")
    assert val_miss == INCREMENT, f"miss call wrong: {val_miss} != {INCREMENT}"
    entries_after_miss = device.num_program_cache_entries()
    assert entries_after_miss == entries_before + 1, "expected exactly one new cache entry on miss"

    # Drop the ONLY Python reference to the semaphore. Without the framework parking it, the
    # GlobalSemaphore destructor would free its L1 here.
    del sem
    gc.collect()

    # Cache HIT: do NOT re-park the handle (we dropped it); the descriptor still bakes the same
    # address. If the framework kept the parked semaphore alive, its L1 still holds 5, so this
    # second accumulation reaches 10.
    build_mesh_pd.current = build_mesh_pd(park_handle=False)
    val_hit = run_and_read()
    logger.info(f"hit call: out={val_hit} (expect {2 * INCREMENT})")
    assert device.num_program_cache_entries() == entries_after_miss, "second call must be a cache hit"
    assert val_hit == 2 * INCREMENT, (
        f"op-internal semaphore did NOT survive the cache hit: out={val_hit}, expected {2 * INCREMENT}. "
        f"The parked GlobalSemaphore's L1 (value 5 from the miss call) was not kept alive by the framework."
    )
