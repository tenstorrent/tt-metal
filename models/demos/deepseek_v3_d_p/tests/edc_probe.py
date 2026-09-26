# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Per-op GDDR EDC checker: a ttnn.generic_op that freezes the device when EDC counters move.

Called after each op under test. The kernel reads MRISC's GDDR EDC counters on every instance of
every device in the mesh and spins forever if any is non-zero, freezing that device at the
offending op so triage can run against it before the next reset clears the evidence.

No previous-value state is kept: MRISC zeroes the counters after training, so every
tt-smi -glx_reset_auto starts them at zero and any non-zero reading is an event since the reset.

The table sits behind the MRISC L1 aperture (MRISC_L1_ADDR = 1<<37, table at +0x8000; see
tt-system-firmware lib/tenstorrent/bh_arc/gddr.c:44-45). Reaching it needs the 64-bit overload of
noc_read_with_state, which writes NOC_TARG_ADDR_MID unmasked; the 32-bit overload applies
NOC_PCIE_MASK and would silently strip bit 37.

WARNING: an event wedges the device, and nothing resets it automatically. Reset promptly.

Env: EDC_PROBE=1 to arm; inert otherwise.
"""

import os

import torch
from loguru import logger

import ttnn

MRISC_L1_ADDR = 1 << 37
TELEMETRY_TABLE = 0x8000
# The table is 32 bytes, but a 32-byte read through the MRISC aperture returns an all-zero table
# on the odd instances; 64 bytes and up returns all 8.
READ_BYTES = 64
NUM_INSTANCES = 8
EDC_WORD_INDEX = 5
CB_INDEX = 0
PROBE_CORE = (0, 0)

KERNEL = r"""
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb         = get_compile_time_arg_val(0);
    constexpr uint32_t addr_lo    = get_compile_time_arg_val(1);
    constexpr uint32_t addr_mid   = get_compile_time_arg_val(2);
    constexpr uint32_t nbytes     = get_compile_time_arg_val(3);
    constexpr uint32_t instances  = get_compile_time_arg_val(4);
    constexpr uint32_t edc_index  = get_compile_time_arg_val(5);
    constexpr uint32_t node_xy[16] = {
        get_compile_time_arg_val(6),  get_compile_time_arg_val(7),  get_compile_time_arg_val(8),
        get_compile_time_arg_val(9),  get_compile_time_arg_val(10), get_compile_time_arg_val(11),
        get_compile_time_arg_val(12), get_compile_time_arg_val(13), get_compile_time_arg_val(14),
        get_compile_time_arg_val(15), get_compile_time_arg_val(16), get_compile_time_arg_val(17),
        get_compile_time_arg_val(18), get_compile_time_arg_val(19), get_compile_time_arg_val(20),
        get_compile_time_arg_val(21)};

    const uint32_t base = get_write_ptr(cb);
    const uint64_t src = ((uint64_t)addr_mid << 32) | (uint64_t)addr_lo;

    noc_read_init_state<NCRISC_RD_CMD_BUF>(NOC_INDEX);
    for (uint32_t i = 0; i < instances; i++) {
        noc_read_with_state<DM_DEDICATED_NOC, NCRISC_RD_CMD_BUF, CQ_NOC_SNDL>(
            NOC_INDEX,
            NOC_XY_ENCODING(
                DYNAMIC_NOC_X(NOC_INDEX, node_xy[2 * i]), DYNAMIC_NOC_Y(NOC_INDEX, node_xy[2 * i + 1])),
            src,
            base + i * nbytes,
            nbytes);
    }
    noc_async_read_barrier();

    volatile tt_l1_ptr uint32_t* words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base);
    for (uint32_t i = 0; i < instances; i++) {
        // Bytes 0/1 are corrected read/write counts, bytes 2/3 the sticky uncorrected flags.
        if (words[i * (nbytes / 4) + edc_index] != 0) {
            while (true) {
                invalidate_l1_cache();
            }
        }
    }
}
"""


def mrisc_node(instance):
    """NOC node publishing MRISC's telemetry table for a GDDR instance.

    tt-system-firmware lib/tenstorrent/bh_arc/noc.c:167-177: the right GDDR column numbers
    noc2axi ports top-to-bottom, so firmware port 0 lands on subchannel 2 there. Instance 0 is a
    documented special case running MRISC on port 2. A wrong node returns an all-zero table
    rather than an error, so this has to be exact.
    """
    right = instance // 4
    firmware_port = 2 if instance == 0 else 0
    return 17 + right, 12 + 3 * (instance % 4) + ((2 - firmware_port) if right else firmware_port)


class EdcProbe:
    def __init__(self, mesh_device):
        core = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(*PROBE_CORE), ttnn.CoreCoord(*PROBE_CORE))])
        landing_bytes = NUM_INSTANCES * READ_BYTES

        # generic_op requires an input and an output tensor. These carry nothing -- the reads land
        # in the CB and the kernel never touches them. They must live in DRAM: even a one-word L1
        # tensor clashes with the model's static CBs, which run to the L1 ceiling on [0-0 - 10-9].
        self.tensors = [
            ttnn.from_torch(
                torch.zeros(1, 1, dtype=torch.int32),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            for _ in range(2)
        ]

        # Plain CB: allocated and freed with the program, so it does not sit in the model's static
        # CB region the way a persistent tensor-backed buffer does.
        cb = ttnn.CBDescriptor(
            total_size=landing_bytes,
            core_ranges=core,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=CB_INDEX, data_format=ttnn.uint32, page_size=landing_bytes)
            ],
        )
        kernel = ttnn.KernelDescriptor(
            kernel_source=KERNEL,
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=core,
            compile_time_args=[
                CB_INDEX,
                (MRISC_L1_ADDR + TELEMETRY_TABLE) & 0xFFFFFFFF,
                (MRISC_L1_ADDR + TELEMETRY_TABLE) >> 32,
                READ_BYTES,
                NUM_INSTANCES,
                EDC_WORD_INDEX,
            ]
            + [xy for instance in range(NUM_INSTANCES) for xy in mrisc_node(instance)],
            config=ttnn.DataMovementConfigDescriptor(
                processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.RISCV_0_default
            ),
        )

        # Every device in the mesh runs the same program, so one launch covers the whole mesh.
        self.mesh_program = ttnn.MeshProgramDescriptor()
        self.mesh_program[
            ttnn.MeshCoordinateRange(
                ttnn.MeshCoordinate(0, 0),
                ttnn.MeshCoordinate(mesh_device.shape[0] - 1, mesh_device.shape[1] - 1),
            )
        ] = ttnn.ProgramDescriptor(cbs=[cb], kernels=[kernel])

        logger.info(f"EdcProbe armed on mesh {mesh_device.shape} core {PROBE_CORE}")

    def check(self):
        ttnn.generic_op(self.tensors, self.mesh_program)


class InertProbe:
    def check(self):
        pass


def make_probe(mesh_device):
    return EdcProbe(mesh_device) if os.environ.get("EDC_PROBE", "0") == "1" else InertProbe()
