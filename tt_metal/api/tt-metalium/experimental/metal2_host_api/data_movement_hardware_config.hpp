// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>

#include <tt-metalium/experimental/metal2_host_api/dataflow_buffer_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/utility/group.hpp>
#include <tt-metalium/kernel_types.hpp>  // For DataMovementProcessor, NOC, etc.

namespace tt::tt_metal::experimental {

// Whether a dedicated-NOC data movement kernel may share its NOC with the other data movement kernel on
// its nodes. See DataMovementHardwareConfig::DataMovement1XXConfig::dedicated_noc_sharing.
enum class DedicatedNocSharing : uint8_t { EXCLUSIVE, SHARED_DISJOINT_TRAFFIC };

// ============================================================================
//  DataMovementHardwareConfig
// ============================================================================
//
// Describes the hardware resources controlled by a data movement ("DM") kernel.
//
// ============================================================================

struct DataMovementHardwareConfig {
    // ---- Generation-specific configs ----
    // Optional and unset by default. Each config applies only to the architecture named in
    // its header, and is ignored on any other.
    // NOTE: The target architecture is selected at program construction time.
    //       See MakeProgramFromSpec for more details.

    // ---- TT-1.x.x specific (Wormhole, Blackhole) ----
    //
    // The common case is handled by role-specific factory functions:
    //  - For a DM kernel that reads from DRAM: CreateReaderDataMovementConfig()
    //  - For a DM kernel that writes to DRAM:  CreateWriterDataMovementConfig()
    //
    // Power users can override these conventions by constructing a
    // DataMovement1XXConfig directly.
    struct DataMovement1XXConfig {
        // The RISC-V core that runs this DM kernel (RISCV_0 or RISCV_1)
        // Each DM kernel on a node must be assigned to a unique RISC-V core
        tt::tt_metal::DataMovementProcessor processor;

        // The physical NOC that this DM kernel uses (NOC_0 or NOC_1)
        tt::tt_metal::NOC noc;

        // NOC ownership model. Leave as DM_DEDICATED_NOC unless you specifically
        // need both DM cores to share a single NOC (e.g. to keep the other NOC
        // free for fabric/CCL traffic). Dynamic mode adds cross-core coordination
        // overhead and must be set identically on both DM kernels on a node.
        // Two dedicated-NOC kernels whose traffic is disjoint can also share a NOC
        // (dedicated_noc_sharing, below).
        tt::tt_metal::NOC_MODE noc_mode = tt::tt_metal::NOC_MODE::DM_DEDICATED_NOC;

        // DM_DEDICATED_NOC only; ignored in DM_DYNAMIC_NOC mode. By default each dedicated-NOC kernel on a
        // node needs a NOC of its own. When both DM kernels on a node set SHARED_DISJOINT_TRAFFIC, they may
        // name the same noc, which keeps the other NOC free without dynamic mode's per-transaction overhead.
        //
        // CAUTION: The caller guarantees that no kind of NoC transaction (reads, writes, atomics) is issued
        //          by both kernels, e.g. one only reads and the other only writes and issues atomics. A kind
        //          issued by both can let a barrier return before its transactions complete. Posted and
        //          non-posted writes, inline writes included, are one kind: in dedicated mode both kernels
        //          issue them from the same command buffers, so one kernel's write can redirect the other's
        //          stateful (set_state / with_state) writes. Only the command buffers are checked: under the
        //          watcher's NoC sanitizer the kernels record the command buffers they program on the shared
        //          NOC, and a command buffer used by both is reported as a watcher error. The
        //          transaction-kind condition itself goes unchecked.
        //
        // The NOC's hardware transaction counters count both kernels' traffic, but noc_async_full_barrier
        // and the end-of-kernel checks (built with watcher or lightweight kernel asserts) compare each RISC's
        // own counts against them. So neither kernel calls a full barrier, and each re-syncs its counters
        // (ncrisc_noc_counters_init) before it returns, once the other kernel has nothing in flight on the
        // NOC. Two other sources of NoC traffic also count against the condition:
        //   - The device profiler pushes a full buffer to DRAM mid-kernel as posted writes on the kernel's
        //     NOC.
        //   - On Blackhole, BRISC firmware sets up remote circular buffers with atomics on BRISC's NOC and
        //     then waits for its own count of them, after it has started the NCRISC kernel. In a program
        //     with remote circular buffers, the NCRISC kernel must not issue atomics on the shared NOC.
        DedicatedNocSharing dedicated_noc_sharing = DedicatedNocSharing::EXCLUSIVE;
    };
    // NOTE: If this kernel is built for a 1st-gen (1xx) architecture, config_1xx
    //       cannot be default-constructed; processor and noc fields have no default.
    std::optional<DataMovement1XXConfig> config_1xx = std::nullopt;

    // ---- TT-2.x.x specific (Quasar and derivatives) ----
    struct DataMovement2XXConfig {
        // Opt-out of DFB implicit sync (on a per-DFB basis)
        //  - Implicit sync enables streamlined kernel-side syntax, but triggers ISR handling.
        //  - Use this control to revert to legacy explicit sync APIs (for specific bound DFBs).
        //  - Opting out is mainly for debug purposes, or for backwards-compatible code style.
        // Any bound DFB not listed here will use implicit sync by default.
        Group<DFBSpecName> disable_dfb_implicit_sync_for;

        // Opt out of DFB implicit sync for ALL the DFBs this kernel binds.
        // (The per-kernel hammer; equivalent to listing every bound DFB above.)
        bool disable_dfb_implicit_sync_for_all = false;
    };
    std::optional<DataMovement2XXConfig> config_2xx = std::nullopt;
};

// Factory helper:
// Default config for a reader DM kernel (i.e. that reads from DRAM)
inline DataMovementHardwareConfig CreateReaderDataMovementConfig() noexcept {
    return DataMovementHardwareConfig{
        // On Wormhole, RISCV_1 runs faster than RISCV_0 due to its dedicated
        // instruction memory. Since DM reader kernels are usually more complex than
        // DM writer kernels, prefer RISCV_1 for readers.
        // NOTE:
        //  - The Wormhole RISCV_1 dedicated instruction memory has a 16 kB size limit,
        //    so a large DM reader kernel may fail to fit. (Runtime error.)
        //  - On Blackhole, RISCV_0 and RISCV_1 have no meaningful performance difference;
        //    reader DM kernels are placed on RISCV_1 by convention only.
        .config_1xx =
            DataMovementHardwareConfig::DataMovement1XXConfig{
                .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,

                // It is more efficient to read from DRAM via NOC_0 on 1st-gen (1xx) devices.
                // This is a subtle consequence of the device topology:
                //  - NOC_0 routes east, then south (rows first)
                //  - NOC_1 routes north, then west (columns first)
                //  - DRAMs are in columns
                // Return data from DRAM reads may cause NOC congestion if it flows column-first.
                // Thus, prefer NOC_0 for DRAM reads.
                .noc = tt::tt_metal::NOC::NOC_0,

                // Dedicated NOC mode is the most bandwidth-efficient mode.
                // (Dynamic NOC mode is generally only used to multiplex both DM cores onto a
                // single NOC in order to reserve the other NOC for fabric traffic.)
                .noc_mode = tt::tt_metal::NOC_MODE::DM_DEDICATED_NOC,
            },
    };
}

// Factory helper:
// Default config for a writer DM kernel (i.e. that writes to DRAM)
inline DataMovementHardwareConfig CreateWriterDataMovementConfig() noexcept {
    return DataMovementHardwareConfig{
        .config_1xx =
            DataMovementHardwareConfig::DataMovement1XXConfig{
                // DM kernels on the same node must be assigned to different RISC-V cores.
                // Since RISCV_1 is preferred for readers, place writers on RISCV_0.
                .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,

                // Since NOC_0 is preferred for DRAM reads, use NOC_1 for DRAM writes.
                .noc = tt::tt_metal::NOC::NOC_1,

                // Prefer dedicated NOC mode for the same reasons as above.
                .noc_mode = tt::tt_metal::NOC_MODE::DM_DEDICATED_NOC,
            },
    };
}

}  // namespace tt::tt_metal::experimental
