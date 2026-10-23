// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Kernel-exit DFB finish on Quasar. Firmware finishes every DFB once the kernel returns: each DM
// publishes how far its implicit-sync traffic got, DM0 posts the credits of any batch that never
// reached the ISR threshold, and every hart drains its tile counters. These cases target the
// situations the old per-object finish got wrong: partial last batches, threads that issue different
// amounts or nothing at all, and traffic spread over temporaries and copies of a DataflowBuffer.

#include "dfb_test_common.hpp"

namespace tt::tt_metal {
namespace {

M2SingleDFBParams implicit_params(
    M2PorCType producer_type, M2PorCType consumer_type, uint32_t num_p, uint32_t num_c, uint32_t ring, uint32_t traffic) {
    return M2SingleDFBParams{
        .producer_type = producer_type,
        .consumer_type = consumer_type,
        .num_producers = num_p,
        .num_consumers = num_c,
        .implicit_sync = true,
        .num_entries = ring,
        .num_entries_in_buffer = traffic,
    };
}

void skip_unless_quasar(distributed::MeshDevice& mesh_device) {
    if (mesh_device.arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Kernel-exit DFB finish is Quasar-only";
    }
}

}  // namespace

// The last batch on each side is smaller than the ISR threshold, so only the kernel-exit finish
// can post (producer) and ack (consumer) it.
TEST_F(UnitMeshFixture, DFBKernelExitPartialBatchDMDM) {
    skip_unless_quasar(this->device());
    for (uint32_t traffic : {7u, 9u, 11u}) {
        run_single_dfb_program_2_0(this->device(), implicit_params(M2PorCType::DM, M2PorCType::DM, 1, 1, 8, traffic));
    }
}

// Two producer threads issue 4 and 3 entries. They share each txn id's batch, so the second batch never
// reaches the threshold even though one producer finished its part of it.
TEST_F(UnitMeshFixture, DFBKernelExitUnevenProducersDMTensix) {
    skip_unless_quasar(this->device());
    run_single_dfb_program_2_0(this->device(), implicit_params(M2PorCType::DM, M2PorCType::TENSIX, 2, 1, 8, 7));
}

// One entry for two producer threads: the second thread issues nothing. The old finish made the busy
// thread wait in sync_threads() for a thread that never got there.
TEST_F(UnitMeshFixture, DFBKernelExitIdleProducerThreadDMTensix) {
    skip_unless_quasar(this->device());
    run_single_dfb_program_2_0(this->device(), implicit_params(M2PorCType::DM, M2PorCType::TENSIX, 2, 1, 8, 1));
}

// Each kernel rotates its transfers over a temporary, a copy passed by value and the main object. The
// old finish drained each temporary on destruction and lost the copies' counts.
TEST_F(UnitMeshFixture, DFBKernelExitTemporariesAndCopiesDMDM) {
    skip_unless_quasar(this->device());
    M2SingleDFBParams p = implicit_params(M2PorCType::DM, M2PorCType::DM, 1, 1, 8, 11);
    p.use_temporaries = true;
    run_single_dfb_program_2_0(this->device(), p);
}

// The same program twice: the second launch must not see the first launch's progress or exit records.
TEST_F(UnitMeshFixture, DFBKernelExitRepeatedLaunchDMDM) {
    skip_unless_quasar(this->device());
    for (int launch = 0; launch < 2; ++launch) {
        run_single_dfb_program_2_0(this->device(), implicit_params(M2PorCType::DM, M2PorCType::DM, 1, 1, 8, 9));
    }
}

}  // namespace tt::tt_metal
