// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// BENCH-LOCAL variant of dataflow_kernel_lib::ReceiverPipe with ONE difference:
// the constructor does NOT kernel-init the `data_ready` flag cell.
//
// WHY IT EXISTS (helper bypass, idea I8). mcast_pipe's ReceiverPipe ctor sets
// data_ready = INVALID and mcast_pipe.hpp:27-29 justifies that as race-free because
// "the sender — the only other writer — is gated behind that ack". Elide the ack
// (PRE_HANDSHAKE=false) and that justification evaporates for a receiver that has NO
// other happens-before edge to the sender (e.g. an mcast-box FILLER core that never
// gathers): the sender could broadcast VALID before that core's ctor runs, and the
// ctor would clobber it -> permanent hang. The fix is the one mcast_pipe already
// prescribes for exactly this situation (line 26): "otherwise the init races and the
// initial value must come from the HOST". The host already creates the semaphore at
// 0 == INVALID, so the ctor set is pure redundancy that must be dropped along with
// the ack.
//
// The proposed kernel_lib change is ONE line in mcast_pipe.inl's ReceiverPipe ctor:
//     if constexpr (DATA_READY_SIGNAL == DataReadySignal::Flag && PRE_HANDSHAKE) {
//         data_ready_.set(INVALID);
//     }
// (plus a header note that the no-handshake path takes its INVALID from host
// CreateSemaphore). This file is that change, expressed locally so the bench can
// measure it without touching the shared library.

#pragma once

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/noc_semaphore.h"
#include "hostdevcommon/common_values.hpp"

namespace bench {

// No PRE_HANDSHAKE template param: this type IS the no-handshake receiver.
// DataReadySignal::Flag only (the op's signal).
template <uint32_t DATA_READY_SEM_ID>
class NoInitReceiverPipe {
public:
    explicit NoInitReceiverPipe(const Noc& noc) : noc_(noc), data_ready_(DATA_READY_SEM_ID) {
        // NO data_ready_.set(INVALID) — the host's CreateSemaphore(..., 0 == INVALID)
        // owns the initial value, because with no ack this core has no
        // happens-before edge to the sender.
    }

    void receive() {
        data_ready_.wait(VALID);
        data_ready_.set(INVALID);  // clear this round's flag before the next block
    }

private:
    Noc noc_;
    Semaphore<> data_ready_;
};

}  // namespace bench
