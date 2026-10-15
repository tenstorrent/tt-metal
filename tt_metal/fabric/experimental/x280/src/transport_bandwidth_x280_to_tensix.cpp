// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/*
 * transport_bandwidth_x280_to_tensix.cpp -- X280 image for the X280 -> Tensix
 * addressable member of the transport_bandwidth suite.
 *
 * Only hart 0 participates. The other three harts return immediately.
 *
 * This image copies the LIM-resident section
 * out of volatile memory, checks cookies and sanity, and hands the state
 * over to the shared frontend under kernels/shared/. The state machine,
 * warm-up discipline, timed loop, boundary specialisation, and result
 * publication all live in the shared frontend so the two transport_bandwidth
 * members share one implementation.
 *
 * The X280-specific pieces -- TLB programming, mtime/rdcycle bracket
 * (§Measurement discipline), NOC store burst -- live in the X280 backend
 * header under kernels/transport_bandwidth/backends/.
 *
 * See docs/benchmarks/design.md § Transport bandwidth and
 * kernels/transport_bandwidth/transport_bandwidth_frontend_cxx.hpp for the
 * contract this image implements against.
 */

extern "C" {
#include "x280_hart_protocol.h"
#include "rt/x280_idle.h"
}

#include "transport_bandwidth_common.h"
#include "transport_bandwidth_frontend_cxx.hpp"
#include "x280_section_load.hpp"
#include "backends/x280_to_tensix.hpp"

namespace {

constexpr uint64_t kParticipants = 1u;

namespace backend = transport_bandwidth::backend_x280_to_tensix;

}  // namespace

/* `hot` because run_member inlines the timed loops here, and GCC lays out
 * loops in a plain `main` as run-once code, top-tested with an extra jump per
 * issue. */
extern "C" __attribute__((hot)) int main(uint64_t hartid) {
    UBENCH_SECTION_STORAGE(transport_bw_x280_to_tensix_section_t, section) store;
    if (!x280_load_section<transport_bw_x280_to_tensix_section_t>(hartid, kParticipants, store.as_uint32)) {
        return 0;
    }

    /* Cookie + sanity check happens before we touch the result region: the
     * only address the section carries for the result region is inside the
     * struct we just copied, so a cookie-mismatched section leaves the
     * host observing DONE-not-set and inferring ARG_LAYOUT_MISMATCH from
     * that. */
    if (!backend::sanity_check(store.section)) {
        return 0;
    }

    volatile transport_bw_result_t* result_v =
        reinterpret_cast<volatile transport_bw_result_t*>(store.section.backend.own_result_addr);
    transport_bw_result_t& r = *const_cast<transport_bw_result_t*>(result_v);

    /* The `state{}` initializer is omitted: state_t carries a 64-byte
     * staging array, and a zero-init would have GCC emit a `memset` call
     * that this freestanding image cannot resolve. `initialize()` writes
     * every scalar field before use and fills the staging array from LIM
     * on the success path.
     *
     * On an initialisation failure, publish ARG_LAYOUT_MISMATCH, or
     * WINDOW_CLEANUP_FAILED when a window it programmed does not clear -- the
     * result address is validated, so the host reads the terminal state rather
     * than inferring it. sanity_check admitted only the listed widths. */
    backend::state_t state;
    if (!backend::initialize(state, store.section, r)) {
        ubench_finish<backend::Common>(
            state, TRANSPORT_BW_STATE_ARG_LAYOUT_MISMATCH, TRANSPORT_BW_STATE_WINDOW_CLEANUP_FAILED);
        return 0;
    }
    transport_bandwidth::dispatch_size(
        backend::store_widths{}, store.section.backend.store_width, [&](auto width) __attribute__((always_inline)) {
            transport_bandwidth::run_member<backend::Backend<decltype(width)::value>>(state);
        });
    return 0;
}
