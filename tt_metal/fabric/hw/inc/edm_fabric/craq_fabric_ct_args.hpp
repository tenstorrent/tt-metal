// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// =============================================================================
// craq_fabric_ct_args.hpp
// =============================================================================
//
// Private CT-arg surface for the CRAQ-Fabric generated kernel.
//
// HARD RULE (see memory/feedback_no_upstream_ct_args.md):
//   The generated kernel MUST NOT take its CT-arg constants from upstream's
//   `fabric_erisc_router_ct_args.hpp`. Every constant the kernel reads must
//   come from OUR builder via OUR macro infrastructure.
//
// This header defines:
//   - `CRAQ_NAMED_CT_ARG(name)` -- our wrapper around the same underlying
//     `get_named_compile_time_arg_val(name)` primitive upstream uses, but
//     spelled differently so that any kernel-side regression to upstream's
//     `NAMED_CT_ARG(...)` macro is visible at lint time.
//   - The `craq_*` constexpr constants the kernel reads instead of the
//     identically-shaped upstream constants. Builder publishes the
//     underlying CT args under the names referenced below.
//
// Backwards-compat notice: this header does NOT replace upstream's CT args
// for upstream's own kernel. It exists solely so the CRAQ-Fabric kernel is
// auditable as a self-contained generated artifact.
// =============================================================================

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>

#ifndef OFFLINE_COMPILATION
// `get_named_compile_time_arg_val` lives behind `api/compile_time_args.h` in
// the device build. Under OFFLINE_COMPILATION the mocks declare a fake at
// global scope; we therefore only include the real header outside the mock.
#include "api/compile_time_args.h"
#endif

// -----------------------------------------------------------------------------
// CRAQ_NAMED_CT_ARG -- our private CT-arg lookup macro.
// -----------------------------------------------------------------------------
//
// Distinct macro name from upstream's `NAMED_CT_ARG` so that:
//   - lint checks can flag any leftover `NAMED_CT_ARG(...)` use in our codegen
//     output as an upstream-coupling violation;
//   - grepping for the macro in our tree finds only our usages;
//   - the underlying primitive is identical, so values are retrieved exactly
//     the same way upstream does.
#define CRAQ_NAMED_CT_ARG(name) get_named_compile_time_arg_val(name)

namespace tt::tt_fabric::craq {

// -----------------------------------------------------------------------------
// Multi-TXQ credit-region addresses.
// -----------------------------------------------------------------------------
//
// On Blackhole 2-erisc with SENDER_TXQ_ID=0 and RECEIVER_TXQ_ID=1, TXQ1 cannot
// do `eth_reg_write` MMIO -- so receiver->sender credit replies must travel
// over ETH *data* into an L1 counter array. The four base addresses below
// describe the per-side L1 layout; `craq_to_senders_credits_base_address` /
// `craq_local_receiver_credits_base_address` are derived as the lower of the
// two sub-region bases, and `craq_total_number_of_receiver_to_sender_credit_num_bytes`
// is the total bulk-send size.
//
// These mirror upstream's identically-named constants in
// `fabric_erisc_router_ct_args.hpp`, but are PUBLISHED BY OUR BUILDER under
// our own names. The values come from the same allocator decisions that
// inform upstream's allocation, but the CT-arg flow is no longer transitive
// through upstream's header.
constexpr std::size_t craq_to_sender_remote_ack_counters_base_address =
    CRAQ_NAMED_CT_ARG("TO_SENDER_REMOTE_ACK_COUNTERS_BASE_ADDR");
constexpr std::size_t craq_to_sender_remote_completion_counters_base_address =
    CRAQ_NAMED_CT_ARG("TO_SENDER_REMOTE_COMPLETION_COUNTERS_BASE_ADDR");

// Bulk-send origin/target = whichever sub-region sits lower in L1.
constexpr std::size_t craq_to_senders_credits_base_address = std::min(
    craq_to_sender_remote_ack_counters_base_address,
    craq_to_sender_remote_completion_counters_base_address);

constexpr std::size_t craq_local_receiver_ack_counters_base_address =
    CRAQ_NAMED_CT_ARG("LOCAL_RECEIVER_ACK_COUNTERS_BASE_ADDR");
constexpr std::size_t craq_local_receiver_completion_counters_base_address =
    CRAQ_NAMED_CT_ARG("LOCAL_RECEIVER_COMPLETION_COUNTERS_BASE_ADDR");

constexpr std::size_t craq_local_receiver_credits_base_address = std::min(
    craq_local_receiver_ack_counters_base_address,
    craq_local_receiver_completion_counters_base_address);

// Two contiguous arrays of equal size: take the size of the first and double.
constexpr std::size_t craq_total_number_of_receiver_to_sender_credit_num_bytes =
    (std::max(craq_local_receiver_ack_counters_base_address,
              craq_local_receiver_completion_counters_base_address) -
     craq_local_receiver_credits_base_address) *
    2;

// -----------------------------------------------------------------------------
// TXQ assignments + multi_txq_enabled.
// -----------------------------------------------------------------------------
//
// SENDER_TXQ_ID/RECEIVER_TXQ_ID are also published by our builder for the
// kernel; the kernel separately declares those at file scope. We re-expose a
// matching constexpr here for `craq_multi_txq_enabled` derivation so the
// outlined headers do not need to read the kernel-scope constants.
constexpr std::size_t craq_sender_txq_id = CRAQ_NAMED_CT_ARG("SENDER_TXQ_ID");
constexpr std::size_t craq_receiver_txq_id = CRAQ_NAMED_CT_ARG("RECEIVER_TXQ_ID");
constexpr bool craq_multi_txq_enabled = (craq_sender_txq_id != craq_receiver_txq_id);

}  // namespace tt::tt_fabric::craq

// Bring the most frequently used names into the unqualified namespace at file
// scope so the kernel body reads cleanly. This is a deliberate trade-off: the
// kernel becomes terse, but the names remain auditable because they all carry
// the `craq_` prefix that is unique to this header.
using tt::tt_fabric::craq::craq_to_sender_remote_ack_counters_base_address;
using tt::tt_fabric::craq::craq_to_sender_remote_completion_counters_base_address;
using tt::tt_fabric::craq::craq_to_senders_credits_base_address;
using tt::tt_fabric::craq::craq_local_receiver_ack_counters_base_address;
using tt::tt_fabric::craq::craq_local_receiver_completion_counters_base_address;
using tt::tt_fabric::craq::craq_local_receiver_credits_base_address;
using tt::tt_fabric::craq::craq_total_number_of_receiver_to_sender_credit_num_bytes;
using tt::tt_fabric::craq::craq_multi_txq_enabled;
