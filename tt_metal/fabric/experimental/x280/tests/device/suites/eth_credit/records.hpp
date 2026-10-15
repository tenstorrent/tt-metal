// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef X280_SUITES_ETH_CREDIT_RECORDS_HPP
#define X280_SUITES_ETH_CREDIT_RECORDS_HPP

#include "harness/measurement/measurement.hpp"
#include "harness/output/output.hpp"

#include "kernels/eth_credit/eth_credit_abi.h"

#include <cstdint>
#include <utility>
#include <vector>

namespace x280::suites::eth_credit {

inline const char* error_name(uint32_t code) {
    switch (code) {
        case EC_ERROR_NONE: return "NONE";
        case EC_ERROR_STALL_TIMEOUT: return "STALL_TIMEOUT";
        case EC_ERROR_BAD_SEQUENCE: return "BAD_SEQUENCE";
        case EC_ERROR_COUNTER_DISTANCE: return "COUNTER_DISTANCE";
        case EC_ERROR_LINK_DOWN: return "LINK_DOWN";
        case EC_ERROR_BAD_ARGS: return "BAD_ARGS";
    }
    return "?";
}

/** One kernel the host names in its reading of a failed run. */
struct Culprit {
    int chip = -1;
    uint32_t kernel_id = 0;
    uint32_t code = EC_ERROR_NONE;
    uint32_t detail = 0;
};

// One kernel's result block as its eth_credit_kernel record reports it, the
// way the kernel wrote it: identity always, and the status, packet counts,
// kept stall sites, window and failure detail only when it wrote the magic.
inline x280::harness::output::RecordSpec kernel_record(
    const char* case_name, int chip, const char* role, const ec_result_t& r) {
    namespace hout = x280::harness::output;
    std::vector<hout::Field> fields{
        {"case", case_name}, {"chip", chip}, {"role", role}, {"magic", r.magic}, {"kernel_id", r.kernel_id}};
    if (r.magic == EC_RESULT_MAGIC) {
        std::vector<hout::Value> packets_in;
        std::vector<hout::Value> packets_out;
        for (uint32_t p = 0; p < EC_MAX_P; ++p) {
            packets_in.push_back(r.packets_in[p]);
            packets_out.push_back(r.packets_out[p]);
        }
        std::vector<hout::Value> sites;
        for (const auto& st : r.stalls) {
            if (st.site == EC_SITE_NONE) {
                continue;
            }
            sites.push_back(hout::Value::object(
                {{"site", st.site}, {"cost", ec_site_is_cost(st.site)}, {"count", st.count}, {"ticks", st.ticks}}));
        }
        fields.insert(
            fields.end(),
            {
                {"state", r.status.state},
                {"code", error_name(r.status.code)},
                {"detail", r.status.detail},
                {"packets_in", hout::Value::array(std::move(packets_in))},
                {"packets_out", hout::Value::array(std::move(packets_out))},
                {"sites", hout::Value::array(std::move(sites))},
                {"first_ts", r.first_ts},
                {"last_ts", r.last_ts},
                {"failure",
                 hout::Value::object(
                     {{"p", r.failure.p},
                      {"expected", r.failure.expected},
                      {"seen", r.failure.seen},
                      {"site", r.failure.site},
                      {"local_ptr", r.failure.local_ptr},
                      {"shadow_ptr", r.failure.shadow_ptr}})},
            });
    }
    return {.suite = "eth_credit", .member = "eth_credit_kernel", .fields = std::move(fields)};
}

// A run's eth_credit_attribution record: the host's origin, stalled and
// missing kernels.
inline x280::harness::output::RecordSpec attribution_record(
    const char* case_name,
    const std::vector<Culprit>& origins,
    const std::vector<Culprit>& stalled,
    const std::vector<Culprit>& missing) {
    namespace hout = x280::harness::output;
    const auto list = [](const std::vector<Culprit>& ks) {
        std::vector<hout::Value> culprits;
        for (const auto& k : ks) {
            culprits.push_back(hout::Value::object(
                {{"chip", k.chip}, {"kernel_id", k.kernel_id}, {"code", error_name(k.code)}, {"detail", k.detail}}));
        }
        return hout::Value::array(std::move(culprits));
    };
    return {
        .suite = "eth_credit",
        .member = "eth_credit_attribution",
        .fields = {
            {"case", case_name}, {"origins", list(origins)}, {"stalled", list(stalled)}, {"missing", list(missing)}}};
}

// One chip's link state for its eth_credit_link record. `erisc` is its
// Ethernet tile. `erisc_clock` is null when no ERISC kernel launched, which
// sampled no clock.
struct LinkState {
    const char* case_name = "";
    int chip = -1;
    x280::harness::output::CorePlacement erisc;
    uint32_t retrain_count_before = 0;
    uint32_t retrain_count_after = 0;
    const char* metal_failure = "";
    const char* undrained = "";
    ec_status_t e0_status{};
    uint32_t l2cpu = 0;
    uint32_t txq0_max_pkt_bytes = 0;
    uint64_t aiclk_hz = 0;
    const x280::harness::measurement::AiclkRunEvidence* erisc_clock = nullptr;
};

// A chip's eth_credit_link record: its ERISC, the link retrain counts around
// the run, how the run ended, E0's status, and the ERISC clock evidence when
// it was sampled.
inline x280::harness::output::RecordSpec link_record(const LinkState& link) {
    namespace hout = x280::harness::output;
    std::vector<hout::Field> fields{
        {"case", link.case_name},
        {"chip", link.chip},
        {"retrain_count_before", link.retrain_count_before},
        {"retrain_count_after", link.retrain_count_after},
        {"metal_failure", link.metal_failure},
        {"undrained", link.undrained},
        {"e0_state", link.e0_status.state},
        {"e0_code", link.e0_status.code},
        {"e0_detail", link.e0_status.detail},
        {"l2cpu", link.l2cpu},
        {"txq0_max_pkt_bytes", link.txq0_max_pkt_bytes},
        {"aiclk_hz", link.aiclk_hz},
    };
    hout::append_placement(fields, "erisc", link.erisc);
    if (link.erisc_clock != nullptr) {
        fields.push_back({"erisc_clock", hout::Value::clock(*link.erisc_clock)});
    }
    return {.suite = "eth_credit", .member = "eth_credit_link", .fields = std::move(fields)};
}

// Which kernel's window a throughput record reports: the receiving forwarder
// (eth_credit), or E1 sinking locally, the Ethernet hop alone (eth_credit_hop).
enum class Receiver { Forwarder, E1 };

// One completed direction's receiver for `p` and where it ran: the sending
// chip's E0 and receiving chip's E1 Ethernet tile, and each chip's X280 tile.
struct ReceiveWindow {
    const char* case_name = "";
    uint32_t p = 0;
    int tx_chip = -1;
    int rx_chip = -1;
    uint32_t x280_hw_index = 0;
    x280::harness::output::CorePlacement tx_erisc;
    x280::harness::output::CorePlacement rx_erisc;
    x280::harness::output::CorePlacement tx_x280;
    x280::harness::output::CorePlacement rx_x280;
    uint32_t x280_pll_mhz = 0;
    ec_shape_t shape{};
    bool bidirectional = false;
    Receiver receiver = Receiver::Forwarder;
};

/**
 * The throughput record of `window.p` at `window.receiver`, whose result block
 * is `forwarder`: identity, the shape, its packet count, the packets it had
 * received when it stamped the window's start (`first_batch_packets`) and raw
 * receive window over its own clock (record_throughput states what the window
 * covers), with both chips' ERISC clock evidence. A request's slot bytes are
 * what crossed the link and `payload_bytes_per_request` the slot less the
 * header.
 *
 * A forwarder's record is eth_credit, on the fixed mtime clock, and a
 * request's useful bytes are the EC_HEADER_BYTES header that reaches the X280.
 * E1's is eth_credit_hop, on its ERISC wall clock converted by
 * `rx_erisc_clock`, and a request's useful bytes are its payload. The run has
 * no warm-up.
 */
inline x280::harness::output::RecordSpec throughput_record(
    const ReceiveWindow& window,
    const ec_result_t& forwarder,
    const x280::harness::measurement::AiclkRunEvidence& tx_erisc_clock,
    const x280::harness::measurement::AiclkRunEvidence& rx_erisc_clock) {
    namespace hout = x280::harness::output;
    const uint32_t packets = forwarder.packets_in[window.p];
    const bool at_e1 = window.receiver == Receiver::E1;
    using Clock = decltype(x280::harness::output::RecordSpec::clock);
    std::vector<hout::Field> fields{
        {"case", window.case_name},
        {"p", window.p},
        {"tx_chip", window.tx_chip},
        {"rx_chip", window.rx_chip},
        {"x280_hw_index", window.x280_hw_index},
        {"x280_pll_mhz", window.x280_pll_mhz},
        {"P", window.shape.P},
        {"N_LINK", window.shape.n_link},
        {"N_DELIVER", window.shape.n_deliver},
        {"SLOT_SIZE", window.shape.slot_size},
        {"BATCH", window.shape.batch},
        {"bidirectional", window.bidirectional},
        {"packets", packets},
        {"first_batch_packets", forwarder.first_batch_packets},
        {"payload_bytes_per_request", window.shape.slot_size - EC_HEADER_BYTES},
        {"receive_window",
         hout::Value::throughput({
             .start_ticks = forwarder.first_ts,
             .end_ticks = forwarder.last_ts,
             .completed = forwarder.status.state == EC_STATE_DONE,
             .useful_bytes_per_request = at_e1 ? window.shape.slot_size - EC_HEADER_BYTES : EC_HEADER_BYTES,
             .slot_bytes_per_request = window.shape.slot_size,
         })},
        {"tx_erisc_clock", hout::Value::clock(tx_erisc_clock)},
        {"rx_erisc_clock", hout::Value::clock(rx_erisc_clock)},
    };
    hout::append_placement(fields, "tx_erisc", window.tx_erisc);
    hout::append_placement(fields, "rx_erisc", window.rx_erisc);
    hout::append_placement(fields, "tx_x280", window.tx_x280);
    hout::append_placement(fields, "rx_x280", window.rx_x280);
    return {
        .suite = "eth_credit",
        .member = at_e1 ? "eth_credit_hop" : "eth_credit",
        .fields = std::move(fields),
        .warmup = hout::NoWarmup{},
        .clock = at_e1 ? Clock{rx_erisc_clock} : Clock{hout::kX280MtimeClock},
    };
}

}  // namespace x280::suites::eth_credit

#endif  // X280_SUITES_ETH_CREDIT_RECORDS_HPP
