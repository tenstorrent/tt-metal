// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef X280_SUITES_ETH_CREDIT_RECORDS_HPP
#define X280_SUITES_ETH_CREDIT_RECORDS_HPP

#include "harness/measurement/measurement.hpp"
#include "harness/output/output.hpp"

#include "kernels/eth_credit/eth_credit_abi.h"

#include <cstdint>
#include <string_view>
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

/** One kernel the host names in its reading of a failed run, and its core. */
struct Culprit {
    int chip = -1;
    uint32_t kernel_id = 0;
    uint32_t code = EC_ERROR_NONE;
    uint32_t detail = 0;
    x280::harness::output::CorePlacement at{};
};

/** The placement-field prefix of a kernel of `role` (ec_role_t): its Ethernet tile or X280 tile. */
inline const char* placement_prefix(uint32_t role) { return role == EC_ROLE_X280_FWD ? "x280" : "erisc"; }

// One kernel's result block as its eth_credit_kernel record reports it, the
// way the kernel wrote it: identity and where it ran (`at`, prefixed x280 for
// role FWD, else erisc) always, and the status, packet counts, kept stall sites,
// window and failure detail only when it wrote the magic.
inline x280::harness::output::RecordSpec kernel_record(
    const char* case_name,
    int chip,
    const char* role,
    const x280::harness::output::CorePlacement& at,
    const ec_result_t& r) {
    namespace hout = x280::harness::output;
    std::vector<hout::Field> fields{
        {"case", case_name}, {"chip", chip}, {"role", role}, {"magic", r.magic}, {"kernel_id", r.kernel_id}};
    hout::append_placement(fields, std::string_view(role) == "FWD" ? "x280" : "erisc", at);
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
                {{"site", st.site},
                 {"cost", ec_site_is_cost(st.site)},
                 {"stall", ec_site_is_stall(st.site)},
                 {"count", st.count},
                 {"ticks", st.ticks}}));
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
// missing kernels, each with where it ran.
inline x280::harness::output::RecordSpec attribution_record(
    const char* case_name,
    const std::vector<Culprit>& origins,
    const std::vector<Culprit>& stalled,
    const std::vector<Culprit>& missing) {
    namespace hout = x280::harness::output;
    const auto list = [](const std::vector<Culprit>& ks) {
        std::vector<hout::Value> culprits;
        for (const auto& k : ks) {
            std::vector<hout::Field> culprit{
                {"chip", k.chip}, {"kernel_id", k.kernel_id}, {"code", error_name(k.code)}, {"detail", k.detail}};
            hout::append_placement(culprit, placement_prefix(ec_kernel_role(k.kernel_id)), k.at);
            culprits.push_back(hout::Value::object(std::move(culprit)));
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

// Which kernel's window a throughput record reports: the receiving forwarder,
// E1 sinking locally, or the sending forwarder, whose window counts what it sent.
enum class Receiver { Forwarder, E1, Sender };

// The bytes of one request a throughput record counts as useful: the
// EC_HEADER_BYTES header a forwarder moves, the slot's payload, or the one
// pointer word a W2 or R1 store moves.
enum class Useful { Header, Payload, Pointer };

// What a throughput record measures: its member, the kernel whose window it
// reports, the writes inside that window (docs/benchmarks/eth_credit.md,
// "The nine writes"), and what of each request is useful.
struct Path {
    const char* member;
    Receiver receiver;
    const char* legs;
    Useful useful;
};

// The whole path, forwarder to forwarder.
inline constexpr Path kFullPath{"eth_credit", Receiver::Forwarder, "W1 W2 W3 W4 W5 W6 R1 R2 R3", Useful::Header};
// E0 sourcing and E1 sinking locally.
inline constexpr Path kEthernetHop{"eth_credit_hop", Receiver::E1, "W3 W4 R2", Useful::Payload};
// The sending forwarder into E0 sinking locally.
inline constexpr Path kNocHop{"eth_credit_leg", Receiver::Sender, "W1 W2 R3", Useful::Header};
// E1 sourcing locally into the receiving forwarder.
inline constexpr Path kReceiveSide{"eth_credit_leg", Receiver::Forwarder, "W5 W6 R1", Useful::Header};
// One write repeated by the sending forwarder with nothing else, one request
// per header (W1) or per pointer store (W2, R1).
inline constexpr Path kW1Alone{"eth_credit_leg", Receiver::Sender, "W1", Useful::Header};
inline constexpr Path kW2Alone{"eth_credit_leg", Receiver::Sender, "W2", Useful::Pointer};
inline constexpr Path kR1Alone{"eth_credit_leg", Receiver::Sender, "R1", Useful::Pointer};
// E0 sourcing over the link into the receiving forwarder: the path without the
// sending forwarder.
inline constexpr Path kE0Source{"eth_credit_leg", Receiver::Forwarder, "W3 W4 W5 W6 R1 R2", Useful::Header};

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
    Path path = kFullPath;
};

/** The bytes of one request `useful` names, for a slot of `shape`. */
inline uint32_t useful_bytes(Useful useful, const ec_shape_t& shape) {
    switch (useful) {
        case Useful::Header: return EC_HEADER_BYTES;
        case Useful::Payload: return shape.slot_size - EC_HEADER_BYTES;
        case Useful::Pointer: return sizeof(uint32_t);
    }
    return 0;
}

/**
 * The `window.path.member` record of `window.p` at `window.path.receiver`,
 * whose result block is `forwarder`: identity, the shape, the writes inside
 * the window (`legs`), its packet count (a sender's sent, else received), the
 * packets of the batch it stamped the window's start after
 * (`first_batch_packets`) and raw window over its own clock (record_throughput
 * states what the window covers), with both chips' ERISC clock evidence. A
 * request's slot bytes are the slot and `payload_bytes_per_request` the slot
 * less the header.
 *
 * A forwarder's record is on the fixed mtime clock, E1's on its ERISC wall
 * clock converted by `rx_erisc_clock`; a request's useful bytes are as
 * `window.path.useful` names. The run has no warm-up.
 */
inline x280::harness::output::RecordSpec throughput_record(
    const ReceiveWindow& window,
    const ec_result_t& forwarder,
    const x280::harness::measurement::AiclkRunEvidence& tx_erisc_clock,
    const x280::harness::measurement::AiclkRunEvidence& rx_erisc_clock) {
    namespace hout = x280::harness::output;
    const Receiver at = window.path.receiver;
    const uint32_t packets = at == Receiver::Sender ? forwarder.packets_out[window.p] : forwarder.packets_in[window.p];
    const bool at_e1 = at == Receiver::E1;
    using Clock = decltype(x280::harness::output::RecordSpec::clock);
    std::vector<hout::Field> fields{
        {"case", window.case_name},
        {"legs", window.path.legs},
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
             .useful_bytes_per_request = useful_bytes(window.path.useful, window.shape),
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
        .member = window.path.member,
        .fields = std::move(fields),
        .warmup = hout::NoWarmup{},
        .clock = at_e1 ? Clock{rx_erisc_clock} : Clock{hout::kX280MtimeClock},
    };
}

/**
 * The eth_credit_round_trip record of sending forwarder `window.p`, which
 * `window` places: identity, the shape, the loop its round trips close
 * (`loop`: `hop`, R3 after W5's acknowledgement, or `consume`, R3 after the
 * receiving forwarder's consume), the round trips it timed (`round_trips`,
 * from the W2 that publishes a packet to the first R3 that covers it) and
 * their summed raw mtime ticks (`round_trip_ticks`, `timed`'s count and
 * ticks), with both chips' ERISC clock evidence, whose AICLK the NoC runs at,
 * on the fixed mtime clock. The run has no warm-up.
 */
inline x280::harness::output::RecordSpec round_trip_record(
    const ReceiveWindow& window,
    const char* loop,
    const ec_stall_t& timed,
    const x280::harness::measurement::AiclkRunEvidence& tx_erisc_clock,
    const x280::harness::measurement::AiclkRunEvidence& rx_erisc_clock) {
    namespace hout = x280::harness::output;
    std::vector<hout::Field> fields{
        {"case", window.case_name},
        {"loop", loop},
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
        {"round_trips", timed.count},
        {"round_trip_ticks", timed.ticks},
        {"tx_erisc_clock", hout::Value::clock(tx_erisc_clock)},
        {"rx_erisc_clock", hout::Value::clock(rx_erisc_clock)},
    };
    hout::append_placement(fields, "tx_erisc", window.tx_erisc);
    hout::append_placement(fields, "rx_erisc", window.rx_erisc);
    hout::append_placement(fields, "tx_x280", window.tx_x280);
    hout::append_placement(fields, "rx_x280", window.rx_x280);
    return {
        .suite = "eth_credit",
        .member = "eth_credit_round_trip",
        .fields = std::move(fields),
        .warmup = hout::NoWarmup{},
        .clock = hout::kX280MtimeClock,
    };
}

}  // namespace x280::suites::eth_credit

#endif  // X280_SUITES_ETH_CREDIT_RECORDS_HPP
