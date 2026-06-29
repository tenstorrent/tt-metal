// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// noc_estimate: a thin CLI around the NoC Estimator (NPE).
//
// Wraps tt::tt_metal::experimental::noc_estimator::estimate_noc_performance() so Claude agents (and
// humans) can query the empirically-measured NoC bandwidth/latency model from the shell without a
// C++ harness. It is the data-movement-roofline source of truth: the same noc_latencies.yaml that
// the validated estimator uses, surfaced as JSON.
//
// One invocation == one "transfer group" (a homogeneous batch of NoC transactions). Compose a whole
// op's target by calling this once per group and combining the results per the perf-roofline-dm
// skill. Emits a single JSON object on stdout; on estimator failure emits {"error": ...} and exits 1.

#include <tt-metalium/experimental/noc_estimator/noc_estimator.hpp>

#include <nlohmann/json.hpp>

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>

using namespace tt::tt_metal::experimental::noc_estimator;
using json = nlohmann::json;

namespace {

// Default AICLK *busy* clocks (MHz) per arch, from umd arch implementations
// (wormhole_implementation.hpp / blackhole_implementation.hpp). tt-sim cannot supply a clock
// (SimulationChip::get_clock() returns 0); on real HW prefer FirmwareInfoProvider::get_aiclk().
constexpr double WORMHOLE_AICLK_MHZ = 1000.0;
constexpr double BLACKHOLE_AICLK_MHZ = 1350.0;

[[noreturn]] void usage_and_exit(int code) {
    std::cerr <<
        R"(noc_estimate - query the NoC Estimator (NPE) measured-bandwidth model, emit JSON.

Usage: noc_estimate [flags]

Flags (defaults mirror NocEstimatorParams):
  --mechanism NAME              UNICAST | MULTICAST | MULTICAST_LINKED        (default UNICAST)
  --pattern NAME                ONE_FROM_ONE | ONE_TO_ONE | ONE_FROM_ALL |
                                ONE_TO_ALL | ALL_TO_ALL | ALL_FROM_ALL |
                                ONE_TO_ROW | ROW_TO_ROW | ONE_TO_COLUMN |
                                COLUMN_TO_COLUMN                              (default ONE_TO_ONE)
  --memory NAME                 L1 | DRAM_INTERLEAVED | DRAM_SHARDED         (default L1)
  --arch NAME                   WORMHOLE_B0 (aliases: WH, wormhole) |
                                BLACKHOLE (aliases: BH, blackhole)           (default WORMHOLE_B0)
  --num-transactions N          total transactions in the group             (default 64)
  --num-transactions-per-barrier N   transactions in flight between barriers (default 1)
  --transaction-size-bytes N    bytes per transaction (page/tile bytes)      (default 512)
  --num-subordinates N          destination cores (for *_ALL / multicast)    (default 1)
  --same-axis[=BOOL]            src/dst share an axis                        (default false)
  --stateful[=BOOL]             stateful transfer mode                       (default false)
  --loopback[=BOOL]             loopback enabled                             (default false)
  --noc-index N                 0 or 1                                       (default 0)
  --aiclk-mhz F                 clock for cycles->ns (default: per-arch busy clock)
  -h, --help                    this message

Output (JSON): bandwidth_bytes_per_cycle, latency_cycles, aiclk_mhz, latency_ns, latency_us,
total_bytes, achieved_GBps, plus an echo of the resolved params. Exit 1 + {"error": ...} on failure.
)";
    std::exit(code);
}

const std::map<std::string, NocMechanism> kMechanisms = {
    {"UNICAST", NocMechanism::UNICAST},
    {"MULTICAST", NocMechanism::MULTICAST},
    {"MULTICAST_LINKED", NocMechanism::MULTICAST_LINKED},
};

const std::map<std::string, NocPattern> kPatterns = {
    {"ONE_FROM_ONE", NocPattern::ONE_FROM_ONE},
    {"ONE_TO_ONE", NocPattern::ONE_TO_ONE},
    {"ONE_FROM_ALL", NocPattern::ONE_FROM_ALL},
    {"ONE_TO_ALL", NocPattern::ONE_TO_ALL},
    {"ALL_TO_ALL", NocPattern::ALL_TO_ALL},
    {"ALL_FROM_ALL", NocPattern::ALL_FROM_ALL},
    {"ONE_TO_ROW", NocPattern::ONE_TO_ROW},
    {"ROW_TO_ROW", NocPattern::ROW_TO_ROW},
    {"ONE_TO_COLUMN", NocPattern::ONE_TO_COLUMN},
    {"COLUMN_TO_COLUMN", NocPattern::COLUMN_TO_COLUMN},
};

const std::map<std::string, MemoryType> kMemories = {
    {"L1", MemoryType::L1},
    {"DRAM_INTERLEAVED", MemoryType::DRAM_INTERLEAVED},
    {"DRAM_SHARDED", MemoryType::DRAM_SHARDED},
};

const std::map<std::string, Architecture> kArches = {
    {"WORMHOLE_B0", Architecture::WORMHOLE_B0},
    {"WH", Architecture::WORMHOLE_B0},
    {"WORMHOLE", Architecture::WORMHOLE_B0},
    {"BLACKHOLE", Architecture::BLACKHOLE},
    {"BH", Architecture::BLACKHOLE},
};

std::string upper(std::string s) {
    for (char& c : s) {
        c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    }
    return s;
}

template <typename T>
T lookup_enum(const std::map<std::string, T>& table, const std::string& raw, const char* what) {
    auto it = table.find(upper(raw));
    if (it == table.end()) {
        throw std::runtime_error(std::string("unknown ") + what + ": '" + raw + "'");
    }
    return it->second;
}

bool parse_bool(const std::string& v) {
    std::string u = upper(v);
    if (u == "1" || u == "TRUE" || u == "YES" || u == "ON") {
        return true;
    }
    if (u == "0" || u == "FALSE" || u == "NO" || u == "OFF") {
        return false;
    }
    throw std::runtime_error("invalid boolean: '" + v + "'");
}

uint32_t parse_u32(const std::string& v, const char* what) {
    try {
        return static_cast<uint32_t>(std::stoul(v));
    } catch (const std::exception&) {
        throw std::runtime_error(std::string("invalid integer for ") + what + ": '" + v + "'");
    }
}

double parse_f64(const std::string& v, const char* what) {
    try {
        return std::stod(v);
    } catch (const std::exception&) {
        throw std::runtime_error(std::string("invalid number for ") + what + ": '" + v + "'");
    }
}

// Split "--flag=value" into {"--flag", "value"}; returns false if there is no '='.
bool split_eq(const std::string& arg, std::string& key, std::string& val) {
    auto pos = arg.find('=');
    if (pos == std::string::npos) {
        return false;
    }
    key = arg.substr(0, pos);
    val = arg.substr(pos + 1);
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    NocEstimatorParams params{};  // struct defaults match NocEstimatorParams
    double aiclk_mhz = -1.0;      // sentinel: resolve from arch below

    // Manual flag parsing: supports "--flag value" and "--flag=value"; bool flags accept either a
    // bare "--flag" (=> true) or "--flag=BOOL".
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        std::string key;
        std::string inline_val;
        const bool has_inline = split_eq(arg, key, inline_val);
        if (!has_inline) {
            key = arg;
        }

        auto next_value = [&](const char* what) -> std::string {
            if (has_inline) {
                return inline_val;
            }
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("missing value for ") + what);
            }
            return argv[++i];
        };

        auto bool_value = [&]() -> bool {
            // "--flag" alone is true; "--flag=false" or "--flag false" honored.
            if (has_inline) {
                return parse_bool(inline_val);
            }
            return true;
        };

        try {
            if (key == "-h" || key == "--help") {
                usage_and_exit(0);
            } else if (key == "--mechanism") {
                params.mechanism = lookup_enum(kMechanisms, next_value("--mechanism"), "mechanism");
            } else if (key == "--pattern") {
                params.pattern = lookup_enum(kPatterns, next_value("--pattern"), "pattern");
            } else if (key == "--memory") {
                params.memory = lookup_enum(kMemories, next_value("--memory"), "memory");
            } else if (key == "--arch") {
                params.arch = lookup_enum(kArches, next_value("--arch"), "arch");
            } else if (key == "--num-transactions") {
                params.num_transactions = parse_u32(next_value("--num-transactions"), "--num-transactions");
            } else if (key == "--num-transactions-per-barrier") {
                params.num_transactions_per_barrier =
                    parse_u32(next_value("--num-transactions-per-barrier"), "--num-transactions-per-barrier");
            } else if (key == "--transaction-size-bytes") {
                params.transaction_size_bytes =
                    parse_u32(next_value("--transaction-size-bytes"), "--transaction-size-bytes");
            } else if (key == "--num-subordinates") {
                params.num_subordinates = parse_u32(next_value("--num-subordinates"), "--num-subordinates");
            } else if (key == "--same-axis") {
                params.same_axis = bool_value();
            } else if (key == "--stateful") {
                params.stateful = bool_value();
            } else if (key == "--loopback") {
                params.loopback = bool_value();
            } else if (key == "--noc-index") {
                params.noc_index = parse_u32(next_value("--noc-index"), "--noc-index");
            } else if (key == "--aiclk-mhz") {
                aiclk_mhz = parse_f64(next_value("--aiclk-mhz"), "--aiclk-mhz");
            } else {
                throw std::runtime_error("unknown flag: '" + key + "' (try --help)");
            }
        } catch (const std::exception& e) {
            std::cerr << "noc_estimate: " << e.what() << "\n";
            return 2;
        }
    }

    if (aiclk_mhz < 0.0) {
        aiclk_mhz = (params.arch == Architecture::BLACKHOLE) ? BLACKHOLE_AICLK_MHZ : WORMHOLE_AICLK_MHZ;
    }

    try {
        NocEstimate est = estimate_noc_performance(params);

        const double total_bytes =
            static_cast<double>(params.transaction_size_bytes) * static_cast<double>(params.num_transactions);
        // aiclk_mhz == 1e6 cycles/s. ns = cycles / (cycles/ns) = cycles * 1000 / aiclk_mhz.
        const double latency_ns = (aiclk_mhz > 0.0) ? est.latency_cycles * 1000.0 / aiclk_mhz : 0.0;
        // GB/s = bytes/cycle * cycles/s / 1e9 = bpc * aiclk_mhz * 1e6 / 1e9 = bpc * aiclk_mhz / 1000.
        const double achieved_gbps = est.bandwidth_bytes_per_cycle * aiclk_mhz / 1000.0;

        json out;
        out["bandwidth_bytes_per_cycle"] = est.bandwidth_bytes_per_cycle;
        out["latency_cycles"] = est.latency_cycles;
        out["aiclk_mhz"] = aiclk_mhz;
        out["latency_ns"] = latency_ns;
        out["latency_us"] = latency_ns / 1000.0;
        out["total_bytes"] = total_bytes;
        out["achieved_GBps"] = achieved_gbps;
        out["params"] = {
            {"mechanism", static_cast<int>(params.mechanism)},
            {"pattern", static_cast<int>(params.pattern)},
            {"memory", static_cast<int>(params.memory)},
            {"arch", static_cast<int>(params.arch)},
            {"num_transactions", params.num_transactions},
            {"num_transactions_per_barrier", params.num_transactions_per_barrier},
            {"transaction_size_bytes", params.transaction_size_bytes},
            {"num_subordinates", params.num_subordinates},
            {"same_axis", params.same_axis},
            {"stateful", params.stateful},
            {"loopback", params.loopback},
            {"noc_index", params.noc_index},
        };
        std::cout << out.dump(2) << "\n";
        return 0;
    } catch (const std::exception& e) {
        json err;
        err["error"] = e.what();
        std::cout << err.dump(2) << "\n";
        return 1;
    }
}
