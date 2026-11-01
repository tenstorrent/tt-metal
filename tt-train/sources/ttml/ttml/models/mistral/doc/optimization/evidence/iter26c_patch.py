"""Iteration 26c: position-aware LPT for the SDPA chain schedules (the hop stagger as an initial per-core load),
chain length and hop load tunable through env knobs for the sweep."""
import re, sys

M = "tt-train/sources/ttml/metal"
P = f"{M}/common/program_utils.hpp"
F = f"{M}/ops/sdpa_fw/device/sdpa_fw_program_factory.cpp"
Q = f"{M}/ops/sdpa_bw/device/sdpa_bw_q_program_factory.cpp"
KV = f"{M}/ops/sdpa_bw/device/sdpa_bw_kv_program_factory.cpp"


def sub(path, old, new, count=1):
    s = open(path).read()
    n = s.count(old)
    if n != count:
        print(f"FAILED {path}: {old[:80]!r} found {n}, expected {count}")
        sys.exit(2)
    open(path, "w").write(s.replace(old, new))


sub(
    P,
    """inline LptSchedule lpt_schedule(const std::vector<uint32_t>& weights, const uint32_t num_cores) {
    TT_FATAL(num_cores > 0, "lpt_schedule: num_cores must be > 0");
    LptSchedule schedule;
    schedule.items_per_core.resize(num_cores);
    std::vector<uint32_t> order(weights.size());
    std::iota(order.begin(), order.end(), 0U);
    std::stable_sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) { return weights[a] > weights[b]; });
    std::vector<uint64_t> load(num_cores, 0U);
""",
    """// `initial_load[c]` (optional) is a per-core load the items are placed on top of (e.g. a chain position's hop
// latency), so the cores that start later get fewer items.
inline LptSchedule lpt_schedule(
    const std::vector<uint32_t>& weights, const uint32_t num_cores, const std::vector<uint64_t>& initial_load = {}) {
    TT_FATAL(num_cores > 0, "lpt_schedule: num_cores must be > 0");
    TT_FATAL(
        initial_load.empty() || initial_load.size() == num_cores, "lpt_schedule: one initial load per core or none");
    LptSchedule schedule;
    schedule.items_per_core.resize(num_cores);
    std::vector<uint32_t> order(weights.size());
    std::iota(order.begin(), order.end(), 0U);
    std::stable_sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) { return weights[a] > weights[b]; });
    std::vector<uint64_t> load(num_cores, 0U);
    if (!initial_load.empty()) {
        load = initial_load;
    }
""",
)
sub(
    P,
    """    for (uint32_t core = 0; core < num_cores; ++core) {
        schedule.max_items = std::max<uint32_t>(schedule.max_items, schedule.items_per_core[core].size());
        schedule.max_load = std::max(schedule.max_load, load[core]);
    }
    return schedule;
}
""",
    """    for (uint32_t core = 0; core < num_cores; ++core) {
        schedule.max_items = std::max<uint32_t>(schedule.max_items, schedule.items_per_core[core].size());
        schedule.max_load = std::max(schedule.max_load, load[core]);
    }
    return schedule;
}

// Chain schedules (SDPA): the cores of a stream form chains of `chain_length` cores; the injector (highest index of
// its chain) starts first and every hop adds latency, so the LPT places the items on top of `hop_load * distance
// from the injector` (iteration 26). Both knobs are tunable through the environment for sweeps.
inline uint32_t env_uint32_or(const char* name, uint32_t fallback) {
    const char* v = std::getenv(name);
    return (v != nullptr && *v != '\\0') ? static_cast<uint32_t>(std::strtoul(v, nullptr, 10)) : fallback;
}

inline std::vector<uint64_t> chain_initial_loads(uint32_t stream_cores, uint32_t chain_length, uint32_t hop_load) {
    std::vector<uint64_t> initial(stream_cores, 0U);
    for (uint32_t c = 0; c < stream_cores; ++c) {
        const uint32_t pos = c % chain_length;
        const uint32_t injector = std::min(c - pos + chain_length - 1U, stream_cores - 1U);
        initial[c] = static_cast<uint64_t>(hop_load) * (injector - c);
    }
    return initial;
}
""",
)
if "#include <cstdlib>" not in open(P).read():
    sub(P, "#include <numeric>\n", "#include <cstdlib>\n#include <numeric>\n")
# ---- forward factory
sub(
    F,
    "constexpr uint32_t kChainLength = 14U;    // cores per K/V forwarding chain",
    'const uint32_t kChainLength = env_uint32_or("TTML_SDPA_CHAIN_LENGTH", 14U);  // cores per K/V forwarding chain\n'
    'const uint32_t kChainHopLoad = env_uint32_or("TTML_SDPA_FW_HOP_LOAD", 5U);     // LPT load per hop, in quarter row-tile units',
)
sub(
    F,
    """                    weights.push_back(((q + 1U + Sk_chunk_t - 1U) / Sk_chunk_t) * Sk_chunk_t + 1U);
                }
            }
            const auto schedule = lpt_schedule(weights, stream_cores);""",
    """                    weights.push_back(4U * (((q + 1U + Sk_chunk_t - 1U) / Sk_chunk_t) * Sk_chunk_t + 1U));
                }
            }
            const auto schedule =
                lpt_schedule(weights, stream_cores, chain_initial_loads(stream_cores, kChainLength, kChainHopLoad));""",
)
# ---- Q and KV factories: chain_schedule gets the hop load
for path, env, default, unit in [
    (Q, "TTML_SDPA_Q_HOP_LOAD", 2, "quarter K-row units"),
    (KV, "TTML_SDPA_KV_HOP_LOAD", 2, "quarter Q-row units"),
]:
    sub(
        path,
        "constexpr uint32_t kChainLength = 14U;    // cores per forwarding chain",
        f'const uint32_t kChainLength = env_uint32_or("TTML_SDPA_CHAIN_LENGTH", 14U);  // cores per forwarding chain\n'
        f'const uint32_t kChainHopLoad = env_uint32_or("{env}", {default}U);    // LPT load per hop, in {unit}',
    )
    sub(
        path,
        "        const auto lpt = lpt_schedule(weights, stream_cores);",
        "        const auto lpt = lpt_schedule(weights, stream_cores, chain_initial_loads(stream_cores, kChainLength, kChainHopLoad));",
    )
print("patched")
