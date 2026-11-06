// slot_alloc_table: the dynamic slot allocator's picks (SLOT_ALLOCATOR.md) for the configurations the study runs,
// next to main's tier pick, the study router's default row today and the study's best measured rows.
// It compiles the host's own allocator source (tt_metal/fabric/builder/fabric_slot_allocator.cpp), so what it prints
// is what the builder does with TT_FABRIC_EXP_DYN_SLOTS=1. Build and run: slot_alloc_table.sh [ARGS].
//
//   --sizes 4352,3072      payload sizes (default 4352,3072)
//   --policy k=v,...       policy overrides, as TT_FABRIC_EXP_DYN_SLOTS_POLICY (SLOT_ALLOCATOR.md)
//   --reasons              every count's reason
//   --only SUBSTRING       configurations whose name contains SUBSTRING
//   --trids                the receivers' transaction IDs instead (TRID_ALLOCATION.md): per router kind, today's
//   uniform
//                          rule against assign_transaction_ids, for main's tier slots and the dynamic slots
//
// Budgets (bytes of L1 for channel buffers): the region start from the study's argument dumps (2D, VC1 and VC2 builds
// 90,912; 1D 89,504) up to MEM_ERISC_MAX_SIZE 455,264 (dev_mem_map.h): 364,352 B and 365,760 B. They give every slot
// count the study logged (2D 81 / 99 / 115 / 137 / 169 at 4352 / 3584 / 3072 / 2560 / 2048 B; 1D 83 / 117 / 174).
// The builder itself uses the budget it computes.

#include "tt_metal/fabric/builder/fabric_slot_allocator.hpp"
#include "tt_metal/fabric/builder/fabric_trid_allocator.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

namespace sa = tt::tt_fabric::slot_allocator;
namespace ta = tt::tt_fabric::trid_allocator;

namespace {

constexpr std::size_t BUDGET_2D = 364352;
constexpr std::size_t BUDGET_1D = 365760;
constexpr std::size_t HEADER_2D = 96;
constexpr std::size_t HEADER_1D = 48;
constexpr std::size_t MAX_SLOTS = 127;  // MAX_CHANNEL_BUFFER_SLOTS

enum class Family { Mesh, Torus, Linear, Ring };

struct Config {
    std::string name;
    Family family;
    std::size_t dim_y, dim_x;
    std::size_t longest_ring;  // not read by the rules (the K + 2 minimum was dropped); kept to name the shape
    bool vc1, vc2;
    bool vc1_first;
};

// One allocation per channel kind: VC0 worker, VC0 forwarding, VC0 receiver, VC1 forwarding, VC1 receiver, VC2
// worker, VC2 receiver (0 where absent).
struct Row {
    std::size_t w = 0, f = 0, r = 0, f1 = 0, r1 = 0, w2 = 0, r2 = 0;
};

bool is_1d(Family f) { return f == Family::Linear || f == Family::Ring; }

std::string show(const Row& row, const Config& c) {
    std::string s = std::to_string(row.w) + "," + std::to_string(row.f) + "," + std::to_string(row.r);
    if (c.vc1) {
        s += "," + std::to_string(row.f1) + "," + std::to_string(row.r1);
    }
    if (c.vc2) {
        s += " + " + std::to_string(row.w2) + "," + std::to_string(row.r2);
    }
    return s;
}

// Channels per VC as the builder sees them: VC0 4 senders in 2D (worker + 3 forwarding), 2 in 1D; VC1 3 forwarding;
// VC2 1 worker; one receiver per VC.
sa::RouterShape shape_of(const Config& c, std::size_t payload) {
    sa::RouterShape s;
    s.budget_bytes = is_1d(c.family) ? BUDGET_1D : BUDGET_2D;
    s.slot_bytes = payload + (is_1d(c.family) ? HEADER_1D : HEADER_2D);
    s.vcs[0] = {1, is_1d(c.family) ? std::size_t{1} : std::size_t{3}, 1};
    if (c.vc1) {
        s.vcs[1] = {0, 3, 1};
    }
    if (c.vc2) {
        s.vcs[2] = {1, 0, 1};
    }
    s.mesh_dims = {c.dim_y, c.dim_x};
    s.two_d = !is_1d(c.family);
    return s;
}

// Main's pick (fabric_static_sized_channels_allocator.cpp, Blackhole): the first tier row that fits, one depth per
// kind, then on Mesh / Torus every spare slot to the worker.
Row main_tier(const Config& c, std::size_t n) {
    struct Tier {
        std::size_t s0, r0, s1, r1, s2, r2;
    };
    static const std::vector<Tier> vc0_only = {
        {32, 32, 0, 0, 0, 0},
        {16, 32, 0, 0, 0, 0},
        {16, 16, 0, 0, 0, 0},
        {8, 16, 0, 0, 0, 0},
        {8, 8, 0, 0, 0, 0},
        {4, 8, 0, 0, 0, 0},
        {4, 4, 0, 0, 0, 0},
        {2, 4, 0, 0, 0, 0},
        {2, 2, 0, 0, 0, 0},
        {1, 2, 0, 0, 0, 0},
        {1, 1, 0, 0, 0, 0}};
    static const std::vector<Tier> vc0_vc1 = {
        {4, 8, 2, 4, 0, 0},
        {4, 8, 2, 2, 0, 0},
        {2, 4, 2, 2, 0, 0},
        {2, 4, 1, 1, 0, 0},
        {2, 2, 1, 1, 0, 0},
        {1, 1, 1, 1, 0, 0}};
    static const std::vector<Tier> vc0_vc1_vc2 = {
        {4, 8, 2, 4, 2, 4},
        {4, 8, 2, 2, 2, 2},
        {2, 4, 2, 2, 2, 2},
        {2, 4, 1, 1, 1, 1},
        {2, 2, 1, 1, 1, 1},
        {1, 1, 1, 1, 1, 1}};
    static const std::vector<Tier> vc0_vc2 = {
        {7, 11, 0, 0, 2, 4},
        {4, 8, 0, 0, 2, 4},
        {4, 8, 0, 0, 2, 2},
        {2, 4, 0, 0, 2, 2},
        {2, 4, 0, 0, 1, 1},
        {2, 2, 0, 0, 1, 1},
        {1, 1, 0, 0, 1, 1}};
    const bool mesh = !is_1d(c.family);
    const auto& table = !mesh              ? vc0_only
                        : (c.vc1 && c.vc2) ? vc0_vc1_vc2
                        : c.vc2            ? vc0_vc2
                        : c.vc1            ? vc0_vc1
                                           : vc0_only;
    const std::size_t s0 = mesh ? 4 : 2, s1 = c.vc1 ? 3 : 0, s2 = c.vc2 ? 1 : 0;
    for (const Tier& t : table) {
        const std::size_t used = s0 * t.s0 + t.r0 + s1 * t.s1 + (c.vc1 ? t.r1 : 0) + s2 * t.s2 + (c.vc2 ? t.r2 : 0);
        if (used > n) {
            continue;
        }
        Row row{t.s0, t.s0, t.r0, t.s1, t.r1, t.s2, t.r2};
        if (mesh) {
            row.w = std::min(t.s0 + (n - used), MAX_SLOTS);
        }
        return row;
    }
    return Row{};
}

// The study router's default today (kernel v2, no slot switch): 1D splits the senders' slots evenly after the tier
// receiver; VC1 / VC2 builds take the study rows with the worker taking the rest; 2D keeps main's pick.
Row study_default(const Config& c, std::size_t n) {
    Row tier = main_tier(c, n);
    if (is_1d(c.family)) {
        const std::size_t senders = n - tier.r;
        Row row = tier;
        row.f = std::min(senders / 2, MAX_SLOTS);
        row.w = std::min(senders - row.f, MAX_SLOTS);
        return row;
    }
    if (!c.vc1 && !c.vc2) {
        return tier;
    }
    Row design = !c.vc2   ? Row{19, 9, 16, 5, 4, 0, 0}
                 : !c.vc1 ? Row{17, 14, 16, 0, 0, 2, 4}
                          : Row{19, 8, 16, 4, 4, 2, 4};
    const std::size_t others =
        3 * design.f + design.r + (c.vc1 ? 3 * design.f1 + design.r1 : 0) + (c.vc2 ? design.w2 + design.r2 : 0);
    if (others + design.w > n) {
        return tier;
    }
    design.w = std::min(n - others, MAX_SLOTS);
    return design;
}

// The study's best measured rows (SLOT_STUDY.md, SLOT_DECISIONS.md), by configuration kind and payload.
std::string study_best(const Config& c, std::size_t payload) {
    static const std::map<std::pair<std::string, std::size_t>, std::string> best = {
        {{"2d", 4352}, "17,16,16 (P3; 20,15,16 trades 3-hop +6% for 1 KiB 1-hop -3..-8%)"},
        {{"2d", 3584}, "20,21,16 (Galaxy 167723: every row at the wire)"},
        {{"2d", 3072}, "20,21,32 (Galaxy 167723)"},
        {{"2d", 2048}, "20,39,32 or 34,32,32 (Galaxy 167648/167673)"},
        {{"1d", 4352}, "26,25,32 (P3; 23,28,32 and 27,24,32 within 0.7)"},
        {{"ring", 4352}, "not window-bound (ALLOCATION_NOTE 4): receiver a power of 2"},
        {{"ring", 3072}, "not window-bound (ALLOCATION_NOTE 4): receiver a power of 2"},
        {{"vc1", 4352}, "19,9,16,5,4 (VC0 first, P1 jobs 376-388)"},
        {{"vc1", 3584}, "14,13,16,8,6 (2-hop at the wire; 256 B 1-hop +12%) or 20,11,16,8,6"},
        {{"vc1", 3072}, "20,15,16,9,7 (Galaxy 167723; 24,15,16,8,6: VC1 rows +2-5%)"},
        {{"vc1", 2048}, "32,18,32,12,10 (Galaxy 167648/167673)"},
        {{"vc1first", 4352}, "18,7,16,7,5 (VC1 to its bound, job 388)"},
        {{"vc2", 4352}, "17,14,16 + 2,4 (the study row; VC2 alone 869; 17,10,16 + 8,10: VC2 246)"},
        {{"vc2", 3584}, "18,11,16 + 16,16 (Galaxy 167723)"},
        {{"vc2", 3072}, "20,14,16 + 16,16 (Galaxy 167723: VC2 at the wire)"},
        {{"vc2", 2048}, "34,20,32 + 16,16 (Galaxy 167648/167673)"},
        {{"vc1vc2", 4352}, "19,8,16,4,4 + 2,4 (the study row, not measured against others)"},
    };
    // VC1 rows were measured on 2x2 splits only (SLOT_ALLOCATOR.md section 5.4).
    if (c.vc1 && (c.dim_y > 2 || c.dim_x > 2)) {
        return "- (VC1 measured on 2x2 splits only)";
    }
    std::string kind = (c.vc1 && c.vc2)                                          ? "vc1vc2"
                       : c.vc1                                                   ? (c.vc1_first ? "vc1first" : "vc1")
                       : c.vc2                                                   ? "vc2"
                       : (c.family == Family::Ring || c.family == Family::Torus) ? "ring"
                       : is_1d(c.family)                                         ? "1d"
                                                                                 : "2d";
    const auto it = best.find({kind, payload});
    return it == best.end() ? "-" : it->second;
}

Row from_allocation(const sa::Allocation& a) {
    using K = sa::ChannelKind;
    return Row{
        a.at(0, K::Worker).slots,
        a.at(0, K::Forwarding).slots,
        a.at(0, K::Receiver).slots,
        a.at(1, K::Forwarding).slots,
        a.at(1, K::Receiver).slots,
        a.at(2, K::Worker).slots,
        a.at(2, K::Receiver).slots};
}

// "G/ids@start" per receiver channel, "|"-separated; "!" marks a channel whose groups share IDs.
std::string show_ids(const ta::Assignment& a) {
    if (a.error) {
        return "ERROR: " + *a.error;
    }
    std::string s;
    for (const auto& c : a.channels) {
        s += (s.empty() ? "" : " | ") + std::to_string(c.group_size) + "/" + std::to_string(c.num_ids) + "@" +
             std::to_string(c.start) + (c.num_ids < c.groups ? "!" : "");
    }
    return s + " (" + std::to_string(a.used_ids) + " IDs)";
}

// The receiver channels of a router: VC0's (first-level acks where the router has deadlock avoidance), then VC1's and
// VC2's (no first-level acks).
std::vector<ta::ReceiverChannel> receivers(const Row& row, const Config& c, bool deadlock_avoidance) {
    std::vector<ta::ReceiverChannel> out = {{row.r, deadlock_avoidance}};
    if (c.vc1) {
        out.push_back({row.r1, false});
    }
    if (c.vc2) {
        out.push_back({row.r2, false});
    }
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    std::vector<std::size_t> sizes = {4352, 3072};
    std::string policy_text;
    std::string only;
    bool reasons = false;
    bool trids = false;
    for (int i = 1; i < argc; i++) {
        const std::string arg = argv[i];
        if (arg == "--reasons") {
            reasons = true;
        } else if (arg == "--trids") {
            trids = true;
        } else if (arg == "--sizes" && i + 1 < argc) {
            sizes.clear();
            std::string list = argv[++i];
            std::size_t start = 0;
            while (start <= list.size()) {
                std::size_t end = list.find(',', start);
                if (end == std::string::npos) {
                    end = list.size();
                }
                sizes.push_back(std::strtoul(list.substr(start, end - start).c_str(), nullptr, 10));
                start = end + 1;
            }
        } else if (arg == "--policy" && i + 1 < argc) {
            policy_text = argv[++i];
        } else if (arg == "--only" && i + 1 < argc) {
            only = argv[++i];
        } else {
            std::fprintf(
                stderr, "usage: %s [--sizes 4352,3072] [--policy k=v,...] [--reasons] [--only NAME]\n", argv[0]);
            return 1;
        }
    }
    sa::Policy base;
    if (const auto err = sa::parse_policy(policy_text, base)) {
        std::fprintf(stderr, "%s\n", err->c_str());
        return 1;
    }

    // Longest ring: the deadlock-avoidance axis lengths the host reads (Ring: every axis; Torus: the wrapped axes).
    const std::vector<Config> configs = {
        {"P150 2x4 mesh", Family::Mesh, 2, 4, 0, false, false, false},
        {"P150 1D line", Family::Linear, 2, 4, 0, false, false, false},
        {"P150 1D ring", Family::Ring, 2, 4, 4, false, false, false},
        {"P150 Y torus (2x4)", Family::Torus, 2, 4, 2, false, false, false},
        {"P150 XY torus (2x4: X ring)", Family::Torus, 2, 4, 4, false, false, false},
        {"Galaxy 8x4 mesh", Family::Mesh, 8, 4, 0, false, false, false},
        {"Galaxy 1D ring", Family::Ring, 8, 4, 8, false, false, false},
        {"Galaxy 8x4 XY torus", Family::Torus, 8, 4, 8, false, false, false},
        {"VC1 split (2x 2x2), VC0 first", Family::Mesh, 2, 2, 0, true, false, false},
        {"VC1 split (2x 2x2), VC1 first", Family::Mesh, 2, 2, 0, true, false, true},
        {"VC1 split (2x 4x4)", Family::Mesh, 4, 4, 0, true, false, false},
        {"VC2 on (P150 2x4, Galaxy 8x4)", Family::Mesh, 8, 4, 0, false, true, false},
        {"VC1 split + VC2 (three VCs)", Family::Mesh, 2, 2, 0, true, true, false},
        {"Multi-mesh XY torus (VC1, 4x4 meshes)", Family::Torus, 4, 4, 4, true, false, false},
    };

    if (trids) {
        // Unit checks of the rule: receivers (slots, first-level acks) -> IDs per channel. A failure stops the table.
        struct Check {
            const char* name;
            std::vector<ta::ReceiverChannel> rx;
            std::vector<std::size_t> want;
        };
        const std::vector<Check> checks = {
            {"2D lone 16", {{16, false}}, {4}},
            {"1D 32", {{32, false}}, {8}},
            {"ring 32a: 32 -> 16 (halve a power of 2)", {{32, true}}, {16}},
            {"ring 20a: 20 -> 16 (round down, not 8)", {{20, true}}, {16}},
            {"ring 24a: 24 -> 16", {{24, true}}, {16}},
            {"torus 16a + VC1 4: 16 -> 8", {{16, true}, {4, false}}, {8, 4}},
            {"20a + 4: 20 -> 16 -> 8", {{20, true}, {4, false}}, {8, 4}},
            {"12a + 8a: 12 -> 8", {{12, true}, {8, true}}, {8, 8}},
            {"three VCs 16 / 6 / 2: fits as is", {{16, false}, {6, false}, {2, false}}, {4, 6, 2}},
        };
        bool all_ok = true;
        std::printf("Unit checks (assign_transaction_ids; receivers as slots, a = first-level acks):\n\n");
        std::printf("| case | result | new |\n| --- | --- | --- |\n");
        for (const Check& k : checks) {
            const ta::Assignment got = ta::assign_transaction_ids(k.rx);
            std::vector<std::size_t> ids;
            for (const auto& c : got.channels) {
                ids.push_back(c.num_ids);
            }
            const bool ok = !got.error && ids == k.want;
            all_ok = all_ok && ok;
            std::printf("| %s | %s | %s |\n", k.name, ok ? "PASS" : "FAIL", show_ids(got).c_str());
        }
        std::printf("\n%s\n\n", all_ok ? "Unit checks: all pass." : "Unit checks: FAILED.");
        if (!all_ok) {
            return 1;
        }
        std::printf(
            "Receiver transaction IDs, per router: G/ids@start per receiver channel (VC0 | VC1 | VC2); ! = groups\n");
        std::printf(
            "share IDs. Uniform = today's rule; new = assign_transaction_ids. Slots: main's tier pick, dynamic.\n\n");
        std::printf("| configuration | payload | router | slots | receivers | uniform (today) | new |\n");
        std::printf("| --- | ---: | --- | --- | --- | --- | --- |\n");
        for (const Config& c : configs) {
            if (!only.empty() && c.name.find(only) == std::string::npos) {
                continue;
            }
            // Router kinds: deadlock avoidance (VC0 first-level acks) on a ring's routers, on a torus's wrapped axes.
            std::vector<std::pair<std::string, bool>> kinds;
            if (c.family == Family::Ring || (c.family == Family::Torus && c.name.find("XY") != std::string::npos)) {
                kinds = {{"deadlock avoidance", true}};
            } else if (c.family == Family::Torus) {
                kinds = {{"N/S (wrapped)", true}, {"E/W", false}};
            } else {
                kinds = {{"no deadlock avoidance", false}};
            }
            for (std::size_t payload : sizes) {
                sa::Policy policy = base;
                if (c.vc1_first) {
                    policy.vc_ratio = {2, 1};
                }
                policy.max_slots = MAX_SLOTS;
                const sa::Allocation a = sa::allocate_slots(shape_of(c, payload), policy);
                const std::vector<std::pair<std::string, Row>> arms = {
                    {"tier", main_tier(c, a.total_slots)}, {"dyn", from_allocation(a)}};
                for (const auto& [arm, row] : arms) {
                    for (const auto& [kind, da] : kinds) {
                        const auto rx = receivers(row, c, da);
                        std::string rx_text;
                        for (const auto& r : rx) {
                            rx_text += (rx_text.empty() ? "" : " | ") + std::to_string(r.slots) +
                                       (r.first_level_ack ? "a" : "");
                        }
                        std::printf(
                            "| %s | %zu | %s | %s | %s | %s | %s |\n",
                            c.name.c_str(),
                            payload,
                            kind.c_str(),
                            arm.c_str(),
                            rx_text.c_str(),
                            show_ids(ta::uniform_transaction_ids(rx, da)).c_str(),
                            show_ids(ta::assign_transaction_ids(rx)).c_str());
                        if (reasons) {
                            for (const auto& ch : ta::assign_transaction_ids(rx).channels) {
                                std::printf("|  |  |  |  |  |  | %s |\n", ch.reason.c_str());
                            }
                        }
                    }
                }
            }
        }
        return 0;
    }

    std::printf("policy: %s\n\n", policy_text.empty() ? "defaults" : policy_text.c_str());
    std::printf(
        "| configuration | payload | slots | main's tier pick | study router today | study best (measured) | dynamic "
        "|\n");
    std::printf("| --- | ---: | ---: | --- | --- | --- | --- |\n");
    std::vector<std::string> detail;
    for (const Config& c : configs) {
        if (!only.empty() && c.name.find(only) == std::string::npos) {
            continue;
        }
        for (std::size_t payload : sizes) {
            sa::Policy policy = base;
            if (c.vc1_first) {
                policy.vc_ratio = {2, 1};  // "VC1 first" is the ratio 2:1
            }
            policy.max_slots = MAX_SLOTS;
            const sa::RouterShape shape = shape_of(c, payload);
            const sa::Allocation a = sa::allocate_slots(shape, policy);
            const std::size_t n = a.total_slots;
            std::string dyn = a.error ? "ERROR: " + *a.error : show(from_allocation(a), c);
            if (a.unused_slots > 0) {
                dyn += " (" + std::to_string(a.unused_slots) + " unused)";
            }
            const std::string tier = show(main_tier(c, n), c);
            const std::string today = show(study_default(c, n), c);
            std::printf(
                "| %s | %zu | %zu | %s | %s | %s | **%s** |\n",
                c.name.c_str(),
                payload,
                n,
                tier.c_str(),
                today == tier ? "= main" : today.c_str(),
                study_best(c, payload).c_str(),
                dyn.c_str());
            if (reasons && !a.error) {
                std::string block = c.name + ", " + std::to_string(payload) + " B:\n";
                for (const std::string& note : a.notes) {
                    block += "    " + note + "\n";
                }
                for (std::size_t vc = 0; vc < sa::NUM_VCS; vc++) {
                    for (std::size_t k = 0; k < sa::NUM_CHANNEL_KINDS; k++) {
                        const sa::Count& count = a.counts[vc][k];
                        if (count.channels == 0) {
                            continue;
                        }
                        block += "    VC" + std::to_string(vc) + " " + sa::to_string(static_cast<sa::ChannelKind>(k)) +
                                 " x" + std::to_string(count.channels) + ": " + std::to_string(count.slots) + " (" +
                                 count.reason + ")\n";
                    }
                }
                detail.push_back(block);
            }
        }
    }
    if (reasons) {
        std::printf("\n");
        for (const std::string& block : detail) {
            std::printf("%s\n", block.c_str());
        }
    }
    return 0;
}
