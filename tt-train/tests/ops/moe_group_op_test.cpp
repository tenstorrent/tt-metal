// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <random>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>
#include <vector>

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"
#include "metal/ops/moe_group/moe_group.hpp"

namespace {

constexpr uint32_t kTileH = 32U;
constexpr uint32_t kSentinel = 0xFFFFFFFFU;
constexpr uint16_t kKSlotSentinel = 0xFFFFU;

class MoeGroupTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        ttml::autograd::ctx().open_device();
        ttml::autograd::ctx().set_seed(42);
    }
    static void TearDownTestSuite() {
        ttml::autograd::ctx().close_device();
    }
};

uint32_t round_up_to(uint32_t x, uint32_t a) {
    return ((x + a - 1U) / a) * a;
}

// Cursor alignment in element-count units — mirrors moe_group_device_operation.cpp.
uint32_t cursor_align_elems() {
    return tt::tt_metal::hal::get_l1_alignment() / sizeof(uint16_t);
}

uint32_t device_grid_cores() {
    auto& device = ttml::autograd::ctx().get_device();
    auto grid = device.compute_with_storage_grid_size();
    return static_cast<uint32_t>(grid.x) * static_cast<uint32_t>(grid.y);
}

// Mirrors moe_group_device_operation.cpp's t_cap formula.
uint32_t compute_t_cap(uint32_t e_local, uint32_t k, uint32_t d, uint32_t b, uint32_t s) {
    const uint32_t num_total_cores = device_grid_cores();
    const uint32_t cur_align = cursor_align_elems();
    const uint32_t unaligned =
        std::min(e_local, k) * d * b * s + e_local * (kTileH + (cur_align - 1U) * num_total_cores);
    return round_up_to(unaligned, kTileH);
}

// num_workers as picked by split_work_to_cores(grid, total_tiles).
uint32_t compute_num_workers(uint32_t e_local, uint32_t k, uint32_t d, uint32_t b, uint32_t s) {
    const uint32_t grid = device_grid_cores();
    const uint32_t t_cap = compute_t_cap(e_local, k, d, b, s);
    return std::min(t_cap / kTileH, grid);
}

struct GroupReference {
    std::vector<float> grouped;         // [T_cap * H], row-major
    std::vector<float> grouped_scores;  // [T_cap]
    std::vector<uint32_t> k_slot;       // [T_cap]
    std::vector<uint32_t> counts;       // [e_local]
    std::vector<uint32_t> offsets;      // [e_local + 1]
    std::vector<uint32_t> plan;         // [T_cap]
    uint32_t t_cap{};
    uint32_t h{};
};

// CPU reference matching moe_group: identical scan layout, per-core padding,
// 32-row tail rounding. `num_total_cores` is the host-side scan core count
// (= num_workers from split_work_to_cores).
GroupReference moe_group_reference(
    const xt::xarray<float>& dispatched,   // [D, B, S, H] (already bf16-roundtripped if needed)
    const xt::xarray<uint32_t>& metadata,  // [D, B, S, K]
    const xt::xarray<float>& scores,       // [D, B, S, K] (bf16-roundtripped)
    const std::vector<uint16_t>& local_expert_ids,
    uint32_t k,
    uint32_t num_total_cores,
    uint32_t t_cap_override) {
    const uint32_t D = static_cast<uint32_t>(dispatched.shape(0));
    const uint32_t B = static_cast<uint32_t>(dispatched.shape(1));
    const uint32_t S = static_cast<uint32_t>(dispatched.shape(2));
    const uint32_t H = static_cast<uint32_t>(dispatched.shape(3));
    const uint32_t E_local = static_cast<uint32_t>(local_expert_ids.size());
    const uint32_t total_rows = D * B * S;
    const uint32_t T_cap = t_cap_override > 0 ? t_cap_override : compute_t_cap(E_local, k, D, B, S);

    // Hit mask + k_slot per (expert, row).
    std::vector<uint8_t> hits(static_cast<size_t>(E_local) * total_rows, 0);
    std::vector<uint32_t> k_slots(static_cast<size_t>(E_local) * total_rows, kKSlotSentinel);
    std::vector<uint32_t> counts(E_local, 0);
    for (uint32_t e = 0; e < E_local; ++e) {
        const uint32_t leid = local_expert_ids[e];
        for (uint32_t t = 0; t < total_rows; ++t) {
            for (uint32_t ki = 0; ki < k; ++ki) {
                const uint32_t md = metadata.flat(t * k + ki);
                if (md == leid) {
                    hits[e * total_rows + t] = 1;
                    k_slots[e * total_rows + t] = ki;
                    ++counts[e];
                    break;
                }
            }
        }
    }

    const uint32_t slice_size = (total_rows + num_total_cores - 1U) / num_total_cores;
    const uint32_t cur_align = cursor_align_elems();

    // local_counts[c, e]
    std::vector<uint32_t> local_counts(num_total_cores * E_local, 0);
    for (uint32_t c = 0; c < num_total_cores; ++c) {
        const uint32_t s_start = c * slice_size;
        if (s_start >= total_rows) {
            continue;
        }
        const uint32_t s_end = std::min(s_start + slice_size, total_rows);
        for (uint32_t e = 0; e < E_local; ++e) {
            uint32_t cc = 0;
            for (uint32_t t = s_start; t < s_end; ++t) cc += hits[e * total_rows + t];
            local_counts[c * E_local + e] = cc;
        }
    }

    std::vector<uint32_t> offsets(E_local + 1, 0);
    for (uint32_t e = 0; e < E_local; ++e) {
        uint32_t running = offsets[e];
        for (uint32_t c = 0; c < num_total_cores; ++c) {
            running += round_up_to(local_counts[c * E_local + e], cur_align);
        }
        offsets[e + 1] = round_up_to(running, kTileH);
    }

    std::vector<uint32_t> plan(T_cap, kSentinel);
    std::vector<float> grouped_scores(T_cap, 0.0F);
    std::vector<uint32_t> k_slot_out(T_cap, kKSlotSentinel);
    std::vector<float> grouped(static_cast<size_t>(T_cap) * H, 0.0F);

    for (uint32_t e = 0; e < E_local; ++e) {
        uint32_t running = offsets[e];
        for (uint32_t c = 0; c < num_total_cores; ++c) {
            const uint32_t s_start = c * slice_size;
            if (s_start >= total_rows)
                continue;
            const uint32_t s_end = std::min(s_start + slice_size, total_rows);
            uint32_t n = 0;
            for (uint32_t t = s_start; t < s_end; ++t) {
                if (!hits[e * total_rows + t])
                    continue;
                plan[running + n] = t;
                k_slot_out[running + n] = k_slots[e * total_rows + t];
                const uint32_t ks = k_slots[e * total_rows + t];
                grouped_scores[running + n] = scores.flat(t * k + ks);
                for (uint32_t hh = 0; hh < H; ++hh) {
                    grouped[(running + n) * H + hh] = dispatched.flat(t * H + hh);
                }
                ++n;
            }
            running += round_up_to(n, cur_align);
        }
    }

    return GroupReference{
        std::move(grouped),
        std::move(grouped_scores),
        std::move(k_slot_out),
        std::move(counts),
        std::move(offsets),
        std::move(plan),
        T_cap,
        H};
}

// Input builders ------------------------------------------------------------

xt::xarray<float> make_dispatched(uint32_t D, uint32_t B, uint32_t S, uint32_t H, uint32_t seed = 0) {
    xt::xarray<float> out = xt::zeros<float>({D, B, S, H});
    // row (d,b,s) value = (d*B*S + b*S + s), broadcast across H. Round-trips
    // cleanly through bf16 for small grids (integers <= 2^7).
    for (uint32_t d = 0; d < D; ++d) {
        for (uint32_t b = 0; b < B; ++b) {
            for (uint32_t s = 0; s < S; ++s) {
                const float v = static_cast<float>(d * B * S + b * S + s);
                for (uint32_t h = 0; h < H; ++h) out(d, b, s, h) = v;
            }
        }
    }
    (void)seed;
    return out;
}

xt::xarray<uint32_t> make_metadata(uint32_t D, uint32_t B, uint32_t S, uint32_t K, uint32_t E, uint32_t seed = 0) {
    xt::xarray<uint32_t> out = xt::zeros<uint32_t>({D, B, S, K});
    std::mt19937 rng(seed + 1);
    std::vector<uint32_t> all(E);
    for (uint32_t e = 0; e < E; ++e) all[e] = e;
    for (uint32_t d = 0; d < D; ++d) {
        for (uint32_t b = 0; b < B; ++b) {
            for (uint32_t s = 0; s < S; ++s) {
                std::shuffle(all.begin(), all.end(), rng);
                for (uint32_t ki = 0; ki < K; ++ki) out(d, b, s, ki) = all[ki];
            }
        }
    }
    return out;
}

xt::xarray<float> make_scores(uint32_t D, uint32_t B, uint32_t S, uint32_t K, uint32_t seed = 0) {
    xt::xarray<float> out = xt::zeros<float>({D, B, S, K});
    std::mt19937 rng(seed + 13);
    std::uniform_real_distribution<float> dist(0.0F, 0.5F);
    for (auto it = out.begin(); it != out.end(); ++it) *it = dist(rng);
    // Round-trip through bf16 so the reference matches what the device sees.
    auto& dev = ttml::autograd::ctx().get_device();
    auto t = ttml::core::from_xtensor<float, ttnn::DataType::BFLOAT16>(out, &dev, ttnn::Layout::ROW_MAJOR);
    return ttml::core::to_xtensor(t);
}

// Run the device op + return host-side outputs as plain arrays.
struct DeviceOutputs {
    std::vector<float> grouped;         // [T_cap * H]
    std::vector<float> grouped_scores;  // [T_cap]
    std::vector<uint32_t> k_slot;       // [T_cap]
    std::vector<uint32_t> counts;       // [E_local]
    std::vector<uint32_t> offsets;      // [E_local + 1]
    std::vector<uint32_t> plan;         // [T_cap]
    uint32_t T_cap{};
    uint32_t H{};
};

DeviceOutputs run_op(
    const xt::xarray<float>& dispatched,
    const xt::xarray<uint32_t>& metadata,
    const xt::xarray<float>& scores,
    const std::vector<uint16_t>& local_expert_ids,
    uint32_t k) {
    auto& dev = ttml::autograd::ctx().get_device();
    const uint32_t E_local = static_cast<uint32_t>(local_expert_ids.size());
    const uint32_t H = static_cast<uint32_t>(dispatched.shape(3));

    auto disp_tt = ttml::core::from_xtensor<float, ttnn::DataType::BFLOAT16>(dispatched, &dev, ttnn::Layout::ROW_MAJOR);
    // metadata / leids as uint16 ROW_MAJOR
    xt::xarray<uint16_t> md16 = xt::cast<uint16_t>(metadata);
    auto md_tt = ttml::core::from_xtensor<uint16_t, ttnn::DataType::UINT16>(md16, &dev, ttnn::Layout::ROW_MAJOR);
    auto sc_tt = ttml::core::from_xtensor<float, ttnn::DataType::BFLOAT16>(scores, &dev, ttnn::Layout::ROW_MAJOR);
    xt::xarray<uint16_t> leids_arr = xt::adapt(local_expert_ids, std::vector<size_t>{local_expert_ids.size()});
    auto leids_tt =
        ttml::core::from_xtensor<uint16_t, ttnn::DataType::UINT16>(leids_arr, &dev, ttnn::Layout::ROW_MAJOR);

    auto [grouped, grouped_scores, k_slot, counts, offsets, plan] =
        ttml::metal::moe_group(disp_tt, md_tt, sc_tt, leids_tt, E_local, k);

    DeviceOutputs out;
    out.H = H;
    auto grouped_rm = ttnn::to_layout(grouped, ttnn::ROW_MAJOR_LAYOUT);
    auto grouped_xt = ttml::core::to_xtensor(grouped_rm);
    out.T_cap = static_cast<uint32_t>(grouped_xt.shape(2));
    out.grouped.assign(grouped_xt.begin(), grouped_xt.end());

    auto gs_xt = ttml::core::to_xtensor(grouped_scores);
    out.grouped_scores.assign(gs_xt.begin(), gs_xt.end());

    auto ks_xt = ttml::core::to_xtensor<uint16_t>(k_slot);
    out.k_slot.assign(ks_xt.begin(), ks_xt.end());

    auto cnts_xt = ttml::core::to_xtensor<uint32_t>(counts);
    out.counts.assign(cnts_xt.begin(), cnts_xt.end());

    auto offs_xt = ttml::core::to_xtensor<uint32_t>(offsets);
    out.offsets.assign(offs_xt.begin(), offs_xt.end());

    auto plan_xt = ttml::core::to_xtensor<uint32_t>(plan);
    out.plan.assign(plan_xt.begin(), plan_xt.end());

    return out;
}

void check_against_reference(
    const xt::xarray<float>& dispatched,
    const xt::xarray<uint32_t>& metadata,
    const xt::xarray<float>& scores,
    const std::vector<uint16_t>& local_expert_ids,
    uint32_t k) {
    const uint32_t D = static_cast<uint32_t>(dispatched.shape(0));
    const uint32_t B = static_cast<uint32_t>(dispatched.shape(1));
    const uint32_t S = static_cast<uint32_t>(dispatched.shape(2));
    const uint32_t E_local = static_cast<uint32_t>(local_expert_ids.size());

    // Round-trip dispatched through bf16 so the reference compares apples-to-apples.
    auto& dev = ttml::autograd::ctx().get_device();
    auto disp_rt = ttml::core::to_xtensor(
        ttml::core::from_xtensor<float, ttnn::DataType::BFLOAT16>(dispatched, &dev, ttnn::Layout::ROW_MAJOR));

    const uint32_t num_workers = compute_num_workers(E_local, k, D, B, S);
    const uint32_t t_cap = compute_t_cap(E_local, k, D, B, S);
    auto ref = moe_group_reference(disp_rt, metadata, scores, local_expert_ids, k, num_workers, t_cap);

    auto out = run_op(dispatched, metadata, scores, local_expert_ids, k);

    ASSERT_EQ(out.T_cap, ref.t_cap);
    ASSERT_EQ(out.counts.size(), ref.counts.size());
    for (size_t i = 0; i < ref.counts.size(); ++i) {
        EXPECT_EQ(out.counts[i], ref.counts[i]) << "counts[" << i << "]";
    }
    ASSERT_EQ(out.offsets.size(), ref.offsets.size());
    for (size_t i = 0; i < ref.offsets.size(); ++i) {
        EXPECT_EQ(out.offsets[i], ref.offsets[i]) << "offsets[" << i << "]";
    }
    ASSERT_EQ(out.plan.size(), ref.plan.size());
    for (size_t i = 0; i < ref.plan.size(); ++i) {
        EXPECT_EQ(out.plan[i], ref.plan[i]) << "plan[" << i << "]";
    }
    ASSERT_EQ(out.k_slot.size(), ref.k_slot.size());
    for (size_t i = 0; i < ref.k_slot.size(); ++i) {
        EXPECT_EQ(out.k_slot[i], ref.k_slot[i]) << "k_slot[" << i << "]";
    }
    ASSERT_EQ(out.grouped_scores.size(), ref.grouped_scores.size());
    for (size_t i = 0; i < ref.grouped_scores.size(); ++i) {
        EXPECT_NEAR(out.grouped_scores[i], ref.grouped_scores[i], 1e-3F) << "grouped_scores[" << i << "]";
    }
    // grouped: only ACTIVE rows (plan[i] != SENTINEL) are guaranteed equal to
    // dispatched[plan[i]]. Pad rows are not part of the op contract — the
    // ungroup op skips them via plan[i] == SENTINEL and weights them by 0
    // grouped_scores, so any value there is harmless.
    ASSERT_EQ(out.grouped.size(), ref.grouped.size());
    const uint32_t H = out.H;
    for (uint32_t i = 0; i < ref.t_cap; ++i) {
        const uint32_t src = ref.plan[i];
        if (src == kSentinel)
            continue;
        for (uint32_t hh = 0; hh < H; ++hh) {
            const float got = out.grouped[i * H + hh];
            const float exp = disp_rt.flat(src * H + hh);
            EXPECT_FLOAT_EQ(got, exp) << "grouped[" << i << ", " << hh << "] (plan=" << src << ")";
        }
    }
}

}  // namespace

TEST_F(MoeGroupTest, BasicShape) {
    constexpr uint32_t D = 2, B = 1, S = 32, H = 64;
    constexpr uint32_t E = 4, K = 2;
    const std::vector<uint16_t> leids = {0, 1};
    auto dispatched = make_dispatched(D, B, S, H);
    auto metadata = make_metadata(D, B, S, K, E);
    auto scores = make_scores(D, B, S, K);
    check_against_reference(dispatched, metadata, scores, leids, K);
}

TEST_F(MoeGroupTest, LargerH) {
    constexpr uint32_t D = 2, B = 1, S = 32, H = 256;
    constexpr uint32_t E = 4, K = 2;
    const std::vector<uint16_t> leids = {0, 1};
    auto dispatched = make_dispatched(D, B, S, H);
    auto metadata = make_metadata(D, B, S, K, E);
    auto scores = make_scores(D, B, S, K);
    check_against_reference(dispatched, metadata, scores, leids, K);
}

TEST_F(MoeGroupTest, NonTileAlignedH) {
    constexpr uint32_t D = 2, B = 1, S = 32, H = 80;
    constexpr uint32_t E = 4, K = 2;
    const std::vector<uint16_t> leids = {0, 1};
    auto dispatched = make_dispatched(D, B, S, H);
    auto metadata = make_metadata(D, B, S, K, E);
    auto scores = make_scores(D, B, S, K);
    check_against_reference(dispatched, metadata, scores, leids, K);
}

TEST_F(MoeGroupTest, ExpertZeroActive) {
    constexpr uint32_t D = 2, B = 1, S = 32, H = 64;
    constexpr uint32_t E = 4, K = 2;
    // local expert 5 is never present in metadata (E=4 → ids in [0,3]).
    const std::vector<uint16_t> leids = {0, 5};
    auto dispatched = make_dispatched(D, B, S, H);
    auto metadata = make_metadata(D, B, S, K, E);
    auto scores = make_scores(D, B, S, K);
    check_against_reference(dispatched, metadata, scores, leids, K);
}

TEST_F(MoeGroupTest, AllTokensActiveForAllExperts) {
    constexpr uint32_t D = 2, B = 1, S = 32, H = 64;
    constexpr uint32_t E = 2, K = 2;  // K == E → every token hits both
    const std::vector<uint16_t> leids = {0, 1};
    auto dispatched = make_dispatched(D, B, S, H);
    auto metadata = make_metadata(D, B, S, K, E);
    auto scores = make_scores(D, B, S, K);
    check_against_reference(dispatched, metadata, scores, leids, K);
}

TEST_F(MoeGroupTest, LargeELocal) {
    constexpr uint32_t D = 2, B = 1, S = 64, H = 64;
    constexpr uint32_t E = 64, K = 4;
    std::vector<uint16_t> leids;
    leids.reserve(32);
    for (uint16_t i = 0; i < 32; ++i) leids.push_back(i);
    auto dispatched = make_dispatched(D, B, S, H);
    auto metadata = make_metadata(D, B, S, K, E);
    auto scores = make_scores(D, B, S, K);
    check_against_reference(dispatched, metadata, scores, leids, K);
}
