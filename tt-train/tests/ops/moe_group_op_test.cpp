// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/tensor/spec/tensor_spec.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <vector>

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"
#include "metal/ops/moe_group/moe_group.hpp"
#include "moe_test_utils.hpp"

namespace {

constexpr uint32_t kTileH = 32U;
constexpr uint32_t kSentinel = 0xFFFFFFFFU;
constexpr uint16_t kKSlotSentinel = 0xFFFFU;

// Float-comparison tolerance for bf16-derived outputs. Single source of truth so
// rtol/atol stay consistent across all checks in this test file.
constexpr float kRtol = 1e-3F;
constexpr float kAtol = 1e-3F;

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
    return tt::round_up(unaligned, kTileH);
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
            running += tt::round_up(local_counts[c * E_local + e], cur_align);
        }
        offsets[e + 1] = tt::round_up(running, kTileH);
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
            running += tt::round_up(n, cur_align);
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

// Build host inputs (dispatched + scores already bf16-roundtripped on the test
// device, so the host reference compares apples-to-apples against device output).
ttml::test_utils::moe::MoeHostInputs make_inputs(
    uint32_t D, uint32_t B, uint32_t S, uint32_t H, uint32_t E, uint32_t K) {
    return ttml::test_utils::moe::make_moe_host_inputs({
        .D = D,
        .B = B,
        .S = S,
        .H = H,
        .E = E,
        .K = K,
        // Row-index broadcast keeps moe_group reorder checks (EXPECT_FLOAT_EQ on grouped rows)
        // trivially debuggable: out row N must equal the constant float(N).
        .dispatched_pattern = ttml::test_utils::moe::DispatchedPattern::RowIndexBroadcast,
        .roundtrip_device = &ttml::autograd::ctx().get_device(),
    });
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

DeviceOutputs run_op(const ttml::test_utils::moe::MoeDeviceInputs& dev_in, uint32_t e_local, uint32_t h, uint32_t k) {
    auto [grouped, grouped_scores, k_slot, counts, offsets, plan] = ttml::metal::moe_group(
        dev_in.dispatched_bf16, dev_in.metadata_u16, dev_in.scores_bf16, dev_in.leids_u16, e_local, k);

    DeviceOutputs out;
    out.H = h;
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

DeviceOutputs run_op(
    const ttml::test_utils::moe::MoeHostInputs& host, const std::vector<uint16_t>& local_expert_ids, uint32_t k) {
    auto& dev = ttml::autograd::ctx().get_device();
    const uint32_t E_local = static_cast<uint32_t>(local_expert_ids.size());
    const uint32_t H = static_cast<uint32_t>(host.dispatched.shape(3));

    auto dev_in = ttml::test_utils::moe::to_device_inputs(host, local_expert_ids, &dev);
    return run_op(dev_in, E_local, H, k);
}

void check_outputs_against_reference(
    const ttml::test_utils::moe::MoeHostInputs& host,
    const std::vector<uint16_t>& local_expert_ids,
    uint32_t k,
    const DeviceOutputs& out) {
    const uint32_t D = static_cast<uint32_t>(host.dispatched.shape(0));
    const uint32_t B = static_cast<uint32_t>(host.dispatched.shape(1));
    const uint32_t S = static_cast<uint32_t>(host.dispatched.shape(2));
    const uint32_t E_local = static_cast<uint32_t>(local_expert_ids.size());

    // host.dispatched is already bf16-roundtripped by make_inputs(), so it matches what the
    // device sees and can be fed straight into the reference.
    const auto& disp_rt = host.dispatched;

    const uint32_t num_workers = compute_num_workers(E_local, k, D, B, S);
    const uint32_t t_cap = compute_t_cap(E_local, k, D, B, S);
    auto ref = moe_group_reference(disp_rt, host.metadata, host.scores, local_expert_ids, k, num_workers, t_cap);

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
    {
        auto out_gs = xt::adapt(out.grouped_scores, std::vector<size_t>{out.grouped_scores.size()});
        auto ref_gs = xt::adapt(ref.grouped_scores, std::vector<size_t>{ref.grouped_scores.size()});
        EXPECT_TRUE(xt::allclose(out_gs, ref_gs, kRtol, kAtol))
            << "grouped_scores allclose failed (rtol=" << kRtol << " atol=" << kAtol << ")";
    }
    // grouped: only ACTIVE rows (plan[i] != SENTINEL) are guaranteed equal to
    // dispatched[plan[i]]. Pad rows are not part of the op contract — the
    // ungroup op skips them via plan[i] == SENTINEL and weights them by 0
    // grouped_scores, so any value there is harmless.
    ASSERT_EQ(out.grouped.size(), ref.grouped.size());
    const uint32_t H = out.H;
    // moe_group only reorders dispatched rows into the grouped layout — no math is
    // performed on the values — so each active grouped row must be bit-identical
    // to the bf16-roundtripped dispatched source. EXPECT_FLOAT_EQ is intentional.
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

void check_against_reference(
    const ttml::test_utils::moe::MoeHostInputs& host, const std::vector<uint16_t>& local_expert_ids, uint32_t k) {
    check_outputs_against_reference(host, local_expert_ids, k, run_op(host, local_expert_ids, k));
}

template <typename T>
ttnn::Tensor make_custom_row_major_tensor(
    const std::vector<T>& data, const ttnn::Tensor& canonical, uint32_t width_alignment) {
    const auto spec = tt::tt_metal::TensorSpec(
        canonical.logical_shape(),
        tt::tt_metal::TensorLayout(
            canonical.dtype(),
            tt::tt_metal::PageConfig(tt::tt_metal::Layout::ROW_MAJOR),
            ttnn::DRAM_MEMORY_CONFIG,
            tt::tt_metal::Alignment({width_alignment})));
    return ttnn::Tensor::from_vector(data, spec, &ttml::autograd::ctx().get_device());
}

}  // namespace

TEST_F(MoeGroupTest, BasicShape) {
    constexpr uint32_t D = 2, B = 1, S = 32, H = 64;
    constexpr uint32_t E = 4, K = 2;
    const std::vector<uint16_t> leids = {0, 1};
    check_against_reference(make_inputs(D, B, S, H, E, K), leids, K);
}

TEST_F(MoeGroupTest, LargerH) {
    constexpr uint32_t D = 2, B = 1, S = 32, H = 256;
    constexpr uint32_t E = 4, K = 2;
    const std::vector<uint16_t> leids = {0, 1};
    check_against_reference(make_inputs(D, B, S, H, E, K), leids, K);
}

TEST_F(MoeGroupTest, NonTileAlignedH) {
    constexpr uint32_t D = 2, B = 1, S = 32, H = 80;
    constexpr uint32_t E = 4, K = 2;
    const std::vector<uint16_t> leids = {0, 1};
    check_against_reference(make_inputs(D, B, S, H, E, K), leids, K);
}

TEST_F(MoeGroupTest, ExpertZeroActive) {
    constexpr uint32_t D = 2, B = 1, S = 32, H = 64;
    constexpr uint32_t E = 4, K = 2;
    // local expert 5 is never present in metadata (E=4 → ids in [0,3]).
    const std::vector<uint16_t> leids = {0, 5};
    check_against_reference(make_inputs(D, B, S, H, E, K), leids, K);
}

TEST_F(MoeGroupTest, AllTokensActiveForAllExperts) {
    constexpr uint32_t D = 2, B = 1, S = 32, H = 64;
    constexpr uint32_t E = 2, K = 2;  // K == E → every token hits both
    const std::vector<uint16_t> leids = {0, 1};
    check_against_reference(make_inputs(D, B, S, H, E, K), leids, K);
}

TEST_F(MoeGroupTest, LargeELocal) {
    constexpr uint32_t D = 2, B = 1, S = 64, H = 64;
    constexpr uint32_t E = 64, K = 4;
    std::vector<uint16_t> leids;
    leids.reserve(32);
    for (uint16_t i = 0; i < 32; ++i) leids.push_back(i);
    check_against_reference(make_inputs(D, B, S, H, E, K), leids, K);
}

TEST_F(MoeGroupTest, CustomInputPageGeometryIsSafeOnColdAndWarmRuns) {
    constexpr uint32_t D = 2, B = 1, S = 32, H = 64;
    constexpr uint32_t E = 4, K = 2;
    const std::vector<uint16_t> leids = {0, 1};
    const auto host = make_inputs(D, B, S, H, E, K);
    auto& device = ttml::autograd::ctx().get_device();
    const uint32_t dram_alignment = tt::tt_metal::hal::get_dram_alignment();

    const xt::xarray<uint16_t> metadata_u16 = xt::cast<uint16_t>(host.metadata);
    const std::vector<uint16_t> metadata_data(metadata_u16.begin(), metadata_u16.end());
    std::vector<::bfloat16> scores_data;
    scores_data.reserve(host.scores.size());
    for (const float value : host.scores) {
        scores_data.emplace_back(value);
    }

    enum class CustomInput { Metadata, Scores, LocalExpertIds };
    const auto make_inputs_with_custom_page = [&](CustomInput custom_input) {
        auto inputs = ttml::test_utils::moe::to_device_inputs(host, leids, &device);
        switch (custom_input) {
            case CustomInput::Metadata:
                inputs.metadata_u16 =
                    make_custom_row_major_tensor(metadata_data, inputs.metadata_u16, 4U * dram_alignment);
                break;
            case CustomInput::Scores:
                inputs.scores_bf16 = make_custom_row_major_tensor(scores_data, inputs.scores_bf16, 4U * dram_alignment);
                break;
            case CustomInput::LocalExpertIds:
                inputs.leids_u16 = make_custom_row_major_tensor(leids, inputs.leids_u16, 2048U);
                break;
        }
        return inputs;
    };

    device.enable_program_cache();
    for (const auto custom_input : {CustomInput::Metadata, CustomInput::Scores, CustomInput::LocalExpertIds}) {
        device.clear_program_cache();

        auto cold_inputs = make_inputs_with_custom_page(custom_input);
        const auto& cold_custom = custom_input == CustomInput::Metadata ? cold_inputs.metadata_u16
                                  : custom_input == CustomInput::Scores ? cold_inputs.scores_bf16
                                                                        : cold_inputs.leids_u16;
        const auto cold_page = tt::tt_metal::TensorAccessorArgs(cold_custom.buffer()).get_compile_time_args().at(1);
        ASSERT_GT(cold_page, dram_alignment);
        const auto entries_before_cold = device.num_program_cache_entries();
        auto cold_out = run_op(cold_inputs, static_cast<uint32_t>(leids.size()), H, K);
        const auto entries_after_cold = device.num_program_cache_entries();
        EXPECT_GT(entries_after_cold, entries_before_cold);
        check_outputs_against_reference(host, leids, K, cold_out);

        auto warm_inputs = make_inputs_with_custom_page(custom_input);
        const auto entries_before_warm = device.num_program_cache_entries();
        auto warm_out = run_op(warm_inputs, static_cast<uint32_t>(leids.size()), H, K);
        const auto entries_after_warm = device.num_program_cache_entries();
        EXPECT_EQ(entries_after_warm, entries_before_warm);
        check_outputs_against_reference(host, leids, K, warm_out);
    }
    device.clear_program_cache();
}
