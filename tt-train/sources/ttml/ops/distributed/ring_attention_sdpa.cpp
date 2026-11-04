// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ring_attention_sdpa.hpp"

#include <fmt/core.h>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <string>
#include <string_view>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-umd-workload/cluster.hpp>
#include <vector>

#include "autograd/auto_context.hpp"
#include "autograd/graph_utils.hpp"
#include "core/compute_kernel_config.hpp"
#include "core/random.hpp"
#include "core/tt_tensor_utils.hpp"
#include "metal/common/const_utils.hpp"
#include "metal/ops/ring_sdpa_bw/ring_sdpa_bw.hpp"
#include "metal/ops/ring_sdpa_fw/ring_sdpa_fw.hpp"
#include "ops/binary_ops.hpp"
#include "ops/distributed/comm_ops.hpp"
#include "ops/scaled_dot_product_attention.hpp"
#include "ttnn/operations/copy/typecast/typecast.hpp"
#include "ttnn/operations/creation/creation.hpp"
#include "ttnn/operations/data_movement/concat/concat.hpp"
#include "ttnn/operations/data_movement/copy/copy.hpp"
#include "ttnn/operations/data_movement/pad/pad.hpp"
#include "ttnn/operations/data_movement/slice/slice.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/binary/binary_composite.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"
#include "ttnn_fixed/distributed/tt_metal.hpp"
#include "ttnn_fixed/distributed/ttnn_ops.hpp"

namespace ttml::ops::distributed {

namespace {

// Pads a (B, H, S, 1) FP32 logsumexp tensor into the (B, H, S, 32) intermediates layout
// the SDPA kernels expect: lse in column 0, the remaining 31 columns are ignored padding.
ttnn::Tensor pad_lse_to_intermediates_layout(const ttnn::Tensor& lse) {
    const ttsl::SmallVector<ttnn::operations::data_movement::PadSpecDim> padding = {
        {0, 0},  // batch
        {0, 0},  // heads
        {0, 0},  // seq_len
        {0, 31}  // width: pad 31 zeros on the right (1 -> 32)
    };
    return ttnn::pad(lse, padding, 0.0F, false, std::nullopt);
}

// Diagnostic per-phase timing of the composite ring attention, enabled with TTML_RING_ATTN_TIMING=1.
// Each lap() synchronizes the mesh so the elapsed wall time is attributed to the ops enqueued since the
// previous lap. The extra synchronizes perturb the timeline slightly; the composite already synchronizes
// around every ring_shift, so the attribution is close to what an untimed run executes.
class RingAttnPhaseTimer {
public:
    explicit RingAttnPhaseTimer(tt::tt_metal::distributed::MeshDevice* device) : m_device(device) {
        if (enabled()) {
            sync();
            m_start = m_last = std::chrono::steady_clock::now();
        }
    }

    static bool enabled() {
        static const bool on = [] {
            const char* env = std::getenv("TTML_RING_ATTN_TIMING");
            return env != nullptr && std::string_view(env) != "0";
        }();
        return on;
    }

    // Returns microseconds since the previous lap (0 when disabled).
    double lap() {
        if (!enabled()) {
            return 0.0;
        }
        sync();
        const auto now = std::chrono::steady_clock::now();
        const double us = std::chrono::duration<double, std::micro>(now - m_last).count();
        m_last = now;
        return us;
    }

    double total_us() const {
        return std::chrono::duration<double, std::micro>(m_last - m_start).count();
    }

private:
    void sync() {
        tt::tt_metal::distributed::Synchronize(*m_device, std::nullopt, std::vector<tt::tt_metal::SubDeviceId>());
    }

    tt::tt_metal::distributed::MeshDevice* m_device;
    std::chrono::steady_clock::time_point m_start{};
    std::chrono::steady_clock::time_point m_last{};
};

std::string format_us_list(const std::vector<double>& values) {
    std::string out = "[";
    for (size_t i = 0; i < values.size(); ++i) {
        out += fmt::format("{}{:.0f}", i == 0 ? "" : ",", values[i]);
    }
    return out + "]";
}

// Rows [begin, end) of a (B, H, S, X) tensor along the sequence dimension.
ttnn::Tensor slice_seq(const ttnn::Tensor& tensor, uint32_t begin, uint32_t end) {
    const auto [batch, heads, seq_len, width] = tensor.logical_shape().to_array_4D();
    TT_FATAL(begin < end && end <= seq_len, "Invalid sequence slice [{}, {}) of length {}", begin, end, seq_len);
    const ttsl::SmallVector<uint32_t> start = {0, 0, begin, 0};
    const ttsl::SmallVector<uint32_t> stop = {batch, heads, end, width};
    const ttsl::SmallVector<uint32_t> step = {1, 1, 1, 1};
    return ttnn::slice(tensor, start, stop, step);
}

ttnn::Tensor concat_seq(const ttnn::Tensor& first, const ttnn::Tensor& second) {
    return ttnn::concat(std::vector<ttnn::Tensor>{first, second}, /*dim=*/2);
}

// Online-softmax merge of one partial attention result into an FP32 accumulator.
// step_intermediates is the (B, H, S, 32) FP32 kernel intermediates (logsumexp in column 0);
// a logsumexp of -inf means "no contribution" and leaves the accumulator unchanged as long
// as the running logsumexp is already finite. The step output is upcast so the whole
// combine stays FP32; mixed-dtype binary ops would otherwise round the products back to bf16.
void merge_partial_into(
    ttnn::Tensor& output_accum,
    ttnn::Tensor& running_lse,
    const ttnn::Tensor& step_output,
    const ttnn::Tensor& step_intermediates) {
    const auto [batch, heads, seq_len, width] = step_intermediates.logical_shape().to_array_4D();
    const ttsl::SmallVector<uint32_t> slice_step = {1, 1, 1, 1};
    const ttsl::SmallVector<uint32_t> lse_start = {0, 0, 0, 0};
    const ttsl::SmallVector<uint32_t> lse_end = {batch, heads, seq_len, 1};
    ttnn::Tensor lse_chunk = ttnn::slice(step_intermediates, lse_start, lse_end, slice_step);

    // Combine via logaddexp: new_lse = log(exp(running_lse) + exp(lse_chunk))
    // Numerically stable form: m = max(a,b); result = m + log(exp(a-m) + exp(b-m))
    ttnn::Tensor m = ttnn::maximum(running_lse, lse_chunk);
    ttnn::Tensor exp_running = ttnn::exp(ttnn::subtract(running_lse, m));
    ttnn::Tensor exp_chunk = ttnn::exp(ttnn::subtract(lse_chunk, m));
    ttnn::Tensor new_lse = ttnn::add(m, ttnn::log(ttnn::add(exp_running, exp_chunk)));

    // Weights for combining outputs: w = exp(lse - new_lse) = Z_i / Z_combined
    ttnn::Tensor old_weight = ttnn::exp(ttnn::subtract(running_lse, new_lse));
    ttnn::Tensor new_weight = ttnn::exp(ttnn::subtract(lse_chunk, new_lse));

    ttnn::Tensor step_output_fp32 = ttnn::typecast(step_output, ttnn::DataType::FLOAT32);
    output_accum = ttnn::add(ttnn::multiply(output_accum, old_weight), ttnn::multiply(step_output_fp32, new_weight));
    running_lse = new_lse;
}

ttnn::Tensor to_fp32(const ttnn::Tensor& tensor) {
    return ttnn::typecast(tensor, ttnn::DataType::FLOAT32);
}

// Zig-zag (load-balanced causal) ring attention. Rank d holds sequence chunks [d | 2N-1-d]
// (C = S_local / 2 tokens each); Q0/Q1, K0/K1, V0/V1 denote the two local halves. With
// src the rank whose K/V this rank holds at a ring step:
//   step 0 (src == d): one causal call over the full local sequence. The local order is
//                      monotone in global position, so the standard causal mask is exact.
//   step s >= 1:       X = (Q1, K0) is always fully visible (chunk 2N-1-d > chunk src) and runs
//                      unmasked on every rank; the one visible same-half block runs unmasked via
//                      zig-zag lane selection: (Q0, K0) when src < d, (Q1, K1) when src > d.
//                      (Q0, K1) is never visible.
// Every rank therefore does two C x C blocks per step, instead of the last rank of the
// contiguous layout doing a full S_local x S_local block every step.
autograd::TensorPtr ring_attention_sdpa_zigzag(
    const autograd::TensorPtr& query,
    const autograd::TensorPtr& key,
    const autograd::TensorPtr& value,
    uint32_t ring_size,
    uint32_t cp_axis_value,
    tt::tt_metal::distributed::MeshDevice* mesh_device) {
    using ttml::metal::AttentionMaskType;
    using ttml::metal::ops::RingLaneSelection;
    constexpr auto kRingDirection = ttml::metal::ops::ring_sdpa_fw::RingDirection::Backward;
    constexpr float kNegInf = -std::numeric_limits<float>::infinity();

    const auto& query_tensor = query->get_value();
    const auto [batch_num, heads, seq_len_local, dim] = query_tensor.logical_shape().to_array_4D();
    TT_FATAL(
        seq_len_local % (2U * tt::constants::TILE_HEIGHT) == 0U,
        "Zig-zag ring attention needs the local sequence length ({}) to split into two tile-aligned halves (a "
        "multiple of {})",
        seq_len_local,
        2U * tt::constants::TILE_HEIGHT);
    const uint32_t half = seq_len_local / 2U;

    RingAttnPhaseTimer fw_timer(mesh_device);
    std::vector<double> fw_sdpa_us;
    std::vector<double> fw_merge_us;
    std::vector<double> fw_shift_us;

    ttnn::Tensor k_current = key->get_value();
    ttnn::Tensor v_current = value->get_value();
    const ttnn::Tensor q_first = slice_seq(query_tensor, 0U, half);
    const ttnn::Tensor q_second = slice_seq(query_tensor, half, seq_len_local);

    const auto dram = ttnn::MemoryConfig(tt::tt_metal::TensorMemoryLayout::INTERLEAVED, ttnn::BufferType::DRAM);
    auto make_intermediates = [&](uint32_t seq_len) {
        return ttnn::empty(
            ttnn::Shape{batch_num, heads, seq_len, 32U},
            ttnn::DataType::FLOAT32,
            ttnn::Layout::TILE,
            mesh_device,
            dram);
    };
    const double fw_setup_us = fw_timer.lap();

    // Step 0: every rank holds its own K/V; one causal call over the full local sequence.
    ttnn::Tensor full_output = ttnn::empty_like(query_tensor);
    ttnn::Tensor full_intermediates = make_intermediates(seq_len_local);
    ttml::metal::ring_sdpa_fw(
        query_tensor,
        k_current,
        v_current,
        ring_size,
        cp_axis_value,
        /*step=*/0U,
        AttentionMaskType::Causal,
        kRingDirection,
        full_output,
        full_intermediates);
    fw_sdpa_us.push_back(fw_timer.lap());

    // FP32 accumulators and running logsumexp per Q half, seeded from step 0.
    const ttnn::Tensor full_output_fp32 = to_fp32(full_output);
    const ttsl::SmallVector<uint32_t> lse_start = {0, 0, 0, 0};
    const ttsl::SmallVector<uint32_t> lse_end = {batch_num, heads, seq_len_local, 1};
    const ttsl::SmallVector<uint32_t> lse_step = {1, 1, 1, 1};
    const ttnn::Tensor full_lse = ttnn::slice(full_intermediates, lse_start, lse_end, lse_step);
    ttnn::Tensor accum_first = slice_seq(full_output_fp32, 0U, half);
    ttnn::Tensor accum_second = slice_seq(full_output_fp32, half, seq_len_local);
    ttnn::Tensor lse_first = slice_seq(full_lse, 0U, half);
    ttnn::Tensor lse_second = slice_seq(full_lse, half, seq_len_local);
    fw_merge_us.push_back(fw_timer.lap());

    // Half-size step buffers, reused every step. Lane outputs start zeroed: a rank that does
    // not compute a lane at a step leaves that lane's buffer as is, and the merge multiplies
    // it by exp(-inf) = 0, which needs a finite value there.
    ttnn::Tensor cross_output = ttnn::empty_like(q_first);
    ttnn::Tensor cross_intermediates = make_intermediates(half);
    ttnn::Tensor lane_output[2] = {ttnn::zeros_like(q_first), ttnn::zeros_like(q_first)};
    ttnn::Tensor lane_intermediates[2] = {make_intermediates(half), make_intermediates(half)};
    const ttnn::Tensor no_contrib_intermediates = pad_lse_to_intermediates_layout(ttnn::full(
        ttnn::Shape{batch_num, heads, half, 1U},
        kNegInf,
        ttnn::DataType::FLOAT32,
        ttnn::Layout::TILE,
        std::ref(*mesh_device)));

    for (uint32_t step = 1; step < ring_size; ++step) {
        k_current = ttnn_fixed::distributed::ring_shift(
            k_current, cp_axis_value, ttnn_fixed::distributed::RingShiftDirection::Backward);
        v_current = ttnn_fixed::distributed::ring_shift(
            v_current, cp_axis_value, ttnn_fixed::distributed::RingShiftDirection::Backward);
        fw_shift_us.push_back(fw_timer.lap());

        const ttnn::Tensor k_first = slice_seq(k_current, 0U, half);
        const ttnn::Tensor k_second = slice_seq(k_current, half, seq_len_local);
        const ttnn::Tensor v_first = slice_seq(v_current, 0U, half);
        const ttnn::Tensor v_second = slice_seq(v_current, half, seq_len_local);

        // Cross block (Q1, K0): fully visible on every rank.
        ttml::metal::ring_sdpa_fw(
            q_second,
            k_first,
            v_first,
            ring_size,
            cp_axis_value,
            step,
            AttentionMaskType::None,
            kRingDirection,
            cross_output,
            cross_intermediates);

        // Same-half block: lane 0 (Q0, K0) where src < d, lane 1 (Q1, K1) where src > d.
        ttnn::copy(no_contrib_intermediates, lane_intermediates[0]);
        ttnn::copy(no_contrib_intermediates, lane_intermediates[1]);
        ttml::metal::ring_sdpa_fw(
            q_first,
            k_first,
            v_first,
            ring_size,
            cp_axis_value,
            step,
            AttentionMaskType::Causal,
            kRingDirection,
            lane_output[0],
            lane_intermediates[0],
            RingLaneSelection::ZigZag,
            ttml::metal::ops::ring_sdpa_fw::RingSDPAFwLane{
                .query = q_second,
                .key = k_second,
                .value = v_second,
                .output = lane_output[1],
                .intermediates = lane_intermediates[1]});
        fw_sdpa_us.push_back(fw_timer.lap());

        merge_partial_into(accum_second, lse_second, cross_output, cross_intermediates);
        merge_partial_into(accum_first, lse_first, lane_output[0], lane_intermediates[0]);
        merge_partial_into(accum_second, lse_second, lane_output[1], lane_intermediates[1]);
        fw_merge_us.push_back(fw_timer.lap());
    }

    auto out = autograd::create_tensor(ttnn::typecast(concat_seq(accum_first, accum_second), query_tensor.dtype()));
    const ttnn::Tensor final_lse = concat_seq(lse_first, lse_second);
    if (RingAttnPhaseTimer::enabled()) {
        const double fw_final_us = fw_timer.lap();
        fmt::print(
            "[RING_ATTN_TIMING] pass=fw layout=zigzag S_local={} ring={} total_us={:.0f} setup_us={:.0f} "
            "final_us={:.0f} sdpa_us={} merge_us={} shift_us={}\n",
            seq_len_local,
            ring_size,
            fw_timer.total_us(),
            fw_setup_us,
            fw_final_us,
            format_us_list(fw_sdpa_us),
            format_us_list(fw_merge_us),
            format_us_list(fw_shift_us));
    }

    autograd::GradFunction grad_fn = [query,
                                      key,
                                      value,
                                      out,
                                      final_lse,
                                      k_current,  // K at the end of the forward ring
                                      v_current,  // V at the end of the forward ring
                                      ring_size,
                                      cp_axis_value,
                                      half,
                                      mesh_device]() mutable {
        tt::tt_metal::distributed::Synchronize(*mesh_device, std::nullopt, std::vector<tt::tt_metal::SubDeviceId>());
        const ttnn::Tensor& grad_output = out->get_grad();
        const ttnn::Tensor& attn_output = out->get_value();
        const ttnn::Tensor& query_tensor = query->get_value();
        const uint32_t seq_len_local = query_tensor.logical_shape()[2];
        RingAttnPhaseTimer bw_timer(mesh_device);
        std::vector<double> bw_sdpa_us;
        std::vector<double> bw_accum_us;
        std::vector<double> bw_shift_us;

        // Global O / logsumexp (see the contiguous backward for why every block must use them),
        // split into the two Q halves.
        const ttnn::Tensor global_intermediates = pad_lse_to_intermediates_layout(final_lse);
        const ttnn::Tensor q_first = slice_seq(query_tensor, 0U, half);
        const ttnn::Tensor q_second = slice_seq(query_tensor, half, seq_len_local);
        const ttnn::Tensor o_first = slice_seq(attn_output, 0U, half);
        const ttnn::Tensor o_second = slice_seq(attn_output, half, seq_len_local);
        const ttnn::Tensor do_first = slice_seq(grad_output, 0U, half);
        const ttnn::Tensor do_second = slice_seq(grad_output, half, seq_len_local);
        const ttnn::Tensor inter_first = slice_seq(global_intermediates, 0U, half);
        const ttnn::Tensor inter_second = slice_seq(global_intermediates, half, seq_len_local);

        ttnn::Tensor grad_Q_accum = ttnn::zeros_like(query_tensor, ttnn::DataType::FLOAT32);
        ttnn::Tensor grad_K_accum = ttnn::zeros_like(key->get_value(), ttnn::DataType::FLOAT32);
        ttnn::Tensor grad_V_accum = ttnn::zeros_like(value->get_value(), ttnn::DataType::FLOAT32);

        // Step buffers. The step-0 call and the cross block run on every rank; the lane
        // buffers are zeroed before every step because each rank fills only one lane.
        ttnn::Tensor grad_Q_full = ttnn::zeros_like(query_tensor);
        ttnn::Tensor grad_K_full = ttnn::zeros_like(key->get_value());
        ttnn::Tensor grad_V_full = ttnn::zeros_like(value->get_value());
        const ttnn::Tensor k_half_like = slice_seq(key->get_value(), 0U, half);
        const ttnn::Tensor v_half_like = slice_seq(value->get_value(), 0U, half);
        ttnn::Tensor grad_Q_cross = ttnn::zeros_like(q_first);
        ttnn::Tensor grad_K_cross = ttnn::zeros_like(k_half_like);
        ttnn::Tensor grad_V_cross = ttnn::zeros_like(v_half_like);
        ttnn::Tensor grad_Q_lane[2] = {ttnn::zeros_like(q_first), ttnn::zeros_like(q_first)};
        ttnn::Tensor grad_K_lane[2] = {ttnn::zeros_like(k_half_like), ttnn::zeros_like(k_half_like)};
        ttnn::Tensor grad_V_lane[2] = {ttnn::zeros_like(v_half_like), ttnn::zeros_like(v_half_like)};
        const ttnn::Tensor zero_Q_half = ttnn::zeros_like(q_first);
        const ttnn::Tensor zero_K_half = ttnn::zeros_like(k_half_like);
        const ttnn::Tensor zero_V_half = ttnn::zeros_like(v_half_like);

        const double bw_setup_us = bw_timer.lap();
        // Reverse ring, as in the contiguous backward: K/V and the travelling dK/dV
        // accumulators shift in the Forward direction; the schedule uses the same
        // src = (d + step) mod N as the forward.
        for (int step = static_cast<int>(ring_size) - 1; step >= 0; --step) {
            const auto step_idx = static_cast<uint32_t>(step);
            if (step_idx == 0U) {
                ttml::metal::ring_sdpa_bw(
                    grad_output,
                    attn_output,
                    query_tensor,
                    k_current,
                    v_current,
                    global_intermediates,
                    ring_size,
                    cp_axis_value,
                    0U,
                    AttentionMaskType::Causal,
                    ttml::metal::ops::ring_sdpa_bw::RingDirection::Backward,
                    grad_Q_full,
                    grad_K_full,
                    grad_V_full);
                bw_sdpa_us.push_back(bw_timer.lap());
                grad_Q_accum = ttnn::add(grad_Q_accum, to_fp32(grad_Q_full));
                grad_K_accum = ttnn::add(grad_K_accum, to_fp32(grad_K_full));
                grad_V_accum = ttnn::add(grad_V_accum, to_fp32(grad_V_full));
                bw_accum_us.push_back(bw_timer.lap());
                continue;
            }

            const ttnn::Tensor k_first = slice_seq(k_current, 0U, half);
            const ttnn::Tensor k_second = slice_seq(k_current, half, seq_len_local);
            const ttnn::Tensor v_first = slice_seq(v_current, 0U, half);
            const ttnn::Tensor v_second = slice_seq(v_current, half, seq_len_local);

            // Cross block (Q1, K0) on every rank.
            ttml::metal::ring_sdpa_bw(
                do_second,
                o_second,
                q_second,
                k_first,
                v_first,
                inter_second,
                ring_size,
                cp_axis_value,
                step_idx,
                AttentionMaskType::None,
                ttml::metal::ops::ring_sdpa_bw::RingDirection::Backward,
                grad_Q_cross,
                grad_K_cross,
                grad_V_cross);

            // Same-half block by lane: (Q0, K0) where src < d, (Q1, K1) where src > d.
            for (uint32_t lane = 0; lane < 2U; ++lane) {
                ttnn::copy(zero_Q_half, grad_Q_lane[lane]);
                ttnn::copy(zero_K_half, grad_K_lane[lane]);
                ttnn::copy(zero_V_half, grad_V_lane[lane]);
            }
            ttml::metal::ring_sdpa_bw(
                do_first,
                o_first,
                q_first,
                k_first,
                v_first,
                inter_first,
                ring_size,
                cp_axis_value,
                step_idx,
                AttentionMaskType::Causal,
                ttml::metal::ops::ring_sdpa_bw::RingDirection::Backward,
                grad_Q_lane[0],
                grad_K_lane[0],
                grad_V_lane[0],
                RingLaneSelection::ZigZag,
                ttml::metal::RingSDPABwLane{
                    .grad_output = do_second,
                    .attn_output = o_second,
                    .query = q_second,
                    .key = k_second,
                    .value = v_second,
                    .intermediates = inter_second,
                    .grad_query = grad_Q_lane[1],
                    .grad_key = grad_K_lane[1],
                    .grad_value = grad_V_lane[1]});
            bw_sdpa_us.push_back(bw_timer.lap());

            // dQ: first half from lane 0; second half from the cross block and lane 1.
            // dK/dV of this step's K/V: first half from the cross block and lane 0, second half
            // from lane 1. Reassembled to full size so the travelling accumulators shift as one.
            grad_Q_accum = ttnn::add(
                grad_Q_accum,
                concat_seq(to_fp32(grad_Q_lane[0]), ttnn::add(to_fp32(grad_Q_cross), to_fp32(grad_Q_lane[1]))));
            grad_K_accum = ttnn::add(
                grad_K_accum,
                concat_seq(ttnn::add(to_fp32(grad_K_cross), to_fp32(grad_K_lane[0])), to_fp32(grad_K_lane[1])));
            grad_V_accum = ttnn::add(
                grad_V_accum,
                concat_seq(ttnn::add(to_fp32(grad_V_cross), to_fp32(grad_V_lane[0])), to_fp32(grad_V_lane[1])));
            bw_accum_us.push_back(bw_timer.lap());

            k_current = ttnn_fixed::distributed::ring_shift(
                k_current, cp_axis_value, ttnn_fixed::distributed::RingShiftDirection::Forward);
            v_current = ttnn_fixed::distributed::ring_shift(
                v_current, cp_axis_value, ttnn_fixed::distributed::RingShiftDirection::Forward);
            grad_K_accum = ttnn_fixed::distributed::ring_shift(
                grad_K_accum, cp_axis_value, ttnn_fixed::distributed::RingShiftDirection::Forward);
            grad_V_accum = ttnn_fixed::distributed::ring_shift(
                grad_V_accum, cp_axis_value, ttnn_fixed::distributed::RingShiftDirection::Forward);
            bw_shift_us.push_back(bw_timer.lap());
        }

        query->add_grad(ttnn::typecast(grad_Q_accum, query_tensor.dtype()));
        key->add_grad(ttnn::typecast(grad_K_accum, key->get_value().dtype()));
        value->add_grad(ttnn::typecast(grad_V_accum, value->get_value().dtype()));
        if (RingAttnPhaseTimer::enabled()) {
            const double bw_final_us = bw_timer.lap();
            fmt::print(
                "[RING_ATTN_TIMING] pass=bw layout=zigzag S_local={} ring={} total_us={:.0f} setup_us={:.0f} "
                "final_us={:.0f} sdpa_us={} accum_us={} shift_us={}\n",
                seq_len_local,
                ring_size,
                bw_timer.total_us(),
                bw_setup_us,
                bw_final_us,
                format_us_list(bw_sdpa_us),
                format_us_list(bw_accum_us),
                format_us_list(bw_shift_us));
        }
    };

    out->set_node(autograd::add_backward_node(std::move(grad_fn), out, query, key, value));
    return out;
}

}  // namespace

autograd::TensorPtr ring_attention_sdpa(
    const autograd::TensorPtr& query,
    const autograd::TensorPtr& key,
    const autograd::TensorPtr& value,
    const std::optional<autograd::TensorPtr>& mask,
    const ttml::metal::AttentionMaskType mask_type,
    RingAttentionLayout layout) {
    if (!autograd::ctx().is_parallelism_context_initialized() ||
        !autograd::ctx().get_parallelism_context().is_cp_enabled()) {
        return ttml::ops::scaled_dot_product_attention(query, key, value, mask);
    }

    const auto& pctx = autograd::ctx().get_parallelism_context();
    const uint32_t cp_axis_value = pctx.get_cp_axis().value();
    const uint32_t ring_size = pctx.get_cp_size();
    const auto& query_tensor = query->get_value();
    auto* mesh_device = query_tensor.device();
    TT_FATAL(mesh_device != nullptr, "Query tensor must be on a mesh device for ring attention");
    TT_FATAL(
        !mask.has_value(),
        "Non-causal mask is not supported in CP mode for now, pass nullopt if you want to use causal mask");
    TT_FATAL(
        mask_type != ttml::metal::AttentionMaskType::Arbitrary,
        "Arbitrary attention mask is not supported in CP mode, use None or Causal");
    tt::tt_metal::distributed::Synchronize(*mesh_device, std::nullopt, std::vector<tt::tt_metal::SubDeviceId>());

    if (layout == RingAttentionLayout::ZigZag && ring_size > 1U) {
        TT_FATAL(
            mask_type == ttml::metal::AttentionMaskType::Causal,
            "Zig-zag ring attention load-balances causal attention; use the contiguous layout without a causal mask");
        return ring_attention_sdpa_zigzag(query, key, value, ring_size, cp_axis_value, mesh_device);
    }

    auto [batch_num, heads, seq_len_local, dim] = query_tensor.logical_shape().to_array_4D();
    RingAttnPhaseTimer fw_timer(mesh_device);
    std::vector<double> fw_sdpa_us;
    std::vector<double> fw_merge_us;
    std::vector<double> fw_shift_us;
    // Initialize current K and V (will be ring-shifted each step)
    // Use raw ttnn::Tensor since we define backward manually
    ttnn::Tensor k_current = key->get_value();
    ttnn::Tensor v_current = value->get_value();

    // Initialize accumulators for online softmax.
    // output_accum: weighted sum of outputs from all steps. Kept FP32: the online-softmax
    // rescaling rewrites the whole accumulator every ring step, so a bf16 accumulator
    // compounds ~ring_size rounding errors into the saved O. Backward consumes that O in
    // u = rowsum(dO * O), where the error is amplified by the softmax-backward
    // cancellation in (dP - u) — enough to push dK outside test tolerance.
    ttnn::Tensor output_accum = ttnn::full(
        ttnn::Shape{batch_num, heads, seq_len_local, dim},
        0.0F,
        ttnn::DataType::FLOAT32,
        ttnn::Layout::TILE,
        std::ref(*mesh_device));

    // global_lse: running logsumexp across all steps
    // lse = log(sum(exp(scale * score_i))) — the log of the softmax normalizer
    // Initialized to -inf (no contribution: exp(-inf) = 0)
    ttnn::Tensor global_lse = ttnn::full(
        ttnn::Shape{batch_num, heads, seq_len_local, 1U},
        -std::numeric_limits<float>::infinity(),
        ttnn::DataType::FLOAT32,
        ttnn::Layout::TILE,
        std::ref(*mesh_device));

    // Allocate output and intermediate tensors (mesh tensors)
    // These will be reused each step
    ttnn::Tensor output_tensor = ttnn::empty_like(query_tensor);
    ttnn::Tensor intermediate_tensor = ttnn::empty(
        ttnn::Shape{batch_num, heads, seq_len_local, 32U},
        ttnn::DataType::FLOAT32,
        ttnn::Layout::TILE,
        mesh_device,
        ttnn::MemoryConfig(tt::tt_metal::TensorMemoryLayout::INTERLEAVED, ttnn::BufferType::DRAM));

    // "no contribution" intermediate: logsumexp = -inf (col 0), rest zeros
    // exp(-inf) = 0, so this chunk contributes nothing to the combined softmax
    ttnn::Tensor col0_neg_inf = ttnn::full(
        ttnn::Shape{batch_num, heads, seq_len_local, 1U},
        -std::numeric_limits<float>::infinity(),
        ttnn::DataType::FLOAT32,
        ttnn::Layout::TILE,
        std::ref(*mesh_device));
    ttnn::Tensor no_contrib_intermediate = pad_lse_to_intermediates_layout(col0_neg_inf);

    const double fw_setup_us = fw_timer.lap();
    for (uint32_t step = 0; step < ring_size; ++step) {
        // For causal masking, initialize intermediate_tensor to "no contribution" values
        // Devices that are skipped will keep these values, indicating zero contribution
        if (mask_type == ttml::metal::AttentionMaskType::Causal) {
            ttnn::copy(no_contrib_intermediate, intermediate_tensor);
        }

        auto [out_tensor, inter_tensor] = ttml::metal::ring_sdpa_fw(
            query_tensor,
            k_current,
            v_current,
            ring_size,
            cp_axis_value,
            step,
            mask_type,
            ttml::metal::ops::ring_sdpa_fw::RingDirection::Backward,
            output_tensor,
            intermediate_tensor);
        fw_sdpa_us.push_back(fw_timer.lap());

        merge_partial_into(output_accum, global_lse, output_tensor, intermediate_tensor);
        fw_merge_us.push_back(fw_timer.lap());

        if (step < ring_size - 1) {
            k_current = ttnn_fixed::distributed::ring_shift(
                k_current, cp_axis_value, ttnn_fixed::distributed::RingShiftDirection::Backward);
            v_current = ttnn_fixed::distributed::ring_shift(
                v_current, cp_axis_value, ttnn_fixed::distributed::RingShiftDirection::Backward);
            fw_shift_us.push_back(fw_timer.lap());
        }
    }

    // Single rounding to the input dtype; the graph (and the saved O used by backward)
    // sees the same dtype as before.
    auto out = autograd::create_tensor(ttnn::typecast(output_accum, query_tensor.dtype()));
    ttnn::Tensor final_lse = global_lse;
    if (RingAttnPhaseTimer::enabled()) {
        const double fw_final_us = fw_timer.lap();
        fmt::print(
            "[RING_ATTN_TIMING] pass=fw S_local={} ring={} total_us={:.0f} setup_us={:.0f} final_us={:.0f} "
            "sdpa_us={} merge_us={} shift_us={}\n",
            seq_len_local,
            ring_size,
            fw_timer.total_us(),
            fw_setup_us,
            fw_final_us,
            format_us_list(fw_sdpa_us),
            format_us_list(fw_merge_us),
            format_us_list(fw_shift_us));
    }

    autograd::GradFunction grad_fn = [query,
                                      key,
                                      value,
                                      out,
                                      final_lse,
                                      k_current,  // K at end of forward (position k1)
                                      v_current,  // V at end of forward (position v1)
                                      ring_size,
                                      cp_axis_value,
                                      mask_type,
                                      mesh_device]() mutable {
        tt::tt_metal::distributed::Synchronize(*mesh_device, std::nullopt, std::vector<tt::tt_metal::SubDeviceId>());
        const auto& grad_output = out->get_grad();
        const auto& attn_output = out->get_value();
        const auto& query_tensor = query->get_value();
        RingAttnPhaseTimer bw_timer(mesh_device);
        std::vector<double> bw_sdpa_us;
        std::vector<double> bw_accum_us;
        std::vector<double> bw_shift_us;

        // FP32 host accumulators: each ring step contributes a bf16 kernel output, but
        // summing them in bf16 rounds the full running magnitude every step. The
        // accumulators are cast back to the input dtype once, after the loop.
        ttnn::Tensor grad_Q_accum = ttnn::zeros_like(query_tensor, ttnn::DataType::FLOAT32);
        ttnn::Tensor grad_K_accum = ttnn::zeros_like(key->get_value(), ttnn::DataType::FLOAT32);
        ttnn::Tensor grad_V_accum = ttnn::zeros_like(value->get_value(), ttnn::DataType::FLOAT32);

        ttnn::Tensor grad_Q_step = ttnn::zeros_like(query_tensor);
        ttnn::Tensor grad_K_step = ttnn::zeros_like(key->get_value());
        ttnn::Tensor grad_V_step = ttnn::zeros_like(value->get_value());

        // Standard ring-flash-attention backward: every step gets the UNSCALED upstream
        // gradient, the GLOBAL forward output, and the GLOBAL logsumexp. The sdpa_bw
        // kernels recompute P = exp(scale*QK^T - lse) from the supplied lse, so the global
        // lse yields the global attention weights restricted to this chunk's columns, and
        // u = rowsum(dO * O_global) is the global softmax-backward correction. Per-chunk
        // contributions then sum to the exact dQ/dK/dV. Feeding per-chunk O/lse here would
        // bias dQ/dK (per-chunk u instead of global) — only dV would come out right.
        ttnn::Tensor global_intermediates = pad_lse_to_intermediates_layout(final_lse);

        // Zero sources for resetting the step buffers before every ring step.
        const ttnn::Tensor zero_Q = ttnn::zeros_like(grad_Q_step);
        const ttnn::Tensor zero_K = ttnn::zeros_like(grad_K_step);
        const ttnn::Tensor zero_V = ttnn::zeros_like(grad_V_step);

        const double bw_setup_us = bw_timer.lap();
        // Loop over ring steps in reverse order (from last to first)
        for (int step = ring_size - 1; step >= 0; --step) {
            const uint32_t step_idx = step;

            // Devices skipped by the causal schedule at this step do not run the kernels,
            // so their step buffers must be zeroed to contribute nothing to the accumulators.
            ttnn::copy(zero_Q, grad_Q_step);
            ttnn::copy(zero_K, grad_K_step);
            ttnn::copy(zero_V, grad_V_step);

            // Use Backward direction (same as forward) since src = (device + step) % ring_size
            auto [grad_Q_result, grad_K_result, grad_V_result] = ttml::metal::ring_sdpa_bw(
                grad_output,
                attn_output,  // Global forward output saved by autograd
                query_tensor,
                k_current,             // K at current ring position
                v_current,             // V at current ring position
                global_intermediates,  // Global logsumexp in intermediates layout
                ring_size,
                cp_axis_value,
                step_idx,
                mask_type,
                ttml::metal::ops::ring_sdpa_bw::RingDirection::Backward,
                grad_Q_step,
                grad_K_step,
                grad_V_step);
            bw_sdpa_us.push_back(bw_timer.lap());

            // The results alias the preallocated step buffers; skipped devices keep the
            // zeros written above. Upcast so the accumulation stays FP32.
            grad_Q_accum = ttnn::add(grad_Q_accum, ttnn::typecast(grad_Q_result, ttnn::DataType::FLOAT32));
            grad_K_accum = ttnn::add(grad_K_accum, ttnn::typecast(grad_K_result, ttnn::DataType::FLOAT32));
            grad_V_accum = ttnn::add(grad_V_accum, ttnn::typecast(grad_V_result, ttnn::DataType::FLOAT32));
            bw_accum_us.push_back(bw_timer.lap());

            // Ring shift K/V and grad accumulators in FORWARD direction
            // K/V: replays the forward pass in reverse (gets K/V for previous step)
            // grad_K/V: routes accumulated gradients back to correct device
            if (step > 0) {
                // Shift K/V forward to get position for next backward iteration
                k_current = ttnn_fixed::distributed::ring_shift(
                    k_current, cp_axis_value, ttnn_fixed::distributed::RingShiftDirection::Forward);
                v_current = ttnn_fixed::distributed::ring_shift(
                    v_current, cp_axis_value, ttnn_fixed::distributed::RingShiftDirection::Forward);

                // Shift grad accumulators
                grad_K_accum = ttnn_fixed::distributed::ring_shift(
                    grad_K_accum, cp_axis_value, ttnn_fixed::distributed::RingShiftDirection::Forward);
                grad_V_accum = ttnn_fixed::distributed::ring_shift(
                    grad_V_accum, cp_axis_value, ttnn_fixed::distributed::RingShiftDirection::Forward);
                bw_shift_us.push_back(bw_timer.lap());
            }
        }

        // Apply gradients, rounded once to the parameter dtype
        query->add_grad(ttnn::typecast(grad_Q_accum, query_tensor.dtype()));
        key->add_grad(ttnn::typecast(grad_K_accum, key->get_value().dtype()));
        value->add_grad(ttnn::typecast(grad_V_accum, value->get_value().dtype()));
        if (RingAttnPhaseTimer::enabled()) {
            const double bw_final_us = bw_timer.lap();
            fmt::print(
                "[RING_ATTN_TIMING] pass=bw S_local={} ring={} total_us={:.0f} setup_us={:.0f} final_us={:.0f} "
                "sdpa_us={} accum_us={} shift_us={}\n",
                query_tensor.logical_shape()[2],
                ring_size,
                bw_timer.total_us(),
                bw_setup_us,
                bw_final_us,
                format_us_list(bw_sdpa_us),
                format_us_list(bw_accum_us),
                format_us_list(bw_shift_us));
        }
    };

    out->set_node(autograd::add_backward_node(std::move(grad_fn), out, query, key, value));

    return out;
}

}  // namespace ttml::ops::distributed
