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
#include <unordered_map>
#include <unordered_set>
#include <tt-metalium/distributed.hpp>
#include <umd/device/cluster.hpp>

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
#include "ttnn/operations/data_movement/copy/copy.hpp"
#include "ttnn/operations/data_movement/pad/pad.hpp"
#include "ttnn/operations/data_movement/slice/slice.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/binary/binary_composite.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/full/full.hpp"
#include "ttnn/operations/full_like/full_like.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"
#include "ttnn_fixed/distributed/tt_metal.hpp"
#include "ttnn_fixed/trivial_ttnn_ops.hpp"
#include "ttnn_fixed/distributed/ttnn_ops.hpp"

namespace ttml::ops::distributed {

namespace {

// A zero tensor of `like`'s shape in `dtype`, filled on the device.
// ttnn::zeros_like builds the tensor on the host and writes it to every chip,
// which for the eight tensors of a 4096-row backward was 25 ms of a 45 ms
// total; moreh_full_like is a device kernel and takes a dispatch.
ttnn::Tensor device_zeros_like(const ttnn::Tensor& like, ttnn::DataType dtype) {
    return ttnn::moreh_full_like(like, 0.0F, dtype, like.layout(), like.memory_config());
}

// The same for a fresh shape and value, on the mesh.
ttnn::Tensor device_full(
    ttnn::MeshDevice* mesh_device, const ttsl::SmallVector<uint32_t>& shape, float value, ttnn::DataType dtype) {
    return ttnn::moreh_full(shape, value, mesh_device, dtype, ttnn::Layout::TILE, ttnn::DRAM_MEMORY_CONFIG);
}

// Where a ring backward's time goes, by phase, when TTML_RING_PROFILE is set.
// Each mark synchronises the mesh, so the profile perturbs what it measures
// (nothing overlaps that would otherwise); what it is good for is the split
// of the total. Off, it costs a getenv and nothing else.
class RingBackwardProfile {
public:
    explicit RingBackwardProfile(tt::tt_metal::distributed::MeshDevice* device) :
        m_device(device), m_enabled(std::getenv("TTML_RING_PROFILE") != nullptr) {
        if (m_enabled) {
            sync();
            m_last = std::chrono::steady_clock::now();
        }
    }

    void mark(const char* phase) {
        if (!m_enabled) {
            return;
        }
        sync();
        const auto now = std::chrono::steady_clock::now();
        auto& slot = m_phases[phase];
        slot.first += std::chrono::duration<double>(now - m_last).count() * 1e3;
        slot.second += 1U;
        m_order.emplace(phase);
        m_last = now;
    }

    ~RingBackwardProfile() {
        if (!m_enabled) {
            return;
        }
        double total = 0.0;
        for (const auto& [name, v] : m_phases) {
            total += v.first;
        }
        fmt::print("[ring backward profile] total {:.2f} ms\n", total);
        for (const auto& name : m_order_vector()) {
            const auto& v = m_phases.at(name);
            fmt::print("  {:<28} {:8.2f} ms  ({:2d} x {:7.1f} us)\n", name, v.first, v.second, v.first / v.second * 1e3);
        }
    }

private:
    void sync() {
        tt::tt_metal::distributed::Synchronize(*m_device, std::nullopt, std::vector<tt::tt_metal::SubDeviceId>());
    }
    std::vector<std::string> m_order_vector() const {
        std::vector<std::string> names;
        for (const auto& n : m_order_seq) {
            names.push_back(n);
        }
        return names;
    }
    struct Order {
        std::vector<std::string>& seq;
        std::unordered_set<std::string>& seen;
        void emplace(const std::string& n) {
            if (seen.insert(n).second) {
                seq.push_back(n);
            }
        }
    };

    tt::tt_metal::distributed::MeshDevice* m_device{};
    bool m_enabled{};
    std::chrono::steady_clock::time_point m_last;
    std::unordered_map<std::string, std::pair<double, uint32_t>> m_phases;
    std::vector<std::string> m_order_seq;
    std::unordered_set<std::string> m_order_seen;
    Order m_order{m_order_seq, m_order_seen};
};

}  // namespace

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

}  // namespace

autograd::TensorPtr ring_attention_sdpa(
    const autograd::TensorPtr& query,
    const autograd::TensorPtr& key,
    const autograd::TensorPtr& value,
    const std::optional<autograd::TensorPtr>& mask,
    const ttml::metal::AttentionMaskType mask_type,
    RingBackwardKind backward_kind,
    uint32_t rows_per_block_tiles,
    ttnn_fixed::distributed::RingShiftTransport shift_transport) {
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

    auto [batch_num, heads, seq_len_local, dim] = query_tensor.logical_shape().to_array_4D();
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
    ttnn::Tensor output_accum = device_full(
        mesh_device, ttsl::SmallVector<uint32_t>{batch_num, heads, seq_len_local, dim}, 0.0F, ttnn::DataType::FLOAT32);

    // global_lse: running logsumexp across all steps
    // lse = log(sum(exp(scale * score_i))) — the log of the softmax normalizer
    // Initialized to -inf (no contribution: exp(-inf) = 0)
    ttnn::Tensor global_lse = device_full(
        mesh_device,
        ttsl::SmallVector<uint32_t>{batch_num, heads, seq_len_local, 1U},
        -std::numeric_limits<float>::infinity(),
        ttnn::DataType::FLOAT32);

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
    ttnn::Tensor col0_neg_inf = device_full(
        mesh_device,
        ttsl::SmallVector<uint32_t>{batch_num, heads, seq_len_local, 1U},
        -std::numeric_limits<float>::infinity(),
        ttnn::DataType::FLOAT32);
    ttnn::Tensor no_contrib_intermediate = pad_lse_to_intermediates_layout(col0_neg_inf);

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

        // Extract logsumexp from column 0 of intermediate
        // Intermediate shape: (B, H, S, 32) FP32, logsumexp in column 0
        const ttsl::SmallVector<uint32_t> slice_step = {1, 1, 1, 1};
        const ttsl::SmallVector<uint32_t> lse_start = {0, 0, 0, 0};
        const ttsl::SmallVector<uint32_t> lse_end = {batch_num, heads, seq_len_local, 1};
        ttnn::Tensor lse_chunk = ttnn::slice(intermediate_tensor, lse_start, lse_end, slice_step);

        // Combine via logaddexp: new_lse = log(exp(global_lse) + exp(lse_chunk))
        // Numerically stable form: m = max(a,b); result = m + log(exp(a-m) + exp(b-m))
        ttnn::Tensor m = ttnn::maximum(global_lse, lse_chunk);
        ttnn::Tensor exp_global = ttnn::exp(ttnn::subtract(global_lse, m));
        ttnn::Tensor exp_chunk = ttnn::exp(ttnn::subtract(lse_chunk, m));
        ttnn::Tensor new_lse = ttnn::add(m, ttnn::log(ttnn::add(exp_global, exp_chunk)));

        // Weights for combining outputs: w = exp(lse - new_lse) = Z_i / Z_combined
        ttnn::Tensor old_weight = ttnn::exp(ttnn::subtract(global_lse, new_lse));
        ttnn::Tensor new_weight = ttnn::exp(ttnn::subtract(lse_chunk, new_lse));

        // Weighted combination of accumulated output and this step's output.
        // The step output is upcast so the whole combine stays FP32; mixed-dtype
        // binary ops would otherwise round the products back to bf16.
        ttnn::Tensor step_output_fp32 = ttnn::typecast(output_tensor, ttnn::DataType::FLOAT32);
        output_accum =
            ttnn::add(ttnn::multiply(output_accum, old_weight), ttnn::multiply(step_output_fp32, new_weight));

        global_lse = new_lse;

        if (step < ring_size - 1) {
            k_current = ttnn_fixed::distributed::ring_shift(
                k_current, cp_axis_value, ttnn_fixed::distributed::RingShiftDirection::Backward, shift_transport);
            v_current = ttnn_fixed::distributed::ring_shift(
                v_current, cp_axis_value, ttnn_fixed::distributed::RingShiftDirection::Backward, shift_transport);
        }
    }

    // Single rounding to the input dtype; the graph (and the saved O used by backward)
    // sees the same dtype as before.
    auto out = autograd::create_tensor(ttnn::typecast(output_accum, query_tensor.dtype()));
    ttnn::Tensor final_lse = global_lse;

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
                                      backward_kind,
                                      rows_per_block_tiles,
                                      shift_transport,
                                      mesh_device]() mutable {
        tt::tt_metal::distributed::Synchronize(*mesh_device, std::nullopt, std::vector<tt::tt_metal::SubDeviceId>());
        RingBackwardProfile profile(mesh_device);
        const auto& grad_output = out->get_grad();
        const auto& attn_output = out->get_value();
        const auto& query_tensor = query->get_value();

        // FP32 host accumulators: each ring step contributes a bf16 kernel output, but
        // summing them in bf16 rounds the full running magnitude every step. The
        // accumulators are cast back to the input dtype once, after the loop.
        ttnn::Tensor grad_Q_accum = device_zeros_like(query_tensor, ttnn::DataType::FLOAT32);
        ttnn::Tensor grad_K_accum = device_zeros_like(key->get_value(), ttnn::DataType::FLOAT32);
        ttnn::Tensor grad_V_accum = device_zeros_like(value->get_value(), ttnn::DataType::FLOAT32);

        const bool cyclic =
            backward_kind == RingBackwardKind::Cyclic || backward_kind == RingBackwardKind::CyclicInPlace;
        const bool in_place = backward_kind == RingBackwardKind::CyclicInPlace;

        // Step buffers and the zero sources that reset them before every ring
        // step: only the ones this backward_kind uses. The two-pass op writes
        // bf16; the cyclic op is FP32 in and out, so its step buffers are too.
        // That is a precision difference between the two paths, in the cyclic
        // path's favour, and it is the op's own dtype rather than a choice
        // made here: its gradients are read back and accumulated into on
        // device, which bf16 would round at every streak start. The in-place
        // cyclic path writes the accumulators and needs neither.
        ttnn::Tensor grad_Q_step, grad_K_step, grad_V_step, zero_Q, zero_K, zero_V;
        ttnn::Tensor grad_Q_step_fp32, grad_K_step_fp32, grad_V_step_fp32, zero_Q_fp32, zero_K_fp32, zero_V_fp32;
        if (!cyclic) {
            grad_Q_step = device_zeros_like(query_tensor, query_tensor.dtype());
            grad_K_step = device_zeros_like(key->get_value(), key->get_value().dtype());
            grad_V_step = device_zeros_like(value->get_value(), value->get_value().dtype());
            zero_Q = device_zeros_like(grad_Q_step, grad_Q_step.dtype());
            zero_K = device_zeros_like(grad_K_step, grad_K_step.dtype());
            zero_V = device_zeros_like(grad_V_step, grad_V_step.dtype());
        } else if (!in_place) {
            grad_Q_step_fp32 = device_zeros_like(query_tensor, ttnn::DataType::FLOAT32);
            grad_K_step_fp32 = device_zeros_like(key->get_value(), ttnn::DataType::FLOAT32);
            grad_V_step_fp32 = device_zeros_like(value->get_value(), ttnn::DataType::FLOAT32);
            zero_Q_fp32 = device_zeros_like(grad_Q_step_fp32, ttnn::DataType::FLOAT32);
            zero_K_fp32 = device_zeros_like(grad_K_step_fp32, ttnn::DataType::FLOAT32);
            zero_V_fp32 = device_zeros_like(grad_V_step_fp32, ttnn::DataType::FLOAT32);
        }

        // Standard ring-flash-attention backward: every step gets the UNSCALED upstream
        // gradient, the GLOBAL forward output, and the GLOBAL logsumexp. The sdpa_bw
        // kernels recompute P = exp(scale*QK^T - lse) from the supplied lse, so the global
        // lse yields the global attention weights restricted to this chunk's columns, and
        // u = rowsum(dO * O_global) is the global softmax-backward correction. Per-chunk
        // contributions then sum to the exact dQ/dK/dV. Feeding per-chunk O/lse here would
        // bias dQ/dK (per-chunk u instead of global) — only dV would come out right.
        ttnn::Tensor global_intermediates = pad_lse_to_intermediates_layout(final_lse);

        // The cyclic backward needs D = rowsum(dO . O), which the two-pass
        // backward computes for itself inside its dQ pass, once per ring step.
        // It is global and constant across steps, so it is computed once here.
        // The product is taken in FP32 rather than the operands' bf16: D is
        // subtracted from dP, so a rounding here lands directly in dS.
        ttnn::Tensor row_scalar;
        if (cyclic) {
            row_scalar = pad_lse_to_intermediates_layout(ttml::ttnn_fixed::sum_ttnn(
                ttnn::multiply(
                    ttnn::typecast(grad_output, ttnn::DataType::FLOAT32),
                    ttnn::typecast(attn_output, ttnn::DataType::FLOAT32)),
                /* dim */ 3,
                /* keep_dim */ true));
        }

        profile.mark("setup (accumulators, D)");

        // Loop over ring steps in reverse order (from last to first)
        for (int step = ring_size - 1; step >= 0; --step) {
            const uint32_t step_idx = step;

            // Devices skipped by the causal schedule at this step do not run the kernels,
            // so their step buffers must be zeroed to contribute nothing to the accumulators.
            if (in_place) {
                // Nothing to zero: the accumulators are the outputs.
            } else if (cyclic) {
                ttnn::copy(zero_Q_fp32, grad_Q_step_fp32);
                ttnn::copy(zero_K_fp32, grad_K_step_fp32);
                ttnn::copy(zero_V_fp32, grad_V_step_fp32);
            } else {
                ttnn::copy(zero_Q, grad_Q_step);
                ttnn::copy(zero_K, grad_K_step);
                ttnn::copy(zero_V, grad_V_step);
            }

            profile.mark("zero step buffers");

            if (in_place) {
                // The step's contribution lands in the accumulators themselves.
                // A chip the causal schedule skips has no program this step
                // and leaves them as they are. The shifted accumulators carry
                // their sums with them, as before.
                auto [gq, gk, gv] = ttml::metal::ring_cyclic_sdpa_bw(
                    query_tensor,
                    k_current,
                    v_current,
                    grad_output,
                    global_intermediates,
                    row_scalar,
                    ring_size,
                    cp_axis_value,
                    step_idx,
                    mask_type,
                    ttml::metal::RingCyclicDirection::Backward,
                    rows_per_block_tiles,
                    /* use_barrier */ false,
                    /* accumulate_into_outputs */ true,
                    grad_Q_accum,
                    grad_K_accum,
                    grad_V_accum);
                grad_Q_accum = gq;
                grad_K_accum = gk;
                grad_V_accum = gv;
                profile.mark(step_idx == 0 ? "kernel, diagonal step" : "kernel, dense step");
            } else if (cyclic) {
                // The cyclic op returns FP32 and accumulates into whatever the
                // step buffers hold, so they are zeroed above and the step's
                // own contribution comes back; the host then adds it, as the
                // two-pass driver does, so the two drivers stay identical in
                // structure. CyclicInPlace is the version without this.
                auto [grad_Q_result, grad_K_result, grad_V_result] = ttml::metal::ring_cyclic_sdpa_bw(
                    query_tensor,
                    k_current,
                    v_current,
                    grad_output,
                    global_intermediates,
                    row_scalar,
                    ring_size,
                    cp_axis_value,
                    step_idx,
                    mask_type,
                    ttml::metal::RingCyclicDirection::Backward,
                    rows_per_block_tiles,
                    /* use_barrier */ false,
                    /* accumulate_into_outputs */ false,
                    grad_Q_step_fp32,
                    grad_K_step_fp32,
                    grad_V_step_fp32);
                profile.mark(step_idx == 0 ? "kernel, diagonal step" : "kernel, dense step");
                grad_Q_accum = ttnn::add(grad_Q_accum, grad_Q_result);
                grad_K_accum = ttnn::add(grad_K_accum, grad_K_result);
                grad_V_accum = ttnn::add(grad_V_accum, grad_V_result);
                profile.mark("accumulate");
            } else {
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
            profile.mark(step_idx == 0 ? "kernel, diagonal step" : "kernel, dense step");

            // The results alias the preallocated step buffers; skipped devices keep the
            // zeros written above. Upcast so the accumulation stays FP32.
            grad_Q_accum = ttnn::add(grad_Q_accum, ttnn::typecast(grad_Q_result, ttnn::DataType::FLOAT32));
            grad_K_accum = ttnn::add(grad_K_accum, ttnn::typecast(grad_K_result, ttnn::DataType::FLOAT32));
            grad_V_accum = ttnn::add(grad_V_accum, ttnn::typecast(grad_V_result, ttnn::DataType::FLOAT32));
            profile.mark("accumulate");
            }

            // Ring shift K/V and grad accumulators in FORWARD direction
            // K/V: replays the forward pass in reverse (gets K/V for previous step)
            // grad_K/V: routes accumulated gradients back to correct device
            if (step > 0) {
                // Shift K/V forward to get position for next backward iteration
                k_current = ttnn_fixed::distributed::ring_shift(
                    k_current, cp_axis_value, ttnn_fixed::distributed::RingShiftDirection::Forward, shift_transport);
                v_current = ttnn_fixed::distributed::ring_shift(
                    v_current, cp_axis_value, ttnn_fixed::distributed::RingShiftDirection::Forward, shift_transport);

                // Shift grad accumulators
                grad_K_accum = ttnn_fixed::distributed::ring_shift(
                    grad_K_accum, cp_axis_value, ttnn_fixed::distributed::RingShiftDirection::Forward, shift_transport);
                grad_V_accum = ttnn_fixed::distributed::ring_shift(
                    grad_V_accum, cp_axis_value, ttnn_fixed::distributed::RingShiftDirection::Forward, shift_transport);
                profile.mark("shift K, V, dK, dV");
            }
        }

        // Apply gradients, rounded once to the parameter dtype
        query->add_grad(ttnn::typecast(grad_Q_accum, query_tensor.dtype()));
        key->add_grad(ttnn::typecast(grad_K_accum, key->get_value().dtype()));
        value->add_grad(ttnn::typecast(grad_V_accum, value->get_value().dtype()));
        profile.mark("finish (typecast, add_grad)");
    };

    out->set_node(autograd::add_backward_node(std::move(grad_fn), out, query, key, value));

    return out;
}

}  // namespace ttml::ops::distributed
