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
#include "ttnn/operations/data_movement/concat/concat.hpp"
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

// Rows [start, start + count) of a (B, H, S, D) tensor, as a copy.
ttnn::Tensor rows_of(const ttnn::Tensor& t, uint32_t start, uint32_t count) {
    const auto [b, h, s, d] = t.logical_shape().to_array_4D();
    (void)s;
    const ttsl::SmallVector<uint32_t> begin = {0, 0, start, 0};
    const ttsl::SmallVector<uint32_t> end = {b, h, start + count, d};
    const ttsl::SmallVector<uint32_t> stride = {1, 1, 1, 1};
    return ttnn::slice(t, begin, end, stride);
}

ttnn::Tensor cat_rows(const ttnn::Tensor& a, const ttnn::Tensor& b) {
    return ttnn::concat(std::vector<ttnn::Tensor>{a, b}, /*dim=*/2);
}

// Fold one step's partial attention (output, intermediates with the lse in
// column 0) into a running FP32 output and lse by the online softmax. A
// partial whose lse is -inf (a chip that ran nothing) leaves both unchanged.
void combine_partial(
    ttnn::Tensor& out_acc,
    ttnn::Tensor& lse_acc,
    const ttnn::Tensor& out_step,
    const ttnn::Tensor& inter_step,
    uint32_t batch,
    uint32_t heads,
    uint32_t rows) {
    const ttsl::SmallVector<uint32_t> slice_step = {1, 1, 1, 1};
    const ttsl::SmallVector<uint32_t> lse_start = {0, 0, 0, 0};
    const ttsl::SmallVector<uint32_t> lse_end = {batch, heads, rows, 1};
    const ttnn::Tensor lse_chunk = ttnn::slice(inter_step, lse_start, lse_end, slice_step);
    const ttnn::Tensor m = ttnn::maximum(lse_acc, lse_chunk);
    const ttnn::Tensor new_lse = ttnn::add(
        m, ttnn::log(ttnn::add(ttnn::exp(ttnn::subtract(lse_acc, m)), ttnn::exp(ttnn::subtract(lse_chunk, m)))));
    const ttnn::Tensor old_weight = ttnn::exp(ttnn::subtract(lse_acc, new_lse));
    const ttnn::Tensor new_weight = ttnn::exp(ttnn::subtract(lse_chunk, new_lse));
    const ttnn::Tensor step_fp32 = ttnn::typecast(out_step, ttnn::DataType::FLOAT32);
    out_acc = ttnn::add(ttnn::multiply(out_acc, old_weight), ttnn::multiply(step_fp32, new_weight));
    lse_acc = new_lse;
}

// The zigzag ring. Every local tensor is [chunk r | chunk 2d - 1 - r], n rows
// each. A step meets the visiting chip's two chunks, and of the four chunk
// pairs exactly two are live everywhere: on the diagonal step the two causal
// triangles (0, 0) and (1, 1) and the full block (1, 0); elsewhere (1, 0)
// and, depending on whether the visitor is an earlier or a later chip,
// (0, 0) or (1, 1). Every chip does the same amount of work at every step
// and none skips, which is the point.
//
// The forward and the two-pass backward go through the single-chip kernels
// on chunk-sized tensors, one pair per launch, the chips chosen by
// ops::ZigzagVisitor. The cyclic backward reads the chunk pairs straight out
// of the two-chunk tensors, two launches a step, accumulating in place.
autograd::TensorPtr ring_attention_sdpa_zigzag(
    const autograd::TensorPtr& query,
    const autograd::TensorPtr& key,
    const autograd::TensorPtr& value,
    RingBackwardKind backward_kind,
    uint32_t rows_per_block_tiles,
    ttnn_fixed::distributed::RingShiftTransport shift_transport) {
    using ttml::metal::AttentionMaskType;
    using ttml::metal::ops::ZigzagVisitor;
    using Direction = ttnn_fixed::distributed::RingShiftDirection;
    const auto& pctx = autograd::ctx().get_parallelism_context();
    const uint32_t cp_axis = pctx.get_cp_axis().value();
    const uint32_t ring_size = pctx.get_cp_size();
    const auto& query_tensor = query->get_value();
    auto* mesh_device = query_tensor.device();
    TT_FATAL(mesh_device != nullptr, "Query tensor must be on a mesh device for ring attention");
    const auto [batch_num, heads, local_rows, dim] = query_tensor.logical_shape().to_array_4D();
    TT_FATAL(
        local_rows % 64U == 0U,
        "zigzag ring attention: the local sequence of {} rows must be two chunks of whole tiles",
        local_rows);
    const uint32_t n = local_rows / 2U;
    tt::tt_metal::distributed::Synchronize(*mesh_device, std::nullopt, std::vector<tt::tt_metal::SubDeviceId>());

    // ------------------------------------------------------------ forward
    const ttnn::Tensor q_lo = rows_of(query_tensor, 0, n);
    const ttnn::Tensor q_hi = rows_of(query_tensor, n, n);
    ttnn::Tensor k_current = key->get_value();
    ttnn::Tensor v_current = value->get_value();

    const auto zeros_n = [&]() {
        return device_full(mesh_device, ttsl::SmallVector<uint32_t>{batch_num, heads, n, dim}, 0.0F, ttnn::DataType::FLOAT32);
    };
    const auto neg_inf_n = [&]() {
        return device_full(
            mesh_device,
            ttsl::SmallVector<uint32_t>{batch_num, heads, n, 1U},
            -std::numeric_limits<float>::infinity(),
            ttnn::DataType::FLOAT32);
    };
    ttnn::Tensor out_lo = zeros_n();
    ttnn::Tensor out_hi = zeros_n();
    ttnn::Tensor lse_lo = neg_inf_n();
    ttnn::Tensor lse_hi = neg_inf_n();
    ttnn::Tensor step_out = ttnn::empty_like(q_lo);
    ttnn::Tensor step_inter = ttnn::empty(
        ttnn::Shape{batch_num, heads, n, 32U},
        ttnn::DataType::FLOAT32,
        ttnn::Layout::TILE,
        mesh_device,
        ttnn::MemoryConfig(tt::tt_metal::TensorMemoryLayout::INTERLEAVED, ttnn::BufferType::DRAM));
    const ttnn::Tensor no_contrib = pad_lse_to_intermediates_layout(neg_inf_n());

    const auto partial = [&](const ttnn::Tensor& q,
                             const ttnn::Tensor& k,
                             const ttnn::Tensor& v,
                             ZigzagVisitor who,
                             AttentionMaskType mask,
                             ttnn::Tensor& out_acc,
                             ttnn::Tensor& lse_acc,
                             uint32_t step) {
        // Chips the launch does not select run nothing and must contribute
        // nothing: an lse of -inf does that.
        ttnn::copy(no_contrib, step_inter);
        ttml::metal::ring_zigzag_sdpa_fw(
            q, k, v, ring_size, cp_axis, step, who, mask, Direction::Backward, step_out, step_inter);
        combine_partial(out_acc, lse_acc, step_out, step_inter, batch_num, heads, n);
    };

    for (uint32_t step = 0; step < ring_size; ++step) {
        const ttnn::Tensor k_lo = rows_of(k_current, 0, n);
        const ttnn::Tensor k_hi = rows_of(k_current, n, n);
        const ttnn::Tensor v_lo = rows_of(v_current, 0, n);
        const ttnn::Tensor v_hi = rows_of(v_current, n, n);
        if (step == 0) {
            partial(q_lo, k_lo, v_lo, ZigzagVisitor::Any, AttentionMaskType::Causal, out_lo, lse_lo, step);
            partial(q_hi, k_hi, v_hi, ZigzagVisitor::Any, AttentionMaskType::Causal, out_hi, lse_hi, step);
            partial(q_hi, k_lo, v_lo, ZigzagVisitor::Any, AttentionMaskType::None, out_hi, lse_hi, step);
        } else {
            partial(q_hi, k_lo, v_lo, ZigzagVisitor::Any, AttentionMaskType::None, out_hi, lse_hi, step);
            partial(q_lo, k_lo, v_lo, ZigzagVisitor::Earlier, AttentionMaskType::None, out_lo, lse_lo, step);
            partial(q_hi, k_hi, v_hi, ZigzagVisitor::Later, AttentionMaskType::None, out_hi, lse_hi, step);
        }
        if (step + 1U < ring_size) {
            k_current = ttnn_fixed::distributed::ring_shift(k_current, cp_axis, Direction::Backward, shift_transport);
            v_current = ttnn_fixed::distributed::ring_shift(v_current, cp_axis, Direction::Backward, shift_transport);
        }
    }

    const ttnn::Tensor out_fp32 = cat_rows(out_lo, out_hi);
    auto out = autograd::create_tensor(ttnn::typecast(out_fp32, query_tensor.dtype()));
    const ttnn::Tensor lse_full = cat_rows(lse_lo, lse_hi);

    // ----------------------------------------------------------- backward
    autograd::GradFunction grad_fn = [query,
                                      key,
                                      value,
                                      out,
                                      lse_lo,
                                      lse_hi,
                                      lse_full,
                                      k_current,
                                      v_current,
                                      ring_size,
                                      cp_axis,
                                      n,
                                      backward_kind,
                                      rows_per_block_tiles,
                                      shift_transport,
                                      mesh_device]() mutable {
        tt::tt_metal::distributed::Synchronize(*mesh_device, std::nullopt, std::vector<tt::tt_metal::SubDeviceId>());
        RingBackwardProfile profile(mesh_device);
        const auto& grad_output = out->get_grad();
        const auto& attn_output = out->get_value();
        const auto& query_tensor = query->get_value();
        const bool two_pass = backward_kind == RingBackwardKind::TwoPass;

        // Cyclic: whole two-chunk tensors, the op addresses the chunks.
        ttnn::Tensor grad_Q_accum;
        ttnn::Tensor grad_K_accum;
        ttnn::Tensor grad_V_accum;
        ttnn::Tensor lse_pad_full;
        ttnn::Tensor row_scalar_full;
        // Two-pass: chunk tensors throughout, since its kernels take whole
        // tensors of one chunk length.
        ttnn::Tensor q_lo, q_hi, dO_lo, dO_hi, O_lo, O_hi, lse_pad_lo, lse_pad_hi;
        ttnn::Tensor dQ_lo, dQ_hi, dK_lo, dK_hi, dV_lo, dV_hi;  // FP32 accumulators
        ttnn::Tensor step_dQ, step_dK, step_dV, zero_step;      // bf16 per-launch outputs
        if (two_pass) {
            q_lo = rows_of(query_tensor, 0, n);
            q_hi = rows_of(query_tensor, n, n);
            dO_lo = rows_of(grad_output, 0, n);
            dO_hi = rows_of(grad_output, n, n);
            O_lo = rows_of(attn_output, 0, n);
            O_hi = rows_of(attn_output, n, n);
            lse_pad_lo = pad_lse_to_intermediates_layout(lse_lo);
            lse_pad_hi = pad_lse_to_intermediates_layout(lse_hi);
            for (ttnn::Tensor* t : {&dQ_lo, &dQ_hi, &dK_lo, &dK_hi, &dV_lo, &dV_hi}) {
                *t = device_zeros_like(q_lo, ttnn::DataType::FLOAT32);
            }
            step_dQ = device_zeros_like(q_lo, q_lo.dtype());
            step_dK = device_zeros_like(q_lo, q_lo.dtype());
            step_dV = device_zeros_like(q_lo, q_lo.dtype());
            zero_step = device_zeros_like(q_lo, q_lo.dtype());
        } else {
            grad_Q_accum = device_zeros_like(query_tensor, ttnn::DataType::FLOAT32);
            grad_K_accum = device_zeros_like(key->get_value(), ttnn::DataType::FLOAT32);
            grad_V_accum = device_zeros_like(value->get_value(), ttnn::DataType::FLOAT32);
            lse_pad_full = pad_lse_to_intermediates_layout(lse_full);
            row_scalar_full = pad_lse_to_intermediates_layout(ttml::ttnn_fixed::sum_ttnn(
                ttnn::multiply(
                    ttnn::typecast(grad_output, ttnn::DataType::FLOAT32),
                    ttnn::typecast(attn_output, ttnn::DataType::FLOAT32)),
                /* dim */ 3,
                /* keep_dim */ true));
        }
        profile.mark("setup (accumulators, D)");

        // One chunk pair through the two-pass kernels: zero the step buffers
        // (a chip the launch does not select leaves them so), run, add.
        const auto two_pass_pair = [&](const ttnn::Tensor& q,
                                       const ttnn::Tensor& k,
                                       const ttnn::Tensor& v,
                                       const ttnn::Tensor& dO,
                                       const ttnn::Tensor& O,
                                       const ttnn::Tensor& lse_pad,
                                       ZigzagVisitor who,
                                       AttentionMaskType mask,
                                       ttnn::Tensor& dQ_acc,
                                       ttnn::Tensor& dK_acc,
                                       ttnn::Tensor& dV_acc,
                                       uint32_t step) {
            ttnn::copy(zero_step, step_dQ);
            ttnn::copy(zero_step, step_dK);
            ttnn::copy(zero_step, step_dV);
            profile.mark("zero step buffers");
            auto [gq, gk, gv] = ttml::metal::ring_zigzag_sdpa_bw(
                dO, O, q, k, v, lse_pad, ring_size, cp_axis, step, who, mask, Direction::Backward, step_dQ, step_dK,
                step_dV);
            profile.mark(step == 0 ? "kernel, diagonal step" : "kernel, dense step");
            dQ_acc = ttnn::add(dQ_acc, ttnn::typecast(gq, ttnn::DataType::FLOAT32));
            dK_acc = ttnn::add(dK_acc, ttnn::typecast(gk, ttnn::DataType::FLOAT32));
            dV_acc = ttnn::add(dV_acc, ttnn::typecast(gv, ttnn::DataType::FLOAT32));
            profile.mark("accumulate");
        };
        // The cyclic op on the two-chunk tensors: the launch's mask picks the
        // pairs (Causal: both triangles; None: the blocks) and zigzag_pair
        // which of the blocks, accumulating in place.
        // dQ stays in the kernels' tile-transposed form across the steps (a
        // zero accumulator is in both forms), and only the very last launch
        // converts it back on the way out; so the snake's head, where every
        // row enters a dense pass, transposes nothing. See the cyclic op's
        // attributes.
        const auto cyclic_launch = [&](AttentionMaskType mask, uint32_t pair, uint32_t step, bool dq_out_transposed) {
            auto [gq, gk, gv] = ttml::metal::ring_cyclic_sdpa_bw(
                query_tensor,
                k_current,
                v_current,
                grad_output,
                lse_pad_full,
                row_scalar_full,
                ring_size,
                cp_axis,
                step,
                mask,
                Direction::Backward,
                rows_per_block_tiles,
                /* use_barrier */ false,
                /* accumulate_into_outputs */ true,
                grad_Q_accum,
                grad_K_accum,
                grad_V_accum,
                ttml::metal::ops::RingLayout::Zigzag,
                pair,
                /* grad_query_in_tile_transposed */ true,
                dq_out_transposed);
            grad_Q_accum = gq;
            grad_K_accum = gk;
            grad_V_accum = gv;
            profile.mark(step == 0 ? "kernel, diagonal step" : "kernel, dense step");
        };
        constexpr uint32_t kAllPairs = 0xFFFFFFFFU;

        for (int step_signed = static_cast<int>(ring_size) - 1; step_signed >= 0; --step_signed) {
            const auto step = static_cast<uint32_t>(step_signed);
            if (two_pass) {
                const ttnn::Tensor k_lo = rows_of(k_current, 0, n);
                const ttnn::Tensor k_hi = rows_of(k_current, n, n);
                const ttnn::Tensor v_lo = rows_of(v_current, 0, n);
                const ttnn::Tensor v_hi = rows_of(v_current, n, n);
                profile.mark("slice visiting chunks");
                if (step == 0) {
                    two_pass_pair(q_lo, k_lo, v_lo, dO_lo, O_lo, lse_pad_lo, ZigzagVisitor::Any, AttentionMaskType::Causal, dQ_lo, dK_lo, dV_lo, step);
                    two_pass_pair(q_hi, k_hi, v_hi, dO_hi, O_hi, lse_pad_hi, ZigzagVisitor::Any, AttentionMaskType::Causal, dQ_hi, dK_hi, dV_hi, step);
                    two_pass_pair(q_hi, k_lo, v_lo, dO_hi, O_hi, lse_pad_hi, ZigzagVisitor::Any, AttentionMaskType::None, dQ_hi, dK_lo, dV_lo, step);
                } else {
                    two_pass_pair(q_hi, k_lo, v_lo, dO_hi, O_hi, lse_pad_hi, ZigzagVisitor::Any, AttentionMaskType::None, dQ_hi, dK_lo, dV_lo, step);
                    two_pass_pair(q_lo, k_lo, v_lo, dO_lo, O_lo, lse_pad_lo, ZigzagVisitor::Earlier, AttentionMaskType::None, dQ_lo, dK_lo, dV_lo, step);
                    two_pass_pair(q_hi, k_hi, v_hi, dO_hi, O_hi, lse_pad_hi, ZigzagVisitor::Later, AttentionMaskType::None, dQ_hi, dK_hi, dV_hi, step);
                }
            } else {
                if (step == 0) {
                    // The causal launch covers every row, so it goes last and
                    // is the one that writes dQ in its natural layout.
                    cyclic_launch(AttentionMaskType::None, kAllPairs, step, /* dq_out_transposed */ true);
                    cyclic_launch(AttentionMaskType::Causal, kAllPairs, step, /* dq_out_transposed */ false);
                } else {
                    cyclic_launch(AttentionMaskType::None, 0U, step, /* dq_out_transposed */ true);
                    cyclic_launch(AttentionMaskType::None, 1U, step, /* dq_out_transposed */ true);
                }
            }

            if (step > 0) {
                k_current = ttnn_fixed::distributed::ring_shift(k_current, cp_axis, Direction::Forward, shift_transport);
                v_current = ttnn_fixed::distributed::ring_shift(v_current, cp_axis, Direction::Forward, shift_transport);
                if (two_pass) {
                    for (ttnn::Tensor* t : {&dK_lo, &dK_hi, &dV_lo, &dV_hi}) {
                        *t = ttnn_fixed::distributed::ring_shift(*t, cp_axis, Direction::Forward, shift_transport);
                    }
                } else {
                    grad_K_accum =
                        ttnn_fixed::distributed::ring_shift(grad_K_accum, cp_axis, Direction::Forward, shift_transport);
                    grad_V_accum =
                        ttnn_fixed::distributed::ring_shift(grad_V_accum, cp_axis, Direction::Forward, shift_transport);
                }
                profile.mark("shift K, V, dK, dV");
            }
        }

        if (two_pass) {
            grad_Q_accum = cat_rows(dQ_lo, dQ_hi);
            grad_K_accum = cat_rows(dK_lo, dK_hi);
            grad_V_accum = cat_rows(dV_lo, dV_hi);
        }
        query->add_grad(ttnn::typecast(grad_Q_accum, query_tensor.dtype()));
        key->add_grad(ttnn::typecast(grad_K_accum, key->get_value().dtype()));
        value->add_grad(ttnn::typecast(grad_V_accum, value->get_value().dtype()));
        profile.mark("finish (typecast, add_grad)");
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
    RingBackwardKind backward_kind,
    uint32_t rows_per_block_tiles,
    ttnn_fixed::distributed::RingShiftTransport shift_transport,
    ttml::metal::ops::RingLayout layout) {
    if (!autograd::ctx().is_parallelism_context_initialized() ||
        !autograd::ctx().get_parallelism_context().is_cp_enabled()) {
        return ttml::ops::scaled_dot_product_attention(query, key, value, mask);
    }
    if (layout == ttml::metal::ops::RingLayout::Zigzag) {
        TT_FATAL(!mask.has_value(), "Non-causal mask is not supported in CP mode");
        TT_FATAL(
            mask_type == ttml::metal::AttentionMaskType::Causal,
            "The zigzag layout balances a causal ring; for an unmasked one the contiguous layout is already "
            "balanced");
        return ring_attention_sdpa_zigzag(query, key, value, backward_kind, rows_per_block_tiles, shift_transport);
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
                    grad_V_accum,
                    ttml::metal::ops::RingLayout::Contiguous,
                    0xFFFFFFFFU,
                    // dQ stays in the kernels' tile-transposed form across the
                    // steps (a zero accumulator is in both forms), and the last
                    // step, the diagonal one, converts it back on the way out;
                    // so the snake's head, where every row enters a dense pass,
                    // transposes nothing. See the cyclic op's attributes.
                    /* grad_query_in_tile_transposed */ true,
                    /* grad_query_out_tile_transposed */ step_idx != 0);
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
