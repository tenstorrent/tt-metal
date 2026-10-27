// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "chunk_gated_delta_rule_bw_program_factory.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

#include "metal/common/program_utils.hpp"

namespace {

constexpr auto kReaderKernelPath =
    "tt-train/sources/ttml/metal/ops/chunk_gated_delta_rule_bw/device/kernels/dataflow/"
    "reader_chunk_gated_delta_rule_bw.cpp";
constexpr auto kWriterKernelPath =
    "tt-train/sources/ttml/metal/ops/chunk_gated_delta_rule_bw/device/kernels/dataflow/"
    "writer_chunk_gated_delta_rule_bw.cpp";
constexpr auto kComputeKernelPath =
    "tt-train/sources/ttml/metal/ops/chunk_gated_delta_rule_bw/device/kernels/compute/"
    "chunk_gated_delta_rule_bw_kernel.cpp";

// L1 below the circular-buffer region that get_max_worker_l1_unreserved_size() still counts: the
// kernel-config ring buffer the five kernel binaries live in, plus slack.
constexpr uint32_t kL1KernelConfigReserve = 80U * 1024U;

// Named-block counts inside the three multi-slot CBs; the kernels derive every slot offset from the
// same values (chunk_gated_delta_rule_bw_common.hpp).
constexpr uint32_t kNumConstMasks = 6U;  // LT, NSTRICT, UT, EYE, BIAS, SUT
constexpr uint32_t kNumVecASlots = 5U;   // decay, gamma, w, beta, dc1
constexpr uint32_t kNumVecBSlots = 9U;   // stage-G column accumulators

constexpr uint32_t kBlockDepth = 1U;   // a resident block CB
constexpr uint32_t kAccumDepth = 2U;   // a block CB updated in place (push the new block, pop the old)
constexpr uint32_t kEgressDepth = 2U;  // blocks in flight to the writer

// Circular-buffer indices; must mirror chunk_gated_delta_rule_bw_common.hpp.
enum Cb : uint32_t {
    kCbConst = 0,
    kCbColOnes,
    kCbStage,
    kCbQIn,
    kCbKIn,
    kCbVIn,
    kCbDoIn,
    kCbGateIn,
    kCbLoadItem,
    kCbLoadVb,
    kCbKA,
    kCbKB,
    kCbKC,
    kCbKD,
    kCbKE,
    kCbKF,
    kCbKTr,
    kCbVA,
    kCbVB,
    kCbVC,
    kCbCA,
    kCbL,
    kCbCC,
    kCbCD,
    kCbCE,
    kCbSA,
    kCbSB,
    kCbVecA,
    kCbVecB,
    kCbVecC,
    kCbEgress,
    kCbGradEgress,
    kNumCbs
};

// The uniform push size of every CB whose block is not a plain product of the block extents.  A
// CB pushed with two different sizes would wrap its fifo, so where a stage needs fewer tiles than
// the block it leaves the tail unused.
struct CbBlocks {
    uint32_t nconst, ncol, maxv, lvb, litem, maxblk, maxblk_g;
};

CbBlocks cb_blocks(uint32_t Ct, uint32_t Kt, uint32_t Vb) {
    return {
        .nconst = kNumConstMasks * Ct * Ct + 2U + Ct,  // masks + scale + zero + Ct row-of-ones
        .ncol = std::max({Ct, Kt, Vb}),
        .maxv = std::max(Ct, Kt) * Vb,
        .lvb = std::max(
            2U * Ct * Kt + Ct * Vb + Kt * Vb + 1U,  // stage-S reverse step
            2U * Ct * Vb + 2U * Kt * Vb),           // stage-G V slice
        .litem = Ct * Ct + 4U * Ct,
        .maxblk = std::max({Ct * Ct, Ct * Kt, Ct * Vb, Kt * Vb, 4U * Ct}),
        // dh0 leaves through the gradient egress as a [Kt, Vb] block.
        .maxblk_g = std::max({Ct * Kt, Ct * Vb, Ct, Kt * Vb}),
    };
}

std::array<uint32_t, kNumCbs> cb_pages(uint32_t Ct, uint32_t Kt, uint32_t Vb) {
    const auto blk = cb_blocks(Ct, Kt, Vb);
    std::array<uint32_t, kNumCbs> pages{};
    pages[kCbConst] = blk.nconst;
    pages[kCbColOnes] = blk.ncol;
    pages[kCbStage] = 1U;
    pages[kCbQIn] = kBlockDepth * Ct * Kt;
    pages[kCbKIn] = kBlockDepth * Ct * Kt;
    pages[kCbVIn] = kBlockDepth * blk.maxv;
    pages[kCbDoIn] = kBlockDepth * Ct * Vb;
    pages[kCbGateIn] = kBlockDepth * 2U * Ct;
    pages[kCbLoadItem] = blk.litem;
    pages[kCbLoadVb] = blk.lvb;
    for (const uint32_t cb : {kCbKA, kCbKB, kCbKC, kCbKD, kCbKE, kCbKF, kCbKTr}) {
        pages[cb] = kAccumDepth * Ct * Kt;
    }
    for (const uint32_t cb : {kCbVA, kCbVB, kCbVC}) {
        pages[cb] = kAccumDepth * Ct * Vb;
    }
    for (const uint32_t cb : {kCbCA, kCbCC, kCbCD, kCbCE}) {
        pages[cb] = kAccumDepth * Ct * Ct;
    }
    pages[kCbL] = Ct * Ct;  // L is never updated in place
    pages[kCbSA] = kAccumDepth * Kt * Vb;
    pages[kCbSB] = kAccumDepth * Kt * Vb;
    pages[kCbVecA] = kNumVecASlots * Ct;
    pages[kCbVecB] = kNumVecBSlots * Ct;
    pages[kCbVecC] = kAccumDepth * Ct;
    pages[kCbEgress] = kEgressDepth * blk.maxblk;
    pages[kCbGradEgress] = kEgressDepth * blk.maxblk_g;
    return pages;
}

bool is_input_dtype_cb(uint32_t cb) {
    return cb == kCbQIn || cb == kCbKIn || cb == kCbVIn || cb == kCbDoIn || cb == kCbGateIn || cb == kCbGradEgress;
}

// The staging CB holds one input tile read for the g/beta gather, at a 64-byte-aligned offset.
uint32_t stage_bytes(uint32_t in_tile) {
    return in_tile + 64U;
}

uint64_t cb_bytes(uint32_t Ct, uint32_t Kt, uint32_t Vb, uint32_t in_tile, uint32_t f32_tile) {
    const auto pages = cb_pages(Ct, Kt, Vb);
    uint64_t total = 0;
    for (uint32_t cb = 0; cb < kNumCbs; ++cb) {
        if (cb == kCbStage) {
            total += stage_bytes(in_tile);
        } else {
            total += static_cast<uint64_t>(pages[cb]) * (is_input_dtype_cb(cb) ? in_tile : f32_tile);
        }
    }
    return total;
}

uint32_t ceil_div(uint32_t a, uint32_t b) {
    return (a + b - 1U) / b;
}

}  // namespace

namespace ttml::metal::ops::chunk_gated_delta_rule_bw::device {

Geometry make_geometry(const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto qs = as_rank4(tensor_args.q.logical_shape());
    const auto vs = as_rank4(tensor_args.v.logical_shape());
    const auto gs = as_rank4(tensor_args.g.logical_shape());

    Geometry geo;
    geo.B = qs[0];
    geo.T = qs[2];
    geo.HV = gs[3];
    geo.K = args.key_head_dim;
    geo.V = vs[3] / geo.HV;
    geo.HK = qs[3] / geo.K;
    geo.G = geo.HV / geo.HK;
    geo.Ct = args.chunk_size / tt::constants::TILE_HEIGHT;
    geo.Kt = ceil_div(geo.K, tt::constants::TILE_WIDTH);
    geo.Vt = ceil_div(geo.V, tt::constants::TILE_WIDTH);
    geo.Tt = ceil_div(geo.T, tt::constants::TILE_HEIGHT);
    geo.NC = ceil_div(geo.T, args.chunk_size);
    geo.BH = geo.B * geo.HV;
    geo.NI = geo.BH * geo.NC;
    geo.neumann_steps = std::max(1U, static_cast<uint32_t>(std::ceil(std::log2(static_cast<double>(args.chunk_size)))));

    const uint32_t in_tile = tt::tile_size(tt::tt_metal::datatype_to_dataformat_converter(tensor_args.q.dtype()));
    const uint32_t f32_tile = tt::tile_size(tt::DataFormat::Float32);
    const uint64_t budget = tt::tt_metal::hal::get_max_worker_l1_unreserved_size() - kL1KernelConfigReserve;
    for (uint32_t vb = geo.Vt; vb > 0U; --vb) {
        if (geo.Vt % vb == 0U && cb_bytes(geo.Ct, geo.Kt, vb, in_tile, f32_tile) <= budget) {
            geo.Vb = vb;
            break;
        }
    }
    TT_FATAL(
        geo.Vb != 0U,
        "chunk_gated_delta_rule_bw: no V block fits L1 (Ct={}, Kt={}, Vt={}, budget={}, min footprint={})",
        geo.Ct,
        geo.Kt,
        geo.Vt,
        budget,
        cb_bytes(geo.Ct, geo.Kt, 1U, in_tile, f32_tile));
    geo.NVB = geo.Vt / geo.Vb;

    const uint32_t CtCt = geo.Ct * geo.Ct;
    const uint32_t CtKt = geo.Ct * geo.Kt;
    const uint32_t CtVt = geo.Ct * geo.Vt;
    const uint32_t KtVt = geo.Kt * geo.Vt;
    // The saved forward replaces Tinv, S and v_new, and v_corr only feeds the forward scan.
    const bool saved = tensor_args.has_saved_forward();
    uint32_t off = 0;
    auto take = [&](uint32_t per_item) {
        const uint32_t base = off;
        off += geo.NI * per_item;
        return base;
    };
    geo.sc_attn = take(saved ? 0U : CtCt);
    geo.sc_kcd = take(CtKt);
    geo.sc_p = take(CtKt);
    geo.sc_vec = take(4U * geo.Ct);
    geo.sc_vcorr = take(saved ? 0U : CtVt);
    geo.sc_u = take(CtVt);
    geo.sc_c = take(KtVt);
    geo.sc_s = take(saved ? 0U : KtVt);
    geo.sc_ds = take(KtVt);
    geo.sc_vnew = take(saved ? 0U : CtVt);
    geo.sc_dvnew = take(CtVt);
    geo.scratch_tiles = std::max(off, 1U);
    return geo;
}

ChunkGatedDeltaRuleBackwardProgramFactory::cached_program_t ChunkGatedDeltaRuleBackwardProgramFactory::create(
    const operation_attributes_t& args, const tensor_args_t& tensor_args, tensor_return_value_t& outputs) {
    const auto& q = tensor_args.q;
    auto* device = q.device();
    tt::tt_metal::Program program{};

    const Geometry geo = make_geometry(args, tensor_args);
    const bool has_h0 = tensor_args.initial_state.has_value();
    const bool has_dht = tensor_args.d_final_state.has_value();
    const bool has_saved = tensor_args.has_saved_forward();

    const tt::DataFormat in_format = tt::tt_metal::datatype_to_dataformat_converter(q.dtype());
    const uint32_t in_tile = tt::tile_size(in_format);
    const uint32_t f32_tile = tt::tile_size(tt::DataFormat::Float32);

    const auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), args.compute_kernel_config);
    const uint32_t dest_limit = dst_full_sync_en ? (fp32_dest_acc_en ? 8U : 16U) : (fp32_dest_acc_en ? 4U : 8U);

    // ---------------- work distribution -----------------------------------
    // Work item wi = bh * NC + chunk.  The scan owner of group bh is the core holding item bh * NC.
    const auto grid = device->compute_with_storage_grid_size();
    const auto [num_cores, all_cores, group1, group2, items1, items2] =
        tt::tt_metal::split_work_to_cores(grid, geo.NI, /*row_wise=*/true);

    struct Assignment {
        tt::tt_metal::CoreCoord core;
        uint32_t start;
        uint32_t count;
    };
    std::vector<Assignment> assign;
    uint32_t cursor = 0;
    for (const auto& [group, count] : {std::pair{group1, items1}, std::pair{group2, items2}}) {
        if (count == 0U) {
            continue;
        }
        for (const auto& core : tt::tt_metal::corerange_to_cores(group, std::nullopt, /*row_wise=*/true)) {
            assign.push_back({core, cursor, count});
            cursor += count;
        }
    }
    TT_FATAL(cursor == geo.NI, "chunk_gated_delta_rule_bw: work split lost items: {} != {}", cursor, geo.NI);

    std::vector<tt::tt_metal::CoreCoord> owner_of(geo.BH);
    std::vector<std::vector<tt::tt_metal::CoreCoord>> members_of(geo.BH);
    for (const auto& a : assign) {
        for (uint32_t wi = a.start; wi < a.start + a.count; ++wi) {
            if (wi % geo.NC == 0U) {
                owner_of[wi / geo.NC] = a.core;
            }
        }
        for (uint32_t bh = a.start / geo.NC; bh <= (a.start + a.count - 1U) / geo.NC; ++bh) {
            members_of[bh].push_back(a.core);
        }
    }

    // ---------------- circular buffers ------------------------------------
    const auto blk = cb_blocks(geo.Ct, geo.Kt, geo.Vb);
    const auto pages = cb_pages(geo.Ct, geo.Kt, geo.Vb);
    for (uint32_t cb = 0; cb < kNumCbs; ++cb) {
        if (cb == kCbStage) {
            create_circular_buffer_bytes(program, all_cores, cb, tt::DataFormat::Float32, stage_bytes(in_tile));
        } else if (is_input_dtype_cb(cb)) {
            create_circular_buffer(program, all_cores, cb, in_format, in_tile, pages[cb]);
        } else {
            create_circular_buffer(program, all_cores, cb, tt::DataFormat::Float32, f32_tile, pages[cb]);
        }
    }

    const uint32_t sem_prep = tt::tt_metal::CreateSemaphore(program, all_cores, 0U);
    const uint32_t sem_scan = tt::tt_metal::CreateSemaphore(program, all_cores, 0U);

    // ---------------- compile-time args (chunk_gated_delta_rule_bw_common.hpp) ----
    const std::vector<uint32_t> common_ct = {
        geo.B,
        geo.T,
        geo.HV,
        geo.K,
        geo.V,
        geo.Ct,
        geo.Kt,
        geo.Vt,
        geo.Vb,
        geo.NVB,
        geo.NC,
        geo.BH,
        geo.neumann_steps,
        static_cast<uint32_t>(q.element_size()),
        dest_limit,
        static_cast<uint32_t>(has_h0),
        static_cast<uint32_t>(has_dht),
        geo.Tt,
        geo.sc_attn,
        geo.sc_kcd,
        geo.sc_p,
        geo.sc_vec,
        geo.sc_vcorr,
        geo.sc_u,
        geo.sc_c,
        geo.sc_s,
        geo.sc_ds,
        geo.sc_vnew,
        geo.sc_dvnew,
        sem_prep,
        sem_scan,
        blk.nconst,
        blk.ncol,
        blk.maxv,
        blk.lvb,
        blk.litem,
        blk.maxblk,
        blk.maxblk_g,
        kNumConstMasks,
        kNumVecASlots,
        kNumVecBSlots,
        geo.HK,
        geo.G,
        static_cast<uint32_t>(has_saved),
        static_cast<uint32_t>(d_out_is_head_major(tensor_args)),
    };

    const auto& scratch = outputs[kScratch];
    // An absent optional tensor borrows a present tensor's accessor so the CT-arg layout is the same
    // in every variant; the kernels guard each use behind HAS_H0 / HAS_DHT / HAS_SAVED.
    auto* h0_buffer = has_h0 ? tensor_args.initial_state->buffer() : q.buffer();
    auto* dht_buffer = has_dht ? tensor_args.d_final_state->buffer() : q.buffer();
    auto* dh0_buffer = has_h0 ? outputs[kDh0].buffer() : outputs[kDq].buffer();
    auto* h_buffer = has_saved ? tensor_args.h->buffer() : scratch.buffer();
    auto* v_new_buffer = has_saved ? tensor_args.v_new->buffer() : scratch.buffer();
    auto* a_buffer = has_saved ? tensor_args.a->buffer() : scratch.buffer();

    std::vector<uint32_t> reader_ct = common_ct;
    for (const auto* buffer :
         {q.buffer(),
          tensor_args.k.buffer(),
          tensor_args.v.buffer(),
          tensor_args.g.buffer(),
          tensor_args.beta.buffer(),
          tensor_args.d_out.buffer(),
          h0_buffer,
          dht_buffer,
          scratch.buffer(),
          h_buffer,
          v_new_buffer,
          a_buffer}) {
        tt::tt_metal::TensorAccessorArgs(buffer).append_to(reader_ct);
    }
    std::vector<uint32_t> writer_ct = common_ct;
    for (const auto* buffer :
         {scratch.buffer(),
          outputs[kDq].buffer(),
          outputs[kDk].buffer(),
          outputs[kDv].buffer(),
          outputs[kDg].buffer(),
          outputs[kDbeta].buffer(),
          dh0_buffer}) {
        tt::tt_metal::TensorAccessorArgs(buffer).append_to(writer_ct);
    }

    const auto reader_id = create_reader_kernel(program, all_cores, reader_ct, {}, kReaderKernelPath);
    const auto writer_id = create_writer_kernel(program, all_cores, writer_ct, {}, kWriterKernelPath);
    const auto compute_id = tt::tt_metal::CreateKernel(
        program,
        kComputeKernelPath,
        all_cores,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .dst_full_sync_en = dst_full_sync_en,
            .math_approx_mode = math_approx_mode,
            .compile_args = common_ct});

    // ---------------- runtime args ----------------------------------------
    const uint32_t scale_bits = std::bit_cast<uint32_t>(args.scale);
    std::vector<tt::tt_metal::CoreCoord> cores;
    cores.reserve(assign.size());
    for (const auto& a : assign) {
        std::vector<uint32_t> groups;  // the bh groups this core has items in
        for (uint32_t bh = a.start / geo.NC; bh <= (a.start + a.count - 1U) / geo.NC; ++bh) {
            groups.push_back(bh);
        }
        std::vector<uint32_t> owned;
        for (uint32_t bh = 0; bh < geo.BH; ++bh) {
            if (owner_of[bh] == a.core) {
                owned.push_back(bh);
            }
        }

        std::vector<uint32_t> reader_rt = {
            q.buffer()->address(),
            tensor_args.k.buffer()->address(),
            tensor_args.v.buffer()->address(),
            tensor_args.g.buffer()->address(),
            tensor_args.beta.buffer()->address(),
            tensor_args.d_out.buffer()->address(),
            has_h0 ? tensor_args.initial_state->buffer()->address() : 0U,
            has_dht ? tensor_args.d_final_state->buffer()->address() : 0U,
            scratch.buffer()->address(),
            h_buffer->address(),
            v_new_buffer->address(),
            a_buffer->address(),
            a.start,
            a.count,
            static_cast<uint32_t>(groups.size()),
            scale_bits,
            static_cast<uint32_t>(owned.size())};
        reader_rt.insert(reader_rt.end(), owned.begin(), owned.end());
        tt::tt_metal::SetRuntimeArgs(program, reader_id, a.core, reader_rt);

        tt::tt_metal::SetRuntimeArgs(program, compute_id, a.core, {a.count, static_cast<uint32_t>(owned.size())});

        std::vector<uint32_t> writer_rt = {
            scratch.buffer()->address(),
            outputs[kDq].buffer()->address(),
            outputs[kDk].buffer()->address(),
            outputs[kDv].buffer()->address(),
            outputs[kDg].buffer()->address(),
            outputs[kDbeta].buffer()->address(),
            has_h0 ? outputs[kDh0].buffer()->address() : 0U,
            a.start,
            a.count,
            static_cast<uint32_t>(groups.size())};
        // For each group this core produces prep for: (bh, owner NoC x, owner NoC y).
        for (const uint32_t bh : groups) {
            const auto owner = device->worker_core_from_logical_core(owner_of[bh]);
            writer_rt.insert(writer_rt.end(), {bh, static_cast<uint32_t>(owner.x), static_cast<uint32_t>(owner.y)});
        }
        writer_rt.push_back(static_cast<uint32_t>(owned.size()));
        writer_rt.insert(writer_rt.end(), owned.begin(), owned.end());
        // For each owned group, the stage-G cores to release.
        for (const uint32_t bh : owned) {
            writer_rt.push_back(static_cast<uint32_t>(members_of[bh].size()));
            for (const auto& member : members_of[bh]) {
                const auto phys = device->worker_core_from_logical_core(member);
                writer_rt.insert(writer_rt.end(), {static_cast<uint32_t>(phys.x), static_cast<uint32_t>(phys.y)});
            }
        }
        tt::tt_metal::SetRuntimeArgs(program, writer_id, a.core, writer_rt);
        cores.push_back(a.core);
    }

    return cached_program_t{
        std::move(program),
        shared_variables_t{.reader_kernel_id = reader_id, .writer_kernel_id = writer_id, .cores = std::move(cores)}};
}

void ChunkGatedDeltaRuleBackwardProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& outputs) {
    auto& program = cached_program.program;
    const uint32_t scale_bits = std::bit_cast<uint32_t>(operation_attributes.scale);
    const auto& shared = cached_program.shared_variables;
    const bool has_h0 = tensor_args.initial_state.has_value();
    const bool has_dht = tensor_args.d_final_state.has_value();
    const bool has_saved = tensor_args.has_saved_forward();

    auto& reader_args = GetRuntimeArgs(program, shared.reader_kernel_id);
    auto& writer_args = GetRuntimeArgs(program, shared.writer_kernel_id);
    for (const auto& core : shared.cores) {
        auto& r = reader_args[core.x][core.y];
        r[0] = tensor_args.q.buffer()->address();
        r[1] = tensor_args.k.buffer()->address();
        r[2] = tensor_args.v.buffer()->address();
        r[3] = tensor_args.g.buffer()->address();
        r[4] = tensor_args.beta.buffer()->address();
        r[5] = tensor_args.d_out.buffer()->address();
        r[6] = has_h0 ? tensor_args.initial_state->buffer()->address() : 0U;
        r[7] = has_dht ? tensor_args.d_final_state->buffer()->address() : 0U;
        r[8] = outputs[kScratch].buffer()->address();
        r[9] = has_saved ? tensor_args.h->buffer()->address() : r[8];
        r[10] = has_saved ? tensor_args.v_new->buffer()->address() : r[8];
        r[11] = has_saved ? tensor_args.a->buffer()->address() : r[8];
        r[15] = scale_bits;

        auto& w = writer_args[core.x][core.y];
        w[0] = outputs[kScratch].buffer()->address();
        w[1] = outputs[kDq].buffer()->address();
        w[2] = outputs[kDk].buffer()->address();
        w[3] = outputs[kDv].buffer()->address();
        w[4] = outputs[kDg].buffer()->address();
        w[5] = outputs[kDbeta].buffer()->address();
        w[6] = has_h0 ? outputs[kDh0].buffer()->address() : 0U;
    }
}

}  // namespace ttml::metal::ops::chunk_gated_delta_rule_bw::device
