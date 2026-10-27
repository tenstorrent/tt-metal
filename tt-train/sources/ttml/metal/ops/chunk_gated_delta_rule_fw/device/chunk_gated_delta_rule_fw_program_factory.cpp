// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "chunk_gated_delta_rule_fw_program_factory.hpp"

#include <algorithm>
#include <set>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

#include "kernels/chunk_gated_delta_rule_fw_common.hpp"
#include "metal/common/program_utils.hpp"

namespace {

constexpr auto kKernelDir = "tt-train/sources/ttml/metal/ops/chunk_gated_delta_rule_fw/device/kernels/";
const std::string kPrepReaderPath = std::string(kKernelDir) + "dataflow/reader_chunk_gated_delta_rule_fw_prep.cpp";
const std::string kPrepWriterPath = std::string(kKernelDir) + "dataflow/writer_chunk_gated_delta_rule_fw_prep.cpp";
const std::string kPrepComputePath = std::string(kKernelDir) + "compute/chunk_gated_delta_rule_fw_prep_kernel.cpp";
const std::string kScanReaderPath = std::string(kKernelDir) + "dataflow/reader_chunk_gated_delta_rule_fw_scan.cpp";
const std::string kScanWriterPath = std::string(kKernelDir) + "dataflow/writer_chunk_gated_delta_rule_fw_scan.cpp";
const std::string kScanComputePath = std::string(kKernelDir) + "compute/chunk_gated_delta_rule_fw_scan_kernel.cpp";

// L1 below the circular-buffer region that get_max_worker_l1_unreserved_size() still counts: the
// kernel-config ring buffer the kernel binaries live in, plus slack.
constexpr uint32_t kL1KernelConfigReserve = 80U * 1024U;

// Double-buffered: the scan's o egress.
constexpr uint32_t kDoubleBuffer = 2U;

uint32_t scratch_tiles(uint32_t Ct, uint32_t Kt, uint32_t Vt) {
    return std::max({Ct * Ct, Ct * Kt, Ct * Vt, Kt * Vt, Kt * Ct});
}

void create_cbs(
    tt::tt_metal::Program& program,
    const tt::tt_metal::CoreRangeSet& cores,
    const std::vector<ttml::metal::ops::chunk_gated_delta_rule_fw::device::CbSpec>& plan) {
    for (const auto& cb : plan) {
        const auto format = cb.bf16 ? tt::DataFormat::Float16_b : tt::DataFormat::Float32;
        create_circular_buffer(program, cores, cb.index, format, tt::tile_size(format), cb.tiles);
    }
}

tt::tt_metal::KernelHandle create_compute(
    tt::tt_metal::Program& program,
    const tt::tt_metal::CoreRangeSet& cores,
    const std::string& path,
    const std::vector<uint32_t>& ct_args,
    const ttnn::DeviceComputeKernelConfig& config,
    tt::ARCH arch) {
    const auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(arch, config);
    return tt::tt_metal::CreateKernel(
        program,
        path,
        cores,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .dst_full_sync_en = dst_full_sync_en,
            .math_approx_mode = math_approx_mode,
            .compile_args = ct_args});
}

}  // namespace

namespace ttml::metal::ops::chunk_gated_delta_rule_fw::device {

std::vector<CbSpec> prep_cb_plan(uint32_t Ct, uint32_t Kt, uint32_t Vt) {
    using namespace ::chunk_gated_delta_rule_fw::prep;
    const uint32_t cc = Ct * Ct, ck = Ct * Kt, cv = Ct * Vt, kc = Kt * Ct;
    const uint32_t scr = scratch_tiles(Ct, Kt, Vt);
    // cb_S/cb_final/cb_s2/cb_s3 (the WY inverse's private scratch) and cb_supd/cb_stmp/cb_ointer
    // (the Ct == 2 block merge) never hold more than one tile in the prep.
    return {
        {cb_q, ck, true},       {cb_k, ck, true},          {cb_v, cv, true},
        {cb_g, Ct, false},      {cb_beta, Ct, false},      {cb_eye, cc, false},
        {cb_tril, cc, false},   {cb_ones, cc, false},      {cb_S, 1U, false},
        {cb_decay, Ct, false},  {cb_decay_exp, Ct, false}, {cb_decayfac, Ct, false},
        {cb_lmask, cc, false},  {cb_Tinv, cc, false},      {cb_vbeta, cv, false},
        {cb_kbeta, ck, false},  {cb_gcum, Ct, false},      {cb_mask, kNumMasks, false},
        {cb_kd, ck, false},     {cb_qdecay, ck, false},    {cb_intra, cc, false},
        {cb_s2, 1U, false},     {cb_dl, 1U, false},        {cb_ointer, 1U, false},
        {cb_kdec_t, kc, false}, {cb_supd, 1U, false},      {cb_stmp, 1U, false},
        {cb_final, 1U, false},  {cb_scr1, scr, false},     {cb_scr2, scr, false},
        {cb_scr3, scr, false},  {cb_s3, 1U, false},
    };
}

std::vector<CbSpec> scan_cb_plan(uint32_t Ct, uint32_t Kt, uint32_t Vt_block) {
    using namespace ::chunk_gated_delta_rule_fw::scan;
    const uint32_t cc = Ct * Ct, ck = Ct * Kt, cv = Ct * Vt_block, kv = Kt * Vt_block, kc = Kt * Ct;
    return {
        {cb_vbeta, cv, false},
        {cb_kd, ck, false},
        {cb_qdecay, ck, false},
        {cb_intra, cc, false},
        {cb_kdec_t, kc, false},
        {cb_dl, 1U, false},
        {cb_Tinv, cc, false},
        {cb_S, kv, false},
        {cb_s2, kv, false},
        {cb_s3, kv, false},
        {cb_out, cv * kDoubleBuffer, false},
        {cb_final, kv, false},
        {cb_vnew, cv, false},
        {cb_ointer, cv, false},
        {cb_supd, kv, false},
        {cb_stmp, kv, false},
        {cb_scr1, scratch_tiles(Ct, Kt, Vt_block), false},
        {cb_hout, kv, false},
        {cb_vnout, cv, false},
        {cb_h0, kv, false},
    };
}

uint32_t cb_plan_bytes(const std::vector<CbSpec>& plan) {
    uint32_t bytes = 0U;
    for (const auto& cb : plan) {
        bytes += cb.tiles * tt::tile_size(cb.bf16 ? tt::DataFormat::Float16_b : tt::DataFormat::Float32);
    }
    return bytes;
}

uint32_t cb_l1_budget() {
    return static_cast<uint32_t>(tt::tt_metal::hal::get_max_worker_l1_unreserved_size()) - kL1KernelConfigReserve;
}

ScanSplit make_scan_split(uint32_t BH, uint32_t Vt, uint32_t num_cores) {
    for (uint32_t nv = Vt; nv > 1U; --nv) {
        if (Vt % nv == 0U && BH * nv <= num_cores) {
            return {nv, Vt / nv};
        }
    }
    return {1U, Vt};
}

// ---------------------------------------------------------------------------------------------
// Prep
// ---------------------------------------------------------------------------------------------
ChunkGatedDeltaRuleForwardPrepProgramFactory::cached_program_t ChunkGatedDeltaRuleForwardPrepProgramFactory::create(
    const ChunkGatedDeltaRuleForwardPrepParams& args,
    const ChunkGatedDeltaRuleForwardPrepInputs& tensor_args,
    tensor_return_value_t& outputs) {
    const auto& q = tensor_args.q;
    auto* device = q.device();
    tt::tt_metal::Program program{};

    const auto& qs = q.logical_shape();
    const uint32_t B = qs[0];
    const uint32_t T = qs[2];
    const uint32_t HV = tensor_args.g.logical_shape()[3];
    const uint32_t K = args.key_head_dim;
    const uint32_t HK = qs[3] / K;
    const uint32_t V = tensor_args.v.logical_shape()[3] / HV;
    const uint32_t Ct = args.chunk_size / tt::constants::TILE_HEIGHT;
    const uint32_t Kt = K / tt::constants::TILE_WIDTH;
    const uint32_t Vt = V / tt::constants::TILE_WIDTH;
    const uint32_t NC = T / args.chunk_size;
    const uint32_t num_items = B * HV * NC;

    // Work item wi = bh * NC + c; every item is independent, so the whole grid shares them.
    const auto grid = device->compute_with_storage_grid_size();
    const auto [num_cores, all_cores, group1, group2, items1, items2] =
        tt::tt_metal::split_work_to_cores(grid, num_items, /*row_wise=*/true);

    create_cbs(program, all_cores, prep_cb_plan(Ct, Kt, Vt));

    std::vector<uint32_t> reader_ct = {Ct, Kt, Vt, NC, HV, HK, static_cast<uint32_t>(tensor_args.g.element_size())};
    for (const auto* buffer :
         {q.buffer(),
          tensor_args.k.buffer(),
          tensor_args.v.buffer(),
          tensor_args.g.buffer(),
          tensor_args.beta.buffer()}) {
        tt::tt_metal::TensorAccessorArgs(buffer).append_to(reader_ct);
    }
    std::vector<uint32_t> writer_ct = {Ct, Kt, Vt};
    for (const auto& out : outputs) {
        tt::tt_metal::TensorAccessorArgs(out.buffer()).append_to(writer_ct);
    }
    const std::vector<uint32_t> compute_ct = {Ct, Kt, Vt};

    const auto reader_id = create_reader_kernel(program, all_cores, reader_ct, {}, kPrepReaderPath);
    const auto writer_id = create_writer_kernel(program, all_cores, writer_ct, {}, kPrepWriterPath);
    const auto compute_id =
        create_compute(program, all_cores, kPrepComputePath, compute_ct, args.compute_kernel_config, device->arch());

    std::vector<tt::tt_metal::CoreCoord> cores;
    cores.reserve(num_cores);
    uint32_t start = 0U;
    for (const auto& [group, count] : {std::pair{group1, items1}, std::pair{group2, items2}}) {
        if (count == 0U) {
            continue;
        }
        for (const auto& core : tt::tt_metal::corerange_to_cores(group, std::nullopt, /*row_wise=*/true)) {
            tt::tt_metal::SetRuntimeArgs(
                program,
                reader_id,
                core,
                {start,
                 count,
                 q.buffer()->address(),
                 tensor_args.k.buffer()->address(),
                 tensor_args.v.buffer()->address(),
                 tensor_args.g.buffer()->address(),
                 tensor_args.beta.buffer()->address()});
            std::vector<uint32_t> writer_rt = {start, count};
            writer_rt.reserve(2U + outputs.size());
            for (const auto& out : outputs) {
                writer_rt.push_back(out.buffer()->address());
            }
            tt::tt_metal::SetRuntimeArgs(program, writer_id, core, writer_rt);
            tt::tt_metal::SetRuntimeArgs(program, compute_id, core, {count});
            cores.push_back(core);
            start += count;
        }
    }
    TT_FATAL(start == num_items, "chunk_gated_delta_rule_fw: work split lost items: {} != {}", start, num_items);

    return cached_program_t{
        std::move(program),
        shared_variables_t{.reader_kernel_id = reader_id, .writer_kernel_id = writer_id, .cores = std::move(cores)}};
}

void ChunkGatedDeltaRuleForwardPrepProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const ChunkGatedDeltaRuleForwardPrepParams&,
    const ChunkGatedDeltaRuleForwardPrepInputs& tensor_args,
    tensor_return_value_t& outputs) {
    auto& program = cached_program.program;
    const auto& shared = cached_program.shared_variables;
    auto& reader_args = GetRuntimeArgs(program, shared.reader_kernel_id);
    auto& writer_args = GetRuntimeArgs(program, shared.writer_kernel_id);
    for (const auto& core : shared.cores) {
        auto& r = reader_args[core.x][core.y];
        r[2] = tensor_args.q.buffer()->address();
        r[3] = tensor_args.k.buffer()->address();
        r[4] = tensor_args.v.buffer()->address();
        r[5] = tensor_args.g.buffer()->address();
        r[6] = tensor_args.beta.buffer()->address();
        auto& w = writer_args[core.x][core.y];
        for (size_t i = 0; i < outputs.size(); ++i) {
            w[2U + i] = outputs[i].buffer()->address();
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Scan
// ---------------------------------------------------------------------------------------------
ChunkGatedDeltaRuleForwardScanProgramFactory::cached_program_t ChunkGatedDeltaRuleForwardScanProgramFactory::create(
    const ChunkGatedDeltaRuleForwardScanParams& args,
    const ChunkGatedDeltaRuleForwardScanInputs& tensor_args,
    tensor_return_value_t& outputs) {
    auto* device = tensor_args.v_beta.device();
    tt::tt_metal::Program program{};

    const auto& vbs = tensor_args.v_beta.logical_shape();  // [BH, NC, C, V]
    const uint32_t BH = vbs[0];
    const uint32_t NC = vbs[1];
    const uint32_t Ct = vbs[2] / tt::constants::TILE_HEIGHT;
    const uint32_t Vt = vbs[3] / tt::constants::TILE_WIDTH;
    const uint32_t Kt = tensor_args.kd.logical_shape()[3] / tt::constants::TILE_WIDTH;
    const bool has_h0 = tensor_args.initial_state.has_value();

    // The recurrence is sequential in time but splits exactly over V: every scan step is column-wise
    // in V and only needs the full-K per-chunk blocks plus that V-block of the state.
    const auto grid = device->compute_with_storage_grid_size();
    const ScanSplit split = make_scan_split(BH, Vt, grid.x * grid.y);
    const uint32_t num_scans = BH * split.num_vblocks;
    std::vector<tt::tt_metal::CoreCoord> cores;
    cores.reserve(num_scans);
    std::set<tt::tt_metal::CoreRange> ranges;
    for (uint32_t i = 0; i < num_scans; ++i) {
        const tt::tt_metal::CoreCoord core{i % grid.x, i / grid.x};
        cores.push_back(core);
        ranges.insert(tt::tt_metal::CoreRange{core, core});
    }
    const tt::tt_metal::CoreRangeSet all_cores{ranges};

    create_cbs(program, all_cores, scan_cb_plan(Ct, Kt, split.vblock_tiles));

    const std::vector<uint32_t> common_ct = {
        Ct,
        Kt,
        split.vblock_tiles,
        Vt,
        static_cast<uint32_t>(has_h0),
        args.num_value_heads,
        NC,
        static_cast<uint32_t>(args.output_head_major)};
    std::vector<uint32_t> reader_ct = common_ct;
    // An absent initial state borrows v_beta's accessor so the compile-time-arg layout is fixed; the
    // reader never touches it without HAS_H0.
    auto* h0_buffer = has_h0 ? tensor_args.initial_state->buffer() : tensor_args.v_beta.buffer();
    for (const auto* buffer :
         {tensor_args.v_beta.buffer(),
          tensor_args.kd.buffer(),
          tensor_args.q_decay.buffer(),
          tensor_args.intra.buffer(),
          tensor_args.k_dec_t.buffer(),
          tensor_args.dl.buffer(),
          tensor_args.a.buffer(),
          h0_buffer}) {
        tt::tt_metal::TensorAccessorArgs(buffer).append_to(reader_ct);
    }
    std::vector<uint32_t> writer_ct = common_ct;
    for (const auto& out : outputs) {
        tt::tt_metal::TensorAccessorArgs(out.buffer()).append_to(writer_ct);
    }

    const auto reader_id = create_reader_kernel(program, all_cores, reader_ct, {}, kScanReaderPath);
    const auto writer_id = create_writer_kernel(program, all_cores, writer_ct, {}, kScanWriterPath);
    create_compute(program, all_cores, kScanComputePath, common_ct, args.compute_kernel_config, device->arch());

    for (uint32_t i = 0; i < num_scans; ++i) {
        const uint32_t bh = i / split.num_vblocks;
        const uint32_t vb = i % split.num_vblocks;
        tt::tt_metal::SetRuntimeArgs(
            program,
            reader_id,
            cores[i],
            {bh,
             vb,
             tensor_args.v_beta.buffer()->address(),
             tensor_args.kd.buffer()->address(),
             tensor_args.q_decay.buffer()->address(),
             tensor_args.intra.buffer()->address(),
             tensor_args.k_dec_t.buffer()->address(),
             tensor_args.dl.buffer()->address(),
             tensor_args.a.buffer()->address(),
             has_h0 ? tensor_args.initial_state->buffer()->address() : 0U});
        std::vector<uint32_t> writer_rt = {bh, vb};
        writer_rt.reserve(2U + outputs.size());
        for (const auto& out : outputs) {
            writer_rt.push_back(out.buffer()->address());
        }
        tt::tt_metal::SetRuntimeArgs(program, writer_id, cores[i], writer_rt);
    }

    return cached_program_t{
        std::move(program),
        shared_variables_t{.reader_kernel_id = reader_id, .writer_kernel_id = writer_id, .cores = std::move(cores)}};
}

void ChunkGatedDeltaRuleForwardScanProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const ChunkGatedDeltaRuleForwardScanParams&,
    const ChunkGatedDeltaRuleForwardScanInputs& tensor_args,
    tensor_return_value_t& outputs) {
    auto& program = cached_program.program;
    const auto& shared = cached_program.shared_variables;
    auto& reader_args = GetRuntimeArgs(program, shared.reader_kernel_id);
    auto& writer_args = GetRuntimeArgs(program, shared.writer_kernel_id);
    const bool has_h0 = tensor_args.initial_state.has_value();
    for (const auto& core : shared.cores) {
        auto& r = reader_args[core.x][core.y];
        r[2] = tensor_args.v_beta.buffer()->address();
        r[3] = tensor_args.kd.buffer()->address();
        r[4] = tensor_args.q_decay.buffer()->address();
        r[5] = tensor_args.intra.buffer()->address();
        r[6] = tensor_args.k_dec_t.buffer()->address();
        r[7] = tensor_args.dl.buffer()->address();
        r[8] = tensor_args.a.buffer()->address();
        r[9] = has_h0 ? tensor_args.initial_state->buffer()->address() : 0U;
        auto& w = writer_args[core.x][core.y];
        for (size_t i = 0; i < outputs.size(); ++i) {
            w[2U + i] = outputs[i].buffer()->address();
        }
    }
}

}  // namespace ttml::metal::ops::chunk_gated_delta_rule_fw::device
