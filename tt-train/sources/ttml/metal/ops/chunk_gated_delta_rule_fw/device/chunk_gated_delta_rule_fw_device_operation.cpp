// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "chunk_gated_delta_rule_fw_device_operation.hpp"

#include <enchantum/enchantum.hpp>

#include "ttnn/device_operation.hpp"

namespace ttml::metal::ops::chunk_gated_delta_rule_fw::device {

namespace {

// The g/beta gather reads value head h as column h of a single tile column, so HV must fit one tile.
constexpr uint32_t kMaxValueHeads = tt::constants::TILE_WIDTH;

void check_tensor(const ttnn::Tensor& tensor, const char* name) {
    TT_FATAL(
        tensor.storage_type() == ttnn::StorageType::DEVICE && tensor.buffer() != nullptr,
        "chunk_gated_delta_rule_fw: {} must be allocated on device",
        name);
    TT_FATAL(
        tensor.layout() == tt::tt_metal::Layout::TILE,
        "chunk_gated_delta_rule_fw: {} must be TILE, got {}",
        name,
        enchantum::to_string(tensor.layout()));
    TT_FATAL(
        tensor.memory_config().memory_layout() == tt::tt_metal::TensorMemoryLayout::INTERLEAVED &&
            tensor.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM,
        "chunk_gated_delta_rule_fw: {} must be DRAM interleaved",
        name);
}

void check_dtype(const ttnn::Tensor& tensor, const char* name, tt::tt_metal::DataType dtype) {
    TT_FATAL(
        tensor.dtype() == dtype,
        "chunk_gated_delta_rule_fw: {} must be {}, got {}",
        name,
        enchantum::to_string(dtype),
        enchantum::to_string(tensor.dtype()));
}

uint32_t num_worker_cores(const ttnn::Tensor& tensor) {
    const auto grid = tensor.device()->compute_with_storage_grid_size();
    return static_cast<uint32_t>(grid.x * grid.y);
}

void check_fits_l1(const std::vector<CbSpec>& plan, const char* program, uint32_t C, uint32_t K, uint32_t V) {
    const uint32_t bytes = cb_plan_bytes(plan);
    TT_FATAL(
        bytes <= cb_l1_budget(),
        "chunk_gated_delta_rule_fw: the {} at chunk_size={} K={} V={} needs {} B of L1 circular buffers, more than "
        "the {} B available",
        program,
        C,
        K,
        V,
        bytes,
        cb_l1_budget());
}

// The scan holds one V-block of the state per core; the block shrinks as B*HV leaves more cores free.
void check_scan_fits_l1(uint32_t BH, uint32_t C, uint32_t K, uint32_t V, uint32_t num_cores) {
    const uint32_t Vt = V / tt::constants::TILE_WIDTH;
    const ScanSplit split = make_scan_split(BH, Vt, num_cores);
    check_fits_l1(
        scan_cb_plan(C / tt::constants::TILE_HEIGHT, K / tt::constants::TILE_WIDTH, split.vblock_tiles),
        "scan",
        C,
        K,
        V);
}

tt::tt_metal::TensorSpec f32_spec(const ttnn::Shape& shape, const tt::tt_metal::MemoryConfig& mem) {
    return {shape, tt::tt_metal::TensorLayout(tt::tt_metal::DataType::FLOAT32, tt::tt_metal::Layout::TILE, mem)};
}

}  // namespace

// ---------------------------------------------------------------------------------------------
// Prep
// ---------------------------------------------------------------------------------------------
void ChunkGatedDeltaRuleForwardPrepDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    for (const auto& [tensor, name] :
         {std::pair{&tensor_args.q, "q"}, std::pair{&tensor_args.k, "k"}, std::pair{&tensor_args.v, "v"}}) {
        check_tensor(*tensor, name);
        check_dtype(*tensor, name, tt::tt_metal::DataType::BFLOAT16);
    }
    check_tensor(tensor_args.g, "g");
    check_tensor(tensor_args.beta, "beta");
    const auto gate_dtype = tensor_args.g.dtype();
    TT_FATAL(
        gate_dtype == tt::tt_metal::DataType::BFLOAT16 || gate_dtype == tt::tt_metal::DataType::FLOAT32,
        "chunk_gated_delta_rule_fw: g must be BFLOAT16 or FLOAT32, got {}",
        enchantum::to_string(gate_dtype));
    check_dtype(tensor_args.beta, "beta", gate_dtype);

    const auto& qs = tensor_args.q.logical_shape();
    const auto& vs = tensor_args.v.logical_shape();
    const auto& gs = tensor_args.g.logical_shape();
    TT_FATAL(
        qs.rank() == 4 && vs.rank() == 4 && gs.rank() == 4 && qs[1] == 1 && vs[1] == 1 && gs[1] == 1,
        "chunk_gated_delta_rule_fw: q/k must be [B,1,T,HK*K], v [B,1,T,HV*V], g/beta [B,1,T,HV]; got q={} v={} g={}",
        qs,
        vs,
        gs);
    const uint32_t B = qs[0];
    const uint32_t T = qs[2];
    const uint32_t HV = gs[3];
    TT_FATAL(
        tensor_args.k.logical_shape() == qs && tensor_args.beta.logical_shape() == gs && vs[0] == B && vs[2] == T &&
            gs[0] == B && gs[2] == T,
        "chunk_gated_delta_rule_fw: shapes disagree: q={} k={} v={} g={} beta={}",
        qs,
        tensor_args.k.logical_shape(),
        vs,
        gs,
        tensor_args.beta.logical_shape());
    TT_FATAL(
        HV >= 1 && HV <= kMaxValueHeads, "chunk_gated_delta_rule_fw: HV={} must be in [1, {}]", HV, kMaxValueHeads);

    const uint32_t C = args.chunk_size;
    TT_FATAL(C == 32U || C == 64U, "chunk_gated_delta_rule_fw: chunk_size must be 32 or 64, got {}", C);
    TT_FATAL(
        T % C == 0U,
        "chunk_gated_delta_rule_fw: T={} must be a multiple of chunk_size={} (whole token tiles move as pages)",
        T,
        C);

    // A page is [32 tokens x 32 dims] of one head; a head dim off the tile grid would address pages
    // straddling two heads.
    const uint32_t K = args.key_head_dim;
    TT_FATAL(vs[3] % HV == 0U, "chunk_gated_delta_rule_fw: v width {} is not a multiple of HV={}", vs[3], HV);
    const uint32_t V = vs[3] / HV;
    TT_FATAL(
        K > 0U && K % tt::constants::TILE_WIDTH == 0U && V % tt::constants::TILE_WIDTH == 0U,
        "chunk_gated_delta_rule_fw: K={} and V={} must be multiples of 32",
        K,
        V);
    TT_FATAL(qs[3] % K == 0U, "chunk_gated_delta_rule_fw: q width {} is not a multiple of K={}", qs[3], K);
    const uint32_t HK = qs[3] / K;
    TT_FATAL(
        HK >= 1U && HV % HK == 0U,
        "chunk_gated_delta_rule_fw: value heads HV={} must be a multiple of key heads HK={}",
        HV,
        HK);

    // The scan runs one core per value head (at least).
    TT_FATAL(
        B * HV <= num_worker_cores(tensor_args.q),
        "chunk_gated_delta_rule_fw: B*HV={} exceeds the {} worker cores",
        B * HV,
        num_worker_cores(tensor_args.q));

    // Each program's CB set must fit one core's L1; the scan is checked here too so an oversized
    // forward fails before the prep is dispatched.
    check_fits_l1(
        prep_cb_plan(C / tt::constants::TILE_HEIGHT, K / tt::constants::TILE_WIDTH, V / tt::constants::TILE_WIDTH),
        "prep",
        C,
        K,
        V);
    check_scan_fits_l1(B * HV, C, K, V, num_worker_cores(tensor_args.q));
}

spec_return_value_t ChunkGatedDeltaRuleForwardPrepDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& qs = tensor_args.q.logical_shape();
    const uint32_t HV = tensor_args.g.logical_shape()[3];
    const uint32_t BH = qs[0] * HV;
    const uint32_t T = qs[2];
    const uint32_t C = args.chunk_size;
    const uint32_t NC = T / C;
    const uint32_t K = args.key_head_dim;
    const uint32_t V = tensor_args.v.logical_shape()[3] / HV;
    const auto& mem = args.output_memory_config;

    spec_return_value_t specs(kNumPrepOutputs, f32_spec(ttnn::Shape({1U, 1U, 1U, 1U}), mem));
    specs[kVBeta] = f32_spec(ttnn::Shape({BH, NC, C, V}), mem);
    specs[kKd] = f32_spec(ttnn::Shape({BH, NC, C, K}), mem);
    specs[kQDecay] = f32_spec(ttnn::Shape({BH, NC, C, K}), mem);
    specs[kIntra] = f32_spec(ttnn::Shape({BH, NC, C, C}), mem);
    specs[kKDecT] = f32_spec(ttnn::Shape({BH, NC, K, C}), mem);
    specs[kDl] = f32_spec(ttnn::Shape({BH, NC, 1U, 1U}), mem);
    specs[kA] = f32_spec(ttnn::Shape({BH, 1U, T, C}), mem);
    specs[kGCumsum] = f32_spec(ttnn::Shape({BH, 1U, T, 1U}), mem);
    return specs;
}

tensor_return_value_t ChunkGatedDeltaRuleForwardPrepDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    tensor_return_value_t outputs;
    for (const auto& spec : compute_output_specs(args, tensor_args)) {
        outputs.push_back(ttnn::create_device_tensor(spec, tensor_args.q.device()));
    }
    return outputs;
}

ttsl::hash::hash_t ChunkGatedDeltaRuleForwardPrepDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    return tt::tt_metal::operation::hash_operation<ChunkGatedDeltaRuleForwardPrepDeviceOperation>(
        args.chunk_size,
        args.key_head_dim,
        args.compute_kernel_config,
        args.output_memory_config,
        tensor_args.q.logical_shape(),
        tensor_args.v.logical_shape(),
        tensor_args.g.logical_shape(),
        tensor_args.g.dtype());
}

// ---------------------------------------------------------------------------------------------
// Scan
// ---------------------------------------------------------------------------------------------
void ChunkGatedDeltaRuleForwardScanDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    for (const auto& [tensor, name] :
         {std::pair{&tensor_args.v_beta, "v_beta"},
          std::pair{&tensor_args.kd, "kd"},
          std::pair{&tensor_args.q_decay, "q_decay"},
          std::pair{&tensor_args.intra, "intra"},
          std::pair{&tensor_args.k_dec_t, "k_dec_t"},
          std::pair{&tensor_args.dl, "dl"},
          std::pair{&tensor_args.a, "a"}}) {
        check_tensor(*tensor, name);
        check_dtype(*tensor, name, tt::tt_metal::DataType::FLOAT32);
    }
    const auto& vbs = tensor_args.v_beta.logical_shape();
    const auto& kds = tensor_args.kd.logical_shape();
    TT_FATAL(vbs.rank() == 4 && kds.rank() == 4, "chunk_gated_delta_rule_fw: v_beta and kd must be rank 4");
    const uint32_t BH = vbs[0];
    const uint32_t NC = vbs[1];
    const uint32_t C = vbs[2];
    const uint32_t V = vbs[3];
    const uint32_t K = kds[3];
    const uint32_t HV = args.num_value_heads;
    TT_FATAL(HV >= 1U && BH % HV == 0U, "chunk_gated_delta_rule_fw: BH={} is not a multiple of HV={}", BH, HV);
    TT_FATAL(
        kds == ttnn::Shape({BH, NC, C, K}) && tensor_args.q_decay.logical_shape() == kds &&
            tensor_args.intra.logical_shape() == ttnn::Shape({BH, NC, C, C}) &&
            tensor_args.k_dec_t.logical_shape() == ttnn::Shape({BH, NC, K, C}) &&
            tensor_args.dl.logical_shape() == ttnn::Shape({BH, NC, 1U, 1U}) &&
            tensor_args.a.logical_shape() == ttnn::Shape({BH, 1U, NC * C, C}),
        "chunk_gated_delta_rule_fw: scan inputs disagree with v_beta {}",
        vbs);
    TT_FATAL(
        BH <= num_worker_cores(tensor_args.v_beta),
        "chunk_gated_delta_rule_fw: B*HV={} exceeds the {} worker cores",
        BH,
        num_worker_cores(tensor_args.v_beta));
    check_scan_fits_l1(BH, C, K, V, num_worker_cores(tensor_args.v_beta));
    if (tensor_args.initial_state.has_value()) {
        const auto& h0 = *tensor_args.initial_state;
        check_tensor(h0, "initial_state");
        check_dtype(h0, "initial_state", tt::tt_metal::DataType::FLOAT32);
        const ttnn::Shape state_shape({BH / HV, HV, K, V});
        TT_FATAL(
            h0.logical_shape() == state_shape,
            "chunk_gated_delta_rule_fw: initial_state must be {}, got {}",
            state_shape,
            h0.logical_shape());
    }
}

spec_return_value_t ChunkGatedDeltaRuleForwardScanDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& vbs = tensor_args.v_beta.logical_shape();
    const uint32_t BH = vbs[0];
    const uint32_t NC = vbs[1];
    const uint32_t C = vbs[2];
    const uint32_t V = vbs[3];
    const uint32_t K = tensor_args.kd.logical_shape()[3];
    const uint32_t HV = args.num_value_heads;
    const uint32_t B = BH / HV;
    const auto& mem = args.output_memory_config;

    spec_return_value_t specs(kNumScanOutputs, f32_spec(ttnn::Shape({1U, 1U, 1U, 1U}), mem));
    specs[kO] =
        f32_spec(args.output_head_major ? ttnn::Shape({BH, 1U, NC * C, V}) : ttnn::Shape({B, 1U, NC * C, HV * V}), mem);
    specs[kFinalState] = f32_spec(ttnn::Shape({B, HV, K, V}), mem);
    specs[kH] = f32_spec(ttnn::Shape({BH, NC, K, V}), mem);
    specs[kVNew] = f32_spec(ttnn::Shape({BH, 1U, NC * C, V}), mem);
    return specs;
}

tensor_return_value_t ChunkGatedDeltaRuleForwardScanDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    tensor_return_value_t outputs;
    for (const auto& spec : compute_output_specs(args, tensor_args)) {
        outputs.push_back(ttnn::create_device_tensor(spec, tensor_args.v_beta.device()));
    }
    return outputs;
}

ttsl::hash::hash_t ChunkGatedDeltaRuleForwardScanDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    return tt::tt_metal::operation::hash_operation<ChunkGatedDeltaRuleForwardScanDeviceOperation>(
        args.num_value_heads,
        args.output_head_major,
        args.compute_kernel_config,
        args.output_memory_config,
        tensor_args.v_beta.logical_shape(),
        tensor_args.kd.logical_shape(),
        tensor_args.initial_state.has_value());
}

}  // namespace ttml::metal::ops::chunk_gated_delta_rule_fw::device

namespace ttnn::prim {

ttml::metal::ops::chunk_gated_delta_rule_fw::device::tensor_return_value_t ttml_chunk_gated_delta_rule_fw_prep(
    const ttnn::Tensor& q,
    const ttnn::Tensor& k,
    const ttnn::Tensor& v,
    const ttnn::Tensor& g,
    const ttnn::Tensor& beta,
    uint32_t chunk_size,
    uint32_t key_head_dim,
    const ttnn::DeviceComputeKernelConfig& compute_kernel_config,
    const tt::tt_metal::MemoryConfig& output_memory_config) {
    using OperationType =
        ttml::metal::ops::chunk_gated_delta_rule_fw::device::ChunkGatedDeltaRuleForwardPrepDeviceOperation;
    auto attributes = OperationType::operation_attributes_t{
        .chunk_size = chunk_size,
        .key_head_dim = key_head_dim,
        .compute_kernel_config = compute_kernel_config,
        .output_memory_config = output_memory_config};
    auto tensor_args = OperationType::tensor_args_t{.q = q, .k = k, .v = v, .g = g, .beta = beta};
    return ttnn::device_operation::launch<OperationType>(attributes, tensor_args);
}

ttml::metal::ops::chunk_gated_delta_rule_fw::device::tensor_return_value_t ttml_chunk_gated_delta_rule_fw_scan(
    const ttml::metal::ops::chunk_gated_delta_rule_fw::device::tensor_return_value_t& prep,
    const std::optional<ttnn::Tensor>& initial_state,
    uint32_t num_value_heads,
    bool output_head_major,
    const ttnn::DeviceComputeKernelConfig& compute_kernel_config,
    const tt::tt_metal::MemoryConfig& output_memory_config) {
    using namespace ttml::metal::ops::chunk_gated_delta_rule_fw::device;
    using OperationType = ChunkGatedDeltaRuleForwardScanDeviceOperation;
    auto attributes = OperationType::operation_attributes_t{
        .num_value_heads = num_value_heads,
        .output_head_major = output_head_major,
        .compute_kernel_config = compute_kernel_config,
        .output_memory_config = output_memory_config};
    auto tensor_args = OperationType::tensor_args_t{
        .v_beta = prep[kVBeta],
        .kd = prep[kKd],
        .q_decay = prep[kQDecay],
        .intra = prep[kIntra],
        .k_dec_t = prep[kKDecT],
        .dl = prep[kDl],
        .a = prep[kA],
        .initial_state = initial_state};
    return ttnn::device_operation::launch<OperationType>(attributes, tensor_args);
}

}  // namespace ttnn::prim
