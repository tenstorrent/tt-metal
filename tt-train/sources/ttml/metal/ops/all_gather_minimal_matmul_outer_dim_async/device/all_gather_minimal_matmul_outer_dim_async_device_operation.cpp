// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "all_gather_minimal_matmul_outer_dim_async_device_operation.hpp"

#include <string_view>
#include <tt-metalium/data_formats/tile_constants.hpp>
#include <tt-metalium/utils/math.hpp>
#include <utility>

#include "kernels/all_gather_minimal_matmul_outer_dim_async_utils.hpp"
#include "metal/common/tensor_validation.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/ccl/common/host/ccl_topology_utils.hpp"
#include "ttnn/operations/ccl/common/host/mesh_ring_plan.hpp"
#include "ttnn/operations/ccl/common/host/moe_utils.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;

namespace {
constexpr std::string_view kOp = "all_gather_minimal_matmul_outer_dim_async";

// Where K and the outer dimension (M of the first operand, N of the second) sit in the input's and the gathered
// tensor's shapes. The first operand is stored [M, K] and the second [N, K] with transpose_b, else [K, N]; dim=-2
// makes the gathered tensor the first operand, dim=-1 the second.
struct OperandDims {
    bool gather_in0;
    int input_outer;
    int input_k;
    int gathered_outer;  // the dim the gathered tensor is sharded along
    int gathered_k;
};

OperandDims operand_dims(
    const ttml::metal::ops::all_gather_minimal_matmul_outer_dim_async::AllGatherMinimalMatmulOuterDimAsyncParams&
        attrs) {
    const bool gather_in0 = attrs.gather_in0();
    const int in1_outer = attrs.transpose_b ? -2 : -1;
    const int in1_k = attrs.transpose_b ? -1 : -2;
    return {
        .gather_in0 = gather_in0,
        .input_outer = gather_in0 ? in1_outer : -2,
        .input_k = gather_in0 ? in1_k : -1,
        .gathered_outer = gather_in0 ? -2 : in1_outer,
        .gathered_k = gather_in0 ? -1 : in1_k};
}
}  // namespace

namespace ttml::metal::ops::all_gather_minimal_matmul_outer_dim_async {
AllGatherMinimalMatmulOuterDimAsyncDeviceOperation::program_factory_t
AllGatherMinimalMatmulOuterDimAsyncDeviceOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return AllGatherMinimalMatmulOuterDimAsyncProgramFactory{};
}

void AllGatherMinimalMatmulOuterDimAsyncDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    const auto& x = tensor_args.input_tensor;
    const auto& g = tensor_args.gathered_tensor;
    const auto& pgb = tensor_args.persistent_gather_buffer;
    const DeviceTensorRequirements req{.dtypes = {DataType::BFLOAT16}, .memory_layout = std::nullopt};
    check_device_tensor(x, kOp, "input", req);
    check_device_tensor(g, kOp, "gathered", req);
    check_device_tensor(pgb, kOp, "persistent_gather_buffer", req);
    check_tile_padding(x, kOp, "input");
    check_tile_padding(g, kOp, "gathered");
    check_same_device(g, x, kOp, "gathered", "input");
    check_same_device(pgb, x, kOp, "persistent_gather_buffer", "input");

    TT_FATAL(
        attrs.dim == -1 || attrs.dim == -2,
        "{}: dim must be -1 (the gathered tensor is the second operand, sharded along its columns) or -2 (it is the "
        "first operand, sharded along its rows), got {}.",
        kOp,
        attrs.dim);
    const OperandDims dims = operand_dims(attrs);
    const auto& x_shape = x.logical_shape();
    const auto& g_shape = g.logical_shape();
    TT_FATAL(
        x_shape.rank() >= 2 && g_shape.rank() >= 2,
        "{}: input and gathered tensor must have rank >= 2, got ranks {} and {}.",
        kOp,
        x_shape.rank(),
        g_shape.rank());
    auto check_matrix = [](const ttnn::Shape& shape, std::string_view name) {
        for (int i = 0; i < static_cast<int>(shape.rank()) - 2; ++i) {
            TT_FATAL(
                shape[i] == 1,
                "{}: the {} tensor must be a matrix: every dim before the last two must be 1, but dim {} is {} (shape "
                "{}).",
                kOp,
                name,
                i,
                shape[i],
                shape);
        }
    };
    check_matrix(g_shape, "gathered");
    if (dims.gather_in0) {
        check_matrix(x_shape, "input");  // as the second operand it has no leading dims to fold into M
    }
    TT_FATAL(
        x_shape[dims.input_k] == g_shape[dims.gathered_k],
        "{}: inner dimensions must match: input K={} vs gathered K={} (dim={}, transpose_b={})",
        kOp,
        x_shape[dims.input_k],
        g_shape[dims.gathered_k],
        attrs.dim,
        attrs.transpose_b);
    TT_FATAL(
        x_shape[dims.input_outer] > 0 && x_shape[dims.input_k] > 0 && g_shape[dims.gathered_outer] > 0,
        "{}: matmul dims must be positive, got input shape {} and gathered shape {}.",
        kOp,
        x_shape,
        g_shape);

    TT_FATAL(
        attrs.topology == ttnn::ccl::Topology::Ring || attrs.topology == ttnn::ccl::Topology::Linear,
        "{}: unsupported topology {}: the op supports Ring and Linear.",
        kOp,
        attrs.topology);

    // The buffer holds the whole gathered tensor: the shard's shape with the gathered dim grown by the ring size.
    ttnn::Shape pgb_shape(g_shape);
    pgb_shape[dims.gathered_outer] = g_shape[dims.gathered_outer] * attrs.ring_size;
    check_shape(pgb, kOp, "persistent_gather_buffer", pgb_shape);
    TT_FATAL(
        g_shape[dims.gathered_outer] % TILE_WIDTH == 0,
        "{}: the gathered shard's dim {} ({}) must be tile-aligned",
        kOp,
        dims.gathered_outer,
        g_shape[dims.gathered_outer]);
    TT_FATAL(
        attrs.ring_size >= 2,
        "{}: the fabric gather needs a ring of at least 2 devices, got ring_size={}.",
        kOp,
        attrs.ring_size);
    TT_FATAL(
        attrs.topology != ttnn::ccl::Topology::Ring || attrs.ring_size > 2,
        "{}: a ring of 2 devices has the same neighbour in both directions, so both senders of a link would share one "
        "fabric router channel; use Topology::Linear",
        kOp);
    TT_FATAL(
        attrs.semaphores.size() == attrs.ring_size,
        "{}: multi_device_global_semaphore must hold one semaphore per ring device ({}), got {}",
        kOp,
        attrs.ring_size,
        attrs.semaphores.size());
    const ttnn::CoreCoord device_grid = x.device()->compute_with_storage_grid_size();
    const ttnn::CoreRange all_cores({0, 0}, {device_grid.x - 1, device_grid.y - 1});
    for (size_t i = 0; i < attrs.semaphores.size(); ++i) {
        const auto& sem = attrs.semaphores[i];
        TT_FATAL(
            sem.device() == x.device(), "{}: multi_device_global_semaphore must belong to the input mesh device", kOp);
        TT_FATAL(
            std::get<0>(sem.attribute_values()).contains(all_cores),
            "{}: multi_device_global_semaphore must be created on the whole compute grid {}",
            kOp,
            all_cores.str());
        // One counter per stripe: a semaphore listed twice would let one stripe's chunks satisfy another's wait.
        for (size_t j = 0; j < i; ++j) {
            TT_FATAL(
                attrs.semaphores[j].address() != sem.address(),
                "{}: multi_device_global_semaphore lists the same semaphore twice (entries {} and {})",
                kOp,
                j,
                i);
        }
    }
    const auto& mesh_shape = x.device()->shape();
    uint32_t non_unit_dims = 0;
    for (size_t i = 0; i < mesh_shape.dims(); ++i) {
        non_unit_dims += mesh_shape[i] > 1 ? 1 : 0;
    }
    TT_FATAL(
        attrs.cluster_axis.has_value() || non_unit_dims <= 1,
        "{}: cluster_axis must be set on a multi-dimensional mesh ({}): it names the axis the tensor is gathered over",
        kOp,
        mesh_shape);
    TT_FATAL(
        attrs.topology != ttnn::ccl::Topology::Ring ||
            (attrs.cluster_axis.has_value() && ttnn::ccl::is_axis_wrap_wired(*x.device(), attrs.cluster_axis.value())),
        "{}: topology=Ring needs cluster_axis set and its wrap-around link wired on this mesh; use Topology::Linear",
        kOp);

    if (attrs.config.has_value()) {
        const auto& cfg = attrs.config.value();
        TT_FATAL(
            cfg.M_block_size > 0 && cfg.K_block_size > 0 && cfg.N_block_size > 0,
            "{}: block sizes must be > 0, got M_block_size={}, K_block_size={}, N_block_size={}.",
            kOp,
            cfg.M_block_size,
            cfg.K_block_size,
            cfg.N_block_size);
        TT_FATAL(
            cfg.subblock_h > 0 && cfg.subblock_w > 0,
            "{}: subblock sizes must be > 0, got subblock_h={}, subblock_w={}.",
            kOp,
            cfg.subblock_h,
            cfg.subblock_w);
        TT_FATAL(
            cfg.M_block_size % cfg.subblock_h == 0,
            "{}: M_block_size ({}) must be divisible by subblock_h ({}).",
            kOp,
            cfg.M_block_size,
            cfg.subblock_h);
        TT_FATAL(
            cfg.N_block_size % cfg.subblock_w == 0,
            "{}: N_block_size ({}) must be divisible by subblock_w ({}).",
            kOp,
            cfg.N_block_size,
            cfg.subblock_w);
        TT_FATAL(
            cfg.compute_with_storage_grid_size.x >= 2 && cfg.compute_with_storage_grid_size.y >= 2,
            "{}: compute_with_storage_grid_size must be at least 2x2, got {}x{}.",
            kOp,
            cfg.compute_with_storage_grid_size.x,
            cfg.compute_with_storage_grid_size.y);
        TT_FATAL(
            cfg.compute_with_storage_grid_size.x <= device_grid.x &&
                cfg.compute_with_storage_grid_size.y <= device_grid.y,
            "{}: compute_with_storage_grid_size ({}x{}) must fit in the device grid ({}x{}).",
            kOp,
            cfg.compute_with_storage_grid_size.x,
            cfg.compute_with_storage_grid_size.y,
            device_grid.x,
            device_grid.y);
        TT_FATAL(
            cfg.subblock_h * cfg.subblock_w <= get_dest_reg_count(attrs.compute_kernel_config),
            "{}: subblock {}x{} needs {} destination registers, but this compute_kernel_config has {}. Use a smaller "
            "subblock_h or subblock_w.",
            kOp,
            cfg.subblock_h,
            cfg.subblock_w,
            cfg.subblock_h * cfg.subblock_w,
            get_dest_reg_count(attrs.compute_kernel_config));
    }

    TT_FATAL(attrs.num_links >= 1, "{}: num_links must be >= 1, got {}.", kOp, attrs.num_links);
    const ttnn::CoreCoord matmul_grid =
        attrs.config.has_value() ? attrs.config->compute_with_storage_grid_size : device_grid;
    TT_FATAL(
        2 * attrs.num_links <= matmul_grid.x,
        "{}: num_links={} needs {} fabric senders (one per link and direction) in one row of the matmul grid, which "
        "has {} cores",
        kOp,
        attrs.num_links,
        2 * attrs.num_links,
        matmul_grid.x);
    const uint32_t K_block_tiles = attrs.config.has_value() ? attrs.config->K_block_size : kDefaultBlockTiles;
    const uint32_t K_blocks = tt::div_up(x.padded_shape()[dims.input_k] / TILE_WIDTH, K_block_tiles);
    TT_FATAL(
        2 * K_blocks <= kLandedCountMask,
        "{}: {} K blocks overflow the stripe semaphores' 16-bit counters; use a larger K_block_size",
        kOp,
        K_blocks);
    const size_t available_links = ttnn::operations::ccl::common::get_num_links(*x.device(), attrs.cluster_axis);
    TT_FATAL(
        attrs.num_links <= available_links,
        "{}: num_links={} but the ring's axis has only {} fabric links per direction",
        kOp,
        attrs.num_links,
        available_links);
}

AllGatherMinimalMatmulOuterDimAsyncDeviceOperation::spec_return_value_t
AllGatherMinimalMatmulOuterDimAsyncDeviceOperation::compute_output_specs(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    const auto& x = tensor_args.input_tensor;
    const auto& pgb = tensor_args.persistent_gather_buffer;
    // [.., M, N]: the first operand's shape with its K replaced by the second operand's N.
    const OperandDims dims = operand_dims(attrs);
    ttnn::Shape output_shape(dims.gather_in0 ? pgb.logical_shape() : x.logical_shape());
    output_shape[-1] = dims.gather_in0 ? x.logical_shape()[dims.input_outer] : pgb.logical_shape()[dims.gathered_outer];
    const auto& memory_config = attrs.output_mem_config.value_or(x.memory_config());
    const auto dtype = attrs.output_dtype.value_or(x.dtype());
    return TensorSpec(output_shape, TensorLayout(dtype, PageConfig(Layout::TILE), memory_config));
}

AllGatherMinimalMatmulOuterDimAsyncDeviceOperation::tensor_return_value_t
AllGatherMinimalMatmulOuterDimAsyncDeviceOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    return ttnn::create_device_tensor(compute_output_specs(attrs, tensor_args), tensor_args.input_tensor.device());
}

// One operand is gathered on cluster_axis, so there it counts as replicated, and the output takes the input's label
// combined with that: whole on every device for a replicated input, split where the input is split.
std::vector<TensorTopology> AllGatherMinimalMatmulOuterDimAsyncDeviceOperation::compute_output_topologies(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    const auto& x = tensor_args.input_tensor;
    const auto& g = tensor_args.gathered_tensor;
    const OperandDims dims = operand_dims(attrs);
    TensorTopology gathered_label =
        ttnn::operations::ccl::common::all_gather_output_topology(g, attrs.cluster_axis, dims.gathered_outer)
            .value_or(g.tensor_topology());
    TensorTopology input_label = x.tensor_topology();
    // What the second operand gives the output is its N, as the output's last dim. So a label that shards the second
    // operand along N (dim -2 of [N, K], dim -1 of [K, N]) moves to the output's last dim: the input's label with
    // dim=-2, and with dim=-1 what the gathered tensor's label keeps on the mesh axes the op does not gather over.
    // The first operand's batch dims and M sit at the same positions in the output, so its label holds as it is.
    const ttnn::Tensor& in1 = dims.gather_in0 ? x : g;
    TensorTopology& in1_label = dims.gather_in0 ? input_label : gathered_label;
    const int in1_rank = static_cast<int>(in1.logical_shape().rank());
    const int in1_outer = dims.gather_in0 ? dims.input_outer : dims.gathered_outer;
    const int output_rank =
        static_cast<int>((dims.gather_in0 ? tensor_args.persistent_gather_buffer : x).logical_shape().rank());
    auto placements = in1_label.placements();
    for (auto& placement : placements) {
        if (ttnn::operations::ccl::common::placement_shards_tensor_dim(
                placement, static_cast<uint32_t>(in1_rank + in1_outer), in1_rank)) {
            placement = tt::tt_metal::distributed::MeshMapperConfig::Shard{output_rank - 1};
        }
    }
    in1_label = TensorTopology(in1_label.distribution_shape(), std::move(placements), in1_label.mesh_coords());
    return {ttnn::operations::ccl::common::union_topology(
        {std::cref(input_label), std::cref(gathered_label)}, x.tensor_topology().mesh_coords())};
}
}  // namespace ttml::metal::ops::all_gather_minimal_matmul_outer_dim_async

namespace ttnn::prim {
ttnn::Tensor ttml_all_gather_minimal_matmul_outer_dim_async(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& gathered_tensor,
    const ttnn::Tensor& persistent_gather_buffer,
    const std::vector<ttnn::GlobalSemaphore>& multi_device_global_semaphore,
    std::optional<uint32_t> cluster_axis,
    ttnn::ccl::Topology topology,
    const std::optional<const experimental::prim::MinimalMatmulConfig>& config,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<const DataType> dtype,
    std::optional<ttnn::DeviceComputeKernelConfig> compute_kernel_config,
    uint32_t num_links,
    bool transpose_b,
    int32_t dim) {
    using OperationType =
        ttml::metal::ops::all_gather_minimal_matmul_outer_dim_async::AllGatherMinimalMatmulOuterDimAsyncDeviceOperation;
    // The ring size below comes from the input's mesh, before validation runs.
    TT_FATAL(
        input_tensor.storage_type() == StorageType::DEVICE && input_tensor.is_allocated(),
        "{}: input must be an allocated device tensor",
        kOp);
    const auto& mesh_shape = input_tensor.device()->shape();
    TT_FATAL(
        !cluster_axis.has_value() || cluster_axis.value() < mesh_shape.dims(),
        "{}: cluster_axis ({}) must be < the mesh rank ({})",
        kOp,
        cluster_axis.value_or(0),
        mesh_shape.dims());
    const uint32_t ring_size = cluster_axis.has_value() ? mesh_shape[cluster_axis.value()]
                                                        : static_cast<uint32_t>(input_tensor.device()->num_devices());
    const auto arch = input_tensor.device()->arch();
    auto kernel_config_val = init_device_compute_kernel_config(
        arch, compute_kernel_config, tt::tt_metal::MathFidelity::HiFi2, false, true, true);

    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{
            .config = config,
            .output_mem_config = memory_config,
            .output_dtype = dtype,
            .compute_kernel_config = kernel_config_val,
            .ring_size = ring_size,
            .topology = topology,
            .cluster_axis = cluster_axis,
            .num_links = num_links,
            .semaphores = multi_device_global_semaphore,
            .transpose_b = transpose_b,
            .dim = dim},
        OperationType::tensor_args_t{
            .input_tensor = input_tensor,
            .gathered_tensor = gathered_tensor,
            .persistent_gather_buffer = persistent_gather_buffer});
}
}  // namespace ttnn::prim
