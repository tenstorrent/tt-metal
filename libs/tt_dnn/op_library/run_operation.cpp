#include <libs/tensor/tensor.hpp>
#include "tt_dnn/op_library/operation.hpp"
#include "tt_dnn/op_library/program_cache.hpp"

namespace tt::tt_metal::operation {

namespace detail {

static Device* get_device(const std::vector<std::reference_wrapper<const Tensor>> &input_tensors) {
    for (auto &input_tensor : input_tensors) {
        if (not input_tensor.get().on_host()) {
            return input_tensor.get().device();
        }
    }
    auto device = AutoPad::GetDefaultDevice();
    TT_ASSERT(device != nullptr, "Requires setting default device if no inputs to op are on device");
    return device;
}


void write_eltwise_unary_runtime_args(Device* device, Program& program, const Tensor &input_tensor, Tensor &output_tensor) {

    tt_metal::Buffer *src0_dram_buffer = input_tensor.buffer();
    auto dram_src0_noc_xy = src0_dram_buffer->noc_coordinates();

    tt_metal::Buffer *dst_dram_buffer = output_tensor.buffer();
    TT_ASSERT(dst_dram_buffer != nullptr, "Output buffer should be allocated on device!");
    auto dram_dst_noc_xy = dst_dram_buffer->noc_coordinates();

    uint32_t num_tiles = input_tensor.volume() / TILE_HW;

    if (program.data_movement_kernels().size() == 2) {
        CoreCoord core = {0, 0};

        tt_metal::WriteRuntimeArgsToDevice(
            device,
            program.data_movement_kernels().at(0),
            core,
            {src0_dram_buffer->address(),
            uint32_t(dram_src0_noc_xy.x),
            uint32_t(dram_src0_noc_xy.y),
            num_tiles, 0,0,0,0,0 } // TODO(AP): [8] is scaler
        );

        tt_metal::WriteRuntimeArgsToDevice(
            device,
            program.data_movement_kernels().at(1),
            core,
            {dst_dram_buffer->address(),
            uint32_t(dram_dst_noc_xy.x),
            uint32_t(dram_dst_noc_xy.y),
            num_tiles }
        );

    } else {
        auto compute_and_storage_grid_size = device->compute_and_storage_grid_size();
        uint32_t num_cores_x = compute_and_storage_grid_size.x;
        uint32_t num_cores_y = compute_and_storage_grid_size.y;
        auto num_cores = std::min(num_tiles, num_cores_x * num_cores_y);
        std::vector<uint32_t> num_tiles_per_core(num_cores, num_tiles / num_cores);
        for(uint32_t i = 0; i < num_tiles % num_cores; i++){
            num_tiles_per_core[i]++;
        }

        for (uint32_t i = 0, num_tiles_written = 0; i < num_cores; i++){
            CoreCoord core = {i / num_cores_y, i % num_cores_y};
            tt_metal::WriteRuntimeArgsToDevice(
                device,
                program.data_movement_kernels().at(i * 2),
                core,
                {src0_dram_buffer->address(),
                uint32_t(dram_src0_noc_xy.x),
                uint32_t(dram_src0_noc_xy.y),
                num_tiles_per_core[i],
                num_tiles_written, 0 /*disable scaler*/ }
            );

            tt_metal::WriteRuntimeArgsToDevice(
                device,
                program.data_movement_kernels().at(i * 2 + 1),
                core,
                {dst_dram_buffer->address(),
                uint32_t(dram_dst_noc_xy.x),
                uint32_t(dram_dst_noc_xy.y),
                num_tiles_per_core[i],
                num_tiles_written }
            );
            num_tiles_written+=num_tiles_per_core[i];
        }
    }
}


Program create_program(
    const Operation& op,
    const std::vector<std::reference_wrapper<const Tensor>>& input_tensors,
    const std::vector<std::optional<std::reference_wrapper<const Tensor>>>& optional_input_tensors,
    std::vector<Tensor>& output_tensors
)
{
    if (optional_input_tensors.empty()) {
        return op.create_program(input_tensors, output_tensors);
    }
    return op.create_program(input_tensors, optional_input_tensors, output_tensors);
}

std::vector<Tensor> run_without_program_cache(
    const Operation& op,
    const std::vector<std::reference_wrapper<const Tensor>> &input_tensors,
    const std::vector<std::optional<std::reference_wrapper<const Tensor>>> &optional_input_tensors) {

    op.validate(input_tensors);

    auto device = detail::get_device(input_tensors);
    auto output_tensors = op.create_output_tensors(input_tensors);

    auto program = create_program(op, input_tensors, optional_input_tensors, output_tensors);
    tt_metal::CompileProgram(device, program);
    tt_metal::ConfigureDeviceWithProgram(device, program);
    tt_metal::LaunchKernels(device, program);

    return output_tensors;
}


std::vector<Tensor> run_with_program_cache(
    const Operation& op,
    const std::vector<std::reference_wrapper<const Tensor>> &input_tensors,
    const std::vector<std::optional<std::reference_wrapper<const Tensor>>> &optional_input_tensors) {

    op.validate(input_tensors);

    auto device = detail::get_device(input_tensors);
    auto output_tensors = op.create_output_tensors(input_tensors);

    Program& program = program_cache::get_or_create(op, input_tensors, output_tensors, device);

    // TODO(arakhmati): write_runtime_args should always be called
    // But at the moment, we need to ignore it for the ops that don't support program caching
    auto program_hash = op.compute_program_hash(input_tensors);
    if (program_hash != "") { // Empty program hash means the op doesn't support program caching
        write_eltwise_unary_runtime_args(device, program, input_tensors.at(0).get(), output_tensors.at(0));
    }

    tt_metal::ConfigureDeviceWithProgram(device, program);
    tt_metal::LaunchKernels(device, program);

    return output_tensors;
}

}

std::vector<Tensor> run(
    const Operation& op,
    const std::vector<std::reference_wrapper<const Tensor>> &input_tensors,
    const std::vector<std::optional<std::reference_wrapper<const Tensor>>> &optional_input_tensors
) {
    if (program_cache::is_enabled()) {
        return detail::run_with_program_cache(op, input_tensors, optional_input_tensors);
    } else {
        return detail::run_without_program_cache(op, input_tensors, optional_input_tensors);
    }
}

std::vector<Tensor> generic_create_output_tensors(
    const Operation& op,
    const std::vector<std::reference_wrapper<const Tensor>> &input_tensors,
    Layout output_layout = tt::tt_metal::Layout::TILE,
    const MemoryConfig &output_mem_config = MemoryConfig{.interleaved = true}
) {
    const auto& input_tensor = input_tensors.at(0).get();
    const auto& output_shapes = op.compute_output_shapes(input_tensors);

    // HACK to avoid copy constructors when using vectors
    // TODO: If we have default initializers for Tensor, we can do: std::vector<Tensor> output_tensor(num_tensors);
    std::vector<Tensor> output_tensors;
    output_tensors.reserve(output_shapes.size());
    for (const auto& output_shape : output_shapes) {
        output_tensors.emplace_back(tt_metal::Tensor(output_shape, input_tensor.dtype(), output_layout, input_tensor.device(), output_mem_config));
    }
    return output_tensors;
}

}
