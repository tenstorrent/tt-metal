#include <sstream>
#include <tt-metalium/universal_kernel.hpp>

#include <iostream>

namespace tt::tt_metal {

UniversalKernelConfigBuilder& UniversalKernelConfigBuilder::add_buffer(
    std::string name, const Buffer& buffer, tt::DataFormat data_format) {
    return add_buffer(std::move(name), &buffer, data_format);
}
UniversalKernelConfigBuilder& UniversalKernelConfigBuilder::add_buffer(
    std::string name, const Buffer* buffer, tt::DataFormat data_format) {
    tensor_data_.emplace_back(
        std::move(name),
        TensorData{
            TensorAccessorArgs(buffer), buffer ? buffer->address() : 0, buffer ? buffer->page_size() : 0, data_format});
    return *this;
}
UniversalKernelConfigBuilder& UniversalKernelConfigBuilder::add_buffer(
    std::string name, const std::shared_ptr<Buffer>& buffer, tt::DataFormat data_format) {
    return add_buffer(std::move(name), buffer.get(), data_format);
}
UniversalKernelConfigBuilder& UniversalKernelConfigBuilder::add_buffer(
    std::string name, const distributed::MeshBuffer& buffer, tt::DataFormat data_format) {
    return add_buffer(std::move(name), &buffer, data_format);
}
UniversalKernelConfigBuilder& UniversalKernelConfigBuilder::add_buffer(
    std::string name, const distributed::MeshBuffer* buffer, tt::DataFormat data_format) {
    tensor_data_.emplace_back(
        std::move(name),
        TensorData{
            TensorAccessorArgs(buffer), buffer ? buffer->address() : 0, buffer ? buffer->page_size() : 0, data_format});
    return *this;
}
UniversalKernelConfigBuilder& UniversalKernelConfigBuilder::add_buffer(
    std::string name, const std::shared_ptr<distributed::MeshBuffer>& buffer, tt::DataFormat data_format) {
    return add_buffer(std::move(name), buffer.get(), data_format);
}

size_t UniversalKernelConfigBuilder::get_runtime_arg_idx(const char* name) const {
    for (size_t i = 0; i < runtime_args_.size(); ++i) {
        if (runtime_args_[i].first == name) {
            return i;
        }
    }
    TT_THROW("Runtime argument {} not found", name);
}
size_t UniversalKernelConfigBuilder::get_common_runtime_arg_idx(const char* name) const {
    for (size_t i = 0; i < common_runtime_args_.size(); ++i) {
        if (common_runtime_args_[i].first == name) {
            return i;
        }
    }
    TT_THROW("Common runtime argument {} not found", name);
}
size_t UniversalKernelConfigBuilder::buffer_addresses_start_runtime_arg_idx() const { return runtime_args_.size(); }

UniversalKernelConfig UniversalKernelConfigBuilder::build() const {
    std::map<std::string, std::string> defines = defines_;
    std::vector<uint32_t> runtime_args;
    std::vector<uint32_t> common_runtime_args;
    std::vector<uint32_t> compile_time_args;

    std::stringstream init_define_ss;

    compile_time_args.reserve(compile_time_args_.size());
    for (const auto& [name, value] : compile_time_args_) {
        compile_time_args.push_back(value);
        init_define_ss << "constexpr uint32_t " << name << " = " << value << "; ";
    }
    runtime_args.reserve(runtime_args_.size());
    for (size_t runtime_arg_idx = 0; runtime_arg_idx < runtime_args_.size(); ++runtime_arg_idx) {
        const auto& [name, value] = runtime_args_[runtime_arg_idx];
        runtime_args.push_back(value);
        init_define_ss << "const uint32_t " << name << " = get_arg_val<uint32_t>(" << runtime_arg_idx << "); ";
    }
    common_runtime_args.reserve(common_runtime_args_.size());
    for (size_t common_runtime_arg_idx = 0; common_runtime_arg_idx < common_runtime_args_.size();
         ++common_runtime_arg_idx) {
        const auto& [name, value] = common_runtime_args_[common_runtime_arg_idx];
        common_runtime_args.push_back(value);
        init_define_ss << "const uint32_t " << name << " = get_common_arg_val<uint32_t>(" << common_runtime_arg_idx
                       << "); ";
    }
    for (size_t tensor_idx = 0; tensor_idx < tensor_data_.size(); ++tensor_idx) {
        const auto& [name, tensor_data] = tensor_data_[tensor_idx];
        size_t cta_offset = compile_time_args.size();
        tensor_data.accessor_args.append_to(compile_time_args);
        runtime_args.push_back(tensor_data.buffer_address);

        init_define_ss << "constexpr auto " << name << "_cb = " << tensor_idx << "; ";
        init_define_ss << "const uint32_t " << name << "_addr = get_arg_val<uint32_t>(" << runtime_args.size() - 1
                       << "); ";
        init_define_ss << "constexpr auto " << name << "_args = TensorAccessorArgs<" << cta_offset << ">(); ";
        init_define_ss << "constexpr uint32_t " << name << "_page_size_bytes = " << tensor_data.page_size_bytes << "; ";
        init_define_ss << "const auto " << name << " = TensorAccessor(" << name << "_args, " << name << "_addr, "
                       << name << "_page_size_bytes); ";
    }

    std::stringstream init_reader_define_ss;
    std::stringstream init_compute_define_ss;

    for (size_t constant_idx = 0; constant_idx < generated_tile_constants_.size(); ++constant_idx) {
        const auto& constant = generated_tile_constants_[constant_idx];
        init_define_ss << "constexpr auto " << constant.name << "_cb = " << constant_idx + tensor_data_.size() << "; ";
        init_define_ss << "constexpr auto " << constant.name << " = ConstantTile(" << constant.name << "_cb, 0); ";
        init_reader_define_ss << constant.generator_code << "; ";
        init_compute_define_ss << "cb_wait_front(" << constant.name << "_cb, 1); ";
    }

    defines["INIT_ARGUMENTS"] = init_define_ss.str();
    defines["TOTAL_NUM_CIRCULAR_BUFFERS"] = std::to_string(tensor_data_.size());

    auto reader_defines = defines;
    reader_defines["INIT_ARGUMENTS"] += init_reader_define_ss.str();
    auto compute_defines = defines;
    compute_defines["INIT_ARGUMENTS"] += init_compute_define_ss.str();

    ReaderDataMovementConfig reader_config(compile_time_args, std::move(reader_defines));
    WriterDataMovementConfig writer_config(compile_time_args, std::move(defines));
    ComputeConfig compute_config{
        .math_fidelity = math_config_.math_fidelity,
        .fp32_dest_acc_en = math_config_.fp32_dest_acc_en,
        .dst_full_sync_en = math_config_.dst_full_sync_en,
        .unpack_to_dest_mode = math_config_.unpack_to_dest_mode,
        .bfp8_pack_precise = math_config_.bfp8_pack_precise,
        .math_approx_mode = math_config_.math_approx_mode,
        .compile_args = std::move(compile_time_args),
        .defines = std::move(compute_defines),
    };
    return {
        .reader_config = std::move(reader_config),
        .writer_config = std::move(writer_config),
        .compute_config = std::move(compute_config),
        .runtime_args = std::move(runtime_args),
        .common_runtime_args = std::move(common_runtime_args),
    };
}

std::vector<CircularBufferConfig> UniversalKernelConfigBuilder::compute_circular_buffers() const {
    std::vector<CircularBufferConfig> circular_buffers;
    circular_buffers.reserve(tensor_data_.size());
    for (size_t tensor_idx = 0; tensor_idx < tensor_data_.size(); ++tensor_idx) {
        const auto& tensor_data = tensor_data_[tensor_idx].second;
        circular_buffers.push_back(
            CircularBufferConfig(2 * tensor_data.page_size_bytes, {{tensor_idx, tensor_data.data_format}})
                .set_page_size(tensor_idx, tensor_data.page_size_bytes));
    }
    for (size_t constant_idx = 0; constant_idx < generated_tile_constants_.size(); ++constant_idx) {
        const auto& generated_tile_constant = generated_tile_constants_[constant_idx];
        uint32_t cb_id = constant_idx + tensor_data_.size();
        uint32_t tile_size = tt::tile_size(generated_tile_constant.data_format);
        circular_buffers.push_back(CircularBufferConfig(tile_size, {{cb_id, generated_tile_constant.data_format}})
                                       .set_page_size(cb_id, tile_size));
    }
    return circular_buffers;
}

}  // namespace tt::tt_metal
