#include <sstream>
#include <tt-metalium/universal_kernel.hpp>

#include <iostream>

namespace tt::tt_metal {

UniversalKernelConfigBuilder& UniversalKernelConfigBuilder::add_buffer(std::string name, const Buffer& buffer) {
    tensor_data_.emplace_back(
        std::move(name), TensorData{TensorAccessorArgs(buffer), buffer.address(), buffer.page_size()});
    return *this;
}
UniversalKernelConfigBuilder& UniversalKernelConfigBuilder::add_buffer(std::string name, const Buffer* buffer) {
    tensor_data_.emplace_back(
        std::move(name),
        TensorData{TensorAccessorArgs(buffer), buffer ? buffer->address() : 0, buffer ? buffer->page_size() : 0});
    return *this;
}
UniversalKernelConfigBuilder& UniversalKernelConfigBuilder::add_buffer(
    std::string name, const std::shared_ptr<Buffer>& buffer) {
    tensor_data_.emplace_back(
        std::move(name),
        TensorData{TensorAccessorArgs(buffer), buffer ? buffer->address() : 0, buffer ? buffer->page_size() : 0});
    return *this;
}
UniversalKernelConfigBuilder& UniversalKernelConfigBuilder::add_buffer(
    std::string name, const distributed::MeshBuffer& buffer) {
    tensor_data_.emplace_back(
        std::move(name), TensorData{TensorAccessorArgs(buffer), buffer.address(), buffer.page_size()});
    return *this;
}
UniversalKernelConfigBuilder& UniversalKernelConfigBuilder::add_buffer(
    std::string name, const distributed::MeshBuffer* buffer) {
    tensor_data_.emplace_back(
        std::move(name),
        TensorData{TensorAccessorArgs(buffer), buffer ? buffer->address() : 0, buffer ? buffer->page_size() : 0});
    return *this;
}
UniversalKernelConfigBuilder& UniversalKernelConfigBuilder::add_buffer(
    std::string name, const std::shared_ptr<distributed::MeshBuffer>& buffer) {
    tensor_data_.emplace_back(
        std::move(name),
        TensorData{TensorAccessorArgs(buffer), buffer ? buffer->address() : 0, buffer ? buffer->page_size() : 0});
    return *this;
}

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

        init_define_ss << "const uint32_t " << name << "_addr = get_arg_val<uint32_t>(" << runtime_args.size() - 1
                       << "); ";
        init_define_ss << "constexpr auto " << name << "_args = TensorAccessorArgs<" << cta_offset << ">(); ";
        init_define_ss << "constexpr uint32_t " << name << "_page_size_bytes = " << tensor_data.page_size_bytes << "; ";
        init_define_ss << "const auto " << name << " = TensorAccessor(" << name << "_args, " << name << "_addr, "
                       << name << "_page_size_bytes); ";
    }

    defines["INIT_ARGUMENTS"] = init_define_ss.str();

    ReaderDataMovementConfig reader_config(compile_time_args, defines);
    WriterDataMovementConfig writer_config(compile_time_args, defines);
    ComputeConfig compute_config{
        .math_fidelity = math_config_.math_fidelity,
        .fp32_dest_acc_en = math_config_.fp32_dest_acc_en,
        .dst_full_sync_en = math_config_.dst_full_sync_en,
        .unpack_to_dest_mode = math_config_.unpack_to_dest_mode,
        .bfp8_pack_precise = math_config_.bfp8_pack_precise,
        .math_approx_mode = math_config_.math_approx_mode,
        .compile_args = compile_time_args,
        .defines = defines,
    };
    return {
        .reader_config = std::move(reader_config),
        .writer_config = std::move(writer_config),
        .compute_config = std::move(compute_config),
        .runtime_args = std::move(runtime_args),
        .common_runtime_args = std::move(common_runtime_args),
    };
}

}  // namespace tt::tt_metal
