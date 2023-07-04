// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <type_traits>

#include "third_party/magic_enum/magic_enum.hpp"
#include "third_party/json/json.hpp"


#include "tensor/tensor.hpp"
#include "tt_dnn/op_library/operation.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tools/profiler/profiler.hpp"

#include "tt_metal/third_party/tracy/public/tracy/Tracy.hpp"
#include "tt_metal/third_party/tracy/public/tracy/TracyC.h"

using json = nlohmann::json;

namespace tt {

namespace tt_metal {

namespace op_profiler {

    enum class OpType {
        python_fallback,
        tt_dnn_cpu,
        tt_dnn_device,
        unknown
    };


    static void set_profiler_location (const string& profilerLocation)
    {
#if defined(PROFILER)
        tt::tt_metal::detail::SetDeviceProfilerDir(profilerLocation);
#endif
    }

#if defined(TRACY_ENABLE)
    inline stack<TracyCZoneCtx> call_stack;
#endif

    static void start_tracy_zone (const string& source,const string& functName, uint32_t lineNum, uint32_t color = 0)
    {
#if defined(TRACY_ENABLE)
        auto tracySrcLoc = ___tracy_alloc_srcloc(lineNum, source.c_str(), source.length(), functName.c_str(), functName.length());
        TracyCZoneCtx ctx =  ___tracy_emit_zone_begin_alloc(tracySrcLoc,1);
        if (color != 0)
        {
            TracyCZoneColor(ctx, color);
        }

        call_stack.push(ctx);
#endif
    }

    static bool stop_tracy_zone (const string& name = "", uint32_t color = 0)
    {
        bool callStackWasEmpty = true;
#if defined(TRACY_ENABLE)
        if (!call_stack.empty())
        {
            callStackWasEmpty = false;
            TracyCZoneCtx ctx =  call_stack.top();
            if (name != "")
            {
                TracyCZoneName(ctx,name.c_str(), name.length());
            }
            if (color != 0)
            {
                TracyCZoneColor(ctx, color);
            }
            TracyCZoneEnd(ctx);
            call_stack.pop();
        }
#endif
        return callStackWasEmpty;

    }

    static void tracy_message(const string& source, uint32_t color = 0xf0f8ff) {
        TracyMessageC(source.c_str(), source.size(), color);
    }

    static void tracy_frame() {
        FrameMark;
    }

#if defined(TRACY_ENABLE)
    static inline json get_kernels_json (const Program& program)
    {
        vector<json> computeKernels;
        vector<json> datamovementKernels;
        for (size_t kernel_id = 0; kernel_id < program.num_kernels(); kernel_id++) {
            auto kernel = tt::tt_metal::detail::GetKernel(program, kernel_id).get();
            if (kernel->processor() == RISCV::COMPUTE) {
                ComputeKernel * computeKernel = static_cast<ComputeKernel*>(kernel);
                MathFidelity mathFidelity = std::get<ComputeConfig>(computeKernel->config()).math_fidelity;
                json computeKernelObj;
                computeKernelObj["math_fidelity"] = fmt::format("{}", magic_enum::enum_name(mathFidelity));
                computeKernelObj["path"] = computeKernel->kernel_path_file_name();
                computeKernelObj["name"] = computeKernel->get_full_kernel_name();
                computeKernels.push_back(computeKernelObj);
            }
            else
            {
                json datamovementKernelObj;
                datamovementKernelObj["path"] = kernel->kernel_path_file_name();
                datamovementKernelObj["name"] = kernel->get_full_kernel_name();
                datamovementKernels.push_back(datamovementKernelObj);
            }
        }
        json ret;
        ret ["compute_kernels"] = computeKernels;
        ret ["datamovement_kernels"] = datamovementKernels;
        return ret;
    }

    static inline json get_tensor_json(const Tensor& tensor)
    {
        json ret;
        string tensorStorageStr;
        if (tensor.storage_type() == StorageType::DEVICE)
        {
            ret["storage_type"]["device_id"] = tensor.device()->id();
            ret["storage_type"]["memory_config"]["buffer_type"] = magic_enum::enum_name(tensor.memory_config().buffer_type);
            ret["storage_type"]["memory_config"]["memory_layout"] = magic_enum::enum_name(tensor.memory_config().memory_layout);
        }
        else
        {
            ret["storage_type"] = fmt::format("{}", magic_enum::enum_name(tensor.storage_type()));
        }

        auto tensor_shape = tensor.get_legacy_shape();
        ret["shape"]["W"] = tensor_shape[0];
        ret["shape"]["Z"] = tensor_shape[1];
        ret["shape"]["Y"] = tensor_shape[2];
        ret["shape"]["X"] = tensor_shape[3];
        ret["layout"] = fmt::format("{}", magic_enum::enum_name(tensor.get_layout()));
        ret["dtype"] = fmt::format("{}", magic_enum::enum_name(tensor.get_dtype()));

        return ret;
    }

    static inline vector<json> get_tensors_json(const vector<Tensor>& tensors)
    {
        ZoneScoped;
        vector<json> ret;
        for(auto& tensor : tensors)
        {
            ret.push_back(get_tensor_json(tensor));
        }
        return ret;
    }

    static inline vector<json> get_tensors_json(const vector<std::optional<const Tensor>>& tensors)
    {
        ZoneScoped;
        vector<json> ret;
        for(auto& tensor : tensors)
        {
            if (tensor.has_value())
            {
                ret.push_back(get_tensor_json(tensor.value()));
            }
        }
        return ret;
    }

    template <bool IsExternal = false, typename Operation>
    inline json get_base_json(
            uint32_t opID,
            const Operation& op,
            const std::vector<Tensor>& input_tensors,
            std::optional<std::reference_wrapper<std::vector<Tensor>>> output_tensors = std::nullopt)
    {
        ZoneScoped;
        json j;
        j["global_call_count"] = opID;

        std::string opName = op.get_type_name();

        if constexpr (!IsExternal)
        {
            auto profiler_info = op.create_profiler_info(input_tensors);
            if (profiler_info.preferred_name.has_value())
            {
                j["op_code"] = profiler_info.preferred_name.value();
            }

            if (profiler_info.parallelization_strategy.has_value())
            {
                j["parallelization_strategy"] = profiler_info.parallelization_strategy.value();
            }
        }

        j["op_code"] = opName;

        json attributesObj;
        if (not op.attributes().empty()) {
            ZoneScopedN("get_attributes_json");
            for (auto&& [name, value] : op.attributes()) {
                std::string nameStr = "";
                if (std::holds_alternative<std::string>(name))
                {
                    nameStr = fmt::format("{}",std::get<std::string>(name));
                }
                else if (std::holds_alternative<const char*>(name))
                {
                    nameStr = fmt::format("{}",std::get<const char*>(name));
                }
                attributesObj[nameStr] = fmt::format("{}",value);
            }
        }

        j["attributes"] = attributesObj;

        j["input_tensors"] = get_tensors_json(input_tensors);

        if (output_tensors.has_value())
        {
            j["output_tensors"] = get_tensors_json(output_tensors.value());
        }
        return j;

    }

    inline std::string op_meta_data_serialized_json(
            uint32_t opID,
            const tt::tt_metal::operation::ExternalOperation& op,
            const std::vector<Tensor>& input_tensors)
    {
        auto j = get_base_json<true>(opID, op, input_tensors);
        j["op_type"] = magic_enum::enum_name(OpType::python_fallback);
        std::string ser = j.dump(4);
        return fmt::format("TT_DNN_FALL_BACK_OP:{}\n{}",j["op_code"], ser);
    }

    inline std::string op_meta_data_serialized_json(
            uint32_t opID,
            const tt::tt_metal::operation::HostOperation& op,
            const std::vector<Tensor>& input_tensors,
            std::vector<Tensor>& output_tensors)
    {
        auto j = get_base_json(opID, op, input_tensors, output_tensors);
        j["op_type"] = magic_enum::enum_name(OpType::tt_dnn_cpu);
        std::string ser = j.dump(4);
        return fmt::format("TT_DNN_HOST_OP:{}\n{}",j["op_code"], ser);
    }

    inline std::string op_meta_data_serialized_json(
            uint32_t opID,
            uint32_t device_id,
            const tt::tt_metal::operation::DeviceOperation& op,
            const std::variant<std::shared_ptr<Program>, std::reference_wrapper<Program>>& program,
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const tt::tt_metal::Tensor>>& optional_input_tensors,
            std::vector<Tensor>& output_tensors)
    {
        ZoneScoped;

        auto j = get_base_json(opID, op, input_tensors, output_tensors);
        j["op_type"] = magic_enum::enum_name(OpType::tt_dnn_device);
        j["device_id"] = device_id;
        if (std::holds_alternative<std::reference_wrapper<Program>>(program))
        {
            j["kernel_info"] = get_kernels_json(std::get<std::reference_wrapper<Program>>(program));
        }
        else if (std::holds_alternative<std::shared_ptr<Program>>(program))
        {
            auto prg = std::get<std::shared_ptr<Program>>(program);
            if (prg != nullptr)
            {
                j["kernel_info"] = get_kernels_json(*prg);
            }
        }

        j["optional_input_tensors"] = get_tensors_json(optional_input_tensors);

        auto perfModel = op.create_op_performance_model(input_tensors, optional_input_tensors, output_tensors);
        j["performance_model"]["compute_ns"] = perfModel.get_compute_ns();
        j["performance_model"]["ideal_ns"] = perfModel.get_ideal_ns();
        j["performance_model"]["bandwidth_ns"] = perfModel.get_bandwidth_ns();
        j["performance_model"]["input_bws"] = perfModel.get_input_bws();
        j["performance_model"]["output_bws"] = perfModel.get_output_bws();

        std::string ser = j.dump(4);
        return fmt::format("TT_DNN_DEVICE_OP:{}\n{}",j["op_code"], ser);
    }

#define TracyOpTTNNDevice(op_id, device_id, operation, program, input_tensors, optional_input_tensors, output_tensors)\
    std::string op_message = op_profiler::op_meta_data_serialized_json(op_id, device_id, operation, program, input_tensors, optional_input_tensors, output_tensors);\
    std::string op_text = fmt::format("id:{}", op_id);\
    ZoneText(op_text.c_str(), op_text.size());\
    TracyMessage(op_message.c_str(), op_message.size());

#define TracyOpTTNNHost(op_id, operation, input_tensors, output_tensors)\
    std::string op_message = op_profiler::op_meta_data_serialized_json(op_id, operation, input_tensors, output_tensors);\
    std::string op_text = fmt::format("id:{}", op_id);\
    ZoneText(op_text.c_str(), op_text.size());\
    TracyMessage(op_message.c_str(), op_message.size());

#define TracyOpTTNNExternal(op_id, op, input_tensors)\
    std::string op_message = op_profiler::op_meta_data_serialized_json(op_id, op, input_tensors);\
    std::string op_text = fmt::format("id:{}", op_id);\
    ZoneText(op_text.c_str(), op_text.size());\
    TracyMessage(op_message.c_str(), op_message.size());

#else

#define TracyOpTTNNDevice(op_id, device_id, operation, program, input_tensors, optional_input_tensors, output_tensors)
#define TracyOpTTNNHost(op_id, operation, input_tensors, output_tensors)
#define TracyOpTTNNExternal(op_id, op, input_tensors)

#endif
}
}
}
