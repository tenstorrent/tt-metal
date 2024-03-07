// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <type_traits>

#include "third_party/magic_enum/magic_enum.hpp"

#include "tt_metal/detail/tt_metal.hpp"
#include "tensor/tensor.hpp"
#include "tools/profiler/profiler.hpp"
#include "tt_metal/detail/tt_metal.hpp"

#include "tt_metal/third_party/tracy/public/tracy/Tracy.hpp"
#include "tt_metal/third_party/tracy/public/tracy/TracyC.h"

namespace tt {

namespace tt_metal {

namespace op_profiler {

    enum class OpType {
        python_fallback,
        tt_dnn_cpu,
        tt_dnn_device,
        custom_zone,
        unknown
    };

    namespace detail {
        static std::filesystem::path const& getLogLocationsRecord()
        {
            static const std::filesystem::path logLocationsRecord =
                string(PROFILER_RUNTIME_ROOT_DIR) + string(PROFILER_LOGS_DIR_NAME) + "/.locations.log";
            return logLocationsRecord;
        }

        static string replace_comma(const string& s)
        {
            string ret = s;
            std::replace( ret.begin(), ret.end(), ',', ';');
            return ret;
        }

        static string join_vector(const vector<string>& strs, string delimiter = "-")
        {
            string ret = "";

            for (auto &str : strs)
            {
                ret += (str + delimiter);
            }
            ret = ret.substr(0,ret.size()-1);

            return ret;
        }

        static string shape_to_str(const Shape& shape)
        {
            return fmt::format("{}", fmt::join(std::begin(shape), std::end(shape), "_"));
        }

        static string tensor_to_str(const Tensor& tensor)
        {
            string tensorStorageStr;
            if (tensor.storage_type() == StorageType::DEVICE)
            {
                tensorStorageStr = fmt::format("DEV_{}_{}_{}",
                        tensor.device()->id(),
                        magic_enum::enum_name(tensor.memory_config().buffer_type),
                        magic_enum::enum_name(tensor.memory_config().memory_layout));
            }
            else
            {
                tensorStorageStr = fmt::format("{}", magic_enum::enum_name(tensor.storage_type()));
            }

            vector<string> tensorStrs = {
                shape_to_str(tensor.get_legacy_shape()),
                fmt::format("{}", magic_enum::enum_name(tensor.get_layout())),
                fmt::format("{}", magic_enum::enum_name(tensor.get_dtype())),
                tensorStorageStr
            };

            return join_vector(tensorStrs, "|");
        }

        static void delete_logs_location_record() { std::filesystem::remove(getLogLocationsRecord()); }

        static void add_log_location_record(const string& logLocation)
        {
            std::ofstream recordFile;
            recordFile.open(getLogLocationsRecord(), std::ios_base::app);
            recordFile << logLocation << std::endl;
            recordFile.close();

        }

        struct OpData {
            string name;
            Profiler profiler = Profiler();
            vector<string> metaDataVector = {};

            int opCallCount;
            int globalCallCount;
            int stackSize;

            vector<string> inputs = {};
            vector<string> outputs = {};
            vector<string> mathFidelities = {};
            vector<string> computeKernelPaths = {};
            vector<string> computeKernelHashes = {};
            vector<string> datamovementKernelPaths = {};
            vector<string> datamovementKernelHashes = {};

            string parlStrategy = "";
            string preferredName = "";

            operation::OpPerformanceModel perf_model;

            OpType type;

            OpData (string opName, int opCount, int globalCount, int stackSizeArg, OpType typeArg):
                name(opName),
                opCallCount(opCount),
                globalCallCount(globalCount),
                stackSize(stackSizeArg),
                type(typeArg)
            {}
        };

        class OpProfiler {

            private:
                const string unknownOpName = "unknown_op";
                OpData unknownOp = OpData (unknownOpName,0,0,0,OpType::unknown);

                string profileFolder = "";
                stack<OpData> opStack;

                unordered_map <string, uint32_t> callCounters;
                int globalCallCount = 0;

                int get_call_count_increment (string opNameArg)
                {
                    auto it = callCounters.find(opNameArg);
                    if (it != callCounters.end())
                    {
                        callCounters.at(opNameArg) ++;
                    }
                    else
                    {
                        callCounters[opNameArg] = 1;
                    }
                    globalCallCount ++;
                    return callCounters.at(opNameArg);
                }

                int get_call_count (string opNameArg) const
                {
                    TT_ASSERT (callCounters.find(opNameArg) != callCounters.end(),
                            "Something is wrong, following op never started: " + opNameArg );
                    return callCounters.at(opNameArg);
                }

                void setup_profiling_folders (
                        string opName,
                        int callCount,
                        Profiler& opProfiler,
                        bool freshTTmetalLogs = true)
                {
                    TT_ASSERT (profileFolder != "", "Bad log folder location, folder has been setup wrong");
                    tt::tt_metal::detail::SetProfilerDir(profileFolder + "/" + opName + "/" + to_string(callCount));
                    if (freshTTmetalLogs)
                    {
                        tt::tt_metal::detail::FreshProfilerHostLog();
                        tt::tt_metal::detail::FreshProfilerDeviceLog();
                    }

                    opProfiler.setOutputDir(profileFolder + "/" + opName);
                    //If it is the first call to this op, freshen the log
                    if (callCount > 1)
                    {
                        opProfiler.setHostNewLogFlag(false);
                    }
                }

                OpData& get_op_data()
                {
#if defined(PROFILER)
                    TT_ASSERT (opStack.size() > 0, "Something is wrong, cannot get op data, op stack is empty");
                    return opStack.top();
#else
                    return unknownOp;
#endif
                }

                vector<pair<string,string>> generate_additional_data()
                {
                    vector<pair<string,string>> additionalFields = {};

                    auto& opData = get_op_data();
                    additionalFields.push_back({"Global Call Count", to_string(opData.globalCallCount)});
                    additionalFields.push_back({"Call Count", to_string(opData.opCallCount)});
                    additionalFields.push_back({"Stack Size", to_string(opData.stackSize)});
                    additionalFields.push_back({"Inputs", join_vector(opData.inputs)});
                    additionalFields.push_back({"Outputs", join_vector(opData.outputs)});
                    additionalFields.push_back({"Math Fidelity", join_vector(opData.mathFidelities)});
                    additionalFields.push_back({"Compute Kernel Paths", join_vector(opData.computeKernelPaths)});
                    additionalFields.push_back({"Compute Kernel Hashes", join_vector(opData.computeKernelHashes)});
                    additionalFields.push_back({"Data Movement Kernel Paths", join_vector(opData.datamovementKernelPaths)});
                    additionalFields.push_back({"Data Movement Kernel Hashes", join_vector(opData.datamovementKernelHashes)});
                    additionalFields.push_back({"Parallelization Strategy", opData.parlStrategy});
                    additionalFields.push_back({"Preferred Name", opData.preferredName});
                    additionalFields.push_back({"Meta Data", join_vector(opData.metaDataVector)});
                    additionalFields.push_back({"Type", fmt::format("{}",magic_enum::enum_name(opData.type))});
                    additionalFields.push_back({"PM Ideal ns", fmt::format("{}", opData.perf_model.get_ideal_ns())});
                    additionalFields.push_back({"PM Compute ns", fmt::format("{}", opData.perf_model.get_compute_ns())});
                    additionalFields.push_back({"PM Bandwidth ns", fmt::format("{}", opData.perf_model.get_bandwidth_ns())});
                    additionalFields.push_back({"PM Req I BW", fmt::format("{}",fmt::join(opData.perf_model.get_input_bws(), "|"))});
                    additionalFields.push_back({"PM Req O BW", fmt::format("{}",fmt::join(opData.perf_model.get_output_bws(), "|"))});
                    return additionalFields;
                }

                void clear_profiler()
                {
                    TT_ASSERT (opStack.size() > 0, "Something is wrong, op stack is empty, clear profiler");

                    opStack.pop();

                    if (opStack.size() > 0)
                    {
                        auto callingOp = get_op_data();
                        auto callingOpName = callingOp.name;
                        auto callingOpCallCount = get_call_count(callingOpName);
                        TT_ASSERT(callingOpCallCount == callingOp.opCallCount,
                                "Something is wrong, op call count from op stack head does not match the expected");

                        // No need to freshen the logs for TT metal they were at start
                        constexpr bool freshTTmetalLogs = false;
                        setup_profiling_folders(
                                callingOpName,
                                callingOpCallCount,
                                callingOp.profiler,
                                freshTTmetalLogs);
                    }
                    else
                    {
                        unknownOp = OpData(unknownOpName, unknownOp.opCallCount + 1, globalCallCount, 0, OpType::unknown);
                        setup_profiling_folders(unknownOpName, globalCallCount, unknownOp.profiler);
                    }
                }

            public:
                OpProfiler ()
                {
#if defined(PROFILER)
                    delete_logs_location_record();
                    set_profiler_location("ops");
#endif
                }

                void start_profiling(const string& opName, OpType opType)
                {
#if defined(PROFILER)
                    auto opNameNoComma = replace_comma(opName);
                    auto callCount = get_call_count_increment(opName);
                    OpData opData = OpData(opNameNoComma, callCount, globalCallCount, opStack.size() + 1, opType);

                    opData.profiler.markStart(opNameNoComma);

                    setup_profiling_folders (opNameNoComma, callCount, opData.profiler);
                    opStack.push(opData);
#endif
                }


                void stop_profiling(const string& opName)
                {
#if defined(PROFILER)
                    auto opNameNoComma = replace_comma(opName);
                    auto& opData = get_op_data();
                    TT_ASSERT (opNameNoComma == opData.name, "Something is wrong, op name mismatch");

                    auto additionalFields = generate_additional_data();
                    opData.profiler.markStop(opNameNoComma, additionalFields);
                    clear_profiler();
#endif
                }

                void append_input_data (const string& input)
                {
#if defined(PROFILER)
                    get_op_data().inputs.push_back(replace_comma(input));
#endif
                }

                void append_output_data (const string& output)
                {
#if defined(PROFILER)
                    get_op_data().outputs.push_back(replace_comma(output));
#endif
                }

                void append_kernel_info (Kernel* kernel)
                {
#if defined(PROFILER)
                    if (kernel->processor() == RISCV::COMPUTE) {
                        ComputeKernel * compute_kernel = static_cast<ComputeKernel*>(kernel);
                        MathFidelity math_fidelity = std::get<ComputeConfig>(compute_kernel->config()).math_fidelity;
                        get_op_data().mathFidelities.push_back(replace_comma(fmt::format("{}", magic_enum::enum_name(math_fidelity))));
                        get_op_data().computeKernelPaths.push_back(replace_comma(compute_kernel->kernel_path_file_name()));
                        get_op_data().computeKernelHashes.push_back(replace_comma(compute_kernel->get_full_kernel_name()));
                    }
                    else
                    {
                        get_op_data().datamovementKernelPaths.push_back(replace_comma(kernel->kernel_path_file_name()));
                        get_op_data().datamovementKernelHashes.push_back(replace_comma(kernel->get_full_kernel_name()));
                    }
#endif
                }

                void set_parallelization_strategy (const string& strategy)
                {
#if defined(PROFILER)
                    get_op_data().parlStrategy = replace_comma(strategy);
#endif
                }

                void set_preferred_name (const string& name)
                {
#if defined(PROFILER)
                    get_op_data().preferredName = replace_comma(name);
#endif
                }

                void append_meta_data(const string& metaData)
                {
#if defined(PROFILER)
                    TT_ASSERT (opStack.size() > 0, "Something is wrong, cannot append meta data, op stack is empty");
                    string noDashMetaData = replace_comma(metaData);
                    std::replace( noDashMetaData.begin(), noDashMetaData.end(), '-', '_');
                    get_op_data().metaDataVector.push_back(noDashMetaData);
#endif
                }

                void set_perf_model (const operation::OpPerformanceModel& m) {
#if defined(PROFILER)
                    get_op_data().perf_model = m;
#endif
                }

                void set_profiler_location(const string& folder)
                {
#if defined(PROFILER)
                    string constructedFolder = string(PROFILER_RUNTIME_ROOT_DIR) + string(PROFILER_LOGS_DIR_NAME) + "/" + folder;
                    int noSlashEnd = constructedFolder.find_last_not_of(" /");
                    auto logFolder = (noSlashEnd == std::string::npos) ? "" : constructedFolder.substr(0, noSlashEnd + 1);
                    auto logFolderDevice = fmt::format("{}_device", logFolder);
                    if ((profileFolder == logFolder) || (profileFolder == logFolderDevice))
                    {
                        //We are in the same active process keep going and append
                        return;
                    }

                    if (getDeviceProfilerState())
                    {
                        profileFolder = logFolderDevice;
                        tt::log_info("Device profiling detected, logs folder location changed to {}", profileFolder);
                        add_log_location_record(logFolderDevice);
                    }
                    else
                    {
                        profileFolder = logFolder;
                        add_log_location_record(logFolder);
                    }

                    if (std::filesystem::is_directory(profileFolder))
                    {
                        std::filesystem::remove_all(profileFolder);
                    }
#endif
                }
        };

        inline OpProfiler operationProfiler;
    }


    static void start_profiling (const string& opName, OpType opType)
    {
#if defined(PROFILER)
        detail::operationProfiler.start_profiling(opName, opType);
#endif
    }

    static void stop_profiling (const string& opName)
    {
#if defined(PROFILER)
        detail::operationProfiler.stop_profiling(opName);
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
#if defined(TRACY_ENABLE)
        TracyMessageC(source.c_str(), source.size(), color);
#endif
    }

    static void tracy_frame() {
#if defined(TRACY_ENABLE)
        FrameMark;
#endif
    }

    static bool get_profiler_flag ()
    {
        return getHostProfilerState();
    }

    static void append_input_data (const Tensor& input)
    {
#if defined(PROFILER)
        detail::operationProfiler.append_input_data(detail::tensor_to_str(input));
#endif
    }

    static void append_input_optional_data (std::optional<const Tensor> input)
    {
#if defined(PROFILER)
        if (input.has_value()) {
            detail::operationProfiler.append_input_data(detail::tensor_to_str(input.value()));
        }
        else
        {
            detail::operationProfiler.append_input_data("");
        }
#endif
    }

    static void append_output_data (const Tensor& output)
    {
#if defined(PROFILER)
        detail::operationProfiler.append_output_data(detail::tensor_to_str(output));
#endif
    }

    static void append_all_tensor_io_data (
        const std::vector<Tensor> &input_tensors,
        const std::vector<std::optional<const Tensor>> &optional_input_tensors,
        const std::vector<Tensor> &output_tensors)
    {
#if defined(PROFILER)
            for (auto& input : input_tensors)
            {
                append_input_data(input);
            }

            for (auto& input : optional_input_tensors)
            {
                append_input_optional_data(input);
            }

            for (auto& output : output_tensors)
            {
                append_output_data(output);
            }
#endif
    }

    static void append_meta_data (const string& metaData)
    {
#if defined(PROFILER)
        detail::operationProfiler.append_meta_data(metaData);
#endif
    }

    template < typename T, typename std::enable_if< std::is_enum<T>::value,bool>::type = true>
    static void set_preferred_name (const T& name)
    {
#if defined(PROFILER)
        detail::operationProfiler.set_preferred_name(fmt::format("{}",magic_enum::enum_name(name)));
#endif
    }

    template < typename T, typename std::enable_if< !std::is_enum<T>::value,bool>::type = true>
    static void set_preferred_name (const T& name)
    {
#if defined(PROFILER)
        detail::operationProfiler.set_preferred_name(fmt::format("{}",name));
#endif
    }

    template < typename T, typename std::enable_if< std::is_enum<T>::value,bool>::type = true>
    static void set_parallelization_strategy (const T& strategy)
    {
#if defined(PROFILER)
        detail::operationProfiler.set_parallelization_strategy(fmt::format("{}",magic_enum::enum_name(strategy)));
#endif
    }

    template < typename T, typename std::enable_if< !std::is_enum<T>::value,bool>::type = true>
    static void set_parallelization_strategy (const T& strategy)
    {
#if defined(PROFILER)
        detail::operationProfiler.set_parallelization_strategy(fmt::format("{}",strategy));
#endif
    }

    static void append_kernel_info (const Program& program)
    {
#if defined(PROFILER)
        for (size_t kernel_id = 0; kernel_id < program.num_kernels(); kernel_id++) {
            auto kernel = tt::tt_metal::detail::GetKernel(program, kernel_id);
            detail::operationProfiler.append_kernel_info(kernel.get());
        }
#endif
    }

    static void set_perf_model(const operation::OpPerformanceModel& m) {
#if defined(PROFILER)
        detail::operationProfiler.set_perf_model(m);
#endif
    }

    static void set_profiler_location (const string& profilerLocation)
    {
#if defined(PROFILER)
        detail::operationProfiler.set_profiler_location(profilerLocation);
#endif
    }

    static void dump_device_profiler_results (Device *device, Program &program)
    {
#if defined(PROFILER)
        if (getDeviceProfilerState())
        {
            tt::tt_metal::DumpDeviceProfileResults(device, program);
        }
#endif
    }

    static void dump_device_profiler_results (Device * device, std::shared_ptr<Program> program)
    {
        dump_device_profiler_results(device, *program);
    }

    class OpProfileScope
    {
        private:
            string scopeName = "";
        public:
            OpProfileScope (const string& scopeNameArg, OpType opType) : scopeName(scopeNameArg)
            {
#if defined(PROFILER)
                start_profiling (scopeName, opType);
#endif
            }

            ~OpProfileScope ()
            {
#if defined(PROFILER)
                stop_profiling (scopeName);
#endif
            }
    };
}

}
}
