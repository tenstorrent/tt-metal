#pragma once

#include <filesystem>
#include <type_traits>

#include "third_party/magic_enum/magic_enum.hpp"
#include "tensor/tensor.hpp"

//TODO(MO): hack until ticket #1184 is in
extern bool enable_fw_profile_hack;

namespace tt {

namespace tt_metal {


namespace op_profiler {

    namespace detail {
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

        static string shape_to_str(const array<uint32_t, 4> shape)
        {
            return to_string(shape[0]) + "_" +\
                to_string(shape[1]) + "_" +\
                to_string(shape[2]) + "_" +\
                to_string(shape[3]);
        }

        static string tensor_to_str(const Tensor& tensor)
        {
            const unordered_map <Layout, string> layout_to_str = {
                {Layout::ROW_MAJOR, "ROW_MAJOR"},
                {Layout::TILE, "TILE"},
                {Layout::CHANNELS_LAST, "CHANNELS_LAST"}
            };

            const unordered_map <DataType, string> dtype_to_str = {
                {DataType::BFLOAT16, "BFLOAT16"},
                {DataType::FLOAT32, "FLOAT32"},
                {DataType::UINT32, "UINT32"},
                {DataType::BFLOAT8_B, "BFLOAT8_B"}
            };

            vector<string> tensorStrs = {
                shape_to_str(tensor.shape()),
                layout_to_str.at(tensor.layout()),
                dtype_to_str.at(tensor.dtype()),
                tensor.storage_type() == StorageType::HOST ? "HOST" : fmt::format("DEV_{}_{}", tensor.device()->pcie_slot(), magic_enum::enum_name(tensor.memory_config().buffer_type)),
            };

            return join_vector(tensorStrs, "|");
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
            string mathFidelity = "";
            string parlStrategy = "";
            string preferredName = "";

            OpData (string opName, int opCount, int globalCount, int stackSizeArg) :
                name(opName),
                opCallCount(opCount),
                globalCallCount(globalCount),
                stackSize(stackSizeArg)
            {}
        };

        class OpProfiler {

            private:
                const string unknownOpName = "unknown_op";
                OpData unknownOp = OpData (unknownOpName,0,0,0);

                bool profileOps = false;
                string profileFolder = "tt_metal/tools/profiler/logs/ops/";
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
                    tt::tt_metal::SetProfilerDir(profileFolder + "/" + opName + "/" + to_string(callCount));
                    if (freshTTmetalLogs)
                    {
                        tt::tt_metal::FreshProfilerHostLog();
                        tt::tt_metal::FreshProfilerDeviceLog();
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
                    if (profileOps)
                    {
                        TT_ASSERT (opStack.size() > 0, "Something is wrong, cannot get op data, op stack is empty");
                        return opStack.top();
                    }
                    return unknownOp;
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
                    additionalFields.push_back({"Math Fidelity", opData.mathFidelity});
                    additionalFields.push_back({"Parallelization Strategy", opData.parlStrategy});
                    additionalFields.push_back({"Preferred Name", opData.preferredName});
                    additionalFields.push_back({"Meta Data", join_vector(opData.metaDataVector)});

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
                        unknownOp = OpData(unknownOpName, unknownOp.opCallCount + 1, globalCallCount, 0);
                        setup_profiling_folders(unknownOpName, globalCallCount, unknownOp.profiler);
                    }
                }

            public:

                void start_profiling(const string& opName)
                {
                    if (profileOps)
                    {
                        auto opNameNoComma = replace_comma(opName);
                        auto callCount = get_call_count_increment(opName);
                        OpData opData = OpData(opNameNoComma, callCount, globalCallCount, opStack.size() + 1);

                        opData.profiler.setHostDoProfile(true);
                        opData.profiler.markStart(opNameNoComma);

                        tt::tt_metal::SetHostProfilerFlag(true);

                        setup_profiling_folders (opNameNoComma, callCount, opData.profiler);

                        opStack.push(opData);
                    }
                }


                void stop_profiling(const string& opName)
                {
                    if (profileOps)
                    {
                        auto opNameNoComma = replace_comma(opName);
                        auto& opData = get_op_data();
                        TT_ASSERT (opNameNoComma == opData.name, "Something is wrong, op name mismatch");

                        auto additionalFields = generate_additional_data();
                        opData.profiler.markStop(opNameNoComma, false);
                        opData.profiler.dumpHostResults(additionalFields);
                        clear_profiler();
                    }
                }

                bool get_profiler_flag () const
                {
                    return profileOps;
                }

                void append_input_data (const string& input)
                {
                    get_op_data().inputs.push_back(replace_comma(input));
                }

                void append_output_data (const string& output)
                {
                    get_op_data().outputs.push_back(replace_comma(output));
                }

                void set_math_fidelity (const string& fidelity)
                {
                    get_op_data().mathFidelity = replace_comma(fidelity);
                }

                void set_parallelization_strategy (const string& strategy)
                {
                    get_op_data().parlStrategy = replace_comma(strategy);
                }

                void set_preferred_name (const string& name)
                {
                    get_op_data().preferredName = replace_comma(name);
                }

                void append_meta_data(const string& metaData)
                {
                    if (profileOps)
                    {
                        TT_ASSERT (opStack.size() > 0, "Something is wrong, cannot append meta data, op stack is empty");
                        string noDashMetaData = replace_comma(metaData);
                        std::replace( noDashMetaData.begin(), noDashMetaData.end(), '-', '_');
                        get_op_data().metaDataVector.push_back(noDashMetaData);
                    }
                }

                void set_profiler_flag(bool doProfile)
                {
                    profileOps = doProfile;

                    //TODO(MO): hack until ticket #1184 is in
                    enable_fw_profile_hack = doProfile;
                }

                void set_profiler_location(const string& profilerLogFolder)
                {
                    if (profileOps)
                    {
                        TT_ASSERT (!(std::filesystem::is_directory(profilerLogFolder)), "Folder " + profilerLogFolder + " exists. Either rename or remove it");
                        profileFolder = profilerLogFolder;
                    }
                }
        };

        inline OpProfiler operationProfiler;
    }

    static void start_profiling (const string& opName)
    {
        detail::operationProfiler.start_profiling(opName);
    }

    static void stop_profiling (const string& opName)
    {
        detail::operationProfiler.stop_profiling(opName);
    }

    static bool get_profiler_flag ()
    {
        return detail::operationProfiler.get_profiler_flag();
    }

    static void append_input_data (const Tensor& input)
    {
        detail::operationProfiler.append_input_data(detail::tensor_to_str(input));
    }

    static void append_input_optional_data (std::optional<const Tensor> input)
    {
        if (input.has_value()) {
            detail::operationProfiler.append_input_data(detail::tensor_to_str(input.value()));
        }
        else
        {
            detail::operationProfiler.append_input_data("");
        }
    }

    static void append_output_data (const Tensor& output)
    {
        detail::operationProfiler.append_output_data(detail::tensor_to_str(output));
    }

    static void append_all_tensor_io_data (
        const std::vector<Tensor> &input_tensors,
        const std::vector<std::optional<const Tensor>> &optional_input_tensors,
        const std::vector<Tensor> &output_tensors)
    {
        if (detail::operationProfiler.get_profiler_flag())
        {
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
        }
    }

    static void append_meta_data (const string& metaData)
    {
        detail::operationProfiler.append_meta_data(metaData);
    }

    template < typename T, typename std::enable_if< std::is_enum<T>::value,bool>::type = true>
    static void set_preferred_name (const T& name)
    {
        detail::operationProfiler.set_preferred_name(fmt::format("{}",magic_enum::enum_name(name)));
    }

    template < typename T, typename std::enable_if< !std::is_enum<T>::value,bool>::type = true>
    static void set_preferred_name (const T& name)
    {
        detail::operationProfiler.set_preferred_name(fmt::format("{}",name));
    }

    template < typename T, typename std::enable_if< std::is_enum<T>::value,bool>::type = true>
    static void set_parallelization_strategy (const T& strategy)
    {
        detail::operationProfiler.set_parallelization_strategy(fmt::format("{}",magic_enum::enum_name(strategy)));
    }

    template < typename T, typename std::enable_if< !std::is_enum<T>::value,bool>::type = true>
    static void set_parallelization_strategy (const T& strategy)
    {
        detail::operationProfiler.set_parallelization_strategy(fmt::format("{}",strategy));
    }

    static void set_math_fidelity (const string& fidelity)
    {
        detail::operationProfiler.set_math_fidelity(fidelity);
    }

    static void set_profiler_flag (bool profilerFlag)
    {
        detail::operationProfiler.set_profiler_flag(profilerFlag);
    }

    static void set_profiler_location (const string& profilerLocation)
    {
        detail::operationProfiler.set_profiler_location(profilerLocation);
    }

    static void dump_device_profiler_results (Device *device, Program &program)
    {
        //TODO: (MO) Added this for now until #1184 is finished to be able to disable device profiling
        const char *TT_METAL_PROFILER = std::getenv("TT_METAL_PROFILER");
        if (detail::operationProfiler.get_profiler_flag() && TT_METAL_PROFILER == nullptr)
        {
            //TODO: (MO) This global is temporary need to update once the new interface is in
            if (HACK_CQ) {
                Finish(*HACK_CQ);
            }
            tt::tt_metal::DumpDeviceProfileResults(device, program);
        }
    }

    class ProfileScope
    {
        private:
            string scopeName = "";
        public:
            ProfileScope (const string& scopeNameArg) : scopeName(scopeNameArg)
            {
                start_profiling (scopeName);
            }

            ~ProfileScope ()
            {
                stop_profiling (scopeName);
            }
    };
}

}
}
