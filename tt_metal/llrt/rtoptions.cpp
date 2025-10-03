// SPDX-FileCopyrightText: © 2023 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "rtoptions.hpp"

#include <ctype.h>
#include <stdio.h>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>

#include "assert.hpp"
#include <umd/device/tt_core_coordinates.h>

using std::vector;

namespace tt {

namespace llrt {

const char* RunTimeDebugFeatureNames[RunTimeDebugFeatureCount] = {
    "DPRINT",
    "READ_DEBUG_DELAY",
    "WRITE_DEBUG_DELAY",
    "ATOMIC_DEBUG_DELAY",
    "ENABLE_L1_DATA_CACHE",
};

const char* RunTimeDebugClassNames[RunTimeDebugClassCount] = {"N/A", "worker", "dispatch", "all"};

constexpr auto TT_METAL_HOME_ENV_VAR = "TT_METAL_HOME";
constexpr auto TT_METAL_KERNEL_PATH_ENV_VAR = "TT_METAL_KERNEL_PATH";
// Set this var to change the cache dir.
constexpr auto TT_METAL_CACHE_ENV_VAR = "TT_METAL_CACHE";
// Used for demonstration purposes and will be removed in the future.
// Env variable to override the core grid configuration
static const char* TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE_ENV_VAR = "TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE";
const std::vector<EnvVarDescriptor>& RunTimeOptions::GetEnvVarTable() {
    static const std::vector<EnvVarDescriptor> ENV_VAR_TABLE = {
       
        {
            "TT_METAL_HOME",
            EnvVarParserType::String,
            [](RunTimeOptions* opts, const char* value) {
                opts->is_root_dir_env_var_set = true;
                opts->root_dir = std::string(value) + "/";
            },
            "Sets the root directory of the TT-Metal installation.",  //description
            "No default (must be set)",                               //default value/behaviour
            "export TT_METAL_HOME=/path/to/tt-metal"                  //how to set the env var/usage example
        },
        {
            "TT_METAL_NULL_KERNELS",
            EnvVarParserType::BoolFlag,
            [](RunTimeOptions* opts, const char* value) { 
                opts->null_kernels = true; 
            },
            "Skip actual kernel execution, useful for testing dispatch logic without running kernels.",
            "false (kernels execute normally)",
            "export TT_METAL_NULL_KERNELS=1"
        },
        {
            "TT_METAL_KERNELS_EARLY_RETURN",
            EnvVarParserType::BoolFlag,
            [](RunTimeOptions* opts, const char* value) { 
                opts->kernels_early_return = true; 
            },
            "Kernels return early, skipping execution but maintaining same size as normal.",
            "false (kernels execute fully)",
            "export TT_METAL_KERNELS_EARLY_RETURN=1"
        },
        {
            "TT_METAL_CLEAR_L1",
            EnvVarParserType::Bool01,
            [](RunTimeOptions* opts, const char* value) {
                opts->clear_l1 = (value[0] == '1');
            },
            "Clear L1 memory on device initialization.",
            "0 (don't clear)",
            "export TT_METAL_CLEAR_L1=1"
        },
        {
            "TT_METAL_CLEAR_DRAM",
            EnvVarParserType::Bool01,
            [](RunTimeOptions* opts, const char* value) {
                opts->clear_dram = (value[0] == '1');
            },
            "Clear DRAM memory on device initialization.",
            "0 (don't clear)",
            "export TT_METAL_CLEAR_DRAM=1"
        },  
        {
            "TT_METAL_WATCHER_TEST_MODE",
            EnvVarParserType::BoolFlag,
            [](RunTimeOptions* opts, const char* value) { 
                opts->test_mode_enabled = true; 
            },
            "Enable test mode for watcher functionality.",
            "false (normal mode)",
            "export TT_METAL_WATCHER_TEST_MODE=1"
        },
        {
            "TT_METAL_KERNEL_MAP",
            EnvVarParserType::BoolFlag,
            [](RunTimeOptions* opts, const char* value) { 
                opts->build_map_enabled = true; 
            },
            "Enable kernel build mapping for debugging.",
            "false (mapping disabled)",
            "export TT_METAL_KERNEL_MAP=1"
        },
        {
            "TT_METAL_DISPATCH_DATA_COLLECTION",
            EnvVarParserType::BoolFlag,
            [](RunTimeOptions* opts, const char* value) { 
                opts->enable_dispatch_data_collection = true; 
            },
            "Enable collection of dispatch debugging data.",
            "false (collection disabled)",
            "export TT_METAL_DISPATCH_DATA_COLLECTION=1"
        },
        {
            "TT_METAL_GTEST_ETH_DISPATCH",
            EnvVarParserType::BoolFlag,
            [](RunTimeOptions* opts, const char* value) { 
                opts->dispatch_core_type = tt_metal::DispatchCoreType::ETH; 
            },
            "Use Ethernet cores for dispatch in tests.",
            "Worker cores (default dispatch type)",
            "export TT_METAL_GTEST_ETH_DISPATCH=1"
        },
        {
            "TT_METAL_SKIP_LOADING_FW",
            EnvVarParserType::BoolFlag,
            [](RunTimeOptions* opts, const char* value) { 
                opts->skip_loading_fw = true; 
            },
            "Skip loading firmware during device initialization.",
            "false (load firmware)",
            "export TT_METAL_SKIP_LOADING_FW=1"
        },
        {
            "TT_METAL_SKIP_DELETING_BUILT_CACHE",
            EnvVarParserType::BoolFlag,
            [](RunTimeOptions* opts, const char* value) { 
                opts->skip_deleting_built_cache = true; 
            },
            "Skip deleting built cache files on cleanup.",
            "false (delete cache)",
            "export TT_METAL_SKIP_DELETING_BUILT_CACHE=1"
        },
        {
            "TT_METAL_ENABLE_HW_CACHE_INVALIDATION",
            EnvVarParserType::BoolFlag,
            [](RunTimeOptions* opts, const char* value) { 
                opts->enable_hw_cache_invalidation = true; 
            },
            "Enable hardware cache invalidation.",
            "false (cache invalidation disabled)",
            "export TT_METAL_ENABLE_HW_CACHE_INVALIDATION=1"
        },
        {
            "TT_METAL_DISABLE_RELAXED_MEM_ORDERING",
            EnvVarParserType::BoolFlag,
            [](RunTimeOptions* opts, const char* value) { 
                opts->disable_relaxed_memory_ordering = true; 
            },
            "Disable relaxed memory ordering optimizations.",
            "false (relaxed ordering enabled)",
            "export TT_METAL_DISABLE_RELAXED_MEM_ORDERING=1"
        },
        {
            "TT_METAL_ENABLE_GATHERING",
            EnvVarParserType::BoolFlag,
            [](RunTimeOptions* opts, const char* value) { 
                opts->enable_gathering = true; 
            },
            "Enable data gathering functionality.",
            "false (gathering disabled)",
            "export TT_METAL_ENABLE_GATHERING=1"
        },
        {
            "TT_METAL_FABRIC_TELEMETRY",
            EnvVarParserType::BoolFlag,
            [](RunTimeOptions* opts, const char* value) { 
                opts->enable_fabric_telemetry = true; 
            },
            "Enable fabric telemetry data collection.",
            "false (telemetry disabled)",
            "export TT_METAL_FABRIC_TELEMETRY=1"
        },
        {
            "TT_METAL_FORCE_REINIT",
            EnvVarParserType::BoolFlag,
            [](RunTimeOptions* opts, const char* value) { 
                opts->force_context_reinit = true; 
            },
            "Force context reinitialization on each run.",
            "false (normal initialization)",
            "export TT_METAL_FORCE_REINIT=1"
        },
        {
            "TT_METAL_FABRIC_BLACKHOLE_TWO_ERISC",
            EnvVarParserType::BoolFlag,
            [](RunTimeOptions* opts, const char* value) { 
                opts->enable_2_erisc_mode_with_fabric = true; 
            },
            "Enable two ERISC mode with fabric on Blackhole architecture.",
            "false (single ERISC mode)",
            "export TT_METAL_FABRIC_BLACKHOLE_TWO_ERISC=1"
        },
        {
            "TT_METAL_LOG_KERNELS_COMPILE_COMMANDS",
            EnvVarParserType::BoolFlag,
            [](RunTimeOptions* opts, const char* value) { 
                opts->log_kernels_compilation_commands = true; 
            },
            "Log kernel compilation commands for debugging.",
            "false (no logging)",
            "export TT_METAL_LOG_KERNELS_COMPILE_COMMANDS=1"
        },
        {
            "TT_METAL_SLOW_DISPATCH_MODE",
            EnvVarParserType::BoolFlag,
            [](RunTimeOptions* opts, const char* value) {
                opts->using_slow_dispatch = true;
            },
            "Use slow dispatch mode for debugging.",
            "false (fast dispatch mode)",
            "export TT_METAL_SLOW_DISPATCH_MODE=1"
        },
        {
            "TT_METAL_DEVICE_PROFILER_DISPATCH",
            EnvVarParserType::Complex,
            [](RunTimeOptions* opts, const char* value) {
                // Only enable dispatch profiling if device profiler is also enabled
                const char* profiler_enabled_str = std::getenv("TT_METAL_DEVICE_PROFILER");
                if (profiler_enabled_str != nullptr && profiler_enabled_str[0] == '1') {
                    if (value && value[0] == '1') {
                        opts->profile_dispatch_cores = true;
                    }
                }
            },
            "Enables profiling of dispatch cores. Requires TT_METAL_DEVICE_PROFILER=1 to be effective.",
            "0 (dispatch profiling disabled)",
            "export TT_METAL_DEVICE_PROFILER_DISPATCH=1"
        },    
        {
            "TT_METAL_SKIP_ETH_CORES_WITH_RETRAIN",
            EnvVarParserType::Bool01,
            [](RunTimeOptions* opts, const char* value) {
                opts->skip_eth_cores_with_retrain = (value[0] == '1');
            },
            "Skip Ethernet cores during retraining process.",
            "true (skip retraining)",
            "export TT_METAL_SKIP_ETH_CORES_WITH_RETRAIN=1"
        },
        {
            "TT_METAL_VALIDATE_PROGRAM_BINARIES",
            EnvVarParserType::Bool01,
            [](RunTimeOptions* opts, const char* value) {
                opts->set_validate_kernel_binaries(value[0] == '1');
            },
            "Validate kernel binary integrity before execution.",
            "0 (no validation)",
            "export TT_METAL_VALIDATE_PROGRAM_BINARIES=1"
        },
        {
            "TT_METAL_DISABLE_DMA_OPS",
            EnvVarParserType::Bool01,
            [](RunTimeOptions* opts, const char* value) {
                if (value[0] == '1') {
                    opts->disable_dma_ops = true;
                }
            },
            "Disable DMA operations for debugging.",
            "0 (DMA enabled)",
            "export TT_METAL_DISABLE_DMA_OPS=1"
        },
        {
            "TT_METAL_ENABLE_ERISC_IRAM",
            EnvVarParserType::Bool01Inverted,
            [](RunTimeOptions* opts, const char* value) {
                bool disabled = (value[0] == '0');
                opts->erisc_iram_enabled = !disabled;
                opts->erisc_iram_enabled_env_var = !disabled;
            },
            "Enable ERISC IRAM functionality (inverted: 0=disabled, 1=enabled).",
            "1 (enabled)",
            "export TT_METAL_ENABLE_ERISC_IRAM=0  # to disable"
        },
        {
            "TT_METAL_CACHE",
            EnvVarParserType::StringWithSuffix,
            [](RunTimeOptions* opts, const char* value) {
                opts->is_cache_dir_env_var_set = true;
                opts->cache_dir_ = std::string(value) + "/tt-metal-cache/";
            },
            "Directory for caching compiled kernels and other build artifacts.",
            "Defaults to system temp directory if not set",
            "export TT_METAL_CACHE=/path/to/cache"
        },
        {
            "TT_METAL_KERNEL_PATH",
            EnvVarParserType::StringWithSuffix,
            [](RunTimeOptions* opts, const char* value) {
                opts->is_kernel_dir_env_var_set = true;
                opts->kernel_dir = std::string(value) + "/";
            },
            "Path to kernel source files.",
            "Uses TT_METAL_HOME/tt_metal/kernels if not set",
            "export TT_METAL_KERNEL_PATH=/path/to/kernels"
        },
        {
            "TT_METAL_SIMULATOR",
            EnvVarParserType::String,
            [](RunTimeOptions* opts, const char* value) {
                opts->simulator_path = std::string(value);
                opts->runtime_target_device_ = tt::TargetDevice::Simulator;
            },
            "Path to simulator executable for testing without hardware.",
            "Hardware mode (no simulator)",
            "export TT_METAL_SIMULATOR=/path/to/simulator"
        },
        {
            "TT_METAL_MOCK_CLUSTER_DESC_PATH",
            EnvVarParserType::String,
            [](RunTimeOptions* opts, const char* value) {
                opts->mock_cluster_desc_path = std::string(value);
                opts->runtime_target_device_ = tt::TargetDevice::Mock;
            },
            "Path to mock cluster descriptor for testing without hardware.",
            "Hardware mode (no mock cluster)",
            "export TT_METAL_MOCK_CLUSTER_DESC_PATH=/path/to/mock_cluster_desc.yaml"
        },
        {
            "TT_METAL_VISIBLE_DEVICES", 
            EnvVarParserType::String,
            [](RunTimeOptions* opts, const char* value) {
                opts->visible_devices = std::string(value);
            },
            "Comma-separated list of device IDs to make visible to the runtime.",
            "All devices visible",
            "export TT_METAL_VISIBLE_DEVICES=0,1,2"
        },
        {
            "ARCH_NAME",
            EnvVarParserType::String,
            [](RunTimeOptions* opts, const char* value) {
                opts->arch_name = std::string(value);
            },
            "Sets the architecture name (only necessary during simulation).",
            "Hardware-detected architecture",
            "export ARCH_NAME=wormhole_b0"
        },
        {
            "TT_METAL_TRACY_MID_RUN_PUSH",
            EnvVarParserType::BoolFlag,
            [](RunTimeOptions* opts, const char* value) {
                opts->tracy_mid_run_push = true;
            },
            "Forces Tracy profiler pushes during execution for real-time profiling.",
            "false (no mid-run pushes)",
            "export TT_METAL_TRACY_MID_RUN_PUSH=1"
        },
        {
            "TT_METAL_GTEST_NUM_HW_CQS",
            EnvVarParserType::UInt,
            [](RunTimeOptions* opts, const char* value) {
                try {
                    opts->set_num_hw_cqs(std::stoi(value));
                } catch (const std::invalid_argument& ia) {
                    TT_THROW("Invalid TT_METAL_GTEST_NUM_HW_CQS: {}", value);
                }
            },
            "Number of hardware command queues to use in tests.",
            "1",
            "export TT_METAL_GTEST_NUM_HW_CQS=4"
        },
        {
            "TT_METAL_ARC_DEBUG_BUFFER_SIZE",
            EnvVarParserType::UInt,
            [](RunTimeOptions* opts, const char* value) {
                sscanf(value, "%u", &opts->arc_debug_buffer_size);
            },
            "Buffer size in DRAM for storing ARC processor debug samples.",
            "0 (disabled)",
            "export TT_METAL_ARC_DEBUG_BUFFER_SIZE=1024"
        },
        {
            "TT_METAL_OPERATION_TIMEOUT_SECONDS",
            EnvVarParserType::Float,
            [](RunTimeOptions* opts, const char* value) {
                float timeout_duration = std::stof(value);
                opts->timeout_duration_for_operations = std::chrono::duration<float>(timeout_duration);
            },
            "Timeout duration for device operations in seconds.",
            "0.0 (no timeout)",
            "export TT_METAL_OPERATION_TIMEOUT_SECONDS=30.0"
        },
        {
            "TT_METAL_RISCV_DEBUG_INFO",
            EnvVarParserType::Complex,     // Use your existing Complex type
            [](RunTimeOptions* opts, const char* value) {
                bool enable_riscv_debug_info = opts->get_inspector_enabled(); // Default from inspector
                if (value != nullptr) {
                    enable_riscv_debug_info = true;                            // Default to true if set
                    if (strcmp(value, "0") == 0) {
                        enable_riscv_debug_info = false;                      // Only "0" = false
                    }
                }
                opts->set_riscv_debug_info_enabled(enable_riscv_debug_info);
            },
            "Enable RISC-V debug info. Defaults to inspector setting, override with 0/1.",
            "Inherits from inspector setting",
            "export TT_METAL_RISCV_DEBUG_INFO=1  # or =0 to disable"
        },
        {
            "TT_METAL_INSPECTOR",
            EnvVarParserType::Complex,
            [](RunTimeOptions* opts, const char* value) {
                if (value != nullptr) {
                    opts->inspector_settings.enabled = true;
                    if (strcmp(value, "0") == 0) {
                        opts->inspector_settings.enabled = false;
                    }
                }
            },
            "Enables or disables the inspector system. Set to '0' to disable, any other value enables it.",
            "true (inspector enabled)",
            "export TT_METAL_INSPECTOR=0"
        },
        {
            "TT_METAL_INSPECTOR_LOG_PATH",
            EnvVarParserType::Complex,
            [](RunTimeOptions* opts, const char* value) {
                if (value != nullptr) {
                    opts->inspector_settings.log_path = std::filesystem::path(value);
                } else {
                    opts->inspector_settings.log_path = std::filesystem::path(opts->get_root_dir()) / "generated/inspector";
                }
            },
            "Sets the log path for inspector output.",
            "Defaults to {TT_METAL_HOME}/generated/inspector if not specified",
            "export TT_METAL_INSPECTOR_LOG_PATH=/path/to/inspector/logs"
        },
        {
            "TT_METAL_INSPECTOR_INITIALIZATION_IS_IMPORTANT",
            EnvVarParserType::Complex,
            [](RunTimeOptions* opts, const char* value) {
                if (value != nullptr) {
                    opts->inspector_settings.initialization_is_important = true;
                    if (strcmp(value, "0") == 0) {
                        opts->inspector_settings.initialization_is_important = false;
                    }
                }
            },
            "Controls whether initialization is considered important for inspector. Set to '0' to disable.",
            "false (initialization not important)",
            "export TT_METAL_INSPECTOR_INITIALIZATION_IS_IMPORTANT=1"
        },
        {
            "TT_METAL_INSPECTOR_WARN_ON_WRITE_EXCEPTIONS",
            EnvVarParserType::Complex,
            [](RunTimeOptions* opts, const char* value) {
                if (value != nullptr) {
                    opts->inspector_settings.warn_on_write_exceptions = true;
                    if (strcmp(value, "0") == 0) {
                        opts->inspector_settings.warn_on_write_exceptions = false;
                    }
                }
            },
            "Controls whether to warn on write exceptions in inspector. Set to '0' to disable warnings.",
            "true (warn on write exceptions)",
            "export TT_METAL_INSPECTOR_WARN_ON_WRITE_EXCEPTIONS=0"
        },
        {
            "TT_MESH_GRAPH_DESC_PATH",
            EnvVarParserType::String,
            [](RunTimeOptions* opts, const char* value) {
                opts->is_custom_fabric_mesh_graph_desc_path_set = true;
                opts->custom_fabric_mesh_graph_desc_path = std::string(value);
            },
            "Custom fabric mesh graph descriptor path.",
            "Default fabric mesh configuration",
            "export TT_MESH_GRAPH_DESC_PATH=/path/to/mesh_desc.yaml"
        },
        {
            "TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE",
            EnvVarParserType::String,
            [](RunTimeOptions* opts, const char* value) {
                opts->is_core_grid_override_todeprecate_env_var_set = true;
                opts->core_grid_override_todeprecate = std::string(value);
            },
            "Override core grid configuration (deprecated).",
            "Hardware-detected core grid",
            "export TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE=custom_grid"
        }, 
        {
            "TT_METAL_DEVICE_PROFILER",
            EnvVarParserType::BoolFlag,
            [](RunTimeOptions* opts, const char* value) {
                opts->profiler_enabled = true;
            },
            "Enables device profiling (requires TRACY_ENABLE compilation flag).",
            "false (profiling disabled)",
            "export TT_METAL_DEVICE_PROFILER=1"
        },
        {
            "TT_METAL_PROFILER_SYNC",
            EnvVarParserType::BoolFlag,
            [](RunTimeOptions* opts, const char* value) {
                opts->profiler_sync_enabled = true;
            },
            "Enables synchronous profiling mode for more accurate timing.",
            "false (asynchronous profiling)",
            "export TT_METAL_PROFILER_SYNC=1"
        },
        {
            "TT_METAL_DEVICE_PROFILER_NOC_EVENTS",
            EnvVarParserType::BoolFlag,
            [](RunTimeOptions* opts, const char* value) {
                opts->profiler_enabled = true;
                opts->profiler_noc_events_enabled = true;
            },
            "Enables NoC (Network-on-Chip) events profiling.",
            "false (NoC events not profiled)",
            "export TT_METAL_DEVICE_PROFILER_NOC_EVENTS=1"
        },
        {
            "TT_METAL_DEVICE_PROFILER_NOC_EVENTS_RPT_PATH",
            EnvVarParserType::String,
            [](RunTimeOptions* opts, const char* value) {
                opts->profiler_noc_events_report_path = std::string(value);
            },
            "Sets the report path for NoC events profiling output files.",
            "Default report location",
            "export TT_METAL_DEVICE_PROFILER_NOC_EVENTS_RPT_PATH=/path/to/reports"
        },
        {
            "TT_METAL_MEM_PROFILER",
            EnvVarParserType::BoolFlag,
            [](RunTimeOptions* opts, const char* value) {
                opts->profiler_buffer_usage_enabled = true;
            },
            "Enables memory/buffer usage profiling for tracking memory allocation patterns.",
            "false (memory profiling disabled)",
            "export TT_METAL_MEM_PROFILER=1"
        },
        {
            "TT_METAL_TRACE_PROFILER",
            EnvVarParserType::BoolFlag,
            [](RunTimeOptions* opts, const char* value) {
                opts->profiler_trace_profiler = true;
            },
            "Enables trace profiler for detailed execution tracing.",
            "false (trace profiling disabled)",
            "export TT_METAL_TRACE_PROFILER=1"
        },
        {
            "TT_METAL_PROFILER_MID_RUN_DUMP",
            EnvVarParserType::BoolFlag,
            [](RunTimeOptions* opts, const char* value) {
                opts->profiler_mid_run_dump = true;
            },
            "Forces Tracy profiler dumps during execution for real-time profiling.",
            "false (no mid-run dumps)",
            "export TT_METAL_PROFILER_MID_RUN_DUMP=1"
        },
        {
            "TT_METAL_WATCHER_DUMP_ALL",
            EnvVarParserType::BoolFlag,
            [](RunTimeOptions* opts, const char* value) {
                opts->watcher_settings.dump_all = true;
            },
            "Enables dumping all watcher data, including potentially unsafe state information.",
            "false (safe data only)",
            "export TT_METAL_WATCHER_DUMP_ALL=1"
        },
        {
            "TT_METAL_WATCHER_APPEND",
            EnvVarParserType::BoolFlag,
            [](RunTimeOptions* opts, const char* value) {
                opts->watcher_settings.append = true;
            },
            "Enables append mode for watcher output files instead of overwriting.",
            "false (overwrite mode)",
            "export TT_METAL_WATCHER_APPEND=1"
        },
        {
            "TT_METAL_WATCHER_NOINLINE",
            EnvVarParserType::BoolFlag,
            [](RunTimeOptions* opts, const char* value) {
                opts->watcher_settings.noinline = true;
            },
            "Disables inlining for watcher functions to reduce binary size.",
            "false (inlining enabled)",
            "export TT_METAL_WATCHER_NOINLINE=1"
        },
        {
            "TT_METAL_WATCHER_PHYS_COORDS",
            EnvVarParserType::BoolFlag,
            [](RunTimeOptions* opts, const char* value) {
                opts->watcher_settings.phys_coords = true;
            },
            "Uses physical coordinates in watcher output instead of logical coordinates.",
            "false (logical coordinates)",
            "export TT_METAL_WATCHER_PHYS_COORDS=1"
        },
        {
            "TT_METAL_WATCHER_TEXT_START",
            EnvVarParserType::BoolFlag,
            [](RunTimeOptions* opts, const char* value) {
                opts->watcher_settings.text_start = true;
            },
            "Includes text start information in watcher output for debugging.",
            "false (no text start info)",
            "export TT_METAL_WATCHER_TEXT_START=1"
        },
        {
            "TT_METAL_WATCHER_SKIP_LOGGING",
            EnvVarParserType::BoolFlag,
            [](RunTimeOptions* opts, const char* value) {
                opts->watcher_settings.skip_logging = true;
            },
            "Disables watcher logging to reduce overhead.",
            "false (logging enabled)",
            "export TT_METAL_WATCHER_SKIP_LOGGING=1"
        },
        {
            "TT_METAL_WATCHER", 
            EnvVarParserType::Complex,
            [](RunTimeOptions* opts, const char* value) {
                if (value != nullptr) {
                    int sleep_val = 0;
                    sscanf(value, "%d", &sleep_val);
                    if (strstr(value, "ms") == nullptr) {
                        sleep_val *= 1000;
                    }
                    opts->watcher_settings.enabled = true;
                    opts->watcher_settings.interval_ms = sleep_val;
                }
            },
            "Enables the watcher system for debugging. When set, enables watcher on all features.",
            "disabled",
            "export TT_METAL_WATCHER=1"
        },
        {
            "TT_METAL_WATCHER_DISABLE_ASSERT",
            EnvVarParserType::Complex,
            [](RunTimeOptions* opts, const char* value) {
                opts->watcher_disabled_features.insert(opts->watcher_assert_str);  
            },
            "Disables watcher assert feature when set to any value.",
            "enabled",
            "export TT_METAL_WATCHER_DISABLE_ASSERT=1"
        },
        {
            "TT_METAL_WATCHER_DISABLE_PAUSE",
            EnvVarParserType::Complex,
            [](RunTimeOptions* opts, const char* value) {
                opts->watcher_disabled_features.insert(opts->watcher_pause_str); 
            },
            "Disables watcher pause feature when set to any value.",
            "enabled",
            "export TT_METAL_WATCHER_DISABLE_PAUSE=1"
        },
        {
            "TT_METAL_WATCHER_DISABLE_RING_BUFFER",
            EnvVarParserType::Complex,
            [](RunTimeOptions* opts, const char* value) {
                opts->watcher_disabled_features.insert(opts->watcher_ring_buffer_str);
            },
            "Disables watcher ring buffer feature when set to any value.",
            "enabled",
            "export TT_METAL_WATCHER_DISABLE_RING_BUFFER=1"
        },
        {
            "TT_METAL_WATCHER_DISABLE_STACK_USAGE",
            EnvVarParserType::Complex,
            [](RunTimeOptions* opts, const char* value) {
                opts->watcher_disabled_features.insert(opts->watcher_stack_usage_str);
            },
            "Disables watcher stack usage tracking when set to any value.",
            "enabled", 
            "export TT_METAL_WATCHER_DISABLE_STACK_USAGE=1"
        },
        {
            "TT_METAL_WATCHER_DISABLE_SANITIZE_NOC",
            EnvVarParserType::Complex,
            [](RunTimeOptions* opts, const char* value) {
                opts->watcher_disabled_features.insert(opts->watcher_noc_sanitize_str);
            },
            "Disables watcher NoC sanitization when set to any value.",
            "enabled",
            "export TT_METAL_WATCHER_DISABLE_SANITIZE_NOC=1" 
        },
        {
            "TT_METAL_WATCHER_DISABLE_SANITIZE_READ_ONLY_L1",
            EnvVarParserType::Complex,
            [](RunTimeOptions* opts, const char* value) {
                opts->watcher_disabled_features.insert(opts->watcher_sanitize_read_only_l1_str);
            },
            "Disables watcher read-only L1 sanitization when set to any value.",
            "enabled",
            "export TT_METAL_WATCHER_DISABLE_SANITIZE_READ_ONLY_L1=1"
        },
        {
            "TT_METAL_WATCHER_DISABLE_SANITIZE_WRITE_ONLY_L1", 
            EnvVarParserType::Complex,
            [](RunTimeOptions* opts, const char* value) {
                opts->watcher_disabled_features.insert(opts->watcher_sanitize_write_only_l1_str);
            },
            "Disables watcher write-only L1 sanitization when set to any value.",
            "enabled",
            "export TT_METAL_WATCHER_DISABLE_SANITIZE_WRITE_ONLY_L1=1"
        },
        {
            "TT_METAL_WATCHER_DISABLE_WAYPOINT",
            EnvVarParserType::Complex,
            [](RunTimeOptions* opts, const char* value) {
                opts->watcher_disabled_features.insert(opts->watcher_waypoint_str);
            },
            "Disables watcher waypoint feature when set to any value.",
            "enabled",
            "export TT_METAL_WATCHER_DISABLE_WAYPOINT=1"
        },
        {
            "TT_METAL_WATCHER_DISABLE_DISPATCH",
            EnvVarParserType::Complex,
            [](RunTimeOptions* opts, const char* value) {
                opts->watcher_disabled_features.insert(opts->watcher_dispatch_str);
            },
            "Disables watcher dispatch feature when set to any value.",
            "enabled",
            "export TT_METAL_WATCHER_DISABLE_DISPATCH=1"
        },
        {
            "TT_METAL_INSPECTOR",
            EnvVarParserType::Complex,
            [](RunTimeOptions* opts, const char* value) {
                opts->inspector_settings.enabled = true;
                if (value != nullptr && strcmp(value, "0") == 0) {
                    opts->inspector_settings.enabled = false;
                }
            },
            "Enables or disables the inspector system. Set to '0' to disable, any other value enables it.",
            "true (enabled)",
            "export TT_METAL_INSPECTOR=1"
        },
        {
            "TT_METAL_INSPECTOR_LOG_PATH", 
            EnvVarParserType::String,
            [](RunTimeOptions* opts, const char* value) {
                opts->inspector_settings.log_path = std::filesystem::path(value);
            },
            "Sets the log path for inspector output.",
            "Defaults to {TT_METAL_HOME}/generated/inspector",
            "export TT_METAL_INSPECTOR_LOG_PATH=/path/to/inspector/logs"
        },
        {
            "TT_METAL_INSPECTOR_INITIALIZATION_IS_IMPORTANT",
            EnvVarParserType::Complex,
            [](RunTimeOptions* opts, const char* value) {
                opts->inspector_settings.initialization_is_important = true;
                if (value != nullptr && strcmp(value, "0") == 0) {
                    opts->inspector_settings.initialization_is_important = false;
                }
            },
            "Controls whether initialization is considered important for inspector. Set to '0' to disable.",
            "false (not important)",
            "export TT_METAL_INSPECTOR_INITIALIZATION_IS_IMPORTANT=1"
        },
        {
            "TT_METAL_INSPECTOR_WARN_ON_WRITE_EXCEPTIONS",
            EnvVarParserType::Complex,
            [](RunTimeOptions* opts, const char* value) {
                opts->inspector_settings.warn_on_write_exceptions = true;
                if (value != nullptr && strcmp(value, "0") == 0) {
                    opts->inspector_settings.warn_on_write_exceptions = false;
                }
            },
            "Controls whether to warn on write exceptions in inspector. Set to '0' to disable warnings.",
            "true (warnings enabled)",
            "export TT_METAL_INSPECTOR_WARN_ON_WRITE_EXCEPTIONS=0"
        },
        {
            "TT_METAL_SKIP_LOADING_FW",
            EnvVarParserType::BoolFlag,
            [](RunTimeOptions* opts, const char* value) {
                opts->skip_loading_fw = true;
            },
            "Skips loading firmware when set, useful for testing scenarios.",
            "false (firmware loading enabled)",
            "export TT_METAL_SKIP_LOADING_FW=1"
        },
        {
            "TT_METAL_SKIP_DELETING_BUILT_CACHE",
            EnvVarParserType::BoolFlag,
            [](RunTimeOptions* opts, const char* value) {
                opts->skip_deleting_built_cache = true;
            },
            "Skips deletion of built cache when set, preserving compiled kernels between runs.",
            "false (cache deletion enabled)",
            "export TT_METAL_SKIP_DELETING_BUILT_CACHE=1"
        },
        {
            "TT_METAL_DPRINT_CORES",
            EnvVarParserType::Complex,
            [](RunTimeOptions* opts, const char* value) {
                // Handled by ParseFeatureEnv() - this is for documentation
            },
            "Specifies worker cores for debug printing. Supports 'all', ranges '(1,1)-(2,2)', or lists '(1,1),(2,2)'.",
            "disabled (no debug printing)",
            "export TT_METAL_DPRINT_CORES=all"
        },
        {
            "TT_METAL_DPRINT_ETH_CORES", 
            EnvVarParserType::Complex,
            [](RunTimeOptions* opts, const char* value) {
                // Handled by ParseFeatureEnv() - this is for documentation
            },
            "Specifies Ethernet cores for debug printing. Same syntax as DPRINT_CORES.",
            "disabled (no debug printing on ETH cores)",
            "export TT_METAL_DPRINT_ETH_CORES=(0,0),(1,0)"
        },
        {
            "TT_METAL_DPRINT_CHIPS",
            EnvVarParserType::Complex,
            [](RunTimeOptions* opts, const char* value) {
                // Handled by ParseFeatureEnv() - this is for documentation  
            },
            "Specifies chip IDs for debug printing. Supports 'all' or comma-separated list of chip IDs.",
            "all chips",
            "export TT_METAL_DPRINT_CHIPS=0,1,2"
        },
        {
            "TT_METAL_DPRINT_RISCVS",
            EnvVarParserType::Complex,
            [](RunTimeOptions* opts, const char* value) {
                // Handled by ParseFeatureEnv() - this is for documentation
            },
            "Specifies RISC-V processors for debug printing. Complex processor selection syntax.",
            "all RISC-V processors",
            "export TT_METAL_DPRINT_RISCVS=BR+NCRISC+TRISC0"
        },
        {
            "TT_METAL_DPRINT_FILE",
            EnvVarParserType::Complex,
            [](RunTimeOptions* opts, const char* value) {
                // Handled by ParseFeatureEnv() - this is for documentation
            },
            "Output file path for debug printing. If not specified, prints to stdout.",
            "stdout",
            "export TT_METAL_DPRINT_FILE=/tmp/debug_output.log"
        },
        {
            "TT_METAL_DPRINT_ONE_FILE_PER_RISC",
            EnvVarParserType::Complex,
            [](RunTimeOptions* opts, const char* value) {
                // Handled by ParseFeatureEnv() - this is for documentation
            },
            "Creates separate output files for each RISC-V processor when set.",
            "false (single output file)",
            "export TT_METAL_DPRINT_ONE_FILE_PER_RISC=1"
        },
        {
            "TT_METAL_DPRINT_PREPEND_DEVICE_CORE_RISC",
            EnvVarParserType::Complex,
            [](RunTimeOptions* opts, const char* value) {
                // Handled by ParseFeatureEnv() - this is for documentation
            },
            "Prepends device/core/RISC information to each debug print line. Set to '0' to disable.",
            "true (prepend enabled)",
            "export TT_METAL_DPRINT_PREPEND_DEVICE_CORE_RISC=0"
        },
        {
            "TT_METAL_WATCHER_ENABLE_NOC_SANITIZE_LINKED_TRANSACTION",
            EnvVarParserType::BoolFlag,
            [](RunTimeOptions* opts, const char* value) {
                opts->watcher_settings.noc_sanitize_linked_transaction = true;
            },
            "Enables NoC sanitization for linked transactions to catch more subtle errors.",
            "false (linked transaction sanitization disabled)",
            "export TT_METAL_WATCHER_ENABLE_NOC_SANITIZE_LINKED_TRANSACTION=1"
        }
        
    };
    return ENV_VAR_TABLE;
}


RunTimeOptions::RunTimeOptions() {
    /*const char* root_dir_str = std::getenv(TT_METAL_HOME_ENV_VAR);
    if (root_dir_str != nullptr) {
        this->is_root_dir_env_var_set = true;
        this->root_dir = std::string(root_dir_str) + "/";
    }*/

    // Check if user has specified a cache path.
    /*const char* cache_dir_str = std::getenv(TT_METAL_CACHE_ENV_VAR);
    if (cache_dir_str != nullptr) {
        this->is_cache_dir_env_var_set = true;
        this->cache_dir_ = std::string(cache_dir_str) + "/tt-metal-cache/";
    }

    const char* kernel_dir_str = std::getenv(TT_METAL_KERNEL_PATH_ENV_VAR);
    if (kernel_dir_str != nullptr) {
        this->is_kernel_dir_env_var_set = true;
        this->kernel_dir = std::string(kernel_dir_str) + "/";
    }*/
    this->system_kernel_dir = "/usr/share/tenstorrent/kernels/";

    /*const char* custom_fabric_mesh_graph_desc_path_str = std::getenv("TT_MESH_GRAPH_DESC_PATH");
    if (custom_fabric_mesh_graph_desc_path_str != nullptr) {
        this->is_custom_fabric_mesh_graph_desc_path_set = true;
        this->custom_fabric_mesh_graph_desc_path = std::string(custom_fabric_mesh_graph_desc_path_str);
    }*/

    /*const char* core_grid_override_todeprecate_str = std::getenv(TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE_ENV_VAR);
    if (core_grid_override_todeprecate_str != nullptr) {
        this->is_core_grid_override_todeprecate_env_var_set = true;
        this->core_grid_override_todeprecate = std::string(core_grid_override_todeprecate_str);
    }*/

    //build_map_enabled = (getenv("TT_METAL_KERNEL_MAP") != nullptr);

    InitializeFromTable();
    //ParseWatcherEnv();
    //ParseInspectorEnv();

    //test_mode_enabled = (getenv("TT_METAL_WATCHER_TEST_MODE") != nullptr);

    profiler_enabled = false;
    profile_dispatch_cores = false;
    profiler_sync_enabled = false;
    profiler_mid_run_dump = false;
    profiler_buffer_usage_enabled = false;
    profiler_trace_profiler = false;
#if defined(TRACY_ENABLE)
    /*const char* profiler_enabled_str = std::getenv("TT_METAL_DEVICE_PROFILER");
    if (profiler_enabled_str != nullptr && profiler_enabled_str[0] == '1') {
        profiler_enabled = true;*/
        /*const char* profile_dispatch_str = std::getenv("TT_METAL_DEVICE_PROFILER_DISPATCH");
        if (profile_dispatch_str != nullptr && profile_dispatch_str[0] == '1') {
            profile_dispatch_cores = true;
        }*/
        /*const char* profiler_sync_enabled_str = std::getenv("TT_METAL_PROFILER_SYNC");
        if (profiler_sync_enabled_str != nullptr && profiler_sync_enabled_str[0] == '1') {
            profiler_sync_enabled = true;
        }
        const char* profiler_trace_profiler_str = std::getenv("TT_METAL_TRACE_PROFILER");
        if (profiler_trace_profiler_str != nullptr && profiler_trace_profiler_str[0] == '1') {
            profiler_trace_profiler = true;
        }
        const char* profiler_mid_run_dump_str = std::getenv("TT_METAL_PROFILER_MID_RUN_DUMP");
        if (profiler_mid_run_dump_str != nullptr && profiler_mid_run_dump_str[0] == '1') {
            profiler_mid_run_dump = true;
        }
    }

    const char *profiler_noc_events_str = std::getenv("TT_METAL_DEVICE_PROFILER_NOC_EVENTS");
    if (profiler_noc_events_str != nullptr && profiler_noc_events_str[0] == '1') {
        profiler_enabled = true;
        profiler_noc_events_enabled = true;
    }

    const char *profiler_noc_events_report_path_str = std::getenv("TT_METAL_DEVICE_PROFILER_NOC_EVENTS_RPT_PATH");
    if (profiler_noc_events_report_path_str != nullptr) {
        profiler_noc_events_report_path = profiler_noc_events_report_path_str;
    }

    const char *profile_buffer_usage_str = std::getenv("TT_METAL_MEM_PROFILER");
    if (profile_buffer_usage_str != nullptr && profile_buffer_usage_str[0] == '1') {
        profiler_buffer_usage_enabled = true;
    }*/
#endif
    TT_FATAL(
        !(get_feature_enabled(RunTimeDebugFeatureDprint) && get_profiler_enabled()),
        "Cannot enable both debug printing and profiling");

    /*null_kernels = (std::getenv("TT_METAL_NULL_KERNELS") != nullptr);

    kernels_early_return = (std::getenv("TT_METAL_KERNELS_EARLY_RETURN") != nullptr);

    this->clear_l1 = false;
    const char* clear_l1_enabled_str = std::getenv("TT_METAL_CLEAR_L1");
    if (clear_l1_enabled_str != nullptr && clear_l1_enabled_str[0] == '1') {
        this->clear_l1 = true;
    }

    this->clear_dram = false;
    const char* clear_dram_enabled_str = std::getenv("TT_METAL_CLEAR_DRAM");
    if (clear_dram_enabled_str != nullptr && clear_dram_enabled_str[0] == '1') {
        this->clear_dram = true;
    }

    const char* skip_eth_cores_with_retrain_str = std::getenv("TT_METAL_SKIP_ETH_CORES_WITH_RETRAIN");
    if (skip_eth_cores_with_retrain_str != nullptr) {
        if (skip_eth_cores_with_retrain_str[0] == '0') {
            skip_eth_cores_with_retrain = false;
        }
        if (skip_eth_cores_with_retrain_str[0] == '1') {
            skip_eth_cores_with_retrain = true;
        }
    }*/

   /* const char* riscv_debug_info_enabled_str = std::getenv("TT_METAL_RISCV_DEBUG_INFO");
    bool enable_riscv_debug_info = get_inspector_enabled();
    if (riscv_debug_info_enabled_str != nullptr) {
        enable_riscv_debug_info = true;
        if (strcmp(riscv_debug_info_enabled_str, "0") == 0) {
            enable_riscv_debug_info = false;
        }
    }
    set_riscv_debug_info_enabled(enable_riscv_debug_info); */

    /*const char* validate_kernel_binaries = std::getenv("TT_METAL_VALIDATE_PROGRAM_BINARIES");
    set_validate_kernel_binaries(validate_kernel_binaries != nullptr && validate_kernel_binaries[0] == '1');

    const char* num_cqs = getenv("TT_METAL_GTEST_NUM_HW_CQS");
    if (num_cqs != nullptr) {
        try {
            set_num_hw_cqs(std::stoi(num_cqs));
        } catch (const std::invalid_argument& ia) {
            TT_THROW("Invalid TT_METAL_GTEST_NUM_HW_CQS: {}", num_cqs);
        }
    }*/

    //using_slow_dispatch = getenv("TT_METAL_SLOW_DISPATCH_MODE") != nullptr;

    /*const char* dispatch_data_collection_str = std::getenv("TT_METAL_DISPATCH_DATA_COLLECTION");
    if (dispatch_data_collection_str != nullptr) {
        enable_dispatch_data_collection = true;
    }*/

    /*if (getenv("TT_METAL_GTEST_ETH_DISPATCH")) {
        this->dispatch_core_type = tt_metal::DispatchCoreType::ETH;
    }

    if (getenv("TT_METAL_SKIP_LOADING_FW")) {
        this->skip_loading_fw = true;
    }

    if (getenv("TT_METAL_SKIP_DELETING_BUILT_CACHE")) {
        this->skip_deleting_built_cache = true;
    }

    if (getenv("TT_METAL_ENABLE_HW_CACHE_INVALIDATION")) {
        this->enable_hw_cache_invalidation = true;
    }*/

    /*if (std::getenv("TT_METAL_SIMULATOR")) {
        this->simulator_path = std::getenv("TT_METAL_SIMULATOR");
        this->runtime_target_device_ = tt::TargetDevice::Simulator;
    }*/

    // Enable mock cluster if TT_METAL_MOCK is set to a descriptor path
    // This is used for initializing UMD without any hardware using a mock cluster descriptor
    /*if (const char* mock_path = std::getenv("TT_METAL_MOCK_CLUSTER_DESC_PATH")) {
        this->mock_cluster_desc_path = std::string(mock_path);
        this->runtime_target_device_ = tt::TargetDevice::Mock;
    }*/

    /*if (auto str = getenv("TT_METAL_ENABLE_ERISC_IRAM")) {
        bool disabled = strcmp(str, "0") == 0;
        this->erisc_iram_enabled = !disabled;
        this->erisc_iram_enabled_env_var = !disabled;
    }
    this->fast_dispatch = (std::getenv("TT_METAL_SLOW_DISPATCH_MODE") == nullptr);*/

    /*if (getenv("TT_METAL_DISABLE_RELAXED_MEM_ORDERING")) {
        this->disable_relaxed_memory_ordering = true;
    }

    if (getenv("TT_METAL_ENABLE_GATHERING")) {
        this->enable_gathering = true;
    } */

    /*const char* arc_debug_enabled_str = std::getenv("TT_METAL_ARC_DEBUG_BUFFER_SIZE");
    if (arc_debug_enabled_str != nullptr) {
        sscanf(arc_debug_enabled_str, "%u", &arc_debug_buffer_size);
    }*/

    /*const char* disable_dma_ops_str = std::getenv("TT_METAL_DISABLE_DMA_OPS");
    if (disable_dma_ops_str != nullptr) {
        if (disable_dma_ops_str[0] == '1') {
            this->disable_dma_ops = true;
        }
    }*/

    /*if (getenv("TT_METAL_FABRIC_TELEMETRY")) {
        enable_fabric_telemetry = true;
    }

    if (getenv("TT_METAL_FORCE_REINIT")) {
        force_context_reinit = true;
    }

    if (getenv("TT_METAL_FABRIC_BLACKHOLE_TWO_ERISC")) {
        this->enable_2_erisc_mode_with_fabric = true;
    }

    if (getenv("TT_METAL_MULTI_AERISC")) {
        log_info(tt::LogMetal, "Enabling experimental multi-erisc mode");
        this->enable_2_erisc_mode = true;
    }

    if (getenv("TT_METAL_LOG_KERNELS_COMPILE_COMMANDS")) {
        this->log_kernels_compilation_commands = true;
    }*/

    /*const char* timeout_duration_for_operations_value = std::getenv("TT_METAL_OPERATION_TIMEOUT_SECONDS");
    float timeout_duration_for_operations =
        timeout_duration_for_operations_value ? std::stof(timeout_duration_for_operations_value) : 0.f;
    this->timeout_duration_for_operations = std::chrono::duration<float>(timeout_duration_for_operations);*/
    VerifyTableDrivenParsing();
}

const std::string& RunTimeOptions::get_root_dir() const {
    if (!this->is_root_dir_specified()) {
        TT_THROW("Env var {} is not set.", TT_METAL_HOME_ENV_VAR);
    }

    return root_dir;
}

const std::string& RunTimeOptions::get_cache_dir() const {
    if (!this->is_cache_dir_specified()) {
        TT_THROW("Env var {} is not set.", TT_METAL_CACHE_ENV_VAR);
    }
    return this->cache_dir_;
}

const std::string& RunTimeOptions::get_kernel_dir() const {
    if (!this->is_kernel_dir_specified()) {
        TT_THROW("Env var {} is not set.", TT_METAL_KERNEL_PATH_ENV_VAR);
    }

    return this->kernel_dir;
}

const std::string& RunTimeOptions::get_core_grid_override_todeprecate() const {
    if (!this->is_core_grid_override_todeprecate()) {
        TT_THROW("Env var {} is not set.", TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE_ENV_VAR);
    }

    return this->core_grid_override_todeprecate;
}

const std::string& RunTimeOptions::get_system_kernel_dir() const { return this->system_kernel_dir; }

void RunTimeOptions::ParseBoolFlag(const char* env_var, bool& target) {
    target = (std::getenv(env_var) != nullptr);
}
void RunTimeOptions::ParseBool01(const char* env_var, bool& target) {
    target = false;
    if (const char* v = std::getenv(env_var)) {
        target = (v[0] == '1');  // strict legacy semantics
    }
}
void RunTimeOptions::ParseString(const char* env_var, std::string& target,const char* suffix, bool* was_set) {
        if (was_set) *was_set = false;
        if (const char* v = std::getenv(env_var)) {
            target.assign(v);
            if (suffix && !target.empty() && target.back() != suffix[0]) {
        // append suffix ONLY if it's not already there
            target += suffix;
        }
        if (was_set) *was_set = true;
    }
}

void RunTimeOptions::ParseInt(const char* env_var, int& target) {
    if (const char* v = std::getenv(env_var)) {
        try {
            target = std::stoi(v);
        } catch (const std::invalid_argument&) {
            TT_THROW("Invalid integer value for environment variable '{}' for {}; expected integer (e.g., 4).", v, env_var);
        } catch (const std::out_of_range&) {
            TT_THROW("Out-of-range integer for environment variable '{}' for {}.", v, env_var);
        }
    }
}

void RunTimeOptions::ParseUInt(const char* env_var, unsigned& target) {
    if (const char* v = std::getenv(env_var)) {
        try {
            long long tmp = std::stoll(v);
            if (tmp < 0) {
                TT_THROW("Invalid integer value for environment variable '{}' for {}; expected unsigned integer.", v, env_var);
            }
            target = static_cast<unsigned>(tmp);
        } catch (const std::invalid_argument&) {
            TT_THROW("Invalid value '{}' for {}; expected unsigned integer (e.g., 4).", v, env_var);
        } catch (const std::out_of_range&) {
            TT_THROW("Out-of-range unsigned integer for environment variable '{}' for {}.", v, env_var);
        }
    }
}

void RunTimeOptions::ParseFloat(const char* env_var, float& target) {
    if (const char* v = std::getenv(env_var)) {
        try {
            target = std::stof(v);
        } catch (const std::invalid_argument&) {
            TT_THROW("Invalid value '{}' for {}; expected float (e.g., 0.5).", v, env_var);
        } catch (const std::out_of_range&) {
            TT_THROW("Out-of-range float '{}' for {}.", v, env_var);
        }
    }
}
void RunTimeOptions::InitializeFromTable() {
    for (const auto& descriptor : GetEnvVarTable()) {
        const char* env_value = std::getenv(descriptor.env_var_name);
        
        // For Complex type, always call the parser (even if env_value is nullptr)
        if (descriptor.parser_type == EnvVarParserType::Complex) {
            descriptor.parser_func(this, env_value);  // Pass nullptr if not set
        }
        // For other types, only call if set
        else if (env_value != nullptr && descriptor.parser_func) {
            descriptor.parser_func(this, env_value);
        }
    }
}
void RunTimeOptions::VerifyTableDrivenParsing() {
    std::cout << "🔍 VERIFICATION: Running table-driven parsing verification..." << std::endl;
    // Store current values (from manual parsing)
    bool manual_null_kernels = null_kernels;
    bool manual_kernels_early_return = kernels_early_return; 
    bool manual_clear_l1 = clear_l1;
    bool manual_clear_dram = clear_dram;
    
    // Reset these variables to defaults
    null_kernels = false;
    kernels_early_return = false;
    clear_l1 = false;
    clear_dram = false;
    
    // Run table-driven parsing
    InitializeFromTable();
    
    // Store table results
    bool table_null_kernels = null_kernels;
    bool table_kernels_early_return = kernels_early_return;
    bool table_clear_l1 = clear_l1;
    bool table_clear_dram = clear_dram;
    
    // Restore manual parsing results
    null_kernels = manual_null_kernels;
    kernels_early_return = manual_kernels_early_return;
    clear_l1 = manual_clear_l1;
    clear_dram = manual_clear_dram;
    
    // Verify they match - if they don't, we have a bug in our table
    TT_ASSERT(table_null_kernels == manual_null_kernels, 
              "TT_METAL_NULL_KERNELS mismatch: table={} vs manual={}", table_null_kernels, manual_null_kernels);
    TT_ASSERT(table_kernels_early_return == manual_kernels_early_return,
              "TT_METAL_KERNELS_EARLY_RETURN mismatch: table={} vs manual={}", table_kernels_early_return, manual_kernels_early_return);
    TT_ASSERT(table_clear_l1 == manual_clear_l1,
              "TT_METAL_CLEAR_L1 mismatch: table={} vs manual={}", table_clear_l1, manual_clear_l1);
    TT_ASSERT(table_clear_dram == manual_clear_dram,
              "TT_METAL_CLEAR_DRAM mismatch: table={} vs manual={}", table_clear_dram, manual_clear_dram);
    std::cout << " VERIFICATION: All checks passed! Tested: null_kernels=" << manual_null_kernels 
              << ", clear_l1=" << manual_clear_l1 << ", clear_dram=" << manual_clear_dram 
              << ", kernels_early_return=" << manual_kernels_early_return << std::endl;          
  
}


void RunTimeOptions::ParseWatcherEnv() {
    const char* watcher_enable_str = getenv("TT_METAL_WATCHER");
    if (watcher_enable_str != nullptr) {
        int sleep_val = 0;
        sscanf(watcher_enable_str, "%d", &sleep_val);
        if (strstr(watcher_enable_str, "ms") == nullptr) {
            sleep_val *= 1000;
        }
        watcher_settings.enabled = true;
        watcher_settings.interval_ms = sleep_val;
    }

    /*watcher_settings.dump_all = (getenv("TT_METAL_WATCHER_DUMP_ALL") != nullptr);
    watcher_settings.append = (getenv("TT_METAL_WATCHER_APPEND") != nullptr);
    watcher_settings.noinline = (getenv("TT_METAL_WATCHER_NOINLINE") != nullptr);
    watcher_settings.phys_coords = (getenv("TT_METAL_WATCHER_PHYS_COORDS") != nullptr);
    watcher_settings.text_start = (getenv("TT_METAL_WATCHER_TEXT_START") != nullptr);
    watcher_settings.skip_logging = (getenv("TT_METAL_WATCHER_SKIP_LOGGING") != nullptr);
    watcher_settings.noc_sanitize_linked_transaction =
        (getenv("TT_METAL_WATCHER_ENABLE_NOC_SANITIZE_LINKED_TRANSACTION") != nullptr);*/
    // Auto unpause is for testing only, no env var.
    watcher_settings.auto_unpause = false;

    // Any watcher features to disabled based on env var.
    std::set all_features = {
        watcher_waypoint_str,
        watcher_noc_sanitize_str,
        watcher_assert_str,
        watcher_pause_str,
        watcher_ring_buffer_str,
        watcher_stack_usage_str,
        watcher_dispatch_str,
        watcher_eth_link_status_str};
    for (const std::string& feature : all_features) {
        std::string env_var("TT_METAL_WATCHER_DISABLE_");
        env_var += feature;
        if (getenv(env_var.c_str()) != nullptr) {
            watcher_disabled_features.insert(feature);
        }
    }

    const char* watcher_debug_delay_str = getenv("TT_METAL_WATCHER_DEBUG_DELAY");
    if (watcher_debug_delay_str != nullptr) {
        sscanf(watcher_debug_delay_str, "%u", &watcher_debug_delay);
        // Assert watcher is also enabled (TT_METAL_WATCHER=1)
        TT_ASSERT(watcher_settings.enabled, "TT_METAL_WATCHER_DEBUG_DELAY requires TT_METAL_WATCHER");
        // Assert TT_METAL_WATCHER_DISABLE_NOC_SANITIZE is either not set or set to 0
        TT_ASSERT(
            watcher_disabled_features.find(watcher_noc_sanitize_str) == watcher_disabled_features.end(),
            "TT_METAL_WATCHER_DEBUG_DELAY requires TT_METAL_WATCHER_DISABLE_NOC_SANITIZE=0");
    }
    if (watcher_settings.noc_sanitize_linked_transaction) {
        TT_ASSERT(
            watcher_disabled_features.find(watcher_noc_sanitize_str) == watcher_disabled_features.end(),
            "TT_METAL_WATCHER_ENABLE_NOC_SANITIZE_LINKED_TRANSACTION requires TT_METAL_WATCHER_DISABLE_NOC_SANITIZE=0");
    }
}

void RunTimeOptions::ParseInspectorEnv() {
    /*const char* inspector_enable_str = getenv("TT_METAL_INSPECTOR");
    if (inspector_enable_str != nullptr) {
        inspector_settings.enabled = true;
        if (strcmp(inspector_enable_str, "0") == 0) {
            inspector_settings.enabled = false;
        }
    }*/

    /*const char* inspector_log_path_str = getenv("TT_METAL_INSPECTOR_LOG_PATH");
    if (inspector_log_path_str != nullptr) {
        inspector_settings.log_path = std::filesystem::path(inspector_log_path_str);
    } else {
        inspector_settings.log_path = std::filesystem::path(get_root_dir()) / "generated/inspector";
    }*/

    /*const char* inspector_initialization_is_important_str = getenv("TT_METAL_INSPECTOR_INITIALIZATION_IS_IMPORTANT");
    if (inspector_initialization_is_important_str != nullptr) {
        inspector_settings.initialization_is_important = true;
        if (strcmp(inspector_initialization_is_important_str, "0") == 0) {
            inspector_settings.initialization_is_important = false;
        }
    } */

    /*const char* inspector_warn_on_write_exceptions_str = getenv("TT_METAL_INSPECTOR_WARN_ON_WRITE_EXCEPTIONS");
    if (inspector_warn_on_write_exceptions_str != nullptr) {
        inspector_settings.warn_on_write_exceptions = true;
        if (strcmp(inspector_warn_on_write_exceptions_str, "0") == 0) {
            inspector_settings.warn_on_write_exceptions = false;
        }
    }*/
}

void RunTimeOptions::ParseFeatureEnv(RunTimeDebugFeatures feature, const tt_metal::Hal& hal) {
    std::string feature_env_prefix("TT_METAL_");
    feature_env_prefix += RunTimeDebugFeatureNames[feature];

    ParseFeatureCoreRange(feature, feature_env_prefix + "_CORES", CoreType::WORKER);
    ParseFeatureCoreRange(feature, feature_env_prefix + "_ETH_CORES", CoreType::ETH);
    ParseFeatureChipIds(feature, feature_env_prefix + "_CHIPS");
    ParseFeatureRiscvMask(feature, feature_env_prefix + "_RISCVS", hal);
    ParseFeatureFileName(feature, feature_env_prefix + "_FILE");
    ParseFeatureOneFilePerRisc(feature, feature_env_prefix + "_ONE_FILE_PER_RISC");
    ParseFeaturePrependDeviceCoreRisc(feature, feature_env_prefix + "_PREPEND_DEVICE_CORE_RISC");

    // Set feature enabled if the user asked for any feature cores
    feature_targets[feature].enabled = false;
    for (auto& core_type_and_all_flag : feature_targets[feature].all_cores) {
        if (core_type_and_all_flag.second != RunTimeDebugClassNoneSpecified) {
            feature_targets[feature].enabled = true;
        }
    }
    for (auto& core_type_and_cores : feature_targets[feature].cores) {
        if (core_type_and_cores.second.size() > 0) {
            feature_targets[feature].enabled = true;
        }
    }

    const char* print_noc_xfers = std::getenv("TT_METAL_RECORD_NOC_TRANSFER_DATA");
    if (print_noc_xfers != nullptr) {
        record_noc_transfer_data = true;
    }
};

void RunTimeOptions::ParseFeatureCoreRange(
    RunTimeDebugFeatures feature, const std::string& env_var, CoreType core_type) {
    char* str = std::getenv(env_var.c_str());
    std::vector<CoreCoord> cores;

    // Check if "all" is specified, rather than a range of cores.
    feature_targets[feature].all_cores[core_type] = RunTimeDebugClassNoneSpecified;
    if (str != nullptr) {
        for (int idx = 0; idx < RunTimeDebugClassCount; idx++) {
            if (strcmp(str, RunTimeDebugClassNames[idx]) == 0) {
                feature_targets[feature].all_cores[core_type] = idx;
                return;
            }
        }
    }
    if (str != nullptr) {
        if (isdigit(str[0])) {
            // Assume this is a single core
            uint32_t x, y;
            if (sscanf(str, "%d,%d", &x, &y) != 2) {
                TT_THROW("Invalid {}", env_var);
            }
            cores.push_back({x, y});
        } else if (str[0] == '(') {
            if (strchr(str, '-')) {
                // Assume this is a range
                CoreCoord start, end;
                if (sscanf(str, "(%zu,%zu)", &start.x, &start.y) != 2) {
                    TT_THROW("Invalid {}", env_var);
                }
                str = strchr(str, '-');
                if (sscanf(str, "-(%zu,%zu)", &end.x, &end.y) != 2) {
                    TT_THROW("Invalid {}", env_var);
                }
                for (uint32_t x = start.x; x <= end.x; x++) {
                    for (uint32_t y = start.y; y <= end.y; y++) {
                        cores.push_back({x, y});
                    }
                }
            } else {
                // Assume this is a list of coordinates (maybe just one)
                while (str != nullptr) {
                    uint32_t x, y;
                    if (sscanf(str, "(%d,%d)", &x, &y) != 2) {
                        TT_THROW("Invalid {}", env_var);
                    }
                    cores.push_back({x, y});
                    str = strchr(str, ',');
                    str = strchr(str + 1, ',');
                    if (str != nullptr) {
                        str++;
                    }
                }
            }
        } else {
            TT_THROW("Invalid {}", env_var);
        }
    }

    // Set the core range
    feature_targets[feature].cores[core_type] = cores;
}

void RunTimeOptions::ParseFeatureChipIds(RunTimeDebugFeatures feature, const std::string& env_var) {
    std::vector<int> chips;
    char* env_var_str = std::getenv(env_var.c_str());

    // If the environment variable is not empty, parse it.
    while (env_var_str != nullptr) {
        // Can also have "all"
        if (strcmp(env_var_str, "all") == 0) {
            feature_targets[feature].all_chips = true;
            break;
        }
        uint32_t chip;
        if (sscanf(env_var_str, "%d", &chip) != 1) {
            TT_THROW("Invalid {}", env_var_str);
        }
        chips.push_back(chip);
        env_var_str = strchr(env_var_str, ',');
        if (env_var_str != nullptr) {
            env_var_str++;
        }
    }

    // Default is no chips are specified is all
    if (chips.size() == 0) {
        feature_targets[feature].all_chips = true;
    }
    feature_targets[feature].chip_ids = chips;
}

void RunTimeOptions::ParseFeatureRiscvMask(
    RunTimeDebugFeatures feature, const std::string& env_var, const tt_metal::Hal& hal) {
    const char* env_var_str = std::getenv(env_var.c_str());

    if (env_var_str != nullptr) {
        feature_targets[feature].processors = hal.parse_processor_set_spec(env_var_str);
    } else {
        auto& processors = feature_targets[feature].processors;
        uint32_t num_core_types = hal.get_programmable_core_type_count();
        for (uint32_t core_type_index = 0; core_type_index < num_core_types; ++core_type_index) {
            auto core_type = hal.get_programmable_core_type(core_type_index);
            uint32_t num_processors = hal.get_num_risc_processors(core_type);
            for (uint32_t processor_index = 0; processor_index < num_processors; ++processor_index) {
                processors.add(core_type, processor_index);
            }
        }
    }
}

void RunTimeOptions::ParseFeatureFileName(RunTimeDebugFeatures feature, const std::string& env_var) {
    char* env_var_str = std::getenv(env_var.c_str());
    feature_targets[feature].file_name = (env_var_str != nullptr) ? std::string(env_var_str) : "";
}

void RunTimeOptions::ParseFeatureOneFilePerRisc(RunTimeDebugFeatures feature, const std::string& env_var) {
    char* env_var_str = std::getenv(env_var.c_str());
    feature_targets[feature].one_file_per_risc = (env_var_str != nullptr);
}

void RunTimeOptions::ParseFeaturePrependDeviceCoreRisc(RunTimeDebugFeatures feature, const std::string &env_var) {
    char *env_var_str = std::getenv(env_var.c_str());
    feature_targets[feature].prepend_device_core_risc =
        (env_var_str != nullptr) ? (strcmp(env_var_str, "1") == 0) : true;
}

uint32_t RunTimeOptions::get_watcher_hash() const {
    // These values will cause kernels / firmware to be recompiled if they change
    // Only the ones which have #define on the device side need to be listed here
    std::string hash_str = "";
    hash_str += std::to_string(watcher_feature_disabled(watcher_waypoint_str));
    hash_str += std::to_string(watcher_feature_disabled(watcher_noc_sanitize_str));
    hash_str += std::to_string(watcher_feature_disabled(watcher_assert_str));
    hash_str += std::to_string(watcher_feature_disabled(watcher_pause_str));
    hash_str += std::to_string(watcher_feature_disabled(watcher_ring_buffer_str));
    hash_str += std::to_string(watcher_feature_disabled(watcher_stack_usage_str));
    hash_str += std::to_string(watcher_feature_disabled(watcher_dispatch_str));
    hash_str += std::to_string(get_watcher_noc_sanitize_linked_transaction());
    hash_str += std::to_string(get_watcher_enabled());
    std::hash<std::string> hash_fn;
    return hash_fn(hash_str);
}

// Can't create a DispatchCoreConfig as part of the RTOptions constructor because the DispatchCoreConfig constructor
// depends on RTOptions settings.
tt_metal::DispatchCoreConfig RunTimeOptions::get_dispatch_core_config() const {
    tt_metal::DispatchCoreConfig dispatch_core_config = tt_metal::DispatchCoreConfig{};
    dispatch_core_config.set_dispatch_core_type(this->dispatch_core_type);
    return dispatch_core_config;
}

}  // namespace llrt

}  // namespace tt

 