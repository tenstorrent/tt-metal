// SPDX-FileCopyrightText: © 2023 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "rtoptions.hpp"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <enchantum/enchantum.hpp>
#include <tt_stl/assert.hpp>
#include <tt-logger/tt-logger.hpp>
#include <umd/device/types/core_coordinates.hpp>

// NOLINTBEGIN(bugprone-branch-clone)

using std::vector;

namespace tt::llrt {

const char* RunTimeDebugFeatureNames[RunTimeDebugFeatureCount] = {
    "DPRINT",
    "READ_DEBUG_DELAY",
    "WRITE_DEBUG_DELAY",
    "ATOMIC_DEBUG_DELAY",
    "ENABLE_L1_DATA_CACHE",
};

const char* RunTimeDebugClassNames[RunTimeDebugClassCount] = {"N/A", "worker", "dispatch", "all"};

// ============================================================================
// ENVIRONMENT VARIABLE IDs
// ============================================================================
// Full definition of EnvVarID enum (forward-declared in rtoptions.hpp)
enum class EnvVarID {
    // ========================================
    // PATH CONFIGURATION
    // ========================================

    TT_METAL_CACHE,                           // Cache directory for compiled kernels
    TT_METAL_KERNEL_PATH,                     // Path to kernel source files
    TT_METAL_LOGS_PATH,                       // Path for generated logs and debug output
    TT_METAL_SIMULATOR,                       // Path to simulator executable
    TT_METAL_MOCK_CLUSTER_DESC_PATH,          // Mock cluster descriptor path
    TT_METAL_VISIBLE_DEVICES,                 // Comma-separated list of visible device IDs
    ARCH_NAME,                                // Architecture name (simulation mode)
    TT_MESH_GRAPH_DESC_PATH,                  // Custom fabric mesh graph descriptor
    TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE,  // Core grid override

    // ========================================
    // KERNEL EXECUTION CONTROL
    // ========================================
    TT_METAL_NULL_KERNELS,          // Skip kernel execution (testing)
    TT_METAL_KERNELS_EARLY_RETURN,  // Kernels return early

    // ========================================
    // MEMORY INITIALIZATION
    // ========================================
    TT_METAL_CLEAR_L1,    // Clear L1 memory on device init
    TT_METAL_CLEAR_DRAM,  // Clear DRAM on device init

    // ========================================
    // DEBUG & TESTING
    // ========================================
    TT_METAL_WATCHER_TEST_MODE,          // Enable watcher test mode
    TT_METAL_KERNEL_MAP,                 // Enable kernel build mapping
    TT_METAL_DISPATCH_DATA_COLLECTION,   // Enable dispatch debug data collection
    TT_METAL_GTEST_ETH_DISPATCH,         // Use Ethernet cores for dispatch in tests
    TT_METAL_SKIP_LOADING_FW,            // Skip firmware loading
    TT_METAL_SKIP_DELETING_BUILT_CACHE,  // Skip cache deletion on cleanup
    TT_METAL_DISABLE_XIP_DUMP,           // Disable XIP dump

    // ========================================
    // HARDWARE CONFIGURATION
    // ========================================
    TT_METAL_ENABLE_HW_CACHE_INVALIDATION,  // Enable HW cache invalidation
    TT_METAL_DISABLE_RELAXED_MEM_ORDERING,  // Disable relaxed memory ordering
    TT_METAL_ENABLE_GATHERING,              // Enable instruction gathering
    TT_METAL_FABRIC_BW_TELEMETRY,           // Enable fabric bandwidth telemetry
    TT_METAL_FABRIC_TELEMETRY,              // Enable fabric telemetry
    TT_FABRIC_PROFILE_RX_CH_FWD,            // Enable fabric RX channel forwarding profiling
    TT_METAL_FORCE_REINIT,                  // Force context reinitialization
    TT_METAL_DISABLE_FABRIC_TWO_ERISC,      // Disable fabric 2-ERISC mode
    TT_METAL_LOG_KERNELS_COMPILE_COMMANDS,  // Log kernel compilation commands
    TT_METAL_SLOW_DISPATCH_MODE,            // Use slow dispatch mode
    TT_METAL_SKIP_ETH_CORES_WITH_RETRAIN,   // Skip Ethernet cores during retrain
    TT_METAL_VALIDATE_PROGRAM_BINARIES,     // Validate kernel binary integrity
    TT_METAL_DISABLE_DMA_OPS,               // Disable DMA operations
    TT_METAL_ENABLE_ERISC_IRAM,             // Enable ERISC IRAM (inverted logic)
    RELIABILITY_MODE,                       // Fabric reliability mode (strict/relaxed)
    TT_METAL_DISABLE_MULTI_AERISC,          // Disable multi-erisc mode (inverted logic, enabled by default)
    TT_METAL_USE_MGD_2_0,                   // Use mesh graph descriptor 2.0
    TT_METAL_FORCE_JIT_COMPILE,             // Force JIT compilation

    // ========================================
    // PROFILING & PERFORMANCE
    // ========================================
    TT_METAL_DEVICE_PROFILER,                      // Enable device profiling
    TT_METAL_DEVICE_PROFILER_DISPATCH,             // Enable dispatch core profiling
    TT_METAL_PROFILER_SYNC,                        // Enable synchronous profiling
    TT_METAL_DEVICE_PROFILER_NOC_EVENTS,           // Enable NoC events profiling
    TT_METAL_DEVICE_PROFILER_NOC_EVENTS_RPT_PATH,  // NoC events report path
    TT_METAL_PROFILE_PERF_COUNTERS,                // Enable Performance Counter profiling
    TT_METAL_MEM_PROFILER,                         // Enable memory/buffer profiling
    TT_METAL_TRACE_PROFILER,                       // Enable trace profiling
    TT_METAL_PROFILER_TRACE_TRACKING,              // Enable trace tracking
    TT_METAL_PROFILER_MID_RUN_DUMP,                // Force mid-run profiler dumps
    TT_METAL_PROFILER_CPP_POST_PROCESS,            // Enable C++ post-processing for profiler
    TT_METAL_PROFILER_SUM,                         // Enable sum profiling
    TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT,       // Maximum number of programs supported by the profiler
    TT_METAL_TRACY_MID_RUN_PUSH,                   // Force Tracy mid-run pushes
    TT_METAL_PROFILER_DISABLE_DUMP_TO_FILES,       // Disable dumping collected device data to files
    TT_METAL_PROFILER_DISABLE_PUSH_TO_TRACY,       // Disable pushing collected device data to Tracy GUI
    TT_METAL_GTEST_NUM_HW_CQS,                     // Number of HW command queues in tests
    TT_METAL_ARC_DEBUG_BUFFER_SIZE,                // ARC processor debug buffer size
    TT_METAL_OPERATION_TIMEOUT_SECONDS,            // Operation timeout duration
    TT_METAL_DISPATCH_TIMEOUT_COMMAND_TO_EXECUTE,  // Terminal command to execute on dispatch timeout.
    TT_METAL_DEVICE_DEBUG_DUMP_ENABLED,            // Enable experimental debug dump mode for profiler

    // ========================================
    // WATCHER SYSTEM
    // ========================================
    TT_METAL_WATCHER,                                         // Enable watcher system
    TT_METAL_WATCHER_DUMP_ALL,                                // Dump all watcher data
    TT_METAL_WATCHER_APPEND,                                  // Append mode for watcher output
    TT_METAL_WATCHER_NOINLINE,                                // Disable watcher function inlining
    TT_METAL_WATCHER_PHYS_COORDS,                             // Use physical coordinates in watcher
    TT_METAL_WATCHER_TEXT_START,                              // Include text start info in watcher
    TT_METAL_WATCHER_SKIP_LOGGING,                            // Disable watcher logging
    TT_METAL_WATCHER_DISABLE_ASSERT,                          // Disable watcher assert feature
    TT_METAL_WATCHER_DISABLE_PAUSE,                           // Disable watcher pause feature
    TT_METAL_WATCHER_DISABLE_RING_BUFFER,                     // Disable watcher ring buffer
    TT_METAL_WATCHER_DISABLE_STACK_USAGE,                     // Disable watcher stack usage tracking
    TT_METAL_WATCHER_DISABLE_SANITIZE_NOC,                    // Disable watcher NoC sanitization
    TT_METAL_WATCHER_DISABLE_SANITIZE_READ_ONLY_L1,           // Disable read-only L1 sanitization
    TT_METAL_WATCHER_DISABLE_SANITIZE_WRITE_ONLY_L1,          // Disable write-only L1 sanitization
    TT_METAL_WATCHER_DISABLE_WAYPOINT,                        // Disable watcher waypoint feature
    TT_METAL_WATCHER_DISABLE_DISPATCH,                        // Disable watcher dispatch feature
    TT_METAL_WATCHER_ENABLE_NOC_SANITIZE_LINKED_TRANSACTION,  // Enable NoC linked transaction sanitization

    // ========================================
    // INSPECTOR
    // ========================================
    TT_METAL_INSPECTOR,                                // Enable/disable inspector
    TT_METAL_INSPECTOR_INITIALIZATION_IS_IMPORTANT,    // Track initialization closely
    TT_METAL_INSPECTOR_WARN_ON_WRITE_EXCEPTIONS,       // Warn on write exceptions
    TT_METAL_RISCV_DEBUG_INFO,                         // Enable RISC-V debug info
    TT_METAL_INSPECTOR_RPC_SERVER_ADDRESS,             // Inspector RPC server address (host:port)
    TT_METAL_INSPECTOR_RPC,                            // Enable/disable inspector RPC server
    TT_METAL_INSPECTOR_SERIALIZE_ON_DISPATCH_TIMEOUT,  // Serialize inspector data on dispatch timeout

    // ========================================
    // DEBUG PRINTING (DPRINT)
    // ========================================
    TT_METAL_DPRINT_CORES,                     // Worker cores for debug printing
    TT_METAL_DPRINT_ETH_CORES,                 // Ethernet cores for debug printing
    TT_METAL_DPRINT_CHIPS,                     // Chip IDs for debug printing
    TT_METAL_DPRINT_RISCVS,                    // RISC-V processors for debug printing
    TT_METAL_DPRINT_FILE,                      // Debug print output file
    TT_METAL_DPRINT_ONE_FILE_PER_RISC,         // Separate file per RISC-V processor
    TT_METAL_DPRINT_PREPEND_DEVICE_CORE_RISC,  // Prepend device/core/RISC info

    // ========================================
    // LIGHTWEIGHT KERNEL DEBUGGING
    // ========================================
    TT_METAL_LIGHTWEIGHT_KERNEL_ASSERTS,  // Enable lightweight kernel asserts

    // ========================================
    // LLK ASSERTIONS
    // ========================================
    TT_METAL_LLK_ASSERTS,  // Enable LLK assertions

    // ========================================
    // DEVICE MANAGER
    // ========================================
    TT_METAL_NUMA_BASED_AFFINITY,

    // ========================================
    // FABRIC CONFIGURATION
    // ========================================
    TT_METAL_FABRIC_ROUTER_SYNC_TIMEOUT_MS,  // Timeout for fabric router sync in milliseconds

    // ========================================
    // JIT BUILD CONFIGURATION
    // ========================================
    TT_METAL_BACKEND_DUMP_RUN_CMD,  // Dump JIT build commands to stdout
};

// Environment variable name for TT-Metal root directory
constexpr auto TT_METAL_RUNTIME_ROOT_ENV_VAR = "TT_METAL_RUNTIME_ROOT";

namespace {
// Helper function to normalize directory paths using std::filesystem
std::string normalize_path(const char* path, const std::string& subdir = "") {
    std::filesystem::path p(path);
    if (!subdir.empty()) {
        p /= subdir;
    }
    return p.lexically_normal().string();
}

// Helper function to check if environment variable value is "1" (enabled)
bool is_env_enabled(const char* value) { return value && value[0] == '1'; }

std::string trim_copy(const std::string& input) {
    auto first = std::find_if_not(input.begin(), input.end(), [](unsigned char ch) { return std::isspace(ch); });
    if (first == input.end()) {
        return "";
    }
    auto last =
        std::find_if_not(input.rbegin(), input.rend(), [](unsigned char ch) { return std::isspace(ch); }).base();
    return std::string(first, last);
}

std::string to_lower_copy(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) { return std::tolower(ch); });
    return value;
}

std::string to_upper_copy(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) { return std::toupper(ch); });
    return value;
}

template <typename IntType>
IntType parse_int_token(const std::string& token, const std::string& context) {
    try {
        long long parsed = std::stoll(token, nullptr, 0);
        return static_cast<IntType>(parsed);
    } catch (const std::invalid_argument&) {
        TT_THROW("Invalid token '{}' while parsing {}", token, context);
    } catch (const std::out_of_range&) {
        TT_THROW("Out-of-range token '{}' while parsing {}", token, context);
    }
}

bool equals_all(const std::string& token) { return to_lower_copy(trim_copy(token)) == "all"; }

}  // namespace

RunTimeOptions::RunTimeOptions() : system_kernel_dir("/usr/share/tenstorrent/kernels/") {
// Default assume package install path
#ifdef TT_METAL_INSTALL_ROOT
    if (std::filesystem::is_directory(std::filesystem::path(TT_METAL_INSTALL_ROOT))) {
        this->root_dir = std::filesystem::path(TT_METAL_INSTALL_ROOT).string();
    }
    log_debug(tt::LogMetal, "initial root_dir: {}", this->root_dir);
#endif

    // ENV Can Override
    const char* root_dir_str = std::getenv(TT_METAL_RUNTIME_ROOT_ENV_VAR);
    if (root_dir_str != nullptr) {
        this->root_dir = std::string(root_dir_str);
        log_debug(tt::LogMetal, "ENV override root_dir: {}", this->root_dir);
    } else if (!g_root_dir.empty()) {
        this->root_dir = g_root_dir;
        log_debug(tt::LogMetal, "API override root_dir: {}", this->root_dir);
    } else if (this->root_dir.empty()) {
        // If the current working directory contains a "tt_metal/" directory,
        // treat the current working directory as the repository root.
        std::filesystem::path current_working_directory = std::filesystem::current_path();
        std::filesystem::path tt_metal_subdirectory = current_working_directory / "tt_metal";
        if (std::filesystem::is_directory(tt_metal_subdirectory)) {
            this->root_dir = current_working_directory.string();
            log_debug(tt::LogMetal, "current working directory fallback root_dir: {}", this->root_dir);
        }
    }

    if (this->root_dir.empty()) {
        log_critical(
            tt::LogMetal,
            "Failed to determine TT-Metal root directory. "
            "Root directory must be set via one of the following methods:\n"
            "1. Automatically determined when using a package install\n"
            "2. Set TT_METAL_RUNTIME_ROOT environment variable to the path containing tt_metal/\n"
            "3. Call RunTimeOptions::set_root_dir() API before creating RunTimeOptions\n"
            "4. Run from the root of the repository\n"
            "Current working directory: {}",
            std::filesystem::current_path().string());
    }

    TT_FATAL(!this->root_dir.empty(), "Root Directory is not set.");

    {
        std::filesystem::path p(root_dir);
        p /= "";  // ensures trailing slash, never duplicates
        this->root_dir = p.string();
    }

    InitializeFromEnvVars();

    if (this->runtime_target_device_ != tt::TargetDevice::Silicon) {
        log_info(tt::LogMetal, "Disabling multi-erisc mode with simulator/mock target device");
        this->enable_2_erisc_mode = false;
    }

    TT_FATAL(
        !(get_feature_enabled(RunTimeDebugFeatureDprint) && get_profiler_enabled()),
        "Cannot enable both debug printing and profiling");
}

void RunTimeOptions::set_root_dir(const std::string& root_dir) {
    std::call_once(g_root_once, [&] { g_root_dir = root_dir; });
}

const std::string& RunTimeOptions::get_root_dir() const { return root_dir; }

const std::string& RunTimeOptions::get_cache_dir() const {
    if (!this->is_cache_dir_specified()) {
        TT_THROW("Env var {} is not set.", "TT_METAL_CACHE");
    }
    return this->cache_dir_;
}

const std::string& RunTimeOptions::get_logs_dir() const { return logs_dir_; }

const std::string& RunTimeOptions::get_kernel_dir() const {
    if (!this->is_kernel_dir_specified()) {
        TT_THROW("Env var {} is not set.", "TT_METAL_KERNEL_PATH");
    }

    return this->kernel_dir;
}

const std::string& RunTimeOptions::get_core_grid_override_todeprecate() const {
    if (!this->is_core_grid_override_todeprecate()) {
        TT_THROW("Env var {} is not set.", enchantum::to_string(EnvVarID::TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE));
    }

    return this->core_grid_override_todeprecate;
}

const std::string& RunTimeOptions::get_system_kernel_dir() const { return this->system_kernel_dir; }

// ============================================================================
// ENVIRONMENT VARIABLE HANDLER
// ============================================================================
// Central handler that processes environment variables based on their ID
// Uses switch statement for efficient dispatch
//
// IMPORTANT: Most cases assume 'value' is non-null (enforced by InitializeFromEnvVars loop guard).
// Only TT_METAL_RISCV_DEBUG_INFO explicitly handle nullptr
// for default value initialization.

void RunTimeOptions::HandleEnvVar(EnvVarID id, const char* value) {
    switch (id) {
        // ========================================
        // PATH CONFIGURATION
        // ========================================

        // TT_METAL_CACHE
        // Directory for caching compiled kernels and other build artifacts.
        // Default: Defaults to system temp directory if not set
        // Usage: export TT_METAL_CACHE=/path/to/cache
        case EnvVarID::TT_METAL_CACHE:
            this->is_cache_dir_env_var_set = true;
            this->cache_dir_ = normalize_path(value, "tt-metal-cache");
            break;

        // TT_METAL_KERNEL_PATH
        // Path to kernel source files.
        // Default: Uses TT_METAL_RUNTIME_ROOT/tt_metal/kernels if not set
        // Usage: export TT_METAL_KERNEL_PATH=/path/to/kernels
        case EnvVarID::TT_METAL_KERNEL_PATH:
            this->is_kernel_dir_env_var_set = true;
            this->kernel_dir = normalize_path(value) + "/";
            break;

        // TT_METAL_LOGS_PATH
        // Directory for generated logs and debug output (dprint, watcher, profiler, etc.)
        // Default: Current working directory if not set
        // Usage: export TT_METAL_LOGS_PATH=/path/to/logs
        case EnvVarID::TT_METAL_LOGS_PATH: this->logs_dir_ = normalize_path(value) + "/"; break;

        // TT_METAL_SIMULATOR
        // Path to simulator executable. When set, overrides mock cluster mode if both are set.
        // Default: Hardware mode (no simulator)
        // Usage: export TT_METAL_SIMULATOR=/path/to/simulator
        case EnvVarID::TT_METAL_SIMULATOR:
            this->simulator_path = std::string(value);
            // Simulator takes precedence over Mock (will be set even if Mock was set first)
            this->runtime_target_device_ = tt::TargetDevice::Simulator;
            break;

        // TT_METAL_MOCK_CLUSTER_DESC_PATH
        // Path to mock cluster descriptor for testing without hardware.
        // Note: If both MOCK and SIMULATOR are set, SIMULATOR takes precedence
        // Default: Hardware mode (no mock cluster)
        // Usage: export TT_METAL_MOCK_CLUSTER_DESC_PATH=/path/to/mock_cluster_desc.yaml
        case EnvVarID::TT_METAL_MOCK_CLUSTER_DESC_PATH:
            this->mock_cluster_desc_path = std::string(value);
            // Only set Mock target if Simulator hasn't been set already
            if (this->simulator_path.empty()) {
                this->runtime_target_device_ = tt::TargetDevice::Mock;
            }
            break;

        // TT_METAL_VISIBLE_DEVICES
        // Comma-separated list of device IDs to make visible to the runtime.
        // Default: All devices visible
        // Usage: export TT_METAL_VISIBLE_DEVICES=0,1,2
        case EnvVarID::TT_METAL_VISIBLE_DEVICES: this->visible_devices = std::string(value); break;

        // ARCH_NAME
        // Sets the architecture name (only necessary during simulation).
        // Default: Hardware-detected architecture
        // Usage: export ARCH_NAME=wormhole_b0
        case EnvVarID::ARCH_NAME: this->arch_name = std::string(value); break;

        // TT_MESH_GRAPH_DESC_PATH
        // Custom fabric mesh graph descriptor path.
        // Default: Default fabric mesh configuration
        // Usage: export TT_MESH_GRAPH_DESC_PATH=/path/to/mesh_desc.yaml
        case EnvVarID::TT_MESH_GRAPH_DESC_PATH:
            this->is_custom_fabric_mesh_graph_desc_path_set = true;
            this->custom_fabric_mesh_graph_desc_path = std::string(value);
            break;

        // TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE
        // Override core grid configuration (this is meant to be deprecated).
        // Default: Hardware-detected core grid
        // Usage: export TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE=custom_grid
        case EnvVarID::TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE:
            this->is_core_grid_override_todeprecate_env_var_set = true;
            this->core_grid_override_todeprecate = std::string(value);
            break;

        // ========================================
        // KERNEL EXECUTION CONTROL
        // ========================================

        // TT_METAL_NULL_KERNELS
        // Skip actual kernel execution, useful for testing dispatch logic without running kernels.
        // Default: false (kernels execute normally)
        // Usage: export TT_METAL_NULL_KERNELS=1
        case EnvVarID::TT_METAL_NULL_KERNELS: this->null_kernels = true; break;

        // TT_METAL_KERNELS_EARLY_RETURN
        // Kernels return early, skipping execution but maintaining same size as normal.
        // Default: false (kernels execute fully)
        // Usage: export TT_METAL_KERNELS_EARLY_RETURN=1
        case EnvVarID::TT_METAL_KERNELS_EARLY_RETURN: this->kernels_early_return = true; break;

        // ========================================
        // MEMORY INITIALIZATION
        // ========================================

        // TT_METAL_CLEAR_L1
        // Clear L1 memory on device initialization.
        // Default: 0 (don't clear)
        // Usage: export TT_METAL_CLEAR_L1=1
        case EnvVarID::TT_METAL_CLEAR_L1: this->clear_l1 = is_env_enabled(value); break;

        // TT_METAL_CLEAR_DRAM
        // Clear DRAM memory on device initialization.
        // Default: 0 (don't clear)
        // Usage: export TT_METAL_CLEAR_DRAM=1
        case EnvVarID::TT_METAL_CLEAR_DRAM: this->clear_dram = is_env_enabled(value); break;
        // ========================================
        // DEBUG & TESTING
        // ========================================

        // TT_METAL_WATCHER_TEST_MODE
        // Enable test mode for watcher functionality.
        // Default: false (normal mode)
        // Usage: export TT_METAL_WATCHER_TEST_MODE=1
        case EnvVarID::TT_METAL_WATCHER_TEST_MODE: this->test_mode_enabled = true; break;

        // TT_METAL_KERNEL_MAP
        // Enable kernel build mapping for debugging.
        // Default: false (mapping disabled)
        // Usage: export TT_METAL_KERNEL_MAP=1
        case EnvVarID::TT_METAL_KERNEL_MAP: this->build_map_enabled = true; break;

        // TT_METAL_DISPATCH_DATA_COLLECTION
        // Enable collection of dispatch debugging data.
        // Default: false (collection disabled)
        // Usage: export TT_METAL_DISPATCH_DATA_COLLECTION=1
        case EnvVarID::TT_METAL_DISPATCH_DATA_COLLECTION: this->enable_dispatch_data_collection = true; break;

        // TT_METAL_GTEST_ETH_DISPATCH
        // Use Ethernet cores for dispatch in tests.
        // Default: Worker cores (default dispatch type)
        // Usage: export TT_METAL_GTEST_ETH_DISPATCH=1
        case EnvVarID::TT_METAL_GTEST_ETH_DISPATCH: this->dispatch_core_type = tt_metal::DispatchCoreType::ETH; break;

        // TT_METAL_SKIP_LOADING_FW
        // Skip loading firmware during device initialization.
        // Default: false (load firmware)
        // Usage: export TT_METAL_SKIP_LOADING_FW=1
        case EnvVarID::TT_METAL_SKIP_LOADING_FW: this->skip_loading_fw = true; break;

        // TT_METAL_SKIP_DELETING_BUILT_CACHE
        // Skip deleting built cache files on cleanup.
        // Default: false (delete cache)
        // Usage: export TT_METAL_SKIP_DELETING_BUILT_CACHE=1
        case EnvVarID::TT_METAL_SKIP_DELETING_BUILT_CACHE: this->skip_deleting_built_cache = true; break;

        // ========================================
        // HARDWARE CONFIGURATION
        // ========================================

        // TT_METAL_ENABLE_HW_CACHE_INVALIDATION
        // Enable hardware to automatically invalidate Blackhole's data cache.
        // Default: false (cache invalidation disabled)
        // Usage: export TT_METAL_ENABLE_HW_CACHE_INVALIDATION=1
        case EnvVarID::TT_METAL_ENABLE_HW_CACHE_INVALIDATION: this->enable_hw_cache_invalidation = true; break;

        // TT_METAL_DISABLE_RELAXED_MEM_ORDERING
        // Disable relaxed memory ordering optimizations on Blackhole.
        // Default: false (relaxed ordering enabled)
        // Usage: export TT_METAL_DISABLE_RELAXED_MEM_ORDERING=1
        case EnvVarID::TT_METAL_DISABLE_RELAXED_MEM_ORDERING: this->disable_relaxed_memory_ordering = true; break;

        // TT_METAL_ENABLE_GATHERING
        // Enable data gathering on Blackhole.
        // Default: false (gathering disabled)
        // Usage: export TT_METAL_ENABLE_GATHERING=1
        case EnvVarID::TT_METAL_ENABLE_GATHERING: this->enable_gathering = true; break;

        // TT_METAL_FABRIC_BW_TELEMETRY
        // Enable fabric bandwidth telemetry data collection.
        // Default: false (telemetry disabled)
        // Usage: export TT_METAL_FABRIC_BW_TELEMETRY=1
        case EnvVarID::TT_METAL_FABRIC_BW_TELEMETRY: this->enable_fabric_bw_telemetry = true; break;

        // TT_METAL_FABRIC_TELEMETRY
        // Enable fabric telemetry data collection (supports structured spec).
        // Default: false (telemetry disabled)
        // Usage: export TT_METAL_FABRIC_TELEMETRY=1 (enable all stats on all chip/eth/erisc) or
        //        export TT_METAL_FABRIC_TELEMETRY="chips=all;eth=0,2,7;erisc=all;stats=ROUTER_STATE|BANDWIDTH"
        case EnvVarID::TT_METAL_FABRIC_TELEMETRY: this->ParseFabricTelemetryEnv(value); break;

        // TT_FABRIC_PROFILE_RX_CH_FWD
        // Enables fabric RX channel forwarding profiling.
        // Default: false
        // Usage: export TT_FABRIC_PROFILE_RX_CH_FWD=1
        case EnvVarID::TT_FABRIC_PROFILE_RX_CH_FWD: this->fabric_profiling_settings.enable_rx_ch_fwd = true; break;

        // RELIABILITY_MODE
        // Sets the fabric reliability mode (STRICT, RELAXED, or DYNAMIC).
        // Default: nullopt (uses system default)
        // Usage: export RELIABILITY_MODE=STRICT (or strict/0), RELAXED (or relaxed/1), DYNAMIC (or dynamic/2)
        case EnvVarID::RELIABILITY_MODE: {
            std::string mode_str(value);
            // Convert to lowercase for case-insensitive comparison
            std::transform(
                mode_str.begin(), mode_str.end(), mode_str.begin(), [](unsigned char c) { return std::tolower(c); });
            if (mode_str == "strict" || mode_str == "0") {
                this->reliability_mode = tt::tt_fabric::FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE;
            } else if (mode_str == "relaxed" || mode_str == "1") {
                this->reliability_mode = tt::tt_fabric::FabricReliabilityMode::RELAXED_SYSTEM_HEALTH_SETUP_MODE;
            } else if (mode_str == "dynamic" || mode_str == "2") {
                this->reliability_mode = tt::tt_fabric::FabricReliabilityMode::DYNAMIC_RECONFIGURATION_SETUP_MODE;
            }
            break;
        }

        // TT_METAL_DISABLE_FABRIC_TWO_ERISC
        // Presence-based override to force-disable fabric 2-ERISC mode, even if defaults would enable
        // Default: false (2-ERISC mode enabled)
        // Usage: export TT_METAL_DISABLE_FABRIC_TWO_ERISC=1
        case EnvVarID::TT_METAL_DISABLE_FABRIC_TWO_ERISC:
            log_info(tt::LogMetal, "Disabling two-erisc fabric mode with TT_METAL_DISABLE_FABRIC_TWO_ERISC");
            this->disable_fabric_2_erisc_mode = true;
            break;

        // TT_METAL_DISABLE_MULTI_AERISC
        // Disable multi-ERISC mode (2-ERISC mode is enabled by default on Blackhole).
        // Use this to fallback to single ERISC mode.
        // Default: enabled (2-ERISC mode)
        // Usage: export TT_METAL_DISABLE_MULTI_AERISC=1
        case EnvVarID::TT_METAL_DISABLE_MULTI_AERISC:
            log_info(tt::LogMetal, "Disabling multi-erisc mode with TT_METAL_DISABLE_MULTI_AERISC");
            this->enable_2_erisc_mode = false;
            break;

        // TT_METAL_USE_MGD_2_0
        // Enables use of Mesh Graph Descriptor 2.0 format for fabric configuration.
        // Default: false (uses MGD 1.0)
        // Usage: export TT_METAL_USE_MGD_2_0=1
        case EnvVarID::TT_METAL_USE_MGD_2_0:
            this->use_mesh_graph_descriptor_2_0 = (std::strncmp(value, "0", 1) != 0);
            break;

        // TT_METAL_FORCE_JIT_COMPILE
        // Force JIT compilation even if dependencies are up-to-date.
        // This overrides the dependency tracking optimization.
        // Default: false (uses dependency tracking)
        // Usage: export TT_METAL_FORCE_JIT_COMPILE=1
        case EnvVarID::TT_METAL_FORCE_JIT_COMPILE: this->force_jit_compile = true; break;

        // TT_METAL_FORCE_REINIT
        // Force context reinitialization on each run.
        // Default: false (normal initialization)
        // Usage: export TT_METAL_FORCE_REINIT=1
        case EnvVarID::TT_METAL_FORCE_REINIT: this->force_context_reinit = true; break;

        // TT_METAL_LOG_KERNELS_COMPILE_COMMANDS
        // Log kernel compilation commands for debugging.
        // Default: false (no logging)
        // Usage: export TT_METAL_LOG_KERNELS_COMPILE_COMMANDS=1
        case EnvVarID::TT_METAL_LOG_KERNELS_COMPILE_COMMANDS: this->log_kernels_compilation_commands = true; break;

        // TT_METAL_SLOW_DISPATCH_MODE
        // Use slow dispatch mode for debugging.
        // Default: false (fast dispatch mode)
        // Usage: export TT_METAL_SLOW_DISPATCH_MODE=1
        case EnvVarID::TT_METAL_SLOW_DISPATCH_MODE:
            this->using_slow_dispatch = true;
            this->fast_dispatch = false;
            break;

        // TT_METAL_SKIP_ETH_CORES_WITH_RETRAIN
        // Skip Ethernet cores during retraining process.
        // Default: true (skip retraining)
        // Usage: export TT_METAL_SKIP_ETH_CORES_WITH_RETRAIN=1
        case EnvVarID::TT_METAL_SKIP_ETH_CORES_WITH_RETRAIN:
            this->skip_eth_cores_with_retrain = is_env_enabled(value);
            break;

        // TT_METAL_VALIDATE_PROGRAM_BINARIES
        // Validate kernel binary integrity before execution.
        // Default: 0 (no validation)
        // Usage: export TT_METAL_VALIDATE_PROGRAM_BINARIES=1
        case EnvVarID::TT_METAL_VALIDATE_PROGRAM_BINARIES:
            this->set_validate_kernel_binaries(is_env_enabled(value));
            break;

        // TT_METAL_DISABLE_DMA_OPS
        // Disable DMA operations for debugging.
        // Default: 0 (DMA enabled)
        // Usage: export TT_METAL_DISABLE_DMA_OPS=1
        case EnvVarID::TT_METAL_DISABLE_DMA_OPS: this->disable_dma_ops = is_env_enabled(value); break;

        // TT_METAL_ENABLE_ERISC_IRAM
        // Enable ERISC IRAM functionality (inverted: 0=disabled, 1=enabled).
        // Default: 1 (enabled)
        // Usage: export TT_METAL_ENABLE_ERISC_IRAM=0  # to disable
        case EnvVarID::TT_METAL_ENABLE_ERISC_IRAM: {
            bool disabled = (value[0] == '0');
            this->erisc_iram_enabled = !disabled;
            this->erisc_iram_enabled_env_var = !disabled;
            break;
        }

        // ========================================
        // PROFILING & PERFORMANCE
        // ========================================

        // TT_METAL_DEVICE_PROFILER
        // Enables device profiling (requires TRACY_ENABLE compilation flag).
        // Default: false (profiling disabled)
        // Usage: export TT_METAL_DEVICE_PROFILER=1
        case EnvVarID::TT_METAL_DEVICE_PROFILER:
#if !defined(TRACY_ENABLE)
            TT_FATAL(false, "TT_METAL_DEVICE_PROFILER requires a Tracy-enabled build of tt-metal.");
#else
            if (is_env_enabled(value)) {
                this->profiler_enabled = true;
            }
#endif
            break;

        // TT_METAL_DEVICE_PROFILER_DISPATCH
        // Enables profiling of dispatch cores. Requires TT_METAL_DEVICE_PROFILER=1 to be effective.
        // Default: 0 (dispatch profiling disabled)
        // Usage: export TT_METAL_DEVICE_PROFILER_DISPATCH=1
        case EnvVarID::TT_METAL_DEVICE_PROFILER_DISPATCH: {
            // Only enable dispatch profiling if device profiler is also enabled
            if (this->profiler_enabled && is_env_enabled(value)) {
                this->profile_dispatch_cores = true;
            }
            break;
        }

        // TT_METAL_PROFILER_SYNC
        // Enables synchronous profiling mode for more accurate timing.
        // Default: false (asynchronous profiling)
        // Usage: export TT_METAL_PROFILER_SYNC=1
        case EnvVarID::TT_METAL_PROFILER_SYNC: {
            // Only enable sync profiling if device profiler is also enabled
            if (this->profiler_enabled && is_env_enabled(value)) {
                this->profiler_sync_enabled = true;
            }
            break;
        }

        // TT_METAL_DEVICE_PROFILER_NOC_EVENTS
        // Enables NoC (Network-on-Chip) events profiling.
        // Default: false (NoC events not profiled)
        // Usage: export TT_METAL_DEVICE_PROFILER_NOC_EVENTS=1
        case EnvVarID::TT_METAL_DEVICE_PROFILER_NOC_EVENTS:
            if (is_env_enabled(value)) {
                this->profiler_enabled = true;
                this->profiler_noc_events_enabled = true;
            }
            break;

        // TT_METAL_DEVICE_PROFILER_NOC_EVENTS_RPT_PATH
        // Sets the report path for NoC events profiling output files.
        // Default: Default report location
        // Usage: export TT_METAL_DEVICE_PROFILER_NOC_EVENTS_RPT_PATH=/path/to/reports
        case EnvVarID::TT_METAL_DEVICE_PROFILER_NOC_EVENTS_RPT_PATH:
            this->profiler_noc_events_report_path = std::string(value);
            break;

        // TT_METAL_MEM_PROFILER
        // Enables memory/buffer usage profiling for tracking memory allocation patterns.
        // Default: false (memory profiling disabled)
        // Usage: export TT_METAL_MEM_PROFILER=1
        case EnvVarID::TT_METAL_MEM_PROFILER:
            if (is_env_enabled(value)) {
                this->profiler_buffer_usage_enabled = true;
            }
            break;

        // TT_METAL_PROFILE_PERF_COUNTERS
        // Enables Performance Counter profiling using a bitfield to select counter groups.
        // Default: 0 (disabled)
        // Usage: export TT_METAL_PROFILE_PERF_COUNTERS=value
        //
        // Valid values (bitfield):
        //   1  (1 << 0) - FPU counters
        //   2  (1 << 1) - PACK counters
        //   4  (1 << 2) - UNPACK counters
        //   8  (1 << 3) - L1 counters
        //   16 (1 << 4) - INSTRN (instruction) counters
        //   31 (0x1F)   - All counter groups (fpu|pack|unpack|l1|instrn)
        //
        // Multiple groups can be combined by OR-ing the values (e.g., 3 = FPU + PACK)
        // Note: Currently, only FPU counters are supported
        case EnvVarID::TT_METAL_PROFILE_PERF_COUNTERS:
            sscanf(value, "%u", &this->profiler_perf_counter_mode);
            if (this->profiler_perf_counter_mode != 0) {
                this->profiler_enabled = true;
            }
            break;

        // TT_METAL_TRACE_PROFILER
        // Enables trace profiler for detailed execution tracing.
        // Default: false (trace profiling disabled)
        // Usage: export TT_METAL_TRACE_PROFILER=1
        case EnvVarID::TT_METAL_TRACE_PROFILER: {
            // Only enable trace profiling if device profiler is also enabled
            if (this->profiler_enabled && is_env_enabled(value)) {
                this->profiler_trace_profiler = true;
            }
            break;
        }

        // TT_METAL_PROFILER_TRACE_TRACKING
        // Enables trace tracking for detailed execution tracing.
        // Default: false (trace tracking disabled)
        // Usage: export TT_METAL_PROFILER_TRACE_TRACKING=1
        case EnvVarID::TT_METAL_PROFILER_TRACE_TRACKING: {
            // Only enable trace tracking if device profiler is also enabled
            if (this->profiler_enabled && is_env_enabled(value)) {
                this->profiler_trace_tracking = true;
            }
            break;
        }

        // TT_METAL_PROFILER_MID_RUN_DUMP
        // Forces Tracy profiler dumps during execution for real-time profiling.
        // Default: false (no mid-run dumps)
        // Usage: export TT_METAL_PROFILER_MID_RUN_DUMP=1
        case EnvVarID::TT_METAL_PROFILER_MID_RUN_DUMP: {
            // Only enable mid-run dumps if device profiler is also enabled
            if (this->profiler_enabled && is_env_enabled(value)) {
                this->profiler_mid_run_dump = true;
            }
            break;
        }

        // TT_METAL_PROFILER_CPP_POST_PROCESS
        // Enable C++ post-processing for profiler output. Requires TT_METAL_DEVICE_PROFILER=1 to be effective.
        // Default: false
        // Usage: export TT_METAL_PROFILER_CPP_POST_PROCESS=1
        case EnvVarID::TT_METAL_PROFILER_CPP_POST_PROCESS: {
            // Only enable C++ post-processing if device profiler is also enabled
            if (this->profiler_enabled && is_env_enabled(value)) {
                this->profiler_cpp_post_process = true;
            }
            break;
        }

        // TT_METAL_PROFILER_SUM
        // Enables sum profiling.
        // Default: false (sum profiling disabled)
        // Usage: export TT_METAL_PROFILER_SUM=1
        case EnvVarID::TT_METAL_PROFILER_SUM: {
            if (this->profiler_enabled && is_env_enabled(value)) {
                this->profiler_sum = true;
            }
            break;
        }

        // TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT
        // Specifies the maximum number of programs supported by the profiler.
        // Default: nullopt (uses profiler default)
        // Usage: export TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=500
        case EnvVarID::TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT: {
            // Only set the program support count if device profiler is also enabled
            if (this->profiler_enabled && value) {
                this->profiler_program_support_count = std::stoi(value);
            }
            break;
        }

        // TT_METAL_TRACY_MID_RUN_PUSH
        // Forces Tracy profiler pushes during execution for real-time profiling.
        // Default: false (no mid-run pushes)
        // Usage: export TT_METAL_TRACY_MID_RUN_PUSH=1
        case EnvVarID::TT_METAL_TRACY_MID_RUN_PUSH: this->tracy_mid_run_push = true; break;

        // TT_METAL_PROFILER_DISABLE_DUMP_TO_FILES
        // Disables dumping collected device data to files.
        // Default: false (dump to files)
        // Usage: export TT_METAL_PROFILER_DISABLE_DUMP_TO_FILES=1
        case EnvVarID::TT_METAL_PROFILER_DISABLE_DUMP_TO_FILES: {
            // Only disable dumping to files if device profiler is also enabled
            if (this->profiler_enabled && is_env_enabled(value)) {
                this->profiler_disable_dump_to_files = true;
            }
            break;
        }

        // TT_METAL_PROFILER_DISABLE_PUSH_TO_TRACY
        // Disables pushing collected device data to Tracy GUI.
        // Default: false (push to Tracy GUI)
        // Usage: export TT_METAL_PROFILER_DISABLE_PUSH_TO_TRACY=1
        case EnvVarID::TT_METAL_PROFILER_DISABLE_PUSH_TO_TRACY: {
            // Only disable pushing to Tracy GUI if device profiler is also enabled
            if (this->profiler_enabled && is_env_enabled(value)) {
                this->profiler_disable_push_to_tracy = true;
            }
            break;
        }

        // TT_METAL_DEVICE_DEBUG_DUMP_ENABLED
        // Enable and sets the polling interval in seconds for experimental debug dump mode for profiler. In this mode,
        // the profiler infrastructure will be used to continuously dump debug packets to a file. Default: false (debug
        // dump mode disabled) Usage: export TT_METAL_DEVICE_DEBUG_DUMP_ENABLED=1
        case EnvVarID::TT_METAL_DEVICE_DEBUG_DUMP_ENABLED: {
            if (is_env_enabled(value)) {
                this->profiler_enabled = true;
                this->profiler_noc_events_enabled = true;
                this->experimental_device_debug_dump_interval_seconds = std::stoi(value);
            }
            break;
        }

        // TT_METAL_GTEST_NUM_HW_CQS
        // Number of hardware command queues to use in tests.
        // Default: 1
        // Usage: export TT_METAL_GTEST_NUM_HW_CQS=4
        case EnvVarID::TT_METAL_GTEST_NUM_HW_CQS: try { this->set_num_hw_cqs(std::stoi(value));
            } catch (const std::invalid_argument& ia) {
                TT_THROW("Invalid TT_METAL_GTEST_NUM_HW_CQS: {}", value);
            } catch (const std::out_of_range&) {
                TT_THROW("TT_METAL_GTEST_NUM_HW_CQS value out of range: {}", value);
            }
            break;

        // TT_METAL_ARC_DEBUG_BUFFER_SIZE
        // Buffer size in DRAM for storing ARC processor debug samples.
        // Default: 0 (disabled)
        // Usage: export TT_METAL_ARC_DEBUG_BUFFER_SIZE=1024
        case EnvVarID::TT_METAL_ARC_DEBUG_BUFFER_SIZE: sscanf(value, "%u", &this->arc_debug_buffer_size); break;

        // TT_METAL_OPERATION_TIMEOUT_SECONDS
        // Timeout duration for device operations in seconds.
        // Default: 0.0 (no timeout)
        // Usage: export TT_METAL_OPERATION_TIMEOUT_SECONDS=30.0
        case EnvVarID::TT_METAL_OPERATION_TIMEOUT_SECONDS: {
            float timeout_duration = std::stof(value);
            this->timeout_duration_for_operations = std::chrono::duration<float>(timeout_duration);
            break;
        }

        // TT_METAL_DISPATCH_TIMEOUT_COMMAND_TO_EXECUTE
        // Terminal command to execute on dispatch timeout.
        // Default: "" (no command)
        // Usage: export TT_METAL_DISPATCH_TIMEOUT_COMMAND_TO_EXECUTE=./tools/tt-triage.py
        case EnvVarID::TT_METAL_DISPATCH_TIMEOUT_COMMAND_TO_EXECUTE:
            this->dispatch_timeout_command_to_execute = std::string(value);
            break;

        // ========================================
        // WATCHER SYSTEM
        // ========================================

        // TT_METAL_WATCHER
        // Enables the watcher system for debugging. When set, enables watcher on all features.
        // Default: disabled
        // Usage: export TT_METAL_WATCHER=1
        case EnvVarID::TT_METAL_WATCHER: {
            int sleep_val = 0;
            sscanf(value, "%d", &sleep_val);
            if (strstr(value, "ms") == nullptr) {
                sleep_val *= 1000;
            }
            this->watcher_settings.enabled = true;
            this->watcher_settings.interval_ms = sleep_val;
            break;
        }

        // TT_METAL_WATCHER_DUMP_ALL
        // Enables dumping all watcher data, including potentially unsafe state information.
        // Default: false (safe data only)
        // Usage: export TT_METAL_WATCHER_DUMP_ALL=1
        case EnvVarID::TT_METAL_WATCHER_DUMP_ALL: this->watcher_settings.dump_all = true; break;

        // TT_METAL_WATCHER_APPEND
        // Enables append mode for watcher output files instead of overwriting.
        // Default: false (overwrite mode)
        // Usage: export TT_METAL_WATCHER_APPEND=1
        case EnvVarID::TT_METAL_WATCHER_APPEND: this->watcher_settings.append = true; break;

        // TT_METAL_WATCHER_NOINLINE
        // Disables inlining for watcher functions to reduce binary size.
        // Default: false (inlining enabled)
        // Usage: export TT_METAL_WATCHER_NOINLINE=1
        case EnvVarID::TT_METAL_WATCHER_NOINLINE: this->watcher_settings.noinline = true; break;

        // TT_METAL_WATCHER_PHYS_COORDS
        // Uses physical coordinates in watcher output instead of logical coordinates.
        // Default: false (logical coordinates)
        // Usage: export TT_METAL_WATCHER_PHYS_COORDS=1
        case EnvVarID::TT_METAL_WATCHER_PHYS_COORDS: this->watcher_settings.phys_coords = true; break;

        // TT_METAL_WATCHER_TEXT_START
        // Includes text start information in watcher output for debugging.
        // Default: false (no text start info)
        // Usage: export TT_METAL_WATCHER_TEXT_START=1
        case EnvVarID::TT_METAL_WATCHER_TEXT_START: this->watcher_settings.text_start = true; break;

        // TT_METAL_WATCHER_SKIP_LOGGING
        // Disables watcher logging to reduce overhead.
        // Default: false (logging enabled)
        // Usage: export TT_METAL_WATCHER_SKIP_LOGGING=1
        case EnvVarID::TT_METAL_WATCHER_SKIP_LOGGING: this->watcher_settings.skip_logging = true; break;

        // TT_METAL_WATCHER_DISABLE_ASSERT
        // Disables watcher assert feature when set to any value.
        // Default: enabled
        // Usage: export TT_METAL_WATCHER_DISABLE_ASSERT=1
        case EnvVarID::TT_METAL_WATCHER_DISABLE_ASSERT:
            this->watcher_disabled_features.insert(this->watcher_assert_str);
            break;

        // TT_METAL_WATCHER_DISABLE_PAUSE
        // Disables watcher pause feature when set to any value.
        // Default: enabled
        // Usage: export TT_METAL_WATCHER_DISABLE_PAUSE=1
        case EnvVarID::TT_METAL_WATCHER_DISABLE_PAUSE:
            this->watcher_disabled_features.insert(this->watcher_pause_str);
            break;

        // TT_METAL_WATCHER_DISABLE_RING_BUFFER
        // Disables watcher ring buffer feature when set to any value.
        // Default: enabled
        // Usage: export TT_METAL_WATCHER_DISABLE_RING_BUFFER=1
        case EnvVarID::TT_METAL_WATCHER_DISABLE_RING_BUFFER:
            this->watcher_disabled_features.insert(this->watcher_ring_buffer_str);
            break;

        // TT_METAL_WATCHER_DISABLE_STACK_USAGE
        // Disables watcher stack usage tracking when set to any value.
        // Default: enabled
        // Usage: export TT_METAL_WATCHER_DISABLE_STACK_USAGE=1
        case EnvVarID::TT_METAL_WATCHER_DISABLE_STACK_USAGE:
            this->watcher_disabled_features.insert(this->watcher_stack_usage_str);
            break;

        // TT_METAL_WATCHER_DISABLE_SANITIZE_NOC
        // Disables watcher NoC sanitization when set to any value.
        // Default: enabled
        // Usage: export TT_METAL_WATCHER_DISABLE_SANITIZE_NOC=1
        case EnvVarID::TT_METAL_WATCHER_DISABLE_SANITIZE_NOC:
            this->watcher_disabled_features.insert(this->watcher_noc_sanitize_str);
            break;

        // TT_METAL_WATCHER_DISABLE_SANITIZE_READ_ONLY_L1
        // Disables watcher read-only L1 sanitization when set to any value.
        // Default: enabled
        // Usage: export TT_METAL_WATCHER_DISABLE_SANITIZE_READ_ONLY_L1=1
        case EnvVarID::TT_METAL_WATCHER_DISABLE_SANITIZE_READ_ONLY_L1:
            this->watcher_disabled_features.insert(this->watcher_sanitize_read_only_l1_str);
            break;

        // TT_METAL_WATCHER_DISABLE_SANITIZE_WRITE_ONLY_L1
        // Disables watcher write-only L1 sanitization when set to any value.
        // Default: enabled
        // Usage: export TT_METAL_WATCHER_DISABLE_SANITIZE_WRITE_ONLY_L1=1
        case EnvVarID::TT_METAL_WATCHER_DISABLE_SANITIZE_WRITE_ONLY_L1:
            this->watcher_disabled_features.insert(this->watcher_sanitize_write_only_l1_str);
            break;

        // TT_METAL_WATCHER_DISABLE_WAYPOINT
        // Disables watcher waypoint feature when set to any value.
        // Default: enabled
        // Usage: export TT_METAL_WATCHER_DISABLE_WAYPOINT=1
        case EnvVarID::TT_METAL_WATCHER_DISABLE_WAYPOINT:
            this->watcher_disabled_features.insert(this->watcher_waypoint_str);
            break;

        // TT_METAL_WATCHER_DISABLE_DISPATCH
        // Disables watcher dispatch feature when set to any value.
        // Default: enabled
        // Usage: export TT_METAL_WATCHER_DISABLE_DISPATCH=1
        case EnvVarID::TT_METAL_WATCHER_DISABLE_DISPATCH:
            this->watcher_disabled_features.insert(this->watcher_dispatch_str);
            break;

        // TT_METAL_WATCHER_ENABLE_NOC_SANITIZE_LINKED_TRANSACTION
        // Enables NoC sanitization for linked transactions to catch more subtle errors.
        // Default: false (linked transaction sanitization disabled)
        // Usage: export TT_METAL_WATCHER_ENABLE_NOC_SANITIZE_LINKED_TRANSACTION=1
        case EnvVarID::TT_METAL_WATCHER_ENABLE_NOC_SANITIZE_LINKED_TRANSACTION:
            this->watcher_settings.noc_sanitize_linked_transaction = true;
            break;

        // ========================================
        // INSPECTOR
        // ========================================

        // TT_METAL_INSPECTOR
        // Enables or disables the inspector system. Set to '0' to disable, any other value enables it.
        // Default: true (enabled)
        // Usage: export TT_METAL_INSPECTOR=1
        case EnvVarID::TT_METAL_INSPECTOR:
            this->inspector_settings.enabled = true;
            if (strcmp(value, "0") == 0) {
                this->inspector_settings.enabled = false;
            }
            break;

        // TT_METAL_INSPECTOR_INITIALIZATION_IS_IMPORTANT
        // Controls whether initialization is considered important for inspector. Set to '0' to disable.
        // Default: false (not important)
        // Usage: export TT_METAL_INSPECTOR_INITIALIZATION_IS_IMPORTANT=1
        case EnvVarID::TT_METAL_INSPECTOR_INITIALIZATION_IS_IMPORTANT:
            this->inspector_settings.initialization_is_important = true;
            if (strcmp(value, "0") == 0) {
                this->inspector_settings.initialization_is_important = false;
            }
            break;

        // TT_METAL_INSPECTOR_WARN_ON_WRITE_EXCEPTIONS
        // Controls whether to warn on write exceptions in inspector. Set to '0' to disable warnings.
        // Default: true (warnings enabled)
        // Usage: export TT_METAL_INSPECTOR_WARN_ON_WRITE_EXCEPTIONS=0
        case EnvVarID::TT_METAL_INSPECTOR_WARN_ON_WRITE_EXCEPTIONS:
            this->inspector_settings.warn_on_write_exceptions = true;
            if (strcmp(value, "0") == 0) {
                this->inspector_settings.warn_on_write_exceptions = false;
            }
            break;

        // TT_METAL_RISCV_DEBUG_INFO
        // Enable RISC-V debug info. Defaults to inspector setting, override with 0/1.
        // Default: Inherits from inspector setting
        // Usage: export TT_METAL_RISCV_DEBUG_INFO=1  # or =0 to disable
        case EnvVarID::TT_METAL_RISCV_DEBUG_INFO: {
            bool enable_riscv_debug_info = this->get_inspector_enabled();  // Default from inspector
            if (value) {
                enable_riscv_debug_info = true;  // Default to true if set
                if (strcmp(value, "0") == 0) {
                    enable_riscv_debug_info = false;  // Only "0" = false
                }
            }
            this->set_riscv_debug_info_enabled(enable_riscv_debug_info);
            break;
        }

        // TT_METAL_INSPECTOR_RPC_SERVER_ADDRESS
        // Sets the RPC server address for the inspector. Format: "host:port" or just "host" (uses default port).
        // Default: localhost:50051
        // Usage: export TT_METAL_INSPECTOR_RPC_SERVER_ADDRESS=127.0.0.1:8080
        case EnvVarID::TT_METAL_INSPECTOR_RPC_SERVER_ADDRESS: {
            // Parse address into host and port
            std::string addr(value);
            size_t colon_pos = addr.find(':');
            if (colon_pos != std::string::npos) {
                this->inspector_settings.rpc_server_host = addr.substr(0, colon_pos);
                try {
                    this->inspector_settings.rpc_server_port = std::stoi(addr.substr(colon_pos + 1));
                } catch (const std::invalid_argument&) {
                    TT_THROW("Invalid port in TT_METAL_INSPECTOR_RPC_SERVER_ADDRESS: {}", value);
                } catch (const std::out_of_range&) {
                    TT_THROW("Port out of range in TT_METAL_INSPECTOR_RPC_SERVER_ADDRESS: {}", value);
                }
            } else {
                this->inspector_settings.rpc_server_host = addr;
            }
            break;
        }

        // TT_METAL_INSPECTOR_RPC
        // Enables or disables the inspector RPC server. Set to '0' to disable, any other value enables it.
        // Default: true (enabled)
        // Usage: export TT_METAL_INSPECTOR_RPC=1
        case EnvVarID::TT_METAL_INSPECTOR_RPC:
            this->inspector_settings.rpc_server_enabled = true;
            if (std::strncmp(value, "0", 1) == 0) {
                this->inspector_settings.rpc_server_enabled = false;
            }
            break;

        // TT_METAL_INSPECTOR_SERIALIZE_ON_DISPATCH_TIMEOUT
        // Enables serialization of inspector state on dispatch timeout. Set to '0' to disable.
        // Default: true (enabled)
        // Usage: export TT_METAL_INSPECTOR_SERIALIZE_ON_DISPATCH_TIMEOUT=1
        case EnvVarID::TT_METAL_INSPECTOR_SERIALIZE_ON_DISPATCH_TIMEOUT:
            this->inspector_settings.serialize_on_dispatch_timeout = true;
            if (std::strncmp(value, "0", 1) == 0) {
                this->inspector_settings.serialize_on_dispatch_timeout = false;
            }
            break;

        // ========================================
        // DEBUG PRINTING (DPRINT)
        // ========================================
        // Note: Most DPRINT variables are handled by ParseFeatureEnv() in constructor
        // These cases exist for documentation and potential future direct handling

        // TT_METAL_DPRINT_CORES
        // Specifies worker cores for debug printing. Supports 'all', ranges '(1,1)-(2,2)', or lists '(1,1),(2,2)'.
        // Default: disabled (no debug printing)
        // Usage: export TT_METAL_DPRINT_CORES=all
        case EnvVarID::TT_METAL_DPRINT_CORES:
            // Handled by ParseFeatureEnv() - this is for documentation
            break;

        // TT_METAL_DPRINT_ETH_CORES
        // Specifies Ethernet cores for debug printing. Same syntax as DPRINT_CORES.
        // Default: disabled (no debug printing on ETH cores)
        // Usage: export TT_METAL_DPRINT_ETH_CORES=(0,0),(1,0)
        case EnvVarID::TT_METAL_DPRINT_ETH_CORES:
            // Handled by ParseFeatureEnv() - this is for documentation
            break;

        // TT_METAL_DPRINT_CHIPS
        // Specifies chip IDs for debug printing. Supports 'all' or comma-separated list of chip IDs.
        // Default: all chips
        // Usage: export TT_METAL_DPRINT_CHIPS=0,1,2
        case EnvVarID::TT_METAL_DPRINT_CHIPS:
            // Handled by ParseFeatureEnv() - this is for documentation
            break;

        // TT_METAL_DPRINT_RISCVS
        // Specifies RISC-V processors for debug printing. Complex processor selection syntax.
        // Default: all RISC-V processors
        // Usage: export TT_METAL_DPRINT_RISCVS=BR+NCRISC+TRISC0
        case EnvVarID::TT_METAL_DPRINT_RISCVS:
            // Handled by ParseFeatureEnv() - this is for documentation
            break;

        // TT_METAL_DPRINT_FILE
        // Output file path for debug printing. If not specified, prints to stdout.
        // Default: stdout
        // Usage: export TT_METAL_DPRINT_FILE=/tmp/debug_output.log
        case EnvVarID::TT_METAL_DPRINT_FILE:
            // Handled by ParseFeatureEnv() - this is for documentation
            break;

        // TT_METAL_DPRINT_ONE_FILE_PER_RISC
        // Creates separate output files for each RISC-V processor when set.
        // Default: false (single output file)
        // Usage: export TT_METAL_DPRINT_ONE_FILE_PER_RISC=1
        case EnvVarID::TT_METAL_DPRINT_ONE_FILE_PER_RISC:
            // Handled by ParseFeatureEnv() - this is for documentation
            break;

        // TT_METAL_DPRINT_PREPEND_DEVICE_CORE_RISC
        // Prepends device/core/RISC information to each debug print line. Set to '0' to disable.
        // Default: true (prepend enabled)
        // Usage: export TT_METAL_DPRINT_PREPEND_DEVICE_CORE_RISC=0
        case EnvVarID::TT_METAL_DPRINT_PREPEND_DEVICE_CORE_RISC:
            // Handled by ParseFeatureEnv() - this is for documentation
            break;

        // TT_METAL_LIGHTWEIGHT_KERNEL_ASSERTS
        // Enables lightweight kernel assertions. If watcher asserts are enabled, they take precedence.
        // Default: false (disabled)
        // Usage: export TT_METAL_LIGHTWEIGHT_KERNEL_ASSERTS=1
        case EnvVarID::TT_METAL_LIGHTWEIGHT_KERNEL_ASSERTS: this->lightweight_kernel_asserts = true; break;

        // TT_METAL_LLK_ASSERTS
        // Enables LLK assertions. If watcher asserts are enabled, they take precedence.
        // Default: false (disabled)
        // Usage: export TT_METAL_LLK_ASSERTS=1
        case EnvVarID::TT_METAL_LLK_ASSERTS: this->enable_llk_asserts = true; break;

        // ========================================
        // DEVICE MANAGER
        // ========================================
        // TT_METAL_NUMA_BASED_AFFINITY
        // Specifies thread binding in DeviceManager
        // Default: disabled
        // Usage: export TT_METAL_NUMA_BASED_AFFINITY=1
        case EnvVarID::TT_METAL_NUMA_BASED_AFFINITY: {
            this->numa_based_affinity = is_env_enabled(value);
            break;
        }

        // ========================================
        // FABRIC CONFIGURATION
        // ========================================
        // TT_METAL_FABRIC_ROUTER_SYNC_TIMEOUT_MS
        // Timeout in milliseconds for fabric router sync
        // Default: 5000ms
        // Usage: export TT_METAL_FABRIC_ROUTER_SYNC_TIMEOUT_MS=8000
        case EnvVarID::TT_METAL_FABRIC_ROUTER_SYNC_TIMEOUT_MS:
            try {
                int parsed_value = std::stoi(value);
                if (parsed_value < 0) {
                    TT_THROW("TT_METAL_FABRIC_ROUTER_SYNC_TIMEOUT_MS must be non-negative: {}", value);
                }
                this->fabric_router_sync_timeout_ms = static_cast<uint32_t>(parsed_value);
            } catch (const std::invalid_argument& ia) {
                TT_THROW("Invalid TT_METAL_FABRIC_ROUTER_SYNC_TIMEOUT_MS: {}", value);
            } catch (const std::out_of_range&) {
                TT_THROW("TT_METAL_FABRIC_ROUTER_SYNC_TIMEOUT_MS value out of range: {}", value);
            }
            break;
        // TT_METAL_DISABLE_XIP_DUMP
        // Disable XIP dump
        // Default: false
        // Usage: export TT_METAL_DISABLE_XIP_DUMP=1
        case EnvVarID::TT_METAL_DISABLE_XIP_DUMP: {
            this->disable_xip_dump = is_env_enabled(value);
            break;
        }

        // TT_METAL_BACKEND_DUMP_RUN_CMD
        // Dump JIT build commands to stdout for debugging kernel compilation.
        // Default: false
        // Usage: export TT_METAL_BACKEND_DUMP_RUN_CMD=1
        case EnvVarID::TT_METAL_BACKEND_DUMP_RUN_CMD: this->dump_build_commands = is_env_enabled(value); break;
    }
}

void RunTimeOptions::InitializeFromEnvVars() {
    // Use enchantum to automatically iterate over all EnvVarID enum values
    for (const auto [id, name] : enchantum::entries_generator<EnvVarID>) {
        const char* value = std::getenv(std::string(name).c_str());

        // Only process if the environment variable is set
        if (value) {
            HandleEnvVar(id, value);
        }
    }

    // Set inspector log path
    this->inspector_settings.log_path = std::filesystem::path(this->get_logs_dir()) / "generated/inspector";

    // TT_METAL_RISCV_DEBUG_INFO: Inherit from inspector if not explicitly set
    if (std::getenv("TT_METAL_RISCV_DEBUG_INFO") == nullptr) {
        HandleEnvVar(EnvVarID::TT_METAL_RISCV_DEBUG_INFO, nullptr);
    }
    ParseWatcherEnv();
}

void RunTimeOptions::ParseWatcherEnv() {
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
            !watcher_disabled_features.contains(watcher_noc_sanitize_str),
            "TT_METAL_WATCHER_DEBUG_DELAY requires TT_METAL_WATCHER_DISABLE_NOC_SANITIZE=0");
    }
    if (watcher_settings.noc_sanitize_linked_transaction) {
        TT_ASSERT(
            !watcher_disabled_features.contains(watcher_noc_sanitize_str),
            "TT_METAL_WATCHER_ENABLE_NOC_SANITIZE_LINKED_TRANSACTION requires TT_METAL_WATCHER_DISABLE_NOC_SANITIZE=0");
    }
}

void RunTimeOptions::ParseFabricTelemetryEnv(const char* value) {
    auto disable_telemetry = [&]() {
        set_enable_fabric_telemetry(false);
        fabric_telemetry_settings = FabricTelemetrySettings{};
    };

    if (value == nullptr) {
        disable_telemetry();
        return;
    }

    std::string spec = trim_copy(value);
    if (spec.empty() || spec == "0") {
        disable_telemetry();
        return;
    }

    FabricTelemetrySettings parsed_settings{};
    parsed_settings.enabled = true;
    set_enable_fabric_telemetry(true);

    if (spec == "1") {
        fabric_telemetry_settings = parsed_settings;
        return;
    }

    auto handle_required_entries = [&](bool condition, const char* key) {
        if (!condition) {
            TT_THROW("TT_METAL_FABRIC_TELEMETRY {}= requires at least one entry", key);
        }
    };

    auto parse_uint32_selection =
        [&](const std::string& raw_value, FabricTelemetrySelection<uint32_t>& selection, const char* key) {
            if (raw_value.empty() || equals_all(raw_value)) {
                selection.set_monitor_all(true);
                return;
            }
            selection.set_monitor_all(false);
            selection.ids.clear();
            std::stringstream value_stream(raw_value);
            std::string token;
            bool parsed_any = false;
            while (std::getline(value_stream, token, ',')) {
                token = trim_copy(token);
                if (token.empty()) {
                    continue;
                }
                int parsed_value = parse_int_token<int>(token, key);
                if (parsed_value < 0) {
                    TT_THROW("TT_METAL_FABRIC_TELEMETRY {}= requires non-negative IDs", key);
                }
                parsed_any = true;
                selection.ids.insert(static_cast<uint32_t>(parsed_value));
            }
            handle_required_entries(parsed_any, key);
        };

    std::stringstream section_stream(spec);
    std::string section;
    while (std::getline(section_stream, section, ';')) {
        section = trim_copy(section);
        if (section.empty()) {
            continue;
        }

        auto eq_pos = section.find('=');
        if (eq_pos == std::string::npos) {
            TT_THROW("Invalid TT_METAL_FABRIC_TELEMETRY segment '{}'. Expected key=value.", section);
        }

        std::string key = trim_copy(section.substr(0, eq_pos));
        std::string raw_value = trim_copy(section.substr(eq_pos + 1));
        std::string key_lower = to_lower_copy(key);

        if (key_lower == "chips") {
            parse_uint32_selection(raw_value, parsed_settings.chips, key_lower.c_str());
        } else if (key_lower == "eth") {
            parse_uint32_selection(raw_value, parsed_settings.channels, key_lower.c_str());
        } else if (key_lower == "erisc") {
            parse_uint32_selection(raw_value, parsed_settings.eriscs, key_lower.c_str());
        } else if (key_lower == "stats") {
            if (raw_value.empty() || equals_all(raw_value)) {
                parsed_settings.stats_mask = FabricTelemetrySettings::kAllStatsMask;
                continue;
            }
            parsed_settings.stats_mask = 0;
            std::replace(raw_value.begin(), raw_value.end(), '|', ',');
            std::stringstream stats_stream(raw_value);
            std::string stats_token;
            while (std::getline(stats_stream, stats_token, ',')) {
                stats_token = trim_copy(stats_token);
                if (stats_token.empty()) {
                    continue;
                }
                std::string upper = to_upper_copy(stats_token);
                if (upper == "ROUTER_STATE") {
                    parsed_settings.stats_mask |= static_cast<uint8_t>(DynamicStatistics::ROUTER_STATE);
                } else if (upper == "BANDWIDTH") {
                    parsed_settings.stats_mask |= static_cast<uint8_t>(DynamicStatistics::BANDWIDTH);
                } else if (upper == "HEARTBEAT_TX") {
                    parsed_settings.stats_mask |= static_cast<uint8_t>(DynamicStatistics::HEARTBEAT_TX);
                } else if (upper == "HEARTBEAT_RX") {
                    parsed_settings.stats_mask |= static_cast<uint8_t>(DynamicStatistics::HEARTBEAT_RX);
                } else {
                    parsed_settings.stats_mask |= static_cast<uint8_t>(parse_int_token<int>(stats_token, "stats"));
                }
            }
        } else {
            TT_THROW("Unknown TT_METAL_FABRIC_TELEMETRY key '{}'", key);
        }
    }

    fabric_telemetry_settings = parsed_settings;
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
        if (!core_type_and_cores.second.empty()) {
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
    if (chips.empty()) {
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

void RunTimeOptions::ParseFeaturePrependDeviceCoreRisc(RunTimeDebugFeatures feature, const std::string& env_var) {
    char* env_var_str = std::getenv(env_var.c_str());
    feature_targets[feature].prepend_device_core_risc =
        (env_var_str != nullptr) ? (strcmp(env_var_str, "1") == 0) : true;
}

uint32_t RunTimeOptions::get_watcher_hash() const {
    // These values will cause kernels / firmware to be recompiled if they change
    // Only the ones which have #define on the device side need to be listed here
    std::string hash_str;
    hash_str += std::to_string(watcher_feature_disabled(watcher_waypoint_str));
    hash_str += std::to_string(watcher_feature_disabled(watcher_noc_sanitize_str));
    hash_str += std::to_string(watcher_feature_disabled(watcher_assert_str));
    hash_str += std::to_string(watcher_feature_disabled(watcher_pause_str));
    hash_str += std::to_string(watcher_feature_disabled(watcher_ring_buffer_str));
    hash_str += std::to_string(watcher_feature_disabled(watcher_stack_usage_str));
    hash_str += std::to_string(watcher_feature_disabled(watcher_dispatch_str));
    hash_str += std::to_string(get_watcher_noc_sanitize_linked_transaction());
    hash_str += std::to_string(get_watcher_enabled());
    hash_str += std::to_string(get_lightweight_kernel_asserts());
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

}  // namespace tt::llrt

// NOLINTEND(bugprone-branch-clone)
