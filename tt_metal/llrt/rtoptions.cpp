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
#include <enchantum/enchantum.hpp>
#include "tt_stl/assert.hpp"
#include <umd/device/tt_core_coordinates.h>

#include <tt-logger/tt-logger.hpp>
#include <tt_stl/assert.hpp>
#include <umd/device/types/core_coordinates.hpp>

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

constexpr auto TT_METAL_RUNTIME_ROOT_ENV_VAR = "TT_METAL_RUNTIME_ROOT";
constexpr auto TT_METAL_KERNEL_PATH_ENV_VAR = "TT_METAL_KERNEL_PATH";
// Set this var to change the cache dir.
constexpr auto TT_METAL_CACHE_ENV_VAR = "TT_METAL_CACHE";
// Used for demonstration purposes and will be removed in the future.
// Env variable to override the core grid configuration
constexpr auto TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE_ENV_VAR = "TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE";

namespace {
// Helper function to normalize directory paths using std::filesystem
std::string normalize_path(const char* path, const std::string& subdir = "") {
    std::filesystem::path p(path);
    if (!subdir.empty()) {
        p /= subdir;
    }
    return p.lexically_normal().string();
}
RunTimeOptions::RunTimeOptions() {
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

    TT_FATAL(!this->root_dir.empty(), "Root Directory is not set.");

    {
        std::filesystem::path p(root_dir);
        p /= "";  // ensures trailing slash, never duplicates
        this->root_dir = p.string();
    }

    // Check if user has specified a cache path.
    const char* cache_dir_str = std::getenv(TT_METAL_CACHE_ENV_VAR);
    if (cache_dir_str != nullptr) {
        this->is_cache_dir_env_var_set = true;
        this->cache_dir_ = std::string(cache_dir_str) + "/tt-metal-cache/";
    }

    const char* kernel_dir_str = std::getenv(TT_METAL_KERNEL_PATH_ENV_VAR);
    if (kernel_dir_str != nullptr) {
        this->is_kernel_dir_env_var_set = true;
        this->kernel_dir = std::string(kernel_dir_str) + "/";
    }
    this->system_kernel_dir = "/usr/share/tenstorrent/kernels/";

    const char* custom_fabric_mesh_graph_desc_path_str = std::getenv("TT_MESH_GRAPH_DESC_PATH");
    if (custom_fabric_mesh_graph_desc_path_str != nullptr) {
        this->is_custom_fabric_mesh_graph_desc_path_set = true;
        this->custom_fabric_mesh_graph_desc_path = std::string(custom_fabric_mesh_graph_desc_path_str);
    }
    p /= "";  // Ensures trailing slash
    return p.string();
}
}  // namespace

RunTimeOptions::RunTimeOptions() :
    system_kernel_dir("/usr/share/tenstorrent/kernels/"),
    profiler_enabled(false),
    profile_dispatch_cores(false),
    profiler_sync_enabled(false),
    profiler_mid_run_dump(false),
    profiler_buffer_usage_enabled(false),
    profiler_trace_profiler(false) {
#if defined(TRACY_ENABLE)
#endif
    TT_FATAL(
        !(get_feature_enabled(RunTimeDebugFeatureDprint) && get_profiler_enabled()),
        "Cannot enable both debug printing and profiling");

    InitializeFromEnvVars();

    null_kernels = (std::getenv("TT_METAL_NULL_KERNELS") != nullptr);

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
    }

    const char* riscv_debug_info_enabled_str = std::getenv("TT_METAL_RISCV_DEBUG_INFO");
    bool enable_riscv_debug_info = get_inspector_enabled();
    if (riscv_debug_info_enabled_str != nullptr) {
        enable_riscv_debug_info = true;
        if (strcmp(riscv_debug_info_enabled_str, "0") == 0) {
            enable_riscv_debug_info = false;
        }
    }
    set_riscv_debug_info_enabled(enable_riscv_debug_info);

    const char* validate_kernel_binaries = std::getenv("TT_METAL_VALIDATE_PROGRAM_BINARIES");
    set_validate_kernel_binaries(validate_kernel_binaries != nullptr && validate_kernel_binaries[0] == '1');

    const char* num_cqs = getenv("TT_METAL_GTEST_NUM_HW_CQS");
    if (num_cqs != nullptr) {
        try {
            set_num_hw_cqs(std::stoi(num_cqs));
        } catch (const std::invalid_argument& ia) {
            TT_THROW("Invalid TT_METAL_GTEST_NUM_HW_CQS: {}", num_cqs);
        }
    }

    using_slow_dispatch = getenv("TT_METAL_SLOW_DISPATCH_MODE") != nullptr;

    const char* dispatch_data_collection_str = std::getenv("TT_METAL_DISPATCH_DATA_COLLECTION");
    if (dispatch_data_collection_str != nullptr) {
        enable_dispatch_data_collection = true;
    }

    if (getenv("TT_METAL_GTEST_ETH_DISPATCH")) {
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
    }

    // Parse RELIABILITY_MODE: "strict" or "relaxed"
    if (const char* reliability_mode_str = getenv("RELIABILITY_MODE"); reliability_mode_str != nullptr) {
        std::string mode(reliability_mode_str);
        if (mode == "relaxed") {
            reliability_mode = tt::tt_fabric::FabricReliabilityMode::RELAXED_SYSTEM_HEALTH_SETUP_MODE;
        } else if (mode == "strict") {
            reliability_mode = tt::tt_fabric::FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE;
        }
    }

    // Enable mock cluster if TT_METAL_MOCK is set to a descriptor path
    // This is used for initializing UMD without any hardware using a mock cluster descriptor
    if (const char* mock_path = std::getenv("TT_METAL_MOCK_CLUSTER_DESC_PATH")) {
        this->mock_cluster_desc_path = std::string(mock_path);
        this->runtime_target_device_ = tt::TargetDevice::Mock;
    }

    // Enable simulator if TT_METAL_SIMULATOR is set to a simulator path
    // This must be set after the mock cluster path is set to have the correct TargetDevice
    if (std::getenv("TT_METAL_SIMULATOR")) {
        this->simulator_path = std::getenv("TT_METAL_SIMULATOR");
        this->runtime_target_device_ = tt::TargetDevice::Simulator;
    }

    if (auto str = getenv("TT_METAL_ENABLE_ERISC_IRAM")) {
        bool disabled = strcmp(str, "0") == 0;
        this->erisc_iram_enabled = !disabled;
        this->erisc_iram_enabled_env_var = !disabled;
    }
    this->fast_dispatch = (std::getenv("TT_METAL_SLOW_DISPATCH_MODE") == nullptr);

    if (getenv("TT_METAL_DISABLE_RELAXED_MEM_ORDERING")) {
        this->disable_relaxed_memory_ordering = true;
    }

    if (getenv("TT_METAL_ENABLE_GATHERING")) {
        this->enable_gathering = true;
    }

    const char* arc_debug_enabled_str = std::getenv("TT_METAL_ARC_DEBUG_BUFFER_SIZE");
    if (arc_debug_enabled_str != nullptr) {
        sscanf(arc_debug_enabled_str, "%u", &arc_debug_buffer_size);
    }

    const char* disable_dma_ops_str = std::getenv("TT_METAL_DISABLE_DMA_OPS");
    if (disable_dma_ops_str != nullptr) {
        if (disable_dma_ops_str[0] == '1') {
            this->disable_dma_ops = true;
        }
    }

    if (getenv("TT_METAL_FABRIC_TELEMETRY")) {
        enable_fabric_telemetry = true;
    }

    if (getenv("TT_FABRIC_PROFILE_RX_CH_FWD")) {
        fabric_profiling_settings.enable_rx_ch_fwd = true;
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
    }

    if (getenv("TT_METAL_USE_MGD_1_0")) {
        this->use_mesh_graph_descriptor_1_0 = true;
    }

    const char* timeout_duration_for_operations_value = std::getenv("TT_METAL_OPERATION_TIMEOUT_SECONDS");
    float timeout_duration_for_operations =
        timeout_duration_for_operations_value ? std::stof(timeout_duration_for_operations_value) : 0.f;
    this->timeout_duration_for_operations = std::chrono::duration<float>(timeout_duration_for_operations);
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

const std::string& RunTimeOptions::get_kernel_dir() const {
    if (!this->is_kernel_dir_specified()) {
        TT_THROW("Env var {} is not set.", "TT_METAL_KERNEL_PATH");
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

// ============================================================================
// ENVIRONMENT VARIABLE HANDLER
// ============================================================================
// Central handler that processes environment variables based on their ID
// Uses switch statement for efficient dispatch

void RunTimeOptions::HandleEnvVar(EnvVarID id, const char* value) {
    switch (id) {
        // ========================================
        // PATH CONFIGURATION
        // ========================================

        // TT_METAL_RUNTIME_ROOT
        // Sets the root directory of the TT-Metal installation.
        // Default: No default (must be set)
        // Usage: export TT_METAL_RUNTIME_ROOT=/path/to/tt-metal
        case EnvVarID::TT_METAL_RUNTIME_ROOT:
            this->is_root_dir_set = true;
            this->root_dir = normalize_path(value);
            break;

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
            this->kernel_dir = normalize_path(value);
            break;

        // TT_METAL_SIMULATOR
        // Path to simulator executable for testing without hardware.
        // Default: Hardware mode (no simulator)
        // Usage: export TT_METAL_SIMULATOR=/path/to/simulator
        case EnvVarID::TT_METAL_SIMULATOR:
            this->simulator_path = std::string(value);
            this->runtime_target_device_ = tt::TargetDevice::Simulator;
            break;

        // TT_METAL_MOCK_CLUSTER_DESC_PATH
        // Path to mock cluster descriptor for testing without hardware.
        // Default: Hardware mode (no mock cluster)
        // Usage: export TT_METAL_MOCK_CLUSTER_DESC_PATH=/path/to/mock_cluster_desc.yaml
        case EnvVarID::TT_METAL_MOCK_CLUSTER_DESC_PATH:
            this->mock_cluster_desc_path = std::string(value);
            this->runtime_target_device_ = tt::TargetDevice::Mock;
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
        // Override core grid configuration (deprecated).
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
        case EnvVarID::TT_METAL_CLEAR_L1: this->clear_l1 = (value[0] == '1'); break;

        // TT_METAL_CLEAR_DRAM
        // Clear DRAM memory on device initialization.
        // Default: 0 (don't clear)
        // Usage: export TT_METAL_CLEAR_DRAM=1
        case EnvVarID::TT_METAL_CLEAR_DRAM: this->clear_dram = (value[0] == '1'); break;
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
        // Enable hardware cache invalidation.
        // Default: false (cache invalidation disabled)
        // Usage: export TT_METAL_ENABLE_HW_CACHE_INVALIDATION=1
        case EnvVarID::TT_METAL_ENABLE_HW_CACHE_INVALIDATION: this->enable_hw_cache_invalidation = true; break;

        // TT_METAL_DISABLE_RELAXED_MEM_ORDERING
        // Disable relaxed memory ordering optimizations.
        // Default: false (relaxed ordering enabled)
        // Usage: export TT_METAL_DISABLE_RELAXED_MEM_ORDERING=1
        case EnvVarID::TT_METAL_DISABLE_RELAXED_MEM_ORDERING: this->disable_relaxed_memory_ordering = true; break;

        // TT_METAL_ENABLE_GATHERING
        // Enable data gathering functionality.
        // Default: false (gathering disabled)
        // Usage: export TT_METAL_ENABLE_GATHERING=1
        case EnvVarID::TT_METAL_ENABLE_GATHERING: this->enable_gathering = true; break;

        // TT_METAL_FABRIC_TELEMETRY
        // Enable fabric telemetry data collection.
        // Default: false (telemetry disabled)
        // Usage: export TT_METAL_FABRIC_TELEMETRY=1
        case EnvVarID::TT_METAL_FABRIC_TELEMETRY: this->enable_fabric_telemetry = true; break;

        // TT_METAL_FORCE_REINIT
        // Force context reinitialization on each run.
        // Default: false (normal initialization)
        // Usage: export TT_METAL_FORCE_REINIT=1
        case EnvVarID::TT_METAL_FORCE_REINIT: this->force_context_reinit = true; break;

        // TT_METAL_FABRIC_BLACKHOLE_TWO_ERISC
        // Enable two ERISC mode with fabric on Blackhole architecture.
        // Default: false (single ERISC mode)
        // Usage: export TT_METAL_FABRIC_BLACKHOLE_TWO_ERISC=1
        case EnvVarID::TT_METAL_FABRIC_BLACKHOLE_TWO_ERISC: this->enable_2_erisc_mode_with_fabric = true; break;

        // TT_METAL_LOG_KERNELS_COMPILE_COMMANDS
        // Log kernel compilation commands for debugging.
        // Default: false (no logging)
        // Usage: export TT_METAL_LOG_KERNELS_COMPILE_COMMANDS=1
        case EnvVarID::TT_METAL_LOG_KERNELS_COMPILE_COMMANDS: this->log_kernels_compilation_commands = true; break;

        // TT_METAL_SLOW_DISPATCH_MODE
        // Use slow dispatch mode for debugging.
        // Default: false (fast dispatch mode)
        // Usage: export TT_METAL_SLOW_DISPATCH_MODE=1
        case EnvVarID::TT_METAL_SLOW_DISPATCH_MODE: this->using_slow_dispatch = true; break;

        // TT_METAL_SKIP_ETH_CORES_WITH_RETRAIN
        // Skip Ethernet cores during retraining process.
        // Default: true (skip retraining)
        // Usage: export TT_METAL_SKIP_ETH_CORES_WITH_RETRAIN=1
        case EnvVarID::TT_METAL_SKIP_ETH_CORES_WITH_RETRAIN:
            this->skip_eth_cores_with_retrain = (value[0] == '1');
            break;

        // TT_METAL_VALIDATE_PROGRAM_BINARIES
        // Validate kernel binary integrity before execution.
        // Default: 0 (no validation)
        // Usage: export TT_METAL_VALIDATE_PROGRAM_BINARIES=1
        case EnvVarID::TT_METAL_VALIDATE_PROGRAM_BINARIES: this->set_validate_kernel_binaries(value[0] == '1'); break;

        // TT_METAL_DISABLE_DMA_OPS
        // Disable DMA operations for debugging.
        // Default: 0 (DMA enabled)
        // Usage: export TT_METAL_DISABLE_DMA_OPS=1
        case EnvVarID::TT_METAL_DISABLE_DMA_OPS:
            if (value[0] == '1') {
                this->disable_dma_ops = true;
            }
            break;

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
        case EnvVarID::TT_METAL_DEVICE_PROFILER: this->profiler_enabled = true; break;

        // TT_METAL_DEVICE_PROFILER_DISPATCH
        // Enables profiling of dispatch cores. Requires TT_METAL_DEVICE_PROFILER=1 to be effective.
        // Default: 0 (dispatch profiling disabled)
        // Usage: export TT_METAL_DEVICE_PROFILER_DISPATCH=1
        case EnvVarID::TT_METAL_DEVICE_PROFILER_DISPATCH: {
            // Only enable dispatch profiling if device profiler is also enabled
            const char* profiler_enabled_str = std::getenv("TT_METAL_DEVICE_PROFILER");
            if (profiler_enabled_str != nullptr && profiler_enabled_str[0] == '1') {
                if (value && value[0] == '1') {
                    this->profile_dispatch_cores = true;
                }
            }
            break;
        }

        // TT_METAL_PROFILER_SYNC
        // Enables synchronous profiling mode for more accurate timing.
        // Default: false (asynchronous profiling)
        // Usage: export TT_METAL_PROFILER_SYNC=1
        case EnvVarID::TT_METAL_PROFILER_SYNC: this->profiler_sync_enabled = true; break;

        // TT_METAL_DEVICE_PROFILER_NOC_EVENTS
        // Enables NoC (Network-on-Chip) events profiling.
        // Default: false (NoC events not profiled)
        // Usage: export TT_METAL_DEVICE_PROFILER_NOC_EVENTS=1
        case EnvVarID::TT_METAL_DEVICE_PROFILER_NOC_EVENTS:
            this->profiler_enabled = true;
            this->profiler_noc_events_enabled = true;
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
        case EnvVarID::TT_METAL_MEM_PROFILER: this->profiler_buffer_usage_enabled = true; break;

        // TT_METAL_TRACE_PROFILER
        // Enables trace profiler for detailed execution tracing.
        // Default: false (trace profiling disabled)
        // Usage: export TT_METAL_TRACE_PROFILER=1
        case EnvVarID::TT_METAL_TRACE_PROFILER: this->profiler_trace_profiler = true; break;

        // TT_METAL_PROFILER_MID_RUN_DUMP
        // Forces Tracy profiler dumps during execution for real-time profiling.
        // Default: false (no mid-run dumps)
        // Usage: export TT_METAL_PROFILER_MID_RUN_DUMP=1
        case EnvVarID::TT_METAL_PROFILER_MID_RUN_DUMP: this->profiler_mid_run_dump = true; break;

        // TT_METAL_TRACY_MID_RUN_PUSH
        // Forces Tracy profiler pushes during execution for real-time profiling.
        // Default: false (no mid-run pushes)
        // Usage: export TT_METAL_TRACY_MID_RUN_PUSH=1
        case EnvVarID::TT_METAL_TRACY_MID_RUN_PUSH: this->tracy_mid_run_push = true; break;

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
            // ========================================
        // WATCHER SYSTEM
        // ========================================

        // TT_METAL_WATCHER
        // Enables the watcher system for debugging. When set, enables watcher on all features.
        // Default: disabled
        // Usage: export TT_METAL_WATCHER=1
        case EnvVarID::TT_METAL_WATCHER:
            if (value != nullptr) {
                int sleep_val = 0;
                sscanf(value, "%d", &sleep_val);
                if (strstr(value, "ms") == nullptr) {
                    sleep_val *= 1000;
                }
                this->watcher_settings.enabled = true;
                this->watcher_settings.interval_ms = sleep_val;
            }
            break;

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
        // INSPECTOR SYSTEM
        // ========================================

        // TT_METAL_INSPECTOR
        // Enables or disables the inspector system. Set to '0' to disable, any other value enables it.
        // Default: true (enabled)
        // Usage: export TT_METAL_INSPECTOR=1
        case EnvVarID::TT_METAL_INSPECTOR:
            if (value != nullptr) {
                this->inspector_settings.enabled = true;
                if (strcmp(value, "0") == 0) {
                    this->inspector_settings.enabled = false;
                }
            }
            break;

        // TT_METAL_INSPECTOR_LOG_PATH
        // Sets the log path for inspector output.
        // Default: Defaults to {TT_METAL_RUNTIME_ROOT}/generated/inspector
        // Usage: export TT_METAL_INSPECTOR_LOG_PATH=/path/to/inspector/logs
        case EnvVarID::TT_METAL_INSPECTOR_LOG_PATH:
            if (value != nullptr) {
                this->inspector_settings.log_path = std::filesystem::path(value);
            } else {
                this->inspector_settings.log_path = std::filesystem::path(this->get_root_dir()) / "generated/inspector";
            }
            break;

        // TT_METAL_INSPECTOR_INITIALIZATION_IS_IMPORTANT
        // Controls whether initialization is considered important for inspector. Set to '0' to disable.
        // Default: false (not important)
        // Usage: export TT_METAL_INSPECTOR_INITIALIZATION_IS_IMPORTANT=1
        case EnvVarID::TT_METAL_INSPECTOR_INITIALIZATION_IS_IMPORTANT:
            if (value != nullptr) {
                this->inspector_settings.initialization_is_important = true;
                if (strcmp(value, "0") == 0) {
                    this->inspector_settings.initialization_is_important = false;
                }
            }
            break;

        // TT_METAL_INSPECTOR_WARN_ON_WRITE_EXCEPTIONS
        // Controls whether to warn on write exceptions in inspector. Set to '0' to disable warnings.
        // Default: true (warnings enabled)
        // Usage: export TT_METAL_INSPECTOR_WARN_ON_WRITE_EXCEPTIONS=0
        case EnvVarID::TT_METAL_INSPECTOR_WARN_ON_WRITE_EXCEPTIONS:
            if (value != nullptr) {
                this->inspector_settings.warn_on_write_exceptions = true;
                if (strcmp(value, "0") == 0) {
                    this->inspector_settings.warn_on_write_exceptions = false;
                }
            }
            break;

        // TT_METAL_RISCV_DEBUG_INFO
        // Enable RISC-V debug info. Defaults to inspector setting, override with 0/1.
        // Default: Inherits from inspector setting
        // Usage: export TT_METAL_RISCV_DEBUG_INFO=1  # or =0 to disable
        case EnvVarID::TT_METAL_RISCV_DEBUG_INFO: {
            bool enable_riscv_debug_info = this->get_inspector_enabled();  // Default from inspector
            if (value != nullptr) {
                enable_riscv_debug_info = true;  // Default to true if set
                if (strcmp(value, "0") == 0) {
                    enable_riscv_debug_info = false;  // Only "0" = false
                }
            }
            this->set_riscv_debug_info_enabled(enable_riscv_debug_info);
            break;
        }

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
        case EnvVarID::RELIABILITY_MODE:
        case EnvVarID::TT_METAL_MULTI_AERISC:
        case EnvVarID::TT_METAL_USE_MGD_2_0:
        case EnvVarID::TT_METAL_INSPECTOR_RPC_SERVER_ADDRESS:
        case EnvVarID::TT_METAL_INSPECTOR_RPC:
        case EnvVarID::COUNT:
            // These variables are not yet implemented or are just markers
            break;
        case EnvVarID::TT_METAL_DPRINT_PREPEND_DEVICE_CORE_RISC:
            // Handled by ParseFeatureEnv() - this is for documentation
            break;
    }
}
void RunTimeOptions::InitializeFromEnvVars() {
    // Use enchantum to automatically iterate over all EnvVarID enum values
    for (const auto [id, name] : enchantum::entries_generator<EnvVarID>) {
        const char* value = std::getenv(name.data());

        // Only process if the environment variable is set
        if (value != nullptr) {
            HandleEnvVar(id, value);
        }
    }

    // TT_METAL_INSPECTOR_LOG_PATH: Set default path if not specified
    if (std::getenv("TT_METAL_INSPECTOR_LOG_PATH") == nullptr) {
        HandleEnvVar(EnvVarID::TT_METAL_INSPECTOR_LOG_PATH, nullptr);
    }

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
    const char* inspector_enable_str = getenv("TT_METAL_INSPECTOR");
    if (inspector_enable_str != nullptr) {
        inspector_settings.enabled = true;
        if (strcmp(inspector_enable_str, "0") == 0) {
            inspector_settings.enabled = false;
        }
    }

    const char* inspector_log_path_str = getenv("TT_METAL_INSPECTOR_LOG_PATH");
    if (inspector_log_path_str != nullptr) {
        inspector_settings.log_path = std::filesystem::path(inspector_log_path_str);
    } else {
        inspector_settings.log_path = std::filesystem::path(get_root_dir()) / "generated/inspector";
    }

    const char* inspector_initialization_is_important_str = getenv("TT_METAL_INSPECTOR_INITIALIZATION_IS_IMPORTANT");
    if (inspector_initialization_is_important_str != nullptr) {
        inspector_settings.initialization_is_important = true;
        if (strcmp(inspector_initialization_is_important_str, "0") == 0) {
            inspector_settings.initialization_is_important = false;
        }
    }

    const char* inspector_warn_on_write_exceptions_str = getenv("TT_METAL_INSPECTOR_WARN_ON_WRITE_EXCEPTIONS");
    if (inspector_warn_on_write_exceptions_str != nullptr) {
        inspector_settings.warn_on_write_exceptions = true;
        if (strcmp(inspector_warn_on_write_exceptions_str, "0") == 0) {
            inspector_settings.warn_on_write_exceptions = false;
        }
    }

    const char* inspector_rpc_server_address_str = getenv("TT_METAL_INSPECTOR_RPC_SERVER_ADDRESS");
    if (inspector_rpc_server_address_str != nullptr) {
        inspector_settings.rpc_server_address = std::string(inspector_rpc_server_address_str);
    }

    const char* inspector_rpc_str = getenv("TT_METAL_INSPECTOR_RPC");
    if (inspector_rpc_str != nullptr) {
        inspector_settings.rpc_server_enabled = true;
        if (std::strncmp(inspector_rpc_str, "0", 1) == 0) {
            inspector_settings.rpc_server_enabled = false;
        }
    }
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
