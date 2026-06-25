// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <fmt/ostream.h>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tt_metal_profiler.hpp>
#include <tt-metalium/work_split.hpp>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>
#include <string>
#include <map>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <sstream>
#include "exhaustive_bf16_generator.hpp"

using namespace tt::tt_metal;

// Timing helper macro
#define START_TIMER(name) auto timer_##name##_start = std::chrono::high_resolution_clock::now()
#define END_TIMER(name)                                                                                      \
    do {                                                                                                     \
        auto timer_##name##_end = std::chrono::high_resolution_clock::now();                                 \
        auto timer_##name##_duration =                                                                       \
            std::chrono::duration_cast<std::chrono::nanoseconds>(timer_##name##_end - timer_##name##_start); \
        fmt::print("TIMING_{}: {:.6f}\n", #name, timer_##name##_duration.count() / 1e6);                     \
    } while (0)

#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

#ifdef KERNEL_VARIANT
#define COMPUTE_KERNEL_PATH \
    OVERRIDE_KERNEL_PREFIX "generic_lut_activation_embedded/kernels/compute/" KERNEL_VARIANT ".cpp"
#else
#define COMPUTE_KERNEL_PATH \
    OVERRIDE_KERNEL_PREFIX "generic_lut_activation_embedded/kernels/compute/piecewise_constant.cpp"
#endif

int main(int argc, char** argv) {
    if (argc < 2) {
        fmt::print(
            stderr,
            "Usage: {} --activation <name> --range-min <min> --range-max <max> [--tiles N] [--precision "
            "bf16|fp32|both]\n",
            argv[0]);
        fmt::print(stderr, "\nRequired arguments:\n");
        fmt::print(stderr, "  --activation <name>       Activation function name\n");
        fmt::print(stderr, "  --range-min <min>         Minimum test input value\n");
        fmt::print(stderr, "  --range-max <max>         Maximum test input value\n");
        fmt::print(stderr, "\nOptional arguments:\n");
        fmt::print(stderr, "  --tiles N                 Number of tiles to process (default: 32)\n");
        fmt::print(
            stderr, "  --batch-tiles 1,8,256     Comma-separated tile counts (runs all in one device session)\n");
        fmt::print(
            stderr, "  --precision bf16|fp32|both Data precision (default: fp32). 'both' loops over bf16 then fp32.\n");
        fmt::print(
            stderr,
            "  --no-dual-eval            Disable dual x-vector evaluation (enabled by default; gives 1.25x SFPU "
            "speedup)\n");
        fmt::print(
            stderr,
            "  --no-adaptive-degree      Disable per-segment adaptive degree optimization (all segments use "
            "POLY_DEGREE)\n");
        return 1;
    }

    uint32_t n_tiles = 32;
    std::vector<uint32_t> batch_tiles;
    std::string precision = "fp32";
    std::string activation_name;
    float test_min = 0.0f;
    float test_max = 0.0f;
    bool use_dual_eval = true;        // Default: dual x-vector evaluation (1.25x speedup)
    bool no_adaptive_degree = false;  // Default: adaptive degree ON (SEG{i}_DEGREE macros active)
    // no_clamp removed: clamping is never needed (segment cascade handles out-of-range inputs)
    bool has_activation = false;
    bool has_range_min = false;
    bool has_range_max = false;

    // Parse arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--activation") {
            if (i + 1 >= argc) {
                fmt::print(stderr, "Error: --activation requires an argument\n");
                return 1;
            }
            activation_name = argv[++i];
            has_activation = true;
        } else if (arg == "--range-min") {
            if (i + 1 >= argc) {
                fmt::print(stderr, "Error: --range-min requires an argument\n");
                return 1;
            }
            test_min = std::stof(argv[++i]);
            has_range_min = true;
        } else if (arg == "--range-max") {
            if (i + 1 >= argc) {
                fmt::print(stderr, "Error: --range-max requires an argument\n");
                return 1;
            }
            test_max = std::stof(argv[++i]);
            has_range_max = true;
        } else if (arg == "--tiles" || arg == "-t") {
            if (i + 1 >= argc) {
                fmt::print(stderr, "Error: --tiles requires an argument\n");
                return 1;
            }
            try {
                n_tiles = std::stoi(argv[++i]);
                if (n_tiles == 0 || n_tiles > 100000) {
                    fmt::print(stderr, "Error: tiles must be between 1 and 100000 (got {})\n", n_tiles);
                    return 1;
                }
            } catch (const std::exception& e) {
                fmt::print(stderr, "Error: invalid tiles value '{}'\n", argv[i]);
                return 1;
            }
        } else if (arg == "--precision" || arg == "-p") {
            if (i + 1 >= argc) {
                fmt::print(stderr, "Error: --precision requires an argument\n");
                return 1;
            }
            precision = argv[++i];
            if (precision != "bf16" && precision != "fp32" && precision != "both") {
                fmt::print(stderr, "Error: precision must be 'bf16', 'fp32', or 'both' (got '{}')\n", precision);
                return 1;
            }
        } else if (arg == "--dual-eval") {
            use_dual_eval = true;  // now redundant (default), kept for compatibility
        } else if (arg == "--no-dual-eval") {
            use_dual_eval = false;
        } else if (arg == "--no-adaptive-degree") {
            no_adaptive_degree = true;
        } else if (arg == "--batch-tiles") {
            if (i + 1 >= argc) {
                fmt::print(stderr, "Error: --batch-tiles requires a comma-separated list of tile counts\n");
                return 1;
            }
            std::string tiles_str = argv[++i];
            std::istringstream ss(tiles_str);
            std::string token;
            while (std::getline(ss, token, ',')) {
                try {
                    uint32_t t = std::stoi(token);
                    if (t == 0 || t > 100000) {
                        fmt::print(stderr, "Error: tile count must be between 1 and 100000 (got {})\n", t);
                        return 1;
                    }
                    batch_tiles.push_back(t);
                } catch (const std::exception& e) {
                    fmt::print(stderr, "Error: invalid tile count '{}' in --batch-tiles\n", token);
                    return 1;
                }
            }
        } else {
            fmt::print(stderr, "Warning: unknown argument '{}'\n", arg);
        }
    }

    if (!has_activation || !has_range_min || !has_range_max) {
        fmt::print(stderr, "Error: --activation, --range-min, and --range-max are required\n");
        return 1;
    }

    // If no --batch-tiles, use single n_tiles value (backward compat)
    if (batch_tiles.empty()) {
        batch_tiles.push_back(n_tiles);
    }
    bool is_batch_mode = (batch_tiles.size() > 1);

    // Build list of precisions to iterate over
    std::vector<std::string> precisions;
    if (precision == "both") {
        precisions = {"bf16", "fp32"};
    } else {
        precisions = {precision};
    }
    bool is_multi_precision = (precisions.size() > 1);

    // Capture DUMP_OUTPUT_CSV base path and profiler base dir before the loop
    std::string dump_csv_base;
    const char* dump_csv_env = std::getenv("DUMP_OUTPUT_CSV");
    if (dump_csv_env) {
        dump_csv_base = dump_csv_env;
    }

    bool pass = true;

    try {
        fmt::print("{}\n", std::string(60, '='));
        fmt::print("Generic LUT Activation Function Example (Embedded Mode)\n");
        fmt::print("{}\n", std::string(60, '='));

        // Embedded mode: LUT constants compiled into kernel, no file loading needed
        fmt::print("Mode: Embedded LUTs (compile-time constants)\n");
        fmt::print("Kernel variant: {}\n", KERNEL_VARIANT);
        fmt::print("Precision: {}\n", precision);
        fmt::print("✓ LUT data embedded in kernel binary (zero L1 overhead)\n");
        fmt::print("✓ LUT precision selected at runtime via JIT compilation\n\n");

        fmt::print("Activation: {} | Test range: [{}, {}]\n", activation_name, test_min, test_max);

        if (is_batch_mode) {
            fmt::print("Batch mode: {} shapes (", batch_tiles.size());
            for (size_t i = 0; i < batch_tiles.size(); i++) {
                if (i > 0) {
                    fmt::print(",");
                }
                fmt::print("{}", batch_tiles[i]);
            }
            fmt::print(" tiles)\n");
        }

        // STEP 2: Create device (once for all shapes)
        START_TIMER(DEVICE_INIT);
        constexpr int device_id = 0;
        auto mesh_device = distributed::MeshDevice::create_unit_mesh(device_id);
        distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
        END_TIMER(DEVICE_INIT);

        // ===== PRECISION LOOP × BATCH LOOP =====
        for (const auto& current_precision : precisions) {
            bool use_bf16 = (current_precision == "bf16");
            tt::DataFormat data_format = use_bf16 ? tt::DataFormat::Float16_b : tt::DataFormat::Float32;

            if (is_multi_precision) {
                fmt::print("\n===== Precision: {} ({}) =====\n", current_precision, use_bf16 ? "bfloat16" : "float32");
            }

            for (uint32_t current_tiles : batch_tiles) {
                // Build prefix: include precision only when looping over multiple precisions
                // IMPORTANT: single-precision batch must keep "BATCH[tiles=N]:" format
                // for backward compat with sweep_helpers.sh extract_shape_timing()
                std::string batch_prefix;
                if (is_multi_precision) {
                    batch_prefix = fmt::format("BATCH[{},tiles={}]:", current_precision, current_tiles);
                } else if (is_batch_mode) {
                    batch_prefix = fmt::format("BATCH[tiles={}]:", current_tiles);
                }

                if (is_batch_mode || is_multi_precision) {
                    fmt::print("\n{}BEGIN\n", batch_prefix);
                }

                START_TIMER(PROGRAM_CREATION);
                Program program = CreateProgram();

                // Tile configuration
                constexpr uint32_t elements_per_tile = tt::constants::TILE_WIDTH * tt::constants::TILE_HEIGHT;
                const uint32_t element_size = use_bf16 ? sizeof(bfloat16) : sizeof(float);
                const uint32_t tile_size_bytes = element_size * elements_per_tile;
                const uint32_t dram_buffer_size = tile_size_bytes * current_tiles;

                fmt::print(
                    "Processing {} tiles ({} elements total, {} bytes per element)\n",
                    current_tiles,
                    current_tiles * elements_per_tile,
                    element_size);

                // MULTI-CORE: Get grid size and split work across cores
                CoreCoord grid_size = mesh_device->compute_with_storage_grid_size();
                fmt::print("Grid size: {}x{}\n", grid_size.x, grid_size.y);

                auto [num_cores, all_cores, core_group_1, core_group_2, tiles_per_core_1, tiles_per_core_2] =
                    split_work_to_cores(grid_size, current_tiles);

                fmt::print("Work split: {} cores total\n", num_cores);
                fmt::print("Group 1: {} cores, {} tiles/core\n", core_group_1.num_cores(), tiles_per_core_1);
                if (!core_group_2.ranges().empty()) {
                    fmt::print("Group 2: {} cores, {} tiles/core\n", core_group_2.num_cores(), tiles_per_core_2);
                }

                // Verify work distribution
                uint32_t total_tiles = core_group_1.num_cores() * tiles_per_core_1;
                if (!core_group_2.ranges().empty()) {
                    total_tiles += core_group_2.num_cores() * tiles_per_core_2;
                }
                TT_FATAL(
                    total_tiles == current_tiles,
                    "Work split mismatch! {} cores × tiles != {} tiles total",
                    total_tiles,
                    current_tiles);

                // Embedded mode: No LUT circular buffer needed (constants in kernel binary)
                fmt::print("✓ Skipping LUT circular buffer (using embedded constants)\n");

                // STEP 5: Create input/output circular buffers on ALL cores
                constexpr uint32_t tiles_per_cb = 2;
                constexpr auto cb_in = tt::CBIndex::c_0;
                constexpr auto cb_out = tt::CBIndex::c_16;

                CreateCircularBuffer(
                    program,
                    all_cores,  // Create on ALL cores
                    CircularBufferConfig(tiles_per_cb * tile_size_bytes, {{cb_in, data_format}})
                        .set_page_size(cb_in, tile_size_bytes));

                CreateCircularBuffer(
                    program,
                    all_cores,  // Create on ALL cores
                    CircularBufferConfig(tiles_per_cb * tile_size_bytes, {{cb_out, data_format}})
                        .set_page_size(cb_out, tile_size_bytes));
                END_TIMER(PROGRAM_CREATION);

                // STEP 6: Allocate DRAM buffers
                START_TIMER(BUFFER_ALLOCATION);
                distributed::DeviceLocalBufferConfig dram_config{
                    .page_size = tile_size_bytes, .buffer_type = BufferType::DRAM};
                distributed::ReplicatedBufferConfig buffer_config{.size = dram_buffer_size};

                auto input_buffer = distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());
                auto output_buffer = distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());
                END_TIMER(BUFFER_ALLOCATION);

                // STEP 7: Create test input data using activation's test range
                START_TIMER(DATA_PREPARATION);
                fmt::print("Creating test input data (range [{}, {}])...\n", test_min, test_max);

                const size_t num_elements = elements_per_tile * current_tiles;

                std::vector<uint8_t> input_buffer_data(dram_buffer_size);

                if (use_bf16) {
                    // Generate EXHAUSTIVE BF16 inputs (all possible BF16 values in range, excluding subnormals)
                    // This ensures we test worst-case bit patterns, not just linearly-spaced values
                    fmt::print("Generating exhaustive BF16 inputs (excluding subnormals)...\n");

                    bfloat16* input_ptr = reinterpret_cast<bfloat16*>(input_buffer_data.data());
                    size_t unique_count = fill_buffer_with_exhaustive_bf16(input_ptr, num_elements, test_min, test_max);

                    fmt::print("  Found {} unique BF16 values in range [{}, {}]\n", unique_count, test_min, test_max);
                    fmt::print(
                        "  Generated {} total elements ({} tiles, repeating exhaustive set)\n",
                        num_elements,
                        current_tiles);
                } else {
                    // FP32: Use linear spacing (exhaustive FP32 not feasible - 2^32 values)
                    fmt::print("Generating linearly-spaced FP32 inputs...\n");
                    float test_range = test_max - test_min;
                    float* input_ptr = reinterpret_cast<float*>(input_buffer_data.data());
                    for (size_t i = 0; i < num_elements; i++) {
                        input_ptr[i] = test_min + test_range * (i / float(num_elements));
                    }
                }
                END_TIMER(DATA_PREPARATION);

                START_TIMER(HOST_TO_DEVICE);
                distributed::EnqueueWriteMeshBuffer(cq, input_buffer, input_buffer_data, false);
                END_TIMER(HOST_TO_DEVICE);

                // STEP 8: Create kernels
                START_TIMER(KERNEL_CREATION);

                // Create reader kernel on ALL cores
                std::vector<uint32_t> reader_compile_time_args;
                TensorAccessorArgs(*input_buffer->get_backing_buffer()).append_to(reader_compile_time_args);
                auto reader = CreateKernel(
                    program,
                    OVERRIDE_KERNEL_PREFIX "generic_lut_activation_embedded/kernels/dataflow/reader.cpp",
                    all_cores,  // Create on ALL cores
                    DataMovementConfig{
                        .processor = DataMovementProcessor::RISCV_0,
                        .noc = NOC::RISCV_0_default,
                        .compile_args = reader_compile_time_args});

                // Create writer kernel on ALL cores
                std::vector<uint32_t> writer_compile_time_args;
                TensorAccessorArgs(*output_buffer->get_backing_buffer()).append_to(writer_compile_time_args);
                auto writer = CreateKernel(
                    program,
                    OVERRIDE_KERNEL_PREFIX "generic_lut_activation_embedded/kernels/dataflow/writer.cpp",
                    all_cores,  // Create on ALL cores
                    DataMovementConfig{
                        .processor = DataMovementProcessor::RISCV_1,
                        .noc = NOC::RISCV_1_default,
                        .compile_args = writer_compile_time_args});

                fmt::print("✓ Created reader/writer kernels on {} cores\n", num_cores);

                // Compute kernel configuration
                // Runtime precision selection via JIT compilation defines
                std::map<std::string, std::string> compute_defines;
                std::vector<UnpackToDestMode> unpack_to_dest_modes(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);

                if (use_bf16) {
                    compute_defines["USE_BF16"] = "1";  // Selects BF16 LUT data and hardware config
                } else {
                    // FP32 mode: hardware handles FP32 via fp32_dest_acc_en + UnpackToDestFp32
                    constexpr auto cb_in_fp32 = tt::CBIndex::c_0;
                    constexpr auto cb_out_fp32 = tt::CBIndex::c_16;
                    unpack_to_dest_modes[static_cast<uint32_t>(cb_in_fp32)] = UnpackToDestMode::UnpackToDestFp32;
                    unpack_to_dest_modes[static_cast<uint32_t>(cb_out_fp32)] = UnpackToDestMode::UnpackToDestFp32;
                }

                // Set dual evaluation flag if requested
                if (use_dual_eval) {
                    compute_defines["USE_DUAL_EVAL"] = "1";
                    fmt::print("✓ Dual x-vector evaluation enabled (exploits SFPU ILP)\n");
                }

                // Disable adaptive per-segment degree optimization if requested
                if (no_adaptive_degree) {
                    compute_defines["DISABLE_ADAPTIVE_DEGREE"] = "1";
                    fmt::print("✓ Adaptive degree optimization disabled (all segments use POLY_DEGREE)\n");
                }

#ifdef KERNEL_VARIANT
                std::string kernel_variant = KERNEL_VARIANT;
                fmt::print("✓ Using kernel variant '{}' with precision={}\n", kernel_variant, precision);
#else
                fmt::print("✓ Using generic kernel with precision={}\n", precision);
#endif

                // Create compute kernel for core_group_1
                auto compute_kernel_1 = CreateKernel(
                    program,
                    COMPUTE_KERNEL_PATH,
                    core_group_1,
                    ComputeConfig{
                        .fp32_dest_acc_en = !use_bf16,                // Enable FP32 accumulator for FP32 mode
                        .unpack_to_dest_mode = unpack_to_dest_modes,  // Enable FP32 unpacking for input/output
                        .math_approx_mode = false,                    // LUT evaluator does not use approximate SFPU ops
                        .defines = compute_defines});

                // Create compute kernel for core_group_2 (if it exists)
                KernelHandle compute_kernel_2 = 0;
                if (!core_group_2.ranges().empty()) {
                    compute_kernel_2 = CreateKernel(
                        program,
                        COMPUTE_KERNEL_PATH,
                        core_group_2,
                        ComputeConfig{
                            .fp32_dest_acc_en = !use_bf16,
                            .unpack_to_dest_mode = unpack_to_dest_modes,
                            .math_approx_mode = use_bf16,
                            .defines = compute_defines});
                    fmt::print("✓ Created compute kernels for group 1 and group 2\n");
                } else {
                    fmt::print("✓ Created compute kernel for group 1 only\n");
                }

                // Set runtime arguments with per-core tile offset tracking
                uint32_t num_cores_y = grid_size.y;
                uint32_t tiles_written = 0;

                fmt::print("\nSetting runtime arguments:\n");
                fmt::print("{}\n", std::string(80, '-'));

                for (uint32_t i = 0; i < num_cores; i++) {
                    // Use TTNN's formula for core coordinates
                    CoreCoord core = {i / num_cores_y, i % num_cores_y};

                    // Check which group this core belongs to
                    uint32_t tiles_this_core = 0;
                    KernelHandle compute_kernel_id = 0;

                    if (core_group_1.contains(core)) {
                        tiles_this_core = tiles_per_core_1;
                        compute_kernel_id = compute_kernel_1;
                        fmt::print(
                            "Core ({},{}) → Group 1: {} tiles, offset {}\n",
                            core.x,
                            core.y,
                            tiles_this_core,
                            tiles_written);
                    } else if (core_group_2.contains(core)) {
                        tiles_this_core = tiles_per_core_2;
                        compute_kernel_id = compute_kernel_2;
                        fmt::print(
                            "Core ({},{}) → Group 2: {} tiles, offset {}\n",
                            core.x,
                            core.y,
                            tiles_this_core,
                            tiles_written);
                    } else {
                        fmt::print("ERROR: Core ({},{}) not in any group!\n", core.x, core.y);
                        TT_FATAL(false, "Core not assigned to any group");
                    }

                    // Set reader args: (input_addr, num_tiles, start_tile_id)
                    SetRuntimeArgs(program, reader, core, {input_buffer->address(), tiles_this_core, tiles_written});

                    // Set writer args: (buffer_addr, num_tiles, start_tile_id)
                    SetRuntimeArgs(program, writer, core, {output_buffer->address(), tiles_this_core, tiles_written});

                    // Set compute args: (num_tiles)
                    SetRuntimeArgs(program, compute_kernel_id, core, {tiles_this_core});

                    tiles_written += tiles_this_core;
                }

                fmt::print("{}\n", std::string(80, '-'));
                fmt::print("Total tiles distributed: {}\n", tiles_written);
                TT_FATAL(
                    tiles_written == current_tiles,
                    "Tile distribution mismatch! {} != {}",
                    tiles_written,
                    current_tiles);

                END_TIMER(KERNEL_CREATION);

                fmt::print("✓ Kernels created and configured\n\n");

                // STEP 9: Execute program
                fmt::print("Executing program with embedded LUT...\n");
                distributed::MeshWorkload workload;
                distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());
                workload.add_program(device_range, std::move(program));

                auto kernel_exec_start = std::chrono::high_resolution_clock::now();
                distributed::EnqueueMeshWorkload(cq, workload, false);
                distributed::Finish(cq);
                auto kernel_exec_end = std::chrono::high_resolution_clock::now();
                auto kernel_exec_duration =
                    std::chrono::duration_cast<std::chrono::nanoseconds>(kernel_exec_end - kernel_exec_start);
                double kernel_exec_ms = kernel_exec_duration.count() / 1e6;

                // Print timing with batch prefix if in batch mode
                fmt::print("{}TIMING_KERNEL_EXECUTION: {:.6f}\n", batch_prefix, kernel_exec_ms);

                fmt::print("✓ Execution complete\n\n");

                // STEP 10: Read and verify results
                START_TIMER(DEVICE_TO_HOST);
                std::vector<uint8_t> output_buffer_data;
                distributed::EnqueueReadMeshBuffer(cq, output_buffer_data, output_buffer, true);
                END_TIMER(DEVICE_TO_HOST);

                // Read mesh device profiler results if enabled
                // In batch mode, all shapes' data accumulates in a single CSV;
                // the shell scripts use time-gap clustering to extract per-shape times.
                if (std::getenv("TT_METAL_DEVICE_PROFILER")) {
                    ReadMeshDeviceProfilerResults(*mesh_device);
                    fmt::print("✓ Device profiler results written\n");
                }

                fmt::print("Results (samples across test range [{}, {}]):\n", test_min, test_max);
                fmt::print("{}\n", std::string(60, '-'));
                fmt::print("{:>5} {:>12} {:>12}\n", "Index", "Input", "Output");
                fmt::print("{}\n", std::string(60, '-'));

                // Helper lambdas to read input/output values
                auto get_input_value = [&](size_t i) -> float {
                    if (use_bf16) {
                        const bfloat16* ptr = reinterpret_cast<const bfloat16*>(input_buffer_data.data());
                        return static_cast<float>(ptr[i]);
                    } else {
                        const float* ptr = reinterpret_cast<const float*>(input_buffer_data.data());
                        return ptr[i];
                    }
                };

                auto get_output_value = [&](size_t i) -> float {
                    if (use_bf16) {
                        const bfloat16* ptr = reinterpret_cast<const bfloat16*>(output_buffer_data.data());
                        return static_cast<float>(ptr[i]);
                    } else {
                        const float* ptr = reinterpret_cast<const float*>(output_buffer_data.data());
                        return ptr[i];
                    }
                };

                // Show samples from start, middle, and end to demonstrate full range
                const int samples[] = {0, 3276, 6553, 9830, 13107, 16384, 19660, 22937, 26214, 29491, 32767};
                for (auto i : samples) {
                    if (static_cast<size_t>(i) < num_elements) {
                        fmt::print("{:5d} {:12.6f} {:12.6f}\n", i, get_input_value(i), get_output_value(i));
                    }
                }

                fmt::print("{}\n", std::string(60, '-'));
                fmt::print("\nProcessed {} tiles ({} elements total)\n", current_tiles, num_elements);

                // Dump full input/output to CSV for accuracy calculation by Python scripts
                if (!dump_csv_base.empty()) {
                    START_TIMER(CSV_DUMP);
                    // Insert precision and/or _tilesN before .csv extension
                    std::string dump_csv_path;
                    size_t ext_pos = dump_csv_base.rfind(".csv");
                    std::string base_no_ext =
                        (ext_pos != std::string::npos) ? dump_csv_base.substr(0, ext_pos) : dump_csv_base;

                    std::string suffix;
                    if (is_multi_precision) {
                        suffix += "_" + current_precision;
                    }
                    if (is_batch_mode) {
                        suffix += "_tiles" + std::to_string(current_tiles);
                    }

                    if (!suffix.empty()) {
                        dump_csv_path = base_no_ext + suffix + ".csv";
                    } else {
                        dump_csv_path = dump_csv_base;
                    }

                    std::ofstream csv_file(dump_csv_path);
                    if (csv_file.is_open()) {
                        // Use large buffer for faster I/O (default is too small for 52M elements)
                        constexpr size_t BUFFER_SIZE = 1 << 20;  // 1MB buffer
                        std::vector<char> file_buffer(BUFFER_SIZE);
                        csv_file.rdbuf()->pubsetbuf(file_buffer.data(), BUFFER_SIZE);

                        csv_file << "input,output\n";

                        // Build output in memory chunks for much faster I/O
                        // ~50x faster than per-element formatted writes
                        constexpr size_t CHUNK_SIZE = 65536;
                        std::string chunk;
                        chunk.reserve(CHUNK_SIZE * 80);  // ~80 chars per line with .17e format

                        for (size_t i = 0; i < num_elements; ++i) {
                            // Use fmt::format_to for faster formatting than iostream
                            // 17 decimal digits for double precision round-trip safety
                            fmt::format_to(
                                std::back_inserter(chunk),
                                "{:.17e},{:.17e}\n",
                                get_input_value(i),
                                get_output_value(i));

                            // Flush chunk periodically
                            if ((i + 1) % CHUNK_SIZE == 0) {
                                csv_file.write(chunk.data(), chunk.size());
                                chunk.clear();
                            }
                        }
                        // Write remaining
                        if (!chunk.empty()) {
                            csv_file.write(chunk.data(), chunk.size());
                        }
                        csv_file.close();
                        fmt::print("\n✓ Output data dumped to: {}\n", dump_csv_path);
                    }
                    END_TIMER(CSV_DUMP);
                }

                if (is_batch_mode || is_multi_precision) {
                    fmt::print("{}END\n", batch_prefix);
                }
            }  // end batch loop
        }  // end precision loop

        // Close device (once for all shapes)
        if (!mesh_device->close()) {
            pass = false;
        }

        fmt::print("\n{}\n", std::string(60, '='));
        if (pass) {
            fmt::print("✓ Test PASSED\n");
        } else {
            fmt::print("✗ Test FAILED\n");
        }
        fmt::print("{}\n", std::string(60, '='));

    } catch (const std::exception& e) {
        fmt::print(stderr, "\n✗ Test failed with exception!\n");
        fmt::print(stderr, "{}\n", e.what());
        return 1;
    }

    return pass ? 0 : 1;
}
