#include <prometheus/counter.h>
#include <prometheus/exposer.h>
#include <prometheus/registry.h>
#include <prometheus/gauge.h>

#include <thread>
#include <chrono>
#include <iostream>
#include <memory>
#include "mesh_command_queue.hpp"
#include "mesh_device.hpp"

#include <functional>

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/bfloat16.hpp>

using namespace tt;
using namespace tt::tt_metal;
using namespace tt::tt_metal::distributed;

MeshWorkload CreateMeshWorkloadInstrumented(prometheus::Counter& counter) {
    counter.Increment();
    return CreateMeshWorkload();
}

void AddProgramToMeshWorkloadInstrumented(
    prometheus::Counter& counter, MeshWorkload& workload, Program&& program, const MeshCoordinateRange& range) {
    counter.Increment();
    AddProgramToMeshWorkload(workload, std::move(program), range);
}

void EnqueueMeshWorkloadInstrumented(
    prometheus::Counter& counter, MeshCommandQueue& cq, MeshWorkload& workload, bool blocking) {
    counter.Increment();
    EnqueueMeshWorkload(cq, workload, blocking);
}

std::shared_ptr<MeshBuffer> MeshBufferCreateInstrumented(
    prometheus::Family<prometheus::Counter>& counter_family,
    const std::string& buffer_label,
    const MeshBufferConfig& sharded_config,
    const DeviceLocalBufferConfig& local_config,
    MeshDevice* mesh_device) {
    // Dynamically attaching a label is slow and not recommended.
    // For the sake of example.
    counter_family.Add({{"buffer_name", buffer_label}}).Increment();
    return MeshBuffer::create(sharded_config, local_config, mesh_device);
}

Program CreateEltwiseAddProgram(
    const std::shared_ptr<MeshBuffer>& a,
    const std::shared_ptr<MeshBuffer>& b,
    const std::shared_ptr<MeshBuffer>& c,
    size_t tile_size_bytes,
    uint32_t num_tiles) {
    auto program = CreateProgram();
    auto target_tensix_core = CoreRange(CoreCoord{0, 0});

    // Add circular buffers for data movement
    constexpr uint32_t src0_cb_index = tt::CBIndex::c_0;
    constexpr uint32_t num_input_tiles = 1;
    CircularBufferConfig cb_src0_config =
        CircularBufferConfig(num_input_tiles * tile_size_bytes, {{src0_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(src0_cb_index, tile_size_bytes);
    CBHandle cb_src0 = tt_metal::CreateCircularBuffer(program, target_tensix_core, cb_src0_config);

    constexpr uint32_t src1_cb_index = tt::CBIndex::c_1;
    CircularBufferConfig cb_src1_config =
        CircularBufferConfig(num_input_tiles * tile_size_bytes, {{src1_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(src1_cb_index, tile_size_bytes);
    CBHandle cb_src1 = tt_metal::CreateCircularBuffer(program, target_tensix_core, cb_src1_config);

    constexpr uint32_t output_cb_index = tt::CBIndex::c_16;
    constexpr uint32_t num_output_tiles = 1;
    CircularBufferConfig cb_output_config =
        CircularBufferConfig(num_output_tiles * tile_size_bytes, {{output_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(output_cb_index, tile_size_bytes);
    CBHandle cb_output = tt_metal::CreateCircularBuffer(program, target_tensix_core, cb_output_config);

    // Add data movement kernels
    KernelHandle reader = CreateKernel(
        program,
        "tt_metal/programming_examples/contributed/vecadd/kernels/interleaved_tile_read.cpp",
        target_tensix_core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});

    KernelHandle writer = CreateKernel(
        program,
        "tt_metal/programming_examples/contributed/vecadd/kernels/tile_write.cpp",
        target_tensix_core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    // Create the eltwise binary kernel
    auto compute = CreateKernel(
        program,
        "tt_metal/programming_examples/contributed/vecadd/kernels/add.cpp",
        target_tensix_core,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = false,
            .math_approx_mode = false,
            .compile_args = {},
            .defines = {{"ELTWISE_OP", "add_tiles"}, {"ELTWISE_OP_TYPE", "EltwiseBinaryType::ELWADD"}}});

    // Set runtime arguments for each device
    SetRuntimeArgs(program, reader, target_tensix_core, {a->address(), b->address(), num_tiles});
    SetRuntimeArgs(program, writer, target_tensix_core, {c->address(), num_tiles});
    SetRuntimeArgs(program, compute, target_tensix_core, {num_tiles});

    return program;
}

int run_programming_example(
    prometheus::Counter& create_mesh_workload_counter,
    prometheus::Counter& add_program_counter,
    prometheus::Counter& enqueue_mesh_workload_counter,
    prometheus::Family<prometheus::Counter>& mesh_buffer_create_counter_family,
    prometheus::Counter& set_runtime_args_counter,
    prometheus::Gauge& num_devices_gauge) {
    auto mesh_device = MeshDevice::create(MeshDeviceConfig(MeshShape(2, 4)));

    // Set the gauge value
    num_devices_gauge.Set(mesh_device->num_devices());

    // Define the global buffer shape and shard shape for distributed buffers
    auto shard_shape = Shape2D{32, 32};
    auto distributed_buffer_shape =
        Shape2D{shard_shape.height() * mesh_device->num_rows(), shard_shape.width() * mesh_device->num_cols()};
    auto num_tiles = 1;
    auto tile_size_bytes = tt::tt_metal::detail::TileSize(tt::DataFormat::Float16_b);
    auto distributed_buffer_size_bytes = mesh_device->num_rows() * mesh_device->num_cols() * tile_size_bytes;

    // Configure device-local buffer settings
    auto local_buffer_config = DeviceLocalBufferConfig{
        .page_size = tile_size_bytes,
        .buffer_type = BufferType::DRAM,
        .buffer_layout = TensorMemoryLayout::INTERLEAVED,
        .bottom_up = false};
    auto distributed_buffer_config = tt::tt_metal::distributed::ShardedBufferConfig{
        .global_size = distributed_buffer_size_bytes,
        .global_buffer_shape = distributed_buffer_shape,
        .shard_shape = shard_shape,
        .shard_orientation = ShardOrientation::ROW_MAJOR};

    auto a = MeshBufferCreateInstrumented(
        mesh_buffer_create_counter_family,
        "a_buffer",
        distributed_buffer_config,
        local_buffer_config,
        mesh_device.get());
    auto b = MeshBufferCreateInstrumented(
        mesh_buffer_create_counter_family,
        "b_buffer",
        distributed_buffer_config,
        local_buffer_config,
        mesh_device.get());
    auto c = MeshBufferCreateInstrumented(
        mesh_buffer_create_counter_family,
        "c_buffer",
        distributed_buffer_config,
        local_buffer_config,
        mesh_device.get());

    // Create and initialize source data
    constexpr float val_to_add = 0.5f;
    std::vector<uint32_t> a_data =
        create_random_vector_of_bfloat16(distributed_buffer_size_bytes, 1 /* rand_max_float */, 0 /* seed */);
    std::vector<uint32_t> b_data = create_constant_vector_of_bfloat16(distributed_buffer_size_bytes, val_to_add);

    // Write data to distributed buffers
    auto& cq = mesh_device->mesh_command_queue();
    EnqueueWriteMeshBuffer(cq, a, a_data, false /* blocking */);
    EnqueueWriteMeshBuffer(cq, b, b_data, false /* blocking */);

    // Create program for distributed computation
    auto program = CreateEltwiseAddProgram(a, b, c, tile_size_bytes, num_tiles);

    // Manual increment for SetRuntimeArgs (still not ideal, but works for demo)
    set_runtime_args_counter.Increment(3);

    // Create mesh workload and broadcast the program across all devices using instrumented calls
    auto mesh_workload = CreateMeshWorkloadInstrumented(create_mesh_workload_counter);
    auto device_range = MeshCoordinateRange(mesh_device->shape());

    AddProgramToMeshWorkloadInstrumented(add_program_counter, mesh_workload, std::move(program), device_range);
    EnqueueMeshWorkloadInstrumented(enqueue_mesh_workload_counter, cq, mesh_workload, false /* blocking */);

    // Read back results
    std::vector<uint32_t> result_data(a_data.size(), 0);
    EnqueueReadMeshBuffer(cq, result_data, c, true /* blocking */);

    // Verify results
    auto transform_to_golden = [val_to_add](const bfloat16& a) { return bfloat16(a.to_float() + val_to_add); };
    std::vector<uint32_t> golden_data =
        pack_bfloat16_vec_into_uint32_vec(unpack_uint32_vec_into_bfloat16_vec(a_data, transform_to_golden));

    // Print partial results so we can see the output is correct (plus or minus some error due to BFP16 precision)
    std::cout << "Partial results: (note we are running under BFP16. It's going to be less accurate)\n";
    bfloat16* a_bf16 = reinterpret_cast<bfloat16*>(a_data.data());
    bfloat16* b_bf16 = reinterpret_cast<bfloat16*>(b_data.data());
    bfloat16* c_bf16 = reinterpret_cast<bfloat16*>(result_data.data());
    bfloat16* golden_bf16 = reinterpret_cast<bfloat16*>(golden_data.data());

    size_t num_failures = 0;
    auto total_values = result_data.size() * 2;
    for (int i = 0; i < total_values; i++) {
        if (!is_close(c_bf16[i].to_float(), golden_bf16[i].to_float())) {
            num_failures++;
        }
    }

    std::cout << "Total values: " << total_values << "\n";
    std::cout << "Distributed elementwise add verification: " << (total_values - num_failures) << " / " << total_values
              << " passed\n";
    if (num_failures > 0) {
        std::cout << "Distributed elementwise add verification failed with " << num_failures << " failures\n";
        throw std::runtime_error("Distributed elementwise add verification failed");
    }

    return 0;
}

int main() {
    std::cout << "Starting metrics demo..." << std::endl;

    auto registry = std::make_shared<prometheus::Registry>();

    auto& loop_counter = prometheus::BuildCounter()
                             .Name("my_example_counter")
                             .Help("Counts how many times we've looped")
                             .Register(*registry)
                             .Add({{"label", "demo"}});

    auto& create_mesh_workload_counter = prometheus::BuildCounter()
                                             .Name("create_mesh_workload_calls")
                                             .Help("Counts calls to CreateMeshWorkload")
                                             .Register(*registry)
                                             .Add({});
    auto& add_program_counter = prometheus::BuildCounter()
                                    .Name("add_program_to_mesh_workload_calls")
                                    .Help("Counts calls to AddProgramToMeshWorkload")
                                    .Register(*registry)
                                    .Add({});
    auto& enqueue_mesh_workload_counter = prometheus::BuildCounter()
                                              .Name("enqueue_mesh_workload_calls")
                                              .Help("Counts calls to EnqueueMeshWorkload")
                                              .Register(*registry)
                                              .Add({});
    auto& mesh_buffer_create_counter_family = prometheus::BuildCounter()
                                                  .Name("mesh_buffer_create_calls")
                                                  .Help("Counts calls to MeshBuffer::create")
                                                  .Register(*registry);
    auto& set_runtime_args_counter = prometheus::BuildCounter()
                                         .Name("set_runtime_args_calls")
                                         .Help("Counts calls to SetRuntimeArgs")
                                         .Register(*registry)
                                         .Add({});
    auto& num_devices_gauge = prometheus::BuildGauge()
                                  .Name("mesh_num_devices")
                                  .Help("Number of devices in the mesh")
                                  .Register(*registry)
                                  .Add({});

    // No idea how 5555 is used, but it is one of 2 ports exposed on docker image.
    // This is forwarded to top-level host port 52677.
    // So can be accessed on e.g. `sjc-snva-t3020:52677`
    prometheus::Exposer exposer{"0.0.0.0:5555"};
    exposer.RegisterCollectable(registry);

    int iteration = 0;
    while (true) {
        std::cout << "Running iteration: " << ++iteration << std::endl;
        try {
            // Pass metric references to the function
            run_programming_example(
                create_mesh_workload_counter,
                add_program_counter,
                enqueue_mesh_workload_counter,
                mesh_buffer_create_counter_family,
                set_runtime_args_counter,
                num_devices_gauge);
        } catch (const std::exception& e) {
            std::cerr << "Error during example execution: " << e.what() << std::endl;
        }
        loop_counter.Increment();
        std::this_thread::sleep_for(std::chrono::seconds(3));
    }
}
