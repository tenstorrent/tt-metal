project(ttml)

set(SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/autograd/auto_context.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/autograd/auto_context.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/autograd/autocast_tensor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/autograd/autocast_tensor.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/autograd/graph.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/autograd/graph.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/autograd/graph_utils.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/autograd/tensor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/autograd/tensor.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/core/clip_grad_norm.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/core/clip_grad_norm.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/core/compute_kernel_config.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/core/compute_kernel_config.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/core/debug.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/core/device.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/core/device.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/core/distributed/distributed.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/core/distributed/distributed.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/core/distributed/ccl_resources.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/core/distributed/ccl_resources.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/core/distributed/socket_manager.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/core/distributed/socket_manager.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/core/mesh_device.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/core/mesh_device.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/core/not_null.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/core/random.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/core/random_sse.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/core/scoped.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/core/system_utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/core/system_utils.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/core/template_utils.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/core/tt_profiler.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/core/tt_profiler.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/core/tt_tensor_utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/core/tt_tensor_utils.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/core/ttnn_all_includes.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/core/xtensor_utils.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/datasets/dataloader.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/datasets/dataset_base.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/datasets/dataset_subset.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/datasets/generators.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/datasets/generators.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/datasets/in_memory_dataset.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/datasets/in_memory_token_dataset.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/datasets/in_memory_token_dataset.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/datasets/utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/datasets/utils.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/init/cpu_initializers.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/init/cpu_initializers.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/init/tensor_initializers.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/init/tensor_initializers.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/models/common/transformer_common.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/models/common/transformer_common.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/models/distributed/gpt2.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/models/distributed/gpt2.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/models/distributed/llama.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/models/distributed/llama.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/models/distributed/pipeline_parallel_llama.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/models/distributed/pipeline_parallel_llama.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/models/base_transformer.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/models/gpt2.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/models/gpt2.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/models/linear_regression.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/models/linear_regression.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/models/llama.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/models/llama.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/models/mlp.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/models/mlp.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/modules/distributed/gpt_block.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/modules/distributed/gpt_block.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/modules/distributed/grouped_query_attention.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/modules/distributed/grouped_query_attention.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/modules/distributed/linear.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/modules/distributed/linear.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/modules/distributed/llama_block.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/modules/distributed/llama_block.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/modules/distributed/multi_head_attention.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/modules/distributed/multi_head_attention.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/modules/dropout_module.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/modules/dropout_module.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/modules/embedding_module.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/modules/embedding_module.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/modules/gpt_block.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/modules/gpt_block.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/modules/grouped_query_attention.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/modules/grouped_query_attention.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/modules/layer_norm_module.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/modules/layer_norm_module.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/modules/linear_module.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/modules/linear_module.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/modules/llama_block.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/modules/llama_block.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/modules/lora_linear_module.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/modules/lora_linear_module.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/modules/module_base.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/modules/module_base.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/modules/multi_head_attention.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/modules/multi_head_attention.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/modules/multi_layer_perceptron.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/modules/multi_layer_perceptron.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/modules/positional_embeddings.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/modules/positional_embeddings.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/modules/rms_norm_module.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/modules/rms_norm_module.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/modules/rotary_embedding.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/modules/rotary_embedding.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/modules/single_head_attention.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/modules/single_head_attention.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/binary_ops.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/binary_ops.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/distributed/comm_ops.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/distributed/comm_ops.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/distributed/pipeline_parallel_comm_ops.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/distributed/pipeline_parallel_comm_ops.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/dropout_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/dropout_op.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/embedding_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/embedding_op.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/layernorm_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/layernorm_op.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/linear_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/linear_op.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/losses.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/losses.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/matmul_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/matmul_op.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/multi_head_utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/multi_head_utils.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/rmsnorm_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/rmsnorm_op.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/rope_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/rope_op.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/scaled_dot_product_attention.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/scaled_dot_product_attention.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/swiglu_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/swiglu_op.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/unary_ops.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/unary_ops.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/sampling_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/sampling_op.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/optimizers/adamw.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/optimizers/adamw.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/optimizers/optimizer_base.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/optimizers/optimizer_base.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/optimizers/remote_optimizer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/optimizers/remote_optimizer.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/optimizers/sgd.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/optimizers/sgd.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/optimizers/no_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/optimizers/no_op.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/optimizers/sgd_fused.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/optimizers/sgd_fused.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/schedulers/lambda_scheduler.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/schedulers/lambda_scheduler.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/schedulers/linear_scheduler.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/schedulers/linear_scheduler.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/schedulers/scheduler_base.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/schedulers/scheduler_base.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/schedulers/sequential_scheduler.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/schedulers/sequential_scheduler.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/schedulers/step_scheduler.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/schedulers/step_scheduler.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/serialization/flatbuffer_file.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/serialization/flatbuffer_file.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/serialization/safetensors.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/serialization/safetensors.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/serialization/serializable.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/serialization/serialization.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/serialization/serialization.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tokenizers/char_tokenizer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tokenizers/char_tokenizer.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tokenizers/char_tokenizer_trainer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tokenizers/char_tokenizer_trainer.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tokenizers/tokenizer_base.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ttnn_fixed/distributed/ttnn_ops.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ttnn_fixed/distributed/ttnn_ops.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ttnn_fixed/distributed/tt_metal.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ttnn_fixed/distributed/tt_metal.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ttnn_fixed/matmuls.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ttnn_fixed/matmuls.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ttnn_fixed/trivial_ttnn_ops.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ttnn_fixed/trivial_ttnn_ops.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/memory_utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/memory_utils.hpp
)

# Manually select only specific files from metal ops
set(METAL_OPS_FILES
    # General
    ${CMAKE_CURRENT_SOURCE_DIR}/metal/operations.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/metal/ttnn_all_includes.hpp
    # Common
    ${CMAKE_CURRENT_SOURCE_DIR}/metal/common/dataflow_utils.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/metal/common/compute_utils.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/metal/common/program_utils.hpp
    # RMSNorm
    # RMSNorm Forward
    ${CMAKE_CURRENT_SOURCE_DIR}/metal/ops/rmsnorm_fw/rmsnorm_fw.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/metal/ops/rmsnorm_fw/rmsnorm_fw.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/metal/ops/rmsnorm_fw/device/rmsnorm_fw_device_operation_types.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/metal/ops/rmsnorm_fw/device/rmsnorm_fw_device_operation.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/metal/ops/rmsnorm_fw/device/rmsnorm_fw_device_operation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/metal/ops/rmsnorm_fw/device/rmsnorm_fw_program_factory.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/metal/ops/rmsnorm_fw/device/rmsnorm_fw_program_factory.cpp
    # RMSNorm Backward
    ${CMAKE_CURRENT_SOURCE_DIR}/metal/ops/rmsnorm_bw/rmsnorm_bw.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/metal/ops/rmsnorm_bw/rmsnorm_bw.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/metal/ops/rmsnorm_bw/device/rmsnorm_bw_device_operation_types.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/metal/ops/rmsnorm_bw/device/rmsnorm_bw_device_operation.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/metal/ops/rmsnorm_bw/device/rmsnorm_bw_device_operation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/metal/ops/rmsnorm_bw/device/rmsnorm_bw_program_factory.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/metal/ops/rmsnorm_bw/device/rmsnorm_bw_program_factory.cpp
    # LayerNorm Backward
    ${CMAKE_CURRENT_SOURCE_DIR}/metal/ops/layernorm_bw/layernorm_bw.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/metal/ops/layernorm_bw/layernorm_bw.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/metal/ops/layernorm_bw/device/layernorm_bw_device_operation_types.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/metal/ops/layernorm_bw/device/layernorm_bw_device_operation.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/metal/ops/layernorm_bw/device/layernorm_bw_device_operation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/metal/ops/layernorm_bw/device/layernorm_bw_program_factory.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/metal/ops/layernorm_bw/device/layernorm_bw_program_factory.cpp
    # LayerNorm Forward
    ${CMAKE_CURRENT_SOURCE_DIR}/metal/ops/layernorm_fw/layernorm_fw.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/metal/ops/layernorm_fw/layernorm_fw.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/metal/ops/layernorm_fw/device/layernorm_fw_device_operation_types.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/metal/ops/layernorm_fw/device/layernorm_fw_device_operation.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/metal/ops/layernorm_fw/device/layernorm_fw_device_operation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/metal/ops/layernorm_fw/device/layernorm_fw_program_factory.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/metal/ops/layernorm_fw/device/layernorm_fw_program_factory.cpp
    # CrossEntropy
    ${CMAKE_CURRENT_SOURCE_DIR}/metal/ops/cross_entropy_fw/cross_entropy_fw.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/metal/ops/cross_entropy_fw/cross_entropy_fw.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/metal/ops/cross_entropy_fw/device/cross_entropy_fw_device_operation_types.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/metal/ops/cross_entropy_fw/device/cross_entropy_fw_device_operation.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/metal/ops/cross_entropy_fw/device/cross_entropy_fw_device_operation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/metal/ops/cross_entropy_fw/device/cross_entropy_fw_program_factory.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/metal/ops/cross_entropy_fw/device/cross_entropy_fw_program_factory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/metal/ops/cross_entropy_bw/cross_entropy_bw.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/metal/ops/cross_entropy_bw/cross_entropy_bw.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/metal/ops/cross_entropy_bw/device/cross_entropy_bw_device_operation_types.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/metal/ops/cross_entropy_bw/device/cross_entropy_bw_device_operation.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/metal/ops/cross_entropy_bw/device/cross_entropy_bw_device_operation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/metal/ops/cross_entropy_bw/device/cross_entropy_bw_program_factory.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/metal/ops/cross_entropy_bw/device/cross_entropy_bw_program_factory.cpp
    # Softmax
    ${CMAKE_CURRENT_SOURCE_DIR}/metal/ops/softmax/softmax.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/metal/ops/softmax/softmax.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/metal/ops/softmax/device/softmax_device_operation_types.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/metal/ops/softmax/device/softmax_device_operation.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/metal/ops/softmax/device/softmax_device_operation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/metal/ops/softmax/device/softmax_program_factory.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/metal/ops/softmax/device/softmax_program_factory.cpp
    # ProfilerNoOp
    ${CMAKE_CURRENT_SOURCE_DIR}/metal/ops/profiler_no_op/profiler_no_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/metal/ops/profiler_no_op/profiler_no_op.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/metal/ops/profiler_no_op/device/profiler_no_op_device_operation_types.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/metal/ops/profiler_no_op/device/profiler_no_op_device_operation.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/metal/ops/profiler_no_op/device/profiler_no_op_device_operation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/metal/ops/profiler_no_op/device/profiler_no_op_program_factory.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/metal/ops/profiler_no_op/device/profiler_no_op_program_factory.cpp
    # SiLU Backward
    ${CMAKE_CURRENT_SOURCE_DIR}/metal/ops/silu_bw/silu_bw.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/metal/ops/silu_bw/silu_bw.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/metal/ops/silu_bw/device/silu_bw_device_operation_types.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/metal/ops/silu_bw/device/silu_bw_device_operation.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/metal/ops/silu_bw/device/silu_bw_device_operation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/metal/ops/silu_bw/device/silu_bw_program_factory.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/metal/ops/silu_bw/device/silu_bw_program_factory.cpp
    # SDPA Forward
    ${CMAKE_CURRENT_SOURCE_DIR}/metal/ops/sdpa_fw/sdpa_fw.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/metal/ops/sdpa_fw/sdpa_fw.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/metal/ops/sdpa_fw/device/sdpa_fw_device_operation_types.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/metal/ops/sdpa_fw/device/sdpa_fw_device_operation.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/metal/ops/sdpa_fw/device/sdpa_fw_device_operation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/metal/ops/sdpa_fw/device/sdpa_fw_program_factory.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/metal/ops/sdpa_fw/device/sdpa_fw_program_factory.cpp
    # SDPA Backward
    ${CMAKE_CURRENT_SOURCE_DIR}/metal/ops/sdpa_bw/sdpa_bw.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/metal/ops/sdpa_bw/sdpa_bw.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/metal/ops/sdpa_bw/device/sdpa_bw_kv_device_operation_types.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/metal/ops/sdpa_bw/device/sdpa_bw_kv_device_operation.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/metal/ops/sdpa_bw/device/sdpa_bw_kv_device_operation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/metal/ops/sdpa_bw/device/sdpa_bw_kv_program_factory.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/metal/ops/sdpa_bw/device/sdpa_bw_kv_program_factory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/metal/ops/sdpa_bw/device/sdpa_bw_q_device_operation_types.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/metal/ops/sdpa_bw/device/sdpa_bw_q_device_operation.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/metal/ops/sdpa_bw/device/sdpa_bw_q_device_operation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/metal/ops/sdpa_bw/device/sdpa_bw_q_program_factory.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/metal/ops/sdpa_bw/device/sdpa_bw_q_program_factory.cpp
    # SGD Fused Optimizer
    ${CMAKE_CURRENT_SOURCE_DIR}/metal/optimizers/sgd_fused/sgd_fused.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/metal/optimizers/sgd_fused/sgd_fused.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/metal/optimizers/sgd_fused/device/sgd_fused_device_operation_types.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/metal/optimizers/sgd_fused/device/sgd_fused_device_operation.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/metal/optimizers/sgd_fused/device/sgd_fused_device_operation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/metal/optimizers/sgd_fused/device/sgd_fused_program_factory.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/metal/optimizers/sgd_fused/device/sgd_fused_program_factory.cpp
    # SwiGLU Forward
    ${CMAKE_CURRENT_SOURCE_DIR}/metal/ops/swiglu_fw/swiglu_fw.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/metal/ops/swiglu_fw/swiglu_fw.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/metal/ops/swiglu_fw/device/swiglu_fw_device_operation_types.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/metal/ops/swiglu_fw/device/swiglu_fw_device_operation.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/metal/ops/swiglu_fw/device/swiglu_fw_device_operation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/metal/ops/swiglu_fw/device/swiglu_fw_program_factory.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/metal/ops/swiglu_fw/device/swiglu_fw_program_factory.cpp
)

list(APPEND SOURCES ${METAL_OPS_FILES})

# Check if TT::Metalium target exists
# If it does not exist, assume that we are building with tt-train as top level project
if(NOT TARGET TT::Metalium)
    find_package(TT-NN)

    if(NOT TT-NN_FOUND)
        if("$ENV{TT_METAL_HOME}" STREQUAL "")
            message(FATAL_ERROR "TT_METAL_HOME is not set")
        endif()

        set(METALIUM_INCLUDE_DIRS
            # Metalium
            "$ENV{TT_METAL_HOME}"
            "$ENV{TT_METAL_HOME}/tt_metal"
            "$ENV{TT_METAL_HOME}/tt_metal/third_party/umd"
            "$ENV{TT_METAL_HOME}/tt_metal/third_party/tracy/public"
            "$ENV{TT_METAL_HOME}/tt_metal/hw/inc/internal/tt-1xx/wormhole"
            "$ENV{TT_METAL_HOME}/tt_metal/hw/inc/internal/tt-1xx/wormhole/wormhole_b0_defines"
            "$ENV{TT_METAL_HOME}/tt_metal/hw/inc/internal/tt-1xx/blackhole"
            "$ENV{TT_METAL_HOME}/tt_metal/api/"
            "$ENV{TT_METAL_HOME}/tt_metal/third_party/umd/device/api"
            "$ENV{TT_METAL_HOME}/tt_metal/hostdevcommon/api"
            "$ENV{TT_METAL_HOME}/tt_metal/include"
            "$ENV{TT_METAL_HOME}/tt_stl"
            # TTNN
            "$ENV{TT_METAL_HOME}/ttnn"
            "$ENV{TT_METAL_HOME}/ttnn/api"
            "$ENV{TT_METAL_HOME}/ttnn/cpp"
            "$ENV{TT_METAL_HOME}/ttnn/cpp/ttnn/deprecated"
            "${reflect_SOURCE_DIR}"
        )

        message(STATUS "Metalium not found, attempting to locate")

        # Define the path to look for the library
        set(METALIUM_LIB_PATH "$ENV{TT_METAL_HOME}/build/lib")

        # Try to find the library
        find_library(TT_METAL_LIBRARY NAMES "tt_metal" PATHS "${METALIUM_LIB_PATH}" NO_DEFAULT_PATH)
        find_library(TTNNCPP_LIBRARY NAMES "_ttnncpp.so" PATHS "${METALIUM_LIB_PATH}" NO_DEFAULT_PATH)
        find_library(DEVICE_LIBRARY NAMES "device" PATHS "${METALIUM_LIB_PATH}" NO_DEFAULT_PATH)
        find_library(TT_STL_LIBRARY NAMES "libtt_stl.so" PATHS "${METALIUM_LIB_PATH}" NO_DEFAULT_PATH)

        if(TT_STL_LIBRARY)
            add_library(Metalium::TT_STL SHARED IMPORTED)
            set_target_properties(
                Metalium::TT_STL
                PROPERTIES
                    IMPORTED_LOCATION
                        "${TT_STL_LIBRARY}"
                    INTERFACE_INCLUDE_DIRECTORIES
                        "$ENV{TT_METAL_HOME}/tt_stl"
            )
            message(STATUS "Successfully found libtt_stl.so at ${TT_STL_LIBRARY}")
        else()
            message(FATAL_ERROR "libtt_stl.so not found in ${METALIUM_LIB_PATH}")
        endif()

        if(TT_METAL_LIBRARY)
            add_library(TT::Metalium SHARED IMPORTED)
            set_target_properties(
                TT::Metalium
                PROPERTIES
                    IMPORTED_LOCATION
                        "${TT_METAL_LIBRARY}"
                    INTERFACE_INCLUDE_DIRECTORIES
                        "${METALIUM_INCLUDE_DIRS}"
            )
            target_link_libraries(
                TT::Metalium
                INTERFACE
                    ${DEVICE_LIBRARY}
                    Metalium::TT_STL
                    nlohmann_json::nlohmann_json
            )
            message(STATUS "Successfully found libtt_metal.so at ${TT_METAL_LIBRARY}")
        else()
            message(FATAL_ERROR "libtt_metal.so not found in ${METALIUM_LIB_PATH}")
        endif()
        if(TTNNCPP_LIBRARY)
            add_library(TTNN::TTNN SHARED IMPORTED)
            set_target_properties(
                TTNN::TTNN
                PROPERTIES
                    IMPORTED_LOCATION
                        "${TTNNCPP_LIBRARY}"
                    INTERFACE_INCLUDE_DIRECTORIES
                        "${METALIUM_INCLUDE_DIRS}"
            )
            target_link_libraries(TTNN::TTNN INTERFACE TT::Metalium)
            message(STATUS "Successfully found _ttnncpp.so at ${TTNNCPP_LIBRARY}")
        else()
            message(FATAL_ERROR "_ttnncpp.so not found in ${METALIUM_LIB_PATH}")
        endif()
    endif()
else()
    message(STATUS "Metalium targets already exists")
endif()

# Include helper functions for flatbuffer generation
include(flatbuffers)

add_library(ttml STATIC ${SOURCES})

target_include_directories(ttml PUBLIC ${PROJECT_SOURCE_DIR})

find_package(
    Python3
    COMPONENTS
        Development
        Development.Module
        Interpreter
    REQUIRED
)

foreach(lib ${BoostPackages})
    target_include_directories(ttml SYSTEM PUBLIC ${Boost${lib}_SOURCE_DIR}/include)
endforeach()

# Define FLATBUFFERS_LARGE_SIZE to use uint64_t for size calculations (must be defined before any flatbuffers headers)
target_compile_definitions(ttml PUBLIC FLATBUFFERS_LARGE_SIZE)

target_link_libraries(
    ttml
    PUBLIC
        dl
        z
        pthread
        atomic
        TT::Metalium
        TTNN::TTNN
        Python3::Python
        fmt::fmt-header-only
        enchantum::enchantum
        yaml-cpp::yaml-cpp
        xtensor
        xtensor-blas
        xtl
        Boost::core
        Boost::container
        tt-logger
        simde::simde
        FlatBuffers::FlatBuffers
)

# Add include directory for TTNN::Core's generated flatbuffer headers
# These are needed because tensor_flatbuffer.hpp includes tensor_generated.h
if(TARGET TTNN::Core)
    # Building with tt-metal from source - get binary directory
    get_target_property(TTNN_CORE_BINARY_DIR TTNN::Core BINARY_DIR)
    if(TTNN_CORE_BINARY_DIR)
        target_include_directories(ttml SYSTEM PRIVATE ${TTNN_CORE_BINARY_DIR}/flatbuffers)
        message(STATUS "Added TTNN::Core flatbuffers directory: ${TTNN_CORE_BINARY_DIR}/flatbuffers")
    endif()
else()
    # Building with installed tt-metal
    # Try to get include directories from TTNN::TTNN which should include the flatbuffers path
    # If the generated headers are installed, they should be in the include directories
    get_target_property(TTNN_INCLUDES TTNN::TTNN INTERFACE_INCLUDE_DIRECTORIES)
    if(TTNN_INCLUDES)
        # Check if any of the include directories contain flatbuffers
        # This is a fallback - ideally the headers should be in a standard location
        message(STATUS "TTNN::Core not found, checking TTNN::TTNN include directories for flatbuffers")
        # Note: If headers are not found, this may indicate an installation issue with tt-metal
    endif()
endif()

# Generate flatbuffer header for ttml_metadata.fbs
# Must be done after target_link_libraries so ttml has access to TTNN::TTNN's include directories
set(FLATBUFFERS_SCHEMA ${CMAKE_CURRENT_SOURCE_DIR}/serialization/ttml_metadata.fbs)
set(FLATBUFFER_GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/serialization)

GENERATE_FBS_HEADER(${FLATBUFFERS_SCHEMA} TARGET ttml OUTPUT_DIR ${FLATBUFFER_GENERATED_DIR})

# Add generated header to target sources and include directory
target_sources(ttml PRIVATE ${FBS_GENERATED_HEADER_FILE})
target_include_directories(ttml PUBLIC ${FLATBUFFER_GENERATED_DIR})

message(STATUS "xtensor_SOURCE_DIR: ${xtensor_SOURCE_DIR}")
message(STATUS "xtl_SOURCE_DIR: ${xtl_SOURCE_DIR}")
message(STATUS "xtensor-blas_SOURCE_DIR: ${xtensor-blas_SOURCE_DIR}")

message(STATUS "cli11_SOURCE_DIR: ${CLI11_SOURCE_DIR}")
target_include_directories(ttml PUBLIC ${CLI11_SOURCE_DIR}/include)

if(ENABLE_LIBCXX)
    target_link_libraries(
        PUBLIC
            ${LIBC++}
            ${LIBC++ABI}
    )
endif()

add_definitions(-DTOKENIZERS_DATA_PATH="${CMAKE_SOURCE_DIR}/data")

set(GPT2_TOKENIZER_URL "https://huggingface.co/openai-community/gpt2/resolve/main/tokenizer.json")
set(GPT2_TOKENIZER_FILE "${CMAKE_SOURCE_DIR}/data/gpt2-tokenizer.json")

# Check if the file already exists before downloading
if(NOT EXISTS "${GPT2_TOKENIZER_FILE}")
    message(STATUS "Downloading Tokenizer text file to ${GPT2_TOKENIZER_FILE}")
    file(
        DOWNLOAD
            ${GPT2_TOKENIZER_URL}
            ${GPT2_TOKENIZER_FILE}
        SHOW_PROGRESS
    )
else()
    message(STATUS "Tokenizer text file already exists at ${GPT2_TOKENIZER_FILE}, skipping download.")
endif()

set(TINYLLAMA_TOKENIZER_URL "https://huggingface.co/TinyLlama/TinyLlama_v1.1/resolve/main/tokenizer.json")
set(TINYLLAMA_TOKENIZER_FILE "${CMAKE_SOURCE_DIR}/data/tinyllama-tokenizer.json")

# Check if the file already exists before downloading
if(NOT EXISTS "${TINYLLAMA_TOKENIZER_FILE}")
    message(STATUS "Downloading Tokenizer text file to ${TINYLLAMA_TOKENIZER_FILE}")
    file(
        DOWNLOAD
            ${TINYLLAMA_TOKENIZER_URL}
            ${TINYLLAMA_TOKENIZER_FILE}
        SHOW_PROGRESS
    )
else()
    message(STATUS "Tokenizer text file already exists at ${TINYLLAMA_TOKENIZER_FILE}, skipping download.")
endif()

set(TTML_SRC_NANOBIND
    ${CMAKE_CURRENT_SOURCE_DIR}/nanobind/__init__.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nanobind/nb_autograd.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nanobind/nb_autograd.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nanobind/nb_export_enum.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nanobind/nb_fwd.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nanobind/nb_models.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nanobind/nb_models.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nanobind/nb_modules.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nanobind/nb_modules.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nanobind/nb_ops.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nanobind/nb_ops.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nanobind/nb_core.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nanobind/nb_core.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nanobind/nb_optimizers.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nanobind/nb_optimizers.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nanobind/nb_util.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nanobind/nb_util.hpp
)
nanobind_add_module(
    _ttml
    STABLE_ABI NB_STATIC
    ${TTML_SRC_NANOBIND}
)
target_link_libraries(
    _ttml
    PRIVATE
        fmt::fmt-header-only
        TT::Metalium
        TTNN::TTNN
        Python3::Python
        ttml
)
target_include_directories(_ttml PUBLIC ${PROJECT_SOURCE_DIR})
install(
    TARGETS
        _ttml
    LIBRARY
        DESTINATION ttml
        COMPONENT ttml
)

# Install Python source files
install(
    DIRECTORY
        ${CMAKE_CURRENT_SOURCE_DIR}/ttml
    DESTINATION ttml
    COMPONENT ttml
    FILES_MATCHING
    PATTERN
    "*.py"
    PATTERN
    "__pycache__"
        EXCLUDE
)

# CPack configuration for creating ttml deb package
set(CPACK_GENERATOR "DEB")
set(CPACK_PACKAGE_NAME "ttml")
set(CPACK_PACKAGE_VERSION "1.0.0")
set(CPACK_PACKAGE_VENDOR "Tenstorrent")
set(CPACK_PACKAGE_DESCRIPTION "TT Machine Learning Library - Training framework for Tenstorrent hardware")
set(CPACK_PACKAGE_CONTACT "support@tenstorrent.com")
set(CPACK_DEBIAN_PACKAGE_MAINTAINER "Tenstorrent")

# Component-based packaging
set(CPACK_DEB_COMPONENT_INSTALL ON)
set(CPACK_COMPONENTS_ALL ttml)

# Debian-specific settings
set(CPACK_DEBIAN_TTML_PACKAGE_NAME "ttml")
set(CPACK_DEBIAN_TTML_FILE_NAME "ttml-${CPACK_PACKAGE_VERSION}-Linux.deb")

# Set the package architecture
set(CPACK_DEBIAN_PACKAGE_ARCHITECTURE "amd64")

include(CPack)
