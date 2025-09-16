project(ttml)

set(SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/autograd/auto_context.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/autograd/auto_context.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/autograd/autocast_tensor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/autograd/autocast_tensor.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/autograd/graph.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/autograd/graph.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/autograd/graph_utils.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/autograd/module_base.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/autograd/module_base.hpp
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
    ${CMAKE_CURRENT_SOURCE_DIR}/core/mesh_device.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/core/mesh_device.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/core/not_null.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/core/random.hpp
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
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/unary_ops.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/unary_ops.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/optimizers/adamw.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/optimizers/adamw.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/optimizers/optimizer_base.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/optimizers/optimizer_base.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/optimizers/sgd.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/optimizers/sgd.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/optimizers/no_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/optimizers/no_op.hpp
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
    ${CMAKE_CURRENT_SOURCE_DIR}/serialization/msgpack_file.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/serialization/msgpack_file.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/serialization/safetensors.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/serialization/safetensors.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/serialization/serializable.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/serialization/serialization.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/serialization/serialization.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tokenizers/bpe_tokenizer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tokenizers/bpe_tokenizer.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tokenizers/char_tokenizer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tokenizers/char_tokenizer.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tokenizers/char_tokenizer_trainer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tokenizers/char_tokenizer_trainer.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tokenizers/tokenizer_base.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ttnn_fixed/distributed/ttnn_ops.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ttnn_fixed/distributed/ttnn_ops.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ttnn_fixed/matmuls.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ttnn_fixed/matmuls.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ttnn_fixed/trivial_ttnn_ops.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ttnn_fixed/trivial_ttnn_ops.hpp
)

# Manually select only specific files from metal ops
set(METAL_OPS_FILES
    # General
    ${CMAKE_CURRENT_SOURCE_DIR}/metal/operations.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/metal/ttnn_all_includes.hpp
    # Common
    ${CMAKE_CURRENT_SOURCE_DIR}/metal/ops/common/program_utils.hpp
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
)

list(APPEND SOURCES ${METAL_OPS_FILES})

# Check if Metalium::Metal target exists
# If it does not exist, assume that we are building with tt-train as top level project
if(NOT TARGET Metalium::Metal)
    if("$ENV{TT_METAL_HOME}" STREQUAL "")
        message(FATAL_ERROR "TT_METAL_HOME is not set")
    endif()

    set(METALIUM_INCLUDE_DIRS
        # Metalium
        "$ENV{TT_METAL_HOME}"
        "$ENV{TT_METAL_HOME}/tt_metal"
        "$ENV{TT_METAL_HOME}/tt_metal/third_party/umd"
        "$ENV{TT_METAL_HOME}/tt_metal/third_party/tracy/public"
        "$ENV{TT_METAL_HOME}/tt_metal/hw/inc/wormhole"
        "$ENV{TT_METAL_HOME}/tt_metal/hw/inc/wormhole/wormhole_b0_defines"
        "$ENV{TT_METAL_HOME}/tt_metal/hw/inc/"
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
    find_library(TTNN_LIBRARY NAMES "_ttnn.so" PATHS "${METALIUM_LIB_PATH}" NO_DEFAULT_PATH)
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
        add_library(Metalium::Metal SHARED IMPORTED)
        set_target_properties(
            Metalium::Metal
            PROPERTIES
                IMPORTED_LOCATION
                    "${TT_METAL_LIBRARY}"
                INTERFACE_INCLUDE_DIRECTORIES
                    "${METALIUM_INCLUDE_DIRS}"
        )
        target_link_libraries(
            Metalium::Metal
            INTERFACE
                ${DEVICE_LIBRARY}
                Metalium::TT_STL
                nlohmann_json::nlohmann_json
        )
        message(STATUS "Successfully found libtt_metal.so at ${TT_METAL_LIBRARY}")
    else()
        message(FATAL_ERROR "libtt_metal.so not found in ${METALIUM_LIB_PATH}")
    endif()
    if(TTNN_LIBRARY)
        add_library(Metalium::TTNN SHARED IMPORTED)
        set_target_properties(
            Metalium::TTNN
            PROPERTIES
                IMPORTED_LOCATION
                    "${TTNN_LIBRARY}"
                INTERFACE_INCLUDE_DIRECTORIES
                    "${METALIUM_INCLUDE_DIRS}"
        )
        target_link_libraries(Metalium::TTNN INTERFACE Metalium::TT_STL)
        message(STATUS "Successfully found _ttnn.so at ${TTNN_LIBRARY}")
    else()
        message(FATAL_ERROR "_ttnn.so not found in ${METALIUM_LIB_PATH}")
    endif()
    if(TTNNCPP_LIBRARY)
        add_library(Metalium::TTNNCPP SHARED IMPORTED)
        set_target_properties(
            Metalium::TTNNCPP
            PROPERTIES
                IMPORTED_LOCATION
                    "${TTNNCPP_LIBRARY}"
                INTERFACE_INCLUDE_DIRECTORIES
                    "${METALIUM_INCLUDE_DIRS}"
        )
        message(STATUS "Successfully found _ttnncpp.so at ${TTNNCPP_LIBRARY}")
    else()
        message(FATAL_ERROR "_ttnncpp.so not found in ${METALIUM_LIB_PATH}")
    endif()
else()
    message(STATUS "Metalium targets already exists")
endif()

add_library(ttml STATIC ${SOURCES})

target_include_directories(ttml PUBLIC ${PROJECT_SOURCE_DIR})

find_package(Python3 REQUIRED Development)

foreach(lib ${BoostPackages})
    target_include_directories(ttml SYSTEM PUBLIC ${Boost${lib}_SOURCE_DIR}/include)
endforeach()

target_link_libraries(
    ttml
    PUBLIC
        dl
        z
        pthread
        atomic
        Metalium::Metal
        Metalium::TTNN
        Metalium::TTNNCPP
        Python3::Python
        fmt::fmt-header-only
        enchantum::enchantum
        yaml-cpp::yaml-cpp
        xtensor
        xtensor-blas
        xtl
        tokenizers_cpp
        wandbcpp
        Boost::core
        Boost::container
        tt-logger
)

message(STATUS "xtensor_SOURCE_DIR: ${xtensor_SOURCE_DIR}")
message(STATUS "xtl_SOURCE_DIR: ${xtl_SOURCE_DIR}")
message(STATUS "xtensor-blas_SOURCE_DIR: ${xtensor-blas_SOURCE_DIR}")

message(STATUS "msgpack_SOURCE_DIR: ${msgpack_SOURCE_DIR}")
target_include_directories(ttml PUBLIC ${msgpack_SOURCE_DIR}/include)
message(STATUS "cli11_SOURCE_DIR: ${CLI11_SOURCE_DIR}")
target_include_directories(ttml PUBLIC ${CLI11_SOURCE_DIR}/include)

target_include_directories(ttml PUBLIC ${TOKENZIER_CPP_PATH}/include)

if(ENABLE_LIBCXX)
    target_link_libraries(
        tokenizers_cpp
        PUBLIC
            ${LIBC++}
            ${LIBC++ABI}
    )
    target_compile_options(tokenizers_cpp PUBLIC -stdlib=libc++)

    target_link_libraries(
        wandbcpp
        PUBLIC
            ${LIBC++}
            ${LIBC++ABI}
    )
    target_compile_options(wandbcpp PUBLIC -stdlib=libc++)
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
