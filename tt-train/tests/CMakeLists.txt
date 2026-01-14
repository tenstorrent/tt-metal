include(CTest)
enable_testing()

set(SOURCES
    core/tensor_utils_test.cpp
    core/n300_utils_test.cpp
    core/scoped_test.cpp
    core/clip_grad_norm_test.cpp
    core/random_tests.cpp
    model/linear_regression_ddp_test.cpp
    model/nano_gpt_test.cpp
    model/model_names_test.cpp
    model/gpt2s_test.cpp
    model/linear_regression_full_test.cpp
    model/weight_tying_test.cpp
    ttnn_fixed/concat_op_test.cpp
    ttnn_fixed/distributed/distributed_ttnn_ops_test.cpp
    ttnn_fixed/trivial_ttnn_ops_test.cpp
    ttnn_fixed/matmuls_test.cpp
    ttnn_fixed/reduce_ops_test.cpp
    ttnn_fixed/dropout_op_test.cpp
    ttnn_fixed/slice_op_test.cpp
    autograd/autograd_tensor.cpp
    autograd/autograd_test.cpp
    autograd/module_base_parameters_test.cpp
    schedulers/schedulers_test.cpp
    tokenizers/char_tokenizer_trainer_test.cpp
    tokenizers/char_tokenizer_test.cpp
    3rd_party/xtensor_test.cpp
    datasets/in_memory_token_dataset_test.cpp
    datasets/random_split_test.cpp
    datasets/utils_test.cpp
    datasets/generators_test.cpp
    datasets/dataloader_test.cpp
    optimizers/adamw_test.cpp
    optimizers/sgd_fused_test.cpp
    optimizers/no_op_test.cpp
    modules/distributed/linear_test.cpp
    serialization/model_serializer_test.cpp
    serialization/flatbuffer_serializer_test.cpp
    serialization/tensor_serializer_test.cpp
    serialization/safetensors_test.cpp
    ops/cross_entropy_fw_op_test.cpp
    ops/positional_embedding_test.cpp
    ops/layer_norm_composite_test.cpp
    ops/distributed/comm_ops_test.cpp
    ops/linear_op_test.cpp
    ops/profiler_no_op_test.cpp
    ops/cross_entropy_bw_op_test.cpp
    ops/rope_test.cpp
    ops/rmsnorm_op_test.cpp
    ops/unary_ops_test.cpp
    ops/embedding_op_test.cpp
    ops/softmax_test.cpp
    ops/silu_op_test.cpp
    ops/sdpa_fw_op_test.cpp
    ops/sdpa_bw_op_test.cpp
    ops/layernorm_bw_fused_op_test.cpp
    ops/layernorm_fw_fused_op_test.cpp
    ops/swiglu_op_test.cpp
    utils/memory_utils_test.cpp
)

add_executable(ttml_tests ${SOURCES})
target_link_libraries(
    ttml_tests
    GTest::gmock_main
    ttml
)
add_definitions(-DTEST_DATA_DIR="${CMAKE_SOURCE_DIR}/data")

# Define the target file location
set(TOKENIZER_URL "https://huggingface.co/togethercomputer/RedPajama-INCITE-Chat-3B-v1/resolve/main/tokenizer.json")
set(TOKENIZER_FILE "${CMAKE_SOURCE_DIR}/data/tokenizer.json")

# Check if the file already exists before downloading
if(NOT EXISTS "${TOKENIZER_FILE}")
    message(STATUS "Downloading Tokenizer text file to ${TOKENIZER_FILE}")
    file(
        DOWNLOAD
            ${TOKENIZER_URL}
            ${TOKENIZER_FILE}
        SHOW_PROGRESS
    )
else()
    message(STATUS "Tokenizer text file already exists at ${TOKENIZER_FILE}, skipping download.")
endif()

add_definitions(-DTEST_DATA_DIR="${CMAKE_SOURCE_DIR}/data")
add_definitions(-DTEST_LOCAL_DATA_DIR="${CMAKE_SOURCE_DIR}/tests/local_data")

# Define the target file location
set(SHAKESPEARE_URL "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt")
set(SHAKESPEARE_FILE "${CMAKE_CURRENT_BINARY_DIR}/shakespeare.txt")

# Check if the file already exists before downloading
if(NOT EXISTS "${SHAKESPEARE_FILE}")
    message(STATUS "Downloading Shakespeare text file to ${SHAKESPEARE_FILE}")
    file(
        DOWNLOAD
            ${SHAKESPEARE_URL}
            ${SHAKESPEARE_FILE}
        SHOW_PROGRESS
    )
else()
    message(STATUS "Shakespeare text file already exists at ${SHAKESPEARE_FILE}, skipping download.")
endif()

include(GoogleTest)
gtest_discover_tests(ttml_tests)
