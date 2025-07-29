project(ttml)

# TODO: remove glob
set(SOURCE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/autograd
    ${CMAKE_CURRENT_SOURCE_DIR}/core
    ${CMAKE_CURRENT_SOURCE_DIR}/datasets
    ${CMAKE_CURRENT_SOURCE_DIR}/init
    ${CMAKE_CURRENT_SOURCE_DIR}/models
    ${CMAKE_CURRENT_SOURCE_DIR}/modules
    ${CMAKE_CURRENT_SOURCE_DIR}/ops
    ${CMAKE_CURRENT_SOURCE_DIR}/optimizers
    ${CMAKE_CURRENT_SOURCE_DIR}/schedulers
    ${CMAKE_CURRENT_SOURCE_DIR}/serialization
    ${CMAKE_CURRENT_SOURCE_DIR}/tokenizers
    ${CMAKE_CURRENT_SOURCE_DIR}/ttnn_fixed
)

set(SOURCES "")
foreach(dir ${SOURCE_DIRS})
    file(
        GLOB_RECURSE DIR_SOURCES
        ${dir}/*.cpp
        ${dir}/*.hpp
    )
    list(APPEND SOURCES ${DIR_SOURCES})
endforeach()

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
        magic_enum::magic_enum
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

set(GPT2_TOKENIZER_URL
    "https://huggingface.co/togethercomputer/RedPajama-INCITE-Chat-3B-v1/resolve/main/tokenizer.json"
)
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
