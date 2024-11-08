project(ttml)

file(
    GLOB_RECURSE SOURCES
    LIST_DIRECTORIES true
    *.hpp
    *.cpp
) # I am a bad person

add_library(ttml STATIC ${SOURCES})

target_include_directories(
    ttml
    SYSTEM
    PUBLIC
        # Metalium
        $ENV{TT_METAL_HOME}
        $ENV{TT_METAL_HOME}/tt_metal
        $ENV{TT_METAL_HOME}/tt_metal/third_party/umd
        $ENV{TT_METAL_HOME}/tt_metal/hw/inc/wormhole
        $ENV{TT_METAL_HOME}/tt_metal/hw/inc/wormhole/wormhole_b0_defines
        $ENV{TT_METAL_HOME}/tt_metal/hw/inc/
        $ENV{TT_METAL_HOME}/tt_metal/third_party/umd/src/firmware/riscv/wormhole
        $ENV{TT_METAL_HOME}/tt_metal/third_party/umd/device
        # TTNN
        $ENV{TT_METAL_HOME}/ttnn/cpp
        $ENV{TT_METAL_HOME}/ttnn/cpp/ttnn/deprecated
        $ENV{TT_METAL_HOME}/tt_metal/third_party/magic_enum
        ${reflect_SOURCE_DIR}
        ${flatbuffers_include_dir}
        ${nng_include_dir}
)
target_include_directories(ttml PUBLIC ${PROJECT_SOURCE_DIR})
target_link_directories(ttml PUBLIC $ENV{TT_METAL_HOME}/build/lib)
find_package(Python REQUIRED Development)

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
)

target_link_libraries(
    ttml
    PUBLIC
        tt_metal
        pthread
        Python::Python
        $ENV{TT_METAL_HOME}/build/lib/_ttnn.so
        fmt
        magic_enum
)

target_link_libraries(ttml PUBLIC yaml-cpp::yaml-cpp)

target_compile_definitions(ttml PUBLIC FMT_HEADER_ONLY)

target_link_libraries(
    ttml
    PUBLIC
        xtensor
        xtl
)

message(STATUS "xtensor_SOURCE_DIR: ${xtensor_SOURCE_DIR}")
message(STATUS "xtl_SOURCE_DIR: ${xtl_SOURCE_DIR}")

message(STATUS "msgpack_SOURCE_DIR: ${msgpack_SOURCE_DIR}")
target_include_directories(ttml PUBLIC ${msgpack_SOURCE_DIR}/include)
message(STATUS "cli11_SOURCE_DIR: ${CLI11_SOURCE_DIR}")
target_include_directories(ttml PUBLIC ${CLI11_SOURCE_DIR}/include)

target_include_directories(ttml PUBLIC ${TOKENZIER_CPP_PATH}/include)

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

target_link_libraries(
    ttml
    PUBLIC
        tokenizers_cpp
        wandbcpp
)
target_link_libraries(
    ttml
    PUBLIC
        Boost::core
        Boost::container
)

add_definitions(-DTOKENIZERS_DATA_PATH="${CMAKE_CURRENT_SOURCE_DIR}/data/tokenizers")
