# Provide unique architecture specific translation units and include paths for each ARCH

set(LLRT_SRC
    llrt.cpp
    core_descriptor.cpp
    rtoptions.cpp
    tlb_config.cpp
    tt_cluster.cpp
    tt_elffile.cpp
    tt_memory.cpp
    hal.cpp
    utils.cpp
)

add_library(llrt OBJECT ${LLRT_SRC})
add_library(Metalium::Metal::LLRT ALIAS llrt)
add_subdirectory(hal)
add_dependencies(llrt hal_generate_dev_msgs_header)

target_link_libraries(
    llrt
    PUBLIC
        umd::device
        Metalium::Metal::Hardware
    PRIVATE
        HAL::1xx
        Tracy::TracyClient
        nlohmann_json::nlohmann_json
        Reflect::Reflect
        yaml-cpp::yaml-cpp
        enchantum::enchantum
        TT::Metalium::Common
        tt-logger::tt-logger
)
target_include_directories(llrt PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(llrt PRIVATE -Wno-int-to-pointer-cast)
