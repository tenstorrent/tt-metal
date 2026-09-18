# Shared *internal* build requirements of Metalium.
#
# Metalium's public API is carried by TT::Metalium.  Its implementation, and the
# internal tests/tools/benchmarks that are deliberately coupled to that
# implementation, additionally need the private header roots and the private
# third-party dependencies declared here.
#
# Consume it with PRIVATE linkage, alongside TT::Metalium:
#
#     target_link_libraries(my_internal_test PRIVATE TT::Metalium TT::Metalium::Private)
#
# Declaring the requirements explicitly (rather than copying
# $<TARGET_PROPERTY:TT::Metalium,INCLUDE_DIRECTORIES>) keeps each dependency's
# own usage requirements intact - most importantly the SYSTEM classification of
# third-party headers, which a bare list of directories does not carry.
#
# This target is build-tree only: it is never installed or exported, and nothing
# here becomes part of Metalium's public interface.

add_library(metalium_private INTERFACE)
add_library(TT::Metalium::Private ALIAS metalium_private)

# Header roots of the implementation.  These mirror the include directories the
# implementation's own compilation targets declare, so internal consumers resolve
# private headers exactly as the implementation does.
target_include_directories(
    metalium_private
    INTERFACE
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/tt_metal
        ${PROJECT_SOURCE_DIR}/tt_metal/api
        ${PROJECT_SOURCE_DIR}/tt_metal/api/tt-metalium
        ${PROJECT_SOURCE_DIR}/tt_metal/common
        ${PROJECT_SOURCE_DIR}/tt_metal/impl
        ${PROJECT_SOURCE_DIR}/tt_metal/impl/debug
        ${PROJECT_SOURCE_DIR}/tt_metal/impl/profiler
        ${PROJECT_SOURCE_DIR}/tt_metal/llrt
        ${PROJECT_SOURCE_DIR}/tt_metal/fabric
        ${PROJECT_SOURCE_DIR}/tt_metal/hw/inc
        ${PROJECT_SOURCE_DIR}/tools/scaleout
        # Generated headers (FlatBuffers schemas of the public API and of impl).
        ${PROJECT_BINARY_DIR}/tt_metal/api
        ${PROJECT_BINARY_DIR}/tt_metal/impl
)

# Cap'n Proto generates these; they are not maintained here, so they keep the
# SYSTEM treatment impl gives them.
target_include_directories(
    metalium_private
    SYSTEM
    INTERFACE
        ${PROJECT_BINARY_DIR}/tt_metal/impl/internal/disaggregation
)

# Third-party dependencies reachable from private headers.  Linking the targets
# (instead of restating their directories) preserves their SYSTEM include
# classification and any compile definitions/options they carry.
target_link_libraries(
    metalium_private
    INTERFACE
        simde::simde # tt_metal/impl/streaming_profiler
        Taskflow::Taskflow # tt_metal/common/executor.hpp
        FlatBuffers::FlatBuffers # tt_metal/impl/lightmetal
)
