# Shared *internal* build requirements of Metalium.
#
# Metalium's public API is carried by TT::Metalium.  Its internal
# tests/tools/benchmarks are deliberately coupled to the implementation and
# additionally need the implementation's private headers and the private
# third-party dependencies behind them.
#
# Consume it with PRIVATE linkage, alongside TT::Metalium:
#
#     target_link_libraries(my_internal_test PRIVATE TT::Metalium TT::Metalium::Private)
#
# This is an aggregate: every requirement comes from the component that owns it,
# by linking that component's target.  Nothing is restated here as a path, so a
# component that changes its header layout or its dependencies keeps its
# consumers correct automatically.  Linking targets (rather than copying
# $<TARGET_PROPERTY:TT::Metalium,INCLUDE_DIRECTORIES>) also keeps the metadata a
# bare directory list drops - most importantly the SYSTEM classification of
# third-party headers.
#
# Because the components below are in this target's dependency graph, none of
# them may link back to it.
#
# This target is build-tree only: it is never installed or exported, and nothing
# here becomes part of Metalium's public interface.

add_library(metalium_private INTERFACE)
add_library(TT::Metalium::Private ALIAS metalium_private)

target_link_libraries(
    metalium_private
    INTERFACE
        # The implementation components whose headers internal consumers include.
        # These are OBJECT libraries; reached through this interface they
        # propagate their usage requirements but not their object files.
        TT::Metalium::Common
        Metalium::Metal::Impl
        Metalium::Metal::LLRT
        TT::Metalium::Fabric
        TT::ScaleoutTools
        # impl links FlatBuffers PRIVATE, but the lightmetal headers internal
        # consumers include pull in <flatbuffers/flatbuffers.h>.
        FlatBuffers::FlatBuffers
)
