# Single source of truth for the layer-completion implementation .cpp files, shared by every build
# target that compiles them: the `_layer_completion` python module (CMakeLists.txt in this dir) and the
# C++ unit / MPI tests under tests/tt_metal. Each consumer `include()`s this file and compiles
# ${LAYER_COMPLETION_IMPL_SRCS} into its own target (with its own include dirs), so the list is
# maintained in exactly one place — add/rename a source here and all consumers pick it up.
#
# LAYER_COMPLETION_DIR is also exported so consumers can put the feature dir (headers) on their include
# path. CMAKE_CURRENT_LIST_DIR resolves to this file's directory regardless of who includes it.
set(LAYER_COMPLETION_DIR ${CMAKE_CURRENT_LIST_DIR})
set(LAYER_COMPLETION_IMPL_SRCS
    ${LAYER_COMPLETION_DIR}/layer_completion_queue.cpp
    ${LAYER_COMPLETION_DIR}/layer_completion_reorder_buffer.cpp
    ${LAYER_COMPLETION_DIR}/layer_completion_router.cpp
)
