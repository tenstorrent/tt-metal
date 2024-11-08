# Place a no-op .clang-tidy file at the root of the CPM cache for good measure
# All of our dependencies added there will have clang-tidy disabled, but
# submodules may call CPMAddPackage without first disabling clang-tidy.
configure_file(
    "${CMAKE_CURRENT_LIST_DIR}/clang-tidy.d/.clang-tidy.disable.yaml"
    "${PROJECT_SOURCE_DIR}/.cpmcache/.clang-tidy"
    COPYONLY
)

# Disable clang-tidy for all generated files; we typically don't have control over them.
configure_file(
    "${CMAKE_CURRENT_LIST_DIR}/clang-tidy.d/.clang-tidy.disable.yaml"
    "${PROJECT_BINARY_DIR}/.clang-tidy"
    COPYONLY
)
