# Place a no-op .clang-tidy file at the root of the CPM cache for good measure
# All of our dependencies added there will have clang-tidy disabled, but
# submodules may call CPMAddPackage without first disabling clang-tidy.
configure_file(
    "${CMAKE_CURRENT_LIST_DIR}/clang-tidy.d/.clang-tidy.disable.yaml"
    "${PROJECT_SOURCE_DIR}/.cpmcache/.clang-tidy"
    COPYONLY
)

# Copy the .clang-tidy to the build dir in case it's not a child of the source dir.
# CMAKE_VERIFY_INTERFACE_HEADER_SETS will generate files in the build dir that we want scanned.
configure_file("${PROJECT_SOURCE_DIR}/.clang-tidy" "${PROJECT_BINARY_DIR}/.clang-tidy" COPYONLY)
