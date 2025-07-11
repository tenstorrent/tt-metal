set(CMAKE_C_CLANG_TIDY "")
set(CMAKE_CXX_CLANG_TIDY "")
set(CMAKE_VERIFY_INTERFACE_HEADER_SETS FALSE)

add_subdirectory(umd)

if(BUILD_MLP_OP_PERF)
    add_subdirectory(mlp-op-perf)
endif()
