set(CMAKE_C_CLANG_TIDY "")
set(CMAKE_CXX_CLANG_TIDY "")
set(CMAKE_VERIFY_INTERFACE_HEADER_SETS FALSE)
add_compile_options(-Wno-unused-variable)

add_subdirectory(umd)
