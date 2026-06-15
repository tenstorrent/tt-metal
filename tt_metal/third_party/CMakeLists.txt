set(CMAKE_C_CLANG_TIDY "")
set(CMAKE_CXX_CLANG_TIDY "")
set(CMAKE_VERIFY_INTERFACE_HEADER_SETS FALSE)
set(TT_UMD_ENABLE_CLANG_TIDY OFF CACHE BOOL "Disable clang-tidy checks for UMD" FORCE)

if(TT_METAL_USE_EMULE)
    set(TT_UMD_BUILD_EMULE ON CACHE BOOL "Enable SWEmuleChip (forwarded from TT_METAL_USE_EMULE)" FORCE)
endif()

add_subdirectory(umd)
