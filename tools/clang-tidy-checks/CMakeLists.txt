cmake_minimum_required(VERSION 3.20)
project(TTAttributeProtocolCheck CXX)

# Find LLVM and Clang
find_package(LLVM REQUIRED CONFIG)
find_package(Clang REQUIRED CONFIG)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake from: ${LLVM_DIR}")
message(STATUS "Using ClangConfig.cmake from: ${Clang_DIR}")

# Build as a shared module that clang-tidy can load via -load
add_library(
    TTAttributeProtocolCheck
    MODULE
    AttributeProtocolCheck.cpp
    Module.cpp
)

target_include_directories(
    TTAttributeProtocolCheck
    PRIVATE
        ${LLVM_INCLUDE_DIRS}
        ${CLANG_INCLUDE_DIRS}
)

target_compile_features(TTAttributeProtocolCheck PRIVATE cxx_std_17)

# On Linux, don't resolve symbols at link time - clang-tidy provides them at runtime
if(UNIX AND NOT APPLE)
    target_link_options(TTAttributeProtocolCheck PRIVATE -Wl,--unresolved-symbols=ignore-all)
endif()

# Don't link against LLVM/Clang libraries - they're provided by clang-tidy at runtime
# Only link system libs
target_compile_definitions(TTAttributeProtocolCheck PRIVATE ${LLVM_DEFINITIONS})
