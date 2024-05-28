
macro(CHECK_COMPILERS)
    message(STATUS "Checking compilers")
    if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "17.0.0" OR CMAKE_CXX_COMPILER_VERSION GREATER_EQUAL "18.0.0")
            message(WARNING "Only Clang-17 is tested right now")
        endif()
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        message(WARNING
            "\n Recommended to use Clang for better performance"
            "\n Either pass in -DCMAKE_CXX_COMPILER=clang++-17"
            "\n Set env variable CXX=clang++-17"
            "\n Check top level CMakeLists and uncomment some lines\n"
        )
        if(CMAKE_CXX_COMPILER_VERSION GREATER_EQUAL "11.0.0")
            message(WARNING "Anything after GCC-9 has not been thoroughly tested!")
        else()
            message(FATAL_ERROR "Any version lower than GCC-11 will not support necessary C++20 features")
        endif()
    else()
        message(FATAL_ERROR "Compiler is not GCC or Clang")
    endif()
endmacro()

macro(CHECK_COMPILER_WARNINGS)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        target_compile_options(compiler_warnings INTERFACE
            -Wsometimes-uninitialized -Wno-c++11-narrowing -Wno-c++23-extensions -Wno-error=local-type-template-args
            -Wno-delete-non-abstract-non-virtual-dtor -Wno-c99-designator -Wno-shift-op-parentheses -Wno-non-c-typedef-for-linkage
            -Wno-deprecated-this-capture -Wno-deprecated-volatile -Wno-deprecated-builtins -Wno-deprecated-declarations # -> extra C++20 build flags
        )
        # -Wsometimes-uninitialized will override the -Wuninitialized added before
    else() # using GCC-11 or higher
        target_compile_options(compiler_warnings INTERFACE
            -Wno-deprecated -Wno-attributes # <-- C++20 warning flags
        )
    endif()
endmacro()
