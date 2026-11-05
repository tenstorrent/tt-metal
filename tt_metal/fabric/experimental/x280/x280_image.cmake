# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# X280 (L2CPU) firmware images, built at build time with tt-metal's SFPI compiler.
#
# GPP_CMD in tt_metal/hw/CMakeLists.txt is scoped to that directory, so the compiler path is
# re-derived here by the same rule; hw/ has already fetched SFPI at configure time.
if(TT_USE_SYSTEM_SFPI)
    set(_x280_sfpi_bin "/opt/tenstorrent/sfpi/compiler/bin")
else()
    set(_x280_sfpi_bin "${PROJECT_SOURCE_DIR}/runtime/sfpi/compiler/bin")
endif()
foreach(_tool gcc objcopy readelf)
    string(TOUPPER "${_tool}" _var)
    set(X280_${_var} "${_x280_sfpi_bin}/riscv-tt-elf-${_tool}")
    if(NOT EXISTS "${X280_${_var}}")
        message(FATAL_ERROR "x280: SFPI ${_tool} not found at ${X280_${_var}}")
    endif()
endforeach()

# Build-tree directory holding firmware-*.{elf,bin}. The runtime tests compile it in.
set(X280_IMAGE_DIR "${PROJECT_BINARY_DIR}/x280")

# SFPI defaults to an RV32 ABI and refuses to compile without -mcpu, so every invocation names
# the X280 core; -march still selects the ISA it compiles for.
set(_x280_target_flags
    -mcpu=sifive-x280
    -march=rv64gc
    -mabi=lp64d
)
set(X280_CFLAGS
    ${_x280_target_flags}
    -nostdlib
    -nostartfiles
    -ffreestanding
    -O2
    -g
    -Wall
    -Werror
    -mcmodel=medany
    -I${CMAKE_CURRENT_LIST_DIR}/include
)
set(_x280_asflags
    ${_x280_target_flags}
    -nostdlib
    -I${CMAKE_CURRENT_LIST_DIR}/include
)
# Neither per-hart entry object zeroes .bss, so images linking one must carry none. The linker
# scripts assert that when this symbol is defined.
set(_x280_ldflags
    ${_x280_target_flags}
    -nostdlib
    -Wl,--defsym=X280_REQUIRE_NO_BSS=1
)
set(_x280_obj_dir "${X280_IMAGE_DIR}/obj")

add_custom_target(x280_images ALL)

# _x280_compile(<out> <source> <flag>...): one object from one C or assembly source, with header
# dependencies through a depfile.
function(_x280_compile out source)
    add_custom_command(
        OUTPUT
            "${out}"
        COMMAND
            "${X280_GCC}" ${ARGN} -MD -MT "${out}" -MF "${out}.d" -c -o "${out}" "${source}"
        DEPENDS
            "${source}"
        DEPFILE "${out}.d"
        COMMENT "x280: compiling ${source}"
        VERBATIM
        COMMAND_EXPAND_LISTS
    )
endfunction()

# The entry objects every image links: entry_hart.o for resident idle, and for active images
# entry_active.o, which enters through the lifecycle harness x280_active_start.o rather than
# calling main directly.
set(_x280_entry "${CMAKE_CURRENT_LIST_DIR}/boot/entry.S")
_x280_compile("${_x280_obj_dir}/entry_hart.o" "${_x280_entry}" ${_x280_asflags} -DX280_PER_HART_PROTOCOL)
# X280_ACTIVE_HARNESS implies X280_PER_HART_PROTOCOL: the harness publishes into the per-hart
# records the trap path also indexes.
_x280_compile("${_x280_obj_dir}/entry_active.o" "${_x280_entry}" ${_x280_asflags} -DX280_PER_HART_PROTOCOL
    -DX280_ACTIVE_HARNESS
)
_x280_compile(
    "${_x280_obj_dir}/x280_active_start.o"
    "${CMAKE_CURRENT_LIST_DIR}/src/x280_active_start.c"
    ${X280_CFLAGS}
)

#[[
x280_add_image(NAME <stem> SOURCES <c-source>... [LIM_IDLE] [DEFINES <macro>...])

Builds ${X280_IMAGE_DIR}/<stem>.elf and <stem>.bin from C sources, and adds them to x280_images.
An image links the active lifecycle harness and ld/x280.ld (.text at 0x08001000), or with
LIM_IDLE the per-hart entry and ld/x280-lim-idle.ld (.text at 0x08000000). DEFINES apply to
this image's sources only.
#]]
function(x280_add_image)
    cmake_parse_arguments(PARSE_ARGV 0 ARG "LIM_IDLE" "NAME" "SOURCES;DEFINES")
    if(NOT ARG_NAME OR NOT ARG_SOURCES OR ARG_UNPARSED_ARGUMENTS)
        message(FATAL_ERROR "x280_add_image: needs NAME and SOURCES; unparsed: '${ARG_UNPARSED_ARGUMENTS}'")
    endif()
    if(ARG_LIM_IDLE)
        set(entry "${_x280_obj_dir}/entry_hart.o")
        set(script "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/ld/x280-lim-idle.ld")
    else()
        set(entry
            "${_x280_obj_dir}/entry_active.o"
            "${_x280_obj_dir}/x280_active_start.o"
        )
        set(script "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/ld/x280.ld")
    endif()
    list(TRANSFORM ARG_DEFINES PREPEND "-D")

    set(objects "")
    foreach(source IN LISTS ARG_SOURCES)
        cmake_path(ABSOLUTE_PATH source BASE_DIRECTORY "${CMAKE_CURRENT_FUNCTION_LIST_DIR}")
        cmake_path(GET source STEM stem)
        set(object "${_x280_obj_dir}/${ARG_NAME}/${stem}.o")
        _x280_compile("${object}" "${source}" ${X280_CFLAGS} ${ARG_DEFINES})
        list(APPEND objects "${object}")
    endforeach()

    set(elf "${X280_IMAGE_DIR}/${ARG_NAME}.elf")
    set(bin "${X280_IMAGE_DIR}/${ARG_NAME}.bin")
    add_custom_command(
        OUTPUT
            "${elf}"
        COMMAND
            "${X280_GCC}" ${_x280_ldflags} -T "${script}" -o "${elf}" ${entry} ${objects}
        DEPENDS
            ${entry}
            ${objects}
            "${script}"
        COMMENT "x280: linking ${ARG_NAME}.elf"
        VERBATIM
        COMMAND_EXPAND_LISTS
    )
    add_custom_command(
        OUTPUT
            "${bin}"
        COMMAND
            "${X280_OBJCOPY}" -O binary "${elf}" "${bin}"
        DEPENDS
            "${elf}"
        VERBATIM
    )
    add_custom_target(x280_image_${ARG_NAME} DEPENDS "${bin}")
    add_dependencies(x280_images x280_image_${ARG_NAME})
endfunction()
