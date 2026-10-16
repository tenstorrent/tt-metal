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
foreach(_tool gcc g++ objcopy readelf)
    string(TOUPPER "${_tool}" _var)
    string(REPLACE "++" "XX" _var "${_var}")
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
    -fno-zero-initialized-in-bss
    -I${CMAKE_CURRENT_LIST_DIR}/include
)
# Every C++ compile adds these: image sources, suite-header checks and refusal checks.
set(_x280_cxxflags
    -std=c++17
    -fno-exceptions
    -fno-rtti
)
set(_x280_asflags
    ${_x280_target_flags}
    -nostdlib
    -I${CMAKE_CURRENT_LIST_DIR}/include
)
# Neither per-hart entry object zeroes .bss, so images linking one must carry none: the compile
# flags keep zero-initialised statics in .data, and the linker scripts assert an empty .bss when
# this symbol is defined.
set(_x280_ldflags
    ${_x280_target_flags}
    -nostdlib
    -Wl,--defsym=X280_REQUIRE_NO_BSS=1
)
set(_x280_obj_dir "${X280_IMAGE_DIR}/obj")

add_custom_target(x280_images ALL)

# _x280_compile(<out> <source> <flag>...): one object from one C, C++ (.cpp, compiled by g++ with
# _x280_cxxflags) or assembly source, with header dependencies through a depfile.
function(_x280_compile out source)
    set(compiler "${X280_GCC}")
    set(language_flags "")
    if(source MATCHES "\\.cpp$")
        set(compiler "${X280_GXX}")
        set(language_flags ${_x280_cxxflags})
    endif()
    add_custom_command(
        OUTPUT
            "${out}"
        COMMAND
            "${compiler}" ${ARGN} ${language_flags} -MD -MT "${out}" -MF "${out}.d" -c -o "${out}" "${source}"
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
# Where an active image runs, and what its entry writes to each hart's L2 prefetcher:
#   lim   ld/x280.ld; nothing.
#   gddr  ld/x280-gddr.ld, the cached local GDDR; the recommended values (tt-isa-documentation
#         BlackholeA0/L2CPUTile/Caches.md § L2 Cache).
# x280_add_image's PLACEMENT picks one per image; X280_IMAGE_PLACEMENT is the default for images
# that name none. Resident idle is in LIM regardless.
set(X280_IMAGE_PLACEMENT "lim" CACHE STRING "Where X280 active images that name no PLACEMENT run: lim or gddr")
set_property(
    CACHE
        X280_IMAGE_PLACEMENT
    PROPERTY
        STRINGS
            lim
            gddr
)
if(NOT X280_IMAGE_PLACEMENT MATCHES "^(lim|gddr)$")
    message(FATAL_ERROR "x280: X280_IMAGE_PLACEMENT is '${X280_IMAGE_PLACEMENT}'; expected lim or gddr")
endif()
# X280_ACTIVE_HARNESS implies X280_PER_HART_PROTOCOL: the harness publishes into the per-hart
# records the trap path also indexes.
set(_x280_active_asflags
    ${_x280_asflags}
    -DX280_PER_HART_PROTOCOL
    -DX280_ACTIVE_HARNESS
)
set(_x280_script_lim "${CMAKE_CURRENT_LIST_DIR}/ld/x280.ld")
set(_x280_entry_lim "${_x280_obj_dir}/entry_active.o")
_x280_compile("${_x280_entry_lim}" "${_x280_entry}" ${_x280_active_asflags})
# Preprocessed, so its region constants are x280_address_map.h's.
set(_x280_script_gddr "${_x280_obj_dir}/x280-gddr.ld")
add_custom_command(
    OUTPUT
        "${_x280_script_gddr}"
    COMMAND
        "${X280_GCC}" -E -P -x assembler-with-cpp -I${CMAKE_CURRENT_LIST_DIR}/include -MD -MT "${_x280_script_gddr}" -MF
        "${_x280_script_gddr}.d" -o "${_x280_script_gddr}" "${CMAKE_CURRENT_LIST_DIR}/ld/x280-gddr.ld"
    DEPENDS
        "${CMAKE_CURRENT_LIST_DIR}/ld/x280-gddr.ld"
    DEPFILE "${_x280_script_gddr}.d"
    COMMENT "x280: preprocessing ld/x280-gddr.ld"
    VERBATIM
)
set(_x280_entry_gddr "${_x280_obj_dir}/entry_active_gddr.o")
_x280_compile(
    "${_x280_entry_gddr}"
    "${_x280_entry}"
    ${_x280_active_asflags}
    -DX280_L2PF_BASIC=0x15811
    -DX280_L2PF_USER=0x38c84e
)
_x280_compile(
    "${_x280_obj_dir}/x280_active_start.o"
    "${CMAKE_CURRENT_LIST_DIR}/src/x280_active_start.c"
    ${X280_CFLAGS}
)
# The pinned-input prepare and teardown operations as a library, for images that run them
# themselves (OBJECTS ${_x280_pinned_input_library}).
set(_x280_pinned_input_library "${_x280_obj_dir}/x280_pinned_input_library.o")
_x280_compile(
    "${_x280_pinned_input_library}"
    "${CMAKE_CURRENT_LIST_DIR}/src/x280_pinned_input.c"
    ${X280_CFLAGS}
    -DX280_PINNED_INPUT_BUILD_LIBRARY=1
)
# x280_add_image records an object's source for the STOP-poll check and refuses an object
# without one.
set_property(
    GLOBAL
    PROPERTY
        "X280_SOURCE_OF:${_x280_pinned_input_library}"
            "${CMAKE_CURRENT_LIST_DIR}/src/x280_pinned_input.c"
)

#[[
x280_add_image(NAME <stem> SOURCES <source>... [LIM_IDLE | PLACEMENT <lim|gddr>] [OBJECTS <object>...]
               [DEFINES <macro>...] [INCLUDES <dir>...] [LINK_OPTIONS <flag>...])

Builds ${X280_IMAGE_DIR}/<stem>.elf and <stem>.bin from C and C++ (.cpp) sources, and adds them
to x280_images. An image links the active lifecycle harness and the script and prefetcher write of
its PLACEMENT, X280_IMAGE_PLACEMENT when it names none: ld/x280.ld (.text at 0x08001000) or
ld/x280-gddr.ld. Each gddr image's name is appended to the global property X280_GDDR_IMAGES.
LIM_IDLE links the per-hart entry and
ld/x280-lim-idle.ld (.text at 0x08000000).
DEFINES and INCLUDES (directories relative to this tree's root) apply to this image's sources only; C++ sources also get -std=c++17 -fno-exceptions -fno-rtti. LINK_OPTIONS
apply to this image's link. OBJECTS are prebuilt objects linked after the image's own, such as
${_x280_pinned_input_library}, the pinned-input operations compiled once with the plain image flags; each
must have its source recorded in the global property X280_SOURCE_OF:<object>. The sources of an
image linking the active harness, its objects' included, are appended to the global property
X280_ACTIVE_IMAGE_SOURCES, which the STOP-poll check scans.
#]]
function(x280_add_image)
    cmake_parse_arguments(PARSE_ARGV 0 ARG "LIM_IDLE" "NAME;PLACEMENT" "SOURCES;OBJECTS;DEFINES;INCLUDES;LINK_OPTIONS")
    if(NOT ARG_NAME OR NOT ARG_SOURCES OR ARG_UNPARSED_ARGUMENTS OR (ARG_LIM_IDLE AND ARG_PLACEMENT))
        message(
            FATAL_ERROR
            "x280_add_image: needs NAME and SOURCES, and at most one of LIM_IDLE and PLACEMENT; unparsed: '${ARG_UNPARSED_ARGUMENTS}'"
        )
    endif()
    if(NOT ARG_PLACEMENT)
        set(ARG_PLACEMENT "${X280_IMAGE_PLACEMENT}")
    endif()
    if(NOT ARG_PLACEMENT MATCHES "^(lim|gddr)$")
        message(FATAL_ERROR "x280_add_image(${ARG_NAME}): PLACEMENT is '${ARG_PLACEMENT}'; expected lim or gddr")
    endif()
    if(ARG_LIM_IDLE)
        set(entry "${_x280_obj_dir}/entry_hart.o")
        set(script "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/ld/x280-lim-idle.ld")
    else()
        set(entry
            "${_x280_entry_${ARG_PLACEMENT}}"
            "${_x280_obj_dir}/x280_active_start.o"
        )
        set(script "${_x280_script_${ARG_PLACEMENT}}")
        if(ARG_PLACEMENT STREQUAL "gddr")
            set_property(
                GLOBAL
                APPEND
                PROPERTY
                    X280_GDDR_IMAGES
                        "${ARG_NAME}"
            )
        endif()
    endif()
    list(TRANSFORM ARG_DEFINES PREPEND "-D")
    list(TRANSFORM ARG_INCLUDES PREPEND "-I${CMAKE_CURRENT_FUNCTION_LIST_DIR}/")

    set(objects "")
    foreach(source IN LISTS ARG_SOURCES)
        cmake_path(ABSOLUTE_PATH source BASE_DIRECTORY "${CMAKE_CURRENT_FUNCTION_LIST_DIR}")
        if(NOT ARG_LIM_IDLE)
            set_property(
                GLOBAL
                APPEND
                PROPERTY
                    X280_ACTIVE_IMAGE_SOURCES
                        "${source}"
            )
        endif()
        cmake_path(GET source STEM stem)
        set(object "${_x280_obj_dir}/${ARG_NAME}/${stem}.o")
        _x280_compile("${object}" "${source}" ${X280_CFLAGS} ${ARG_INCLUDES} ${ARG_DEFINES})
        list(APPEND objects "${object}")
    endforeach()
    foreach(object IN LISTS ARG_OBJECTS)
        get_property(source GLOBAL PROPERTY "X280_SOURCE_OF:${object}")
        if(NOT source)
            message(FATAL_ERROR "x280_add_image(${ARG_NAME}): OBJECTS entry ${object} has no recorded source")
        endif()
        if(NOT ARG_LIM_IDLE)
            set_property(
                GLOBAL
                APPEND
                PROPERTY
                    X280_ACTIVE_IMAGE_SOURCES
                        "${source}"
            )
        endif()
        list(APPEND objects "${object}")
    endforeach()

    set(elf "${X280_IMAGE_DIR}/${ARG_NAME}.elf")
    set(bin "${X280_IMAGE_DIR}/${ARG_NAME}.bin")
    add_custom_command(
        OUTPUT
            "${elf}"
        COMMAND
            "${X280_GCC}" ${_x280_ldflags} ${ARG_LINK_OPTIONS} -T "${script}" -o "${elf}" ${entry} ${objects}
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
