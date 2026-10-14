# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# CTest x280_image_layout: resident idle's .text starts the LIM at 0x08000000, and an active
# image's .text follows the idle page at 0x08001000.
# Usage: cmake -DREADELF=<riscv readelf> -DIMAGE_DIR=<X280_IMAGE_DIR> -P x280_image_layout.cmake

foreach(_pair IN ITEMS "firmware-lim-idle.elf=08000000" "firmware.elf=08001000")
    string(REPLACE "=" ";" _pair "${_pair}")
    list(GET _pair 0 _elf)
    list(GET _pair 1 _addr)
    execute_process(
        COMMAND
            "${READELF}" -SW "${IMAGE_DIR}/${_elf}"
        OUTPUT_VARIABLE _sections
        RESULT_VARIABLE _rc
    )
    if(NOT _rc EQUAL 0)
        message(FATAL_ERROR "FAIL: readelf -S ${IMAGE_DIR}/${_elf} exited ${_rc}")
    endif()
    if(NOT _sections MATCHES "\\] \\.text +PROGBITS +0*${_addr} ")
        message(FATAL_ERROR "FAIL: ${_elf} .text is not at 0x${_addr}:\n${_sections}")
    endif()
    message(STATUS "PASS: ${_elf} .text at 0x${_addr}")
endforeach()
