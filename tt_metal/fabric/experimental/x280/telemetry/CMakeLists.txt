# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Experimental telemetry image and the driver that primes the device or loads the image.
# Added from x280/CMakeLists.txt, so x280_add_image and x280_device are in scope.

# x280_add_image resolves relative SOURCES against x280/, so pass an absolute path.
# Produces ${X280_IMAGE_DIR}/firmware-telemetry.bin.
x280_add_image(NAME firmware-telemetry SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/telemetry.c)

# Adds the host driver which loads the fimrware, and idle firmware if necessary.
add_executable(x280_telemetry_driver host/telemetry_driver.cpp)
target_link_libraries(
    x280_telemetry_driver
    PRIVATE
        x280_device
        TT::Metalium
)
target_compile_definitions(x280_telemetry_driver PRIVATE X280_IMAGE_DIR="${X280_IMAGE_DIR}")
add_dependencies(x280_telemetry_driver x280_images)

# Adds the host application which reads the telemetry table and displays it.
add_executable(x280_telemetry_app host/telemetry_app.cpp)
target_link_libraries(
    x280_telemetry_app
    PRIVATE
        x280_device
        TT::Metalium
)
target_include_directories(x280_telemetry_app PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
