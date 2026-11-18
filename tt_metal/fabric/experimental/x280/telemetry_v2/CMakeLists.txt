# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Firmware image. Produces ${X280_IMAGE_DIR}/firmware-telemetry-v2.bin.
# x280_add_image resolves relative SOURCES against x280/, so pass an absolute path.
x280_add_image(NAME firmware-telemetry-v2 STANDALONE SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/telemetry_firmware.c)

# Host loader for the standalone telemetry firmware.
add_executable(x280_telemetry_v2_loader host/loader.cpp)

target_link_libraries(
    x280_telemetry_v2_loader
    PRIVATE
        x280_device
        TT::Metalium
)

# Test app: reads the heartbeat after the loader has exited. It uses the runtime's private Metal
# transport header (src/device), as the device tests do.
add_executable(x280_telemetry_v2_heartbeat_check host/heartbeat_check.cpp)

target_include_directories(
    x280_telemetry_v2_heartbeat_check
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/..
        ${CMAKE_CURRENT_SOURCE_DIR}/../src/device
        "$<TARGET_PROPERTY:Metalium::Metal,INCLUDE_DIRECTORIES>"
)

target_link_libraries(
    x280_telemetry_v2_heartbeat_check
    PRIVATE
        x280_device
        TT::Metalium
)

# Client App: Reads the table that firmware updates.
add_executable(x280_telemetry_client host/telemetry_client.cpp)

target_include_directories(
    x280_telemetry_client
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/..
        ${CMAKE_CURRENT_SOURCE_DIR}/../src/device
        "$<TARGET_PROPERTY:Metalium::Metal,INCLUDE_DIRECTORIES>"
)


target_link_libraries(
    x280_telemetry_client
    PRIVATE
        x280_device
        TT::Metalium
)

# FTXUI for the terminal UI. Only x280_telemetry_tui links it; nothing else in Metal does.
if(NOT TARGET ftxui::screen)
    CPMAddPackage(
        NAME ftxui
        GITHUB_REPOSITORY ArthurSonzogni/FTXUI
        GIT_TAG v6.1.9
        SYSTEM YES
        OPTIONS
            "FTXUI_BUILD_EXAMPLES OFF"
            "FTXUI_BUILD_DOCS OFF"
            "FTXUI_BUILD_TESTS OFF"
            "FTXUI_ENABLE_INSTALL OFF"
            "CMAKE_MESSAGE_LOG_LEVEL NOTICE"
    )
endif()

# Terminal UI: finds the running standalone firmware and shows its telemetry table live. Read-only.
add_executable(x280_telemetry_tui host/telemetry_tui.cpp)

target_include_directories(
    x280_telemetry_tui
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/..
        ${CMAKE_CURRENT_SOURCE_DIR}/../src/device
        "$<TARGET_PROPERTY:Metalium::Metal,INCLUDE_DIRECTORIES>"
)

target_link_libraries(
    x280_telemetry_tui
    PRIVATE
        x280_device
        TT::Metalium
        ftxui::screen
        ftxui::dom
        ftxui::component
)

# Where the loader finds the firmware image by default.
target_compile_definitions(x280_telemetry_v2_loader PRIVATE X280_IMAGE_DIR="${X280_IMAGE_DIR}")

# Building the loader also builds the firmware image it will load.
add_dependencies(x280_telemetry_v2_loader x280_image_firmware-telemetry-v2)
