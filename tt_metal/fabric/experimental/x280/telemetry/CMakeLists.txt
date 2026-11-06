# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Experimental telemetry image and the terminal UI that loads it and shows its table.
# Added from x280/CMakeLists.txt, so x280_add_image and x280_device are in scope.

# x280_add_image resolves relative SOURCES against x280/, so pass an absolute path.
# Produces ${X280_IMAGE_DIR}/firmware-telemetry2.bin, which x280_telemetry_tui loads by default.
x280_add_image(NAME firmware-telemetry2 SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/telemetry_v2.c)

# FTXUI for the terminal UI. Only x280_telemetry_tui links it; nothing else in Metal does.
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

# Primes the chips, or loads the firmware and shows the tables in a live terminal UI.
add_executable(
    x280_telemetry_tui
        host/telemetry_tui.cpp
        host/telemetry_host_lib.cpp
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
target_include_directories(x280_telemetry_tui PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_compile_definitions(x280_telemetry_tui PRIVATE X280_IMAGE_DIR="${X280_IMAGE_DIR}")
add_dependencies(x280_telemetry_tui x280_images)

# Legacy programs, kept in host/legacy and not built. Uncomment a block to build it again.

# v1 firmware image, loaded by x280_telemetry_loader.
#x280_add_image(NAME firmware-telemetry SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/telemetry.c)

# Host loader which loads the firmware, and idle firmware if necessary.
#add_executable(x280_telemetry_loader host/legacy/telemetry_loader.cpp)
#target_link_libraries(
#    x280_telemetry_loader
#    PRIVATE
#        x280_device
#        TT::Metalium
#)
#target_compile_definitions(x280_telemetry_loader PRIVATE X280_IMAGE_DIR="${X280_IMAGE_DIR}")
#add_dependencies(x280_telemetry_loader x280_images)

# Host application which reads the telemetry table and displays it.
#add_executable(x280_telemetry_app host/legacy/telemetry_app.cpp)
#target_link_libraries(
#    x280_telemetry_app
#    PRIVATE
#        x280_device
#        TT::Metalium
#)
#target_include_directories(x280_telemetry_app PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")

# Host application which loads the firmware and prints the tables until Ctrl-C.
# Shares telemetry_host_lib.cpp with the TUI.
#add_executable(
#    x280_telemetry_host_app
#        host/legacy/telemetry_host_app.cpp
#        host/telemetry_host_lib.cpp
#)
#target_link_libraries(
#    x280_telemetry_host_app
#    PRIVATE
#        x280_device
#        TT::Metalium
#)
#target_include_directories(x280_telemetry_host_app PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
#target_compile_definitions(x280_telemetry_host_app PRIVATE X280_IMAGE_DIR="${X280_IMAGE_DIR}")
#add_dependencies(x280_telemetry_host_app x280_images)
