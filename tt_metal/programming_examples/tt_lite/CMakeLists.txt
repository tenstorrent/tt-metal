# --- capture_and_dump: captures a trace and writes a .ttb file ---
add_executable(tt_lite_capture capture_and_dump.cpp)
target_link_libraries(tt_lite_capture PUBLIC tt_metal pthread)
target_include_directories(tt_lite_capture PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/tt_metal/impl)
if(TARGET TT::CommonPCH)
    tt_reuse_precompile_headers(tt_lite_capture TT::CommonPCH)
endif()
set_target_properties(
    tt_lite_capture PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/programming_examples/tt_lite
)

# --- replay_from_file: loads a .ttb file and replays the trace ---
add_executable(tt_lite_replay replay_from_file.cpp)
target_link_libraries(tt_lite_replay PUBLIC tt_metal pthread)
target_include_directories(tt_lite_replay PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/tt_metal/impl)
if(TARGET TT::CommonPCH)
    tt_reuse_precompile_headers(tt_lite_replay TT::CommonPCH)
endif()
set_target_properties(
    tt_lite_replay PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/programming_examples/tt_lite
)
