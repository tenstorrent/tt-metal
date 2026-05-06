add_executable(triage_open_and_close open_and_close.cpp)
target_link_libraries(triage_open_and_close PUBLIC TT::Metalium)

add_executable(triage_hold_device hold_device.cpp)
target_link_libraries(triage_hold_device PUBLIC TT::Metalium)
