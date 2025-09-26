project(gpt2s_inference)

set(SOURCES main.cpp)

add_executable(gpt2s_inference ${SOURCES})
target_link_libraries(gpt2s_inference PRIVATE ttml)
