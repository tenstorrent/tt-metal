project(llm_inference)

set(SOURCES main.cpp)

add_executable(llm_inference ${SOURCES})
target_link_libraries(llm_inference PRIVATE ttml)
