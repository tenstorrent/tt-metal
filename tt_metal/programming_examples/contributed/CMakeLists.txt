
set(VECADD_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/vecadd/vecadd.cpp)

CREATE_PGM_EXAMPLES_EXE("${VECADD_SRCS}" "contributed") # output binaries to build/programming_examples/contributed

add_custom_target(contributed DEPENDS ${PROGRAMMING_EXAMPLES_TEST_TARGETS})
