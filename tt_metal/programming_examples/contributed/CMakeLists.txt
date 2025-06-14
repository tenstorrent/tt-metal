set(VECADD_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/vecadd/vecadd.cpp)
set(MULTICAST_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/multicast/multicast.cpp)

CREATE_PGM_EXAMPLES_EXE("${VECADD_SRCS}" "contributed") # output binaries to build/programming_examples/contributed
CREATE_PGM_EXAMPLES_EXE("${MULTICAST_SRCS}" "contributed")

add_custom_target(contributed DEPENDS ${PROGRAMMING_EXAMPLES_TEST_TARGETS})
