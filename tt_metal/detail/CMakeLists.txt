set(DETAIL_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/reports/compilation_reporter.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/reports/memory_reporter.cpp
)

add_library(detail OBJECT ${DETAIL_SRC})
target_link_libraries(detail PUBLIC common)
