
set(DETAIL_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/reports/compilation_reporter.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/reports/memory_reporter.cpp)

add_library(detail OBJECT ${DETAIL_SRC})
target_link_libraries(detail PUBLIC common)

target_include_directories(detail PUBLIC
                                    ${CMAKE_SOURCE_DIR}
                                    ${CMAKE_SOURCE_DIR}/tt_metal
                                    ${CMAKE_CURRENT_SOURCE_DIR})
