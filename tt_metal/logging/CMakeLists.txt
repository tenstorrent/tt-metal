# Include source file lists (module owners maintain sources.cmake)
include(sources.cmake)

add_library(metalium-logging OBJECT ${LOGGING_SOURCES})
add_library(TT::Metalium::Logging ALIAS metalium-logging)

target_link_libraries(metalium-logging PRIVATE tt-logger::tt-logger)
