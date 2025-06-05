add_library(metalium-logging OBJECT logging.cpp)
add_library(TT::Metalium::Logging ALIAS metalium-logging)

target_link_libraries(metalium-logging PRIVATE tt-logger::tt-logger)
