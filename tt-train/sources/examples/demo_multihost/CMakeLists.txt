project(demo_multihost)

set(SOURCES
    main.cpp
    roles/aggregator.cpp
    roles/aggregator.hpp
    roles/optimizer.cpp
    roles/optimizer.hpp
    roles/worker.cpp
    roles/worker.hpp
)

add_executable(demo_multihost ${SOURCES})
target_link_libraries(demo_multihost PRIVATE ttml)
