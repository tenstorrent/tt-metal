# Include source file lists (module owners maintain sources.cmake)
include(sources.cmake)

add_library(detail OBJECT ${DETAIL_SRC})
target_link_libraries(
    detail
    PUBLIC
        common
    PRIVATE
        Metalium::Metal::Impl
        TT::Metalium::HostDevCommon
)
