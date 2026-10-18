include(sources.cmake)

add_library(detail OBJECT ${DETAIL_SRC})
tt_reuse_precompile_headers(detail metal_common_pch)
target_link_libraries(
    detail
    PUBLIC
        common
    PRIVATE
        Metalium::Metal::Impl
        TT::Metalium::HostDevCommon
)
