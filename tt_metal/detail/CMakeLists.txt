include(sources.cmake)

add_library(detail OBJECT ${DETAIL_SRC})
target_precompile_headers(detail REUSE_FROM metal_common_pch)
target_link_libraries(
    detail
    PUBLIC
        common
    PRIVATE
        Metalium::Metal::Impl
        TT::Metalium::HostDevCommon
)
