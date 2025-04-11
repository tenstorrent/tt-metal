# Smoke
add_executable(tt-metalium-validation-smoke)

target_link_libraries(
    tt-metalium-validation-smoke
    PRIVATE
        TT::Metalium::Test::Dispatch::Smoke
        TT::Metalium::Test::STL::Smoke
)

install(
    TARGETS
        tt-metalium-validation-smoke
    RUNTIME
        DESTINATION libexec/tt-metalium/validation
        COMPONENT metalium-validation
)

# Sanity
add_executable(tt-metalium-validation-sanity)

target_link_libraries(tt-metalium-validation-sanity PRIVATE TT::Metalium::Test::Dispatch::Sanity)

install(
    TARGETS
        tt-metalium-validation-sanity
    RUNTIME
        DESTINATION libexec/tt-metalium/validation
        COMPONENT metalium-validation
)
