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

# Basic
add_executable(tt-metalium-validation-basic)

target_link_libraries(tt-metalium-validation-basic PRIVATE TT::Metalium::Test::Dispatch::Basic)

install(
    TARGETS
        tt-metalium-validation-basic
    RUNTIME
        DESTINATION libexec/tt-metalium/validation
        COMPONENT metalium-validation
)
