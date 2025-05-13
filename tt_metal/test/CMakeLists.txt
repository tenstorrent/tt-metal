# Spell out the RPATH so that shlibdeps can find deps across package components.
set(CMAKE_INSTALL_RPATH "$ORIGIN/../lib")

# Smoke
add_executable(tt-metalium-validation-smoke)

target_link_libraries(
    tt-metalium-validation-smoke
    PRIVATE
        TT::SystemHealth::Smoke
        TT::Metalium::Test::Device::Smoke
        TT::Metalium::Test::Dispatch::Smoke
        TT::Metalium::Test::STL::Smoke
)

install(TARGETS tt-metalium-validation-smoke RUNTIME COMPONENT metalium-validation)

# Basic
add_executable(tt-metalium-validation-basic)

target_link_libraries(tt-metalium-validation-basic PRIVATE TT::Metalium::Test::Dispatch::Basic)

install(TARGETS tt-metalium-validation-basic RUNTIME COMPONENT metalium-validation)
