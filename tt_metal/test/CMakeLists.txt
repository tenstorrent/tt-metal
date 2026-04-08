# Spell out the RPATH so that shlibdeps can find deps across package components.
set(CMAKE_INSTALL_RPATH "$ORIGIN/../lib")

# Smoke
add_executable(tt-metalium-validation-smoke)

target_link_libraries(
    tt-metalium-validation-smoke
    PRIVATE
        TT::SystemHealth::Smoke
        TT::Metalium::Test::Fabric::Smoke
        TT::Metalium::Test::API
        TT::Metalium::Test::Device::Smoke
        TT::Metalium::Test::Dispatch::Smoke
        TT::Metalium::Test::SFPI
        TT::Metalium::Test::STL::Smoke
)

install(TARGETS tt-metalium-validation-smoke RUNTIME COMPONENT metalium-validation)

# Basic
add_executable(tt-metalium-validation-basic)

target_link_libraries(tt-metalium-validation-basic PRIVATE TT::Metalium::Test::Dispatch::Basic)

install(TARGETS tt-metalium-validation-basic RUNTIME COMPONENT metalium-validation)

install(
    FILES
        lsan.supp
    DESTINATION ${CMAKE_INSTALL_DATADIR}/tt-metalium
    CONFIGURATIONS
        ASan
        ASanCoverage
    COMPONENT metalium-runtime
)

# Install cluster descriptor YAML files for mock device testing.
# These are required by configure_mock_mode(), which looks for files in:
# get_root_dir() + "/tt_metal/third_party/umd/tests/cluster_descriptor_examples/"
install(
    DIRECTORY
        ${PROJECT_SOURCE_DIR}/tt_metal/third_party/umd/tests/cluster_descriptor_examples/
    DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/tt_metal/third_party/umd/tests/cluster_descriptor_examples
    COMPONENT metalium-validation
    FILES_MATCHING
    PATTERN
    "*.yaml"
)

# Install SOC descriptor YAML files for mock device testing.
# These are required by get_soc_description_file(), which looks for files in:
# get_root_dir() + "/tt_metal/soc_descriptors/"
install(
    FILES
        ${PROJECT_SOURCE_DIR}/tt_metal/soc_descriptors/quasar_32_arch.yaml
    DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/tt_metal/soc_descriptors
    COMPONENT metalium-validation
)
