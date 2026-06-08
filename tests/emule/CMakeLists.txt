# Emule CTest registrations — reuses existing test executables with emule env vars.
# No source compilation; just CTest entries with --gtest_filter and ENVIRONMENT.

set(EMULE_ENV
    "TT_METAL_RUNTIME_ROOT=${PROJECT_SOURCE_DIR}"
    "TT_METAL_EMULE_MODE=1"
    "TT_METAL_SLOW_DISPATCH_MODE=1"
    "TT_METAL_MOCK_CLUSTER_DESC_PATH=${PROJECT_SOURCE_DIR}/tt_metal/third_party/umd/tests/cluster_descriptor_examples/wormhole_N150.yaml"
)

set(EMULE_LABEL "emule")

# --- Tier 2: tt_metal API buffer tests (from unit_tests_api) ---

if(TT_METAL_BUILD_TESTS)
    set(EMULE_API_TESTS
        "MeshDeviceFixture.TestSimpleDramBufferLo"
        "MeshDeviceFixture.TestSimpleDramBufferHi"
        "MeshDeviceFixture.TestSimpleL1BufferLo"
        "MeshDeviceFixture.TestSimpleL1BufferHi"
        "MeshDeviceFixture.TensixTestSimpleL1ReadWriteTileLo"
        "MeshDeviceFixture.TensixTestSimpleL1ReadWriteTileHi"
        "MeshDeviceFixture.TensixTestSimpleL1ReadWritex2y2TileLo"
        "MeshDeviceFixture.TensixTestSimpleL1ReadWritex2y2TileHi"
    )

    foreach(TEST_FILTER ${EMULE_API_TESTS})
        string(REPLACE "." "_" TEST_NAME "${TEST_FILTER}")
        add_test(
            NAME "emule/${TEST_NAME}"
            COMMAND
                unit_tests_api --gtest_filter=${TEST_FILTER}
        )
        set_tests_properties(
            "emule/${TEST_NAME}"
            PROPERTIES
                ENVIRONMENT
                    "${EMULE_ENV}"
                LABELS
                    "${EMULE_LABEL}"
        )
    endforeach()
endif()

# --- Tier 3: ttnn tests (from unit_tests_ttnn) ---
# Parametrized tests have suite prefixes (e.g. "AddUnaryTests/AddUnaryFixture"),
# so use "*Fixture.*" to match regardless of prefix.

if(TTNN_BUILD_TESTS)
    set(EMULE_TTNN_FILTERS
        "*Add1DTensorAndScalarFixture.*"
        "*AddUnaryFixture.*"
        "*SubUnaryFixture.*"
        "*RsubUnaryFixture.*"
        "SingleTileMatmulFixture.*"
        "MultiTileMatmulFixture.*"
        "*MatmulSweepFixture.*"
        "*RelationalUnaryFixture.*"
    )

    set(EMULE_TTNN_NAMES
        "Add1DTensorAndScalarFixture"
        "AddUnaryFixture"
        "SubUnaryFixture"
        "RsubUnaryFixture"
        "SingleTileMatmulFixture"
        "MultiTileMatmulFixture"
        "MatmulSweepFixture"
        "RelationalUnaryFixture"
    )

    list(LENGTH EMULE_TTNN_FILTERS _len)
    math(EXPR _last "${_len} - 1")

    foreach(_i RANGE ${_last})
        list(GET EMULE_TTNN_FILTERS ${_i} TEST_FILTER)
        list(GET EMULE_TTNN_NAMES ${_i} TEST_NAME)
        add_test(
            NAME "emule/${TEST_NAME}"
            COMMAND
                unit_tests_ttnn --gtest_filter=${TEST_FILTER}
            WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        )
        set_tests_properties(
            "emule/${TEST_NAME}"
            PROPERTIES
                ENVIRONMENT
                    "${EMULE_ENV}"
                LABELS
                    "${EMULE_LABEL}"
        )
    endforeach()
endif()
