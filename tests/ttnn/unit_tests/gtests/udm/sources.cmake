# Source files for ttnn udm tests
# Module owners should update this file when adding/removing/renaming source files

set(UNIT_TESTS_TTNN_UDM_SOURCES
    copy/test_udm_copy.cpp
    eltwise/test_udm_add.cpp
    reduction/sharded/test_udm_reduction.cpp
    reduction/interleaved/test_udm_reduction_interleaved.cpp
)
