include $(TT_METAL_HOME)/tests/ttnn/unit_tests/module.mk

# Only builds the tests, and specifically tests/ttnn/unit_tests in tests/ttnn/unit_tests/module.mk
.PHONY: tests/ttnn
tests/ttnn: tests/ttnn/unit_tests
