export TT_METAL_HOME=$(shell pwd)
export TT_METAL_RUNTIME_ROOT=$(TT_METAL_HOME)
export PYTHONPATH=$(TT_METAL_HOME)

BIN=$(TT_METAL_HOME)/build_Release/test/tt_metal/tt_fabric/test_infra/test_tt_fabric
TEST_DIR=$(TT_METAL_HOME)/tests/tt_metal/tt_metal/perf_microbenchmark/routing

metal:
	$(TT_METAL_HOME)/build_metal.sh --build-metal-tests -c --development

metal_debug:
	$(TT_METAL_HOME)/build_metal.sh --build-metal-tests -c --debug

help:
	TT_METAL_CLEAR_L1=1 $(BIN) --help

sanity:
	TT_METAL_CLEAR_L1=1 $(BIN) --test_config $(TEST_DIR)/test_fabric_sanity_at_least_2x2_mesh.yaml

ubench:
	TT_METAL_CLEAR_L1=1 $(BIN) --test_config $(TEST_DIR)/test_fabric_ubench_at_least_2x2_mesh.yaml --dump-built-tests

test:
	TT_METAL_CLEAR_L1=1 $(BIN) --test_config $(TEST_DIR)/test.yaml --dump-built-tests

debug:
	env TT_FABRIC_PROFILER=RECEIVER_CHANNEL_FORWARD TT_METAL_CLEAR_L1=1 gdb --args $(BIN) --test_config $(TEST_DIR)/test_fabric_ubench_at_least_2x2_mesh.yaml --dump-built-tests

# Must be run from within the python venv (source $(TT_METAL_HOME)/python_env/bin/activate)
pytest:
	python -m tracy -r -m pytest tests/nightly/tg/ccl/test_minimal_all_gather_async.py

2x2:
	TT_METAL_RUNTIME_ROOT=$(TT_METAL_HOME) tt-run --mpi-args "$(mpi_args)" --rank-binding $(TT_METAL_HOME)/tests/tt_metal/distributed/config/2x2_multiprocess_rank_bindings.yaml $(BIN) --test_config $(TEST_DIR)/test_t3k_2x2.yaml --dump-built-tests

reset:
	tt-smi -r

pre-commit:
	pre-commit run --all-files

clean:
	$(TT_METAL_HOME)/build_metal.sh --clean
