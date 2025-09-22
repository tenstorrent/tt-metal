export TT_METAL_HOME=/localdev/jhai/tt-metal
# export TT_METAL_HOME=$(shell pwd)/../../../../..
export PYTHONPATH=$(TT_METAL_HOME)

metal:
	cd $(TT_METAL_HOME); \
	./build_metal.sh --build-metal-tests; \
	cd $(TT_METAL_HOME)/tests/tt_metal/tt_metal/perf_microbenchmark/routing

metal_debug:
	cd $(TT_METAL_HOME); \
	./build_metal.sh --build-metal-tests --debug; \
	cd $(TT_METAL_HOME)/tests/tt_metal/tt_metal/perf_microbenchmark/routing

sanity:
	TT_METAL_CLEAR_L1=1 $(TT_METAL_HOME)/build/test/tt_metal/perf_microbenchmark/routing/test_tt_fabric --test_config ${TT_METAL_HOME}/tests/tt_metal/tt_metal/perf_microbenchmark/routing/test_fabric_sanity_at_least_2x2_mesh.yaml

ubench:
	TT_METAL_CLEAR_L1=1 $(TT_METAL_HOME)/build/test/tt_metal/perf_microbenchmark/routing/test_tt_fabric --test_config ${TT_METAL_HOME}/tests/tt_metal/tt_metal/perf_microbenchmark/routing/test_fabric_ubench_at_least_2x2_mesh.yaml

test: reset
	TT_METAL_CLEAR_L1=1 $(TT_METAL_HOME)/build/test/tt_metal/perf_microbenchmark/routing/test_tt_fabric --test_config ${TT_METAL_HOME}/tests/tt_metal/tt_metal/perf_microbenchmark/routing/test_fabric_iterations.yaml

debug: reset
	env TT_METAL_CLEAR_L1=1 gdb --args $(TT_METAL_HOME)/build/test/tt_metal/perf_microbenchmark/routing/test_tt_fabric --test_config ${TT_METAL_HOME}/tests/tt_metal/tt_metal/perf_microbenchmark/routing/test_fabric_iterations.yaml --dump-built-tests

debug_ubench: reset
	env TT_METAL_CLEAR_L1=1 gdb --args $(TT_METAL_HOME)/build/test/tt_metal/perf_microbenchmark/routing/test_tt_fabric --test_config ${TT_METAL_HOME}/tests/tt_metal/tt_metal/perf_microbenchmark/routing/test_fabric_ubench_at_least_2x2_mesh.yaml --dump-built-tests

reset:
	tt-smi -r

clean:
