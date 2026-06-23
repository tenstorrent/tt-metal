export TT_METAL_HOME=/localdev/jhai/tt-metal
export TT_METAL_RUNTIME_ROOT=$(TT_METAL_HOME)
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

reset:
	tt-smi -r
	# sudo ipmitool raw 0x30 0x8B 0xF 0xFF 0x0 0xF | sleep 180

test_list:
	$(TT_METAL_HOME)/build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_list_tests

test_1d:
	# export TT_METAL_DPRINT_CORES=0,0; \
	# export TT_METAL_DPRINT_CHIPS=0,1,2,3,4,5,6,7; \
	$(TT_METAL_HOME)/build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="NightlyFabric1DFixture.TestLinearFabric*"
	# $(TT_METAL_HOME)/build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="NightlyFabric2DFixture.TestMeshFabric*"
	# $(TT_METAL_HOME)/build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="NightlyFabric2DFixture.TestMeshFabricMulticastNocFusedScatterWriteAtomicInc*"
	# $(TT_METAL_HOME)/build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="NightlyFabric1DFixture.TestLinearFabricSparseMulticastNocUnicastWrite"
	# $(TT_METAL_HOME)/build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="NightlyFabric1DFixture.TestLinearFabricMulticastNocUnicastWrite"

pytest:
	# export TT_METAL_DPRINT_CORES=all; \
	# export TT_METAL_DPRINT_CHIPS=all; \
	# export TT_METAL_WATCHER=5; \
	pytest $(TT_METAL_HOME)/tests/nightly/t3000/ccl/test_deepseek_moe_reduce_scatter.py

clean:
	cd $(TT_METAL_HOME); \
	./build_metal.sh --clean; \
	cd $(TT_METAL_HOME)/tests/tt_metal/tt_metal/perf_microbenchmark/routing
