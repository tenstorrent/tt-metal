source /home/aliu/tt-metal/bin/activate

pytest -svv "tests/ttnn/unit_tests/operations/ccl/test_all_to_all_dispatch_6U.py::test_all_to_all_dispatch_no_trace[silicon_arch_name=wormhole_b0-dram-l1-dtype=DataType.BFLOAT16-topology=None-num_links=1-s2-hidden_size=7168-select_experts_k=8-experts_per_device=8-batches_per_device=16-cluster_axis=0-8x8_grid-trace_mode=False-fabric_2d]"

pytest -svv "tests/ttnn/unit_tests/operations/ccl/test_all_to_all_dispatch_6U.py::test_all_to_all_dispatch_no_trace[silicon_arch_name=wormhole_b0-dram-l1-dtype=DataType.BFLOAT16-topology=None-num_links=1-s2-hidden_size=7168-select_experts_k=8-experts_per_device=8-batches_per_device=16-cluster_axis=1-8x8_grid-trace_mode=False-fabric_2d]"

#pytest -svv "tests/ttnn/unit_tests/operations/ccl/test_all_to_all_combine_6U.py::test_all_to_all_combine_no_trace[silicon_arch_name=wormhole_b0-dram-l1-dtype=DataType.BFLOAT16-topology=None-num_links=1-num_iters=2-local_reduce=True-seq=1-hidden_size=7000-select_experts_k=8-experts_per_device=8-batches_per_device=8-axis=1-8x8_grid-trace_mode=False-fabric_2d]"
