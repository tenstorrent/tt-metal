# Guide for adding op unit tests for new fused ops based on an existing model/module

The following should be provided by the user in order to complete this task successfully:
- The name of the new fused op
- The sequence of ops that comprise the new fused op
- The test command for the containing module of the sequence of ops

Before you start, please read the example test in models/demos/deepseek_v3/tests/fused_op_unit_tests/mla/test_ds_fused_wqkva.py carefully. The new test should be similar to the example test, just for a different fused op.
Example test command for test_mla.py:
```
source ../setup_metal.sh
export DEEPSEEK_V3_HF_MODEL="/proj_sw/user_dev/deepseek-ai/DeepSeek-R1-0528/" && export DEEPSEEK_V3_CACHE="/proj_sw/user_dev/deepseek-v3-cache2" && export TT_METAL_RUNTIME_ROOT=/proj_sw/user_dev/jrock/tt-metal && export MESH_DEVICE=TG && pytest /proj_sw/user_dev/jrock/tt-metal/models/demos/deepseek_v3/tests/test_mla.py::test_forward_pass -k "decode" 2>&1 | tee /proj_sw/user_dev/jrock/tt-metal/logs/ds_mla_$(date +%Y%m%d_%H%M%S).log
```

Follow these steps to add a new fused op unit test:
1. Run the provided module test command to capture the baseline accuracy and verify that the test is working.
2. Identify the ttnn operations that will be fused into a new operation, this should be provided by the user;
    1. Finde the sequence of ops in the module for decode; if it's unclear which ones should be fused, let the user know immediately.
    2. Find the sequence of ops in the module for prefill; if the same sequence of ops does not exist for prefill, then ignore all prefill instructions and inform the user in the final summary that this was a decode only fused op sequence.
3. Create a new file for the new fused op unit test under MODEL_FOLDER/tests/fused_op_unit_tests/test_NEW_FUSED_OP_NAME.py
4. In the fused op unit test file, create a PyTorch reference for the newly fused op based on the sequence of ttnn ops.
	- Make sure to use the PyTorch reference as a basis here. To figure out what reference code is used, look at the containing module's test, e.g. test_mla.py for a fused op that's contained in the MLA module. Then figure out which exact portion of the reference model's code corresponds to our new fused op. It might be necessary to modify the reference code slightly to get the correct reference implementation since the exact operations used in the ttnn model and in the reference model my differer.
	- In the fused op unit test file, create a new reference function "NEW_FUSED_OP_reference". Inputs/outputs to the function should be PyTorch tensors and function parameters of that sequence of ops; make sure to parameterize everything that's parameterized in the module as well, no hardcoding unless it is hardcoded in the model too.
5. In the fused op unit test file, create a function "NEW_FUSED_OP_ttnn" containing the sequence of ttnn ops that correspond to the new fused op. Inputs and outputs correspond to the inputs/output of the sequence of ops for the new fused op; inputs/outputs should be ttnn tensors and function parameters.
6. *Verify NEW_FUSED_OP_ttnn* by replacing the identified sequence of ops within the module code with a *function call to NEW_FUSED_OP_ttnn of the newly created test file*, and run the module test to verify that the code is running as expected and producing good outputs (no change to the baseline!). If it's not passing, debug NEW_FUSED_OP_ttnn to make it work. Add the result of this including a log file in the final summary of work. Do not proceed further in the TODOs unless this passes. If it passes, revert those changes in the module and continue.
7. In the fused op unit test file, implement a pytest that:
	1. Creates all input tensors and function parameters (based on the actual use in the module code)
      - Parameters to the ops should be variables in an easily readable section in the code
      - Use pytest parameters for expected_pcc, expected_perf, mode+seqlen(add decode with seqlen 1, prefill with seqlen 128/1024/8k/128k)
    2. Add a pytest parameter option to use real model weights/random weights
	3. Tests the NEW_FUSED_OP_ttnn using the reference code in NEW_FUSED_OP_reference
    4. Contains a performance measurement wrapper, so that we measure device performance. See following notes on performance measurements.
    5. Supports trace mode; add a pytest parameter to turn tracing on/off
      - If it fails due to trace_region_size being too small, set the trace_region_size pytest parameter (see example test) based on the required size as printed in the log.
    6. Contains a pytet parameter to turn program_caching on/off, this is only done when trace if off, with trace mode program_cache must be enabled too; use device.disable_and_clear_program_cache() for disabling, it's enabled by default
	7. Compares PCC, ATOL, performance
8. *Verify NEW_FUSED_OP_reference* and the test code itself by running the unit test and comparing pcc to NEW_FUSED_OP_ttnn. The PCC should typically by > 0.99, otherwise there's likely something wrong. Add the result of this including a log file in the final summary of work. Do not proceed further in the TODOs unless this passes.
	1. Update the expected_pcc with the pcc value from the test (if it's > 0.99 otherwise, debug the test to fix it!)
	2. Use the current perf as expected_perf but add a TODO comment to add the actual target (based on theoretical numbers)
9. *Verify* that the fused op unit test as well as the sequence of ops within the module *use the exact same configuration including input shapes, dtype, memory_config, and buffer type*. Add the result of this including a log file in the final summary of work. Update the test configurations if there are any mismatches, do not change the sequence in the module, consider this the ground truth.
    1. Run the device perf test of the fused op unit test for prefill (shortest seqlen in fused op unit test) and decode, copy the generated csv files into a newly created folder.
    2. Run the module test with 1 iteration, both for prefill (same seqlen as in fused op unit test csv) and decode, copy the generated csv files into the same folder as in the last step.
    3. Compare for each op that all properties are identical, i.e. fused op unit test and module test match in terms of op input/output properties both for prefill and for decode. Take a look at example_compare_fused_wqkva_configs.py to see how that was done for an example fused op unit test.
10. Add a single device test
    1. Create a new pytest in the file with the same name + "_single_device"
    2. If the sequence of ops contains a CCL, skip the single device test with an appropriate skip message. The following points only affect tests that can be executed on single device
    3. The test takes the first device from the mesh_device fixture and runs the ops only on that device
    4. The input shape to the single device test is the chunk of the input from the multi device test that resides on the first device, hence the chunk that is processed but the first device. In order to find the correct shape, run the device perf test (multi device) and extract the input shape to the first op from the generated ops_perf_report, this is already the per device shape. Do the same for all matmul input_tensor_b shapes.
    5. Restructure the existing code to maintain a clean test that re-uses common parts of the code. Be very careful not to change any functionality in the existing test.
    6. Run the single device test and verify that both the PCC and the perf are the same as for the multi device test.
11. Add a single device test for device performance, see step 10 for how to do that.
12. If single device tests are not skipped, *verify* the single device tests by running the single device, device perf test that generated the csv file and compare it to the multi device perf csv. All shapes must match, create a helper script to verify that.
13. Print the summary for all verification steps clearly representing the results and the links to logs for all successful verification steps.
14. List anything that was unexpected and/or any workarouds you needed to make the fused op unit test work.

Notes on performance measurements:
- Performance measurements use three metrics: e2e_duration, kernel_duration, op_to_op_latency
- e2e_duration: average duration of the whole fused op (sequence of ops), end to end. This is measured using profiler.start and profiler.end calls, use 10 warumup iterations and 100 measurement iterations.
- Device performance metrics: kernel_duration, op_to_op_latency
    - Both device performance metrics require tracy profiler to run device performance and post processing of the resulting performance table; uses run_device_profiler helper to run the test function with profiler and subsequent post processing of the results; averages each op's metrics over all iterations
    - Warumup and measurement iterations
        - Use 10 warumup iteration and 10 measurement iterations
        - Signposts are used to ignore the warumup iterations in the post processing step and use the following iterations for perf measurement averages
        - Per iteration, use the max over devices for all ops except for CCLs (AllGather, ReduceScatter, AllReduce, AllToAll) where to use the average over all devices
        - Then average over all measurement iterations to compute the average duration per op per iteration
    - total_kernel_duration/total_op_to_op_latency: the sum of all op's average durations per op per iteration
    - Both total_kernel_duration and total_op_to_op_latency are printed, asserted and uploaded via benchmark_data.add_measurement
- Decode uses trace mode for perf measurements, prefill uses non_trace mode for perf measurements

Notes on using TT hardware:
- If running tests, set a timeout of 15 minutes.
- If there's a machine issue, you'll need to reset the machine using "tt-smi -glx_reset"
- Important: Running the test may take a few minutes, if you kill it the device might need a reset. In general, if there is no log output for more than 5 minutes the test likely hangs and needs to be killed + device reset, but if there's log output keep it running.
- Run each job with piping the output to a log file, i.e. add " 2>&1 | tee $TT_METAL_HOME/logs/ds_mla_$(date +%Y%m%d_%H%M%S).log" to each command
