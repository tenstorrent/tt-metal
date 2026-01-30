# Guide for adding op unit tests for new fused ops based on an existing model/module

The following should be provided by the user in order to complete this task successfully:
- The module and name of the new fused op or function call containing the sequence of ops

Before you start, please read the example test in models/demos/deepseek_v3/tests/fused_op_unit_tests/mla/test_ds_fused_wqkva.py carefully. The new test should be similar to the example test, just for a different fused op.

Example test command for running the module test for test_mla.py:
```
source ../setup_metal.sh && pytest /proj_sw/user_dev/jrock/tt-metal/models/demos/deepseek_v3/tests/test_mla.py::test_forward_pass -k "decode" 2>&1 | tee /proj_sw/user_dev/jrock/tt-metal/logs/ds_mla_$(date +%Y%m%d_%H%M%S).log
```
Change the module test command based on the module that the new fused op is contained in.

## Steps to add a new fused op unit test
0. Check if the unit test already exists! If a unit test for the given op/ sequence of ops already exists, use the following list as a checklist to verify that all steps have been completed successfully. Look up artifacts in tests/fused_op_unit_tests/MODULE/test_results to verify
  1. Compare the OP_NAME_results.csv to the README.md's row corresponding to the unit test. If they match, continue. If they don't match, update the README.md based on the OP_NAME_results.csv.
  2. Compare the unit test code with the OP_NAME_results.csv and verify if the OP_NAME_results.csv was generated from the current status of the unit tests. If the function calls and parameter conbinations match, continue. If they don't match, run the tests and generate a _results.csv file. See section "Generating results csv"
1. Find the corresponding module test of the module containing the fused op. Look for the test in models/demos/deepseek_v3/tests/test_*.py. Run the provided module test command to capture the baseline accuracy and verify that the test is working.
2. Identify the ttnn operations that will be fused into a new operation, this should be provided by the user / by the function the user provided
    1. Finde the sequence of ops in the module for decode; if it's unclear which ones should be fused, let the user know immediately.
    2. Find the sequence of ops in the module for prefill; if the same sequence of ops does not exist for prefill, then ignore all prefill instructions and inform the user in the final summary that this was a decode only fused op sequence.
3. Create a new file for the new fused op unit test under tests/fused_op_unit_tests/MODULE/test_NEW_FUSED_OP_NAME.py
4. In the fused op unit test file, create a PyTorch reference for the newly fused op based on the sequence of ttnn ops.
	- Make sure to use the PyTorch reference as a basis here. To figure out what reference code is used, look at the containing module's test, e.g. test_mla.py for a fused op that's contained in the MLA module. Then figure out which exact portion of the reference model's code corresponds to our new fused op. It might be necessary to modify the reference code slightly to get the correct reference implementation since the exact operations used in the ttnn model and in the reference model my differer.
	- In the fused op unit test file, create a new reference function "NEW_FUSED_OP_reference". Inputs/outputs to the function should be PyTorch tensors and function parameters of that sequence of ops; make sure to parameterize everything that's parameterized in the module as well, no hardcoding unless it is hardcoded in the model too.
5. In the fused op unit test file, create a function "NEW_FUSED_OP_ttnn" calling a function in the module code containing the sequence of ttnn ops that correspond to the new fused op. If there is no function in the module code that already contains the sequence of ops, then create a function there and call it from the unit test as well as the module itself; If it already esists just call it from the unit test. Inputs and outputs correspond to the inputs/output of the sequence of ops for the new fused op; inputs/outputs should be ttnn tensors and function parameters.
6. In the fused op unit test file, implement a pytest that:
	1. Creates all input tensors and function parameters (based on the actual use in the module code)
      - Parameters to the ops should be variables in an easily readable section in the code
      - Use pytest parameters for expected_pcc, expected_perf, mode+seqlen(add decode with seqlen 1, prefill with seqlen 128/1024/8k/32k/128k)
    2. Runs 100 iterations and checks/asserts PCC and ATOL (on the last iteration)
    3. Add a pytest parameter option to use real model weights/random weights
	  4. Tests the NEW_FUSED_OP_ttnn using the reference code in NEW_FUSED_OP_reference
    5. Contains a performance measurement wrapper, so that we measure device performance. Only measure for decode-1 and prefill-128. See following notes on performance measurements.
    6. Supports trace mode; add a pytest parameter to turn tracing on/off
      - If it fails due to trace_region_size being too small, set the trace_region_size pytest parameter (see example test) based on the required size as printed in the log.
    7. Contains a pytest parameter to turn program_caching on/off, this is only done when trace if off, with trace mode program_cache must be enabled too; use device.disable_and_clear_program_cache() for disabling, it's enabled by default
	8. Asserts on PCC, ATOL, performance
7. *Verify NEW_FUSED_OP_reference* and the test code itself by running the unit test and comparing pcc to NEW_FUSED_OP_ttnn. The PCC should typically by > 0.98, otherwise there's likely something wrong. Add the result of this including a log file in the final summary of work. Do not proceed further in the TODOs unless this passes.
	1. Update the expected_pcc with the pcc value from the test (if it's > 0.98 otherwise, debug the test to fix it!)
	2. Use the current perf as expected_perf but add a TODO comment to add the actual target (based on theoretical numbers)
8. *Verify* that the fused op unit test as well as the sequence of ops within the module *use the exact same configuration including input shapes, dtype, memory_config, and buffer type*. Add the result of this step including a log file in a folder "tests/fused_op_unit_tests/MODULE/test_results/FUSED_OP/verification" within the same module folder as the op test, where FUSED_OP is the fused op name. Update the test inputs if there are any mismatches, do not change the sequence in the module, consider this the ground truth.
    1. Run the device perf test of the fused op unit test for prefill (128 sequence length) and decode, copy the generated csv files into a the verification folder.
    2. Run the module test with 1 iteration, both for prefill (128 sequence length) and decode, copy the generated csv files into the test_results folder.
    3. Compare for each op that all properties are identical, i.e. fused op unit test and module test match in terms of op input/output properties both for prefill and for decode. Take a look at example_compare_fused_wqkva_configs.py to see how that was done for an example fused op unit test.
    4. Print a short summary for each input including all tensor properties (shape, dype, memory_config, buffer type, sharding details if sharded) and add a comment that explicitly states whether this check passed or failed.
8. Add a single device test if there is no CCL in the fused op; if there is a CCL, then add a comment describing this, skip the steps for singlle device tests below
    1. Create a new pytest in the file with the same name + "_single_device"
    2. The test takes the first device from the mesh_device fixture and runs the ops only on that device
    3. The input shape to the single device test is the chunk of the input from the multi device test that resides on the first device, hence the chunk that is processed but the first device. In order to find the correct shape, run the device perf test (multi device) and extract the input shape to the first op from the generated ops_perf_report, this is already the per device shape. Do the same for all matmul input_tensor_b shapes.
    4. Restructure the existing code to maintain a clean test that re-uses common parts of the code. Be very careful not to change any functionality in the existing test.
    5. Run the single device test and verify that both the PCC and the perf are the same as for the multi device test.
9. Add a single device test for device performance.
10. If single device tests are not skipped, *verify* the single device tests by running the single device - device perf test that generated the csv file and compare it to the multi device perf csv. All shapes must match, create a helper script to verify that. Add all artifacts to "tests/fused_op_unit_tests/MODULE/test_results/FUSED_OP/verification".
11. Add skips on CI (os.getenv("CI") == "true") to all tests except for {decode/prefill-128} + program_cache + tracing + real_weights + {normal version (accuracy check) / device perf}
12. Update the results, see section 'Update resutls' for instructions.
14. List anything that was unexpected and/or any workarouds you needed to make the fused op unit test work and print an overall summary of the restuls/links to the readme and the restuls file.

## Update resutls
Path results csv: 'models/demos/deepseek_v3/tests/fused_op_unit_tests/MODULE/test_results/OP_NAME_results.csv', where MODULE is the module and OP_NAME is the op name
Remove any previous results csv file for this op if existing, and remove all logs in the 'logs' subfolder for this op. Test all test funcitons and parameter configurations for the op. Since some tests take very long (especially the long sequence lengths), test each parameter combination separately; use CI=false. Create a csv summary of results in 'models/demos/deepseek_v3/tests/fused_op_unit_tests/MODULE/test_results/OP_NAME_results.csv', where MODULE is the module and OP_NAME is the op name. Before each run, reset the machine using 'tt-smi -glx_reset'. Set a 20 mins timeout, if you need a longer timeout update the timeout in AGENTS_GUIDE_ADD_TEST.md with the new timeout, look at the output log before killing the run, leave it running if the log changed within the last 10 mins. Use the following structure for the csv: test_name,status,pcc,e2e perf, device perf,failure_reason (optional),comment (optional), link, timestamp. Add one line per concrete test (parameter combination) with all details filled in. In 'comment' column add any comments for potential fixes if the test is failing or any other comments of interest. Copy the log file for each test configuration into the logs sub-folder and add a link to the log file to the 'link' column in the results csv. Update the README.md with the new status of the test, or create a new row if it did not exist before; see Update instructions in the README.md for more details.

## Performance measurements:
- Performance measurements use three metrics: e2e_duration, kernel_duration, op_to_op_latency
- e2e_duration: average duration of the whole fused op (sequence of ops) end to end. This is measured using profiler.start and profiler.end calls, use 10 warumup iterations and 100 measurement iterations.
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

## Skipping tests
- Make sure to skip the test by using marks=pytest.mark.skip(...); skipping inside the unit test takes very long due to device init/teardown and should be avoided
- All skips for long sequence lengths should only be skipped on CI, not on local execution or when testing the new test file and updating the README.md
- For tests that are not applicable (e.g. single device tests for ops containing a CCL) do not skip, instead just write a comment about the test not being applicable for that op including the reason

## Using TT hardware
- Before running check if the following env variables are set: MESH_DEVICE, DEEPSEEK_V3_CACHE, DEEPSEEK_V3_HF_MODEL; if not set run `source ../setup_metal.sh` and check again. Only if you still don't have the env variables set, stop and ask the user for details.
- If running tests, set a timeout of 15 minutes.
- If there's a machine issue, you'll need to reset the machine using "tt-smi -glx_reset"
- Important: Running the test may take a few minutes, if you kill it the device might need a reset. In general, if there is no log output for more than 5 minutes the test likely hangs and needs to be killed + device reset, but if there's log output keep it running.
- Run each job with piping the output to a log file, i.e. add " 2>&1 | tee $TT_METAL_HOME/logs/ds_mla_$(date +%Y%m%d_%H%M%S).log" to each command

## Summary of fused uni test features
  - Accuracy: compares ttnn vs reference with PCC and ATOL checks, asserts on both.
  - Iterations: runs 100 iterations; accuracy checks are asserted on the last iteration.
  - Modes/seqlen coverage: parameters for decode (seqlen 1) and prefill (128/1024/8k/32k/128k) when applicable.
  - Weights coverage: parameter to run with real model weights vs random weights.
  - Trace support: parameterized trace on/off; trace mode must have program cache on; adjustable trace region size if needed.
  - Program cache control: parameterized on/off (only when trace is off); uses device.disable_and_clear_program_cache() when
    disabling.
  - Performance checks: includes e2e perf (profiler start/end) and device perf (kernel_duration, op_to_op_latency) with warmup/
    measurement rules, asserts expected perf, and uploads metrics.
  - Decode/prefill perf modes: decode perf uses trace; prefill perf uses non-trace.
  - Single-device tests: if no CCL, includes a _single_device accuracy test and a single-device perf test, and verifies parity
    with multi-device results.
  - Config parity validation: unit test inputs/params must match module sequence exactly (shape, dtype, memory_config, buffer
    type, sharding), verified and documented.
  - CI skipping: all tests skipped on CI except decode/prefill-128 with program_cache + tracing + real_weights (accuracy + device
    perf).
