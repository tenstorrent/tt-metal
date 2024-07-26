# PyTorch Sweep Tests
This project contains infra for running sweep tests for ops in `tt_lib`.

## Description
Sweep tests are used to compare and validate `tt_lib` ops against a PyTorch golden. It mainly consists of two parts:
- A `run_test` function that runs an op through `tt_lib` and `pytorch` and compares the results.
- A lightweight wrapper around `run_test` that can perform sweeps of tests (mostly across a range of input shapes) and dump the results to output CSV's.

## Setup
Follow the `Getting Started` page to setup the `python_env`. Then, run:
```
source build/python_env/bin/activate
```

## single YAML file

The script used for single sweep run is
`tests/ttnn/python_api_testing/sweep_tests/run_sweep_test.py`

Command for running a single sweep test has the following pattern
`pytest test_script --input-path yaml_path --input-method cli --cli-input output_dir`

This script has two crucial parameters:
1) yaml_path - The YAML file defining op sweep test itself
Path pattern for ttnn ops:
`tests/ttnn/python_api_testing/sweep_tests/test_configs/ci_sweep_tests_state/device_name/my_sweep.yaml` 
Various test variants:
  1.1) state: a) working - Passes in all cases b)broken - Fails at least in one test case for any reason - Low PCC, specified or unspecified error
  1.2) device_name - Variant of the sweeps for specific device: a) grayskull b) wormhole

2) output_dir - Optional folder name to dump result CSV's in. If not specified, it will default to `pytorch_test_folder`. If this folder (or any folder specified) already exists, the tests will not run. In case if provided - after sweep run, here we can find the test results within the generated CSV file as outcome.


example:

`pytest tests/ttnn/python_api_testing/sweep_tests/run_sweep_test.py --input-path tests/ttnn/python_api_testing/sweep_tests/test_configs/ci_sweep_tests_working/grayskull/ttnn_eltwise_add_test.yaml --input-method cli --cli-input results_ttnn_add`

## multi YAML files

The script used for multiple YAML sweeps at once:
`tests/ttnn/python_api_testing/sweep_tests/run_sweep_tests.py`

The command pattern:
`python test_script -d path_to_yaml_dir -r output_dir`

This script has two crucial parameters:
1) path_to_yaml_dir - Folder containing the YAML files of the sweeps which we want to run at once
2) output_folder - After sweep run, here we can find the test results within the generated CSV file as outcome

example:

`python tests/ttnn/python_api_testing/sweep_tests/run_sweep_tests.py -d tests/ttnn/python_api_testing/sweep_tests/test_configs/ci_sweep_tests_working/grayskull/ -r result_sweeps`


## Adding test configs
Here is an example of a test config:
```
---
test-list:
  eltwise-add:
    shape:
      start-shape: [1, 1, 32, 32]
      end-shape: [1, 1, 64, 64]
      interval: 32
      num-shapes: 2
      num-samples: all
      args-sampling-strategy: "all"
      method: default
    datagen:
      function: gen_rand
      args:
        low: -100
        high: 100
    comparison:
      function: comp_pcc
      args:
        pcc: 0.99
    args-gen: gen_dtype_layout_device
    sanitize-args: True
    args:
        data-layout: ["TILE"]
        data-type: ["BFLOAT16"]
        buffer-type: ["DRAM", "L1", "SYSTEM_MEMORY"]
        out-buffer-type: ["DRAM"]
    output-file: eltwise_add_sweep.csv
```

- _eltwise-add_: Maps to `tt_lib` and `pytorch` ops in `python_api_testing/sweep_tests/op_map.py`
- _shape_ and _datagen_: Passed verbatim as dictionaries to `shapes_and_datagen` function in `python_api_testing/sweep_tests/common.py`. Data from _args_ dictionary is also passed to `shapes_and_datagen` to control data-layout, data-type and memory config of input arguments (also called input config).
  - `shapes_and_datagen` is a generator that yields a matching list of shapes, datagen functions, and input configs to sweep over for the test. The datagen function is mapped to functions in `python_api_testing/sweep_tests/generation_funcs.py` and used for populating the shapes with input data.
  - In general, a sweep is performed across the start and end shapes and inputs are setup according to data-layout, data-type and memory configs defined in _args_ section. Take a look at the code for how these fields are handled, but at a high-level:
    - `start-shape` and `end-shape`: Ranges for each dim; how the sweep happens depends on `method`.
      - `shape-list`: Use this field to ignore all other parameters and use hardcoded list of shapes.
    - `interval`: Defaults to 32; step used to sweep across each dim.
    - `num-shapes`: Number of input shapes (number of inputs). Eg. unary operation has `num-shapes` equal to 1, and binary operation has number of shapes equal to 2.
    - `num-samples`: Number of generated test runs with different inputs (different shapes and configs). Defaults to `all`, which means that sweep will go through all input shapes defined by `start-shape`, `end-shape` and `interval`. If number is provided there will be (roughly) `num-samples` test runs generated.
    - `args-sampling-strategy`: Strategy how arguments to operation will be configured. If set to `all` (which is default) sweep will go through all combinations for all inputs of data-layout, data-type, and memory configs defined in _args_ dictionary. If set to `random` sweep will randomly choose data-layout, data-type, and memory configs for each input to generate `num-samples` test runs.
    - `method`: Defaults to `default`; determines how the shapes are swept across
- _comparison_: Maps to `python_api_testing/sweep_tests/comparison_funcs.py`.
- _args-gen_: Maps to `python_api_testing/sweep_tests/generation_funcs.py`. You can choose what function is used to generate arguments for the test (dtype, layout etc). For example, for `glu` you might want to generate a `dim` parameter, so default `args-gen` won't be useful.
- _sanitize-args_: `True` if `args-gen` is doing arg sanitization, `False` otherwise. If not stated, the default is `True`. When ON, args sanitization won't allow some problematic combinations of args (Eg. `ROW_MAJOR` layout for `BFLOAT8_B` and `BFLOAT4_B`). But args sanitization might be an obstacle when we want to create some more flexible tests.
- _args_: Defines how arguments to operation can be configured in terms of data-layout, data_type and memory config.
  - `data-layout`: Data layout each input argument can take. Can be TILE or ROW_MAJOR.
  - `data-type`: Data type each input argument can take. Can be one of the follwoing: BFLOAT16, BFLOAT8_B, BFLOAT4_B, FLOAT32, UINT32, UINT16 or INT32.
  - `buffer-type`: Buffer type each input argument can take. Can be DRAM, L1, or SYSTEM_MEMORY.
  - `out-buffer-type`: Buffer type output can take. Can be DRAM, L1, or SYSTEM_MEMORY.
- _output-file_: Name of the output csv dumped inside the output folder. You can write results for additional tests to the same file if you provide the same output file path.


### _datagen_ section of test configs notes

_datagen_ section can be written for each input separately as well. Eg:

```
  ...
  datagen:
    - input_1:
      function: gen_rand
      args:
        low: -100
        high: 100
    - input_2:
      function: gen_rand
      args:
        low: -10
        high: 10
  ...
```


### _args_ section of test configs notes

_args_ section can be written for each input separately as well. Eg:

```
  comparison:
      function: comp_pcc
      args:
        pcc: 0.99
  args:
    inputs:
      - input-1:
        data-layout: ["TILE"]
        data-type: ["BFLOAT16"]
        buffer-type: ["DRAM", "L1", "SYSTEM_MEMORY"]
      - input-2:
        data-layout: ["TILE"]
        data-type: ["BFLOAT16"]
        buffer-type: ["DRAM", "L1", "SYSTEM_MEMORY"]
    out-buffer-type: ["DRAM", "L1"]
  output-file: eltwise_add_sweep.csv
```

### _coregrid_ coordinates range specification

It is also possible (optionally) to specify ranges for core grid coordinates in case of operations which support this kind of settings (based on TTNN operation documentation). On the test level, we are able to specify additional section ‘core grid’ in YAML file.

An example for _coregrid_ parameter specification is given (in this case for matmul operation)
```
  ...
  - ttnn-matmul:
      coregrid:
        xmin: 2
        xmax: 8
        ymin: 3
        ymax: 7
      shape:
        start-shape: [1, 1, 32, 32, 32]
        end-shape: [6, 12, 512, 512, 512]
  ...
```

In ‘coregrid’ section there is the set of parameters related to coregrid which can be specified:
- ‘xmin’ : minimal value of core grid’s x coordinate
- ‘xmax’: maximal value of core grid’s x coordinate
- ‘ymin’ : minimal value of core grid’s y coordinate
- ‘ymax’: maximal value of core grid’s y coordinate

As outcome, the resulting coordinates for x will be in range [xmin, xmax], while y will be in range [ymin, ymax]. For given example, x would be in range [2, 8], while y would be in range [3, 7].
