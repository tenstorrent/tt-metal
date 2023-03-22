# PyTorch Sweep Tests
This project contains infra for running sweep tests for ops in `gp.ai`.

## Description
Sweep tests are used to compare and validate `ttlib` ops against a PyTorch golden. It mainly consists of two parts:
- A `run_test` function that runs an op through `ttlib` and `pytorch` and compares the results.
- A lightweight wrapper around `run_test` that can perform sweeps of tests (mostly across a range of input shapes) and dump the results to output CSV's.

## Setup
Follow the `Getting Started` page to setup the `python_env`. Then, run:
```
source build/python_env/bin/activate
pip install -r python_api_testing/requirements.txt
```

## Running tests
The tests are run through a python script:
```
python python_api_testing/sweep_tests/run_pytorch_test.py -i python_api_testing/sweep_tests/test_configs/<test_config.yaml> -o <output_folder>
```

The inputs are:
- `-i, --input-test-config`: Input test config containing list of tests to run (see below for description on what this config should look like).
- `-o, --output-folder-path`: Optional folder name to dump result CSV's in. If not specified, it will default to `pytorch_test_folder`. If this folder (or any folder specified) already exists, the tests will not run.


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
    output-file: eltwise_add_sweep.csv
```

- _eltwise-add_: Maps to `ttlib` and `pytorch` ops in `python_api_testing/sweep_tests/op_map.py`
- _shape_ and _datagen_: Passed verbatim as dictionaries to `shapes_and_datagen` function in `python_api_testing/sweep_tests/common.py`.
  - This function is a generator that yields a matching list of shapes and datagen functions to sweep over for the test. The datagen function is mapped to functions in `python_api_testing/sweep_tests/generation_funcs.py` and used for populating the shapes with input data.
  - In general, a sweep is performed across the start and end shapes. Take a look at the code for how these fields are handled, but at a high-level:
    - `start-shape` and `end-shape`: Ranges for each dim; how the sweep happens depends on `method`.
      - `shape-list`: Use this field to ignore all other parameters and use hardcoded list of shapes.
    - `interval`: Defaults to 32; step used to sweep across each dim.
    - `num-shapes`: Used to duplicate (if applicable) the output shapes. For example, eltwise binary tests has two inputs. This is also used to duplicate the number of datagen functions if only one is provided.
    - `num-samples`: Defaults to `all`; if you provide a number, it will randomly select that number of shapes from the list of all available shapes.
    - `method`: Defaults to `default`; determines how the shapes are swept across
- _comparison_: Maps to `python_api_testing/sweep_tests/comparison_funcs.py`.
- _output-file_: Name of the output csv dumped inside the output folder. You can write results for additional tests to the same file if you provide the same output file path.
