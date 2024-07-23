# Sweep Framework

## Op Test Library

Sweep test module files are placed in the `tests/sweep_framework/sweeps/` folder. The name of the `.py` file will be the name of the module. They use the following template:

### Parameters

**LIMITATION: The suites must be made up of a maximum of 10,000 individual permutations. If you wish to test on more than 10,000 vectors, split the inputs into more than one suite.**

#### Example
```
parameters = {
    "suite_1": {
        "batch_sizes": [(1,)],
        "height": [384, 1024],
        "width": [1024, 4096],
        "broadcast": [None, "h", "w", "hw"],
        "input_a_dtype": [ttnn.bfloat16],
        "input_b_dtype": [ttnn.bfloat16],
        "input_a_layout": [ttnn.TILE_LAYOUT],
        "input_b_layout": [ttnn.TILE_LAYOUT],
        "input_b_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
    },
    "suite_2": {
        "batch_sizes": [(1,)],
        "height": [1024, 1024],
        "width": [1024, 2048],
        "broadcast": [None, "h", "hw"],
        "input_a_dtype": [ttnn.bfloat16],
        "input_b_dtype": [ttnn.bfloat16],
        "input_a_layout": [ttnn.ROW_MAJOR_LAYOUT],
        "input_b_layout": [ttnn.ROW_MAJOR_LAYOUT],
        "input_b_memory_config": [ttnn.L1_MEMORY_CONFIG],
        "input_a_memory_config": [ttnn.L1_MEMORY_CONFIG],
        "output_memory_config": [
            ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(
                    ttnn.CoreRangeSet(
                        {
                            ttnn.CoreRange(ttnn.CoreCoord(1, 3), ttnn.CoreCoord(1, 4)),
                            ttnn.CoreRange(ttnn.CoreCoord(2, 3), ttnn.CoreCoord(2, 4)),
                        }
                    ),
                    [64, 64],
                    ttnn.ShardOrientation.ROW_MAJOR,
                    False,
                ),
            )
        ],
    },
}
```

Parameters must be seperated into "suites" (in this exmaple, dram and l1 are the named suites). These suites allow for granularity when running test suites with the runner. A common use case may be to define post-commit, nightly, and weekly type suites where each suite is larger than the previous, and can be run less frequently.

Each suite dictionary must contain a list of input parameters, and all permutations of these will be generated.

It is possible to define generator functions for these input parameters seperately, and pass the generators into the parameter field:

#### Example

```
def get_block_sharded_specs(
    batch_sizes_choices: List[Tuple[int, ...]],
    m_size_choices: List[int],
    k_size_choices: List[int],
    num_cores_choices: List[int],
) -> Tuple[Tuple[int, ...], int, int, int, int, int]:
    for batch_sizes, m_size, k_size in itertools.product(batch_sizes_choices, m_size_choices, k_size_choices):
        total_height = functools.reduce(operator.mul, batch_sizes) * m_size
        for per_core_height, num_cores_height in get_per_core_size_and_num_cores(
            total_height, num_cores_choices, max_per_core_size=1024
        ):
            for per_core_width, num_cores_width in get_per_core_size_and_num_cores(
                k_size, num_cores_choices, max_per_core_size=1024
            ):
                yield (batch_sizes, m_size, k_size, per_core_height, per_core_width, num_cores_height, num_cores_width)

"block_sharded_specs": list(
        get_block_sharded_specs(
            [(1,), (2,)],
            [x if x > 0 else 32 for x in range(0, 2048, 384)],
            [x if x > 0 else 32 for x in range(0, 2048, 256)],
            [1, 4, 7],
        )
    )
```

### Vector Validation
Each op test file can optionally have a `invalidate_vector` function which takes in a vector and determines if it is an invalid combination. The function must return a tuple of `(True, REASON STRING)` if the vector is invalid or `(False, None)` if the vector is valid.

#### Example
```
def invalidate_vector(test_vector) -> Tuple[bool, Optional[str]]:
    if test_vector["broadcast"] in {"w", "hw"} and test_vector["input_b_layout"] == ttnn.ROW_MAJOR_LAYOUT:
        return True, "Broadcasting along width is not supported for row major layout"
    return False, None
```

Vectors marked invalid will not be run by the test runner, but it will be recorded that it was skipped due to its invalidity.

### Run Function

The run function will be called by the test runner with all defined parameters passed in, along with the device.

This is where to define the test case itself including setup and teardown and golden comparison.

The runner expects one of two returns from the run function:

- Tuple[bool, Optional[str]] where the bool is the test pass/fail (True or False), and the string is the PCC. (e.g. (True, "0.999"))

- List[Tuple[bool, Optional[str]], Integer] where the tuple is the same as above, and the Integer is the e2e perf in nanoseconds.

See below example for how to measure the e2e perf and return it properly.

#### Example
```
from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time

def run(
    batch_sizes,
    height,
    width,
    broadcast,
    input_a_dtype,
    input_b_dtype,
    input_a_layout,
    input_b_layout,
    input_b_memory_config,
    input_a_memory_config,
    output_memory_config,
    *,
    device,
) -> Tuple[bool, Optional[str]]:
    input_shape_a = (*batch_sizes, height, width)
    input_shape_b = (*batch_sizes, height, width)
    if broadcast == "hw":
        input_shape_b = (*batch_sizes, 1, 1)
    elif broadcast == "h":
        input_shape_b = (*batch_sizes, 1, width)
    elif broadcast == "w":
        input_shape_b = (*batch_sizes, height, 1)

    torch_input_tensor_a = torch_random(input_shape_a, -0.1, 0.1, dtype=torch.float32)
    torch_input_tensor_b = torch_random(input_shape_b, -0.1, 0.1, dtype=torch.float32)
    torch_output_tensor = torch.add(torch_input_tensor_a, torch_input_tensor_b)

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=input_a_dtype,
        layout=input_a_layout,
        device=device,
        memory_config=input_b_memory_config,
    )
    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b,
        dtype=input_b_dtype,
        layout=input_b_layout,
        device=device,
        memory_config=input_a_memory_config,
    )

    start_time = start_measuring_time()
    output_tensor = ttnn.add(input_tensor_a, input_tensor_b, memory_config=output_memory_config)
    e2e_perf = stop_measuring_time(start_time)

    output_tensor = ttnn.to_torch(output_tensor)

    return [check_with_pcc(torch_output_tensor, output_tensor, 0.999), e2e_perf]
```

## Test Vector Generation

The test vector generator takes in lists of parameters from each sweep test file and generates all permutations of these parameters, and stores serialized versions of them as suites in the test vector database.

The test vectors are stored in a seperate Elasticsearch index based on the module. The test vector ids are the sha224 hash of the vector itself. This ID is used to uniquely identify each vector, and detect changes in the suite. If the parameter generator is run multiple times on the same parameter set from the op test file, only the first run will generate vectors. Only if the suite is changed, will the old vectors be marked "archived", and the suite will be updated with the new inputs. The runner will only detect and run vectors that are marked "current".

### Usage

**NOTE: The environemnt variable ELASTIC_PASSWORD must be set to connect to the Elasticsearch database which is used to store and retrieve test data.**

To run the test vector generator:

`python3 tests/sweep_framework/parameter_generator.py`

Options:

`--module-name <sweep_name>` OPTIONAL: Select the sweep file to generate parameters for. This should be only the name and not extension of the file. If not set, the generator will generate vectors for all sweep files in the sweeps folder.

`--elastic <elastic_url>` OPTIONAL: Default is `http://localhost:9200`, which in almost all cases should be overridden unless running with a local instance of Elasticsearch.

## Test Runner

The test runner reads in test vectors from the test vector database and executes the tests sequentially by calling the op test's run function with the specified vectors.

### Features

- Hang Detection / Timeout: Default timeout for one single test is 30 seconds. This can be overridden by setting a global `TIMEOUT` variable in the test file. Test processes are killed after this timeout and tt-smi is automatically run to reset the chip after a hang, before continuing the test suite.
- Result Classification: Tests will be assigned one of the following statuses after a run:
    1. PASS: The test met expected criteria. In this case the test message reponse is stored with the status, typically this is a PCC value.
    2. FAIL: ASSERT / EXCEPTION: The test failed due to an assertion in the op itself, failed PCC assertion, or any other exception that is raised during execution. The exception is stored with the test result.
    3. FAIL: CRASH / HANG: The test timed out and is assumed to be in a hung state.
    4. NOT RUN: The test was run with a vector that is marked as invalid. The invalid reason given from the op test file is stored with the result.
    5. FAIL_L1_OUT_OF_MEM: The test failed specifically due to an L1 Out of Memory error.
    6. FAIL_WATCHER: The test failed due to a Watcher raised exception. This only occurs if `--watcher` is passed in the run command.
- Granularity of Testing: Tests can be run by all sweeps, individual sweep, or individual suite to allow for faster/slower test runs, spanning larger/smaller suites of tests.
- Git Hash information is stored with each test run, so it is easy to see on which commit the test is breaking/passing.
- Data Aggreation: Results are accumulated in a database that can be queried to see desired details of test runs.


### Usage

**NOTE: The environemnt variable ELASTIC_PASSWORD must be set to connect to the Elasticsearch database which is used to store and retrieve test data.**

To run the test runner:
`python3 tests/sweep_framework/runner.py`

Options:

`--module-name <sweep_name>` OPTIONAL: If set, only the vectors that exist for the specified sweep will be run. If not set, all vectors for all sweeps will be run.

`--suite-name <suite_name>` OPTIONAL: This must be set in conjunction with module name. If set, only the vectors from the specified suite will be run.

`--elastic <elastic_url>` OPTIONAL: Default is `http://localhost:9200`, which in almost all cases should be overridden unless running with a local instance of Elasticsearch.

`--vector-id <vector_id>` OPTIONAL This must be set in conjunction with module name. If set, only the vector specified will be run.

`--arch <architecture>` REQUIRED: This will determine which tt-smi binaries to run on hangs. OPTIONS: `["grayskull", "wormhole", "wormhole_b0", "blackhole"]`

`--watcher` OPTIONAL: This will run the tests with Watcher enabled. Watcher logs will be written to `generated/watcher/` and any Watcher exceptions raised will be caught.

`--perf` OPTIONAL: This will enable e2e perf testing on the op tests that are written to support it. Each test will be run twice and the second result will be kept to avoid measuring compile time.

`--dry-run` EXPERIMENTAL: This flag will print all the test vectors that would be run without the flag. This is best used piping stdout to a text file to avoid flooding the terminal.

## Query Tool

**NOTE: The environemnt variable ELASTIC_PASSWORD must be set to connect to the Elasticsearch database which is used to store and retrieve test data.**

This tool is used to query the database to see information on test vectors and test runs.

You can specify which module, suite, or individual vector or run to query, and you will see varying levels of detail.

The summary command will show the test status totals from the latest run by default, or all runs if `--all` is included in the command.

The detail command will show a list of all individual test runs, which can be narrowed down by module and suite.

The vector and result commands will show a detailed view of the data, including each input parameter for a test vector, or a long form exception for result.

### Usage

`query.py [OPTIONS] COMMAND [ARGS]...`

Options:

  `--module-name TEXT`  Name of the module to be queried.

  `--suite-name TEXT`   Suite name to filter by.

  `--vector-id TEXT`    Individual Vector ID to filter by.

  `--run-id TEXT`       Individual Run ID to filter by.

  `--elastic TEXT`      Elastic Connection String

  `--all`               Displays total run statistics instead of the most recent run.

  `--help`              Show this message and exit.

Commands:
  `detail`
  `result`
  `summary`
  `vector`

#### Examples

```
$ python3 tests/sweep_framework/query.py --elastic http://172.18.0.2:9200 --module-name add summary

+------+------+-------------------------+-------------------+---------+
|      | PASS | FAIL (ASSERT/EXCEPTION) | FAIL (CRASH/HANG) | NOT RUN |
+------+------+-------------------------+-------------------+---------+
| dram |  16  |            0            |         0         |    0    |
+------+------+-------------------------+-------------------+---------+
|  l1  |  0   |            8            |         0         |    4    |
+------+------+-------------------------+-------------------+---------+
```

```
$ python3 tests/sweep_framework/query.py --elastic http://172.18.0.2:9200 --module-name matmul_default_sharded --all summary

+---------+------+-------------------------+-------------------+---------+
|         | PASS | FAIL (ASSERT/EXCEPTION) | FAIL (CRASH/HANG) | NOT RUN |
+---------+------+-------------------------+-------------------+---------+
| default | 192  |            0            |         0         |    0    |
+---------+------+-------------------------+-------------------+---------+
```

```
$ python3 tests/sweep_framework/query.py --elastic http://172.18.0.2:9200 --module-name add --suite-name dram summary

+----------------------+------+-------------------------+-------------------+---------+
| vector_id            | PASS | FAIL (ASSERT/EXCEPTION) | FAIL (CRASH/HANG) | NOT RUN |
+----------------------+------+-------------------------+-------------------+---------+
| 41a9k5ABKWg1nzaMCbUo |  1   |            0            |         0         |    0    |
+----------------------+------+-------------------------+-------------------+---------+
| 51a9k5ABKWg1nzaMCbVF |  1   |            0            |         0         |    0    |
+----------------------+------+-------------------------+-------------------+---------+
| 5Fa9k5ABKWg1nzaMCbU- |  1   |            0            |         0         |    0    |
+----------------------+------+-------------------------+-------------------+---------+
| 5Va9k5ABKWg1nzaMCbVA |  1   |            0            |         0         |    0    |
+----------------------+------+-------------------------+-------------------+---------+
| 5la9k5ABKWg1nzaMCbVD |  1   |            0            |         0         |    0    |
+----------------------+------+-------------------------+-------------------+---------+
| 61a9k5ABKWg1nzaMCbVP |  1   |            0            |         0         |    0    |
+----------------------+------+-------------------------+-------------------+---------+
| 6Fa9k5ABKWg1nzaMCbVI |  1   |            0            |         0         |    0    |
+----------------------+------+-------------------------+-------------------+---------+
| 6Va9k5ABKWg1nzaMCbVL |  1   |            0            |         0         |    0    |
+----------------------+------+-------------------------+-------------------+---------+
| 6la9k5ABKWg1nzaMCbVN |  1   |            0            |         0         |    0    |
+----------------------+------+-------------------------+-------------------+---------+
| 71a9k5ABKWg1nzaMCbVZ |  1   |            0            |         0         |    0    |
+----------------------+------+-------------------------+-------------------+---------+
| 7Fa9k5ABKWg1nzaMCbVR |  1   |            0            |         0         |    0    |
+----------------------+------+-------------------------+-------------------+---------+
| 7Va9k5ABKWg1nzaMCbVU |  1   |            0            |         0         |    0    |
+----------------------+------+-------------------------+-------------------+---------+
| 7la9k5ABKWg1nzaMCbVW |  1   |            0            |         0         |    0    |
+----------------------+------+-------------------------+-------------------+---------+
| 8Fa9k5ABKWg1nzaMCbVa |  1   |            0            |         0         |    0    |
+----------------------+------+-------------------------+-------------------+---------+
| 8Va9k5ABKWg1nzaMCbVd |  1   |            0            |         0         |    0    |
+----------------------+------+-------------------------+-------------------+---------+
| 8la9k5ABKWg1nzaMCbVf |  1   |            0            |         0         |    0    |
+----------------------+------+-------------------------+-------------------+---------+
```


```
$ python3 tests/sweep_framework/query.py --elastic http://172.18.0.2:9200 --module-name add detail

                        Sweep   Suite        Vector ID              Timestamp               Status                                              Details                                       Git Hash
---------------------- ------- ------- ---------------------- --------------------- ----------------------- -------------------------------------------------------------------------------- -----------
 t1bSmJABKWg1nzaMS7eo    add    dram    41a9k5ABKWg1nzaMCbUo   2024-07-09_18-47-13           PASS                                                 1.0                                         4e3cc801c
 uFbSmJABKWg1nzaMS7e5    add    dram    5Fa9k5ABKWg1nzaMCbU-   2024-07-09_18-47-13           PASS                                                 1.0                                         4e3cc801c
 uVbSmJABKWg1nzaMS7e7    add    dram    5Va9k5ABKWg1nzaMCbVA   2024-07-09_18-47-14           PASS                                                 1.0                                         4e3cc801c
 ulbSmJABKWg1nzaMS7e9    add    dram    5la9k5ABKWg1nzaMCbVD   2024-07-09_18-47-14           PASS                                                 1.0                                         4e3cc801c
 u1bSmJABKWg1nzaMS7fA    add    dram    51a9k5ABKWg1nzaMCbVF   2024-07-09_18-47-14           PASS                                                 1.0                                         4e3cc801c
 vFbSmJABKWg1nzaMS7fC    add    dram    6Fa9k5ABKWg1nzaMCbVI   2024-07-09_18-47-14           PASS                                                 1.0                                         4e3cc801c
 vVbSmJABKWg1nzaMS7fE    add    dram    6Va9k5ABKWg1nzaMCbVL   2024-07-09_18-47-14           PASS                                                 1.0                                         4e3cc801c
 vlbSmJABKWg1nzaMS7fG    add    dram    6la9k5ABKWg1nzaMCbVN   2024-07-09_18-47-14           PASS                                                 1.0                                         4e3cc801c
 v1bSmJABKWg1nzaMS7fI    add    dram    61a9k5ABKWg1nzaMCbVP   2024-07-09_18-47-14           PASS                                                 1.0                                         4e3cc801c
 wFbSmJABKWg1nzaMS7fL    add    dram    7Fa9k5ABKWg1nzaMCbVR   2024-07-09_18-47-14           PASS                                                 1.0                                         4e3cc801c
 wVbSmJABKWg1nzaMS7fN    add    dram    7Va9k5ABKWg1nzaMCbVU   2024-07-09_18-47-14           PASS                                                 1.0                                         4e3cc801c
 wlbSmJABKWg1nzaMS7fQ    add    dram    7la9k5ABKWg1nzaMCbVW   2024-07-09_18-47-15           PASS                                                 1.0                                         4e3cc801c
 w1bSmJABKWg1nzaMS7fS    add    dram    71a9k5ABKWg1nzaMCbVZ   2024-07-09_18-47-15           PASS                                                 1.0                                         4e3cc801c
 xFbSmJABKWg1nzaMS7fU    add    dram    8Fa9k5ABKWg1nzaMCbVa   2024-07-09_18-47-15           PASS                                                 1.0                                         4e3cc801c
 xVbSmJABKWg1nzaMS7fW    add    dram    8Va9k5ABKWg1nzaMCbVd   2024-07-09_18-47-15           PASS                                                 1.0                                         4e3cc801c
 xlbSmJABKWg1nzaMS7fY    add    dram    8la9k5ABKWg1nzaMCbVf   2024-07-09_18-47-15           PASS                                                 1.0                                         4e3cc801c
 x1bSmJABKWg1nzaMVbe2    add     l1     81a9k5ABKWg1nzaMCbVh   2024-07-09_18-47-18   FAIL_ASSERT_EXCEPTION                 TT_FATAL @ ../ttnn/cpp/ttnn/op_library/binary/bina                 4e3cc801c
 yFbSmJABKWg1nzaMVbfJ    add     l1     9Fa9k5ABKWg1nzaMCbV0   2024-07-09_18-47-18   FAIL_ASSERT_EXCEPTION                 TT_FATAL @ ../tt_metal/common/math.hpp:16: b > 0 b                 4e3cc801c
 yVbSmJABKWg1nzaMVbfM    add     l1     9Va9k5ABKWg1nzaMCbV3   2024-07-09_18-47-18          NOT_RUN          INVALID VECTOR: Broadcasting along width is not supported for row major layout   4e3cc801c
 ylbSmJABKWg1nzaMVbfN    add     l1     9la9k5ABKWg1nzaMCbWI   2024-07-09_18-47-18   FAIL_ASSERT_EXCEPTION                 TT_FATAL @ ../ttnn/cpp/ttnn/op_library/binary/bina                 4e3cc801c
 y1bSmJABKWg1nzaMVbfQ    add     l1     91a9k5ABKWg1nzaMCbWL   2024-07-09_18-47-18   FAIL_ASSERT_EXCEPTION                 TT_FATAL @ ../tt_metal/common/math.hpp:16: b > 0 b                 4e3cc801c
 zFbSmJABKWg1nzaMVbfT    add     l1     -Fa9k5ABKWg1nzaMCbWO   2024-07-09_18-47-18          NOT_RUN          INVALID VECTOR: Broadcasting along width is not supported for row major layout   4e3cc801c
 zVbSmJABKWg1nzaMVbfU    add     l1     -Va9k5ABKWg1nzaMCbWR   2024-07-09_18-47-18   FAIL_ASSERT_EXCEPTION                 TT_FATAL @ ../ttnn/cpp/ttnn/op_library/binary/bina                 4e3cc801c
 zlbSmJABKWg1nzaMVbfX    add     l1     -la9k5ABKWg1nzaMCbWU   2024-07-09_18-47-18   FAIL_ASSERT_EXCEPTION                 TT_FATAL @ ../tt_metal/common/math.hpp:16: b > 0 b                 4e3cc801c
 z1bSmJABKWg1nzaMVbfZ    add     l1     -1a9k5ABKWg1nzaMCbWW   2024-07-09_18-47-18          NOT_RUN          INVALID VECTOR: Broadcasting along width is not supported for row major layout   4e3cc801c
 0FbSmJABKWg1nzaMVbfa    add     l1     _Fa9k5ABKWg1nzaMCbWZ   2024-07-09_18-47-18   FAIL_ASSERT_EXCEPTION                 TT_FATAL @ ../ttnn/cpp/ttnn/op_library/binary/bina                 4e3cc801c
 0VbSmJABKWg1nzaMVbfc    add     l1     _Va9k5ABKWg1nzaMCbWb   2024-07-09_18-47-18   FAIL_ASSERT_EXCEPTION                 TT_FATAL @ ../tt_metal/common/math.hpp:16: b > 0 b                 4e3cc801c
 0lbSmJABKWg1nzaMVbff    add     l1     _la9k5ABKWg1nzaMCbWd   2024-07-09_18-47-18          NOT_RUN          INVALID VECTOR: Broadcasting along width is not supported for row major layout   4e3cc801c
```


```
$ python3 tests/sweep_framework/query.py --elastic http://172.18.0.2:9200 --module-name add --suite-name dram detail

    run_id              Sweep   Suite        Vector ID              Timestamp        Status   Details   Git Hash
---------------------- ------- ------- ---------------------- --------------------- -------- --------- -----------
 t1bSmJABKWg1nzaMS7eo    add    dram    41a9k5ABKWg1nzaMCbUo   2024-07-09_18-47-13    PASS      1.0     4e3cc801c
 uFbSmJABKWg1nzaMS7e5    add    dram    5Fa9k5ABKWg1nzaMCbU-   2024-07-09_18-47-13    PASS      1.0     4e3cc801c
 uVbSmJABKWg1nzaMS7e7    add    dram    5Va9k5ABKWg1nzaMCbVA   2024-07-09_18-47-14    PASS      1.0     4e3cc801c
 ulbSmJABKWg1nzaMS7e9    add    dram    5la9k5ABKWg1nzaMCbVD   2024-07-09_18-47-14    PASS      1.0     4e3cc801c
 u1bSmJABKWg1nzaMS7fA    add    dram    51a9k5ABKWg1nzaMCbVF   2024-07-09_18-47-14    PASS      1.0     4e3cc801c
 vFbSmJABKWg1nzaMS7fC    add    dram    6Fa9k5ABKWg1nzaMCbVI   2024-07-09_18-47-14    PASS      1.0     4e3cc801c
 vVbSmJABKWg1nzaMS7fE    add    dram    6Va9k5ABKWg1nzaMCbVL   2024-07-09_18-47-14    PASS      1.0     4e3cc801c
 vlbSmJABKWg1nzaMS7fG    add    dram    6la9k5ABKWg1nzaMCbVN   2024-07-09_18-47-14    PASS      1.0     4e3cc801c
 v1bSmJABKWg1nzaMS7fI    add    dram    61a9k5ABKWg1nzaMCbVP   2024-07-09_18-47-14    PASS      1.0     4e3cc801c
 wFbSmJABKWg1nzaMS7fL    add    dram    7Fa9k5ABKWg1nzaMCbVR   2024-07-09_18-47-14    PASS      1.0     4e3cc801c
 wVbSmJABKWg1nzaMS7fN    add    dram    7Va9k5ABKWg1nzaMCbVU   2024-07-09_18-47-14    PASS      1.0     4e3cc801c
 wlbSmJABKWg1nzaMS7fQ    add    dram    7la9k5ABKWg1nzaMCbVW   2024-07-09_18-47-15    PASS      1.0     4e3cc801c
 w1bSmJABKWg1nzaMS7fS    add    dram    71a9k5ABKWg1nzaMCbVZ   2024-07-09_18-47-15    PASS      1.0     4e3cc801c
 xFbSmJABKWg1nzaMS7fU    add    dram    8Fa9k5ABKWg1nzaMCbVa   2024-07-09_18-47-15    PASS      1.0     4e3cc801c
 xVbSmJABKWg1nzaMS7fW    add    dram    8Va9k5ABKWg1nzaMCbVd   2024-07-09_18-47-15    PASS      1.0     4e3cc801c
 xlbSmJABKWg1nzaMS7fY    add    dram    8la9k5ABKWg1nzaMCbVf   2024-07-09_18-47-15    PASS      1.0     4e3cc801c
 M1bZmJABKWg1nzaMM7hD    add    dram    41a9k5ABKWg1nzaMCbUo   2024-07-09_18-54-45    PASS      1.0     4e3cc801c
 NFbZmJABKWg1nzaMM7hI    add    dram    5Fa9k5ABKWg1nzaMCbU-   2024-07-09_18-54-46    PASS      1.0     4e3cc801c
 NVbZmJABKWg1nzaMM7hM    add    dram    5Va9k5ABKWg1nzaMCbVA   2024-07-09_18-54-46    PASS      1.0     4e3cc801c
 NlbZmJABKWg1nzaMM7hP    add    dram    5la9k5ABKWg1nzaMCbVD   2024-07-09_18-54-46    PASS      1.0     4e3cc801c
 N1bZmJABKWg1nzaMM7hR    add    dram    51a9k5ABKWg1nzaMCbVF   2024-07-09_18-54-47    PASS      1.0     4e3cc801c
 OFbZmJABKWg1nzaMM7hU    add    dram    6Fa9k5ABKWg1nzaMCbVI   2024-07-09_18-54-47    PASS      1.0     4e3cc801c
 OVbZmJABKWg1nzaMM7hV    add    dram    6Va9k5ABKWg1nzaMCbVL   2024-07-09_18-54-47    PASS      1.0     4e3cc801c
 OlbZmJABKWg1nzaMM7hX    add    dram    6la9k5ABKWg1nzaMCbVN   2024-07-09_18-54-47    PASS      1.0     4e3cc801c
 O1bZmJABKWg1nzaMM7hb    add    dram    61a9k5ABKWg1nzaMCbVP   2024-07-09_18-54-47    PASS      1.0     4e3cc801c
 PFbZmJABKWg1nzaMM7he    add    dram    7Fa9k5ABKWg1nzaMCbVR   2024-07-09_18-54-47    PASS      1.0     4e3cc801c
 PVbZmJABKWg1nzaMM7hg    add    dram    7Va9k5ABKWg1nzaMCbVU   2024-07-09_18-54-47    PASS      1.0     4e3cc801c
 PlbZmJABKWg1nzaMM7hi    add    dram    7la9k5ABKWg1nzaMCbVW   2024-07-09_18-54-47    PASS      1.0     4e3cc801c
 P1bZmJABKWg1nzaMM7hk    add    dram    71a9k5ABKWg1nzaMCbVZ   2024-07-09_18-54-47    PASS      1.0     4e3cc801c
 QFbZmJABKWg1nzaMM7hm    add    dram    8Fa9k5ABKWg1nzaMCbVa   2024-07-09_18-54-48    PASS      1.0     4e3cc801c
 QVbZmJABKWg1nzaMM7hp    add    dram    8Va9k5ABKWg1nzaMCbVd   2024-07-09_18-54-48    PASS      1.0     4e3cc801c
 QlbZmJABKWg1nzaMM7hq    add    dram    8la9k5ABKWg1nzaMCbVf   2024-07-09_18-54-48    PASS      1.0     4e3cc801c
```

```
$ python3 tests/sweep_framework/query.py --elastic http://172.18.0.2:9200 --module-name add --vector-id 6Va9k5ABKWg1nzaMCbVL vector

{'sweep_name': 'add',
 'timestamp': '2024-07-08_19-05-57',
 'batch_sizes': '(1,)',
 'height': '384',
 'width': '4096',
 'broadcast': 'w',
 'input_a_dtype': 'DataType.BFLOAT16',
 'input_b_dtype': 'DataType.BFLOAT16',
 'input_a_layout': 'Layout.TILE',
 'input_b_layout': 'Layout.TILE',
 'input_b_memory_config': {'type': 'tt_lib.tensor.MemoryConfig',
                           'memory_layout': 'TensorMemoryLayout.INTERLEAVED',
                           'buffer_type': 'BufferType.DRAM'},
 'input_a_memory_config': {'type': 'tt_lib.tensor.MemoryConfig',
                           'memory_layout': 'TensorMemoryLayout.INTERLEAVED',
                           'buffer_type': 'BufferType.DRAM'},
 'output_memory_config': {'type': 'tt_lib.tensor.MemoryConfig',
                          'memory_layout': 'TensorMemoryLayout.INTERLEAVED',
                          'buffer_type': 'BufferType.DRAM'},
 'suite_name': 'dram',
 'status': 'VectorStatus.VALID'}
```

```
$ python3 tests/sweep_framework/query.py --elastic http://172.18.0.2:9200 --module-name add --run-id TlbZmJABKWg1nzaMPbiV result

{'sweep_name': 'add',
 'suite_name': 'l1',
 'vector_id': '_la9k5ABKWg1nzaMCbWd',
 'status': 'TestStatus.NOT_RUN',
 'message': 'INVALID VECTOR: Broadcasting along width is not supported for row '
            'major layout',
 'timestamp': '2024-07-09_18-54-51',
 'git_hash': '4e3cc801c'}
```

## Database

Elasticsearch on Apache Lucene database. @Bill/@Raymond Kim to add more details.
