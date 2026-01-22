# Sweep Framework

## Op Test Library

Sweep test module files are placed in the `tests/sweep_framework/sweeps/` folder. The name of the `.py` file will be the name of the module. They use the following template:

### Parameters

**LIMITATION: The suites must be made up of a maximum of 10,000 individual permutations. If you wish to test on more than 10,000 vectors, split the inputs into more than one suite.**

**LIMITATION: All parameters should be top level. Do not nest ttnn parameters in tuples or dictionaries, or any ttnn types within will not be serialized correctly. Instead, make them top level. If you require nested data because you want to avoid the cross-product of all individual parameters, split them into separate suites, and add in general parameters after.**

#### Example (INCORRECT, nested ttnn types in dicts/tuples, no separate suites):
```
parameters = {
    "default": {
        "matmul_specs": [
            # mcast 2d
            (
                (2, 3),
                (1600, 224, 896),
                False,
                dict(core_grid=ttnn.CoreGrid(y=5, x=7), strategy=ttnn.ShardStrategy.BLOCK),
                None,
            ),
            # mcast 2d transposed
            (
                (2, 3),
                (1600, 224, 896),
                False,
                dict(core_grid=ttnn.CoreGrid(y=5, x=7), strategy=ttnn.ShardStrategy.BLOCK, orientation=ttnn.ShardOrientation.COL_MAJOR),
                None,
            )
        ],
        "compute_kernel_config": [None],
        "input_a_dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
        "input_b_dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
        "output_dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
        "input_layout": [ttnn.TILE_LAYOUT],
    }
}
```

#### Example (CORRECT,  ttnn types are top level, and more sensible suite definitions are used):
```
# Matmul specific parameters
parameters = {
    "mcast_2d": {
        "batch_sizes": [(2, 3)],
        "input_shapes": [(1600, 224, 896)],
        "batch_matrix_multiply": [False],
        "input_a_sharded_core_grid": [ttnn.CoreGrid(y=5, x=7)],
        "input_a_sharded_strategy": [ttnn.ShardStrategy.BLOCK],
        "input_b_sharded_memory_config_specs": [None]
    },
    "mcast_2d_transposed": {
        "batch_sizes": [(2, 3)],
        "input_shapes": [(1600, 224, 896)],
        "batch_matrix_multiply": [False],
        "input_a_sharded_core_grid": [ttnn.CoreGrid(y=5, x=7)],
        "input_a_sharded_strategy": [ttnn.ShardStrategy.BLOCK],
        "input_a_sharded_orientation": [ttnn.ShardOrientation.COL_MAJOR],
        "input_b_sharded_memory_config_specs": [None]
    }
}

# Add general parameters
general = {
    "compute_kernel_config": [None],
    "input_a_dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
    "input_b_dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
    "output_dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
    "input_layout": [ttnn.TILE_LAYOUT]
}
for p in parameters.values():
    p.update(general)
```

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
                ),
            )
        ],
    },
}
```

Parameters must be separated into "suites" (in this example, suite_1 and suite_2 are the named suites). These suites allow for granularity when running test suites with the runner. A common use case may be to define post-commit, nightly, and weekly type suites where each suite is larger than the previous, and can be run less frequently.

Each suite dictionary must contain a list of input parameters, and all permutations of these will be generated.

It is possible to define generator functions for these input parameters separately, and pass the generators into the parameter field:

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

### Device Fixture
Each op test file can optionally have a `mesh_device_fixture` generator which will be picked up by the infra for when a developer wants to use a custom (multi-chip, mesh, or otherwise) device configuration.

This function should have two stages, setup and teardown of the device. These stages will be executed before, and after the test suite is executed. They are separated by the yield statement.

The `yield` statement in the generator should yield all of the devices at once, if there are multiple. This object will be passed to `run()` when the tests are executed as the `device` parameter.

The `yield` statement must give a tuple of your device object, and a label for this device configuration. See the below example.

#### Example

```
def mesh_device_fixture():
    # SETUP (called before test suite is executed)
    import tt_lib as ttl

    assert ttnn.get_num_devices() >= 8, "Not T3000!"

    num_devices_requested = 8

    mesh_device = ttnn.open_mesh_device(
        ttnn.MeshShape(1, num_devices_requested),
    )

    print("ADD: Opened device mesh")
    # YIELD to test infrastructure
    # IMPORTANT: Whatever device object(s) you want to pass to your run function need to be ONE object here, as this generator will only be referenced once before executing the tests.
    # i.e. If you have four separate devices to use in your test, use 'yield ([device1, device2, device3, device4], "4 Device Setup")' inside of a list.
    yield (mesh_device, "T3000 Mesh")

    # TEARDOWN (called after test suite is finished executing)
    print("ADD: Closing device mesh")

    ttnn.close_mesh_device(mesh_device)
    del mesh_device
```

### Run Function

The run function will be called by the test runner with all defined parameters passed in, along with the device.

This is where to define the test case itself including setup and teardown and golden comparison.

If you defined a `mesh_device_fixture` generator, the object you yielded will be passed into this function as `device`. Otherwise, `device` will be the default ttnn device opened by the infra.

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

The test vector generator takes in lists of parameters from each sweep test file and generates all permutations of these parameters, and stores serialized versions of them as suites in JSON files.

**NOTE: Elasticsearch support has been removed. Test vectors are now always exported to disk in JSON format under `tests/sweep_framework/vectors_export/`.**

The test vector ids are the sha224 hash of the vector itself. This ID is used to uniquely identify each vector, and detect changes in the suite. If the parameter generator is run multiple times on the same parameter set from the op test file, it will update the JSON files with the new vectors.

### Usage

To run the test vector generator:

`python3 tests/sweep_framework/sweeps_parameter_generator.py`

Options:

`--module-name <sweep_name>` OPTIONAL: Select the sweep file to generate parameters for. This should be only the name and not extension of the file. If not set, the generator will generate vectors for all sweep files in the sweeps folder.

`--dump-file` OPTIONAL: [DEPRECATED] This flag is deprecated and no longer needed. Vectors are always dumped to disk by default. Elasticsearch support has been removed.

`--tag <tag>` OPTIONAL: This setting is used to assign a custom tag that will be assigned to your test vectors. This is to keep copies of vectors separate from other developers / CI. By default, this will be your username. You are able to specify a tag when running tests using the runner.

## Test Runner

The test runner reads in test vectors from the test vector database and executes the tests sequentially by calling the op test's run function with the specified vectors.

### Features

- Hang Detection / Timeout: Default timeout for one single test is 30 seconds. This can be overridden by setting a global `TIMEOUT` variable in the test file. Test processes are killed after this timeout and tt-smi is automatically run to reset the chip after a hang, before continuing the test suite.
- NOTE ON HANGS: When specifying one test vector to run, hang detection will be disabled. This is because the test is run in the parent process to allow debug tools like gdb/lldb to be used easily.
- NOTE ON TT-SMI: To ensure the best stability, set the TT-SMI reset command in an environment variable `TT_SMI_RESET_COMMAND`. For example `TT_SMI_RESET_COMMAND="tt-smi -tr 0"`. If this is not set, the system will attempt to find tt-smi and use default flags, but this is not guaranteed to work on every system because of version mismatches of tt-smi.
- Result Classification: Tests will be assigned one of the following statuses after a run:
    1. PASS: The test met expected criteria. In this case the test message response is stored with the status, typically this is a PCC value.
    2. FAIL: ASSERT / EXCEPTION: The test failed due to an assertion in the op itself, failed PCC assertion, or any other exception that is raised during execution. The exception is stored with the test result.
    3. FAIL: CRASH / HANG: The test timed out and is assumed to be in a hung state.
    4. NOT RUN: The test was run with a vector that is marked as invalid. The invalid reason given from the op test file is stored with the result.
    5. FAIL_L1_OUT_OF_MEM: The test failed specifically due to an L1 Out of Memory error.
    6. FAIL_WATCHER: The test failed due to a Watcher raised exception. This only occurs if `--watcher` is passed in the run command.
- Memory Profiling: Per-core and device-level memory usage can be captured using the `--measure-memory` flag. This uses graph trace in NO_DISPATCH mode to profile memory requirements without execution overhead. Memory data is stored in test results with multiple granular metrics and can be analyzed in downstream dashboards. This is useful for:
    - Identifying memory-intensive operations and configurations per core
    - Tracking memory usage trends and regressions across commits
    - Understanding actual vs theoretical worst-case memory usage
    - Optimizing memory layouts and detecting L1 memory limit violations
    - Analyzing memory distribution patterns (sequential vs parallel execution)
- Granularity of Testing: Tests can be run by all sweeps, individual sweep, or individual suite to allow for faster/slower test runs, spanning larger/smaller suites of tests.
- Git Hash information is stored with each test run, so it is easy to see on which commit the test is breaking/passing.
- Data Aggregation: Results are accumulated in a database that can be queried to see desired details of test runs.


### Usage

**NOTE: Elasticsearch support has been removed from the parameter generator. Vectors are now always exported to disk in JSON format.**

Go to [`tests/README.md`](../README.md) for the latest information on how to run the sweeps_runner.

#### Memory Profiling

To capture per-core and device-level memory usage during sweep runs, add the `--measure-memory` flag:

```bash
python3 tests/sweep_framework/sweeps_runner.py \
  --module-name <module_name> \
  --vector-source vectors_export \
  --result-dest results_export \
  --measure-memory \
  --tag <your_tag>
```

**How it works:**
- Uses ttnn graph trace with NO_DISPATCH mode (fast, no device execution overhead)
- Captures both per-core and device-level memory metrics for each test vector
- Uses `extract_resource_usage_per_core()` for worst-case per-core analysis
- Uses `extract_peak_L1_memory_usage()` for actual observed device memory
- Memory capture failures are non-fatal (returns null but test continues)

**Memory metrics captured:**

| Metric Name | Description | Use Case |
|-------------|-------------|----------|
| `peak_l1_memory_per_core_bytes` | Peak total (CB+L1) per core | Per-core memory budget validation |
| `peak_cb_per_core_bytes` | Peak circular buffer per core | CB-specific analysis |
| `peak_l1_buffers_per_core_bytes` | Peak L1 buffer per core | L1-specific analysis |
| `num_cores` | Number of cores used | Context for other metrics |
| `peak_l1_memory_aggregate_bytes` | Worst-case if all cores peak together | Conservative capacity planning |
| `peak_l1_memory_device_bytes` | Actual observed peak across device | Realistic usage tracking |

**Understanding the metrics:**

- **Per-core metrics** (`peak_l1_memory_per_core_bytes`, etc.): Show worst-case memory per individual core
- **Aggregate** (`peak_l1_memory_aggregate_bytes`): Theoretical maximum if all cores peak simultaneously (= `peak_l1_memory_per_core_bytes × num_cores`)
- **Device** (`peak_l1_memory_device_bytes`): Actual peak observed during execution
- **Key insight**: If `aggregate ≈ device`, cores peak together (parallel). If `aggregate >> device`, execution is sequential.

**Example result with memory metrics:**
```json
{
  "status": "pass",
  "metrics": [
    {
      "metric_name": "peak_l1_memory_per_core_bytes",
      "metric_value": 18432
    },
    {
      "metric_name": "peak_cb_per_core_bytes",
      "metric_value": 18432
    },
    {
      "metric_name": "peak_l1_buffers_per_core_bytes",
      "metric_value": 0
    },
    {
      "metric_name": "num_cores",
      "metric_value": 64
    },
    {
      "metric_name": "peak_l1_memory_aggregate_bytes",
      "metric_value": 1179648
    },
    {
      "metric_name": "peak_l1_memory_device_bytes",
      "metric_value": 18432
    },
    {
      "metric_name": "e2e_perf_ms",
      "metric_value": 2.5
    }
  ]
}
```

**Interpreting results:**
In the example above:
- Each core uses at most 18KB (well within 256KB L1 limit ✓)
- Circular buffers account for all memory (18KB), L1 buffers are 0
- 64 cores are used
- Aggregate (1.18MB) >> Device (18KB), indicating **sequential execution** (only ~1 core active at a time)
- Low memory pressure, good efficiency

**Use cases:**
- Identify memory-intensive operation configurations
- Track memory usage trends across commits
- Detect memory regressions in CI
- Optimize memory layouts before running on hardware

## CI Execution

The sweep framework runs automatically in CI via the [ttnn - run sweeps](https://github.com/tenstorrent/tt-metal/actions/workflows/ttnn-run-sweeps.yaml) workflow. There are four distinct run types:

### Run Types

| Run Type | Schedule | Suite | Description |
|----------|----------|-------|-------------|
| **Nightly** | Mon, Tue, Thu, Fri @ 4:30 AM UTC | `nightly` | Regular sweep testing with standard parameters |
| **Comprehensive** | Wed, Sat @ 4:00 AM UTC | All suites | Exhaustive testing with all suite combinations |
| **Model Traced** | Daily @ 4:00 AM UTC | `model_traced` | Tests with configurations traced from real models |
| **Lead Models** | Daily @ 3:00 AM UTC | `model_traced` | Tests from prioritized lead models with multi-chip support |

### Lead Models

Lead models are prioritized models whose traced configurations receive special treatment in CI, including automatic routing to multi-chip runners.

**Configuration:** Lead models are defined in `tests/sweep_framework/framework/constants.py`:

```python
LEAD_MODELS = [
    "deepseek_v3",
]
```

To add a new lead model, add its directory name pattern to this list. The pattern is matched against the `source` path in traced operations.

**Generation:** To generate vectors for lead models only:

```bash
python3 tests/sweep_framework/sweeps_parameter_generator.py --model-traced lead --tag ci-main
```

### Multi-Chip Runner Assignment

For lead model runs, the framework automatically routes operations to appropriate hardware based on their `mesh_device_shape`:

| Mesh Shape | Description | Runner |
|------------|-------------|--------|
| `[1, 1]` | Single-chip | N150 |
| `[1, 2]`, `[1, 4]`, `[2, 4]` | Small multi-chip | Galaxy |
| `[4, 8]`, `[8, 4]` | 32-chip | Galaxy (topology-6u) |

**How it works:**

1. **Vector Generation:** The parameter generator reads `mesh_device_shape` from `traced_machine_info.tensor_placements` in the master JSON
2. **File Routing:** Vectors are grouped by mesh shape and written to separate files (e.g., `op__mesh_4x8.json`)
3. **Matrix Computation:** `framework/compute_sweep_matrix.py` creates a GitHub Actions matrix that maps mesh shapes to runners
4. **Execution:** CI spawns parallel jobs on appropriate hardware based on the matrix

**Runner Configuration:** The mesh-to-runner mapping is defined in `framework/compute_sweep_matrix.py`:

```python
def get_lead_models_mesh_runner_config():
    return [
        {
            "mesh_shapes": ["1x1"],
            "runs_on": "tt-ubuntu-2204-n150-stable",
            ...
        },
        {
            "mesh_shapes": ["1x2", "1x4", "2x4", "4x8", "8x4"],
            "runs_on": ["topology-6u", "arch-wormhole_b0", "in-service", "pipeline-functional"],
            ...
        },
    ]
```

For more details on the master JSON format and tensor placements, see the [Model Tracer README](../../model_tracer/README.md).

## FAQ / Troubleshooting

- If you see an error like the following, it means you did not re-create your environment using the `create_venv.sh` script. Either re-create your python environment, or manually install the dependencies using `pip install beautifultable termcolor`:
```
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
ModuleNotFoundError: No module named 'beautifultable'
```

- TTNN class nanobinds need to use the `tt_nanobind_class` wrapper to enable serialization/deserialization within this framework. There is a template in `tt_lib_bindings_tensor.cpp` which should replace `nb::class_` and automatically add these bindings to your type. (TODO: Move the template from this location to a common location.) Enum types do not need these nanobinds.

- Code within the `invalidate_vector` function or any code / generators used within `parameters` must NOT include any device code. These functions are intended to be run on CPU only, without access to a device. If you wish to filter tests by device architecture, see the [Device Fixture](#device-fixture) section.

- Before merging new tests / modified tests to main, please verify that your changes can generate tests and execute by testing locally, and also using the [ttnn - run sweeps](https://github.com/tenstorrent/tt-metal/actions/workflows/ttnn-run-sweeps.yaml) pipeline with your branch and modified test(s) selected.
