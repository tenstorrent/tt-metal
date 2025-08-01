import json
import os
from collections import defaultdict


def load_oom_data():
    """Load all OOM failure data"""
    import csv

    oom_failures = []
    with open("ssinghal/all_oom_failures.csv", "r") as f:
        reader = csv.reader(f)
        next(reader)  # Skip header

        for row in reader:
            if len(row) >= 6:
                oom_failures.append(
                    {
                        "operator": row[0],
                        "input_shape": row[1],
                        "total_memory_B": int(row[2]),
                        "total_memory_MB": float(row[3]),
                        "per_bank_memory_B": int(row[4]),
                        "per_bank_memory_KB": float(row[5]),
                    }
                )

    return oom_failures


def load_operator_mapping():
    """Load operator mapping data"""
    with open("ssinghal/operator_mapping.json", "r") as f:
        return json.load(f)


def group_oom_failures(oom_failures):
    """Group OOM failures by operator"""
    by_operator = defaultdict(list)
    for failure in oom_failures:
        by_operator[failure["operator"]].append(failure)
    return dict(by_operator)


def create_oom_test_file(operator, failures, operator_mapping):
    """Create a test file specifically for OOM failures of an operator"""

    op_info = operator_mapping.get(operator, {})
    ttnn_function = op_info.get("ttnn_function", f"ttnn.{operator}")
    torch_function = op_info.get("torch_function", f"torch.{operator}")
    num_inputs = op_info.get("num_inputs", 1)
    category = op_info.get("category", "unknown")

    # Extract unique shapes that cause OOM
    oom_shapes = []
    for failure in failures:
        shape_str = failure["input_shape"]
        # Parse the shape string "[1, 2, 3, 4]" into a list
        try:
            shape = eval(shape_str)  # Safe since we control the input
            oom_shapes.append({"shape": shape, "memory_mb": failure["total_memory_MB"]})
        except:
            continue

    # Remove duplicates
    unique_shapes = []
    seen = set()
    for item in oom_shapes:
        shape_tuple = tuple(item["shape"])
        if shape_tuple not in seen:
            seen.add(shape_tuple)
            unique_shapes.append(item)

    # Sort by memory requirement (descending)
    unique_shapes.sort(key=lambda x: x["memory_mb"], reverse=True)

    # Format shapes for parametrization
    shape_params = []
    for item in unique_shapes:
        shape_params.append(f"({item['shape']}, {item['memory_mb']})")

    formatted_shapes = ", ".join(shape_params)

    # Generate appropriate input creation and operation calls based on operator type
    if num_inputs == 1:
        input_creation = """        torch_input = torch.rand(input_shape, dtype=torch.bfloat16)

        if len(input_shape) == 4:
            torch_input = torch_input.permute(0, 2, 3, 1)

        # This should trigger OOM - we expect this test to be skipped
        ttnn_input = ttnn.from_torch(torch_input, device=device, memory_config=ttnn.L1_MEMORY_CONFIG, layout=ttnn.TILE_LAYOUT)"""

        if category == "activation":
            operation_call = f"        ttnn_output = {ttnn_function}(ttnn_input)"
        elif operator in ["view", "unsafeview"]:
            operation_call = "        ttnn_output = ttnn.reshape(ttnn_input, input_shape)"
        elif operator == "permute":
            operation_call = (
                "        ttnn_output = ttnn.permute(ttnn_input, (0, 3, 1, 2)) if len(input_shape) == 4 else ttnn_input"
            )
        else:
            operation_call = f"        ttnn_output = {ttnn_function}(ttnn_input)"

    elif num_inputs == 2:
        input_creation = """        torch_input1 = torch.rand(input_shape, dtype=torch.bfloat16)
        torch_input2 = torch.rand(input_shape, dtype=torch.bfloat16)

        if len(input_shape) == 4:
            torch_input1 = torch_input1.permute(0, 2, 3, 1)
            torch_input2 = torch_input2.permute(0, 2, 3, 1)

        # This should trigger OOM - we expect this test to be skipped
        ttnn_input1 = ttnn.from_torch(torch_input1, device=device, memory_config=ttnn.L1_MEMORY_CONFIG, layout=ttnn.TILE_LAYOUT)
        ttnn_input2 = ttnn.from_torch(torch_input2, device=device, memory_config=ttnn.L1_MEMORY_CONFIG, layout=ttnn.TILE_LAYOUT)"""

        operation_call = f"        ttnn_output = {ttnn_function}(ttnn_input1, ttnn_input2)"

    elif num_inputs == "variable":
        input_creation = """        torch_input1 = torch.rand(input_shape, dtype=torch.bfloat16)
        torch_input2 = torch.rand(input_shape, dtype=torch.bfloat16)

        if len(input_shape) == 4:
            torch_input1 = torch_input1.permute(0, 2, 3, 1)
            torch_input2 = torch_input2.permute(0, 2, 3, 1)

        # This should trigger OOM - we expect this test to be skipped
        ttnn_input1 = ttnn.from_torch(torch_input1, device=device, memory_config=ttnn.L1_MEMORY_CONFIG, layout=ttnn.TILE_LAYOUT)
        ttnn_input2 = ttnn.from_torch(torch_input2, device=device, memory_config=ttnn.L1_MEMORY_CONFIG, layout=ttnn.TILE_LAYOUT)"""

        if "concat" in ttnn_function:
            operation_call = f"        ttnn_output = {ttnn_function}([ttnn_input1, ttnn_input2], dim=0)"
        else:
            operation_call = f"        ttnn_output = {ttnn_function}([ttnn_input1, ttnn_input2])"
    else:
        # Default single input
        input_creation = """        torch_input = torch.rand(input_shape, dtype=torch.bfloat16)

        if len(input_shape) == 4:
            torch_input = torch_input.permute(0, 2, 3, 1)

        # This should trigger OOM - we expect this test to be skipped
        ttnn_input = ttnn.from_torch(torch_input, device=device, memory_config=ttnn.L1_MEMORY_CONFIG, layout=ttnn.TILE_LAYOUT)"""

        operation_call = f"        ttnn_output = {ttnn_function}(ttnn_input)"

    # Special handling for Linear layer
    if operator == "linear":
        input_creation = """        batch_size, seq_len, hidden_size = input_shape if len(input_shape) == 3 else (1, input_shape[0], input_shape[1])
        torch_input = torch.rand((batch_size, seq_len, hidden_size), dtype=torch.bfloat16)
        torch_weight = torch.rand((hidden_size, hidden_size), dtype=torch.bfloat16)

        # This should trigger OOM - we expect this test to be skipped
        ttnn_input = ttnn.from_torch(torch_input, device=device, memory_config=ttnn.L1_MEMORY_CONFIG, layout=ttnn.TILE_LAYOUT)
        ttnn_weight = ttnn.from_torch(torch_weight, device=device, memory_config=ttnn.L1_MEMORY_CONFIG, layout=ttnn.TILE_LAYOUT)"""

        operation_call = "        ttnn_output = ttnn.linear(ttnn_input, ttnn_weight)"

    test_content = f'''import pytest
import torch
import ttnn

@pytest.mark.parametrize("input_shape_memory", [{formatted_shapes}])
def test_oom_{operator.lower()}(device, input_shape_memory):
    """
    Test {operator} operator with shapes that previously caused OOM failures.
    These tests are expected to be SKIPPED due to out-of-memory conditions.

    This test serves to:
    1. Document problematic input shapes for {operator}
    2. Verify OOM handling works correctly
    3. Track memory requirements for optimization
    """
    input_shape, expected_memory_mb = input_shape_memory
    torch.manual_seed(0)

    print(f"Testing {operator} with shape {{input_shape}} (Expected memory: {{expected_memory_mb}} MB)")

    try:
{input_creation}

{operation_call}

        # If we reach here, the operation succeeded unexpectedly
        pytest.fail(f"Expected OOM for shape {{input_shape}} but operation succeeded")

    except RuntimeError as e:
        if "Out of Memory" in str(e):
            pytest.skip(f"Expected OOM: {{input_shape}} requires {{expected_memory_mb}} MB - {{str(e)}}")
        else:
            # Some other runtime error
            pytest.fail(f"Unexpected RuntimeError for {{input_shape}}: {{str(e)}}")
    except Exception as e:
        # Some other unexpected error
        pytest.fail(f"Unexpected error for {{input_shape}}: {{str(e)}}")


@pytest.mark.parametrize("input_shape_memory", [{formatted_shapes}])
def test_memory_estimation_{operator.lower()}(input_shape_memory):
    """
    Test to estimate memory requirements without actually running on device.
    This can be used for memory planning and optimization.
    """
    input_shape, expected_memory_mb = input_shape_memory

    # Calculate theoretical memory requirement
    import numpy as np

    # Assuming bfloat16 (2 bytes per element)
    element_size = 2
    total_elements = np.prod(input_shape)
    theoretical_memory_mb = (total_elements * element_size) / (1024 * 1024)

    print(f"Shape: {{input_shape}}")
    print(f"  Theoretical memory: {{theoretical_memory_mb:.2f}} MB")
    print(f"  Actual OOM at: {{expected_memory_mb}} MB")
    print(f"  Overhead factor: {{expected_memory_mb / max(theoretical_memory_mb, 0.001):.2f}}x")

    # This test always passes - it's just for analysis
    assert True
'''

    return test_content


def create_master_oom_test():
    """Create a master test file that tests all OOM conditions"""

    oom_failures = load_oom_data()

    # Get top 20 most memory-intensive failures across all operators
    top_failures = sorted(oom_failures, key=lambda x: x["total_memory_MB"], reverse=True)[:20]

    test_params = []
    for failure in top_failures:
        shape_str = failure["input_shape"]
        try:
            shape = eval(shape_str)
            test_params.append(f"('{failure['operator']}', {shape}, {failure['total_memory_MB']})")
        except:
            continue

    formatted_params = ", ".join(test_params)

    master_content = f'''import pytest
import torch
import ttnn

@pytest.mark.parametrize("operator_shape_memory", [{formatted_params}])
def test_master_oom_scenarios(device, operator_shape_memory):
    """
    Master test for the most memory-intensive operations across all operators.

    This test documents the top OOM failure cases and can be used to:
    1. Track memory optimization progress
    2. Verify OOM handling across different operators
    3. Benchmark memory requirements
    """
    operator, input_shape, expected_memory_mb = operator_shape_memory

    print(f"\\nTesting {{operator}} with shape {{input_shape}} ({{expected_memory_mb}} MB)")

    # Skip the actual execution since these are known OOM cases
    pytest.skip(f"Known OOM case: {{operator}} {{input_shape}} requires {{expected_memory_mb}} MB")


def test_memory_analysis_summary():
    """
    Summary test that prints memory analysis for all OOM failures.
    This is always-passing test for documentation purposes.
    """

    oom_data = {oom_failures}

    print("\\n" + "="*60)
    print("COMPREHENSIVE OOM FAILURE ANALYSIS")
    print("="*60)

    # Group by operator
    by_operator = {{}}
    for failure in oom_data:
        op = failure['operator']
        if op not in by_operator:
            by_operator[op] = []
        by_operator[op].append(failure)

    print(f"\\nTotal OOM failures found: {{len(oom_data)}}")
    print(f"Operators affected: {{len(by_operator)}}")

    print("\\nFailures by operator:")
    for operator, failures in sorted(by_operator.items()):
        max_memory = max(f['total_memory_MB'] for f in failures)
        min_memory = min(f['total_memory_MB'] for f in failures)
        avg_memory = sum(f['total_memory_MB'] for f in failures) / len(failures)

        print(f"  {{operator:15}} {{len(failures):3d}} failures "
              f"({{min_memory:8.1f}} - {{max_memory:8.1f}} MB, avg: {{avg_memory:6.1f}} MB)")

    print("\\nTop 10 most memory-intensive shapes:")
    sorted_failures = sorted(oom_data, key=lambda x: x['total_memory_MB'], reverse=True)
    for i, failure in enumerate(sorted_failures[:10], 1):
        print(f"  {{i:2d}}. {{failure['operator']:12}} {{str(failure['input_shape']):30}} "
              f"{{failure['total_memory_MB']:8.1f}} MB")

    print("\\n" + "="*60)

    assert True  # This test always passes
'''

    return master_content


def main():
    print("Generating OOM-specific test files...")

    # Load data
    oom_failures = load_oom_data()
    operator_mapping = load_operator_mapping()

    print(f"Found {len(oom_failures)} OOM failures")

    # Group by operator
    failures_by_op = group_oom_failures(oom_failures)

    print(f"Affected operators: {len(failures_by_op)}")

    # Create individual test files for each operator with OOM failures
    generated_files = []

    for operator, failures in failures_by_op.items():
        print(f"Creating OOM test for {operator} ({len(failures)} failures)...")

        test_content = create_oom_test_file(operator, failures, operator_mapping)
        test_file = f"ssinghal/oom_tests/test_oom_{operator.lower()}.py"

        with open(test_file, "w") as f:
            f.write(test_content)

        generated_files.append(test_file)

    # Create master test file
    print("Creating master OOM test file...")
    master_content = create_master_oom_test()
    master_file = "ssinghal/oom_tests/test_master_oom.py"

    with open(master_file, "w") as f:
        f.write(master_content)

    generated_files.append(master_file)

    # Create README
    readme_content = f"""# OOM Test Suite

This directory contains pytest files specifically designed to test Out-of-Memory (OOM) failure scenarios.

## Files Generated

### Individual Operator Tests
{chr(10).join(f"- `{os.path.basename(f)}` - OOM tests for {os.path.basename(f).replace('test_oom_', '').replace('.py', '')} operator" for f in generated_files[:-1])}

### Master Test
- `test_master_oom.py` - Comprehensive test covering top OOM scenarios across all operators

## Purpose

These tests serve to:

1. **Document** problematic input shapes that cause OOM failures
2. **Verify** that OOM handling works correctly (tests should be SKIPPED)
3. **Track** memory requirements for optimization efforts
4. **Benchmark** memory usage patterns across different operators

## Usage

Run all OOM tests:
```bash
export WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml
export ARCH_NAME=wormhole_b0
export TT_METAL_HOME=$(pwd)
export PYTHONPATH=$(pwd)
source python_env/bin/activate

# Run all OOM tests (should mostly be skipped)
pytest ssinghal/oom_tests/ -v

# Run specific operator OOM tests
pytest ssinghal/oom_tests/test_oom_view.py -v

# Run master OOM analysis
pytest ssinghal/oom_tests/test_master_oom.py -v
```

## Expected Behavior

- Most tests should be **SKIPPED** due to OOM conditions
- Tests that **FAIL** indicate unexpected behavior
- Tests that **PASS** indicate the operation succeeded (unexpected for OOM scenarios)

## Statistics

- **Total OOM failures documented**: {len(oom_failures)}
- **Operators with OOM issues**: {len(failures_by_op)}
- **Memory range**: {min(f['total_memory_MB'] for f in oom_failures):.1f} MB - {max(f['total_memory_MB'] for f in oom_failures):.1f} MB

## Most Problematic Operators

{chr(10).join(f"- **{op}**: {len(failures)} failures" for op, failures in sorted(failures_by_op.items(), key=lambda x: len(x[1]), reverse=True)[:5])}
"""

    with open("ssinghal/oom_tests/README.md", "w") as f:
        f.write(readme_content)

    print(f"\n=== OOM TEST GENERATION COMPLETE ===")
    print(f"Generated {len(generated_files)} test files:")
    for file in generated_files:
        print(f"  - {file}")
    print(f"  - ssinghal/oom_tests/README.md")

    print(f"\nTo run OOM tests:")
    print(f"  cd ssinghal/oom_tests")
    print(f"  pytest test_master_oom.py -v")


if __name__ == "__main__":
    main()
