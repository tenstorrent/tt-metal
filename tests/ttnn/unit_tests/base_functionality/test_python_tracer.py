# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import json
import shutil
import pytest
import torch

import ttnn
import ttnn.operation_tracer

pytestmark = [pytest.mark.use_module_device, pytest.mark.requires_fast_runtime_mode_off]

# Constants for operation names used in trace file matching
TRACE_FILE_PATTERN_TTNN_RAND = "ttnn_rand"
TRACE_FILE_PATTERN_TTNN_ADD = "ttnn_add"
TRACE_FILE_PATTERN_TTNN_FROM_TORCH = "ttnn_from_torch"
TRACE_FILE_PATTERN_TTNN_TO_DEVICE = "ttnn_to_device"
OPERATION_NAME_TTNN_RAND = "ttnn.rand"
OPERATION_NAME_TTNN_ADD = "ttnn.add"
OPERATION_NAME_TTNN_FROM_TORCH = "ttnn.from_torch"
OPERATION_NAME_TTNN_TO_DEVICE = "ttnn.to_device"


def count_elements(obj) -> int:
    """Recursively count elements in a nested list structure."""
    if isinstance(obj, list):
        return sum(count_elements(item) for item in obj)
    else:
        return 1


@pytest.fixture(scope="function", autouse=True)
def enable_tracing_for_test(request):
    """Enable tracing for each test and restore it after.

    Only enables tracing for tests that need it (test_operation_parameter_tracing).
    Other tests will have tracing disabled.
    """
    # Save original state
    original_trace_flag = ttnn.operation_tracer._ENABLE_TRACE
    original_serialize_values = ttnn.operation_tracer._SERIALIZE_TENSOR_VALUES

    # Only enable tracing for the specific test that needs it
    if (
        "test_operation_parameter_tracing" in request.node.name
        or "test_default_no_tensor_values" in request.node.name
        or "test_from_torch_to_device_tracing" in request.node.name
    ):
        ttnn.operation_tracer._ENABLE_TRACE = True
        # For test_operation_parameter_tracing, enable tensor value serialization (to test with values)
        # For test_default_no_tensor_values, use default (False - no values) to test default behavior
        if "test_operation_parameter_tracing" in request.node.name:
            ttnn.operation_tracer.enable_tensor_value_serialization(True)
        else:
            # Use default (False) - no values serialized by default
            ttnn.operation_tracer.enable_tensor_value_serialization(False)
    else:
        ttnn.operation_tracer._ENABLE_TRACE = False

    yield

    # Restore original state after test
    ttnn.operation_tracer._ENABLE_TRACE = original_trace_flag
    ttnn.operation_tracer.enable_tensor_value_serialization(original_serialize_values)


@pytest.mark.parametrize(
    "shape_a,shape_b,dtype",
    [
        ([2, 3], [2, 3], ttnn.bfloat16),
        ([4, 5], [4, 5], ttnn.bfloat16),
        ([1, 8], [1, 8], ttnn.float32),
        ([3, 4], [3, 4], ttnn.int32),
        ([2, 2, 2], [2, 2, 2], ttnn.bfloat16),
    ],
    ids=["2x3_bfloat16", "4x5_bfloat16", "1x8_float32", "3x4_int32", "2x2x2_bfloat16"],
)
def test_operation_parameter_tracing(tmp_path, device, shape_a, shape_b, dtype):
    """Test that operation parameters are traced when --trace-params flag is used."""
    # Reset counter for predictable test numbering
    # Note: Tracing is enabled by the autouse fixture
    original_operation_counter = ttnn.operation_tracer._OPERATION_COUNTER
    ttnn.operation_tracer._OPERATION_COUNTER = 0

    # Override the log directory in CONFIG to use tmp_path
    original_report_path = None
    if hasattr(ttnn.CONFIG, "root_report_path"):
        original_report_path = ttnn.CONFIG.root_report_path
        ttnn.CONFIG.root_report_path = str(tmp_path)

    # The trace directory will be created under root_report_path / "operation_parameters"
    trace_dir = tmp_path / "operation_parameters"

    tensor_a = ttnn.rand(shape=shape_a, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tensor_b = ttnn.rand(shape=shape_b, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    # Execute add operation - result is traced but not used in test assertions
    ttnn.add(tensor_a, tensor_b)
    ttnn.synchronize_device(device)

    # Find the trace files (format: {number}_{operation_name}_{timestamp}.json)
    trace_files = sorted(trace_dir.glob("*_ttnn_*.json")) if trace_dir.exists() else []

    # Find rand and add operation files
    rand_files = [f for f in trace_files if TRACE_FILE_PATTERN_TTNN_RAND in f.name]
    add_files = [f for f in trace_files if TRACE_FILE_PATTERN_TTNN_ADD in f.name]

    assert len(rand_files) >= 2, f"Expected at least 2 rand trace files, found {len(rand_files)}"
    assert len(add_files) >= 1, f"Expected at least 1 add trace file, found {len(add_files)}"

    # Check the first rand operation trace
    rand_file_1 = rand_files[0]
    with open(rand_file_1, "r") as f:
        operation_data_1 = json.load(f)

    assert operation_data_1["operation_name"] == OPERATION_NAME_TTNN_RAND
    assert "operation_id" in operation_data_1, "Expected operation_id in trace file"

    # Check that shape parameter is captured
    # The shape should be in kwargs (ttnn.rand takes all params as kwargs)
    assert "kwargs" in operation_data_1, "Expected kwargs in operation data"
    assert "shape" in operation_data_1["kwargs"], "Expected shape in kwargs"
    shape_value = operation_data_1["kwargs"]["shape"]
    assert shape_value == shape_a, f"Expected shape {shape_a}, got {shape_value}"

    # Note: ttnn.rand doesn't have tensor parameters (it creates and returns a tensor)

    # Check the second rand operation trace
    rand_file_2 = rand_files[1]
    with open(rand_file_2, "r") as f:
        operation_data_2 = json.load(f)

    assert operation_data_2["operation_name"] == OPERATION_NAME_TTNN_RAND
    shape_value_2 = operation_data_2["kwargs"]["shape"]
    assert shape_value_2 == shape_b, f"Expected shape {shape_b}, got {shape_value_2}"

    # Check the add operation trace
    add_file = add_files[0]
    with open(add_file, "r") as f:
        operation_data_add = json.load(f)

    assert operation_data_add["operation_name"] == OPERATION_NAME_TTNN_ADD
    assert len(operation_data_add["args"]) >= 2, "Expected at least 2 args for add operation"

    # Check that both tensors a and b are captured
    # They should be in args as tensor data (now embedded directly in JSON)
    arg_0 = operation_data_add["args"][0]["value"]
    arg_1 = operation_data_add["args"][1]["value"]

    assert arg_0["type"] == "ttnn.Tensor", f"Expected tensor type, got {arg_0.get('type')}"
    assert arg_1["type"] == "ttnn.Tensor", f"Expected tensor type, got {arg_1.get('type')}"

    # Verify tensor data contains correct information
    for i, tensor_data in enumerate([arg_0, arg_1]):
        # Check that required fields exist
        assert "shape" in tensor_data, f"Tensor data {i} missing 'shape' field"
        assert "dtype" in tensor_data, f"Tensor data {i} missing 'dtype' field"
        assert "values" in tensor_data, f"Tensor data {i} missing 'values' field"

        # Get expected shape
        expected_shape = shape_a if i == 0 else shape_b
        actual_shape = tensor_data["shape"]

        # Verify shape matches
        assert (
            actual_shape == expected_shape
        ), f"Shape mismatch in tensor {i}: expected {expected_shape}, got {actual_shape}"

        # Calculate expected number of elements
        expected_elements = 1
        for dim in expected_shape:
            expected_elements *= dim

        # Verify values structure matches shape
        values = tensor_data["values"]

        actual_elements = count_elements(values)
        assert (
            actual_elements == expected_elements
        ), f"Element count mismatch in tensor {i}: expected {expected_elements} elements, got {actual_elements}"

        # Verify values are numeric (float or int)
        def verify_numeric(obj):
            if isinstance(obj, list):
                for item in obj:
                    verify_numeric(item)
            else:
                assert isinstance(obj, (int, float)), f"Expected numeric value, got {type(obj)}: {obj}"

        verify_numeric(values)

        # Verify dtype is a string
        assert isinstance(tensor_data["dtype"], str), f"Expected dtype to be string, got {type(tensor_data['dtype'])}"

    # Verify return value is captured for add operation
    assert "return_value" in operation_data_add, "Expected return_value in add operation data"
    return_value_data = operation_data_add["return_value"]
    assert (
        return_value_data["type"] == "ttnn.Tensor"
    ), f"Expected return value to be tensor, got {return_value_data.get('type')}"
    assert "shape" in return_value_data, "Return value missing 'shape' field"
    assert "dtype" in return_value_data, "Return value missing 'dtype' field"
    assert "values" in return_value_data, "Return value missing 'values' field"

    # Verify return value shape matches input shapes (add should preserve shape)
    return_shape = return_value_data["shape"]
    assert return_shape == shape_a, f"Return value shape mismatch: expected {shape_a}, got {return_shape}"

    # Verify return value has correct number of elements
    expected_return_elements = 1
    for dim in shape_a:
        expected_return_elements *= dim

    return_values = return_value_data["values"]

    actual_return_elements = count_elements(return_values)
    assert (
        actual_return_elements == expected_return_elements
    ), f"Return value element count mismatch: expected {expected_return_elements}, got {actual_return_elements}"

    # Clean up trace directory
    if trace_dir.exists():
        shutil.rmtree(trace_dir)

    # Restore original state
    ttnn.operation_tracer._OPERATION_COUNTER = original_operation_counter
    if original_report_path is not None:
        ttnn.CONFIG.root_report_path = original_report_path


def test_default_no_tensor_values(tmp_path, device):
    """Test that tensor values are excluded by default (only metadata is serialized)."""
    # Reset counter for predictable test numbering
    original_operation_counter = ttnn.operation_tracer._OPERATION_COUNTER
    ttnn.operation_tracer._OPERATION_COUNTER = 0

    # Override the log directory in CONFIG to use tmp_path
    original_report_path = None
    if hasattr(ttnn.CONFIG, "root_report_path"):
        original_report_path = ttnn.CONFIG.root_report_path
        ttnn.CONFIG.root_report_path = str(tmp_path)

    # The trace directory will be created under root_report_path / "operation_parameters"
    trace_dir = tmp_path / "operation_parameters"

    # Create tensors and perform operation
    shape_a = [2, 3]
    shape_b = [2, 3]
    tensor_a = ttnn.rand(shape=shape_a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tensor_b = ttnn.rand(shape=shape_b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    # Execute add operation - result is traced but not used in test assertions
    ttnn.add(tensor_a, tensor_b)
    ttnn.synchronize_device(device)

    # Find the trace files
    trace_files = sorted(trace_dir.glob("*_ttnn_*.json")) if trace_dir.exists() else []

    # Find add operation file
    add_files = [f for f in trace_files if TRACE_FILE_PATTERN_TTNN_ADD in f.name]
    assert len(add_files) >= 1, f"Expected at least 1 add trace file, found {len(add_files)}"

    # Check the add operation trace
    add_file = add_files[0]
    with open(add_file, "r") as f:
        operation_data_add = json.load(f)

    assert operation_data_add["operation_name"] == OPERATION_NAME_TTNN_ADD

    # Check that tensor data exists but values are NOT included
    arg_0 = operation_data_add["args"][0]["value"]
    arg_1 = operation_data_add["args"][1]["value"]

    assert arg_0["type"] == "ttnn.Tensor", f"Expected tensor type, got {arg_0.get('type')}"
    assert arg_1["type"] == "ttnn.Tensor", f"Expected tensor type, got {arg_1.get('type')}"

    # Verify tensor metadata exists
    for i, tensor_data in enumerate([arg_0, arg_1]):
        assert "shape" in tensor_data, f"Tensor data {i} missing 'shape' field"
        assert "dtype" in tensor_data, f"Tensor data {i} missing 'dtype' field"
        # Verify values are NOT included
        assert (
            "values" not in tensor_data
        ), f"Tensor data {i} should not have 'values' field when serialization is disabled"

    # Verify return value also doesn't have values
    assert "return_value" in operation_data_add, "Expected return_value in add operation data"
    return_value_data = operation_data_add["return_value"]
    assert (
        return_value_data["type"] == "ttnn.Tensor"
    ), f"Expected return value to be tensor, got {return_value_data.get('type')}"
    assert "shape" in return_value_data, "Return value missing 'shape' field"
    assert "dtype" in return_value_data, "Return value missing 'dtype' field"
    assert (
        "values" not in return_value_data
    ), "Return value should not have 'values' field when serialization is disabled"

    # Clean up trace directory
    if trace_dir.exists():
        shutil.rmtree(trace_dir)

    # Restore original state
    ttnn.operation_tracer._OPERATION_COUNTER = original_operation_counter
    if original_report_path is not None:
        ttnn.CONFIG.root_report_path = original_report_path


def test_tracing_disabled_no_files_created(tmp_path, device):
    """Test that no trace files are created when tracing is disabled."""
    # Ensure tracing is disabled (fixture should handle this, but be explicit)
    original_trace_flag = ttnn.operation_tracer._ENABLE_TRACE
    ttnn.operation_tracer._ENABLE_TRACE = False

    # Override the log directory in CONFIG to use tmp_path
    original_report_path = None
    if hasattr(ttnn.CONFIG, "root_report_path"):
        original_report_path = ttnn.CONFIG.root_report_path
        ttnn.CONFIG.root_report_path = str(tmp_path)

    # The trace directory would be created under root_report_path / "operation_parameters"
    trace_dir = tmp_path / "operation_parameters"

    # Perform some operations - results not needed, we just verify no trace files created
    a = ttnn.rand(shape=[2, 3], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    b = ttnn.rand(shape=[2, 3], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn.add(a, b)
    ttnn.synchronize_device(device)

    # Verify NO trace files were created
    trace_files = list(trace_dir.glob("*_ttnn_*.json")) if trace_dir.exists() else []
    assert (
        len(trace_files) == 0
    ), f"Expected no trace files when tracing is disabled, but found {len(trace_files)} files: {[f.name for f in trace_files]}"

    # Also verify the directory doesn't exist (or is empty)
    if trace_dir.exists():
        all_files = list(trace_dir.glob("*"))
        assert len(all_files) == 0, f"Expected trace directory to be empty, but found: {[f.name for f in all_files]}."

    # Restore original state
    ttnn.operation_tracer._ENABLE_TRACE = original_trace_flag
    if original_report_path is not None:
        ttnn.CONFIG.root_report_path = original_report_path


def test_from_torch_to_device_tracing(tmp_path, device):
    """Test that from_torch and to_device operations are traced correctly."""
    # Enable tracing (module already imported at top of file)
    original_trace_flag = ttnn.operation_tracer._ENABLE_TRACE
    ttnn.operation_tracer._ENABLE_TRACE = True

    # Set up trace directory
    original_report_path = None
    if hasattr(ttnn.CONFIG, "root_report_path"):
        original_report_path = ttnn.CONFIG.root_report_path
        ttnn.CONFIG.root_report_path = str(tmp_path)

    trace_dir = tmp_path / "operation_parameters"

    # Reset counter
    original_counter = ttnn.operation_tracer._OPERATION_COUNTER
    ttnn.operation_tracer._OPERATION_COUNTER = 0

    # Create tensors using from_torch and to_device
    torch_tensor = torch.randn(2, 3).bfloat16()
    tensor_host = ttnn.from_torch(torch_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    # Execute to_device - result is traced but not used in test assertions
    ttnn.to_device(tensor_host, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.synchronize_device(device)

    # Verify trace files were created
    trace_files = sorted(trace_dir.glob("*_ttnn_*.json")) if trace_dir.exists() else []

    # Should have 2 files: 1 from_torch + 1 to_device
    assert len(trace_files) >= 2, f"Expected at least 2 trace files, found {len(trace_files)}"

    # Find operation files
    from_torch_files = [f for f in trace_files if TRACE_FILE_PATTERN_TTNN_FROM_TORCH in f.name]
    to_device_files = [f for f in trace_files if TRACE_FILE_PATTERN_TTNN_TO_DEVICE in f.name]

    assert len(from_torch_files) >= 1, "Expected at least 1 from_torch trace file"
    assert len(to_device_files) >= 1, "Expected at least 1 to_device trace file"

    # Check from_torch trace
    with open(from_torch_files[0], "r") as f:
        from_torch_data = json.load(f)

    assert from_torch_data["operation_name"] == OPERATION_NAME_TTNN_FROM_TORCH
    assert len(from_torch_data["args"]) >= 1, "Expected at least 1 arg for from_torch"
    # First arg should be a torch.Tensor
    first_arg = from_torch_data["args"][0]["value"]
    assert first_arg["type"] == "torch.Tensor", f"Expected torch.Tensor, got {first_arg.get('type')}"
    assert first_arg["shape"] == [2, 3], f"Expected shape [2, 3], got {first_arg.get('shape')}"

    # Check to_device trace
    with open(to_device_files[0], "r") as f:
        to_device_data = json.load(f)

    assert to_device_data["operation_name"] == OPERATION_NAME_TTNN_TO_DEVICE
    assert len(to_device_data["args"]) >= 1, "Expected at least 1 arg for to_device"
    # First arg should be a ttnn.Tensor
    first_arg = to_device_data["args"][0]["value"]
    assert first_arg["type"] == "ttnn.Tensor", f"Expected ttnn.Tensor, got {first_arg.get('type')}"

    # Clean up trace directory
    if trace_dir.exists():
        shutil.rmtree(trace_dir)

    # Restore
    ttnn.operation_tracer._OPERATION_COUNTER = original_counter
    ttnn.operation_tracer._ENABLE_TRACE = original_trace_flag
    if original_report_path is not None:
        ttnn.CONFIG.root_report_path = original_report_path
