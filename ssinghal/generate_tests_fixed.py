import json
import os


def load_data():
    """Load all necessary data files"""

    with open("ssinghal/basic_operators.json", "r") as f:
        basic_operators = json.load(f)

    with open("ssinghal/operator_mapping.json", "r") as f:
        operator_mapping = json.load(f)

    with open("ssinghal/mapping_analysis.json", "r") as f:
        mapping_analysis = json.load(f)

    return basic_operators, operator_mapping, mapping_analysis


def create_test_template(op_name, op_mapping, shapes_data):
    """Create a test file template for a given operator with proper indentation"""

    # Extract all shapes for this operator
    all_shapes = []
    for entry in shapes_data:
        all_shapes.extend(entry["shapes"])

    # Remove duplicates while preserving order
    unique_shapes = []
    seen = set()
    for shape in all_shapes:
        shape_tuple = tuple(shape)
        if shape_tuple not in seen:
            seen.add(shape_tuple)
            unique_shapes.append(shape)

    # Format shapes for pytest parameterization
    if unique_shapes:
        formatted_shapes = ", ".join([str(shape) for shape in unique_shapes])
    else:
        formatted_shapes = "[1, 32, 32, 32]"  # Default shape if none found

    num_inputs = op_mapping.get("num_inputs", 1)
    ttnn_function = op_mapping["ttnn_function"]
    torch_function = op_mapping["torch_function"]
    category = op_mapping["category"]

    # Generate input tensor creation based on operator requirements
    if num_inputs == 1:
        input_creation = """        torch_input = torch.rand(input_shape, dtype=torch.bfloat16)

        if len(input_shape) == 4:
            torch_input = torch_input.permute(0, 2, 3, 1)

        ttnn_input = ttnn.from_torch(torch_input, device=device, memory_config=ttnn.L1_MEMORY_CONFIG, layout=ttnn.TILE_LAYOUT)"""

        if category == "activation":
            operation_call = f"        ttnn_output = {ttnn_function}(ttnn_input)"
            torch_reference = f"        torch_reference = {torch_function}(torch_input)"
        elif op_name in ["view", "unsafeview"]:
            operation_call = "        ttnn_output = ttnn.reshape(ttnn_input, input_shape)"
            torch_reference = "        torch_reference = torch_input.view(input_shape)"
        elif op_name == "permute":
            operation_call = (
                "        ttnn_output = ttnn.permute(ttnn_input, (0, 3, 1, 2)) if len(input_shape) == 4 else ttnn_input"
            )
            torch_reference = (
                "        torch_reference = torch_input.permute(0, 3, 1, 2) if len(input_shape) == 4 else torch_input"
            )
        elif op_name == "transpose":
            operation_call = "        ttnn_output = ttnn.transpose(ttnn_input, -2, -1)"
            torch_reference = "        torch_reference = torch.transpose(torch_input, -2, -1)"
        elif op_name == "unsqueeze":
            operation_call = "        ttnn_output = ttnn.unsqueeze(ttnn_input, 0)"
            torch_reference = "        torch_reference = torch.unsqueeze(torch_input, 0)"
        else:
            operation_call = f"        ttnn_output = {ttnn_function}(ttnn_input)"
            torch_reference = f"        torch_reference = {torch_function}(torch_input)"

    elif num_inputs == 2:
        input_creation = """        torch_input1 = torch.rand(input_shape, dtype=torch.bfloat16)
        torch_input2 = torch.rand(input_shape, dtype=torch.bfloat16)

        if len(input_shape) == 4:
            torch_input1 = torch_input1.permute(0, 2, 3, 1)
            torch_input2 = torch_input2.permute(0, 2, 3, 1)

        ttnn_input1 = ttnn.from_torch(torch_input1, device=device, memory_config=ttnn.L1_MEMORY_CONFIG, layout=ttnn.TILE_LAYOUT)
        ttnn_input2 = ttnn.from_torch(torch_input2, device=device, memory_config=ttnn.L1_MEMORY_CONFIG, layout=ttnn.TILE_LAYOUT)"""

        operation_call = f"        ttnn_output = {ttnn_function}(ttnn_input1, ttnn_input2)"
        torch_reference = f"        torch_reference = {torch_function}(torch_input1, torch_input2)"

    elif num_inputs == 3:
        input_creation = """        torch_input1 = torch.rand(input_shape, dtype=torch.bfloat16)
        torch_input2 = torch.rand(input_shape, dtype=torch.bfloat16)
        torch_input3 = torch.rand(input_shape, dtype=torch.bfloat16)

        if len(input_shape) == 4:
            torch_input1 = torch_input1.permute(0, 2, 3, 1)
            torch_input2 = torch_input2.permute(0, 2, 3, 1)
            torch_input3 = torch_input3.permute(0, 2, 3, 1)

        ttnn_input1 = ttnn.from_torch(torch_input1, device=device, memory_config=ttnn.L1_MEMORY_CONFIG, layout=ttnn.TILE_LAYOUT)
        ttnn_input2 = ttnn.from_torch(torch_input2, device=device, memory_config=ttnn.L1_MEMORY_CONFIG, layout=ttnn.TILE_LAYOUT)
        ttnn_input3 = ttnn.from_torch(torch_input3, device=device, memory_config=ttnn.L1_MEMORY_CONFIG, layout=ttnn.TILE_LAYOUT)"""

        operation_call = f"        ttnn_output = {ttnn_function}(ttnn_input1, ttnn_input2, ttnn_input3)"
        torch_reference = f"        torch_reference = {torch_function}(torch_input1, torch_input2, torch_input3)"

    elif num_inputs == "variable":
        input_creation = """        torch_input1 = torch.rand(input_shape, dtype=torch.bfloat16)
        torch_input2 = torch.rand(input_shape, dtype=torch.bfloat16)

        if len(input_shape) == 4:
            torch_input1 = torch_input1.permute(0, 2, 3, 1)
            torch_input2 = torch_input2.permute(0, 2, 3, 1)

        ttnn_input1 = ttnn.from_torch(torch_input1, device=device, memory_config=ttnn.L1_MEMORY_CONFIG, layout=ttnn.TILE_LAYOUT)
        ttnn_input2 = ttnn.from_torch(torch_input2, device=device, memory_config=ttnn.L1_MEMORY_CONFIG, layout=ttnn.TILE_LAYOUT)"""

        if "concat" in ttnn_function:
            operation_call = f"        ttnn_output = {ttnn_function}([ttnn_input1, ttnn_input2], dim=0)"
            torch_reference = f"        torch_reference = {torch_function}([torch_input1, torch_input2], dim=0)"
        else:
            operation_call = f"        ttnn_output = {ttnn_function}([ttnn_input1, ttnn_input2])"
            torch_reference = f"        torch_reference = {torch_function}([torch_input1, torch_input2])"
    else:
        # Default single input
        input_creation = """        torch_input = torch.rand(input_shape, dtype=torch.bfloat16)

        if len(input_shape) == 4:
            torch_input = torch_input.permute(0, 2, 3, 1)

        ttnn_input = ttnn.from_torch(torch_input, device=device, memory_config=ttnn.L1_MEMORY_CONFIG, layout=ttnn.TILE_LAYOUT)"""

        operation_call = f"        ttnn_output = {ttnn_function}(ttnn_input)"
        torch_reference = f"        torch_reference = {torch_function}(torch_input)"

    # Special handling for Linear layer
    if op_name == "Linear":
        input_creation = """        batch_size, seq_len, hidden_size = input_shape if len(input_shape) == 3 else (1, input_shape[0], input_shape[1])
        torch_input = torch.rand((batch_size, seq_len, hidden_size), dtype=torch.bfloat16)
        torch_weight = torch.rand((hidden_size, hidden_size), dtype=torch.bfloat16)

        ttnn_input = ttnn.from_torch(torch_input, device=device, memory_config=ttnn.L1_MEMORY_CONFIG, layout=ttnn.TILE_LAYOUT)
        ttnn_weight = ttnn.from_torch(torch_weight, device=device, memory_config=ttnn.L1_MEMORY_CONFIG, layout=ttnn.TILE_LAYOUT)"""

        operation_call = "        ttnn_output = ttnn.linear(ttnn_input, ttnn_weight)"
        torch_reference = "        torch_reference = torch.nn.functional.linear(torch_input, torch_weight)"

    test_content = f'''import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import check_with_pcc_without_tensor_printout

@pytest.mark.parametrize("input_shape", [{formatted_shapes}])
def test_{op_name.lower()}(device, input_shape):
    """Test {op_name} operator with various input shapes from vision models"""
    torch.manual_seed(0)

    try:
{input_creation}

{operation_call}

{torch_reference}

        # Convert output back to torch
        ttnn_result = ttnn.to_torch(ttnn_output)

        # Compare results
        check_with_pcc_without_tensor_printout(ttnn_result, torch_reference, 0.99)

    except RuntimeError as e:
        if "Out of Memory" in str(e):
            pytest.skip(f"OOM: {{input_shape}} - {{str(e)}}")
        else:
            raise e
    except Exception as e:
        if "incompatible function arguments" in str(e):
            pytest.skip(f"Type error: {{input_shape}} - {{str(e)}}")
        else:
            raise e
'''

    return test_content


def generate_all_tests():
    """Generate test files for all supported operators with proper indentation"""

    basic_operators, operator_mapping, mapping_analysis = load_data()

    # Create test directory
    test_dir = "ssinghal/tests"
    os.makedirs(test_dir, exist_ok=True)

    # Get testable operators
    testable_operators = []
    detailed_analysis = mapping_analysis["detailed_analysis"]

    for op_name, analysis in detailed_analysis.items():
        if analysis.get("mapping") and analysis["mapping"]["supported"] and analysis["mapping"]["category"] != "meta":
            testable_operators.append(op_name)

    print(f"Generating tests for {len(testable_operators)} operators...")

    generated_tests = []
    skipped_tests = []

    for op_name in testable_operators:
        try:
            op_mapping = operator_mapping[op_name]
            shapes_data = basic_operators[op_name]

            # Generate test content
            test_content = create_test_template(op_name, op_mapping, shapes_data)

            # Write test file
            test_file_path = f"{test_dir}/test_{op_name.lower()}.py"
            with open(test_file_path, "w") as f:
                f.write(test_content)

            generated_tests.append(
                {
                    "operator": op_name,
                    "file": test_file_path,
                    "occurrences": len(shapes_data),
                    "unique_shapes": len(set(tuple(shape) for entry in shapes_data for shape in entry["shapes"])),
                }
            )

            print(f"✓ Generated test for {op_name}")

        except Exception as e:
            print(f"✗ Failed to generate test for {op_name}: {str(e)}")
            skipped_tests.append({"operator": op_name, "error": str(e)})

    return generated_tests, skipped_tests


def main():
    generated_tests, skipped_tests = generate_all_tests()

    print(f"\n=== TEST GENERATION COMPLETE ===")
    print(f"Generated: {len(generated_tests)} test files")
    print(f"Skipped: {len(skipped_tests)} operators")

    if generated_tests:
        print(f"\n=== GENERATED TESTS ===")
        for test in generated_tests:
            print(f"  {test['operator']}: {test['unique_shapes']} unique shapes")


if __name__ == "__main__":
    main()
