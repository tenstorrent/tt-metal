import csv
import json
import re
from collections import defaultdict


def extract_operators_from_csv():
    """Extract actual operators from the CSV file by taking the last part after '.'"""

    operators_data = defaultdict(list)

    with open("all_cnn_models.csv", "r") as f:
        reader = csv.reader(f)
        try:
            next(reader)  # Skip header
        except StopIteration:
            return {}

        for row in reader:
            if len(row) < 3:
                continue

            unique_name, input_shapes_str, model_name = row

            # Skip empty or input tensor entries
            if not unique_name or unique_name.startswith("input_tensor"):
                continue

            # Extract operator name (last part after '.')
            if "." in unique_name:
                op_name = unique_name.split(".")[-1]
            else:
                op_name = unique_name

            # Skip numbered entries that aren't real operators
            if op_name.replace("_", "").replace("-", "").isdigit():
                continue

            # Parse input shapes
            if input_shapes_str and input_shapes_str != "[]":
                # Parse torch.Size entries
                matches = re.findall(r"torch\.Size\(\[([^\]]*)\]\)", input_shapes_str)

                if matches:
                    shapes = []
                    for match in matches:
                        if match.strip():  # Non-empty
                            try:
                                shape = [int(x.strip()) for x in match.split(",")]
                                shapes.append(shape)
                            except ValueError:
                                continue

                    if shapes:
                        operators_data[op_name].append(
                            {"shapes": shapes, "model": model_name, "full_name": unique_name}
                        )

    return operators_data


def categorize_operators(operators_data):
    """Categorize operators into basic vs complex module names"""

    basic_ops = {}
    complex_ops = {}

    for op_name, data in operators_data.items():
        # Basic operators: no underscores or module naming patterns
        if not any(x in op_name for x in ["_", "Module", "Sequential", "BatchNorm", "Conv", "Layer", "Block"]):
            basic_ops[op_name] = data
        else:
            complex_ops[op_name] = data

    return basic_ops, complex_ops


def main():
    print("Extracting operators from CSV...")
    operators_data = extract_operators_from_csv()

    print(f"Total operators found: {len(operators_data)}")

    # Categorize operators
    basic_ops, complex_ops = categorize_operators(operators_data)

    print(f"Basic operators: {len(basic_ops)}")
    print(f"Complex module operators: {len(complex_ops)}")

    # Save all data
    with open("ssinghal/all_operators.json", "w") as f:
        json.dump(operators_data, f, indent=2)

    with open("ssinghal/basic_operators.json", "w") as f:
        json.dump(basic_ops, f, indent=2)

    with open("ssinghal/complex_operators.json", "w") as f:
        json.dump(complex_ops, f, indent=2)

    # Print basic operators summary
    print("\n=== BASIC OPERATORS SUMMARY ===")
    for op_name in sorted(basic_ops.keys()):
        count = len(basic_ops[op_name])
        print(f"  {op_name}: {count} occurrences")

    print(f"\nSaved data to:")
    print(f"  - ssinghal/all_operators.json")
    print(f"  - ssinghal/basic_operators.json")
    print(f"  - ssinghal/complex_operators.json")


if __name__ == "__main__":
    main()
