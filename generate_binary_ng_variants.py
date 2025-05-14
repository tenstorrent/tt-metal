import os

# File paths
original_file = "/home/ubuntu/tt-metal/tests/sweep_framework/sweeps/eltwise/binary_ng/binary_ng.py"
base_dir = os.path.dirname(original_file)

# List of operations
op_names = [
    "hypot",
    "xlogy",
    "minimum",
    "maximum",
    "atan2",
    "nextafter",
    "addalpha",
    "subalpha",
    "isclose",
    "remainder",
    "fmod",
    "div",
    "div_no_nan",
    "scatter",
    "outer",
    "gcd",
    "lcm",
]

# Read original content
with open(original_file, "r") as file:
    original_content = file.read()

# Loop through each operation name and create a new file
for op in op_names:
    new_content = original_content.replace("op_name_here", op)
    new_filename = f"binary_ng_{op}_bcast.py"
    new_filepath = os.path.join(base_dir, new_filename)

    with open(new_filepath, "w") as new_file:
        new_file.write(new_content)

    print(f"✅ Created: {new_filepath}")
