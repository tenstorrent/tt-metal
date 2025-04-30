import os
import re

# List of operations
ops = [
    # "hypot",
    # "xlogy",
    "minimum",
    "maximum",
    # "atan2",
    # "nextafter",
    # "addalpha",
    # "subalpha",
    # "isclose",
    # "remainder",
    # "fmod",
    # "div",
    # "div_no_nan",
    # "scatter",
    # "outer",
    # "gcd",
    # "lcm",
]


# Function to modify the file
def modify_file(file_path):
    with open(file_path, "r") as f:
        content = f.read()

    # Modify only if the exact range [-100, 100] is found
    modified_content = re.sub(
        r"low=-100,\s*high=100",  # Match `low=-100, high=100`
        "low=-1000000, high=1000000",  # Replace with `low=-1000, high=1000`
        content,
    )

    # If modifications were made, overwrite the file
    if content != modified_content:
        with open(file_path, "w") as f:
            f.write(modified_content)
        print(f"Modified {file_path}")


# Loop through each operation and run the commands
for op in ops:
    bitwise = ""
    if op.startswith("bitwise"):
        bitwise = "bitwise."
    module_name = f"eltwise.binary_ng.{bitwise}binary_ng_{op}_bcast"
    file_path = f"tests/sweep_framework/sweeps/eltwise/binary_ng/binary_ng_{op}.py"  # Assuming file structure

    # Modify the file if it exists
    # if os.path.exists(file_path):
    # modify_file(file_path)

    # Run the commands
    command1 = f"python3 tests/sweep_framework/sweeps_parameter_generator.py --module-name {module_name} --elastic cloud --clean"
    command2 = f"python3 tests/sweep_framework/sweeps_runner.py --module-name {module_name} --elastic cloud"

    os.system(command1)
    os.system(command2)
