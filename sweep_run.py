import os
import re

# List of operations
ops = ["i0", "i1", "erf", "tan", "abs_int32"]


# Function to modify the file
def modify_file(file_path):
    with open(file_path, "r") as f:
        content = f.read()

    # Modify only if the exact range [-100, 100] is found
    modified_content = re.sub(
        r"low=-100,\s*high=100",  # Match `low=-100, high=100`
        "low=-1000, high=1000",  # Replace with `low=-1000, high=1000`
        content,
    )

    # If modifications were made, overwrite the file
    if content != modified_content:
        with open(file_path, "w") as f:
            f.write(modified_content)
        print(f"Modified {file_path}")


# Loop through each operation and run the commands
for op in ops:
    module_name = f"eltwise.unary.{op}.{op}_survey"
    file_path = f"tests/sweep_framework/sweeps/eltwise/unary/{op}/{op}_survey.py"  # Assuming file structure

    # Modify the file if it exists
    # if os.path.exists(file_path):
    #     modify_file(file_path)

    # Run the commands
    command1 = f"python3 tests/sweep_framework/sweeps_parameter_generator.py --module-name {module_name} --elastic cloud --clean"
    command2 = f"python3 tests/sweep_framework/sweeps_runner.py --module-name {module_name} --elastic cloud"

    os.system(command1)
    os.system(command2)
