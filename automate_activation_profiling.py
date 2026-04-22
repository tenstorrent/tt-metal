#!/usr/bin/env python3

import re
import subprocess
import time
import sys
from pathlib import Path


def extract_shape_entries(file_content):
    """Extract shape and memory config entries from the parametrize decorator"""
    lines = file_content.split("\n")

    # Find the parametrize block with input_shapes
    start_idx = None
    end_idx = None

    for i, line in enumerate(lines):
        if '"input_shapes, memory_config_type"' in line:
            # Look for the opening bracket on the next line
            for j in range(i + 1, min(i + 5, len(lines))):
                if "[" in lines[j]:
                    start_idx = j + 1  # Start after the opening bracket
                    break
            break

    if start_idx is None:
        raise ValueError("Could not find shape parametrize block start")

    # Find the end of this parametrize block
    for i in range(start_idx, len(lines)):
        line = lines[i].strip()
        if line == "]," or line == "]":
            end_idx = i
            break

    if end_idx is None:
        raise ValueError("Could not find end of shape parametrize block")

    # Extract shape entries
    entries = []
    for i in range(start_idx, end_idx):
        line = lines[i]
        stripped_line = line.strip()

        # Skip empty lines and pure comment lines
        if not stripped_line or (stripped_line.startswith("#") and "torch.Size" not in stripped_line):
            continue

        # Check if this line contains a shape entry
        if "torch.Size" in stripped_line and '"' in stripped_line:
            # Extract shape and memory config
            shape_match = re.search(r"torch\.Size\(\[([^\]]+)\]\)", stripped_line)
            config_match = re.search(r'"([^"]+)"', stripped_line)

            if shape_match and config_match:
                shape_str = shape_match.group(1)
                memory_config = config_match.group(1)

                entries.append(
                    {
                        "line_num": i + 1,
                        "line_content": stripped_line,
                        "shape": shape_str,
                        "memory_config": memory_config,
                        "is_commented": stripped_line.startswith("#"),
                    }
                )

    return entries, start_idx + 1, end_idx + 1


def extract_activation_entries(file_content):
    """Extract all activation function entries from the parametrize decorator"""

    # Find the start and end of the activation function parametrize block
    lines = file_content.split("\n")

    # Find the parametrize block with activation_func
    start_idx = None
    end_idx = None

    for i, line in enumerate(lines):
        if '"activation_func, func_name, params, has_approx"' in line:
            # Look for the opening bracket on the next line
            for j in range(i + 1, min(i + 5, len(lines))):
                if "[" in lines[j]:
                    start_idx = j + 1  # Start after the opening bracket
                    break
            break

    if start_idx is None:
        raise ValueError("Could not find activation function parametrize block start")

    # Find the end of this parametrize block (look for ],)
    for i in range(start_idx, len(lines)):
        line = lines[i].strip()
        if line == "]," or line == "]":
            end_idx = i
            break

    if end_idx is None:
        raise ValueError("Could not find end of activation function parametrize block")

    # Extract entries from the block
    entries = []
    for i in range(start_idx, end_idx):
        line = lines[i]
        stripped_line = line.strip()

        # Skip empty lines and pure comment lines (comments without function calls)
        if not stripped_line or (stripped_line.startswith("#") and "(ttnn." not in stripped_line):
            continue

        # Check if this line contains an activation function entry
        if "(ttnn." in stripped_line and '"' in stripped_line:
            # Extract the function name from the string in the tuple
            func_name_match = re.search(r'"([^"]+)"', stripped_line)
            if func_name_match:
                func_name = func_name_match.group(1)
                entries.append(
                    {
                        "line_num": i + 1,  # Convert to 1-based line numbering
                        "line_content": stripped_line,
                        "func_name": func_name,
                        "is_commented": stripped_line.startswith("#"),
                    }
                )

    return entries, start_idx + 1, end_idx + 1  # Convert to 1-based


def modify_shape_entries(file_path, shape_entries, active_shape_idx, shape_start_line, shape_end_line):
    """Modify the test file to have only one shape entry active"""
    with open(file_path, "r") as f:
        lines = f.readlines()

    # Comment out all shape entries
    for i in range(shape_start_line - 1, shape_end_line):
        if i < len(lines):
            line = lines[i].strip()
            if "torch.Size" in line and '"' in line:
                if not lines[i].strip().startswith("#"):
                    lines[i] = "        # " + lines[i].lstrip()

    # Uncomment the active shape entry
    if active_shape_idx < len(shape_entries):
        target_line_idx = shape_entries[active_shape_idx]["line_num"] - 1
        if target_line_idx < len(lines):
            line = lines[target_line_idx]
            if line.strip().startswith("#"):
                content_start = line.find("(torch.Size")
                if content_start > 0:
                    indent = " " * 8
                    lines[target_line_idx] = indent + line[content_start:]

    with open(file_path, "w") as f:
        f.writelines(lines)


def modify_activation_entries(
    file_path, activation_entries, active_activation_idx, activation_start_line, activation_end_line
):
    """Modify the test file to have only one activation entry active"""
    with open(file_path, "r") as f:
        lines = f.readlines()

    # Comment out all activation function entries
    for i in range(activation_start_line - 1, activation_end_line):  # Convert back to 0-based
        if i < len(lines):
            line = lines[i].strip()
            if "(ttnn." in line and '"' in line:
                # Comment this line if it's not already commented
                if not lines[i].strip().startswith("#"):
                    lines[i] = "        # " + lines[i].lstrip()

    # Uncomment the active entry
    if active_activation_idx < len(activation_entries):
        target_line_idx = activation_entries[active_activation_idx]["line_num"] - 1  # Convert to 0-based
        if target_line_idx < len(lines):
            line = lines[target_line_idx]
            # Remove comment if present
            if line.strip().startswith("#"):
                # Find the position of the actual content after the comment
                content_start = line.find("(ttnn.")
                if content_start > 0:
                    indent = " " * 8  # Standard pytest parametrize indentation
                    lines[target_line_idx] = indent + line[content_start:]

    # Write back to file
    with open(file_path, "w") as f:
        f.writelines(lines)


def generate_profile_name(func_name, shape_str, memory_config):
    """Generate a dynamic profile name based on function, shape, and memory config"""
    # Convert shape string to a cleaner format
    # e.g., "1, 16, 320, 320" -> "1_16_320_320"
    shape_clean = shape_str.replace(" ", "").replace(",", "_")

    # Convert memory config to a shorter format
    # e.g., "height_sharded" -> "HEIGHT_SHARDED"
    config_clean = memory_config.upper().replace("_", "_")

    # Generate the full name
    profile_name = f"{func_name}_{shape_clean}_{config_clean}"

    return profile_name


def run_profiling_command(profile_name, workspace_dir):
    """Run the profiling command for the given function with dynamic profile name"""

    # Full command with environment setup
    cmd = f'''cd {workspace_dir} && source python_env/bin/activate && rm -rf /home/ubuntu/.cache/tt-metal-cache && ./tools/tracy/profile_this.py -n {profile_name} -c "pytest {workspace_dir}/test_all_activations.py"'''

    print(f"\n{'='*60}")
    print(f"Running profiling for: {profile_name}")
    print(f"Command: {cmd}")
    print(f"{'='*60}")

    try:
        # Run with bash to handle source command properly
        result = subprocess.run(
            cmd,
            shell=True,
            executable="/bin/bash",  # Use bash to handle source command
            cwd=str(workspace_dir),
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout (increased for environment setup)
        )

        print(f"Exit code: {result.returncode}")
        if result.stdout:
            print("STDOUT:")
            print(result.stdout[-2000:])  # Show last 2000 chars to avoid too much output
        if result.stderr:
            print("STDERR:")
            print(result.stderr[-1000:])  # Show last 1000 chars of errors

        return result.returncode == 0

    except subprocess.TimeoutExpired:
        print(f"TIMEOUT: Profiling for {func_name} took too long")
        return False
    except Exception as e:
        print(f"ERROR running profiling for {func_name}: {e}")
        return False


def test_environment_setup(workspace_dir):
    """Test if the environment setup works correctly"""

    print("Testing environment setup...")

    cmd = f'cd {workspace_dir} && source python_env/bin/activate && echo "Environment activated successfully" && which python3 && python3 -c "import ttnn; print(\'TTNN imported successfully\')"'

    try:
        result = subprocess.run(
            cmd,
            shell=True,
            executable="/bin/bash",
            cwd=str(workspace_dir),
            capture_output=True,
            text=True,
            timeout=60,
        )

        print(f"Environment test exit code: {result.returncode}")
        if result.stdout:
            print("STDOUT:")
            print(result.stdout)
        if result.stderr:
            print("STDERR:")
            print(result.stderr)

        return result.returncode == 0

    except Exception as e:
        print(f"Environment test failed: {e}")
        return False


def main():
    # Dynamically determine the workspace path from the script's location
    script_dir = Path(__file__).resolve().parent
    test_file = script_dir / "test_all_activations.py"

    print(f"Working directory: {script_dir}")
    print(f"Reading test file: {test_file}")
    print("Analyzing test_all_activations.py...")

    # Test environment setup first (can be disabled for debugging)
    # if not test_environment_setup(script_dir):
    #     print("❌ Environment setup test failed! Please check venv_start script.")
    #     return

    # Read the original file
    with open(test_file, "r") as f:
        original_content = f.read()

    try:
        # Extract shape entries
        shape_entries, shape_start_line, shape_end_line = extract_shape_entries(original_content)

        print(f"\nFound {len(shape_entries)} shape configuration entries:")
        for i, entry in enumerate(shape_entries):
            status = "COMMENTED" if entry["is_commented"] else "ACTIVE"
            print(
                f"  {i+1:2d}. Shape: [{entry['shape']:<20}] Memory: {entry['memory_config']:<20} (line {entry['line_num']}) [{status}]"
            )

        if not shape_entries:
            print("No shape entries found!")
            return

        # Extract all activation function entries
        activation_entries, activation_start_line, activation_end_line = extract_activation_entries(original_content)

        print(f"\nFound {len(activation_entries)} activation function entries:")
        for i, entry in enumerate(activation_entries):
            status = "COMMENTED" if entry["is_commented"] else "ACTIVE"
            print(f"  {i+1:2d}. {entry['func_name']:<20} (line {entry['line_num']}) [{status}]")

        if not activation_entries:
            print("No activation function entries found!")
            return

        # Calculate total combinations
        total_combinations = len(shape_entries) * len(activation_entries)
        print(f"\n{'='*60}")
        print(f"Total combinations to profile: {total_combinations}")
        print(f"  Shapes: {len(shape_entries)}")
        print(f"  Activations: {len(activation_entries)}")
        print(f"{'='*60}")

        # Ask for confirmation
        response = input(f"\nProceed with profiling {total_combinations} combinations? (y/N): ")
        if response.lower() != "y":
            print("Aborted.")
            return

        successful_profiles = []
        failed_profiles = []

        combination_count = 0

        # Process each combination of shape and activation
        for shape_idx, shape_entry in enumerate(shape_entries):
            for activation_idx, activation_entry in enumerate(activation_entries):
                combination_count += 1

                # Generate dynamic profile name
                profile_name = generate_profile_name(
                    activation_entry["func_name"], shape_entry["shape"], shape_entry["memory_config"]
                )

                print(f"\n{'#'*60}")
                print(f"Processing {combination_count}/{total_combinations}")
                print(f"  Shape: [{shape_entry['shape']}] Config: {shape_entry['memory_config']}")
                print(f"  Activation: {activation_entry['func_name']}")
                print(f"  Profile Name: {profile_name}")
                print(f"{'#'*60}")

                # Modify the test file to activate only this combination
                modify_shape_entries(test_file, shape_entries, shape_idx, shape_start_line, shape_end_line)
                modify_activation_entries(
                    test_file, activation_entries, activation_idx, activation_start_line, activation_end_line
                )

                # Run the profiling
                success = run_profiling_command(profile_name, script_dir)

                if success:
                    successful_profiles.append(profile_name)
                    print(f"✅ SUCCESS: {profile_name}")
                else:
                    failed_profiles.append(profile_name)
                    print(f"❌ FAILED: {profile_name}")

                # Add a small delay between runs
                time.sleep(2)

        # Summary
        print(f"\n{'='*60}")
        print("PROFILING COMPLETE")
        print(f"{'='*60}")
        print(f"Successful: {len(successful_profiles)}/{total_combinations}")
        for profile in successful_profiles:
            print(f"  ✅ {profile}")

        if failed_profiles:
            print(f"\nFailed: {len(failed_profiles)}/{total_combinations}")
            for profile in failed_profiles:
                print(f"  ❌ {profile}")

        print(f"\nProfile results should be in: {script_dir}/generated/profiler/reports/")

    finally:
        # Restore the original file
        print(f"\nRestoring original test file...")
        with open(test_file, "w") as f:
            f.write(original_content)
        print("✅ Original file restored")


if __name__ == "__main__":
    main()
