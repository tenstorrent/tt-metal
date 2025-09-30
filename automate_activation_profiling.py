#!/usr/bin/env python3

import re
import subprocess
import time
import sys
from pathlib import Path


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


def modify_test_file(file_path, entries, active_entry_idx, start_line, end_line):
    """Modify the test file to have only one entry active"""

    with open(file_path, "r") as f:
        lines = f.readlines()

    # Comment out all activation function entries
    for i in range(start_line - 1, end_line):  # Convert back to 0-based
        if i < len(lines):
            line = lines[i].strip()
            if "(ttnn." in line and '"' in line:
                # Comment this line if it's not already commented
                if not lines[i].strip().startswith("#"):
                    lines[i] = "        # " + lines[i].lstrip()

    # Uncomment the active entry
    if active_entry_idx < len(entries):
        target_line_idx = entries[active_entry_idx]["line_num"] - 1  # Convert to 0-based
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


def run_profiling_command(func_name):
    """Run the profiling command for the given function"""

    # Full command with environment setup
    cmd = f'''cd /home/ubuntu/logical_not/tt-metal && source python_env/bin/activate && rm -rf /home/ubuntu/.cache/tt-metal-cache && ./tools/tracy/profile_this.py -n {func_name}_16_320_320_YOLOV4_HEIGHT_SHARDED_ACT -c "pytest /home/ubuntu/logical_not/tt-metal/test_all_activations.py"'''

    print(f"\n{'='*60}")
    print(f"Running profiling for: {func_name}")
    print(f"Command: {cmd}")
    print(f"{'='*60}")

    try:
        # Run with bash to handle source command properly
        result = subprocess.run(
            cmd,
            shell=True,
            executable="/bin/bash",  # Use bash to handle source command
            cwd="/home/ubuntu/logical_not/tt-metal",
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


def test_environment_setup():
    """Test if the environment setup works correctly"""

    print("Testing environment setup...")

    cmd = 'cd /home/ubuntu/logical_not/tt-metal && source python_env/bin/activate && echo "Environment activated successfully" && which python3 && python3 -c "import ttnn; print(\'TTNN imported successfully\')"'

    try:
        result = subprocess.run(
            cmd,
            shell=True,
            executable="/bin/bash",
            cwd="/home/ubuntu/logical_not/tt-metal",
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
    test_file = "/home/ubuntu/logical_not/tt-metal/test_all_activations.py"

    print("Analyzing test_all_activations.py...")

    # Test environment setup first
    if not test_environment_setup():
        print("❌ Environment setup test failed! Please check venv_start script.")
        return

    # Read the original file
    with open(test_file, "r") as f:
        original_content = f.read()

    try:
        # Extract all activation function entries
        entries, start_line, end_line = extract_activation_entries(original_content)

        print(f"Found {len(entries)} activation function entries:")
        for i, entry in enumerate(entries):
            status = "COMMENTED" if entry["is_commented"] else "ACTIVE"
            print(f"  {i+1:2d}. {entry['func_name']:<20} (line {entry['line_num']}) [{status}]")

        if not entries:
            print("No activation function entries found!")
            return

        print(f"\nWill process entries from line {start_line} to {end_line}")

        # Ask for confirmation
        response = input(f"\nProceed with profiling {len(entries)} activation functions? (y/N): ")
        if response.lower() != "y":
            print("Aborted.")
            return

        successful_profiles = []
        failed_profiles = []

        # Process each entry
        for i, entry in enumerate(entries):
            print(f"\n{'#'*60}")
            print(f"Processing {i+1}/{len(entries)}: {entry['func_name']}")
            print(f"{'#'*60}")

            # Modify the test file to activate only this entry
            modify_test_file(test_file, entries, i, start_line, end_line)

            # Run the profiling
            success = run_profiling_command(entry["func_name"])

            if success:
                successful_profiles.append(entry["func_name"])
                print(f"✅ SUCCESS: {entry['func_name']}")
            else:
                failed_profiles.append(entry["func_name"])
                print(f"❌ FAILED: {entry['func_name']}")

            # Add a small delay between runs
            time.sleep(2)

        # Summary
        print(f"\n{'='*60}")
        print("PROFILING COMPLETE")
        print(f"{'='*60}")
        print(f"Successful: {len(successful_profiles)}")
        for func in successful_profiles:
            print(f"  ✅ {func}")

        if failed_profiles:
            print(f"\nFailed: {len(failed_profiles)}")
            for func in failed_profiles:
                print(f"  ❌ {func}")

        print(f"\nProfile results should be in: /home/ubuntu/logical_not/tt-metal/generated/profiler/reports/")

    finally:
        # Restore the original file
        print(f"\nRestoring original test file...")
        with open(test_file, "w") as f:
            f.write(original_content)
        print("✅ Original file restored")


if __name__ == "__main__":
    main()
