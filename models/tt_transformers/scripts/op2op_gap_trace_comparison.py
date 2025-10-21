#!/usr/bin/env python3
"""
Script to run tracy performance profiling with customizable parameters.
Runs tracy profiling separately for each specified sequence length.
Tracy signposts are always enabled for op2op gap analysis.

Usage:
    python op2op_gap_comparison.py --hf-model MODEL_NAME --mesh-device DEVICE_ID --seq-lens SEQ_LEN [SEQ_LEN ...] --test-path TEST_PATH

Examples:
    # Single sequence length
    python op2op_gap_trace_comparison.py --hf-model "meta-llama/Llama-3.1-8B-Instruct" --mesh-device "T3K" --seq-lens 128 \
        --test-path "models/tt_transformers/demo/simple_text_demo.py"

    # Multiple sequence lengths (runs tracy separately for each)
    python op2op_gap_trace_comparison.py --hf-model "meta-llama/Llama-3.1-8B-Instruct" --mesh-device "T3K" --seq-lens 128 256 512 \
        --test-path "models/tt_transformers/demo/simple_text_demo.py"

    # Decode lengths (1, 2, 4, 8 are automatically mapped to 1024, 2048, 4096, 8192)
    python op2op_gap_trace_comparison.py --hf-model "meta-llama/Llama-3.1-8B-Instruct" --mesh-device "T3K" --seq-lens 1 2 4 8 \
        --test-path "models/tt_transformers/demo/simple_text_demo.py"

    # Programmatic usage - import and use the function directly
    from op2op_gap_trace_comparison import run_tracy_profiling
    csv_paths = run_tracy_profiling("meta-llama/Llama-3.1-8B-Instruct", "T3K", 128)
    print(f"Generated CSV files: {csv_paths}")
"""
import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


def filter_csv_by_signposts(csv_path):
    """
    Filter CSV to keep only rows between start_prefill_perf_test and end_prefill_perf_test signposts.

    Args:
        csv_path: Path to the CSV file to filter

    Returns:
        bool: True if filtering was successful, False otherwise
    """
    try:
        # Read the CSV file
        with open(csv_path, "r", newline="") as f:
            reader = csv.reader(f)
            rows = list(reader)

        if len(rows) <= 1:
            print(f"CSV file {csv_path} has no data rows to filter")
            return False

        # Keep the header row
        header = rows[0]
        data_rows = rows[1:]

        # Find the signpost rows
        start_idx = None
        end_idx = None

        for i, row in enumerate(data_rows):
            # Check if any column contains the signpost markers
            row_str = ",".join(row)
            if "start_prefill_perf_test" in row_str:
                start_idx = i
            elif "end_prefill_perf_test" in row_str:
                end_idx = i
                break  # Stop once we find the end marker

        if start_idx is None or end_idx is None:
            print(f"Warning: Could not find signpost markers in {csv_path}")
            print(f"  start_prefill_perf_test found: {start_idx is not None}")
            print(f"  end_prefill_perf_test found: {end_idx is not None}")
            return False

        # Keep only rows between the signposts (exclusive)
        filtered_rows = data_rows[start_idx + 1 : end_idx]

        print(f"Filtered CSV: kept {len(filtered_rows)} rows (removed {len(data_rows) - len(filtered_rows)} rows)")

        # Write the filtered data back to the file
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(filtered_rows)

        return True

    except Exception as e:
        print(f"Error filtering CSV {csv_path}: {e}")
        return False


def rename_csv_files(csv_paths, mesh_device, hf_model, seq_len):
    """
    Rename CSV files to include the mesh device and model in the filename.

    Args:
        csv_paths: List of CSV file paths to rename
        mesh_device: String indicating the mesh device ID (e.g., "0", "1", "0,1")
        hf_model: String indicating the HuggingFace model name
        seq_len: Sequence length used in the test

    Returns:
        List of renamed CSV file paths
    """
    renamed_paths = []

    for csv_path in csv_paths:
        if not csv_path or not os.path.exists(csv_path):
            continue

        # Get the directory and filename
        csv_dir = os.path.dirname(csv_path)
        csv_filename = os.path.basename(csv_path)

        # Check if it's an ops_perf_results CSV file
        if csv_filename.startswith("ops_perf_results"):
            # Create new filename with mesh device and model
            # Clean up model name for filename (replace slashes and special chars)
            clean_model = hf_model.replace("/", "-").replace(" ", "_").replace(":", "-")

            # Clean up mesh device for filename (replace commas with underscores)
            clean_mesh = mesh_device.replace(",", "_")

            # Create the new filename: isl-{seq_len}_mesh-device-{device}_model-{model}.csv
            # Remove the timestamp part by creating a completely new filename
            new_filename = f"isl-{seq_len}_mesh-device-{clean_mesh}_model-{clean_model}.csv"
            new_path = os.path.join(csv_dir, new_filename)

            try:
                # Rename the file
                shutil.move(csv_path, new_path)
                renamed_paths.append(new_path)
                print(f"Renamed CSV file: {csv_path} -> {new_path}")

                # Filter the CSV to keep only rows between signposts
                print(f"Filtering CSV to keep only rows between signposts...")
                filter_csv_by_signposts(new_path)

            except Exception as e:
                print(f"Error renaming {csv_path}: {e}")
                renamed_paths.append(csv_path)  # Keep original path if rename fails
        else:
            # Keep original path if it's not an ops_perf_results file
            renamed_paths.append(csv_path)

    return renamed_paths


def extract_csv_from_output(output_text):
    """
    Extract CSV file paths from tracy console output.

    Args:
        output_text: The console output text from tracy

    Returns:
        List of CSV file paths found in the output
    """
    csv_paths = []

    # Look for the specific tracy pattern:
    # "tracy.process_ops_logs:generate_reports:909 - OPs csv generated at: /path/to/file.csv"
    # The path will always contain "generated/profiler/reports/" somewhere in it
    patterns = [
        r"OPs csv generated at:\s*(.+\.csv)",
    ]

    for pattern in patterns:
        matches = re.findall(pattern, output_text)
        csv_paths.extend(matches)

    # Clean up paths (remove quotes, extra whitespace)
    csv_paths = [path.strip().strip("\"'") for path in csv_paths]

    # Filter to only include paths that contain "generated/profiler/reports/"
    csv_paths = [path for path in csv_paths if "generated/profiler/reports/" in path]

    # Remove duplicates and return
    return list(set(csv_paths))


def find_tracy_csv_files(tt_metal_root):
    """
    Find tracy generated CSV files with various naming patterns:
    - isl-128_mesh-device-{device}_model-{model}_ops_perf_results_YYYY_MM_DD_HH_MM_SS.csv

    Args:
        tt_metal_root: Path to tt-metal root directory

    Returns:
        List of absolute paths to CSV files
    """
    csv_paths = []

    # Look for tracy generated CSV files with pattern: ops_perf_results_YYYY_MM_DD_HH_MM_SS.csv
    reports_dir = tt_metal_root / "generated" / "profiler" / "reports"

    if reports_dir.exists():
        # Look for directories with timestamp pattern YYYY_MM_DD_HH_MM_SS
        timestamp_dirs = [d for d in reports_dir.iterdir() if d.is_dir() and d.name.count("_") == 5]

        # Sort by modification time to get the most recent first
        timestamp_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)

        for timestamp_dir in timestamp_dirs:
            # Look for ops_perf_results_*.csv files in each timestamp directory
            csv_files = list(timestamp_dir.glob("ops_perf_results_*.csv"))
            csv_paths.extend([str(f.absolute()) for f in csv_files])

            # Also look for renamed files with mesh device and model
            renamed_csv_files = list(timestamp_dir.glob("isl-128_mesh-*_model-*_ops_perf_results_*.csv"))
            csv_paths.extend([str(f.absolute()) for f in renamed_csv_files])

    # Also search recursively in case the structure is different
    csv_patterns = [
        "**/ops_perf_results_*.csv",
        "**/isl-128_mesh-*_model-*_ops_perf_results_*.csv",
        "generated/profiler/reports/**/ops_perf_results_*.csv",
        "generated/profiler/reports/**/isl-128_mesh-*_model-*_ops_perf_results_*.csv",
    ]

    for pattern in csv_patterns:
        csv_files = list(tt_metal_root.glob(pattern))
        csv_paths.extend([str(f.absolute()) for f in csv_files])

    # Remove duplicates and sort
    return sorted(list(set(csv_paths)))


def run_tracy_profiling(hf_model, mesh_device, included_seq_lens, test_path=None):
    """
    Run tracy performance profiling and return CSV paths as a list.

    Args:
        hf_model: HuggingFace model name
        mesh_device: Mesh device ID
        included_seq_lens: Sequence length to include in the test
        test_path: Path to test file (optional)

    Returns:
        List of CSV file paths generated by tracy
    """
    # Build environment variables

    env_vars = {
        "HF_MODEL": hf_model,
        "MESH_DEVICE": mesh_device,
        "INCLUDED_SEQ_LENS": str(included_seq_lens),
        "TEST_FILE": test_path,
        "ENABLE_TRACY_SIGNPOSTS": "1",
    }

    # Construct the pytest command - always use trace-prefill and performance
    # pytest_cmd = f"pytest {test_path} -k 'trace-prefill and performance'"
    pytest_cmd = f"pytest {test_path} -k 'batch-1 and performance' --num_layers 1 --max_generated_tokens 2"

    # Construct the full tracy command
    tracy_cmd = f'python -m tracy -r -m -v -p "{pytest_cmd}"'

    # Print the command and environment variables
    print("\n" + "=" * 80)
    print("Command to be executed:")
    print("=" * 80)
    print(f"Command: {tracy_cmd}")
    print(f"\nEnvironment variables:")
    for key, value in env_vars.items():
        print(f"  {key}={value}")
    print("=" * 80 + "\n")

    try:
        # Set environment variables and run the command
        env = os.environ.copy()
        env.update(env_vars)

        print("Executing command...")
        result = subprocess.run(
            tracy_cmd,
            shell=True,
            env=env,
            cwd=Path(__file__).parent.parent.parent.parent,  # Go to tt-metal root
            capture_output=True,
            text=True,
        )

        # Only print output if there was an error
        if result.returncode != 0:
            print("\nCommand failed with error:")
            if result.stdout:
                print("stdout:")
                print(result.stdout)
            if result.stderr:
                print("stderr:")
                print(result.stderr)

        # Extract CSV paths from the console output
        csv_paths = []
        if result.returncode == 0:
            print("Command completed successfully. Extracting CSV paths from output...")

            # Extract CSV paths from stdout
            stdout_csv_paths = extract_csv_from_output(result.stdout)
            csv_paths.extend(stdout_csv_paths)

            # Also check stderr in case CSV info is printed there
            stderr_csv_paths = extract_csv_from_output(result.stderr)
            csv_paths.extend(stderr_csv_paths)

            # Remove duplicates
            csv_paths = list(set(csv_paths))

            if csv_paths:
                print(f"Found {len(csv_paths)} CSV file(s) from command output:")
                for csv_path in csv_paths:
                    print(f"  {csv_path}")

                # Rename CSV files with mesh device and model
                print(f"\nRenaming CSV files with mesh-device: {mesh_device}, model: {hf_model}")
                csv_paths = rename_csv_files(csv_paths, mesh_device, hf_model, included_seq_lens)
            else:
                print("No CSV file paths found in command output.")
                print("Searching for CSV files in generated/profiler/reports directory...")

                # Fallback: search for CSV files in the expected directory structure
                tt_metal_root = Path(__file__).parent.parent.parent.parent
                fallback_csv_paths = find_tracy_csv_files(tt_metal_root)

                if fallback_csv_paths:
                    print(f"Found {len(fallback_csv_paths)} CSV file(s) in reports directory:")
                    for csv_path in fallback_csv_paths:
                        print(f"  {csv_path}")

                    # Rename the fallback CSV files
                    print(f"\nRenaming fallback CSV files with mesh-device: {mesh_device}, model: {hf_model}")
                    csv_paths = rename_csv_files(fallback_csv_paths, mesh_device, hf_model, included_seq_lens)
                else:
                    print("No CSV files found in reports directory either.")

        return csv_paths

    except KeyboardInterrupt:
        print("\nCommand interrupted by user")
        return []
    except Exception as e:
        print(f"Error executing command: {e}")
        return []


def main():
    parser = argparse.ArgumentParser(
        description="Run tracy performance profiling with customizable parameters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Required arguments
    parser.add_argument(
        "--hf-model", required=True, help="HuggingFace model name (e.g., 'meta-llama/Llama-3.1-8B-Instruct')"
    )

    parser.add_argument("--mesh-device", required=True, help="Mesh device ID (e.g.,N150, N300, T3K, TG)")

    parser.add_argument(
        "--seq-lens",
        type=int,
        nargs="+",
        choices=[128, 256, 512, 1, 2, 4, 8],
        required=True,
        help="Sequence lengths to include in the test - can specify multiple (128, 256, 512, 1, 2, 4, 8)",
    )

    # Optional arguments
    parser.add_argument(
        "--test-path",
        default="models/tt_transformers/demo/simple_text_demo.py",
        help="Path to the test file (default: models/tt_transformers/demo/simple_text_demo.py)",
    )

    parser.add_argument(
        "--dry-run", action="store_true", help="Print the command that would be executed without running it"
    )

    args = parser.parse_args()

    if args.dry_run:
        print("\n" + "=" * 80)
        print("DRY RUN MODE - Commands that would be executed:")
        print("=" * 80)

        len_mapper = {
            1: 1024,
            2: 2048,
            4: 4096,
            8: 8192,
        }

        for seq_len in args.seq_lens:
            mapped_seq_len = len_mapper.get(seq_len, seq_len)

            print(f"\nFor seq_len: {seq_len} (mapped to {mapped_seq_len}):")
            print("-" * 80)

            # Build environment variables
            env_vars = {
                "HF_MODEL": args.hf_model,
                "MESH_DEVICE": args.mesh_device,
                "INCLUDED_SEQ_LENS": str(mapped_seq_len),
                "TEST_FILE": args.test_path,
                "ENABLE_TRACY_SIGNPOSTS": "1",
            }

            # Construct the pytest command
            pytest_cmd = f"pytest {args.test_path} -k 'batch-1 and performance' --num_layers 1 --max_generated_tokens 2"

            # Construct the full tracy command
            tracy_cmd = f'python -m tracy -r -m -v -p "{pytest_cmd}"'

            print(f"Command: {tracy_cmd}")
            print(f"\nEnvironment variables:")
            for key, value in env_vars.items():
                print(f"  {key}={value}")

        print("\n" + "=" * 80)
        print("Dry run complete - no commands executed")
        print("=" * 80 + "\n")
        return 0

    # Run tracy profiling and get CSV paths
    len_mapper = {
        1: 1024,
        2: 2048,
        4: 4096,
        8: 8192,
    }

    all_csv_paths = []

    # Run tracy profiling separately for each seq_len
    for seq_len in args.seq_lens:
        print(f"\n{'='*80}")
        print(f"Running tracy profiling for seq_len: {seq_len}")
        print(f"{'='*80}\n")

        # Map the seq_len if needed
        mapped_seq_len = len_mapper.get(seq_len, seq_len)

        csv_paths = run_tracy_profiling(
            hf_model=args.hf_model,
            mesh_device=args.mesh_device,
            included_seq_lens=mapped_seq_len,
            test_path=args.test_path,
        )

        all_csv_paths.extend(csv_paths)

        if csv_paths:
            print(f"\nCompleted seq_len {seq_len}. Generated CSV files: {csv_paths}")
        else:
            print(f"\nWarning: No CSV files found for seq_len {seq_len}")

    print(f"\n{'='*80}")
    print(f"All tracy profiling runs completed!")
    print(f"Total CSV files generated: {len(all_csv_paths)}")
    for csv_path in all_csv_paths:
        print(f"  - {csv_path}")
    print(f"{'='*80}\n")

    # Return success/failure based on whether CSV paths were found
    return 0 if all_csv_paths else 1


if __name__ == "__main__":
    sys.exit(main())
