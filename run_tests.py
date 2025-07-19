#!/usr/bin/env python3
import subprocess
import sys
from datetime import datetime


def run_command(command, log_file_name):
    """
    Executes a shell command and redirects its stdout and stderr to a log file.

    Args:
        command (list): The command to execute as a list of strings.
        log_file_name (str): The path to the log file.

    Returns:
        bool: True if the command was successful, False otherwise.
    """
    try:
        # Open the log file in write mode
        with open(log_file_name, "w") as log_file:
            # Execute the command
            # stdout is redirected to the log file
            # stderr is redirected to stdout, so it also goes to the log file
            result = subprocess.run(
                command,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                check=True,  # Raise an exception for non-zero exit codes
                text=True,  # Decode stdout/stderr as text
            )
        print(f"  -> SUCCESS: Command completed successfully.")
        return True
    except FileNotFoundError:
        print(f"  -> ERROR: Command not found. Is 'python' in your system's PATH?")
        return False
    except subprocess.CalledProcessError as e:
        print(f"  -> FAILED: Command exited with error code {e.returncode}.")
        print(f"  -> Check log for details: {log_file_name}")
        return False
    except Exception as e:
        print(f"  -> An unexpected error occurred: {e}")
        return False


def main():
    """Main function to define and run all test commands."""

    # List of module names to be tested
    module_names = [
        "data_movement.backward.concat_bw.concat_bw",
        "data_movement.concat.concat_interleaved",
        "data_movement.concat.concat_interleaved_n_tensors",
        "data_movement.concat.concat_pytorch2",
        "data_movement.concat.concat_sharded",
        "data_movement.copy.copy",
        "data_movement.embedding.embedding_pytorch2",
        "data_movement.expand.expand_pytorch2",
        "data_movement.fill.fill_pytorch2",
        "data_movement.index_select.index_select_pytorch2",
        "data_movement.interleaved_to_sharded.interleaved_to_sharded_e2e",
        "data_movement.nonzero.nonzero",
        "data_movement.permute.permute",
        "data_movement.permute.permute_pytorch2_rm",
        "data_movement.permute.permute_pytorch2_tiled",
        "data_movement.repeat.repeat",
        "data_movement.repeat.repeat_pytorch2",
        "data_movement.repeat_interleave.repeat_interleave",
        "data_movement.reshape.reshape",
        "data_movement.slice.slice_forge",
        "data_movement.slice.slice_pytorch2_rm",
        "data_movement.slice.slice_pytorch2_tiled",
        "data_movement.split.split_pytorch2",
        "data_movement.squeeze.squeeze_pytorch2",
        "data_movement.stack.stack_pytorch2",
        "data_movement.transpose.t_pytorch2",
        "data_movement.transpose.transpose_forge",
        "data_movement.transpose.transpose_interleaved",
        "data_movement.transpose.transpose_pytorch2",
        "data_movement.unsqueeze.unsqueeze_pytorch2",
        "data_movement.view.view_pytorch2",
        "data_movement.view.view_tt_torch",
    ]

    print("--- Starting Test Execution ---")

    # Loop through each module name and execute the corresponding command
    for i, module_name in enumerate(module_names):
        # Generate a timestamp for the log file name
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # Construct the log file name
        log_file_name = f"test_run_{module_name}_{timestamp}.log"

        # Construct the command as a list of arguments
        command = [
            "python",
            "-u",  # Unbuffered binary stdout and stderr
            "tests/sweep_framework/sweeps_runner.py",
            "--database",
            "postgres",
            "--module-name",
            module_name,
        ]

        print(f"\n[{i+1}/{len(module_names)}] Running test for module: {module_name}")

        # Execute the command
        success = run_command(command, log_file_name)

        # If a command fails, stop the script
        if not success:
            print("\n--- Halting script due to test failure. ---")
            sys.exit(1)  # Exit with a non-zero status code to indicate failure

    print("\n--- All tests completed successfully. ---")


if __name__ == "__main__":
    main()
