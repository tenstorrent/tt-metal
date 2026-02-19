# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

#!/usr/bin/env python3
"""
Generic Operations Tracer

Takes any model test path and extracts ttnn operations by running it with --trace-params enabled.
Automatically detects if it's a pytest test or standalone Python script.
Uses the new simple parameter tracer instead of graph tracing.

Usage:
    python generic_ops_tracer.py <test_path> [--output-dir <dir>] [--store]

Examples (Pytest):
    python generic_ops_tracer.py models/demos/wormhole/distilbert/demo/demo.py::test_demo
    python generic_ops_tracer.py models/demos/vision/classification/resnet50/wormhole/demo/demo.py::test_demo_sample
    python generic_ops_tracer.py /path/to/test.py::test_function --store

Examples (Standalone Python):
    python generic_ops_tracer.py models/demos/vision/classification/resnet50/wormhole/demo/demo.py
    python generic_ops_tracer.py models/experimental/some_model/run_model.py --store
    python generic_ops_tracer.py /path/to/script.py --output-dir ./my_traces
"""

import sys
import os
import subprocess
import json
import hashlib
from tqdm import tqdm
import argparse
from datetime import datetime
from pathlib import Path


def get_base_dir():
    """Get the tt-metal base directory from PYTHONPATH or current working directory"""
    pythonpath = os.environ.get("PYTHONPATH", "")
    if pythonpath:
        paths = pythonpath.split(":")
        for path in paths:
            if "tt-metal" in path:
                if path.endswith("tt-metal"):
                    return path
                parts = path.split("tt-metal")
                if parts:
                    return parts[0] + "tt-metal"
    current_dir = os.getcwd()
    if "tt-metal" in current_dir:
        parts = current_dir.split("tt-metal")
        return parts[0] + "tt-metal"
    return current_dir


BASE_DIR = get_base_dir()


def get_machine_info():
    """Get machine info (board type, device series, card count, and device count) using tt-smi command."""
    try:
        result = subprocess.run(["tt-smi", "-ls"], capture_output=True, text=True, timeout=10)
        if result.returncode != 0 or not result.stdout.strip():
            return None

        # Parse "All available boards" section for total device count
        all_devices = []
        in_all_boards = False

        # Parse "Boards that can be reset" section for card count
        in_reset_table = False
        machines = {}

        for line in result.stdout.split("\n"):
            # Track when we enter "All available boards" section
            if "All available boards on host" in line:
                in_all_boards = True
                in_reset_table = False
                continue

            # Track when we enter "Boards that can be reset" section
            if "Boards that can be reset" in line:
                in_all_boards = False
                in_reset_table = True
                continue

            # Parse device rows in "All available boards" section
            if in_all_boards and line.strip().startswith("│"):
                parts = [p.strip() for p in line.split("│") if p.strip()]
                if len(parts) >= 3:
                    pci_dev_id = parts[0]
                    board_type = parts[1]
                    device_series = parts[2].rstrip("LR").strip()  # Remove L/R suffix
                    if board_type and device_series and board_type != "Board Type" and pci_dev_id != "PCI Dev ID":
                        all_devices.append((board_type, device_series))

            # Count cards from "Boards that can be reset" section
            if in_reset_table and line.strip().startswith("│"):
                parts = [p.strip() for p in line.split("│") if p.strip()]
                if len(parts) >= 3:
                    board_type = parts[1]
                    device_series = parts[2].rstrip("LR").strip()
                    if board_type and device_series and board_type != "Board Type":
                        key = (board_type, device_series)
                        machines[key] = machines.get(key, 0) + 1

        if machines and all_devices:
            (board_type, device_series), card_count = max(machines.items(), key=lambda x: x[1])
            # Count total devices from "All available boards" section
            device_count = len(all_devices)

            return {
                "board_type": board_type,
                "device_series": device_series,
                "card_count": card_count,
                "device_count": device_count,
            }
        return None
    except Exception:
        return None


def load_valid_operations():
    """Load valid operations from Allops.txt.

    Returns a set of normalized operation names (dot notation) for efficient lookup.
    """
    valid_ops = set()
    allops_file = os.path.join(BASE_DIR, "tests/sweep_framework/Allops.txt")

    try:
        with open(allops_file, "r") as f:
            for line in f:
                op_name = line.strip()
                if op_name:
                    # Normalize to dot notation for consistent comparison
                    normalized = normalize_op_name(op_name)
                    valid_ops.add(normalized)
        print(f"📋 Loaded {len(valid_ops)} valid operations from Allops.txt")
        return valid_ops
    except FileNotFoundError:
        print(f"⚠️ Allops.txt not found at {allops_file}, will include all ttnn operations")
        return None
    except Exception as e:
        print(f"⚠️ Error loading Allops.txt: {e}, will include all ttnn operations")
        return None


def normalize_op_name(op_name: str) -> str:
    """Normalize operation name to use dot notation.

    Converts C++ style (ttnn::op) to Python style (ttnn.op) for consistent comparison.
    """
    return op_name.replace("::", ".")


def get_excluded_operations():
    """Operations to exclude from tracing.

    Uses dot notation. Will be normalized during comparison to handle both formats.
    """
    return {
        # Memory management operations
        "ttnn.allocate_tensor_on_device",
        "ttnn.deallocate",
        "ttnn.move",
        "ttnn.reallocate",
        "ttnn.copy_host_to_device_tensor",
        "ttnn.copy_device_to_host_tensor",
        # Data conversion operations
        "ttnn.to_device",
        "ttnn.to_dtype",
        "ttnn.to_layout",
        "ttnn.to_memory_config",
        "ttnn.to_torch",
        "ttnn.from_device",
        "ttnn.from_torch",
        # Utility operations
        "ttnn.view",
        "ttnn.dump_tensor",
        "ttnn.load_tensor",
        "ttnn.as_tensor",
        # Other excluded operations
        "ttnn.unary_chain",
        "ttnn.pearson_correlation_coefficient",
        "ttnn.dram_prefetcher",
        "ttnn.complex_tensor",
        # Primitive/example operations
        "ttnn.prim.binary",
        "ttnn.prim.example",
        "ttnn.prim.example_multiple_return",
        "ttnn.composite_example",
        "ttnn.composite_example_multiple_return",
    }


def is_valid_operation(op_name, valid_operations, excluded_operations):
    """Check if operation should be included in the trace.

    Normalizes operation names to handle both C++ (::) and Python (.) formats.

    Args:
        op_name: Operation name to check (can be :: or . notation)
        valid_operations: Pre-normalized set of valid operations (dot notation) or None
        excluded_operations: Set of excluded operations (dot notation)
    """
    # Normalize the op_name once
    normalized_op = normalize_op_name(op_name)

    # Check exclusions first (already normalized)
    if normalized_op in excluded_operations:
        return False

    if valid_operations is None:
        return op_name.startswith("ttnn::") or op_name.startswith("ttnn.") or op_name.startswith("ttnn::experimental::")

    # valid_operations is already normalized in load_valid_operations(), so direct lookup
    return normalized_op in valid_operations


def collect_operation_jsons(trace_dir):
    """Collect all operation JSON files from the trace directory"""
    trace_path = Path(trace_dir)
    if not trace_path.exists():
        return []

    # Find all JSON files in the operation_parameters directory
    json_files = sorted(trace_path.glob("*.json"))
    return json_files


def convert_json_to_master_format(json_file, test_source, machine_info):
    """Convert individual JSON file to master format"""
    try:
        with open(json_file, "r") as f:
            data = json.load(f)

        operation_name = data.get("operation_name", "unknown")

        # Convert args format from new tracer to master format
        arguments = {}

        # Track mesh_device info (extracted from args)
        mesh_device_info = None

        # Add positional args with arg0, arg1, arg2, etc. labels
        for arg in data.get("args", []):
            position = arg.get("position", 0)
            arg_key = f"arg{position}"
            arg_value = arg.get("value", {})

            # Extract mesh_device info from tensor arguments
            if isinstance(arg_value, dict) and "mesh_device" in arg_value:
                mesh_data = arg_value["mesh_device"]

                # Extract device info (only once, they should all be the same)
                if mesh_device_info is None:
                    mesh_device_info = {
                        "device_ids": mesh_data.get("device_ids", []),
                        "device_count": len(mesh_data.get("device_ids", [])),
                        "mesh_device_shape": mesh_data.get("shape", []),
                    }

                # Extract tensor placement info and store it per-tensor
                placements = mesh_data.get("placements", [])
                distribution_shape = mesh_data.get("distribution_shape", [])
                mesh_shape = mesh_data.get("shape", [])

                # Remove mesh_device from the argument value
                arg_value_clean = {k: v for k, v in arg_value.items() if k != "mesh_device"}

                # Add per-tensor placement info if it exists
                if placements:
                    arg_value_clean["tensor_placement"] = {
                        "placement": str(placements),
                        "distribution_shape": str(distribution_shape),
                        "mesh_device_shape": str(mesh_shape),
                    }

                # Remove redundant shape if it matches original_shape
                if "shape" in arg_value_clean and "original_shape" in arg_value_clean:
                    if arg_value_clean["shape"] == arg_value_clean["original_shape"]:
                        del arg_value_clean["shape"]

                # Remove redundant dtype if it matches original_dtype
                if "dtype" in arg_value_clean and "original_dtype" in arg_value_clean:
                    if arg_value_clean["dtype"] == arg_value_clean["original_dtype"]:
                        del arg_value_clean["dtype"]

                arguments[arg_key] = arg_value_clean
            else:
                # Also clean up non-mesh tensors
                if isinstance(arg_value, dict):
                    arg_value_clean = arg_value.copy()

                    # Remove redundant shape if it matches original_shape
                    if "shape" in arg_value_clean and "original_shape" in arg_value_clean:
                        if arg_value_clean["shape"] == arg_value_clean["original_shape"]:
                            del arg_value_clean["shape"]

                    # Remove redundant dtype if it matches original_dtype
                    if "dtype" in arg_value_clean and "original_dtype" in arg_value_clean:
                        if arg_value_clean["dtype"] == arg_value_clean["original_dtype"]:
                            del arg_value_clean["dtype"]

                    arguments[arg_key] = arg_value_clean
                else:
                    arguments[arg_key] = arg_value

        # Add kwargs as named arguments (they come after positional args)
        kwargs = data.get("kwargs", {})
        for key, value in kwargs.items():
            # Also check kwargs for mesh_device info
            if isinstance(value, dict) and "mesh_device" in value:
                mesh_data = value["mesh_device"]

                if mesh_device_info is None:
                    mesh_device_info = {
                        "device_ids": mesh_data.get("device_ids", []),
                        "device_count": len(mesh_data.get("device_ids", [])),
                        "mesh_device_shape": mesh_data.get("shape", []),
                    }

                placements = mesh_data.get("placements", [])
                distribution_shape = mesh_data.get("distribution_shape", [])
                mesh_shape = mesh_data.get("shape", [])

                value_clean = {k: v for k, v in value.items() if k != "mesh_device"}

                # Add per-tensor placement info if it exists
                if placements:
                    value_clean["tensor_placement"] = {
                        "placement": str(placements),
                        "distribution_shape": str(distribution_shape),
                        "mesh_device_shape": str(mesh_shape),
                    }

                # Remove redundant shape if it matches original_shape
                if "shape" in value_clean and "original_shape" in value_clean:
                    if value_clean["shape"] == value_clean["original_shape"]:
                        del value_clean["shape"]

                # Remove redundant dtype if it matches original_dtype
                if "dtype" in value_clean and "original_dtype" in value_clean:
                    if value_clean["dtype"] == value_clean["original_dtype"]:
                        del value_clean["dtype"]

                arguments[key] = value_clean
            else:
                # Also clean up non-mesh tensors in kwargs
                if isinstance(value, dict):
                    value_clean = value.copy()

                    # Remove redundant shape if it matches original_shape
                    if "shape" in value_clean and "original_shape" in value_clean:
                        if value_clean["shape"] == value_clean["original_shape"]:
                            del value_clean["shape"]

                    # Remove redundant dtype if it matches original_dtype
                    if "dtype" in value_clean and "original_dtype" in value_clean:
                        if value_clean["dtype"] == value_clean["original_dtype"]:
                            del value_clean["dtype"]

                    arguments[key] = value_clean
                else:
                    arguments[key] = value

        # Merge mesh_device info into machine_info
        enhanced_machine_info = machine_info.copy() if machine_info else {}

        if mesh_device_info:
            enhanced_machine_info.update(mesh_device_info)

        # Note: tensor_placements are now stored per-tensor in the arguments
        # instead of globally in machine_info, to avoid ambiguity

        return {
            "operation": operation_name,
            "arguments": arguments,
            "source": test_source,
            "machine_info": enhanced_machine_info,
        }
    except Exception as e:
        print(f"⚠️ Error processing {json_file}: {e}")
        return None


def update_master_file(master_file_path, operations, test_source):
    """Update master JSON file with operations"""
    import hashlib

    # Load existing master data
    master_data = {"operations": {}, "metadata": {"models": [], "unique_operations": 0, "total_configurations": 0}}

    if os.path.exists(master_file_path) and os.path.getsize(master_file_path) > 0:
        try:
            with open(master_file_path, "r") as f:
                content = f.read().strip()
                if content:
                    master_data = json.loads(content)
        except Exception as e:
            print(f"⚠️ Warning: Could not load existing master JSON: {e}")
            print(f"   Starting with empty master data")

    # Find the current max config_id
    max_config_id = 0
    for op_data in master_data.get("operations", {}).values():
        for config in op_data.get("configurations", []):
            if isinstance(config, dict) and "config_id" in config:
                max_config_id = max(max_config_id, config["config_id"])

    # Group operations by operation name
    new_configs_added = 0
    next_config_id = max_config_id + 1

    print(f"\n💾 Updating master JSON with {len(operations)} operations...")
    for operation in tqdm(operations, desc="Updating master", unit="op"):
        if not operation:
            continue

        op_name = operation.get("operation", "unknown")
        op_args = operation.get("arguments", [])

        # Initialize operation entry if not exists
        if op_name not in master_data["operations"]:
            master_data["operations"][op_name] = {"configurations": []}

        # Create argument signature for deduplication
        args_str = json.dumps(op_args, sort_keys=True, default=str)
        arg_signature = hashlib.md5(args_str.encode()).hexdigest()

        # Check if this configuration already exists
        matching_config = None
        for existing_config in master_data["operations"][op_name]["configurations"]:
            if isinstance(existing_config, dict) and "arguments" in existing_config:
                existing_args = existing_config["arguments"]
                existing_sig = hashlib.md5(json.dumps(existing_args, sort_keys=True, default=str).encode()).hexdigest()
                if existing_sig == arg_signature:
                    matching_config = existing_config
                    break

        if matching_config is None:
            # New configuration - assign new config_id
            # Compute config_hash for stable tracking (same logic as load_ttnn_ops_data_v2.py)
            machine_info = operation.get("machine_info")

            # Extract hardware tuple
            hardware = None
            if machine_info:
                board_type = machine_info.get("board_type")
                if board_type:
                    device_series = machine_info.get("device_series")
                    if isinstance(device_series, list):
                        device_series = device_series[0] if device_series else None
                    hardware = (board_type, device_series, machine_info.get("card_count", 1))

            # Extract mesh config
            mesh_config = None
            if machine_info and "tensor_placements" in machine_info:
                placements = machine_info.get("tensor_placements", [])
                if placements:
                    p = placements[0]
                    mesh_shape_str = p.get("mesh_device_shape")
                    if mesh_shape_str:
                        try:
                            mesh_shape = (
                                json.loads(mesh_shape_str) if isinstance(mesh_shape_str, str) else mesh_shape_str
                            )
                            if mesh_shape:
                                import re

                                placement_str = p.get("placement", "")
                                shard_dim = None
                                if "PlacementShard" in placement_str:
                                    match = re.search(r"PlacementShard\((\d+)\)", placement_str)
                                    if match:
                                        shard_dim = int(match.group(1))
                                mesh_config = {
                                    "mesh_shape": mesh_shape,
                                    "placement_type": "shard" if shard_dim is not None else "replicate",
                                    "shard_dim": shard_dim,
                                }
                        except:
                            pass

            # Compute SHA-256 hash
            normalized = {"operation": op_name, "arguments": op_args, "hardware": hardware, "mesh": mesh_config}
            config_hash = hashlib.sha256(json.dumps(normalized, sort_keys=True).encode()).hexdigest()

            config_entry = {
                "config_id": next_config_id,
                "config_hash": config_hash,
                "arguments": op_args,
                "executions": [
                    {
                        "source": test_source,
                        "machine_info": machine_info,
                        "count": operation.get("execution_count", 1),
                    }
                ],
            }

            master_data["operations"][op_name]["configurations"].append(config_entry)
            new_configs_added += 1
            next_config_id += 1
        else:
            # Configuration exists - check if this (source, machine_info) pair exists
            if isinstance(matching_config, dict):
                # Get or create executions list
                if "executions" not in matching_config:
                    # Migrate old format to new format
                    old_source = matching_config.get("source", "")
                    old_machine_info = matching_config.get("machine_info")
                    old_count = matching_config.get("execution_count", 1)

                    # Handle old format where source could be string or list
                    if isinstance(old_source, str):
                        sources = [old_source] if old_source else []
                    else:
                        sources = old_source if old_source else []

                    # Handle old format where machine_info could be dict or list
                    if isinstance(old_machine_info, dict):
                        machines = [old_machine_info]
                    elif isinstance(old_machine_info, list):
                        machines = old_machine_info
                    else:
                        machines = []

                    # Create executions from old format (best effort - can't recover exact pairs)
                    matching_config["executions"] = []
                    if sources and machines:
                        # Create all combinations (we lost the original pairing)
                        for src in sources:
                            for machine in machines:
                                matching_config["executions"].append(
                                    {
                                        "source": src,
                                        "machine_info": machine,
                                        "count": old_count,
                                    }
                                )
                    elif sources:
                        for src in sources:
                            matching_config["executions"].append(
                                {
                                    "source": src,
                                    "machine_info": None,
                                    "count": old_count,
                                }
                            )

                    # Remove old fields
                    matching_config.pop("source", None)
                    matching_config.pop("machine_info", None)
                    matching_config.pop("execution_count", None)

                # Check if this (source, machine_info) pair already exists
                new_source = test_source
                new_machine_info = operation.get("machine_info")
                new_count = operation.get("execution_count", 1)

                found_execution = None
                for execution in matching_config["executions"]:
                    if execution["source"] == new_source:
                        # Check if machine_info matches
                        exec_machine = execution.get("machine_info")
                        if exec_machine is None and new_machine_info is None:
                            found_execution = execution
                            break
                        elif exec_machine and new_machine_info:
                            # Compare complete machine_info (all fields must match)
                            # Convert to JSON strings for deep comparison
                            exec_machine_str = json.dumps(exec_machine, sort_keys=True, default=str)
                            new_machine_str = json.dumps(new_machine_info, sort_keys=True, default=str)
                            if exec_machine_str == new_machine_str:
                                found_execution = execution
                                break

                if found_execution:
                    # Update existing execution - take max count
                    found_execution["count"] = max(found_execution.get("count", 1), new_count)
                else:
                    # Add new execution entry
                    matching_config["executions"].append(
                        {
                            "source": new_source,
                            "machine_info": new_machine_info,
                            "count": new_count,
                        }
                    )

    # Update metadata
    if test_source not in master_data["metadata"]["models"]:
        master_data["metadata"]["models"].append(test_source)

    total_configurations = sum(len(op_data["configurations"]) for op_data in master_data["operations"].values())
    unique_operations = len(master_data["operations"])

    # Create operation summary with config counts
    operations_summary = {}
    for op_name, op_data in sorted(master_data["operations"].items()):
        config_count = len(op_data["configurations"])
        operations_summary[op_name] = config_count

    master_data["metadata"]["unique_operations"] = unique_operations
    master_data["metadata"]["total_configurations"] = total_configurations
    master_data["metadata"]["operations_summary"] = operations_summary
    master_data["metadata"]["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Save master file
    try:
        with open(master_file_path, "w") as f:
            json.dump(master_data, f, indent=2, default=str)
    except Exception as e:
        print(f"❌ Error saving master file: {e}")

    return new_configs_added


def detect_pytest_tests(test_path):
    """Detect if a file/path contains pytest test cases"""
    try:
        python_cmd = os.path.join(BASE_DIR, "python_env/bin/python")
        result = subprocess.run(
            [python_cmd, "-m", "pytest", test_path, "--collect-only", "-q"],
            cwd=BASE_DIR,
            capture_output=True,
            text=True,
            timeout=60,
        )

        if result.returncode == 0:
            output = result.stdout.lower()
            if "test" in output or "collected" in output:
                if "no tests collected" in output or "collected 0" in output:
                    return False
                return True
        return False
    except Exception:
        return False


def run_test_with_tracing(test_path, output_dir, keep_traces=False, debug_mode=False, extra_args=None):
    """Run test with --trace-params flag and collect operation JSONs"""
    extra_args = extra_args or []

    print(f"🚀 Running test with parameter tracing...")

    # Show deprecation warning if debug flag was used
    if debug_mode:
        print(f"⚠️  Note: --debug flag is deprecated (live output is now always enabled)")

    # Use python executable from tt-metal environment
    # Try to find python_env, fall back to system python3 if not found (e.g., in Docker/CI)
    python_env_path = os.path.join(BASE_DIR, "python_env/bin/python")
    if os.path.exists(python_env_path):
        python_cmd = python_env_path
    else:
        # Fallback to system python3 (used in Docker containers)
        python_cmd = "python3"

    # Create a unique subdirectory for this run based on source name and timestamp
    # This prevents conflicts with previous runs
    test_basename = os.path.basename(test_path).replace(".py", "").replace("::", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_subdir = f"{test_basename}_{timestamp}"

    # Trace directory with unique subdirectory
    trace_dir = os.path.join(BASE_DIR, "generated/ttnn/reports/operation_parameters", unique_subdir)
    os.makedirs(trace_dir, exist_ok=True)

    print(f"📂 Trace directory: {trace_dir}")

    # Detect if this is a pytest test or standalone script
    is_pytest = "::" in test_path or detect_pytest_tests(test_path)

    if is_pytest:
        print(f"✅ Detected pytest test cases, running with pytest...")
        if extra_args:
            print(f"📎 Passing additional arguments: {' '.join(extra_args)}")

        cmd = [python_cmd, "-m", "pytest", test_path, "-v", "-s", "--trace-params"] + extra_args
    else:
        print(f"✅ No pytest cases detected, running as standalone Python script...")
        cmd = [python_cmd, test_path, "--trace-params"] + extra_args

    # Set environment variable to specify custom trace directory
    # The operation_tracer.py checks TTNN_OPERATION_TRACE_DIR env var
    env = os.environ.copy()
    env["TTNN_OPERATION_TRACE_DIR"] = trace_dir

    # Disable fast runtime mode to enable operation tracing
    # Fast mode skips the tracing decorator for performance
    env["TTNN_CONFIG_OVERRIDES"] = '{"enable_fast_runtime_mode": false}'

    # Run the command with custom environment (always show live output now)
    # Use a custom command wrapper with tee to capture output while showing it live
    import tempfile
    import re

    # Create a temp file to capture output
    tmp_output_fd, tmp_output_path = tempfile.mkstemp(suffix=".log", text=True)
    os.close(tmp_output_fd)  # Close fd, we'll open as file

    try:
        # Build command with tee to show output live AND save to file
        # Convert cmd list to properly quoted string for shell
        cmd_str = " ".join(f'"{arg}"' if " " in arg else arg for arg in cmd)
        tee_cmd = f"{cmd_str} 2>&1 | tee {tmp_output_path}"

        # Run with shell=True to use tee
        result = subprocess.run(tee_cmd, shell=True, cwd=BASE_DIR, text=True, env=env)

        # Read the captured output to parse test statistics
        with open(tmp_output_path, "r") as f:
            output_text = f.read()

        # Parse pytest results from output
        test_stats = {"passed": 0, "failed": 0, "total": 0}

        # Look for pytest summary line like: "1 failed, 1 passed in X.XXs"
        summary_match = re.search(r"(\d+)\s+failed.*?(\d+)\s+passed", output_text)
        if summary_match:
            test_stats["failed"] = int(summary_match.group(1))
            test_stats["passed"] = int(summary_match.group(2))
            test_stats["total"] = test_stats["passed"] + test_stats["failed"]
        else:
            # Check for only passed
            passed_match = re.search(r"(\d+)\s+passed", output_text)
            if passed_match:
                test_stats["passed"] = int(passed_match.group(1))
                test_stats["total"] = test_stats["passed"]
            # Check for only failed
            failed_match = re.search(r"(\d+)\s+failed", output_text)
            if failed_match:
                test_stats["failed"] = int(failed_match.group(1))
                test_stats["total"] += test_stats["failed"]
    finally:
        # Clean up temp file
        try:
            os.remove(tmp_output_path)
        except OSError:
            # Best-effort cleanup: ignore failures to remove the temp file
            pass

    # Collect generated JSON files from the unique subdirectory
    json_files = collect_operation_jsons(trace_dir)

    print(f"📊 Found {len(json_files)} operation trace files")

    # Create metadata file with source and machine info
    # This will be used when importing traces with --load
    metadata = {
        "test_source": test_path,
        "timestamp": datetime.now().isoformat(),
        "machine_info": get_machine_info(),
        "trace_count": len(json_files),
    }

    # Check for HF_MODEL and LLAMA_DIR environment variables
    if "models/tt_transformers/demo/simple_text_demo.py" in test_path:
        hf_model = os.environ.get("HF_MODEL")
        llama_dir = os.environ.get("LLAMA_DIR")
        if hf_model:
            metadata["HF_MODEL"] = hf_model
        if llama_dir:
            metadata["LLAMA_DIR"] = llama_dir

    # Write metadata file
    metadata_file = os.path.join(trace_dir, "_trace_metadata.json")
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    return {
        "success": result.returncode == 0,
        "exit_code": result.returncode,
        "trace_files": json_files,
        "trace_dir": trace_dir,
        "keep_traces": keep_traces,
        "output_dir": output_dir,
        "test_stats": test_stats,
    }


def parse_shard_spec_string(shard_spec_str):
    """
    Parse a ShardSpec string representation into a proper dictionary.

    Handles both correct and malformed ShardSpec strings from C++ operator<<.
    The C++ bug (missing closing brace) has been fixed, but this handles old traces.

    Example input:
    'ShardSpec{grid=[{"start":{"x":0,"y":0},"end":{"x":7,"y":7}}], shape=[32, 32], orientation=ShardOrientation::ROW_MAJOR}'

    Returns:
    {
        "grid": [{"start": {"x": 0, "y": 0}, "end": {"x": 7, "y": 7}}],
        "shape": [32, 32],
        "orientation": "ROW_MAJOR"
    }
    """
    import re

    if not isinstance(shard_spec_str, str) or not shard_spec_str.startswith("ShardSpec{"):
        return shard_spec_str

    try:
        result = {}

        # Extract grid - find the position and manually parse the JSON array
        eq_pos = shard_spec_str.find("grid=")
        if eq_pos != -1:
            grid_start = eq_pos + 5  # Move to '[' after '='
            # Find the matching ']' for the grid array by looking for '], shape='
            # (not '}], shape=' because there might be multiple ranges without } before ])
            shape_pos = shard_spec_str.find("], shape=", grid_start)
            if shape_pos != -1:
                grid_json = shard_spec_str[grid_start : shape_pos + 1]  # +1 to include ']'

                # Fix malformed JSON: if we have unbalanced braces, add missing '}'
                # This handles old traces where the C++ operator<< had a bug (now fixed in buffer.cpp)
                open_count = grid_json.count("{")
                close_count = grid_json.count("}")
                if open_count > close_count:
                    # Try to fix by adding missing closing braces
                    # Common C++ bug: {"end":{"x":7,"y":2}, should be {"end":{"x":7,"y":2}},
                    # Pattern: "y":<number>} needs an extra } when followed by , or ]

                    # Strategy 1: Smart pattern matching for end coordinates
                    # Find "y":<number>} followed by either (, {) or (]) and add }
                    test_json = re.sub(r'("y":\d+)\}(,\s*\{|])', r"\1}}\2", grid_json)

                    try:
                        result["grid"] = json.loads(test_json)
                    except json.JSONDecodeError as e:
                        # Strategy 2: If strategy 1 failed, add all missing braces before final ']'
                        missing = open_count - close_count
                        test_json = grid_json[:-1] + ("}" * missing) + grid_json[-1]
                        try:
                            result["grid"] = json.loads(test_json)
                        except json.JSONDecodeError as e2:
                            # Both strategies failed, log and skip
                            print(f"⚠️ Warning: Could not fix malformed grid JSON: {e2}")
                else:
                    # No missing braces, try to parse normally
                    try:
                        result["grid"] = json.loads(grid_json)
                    except json.JSONDecodeError as e:
                        # Log warning if parsing fails
                        print(f"⚠️ Warning: Could not parse grid JSON: {e}")

        # Extract shape - it's an array like [128, 576]
        shape_match = re.search(r"shape=\[(\d+),\s*(\d+)\]", shard_spec_str)
        if shape_match:
            result["shape"] = [int(shape_match.group(1)), int(shape_match.group(2))]

        # Extract orientation
        orientation_match = re.search(r"orientation=ShardOrientation::(\w+)", shard_spec_str)
        if orientation_match:
            result["orientation"] = orientation_match.group(1)

        return result if result else shard_spec_str

    except Exception as e:
        # Silently return original if parsing fails
        return shard_spec_str


def fix_memory_config_recursive(obj, fixed_count_ref):
    """
    Recursively search for memory_config with shard_spec strings and fix them.
    """
    if isinstance(obj, dict):
        # Check if this dict is a memory_config with shard_spec
        if "shard_spec" in obj and isinstance(obj["shard_spec"], str):
            if obj["shard_spec"].startswith("ShardSpec{"):
                parsed = parse_shard_spec_string(obj["shard_spec"])
                if isinstance(parsed, dict):
                    obj["shard_spec"] = parsed
                    fixed_count_ref[0] += 1

        # Recurse into all values
        for value in obj.values():
            fix_memory_config_recursive(value, fixed_count_ref)

    elif isinstance(obj, list):
        # Recurse into all items
        for item in obj:
            fix_memory_config_recursive(item, fixed_count_ref)


def fix_infinity_in_json_file(json_file):
    """
    Pre-process JSON file to fix invalid -Infinity, Infinity, and NaN values.
    These need to be strings for valid JSON.
    """
    import re

    print(f"🔧 Pre-processing JSON to fix infinity/nan values...")

    try:
        # Read the file as text
        with open(json_file, "r") as f:
            content = f.read()

        # Count occurrences (use same patterns as replacements)
        infinity_count = len(re.findall(r":\s*-Infinity\b", content)) + len(re.findall(r":\s*Infinity\b", content))
        nan_count = len(re.findall(r":\s*NaN\b", content))

        if infinity_count == 0 and nan_count == 0:
            print(f"   No infinity/nan values to fix")
            return 0

        # Replace invalid JSON values with strings
        # Match patterns like: "value": -Infinity
        content = re.sub(r":\s*-Infinity\b", ': "-inf"', content)
        content = re.sub(r":\s*Infinity\b", ': "inf"', content)
        content = re.sub(r":\s*NaN\b", ': "nan"', content)

        # Write back
        with open(json_file, "w") as f:
            f.write(content)

        print(f"✅ Fixed {infinity_count} infinity and {nan_count} NaN values")
        return infinity_count + nan_count

    except Exception as e:
        print(f"❌ Error fixing infinity/nan values: {e}")
        import traceback

        traceback.print_exc()
        return 0


def fix_memory_config_in_json(json_file):
    """
    Fix memory_config entries in the master JSON file by parsing shard_spec strings.
    This function modifies the JSON in-place.
    """
    print(f"🔧 Fixing memory config entries in {os.path.basename(json_file)}...")

    try:
        # First, fix any infinity/nan values that would prevent JSON loading
        fix_infinity_in_json_file(json_file)

        # Now load and process the JSON
        with open(json_file, "r") as f:
            data = json.load(f)

        # Use a list to pass by reference for counting
        fixed_count_ref = [0]

        # Recursively fix all shard_spec entries
        fix_memory_config_recursive(data, fixed_count_ref)

        # Write back the fixed JSON
        with open(json_file, "w") as f:
            json.dump(data, f, indent=2)

        print(f"✅ Fixed {fixed_count_ref[0]} shard_spec entries")
        return fixed_count_ref[0]

    except Exception as e:
        print(f"❌ Error fixing memory config: {e}")
        import traceback

        traceback.print_exc()
        return 0


def main():
    parser = argparse.ArgumentParser(
        description="TTNN Operations Tracer - Extract operation configurations from model tests or scripts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples (Pytest tests):
    python model_tracer/generic_ops_tracer.py test.py -k "test_pow"
    python model_tracer/generic_ops_tracer.py test.py --store

Examples (Standalone Python scripts):
    python model_tracer/generic_ops_tracer.py model.py
    python model_tracer/generic_ops_tracer.py model.py --store --output-dir ./my_traces

Examples (Import existing traces):
    python model_tracer/generic_ops_tracer.py --load /path/to/traces
        """,
    )
    parser.add_argument("test_path", nargs="?", help="Path to test file or script")
    parser.add_argument(
        "--output-dir",
        "-o",
        default="./model_tracer/traced_operations",
        help="Directory to save master JSON (default: ./model_tracer/traced_operations)",
    )
    parser.add_argument(
        "--store", "--keep-traces", action="store_true", help="Keep individual trace files (default: delete them)"
    )
    parser.add_argument("-d", "--debug", action="store_true", help="[DEPRECATED] Live output is now always enabled")
    parser.add_argument(
        "--load",
        "--from-trace-dir",
        type=str,
        help="Process existing trace directory and add to master JSON (skips test execution). "
        "Useful for importing traces collected on other machines with --store flag.",
    )

    # Handle explicit separator
    if "--" in sys.argv:
        separator_index = sys.argv.index("--")
        tracer_argv = sys.argv[1:separator_index]
        extra_args = sys.argv[separator_index + 1 :]
        args = parser.parse_args(tracer_argv)
    else:
        args, extra_args = parser.parse_known_args()

    # Either test_path or load must be provided
    if not args.test_path and not args.load:
        print("❌ Error: Either test_path or --load is required")
        parser.print_help()
        return 1

    if args.load and args.test_path:
        print("❌ Error: Cannot specify both test_path and --load")
        parser.print_help()
        return 1

    print("🚀 TTNN Operations Tracer (New Simple Tracer)")
    print("=" * 50)

    # Handle two modes: run test or process existing traces
    if args.load:
        print(f"📂 Processing existing traces from: {args.load}")
        if not os.path.isdir(args.load):
            print(f"❌ Error: Trace directory not found: {args.load}")
            return 1
        trace_dir = args.load
        # Find all JSON files in the trace directory, excluding metadata
        trace_files = [
            os.path.join(trace_dir, f)
            for f in os.listdir(trace_dir)
            if f.endswith(".json") and not f.startswith("_trace_")
        ]
        if not trace_files:
            print(f"❌ Error: No JSON trace files found in {args.load}")
            return 1
        result = {
            "success": True,
            "trace_files": sorted(trace_files),
            "trace_dir": trace_dir,
            "keep_traces": True,  # Always keep when processing existing traces
        }
        print(f"✅ Found {len(trace_files)} trace files")
    else:
        print(f"📁 {os.path.basename(args.test_path)}")
        if args.store:
            print(f"💾 Keeping individual trace files")
        if extra_args:
            print(f"📎 Extra arguments: {' '.join(extra_args)}")
        test_source = args.test_path

    print("=" * 50)

    try:
        # Run test with tracing (unless processing existing traces)
        if not args.load:
            result = run_test_with_tracing(args.test_path, args.output_dir, args.store, args.debug, extra_args)

        print("\n" + "=" * 50)
        print("📋 RESULTS")
        print("=" * 50)

        # Display test results if we ran tests (not from existing traces)
        if not args.load and "test_stats" in result:
            stats = result["test_stats"]
            if stats["total"] > 0:
                print(f"Test Results: ✅ {stats['passed']} passed, ❌ {stats['failed']} failed (Total: {stats['total']})")
            else:
                # Fallback if we couldn't parse the output
                print(f"Test Result: {'✅ PASSED' if result['success'] else '❌ FAILED'}")

        print(f"📊 Collected {len(result['trace_files'])} operation trace files")

        if result["trace_files"]:
            # Load valid operations and excluded operations
            valid_operations = load_valid_operations()
            excluded_operations = get_excluded_operations()
            machine_info = get_machine_info()

            # Extract test source name and possibly override machine_info from metadata
            if args.load:
                # Try to load metadata file if it exists
                metadata_file = os.path.join(args.load, "_trace_metadata.json")
                if os.path.exists(metadata_file):
                    try:
                        with open(metadata_file, "r") as f:
                            metadata = json.load(f)

                        # Use test_source from metadata
                        test_source = metadata.get("test_source", os.path.basename(os.path.abspath(args.load)))

                        # Convert to relative path if needed
                        if os.path.isabs(test_source) and BASE_DIR in test_source:
                            test_source = os.path.relpath(test_source, BASE_DIR)

                        # Append HF_MODEL/LLAMA_DIR from metadata if present
                        env_tags = []
                        if "HF_MODEL" in metadata:
                            env_tags.append(f"[HF_MODEL:{metadata['HF_MODEL']}]")
                        if "LLAMA_DIR" in metadata:
                            env_tags.append(f"[LLAMA_DIR:{metadata['LLAMA_DIR']}]")
                        if env_tags:
                            test_source = f"{test_source} {' '.join(env_tags)}"

                        # Use machine_info from metadata if present
                        if "machine_info" in metadata:
                            machine_info = metadata["machine_info"]
                            print(f"📋 Loaded metadata from trace directory")
                            print(f"   Original source: {metadata.get('test_source')}")
                            if "machine_info" in metadata and metadata["machine_info"]:
                                machine_desc = (
                                    metadata["machine_info"][0]
                                    if isinstance(metadata["machine_info"], list)
                                    else metadata["machine_info"]
                                )
                                if isinstance(machine_desc, dict):
                                    print(
                                        f"   Machine: {machine_desc.get('board_type')} {machine_desc.get('device_series')}"
                                    )
                    except Exception as e:
                        print(f"⚠️ Could not load metadata file: {e}")
                        test_source = os.path.basename(os.path.abspath(args.load))
                else:
                    # Fallback to directory name if no metadata
                    test_source = os.path.basename(os.path.abspath(args.load))
            else:
                test_source = args.test_path
                if os.path.isabs(test_source) and BASE_DIR in test_source:
                    test_source = os.path.relpath(test_source, BASE_DIR)

                # Check for HF_MODEL and LLAMA_DIR environment variables and append if set
                # Only capture for models/tt_transformers/demo/simple_text_demo.py
                # This helps identify which specific HuggingFace model or Llama directory was used
                if "models/tt_transformers/demo/simple_text_demo.py" in test_source:
                    hf_model = os.environ.get("HF_MODEL")
                    llama_dir = os.environ.get("LLAMA_DIR")

                    # Append whichever environment variables are available
                    env_tags = []
                    if hf_model:
                        env_tags.append(f"[HF_MODEL:{hf_model}]")
                    if llama_dir:
                        env_tags.append(f"[LLAMA_DIR:{llama_dir}]")

                    if env_tags:
                        test_source = f"{test_source} {' '.join(env_tags)}"

            # Convert and filter operations
            all_operations = []
            filtered_operations = []

            print(f"\n📝 Processing {len(result['trace_files'])} trace files...")
            for json_file in tqdm(result["trace_files"], desc="Converting JSONs", unit="file"):
                operation = convert_json_to_master_format(json_file, test_source, machine_info)
                if operation:
                    all_operations.append(operation)
                    op_name = operation.get("operation", "")
                    if is_valid_operation(op_name, valid_operations, excluded_operations):
                        filtered_operations.append(operation)

            print(f"🎯 Filtered to {len(filtered_operations)} valid operations (from {len(all_operations)} total)")

            # Count execution occurrences within this run
            import hashlib

            execution_counts = {}  # signature -> count

            print("\n🔢 Counting execution frequencies...")
            for operation in tqdm(filtered_operations, desc="Counting executions", unit="op"):
                op_name = operation.get("operation", "unknown")
                op_args = operation.get("arguments", {})
                args_str = json.dumps(op_args, sort_keys=True, default=str)
                signature = f"{op_name}::{hashlib.md5(args_str.encode()).hexdigest()}"
                execution_counts[signature] = execution_counts.get(signature, 0) + 1

            # Add execution count to each operation (max count seen in this run)
            for operation in tqdm(filtered_operations, desc="Adding exec counts", unit="op", leave=False):
                op_name = operation.get("operation", "unknown")
                op_args = operation.get("arguments", {})
                args_str = json.dumps(op_args, sort_keys=True, default=str)
                signature = f"{op_name}::{hashlib.md5(args_str.encode()).hexdigest()}"
                operation["execution_count"] = execution_counts[signature]

            # Deduplicate operations with same config (keep one with execution count)
            print("\n🔍 Deduplicating configurations...")
            unique_operations = {}
            for operation in tqdm(filtered_operations, desc="Deduplicating", unit="op"):
                op_name = operation.get("operation", "unknown")
                op_args = operation.get("arguments", {})
                args_str = json.dumps(op_args, sort_keys=True, default=str)
                signature = f"{op_name}::{hashlib.md5(args_str.encode()).hexdigest()}"
                if signature not in unique_operations:
                    unique_operations[signature] = operation

            filtered_operations_unique = list(unique_operations.values())
            print(f"✅ Reduced to {len(filtered_operations_unique)} unique configurations")

            # Fix shard_spec strings in operations BEFORE adding to master JSON
            # This ensures consistent hashing with existing configs
            print("\n🔧 Normalizing memory configs in operations...")
            fixed_count = [0]  # Mutable reference for counting
            for operation in filtered_operations_unique:
                if "arguments" in operation:
                    fix_memory_config_recursive(operation["arguments"], fixed_count)
            if fixed_count[0] > 0:
                print(f"   Fixed {fixed_count[0]} shard_spec entries in operations")

            # Update master JSON
            os.makedirs(args.output_dir, exist_ok=True)
            master_file = os.path.join(args.output_dir, "ttnn_operations_master.json")
            new_configs_added = update_master_file(master_file, filtered_operations_unique, test_source)

            print(f"📝 Added {new_configs_added} new unique configurations to {master_file}")
            print(f"   Source: {test_source}")

            # Cleanup individual trace files and subdirectory if not storing
            # Never cleanup when processing existing traces (--load)
            if not args.load and not result["keep_traces"]:
                print("\n🧹 Cleaning up individual trace files...")
                cleaned_count = 0
                for trace_file in tqdm(result["trace_files"], desc="Removing files", unit="file"):
                    try:
                        os.remove(trace_file)
                        cleaned_count += 1
                    except OSError:
                        # Best-effort cleanup: ignore failures
                        pass
                if cleaned_count > 0:
                    print(f"✅ Cleaned up {cleaned_count} trace file(s)")

                # Also remove the metadata file and subdirectory
                trace_dir = result.get("trace_dir")
                if trace_dir and os.path.exists(trace_dir):
                    try:
                        # Remove metadata file if it exists
                        metadata_file = os.path.join(trace_dir, "_trace_metadata.json")
                        if os.path.exists(metadata_file):
                            os.remove(metadata_file)

                        # Check if directory is empty now
                        if not os.listdir(trace_dir):
                            os.rmdir(trace_dir)
                            print(f"✅ Cleaned up trace directory: {os.path.basename(trace_dir)}")
                        else:
                            print(f"⚠️ Trace directory not empty, keeping: {os.path.basename(trace_dir)}")
                    except Exception as e:
                        print(f"⚠️ Could not remove trace directory: {e}")

                if cleaned_count > 0:
                    print("💡 Tip: Use --store flag to keep individual trace files")

            print(f"\n✅ Operations extracted successfully!")
            print(f"📄 Master file: {master_file}")

            # Fix memory config shard_spec entries in the master JSON
            fix_memory_config_in_json(master_file)

        # Always return 0 (success) as long as we processed traces
        # Test failures don't affect tracer success
        return 0

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
