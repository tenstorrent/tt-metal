# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

#!/usr/bin/env python3
"""
Generic Operations Tracer

Takes any model test path and extracts ttnn operations by running pytest
with tracing enabled. No model-specific code or device initialization needed.

Usage:
    python generic_ops_tracer.py <test_path> [--output-dir <dir>] [--store]

Examples:
    python generic_ops_tracer.py models/demos/wormhole/distilbert/demo/demo.py::test_demo
    python generic_ops_tracer.py models/demos/wormhole/resnet50/demo/demo.py::test_demo_sample
    python generic_ops_tracer.py /path/to/test.py::test_function --store
    python generic_ops_tracer.py /path/to/test.py::test_function --output-dir ./my_traces --store
"""

import sys
import os
import subprocess
import json
import tempfile
import argparse
from datetime import datetime


def get_base_dir():
    """Get the tt-metal base directory from PYTHONPATH or current working directory"""
    pythonpath = os.environ.get("PYTHONPATH", "")
    if pythonpath:
        # PYTHONPATH might contain multiple paths separated by ':'
        paths = pythonpath.split(":")
        for path in paths:
            # Look for tt-metal directory
            if "tt-metal" in path:
                # Extract the tt-metal base directory
                if path.endswith("tt-metal"):
                    return path
                # Handle cases like /home/ubuntu/tt-metal/python_env/lib/python3.X/site-packages
                parts = path.split("tt-metal")
                if parts:
                    return parts[0] + "tt-metal"
    # Fallback: assume we're running from within tt-metal and find it
    current_dir = os.getcwd()
    if "tt-metal" in current_dir:
        parts = current_dir.split("tt-metal")
        return parts[0] + "tt-metal"
    # Last resort: use current directory
    return current_dir


BASE_DIR = get_base_dir()


def get_machine_info():
    """
    Get machine info (board type, device series, and card count) using tt-smi command.
    Returns a dict with 'board_type', 'device_series', and 'card_count' or None on failure.
    Gracefully handles command not found or other errors.
    """
    try:
        # Run the bash command to extract machine info with card count
        cmd = """
        tt-smi -ls \\
        | sed 's/│/|/g' \\
        | awk -F'|' '
        /Boards that can be reset:/ {in_table=1; next}
        in_table && $0 ~ /^\\|/ {
            gsub(/^[ \\t]+|[ \\t]+$/, "", $3)
            gsub(/^[ \\t]+|[ \\t]+$/, "", $4)
            sub(/[[:space:]]+L$/, "", $4)
            if ($3 != "") machines[$3" "$4]++
        }
        END {
            for (m in machines) print m, machines[m], (machines[m] > 1 ? "cards" : "card")
        }'
        """

        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)  # 10 second timeout

        if result.returncode == 0 and result.stdout.strip():
            # Parse the output: "Wormhole n300 1 card" or "Blackhole tt-galaxy-bh 32 cards"
            lines = result.stdout.strip().split("\n")
            if lines:
                # Take the first line (should be the primary board)
                parts = lines[0].strip().split()
                if len(parts) >= 3:
                    board_type = parts[0]  # e.g., "Wormhole" or "Blackhole"
                    device_series = parts[1]  # e.g., "n300", "n150", "tt-galaxy-bh"
                    card_count = int(parts[2])  # e.g., 1, 2, 32
                    return {"board_type": board_type, "device_series": device_series, "card_count": card_count}

        # If we get here, command didn't produce expected output
        return None

    except subprocess.TimeoutExpired:
        # Command took too long
        return None
    except FileNotFoundError:
        # tt-smi command not found
        return None
    except Exception:
        # Any other error - silently fail
        return None


def create_tracing_plugin(output_dir):
    """
    Create a pytest plugin that captures operations during test execution.

    Args:
        output_dir: Directory to save trace outputs

    Returns:
        str: Path to the created plugin file
    """

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    plugin_content = '''
import pytest
import ttnn
from ttnn.graph_tracer_utils import GraphTracerUtils
import json
import os
from datetime import datetime

BASE_DIR_PLACEHOLDER = "BASE_DIR_VALUE"

class OperationsTracingPlugin:
    def __init__(self):
        self.trace_active = False
        self.output_dir = "OUTPUT_DIR_PLACEHOLDER"
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.test_counter = 0  # Counter to make each trace file unique
        self.valid_operations = self.load_valid_operations()
        self.current_test_source = None  # Will be set in pytest_runtest_setup

        # Operations to exclude from tracing (even if in Allops.txt)
        self.excluded_operations = {
            'ttnn::unary_chain',
            'ttnn::view',
            'ttnn::pearson_correlation_coefficient',
            'ttnn::dump_tensor',
            'ttnn::dram_prefetcher',
            'ttnn::complex_tensor',
            'ttnn::as_tensor',
            'ttnn::allocate_tensor_on_device',
            'ttnn::to_device',
            'ttnn::to_dtype',
            'ttnn::to_layout',
            'ttnn::to_memory_config',
            'ttnn::to_torch',
            'ttnn::prim::binary',
            'ttnn::prim::example',
            'ttnn::prim::example_multiple_return',
            'ttnn::from_device',
            'ttnn::from_torch',
            'ttnn::composite_example',
            'ttnn::composite_example_multiple_return',
            # Memory/Resource Management
            'ttnn::deallocate',
            'ttnn::move',
            'ttnn::reallocate',
            # Utility Operations
            'ttnn::load_tensor'
        }

    def load_valid_operations(self):
        """Load valid operations from Allops.txt"""
        valid_ops = set()
        allops_file = os.path.join(BASE_DIR_PLACEHOLDER, "tests/sweep_framework/Allops.txt")

        try:
            with open(allops_file, 'r') as f:
                for line in f:
                    op_name = line.strip()
                    if op_name:  # Skip empty lines
                        # Convert from dot notation (ttnn.add) to double-colon notation (ttnn::add)
                        op_name_colons = op_name.replace('.', '::')
                        valid_ops.add(op_name_colons)

            print(f"📋 Loaded {len(valid_ops)} valid operations from Allops.txt")
            return valid_ops

        except FileNotFoundError:
            print(f"⚠️ Allops.txt not found at {allops_file}, falling back to prefix filtering")
            return None
        except Exception as e:
            print(f"⚠️ Error loading Allops.txt: {e}, falling back to prefix filtering")
            return None

    def is_valid_operation(self, op_name):
        """Check if operation is in the valid operations list and not excluded"""
        # First check if it's in our exclusion list
        if op_name in self.excluded_operations:
            return False

        if self.valid_operations is None:
            # Fallback to old filtering logic
            return op_name.startswith('ttnn::') or op_name.startswith('ttnn::experimental::')

        return op_name in self.valid_operations

    def fix_unparsed_elements(self, obj):
        """Pre-process to fix UnparsedElements before main cleaning"""
        if isinstance(obj, dict):
            # Check if this is an UnparsedElement
            if "UnparsedElement" in obj:
                unparsed_data = obj["UnparsedElement"]
                element_info = unparsed_data.get("element_info", "")

                # Convert to string if needed
                if not isinstance(element_info, str):
                    element_info = str(element_info)

                # Try to parse with regex fixes
                if element_info and element_info.startswith('{'):
                    try:
                        import re
                        import json as json_module

                        fixed_json_str = element_info
                        # Apply regex fixes
                        fixed_json_str = re.sub(r':\s*"{\s*([^}]+)\s*}"', r': "[\1]"', fixed_json_str)
                        fixed_json_str = re.sub(r'"grid"\s*:\s*\{(\[.*?\](?:\s*,\s*\[.*?\])*)\}', r'"grid":[\1]', fixed_json_str)
                        fixed_json_str = re.sub(r'(\{[^}]+\})\s*-\s*(\{[^}]+\})', r'\1, \2', fixed_json_str)

                        # Parse and return the fixed data
                        parsed_data = json_module.loads(fixed_json_str)
                        # Recursively fix any nested UnparsedElements
                        fixed_result = self.fix_unparsed_elements(parsed_data)
                        # Debug: confirm success
                        if "SHARDED" in element_info:
                            print("✅ Fixed sharded UnparsedElement")
                        return fixed_result
                    except Exception as e:
                        if "SHARDED" in element_info:
                            print(f"❌ Failed to fix sharded UnparsedElement: {str(e)[:80]}")
                        pass

                # If parsing failed, return as-is
                return obj
            else:
                # Recursively fix nested structures
                return {k: self.fix_unparsed_elements(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.fix_unparsed_elements(item) for item in obj]
        else:
            return obj

    def clean_operation_data(self, operation):
        """Clean operation data to ensure it's JSON serializable"""
        if not isinstance(operation, dict):
            return None

        # First, fix all UnparsedElements
        operation = self.fix_unparsed_elements(operation)

        def clean_recursive(obj):
            """Recursively clean objects to ensure JSON serialization"""
            if isinstance(obj, dict):
                cleaned = {}
                for key, value in obj.items():
                    try:
                        # Try to clean the value recursively
                        cleaned_value = clean_recursive(value)
                        # Test if this specific value is JSON serializable
                        json.dumps(cleaned_value)
                        cleaned[key] = cleaned_value
                    except Exception as e:
                        # For problematic values, try to parse as JSON first
                        # Don't truncate yet - we need the full string for parsing
                        value_str = str(value)

                        # Try to parse as JSON if it looks like JSON
                        if value_str.startswith('{') and value_str.endswith('}'):
                            try:
                                # First try direct JSON parsing
                                parsed_data = json.loads(value_str)
                                # Recursively clean the parsed data
                                cleaned[key] = clean_recursive(parsed_data)
                                continue
                            except:
                                # Try to fix common C++ representation issues
                                try:
                                    import re
                                    fixed_json_str = value_str

                                    # Fix C++ style braces in values like "{32, 32}" -> "[32, 32]"
                                    fixed_json_str = re.sub(r':\s*"{\s*([^}]+)\s*}"', r': "[\1]"', fixed_json_str)

                                    # Fix grid format: "grid":{[...], [...]} -> "grid":[[...], [...]]
                                    # This handles CoreRangeSet structures with multiple ranges
                                    fixed_json_str = re.sub(r'"grid"\s*:\s*\{(\[.*?\](?:\s*,\s*\[.*?\])*)\}', r'"grid":[\1]', fixed_json_str)

                                    # Fix grid ranges like [{"x":0,"y":0} - {"x":7,"y":7}] -> [{"x":0,"y":0}, {"x":7,"y":7}]
                                    fixed_json_str = re.sub(r'(\{[^}]+\})\s*-\s*(\{[^}]+\})', r'\1, \2', fixed_json_str)

                                    parsed_data = json.loads(fixed_json_str)
                                    cleaned[key] = clean_recursive(parsed_data)
                                    continue
                                except:
                                    pass

                        # If JSON parsing fails, create UnparsedElement with full string for later parsing
                        cleaned[key] = {
                            "UnparsedElement": {
                                "error": str(e),
                                "element_info": value_str  # Keep full string for sweep test parsing
                            }
                        }
                return cleaned
            elif isinstance(obj, list):
                return [clean_recursive(item) for item in obj]
            elif isinstance(obj, (str, int, float, bool)) or obj is None:
                return obj
            else:
                # For non-JSON serializable objects, convert to string
                return str(obj)

        try:
            cleaned_op = clean_recursive(operation)
            # Final test if the entire cleaned operation is JSON serializable
            json.dumps(cleaned_op)
            return cleaned_op
        except (TypeError, ValueError) as e:
            # If even the cleaned version fails, return a minimal version
            return {
                "operation": str(operation.get('operation', 'unknown')),
                "arguments": [],
                "error": f"Complete serialization failure: {str(e)}"
            }

    def get_operation_signature(self, operation):
        """Generate a unique signature for an operation based on name and key arguments"""
        import hashlib

        if not isinstance(operation, dict) or 'operation' not in operation:
            return None

        # Create signature from operation name and arguments
        op_name = operation['operation']
        args_str = str(operation.get('arguments', []))

        # Create hash of operation name + arguments
        signature = hashlib.md5(f"{op_name}{args_str}".encode()).hexdigest()
        return signature

    def get_arguments_signature(self, arguments):
        """Generate a unique signature for arguments only"""
        import hashlib

        args_str = str(arguments)
        signature = hashlib.md5(args_str.encode()).hexdigest()
        return signature

    def _merge_machine_info(self, existing_config, new_machine_info):
        """
        Merge machine info into an existing configuration.

        Handles smart merging:
        - If same board_type, merge device_series into a list
        - If different board_type, create list of machine_info dicts
        """
        if 'machine_info' not in existing_config:
            # No existing machine info, just add as list
            existing_config['machine_info'] = [new_machine_info]
            return

        existing_machine_info = existing_config['machine_info']

        # Handle legacy single-dict format - convert to list
        if isinstance(existing_machine_info, dict):
            existing_machine_info = [existing_machine_info]
            existing_config['machine_info'] = existing_machine_info

        # Now existing_machine_info is a list
        new_board_type = new_machine_info.get('board_type')
        new_device_series = new_machine_info.get('device_series')

        # Find if we have an entry with matching board_type
        matching_board_entry = None
        for entry in existing_machine_info:
            if entry.get('board_type') == new_board_type:
                matching_board_entry = entry
                break

        if matching_board_entry:
            # Same board type - merge device_series
            existing_series = matching_board_entry.get('device_series')

            # Convert to list if needed
            if not isinstance(existing_series, list):
                existing_series = [existing_series]
                matching_board_entry['device_series'] = existing_series

            # Add new device_series if not already present
            if new_device_series not in existing_series:
                existing_series.append(new_device_series)
                # Keep sorted for consistency
                existing_series.sort()
        else:
            # Different board type - add as new entry
            existing_machine_info.append(new_machine_info)

    def update_master_file(self, master_file_path, new_operations, test_name):
        """Update master file with unique operation configurations grouped by operation name"""

        # Load existing master data with grouped structure
        master_data = {"operations": {}, "metadata": {"models": [], "total_operations": 0, "unique_operations": 0}}

        # Load existing master file with retry logic for file locking
        max_retries = 5
        retry_delay = 0.1  # 100ms

        if os.path.exists(master_file_path):
            for attempt in range(max_retries):
                try:
                    with open(master_file_path, 'r') as f:
                        master_data = json.load(f)
                    break  # Success, exit retry loop
                except (IOError, json.JSONDecodeError) as e:
                    if attempt < max_retries - 1:
                        # Wait and retry (another test might be writing)
                        import time
                        time.sleep(retry_delay)
                        continue
                    else:
                        # Last attempt failed
                        print(f"⚠️ Could not load existing master file after {max_retries} attempts: {str(e)}. Starting fresh.")
                        master_data = {"operations": {}, "metadata": {"models": [], "total_operations": 0, "unique_operations": 0}}
                        break

            # Handle legacy format conversion
            try:
                if 'content' in master_data and 'operations' not in master_data:
                    print("🔄 Converting legacy format to grouped format...")
                    operations_dict = {}
                    for op in master_data.get('content', []):
                        op_name = op.get('operation', 'unknown')
                        if op_name not in operations_dict:
                            operations_dict[op_name] = {"configurations": []}
                        # Convert old format (list) to new format (dict with source)
                        op_args = op.get('arguments', [])
                        if isinstance(op_args, list) and len(op_args) > 0:
                            # Check if already in new format
                            if isinstance(op_args[0], dict) and 'arguments' in op_args[0]:
                                operations_dict[op_name]["configurations"].extend(op_args)
                            else:
                                # Old format - convert to new format with unknown source
                                operations_dict[op_name]["configurations"].append({
                                    "arguments": op_args,
                                    "source": "unknown"
                                })
                    master_data = {
                        "operations": operations_dict,
                        "metadata": master_data.get('metadata', {"models": [], "total_operations": 0, "unique_operations": 0})
                    }

                # Convert old format configurations (list) to new format (dict with source)
                # This handles existing master files that have list format
                for op_name, op_data in master_data.get('operations', {}).items():
                    configs = op_data.get('configurations', [])
                    if configs and isinstance(configs[0], list):
                        print(f"🔄 Converting {op_name} configurations to new format with source tags...")
                        converted_configs = []
                        for config in configs:
                            if isinstance(config, list):
                                converted_configs.append({
                                    "arguments": config,
                                    "source": "unknown"  # Legacy configs don't have source info
                                })
                            elif isinstance(config, dict) and 'arguments' in config:
                                # Already in new format
                                converted_configs.append(config)
                            else:
                                # Fallback: wrap in new format
                                converted_configs.append({
                                    "arguments": config,
                                    "source": "unknown"
                                })
                        master_data['operations'][op_name]['configurations'] = converted_configs

            except (json.JSONDecodeError, IOError) as e:
                print(f"⚠️ Error processing master file format: {str(e)}. Starting fresh.")
                master_data = {"operations": {}, "metadata": {"models": [], "total_operations": 0, "unique_operations": 0}}
        else:
            # File doesn't exist, start fresh
            master_data = {"operations": {}, "metadata": {"models": [], "total_operations": 0, "unique_operations": 0}}

        # Group new operations by operation name and collect unique configurations
        new_configs_added = 0

        for operation in new_operations:
            # Clean the operation data first
            clean_op = self.clean_operation_data(operation)
            if clean_op:
                op_name = clean_op.get('operation', 'unknown')
                op_args = clean_op.get('arguments', [])

                # Initialize operation entry if not exists
                if op_name not in master_data['operations']:
                    master_data['operations'][op_name] = {"configurations": []}

                # Check if this argument configuration already exists
                arg_signature = self.get_arguments_signature(op_args)

                # Try to get machine info for the new configuration
                new_machine_info = get_machine_info()

                # Find matching configuration to merge machine info
                matching_config = None
                for existing_config in master_data['operations'][op_name]["configurations"]:
                    # Handle both old format (list) and new format (dict with source)
                    if isinstance(existing_config, list):
                        existing_args = existing_config
                    elif isinstance(existing_config, dict) and 'arguments' in existing_config:
                        existing_args = existing_config['arguments']
                    else:
                        existing_args = existing_config

                    existing_sig = self.get_arguments_signature(existing_args)
                    if existing_sig == arg_signature:
                        matching_config = existing_config
                        break

                if matching_config is None:
                    # New configuration - add it
                    config_entry = {
                        "arguments": op_args,
                        "source": test_name
                    }

                    if new_machine_info:
                        config_entry["machine_info"] = [new_machine_info]

                    master_data['operations'][op_name]["configurations"].append(config_entry)
                    new_configs_added += 1
                else:
                    # Configuration exists - merge machine info if needed
                    if new_machine_info and isinstance(matching_config, dict):
                        self._merge_machine_info(matching_config, new_machine_info)

        # Update metadata
        if test_name not in master_data['metadata']['models']:
            master_data['metadata']['models'].append(test_name)

        # Calculate statistics from grouped operations
        total_configurations = sum(len(op_data["configurations"]) for op_data in master_data['operations'].values())
        unique_operations = len(master_data['operations'])

        master_data['metadata']['unique_operations'] = unique_operations
        master_data['metadata']['total_configurations'] = total_configurations
        master_data['metadata']['total_operations'] = master_data['metadata'].get('total_operations', 0) + len(new_operations)
        master_data['metadata']['last_updated'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Count configurations per operation type
        op_types = {}
        for op_name, op_data in master_data['operations'].items():
            op_types[op_name] = len(op_data["configurations"])

        master_data['metadata']['operation_types'] = op_types

        # Save updated master file with file locking to prevent race conditions
        # Use atomic write (write to temp file, then rename) to prevent corruption
        import tempfile
        import shutil

        try:
            # Write to temporary file first (atomic operation)
            temp_file = master_file_path + '.tmp'
            with open(temp_file, 'w') as f:
                json.dump(master_data, f, indent=2, default=str)

            # Atomic rename (replaces existing file atomically)
            shutil.move(temp_file, master_file_path)
        except (IOError, TypeError, OSError) as e:
            print(f"❌ Error saving master file: {e}")
            # Clean up temp file if it exists
            temp_file = master_file_path + '.tmp'
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except:
                    pass
            # Try to save without problematic data
            try:
                # Create a simplified version if full serialization fails
                simplified_data = {{
                    "operations": {},
                    "metadata": master_data.get('metadata', {})
                }}
                temp_file = master_file_path + '.tmp'
                with open(temp_file, 'w') as f:
                    json.dump(simplified_data, f, indent=2, default=str)
                shutil.move(temp_file, master_file_path)
                print("💾 Saved simplified master file without problematic operations")
            except Exception as e2:
                print(f"❌ Failed to save even simplified master file: {e2}")
                # Clean up temp file
                temp_file = master_file_path + '.tmp'
                if os.path.exists(temp_file):
                    try:
                        os.remove(temp_file)
                    except:
                        pass

        return new_configs_added

    def pytest_runtest_setup(self, item):
        """Start tracing before each test"""
        # Extract test name/path for source tagging
        # Use nodeid which includes the full path (e.g., "models/demo.py::test_function")
        nodeid = getattr(item, 'nodeid', item.name)
        # Clean up the source name for readability
        if '::' in nodeid:
            # Extract file path and test name
            parts = nodeid.split('::')
            source_path = parts[0] if len(parts) > 0 else item.name
        else:
            source_path = nodeid

        # Normalize absolute paths to relative paths (if within BASE_DIR)
        if os.path.isabs(source_path) and BASE_DIR_PLACEHOLDER in source_path:
            try:
                source_path = os.path.relpath(source_path, BASE_DIR_PLACEHOLDER)
            except ValueError:
                # If relpath fails, keep original
                pass

        # Check for HF_MODEL and LLAMA_DIR environment variables and append if set
        # Only capture for models/tt_transformers/demo/simple_text_demo.py
        # This helps identify which specific HuggingFace model or Llama directory was used
        hf_model = None
        llama_dir = None
        if 'models/tt_transformers/demo/simple_text_demo.py' in source_path:
            hf_model = os.environ.get('HF_MODEL', None)
            llama_dir = os.environ.get('LLAMA_DIR', None)

            # Append whichever environment variables are available
            env_tags = []
            if hf_model:
                env_tags.append(f"[HF_MODEL:{hf_model}]")
            if llama_dir:
                env_tags.append(f"[LLAMA_DIR:{llama_dir}]")

            if env_tags:
                source_path = f"{source_path} {' '.join(env_tags)}"

        self.current_test_source = source_path

        print(f"\\n🔍 Starting operations trace for: {item.name}")
        print(f"📝 Source tag: {self.current_test_source}")
        print(f"🔢 Test number: {self.test_counter + 1}")  # Show which test number this is
        if hf_model:
            print(f"🤗 HuggingFace Model: {hf_model}")
        if llama_dir:
            print(f"🦙 Llama Directory: {llama_dir}")
        os.makedirs(self.output_dir, exist_ok=True)

        # Begin graph capture
        ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
        self.trace_active = True

    def pytest_runtest_teardown(self, item, nextitem):
        """Capture operations after each test"""
        if not self.trace_active:
            return

        try:
            print("📊 Capturing operations...")
            captured_graph = ttnn.graph.end_graph_capture()
            trace_data = GraphTracerUtils.serialize_graph(captured_graph)

            # Filter the trace data to only include TTNN operations
            if isinstance(trace_data, dict) and 'content' in trace_data:
                original_operations = trace_data['content']
                filtered_operations = []

                for op in original_operations:
                    if isinstance(op, dict) and 'operation' in op:
                        op_name = op['operation']
                        # Include only valid operations from Allops.txt
                        if self.is_valid_operation(op_name):
                            filtered_operations.append(op)

                # Update trace_data with filtered operations
                trace_data['content'] = filtered_operations
                print(f"🎯 Filtered to {len(filtered_operations)} TTNN operations (from {len(original_operations)} total)")

                # Update master JSON file with unique configurations
                master_file = os.path.join(self.output_dir, 'ttnn_operations_master.json')
                # Use the source tag we set in pytest_runtest_setup
                # If current_test_source is None (test skipped before setup), use item.name as fallback
                test_source = getattr(self, 'current_test_source', None)
                if test_source is None:
                    # Fallback: extract source from item.nodeid or use item.name
                    nodeid = getattr(item, 'nodeid', item.name)
                    if '::' in nodeid:
                        parts = nodeid.split('::')
                        test_source = parts[0] if len(parts) > 0 else item.name
                    else:
                        test_source = nodeid

                    # Normalize absolute paths to relative paths (if within BASE_DIR)
                    if os.path.isabs(test_source) and BASE_DIR_PLACEHOLDER in test_source:
                        try:
                            test_source = os.path.relpath(test_source, BASE_DIR_PLACEHOLDER)
                        except ValueError:
                            # If relpath fails, keep original
                            pass

                    # Check for HF_MODEL and LLAMA_DIR environment variables and append if set (fallback case)
                    # Only capture for models/tt_transformers/demo/simple_text_demo.py
                    hf_model = None
                    llama_dir = None
                    if 'models/tt_transformers/demo/simple_text_demo.py' in test_source:
                        hf_model = os.environ.get('HF_MODEL', None)
                        llama_dir = os.environ.get('LLAMA_DIR', None)

                        # Append whichever environment variables are available
                        env_tags = []
                        if hf_model:
                            env_tags.append(f"[HF_MODEL:{hf_model}]")
                        if llama_dir:
                            env_tags.append(f"[LLAMA_DIR:{llama_dir}]")

                        if env_tags:
                            test_source = f"{test_source} {' '.join(env_tags)}"

                new_configs_added = self.update_master_file(master_file, filtered_operations, test_source)
                print(f"📝 Added {new_configs_added} new unique configurations to master file (source: {test_source})")
                print(f"   📊 Captured {len(filtered_operations)} operations from this test")

            # Generate trace filename - sanitize the test name
            test_name = item.name.replace("[", "_").replace("]", "_").replace(":", "_").replace("/", "_").replace("-", "_")
            # Limit filename length
            if len(test_name) > 100:
                test_name = test_name[:100]
            # Increment counter for each test to ensure unique filenames
            self.test_counter += 1
            trace_file = os.path.join(self.output_dir, f"{test_name}_filtered_ops_{self.timestamp}_{self.test_counter:03d}.json")
            print(f"📁 Creating trace file #{self.test_counter}: {os.path.basename(trace_file)}")

            # Save trace data (clean it first to ensure JSON serialization)
            try:
                # Clean the trace data the same way we do for master file
                cleaned_trace_data = trace_data.copy()
                if 'content' in cleaned_trace_data:
                    cleaned_operations = []
                    for op in cleaned_trace_data['content']:
                        cleaned_op = self.clean_operation_data(op)
                        if cleaned_op:
                            cleaned_operations.append(cleaned_op)
                    cleaned_trace_data['content'] = cleaned_operations

                with open(trace_file, 'w') as f:
                    json.dump(cleaned_trace_data, f, indent=2, default=str)
                print(f"💾 Operations saved to: {trace_file}")
                file_format = "JSON"

            except (TypeError, ValueError) as e:
                # Fallback to string representation
                trace_file_txt = trace_file.replace('.json', '_repr.txt')
                with open(trace_file_txt, 'w') as f:
                    f.write(str(trace_data))
                trace_file = trace_file_txt
                print(f"💾 Operations saved to: {trace_file} (as text)")
                file_format = "Text"

            # Analyze operations
            if isinstance(trace_data, dict) and 'content' in trace_data:
                operations = trace_data['content']
                print(f"📈 Captured {len(operations)} operations")

                # Count operation types and filter relevant operations
                op_counts = {}
                filtered_op_counts = {}

                for op in operations:
                    if isinstance(op, dict) and 'operation' in op:
                        op_name = op['operation']
                        op_counts[op_name] = op_counts.get(op_name, 0) + 1

                        # Include only valid operations from Allops.txt
                        if self.is_valid_operation(op_name):
                            filtered_op_counts[op_name] = filtered_op_counts.get(op_name, 0) + 1

                print("\\n📋 ALL OPERATIONS:")
                print("=" * 60)
                for op_name, count in sorted(op_counts.items(), key=lambda x: x[1], reverse=True):
                    print(f"{op_name}: {count}x")
                print("=" * 60)

                print("\\n🎯 VALID OPERATIONS (from Allops.txt):")
                print("=" * 60)
                if filtered_op_counts:
                    for op_name, count in sorted(filtered_op_counts.items(), key=lambda x: x[1], reverse=True):
                        print(f"{op_name}: {count}x")
                else:
                    print("No valid operations found")
                print("=" * 60)

                # File info
                file_size = os.path.getsize(trace_file)
                print(f"📁 File: {trace_file}")
                print(f"📊 Size: {file_size:,} bytes ({file_format})")
                print(f"🔧 Total Operations: {len(operations)}")
                print(f"📋 All Op Types: {len(op_counts)}")
                print(f"🎯 Valid Op Types: {len(filtered_op_counts)}") # From Allops.txt

            else:
                print("⚠️ No operations captured or unexpected format")

        except Exception as e:
            print(f"❌ Error capturing operations: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.trace_active = False

def pytest_configure(config):
    """Register the tracing plugin"""
    config.pluginmanager.register(OperationsTracingPlugin(), "operations_tracer")
'''

    # Write plugin to tt-metal directory
    plugin_file = os.path.join(BASE_DIR, "conftest_tracer.py")
    os.makedirs(output_dir, exist_ok=True)

    with open(plugin_file, "w") as f:
        content = plugin_content.replace("OUTPUT_DIR_PLACEHOLDER", output_dir)
        content = content.replace("BASE_DIR_VALUE", BASE_DIR)
        f.write(content)

    return plugin_file


def run_test_with_tracing(test_path, output_dir, keep_traces=False):
    """
    Run pytest with operations tracing enabled.

    Args:
        test_path: Path to test (e.g., /path/to/test.py::test_function)
        output_dir: Directory to save trace outputs
        keep_traces: If True, keep individual trace files after adding to master JSON

    Returns:
        dict: Results of the test run
    """

    print(f"🚀 Running test with operations tracing...")
    plugin_file = create_tracing_plugin(output_dir)

    # Run pytest from tt-metal directory with our plugin
    # Use the same python executable that's running this script
    python_cmd = os.path.join(BASE_DIR, "python_env/bin/python")
    result = subprocess.run(
        [python_cmd, "-m", "pytest", test_path, "-v", "-s", "--tb=short", "-p", "conftest_tracer"],
        cwd=BASE_DIR,
        capture_output=True,
        text=True,
    )

    # Check for created trace files - get all files from current run
    # Use timestamp in filename to group files from same run (more reliable than mtime)
    trace_files = []
    if os.path.exists(output_dir):
        import time
        import re

        current_time = time.time()
        files_with_timestamp = []

        # Extract timestamp from filename pattern: name_filtered_ops_TIMESTAMP_COUNTER.json
        timestamp_pattern = r"_filtered_ops_(\d{8}_\d{6})_(\d{3})\.(json|txt)$"

        for f in os.listdir(output_dir):
            if ("_ops_" in f and (f.endswith(".json") or f.endswith(".txt"))) and f != "conftest.py":
                file_path = os.path.join(output_dir, f)
                file_time = os.path.getmtime(file_path)

                # Try to extract timestamp from filename
                match = re.search(timestamp_pattern, f)
                if match:
                    file_timestamp_str = match.group(1)  # e.g., "20251120_070439"
                    file_counter = match.group(2)  # e.g., "001"
                    files_with_timestamp.append((file_timestamp_str, file_counter, file_time, file_path))
                else:
                    # Fallback: use modification time for files without timestamp pattern
                    # Only include if created in last 60 seconds
                    if current_time - file_time < 60:
                        files_with_timestamp.append((None, None, file_time, file_path))

        if files_with_timestamp:
            # Group files by timestamp (files with same timestamp are from same run)
            timestamp_groups = {}
            for ts_str, counter, mtime, file_path in files_with_timestamp:
                if ts_str:
                    if ts_str not in timestamp_groups:
                        timestamp_groups[ts_str] = []
                    timestamp_groups[ts_str].append((int(counter), mtime, file_path))

            # Get the most recent timestamp group (current run)
            if timestamp_groups:
                # Sort by modification time of first file in each group
                sorted_groups = sorted(
                    timestamp_groups.items(), key=lambda x: max(f[1] for f in x[1]), reverse=True  # Max mtime in group
                )

                # Take files from the most recent timestamp group
                most_recent_timestamp, files_in_group = sorted_groups[0]
                # Sort by counter to maintain test order
                files_in_group.sort(key=lambda x: x[0])  # Sort by counter
                trace_files = [f[2] for f in files_in_group]  # Extract file paths
                print(f"🔍 Found {len(trace_files)} trace file(s) with timestamp {most_recent_timestamp}")
            else:
                # Fallback: use modification time for files without timestamp
                files_with_time = [(mtime, file_path) for _, _, mtime, file_path in files_with_timestamp]
                files_with_time.sort(reverse=True)
                if files_with_time:
                    most_recent_time = files_with_time[0][0]
                    trace_files = [
                        file_path for file_time, file_path in files_with_time if most_recent_time - file_time < 60
                    ]

    return {
        "success": result.returncode == 0,
        "exit_code": result.returncode,
        "trace_files": trace_files,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "plugin_file": plugin_file,
        "keep_traces": keep_traces,
    }


def main():
    parser = argparse.ArgumentParser(
        description="TTNN Operations Tracer - Extract operation configurations from model tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python model_tracer/generic_ops_tracer.py models/demos/wormhole/distilbert/demo/demo.py::test_demo
    python model_tracer/generic_ops_tracer.py /path/to/test.py::test_function --store
    python model_tracer/generic_ops_tracer.py /path/to/test.py::test_function --output-dir ./my_traces --store
        """,
    )
    parser.add_argument("test_path", help="Path to test file (e.g., /path/to/test.py::test_function)")
    parser.add_argument(
        "--output-dir",
        "-o",
        default="./model_tracer/traced_operations",
        help="Directory to save trace outputs (default: ./model_tracer/traced_operations)",
    )
    parser.add_argument(
        "--store",
        "--keep-traces",
        action="store_true",
        help="Keep individual trace files after adding to master JSON (default: delete them)",
    )

    args = parser.parse_args()

    print("🚀 TTNN Operations Tracer")
    print("=" * 50)
    print(f"📁 {os.path.basename(args.test_path)}")
    if args.store:
        print(f"💾 Keeping individual trace files")
    print("=" * 50)

    try:
        result = run_test_with_tracing(args.test_path, args.output_dir, args.store)

        print("\\n" + "=" * 50)
        print("📋 RESULTS")
        print("=" * 50)

        print(f"Test Result: {'✅ PASSED' if result['success'] else '❌ FAILED'}")

        # Show all trace files if multiple tests ran
        if result["trace_files"]:
            total_operations = 0
            all_op_counts = {}

            print(f"📊 Found {len(result['trace_files'])} trace file(s) from {len(result['trace_files'])} test(s):")

            # Aggregate operations from all trace files
            for idx, trace_file in enumerate(result["trace_files"], 1):
                file_size = os.path.getsize(trace_file)
                print(f"\n   Test {idx}: {os.path.basename(trace_file)} ({file_size:,} bytes)")

                # Try to show operation count and types
                if trace_file.endswith(".json"):
                    try:
                        with open(trace_file, "r") as f:
                            data = json.load(f)
                        if isinstance(data, dict) and "content" in data:
                            operations = data["content"]
                            total_operations += len(operations)

                            op_counts = {}
                            for op in operations:
                                if isinstance(op, dict) and "operation" in op:
                                    op_name = op["operation"]
                                    op_counts[op_name] = op_counts.get(op_name, 0) + 1
                                    # Aggregate across all tests
                                    all_op_counts[op_name] = all_op_counts.get(op_name, 0) + op_counts[op_name]

                            print(f"      📊 Captured: {len(operations)} operations, {len(op_counts)} unique types")

                    except Exception as e:
                        print(f"      ⚠️ Could not read trace file: {e}")

            # Show aggregated summary if multiple tests
            if len(result["trace_files"]) > 1:
                print(f"\n📊 Total across all tests: {total_operations} operations, {len(all_op_counts)} unique types")

            # Show unique configurations from the last test (or aggregate if multiple)
            if all_op_counts:
                print("\n🔧 Unique Configurations:")
                sorted_ops = sorted(all_op_counts.items(), key=lambda x: x[1], reverse=True)
                for op_name, exec_count in sorted_ops:
                    print(f"   • {op_name}: {exec_count}x executed")

        if result["success"] and result["trace_files"]:
            print("\\n✅ Operations extracted successfully!")

            # Cleanup individual trace files if --store flag not set
            if not result["keep_traces"]:
                print("\\n🧹 Cleaning up individual trace files...")
                cleaned_count = 0
                for trace_file in result["trace_files"]:
                    try:
                        # Only delete trace files (not master JSON)
                        if "ttnn_operations_master.json" not in trace_file:
                            os.remove(trace_file)
                            cleaned_count += 1
                            print(f"   Deleted: {os.path.basename(trace_file)}")
                    except Exception as e:
                        print(f"   ⚠️ Could not delete {os.path.basename(trace_file)}: {e}")

                if cleaned_count > 0:
                    print(f"✅ Cleaned up {cleaned_count} trace file(s)")
                    print("💡 Tip: Use --store flag to keep individual trace files")
        elif result["success"] and not result["trace_files"]:
            print("\\n⚠️ Test passed but no operations captured")
        else:
            print("\\n❌ Test failed or operations not captured")

        return 0 if result["success"] else 1

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return 1

    finally:
        # Clean up plugin file
        try:
            if "result" in locals() and "plugin_file" in result:
                plugin_file = result["plugin_file"]
            else:
                plugin_file = os.path.join(BASE_DIR, "conftest_tracer.py")

            if os.path.exists(plugin_file):
                os.unlink(plugin_file)
        except:
            pass


if __name__ == "__main__":
    sys.exit(main())
