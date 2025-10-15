#!/usr/bin/env python3
"""
Generic Operations Tracer

Takes any model test path and extracts ttnn operations by running pytest
with tracing enabled. No model-specific code or device initialization needed.

Usage:
    python generic_ops_tracer.py <test_path> [output_dir]

Examples:
    python generic_ops_tracer.py /home/ubuntu/tt-metal/models/demos/wormhole/distilbert/demo/demo.py::test_demo
    python generic_ops_tracer.py /home/ubuntu/tt-metal/models/demos/wormhole/resnet50/demo/demo.py::test_demo_sample
"""

import sys
import os
import subprocess
import json
import tempfile
from datetime import datetime

# Set up environment
sys.path.insert(0, "/home/ubuntu/tt-metal")
os.chdir("/home/ubuntu/tt-metal")

os.environ["ARCH_NAME"] = "wormhole_b0"
os.environ["TT_METAL_HOME"] = "/home/ubuntu/tt-metal"
os.environ["PYTHONPATH"] = "/home/ubuntu/tt-metal"
os.environ["TT_METAL_ENV"] = "dev"
os.environ["TT_METAL_GRAPH_TRACE"] = "1"


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

class OperationsTracingPlugin:
    def __init__(self):
        self.trace_active = False
        self.output_dir = "OUTPUT_DIR_PLACEHOLDER"
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.valid_operations = self.load_valid_operations()

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
        allops_file = "/home/ubuntu/tt-metal/tests/sweep_framework/Allops.txt"

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

    def update_master_file(self, master_file_path, new_operations, test_name):
        """Update master file with unique operation configurations grouped by operation name"""

        # Load existing master data with grouped structure
        master_data = {"operations": {}, "metadata": {"models": [], "total_operations": 0, "unique_operations": 0}}

        if os.path.exists(master_file_path):
            try:
                with open(master_file_path, 'r') as f:
                    master_data = json.load(f)

                # Handle legacy format conversion
                if 'content' in master_data and 'operations' not in master_data:
                    print("🔄 Converting legacy format to grouped format...")
                    operations_dict = {}
                    for op in master_data.get('content', []):
                        op_name = op.get('operation', 'unknown')
                        if op_name not in operations_dict:
                            operations_dict[op_name] = {"configurations": []}
                        operations_dict[op_name]["configurations"].append(op.get('arguments', []))
                    master_data = {
                        "operations": operations_dict,
                        "metadata": master_data.get('metadata', {"models": [], "total_operations": 0, "unique_operations": 0})
                    }

            except (json.JSONDecodeError, IOError) as e:
                print(f"⚠️ Could not load existing master file: {str(e)}. Starting fresh.")
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
                existing_signatures = set()

                for existing_config in master_data['operations'][op_name]["configurations"]:
                    existing_sig = self.get_arguments_signature(existing_config)
                    existing_signatures.add(existing_sig)

                # Add configuration if it's unique
                if arg_signature not in existing_signatures:
                    master_data['operations'][op_name]["configurations"].append(op_args)
                    new_configs_added += 1

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

        # Save updated master file
        try:
            with open(master_file_path, 'w') as f:
                json.dump(master_data, f, indent=2, default=str)
        except (IOError, TypeError) as e:
            print(f"❌ Error saving master file: {e}")
            # Try to save without problematic data
            try:
                # Create a simplified version if full serialization fails
                simplified_data = {{
                    "operations": {},
                    "metadata": master_data.get('metadata', {})
                }}
                with open(master_file_path, 'w') as f:
                    json.dump(simplified_data, f, indent=2, default=str)
                print("💾 Saved simplified master file without problematic operations")
            except Exception as e2:
                print(f"❌ Failed to save even simplified master file: {e2}")

        return new_configs_added

    def pytest_runtest_setup(self, item):
        """Start tracing before each test"""
        print(f"\\n🔍 Starting operations trace for: {item.name}")
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
                new_configs_added = self.update_master_file(master_file, filtered_operations, item.name)
                print(f"📝 Added {new_configs_added} new unique configurations to master file")

            # Generate trace filename - sanitize the test name
            test_name = item.name.replace("[", "_").replace("]", "_").replace(":", "_").replace("/", "_").replace("-", "_")
            # Limit filename length
            if len(test_name) > 100:
                test_name = test_name[:100]
            trace_file = os.path.join(self.output_dir, f"{test_name}_filtered_ops_{self.timestamp}.json")

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
    plugin_file = "/home/ubuntu/tt-metal/conftest_tracer.py"
    os.makedirs(output_dir, exist_ok=True)

    with open(plugin_file, "w") as f:
        f.write(plugin_content.replace("OUTPUT_DIR_PLACEHOLDER", output_dir))

    return plugin_file


def run_test_with_tracing(test_path, output_dir):
    """
    Run pytest with operations tracing enabled.

    Args:
        test_path: Path to test (e.g., /path/to/test.py::test_function)
        output_dir: Directory to save trace outputs

    Returns:
        dict: Results of the test run
    """

    print(f"🚀 Running test with operations tracing...")
    plugin_file = create_tracing_plugin(output_dir)

    # Run pytest from tt-metal directory with our plugin
    # Use the same python executable that's running this script
    python_cmd = "/home/ubuntu/tt-metal/python_env/bin/python"
    result = subprocess.run(
        [python_cmd, "-m", "pytest", test_path, "-v", "-s", "--tb=short", "-p", "conftest_tracer"],
        cwd="/home/ubuntu/tt-metal",
        capture_output=True,
        text=True,
    )

    # Check for created trace files - get the most recent one
    trace_files = []
    if os.path.exists(output_dir):
        files_with_time = []
        for f in os.listdir(output_dir):
            if ("_ops_" in f and (f.endswith(".json") or f.endswith(".txt"))) and f != "conftest.py":
                file_path = os.path.join(output_dir, f)
                file_time = os.path.getmtime(file_path)
                files_with_time.append((file_time, file_path))

        # Sort by modification time (most recent first) and take only recent files
        files_with_time.sort(reverse=True)
        if files_with_time:
            # Only include files from the last 30 seconds (current run)
            import time

            current_time = time.time()
            for file_time, file_path in files_with_time:
                if current_time - file_time < 30:  # Files created in last 30 seconds
                    trace_files.append(file_path)

    return {
        "success": result.returncode == 0,
        "exit_code": result.returncode,
        "trace_files": trace_files,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "plugin_file": plugin_file,
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python generic_ops_tracer.py <test_path> [output_dir]")
        print("\\nExamples:")
        print(
            "  python generic_ops_tracer.py /home/ubuntu/tt-metal/models/demos/wormhole/distilbert/demo/demo.py::test_demo"
        )
        print(
            "  python generic_ops_tracer.py /home/ubuntu/tt-metal/models/demos/wormhole/resnet50/demo/demo.py::test_demo_sample"
        )
        return 1

    test_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "./traced_operations"

    print("🚀 TTNN Operations Tracer")
    print("=" * 50)
    print(f"📁 {os.path.basename(test_path)}")
    print("=" * 50)

    try:
        result = run_test_with_tracing(test_path, output_dir)

        print("\\n" + "=" * 50)
        print("📋 RESULTS")
        print("=" * 50)

        print(f"Test Result: {'✅ PASSED' if result['success'] else '❌ FAILED'}")

        # Show current trace file details only
        if result["trace_files"]:
            trace_file = result["trace_files"][0]  # Show only the latest/current file
            file_size = os.path.getsize(trace_file)

            # Try to show operation count and types
            if trace_file.endswith(".json"):
                try:
                    with open(trace_file, "r") as f:
                        data = json.load(f)
                    if isinstance(data, dict) and "content" in data:
                        operations = data["content"]
                        op_counts = {}
                        for op in operations:
                            if isinstance(op, dict) and "operation" in op:
                                op_name = op["operation"]
                                op_counts[op_name] = op_counts.get(op_name, 0) + 1

                        print(f"📊 Captured: {len(operations)} operations, {len(op_counts)} unique types")
                        print(f"💾 File: {os.path.basename(trace_file)} ({file_size:,} bytes)")

                        # Count unique configurations in current test only
                        if op_counts:
                            print("🔧 Unique Configurations (current test):")
                            sorted_ops = sorted(op_counts.items(), key=lambda x: x[1], reverse=True)
                            for op_name, exec_count in sorted_ops:
                                # Count unique argument signatures in current test
                                unique_args = set()
                                for op in operations:
                                    if isinstance(op, dict) and op.get("operation") == op_name:
                                        args_signature = str(op.get("arguments", []))
                                        unique_args.add(args_signature)

                                unique_count = len(unique_args)
                                print(f"   • {op_name}: {unique_count} unique configs ({exec_count}x executed)")

                except:
                    print(f"💾 File: {os.path.basename(trace_file)} ({file_size:,} bytes)")

        if result["success"] and result["trace_files"]:
            print("\\n✅ Operations extracted successfully!")
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
                plugin_file = "/home/ubuntu/tt-metal/conftest_tracer.py"

            if os.path.exists(plugin_file):
                os.unlink(plugin_file)
        except:
            pass


if __name__ == "__main__":
    sys.exit(main())
