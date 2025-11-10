# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

#!/usr/bin/env python3
"""
Master Configuration Loader for Sweep Tests

This module provides utilities to load real-world operation configurations
from the master JSON file and convert them into sweep test parameters.
"""

import json
import re
import os
import sys
import ttnn
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
from .operation_parameter_extractors import OperationParameterExtractors

# Get the base directory dynamically - import from model_tracer
try:
    # Try direct import (if model_tracer is in PYTHONPATH or same parent)
    sys.path.insert(0, str(Path(__file__).parent.parent / "model_tracer"))
    from generic_ops_tracer import get_base_dir
except ImportError:
    # Fallback: define inline if generic_ops_tracer not found
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


@dataclass
class TensorConfig:
    """Represents a tensor configuration extracted from master JSON"""

    shape: List[int]
    dtype: str
    layout: str
    memory_config: Dict


class MasterConfigLoader:
    """Loads and converts master JSON configurations to sweep test parameters"""

    def __init__(self, master_file_path: str = None):
        if master_file_path is None:
            master_file_path = os.path.join(BASE_DIR, "model_tracer/traced_operations/ttnn_operations_master.json")
        self.master_file_path = master_file_path
        self.master_data = None
        self.traced_configs_cache = {}  # Cache configs by operation name

    def load_master_data(self):
        """Load the master JSON file"""
        if self.master_data is None:
            try:
                with open(self.master_file_path, "r") as f:
                    self.master_data = json.load(f)
                print(f"✅ Loaded master data with {len(self.master_data.get('operations', {}))} operations")
            except FileNotFoundError:
                print(f"❌ Master file not found: {self.master_file_path}")
                self.master_data = {"operations": {}}
            except json.JSONDecodeError as e:
                print(f"❌ Error parsing master JSON: {e}")
                self.master_data = {"operations": {}}

    def get_operation_configs(self, operation_name: str) -> List[List[Dict]]:
        """Get all configurations for a specific operation"""
        self.load_master_data()

        # Try exact match first
        if operation_name in self.master_data.get("operations", {}):
            return self.master_data["operations"][operation_name].get("configurations", [])

        # Try with ttnn:: prefix
        ttnn_op_name = f"ttnn::{operation_name}"
        if ttnn_op_name in self.master_data.get("operations", {}):
            return self.master_data["operations"][ttnn_op_name].get("configurations", [])

        # Try without prefix if it starts with ttnn::
        if operation_name.startswith("ttnn::"):
            base_name = operation_name[6:]  # Remove "ttnn::"
            if base_name in self.master_data.get("operations", {}):
                return self.master_data["operations"][base_name].get("configurations", [])

        print(f"⚠️ No configurations found for operation: {operation_name}")
        return []

    def parse_dtype(self, dtype_str: str) -> Any:
        """Convert dtype string to ttnn dtype"""
        dtype_mapping = {
            "DataType::BFLOAT16": ttnn.bfloat16,
            "DataType::FLOAT32": ttnn.float32,
            "DataType::INT32": ttnn.int32,
            "DataType::UINT32": ttnn.uint32,
            "DataType::BFLOAT8_B": ttnn.bfloat8_b,
            "DataType::UINT16": ttnn.uint16,
        }
        return dtype_mapping.get(dtype_str, ttnn.bfloat16)

    def parse_layout(self, layout_str: str) -> Any:
        """Convert layout string to ttnn layout"""
        if "TILE" in layout_str:
            return ttnn.TILE_LAYOUT
        elif "ROW_MAJOR" in layout_str:
            return ttnn.ROW_MAJOR_LAYOUT
        else:
            return ttnn.TILE_LAYOUT  # Default

    def parse_memory_config(self, memory_config: Dict, tensor_shape: list = None) -> Any:
        """Convert memory config dict to ttnn memory config

        Args:
            memory_config: Memory config dictionary from master JSON
            tensor_shape: Tensor shape needed for creating sharded configs
        """
        try:
            buffer_type = memory_config.get("buffer_type", "BufferType::DRAM")
            memory_layout = memory_config.get("memory_layout", "TensorMemoryLayout::INTERLEAVED")

            # Map buffer types
            if "DRAM" in buffer_type:
                buffer_type_ttnn = ttnn.BufferType.DRAM
            elif "L1" in buffer_type:
                buffer_type_ttnn = ttnn.BufferType.L1
            else:
                buffer_type_ttnn = ttnn.BufferType.DRAM

            # Map memory layouts
            if "INTERLEAVED" in memory_layout:
                memory_layout_ttnn = ttnn.TensorMemoryLayout.INTERLEAVED
                return ttnn.MemoryConfig(memory_layout_ttnn, buffer_type_ttnn)
            elif "BLOCK_SHARDED" in memory_layout:
                sharding_strategy = ttnn.ShardStrategy.BLOCK
            elif "WIDTH_SHARDED" in memory_layout:
                sharding_strategy = ttnn.ShardStrategy.WIDTH
            elif "HEIGHT_SHARDED" in memory_layout:
                sharding_strategy = ttnn.ShardStrategy.HEIGHT
            else:
                memory_layout_ttnn = ttnn.TensorMemoryLayout.INTERLEAVED
                return ttnn.MemoryConfig(memory_layout_ttnn, buffer_type_ttnn)

            # For sharded configs, we need to create them properly with shard spec
            shard_spec = memory_config.get("shard_spec")
            nd_shard_spec = memory_config.get("nd_shard_spec")  # Prefer nd_shard_spec if available

            # Use nd_shard_spec if available as it has cleaner format
            if nd_shard_spec and isinstance(nd_shard_spec, dict):
                shard_spec = nd_shard_spec

            if shard_spec and shard_spec != "std::nullopt" and tensor_shape:
                import re

                # Extract shard shape - prefer cleaner array format from nd_shard_spec
                shard_shape = None
                if "shard_shape" in shard_spec:
                    # nd_shard_spec format: "shard_shape": [224, 224]
                    shard_shape_data = shard_spec["shard_shape"]
                    if isinstance(shard_shape_data, list) and len(shard_shape_data) >= 2:
                        shard_shape = shard_shape_data[:2]
                elif "shape" in shard_spec:
                    # Regular shard_spec format: "shape": "{224, 224}"
                    shard_shape_str = shard_spec["shape"]
                    if isinstance(shard_shape_str, str):
                        numbers = re.findall(r"\d+", shard_shape_str)
                        if len(numbers) >= 2:
                            shard_shape = [int(numbers[0]), int(numbers[1])]

                # Parse orientation
                orientation_str = shard_spec.get("orientation", "ShardOrientation::ROW_MAJOR")
                shard_orientation = ttnn.ShardOrientation.ROW_MAJOR
                if "COL_MAJOR" in str(orientation_str):
                    shard_orientation = ttnn.ShardOrientation.COL_MAJOR

                # Parse grid to extract core range
                grid_data = shard_spec.get("grid", [])
                core_grid = None

                if grid_data and shard_shape and len(tensor_shape) >= 2:
                    try:
                        # Grid can be in two formats:
                        # 1. Simple: [{"x":0,"y":0}, {"x":7,"y":7}] - single range
                        # 2. Complex: [[{"x":0,"y":0}, {"x":7,"y":5}], [{"x":0,"y":6}, {"x":0,"y":6}]] - multiple ranges

                        if isinstance(grid_data, list) and len(grid_data) > 0:
                            # Check if it's a complex grid (list of lists)
                            if isinstance(grid_data[0], list):
                                # Multiple core ranges - use the first range for now
                                first_range = grid_data[0]
                                if len(first_range) >= 2:
                                    start_coords = first_range[0]
                                    end_coords = first_range[1]
                                else:
                                    start_coords = end_coords = first_range[0] if first_range else {}
                            else:
                                # Simple grid - single range
                                if len(grid_data) >= 2:
                                    start_coords = grid_data[0]
                                    end_coords = grid_data[1]
                                else:
                                    start_coords = end_coords = grid_data[0] if grid_data else {}

                            # Extract coordinates if we have dicts
                            if isinstance(start_coords, dict) and isinstance(end_coords, dict):
                                start_x = start_coords.get("x", 0)
                                start_y = start_coords.get("y", 0)
                                end_x = end_coords.get("x", 0)
                                end_y = end_coords.get("y", 0)

                                # Create CoreGrid from the range
                                # CoreGrid expects (y, x) format and represents number of cores, not end coordinates
                                num_cores_y = end_y - start_y + 1
                                num_cores_x = end_x - start_x + 1
                                core_grid = ttnn.CoreGrid(y=num_cores_y, x=num_cores_x)
                        elif isinstance(grid_data, str):
                            # Try to parse string representation
                            coords = re.findall(r'x["\s]*:\s*(\d+).*?y["\s]*:\s*(\d+)', grid_data)
                            if len(coords) >= 2:
                                start_x, start_y = int(coords[0][0]), int(coords[0][1])
                                end_x, end_y = int(coords[1][0]), int(coords[1][1])
                                num_cores_y = end_y - start_y + 1
                                num_cores_x = end_x - start_x + 1
                                core_grid = ttnn.CoreGrid(y=num_cores_y, x=num_cores_x)

                        if core_grid and shard_shape:
                            # Create ShardSpec manually using the EXACT shard_shape from traced data
                            # Don't let TTNN calculate it - use the actual traced values!
                            from ttnn import CoreCoord, CoreRange, CoreRangeSet

                            # Build CoreRangeSet from grid_data (handle both simple and complex grids)
                            core_ranges = []
                            if isinstance(grid_data[0], list):
                                # Complex multi-range grid
                                for range_pair in grid_data:
                                    if len(range_pair) >= 2:
                                        start = CoreCoord(range_pair[0]["x"], range_pair[0]["y"])
                                        end = CoreCoord(range_pair[1]["x"], range_pair[1]["y"])
                                        core_ranges.append(CoreRange(start, end))
                            else:
                                # Simple single-range grid
                                start = CoreCoord(grid_data[0]["x"], grid_data[0]["y"])
                                end = CoreCoord(grid_data[1]["x"], grid_data[1]["y"])
                                core_ranges.append(CoreRange(start, end))

                            core_range_set = CoreRangeSet(set(core_ranges))

                            # Create ShardSpec with EXACT traced shard_shape
                            shard_spec = ttnn.ShardSpec(core_range_set, shard_shape, shard_orientation)

                            # Map strategy to TensorMemoryLayout
                            if sharding_strategy == ttnn.ShardStrategy.HEIGHT:
                                memory_layout_ttnn = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
                            elif sharding_strategy == ttnn.ShardStrategy.WIDTH:
                                memory_layout_ttnn = ttnn.TensorMemoryLayout.WIDTH_SHARDED
                            elif sharding_strategy == ttnn.ShardStrategy.BLOCK:
                                memory_layout_ttnn = ttnn.TensorMemoryLayout.BLOCK_SHARDED
                            else:
                                memory_layout_ttnn = ttnn.TensorMemoryLayout.INTERLEAVED

                            # Create MemoryConfig with our custom ShardSpec
                            sharded_config = ttnn.MemoryConfig(memory_layout_ttnn, buffer_type_ttnn, shard_spec)
                            return sharded_config
                    except Exception as e:
                        print(f"⚠️ Could not create sharded config: {e}, falling back to interleaved")
                        return ttnn.DRAM_MEMORY_CONFIG

            # Fallback to interleaved if we can't create proper sharded config
            return ttnn.DRAM_MEMORY_CONFIG

        except Exception as e:
            print(f"⚠️ Error parsing memory config, using DRAM default: {e}")
            return ttnn.DRAM_MEMORY_CONFIG

    def _count_tensor_inputs(self, configs: List) -> int:
        """
        Count the number of tensor inputs by checking the first config.

        Args:
            configs: List of operation configurations

        Returns:
            Number of tensor inputs (0, 1, 2, 3, etc.)
        """
        if not configs or len(configs) == 0:
            return 0

        # Check first config for number of tensor arguments
        first_config = configs[0]
        tensor_count = 0

        for arg in first_config:
            tensor_config = self.extract_tensor_config(arg)
            if tensor_config:
                tensor_count += 1

        return tensor_count

    def _is_binary_operation(self, configs: List) -> bool:
        """
        Detect if an operation is binary (2 tensor inputs) by checking the first config.

        Args:
            configs: List of operation configurations

        Returns:
            True if operation has 2 tensor inputs, False otherwise
        """
        return self._count_tensor_inputs(configs) == 2

    def get_suite_parameters(
        self, operation_name: str, suite_name: str = "model_traced", all_cases: bool = False
    ) -> Dict:
        """
        Get ready-to-use sweep test parameters for an operation.
        This is the simplified interface for sweep tests.
        Automatically detects unary vs binary operations.

        Args:
            operation_name: Name of the operation (e.g., 'sigmoid_accurate', 'add')
            suite_name: Name of the test suite (default: 'model_traced')
            all_cases: If False (default), returns unique input configs (N tests, deduplicated).
                      If True, returns all combinations as Cartesian product (N×M×... tests).

        Returns:
            Dictionary with all necessary parameters ready to add to test parameters

        Example (Unary):
            # Default: Run unique input configs (deduplicated, one test per unique input)
            loader = MasterConfigLoader()
            model_traced_params = loader.get_suite_parameters("sigmoid_accurate")  # 30 configs (30 unique inputs)

            # Run all combinations (Cartesian product of all parameters)
            model_traced_params = loader.get_suite_parameters("sigmoid_accurate", all_cases=True)  # Many tests

        Example (Binary):
            # Default: Run unique input pair configs (deduplicated)
            model_traced_params = loader.get_suite_parameters("add")

            # Or: Run all combinations
            model_traced_params = loader.get_suite_parameters("add", all_cases=True)

            parameters = {
                "model_traced": model_traced_params,
            }
        """
        # Register this instance as global so get_traced_config() can access it
        get_global_loader(self)

        try:
            configs = self.get_operation_configs(operation_name)

            if not configs:
                print(f"⚠️ No traced configurations found for {operation_name}")
                # Return empty lists - sweep tests will handle defaults
                return {
                    "input_shape": [[1, 32, 32]],
                    "input_a_dtype": [],
                    "input_a_layout": [],
                    "input_a_memory_config": [],
                    "output_memory_config": [],
                }

            # Special handling for operations with complex parameter structures
            if operation_name in ["conv2d", "ttnn::conv2d"]:
                print(f"🔧 Detected conv2d operation with special parameter structure")
                return self._get_conv2d_suite_parameters(
                    operation_name, configs, all_cases, deduplicate_inputs=not all_cases
                )
            elif operation_name in ["linear", "ttnn::linear"]:
                print(f"🔧 Detected linear operation with special parameter structure")
                return self._get_operation_suite_parameters(
                    operation_name, configs, all_cases, deduplicate_inputs=not all_cases
                )
            elif operation_name in ["embedding", "ttnn::embedding"]:
                print(f"🔧 Detected embedding operation with special parameter structure")
                return self._get_operation_suite_parameters(
                    operation_name, configs, all_cases, deduplicate_inputs=not all_cases
                )

            # Detect the number of tensor inputs
            tensor_count = self._count_tensor_inputs(configs)

            # By default, deduplicate inputs unless running all_cases (Cartesian product)
            deduplicate_inputs = not all_cases

            if tensor_count == 0:
                print(
                    f"⚠️  No tensor inputs detected for {operation_name} - operation may have non-standard parameters"
                )
                print(f"    Treating as unary operation with first argument as input")
                return self._get_unary_suite_parameters(operation_name, configs, all_cases, deduplicate_inputs)
            elif tensor_count == 1:
                print(f"🔧 Detected unary operation: {operation_name} (1 tensor input)")
                return self._get_unary_suite_parameters(operation_name, configs, all_cases, deduplicate_inputs)
            elif tensor_count == 2:
                print(f"🔧 Detected binary operation: {operation_name} (2 tensor inputs)")
                return self._get_binary_suite_parameters(operation_name, configs, all_cases, deduplicate_inputs)
            elif tensor_count >= 3:
                print(f"🔧 Detected multi-input operation: {operation_name} ({tensor_count} tensor inputs)")
                return self._get_multi_input_suite_parameters(
                    operation_name, configs, tensor_count, all_cases, deduplicate_inputs
                )
            else:
                # Fallback - shouldn't reach here
                print(f"⚠️  Unable to determine operation type for {operation_name}, defaulting to unary")
                return self._get_unary_suite_parameters(operation_name, configs, all_cases, deduplicate_inputs)

        except Exception as e:
            print(f"❌ Error loading configurations for {operation_name}: {e}")
            import traceback

            traceback.print_exc()
            return {"traced_config_name": []}

    def _get_unary_suite_parameters(
        self, operation_name: str, configs: List, all_cases: bool, deduplicate_inputs: bool = False
    ) -> Dict:
        """
        Get parameters for unary operations (single tensor input).
        """
        # Extract PAIRED configurations (shape + dtype + layout + memory_config)
        # This ensures we get N exact configs, not a Cartesian product
        paired_configs = []
        failed_configs = 0
        seen_input_signatures = set() if deduplicate_inputs else None

        for config_idx, config in enumerate(configs):
            try:
                # Extract first tensor from each config
                # Config is a list of arguments: [{"UnparsedElement": ...}, {"arg1": "nullopt"}, ...]
                tensor_config = None
                for arg in config:
                    # arg is a dict, could be {"UnparsedElement": ...} or {"arg0": {...}} etc.
                    # Pass the entire arg dict to extract_tensor_config
                    tensor_config = self.extract_tensor_config(arg)
                    if tensor_config:
                        break  # Found tensor, proceed to parse it

                if not tensor_config:
                    failed_configs += 1
                    continue

                # Parse the config to TTNN types
                try:
                    parsed_dtype = self.parse_dtype(tensor_config.dtype)
                    parsed_layout = self.parse_layout(tensor_config.layout)
                    parsed_mem_config = self.parse_memory_config(tensor_config.memory_config, tensor_config.shape)

                    if parsed_dtype and parsed_layout and parsed_mem_config:
                        # If deduplicating, check if we've seen this input before
                        if deduplicate_inputs:
                            import hashlib

                            input_sig = hashlib.md5(
                                str((tensor_config.shape, parsed_dtype, parsed_layout, parsed_mem_config)).encode()
                            ).hexdigest()
                            if input_sig in seen_input_signatures:
                                continue  # Skip this config, we already have one with this input
                            seen_input_signatures.add(input_sig)

                        # Determine output memory config based on operation
                        if operation_name == "sharded_to_interleaved":
                            # This operation converts sharded to interleaved, so output must be INTERLEAVED
                            output_mem_config = ttnn.DRAM_MEMORY_CONFIG  # Interleaved DRAM
                        else:
                            # For most unary ops, output matches input
                            output_mem_config = parsed_mem_config

                        # Override layout for operations that require ROW_MAJOR
                        if operation_name in [
                            "pad",
                            "tilize_with_val_padding",
                            "ttnn::pad",
                            "ttnn::tilize_with_val_padding",
                        ]:
                            # These operations require ROW_MAJOR layout
                            parsed_layout = ttnn.ROW_MAJOR_LAYOUT

                        config_dict = {
                            "shape": tensor_config.shape,
                            "dtype": parsed_dtype,
                            "layout": parsed_layout,
                            "memory_config": parsed_mem_config,
                            "output_memory_config": output_mem_config,
                        }

                        # Extract operation-specific parameters for certain operations
                        if operation_name == "permute" or operation_name == "ttnn::permute":
                            # For permute, extract dims from arg1
                            dims = self._extract_permute_dims(config)
                            if dims:
                                config_dict["dims"] = dims
                            else:
                                # Fallback to default if extraction fails
                                config_dict["dims"] = [0, 1, 3, 2]  # N, C, W, H -> N, C, H, W
                        elif operation_name == "untilize_with_unpadding":
                            # For untilize_with_unpadding, extract end_shape from arg1
                            end_shape = self._extract_shape_parameter(config, arg_name="arg1")
                            if end_shape:
                                config_dict["end_shape"] = end_shape
                        elif operation_name == "transpose":
                            # For transpose, extract dim0 and dim1 from arg1 and arg2
                            dim0 = self._extract_int_parameter(config, "arg1")
                            dim1 = self._extract_int_parameter(config, "arg2")
                            if dim0 is not None and dim1 is not None:
                                config_dict["dim0"] = dim0
                                config_dict["dim1"] = dim1
                        elif operation_name == "reshape":
                            # For reshape, extract target_shape from arg1
                            target_shape = self._extract_shape_parameter(config, arg_name="arg1")
                            if target_shape:
                                config_dict["target_shape"] = target_shape
                        elif operation_name == "pad":
                            # For pad, extract padding (arg1) and value (arg2)
                            padding = None
                            value = None
                            for arg in config:
                                if isinstance(arg, dict):
                                    if "arg1" in arg:
                                        padding = self._parse_list_from_string(arg["arg1"])
                                        # Normalize padding format
                                        if padding and isinstance(padding, list):
                                            # Check if it's a flat list that needs conversion
                                            if len(padding) == 4 and all(isinstance(x, int) for x in padding):
                                                # Convert [front_H, back_H, front_W, back_W] to [[0,0], [0,0], [front_H, back_H], [front_W, back_W]]
                                                padding = [
                                                    [0, 0],
                                                    [0, 0],
                                                    [padding[0], padding[1]],
                                                    [padding[2], padding[3]],
                                                ]
                                    if "arg2" in arg:
                                        value = self._parse_numeric_value(arg["arg2"])
                                        # Value must be a single float, not a list
                                        if isinstance(value, list):
                                            # If all elements are the same, use that value
                                            if len(set(value)) == 1:
                                                value = float(value[0])
                                            else:
                                                # Use the first element (or could skip this config)
                                                value = float(value[0])
                                        elif value is not None:
                                            value = float(value)
                            if padding is not None and value is not None:
                                config_dict["padding"] = padding
                                config_dict["value"] = value
                        elif operation_name == "tilize_with_val_padding":
                            # For tilize_with_val_padding, extract padded_shape (arg1) and pad_value (arg2)
                            padded_shape = self._extract_shape_parameter(config, arg_name="arg1")
                            pad_value = None
                            for arg in config:
                                if isinstance(arg, dict) and "arg2" in arg:
                                    pad_value = self._parse_numeric_value(arg["arg2"])
                                    break
                            if padded_shape and pad_value is not None:
                                config_dict["padded_shape"] = padded_shape
                                config_dict["pad_value"] = pad_value

                        paired_configs.append(config_dict)
                    else:
                        failed_configs += 1
                except (AttributeError, Exception) as e:
                    failed_configs += 1
            except Exception as e:
                failed_configs += 1

        # Build parameter dictionary based on all_cases flag
        if paired_configs:
            # Store configs in instance cache for lookup
            self.traced_configs_cache[operation_name] = paired_configs

            if all_cases:
                # Return separate lists for Cartesian product
                # Extract UNIQUE values for each parameter type
                unique_shapes = []
                seen_shapes = set()
                dtypes = set()
                layouts = set()
                unique_memory_configs = []
                seen_mem_configs = set()
                unique_output_memory_configs = []
                seen_output_mem_configs = set()

                for cfg in paired_configs:
                    # Track unique shapes
                    shape_tuple = tuple(cfg["shape"])
                    if shape_tuple not in seen_shapes:
                        unique_shapes.append(cfg["shape"])
                        seen_shapes.add(shape_tuple)

                    dtypes.add(cfg["dtype"])
                    layouts.add(cfg["layout"])

                    # Track unique memory configs (using str representation as key)
                    mem_config_str = str(cfg["memory_config"])
                    if mem_config_str not in seen_mem_configs:
                        unique_memory_configs.append(cfg["memory_config"])
                        seen_mem_configs.add(mem_config_str)

                    output_mem_config_str = str(cfg["output_memory_config"])
                    if output_mem_config_str not in seen_output_mem_configs:
                        unique_output_memory_configs.append(cfg["output_memory_config"])
                        seen_output_mem_configs.add(output_mem_config_str)

                result = {
                    "input_shape": unique_shapes,
                    "input_a_dtype": list(dtypes),
                    "input_a_layout": list(layouts),
                    "input_a_memory_config": unique_memory_configs,
                    "output_memory_config": unique_output_memory_configs,
                }

                total_tests = (
                    len(unique_shapes)
                    * len(dtypes)
                    * len(layouts)
                    * len(unique_memory_configs)
                    * len(unique_output_memory_configs)
                )
                print(f"✅ Loaded {len(paired_configs)} traced configurations for {operation_name} (model_traced suite)")
                print(f"   📊 all_cases=True: Will generate ~{total_tests} test vectors (Cartesian product)")
                if failed_configs > 0:
                    print(f"⚠️ Failed to parse {failed_configs} configurations")
            else:
                # Return individual parameter lists to match sweep file expectations
                # Extract each parameter type into separate lists
                input_shapes = []
                input_a_dtypes = []
                input_a_layouts = []
                input_a_memory_configs = []
                output_memory_configs = []
                traced_config_names = []
                dims_list = [] if (operation_name == "permute" or operation_name == "ttnn::permute") else None
                end_shape_list = [] if operation_name == "untilize_with_unpadding" else None
                dim0_list = [] if operation_name == "transpose" else None
                dim1_list = [] if operation_name == "transpose" else None
                target_shape_list = [] if operation_name == "reshape" else None
                padding_list = [] if operation_name == "pad" else None
                value_list = [] if operation_name == "pad" else None
                padded_shape_list = [] if operation_name == "tilize_with_val_padding" else None
                pad_value_list = [] if operation_name == "tilize_with_val_padding" else None

                for idx, cfg in enumerate(paired_configs):
                    input_shapes.append(cfg["shape"])
                    input_a_dtypes.append(cfg["dtype"])
                    input_a_layouts.append(cfg["layout"])
                    input_a_memory_configs.append(cfg["memory_config"])
                    output_memory_configs.append(cfg["output_memory_config"])
                    traced_config_names.append(f"{operation_name}_traced_{idx}")
                    if (operation_name == "permute" or operation_name == "ttnn::permute") and "dims" in cfg:
                        dims_list.append(cfg["dims"])
                    if operation_name == "untilize_with_unpadding" and "end_shape" in cfg:
                        end_shape_list.append(cfg["end_shape"])
                    if operation_name == "transpose":
                        if "dim0" in cfg and "dim1" in cfg:
                            dim0_list.append(cfg["dim0"])
                            dim1_list.append(cfg["dim1"])
                    if operation_name == "reshape" and "target_shape" in cfg:
                        target_shape_list.append(cfg["target_shape"])
                    if operation_name == "pad":
                        if "padding" in cfg and "value" in cfg:
                            padding_list.append(cfg["padding"])
                            value_list.append(cfg["value"])
                    if operation_name == "tilize_with_val_padding":
                        if "padded_shape" in cfg and "pad_value" in cfg:
                            padded_shape_list.append(cfg["padded_shape"])
                            pad_value_list.append(cfg["pad_value"])

                # Convert to exact configurations format (prevents Cartesian product)
                # Use comma-separated parameter names to pass tuples of values together
                param_names = [
                    "input_shape",
                    "input_a_dtype",
                    "input_a_layout",
                    "input_a_memory_config",
                    "output_memory_config",
                ]
                param_lists = [
                    input_shapes,
                    input_a_dtypes,
                    input_a_layouts,
                    input_a_memory_configs,
                    output_memory_configs,
                ]

                # Add operation-specific parameters
                if (operation_name == "permute" or operation_name == "ttnn::permute") and dims_list:
                    param_names.append("dims")
                    param_lists.append(dims_list)
                if operation_name == "untilize_with_unpadding" and end_shape_list:
                    param_names.append("end_shape")
                    param_lists.append(end_shape_list)
                if operation_name == "transpose" and dim0_list and dim1_list:
                    param_names.extend(["dim0", "dim1"])
                    param_lists.extend([dim0_list, dim1_list])
                if operation_name == "reshape" and target_shape_list:
                    param_names.append("target_shape")
                    param_lists.append(target_shape_list)
                if operation_name == "pad" and padding_list and value_list:
                    param_names.extend(["padding", "value"])
                    param_lists.extend([padding_list, value_list])
                if operation_name == "tilize_with_val_padding" and padded_shape_list and pad_value_list:
                    param_names.extend(["padded_shape", "pad_value"])
                    param_lists.extend([padded_shape_list, pad_value_list])

                # NOTE: traced_config_name is metadata only, not passed to run()
                # param_names.append("traced_config_name")
                # param_lists.append(traced_config_names)

                # Create tuples of exact configurations
                exact_configs = list(zip(*param_lists))
                param_key = ",".join(param_names)

                result = {param_key: exact_configs}

                print(f"✅ Loaded {len(paired_configs)} traced configurations for {operation_name} (model_traced suite)")
                dedup_msg = " (unique inputs)" if deduplicate_inputs else " (all input/output pairs)"
                print(f"   📊 Will generate {len(paired_configs)} test vectors{dedup_msg}")
                if failed_configs > 0:
                    print(f"⚠️ Failed to parse {failed_configs} configurations")

            return result
        else:
            # No configs successfully parsed, return empty
            print(f"⚠️ No configurations could be parsed for {operation_name} (TTNN may not be initialized)")
            return {
                "traced_config": [],
            }

    def _get_binary_suite_parameters(
        self, operation_name: str, configs: List, all_cases: bool, deduplicate_inputs: bool = False
    ) -> Dict:
        """
        Get parameters for binary operations (two tensor inputs).
        Handles operations like add, multiply, etc. that take two tensors.
        """
        # Extract configurations for BOTH tensors (arg0 and arg1)
        paired_configs = []
        failed_configs = 0

        for config_idx, config in enumerate(configs):
            try:
                # Extract BOTH tensors from each config
                tensor_configs = []
                for arg in config:
                    tensor_config = self.extract_tensor_config(arg)
                    if tensor_config:
                        tensor_configs.append(tensor_config)
                        if len(tensor_configs) >= 2:
                            break  # We have both tensors

                if len(tensor_configs) < 2:
                    failed_configs += 1
                    continue

                # Parse both tensor configs
                try:
                    # First tensor (input_a)
                    parsed_dtype_a = self.parse_dtype(tensor_configs[0].dtype)
                    parsed_layout_a = self.parse_layout(tensor_configs[0].layout)
                    parsed_mem_config_a = self.parse_memory_config(
                        tensor_configs[0].memory_config, tensor_configs[0].shape
                    )

                    # Second tensor (input_b)
                    parsed_dtype_b = self.parse_dtype(tensor_configs[1].dtype)
                    parsed_layout_b = self.parse_layout(tensor_configs[1].layout)
                    parsed_mem_config_b = self.parse_memory_config(
                        tensor_configs[1].memory_config, tensor_configs[1].shape
                    )

                    if all(
                        [
                            parsed_dtype_a,
                            parsed_layout_a,
                            parsed_mem_config_a,
                            parsed_dtype_b,
                            parsed_layout_b,
                            parsed_mem_config_b,
                        ]
                    ):
                        paired_configs.append(
                            {
                                "shape_a": tensor_configs[0].shape,
                                "shape_b": tensor_configs[1].shape,
                                "dtype_a": parsed_dtype_a,
                                "dtype_b": parsed_dtype_b,
                                "layout_a": parsed_layout_a,
                                "layout_b": parsed_layout_b,
                                "memory_config_a": parsed_mem_config_a,
                                "memory_config_b": parsed_mem_config_b,
                                "output_memory_config": parsed_mem_config_a,  # Use first input's memory config as default
                            }
                        )
                    else:
                        failed_configs += 1
                except (AttributeError, Exception) as e:
                    failed_configs += 1
            except Exception as e:
                failed_configs += 1

        # Build parameter dictionary based on all_cases flag
        if paired_configs:
            # Store configs in instance cache for lookup (for binary ops)
            self.traced_configs_cache[operation_name] = paired_configs

            if all_cases:
                # Return separate lists for Cartesian product
                # Extract UNIQUE values for each parameter type
                unique_shapes_a = []
                unique_shapes_b = []
                seen_shape_pairs = set()
                dtypes_a = set()
                dtypes_b = set()
                layouts_a = set()
                layouts_b = set()
                unique_memory_configs_a = []
                unique_memory_configs_b = []
                seen_mem_config_pairs = set()

                for cfg in paired_configs:
                    # Track unique shape pairs
                    shape_pair = (tuple(cfg["shape_a"]), tuple(cfg["shape_b"]))
                    if shape_pair not in seen_shape_pairs:
                        unique_shapes_a.append(cfg["shape_a"])
                        unique_shapes_b.append(cfg["shape_b"])
                        seen_shape_pairs.add(shape_pair)

                    dtypes_a.add(cfg["dtype_a"])
                    dtypes_b.add(cfg["dtype_b"])
                    layouts_a.add(cfg["layout_a"])
                    layouts_b.add(cfg["layout_b"])

                    # Track unique memory config pairs
                    mem_config_pair = (str(cfg["memory_config_a"]), str(cfg["memory_config_b"]))
                    if mem_config_pair not in seen_mem_config_pairs:
                        unique_memory_configs_a.append(cfg["memory_config_a"])
                        unique_memory_configs_b.append(cfg["memory_config_b"])
                        seen_mem_config_pairs.add(mem_config_pair)

                # For binary operations, input_shape is a dict with "self" and "other"
                input_shapes = [{"self": sa, "other": sb} for sa, sb in zip(unique_shapes_a, unique_shapes_b)]

                result = {
                    "input_shape": input_shapes,
                    "input_a_dtype": list(dtypes_a),
                    "input_b_dtype": list(dtypes_b),
                    "input_a_layout": list(layouts_a),
                    "input_b_layout": list(layouts_b),
                    "input_a_memory_config": unique_memory_configs_a,
                    "input_b_memory_config": unique_memory_configs_b,
                }

                total_tests = (
                    len(input_shapes)
                    * len(dtypes_a)
                    * len(dtypes_b)
                    * len(layouts_a)
                    * len(layouts_b)
                    * len(unique_memory_configs_a)
                    * len(unique_memory_configs_b)
                )
                print(f"✅ Loaded {len(paired_configs)} traced configurations for {operation_name} (model_traced suite)")
                print(f"   📊 all_cases=True: Will generate ~{total_tests} test vectors (Cartesian product)")
                if failed_configs > 0:
                    print(f"⚠️ Failed to parse {failed_configs} configurations")
            else:
                # Return individual parameter lists to match sweep file expectations
                # Extract each parameter type into separate lists
                input_shapes = []
                input_a_dtypes = []
                input_b_dtypes = []
                input_a_layouts = []
                input_b_layouts = []
                input_a_memory_configs = []
                input_b_memory_configs = []
                output_memory_configs = []
                traced_config_names = []

                for idx, cfg in enumerate(paired_configs):
                    input_shapes.append({"self": cfg["shape_a"], "other": cfg["shape_b"]})
                    input_a_dtypes.append(cfg["dtype_a"])
                    input_b_dtypes.append(cfg["dtype_b"])
                    input_a_layouts.append(cfg["layout_a"])
                    input_b_layouts.append(cfg["layout_b"])
                    input_a_memory_configs.append(cfg["memory_config_a"])
                    input_b_memory_configs.append(cfg["memory_config_b"])
                    output_memory_configs.append(cfg["output_memory_config"])
                    traced_config_names.append(f"{operation_name}_traced_{idx}")

                # Convert to exact configurations format (prevents Cartesian product)
                # Use comma-separated parameter names to pass tuples of values together
                param_names = [
                    "input_shape",
                    "input_a_dtype",
                    "input_b_dtype",
                    "input_a_layout",
                    "input_b_layout",
                    "input_a_memory_config",
                    "input_b_memory_config",
                    "output_memory_config",
                    # NOTE: traced_config_name is metadata only, not passed to run()
                    # "traced_config_name",
                ]
                param_lists = [
                    input_shapes,
                    input_a_dtypes,
                    input_b_dtypes,
                    input_a_layouts,
                    input_b_layouts,
                    input_a_memory_configs,
                    input_b_memory_configs,
                    output_memory_configs,
                    # traced_config_names,
                ]

                # Create tuples of exact configurations
                exact_configs = list(zip(*param_lists))
                param_key = ",".join(param_names)

                result = {param_key: exact_configs}

                print(f"✅ Loaded {len(paired_configs)} traced configurations for {operation_name} (model_traced suite)")
                dedup_msg = " (unique input pairs)" if deduplicate_inputs else " (all input/output pairs)"
                print(f"   📊 Will generate {len(paired_configs)} test vectors{dedup_msg}")
                if failed_configs > 0:
                    print(f"⚠️ Failed to parse {failed_configs} configurations")

            return result
        else:
            # No configs successfully parsed, return empty
            print(f"⚠️ No configurations could be parsed for {operation_name} (TTNN may not be initialized)")
            return {
                "traced_config_name": [],
            }

    def _get_multi_input_suite_parameters(
        self, operation_name: str, configs: List, tensor_count: int, all_cases: bool, deduplicate_inputs: bool = False
    ) -> Dict:
        """
        Get parameters for multi-input operations (3+ tensor inputs).
        Handles operations like where (ternary), addcmul, etc.
        """
        # Extract configurations for ALL tensors
        paired_configs = []
        failed_configs = 0

        for config_idx, config in enumerate(configs):
            try:
                # Extract ALL tensors from each config
                tensor_configs = []
                for arg in config:
                    tensor_config = self.extract_tensor_config(arg)
                    if tensor_config:
                        tensor_configs.append(tensor_config)
                        if len(tensor_configs) >= tensor_count:
                            break

                if len(tensor_configs) < tensor_count:
                    failed_configs += 1
                    continue

                # Parse all tensor configs
                parsed_config = {}
                for i, tc in enumerate(tensor_configs):
                    suffix = chr(97 + i)  # a, b, c, d, ...
                    try:
                        parsed_config[f"shape_{suffix}"] = tc.shape
                        parsed_config[f"dtype_{suffix}"] = self.parse_dtype(tc.dtype)
                        parsed_config[f"layout_{suffix}"] = self.parse_layout(tc.layout)
                        parsed_config[f"memory_config_{suffix}"] = self.parse_memory_config(tc.memory_config, tc.shape)
                    except Exception as e:
                        failed_configs += 1
                        break

                if len(parsed_config) == tensor_count * 4:  # shape, dtype, layout, mem_config for each tensor
                    paired_configs.append(parsed_config)

            except Exception as e:
                failed_configs += 1

        # Build parameter dictionary based on all_cases flag
        if paired_configs:
            # Store configs in instance cache for lookup
            self.traced_configs_cache[operation_name] = paired_configs

            if all_cases:
                # Return separate lists for Cartesian product
                result = {}

                # Build input_shape as list of dicts
                unique_shapes = []
                seen_shapes = set()
                for cfg in paired_configs:
                    shape_tuple = tuple([tuple(cfg[f"shape_{chr(97+i)}"]) for i in range(tensor_count)])
                    if shape_tuple not in seen_shapes:
                        shape_dict = {f"input_{chr(97+i)}": cfg[f"shape_{chr(97+i)}"] for i in range(tensor_count)}
                        unique_shapes.append(shape_dict)
                        seen_shapes.add(shape_tuple)

                result["input_shape"] = unique_shapes

                # Add dtypes, layouts, and memory configs for each input
                for i in range(tensor_count):
                    suffix = chr(97 + i)  # a, b, c, ...
                    result[f"input_{suffix}_dtype"] = list(set(cfg[f"dtype_{suffix}"] for cfg in paired_configs))
                    result[f"input_{suffix}_layout"] = list(set(cfg[f"layout_{suffix}"] for cfg in paired_configs))

                    unique_mem_configs = []
                    seen = set()
                    for cfg in paired_configs:
                        mc_str = str(cfg[f"memory_config_{suffix}"])
                        if mc_str not in seen:
                            unique_mem_configs.append(cfg[f"memory_config_{suffix}"])
                            seen.add(mc_str)
                    result[f"input_{suffix}_memory_config"] = unique_mem_configs

                total_tests = len(unique_shapes)
                for i in range(tensor_count):
                    suffix = chr(97 + i)
                    total_tests *= len(result[f"input_{suffix}_dtype"])
                    total_tests *= len(result[f"input_{suffix}_layout"])
                    total_tests *= len(result[f"input_{suffix}_memory_config"])

                print(f"✅ Loaded {len(paired_configs)} traced configurations for {operation_name} (model_traced suite)")
                print(f"   📊 all_cases=True: Will generate ~{total_tests} test vectors (Cartesian product)")
                if failed_configs > 0:
                    print(f"⚠️ Failed to parse {failed_configs} configurations")
            else:
                # Return individual parameter lists to match sweep file expectations
                # Extract each parameter type into separate lists
                input_shapes = []
                dtypes = [[] for _ in range(tensor_count)]
                layouts = [[] for _ in range(tensor_count)]
                memory_configs = [[] for _ in range(tensor_count)]
                traced_config_names = []

                for idx, cfg in enumerate(paired_configs):
                    # Build input_shape dict
                    input_shape = {f"input_{chr(97+i)}": cfg[f"shape_{chr(97+i)}"] for i in range(tensor_count)}
                    input_shapes.append(input_shape)

                    # Extract dtypes, layouts, and memory configs for each input
                    for i in range(tensor_count):
                        suffix = chr(97 + i)
                        dtypes[i].append(cfg[f"dtype_{suffix}"])
                        layouts[i].append(cfg[f"layout_{suffix}"])
                        memory_configs[i].append(cfg[f"memory_config_{suffix}"])

                    traced_config_names.append(f"{operation_name}_traced_{idx}")

                # Convert to exact configurations format (prevents Cartesian product)
                # Use comma-separated parameter names to pass tuples of values together
                param_names = ["input_shape"]
                param_lists = [input_shapes]

                # Add dtypes, layouts, and memory configs for each input
                for i in range(tensor_count):
                    suffix = chr(97 + i)
                    param_names.extend(
                        [f"input_{suffix}_dtype", f"input_{suffix}_layout", f"input_{suffix}_memory_config"]
                    )
                    param_lists.extend([dtypes[i], layouts[i], memory_configs[i]])

                # NOTE: traced_config_name is metadata only, not passed to run()
                # param_names.append("traced_config_name")
                # param_lists.append(traced_config_names)

                # Create tuples of exact configurations
                exact_configs = list(zip(*param_lists))
                param_key = ",".join(param_names)

                result = {param_key: exact_configs}

                print(f"✅ Loaded {len(paired_configs)} traced configurations for {operation_name} (model_traced suite)")
                print(f"   📊 Will generate {len(paired_configs)} test vectors")
                if failed_configs > 0:
                    print(f"⚠️ Failed to parse {failed_configs} configurations")

            return result
        else:
            # No configs successfully parsed, return empty
            print(f"⚠️ No configurations could be parsed for {operation_name} (TTNN may not be initialized)")
            return {
                "traced_config_name": [],
            }

    def get_traced_config(self, traced_config_name: str) -> Optional[Dict]:
        """
        Look up a traced config by its name.
        Handles both unary and binary operations.

        Args:
            traced_config_name: String name like "sigmoid_accurate_traced_0" or "add_traced_0"

        Returns:
            For unary ops: Dictionary with 'shape', 'dtype', 'layout', 'memory_config', 'output_memory_config'
            For binary ops: Dictionary with 'shape_a', 'shape_b', 'dtype_a', 'dtype_b', etc.
            Returns None if not found
        """
        if traced_config_name is None:
            return None

        # Extract operation name and index from name like "sigmoid_accurate_traced_0"
        parts = traced_config_name.split("_traced_")
        if len(parts) != 2:
            return None

        operation_name = parts[0]
        try:
            config_idx = int(parts[1])
        except ValueError:
            return None

        # Look up in cache
        if operation_name in self.traced_configs_cache:
            configs = self.traced_configs_cache[operation_name]
            if config_idx < len(configs):
                config = configs[config_idx]
                # Add a flag to indicate if it's binary
                if "shape_a" in config and "shape_b" in config:
                    config["is_binary"] = True
                else:
                    config["is_binary"] = False
                return config

        return None

    def _extract_permute_dims(self, config: List) -> Optional[List[int]]:
        """Extract dims parameter from permute operation config"""
        try:
            # Look for arg1 which should contain the dims parameter
            for arg in config:
                if isinstance(arg, dict) and "arg1" in arg:
                    dims_str = arg["arg1"]
                    # The dims are in format '[0, 2, 3, 1]' or similar
                    if isinstance(dims_str, str) and dims_str.startswith("[") and dims_str.endswith("]"):
                        # Parse the list string
                        dims_str = dims_str.strip("[]")
                        if dims_str:
                            dims = [int(x.strip()) for x in dims_str.split(",")]
                            return dims
            return None
        except Exception as e:
            return None

    def _parse_list_from_string(self, value) -> Optional[List]:
        """Parse a list from string representation or return if already a list"""
        try:
            # If already a list, return it
            if isinstance(value, list):
                return value
            # If string, try to parse it
            if isinstance(value, str):
                value = value.strip()
                if value.startswith("[") and value.endswith("]"):
                    # Use json.loads for safer parsing
                    import json

                    return json.loads(value.replace("'", '"'))
            return None
        except Exception as e:
            return None

    def _parse_numeric_value(self, value):
        """Parse numeric value from string or return if already numeric"""
        try:
            # If already a number, return it
            if isinstance(value, (int, float)):
                return value
            # If list, check if it's a numeric list or parse each element
            if isinstance(value, list):
                # Could be a list of numbers for value parameter
                return value
            # If string, try to parse it
            if isinstance(value, str):
                value = value.strip()
                # Try as list first
                if value.startswith("["):
                    parsed = self._parse_list_from_string(value)
                    if parsed is not None:
                        return parsed
                # Try as float
                if "." in value:
                    return float(value)
                # Try as int
                return int(value)
            return None
        except Exception as e:
            return None

    def _extract_shape_parameter(self, config: List, arg_name: str = "arg1") -> Optional[List[int]]:
        """Extract Shape parameter from config (e.g., for untilize_with_unpadding end_shape, reshape target_shape)"""
        try:
            for arg in config:
                if isinstance(arg, dict) and arg_name in arg:
                    shape_data = arg[arg_name]
                    # Handle dict with 'Shape' key
                    if isinstance(shape_data, dict) and "Shape" in shape_data:
                        shape = shape_data["Shape"]
                        if isinstance(shape, list):
                            return shape
                    # Handle string representation of list
                    elif isinstance(shape_data, str):
                        parsed = self._parse_list_from_string(shape_data)
                        if parsed is not None and isinstance(parsed, list):
                            return parsed
                    # Handle direct list
                    elif isinstance(shape_data, list):
                        return shape_data
            return None
        except Exception as e:
            return None

    def _extract_int_parameter(self, config: List, arg_name: str) -> Optional[int]:
        """Extract integer parameter from config (e.g., for transpose dim0, dim1)"""
        try:
            for arg in config:
                if isinstance(arg, dict) and arg_name in arg:
                    value = arg[arg_name]
                    if isinstance(value, (int, str)):
                        return int(value)
            return None
        except Exception as e:
            return None

    def _get_conv2d_suite_parameters(
        self, operation_name: str, configs: List, all_cases: bool, deduplicate_inputs: bool = False
    ) -> Dict:
        """Get parameters for conv2d operation which uses input_specs format"""
        try:
            input_specs_list = []

            for config in configs:
                params = self._extract_conv2d_parameters(config)
                if params:
                    # Build input_specs list:
                    # [batch_size, output_channels, input_channels, input_height, input_width,
                    #  kernel_height, kernel_width, stride_h, stride_w, pad_h, pad_w, groups, dilation_h, dilation_w, bias]
                    input_spec = [
                        params["batch_size"],
                        params["output_channels"],
                        params["input_channels"],
                        params["input_height"],
                        params["input_width"],
                        params["kernel_height"],
                        params["kernel_width"],
                        params["stride_h"],
                        params["stride_w"],
                        params["pad_h"],
                        params["pad_w"],
                        params["groups"],
                        params["dilation_h"],
                        params["dilation_w"],
                        params["has_bias"],
                    ]
                    input_specs_list.append(input_spec)

            if input_specs_list:
                print(
                    f"✅ Loaded {len(input_specs_list)} traced configurations for {operation_name} (model_traced suite)"
                )
                return {
                    "input_specs": input_specs_list,
                    "is_conv1d": [False] * len(input_specs_list),
                }

            return {"input_specs": [], "is_conv1d": []}
        except Exception as e:
            print(f"❌ Error extracting conv2d parameters: {e}")
            import traceback

            traceback.print_exc()
            return {"input_specs": [], "is_conv1d": []}

    def _get_linear_suite_parameters(
        self, operation_name: str, configs: List, all_cases: bool, deduplicate_inputs: bool = False
    ) -> Dict:
        """Get parameters for linear operation"""
        try:
            paired_configs = []

            for config in configs:
                # Extract base tensor config for input tensor
                tensor_config = None
                for arg in config:
                    tc = self.extract_tensor_config(arg)
                    if tc:
                        tensor_config = tc
                        break

                if not tensor_config:
                    continue

                # Extract linear-specific parameters
                linear_params = self._extract_linear_parameters(config)
                if not linear_params:
                    continue

                # Parse tensor config
                parsed_dtype = self.parse_dtype(tensor_config.dtype)
                parsed_layout = self.parse_layout(tensor_config.layout)
                parsed_mem_config = self.parse_memory_config(tensor_config.memory_config, tensor_config.shape)

                if parsed_dtype and parsed_layout and parsed_mem_config:
                    config_dict = {
                        "input_shape": linear_params["input_shape"],
                        "weight_shape": linear_params["weight_shape"],
                        "bias_shape": linear_params["bias_shape"],
                        "input_a_dtype": parsed_dtype,
                        "input_b_dtype": parsed_dtype,  # Assume same dtype for weight
                        "input_a_layout": parsed_layout,
                        "input_b_layout": parsed_layout,  # Assume same layout for weight
                        "input_a_memory_config": parsed_mem_config,
                        "input_b_memory_config": parsed_mem_config,  # Assume same memory config
                        "output_memory_config": parsed_mem_config,
                        "transpose_a": linear_params["transpose_a"],
                        "transpose_b": linear_params["transpose_b"],
                        "has_bias": linear_params["has_bias"],
                    }
                    paired_configs.append(config_dict)

            if paired_configs:
                print(f"✅ Loaded {len(paired_configs)} traced configurations for {operation_name} (model_traced suite)")

                # Build parameter dict
                param_names = [
                    "input_shape,weight_shape,bias_shape,input_a_dtype,input_b_dtype,input_a_layout,input_b_layout,"
                    + "input_a_memory_config,input_b_memory_config,output_memory_config,transpose_a,transpose_b,has_bias"
                ]
                param_lists = [
                    [
                        (
                            cfg["input_shape"],
                            cfg["weight_shape"],
                            cfg["bias_shape"],
                            cfg["input_a_dtype"],
                            cfg["input_b_dtype"],
                            cfg["input_a_layout"],
                            cfg["input_b_layout"],
                            cfg["input_a_memory_config"],
                            cfg["input_b_memory_config"],
                            cfg["output_memory_config"],
                            cfg["transpose_a"],
                            cfg["transpose_b"],
                            cfg["has_bias"],
                        )
                        for cfg in paired_configs
                    ]
                ]

                return {param_names[0]: param_lists[0]}

            return {}
        except Exception as e:
            print(f"❌ Error extracting linear parameters: {e}")
            import traceback

            traceback.print_exc()
            return {}

    def _get_operation_suite_parameters(
        self, operation_name: str, configs: List, all_cases: bool, deduplicate_inputs: bool = False
    ) -> Dict:
        """Get parameters for operations using the OperationParameterExtractors registry"""
        try:
            # Clean operation name (remove namespace prefix if present)
            clean_op_name = operation_name.replace("ttnn::", "")

            # First extract parameters from each config
            extracted_params = []
            for config in configs:
                params = OperationParameterExtractors.extract_parameters(clean_op_name, config)
                if params:
                    extracted_params.append(params)

            # Then transform the extracted parameters
            if extracted_params:
                transformed_configs = OperationParameterExtractors.transform_parameters(clean_op_name, extracted_params)

                if transformed_configs:
                    print(
                        f"✅ Loaded {len(transformed_configs)} traced configurations for {operation_name} (model_traced suite)"
                    )

                    # For embedding, we have a specific parameter format
                    if clean_op_name == "embedding":
                        param_names = [
                            "input_shape,input_a_dtype,input_b_dtype,input_a_layout,input_b_layout,"
                            + "input_a_memory_config,input_b_memory_config,output_memory_config"
                        ]
                        param_lists = [
                            [
                                (
                                    cfg["input_shape"],
                                    cfg["input_a_dtype"],
                                    cfg["input_b_dtype"],
                                    cfg["input_a_layout"],
                                    cfg["input_b_layout"],
                                    cfg["input_a_memory_config"],
                                    cfg["input_b_memory_config"],
                                    cfg["output_memory_config"],
                                )
                                for cfg in transformed_configs
                            ]
                        ]
                        return {param_names[0]: param_lists[0]}

                    # For linear, we have a specific parameter format
                    elif clean_op_name == "linear":
                        param_names = [
                            "input_shape,weight_shape,bias_shape,input_a_dtype,input_b_dtype,input_a_layout,input_b_layout,"
                            + "input_a_memory_config,input_b_memory_config,output_memory_config,transpose_a,transpose_b,has_bias"
                        ]
                        param_lists = [
                            [
                                (
                                    cfg["input_shape"],
                                    cfg["weight_shape"],
                                    cfg["bias_shape"],
                                    cfg["input_a_dtype"],
                                    cfg["input_b_dtype"],
                                    cfg["input_a_layout"],
                                    cfg["input_b_layout"],
                                    cfg["input_a_memory_config"],
                                    cfg["input_b_memory_config"],
                                    cfg["output_memory_config"],
                                    cfg["transpose_a"],
                                    cfg["transpose_b"],
                                    cfg["has_bias"],
                                )
                                for cfg in transformed_configs
                            ]
                        ]
                        return {param_names[0]: param_lists[0]}

                    # For other operations, return the transformed configs directly
                    # This would need to be customized per operation
                    return {}

            return {}
        except Exception as e:
            print(f"❌ Error extracting {operation_name} parameters: {e}")
            import traceback

            traceback.print_exc()
            return {}

    def _extract_conv2d_parameters(self, config: List) -> Optional[Dict]:
        """Extract all parameters for conv2d operation"""
        try:
            # Conv2d parameter mapping:
            # arg3: input_channels, arg4: output_channels, arg5: batch_size
            # arg6: input_height, arg7: input_width
            # arg8: [kernel_h, kernel_w], arg9: [stride_h, stride_w]
            # arg10: [pad_h1, pad_h2, pad_w1, pad_w2], arg11: [dilation_h, dilation_w]
            # arg12: groups, arg14: bias tensor (optional)

            params = {}
            for arg in config:
                if not isinstance(arg, dict):
                    continue
                if "arg3" in arg:
                    params["input_channels"] = int(arg["arg3"]) if isinstance(arg["arg3"], (int, str)) else None
                if "arg4" in arg:
                    params["output_channels"] = int(arg["arg4"]) if isinstance(arg["arg4"], (int, str)) else None
                if "arg5" in arg:
                    params["batch_size"] = int(arg["arg5"]) if isinstance(arg["arg5"], (int, str)) else None
                if "arg6" in arg:
                    params["input_height"] = int(arg["arg6"]) if isinstance(arg["arg6"], (int, str)) else None
                if "arg7" in arg:
                    params["input_width"] = int(arg["arg7"]) if isinstance(arg["arg7"], (int, str)) else None
                if "arg8" in arg:
                    kernel = self._parse_list_from_string(arg["arg8"]) if isinstance(arg["arg8"], str) else arg["arg8"]
                    if kernel and len(kernel) >= 2:
                        params["kernel_height"] = kernel[0]
                        params["kernel_width"] = kernel[1]
                if "arg9" in arg:
                    stride = self._parse_list_from_string(arg["arg9"]) if isinstance(arg["arg9"], str) else arg["arg9"]
                    if stride and len(stride) >= 2:
                        params["stride_h"] = stride[0]
                        params["stride_w"] = stride[1]
                if "arg10" in arg:
                    padding = (
                        self._parse_list_from_string(arg["arg10"]) if isinstance(arg["arg10"], str) else arg["arg10"]
                    )
                    if padding:
                        if len(padding) >= 4:
                            # Format: [pad_h1, pad_h2, pad_w1, pad_w2]
                            params["pad_h"] = padding[0]  # Use first pad value
                            params["pad_w"] = padding[2]  # Use third pad value
                        elif len(padding) >= 2:
                            # Format: [pad_h, pad_w]
                            params["pad_h"] = padding[0]
                            params["pad_w"] = padding[1]
                if "arg11" in arg:
                    dilation = (
                        self._parse_list_from_string(arg["arg11"]) if isinstance(arg["arg11"], str) else arg["arg11"]
                    )
                    if dilation and len(dilation) >= 2:
                        params["dilation_h"] = dilation[0]
                        params["dilation_w"] = dilation[1]
                if "arg12" in arg:
                    params["groups"] = int(arg["arg12"]) if isinstance(arg["arg12"], (int, str)) else None
                if "arg14" in arg and isinstance(arg["arg14"], dict):
                    # Bias tensor exists
                    params["has_bias"] = True

            # Set has_bias to False if not found
            if "has_bias" not in params:
                params["has_bias"] = False

            # Check if we have all required params
            required = [
                "batch_size",
                "output_channels",
                "input_channels",
                "input_height",
                "input_width",
                "kernel_height",
                "kernel_width",
                "stride_h",
                "stride_w",
                "pad_h",
                "pad_w",
                "groups",
                "dilation_h",
                "dilation_w",
            ]
            if all(k in params for k in required):
                return params
            return None
        except Exception as e:
            return None

    def _extract_linear_parameters(self, config: List) -> Optional[Dict]:
        """Extract all parameters for linear operation"""
        try:
            # Linear parameter mapping:
            # arg0: input tensor, arg1: weight tensor, arg2: bias tensor (optional)
            # arg3: transpose_a, arg4: transpose_b

            params = {}

            # Extract transpose flags
            for arg in config:
                if not isinstance(arg, dict):
                    continue
                if "arg3" in arg:
                    params["transpose_a"] = bool(int(arg["arg3"])) if isinstance(arg["arg3"], (int, str)) else False
                if "arg4" in arg:
                    params["transpose_b"] = bool(int(arg["arg4"])) if isinstance(arg["arg4"], (int, str)) else False

            # Extract tensor shapes
            tensor_shapes = []
            for arg in config:
                if isinstance(arg, dict):
                    # Check for direct tensor
                    if "arg1" in arg or "arg2" in arg:
                        for key in ["arg1", "arg2"]:
                            if key in arg and isinstance(arg[key], dict) and "Tensor" in arg[key]:
                                tensor_spec = arg[key]["Tensor"]["tensor_spec"]
                                shape = tensor_spec["logical_shape"]
                                tensor_shapes.append(shape)
                    # Check for UnparsedElement (input tensor)
                    if "UnparsedElement" in arg:
                        tc = self.extract_tensor_config(arg)
                        if tc and hasattr(tc, "shape"):
                            tensor_shapes.insert(0, tc.shape)

            if len(tensor_shapes) >= 2:
                params["input_shape"] = tensor_shapes[0]
                params["weight_shape"] = tensor_shapes[1]
                params["bias_shape"] = tensor_shapes[2] if len(tensor_shapes) >= 3 else None
                params["has_bias"] = len(tensor_shapes) >= 3
                return params

            return None
        except Exception as e:
            return None

    def extract_tensor_config(self, arg_data: Dict) -> Optional[TensorConfig]:
        """Extract tensor configuration from argument data"""
        if not isinstance(arg_data, dict):
            return None

        # Handle UnparsedElement by parsing its element_info string
        if "UnparsedElement" in arg_data:
            unparsed_data = arg_data["UnparsedElement"]
            element_info = unparsed_data.get("element_info", "")

            if element_info and element_info.startswith("{"):
                try:
                    # Apply regex fixes for C++ style formats
                    fixed_json_str = element_info
                    # Fix C++ style braces in values like "{32, 32}" -> "[32, 32]"
                    fixed_json_str = re.sub(r':\s*"{\s*([^}]+)\s*}"', r': "[\1]"', fixed_json_str)
                    # Fix grid format: "grid":{[...], [...]} -> "grid":[[...], [...]]
                    # This handles CoreRangeSet structures with multiple ranges
                    fixed_json_str = re.sub(
                        r'"grid"\s*:\s*\{(\[.*?\](?:\s*,\s*\[.*?\])*)\}', r'"grid":[\1]', fixed_json_str
                    )
                    # Fix grid ranges like {"x":0,"y":0} - {"x":7,"y":7} -> {"x":0,"y":0}, {"x":7,"y":7}
                    # Using a more specific pattern to handle coordinate objects
                    fixed_json_str = re.sub(
                        r'(\{"x":\d+,"y":\d+\})\s*-\s*(\{"x":\d+,"y":\d+\})', r"\1, \2", fixed_json_str
                    )

                    # Parse the fixed JSON
                    parsed_data = json.loads(fixed_json_str)

                    # Extract arg0 (first argument) which contains the tensor
                    for key, value in parsed_data.items():
                        if isinstance(value, dict) and "Tensor" in value:
                            arg_data = value
                            break
                except Exception as e:
                    return None

        # Handle nested structure like {arg0: {Tensor: ...}} or {arg1: {Tensor: ...}}
        # Check if any of the keys are argument names (arg0, arg1, etc.)
        if "Tensor" not in arg_data:
            # Look for nested tensor in argument keys
            for key, value in arg_data.items():
                if key.startswith("arg") and isinstance(value, dict) and "Tensor" in value:
                    arg_data = value
                    break

        if "Tensor" not in arg_data:
            return None

        tensor_data = arg_data["Tensor"]
        tensor_spec = tensor_data.get("tensor_spec", {})
        tensor_layout = tensor_spec.get("tensor_layout", {})

        # Extract shape
        shape = tensor_spec.get("logical_shape", [])
        if not shape:
            return None

        # Extract other properties
        dtype_str = tensor_layout.get("dtype", "DataType::BFLOAT16")
        memory_config = tensor_layout.get("memory_config", {})

        # Determine layout (simplified - would need more logic for accurate detection)
        layout = "TILE"  # Default assumption for most ops

        return TensorConfig(shape=shape, dtype=dtype_str, layout=layout, memory_config=memory_config)

    def convert_to_sweep_parameters(self, operation_name: str, max_configs: int = 50) -> Dict[str, Dict]:
        """Convert master JSON configs to sweep test parameters"""
        configs = self.get_operation_configs(operation_name)

        if not configs:
            return {}

        # Limit number of configurations to avoid overwhelming tests
        configs = configs[:max_configs]

        sweep_params = {"master_configs": {}}

        # Extract unique values for each parameter type
        input_shapes = []
        input_dtypes = []
        input_layouts = []
        input_memory_configs = []
        output_memory_configs = []

        for config in configs:
            try:
                # Parse each argument in the configuration
                for arg in config:
                    for arg_name, arg_data in arg.items():
                        if arg_name.startswith("arg") and isinstance(arg_data, dict):
                            tensor_config = self.extract_tensor_config(arg_data)
                            if tensor_config:
                                # Add to our lists if not already present
                                if tensor_config.shape not in input_shapes:
                                    input_shapes.append(tensor_config.shape)

                                dtype_ttnn = self.parse_dtype(tensor_config.dtype)
                                if dtype_ttnn not in input_dtypes:
                                    input_dtypes.append(dtype_ttnn)

                                layout_ttnn = self.parse_layout(tensor_config.layout)
                                if layout_ttnn not in input_layouts:
                                    input_layouts.append(layout_ttnn)

                                memory_config_ttnn = self.parse_memory_config(tensor_config.memory_config)
                                if memory_config_ttnn not in input_memory_configs:
                                    input_memory_configs.append(memory_config_ttnn)

            except Exception as e:
                print(f"⚠️ Error processing config: {e}")
                continue

        # Create sweep parameters structure
        if input_shapes:
            sweep_params["master_configs"]["input_shapes"] = input_shapes[:20]  # Limit to 20 shapes
        if input_dtypes:
            sweep_params["master_configs"]["input_dtypes"] = input_dtypes
        if input_layouts:
            sweep_params["master_configs"]["input_layouts"] = input_layouts
        if input_memory_configs:
            sweep_params["master_configs"]["input_memory_configs"] = input_memory_configs[:10]  # Limit memory configs

        # Add default output memory configs (could be derived from inputs)
        sweep_params["master_configs"]["output_memory_configs"] = [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG]

        print(f"✅ Generated sweep parameters for {operation_name}:")
        print(f"   • {len(input_shapes)} unique input shapes")
        print(f"   • {len(input_dtypes)} unique dtypes")
        print(f"   • {len(input_layouts)} unique layouts")
        print(f"   • {len(input_memory_configs)} unique memory configs")

        return sweep_params

    def get_raw_configurations(self, operation_name: str, max_configs: int = 10) -> List[Dict]:
        """Get raw configurations for direct use in sweep tests"""
        configs = self.get_operation_configs(operation_name)

        if not configs:
            return []

        # Limit and return raw configs
        return configs[:max_configs]


# Convenience functions for easy integration
def load_master_configs_for_operation(operation_name: str, max_configs: int = 50) -> Dict[str, Dict]:
    """Convenience function to load master configs for an operation"""
    loader = MasterConfigLoader()
    return loader.convert_to_sweep_parameters(operation_name, max_configs)


def get_master_configs_raw(operation_name: str, max_configs: int = 10) -> List[Dict]:
    """Convenience function to get raw master configs for an operation"""
    loader = MasterConfigLoader()
    return loader.get_raw_configurations(operation_name, max_configs)


# Example usage function
def create_master_based_sweep_test(operation_name: str):
    """Example of how to create a sweep test using master configs"""

    # Load master configurations
    master_params = load_master_configs_for_operation(operation_name)

    if not master_params:
        print(f"❌ No master configurations found for {operation_name}")
        return None

    # Create combined parameters (master + some manual ones for comparison)
    parameters = {
        **master_params,
        "manual_suite": {
            "input_shapes": [[1, 32, 32], [1, 1, 1024]],  # Some manual test cases
            "input_dtypes": [ttnn.bfloat16],
            "input_layouts": [ttnn.TILE_LAYOUT],
            "input_memory_configs": [ttnn.DRAM_MEMORY_CONFIG],
            "output_memory_configs": [ttnn.DRAM_MEMORY_CONFIG],
        },
    }

    return parameters


# Global instance for easy access from sweep tests
_global_loader = None


def get_global_loader(instance: MasterConfigLoader = None) -> MasterConfigLoader:
    """
    Get or create the global MasterConfigLoader instance.

    Args:
        instance: If provided, sets this as the global instance

    Returns:
        The global MasterConfigLoader instance
    """
    global _global_loader
    if instance is not None:
        _global_loader = instance
    if _global_loader is None:
        _global_loader = MasterConfigLoader()
    return _global_loader


def get_traced_config(traced_config_name: str) -> Optional[Dict]:
    """
    Convenience function to look up traced config from global loader.
    Use this in your sweep test's run() function.

    Args:
        traced_config_name: String name like "sigmoid_accurate_traced_0"

    Returns:
        Dictionary with 'shape', 'dtype', 'layout', 'memory_config', 'output_memory_config'
        or None if not found

    Example:
        ```python
        def run(traced_config_name=None, ...):
            if traced_config_name:
                cfg = get_traced_config(traced_config_name)
                input_shape = cfg['shape']
                input_a_dtype = cfg['dtype']
                # ... etc
        ```
    """
    return get_global_loader().get_traced_config(traced_config_name)


def unpack_traced_config(
    traced_config_name: str, use_defaults: bool = False
) -> Tuple[Optional[list], Optional[any], Optional[any], Optional[any], Optional[any]]:
    """
    Convenience function to unpack a traced config directly into values.
    This is the SIMPLEST way to use traced configs in UNARY operations.

    Args:
        traced_config_name: String name like "sigmoid_accurate_traced_0"
        use_defaults: If True and config not found, returns default values instead of None

    Returns:
        Tuple of (shape, dtype, layout, input_memory_config, output_memory_config)
        or (None, None, None, None, None) if not found and use_defaults=False
        or default values if not found and use_defaults=True

    Example:
        ```python
        def run(input_shape=None, input_a_dtype=None, ..., traced_config_name=None, *, device):
            # Unpack traced config if provided, otherwise use defaults
            input_shape, input_a_dtype, input_a_layout, input_a_memory_config, output_memory_config = (
                unpack_traced_config(traced_config_name) if traced_config_name
                else (input_shape, input_a_dtype, input_a_layout, input_a_memory_config, output_memory_config)
            )

            # Or even simpler: let defaults be None and handle them later
        ```
    """
    cfg = get_traced_config(traced_config_name)
    if cfg:
        # Check if it's binary - if so, return None to avoid confusion
        if cfg.get("is_binary", False):
            print(f"⚠️ Use unpack_binary_traced_config() for binary operation configs")
            return (None, None, None, None, None)
        return (cfg["shape"], cfg["dtype"], cfg["layout"], cfg["memory_config"], cfg["output_memory_config"])

    if use_defaults:
        # Return sensible defaults
        return (
            [1, 32, 32],  # shape
            ttnn.bfloat16,  # dtype
            ttnn.TILE_LAYOUT,  # layout
            ttnn.DRAM_MEMORY_CONFIG,  # input_memory_config
            ttnn.DRAM_MEMORY_CONFIG,  # output_memory_config
        )

    return (None, None, None, None, None)


def unpack_binary_traced_config(traced_config_name: str, use_defaults: bool = False) -> Tuple:
    """
    Convenience function to unpack a traced config for BINARY operations (like add, multiply).

    Args:
        traced_config_name: String name like "add_traced_0"
        use_defaults: If True and config not found, returns default values instead of None

    Returns:
        Tuple of (input_shape_dict, input_a_dtype, input_b_dtype, input_a_layout, input_b_layout,
                  input_a_memory_config, input_b_memory_config)
        where input_shape_dict is {"self": shape_a, "other": shape_b}

    Example:
        ```python
        def run(input_shape=None, input_a_dtype=None, input_b_dtype=None, ...,
                traced_config_name=None, *, device):
            if traced_config_name:
                input_shape, input_a_dtype, input_b_dtype, input_a_layout, input_b_layout, \\
                    input_a_memory_config, input_b_memory_config = unpack_binary_traced_config(traced_config_name)
        ```
    """
    cfg = get_traced_config(traced_config_name)
    if cfg:
        # Check if it's actually binary
        if not cfg.get("is_binary", False):
            print(f"⚠️ Use unpack_traced_config() for unary operation configs")
            return (None, None, None, None, None, None, None)

        # Return binary config
        input_shape = {"self": cfg["shape_a"], "other": cfg["shape_b"]}
        return (
            input_shape,
            cfg["dtype_a"],
            cfg["dtype_b"],
            cfg["layout_a"],
            cfg["layout_b"],
            cfg["memory_config_a"],
            cfg["memory_config_b"],
        )

    if use_defaults:
        # Return sensible defaults for binary operations
        return (
            {"self": [1, 32, 32], "other": [1, 32, 32]},  # input_shape
            ttnn.bfloat16,  # input_a_dtype
            ttnn.bfloat16,  # input_b_dtype
            ttnn.TILE_LAYOUT,  # input_a_layout
            ttnn.TILE_LAYOUT,  # input_b_layout
            ttnn.DRAM_MEMORY_CONFIG,  # input_a_memory_config
            ttnn.DRAM_MEMORY_CONFIG,  # input_b_memory_config
        )

    return (None, None, None, None, None, None, None)


if __name__ == "__main__":
    # Example usage
    loader = MasterConfigLoader()

    # Test with add operation
    add_params = loader.convert_to_sweep_parameters("add")
    print(f"\n📊 Master-based parameters for 'add': {len(add_params)} suites")

    # Test with transpose operation
    transpose_params = loader.convert_to_sweep_parameters("transpose")
    print(f"\n📊 Master-based parameters for 'transpose': {len(transpose_params)} suites")
