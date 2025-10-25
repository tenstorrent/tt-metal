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
import ttnn
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class TensorConfig:
    """Represents a tensor configuration extracted from master JSON"""

    shape: List[int]
    dtype: str
    layout: str
    memory_config: Dict


class MasterConfigLoader:
    """Loads and converts master JSON configurations to sweep test parameters"""

    def __init__(
        self, master_file_path: str = "/home/ubuntu/tt-metal/model_tracer/traced_operations/ttnn_operations_master.json"
    ):
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

    def _is_binary_operation(self, configs: List) -> bool:
        """
        Detect if an operation is binary (2 tensor inputs) by checking the first config.

        Args:
            configs: List of operation configurations

        Returns:
            True if operation has 2 tensor inputs, False otherwise
        """
        if not configs or len(configs) == 0:
            return False

        # Check first config for number of tensor arguments
        first_config = configs[0]
        tensor_count = 0

        for arg in first_config:
            tensor_config = self.extract_tensor_config(arg)
            if tensor_config:
                tensor_count += 1
                if tensor_count >= 2:
                    return True

        return False

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
            all_cases: If False (default), returns exact traced configs (N tests).
                      If True, returns all combinations as Cartesian product (N×M×... tests).

        Returns:
            Dictionary with all necessary parameters ready to add to test parameters

        Example (Unary):
            # Default: Run exact 30 traced configs
            loader = MasterConfigLoader()
            model_traced_params = loader.get_suite_parameters("sigmoid_accurate")

            # Or: Run all combinations (30 shapes × dtypes × layouts × memory_configs)
            model_traced_params = loader.get_suite_parameters("sigmoid_accurate", all_cases=True)

        Example (Binary):
            # Default: Run exact 6 traced configs for add (paired inputs)
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

            # Detect if this is a binary operation
            is_binary = self._is_binary_operation(configs)

            if is_binary:
                print(f"🔧 Detected binary operation: {operation_name}")
                return self._get_binary_suite_parameters(operation_name, configs, all_cases)
            else:
                print(f"🔧 Detected unary operation: {operation_name}")
                return self._get_unary_suite_parameters(operation_name, configs, all_cases)

        except Exception as e:
            print(f"❌ Error loading configurations for {operation_name}: {e}")
            import traceback

            traceback.print_exc()
            return {"traced_config_name": []}

    def _get_unary_suite_parameters(self, operation_name: str, configs: List, all_cases: bool) -> Dict:
        """
        Get parameters for unary operations (single tensor input).
        """
        # Extract PAIRED configurations (shape + dtype + layout + memory_config)
        # This ensures we get N exact configs, not a Cartesian product
        paired_configs = []
        failed_configs = 0

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
                        paired_configs.append(
                            {
                                "shape": tensor_config.shape,
                                "dtype": parsed_dtype,
                                "layout": parsed_layout,
                                "memory_config": parsed_mem_config,
                                "output_memory_config": parsed_mem_config,  # For unary ops, output matches input
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
            # Store configs in instance cache for lookup
            self.traced_configs_cache[operation_name] = paired_configs

            if all_cases:
                # Return separate lists for Cartesian product
                # Extract unique values for each parameter type
                shapes = []
                dtypes = set()
                layouts = set()
                memory_configs = []
                output_memory_configs = []

                for cfg in paired_configs:
                    shapes.append(cfg["shape"])
                    dtypes.add(cfg["dtype"])
                    layouts.add(cfg["layout"])
                    memory_configs.append(cfg["memory_config"])
                    output_memory_configs.append(cfg["output_memory_config"])

                result = {
                    "input_shape": shapes,
                    "input_a_dtype": list(dtypes),
                    "input_a_layout": list(layouts),
                    "input_a_memory_config": memory_configs,
                    "output_memory_config": output_memory_configs,
                }

                total_tests = (
                    len(shapes) * len(dtypes) * len(layouts) * len(memory_configs) * len(output_memory_configs)
                )
                print(f"✅ Loaded {len(paired_configs)} traced configurations for {operation_name} (model_traced suite)")
                print(f"   📊 all_cases=True: Will generate ~{total_tests} test vectors (Cartesian product)")
                if failed_configs > 0:
                    print(f"⚠️ Failed to parse {failed_configs} configurations")
            else:
                # Return config names for exact paired configs (default)
                traced_config_names = [f"{operation_name}_traced_{i}" for i in range(len(paired_configs))]

                result = {
                    "traced_config_name": traced_config_names,
                }

                print(f"✅ Loaded {len(paired_configs)} traced configurations for {operation_name} (model_traced suite)")
                print(f"   📊 Will generate {len(paired_configs)} test vectors (exact traced configs)")
                if failed_configs > 0:
                    print(f"⚠️ Failed to parse {failed_configs} configurations")

            return result
        else:
            # No configs successfully parsed, return empty
            print(f"⚠️ No configurations could be parsed for {operation_name} (TTNN may not be initialized)")
            return {
                "traced_config": [],
            }

    def _get_binary_suite_parameters(self, operation_name: str, configs: List, all_cases: bool) -> Dict:
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
                shapes_a = []
                shapes_b = []
                dtypes_a = set()
                dtypes_b = set()
                layouts_a = set()
                layouts_b = set()
                memory_configs_a = []
                memory_configs_b = []

                for cfg in paired_configs:
                    shapes_a.append(cfg["shape_a"])
                    shapes_b.append(cfg["shape_b"])
                    dtypes_a.add(cfg["dtype_a"])
                    dtypes_b.add(cfg["dtype_b"])
                    layouts_a.add(cfg["layout_a"])
                    layouts_b.add(cfg["layout_b"])
                    memory_configs_a.append(cfg["memory_config_a"])
                    memory_configs_b.append(cfg["memory_config_b"])

                # For binary operations, input_shape is a dict with "self" and "other"
                input_shapes = [{"self": sa, "other": sb} for sa, sb in zip(shapes_a, shapes_b)]

                result = {
                    "input_shape": input_shapes,
                    "input_a_dtype": list(dtypes_a),
                    "input_b_dtype": list(dtypes_b),
                    "input_a_layout": list(layouts_a),
                    "input_b_layout": list(layouts_b),
                    "input_a_memory_config": memory_configs_a,
                    "input_b_memory_config": memory_configs_b,
                }

                total_tests = (
                    len(input_shapes)
                    * len(dtypes_a)
                    * len(dtypes_b)
                    * len(layouts_a)
                    * len(layouts_b)
                    * len(memory_configs_a)
                    * len(memory_configs_b)
                )
                print(f"✅ Loaded {len(paired_configs)} traced configurations for {operation_name} (model_traced suite)")
                print(f"   📊 all_cases=True: Will generate ~{total_tests} test vectors (Cartesian product)")
                if failed_configs > 0:
                    print(f"⚠️ Failed to parse {failed_configs} configurations")
            else:
                # Return config names for exact paired configs (default)
                traced_config_names = [f"{operation_name}_traced_{i}" for i in range(len(paired_configs))]

                result = {
                    "traced_config_name": traced_config_names,
                }

                print(f"✅ Loaded {len(paired_configs)} traced configurations for {operation_name} (model_traced suite)")
                print(f"   📊 Will generate {len(paired_configs)} test vectors (exact traced configs)")
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
