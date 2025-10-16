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

    def __init__(self, master_file_path: str = "/home/ubuntu/tt-metal/traced_operations/ttnn_operations_master.json"):
        self.master_file_path = master_file_path
        self.master_data = None

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

    def extract_tensor_config(self, arg_data: Dict) -> Optional[TensorConfig]:
        """Extract tensor configuration from argument data"""
        if not isinstance(arg_data, dict) or "Tensor" not in arg_data:
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


if __name__ == "__main__":
    # Example usage
    loader = MasterConfigLoader()

    # Test with add operation
    add_params = loader.convert_to_sweep_parameters("add")
    print(f"\n📊 Master-based parameters for 'add': {len(add_params)} suites")

    # Test with transpose operation
    transpose_params = loader.convert_to_sweep_parameters("transpose")
    print(f"\n📊 Master-based parameters for 'transpose': {len(transpose_params)} suites")
