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
from framework.constants import LEAD_MODELS

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
    storage_type: str = "StorageType::DEVICE"  # Default to DEVICE for backward compatibility


class MasterConfigLoader:
    """Loads and converts master JSON configurations to sweep test parameters

    Class Attributes:
        lead_models_only: When True, filters configurations to only include
            those from lead models (e.g., deepseek_v3). Set via set_lead_models_filter()
            before importing sweep modules that use this loader.
    """

    # Class-level filter setting (replaces environment variable approach)
    # This is set by sweeps_parameter_generator.py before importing sweep modules
    _lead_models_only: bool = False

    @classmethod
    def set_lead_models_filter(cls, enabled: bool) -> None:
        """Set the lead models filter.

        Args:
            enabled: If True, only configurations from lead models will be loaded.
                    If False, all configurations will be loaded.

        Note:
            This must be called BEFORE importing sweep modules that use MasterConfigLoader,
            as the filtering happens at module load time when get_suite_parameters() is called.
        """
        cls._lead_models_only = enabled

    @classmethod
    def get_lead_models_filter(cls) -> bool:
        """Get the current lead models filter setting."""
        return cls._lead_models_only

    @staticmethod
    def _source_matches_lead_models(source) -> bool:
        """Check if source matches any lead model pattern.

        Args:
            source: Either a string path or a list of string paths

        Returns:
            True if any source path contains a lead model pattern
        """
        # Normalize source to a list
        sources = source if isinstance(source, list) else [source]

        for src in sources:
            if not isinstance(src, str):
                continue
            src_lower = src.lower()
            if any(pattern.lower() in src_lower for pattern in LEAD_MODELS):
                return True
        return False

    def _matches_operation(self, operation_name: str, base_name: str) -> bool:
        """Check if operation_name matches any variant of base_name.

        Handles common patterns like:
        - "add" matches ["add", "ttnn::add"]
        - "linear" matches ["linear", "ttnn::linear"]
        - "nlp_create_qkv_heads" matches ["nlp_create_qkv_heads", "experimental::nlp_create_qkv_heads", "ttnn::experimental::nlp_create_qkv_heads"]

        Args:
            operation_name: The operation name to check
            base_name: The base operation name

        Returns:
            True if operation_name is any variant of base_name
        """
        variants = [
            base_name,
            f"ttnn::{base_name}",
            f"experimental::{base_name}",
            f"ttnn::experimental::{base_name}",
            f"transformer::{base_name}",
            f"ttnn::transformer::{base_name}",
        ]
        return operation_name in variants

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
            configs = self.master_data["operations"][operation_name].get("configurations", [])
            # Convert new format (dict with source) to old format (list) for backward compatibility
            return self._normalize_configs(configs)

        # Try with ttnn:: prefix
        ttnn_op_name = f"ttnn::{operation_name}"
        if ttnn_op_name in self.master_data.get("operations", {}):
            configs = self.master_data["operations"][ttnn_op_name].get("configurations", [])
            return self._normalize_configs(configs)

        # Try with ttnn::experimental:: namespace (e.g., ttnn::experimental::create_qkv_heads)
        experimental_full_op_name = f"ttnn::experimental::{operation_name}"
        if experimental_full_op_name in self.master_data.get("operations", {}):
            configs = self.master_data["operations"][experimental_full_op_name].get("configurations", [])
            return self._normalize_configs(configs)

        # Try with experimental:: namespace (e.g., experimental::nlp_concat_heads)
        if operation_name.startswith("experimental::"):
            experimental_op_name = f"ttnn::{operation_name}"
            if experimental_op_name in self.master_data.get("operations", {}):
                configs = self.master_data["operations"][experimental_op_name].get("configurations", [])
                return self._normalize_configs(configs)

        # Try with transformer:: namespace (e.g., transformer::paged_scaled_dot_product_attention_decode)
        transformer_op_name = f"ttnn::transformer::{operation_name}"
        if transformer_op_name in self.master_data.get("operations", {}):
            configs = self.master_data["operations"][transformer_op_name].get("configurations", [])
            return self._normalize_configs(configs)

        # Try without prefix if it starts with ttnn::
        if operation_name.startswith("ttnn::"):
            base_name = operation_name[6:]  # Remove "ttnn::"
            if base_name in self.master_data.get("operations", {}):
                configs = self.master_data["operations"][base_name].get("configurations", [])
                return self._normalize_configs(configs)
            # Also try with transformer:: namespace
            transformer_base = f"ttnn::transformer::{base_name}"
            if transformer_base in self.master_data.get("operations", {}):
                configs = self.master_data["operations"][transformer_base].get("configurations", [])
                return self._normalize_configs(configs)

        print(f"⚠️ No configurations found for operation: {operation_name}")
        return []

    def _normalize_configs(self, configs: List) -> List[Tuple[List[Dict], str]]:
        """
        Normalize configurations to always return list of (argument list, source) tuples.
        Handles both old format (list) and new format (dict with source or contexts).

        Args:
            configs: List of configurations (either list of args or dict with 'arguments' and 'source'/'contexts')

        Returns:
            List of (arguments, source, machine_info) tuples for traceability
        """
        # Check if we should filter for lead models only
        # Uses class-level setting instead of environment variable for cleaner control
        lead_models_only = MasterConfigLoader._lead_models_only

        normalized = []
        for config in configs:
            if isinstance(config, dict) and "arguments" in config:
                # Check if this config has the new contexts format
                if "contexts" in config:
                    # New contexts format: expand each context into separate tuples
                    arguments = config["arguments"]
                    for context in config["contexts"]:
                        # Extract source (should be a list in new format)
                        source_list = context.get("source", ["unknown"])
                        source = source_list[0] if isinstance(source_list, list) and len(source_list) > 0 else "unknown"

                        # Extract machine_info
                        machine_info = context.get("machine_info", None)

                        # Filter for lead models if requested
                        if lead_models_only:
                            if not self._source_matches_lead_models(source_list):
                                continue  # Skip this context

                        normalized.append((arguments, source, machine_info))
                else:
                    # Old single source/machine_info format
                    source = config.get("source", "unknown")
                    machine_info = config.get("machine_info", None)

                    # Filter for lead models if requested
                    if lead_models_only:
                        if not self._source_matches_lead_models(source):
                            continue  # Skip this config

                    normalized.append((config["arguments"], source, machine_info))
            elif isinstance(config, list):
                # Old format: use as-is with unknown source and no machine_info
                # Skip if lead_models_only since we can't determine source
                if not lead_models_only:
                    normalized.append((config, "unknown", None))
            else:
                # Fallback: wrap in list with unknown source and no machine_info
                normalized.append((config if isinstance(config, list) else [config], "unknown", None))
        return normalized

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
        # Handle case where layout is already a Layout object
        if hasattr(layout_str, "__class__") and "Layout" in str(layout_str.__class__):
            return layout_str

        # Handle string case
        layout_str_converted = str(layout_str)
        if "TILE" in layout_str_converted:
            return ttnn.TILE_LAYOUT
        elif "ROW_MAJOR" in layout_str_converted:
            return ttnn.ROW_MAJOR_LAYOUT
        else:
            return ttnn.TILE_LAYOUT  # Default

    def _is_valid_sharding_config(self, memory_config: Dict, tensor_shape: list = None) -> bool:
        """
        Check if a sharding configuration is valid for the current hardware.
        Since traced configs come from real model runs that worked, we trust them as-is.
        No validation is performed - all traced configs are considered valid.

        Args:
            memory_config: Memory config dictionary
            tensor_shape: Tensor shape (optional, for validation)

        Returns:
            True - all traced configs are considered valid
        """
        # Trust traced configs - they come from real model runs that worked
        # No validation needed - use configs directly as requested by user
        return True

    def _is_valid_ttnn_memory_config(self, mem_config, operation_name: str = None) -> bool:
        """
        Check if a TTNN MemoryConfig object is valid for the operation.
        Used to validate parsed memory configs before adding to test parameters.

        Args:
            mem_config: TTNN MemoryConfig object
            operation_name: Name of the operation (for operation-specific checks)

        Returns:
            True if valid, False otherwise
        """
        try:
            if not mem_config:
                return False

            # Trust traced configs - no validation needed
            # Traced configs come from real model runs that worked
            return True
        except Exception:
            return True  # If we can't check, assume valid

    def parse_memory_config(self, memory_config: Dict, tensor_shape: list = None) -> Any:
        """Convert memory config dict to ttnn memory config

        Args:
            memory_config: Memory config dictionary from master JSON
            tensor_shape: Tensor shape needed for creating sharded configs
        """
        try:
            # Validate sharding config before parsing
            if not self._is_valid_sharding_config(memory_config, tensor_shape):
                # Invalid sharding config - return DRAM interleaved as fallback
                return ttnn.DRAM_MEMORY_CONFIG

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

            # Check if shard_spec is nullopt - BLOCK_SHARDED can have nullopt shard_spec (default sharding)
            if shard_spec == "std::nullopt" or not shard_spec:
                # No shard spec - for BLOCK_SHARDED, return config without shard_spec
                if "BLOCK_SHARDED" in memory_layout:
                    # BLOCK_SHARDED without explicit shard_spec is valid (uses default sharding)
                    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.BLOCK_SHARDED, buffer_type_ttnn)
                else:
                    # Other sharded layouts need shard_spec, fall back to interleaved
                    return ttnn.DRAM_MEMORY_CONFIG

            if tensor_shape:
                # Extract shard shape - prefer cleaner array format from nd_shard_spec
                shard_shape = None
                if "shard_shape" in shard_spec:
                    # nd_shard_spec format: "shard_shape": [224, 224]
                    shard_shape_data = shard_spec["shard_shape"]
                    if isinstance(shard_shape_data, list) and len(shard_shape_data) >= 2:
                        shard_shape = shard_shape_data[:2]
                elif "shape" in shard_spec:
                    # Regular shard_spec format: can be "shape": "{224, 224}" or "shape": "[224, 224]"
                    shard_shape_str = shard_spec["shape"]
                    if isinstance(shard_shape_str, str):
                        # Extract all numbers from the string (works for both {} and [] formats)
                        numbers = re.findall(r"\d+", shard_shape_str)
                        if len(numbers) >= 2:
                            shard_shape = [int(numbers[0]), int(numbers[1])]
                    elif isinstance(shard_shape_str, list) and len(shard_shape_str) >= 2:
                        # Shape is already a list
                        shard_shape = shard_shape_str[:2]

                # Use shard shape directly from traced config - no validation or adjustment
                # Traced configs come from real model runs that worked, so use them as-is
                # shard_shape is already extracted above, use it directly

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
                            # Grid can be in three formats:
                            # 1. Simple: [{"x":0,"y":0}, {"x":7,"y":7}] - direct coordinates
                            # 2. Complex: [[{"x":0,"y":0}, {"x":7,"y":5}], [...]] - multiple ranges
                            # 3. CoreRange: [{"start": {"x":0,"y":0}, "end": {"x":7,"y":3}}] - CoreRange format

                            # Check for CoreRange format first (has "start" and "end" keys)
                            if isinstance(grid_data[0], dict) and "start" in grid_data[0] and "end" in grid_data[0]:
                                # CoreRange format
                                start_coords = grid_data[0].get("start", {})
                                end_coords = grid_data[0].get("end", {})
                            # Check if it's a complex grid (list of lists)
                            elif isinstance(grid_data[0], list):
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

                                # Note: We don't validate coordinates here - let TTNN fail naturally
                                # with a clear error message if coordinates exceed hardware limits.
                                # This provides better debugging information than silently falling back.

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

                        if core_grid and shard_shape and isinstance(grid_data, list) and len(grid_data) > 0:
                            # Create ShardSpec manually using the EXACT shard_shape from traced data
                            # Don't let TTNN calculate it - use the actual traced values!
                            from ttnn import CoreCoord, CoreRange, CoreRangeSet

                            # Build CoreRangeSet from grid_data (handle all three formats)
                            core_ranges = []
                            if isinstance(grid_data[0], dict) and "start" in grid_data[0] and "end" in grid_data[0]:
                                # CoreRange format: [{"start": {...}, "end": {...}}]
                                for range_obj in grid_data:
                                    if isinstance(range_obj, dict) and "start" in range_obj and "end" in range_obj:
                                        start_coords = range_obj["start"]
                                        end_coords = range_obj["end"]
                                        if isinstance(start_coords, dict) and isinstance(end_coords, dict):
                                            start = CoreCoord(start_coords["x"], start_coords["y"])
                                            end = CoreCoord(end_coords["x"], end_coords["y"])
                                            core_ranges.append(CoreRange(start, end))
                            elif isinstance(grid_data[0], list):
                                # Complex multi-range grid: [[{...}, {...}], ...]
                                for range_pair in grid_data:
                                    if (
                                        len(range_pair) >= 2
                                        and isinstance(range_pair[0], dict)
                                        and isinstance(range_pair[1], dict)
                                    ):
                                        start = CoreCoord(range_pair[0]["x"], range_pair[0]["y"])
                                        end = CoreCoord(range_pair[1]["x"], range_pair[1]["y"])
                                        core_ranges.append(CoreRange(start, end))
                            elif len(grid_data) >= 2:
                                # Simple single-range grid: [{...}, {...}]
                                if isinstance(grid_data[0], dict) and isinstance(grid_data[1], dict):
                                    start = CoreCoord(grid_data[0]["x"], grid_data[0]["y"])
                                    end = CoreCoord(grid_data[1]["x"], grid_data[1]["y"])
                                    core_ranges.append(CoreRange(start, end))

                            # Only create CoreRangeSet if we have valid core_ranges
                            if not core_ranges:
                                raise ValueError("Could not parse core ranges from grid_data")

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
        Only counts consecutive tensor arguments from the start, before the first non-tensor argument.
        This prevents counting output pre-allocation tensors that appear later in the argument list.

        Args:
            configs: List of operation configurations (list of (arguments, source) tuples)

        Returns:
            Number of tensor inputs (0, 1, 2, 3, etc.)
        """
        if not configs or len(configs) == 0:
            return 0

        # Check first config for number of tensor arguments
        # configs is a list of (arguments, source, machine_info) tuples
        first_config_args, first_source, first_machine_info = configs[0]
        tensor_count = 0

        # Only count consecutive tensors from the start
        # Stop when we hit the first non-tensor (this prevents counting output tensors)
        for arg in first_config_args:
            tensor_config = self.extract_tensor_config(arg)
            if tensor_config:
                tensor_count += 1
            else:
                # Stop at first non-tensor argument
                break

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
            if self._matches_operation(operation_name, "conv2d"):
                print(f"🔧 Detected conv2d operation with special parameter structure")
                return self._get_conv2d_suite_parameters(operation_name, configs, all_cases, deduplicate_inputs=False)
            elif self._matches_operation(operation_name, "linear"):
                print(f"🔧 Detected linear operation with special parameter structure")
                return self._get_operation_suite_parameters(
                    operation_name, configs, all_cases, deduplicate_inputs=False
                )
            elif self._matches_operation(operation_name, "embedding"):
                print(f"🔧 Detected embedding operation with special parameter structure")
                return self._get_operation_suite_parameters(
                    operation_name, configs, all_cases, deduplicate_inputs=False
                )
            elif self._matches_operation(operation_name, "concat"):
                print(f"🔧 Detected concat operation with vector of tensors input")
                return self._get_concat_suite_parameters(operation_name, configs, all_cases, deduplicate_inputs=False)
            elif self._matches_operation(operation_name, "nlp_create_qkv_heads"):
                print(f"🔧 Detected nlp_create_qkv_heads operation - extracting num_q_heads and num_kv_heads")
                return self._get_nlp_create_qkv_heads_suite_parameters(
                    operation_name, configs, all_cases, deduplicate_inputs=False
                )
            elif self._matches_operation(operation_name, "nlp_create_qkv_heads_decode"):
                print(f"🔧 Detected nlp_create_qkv_heads_decode operation - extracting num_heads and num_kv_heads")
                return self._get_nlp_create_qkv_heads_decode_suite_parameters(
                    operation_name, configs, all_cases, deduplicate_inputs=False
                )
            elif self._matches_operation(operation_name, "create_qkv_heads"):
                print(f"🔧 Detected create_qkv_heads operation - extracting num_heads and num_kv_heads")
                return self._get_create_qkv_heads_suite_parameters(
                    operation_name, configs, all_cases, deduplicate_inputs=False
                )
            elif self._matches_operation(operation_name, "paged_scaled_dot_product_attention_decode"):
                print(
                    f"🔧 Detected paged_scaled_dot_product_attention_decode operation - using operation-specific extractor"
                )
                return self._get_operation_suite_parameters(
                    operation_name, configs, all_cases, deduplicate_inputs=False
                )
            elif self._matches_operation(operation_name, "scaled_dot_product_attention_decode"):
                print(f"🔧 Detected scaled_dot_product_attention_decode operation - using operation-specific extractor")
                return self._get_operation_suite_parameters(
                    operation_name, configs, all_cases, deduplicate_inputs=False
                )
            elif self._matches_operation(operation_name, "fill"):
                print(f"🔧 Detected fill operation - extracting fill_value parameter")
                return self._get_fill_suite_parameters(operation_name, configs, all_cases, deduplicate_inputs=False)
            elif self._matches_operation(operation_name, "split"):
                print(f"🔧 Detected split operation - extracting split_size and dim parameters")
                return self._get_split_suite_parameters(operation_name, configs, all_cases, deduplicate_inputs=False)
            elif self._matches_operation(operation_name, "scatter"):
                print(f"🔧 Detected scatter operation - extracting dim parameter")
                return self._get_scatter_suite_parameters(operation_name, configs, all_cases, deduplicate_inputs=False)
            elif self._matches_operation(operation_name, "attention_softmax_"):
                print(f"🔧 Detected attention_softmax_ operation - extracting head_size parameter")
                return self._get_attention_softmax_suite_parameters(
                    operation_name, configs, all_cases, deduplicate_inputs=False
                )
            elif self._matches_operation(operation_name, "fast_reduce_nc"):
                print(f"🔧 Detected fast_reduce_nc operation - extracting dims parameter")
                return self._get_fast_reduce_nc_suite_parameters(
                    operation_name, configs, all_cases, deduplicate_inputs=False
                )
            elif self._matches_operation(operation_name, "repeat"):
                print(f"🔧 Detected repeat operation - extracting repeat vector (no deduplication)")
                return self._get_unary_suite_parameters(operation_name, configs, all_cases, deduplicate_inputs=False)
            elif self._matches_operation(
                operation_name, "scaled_dot_product_attention"
            ) and not self._matches_operation(operation_name, "decode"):
                print(
                    f"🔧 Detected scaled_dot_product_attention operation - using operation-specific extractor with is_causal and scale"
                )
                return self._get_scaled_dot_product_attention_suite_parameters(
                    operation_name, configs, all_cases, deduplicate_inputs=False
                )
            elif self._matches_operation(operation_name, "paged_update_cache"):
                print(
                    f"🔧 Detected paged_update_cache operation (multi-input with non-consecutive tensors) - using operation-specific extractor"
                )
                return self._get_operation_suite_parameters(
                    operation_name, configs, all_cases, deduplicate_inputs=False
                )

            # Detect the number of tensor inputs
            tensor_count = self._count_tensor_inputs(configs)

            # Disable deduplication for model_traced - master JSON has already deduplicated
            # and each config has a unique config_id to prevent vector hash collisions
            deduplicate_inputs = False

            if tensor_count == 0:
                print(
                    f"⚠️  No tensor inputs detected for {operation_name} - operation may have non-standard parameters"
                )
                print(f"    Treating as unary operation with first argument as input")
                return self._get_unary_suite_parameters(operation_name, configs, all_cases, deduplicate_inputs)
            elif tensor_count == 1:
                # Special case: scale_mask_softmax_in_place has 1 tensor + scale + optional mask
                if self._matches_operation(operation_name, "scale_mask_softmax_in_place"):
                    print(
                        f"🔧 Detected scale_mask_softmax_in_place operation: {operation_name} (1 tensor input + scale + optional mask)"
                    )
                    return self._get_operation_suite_parameters(operation_name, configs, all_cases, deduplicate_inputs)
                # Special case: permute has registered extractor for dims parameter
                elif self._matches_operation(operation_name, "permute"):
                    print(f"🔧 Detected permute operation: {operation_name} (1 tensor input + dims parameter)")
                    # Use generic unary path which will call the registered extractor
                    # Enable deduplication for permute since dims parameter is part of the signature
                    return self._get_unary_suite_parameters(operation_name, configs, all_cases, deduplicate_inputs=True)
                print(f"🔧 Detected unary operation: {operation_name} (1 tensor input)")
                return self._get_unary_suite_parameters(operation_name, configs, all_cases, deduplicate_inputs)
            elif tensor_count == 2:
                # Special case: update_cache has 2 tensors + 2 scalars, needs custom extraction
                if self._matches_operation(operation_name, "update_cache"):
                    print(f"🔧 Detected update_cache operation: {operation_name} (2 tensor inputs + scalars)")
                    return self._get_operation_suite_parameters(operation_name, configs, all_cases, deduplicate_inputs)
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

        for config_idx, (config, source, machine_info) in enumerate(configs):
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
                        # Hardcode specific operation requirements
                        # tilize and tilize_with_val_padding: JSON doesn't have layout field, but these ops require ROW_MAJOR_LAYOUT
                        if self._matches_operation(operation_name, "tilize") or self._matches_operation(
                            operation_name, "tilize_with_val_padding"
                        ):
                            parsed_layout = ttnn.ROW_MAJOR_LAYOUT

                        # pad: If padding has front padding (non-zero first element), use ROW_MAJOR layout
                        # (TILE layout doesn't support front padding, but ROW_MAJOR does)
                        # IMPORTANT: Pad has two formats:
                        # 1. padding format: arg1 is nested list like [[0,0], [0,13], [0,0], [0,0]]
                        # 2. output_padded_shape format: arg1 is flat list like [1, 96, 32, 64] (output shape)
                        # We need to detect which format and only check front padding for format 1
                        if self._matches_operation(operation_name, "pad"):
                            # Extract arg1 from config to determine format
                            arg1_parsed = None
                            for arg in config:
                                if isinstance(arg, dict) and "arg1" in arg:
                                    arg1_str = arg["arg1"]
                                    # Parse arg1 string/list
                                    if isinstance(arg1_str, str):
                                        try:
                                            import ast

                                            arg1_parsed = ast.literal_eval(arg1_str)
                                        except Exception:
                                            arg1_parsed = OperationParameterExtractors._parse_list_from_string(arg1_str)
                                    elif isinstance(arg1_str, list):
                                        arg1_parsed = arg1_str
                                    break

                            # Determine format: nested list = padding format, flat 4-element list = output_padded_shape format
                            is_padding_format = False
                            if arg1_parsed and isinstance(arg1_parsed, list):
                                if len(arg1_parsed) > 0 and isinstance(arg1_parsed[0], (list, tuple)):
                                    # Nested list - this is padding format
                                    is_padding_format = True

                            # Only check for front padding if using padding format
                            if is_padding_format and arg1_parsed:
                                has_front_padding = False
                                for dim_pad in arg1_parsed:
                                    if isinstance(dim_pad, (list, tuple)) and len(dim_pad) >= 1:
                                        if dim_pad[0] != 0:  # Front padding is non-zero
                                            has_front_padding = True
                                            break

                                if has_front_padding:
                                    parsed_layout = ttnn.ROW_MAJOR_LAYOUT

                        # upsample: C++ code requires INTERLEAVED memory layout (see upsample_op.cpp:22-23)
                        # Also, if shape is not tile-aligned, use ROW_MAJOR layout (TILE layout requires tile-aligned shapes)
                        if self._matches_operation(operation_name, "upsample"):
                            parsed_mem_config = ttnn.DRAM_MEMORY_CONFIG  # INTERLEAVED DRAM

                            # Check if shape is tile-aligned
                            if (
                                tensor_config.shape
                                and isinstance(tensor_config.shape, list)
                                and len(tensor_config.shape) >= 4
                            ):
                                h, w = tensor_config.shape[1], tensor_config.shape[2]
                                if h % 32 != 0 or w % 32 != 0:
                                    # Shape is not tile-aligned, use ROW_MAJOR layout
                                    parsed_layout = ttnn.ROW_MAJOR_LAYOUT

                        # Determine output memory config based on operation
                        # First, try to extract output memory config from arg1
                        output_mem_config = None

                        # Extract from arg1 for operations that have output memory config in arg1
                        # (interleaved_to_sharded, nlp_concat_heads, etc.)
                        for arg in config:
                            if isinstance(arg, dict) and "arg1" in arg:
                                if isinstance(arg["arg1"], dict) and "MemoryConfig" in arg["arg1"]:
                                    try:
                                        output_mem_config = self.parse_memory_config(
                                            arg["arg1"]["MemoryConfig"], tensor_config.shape
                                        )
                                        break
                                    except Exception as e:
                                        # If parsing fails, continue to next arg or use default
                                        pass

                        # If not extracted from arg1, use operation-specific defaults
                        if output_mem_config is None:
                            if operation_name == "sharded_to_interleaved":
                                # This operation converts sharded to interleaved, so output must be INTERLEAVED
                                output_mem_config = ttnn.DRAM_MEMORY_CONFIG  # Interleaved DRAM
                            elif self._matches_operation(operation_name, "upsample"):
                                # upsample output also needs INTERLEAVED
                                output_mem_config = ttnn.DRAM_MEMORY_CONFIG
                            elif self._matches_operation(operation_name, "untilize_with_unpadding"):
                                # untilize_with_unpadding: Output memory config must be INTERLEAVED for block sharded input
                                # (see untilize_with_unpadding_op.cpp:37)
                                if parsed_mem_config.memory_layout in [
                                    ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                                    ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                                    ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                                ]:
                                    output_mem_config = ttnn.DRAM_MEMORY_CONFIG  # INTERLEAVED DRAM
                                else:
                                    output_mem_config = parsed_mem_config
                            else:
                                # For most unary ops (including nlp_concat_heads), output matches input
                                # unless explicitly specified in arg1 (which is checked above)
                                output_mem_config = parsed_mem_config

                        # Extract storage_type from tensor_config
                        storage_type_str = (
                            tensor_config.storage_type
                            if hasattr(tensor_config, "storage_type")
                            else "StorageType::DEVICE"
                        )

                        config_dict = {
                            "shape": tensor_config.shape,
                            "dtype": tensor_config.dtype,  # Store the string, not the parsed object
                            "layout": parsed_layout,
                            "memory_config": parsed_mem_config,
                            "output_memory_config": output_mem_config,
                            "storage_type": storage_type_str,
                            "traced_source": source,
                            "traced_machine_info": machine_info,
                        }

                        # Extract operation-specific parameters using registry extractors
                        clean_op_name = operation_name.replace("ttnn::", "")
                        op_params = OperationParameterExtractors.extract_parameters(clean_op_name, config)
                        if not op_params:
                            # Try with full operation name
                            op_params = OperationParameterExtractors.extract_parameters(operation_name, config)

                        if op_params:
                            # Merge extracted parameters into config_dict
                            config_dict.update(op_params)

                            # Special handling for permute - validate and fix dims
                            if (
                                operation_name == "permute" or operation_name == "ttnn::permute"
                            ) and "dims" in op_params:
                                dims = op_params["dims"]
                                shape = config_dict.get("shape", [])
                                if isinstance(shape, list):
                                    ndim = len(shape)
                                    # If dims is None or mismatched, use identity permutation
                                    if dims is None or not isinstance(dims, list) or len(dims) != ndim:
                                        config_dict["dims"] = list(range(ndim))

                            # Special handling for reshape - if validation failed, skip this config
                            if operation_name == "reshape" and "target_shape" not in op_params:
                                failed_configs += 1
                                continue

                        # If deduplicating, check if we've seen this config before
                        # For reshape, include target_shape in deduplication signature
                        # because each (input_shape, target_shape) pair is unique
                        if deduplicate_inputs:
                            import hashlib

                            if operation_name == "reshape" and "target_shape" in config_dict:
                                # For reshape, deduplicate based on (input, target_shape) pair
                                target_shape = config_dict["target_shape"]
                                input_sig = hashlib.md5(
                                    str(
                                        (
                                            tensor_config.shape,
                                            parsed_dtype,
                                            parsed_layout,
                                            parsed_mem_config,
                                            target_shape,
                                        )
                                    ).encode()
                                ).hexdigest()
                            elif self._matches_operation(operation_name, "repeat") and "repeat_shape" in config_dict:
                                # For repeat, deduplicate based on (input, repeat_shape) pair
                                repeat_shape = config_dict["repeat_shape"]
                                input_sig = hashlib.md5(
                                    str(
                                        (
                                            tensor_config.shape,
                                            parsed_dtype,
                                            parsed_layout,
                                            parsed_mem_config,
                                            repeat_shape,
                                        )
                                    ).encode()
                                ).hexdigest()
                            elif (
                                operation_name == "permute" or operation_name == "ttnn::permute"
                            ) and "dims" in config_dict:
                                # For permute, deduplicate based on (input, dims) pair
                                dims = config_dict["dims"]
                                input_sig = hashlib.md5(
                                    str(
                                        (
                                            tensor_config.shape,
                                            parsed_dtype,
                                            parsed_layout,
                                            parsed_mem_config,
                                            dims,
                                        )
                                    ).encode()
                                ).hexdigest()
                            else:
                                # For other operations, deduplicate based on input signature only
                                input_sig = hashlib.md5(
                                    str((tensor_config.shape, parsed_dtype, parsed_layout, parsed_mem_config)).encode()
                                ).hexdigest()

                            if input_sig in seen_input_signatures:
                                continue  # Skip this config, we already have one with this signature
                            seen_input_signatures.add(input_sig)

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
                        # Convert to tuple so it serializes as a string for proper deserialization
                        unique_shapes.append(shape_tuple)
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
                storage_types = []
                traced_source_list = []
                traced_machine_info_list = []
                traced_config_names = []
                config_ids = []  # Unique IDs to prevent hash collisions
                dims_list = [] if (operation_name == "permute" or operation_name == "ttnn::permute") else None
                end_shape_list = [] if operation_name == "untilize_with_unpadding" else None
                dim0_list = [] if operation_name == "transpose" else None
                dim1_list = [] if operation_name == "transpose" else None
                target_shape_list = [] if operation_name == "reshape" else None
                # pad specific parameters (support both padding and output_padded_shape formats)
                padding_list = [] if self._matches_operation(operation_name, "pad") else None
                output_padded_shape_list = [] if self._matches_operation(operation_name, "pad") else None
                input_tensor_start_list = [] if self._matches_operation(operation_name, "pad") else None
                value_list = [] if self._matches_operation(operation_name, "pad") else None
                padded_shape_list = [] if operation_name == "tilize_with_val_padding" else None
                pad_value_list = [] if operation_name == "tilize_with_val_padding" else None
                num_heads_list = (
                    []
                    if operation_name
                    in [
                        "nlp_concat_heads_decode",
                        "experimental::nlp_concat_heads_decode",
                        "ttnn::experimental::nlp_concat_heads_decode",
                        "split_query_key_value_and_split_heads",
                        "experimental::split_query_key_value_and_split_heads",
                        "ttnn::experimental::split_query_key_value_and_split_heads",
                    ]
                    else None
                )
                kv_input_height_list = (
                    []
                    if operation_name
                    in [
                        "split_query_key_value_and_split_heads",
                        "experimental::split_query_key_value_and_split_heads",
                        "ttnn::experimental::split_query_key_value_and_split_heads",
                    ]
                    else None
                )
                # max_pool2d specific parameters
                batch_size_list = [] if self._matches_operation(operation_name, "max_pool2d") else None
                input_h_list = [] if self._matches_operation(operation_name, "max_pool2d") else None
                input_w_list = [] if self._matches_operation(operation_name, "max_pool2d") else None
                channels_list = [] if self._matches_operation(operation_name, "max_pool2d") else None
                kernel_size_list = [] if self._matches_operation(operation_name, "max_pool2d") else None
                stride_list = [] if self._matches_operation(operation_name, "max_pool2d") else None
                padding_list_maxpool = [] if self._matches_operation(operation_name, "max_pool2d") else None
                dilation_list = [] if self._matches_operation(operation_name, "max_pool2d") else None
                applied_shard_scheme_list = [] if self._matches_operation(operation_name, "max_pool2d") else None
                # upsample specific parameters
                scale_factor_list = [] if self._matches_operation(operation_name, "upsample") else None
                mode_list = [] if self._matches_operation(operation_name, "upsample") else None
                # typecast specific parameters
                output_dtype_list = [] if self._matches_operation(operation_name, "typecast") else None
                # gt specific parameters
                scalar_list = [] if self._matches_operation(operation_name, "gt") else None
                # where specific parameters
                scalar_if_true_list = [] if self._matches_operation(operation_name, "where") else None
                scalar_if_false_list = [] if self._matches_operation(operation_name, "where") else None
                # multiply_ specific parameters (scalar multiply)
                scalar_value_list = [] if self._matches_operation(operation_name, "multiply_") else None
                # repeat specific parameters
                repeat_shape_list = [] if self._matches_operation(operation_name, "repeat") else None
                # New operation parameters
                exponent_list = [] if self._matches_operation(operation_name, "pow") else None
                min_list = [] if self._matches_operation(operation_name, "clamp") else None
                max_list = [] if self._matches_operation(operation_name, "clamp") else None
                # rms_norm specific parameters
                program_config_list = (
                    []
                    if (
                        operation_name
                        in [
                            "rms_norm_pre_all_gather",
                            "ttnn::rms_norm_pre_all_gather",
                            "rms_norm_post_all_gather",
                            "ttnn::rms_norm_post_all_gather",
                        ]
                    )
                    else None
                )
                # dim parameter is used by multiple operations
                dim_list = (
                    []
                    if (
                        self._matches_operation(operation_name, "argmax")
                        or self._matches_operation(operation_name, "sum")
                        or self._matches_operation(operation_name, "std")
                        or self._matches_operation(operation_name, "softmax")
                    )
                    else None
                )
                # group_norm parameters (all 16 traced arguments)
                num_groups_list = [] if self._matches_operation(operation_name, "group_norm") else None
                epsilon_list = [] if self._matches_operation(operation_name, "group_norm") else None
                input_mask_shape_list = [] if self._matches_operation(operation_name, "group_norm") else None
                input_mask_dtype_list = [] if self._matches_operation(operation_name, "group_norm") else None
                input_mask_layout_list = [] if self._matches_operation(operation_name, "group_norm") else None
                input_mask_memory_config_list = [] if self._matches_operation(operation_name, "group_norm") else None
                weight_shape_list = [] if self._matches_operation(operation_name, "group_norm") else None
                weight_dtype_list = [] if self._matches_operation(operation_name, "group_norm") else None
                weight_layout_list = [] if self._matches_operation(operation_name, "group_norm") else None
                weight_memory_config_list = [] if self._matches_operation(operation_name, "group_norm") else None
                bias_shape_list = [] if self._matches_operation(operation_name, "group_norm") else None
                bias_dtype_list = [] if self._matches_operation(operation_name, "group_norm") else None
                bias_layout_list = [] if self._matches_operation(operation_name, "group_norm") else None
                bias_memory_config_list = [] if self._matches_operation(operation_name, "group_norm") else None
                reciprocals_shape_list = [] if self._matches_operation(operation_name, "group_norm") else None
                reciprocals_dtype_list = [] if self._matches_operation(operation_name, "group_norm") else None
                reciprocals_layout_list = [] if self._matches_operation(operation_name, "group_norm") else None
                reciprocals_memory_config_list = [] if self._matches_operation(operation_name, "group_norm") else None
                inplace_list = [] if self._matches_operation(operation_name, "group_norm") else None
                num_out_blocks_list = [] if self._matches_operation(operation_name, "group_norm") else None
                use_welford_list = [] if self._matches_operation(operation_name, "group_norm") else None

                invalid_configs = []
                for idx, cfg in enumerate(paired_configs):
                    # For reshape, only include configs that have target_shape
                    # Note: We don't filter based on element count matching here because
                    # the extractor already validates this, and configs from real models are valid
                    if operation_name == "reshape":
                        if "target_shape" not in cfg:
                            continue  # Skip configs without target_shape

                    # For pad, only include configs that have pad parameters
                    # This ensures alignment with the pad parameter lists built later
                    if operation_name == "pad" or operation_name == "ttnn::pad":
                        has_padding = "padding" in cfg and "value" in cfg
                        has_output_format = (
                            "output_padded_shape" in cfg and "input_tensor_start" in cfg and "value" in cfg
                        )
                        if not (has_padding or has_output_format):
                            continue  # Skip configs without pad parameters

                    # Validate and report invalid configs (but don't filter - let them fail)
                    mem_config = cfg.get("memory_config")
                    output_mem_config = cfg.get("output_memory_config")
                    shape = cfg.get("shape", [])
                    layout = cfg.get("layout")

                    # Check for invalid shard specs (too many cores, non-tile-aligned shard shapes)
                    invalid_reasons = []

                    # Trust traced configs - no validation needed
                    # Traced configs come from real model runs that worked
                    # Input and output memory configs are used directly from config

                    # Check operation-specific requirements (report but don't convert)
                    # Note: tilize and upsample are hardcoded above, so these checks are just for reporting
                    if self._matches_operation(operation_name, "upsample"):
                        if isinstance(shape, list) and len(shape) >= 4:
                            h, w = shape[1], shape[2]
                            if h % 32 != 0 or w % 32 != 0:
                                invalid_reasons.append(f"shape H={h}, W={w} not tile-aligned (must be multiples of 32)")

                    # Report invalid configs
                    if invalid_reasons:
                        invalid_configs.append({"index": idx, "shape": shape, "reasons": invalid_reasons})

                    # Convert shape to tuple so it serializes as a string for proper deserialization
                    input_shapes.append(tuple(cfg["shape"]))
                    # Parse dtype/layout strings to ttnn objects
                    parsed_dtype = self.parse_dtype(cfg["dtype"])
                    parsed_layout = self.parse_layout(cfg["layout"])

                    # Override UINT16 TILE to ROW_MAJOR for reshape to avoid device operation assertion
                    # The reshape device operation doesn't support UINT16, but ROW_MAJOR uses view path
                    if (
                        operation_name == "reshape"
                        and parsed_dtype == ttnn.uint16
                        and parsed_layout == ttnn.TILE_LAYOUT
                    ):
                        parsed_layout = ttnn.ROW_MAJOR_LAYOUT

                    input_a_dtypes.append(parsed_dtype)
                    input_a_layouts.append(parsed_layout)
                    # Parse memory configs to ttnn objects (for proper serialization)
                    # Check if already a MemoryConfig object (from some extractors)
                    mem_config = cfg["memory_config"]
                    if isinstance(mem_config, dict):
                        mem_config = self.parse_memory_config(mem_config, cfg["shape"])
                    input_a_memory_configs.append(mem_config)

                    out_mem_config = cfg["output_memory_config"]
                    if isinstance(out_mem_config, dict):
                        out_mem_config = self.parse_memory_config(out_mem_config, cfg["shape"])
                    output_memory_configs.append(out_mem_config)
                    storage_types.append(cfg.get("storage_type", "StorageType::DEVICE"))
                    traced_source_list.append(cfg.get("traced_source", "unknown"))
                    traced_machine_info_list.append(cfg.get("traced_machine_info", None))
                    traced_config_names.append(f"{operation_name}_traced_{idx}")
                    config_ids.append(f"config_{idx}")  # Unique ID to prevent hash collisions
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
                    if self._matches_operation(operation_name, "repeat") and "repeat_shape" in cfg:
                        repeat_shape_list.append(cfg["repeat_shape"])
                    if operation_name in [
                        "rms_norm_pre_all_gather",
                        "ttnn::rms_norm_pre_all_gather",
                        "rms_norm_post_all_gather",
                        "ttnn::rms_norm_post_all_gather",
                    ]:
                        if "program_config" in cfg:
                            program_config_list.append(cfg["program_config"])
                        else:
                            program_config_list.append(None)
                    if operation_name == "pad" or operation_name == "ttnn::pad":
                        if "padding" in cfg and "value" in cfg:
                            # Using padding format
                            padding_list.append(cfg["padding"])
                            value_list.append(cfg["value"])
                        elif "output_padded_shape" in cfg and "input_tensor_start" in cfg and "value" in cfg:
                            # Using output_padded_shape format
                            output_padded_shape_list.append(cfg["output_padded_shape"])
                            input_tensor_start_list.append(cfg["input_tensor_start"])
                            value_list.append(cfg["value"])
                        # Note: If neither format matches, this config will be skipped (no pad params added)
                    if operation_name == "tilize_with_val_padding":
                        if "padded_shape" in cfg and "pad_value" in cfg:
                            padded_shape_list.append(cfg["padded_shape"])
                            pad_value_list.append(cfg["pad_value"])
                    if (
                        operation_name
                        in [
                            "nlp_concat_heads_decode",
                            "experimental::nlp_concat_heads_decode",
                            "ttnn::experimental::nlp_concat_heads_decode",
                        ]
                        and "num_heads" in cfg
                    ):
                        num_heads_list.append(cfg["num_heads"])
                    # Extract split_query_key_value_and_split_heads parameters
                    if operation_name in [
                        "split_query_key_value_and_split_heads",
                        "experimental::split_query_key_value_and_split_heads",
                        "ttnn::experimental::split_query_key_value_and_split_heads",
                    ]:
                        if "num_heads" in cfg:
                            num_heads_list.append(cfg["num_heads"])
                        if "kv_input_height" in cfg:
                            kv_input_height_list.append(cfg["kv_input_height"])
                        else:
                            # Default value if not extracted (will be None)
                            kv_input_height_list.append(None)
                    # Extract max_pool2d parameters
                    if self._matches_operation(operation_name, "max_pool2d"):
                        if "batch_size" in cfg:
                            batch_size_list.append(cfg["batch_size"])
                        if "input_h" in cfg:
                            input_h_list.append(cfg["input_h"])
                        if "input_w" in cfg:
                            input_w_list.append(cfg["input_w"])
                        if "channels" in cfg:
                            channels_list.append(cfg["channels"])
                        if "kernel_size" in cfg:
                            kernel_size_list.append(cfg["kernel_size"])
                        if "stride" in cfg:
                            stride_list.append(cfg["stride"])
                        if "padding" in cfg:
                            padding_list_maxpool.append(cfg["padding"])
                        if "dilation" in cfg:
                            dilation_list.append(cfg["dilation"])
                        if "applied_shard_scheme" in cfg:
                            applied_shard_scheme_list.append(cfg["applied_shard_scheme"])
                    # Extract upsample parameters
                    if self._matches_operation(operation_name, "upsample"):
                        if "scale_factor" in cfg:
                            scale_factor_list.append(cfg["scale_factor"])
                        if "mode" in cfg:
                            mode_list.append(cfg["mode"])
                    # Extract typecast parameters
                    if self._matches_operation(operation_name, "typecast"):
                        if "output_dtype" in cfg:
                            # Parse output_dtype string to TTNN dtype
                            output_dtype_str = cfg["output_dtype"]
                            parsed_output_dtype = self.parse_dtype(f"DataType::{output_dtype_str}")
                            if parsed_output_dtype:
                                output_dtype_list.append(parsed_output_dtype)
                    # Extract permute parameters (already extracted above in main loop)
                    # Extract gt parameters
                    if self._matches_operation(operation_name, "gt"):
                        if "scalar" in cfg:
                            scalar_list.append(cfg["scalar"])
                    # Extract where parameters
                    if self._matches_operation(operation_name, "where"):
                        if "scalar_if_true" in cfg:
                            scalar_if_true_list.append(cfg["scalar_if_true"])
                        if "scalar_if_false" in cfg:
                            scalar_if_false_list.append(cfg["scalar_if_false"])
                    # Extract multiply_ parameters (scalar value)
                    if self._matches_operation(operation_name, "multiply_"):
                        if "scalar_value" in cfg:
                            scalar_value_list.append(cfg["scalar_value"])
                    # Extract pow parameters
                    if self._matches_operation(operation_name, "pow"):
                        if "exponent" in cfg:
                            exponent_list.append(cfg["exponent"])
                    # Extract clamp parameters
                    if self._matches_operation(operation_name, "clamp"):
                        if "min" in cfg:
                            min_list.append(cfg["min"])
                        if "max" in cfg:
                            max_list.append(cfg["max"])
                    # Extract dimension parameters (for argmax, sum, std, softmax)
                    if (
                        self._matches_operation(operation_name, "argmax")
                        or self._matches_operation(operation_name, "sum")
                        or self._matches_operation(operation_name, "std")
                        or self._matches_operation(operation_name, "softmax")
                    ):
                        if "dim" in cfg:
                            dim_list.append(cfg["dim"])
                    # Extract group_norm parameters (all 16 arguments)
                    # NOTE: All optional parameters must be appended for EVERY config (use None if missing)
                    # to ensure zip(*param_lists) doesn't truncate
                    if self._matches_operation(operation_name, "group_norm"):
                        num_groups_list.append(cfg.get("num_groups", None))
                        epsilon_list.append(cfg.get("epsilon", None))
                        # Optional tensor parameters - must append for every config
                        input_mask_shape_list.append(
                            tuple(cfg["input_mask_shape"]) if "input_mask_shape" in cfg else None
                        )
                        input_mask_dtype_list.append(
                            self.parse_dtype(cfg["input_mask_dtype"]) if "input_mask_dtype" in cfg else None
                        )
                        input_mask_layout_list.append(
                            self.parse_layout(cfg["input_mask_layout"]) if "input_mask_layout" in cfg else None
                        )
                        if "input_mask_memory_config" in cfg:
                            mem_cfg = cfg["input_mask_memory_config"]
                            if isinstance(mem_cfg, dict):
                                mem_cfg = self.parse_memory_config(mem_cfg, cfg.get("input_mask_shape", []))
                            input_mask_memory_config_list.append(mem_cfg)
                        else:
                            input_mask_memory_config_list.append(None)
                        weight_shape_list.append(tuple(cfg["weight_shape"]) if "weight_shape" in cfg else None)
                        weight_dtype_list.append(
                            self.parse_dtype(cfg["weight_dtype"]) if "weight_dtype" in cfg else None
                        )
                        weight_layout_list.append(
                            self.parse_layout(cfg["weight_layout"]) if "weight_layout" in cfg else None
                        )
                        if "weight_memory_config" in cfg:
                            mem_cfg = cfg["weight_memory_config"]
                            if isinstance(mem_cfg, dict):
                                mem_cfg = self.parse_memory_config(mem_cfg, cfg.get("weight_shape", []))
                            weight_memory_config_list.append(mem_cfg)
                        else:
                            weight_memory_config_list.append(None)
                        bias_shape_list.append(tuple(cfg["bias_shape"]) if "bias_shape" in cfg else None)
                        bias_dtype_list.append(self.parse_dtype(cfg["bias_dtype"]) if "bias_dtype" in cfg else None)
                        bias_layout_list.append(self.parse_layout(cfg["bias_layout"]) if "bias_layout" in cfg else None)
                        if "bias_memory_config" in cfg:
                            mem_cfg = cfg["bias_memory_config"]
                            if isinstance(mem_cfg, dict):
                                mem_cfg = self.parse_memory_config(mem_cfg, cfg.get("bias_shape", []))
                            bias_memory_config_list.append(mem_cfg)
                        else:
                            bias_memory_config_list.append(None)
                        reciprocals_shape_list.append(
                            tuple(cfg["reciprocals_shape"]) if "reciprocals_shape" in cfg else None
                        )
                        reciprocals_dtype_list.append(
                            self.parse_dtype(cfg["reciprocals_dtype"]) if "reciprocals_dtype" in cfg else None
                        )
                        reciprocals_layout_list.append(
                            self.parse_layout(cfg["reciprocals_layout"]) if "reciprocals_layout" in cfg else None
                        )
                        if "reciprocals_memory_config" in cfg:
                            mem_cfg = cfg["reciprocals_memory_config"]
                            if isinstance(mem_cfg, dict):
                                mem_cfg = self.parse_memory_config(mem_cfg, cfg.get("reciprocals_shape", []))
                            reciprocals_memory_config_list.append(mem_cfg)
                        else:
                            reciprocals_memory_config_list.append(None)
                        inplace_list.append(cfg.get("inplace", None))
                        num_out_blocks_list.append(cfg.get("num_out_blocks", None))
                        use_welford_list.append(cfg.get("use_welford", None))

                # Convert to exact configurations format (prevents Cartesian product)
                # Use comma-separated parameter names to pass tuples of values together
                param_names = [
                    "input_shape",
                    "input_a_dtype",
                    "input_a_layout",
                    "input_a_memory_config",
                    "output_memory_config",
                    "storage_type",
                    "traced_source",
                    "traced_machine_info",
                    "config_id",  # Unique ID to prevent hash collisions
                ]
                param_lists = [
                    input_shapes,
                    input_a_dtypes,
                    input_a_layouts,
                    input_a_memory_configs,
                    output_memory_configs,
                    storage_types,
                    traced_source_list,
                    traced_machine_info_list,
                    config_ids,  # Add unique config IDs
                ]

                # Add operation-specific parameters
                # (permute dims handling moved to later section with other op-specific params)
                if operation_name == "untilize_with_unpadding" and end_shape_list:
                    param_names.append("end_shape")
                    param_lists.append(end_shape_list)
                if operation_name == "transpose" and dim0_list and dim1_list:
                    param_names.extend(["dim0", "dim1"])
                    param_lists.extend([dim0_list, dim1_list])
                if operation_name == "reshape" and target_shape_list:
                    param_names.append("target_shape")
                    param_lists.append(target_shape_list)
                if operation_name == "pad" or operation_name == "ttnn::pad":
                    # FIX: Don't mix formats! Each config must keep its own parameters.
                    # Build COMPLETE parameter sets for BOTH formats in the SAME config
                    # This ensures each row in the zipped result has consistent parameters

                    padding_complete = []
                    value_complete = []
                    output_padded_shape_complete = []
                    input_tensor_start_complete = []

                    for idx, cfg in enumerate(paired_configs):
                        if "padding" in cfg and "value" in cfg:
                            # This config uses padding format - also store None for output_padded_shape
                            padding_complete.append(cfg["padding"])
                            value_complete.append(cfg["value"])
                            output_padded_shape_complete.append(None)
                            input_tensor_start_complete.append(None)
                        elif "output_padded_shape" in cfg and "input_tensor_start" in cfg and "value" in cfg:
                            # This config uses output_padded_shape format - also store None for padding
                            padding_complete.append(None)
                            value_complete.append(cfg["value"])
                            output_padded_shape_complete.append(cfg["output_padded_shape"])
                            input_tensor_start_complete.append(cfg["input_tensor_start"])
                        else:
                            # Config has neither format - this should not happen if first loop filtering works correctly
                            # But keep this as a safety check
                            continue

                    # Add ALL pad parameters (both formats) to support mixed configs
                    # The sweep test will check which ones are None and use the appropriate format
                    param_names.extend(["padding", "output_padded_shape", "input_tensor_start", "value"])
                    param_lists.extend(
                        [padding_complete, output_padded_shape_complete, input_tensor_start_complete, value_complete]
                    )
                if operation_name == "tilize_with_val_padding" and padded_shape_list and pad_value_list:
                    param_names.extend(["padded_shape", "pad_value"])
                    param_lists.extend([padded_shape_list, pad_value_list])
                if (
                    operation_name
                    in [
                        "nlp_concat_heads_decode",
                        "experimental::nlp_concat_heads_decode",
                        "ttnn::experimental::nlp_concat_heads_decode",
                    ]
                    and num_heads_list
                ):
                    param_names.append("num_heads")
                    param_lists.append(num_heads_list)
                # Add split_query_key_value_and_split_heads parameters
                if operation_name in [
                    "split_query_key_value_and_split_heads",
                    "experimental::split_query_key_value_and_split_heads",
                    "ttnn::experimental::split_query_key_value_and_split_heads",
                ]:
                    if num_heads_list:
                        param_names.append("num_heads")
                        param_lists.append(num_heads_list)
                    if kv_input_height_list:
                        param_names.append("kv_input_height")
                        param_lists.append(kv_input_height_list)
                # Add max_pool2d parameters
                if self._matches_operation(operation_name, "max_pool2d"):
                    if batch_size_list and input_h_list and input_w_list and channels_list:
                        param_names.extend(["batch_size", "input_h", "input_w", "channels"])
                        param_lists.extend([batch_size_list, input_h_list, input_w_list, channels_list])
                    if kernel_size_list:
                        param_names.append("kernel_size")
                        param_lists.append(kernel_size_list)
                    if stride_list:
                        param_names.append("stride")
                        param_lists.append(stride_list)
                    if padding_list_maxpool:
                        param_names.append("padding")
                        param_lists.append(padding_list_maxpool)
                    if dilation_list:
                        param_names.append("dilation")
                        param_lists.append(dilation_list)
                    if applied_shard_scheme_list:
                        param_names.append("applied_shard_scheme")
                        param_lists.append(applied_shard_scheme_list)
                # Add upsample parameters
                if self._matches_operation(operation_name, "upsample"):
                    if scale_factor_list:
                        param_names.append("scale_factor")
                        param_lists.append(scale_factor_list)
                    if mode_list:
                        param_names.append("mode")
                        param_lists.append(mode_list)
                # Add typecast parameters
                if self._matches_operation(operation_name, "typecast"):
                    if output_dtype_list:
                        param_names.append("output_dtype")
                        param_lists.append(output_dtype_list)
                # Add permute parameters
                if self._matches_operation(operation_name, "permute"):
                    if dims_list is not None and len(dims_list) > 0:
                        param_names.append("dims")
                        param_lists.append(dims_list)
                # Add gt parameters
                if self._matches_operation(operation_name, "gt"):
                    if scalar_list:
                        param_names.append("scalar")
                        param_lists.append(scalar_list)
                # Add where parameters
                if self._matches_operation(operation_name, "where"):
                    if scalar_if_true_list:
                        param_names.append("scalar_if_true")
                        param_lists.append(scalar_if_true_list)
                    if scalar_if_false_list:
                        param_names.append("scalar_if_false")
                        param_lists.append(scalar_if_false_list)
                # Add multiply_ parameters
                if self._matches_operation(operation_name, "multiply_"):
                    if scalar_value_list:
                        param_names.append("scalar_value")
                        param_lists.append(scalar_value_list)
                # Add repeat parameters (repeat vector as 'repeat_shape')
                if self._matches_operation(operation_name, "repeat"):
                    if repeat_shape_list:
                        param_names.append("repeat_shape")
                        param_lists.append(repeat_shape_list)
                # Add rms_norm program_config
                if operation_name in [
                    "rms_norm_pre_all_gather",
                    "ttnn::rms_norm_pre_all_gather",
                    "rms_norm_post_all_gather",
                    "ttnn::rms_norm_post_all_gather",
                ]:
                    if program_config_list:
                        param_names.append("program_config")
                        param_lists.append(program_config_list)
                # Add pow parameters
                if self._matches_operation(operation_name, "pow"):
                    if exponent_list:
                        param_names.append("exponent")
                        param_lists.append(exponent_list)
                # Add clamp parameters
                if self._matches_operation(operation_name, "clamp"):
                    if min_list:
                        param_names.append("min")
                        param_lists.append(min_list)
                    if max_list:
                        param_names.append("max")
                        param_lists.append(max_list)
                # Add dimension parameters (for argmax, sum, std, softmax)
                if (
                    self._matches_operation(operation_name, "argmax")
                    or self._matches_operation(operation_name, "sum")
                    or self._matches_operation(operation_name, "std")
                    or self._matches_operation(operation_name, "softmax")
                ):
                    if dim_list:
                        param_names.append("dim")
                        param_lists.append(dim_list)
                # Add group_norm parameters (all 16 arguments)
                # Always add all parameters (not conditionally) to ensure zip doesn't truncate
                if self._matches_operation(operation_name, "group_norm"):
                    param_names.extend(
                        [
                            "num_groups",
                            "epsilon",
                            "input_mask_shape",
                            "input_mask_dtype",
                            "input_mask_layout",
                            "input_mask_memory_config",
                            "weight_shape",
                            "weight_dtype",
                            "weight_layout",
                            "weight_memory_config",
                            "bias_shape",
                            "bias_dtype",
                            "bias_layout",
                            "bias_memory_config",
                            "reciprocals_shape",
                            "reciprocals_dtype",
                            "reciprocals_layout",
                            "reciprocals_memory_config",
                            "inplace",
                            "num_out_blocks",
                            "use_welford",
                        ]
                    )
                    param_lists.extend(
                        [
                            num_groups_list,
                            epsilon_list,
                            input_mask_shape_list,
                            input_mask_dtype_list,
                            input_mask_layout_list,
                            input_mask_memory_config_list,
                            weight_shape_list,
                            weight_dtype_list,
                            weight_layout_list,
                            weight_memory_config_list,
                            bias_shape_list,
                            bias_dtype_list,
                            bias_layout_list,
                            bias_memory_config_list,
                            reciprocals_shape_list,
                            reciprocals_dtype_list,
                            reciprocals_layout_list,
                            reciprocals_memory_config_list,
                            inplace_list,
                            num_out_blocks_list,
                            use_welford_list,
                        ]
                    )

                # NOTE: traced_config_name is metadata only, not passed to run()
                # param_names.append("traced_config_name")
                # param_lists.append(traced_config_names)

                # Create tuples of exact configurations (prevents Cartesian product)
                # This ensures we only test the exact traced configs, not all combinations
                exact_configs = list(zip(*param_lists))
                param_key = ",".join(param_names)
                result = {param_key: exact_configs}

                print(f"✅ Loaded {len(paired_configs)} traced configurations for {operation_name} (model_traced suite)")
                dedup_msg = " (unique inputs)" if deduplicate_inputs else " (all input/output pairs)"
                valid_configs = len(input_shapes) if input_shapes else 0
                print(f"   📊 Will generate {valid_configs} test vectors{dedup_msg}")
                if failed_configs > 0:
                    print(f"⚠️ Failed to parse {failed_configs} configurations")
                if invalid_configs:
                    print(
                        f"⚠️ Found {len(invalid_configs)} configurations with potential issues (will fail at runtime):"
                    )
                    for inv_cfg in invalid_configs[:5]:  # Show first 5
                        print(f"   Config {inv_cfg['index']}: shape={inv_cfg['shape']}")
                        for reason in inv_cfg["reasons"]:
                            print(f"      - {reason}")
                    if len(invalid_configs) > 5:
                        print(f"   ... and {len(invalid_configs) - 5} more")

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

        for config_idx, (config, source, machine_info) in enumerate(configs):
            try:
                # Extract BOTH tensors from each config
                tensor_configs = []
                scalar_value = None
                for arg in config:
                    tensor_config = self.extract_tensor_config(arg)
                    if tensor_config:
                        tensor_configs.append(tensor_config)
                        if len(tensor_configs) >= 2:
                            break  # We have both tensors
                    # Check for scalar values (for tensor-scalar operations like multiply)
                    elif len(tensor_configs) == 1 and scalar_value is None:
                        for arg_key, arg_val in arg.items():
                            # Check for numeric scalar (int or float)
                            if isinstance(arg_val, (int, float)):
                                scalar_value = arg_val
                                break
                            # Check for string representation of numeric value
                            elif isinstance(arg_val, str):
                                try:
                                    # Try to parse as float
                                    scalar_value = float(arg_val)
                                    break
                                except (ValueError, TypeError):
                                    # Non-numeric string: intentionally ignore and continue searching for a scalar
                                    pass

                # Accept either 2 tensors OR 1 tensor + 1 scalar
                if len(tensor_configs) < 1:
                    failed_configs += 1
                    continue

                if len(tensor_configs) < 2 and scalar_value is None:
                    failed_configs += 1
                    continue

                # Parse tensor configs (handle both tensor-tensor and tensor-scalar)
                try:
                    # First tensor (input_a) - always present
                    parsed_dtype_a = self.parse_dtype(tensor_configs[0].dtype)
                    parsed_layout_a = self.parse_layout(tensor_configs[0].layout)
                    parsed_mem_config_a = self.parse_memory_config(
                        tensor_configs[0].memory_config, tensor_configs[0].shape
                    )

                    # Second input - either tensor or scalar
                    if len(tensor_configs) >= 2:
                        # Tensor-tensor operation
                        parsed_dtype_b = self.parse_dtype(tensor_configs[1].dtype)
                        parsed_layout_b = self.parse_layout(tensor_configs[1].layout)
                        parsed_mem_config_b = self.parse_memory_config(
                            tensor_configs[1].memory_config, tensor_configs[1].shape
                        )
                        shape_b = tensor_configs[1].shape
                    else:
                        # Tensor-scalar operation (use None for scalar placeholders)
                        parsed_dtype_b = None
                        parsed_layout_b = None
                        parsed_mem_config_b = None
                        shape_b = None

                    if parsed_dtype_a and parsed_layout_a and parsed_mem_config_a:
                        # Build config dict
                        config_dict = {
                            "shape_a": tensor_configs[0].shape,
                            "shape_b": shape_b,
                            "dtype_a": parsed_dtype_a,
                            "dtype_b": parsed_dtype_b,
                            "layout_a": parsed_layout_a,
                            "layout_b": parsed_layout_b,
                            "memory_config_a": parsed_mem_config_a,
                            "memory_config_b": parsed_mem_config_b,
                            "output_memory_config": parsed_mem_config_a,  # Use first input's memory config as default
                            "traced_source": source,
                            "traced_machine_info": machine_info,
                        }

                        # Add scalar value if present
                        if scalar_value is not None:
                            config_dict["scalar"] = scalar_value

                        paired_configs.append(config_dict)
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

                # For binary operations, pass shapes as a dict with "self" and "other" keys
                # Convert shapes to tuples for proper serialization
                input_shapes = [
                    {"self": tuple(sa), "other": tuple(sb)} for sa, sb in zip(unique_shapes_a, unique_shapes_b)
                ]
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
                    len(unique_shapes_a)
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
                traced_source_list = []
                traced_machine_info_list = []
                traced_config_names = []

                # Separate lists for optional scalar parameter
                scalars = []

                # Create unique config IDs to prevent unintended hash collisions
                # This ensures that even if two configs have identical parameters,
                # they generate different test vectors if they come from different traced configs
                config_ids = []

                for idx, cfg in enumerate(paired_configs):
                    # Handle both tensor-tensor and tensor-scalar operations
                    if cfg["shape_b"] is not None:
                        # Tensor-tensor: Pass shapes as a dict with "self" and "other" keys
                        input_shapes.append({"self": tuple(cfg["shape_a"]), "other": tuple(cfg["shape_b"])})
                    else:
                        # Tensor-scalar: Pass "other" as None to indicate scalar operation
                        input_shapes.append({"self": tuple(cfg["shape_a"]), "other": None})

                    # Parse dtype/layout strings to ttnn objects
                    input_a_dtypes.append(self.parse_dtype(cfg["dtype_a"]))
                    input_b_dtypes.append(self.parse_dtype(cfg["dtype_b"]))
                    input_a_layouts.append(self.parse_layout(cfg["layout_a"]))
                    input_b_layouts.append(self.parse_layout(cfg["layout_b"]))
                    input_a_memory_configs.append(cfg["memory_config_a"])
                    input_b_memory_configs.append(cfg["memory_config_b"])
                    output_memory_configs.append(cfg["output_memory_config"])
                    traced_source_list.append(cfg.get("traced_source", "unknown"))
                    traced_machine_info_list.append(cfg.get("traced_machine_info", None))
                    traced_config_names.append(f"{operation_name}_traced_{idx}")
                    # Add unique config ID to ensure unique hashes
                    config_ids.append(f"config_{idx}")

                    # Add scalar value if present (will be None for tensor-tensor ops)
                    scalars.append(cfg.get("scalar", None))

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
                    "scalar",  # For tensor-scalar operations (None for tensor-tensor)
                    "traced_source",
                    "traced_machine_info",
                    "config_id",  # Unique ID to prevent hash collisions
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
                    scalars,  # Add scalar values
                    traced_source_list,
                    traced_machine_info_list,
                    config_ids,  # Add unique config IDs
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
                    print(
                        f"⚠️ Failed to parse {failed_configs} configurations (including invalid configs filtered out)"
                    )
                # For matmul, add note if configs were filtered
                if self._matches_operation(operation_name, "matmul") and len(paired_configs) == 0 and len(configs) > 0:
                    print(f"   ℹ️ All matmul configs were filtered out (input_b must be INTERLEAVED)")

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
        For paged_update_cache, the 4th tensor is optional (min 3, max 4).
        """
        # Extract configurations for ALL tensors
        paired_configs = []
        failed_configs = 0

        # For paged_update_cache, allow 3 or 4 tensors (4th is optional)
        is_paged_update_cache = "paged_update_cache" in operation_name.lower()
        min_tensor_count = tensor_count - 1 if is_paged_update_cache else tensor_count

        for config_idx, (config, source, machine_info) in enumerate(configs):
            try:
                # Extract ALL tensors from each config
                tensor_configs = []
                for arg in config:
                    tensor_config = self.extract_tensor_config(arg)
                    if tensor_config:
                        tensor_configs.append(tensor_config)
                        # Continue collecting up to tensor_count (don't break early)
                        if len(tensor_configs) >= tensor_count:
                            break

                # Accept min_tensor_count to tensor_count tensors
                if len(tensor_configs) < min_tensor_count:
                    failed_configs += 1
                    continue

                # Parse all tensor configs (actual count may be less than tensor_count if optional)
                parsed_config = {"traced_source": source, "traced_machine_info": machine_info}
                actual_tensor_count = len(tensor_configs)
                parse_failed = False

                for i, tc in enumerate(tensor_configs):
                    suffix = chr(97 + i)  # a, b, c, d, ...
                    try:
                        parsed_config[f"shape_{suffix}"] = tc.shape
                        parsed_config[f"dtype_{suffix}"] = self.parse_dtype(tc.dtype)
                        parsed_config[f"layout_{suffix}"] = self.parse_layout(tc.layout)
                        parsed_config[f"memory_config_{suffix}"] = self.parse_memory_config(tc.memory_config, tc.shape)
                    except Exception as e:
                        failed_configs += 1
                        parse_failed = True
                        break

                if not parse_failed:
                    # Verify we have all required fields for the actual tensors parsed
                    expected_fields = (
                        actual_tensor_count * 4 + 2
                    )  # shape, dtype, layout, mem_config for each tensor + traced_source + traced_machine_info
                    if len(parsed_config) == expected_fields:
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

                # Build input_shape as list of dicts (like add_llama.py format)
                # BUT convert lists to tuples for serialization
                unique_shapes = []
                seen_shapes = set()
                for cfg in paired_configs:
                    shape_tuple = tuple([tuple(cfg[f"shape_{chr(97+i)}"]) for i in range(tensor_count)])
                    if shape_tuple not in seen_shapes:
                        # Create dict with tuple values (not list values) so eval() works
                        shape_dict = {
                            f"input_{chr(97+i)}": tuple(cfg[f"shape_{chr(97+i)}"]) for i in range(tensor_count)
                        }
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
                traced_source_list = []
                traced_machine_info_list = []
                traced_config_names = []
                config_ids = []  # Unique IDs to prevent hash collisions

                for idx, cfg in enumerate(paired_configs):
                    # Determine actual tensor count in this config (may be less than expected for optional tensors)
                    actual_tensor_count = sum(1 for i in range(tensor_count) if f"shape_{chr(97+i)}" in cfg)

                    # Build input_shape dict with tuple values (not list values) for serialization
                    input_shape = {
                        f"input_{chr(97+i)}": tuple(cfg[f"shape_{chr(97+i)}"]) for i in range(actual_tensor_count)
                    }
                    input_shapes.append(input_shape)

                    # Extract dtypes, layouts, and memory configs for each input (up to actual count)
                    for i in range(actual_tensor_count):
                        suffix = chr(97 + i)
                        dtypes[i].append(cfg[f"dtype_{suffix}"])
                        layouts[i].append(cfg[f"layout_{suffix}"])
                        memory_configs[i].append(cfg[f"memory_config_{suffix}"])

                    # For missing optional tensors, append None
                    for i in range(actual_tensor_count, tensor_count):
                        dtypes[i].append(None)
                        layouts[i].append(None)
                        memory_configs[i].append(None)

                    traced_source_list.append(cfg.get("traced_source", "unknown"))
                    traced_machine_info_list.append(cfg.get("traced_machine_info", None))
                    traced_config_names.append(f"{operation_name}_traced_{idx}")
                    config_ids.append(f"config_{idx}")  # Unique ID to prevent hash collisions

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

                # Add traced_source and traced_machine_info for traceability
                param_names.append("traced_source")
                param_lists.append(traced_source_list)
                param_names.append("traced_machine_info")
                param_lists.append(traced_machine_info_list)
                param_names.append("config_id")  # Unique ID to prevent hash collisions
                param_lists.append(config_ids)

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

    def _get_conv2d_suite_parameters(
        self, operation_name: str, configs: List, all_cases: bool, deduplicate_inputs: bool = False
    ) -> Dict:
        """Get parameters for conv2d operation which uses input_specs format"""
        try:
            input_specs_list = []
            compute_configs_list = []
            dtypes_list = []
            config_tensors_in_dram_list = []
            traced_source_list = []
            traced_machine_info_list = []

            for config, source, machine_info in configs:
                params = self._extract_conv2d_parameters(config)
                if params:
                    # Build input_specs list:
                    # [batch_size, output_channels, input_channels, input_height, input_width,
                    #  kernel_height, kernel_width, stride_h, stride_w, pad_h, pad_w, groups, dilation_h, dilation_w, bias]
                    # Use tuple so it serializes as a string for proper deserialization
                    input_spec = (
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
                    )
                    input_specs_list.append(input_spec)
                    # Extract compute_config if available
                    compute_configs_list.append(params.get("compute_config"))
                    # Extract dtype if available
                    dtypes_list.append(params.get("dtype", "bfloat16"))
                    # Extract config_tensors_in_dram if available (default True for model_traced to help with OOM)
                    config_tensors_in_dram_list.append(params.get("config_tensors_in_dram", True))
                    # Track source for traceability
                    traced_source_list.append(source)
                    # Track machine_info for traceability
                    traced_machine_info_list.append(machine_info)

            if input_specs_list:
                print(
                    f"✅ Loaded {len(input_specs_list)} traced configurations for {operation_name} (model_traced suite)"
                )
                # Pair input_specs with is_conv1d, compute_config, dtype, config_tensors_in_dram, traced_source, traced_machine_info, and config_id to prevent Cartesian product
                # Use comma-separated parameter name to pass tuples together
                config_ids = [f"config_{idx}" for idx in range(len(input_specs_list))]
                paired_configs = list(
                    zip(
                        input_specs_list,
                        [False] * len(input_specs_list),
                        compute_configs_list,
                        dtypes_list,
                        config_tensors_in_dram_list,
                        traced_source_list,
                        traced_machine_info_list,
                        config_ids,
                    )
                )
                return {
                    "input_specs,is_conv1d,compute_config,dtype,config_tensors_in_dram,traced_source,traced_machine_info,config_id": paired_configs,
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

            for config, source, machine_info in configs:
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
                linear_params = OperationParameterExtractors._extract_linear_parameters(config)
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
                        "traced_source": source,
                        "traced_machine_info": machine_info,
                    }
                    paired_configs.append(config_dict)

            if paired_configs:
                print(f"✅ Loaded {len(paired_configs)} traced configurations for {operation_name} (model_traced suite)")

                # Build parameter dict
                param_names = [
                    "input_shape,weight_shape,bias_shape,input_a_dtype,input_b_dtype,input_a_layout,input_b_layout,"
                    + "input_a_memory_config,input_b_memory_config,output_memory_config,transpose_a,transpose_b,has_bias,traced_source,traced_machine_info"
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
                            cfg["traced_source"],
                            cfg["traced_machine_info"],
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

            # First extract parameters from each config, tracking sources and machine_info
            extracted_params = []
            extracted_sources = []
            extracted_machine_infos = []
            for config, source, machine_info in configs:
                params = OperationParameterExtractors.extract_parameters(clean_op_name, config)
                if params:
                    extracted_params.append(params)
                    extracted_sources.append(source)
                    extracted_machine_infos.append(machine_info)

            # Then transform the extracted parameters
            if extracted_params:
                transformed_configs = OperationParameterExtractors.transform_parameters(
                    clean_op_name,
                    extracted_params,
                    parse_dtype=self.parse_dtype,
                    parse_layout=self.parse_layout,
                    parse_memory_config=self.parse_memory_config,
                )

                if transformed_configs:
                    print(
                        f"✅ Loaded {len(transformed_configs)} traced configurations for {operation_name} (model_traced suite)"
                    )

                    # For embedding, return tuples to prevent cartesian product explosion
                    if clean_op_name == "embedding":
                        param_tuples = []

                        for idx, cfg in enumerate(transformed_configs):
                            # Convert input_shape dict to embedding_args tuple format
                            # input_shape is {"self": [batch_size, seq_length], "other": [num_embeddings, embeddings_dim]}
                            # embedding_args should be (batch_size, seq_length, embeddings_dim, num_embeddings)
                            input_shape_dict = cfg["input_shape"]
                            if (
                                isinstance(input_shape_dict, dict)
                                and "self" in input_shape_dict
                                and "other" in input_shape_dict
                            ):
                                self_shape = input_shape_dict["self"]
                                other_shape = input_shape_dict["other"]
                                # Handle both list and tuple formats
                                if isinstance(self_shape, (list, tuple)) and len(self_shape) >= 2:
                                    batch_size = self_shape[0] if isinstance(self_shape[0], int) else self_shape[-2]
                                    seq_length = self_shape[-1]
                                else:
                                    continue
                                if isinstance(other_shape, (list, tuple)) and len(other_shape) >= 2:
                                    num_embeddings = other_shape[-2]
                                    embeddings_dim = other_shape[-1]
                                else:
                                    continue
                                embedding_args = (batch_size, seq_length, embeddings_dim, num_embeddings)
                            else:
                                continue

                            # Create tuple with all parameters to keep configs together
                            param_tuples.append(
                                (
                                    embedding_args,
                                    cfg["input_a_dtype"],
                                    cfg["input_b_dtype"],
                                    cfg["input_b_dtype"],  # Output dtype matches weight dtype
                                    cfg["input_a_layout"],
                                    cfg["input_b_layout"],
                                    cfg["input_a_memory_config"],
                                    cfg["input_b_memory_config"],
                                    cfg["output_memory_config"],
                                    extracted_sources[idx] if idx < len(extracted_sources) else "unknown",
                                    extracted_machine_infos[idx] if idx < len(extracted_machine_infos) else None,
                                )
                            )

                        # Return as comma-separated parameter name with tuple list (like linear)
                        param_name = "embedding_args,input_dtype,weight_dtype,output_dtype,input_layout,weight_layout,input_memory_config,weight_memory_config,output_memory_config,traced_source,traced_machine_info"
                        return {param_name: param_tuples}

                    elif clean_op_name == "linear":
                        param_names = [
                            "input_shape,weight_shape,bias_shape,input_a_dtype,input_b_dtype,input_a_layout,input_b_layout,"
                            + "input_a_memory_config,input_b_memory_config,output_memory_config,transpose_a,transpose_b,has_bias,traced_source,traced_machine_info,config_id"
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
                                    extracted_sources[idx] if idx < len(extracted_sources) else "unknown",
                                    extracted_machine_infos[idx] if idx < len(extracted_machine_infos) else None,
                                    f"config_{idx}",  # Unique config ID
                                )
                                for idx, cfg in enumerate(transformed_configs)
                            ]
                        ]
                        return {param_names[0]: param_lists[0]}

                    # For all_gather_async, we need to return parameters in the correct format
                    elif clean_op_name == "experimental::all_gather_async" or clean_op_name == "all_gather_async":
                        # Extract parameters from transformed configs
                        input_shapes = []
                        input_a_dtypes = []
                        input_a_layouts = []
                        input_a_memory_configs = []
                        output_memory_configs = []

                        for cfg in transformed_configs:
                            input_shapes.append(cfg.get("input_shape"))
                            input_a_dtypes.append(cfg.get("input_dtype"))
                            input_a_layouts.append(cfg.get("input_layout", ttnn.TILE_LAYOUT))
                            input_a_memory_configs.append(cfg.get("input_memory_config"))
                            output_memory_configs.append(cfg.get("output_memory_config"))

                        # Create tuples of exact configurations
                        param_names = [
                            "input_shape,input_a_dtype,input_a_layout,input_a_memory_config,output_memory_config,traced_source,traced_machine_info"
                        ]
                        param_lists = [
                            [
                                (
                                    cfg.get("input_shape"),
                                    cfg.get("input_dtype"),
                                    cfg.get("input_layout", ttnn.TILE_LAYOUT),
                                    cfg.get("input_memory_config"),
                                    cfg.get("output_memory_config"),
                                    extracted_sources[idx] if idx < len(extracted_sources) else "unknown",
                                    extracted_machine_infos[idx] if idx < len(extracted_machine_infos) else None,
                                )
                                for idx, cfg in enumerate(transformed_configs)
                            ]
                        ]
                        return {param_names[0]: param_lists[0]}

                    # For paged_scaled_dot_product_attention_decode, extract parameters from transformed configs
                    elif (
                        clean_op_name == "transformer::paged_scaled_dot_product_attention_decode"
                        or clean_op_name == "paged_scaled_dot_product_attention_decode"
                    ):
                        # Extract parameters from transformed configs
                        input_shapes = []
                        input_a_dtypes = []
                        input_a_layouts = []
                        input_a_memory_configs = []
                        input_b_dtypes = []
                        input_b_layouts = []
                        input_b_memory_configs = []
                        input_c_dtypes = []
                        input_c_layouts = []
                        input_c_memory_configs = []
                        input_d_dtypes = []
                        input_d_layouts = []
                        input_d_memory_configs = []
                        input_e_dtypes = []
                        input_e_layouts = []
                        input_e_memory_configs = []
                        output_memory_configs = []

                        for cfg in transformed_configs:
                            input_shapes.append(cfg.get("input_shape"))
                            input_a_dtypes.append(cfg.get("input_a_dtype"))
                            input_a_layouts.append(cfg.get("input_a_layout", ttnn.TILE_LAYOUT))
                            input_a_memory_configs.append(cfg.get("input_a_memory_config"))
                            input_b_dtypes.append(cfg.get("input_b_dtype"))
                            input_b_layouts.append(cfg.get("input_b_layout", ttnn.TILE_LAYOUT))
                            input_b_memory_configs.append(cfg.get("input_b_memory_config"))
                            input_c_dtypes.append(cfg.get("input_c_dtype"))
                            input_c_layouts.append(cfg.get("input_c_layout", ttnn.TILE_LAYOUT))
                            input_c_memory_configs.append(cfg.get("input_c_memory_config"))
                            input_d_dtypes.append(cfg.get("input_d_dtype"))
                            input_d_layouts.append(cfg.get("input_d_layout", ttnn.TILE_LAYOUT))
                            input_d_memory_configs.append(cfg.get("input_d_memory_config"))
                            input_e_dtypes.append(cfg.get("input_e_dtype"))
                            input_e_layouts.append(cfg.get("input_e_layout", ttnn.TILE_LAYOUT))
                            input_e_memory_configs.append(cfg.get("input_e_memory_config"))
                            output_memory_configs.append(cfg.get("output_memory_config"))

                        # Create tuples of exact configurations
                        param_names = [
                            "input_shape,input_a_dtype,input_a_layout,input_a_memory_config,input_b_dtype,input_b_layout,input_b_memory_config,input_c_dtype,input_c_layout,input_c_memory_config,input_d_dtype,input_d_layout,input_d_memory_config,input_e_dtype,input_e_layout,input_e_memory_config,output_memory_config,traced_source,traced_machine_info"
                        ]
                        param_lists = [
                            [
                                (
                                    cfg.get("input_shape"),
                                    cfg.get("input_a_dtype"),
                                    cfg.get("input_a_layout", ttnn.TILE_LAYOUT),
                                    cfg.get("input_a_memory_config"),
                                    cfg.get("input_b_dtype"),
                                    cfg.get("input_b_layout", ttnn.TILE_LAYOUT),
                                    cfg.get("input_b_memory_config"),
                                    cfg.get("input_c_dtype"),
                                    cfg.get("input_c_layout", ttnn.TILE_LAYOUT),
                                    cfg.get("input_c_memory_config"),
                                    cfg.get("input_d_dtype"),
                                    cfg.get("input_d_layout", ttnn.TILE_LAYOUT),
                                    cfg.get("input_d_memory_config"),
                                    cfg.get("input_e_dtype"),
                                    cfg.get("input_e_layout", ttnn.TILE_LAYOUT),
                                    cfg.get("input_e_memory_config"),
                                    cfg.get("output_memory_config"),
                                    extracted_sources[idx] if idx < len(extracted_sources) else "unknown",
                                    extracted_machine_infos[idx] if idx < len(extracted_machine_infos) else None,
                                )
                                for idx, cfg in enumerate(transformed_configs)
                            ]
                        ]
                        return {param_names[0]: param_lists[0]}

                    # For scaled_dot_product_attention_decode (non-paged), extract parameters including scalar params
                    elif (
                        clean_op_name == "transformer::scaled_dot_product_attention_decode"
                        or clean_op_name == "scaled_dot_product_attention_decode"
                    ):
                        # Build parameter tuples including scalar parameters
                        param_names = [
                            "input_shape,input_a_dtype,input_a_layout,input_a_memory_config,input_b_dtype,input_b_layout,input_b_memory_config,input_c_dtype,input_c_layout,input_c_memory_config,input_d_dtype,input_d_layout,input_d_memory_config,output_memory_config,scale,k_chunk_size,is_causal,traced_source,traced_machine_info"
                        ]
                        param_lists = [
                            [
                                (
                                    cfg.get("input_shape"),
                                    cfg.get("input_a_dtype"),
                                    cfg.get("input_a_layout", ttnn.TILE_LAYOUT),
                                    cfg.get("input_a_memory_config"),
                                    cfg.get("input_b_dtype"),
                                    cfg.get("input_b_layout", ttnn.TILE_LAYOUT),
                                    cfg.get("input_b_memory_config"),
                                    cfg.get("input_c_dtype"),
                                    cfg.get("input_c_layout", ttnn.TILE_LAYOUT),
                                    cfg.get("input_c_memory_config"),
                                    cfg.get("input_d_dtype"),
                                    cfg.get("input_d_layout", ttnn.TILE_LAYOUT),
                                    cfg.get("input_d_memory_config"),
                                    cfg.get("output_memory_config"),
                                    cfg.get("scale"),  # Scalar parameter
                                    cfg.get("k_chunk_size"),  # Scalar parameter
                                    cfg.get("is_causal"),  # Scalar parameter
                                    extracted_sources[idx] if idx < len(extracted_sources) else "unknown",
                                    extracted_machine_infos[idx] if idx < len(extracted_machine_infos) else None,
                                )
                                for idx, cfg in enumerate(transformed_configs)
                            ]
                        ]
                        return {param_names[0]: param_lists[0]}

                    # For paged_update_cache (4 tensor inputs)
                    elif clean_op_name == "experimental::paged_update_cache" or clean_op_name == "paged_update_cache":
                        param_names = [
                            "input_shape,input_a_dtype,input_a_layout,input_a_memory_config,input_b_dtype,input_b_layout,input_b_memory_config,input_c_dtype,input_c_layout,input_c_memory_config,input_d_dtype,input_d_layout,input_d_memory_config,output_memory_config,traced_source,traced_machine_info"
                        ]
                        param_lists = [
                            [
                                (
                                    cfg.get("input_shape"),
                                    cfg.get("input_a_dtype"),
                                    cfg.get("input_a_layout", ttnn.TILE_LAYOUT),
                                    cfg.get("input_a_memory_config"),
                                    cfg.get("input_b_dtype"),
                                    cfg.get("input_b_layout", ttnn.TILE_LAYOUT),
                                    cfg.get("input_b_memory_config"),
                                    cfg.get("input_c_dtype"),
                                    cfg.get("input_c_layout", ttnn.TILE_LAYOUT),
                                    cfg.get("input_c_memory_config"),
                                    cfg.get("input_d_dtype"),
                                    cfg.get("input_d_layout", ttnn.TILE_LAYOUT),
                                    cfg.get("input_d_memory_config"),
                                    cfg.get("output_memory_config"),
                                    extracted_sources[idx] if idx < len(extracted_sources) else "unknown",
                                    extracted_machine_infos[idx] if idx < len(extracted_machine_infos) else None,
                                )
                                for idx, cfg in enumerate(transformed_configs)
                            ]
                        ]
                        return {param_names[0]: param_lists[0]}

                    # For scale_mask_softmax_in_place (1 tensor input + scale + optional mask)
                    elif (
                        clean_op_name == "scale_mask_softmax_in_place"
                        or clean_op_name == "ttnn::scale_mask_softmax_in_place"
                    ):
                        param_names = [
                            "input_shape,input_a_dtype,input_a_layout,input_a_memory_config,mask_shape,input_b_dtype,input_b_layout,input_b_memory_config,output_memory_config,scalar,traced_source,traced_machine_info,config_id"
                        ]
                        param_lists = [
                            [
                                (
                                    cfg.get("input_shape"),
                                    cfg.get("input_a_dtype"),
                                    cfg.get("input_a_layout", ttnn.TILE_LAYOUT),
                                    cfg.get("input_a_memory_config"),
                                    cfg.get("mask_shape"),  # Optional - can be None
                                    cfg.get("input_b_dtype"),  # Optional
                                    cfg.get("input_b_layout", ttnn.TILE_LAYOUT),  # Optional
                                    cfg.get("input_b_memory_config"),  # Optional
                                    cfg.get("output_memory_config"),
                                    cfg.get("scalar"),  # Scale value
                                    extracted_sources[idx] if idx < len(extracted_sources) else "unknown",
                                    extracted_machine_infos[idx] if idx < len(extracted_machine_infos) else None,
                                    f"config_{idx}",  # Unique config ID
                                )
                                for idx, cfg in enumerate(transformed_configs)
                            ]
                        ]
                        return {param_names[0]: param_lists[0]}

                    # For update_cache (2 tensor inputs + 2 scalars)
                    elif clean_op_name == "update_cache" or clean_op_name == "ttnn::update_cache":
                        param_names = [
                            "input_shape,input_a_dtype,input_a_layout,input_a_memory_config,input_b_dtype,input_b_layout,input_b_memory_config,output_memory_config,scalar,traced_source,traced_machine_info,config_id"
                        ]
                        param_lists = [
                            [
                                (
                                    cfg.get("input_shape"),
                                    cfg.get("input_a_dtype"),
                                    cfg.get("input_a_layout", ttnn.TILE_LAYOUT),
                                    cfg.get("input_a_memory_config"),
                                    cfg.get("input_b_dtype"),
                                    cfg.get("input_b_layout", ttnn.TILE_LAYOUT),
                                    cfg.get("input_b_memory_config"),
                                    cfg.get("output_memory_config"),
                                    cfg.get("scalar"),  # Dict with update_index and batch_offset
                                    extracted_sources[idx] if idx < len(extracted_sources) else "unknown",
                                    extracted_machine_infos[idx] if idx < len(extracted_machine_infos) else None,
                                    f"config_{idx}",  # Unique config ID
                                )
                                for idx, cfg in enumerate(transformed_configs)
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

    def _get_concat_suite_parameters(
        self, operation_name: str, configs: List, all_cases: bool, deduplicate_inputs: bool = False
    ) -> Dict:
        """Get parameters for concat operation which takes a vector of tensors"""
        try:
            paired_configs = []
            failed_configs = 0

            for config_idx, (config, source, machine_info) in enumerate(configs):
                try:
                    # Concat takes a vector of tensors as arg0, dim as arg1, memory_config as arg2
                    # Extract vector of tensors from arg0 (may be UnparsedElement)
                    tensor_configs = []
                    dim = None
                    memory_config = None

                    # Extract dim from arg1
                    for arg in config:
                        if isinstance(arg, dict):
                            if "arg1" in arg:
                                dim_val = arg["arg1"]
                                if isinstance(dim_val, (int, str)) and dim_val != "nullopt":
                                    try:
                                        dim = int(dim_val)
                                    except (ValueError, TypeError):
                                        pass
                            if "arg2" in arg:
                                mem_config_data = arg["arg2"]
                                if isinstance(mem_config_data, dict) and "MemoryConfig" in mem_config_data:
                                    memory_config = self.parse_memory_config(mem_config_data["MemoryConfig"], None)

                            # Extract vector of tensors from arg0
                            if "arg0" in arg:
                                arg0_data = arg["arg0"]

                                # Check if it's a list (which it should be after our parsing fix)
                                if isinstance(arg0_data, list):
                                    if len(arg0_data) > 0 and isinstance(arg0_data[0], dict):
                                        if "tensor_spec" in arg0_data[0]:
                                            tensor_configs = arg0_data

                                # Check if it's a string (simplified representation)
                                elif (
                                    isinstance(arg0_data, str)
                                    and arg0_data.startswith("[{")
                                    and "tensor_spec" in arg0_data
                                ):
                                    # Try to parse the JSON array string
                                    try:
                                        tensor_array = json.loads(arg0_data)
                                        tensor_configs = tensor_array
                                    except (json.JSONDecodeError, ValueError):
                                        pass

                            # Check for UnparsedElement in arg0
                            if "UnparsedElement" in arg:
                                from tests.sweep_framework.operation_parameter_extractors import (
                                    OperationParameterExtractors,
                                )

                                unparsed_data = arg["UnparsedElement"]
                                tensor_vector = OperationParameterExtractors.extract_tensor_vector_from_unparsed(
                                    unparsed_data
                                )
                                if tensor_vector:
                                    tensor_configs = tensor_vector

                    # Extract shapes, dtypes, layouts, and memory_configs from tensor vector
                    if tensor_configs and len(tensor_configs) >= 2 and dim is not None:
                        # Build input_shape dict with all tensor shapes
                        input_shape_dict = {}
                        input_dtypes = []
                        input_layouts = []
                        input_memory_configs = []

                        for i, tensor_obj in enumerate(tensor_configs):
                            if "tensor_spec" in tensor_obj:
                                tensor_spec = tensor_obj["tensor_spec"]
                                tensor_layout = tensor_spec.get("tensor_layout", {})

                                shape = tensor_spec.get("logical_shape", [])
                                dtype_str = tensor_layout.get("dtype", "")
                                layout_str = tensor_layout.get("layout", "")
                                mem_config_dict = tensor_layout.get("memory_config", {})

                                # Store shape with key like input_a, input_b, etc.
                                suffix = chr(97 + i)  # a, b, c, ...
                                input_shape_dict[f"input_{suffix}"] = shape

                                # Parse and store dtype, layout, memory_config
                                if dtype_str:
                                    dtype_str_clean = dtype_str.replace("DataType::", "")
                                    input_dtypes.append(self.parse_dtype(f"DataType::{dtype_str_clean}"))
                                else:
                                    input_dtypes.append(None)

                                if layout_str:
                                    layout_str_clean = layout_str.replace("Layout::", "")
                                    input_layouts.append(self.parse_layout(layout_str_clean))
                                else:
                                    input_layouts.append(ttnn.TILE_LAYOUT)

                                if mem_config_dict:
                                    input_memory_configs.append(self.parse_memory_config(mem_config_dict, shape))
                                else:
                                    input_memory_configs.append(None)

                        # Create config dict with all extracted information
                        config_dict = {
                            "input_shape": input_shape_dict,
                            "dim": dim,
                            "output_memory_config": memory_config or ttnn.DRAM_MEMORY_CONFIG,
                            "traced_source": source,
                            "traced_machine_info": machine_info,
                        }

                        # Add dtype, layout, memory_config for each input (at least 2)
                        if len(input_dtypes) >= 2:
                            config_dict["input_a_dtype"] = input_dtypes[0]
                            config_dict["input_b_dtype"] = input_dtypes[1]
                            config_dict["input_a_layout"] = input_layouts[0]
                            config_dict["input_b_layout"] = input_layouts[1]
                            config_dict["input_a_memory_config"] = input_memory_configs[0] or ttnn.DRAM_MEMORY_CONFIG
                            config_dict["input_b_memory_config"] = input_memory_configs[1] or ttnn.DRAM_MEMORY_CONFIG

                        paired_configs.append(config_dict)

                except Exception as e:
                    failed_configs += 1
                    continue

            if paired_configs:
                print(f"✅ Loaded {len(paired_configs)} traced configurations for {operation_name} (model_traced suite)")

                # Build parameter dict - include input_shape and all tensor parameters
                param_names = [
                    "input_shape",
                    "dim",
                    "input_a_dtype",
                    "input_a_layout",
                    "input_a_memory_config",
                    "input_b_dtype",
                    "input_b_layout",
                    "input_b_memory_config",
                    "output_memory_config",
                    "traced_source",
                    "traced_machine_info",
                ]
                param_lists = [
                    [cfg.get("input_shape") for cfg in paired_configs],
                    [cfg.get("dim") for cfg in paired_configs],
                    [cfg.get("input_a_dtype") for cfg in paired_configs],
                    [cfg.get("input_a_layout") for cfg in paired_configs],
                    [cfg.get("input_a_memory_config") for cfg in paired_configs],
                    [cfg.get("input_b_dtype") for cfg in paired_configs],
                    [cfg.get("input_b_layout") for cfg in paired_configs],
                    [cfg.get("input_b_memory_config") for cfg in paired_configs],
                    [cfg.get("output_memory_config") for cfg in paired_configs],
                    [cfg.get("traced_source", "unknown") for cfg in paired_configs],
                    [cfg.get("traced_machine_info") for cfg in paired_configs],
                ]

                # Create tuples of exact configurations
                exact_configs = list(zip(*param_lists))
                param_key = ",".join(param_names)

                return {param_key: exact_configs}

            return {}
        except Exception as e:
            print(f"❌ Error extracting concat parameters: {e}")
            import traceback

            traceback.print_exc()
            return {}

    def _get_fill_suite_parameters(
        self, operation_name: str, configs: List, all_cases: bool, deduplicate_inputs: bool = False
    ) -> Dict:
        """Get parameters for fill operation which requires fill_value parameter"""
        # Extract fill_value from arg1
        fill_values = []
        for config in configs:
            if isinstance(config, dict) and "arguments" in config:
                args = config["arguments"]
                if len(args) > 1 and isinstance(args[1], dict) and "arg1" in args[1]:
                    fill_value = args[1]["arg1"]
                    try:
                        # Convert string to float
                        fill_values.append(float(fill_value))
                    except:
                        fill_values.append(0.0)

        # Get base parameters using unary operation logic
        params = self._get_operation_suite_parameters(operation_name, configs, all_cases, deduplicate_inputs)

        # Add fill_value parameter
        if fill_values:
            params["fill_value"] = fill_values

        return params

    def _get_split_suite_parameters(
        self, operation_name: str, configs: List, all_cases: bool, deduplicate_inputs: bool = False
    ) -> Dict:
        """Get parameters for split operation which requires split_size and dim parameters"""
        split_sizes = []
        dims = []
        for config in configs:
            if isinstance(config, dict) and "arguments" in config:
                args = config["arguments"]
                # arg1 is split_size/num_chunks, arg2 is dim
                if len(args) > 1 and isinstance(args[1], dict) and "arg1" in args[1]:
                    try:
                        split_sizes.append(int(args[1]["arg1"]))
                    except:
                        split_sizes.append(1)
                if len(args) > 2 and isinstance(args[2], dict) and "arg2" in args[2]:
                    try:
                        dims.append(int(args[2]["arg2"]))
                    except:
                        dims.append(0)

        # Get base parameters
        params = self._get_operation_suite_parameters(operation_name, configs, all_cases, deduplicate_inputs)

        # Add operation-specific parameters
        if split_sizes:
            params["split_size"] = split_sizes
        if dims:
            params["dim"] = dims

        return params

    def _get_scatter_suite_parameters(
        self, operation_name: str, configs: List, all_cases: bool, deduplicate_inputs: bool = False
    ) -> Dict:
        """Get parameters for scatter operation which has 3 tensor inputs and dim parameter"""
        dims = []
        for config in configs:
            if isinstance(config, dict) and "arguments" in config:
                args = config["arguments"]
                # arg1 is dim (arg0=input, arg2=index, arg3=src)
                if len(args) > 1 and isinstance(args[1], dict) and "arg1" in args[1]:
                    try:
                        dims.append(int(args[1]["arg1"]))
                    except:
                        dims.append(0)

        # Get base parameters - scatter is a 3-tensor operation
        params = self._get_operation_suite_parameters(operation_name, configs, all_cases, deduplicate_inputs)

        # Add dim parameter
        if dims:
            params["dim"] = dims

        return params

    def _get_attention_softmax_suite_parameters(
        self, operation_name: str, configs: List, all_cases: bool, deduplicate_inputs: bool = False
    ) -> Dict:
        """Get parameters for attention_softmax_ operation which requires head_size and attention_mask"""
        scalars = []
        for config in configs:
            if isinstance(config, dict) and "arguments" in config:
                args = config["arguments"]
                # arg0 is input tensor, arg1 is head_size (scalar), arg2 is attention_mask tensor
                if len(args) > 1 and isinstance(args[1], dict) and "arg1" in args[1]:
                    try:
                        head_size_value = args[1]["arg1"]
                        scalars.append(int(head_size_value) if head_size_value else None)
                    except:
                        scalars.append(None)

        # Get base parameters using the binary operation logic (has 2 tensor inputs)
        params = self._get_binary_suite_parameters(operation_name, configs, all_cases, deduplicate_inputs)

        # Add scalar parameter (head_size)
        if scalars:
            params["scalar"] = scalars

        return params

    def _get_fast_reduce_nc_suite_parameters(
        self, operation_name: str, configs: List, all_cases: bool, deduplicate_inputs: bool = False
    ) -> Dict:
        """Get parameters for fast_reduce_nc operation which requires dims list parameter"""
        dims_list = []
        for config in configs:
            if isinstance(config, dict) and "arguments" in config:
                args = config["arguments"]
                # arg1 is dims list
                if len(args) > 1 and isinstance(args[1], dict) and "arg1" in args[1]:
                    dims_value = args[1]["arg1"]
                    if isinstance(dims_value, list):
                        dims_list.append(dims_value)
                    elif isinstance(dims_value, str):
                        try:
                            import ast

                            dims_list.append(ast.literal_eval(dims_value))
                        except:
                            dims_list.append([0, 1])

        # Get base parameters
        params = self._get_operation_suite_parameters(operation_name, configs, all_cases, deduplicate_inputs)

        # Add dims parameter
        if dims_list:
            params["dims"] = dims_list

        return params

    def _get_nlp_create_qkv_heads_suite_parameters(
        self, operation_name: str, configs: List, all_cases: bool, deduplicate_inputs: bool = False
    ) -> Dict:
        """Get parameters for nlp_create_qkv_heads operation which requires num_q_heads and num_kv_heads parameters"""
        try:
            paired_configs = []
            failed_configs = 0

            for config_idx, (config, source, machine_info) in enumerate(configs):
                try:
                    # Extract input tensor (arg0)
                    tensor_config = None
                    num_q_heads = None
                    num_kv_heads = None

                    for arg in config:
                        if isinstance(arg, dict):
                            if "arg0" in arg:
                                tensor_config = self.extract_tensor_config(arg["arg0"])
                            if "arg2" in arg:
                                # arg2 is num_q_heads
                                num_q_heads_val = arg["arg2"]
                                if isinstance(num_q_heads_val, (int, str)) and num_q_heads_val != "nullopt":
                                    try:
                                        num_q_heads = int(num_q_heads_val)
                                    except (ValueError, TypeError):
                                        pass
                            if "arg3" in arg:
                                # arg3 is num_kv_heads
                                num_kv_heads_val = arg["arg3"]
                                if isinstance(num_kv_heads_val, (int, str)) and num_kv_heads_val != "nullopt":
                                    try:
                                        num_kv_heads = int(num_kv_heads_val)
                                    except (ValueError, TypeError):
                                        pass

                    if not tensor_config:
                        failed_configs += 1
                        continue

                    # Infer num_q_heads and num_kv_heads from input shape if not provided
                    if tensor_config.shape and len(tensor_config.shape) >= 4:
                        hidden_dim = tensor_config.shape[3]  # [B, 1, S, hidden_dim]

                        # If num_kv_heads is None but num_q_heads is provided, assume MHA
                        if num_kv_heads is None and num_q_heads is not None:
                            # MHA: hidden_dim = 3 * num_q_heads * head_dim
                            head_dim = hidden_dim // (3 * num_q_heads)
                            num_kv_heads = num_q_heads
                        # If both are None, try to infer from hidden_dim
                        elif num_q_heads is None and num_kv_heads is None:
                            # Common pattern: assume head_dim=64 and MHA
                            # hidden_dim = 3 * num_heads * head_dim
                            # Try common head_dim values: 64, 128, 32
                            for head_dim in [64, 128, 32, 96]:
                                if hidden_dim % (3 * head_dim) == 0:
                                    num_q_heads = hidden_dim // (3 * head_dim)
                                    num_kv_heads = num_q_heads
                                    break

                    # Parse tensor config
                    parsed_dtype = self.parse_dtype(tensor_config.dtype)
                    parsed_layout = self.parse_layout(tensor_config.layout)
                    parsed_mem_config = self.parse_memory_config(tensor_config.memory_config, tensor_config.shape)

                    # Extract output memory config from arg5 if present
                    output_mem_config = parsed_mem_config
                    for arg in config:
                        if isinstance(arg, dict) and "arg5" in arg:
                            mem_config_data = arg["arg5"]
                            if isinstance(mem_config_data, dict) and "MemoryConfig" in mem_config_data:
                                output_mem_config = self.parse_memory_config(
                                    mem_config_data["MemoryConfig"], tensor_config.shape
                                )
                                break

                    if parsed_dtype and parsed_layout and parsed_mem_config:
                        config_dict = {
                            "shape": tensor_config.shape,
                            "dtype": parsed_dtype,
                            "layout": parsed_layout,
                            "memory_config": parsed_mem_config,
                            "output_memory_config": output_mem_config,
                            "num_q_heads": num_q_heads,
                            "num_kv_heads": num_kv_heads,
                            "traced_source": source,
                            "traced_machine_info": machine_info,
                        }
                        paired_configs.append(config_dict)

                except Exception as e:
                    failed_configs += 1
                    continue

            if paired_configs:
                print(f"✅ Loaded {len(paired_configs)} traced configurations for {operation_name} (model_traced suite)")

                # Build parameter dict
                param_names = [
                    "input_shape",
                    "input_a_dtype",
                    "input_a_layout",
                    "input_a_memory_config",
                    "num_q_heads",
                    "num_kv_heads",
                    "output_memory_config",
                    "traced_source",
                ]
                param_lists = [
                    [cfg["shape"] for cfg in paired_configs],
                    [cfg["dtype"] for cfg in paired_configs],
                    [cfg["layout"] for cfg in paired_configs],
                    [cfg["memory_config"] for cfg in paired_configs],
                    [cfg["num_q_heads"] for cfg in paired_configs],
                    [cfg["num_kv_heads"] for cfg in paired_configs],
                    [cfg["output_memory_config"] for cfg in paired_configs],
                    [cfg.get("traced_source", "unknown") for cfg in paired_configs],
                ]

                # Create tuples of exact configurations
                exact_configs = list(zip(*param_lists))
                param_key = ",".join(param_names)

                return {param_key: exact_configs}

            return {}
        except Exception as e:
            print(f"❌ Error extracting nlp_create_qkv_heads parameters: {e}")
            import traceback

            traceback.print_exc()
            return {}

    def _get_nlp_create_qkv_heads_decode_suite_parameters(
        self, operation_name: str, configs: List, all_cases: bool, deduplicate_inputs: bool = False
    ) -> Dict:
        """Get parameters for nlp_create_qkv_heads_decode operation which requires num_heads and num_kv_heads parameters"""
        try:
            paired_configs = []
            failed_configs = 0

            for config_idx, (config, source, machine_info) in enumerate(configs):
                try:
                    # Extract input tensor (arg0)
                    tensor_config = None
                    num_heads = None
                    num_kv_heads = None

                    for arg in config:
                        if isinstance(arg, dict):
                            if "arg0" in arg:
                                tensor_config = self.extract_tensor_config(arg["arg0"])
                            if "arg1" in arg:
                                # arg1 is num_q_heads (called num_heads in function)
                                num_heads_val = arg["arg1"]
                                if isinstance(num_heads_val, (int, str)) and num_heads_val != "nullopt":
                                    try:
                                        num_heads = int(num_heads_val)
                                    except (ValueError, TypeError):
                                        pass
                            if "arg2" in arg:
                                # arg2 is num_kv_heads
                                num_kv_heads_val = arg["arg2"]
                                if isinstance(num_kv_heads_val, (int, str)) and num_kv_heads_val != "nullopt":
                                    try:
                                        num_kv_heads = int(num_kv_heads_val)
                                    except (ValueError, TypeError):
                                        pass

                    if not tensor_config:
                        failed_configs += 1
                        continue
                    # Allow None values for num_heads and num_kv_heads - test files will infer them

                    # Parse tensor config
                    parsed_dtype = self.parse_dtype(tensor_config.dtype)
                    parsed_layout = self.parse_layout(tensor_config.layout)
                    parsed_mem_config = self.parse_memory_config(tensor_config.memory_config, tensor_config.shape)

                    # Extract output memory config from arg6 if present
                    output_mem_config = parsed_mem_config
                    for arg in config:
                        if isinstance(arg, dict) and "arg6" in arg:
                            mem_config_data = arg["arg6"]
                            if isinstance(mem_config_data, dict) and "MemoryConfig" in mem_config_data:
                                output_mem_config = self.parse_memory_config(
                                    mem_config_data["MemoryConfig"], tensor_config.shape
                                )
                                break

                    if parsed_dtype and parsed_layout and parsed_mem_config:
                        config_dict = {
                            "shape": tensor_config.shape,
                            "dtype": parsed_dtype,
                            "layout": parsed_layout,
                            "memory_config": parsed_mem_config,
                            "output_memory_config": output_mem_config,
                            "num_heads": num_heads,
                            "num_kv_heads": num_kv_heads,
                            "traced_source": source,
                            "traced_machine_info": machine_info,
                        }
                        paired_configs.append(config_dict)

                except Exception as e:
                    failed_configs += 1
                    continue

            if paired_configs:
                print(f"✅ Loaded {len(paired_configs)} traced configurations for {operation_name} (model_traced suite)")

                # Build parameter dict
                param_names = [
                    "input_shape",
                    "input_a_dtype",
                    "input_a_layout",
                    "input_a_memory_config",
                    "num_heads",
                    "num_kv_heads",
                    "output_memory_config",
                    "traced_source",
                ]
                param_lists = [
                    [cfg["shape"] for cfg in paired_configs],
                    [cfg["dtype"] for cfg in paired_configs],
                    [cfg["layout"] for cfg in paired_configs],
                    [cfg["memory_config"] for cfg in paired_configs],
                    [cfg["num_heads"] for cfg in paired_configs],
                    [cfg["num_kv_heads"] for cfg in paired_configs],
                    [cfg["output_memory_config"] for cfg in paired_configs],
                    [cfg.get("traced_source", "unknown") for cfg in paired_configs],
                ]

                # Create tuples of exact configurations
                exact_configs = list(zip(*param_lists))
                param_key = ",".join(param_names)

                return {param_key: exact_configs}

            return {}
        except Exception as e:
            print(f"❌ Error extracting nlp_create_qkv_heads_decode parameters: {e}")
            import traceback

            traceback.print_exc()
            return {}

    def _get_create_qkv_heads_suite_parameters(
        self, operation_name: str, configs: List, all_cases: bool, deduplicate_inputs: bool = False
    ) -> Dict:
        """Get parameters for create_qkv_heads operation which requires num_heads and num_kv_heads parameters

        Argument mapping for ttnn::experimental::create_qkv_heads:
        - arg0: input tensor
        - arg1: num_heads
        - arg2: num_kv_heads
        - arg3: transpose_k_heads
        - arg4: output memory config
        """
        try:
            paired_configs = []
            failed_configs = 0

            for config_idx, (config, source, machine_info) in enumerate(configs):
                try:
                    # Extract input tensor (arg0)
                    tensor_config = None
                    num_heads = None
                    num_kv_heads = None
                    transpose_k_heads = False  # Default to False

                    for arg in config:
                        if isinstance(arg, dict):
                            if "arg0" in arg:
                                tensor_config = self.extract_tensor_config(arg["arg0"])
                            if "arg1" in arg:
                                # arg1 is num_heads
                                num_heads_val = arg["arg1"]
                                if isinstance(num_heads_val, (int, str)) and num_heads_val != "nullopt":
                                    try:
                                        num_heads = (
                                            int(num_heads_val.strip('"'))
                                            if isinstance(num_heads_val, str)
                                            else num_heads_val
                                        )
                                    except (ValueError, TypeError, AttributeError):
                                        pass
                            if "arg2" in arg:
                                # arg2 is num_kv_heads
                                num_kv_heads_val = arg["arg2"]
                                if isinstance(num_kv_heads_val, (int, str)) and num_kv_heads_val != "nullopt":
                                    try:
                                        num_kv_heads = (
                                            int(num_kv_heads_val.strip('"'))
                                            if isinstance(num_kv_heads_val, str)
                                            else num_kv_heads_val
                                        )
                                    except (ValueError, TypeError, AttributeError):
                                        pass
                            if "arg3" in arg:
                                # arg3 is transpose_k_heads (boolean as int)
                                transpose_k_heads_val = arg["arg3"]
                                if isinstance(transpose_k_heads_val, (int, str)) and transpose_k_heads_val != "nullopt":
                                    try:
                                        transpose_k_heads = bool(int(transpose_k_heads_val))
                                    except (ValueError, TypeError):
                                        pass

                    if tensor_config and num_heads is not None and num_kv_heads is not None:
                        paired_config = {
                            "input_shape": tensor_config.shape,
                            "input_a_dtype": tensor_config.dtype,
                            "input_a_layout": tensor_config.layout,
                            "input_a_memory_config": tensor_config.memory_config,
                            "output_memory_config": tensor_config.memory_config,  # Default to input's memory config
                            "num_heads": num_heads,
                            "num_kv_heads": num_kv_heads,
                            "transpose_k_heads": transpose_k_heads,
                            "traced_source": source or "unknown",
                            "traced_machine_info": machine_info or {},
                        }
                        paired_configs.append(paired_config)
                    else:
                        failed_configs += 1

                except Exception as e:
                    failed_configs += 1
                    continue

            if failed_configs > 0:
                print(f"⚠️  Failed to parse {failed_configs}/{len(configs)} configs for {operation_name}")

            print(f"✅ Loaded {len(paired_configs)} traced configurations for {operation_name} (model_traced suite)")

            # Build parameter lists
            input_shape_list = []
            input_a_dtype_list = []
            input_a_layout_list = []
            input_a_memory_config_list = []
            output_memory_config_list = []
            num_heads_list = []
            num_kv_heads_list = []
            transpose_k_heads_list = []
            traced_source_list = []
            traced_machine_info_list = []

            for cfg in paired_configs:
                input_shape_list.append(cfg["input_shape"])
                # Parse dtype/layout strings to ttnn objects
                input_a_dtype_list.append(self.parse_dtype(cfg["input_a_dtype"]))
                input_a_layout_list.append(self.parse_layout(cfg["input_a_layout"]))
                input_a_memory_config_list.append(cfg["input_a_memory_config"])
                output_memory_config_list.append(cfg["output_memory_config"])
                num_heads_list.append(cfg["num_heads"])
                num_kv_heads_list.append(cfg["num_kv_heads"])
                transpose_k_heads_list.append(cfg["transpose_k_heads"])
                traced_source_list.append(cfg["traced_source"])
                traced_machine_info_list.append(cfg["traced_machine_info"])

            if all_cases:
                # For all_cases, Cartesian product (but for model_traced, it's typically 1:1)
                param_key = "all_test_cases"
            else:
                # For exact configs, zip the parameters
                param_key = "exact_test_cases"

            if paired_configs:
                # Create comma-separated parameter key and zip all parameters together
                param_names = [
                    "input_shape",
                    "input_a_dtype",
                    "input_a_layout",
                    "input_a_memory_config",
                    "output_memory_config",
                    "num_heads",
                    "num_kv_heads",
                    "transpose_k_heads",
                    "traced_source",
                    "traced_machine_info",
                ]
                param_lists = [
                    input_shape_list,
                    input_a_dtype_list,
                    input_a_layout_list,
                    input_a_memory_config_list,
                    output_memory_config_list,
                    num_heads_list,
                    num_kv_heads_list,
                    transpose_k_heads_list,
                    traced_source_list,
                    traced_machine_info_list,
                ]

                # Create tuples of exact configurations (prevents Cartesian product)
                exact_configs = list(zip(*param_lists))
                param_key = ",".join(param_names)

                print(f"   📊 Will generate {len(paired_configs)} test vectors (unique inputs)")

                return {param_key: exact_configs}

            return {}
        except Exception as e:
            print(f"❌ Error extracting create_qkv_heads parameters: {e}")
            import traceback

            traceback.print_exc()
            return {}

    def _get_scaled_dot_product_attention_suite_parameters(
        self, operation_name: str, configs: List, all_cases: bool, deduplicate_inputs: bool = False
    ) -> Dict:
        """Get parameters for scaled_dot_product_attention operation with is_causal and scale parameters"""
        try:
            paired_configs = []
            failed_configs = 0
            seen_input_signatures = set() if deduplicate_inputs else None

            for config_idx, (config, source, machine_info) in enumerate(configs):
                try:
                    # Extract Q, K, V tensor configs (first 3 args)
                    tensor_configs = []
                    for arg in config:
                        tensor_config = self.extract_tensor_config(arg)
                        if tensor_config:
                            tensor_configs.append(tensor_config)
                            if len(tensor_configs) >= 3:
                                break

                    if len(tensor_configs) < 3:
                        failed_configs += 1
                        continue

                    # Extract scalar parameters: is_causal (arg4) and scale (arg5)
                    is_causal = True  # Default
                    scale = None  # Will be calculated if None

                    for arg in config:
                        if isinstance(arg, dict):
                            # arg4 is is_causal (boolean as int)
                            if "arg4" in arg:
                                is_causal_val = arg["arg4"]
                                if isinstance(is_causal_val, (int, str)) and is_causal_val != "nullopt":
                                    try:
                                        is_causal = bool(int(is_causal_val))
                                    except ValueError:
                                        pass
                            # arg5 is scale (float)
                            if "arg5" in arg:
                                scale_val = arg["arg5"]
                                if isinstance(scale_val, (int, float, str)) and scale_val != "nullopt":
                                    try:
                                        scale = float(scale_val)
                                    except ValueError:
                                        pass

                    # Parse tensor configs
                    config_dict = {
                        "input_shape": {
                            "input_a": tensor_configs[0].shape,
                            "input_b": tensor_configs[1].shape,
                            "input_c": tensor_configs[2].shape,
                        },
                        "input_a_dtype": self.parse_dtype(tensor_configs[0].dtype),
                        "input_a_layout": self.parse_layout(tensor_configs[0].layout),
                        "input_a_memory_config": self.parse_memory_config(
                            tensor_configs[0].memory_config, tensor_configs[0].shape
                        ),
                        "input_b_dtype": self.parse_dtype(tensor_configs[1].dtype),
                        "input_b_layout": self.parse_layout(tensor_configs[1].layout),
                        "input_b_memory_config": self.parse_memory_config(
                            tensor_configs[1].memory_config, tensor_configs[1].shape
                        ),
                        "input_c_dtype": self.parse_dtype(tensor_configs[2].dtype),
                        "input_c_layout": self.parse_layout(tensor_configs[2].layout),
                        "input_c_memory_config": self.parse_memory_config(
                            tensor_configs[2].memory_config, tensor_configs[2].shape
                        ),
                        "output_memory_config": self.parse_memory_config(
                            tensor_configs[0].memory_config, tensor_configs[0].shape
                        ),  # Use Q's config
                        "is_causal": is_causal,
                        "scale": scale,
                        "traced_source": source,
                        "traced_machine_info": machine_info,
                    }

                    if deduplicate_inputs:
                        import hashlib

                        input_sig = hashlib.md5(
                            str(
                                (
                                    tensor_configs[0].shape,
                                    tensor_configs[1].shape,
                                    tensor_configs[2].shape,
                                    config_dict["input_a_dtype"],
                                    config_dict["input_b_dtype"],
                                    config_dict["input_c_dtype"],
                                    is_causal,
                                    scale,
                                )
                            ).encode()
                        ).hexdigest()
                        if input_sig in seen_input_signatures:
                            continue
                        seen_input_signatures.add(input_sig)

                    paired_configs.append(config_dict)
                except Exception as e:
                    failed_configs += 1
                    print(f"Error processing scaled_dot_product_attention config: {e}")
                    import traceback

                    traceback.print_exc()
                    continue

            if paired_configs:
                self.traced_configs_cache[operation_name] = paired_configs

                # Build parameter lists
                input_shapes = []
                input_a_dtypes = []
                input_a_layouts = []
                input_a_memory_configs = []
                input_b_dtypes = []
                input_b_layouts = []
                input_b_memory_configs = []
                input_c_dtypes = []
                input_c_layouts = []
                input_c_memory_configs = []
                output_memory_configs = []
                is_causal_list = []
                scale_list = []
                traced_source_list = []
                traced_machine_info_list = []

                for cfg in paired_configs:
                    input_shapes.append(cfg["input_shape"])
                    input_a_dtypes.append(cfg["input_a_dtype"])
                    input_a_layouts.append(cfg["input_a_layout"])
                    input_a_memory_configs.append(cfg["input_a_memory_config"])
                    input_b_dtypes.append(cfg["input_b_dtype"])
                    input_b_layouts.append(cfg["input_b_layout"])
                    input_b_memory_configs.append(cfg["input_b_memory_config"])
                    input_c_dtypes.append(cfg["input_c_dtype"])
                    input_c_layouts.append(cfg["input_c_layout"])
                    input_c_memory_configs.append(cfg["input_c_memory_config"])
                    output_memory_configs.append(cfg["output_memory_config"])
                    is_causal_list.append(cfg["is_causal"])
                    scale_list.append(cfg["scale"])
                    traced_source_list.append(cfg["traced_source"])
                    traced_machine_info_list.append(cfg["traced_machine_info"])

                param_names = [
                    "input_shape",
                    "input_a_dtype",
                    "input_a_layout",
                    "input_a_memory_config",
                    "input_b_dtype",
                    "input_b_layout",
                    "input_b_memory_config",
                    "input_c_dtype",
                    "input_c_layout",
                    "input_c_memory_config",
                    "output_memory_config",
                    "is_causal",
                    "scale",
                    "traced_source",
                    "traced_machine_info",
                ]
                param_lists = [
                    input_shapes,
                    input_a_dtypes,
                    input_a_layouts,
                    input_a_memory_configs,
                    input_b_dtypes,
                    input_b_layouts,
                    input_b_memory_configs,
                    input_c_dtypes,
                    input_c_layouts,
                    input_c_memory_configs,
                    output_memory_configs,
                    is_causal_list,
                    scale_list,
                    traced_source_list,
                    traced_machine_info_list,
                ]

                exact_configs = list(zip(*param_lists))
                param_key = ",".join(param_names)
                result = {param_key: exact_configs}

                print(f"✅ Loaded {len(paired_configs)} traced configurations for {operation_name} (model_traced suite)")
                dedup_msg = " (unique inputs)" if deduplicate_inputs else " (all input/output pairs)"
                valid_configs = len(input_shapes) if input_shapes else 0
                print(f"   📊 Will generate {valid_configs} test vectors{dedup_msg}")
                if failed_configs > 0:
                    print(f"⚠️ Failed to parse {failed_configs} configurations")
                return result
            return {}
        except Exception as e:
            print(f"❌ Error extracting scaled_dot_product_attention parameters: {e}")
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
            # arg12: groups, arg13: dtype, arg14: bias tensor (optional)
            # arg15: conv_config (unsupported type), arg16: compute_config

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
                    kernel = (
                        OperationParameterExtractors._parse_list_from_string(arg["arg8"])
                        if isinstance(arg["arg8"], str)
                        else arg["arg8"]
                    )
                    if kernel and len(kernel) >= 2:
                        params["kernel_height"] = kernel[0]
                        params["kernel_width"] = kernel[1]
                if "arg9" in arg:
                    stride = (
                        OperationParameterExtractors._parse_list_from_string(arg["arg9"])
                        if isinstance(arg["arg9"], str)
                        else arg["arg9"]
                    )
                    if stride and len(stride) >= 2:
                        params["stride_h"] = stride[0]
                        params["stride_w"] = stride[1]
                if "arg10" in arg:
                    padding = (
                        OperationParameterExtractors._parse_list_from_string(arg["arg10"])
                        if isinstance(arg["arg10"], str)
                        else arg["arg10"]
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
                        OperationParameterExtractors._parse_list_from_string(arg["arg11"])
                        if isinstance(arg["arg11"], str)
                        else arg["arg11"]
                    )
                    if dilation and len(dilation) >= 2:
                        params["dilation_h"] = dilation[0]
                        params["dilation_w"] = dilation[1]
                if "arg12" in arg:
                    params["groups"] = int(arg["arg12"]) if isinstance(arg["arg12"], (int, str)) else None
                if "arg13" in arg:
                    # Extract dtype (e.g., "DataType::BFLOAT8_B" -> "bfloat8_b")
                    dtype_str = str(arg["arg13"])
                    if "BFLOAT8_B" in dtype_str or "bfloat8_b" in dtype_str:
                        params["dtype"] = "bfloat8_b"
                    elif "BFLOAT16" in dtype_str or "bfloat16" in dtype_str:
                        params["dtype"] = "bfloat16"
                    elif "FLOAT32" in dtype_str or "float32" in dtype_str:
                        params["dtype"] = "float32"
                if "arg14" in arg and isinstance(arg["arg14"], dict):
                    # Bias tensor exists
                    params["has_bias"] = True
                if "arg16" in arg and isinstance(arg["arg16"], dict):
                    # Extract compute_config (WormholeComputeKernelConfig)
                    compute_config_dict = arg["arg16"]
                    if "WormholeComputeKernelConfig" in compute_config_dict:
                        wormhole_config = compute_config_dict["WormholeComputeKernelConfig"]
                        params["compute_config"] = {
                            "math_fidelity": wormhole_config.get("math_fidelity", "LoFi"),
                            "math_approx_mode": wormhole_config.get("math_approx_mode", 0),
                            "fp32_dest_acc_en": wormhole_config.get("fp32_dest_acc_en", 1),
                            "packer_l1_acc": wormhole_config.get("packer_l1_acc", 1),
                            "dst_full_sync_en": wormhole_config.get("dst_full_sync_en", 0),
                            "throttle_level": wormhole_config.get("throttle_level", "ThrottleLevel::NO_THROTTLE"),
                        }

            # Set has_bias to False if not found
            if "has_bias" not in params:
                params["has_bias"] = False

            # Set default dtype if not found
            if "dtype" not in params:
                params["dtype"] = "bfloat16"

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

    def extract_tensor_config(self, arg_data: Dict) -> Optional[TensorConfig]:
        """Extract tensor configuration from argument data

        Note: UnparsedElements are now fixed by the tracer's post-processing,
        so this method only handles already-clean data structures.
        """
        if not isinstance(arg_data, dict):
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

        # Extract layout from tensor_data (top level, added by graph tracer)
        layout_str = tensor_data.get("layout", "")
        if layout_str:
            # Clean up layout string: "Layout::ROW_MAJOR" -> "ROW_MAJOR"
            layout = layout_str.replace("Layout::", "")
        else:
            # Fallback: Determine layout (simplified - would need more logic for accurate detection)
            layout = "TILE"  # Default assumption for most ops

        # Extract storage_type from tensor_data (top level, added by graph tracer)
        storage_type = tensor_data.get("storage_type", "StorageType::DEVICE")
        if not storage_type:
            storage_type = "StorageType::DEVICE"  # Default to DEVICE

        return TensorConfig(
            shape=shape, dtype=dtype_str, layout=layout, memory_config=memory_config, storage_type=storage_type
        )

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
