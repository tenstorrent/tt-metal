# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

#!/usr/bin/env python3
"""
Master Configuration Loader for Sweep Tests

This module provides utilities to load real-world operation configurations
from the master JSON file and convert them into sweep test parameters.
"""

import json
import os
import sys
import ttnn
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
from tests.sweep_framework.framework.constants import LEAD_MODELS

# Set up logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

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
    storage_type: str = "StorageType::DEVICE"  # Default to DEVICE
    tensor_placement: Optional[Dict] = None  # Per-tensor placement info (PlacementShard/Replicate)


class MasterConfigLoader:
    """Loads and converts master JSON configurations to sweep test parameters

    Class Attributes:
        lead_models_only: When True, filters configurations to only include
            those from lead models (e.g., deepseek_v3). Set via set_lead_models_filter()
            before importing sweep modules that use this loader.
        _use_database: When True, loads configurations from PostgreSQL database
            instead of the JSON file. Set via set_database_mode().
        _mesh_filter: When set, filters configurations to a specific mesh shape
            at the database query level. Set via set_mesh_filter().
    """

    # Class-level filter setting (replaces environment variable approach)
    # This is set by sweeps_parameter_generator.py before importing sweep modules
    _lead_models_only: bool = False

    # Database mode settings (Phase 2)
    _use_database: bool = False

    # Mesh filter for server-side filtering (Phase 3)
    _mesh_filter: Optional[Tuple[int, int]] = None

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

    @classmethod
    def set_database_mode(cls, enabled: bool) -> None:
        """Enable or disable database loading mode.

        Args:
            enabled: If True, load configurations from PostgreSQL database.
                    If False, load from JSON file (default behavior).

        Note:
            Database mode requires TTNN_OPS_DATABASE_URL or POSTGRES_* environment
            variables to be set. If database connection fails, it will fall back
            to JSON file loading.
        """
        cls._use_database = enabled

    @classmethod
    def get_database_mode(cls) -> bool:
        """Get the current database mode setting."""
        return cls._use_database

    @classmethod
    def set_mesh_filter(cls, mesh_shape: Optional[Tuple[int, int]]) -> None:
        """Set mesh shape filter for server-side filtering.

        Args:
            mesh_shape: Tuple of (rows, cols) to filter configs by mesh shape,
                       e.g., (2, 4) for 2x4 mesh. Set to None to disable filtering.

        Note:
            This filter is applied at the database query level when in database mode,
            reducing data transfer.
        """
        cls._mesh_filter = mesh_shape

    @classmethod
    def get_mesh_filter(cls) -> Optional[Tuple[int, int]]:
        """Get the current mesh filter setting."""
        return cls._mesh_filter

    @staticmethod
    def _parse_list_from_string(s: str) -> List:
        """Simple helper to parse list strings like '[1, 2, 3]'"""
        try:
            import ast

            return ast.literal_eval(s)
        except (ValueError, SyntaxError, TypeError):
            # Fallback for simple cases
            if isinstance(s, list):
                return s
            return []

    @staticmethod
    def _extract_tensor_config(arg_data: Dict) -> Optional[TensorConfig]:
        """Extract tensor configuration from argument data"""
        if not isinstance(arg_data, dict):
            return None
        if arg_data.get("type") != "ttnn.Tensor":
            return None
        try:
            return TensorConfig(
                shape=arg_data.get("original_shape", arg_data.get("shape", [])),
                dtype=arg_data.get("original_dtype", arg_data.get("dtype", "")),
                layout=arg_data.get("layout", ""),
                memory_config=arg_data.get("memory_config", {}),
                storage_type=arg_data.get("storage_type", "StorageType.DEVICE"),
                tensor_placement=arg_data.get("tensor_placement"),  # Extract placement info
            )
        except Exception:
            return None

    @staticmethod
    def _extract_arguments(config_or_arguments: Dict) -> Dict[str, Any]:
        """Extract arguments from master JSON config

        Arguments use:
        - arg0, arg1, arg2, ... for positional arguments
        - Actual parameter names for keyword arguments (e.g., bias, memory_config, stride)

        Args:
            config_or_arguments: Either the full config dict (with 'arguments' key)
                                 or just the arguments dict directly
        """
        # Handle both full config and arguments dict
        if "arguments" in config_or_arguments:
            arguments = config_or_arguments["arguments"]
        else:
            # Already the arguments dict
            arguments = config_or_arguments

        extracted = {}

        for arg_name, arg_value in arguments.items():
            # Try to extract tensor config
            tensor_config = MasterConfigLoader._extract_tensor_config(arg_value)
            if tensor_config:
                extracted[arg_name] = tensor_config
            else:
                # Keep raw value for non-tensor arguments
                extracted[arg_name] = arg_value

        return extracted

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
        """Initialize the MasterConfigLoader for V2 format.

        Args:
            master_file_path: Explicit path to JSON file. If None, uses ttnn_operations_master_v2.json
        """
        if master_file_path is None:
            traced_dir = os.path.join(BASE_DIR, "model_tracer", "traced_operations")
            v2_path = os.path.join(traced_dir, "ttnn_operations_master_v2.json")
            reconstructed_v2_path = os.path.join(traced_dir, "ttnn_operations_master_v2_reconstructed.json")

            # Prefer reconstructed V2 if it exists (from database)
            if os.path.exists(reconstructed_v2_path):
                logger.info(f"✅ Using V2 reconstructed JSON from database: {reconstructed_v2_path}")
                master_file_path = reconstructed_v2_path
            elif os.path.exists(v2_path):
                logger.info(f"✅ Using V2 JSON: {v2_path}")
                master_file_path = v2_path
            else:
                logger.warning(f"⚠️  V2 JSON not found at {v2_path}")
                logger.warning(
                    f"   Please ensure ttnn_operations_master_v2.json exists in model_tracer/traced_operations/"
                )
                master_file_path = v2_path  # Set it anyway, will fail gracefully in load_master_data

        self.master_file_path = master_file_path
        self.master_data = None
        self.traced_configs_cache = {}  # Cache configs by operation name

    def load_master_data(self):
        """Load the master JSON file

        Note: If the file is not found or corrupted, we continue with empty data
        to allow the system to function in degraded mode (no traced configs).
        """
        if self.master_data is None:
            try:
                with open(self.master_file_path, "r") as f:
                    self.master_data = json.load(f)
                logger.info(f"✅ Loaded master data with {len(self.master_data.get('operations', {}))} operations")
            except FileNotFoundError:
                logger.error(f"❌ Master file not found: {self.master_file_path}")
                logger.warning(f"⚠️  Continuing with empty master data (degraded mode)")
                self.master_data = {"operations": {}}
            except json.JSONDecodeError as e:
                logger.error(f"❌ Error parsing master JSON: {e}")
                logger.warning(f"⚠️  Continuing with empty master data (degraded mode)")
                self.master_data = {"operations": {}}

    def get_operation_configs(self, operation_name: str) -> List[List[Dict]]:
        """Get all configurations for a specific operation"""
        self.load_master_data()

        # Try exact match first
        if operation_name in self.master_data.get("operations", {}):
            configs = self.master_data["operations"][operation_name].get("configurations", [])
            return self._normalize_configs(configs)

        # Try with ttnn. prefix and ttnn:: prefix
        for prefix in ["ttnn.", "ttnn::"]:
            ttnn_op_name = f"{prefix}{operation_name}"
            if ttnn_op_name in self.master_data.get("operations", {}):
                configs = self.master_data["operations"][ttnn_op_name].get("configurations", [])
                return self._normalize_configs(configs)

        # Try with both :: and . delimiters for experimental namespace
        # (e.g., experimental::create_qkv_heads or experimental.create_qkv_heads)
        for delimiter in ["::", "."]:
            experimental_full_op_name = f"ttnn{delimiter}experimental{delimiter}{operation_name}"
            if experimental_full_op_name in self.master_data.get("operations", {}):
                configs = self.master_data["operations"][experimental_full_op_name].get("configurations", [])
                return self._normalize_configs(configs)

        # Try with experimental:: or experimental. namespace (e.g., experimental::nlp_concat_heads)
        if operation_name.startswith("experimental::") or operation_name.startswith("experimental."):
            # Get the base name without experimental prefix
            if "::" in operation_name:
                base_name = operation_name.split("::", 1)[1]
            else:
                base_name = operation_name.split(".", 1)[1]

            # Try both delimiter formats
            for delimiter in ["::", "."]:
                experimental_op_name = f"ttnn{delimiter}experimental{delimiter}{base_name}"
                if experimental_op_name in self.master_data.get("operations", {}):
                    configs = self.master_data["operations"][experimental_op_name].get("configurations", [])
                    return self._normalize_configs(configs)

        # Try with both :: and . delimiters for transformer namespace
        # (e.g., transformer::scaled_dot_product_attention or transformer.scaled_dot_product_attention)
        for delimiter in ["::", "."]:
            transformer_op_name = f"ttnn{delimiter}transformer{delimiter}{operation_name}"
            if transformer_op_name in self.master_data.get("operations", {}):
                configs = self.master_data["operations"][transformer_op_name].get("configurations", [])
                return self._normalize_configs(configs)

        # Try with transformer:: or transformer. namespace if it starts with that prefix
        if operation_name.startswith("transformer::") or operation_name.startswith("transformer."):
            # Get the base name without transformer prefix
            if "::" in operation_name:
                base_name = operation_name.split("::", 1)[1]
            else:
                base_name = operation_name.split(".", 1)[1]

            # Try both delimiter formats
            for delimiter in ["::", "."]:
                transformer_op_name = f"ttnn{delimiter}transformer{delimiter}{base_name}"
                if transformer_op_name in self.master_data.get("operations", {}):
                    configs = self.master_data["operations"][transformer_op_name].get("configurations", [])
                    return self._normalize_configs(configs)

        # Try without prefix if it starts with ttnn:: or ttnn.
        if operation_name.startswith("ttnn::") or operation_name.startswith("ttnn."):
            prefix_len = 6  # Length of "ttnn::" or "ttnn."
            base_name = operation_name[prefix_len:]
            if base_name in self.master_data.get("operations", {}):
                configs = self.master_data["operations"][base_name].get("configurations", [])
                return self._normalize_configs(configs)
            # Also try with transformer:: and transformer. namespaces
            for delimiter in ["::", "."]:
                transformer_base = f"ttnn{delimiter}transformer{delimiter}{base_name}"
                if transformer_base in self.master_data.get("operations", {}):
                    configs = self.master_data["operations"][transformer_base].get("configurations", [])
                    return self._normalize_configs(configs)

        logger.warning(f"⚠️ No configurations found for operation: {operation_name}")
        return []

    def _normalize_configs(self, configs: List) -> List[Tuple[List[Dict], str, Any, str]]:
        """
        Normalize configurations to always return list of (argument list, source, machine_info, config_hash) tuples.
        Handles both old format (list) and new format (dict with source or contexts).

        Args:
            configs: List of configurations (dict with 'arguments' and 'source')

        Returns:
            List of (arguments, source, machine_info, config_hash) tuples for traceability
        """
        # Check if we should filter for lead models only
        # Uses class-level setting instead of environment variable for cleaner control
        lead_models_only = MasterConfigLoader._lead_models_only

        normalized = []
        for config in configs:
            if isinstance(config, dict) and "arguments" in config:
                arguments = config["arguments"]
                config_hash = config.get("config_hash", None)

                # Check for new "executions" format (explicit source/machine pairs with counts)
                if "executions" in config:
                    # Newest format: expand each execution into separate tuples
                    for execution in config["executions"]:
                        source = execution.get("source", "unknown")
                        machine_info = execution.get("machine_info", None)
                        # count is tracked but not passed to sweep tests

                        # Filter for lead models if requested
                        if lead_models_only:
                            if not self._source_matches_lead_models(source):
                                continue  # Skip this execution

                        normalized.append((arguments, source, machine_info, config_hash))

                # Check for mid-level "contexts" format (multiple execution contexts)
                elif "contexts" in config:
                    # Contexts format: expand each context into separate tuples
                    for context in config["contexts"]:
                        # Extract source (should be a list)
                        source_list = context.get("source", ["unknown"])
                        source = source_list[0] if isinstance(source_list, list) and len(source_list) > 0 else "unknown"

                        # Extract machine_info
                        machine_info = context.get("machine_info", None)

                        # Filter for lead models if requested
                        if lead_models_only:
                            if not self._source_matches_lead_models(source_list):
                                continue  # Skip this context

                        normalized.append((arguments, source, machine_info, config_hash))

                else:
                    # Old format: single source/machine_info (pairing may be lost)
                    source = config.get("source", "unknown")
                    machine_info = config.get("machine_info", None)

                    # Filter for lead models if requested
                    if lead_models_only:
                        if not self._source_matches_lead_models(source):
                            continue  # Skip this config

                    normalized.append((arguments, source, machine_info, config_hash))
            elif isinstance(config, list):
                # Legacy list format: use as-is with unknown source
                # Skip if lead_models_only since we can't determine source
                if not lead_models_only:
                    normalized.append((config, "unknown", None, None))
            else:
                # Fallback: wrap in list with unknown source and no machine_info
                normalized.append((config if isinstance(config, list) else [config], "unknown", None, None))
        return normalized

    def parse_dtype(self, dtype_str: str) -> Any:
        """Convert dtype string to ttnn dtype"""
        dtype_mapping = {
            # Legacy tracer format (namespace-style)
            "DataType::BFLOAT16": ttnn.bfloat16,
            "DataType::FLOAT32": ttnn.float32,
            "DataType::INT32": ttnn.int32,
            "DataType::UINT32": ttnn.uint32,
            "DataType::BFLOAT8_B": ttnn.bfloat8_b,
            "DataType::UINT16": ttnn.uint16,
            # V2 tracer format (dot-style)
            "DataType.BFLOAT16": ttnn.bfloat16,
            "DataType.FLOAT32": ttnn.float32,
            "DataType.INT32": ttnn.int32,
            "DataType.UINT32": ttnn.uint32,
            "DataType.BFLOAT8_B": ttnn.bfloat8_b,
            "DataType.UINT16": ttnn.uint16,
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
        if not mem_config:
            return False

        # Trust traced configs - no validation needed
        # Traced configs come from real model runs that worked
        return True

    def parse_memory_config(self, memory_config: Dict, tensor_shape: list = None) -> Any:
        """Convert memory config dict to ttnn memory config

        Parses both INTERLEAVED and SHARDED memory configs from the master JSON.
        Now properly handles shard_spec with grid, shape, and orientation.

        Raises exceptions on parsing errors instead of falling back to defaults,
        making it easier to identify and fix issues.

        Args:
            memory_config: Memory config dictionary from master JSON
            tensor_shape: Tensor shape (not used currently)

        Raises:
            ValueError: If memory config is invalid or incomplete
        """
        # If empty or missing, return default
        if not memory_config or not isinstance(memory_config, dict):
            return ttnn.DRAM_MEMORY_CONFIG

        buffer_type = memory_config.get("buffer_type")
        memory_layout = memory_config.get("memory_layout")

        # Validate required fields
        if not buffer_type:
            raise ValueError(f"Missing buffer_type in memory_config: {memory_config}")
        if not memory_layout:
            raise ValueError(f"Missing memory_layout in memory_config: {memory_config}")

        # Map buffer types - fail if unknown
        if "DRAM" in buffer_type:
            buffer_type_ttnn = ttnn.BufferType.DRAM
        elif "L1" in buffer_type:
            buffer_type_ttnn = ttnn.BufferType.L1
        else:
            raise ValueError(f"Unknown buffer_type: {buffer_type}")

        # Parse INTERLEAVED configs
        if "INTERLEAVED" in memory_layout:
            memory_layout_ttnn = ttnn.TensorMemoryLayout.INTERLEAVED
            return ttnn.MemoryConfig(memory_layout_ttnn, buffer_type_ttnn)

        # Parse SHARDED configs
        elif "SHARDED" in memory_layout:
            # Parse shard_spec - REQUIRED for sharded configs
            shard_spec_dict = memory_config.get("shard_spec")
            if not shard_spec_dict or not isinstance(shard_spec_dict, dict):
                raise ValueError(
                    f"Sharded memory layout '{memory_layout}' requires shard_spec, " f"but got: {shard_spec_dict}"
                )

            # Extract grid, shape, and orientation from shard_spec
            grid_list = shard_spec_dict.get("grid")
            shard_shape = shard_spec_dict.get("shape")
            orientation_str = shard_spec_dict.get("orientation")

            # Validate required shard_spec fields
            if not grid_list:
                raise ValueError(f"Missing 'grid' in shard_spec: {shard_spec_dict}")
            if not shard_shape:
                raise ValueError(f"Missing 'shape' in shard_spec: {shard_spec_dict}")
            if not orientation_str:
                raise ValueError(f"Missing 'orientation' in shard_spec: {shard_spec_dict}")

            # Create CoreRangeSet from grid
            # grid is a list of ranges like [{"start": {"x": 0, "y": 0}, "end": {"x": 7, "y": 7}}]
            core_ranges = set()
            for range_dict in grid_list:
                start = range_dict.get("start")
                end = range_dict.get("end")

                if not start or not end:
                    raise ValueError(f"Invalid grid range (missing start/end): {range_dict}")
                if "x" not in start or "y" not in start:
                    raise ValueError(f"Invalid grid start (missing x/y): {start}")
                if "x" not in end or "y" not in end:
                    raise ValueError(f"Invalid grid end (missing x/y): {end}")

                core_range = ttnn.CoreRange(ttnn.CoreCoord(start["x"], start["y"]), ttnn.CoreCoord(end["x"], end["y"]))
                core_ranges.add(core_range)

            shard_grid = ttnn.CoreRangeSet(core_ranges)

            # Map orientation
            if orientation_str == "COL_MAJOR":
                orientation = ttnn.ShardOrientation.COL_MAJOR
            elif orientation_str == "ROW_MAJOR":
                orientation = ttnn.ShardOrientation.ROW_MAJOR
            else:
                raise ValueError(f"Unknown orientation: {orientation_str}")

            # Create ShardSpec
            shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, orientation)

            # Map memory layout
            if "WIDTH_SHARDED" in memory_layout:
                memory_layout_ttnn = ttnn.TensorMemoryLayout.WIDTH_SHARDED
            elif "HEIGHT_SHARDED" in memory_layout:
                memory_layout_ttnn = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
            elif "BLOCK_SHARDED" in memory_layout:
                memory_layout_ttnn = ttnn.TensorMemoryLayout.BLOCK_SHARDED
            else:
                raise ValueError(f"Unknown sharded layout: {memory_layout}")

            # Create and return sharded memory config
            return ttnn.MemoryConfig(memory_layout_ttnn, buffer_type_ttnn, shard_spec)

        else:
            raise ValueError(f"Unknown memory_layout: {memory_layout}")

    @staticmethod
    def parse_special_float(value):
        """Parse special float string values like "inf", "-inf", "nan" to Python floats"""
        if isinstance(value, str):
            if value in ["inf", "Infinity"]:
                return float("inf")
            elif value in ["-inf", "-Infinity"]:
                return float("-inf")
            elif value in ["nan", "NaN"]:
                return float("nan")
        return value

    @staticmethod
    def parse_enum_value(value, default=None):
        """Parse enum types that were captured as {'type': 'EnumName'} during tracing.

        For enums without captured values, we return a sensible default.
        Currently handles:
        - Topology: defaults to Linear
        """
        if isinstance(value, dict) and "type" in value:
            enum_type = value["type"]

            # Check if there's a 'value' field with the actual enum value
            if "value" in value:
                enum_value = value["value"]

                # Parse based on type
                if enum_type == "Topology":
                    if "Linear" in enum_value or "LINEAR" in enum_value:
                        return ttnn.Topology.Linear
                    elif "Ring" in enum_value or "RING" in enum_value:
                        return ttnn.Topology.Ring
                    else:
                        return default if default is not None else ttnn.Topology.Linear
            else:
                # No value field - use type-based defaults
                if enum_type == "Topology":
                    # Default to Linear topology (most common, safest)
                    return default if default is not None else ttnn.Topology.Linear

        # Not an enum dict, return as-is
        return value

    def _get_generic_parameters(self, operation_name: str, configs: List) -> Dict:
        """Generic parameter extraction for all operations.

        Simple logic:
        - If argument is a tensor → extract tensor parameters
        - Otherwise → pass through as-is (with special float handling)
        """
        traced_config_list = []

        for config_args, source, machine_info, config_hash in configs:
            try:
                config_dict = {}
                positional_tensors = []

                # Separate positional args (arg0, arg1, ...) from named kwargs
                positional_args = {}
                named_kwargs = {}

                for key, value in config_args.items():
                    if key.startswith("arg") and key[3:].isdigit():
                        positional_args[key] = value
                    else:
                        named_kwargs[key] = value

                # Process positional tensor arguments (arg0, arg1, arg2, ...)
                # These become input_a_*, input_b_*, input_c_*, ...
                arg_idx = 0
                while f"arg{arg_idx}" in positional_args:
                    arg_value = positional_args[f"arg{arg_idx}"]

                    # Try to extract as tensor
                    tensor_config = self._extract_tensor_config(arg_value)
                    if tensor_config:
                        # It's a tensor - parse and store
                        parsed_dtype = self.parse_dtype(tensor_config.dtype)
                        parsed_layout = self.parse_layout(tensor_config.layout)
                        parsed_mem_config = self.parse_memory_config(tensor_config.memory_config, tensor_config.shape)

                        # Skip this config if memory_config parsing returned None
                        # (happens with mesh-sharded tensors missing grid info)
                        if parsed_mem_config is None:
                            raise ValueError(
                                f"Memory config parsing returned None (likely mesh-sharded tensor without grid)"
                            )

                        positional_tensors.append(
                            {
                                "shape": tuple(tensor_config.shape),
                                "dtype": parsed_dtype,
                                "layout": parsed_layout,
                                "memory_config": parsed_mem_config,
                                "tensor_placement": tensor_config.tensor_placement,
                            }
                        )
                    else:
                        # Positional arg that's not a tensor (rare) - store with arg name
                        config_dict[f"arg{arg_idx}"] = self.parse_special_float(arg_value)

                    arg_idx += 1

            except Exception as e:
                # Add config_hash context to error
                config_hash_display = config_hash[:16] + "..." if config_hash else "unknown"
                logger.warning(f"⚠️ Skipping config_hash={config_hash_display} due to error: {e}")
                continue

            # Skip if no tensors found
            if not positional_tensors:
                continue

            # Add positional tensor parameters with consistent naming
            # Always use input_a_, input_b_, input_c_ pattern (even for unary)
            for i, tensor in enumerate(positional_tensors):
                suffix = chr(97 + i)  # a, b, c, ...
                config_dict[f"input_{suffix}_shape"] = tensor["shape"]
                config_dict[f"input_{suffix}_dtype"] = tensor["dtype"]
                config_dict[f"input_{suffix}_layout"] = tensor["layout"]
                config_dict[f"input_{suffix}_memory_config"] = tensor["memory_config"]
                config_dict[f"input_{suffix}_tensor_placement"] = tensor.get("tensor_placement")

            config_dict["output_memory_config"] = positional_tensors[0]["memory_config"]

            # Process named keyword arguments
            # Named tensor kwargs preserve their semantic names (e.g., weight, bias_tensor)
            # Scalar/bool kwargs are passed as-is
            for key, value in named_kwargs.items():
                # Check if it's a tensor kwarg
                tensor_config = self._extract_tensor_config(value)
                if tensor_config:
                    # Named tensor kwarg - use its name (e.g., weight_shape, bias_shape)
                    parsed_dtype = self.parse_dtype(tensor_config.dtype)
                    parsed_layout = self.parse_layout(tensor_config.layout)
                    parsed_mem_config = self.parse_memory_config(tensor_config.memory_config, tensor_config.shape)

                    if parsed_mem_config is None:
                        logger.warning(f"⚠️ Skipping named tensor kwarg '{key}' due to unparseable memory_config")
                        continue

                    config_dict[f"{key}_shape"] = tuple(tensor_config.shape)
                    config_dict[f"{key}_dtype"] = parsed_dtype
                    config_dict[f"{key}_layout"] = parsed_layout
                    config_dict[f"{key}_memory_config"] = parsed_mem_config
                    config_dict[f"{key}_tensor_placement"] = tensor_config.tensor_placement
                else:
                    # Scalar/bool kwarg - pass as-is (preserve name)
                    # Try enum parsing first, then float parsing
                    parsed_value = self.parse_enum_value(value)
                    if parsed_value == value:  # If enum parsing didn't change it, try float
                        parsed_value = self.parse_special_float(value)
                    config_dict[key] = parsed_value

            # Add metadata
            config_dict["traced_source"] = source
            config_dict["traced_machine_info"] = machine_info
            config_dict["config_hash"] = config_hash

            traced_config_list.append(config_dict)

        if not traced_config_list:
            return {"traced_config": []}

        logger.info(f"✅ Loaded {len(traced_config_list)} traced configurations for {operation_name}")

        # Collect all unique keys across all configs
        all_keys = []
        seen_keys = set()
        for cfg in traced_config_list:
            for k in cfg.keys():
                if k not in seen_keys:
                    all_keys.append(k)
                    seen_keys.add(k)

        # Build tuples with all parameters
        param_values = [[cfg.get(k) for k in all_keys] for cfg in traced_config_list]
        return {",".join(all_keys): [tuple(v) for v in param_values]}

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
                logger.warning(f"⚠️ No traced configurations found for {operation_name}")
                # Return empty lists - sweep tests will handle defaults
                return {
                    "input_shape": [[1, 32, 32]],
                    "input_a_dtype": [],
                    "input_a_layout": [],
                    "input_a_memory_config": [],
                    "output_memory_config": [],
                }

            # Use single generic extraction for all operations
            return self._get_generic_parameters(operation_name, configs)

        except Exception as e:
            logger.error(f"❌ Error loading configurations for {operation_name}: {e}")
            logger.error(f"   This error occurred while processing one or more configs.")
            logger.error(f"   Check logs above for specific config_hash that failed.")
            import traceback

            traceback.print_exc()
            return {"traced_config_name": []}

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
                                # Skip if memory_config parsing returned None
                                if memory_config_ttnn is None:
                                    raise ValueError(
                                        f"Memory config parsing returned None (likely mesh-sharded tensor without grid)"
                                    )
                                if memory_config_ttnn not in input_memory_configs:
                                    input_memory_configs.append(memory_config_ttnn)

            except Exception as e:
                logger.warning(f"⚠️ Error processing config: {e}")
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

        logger.info(f"✅ Generated sweep parameters for {operation_name}:")
        logger.info(f"   • {len(input_shapes)} unique input shapes")
        logger.info(f"   • {len(input_dtypes)} unique dtypes")
        logger.info(f"   • {len(input_layouts)} unique layouts")
        logger.info(f"   • {len(input_memory_configs)} unique memory configs")

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
        logger.error(f"❌ No master configurations found for {operation_name}")
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


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    loader = MasterConfigLoader()

    # Test with add operation
    add_params = loader.convert_to_sweep_parameters("add")
    logger.info(f"\n📊 Master-based parameters for 'add': {len(add_params)} suites")

    # Test with transpose operation
    transpose_params = loader.convert_to_sweep_parameters("transpose")
    logger.info(f"\n📊 Master-based parameters for 'transpose': {len(transpose_params)} suites")
