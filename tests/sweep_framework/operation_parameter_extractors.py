# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Operation-specific parameter extractors for model traced sweep tests.

This module provides a registry system for operation-specific parameter extraction.
Users can easily register new operation extractors by providing the operation name
and extraction functions.

Usage:
    from operation_parameter_extractors import OperationParameterExtractors

    # Register a new operation extractor
    OperationParameterExtractors.register_extractor(
        "my_operation",
        extract_func=my_extract_function,
        transform_func=my_transform_function
    )

    # Use the extractor
    params = OperationParameterExtractors.extract_parameters("my_operation", config)
    transformed = OperationParameterExtractors.transform_parameters("my_operation", configs)
"""

from typing import Dict, List, Any, Optional, Callable
import json
import re


class TensorConfig:
    """Represents a tensor configuration extracted from master JSON"""

    def __init__(self, shape: List[int], dtype: str, layout: str, memory_config: Dict):
        self.shape = shape
        self.dtype = dtype
        self.layout = layout
        self.memory_config = memory_config


class OperationParameterExtractors:
    """Registry and collection of parameter extraction methods for different operations"""

    # Registry for operation extractors
    _extractors = {}
    _transformers = {}

    @classmethod
    def register_extractor(
        cls,
        operation_name: str,
        extract_func: Optional[Callable[[List], Optional[Dict]]] = None,
        transform_func: Optional[Callable[[List], List[Dict]]] = None,
    ):
        """Register extraction and transformation functions for an operation

        Args:
            operation_name: Name of the operation (e.g., 'embedding', 'linear')
            extract_func: Function to extract parameters from a single config
            transform_func: Function to transform raw configs to sweep format
        """
        if extract_func:
            cls._extractors[operation_name] = extract_func
        if transform_func:
            cls._transformers[operation_name] = transform_func

    @classmethod
    def extract_parameters(cls, operation_name: str, config: List) -> Optional[Dict]:
        """Extract parameters for a specific operation"""
        if operation_name in cls._extractors:
            return cls._extractors[operation_name](config)
        return None

    @classmethod
    def transform_parameters(
        cls, operation_name: str, configs: List, parse_dtype=None, parse_layout=None, parse_memory_config=None
    ) -> List[Dict]:
        """Transform parameters for a specific operation

        Args:
            operation_name: Name of the operation
            configs: List of extracted parameter dicts
            parse_dtype: Optional function to parse dtype strings to TTNN types
            parse_layout: Optional function to parse layout strings to TTNN types
            parse_memory_config: Optional function to parse memory config dicts to TTNN MemoryConfig objects
        """
        if operation_name in cls._transformers:
            transformer = cls._transformers[operation_name]
            # Check if transformer accepts parser functions
            import inspect

            sig = inspect.signature(transformer)
            if len(sig.parameters) > 1:  # More than just 'configs'
                return transformer(
                    configs, parse_dtype=parse_dtype, parse_layout=parse_layout, parse_memory_config=parse_memory_config
                )
            else:
                return transformer(configs)
        return []

    @classmethod
    def list_registered_operations(cls) -> List[str]:
        """List all registered operations"""
        return list(set(cls._extractors.keys()) | set(cls._transformers.keys()))

    # Built-in extractors for common operations

    @staticmethod
    def _extract_embedding_parameters(config: List) -> Optional[Dict]:
        """Extract all parameters for embedding operation"""
        try:
            params = {}

            # Extract tensor configs from arg0 (indices) and arg1 (weights)
            indices_config = None
            weights_config = None

            for arg in config:
                if isinstance(arg, dict):
                    if "arg0" in arg:
                        indices_config = OperationParameterExtractors.extract_tensor_config(arg["arg0"])
                    if "arg1" in arg:
                        weights_config = OperationParameterExtractors.extract_tensor_config(arg["arg1"])

            if indices_config and weights_config:
                # Create input_shape dict format expected by transform
                params["input_shape"] = {"self": indices_config.shape, "other": weights_config.shape}
                params["indices_dtype"] = indices_config.dtype
                params["weights_dtype"] = weights_config.dtype
                params["indices_layout"] = indices_config.layout
                params["weights_layout"] = weights_config.layout
                params["indices_memory_config"] = indices_config.memory_config
                params["weights_memory_config"] = weights_config.memory_config

                # Extract output memory config from arg6 if present
                output_memory_config = None
                for arg in config:
                    if isinstance(arg, dict) and "arg6" in arg:
                        mem_config_data = arg["arg6"]
                        if isinstance(mem_config_data, dict) and "MemoryConfig" in mem_config_data:
                            output_memory_config = mem_config_data["MemoryConfig"]
                            break

                params["output_memory_config"] = output_memory_config or weights_config.memory_config

                return params

            return None
        except Exception as e:
            print(f"Error extracting embedding parameters: {e}")
            return None

    @staticmethod
    def _transform_embedding_parameters(
        configs: List, parse_dtype=None, parse_layout=None, parse_memory_config=None
    ) -> List[Dict]:
        """Transform embedding traced configs to run function format

        Args:
            configs: List of extracted parameter dicts
            parse_dtype: Optional function to parse dtype strings to TTNN types
            parse_layout: Optional function to parse layout strings to TTNN types
            parse_memory_config: Optional function to parse memory config dicts to TTNN MemoryConfig objects
        """
        transformed_configs = []

        for config in configs:
            try:
                if not isinstance(config, dict):
                    continue

                # Extract from the extracted params dict
                input_shape_dict = config.get("input_shape", {})
                if not input_shape_dict or "self" not in input_shape_dict or "other" not in input_shape_dict:
                    continue

                indices_shape = input_shape_dict["self"]
                weights_shape = input_shape_dict["other"]

                # Parse dtypes and layouts from the config
                indices_dtype_str = config.get("indices_dtype", "DataType::UINT32")
                weights_dtype_str = config.get("weights_dtype", "DataType::BFLOAT16")
                indices_layout_str = config.get("indices_layout", "Layout::TILE")
                weights_layout_str = config.get("weights_layout", "Layout::TILE")

                # Parse memory configs
                indices_mem_config = config.get("indices_memory_config", {})
                weights_mem_config = config.get("weights_memory_config", {})
                output_mem_config = config.get("output_memory_config", weights_mem_config)

                transformed_config = {
                    "input_shape": input_shape_dict,  # Keep as dict with 'self' and 'other'
                    "input_a_dtype": indices_dtype_str,
                    "input_b_dtype": weights_dtype_str,
                    "input_a_layout": indices_layout_str,
                    "input_b_layout": weights_layout_str,
                    "input_a_memory_config": indices_mem_config,
                    "input_b_memory_config": weights_mem_config,
                    "output_memory_config": output_mem_config,
                }

                # Apply parsers if provided
                if parse_dtype:
                    transformed_config["input_a_dtype"] = parse_dtype(indices_dtype_str)
                    transformed_config["input_b_dtype"] = parse_dtype(weights_dtype_str)
                if parse_layout:
                    transformed_config["input_a_layout"] = parse_layout(indices_layout_str)
                    transformed_config["input_b_layout"] = parse_layout(weights_layout_str)
                if parse_memory_config:
                    transformed_config["input_a_memory_config"] = parse_memory_config(indices_mem_config, indices_shape)
                    transformed_config["input_b_memory_config"] = parse_memory_config(weights_mem_config, weights_shape)
                    transformed_config["output_memory_config"] = parse_memory_config(output_mem_config, weights_shape)

                transformed_configs.append(transformed_config)

            except Exception as e:
                print(f"Error transforming embedding config: {e}")
                continue

        return transformed_configs

    @staticmethod
    def _extract_linear_parameters(config: List) -> Optional[Dict]:
        """Extract all parameters for linear operation"""
        try:
            params = {}

            # Extract transpose flags (arg3: transpose_a, arg4: transpose_b)
            for arg in config:
                if not isinstance(arg, dict):
                    continue
                if "arg3" in arg:
                    transpose_a_val = arg["arg3"]
                    params["transpose_a"] = (
                        bool(int(transpose_a_val))
                        if isinstance(transpose_a_val, (int, str)) and transpose_a_val != "nullopt"
                        else False
                    )
                if "arg4" in arg:
                    transpose_b_val = arg["arg4"]
                    params["transpose_b"] = (
                        bool(int(transpose_b_val))
                        if isinstance(transpose_b_val, (int, str)) and transpose_b_val != "nullopt"
                        else False
                    )

            # Extract tensor shapes from the traced config
            # arg0: input tensor, arg1: weight tensor, arg2: bias tensor (optional)
            tensor_shapes = []

            # Extract from arg0 (input tensor)
            for arg in config:
                if isinstance(arg, dict) and "arg0" in arg:
                    tensor_config = OperationParameterExtractors.extract_tensor_config(arg["arg0"])
                    if tensor_config:
                        tensor_shapes.append(tensor_config.shape)
                        break

            # Extract from arg1 (weight tensor) - this might be UnparsedElement
            # In the traced config, arg1 might be in a dict with "arg1" key, or directly as UnparsedElement
            for arg in config:
                if isinstance(arg, dict):
                    # Case 1: {"arg1": {...}}
                    if "arg1" in arg:
                        tensor_config = OperationParameterExtractors.extract_tensor_config(arg["arg1"])
                        if tensor_config:
                            tensor_shapes.append(tensor_config.shape)
                            break
                    # Case 2: {"UnparsedElement": {...}} - this might be arg1
                    elif "UnparsedElement" in arg:
                        tensor_config = OperationParameterExtractors.extract_tensor_config(arg)
                        if tensor_config:
                            tensor_shapes.append(tensor_config.shape)
                            break

            # Check for bias (arg2)
            has_bias = False
            for arg in config:
                if isinstance(arg, dict) and "arg2" in arg:
                    bias_val = arg["arg2"]
                    if bias_val != "nullopt" and bias_val is not None:
                        has_bias = True
                        # For bias, we typically expect [out_features] shape
                        # This is derived from weight shape, we'll handle this in transform
                    break

            if len(tensor_shapes) >= 2:
                input_shape = tensor_shapes[0]
                weight_shape = tensor_shapes[1]

                params["input_shape"] = input_shape
                params["weight_shape"] = weight_shape
                params["bias_shape"] = None  # Will be computed in transform
                params["has_bias"] = has_bias

                return params

            return None
        except Exception as e:
            print(f"Error extracting linear parameters: {e}")
            return None

    @staticmethod
    def _transform_linear_parameters(
        configs: List, parse_dtype=None, parse_layout=None, parse_memory_config=None
    ) -> List[Dict]:
        """Transform linear traced configs to run function format

        Args:
            configs: List of extracted parameter dicts
            parse_dtype: Optional function to parse dtype strings to TTNN types
            parse_layout: Optional function to parse layout strings to TTNN types
            parse_memory_config: Optional function to parse memory config dicts/strings to TTNN MemoryConfig objects
        """
        transformed_configs = []

        for config in configs:
            try:
                # config is expected to be the params dict from _extract_linear_parameters
                if not isinstance(config, dict):
                    continue

                input_shape = config["input_shape"]
                weight_shape = config["weight_shape"]
                has_bias = config["has_bias"]
                transpose_a = config.get("transpose_a", False)
                transpose_b = config.get("transpose_b", False)

                # For linear operations, TTNN can handle 4D tensors directly
                # The shapes follow matmul semantics: input[..., m, k] @ weight[..., k, n] -> result[..., m, n]
                # So we keep the original shapes from tracing

                processed_input_shape = input_shape
                processed_weight_shape = weight_shape

                # For bias, if present, it should match the output features (last dim of weight)
                bias_shape = None
                if has_bias:
                    # For matmul semantics, bias should match the last dimension of the weight tensor
                    # weight shape is [..., k, n], so bias should be [n]
                    if len(processed_weight_shape) >= 2:
                        bias_shape = [processed_weight_shape[-1]]
                    else:
                        bias_shape = None

                # Get memory configs - handle both string and dict formats
                input_a_mem_cfg = config.get("input_a_memory_config", "MemoryConfig.INTERLEAVED")
                input_b_mem_cfg = config.get("input_b_memory_config", "MemoryConfig.INTERLEAVED")
                output_mem_cfg = config.get("output_memory_config", "MemoryConfig.INTERLEAVED")

                # Convert string memory configs to dict format if needed
                if isinstance(input_a_mem_cfg, str):
                    if "INTERLEAVED" in input_a_mem_cfg:
                        input_a_mem_cfg = {
                            "memory_layout": "TensorMemoryLayout::INTERLEAVED",
                            "buffer_type": "BufferType::DRAM",
                        }
                    elif "WIDTH_SHARDED" in input_a_mem_cfg:
                        input_a_mem_cfg = {
                            "memory_layout": "TensorMemoryLayout::WIDTH_SHARDED",
                            "buffer_type": "BufferType::L1",
                        }
                    elif "HEIGHT_SHARDED" in input_a_mem_cfg:
                        input_a_mem_cfg = {
                            "memory_layout": "TensorMemoryLayout::HEIGHT_SHARDED",
                            "buffer_type": "BufferType::L1",
                        }
                    else:
                        input_a_mem_cfg = {
                            "memory_layout": "TensorMemoryLayout::INTERLEAVED",
                            "buffer_type": "BufferType::DRAM",
                        }

                if isinstance(input_b_mem_cfg, str):
                    if "INTERLEAVED" in input_b_mem_cfg:
                        input_b_mem_cfg = {
                            "memory_layout": "TensorMemoryLayout::INTERLEAVED",
                            "buffer_type": "BufferType::DRAM",
                        }
                    elif "WIDTH_SHARDED" in input_b_mem_cfg:
                        input_b_mem_cfg = {
                            "memory_layout": "TensorMemoryLayout::WIDTH_SHARDED",
                            "buffer_type": "BufferType::L1",
                        }
                    elif "HEIGHT_SHARDED" in input_b_mem_cfg:
                        input_b_mem_cfg = {
                            "memory_layout": "TensorMemoryLayout::HEIGHT_SHARDED",
                            "buffer_type": "BufferType::L1",
                        }
                    else:
                        input_b_mem_cfg = {
                            "memory_layout": "TensorMemoryLayout::INTERLEAVED",
                            "buffer_type": "BufferType::DRAM",
                        }

                if isinstance(output_mem_cfg, str):
                    if "INTERLEAVED" in output_mem_cfg:
                        output_mem_cfg = {
                            "memory_layout": "TensorMemoryLayout::INTERLEAVED",
                            "buffer_type": "BufferType::DRAM",
                        }
                    elif "WIDTH_SHARDED" in output_mem_cfg:
                        output_mem_cfg = {
                            "memory_layout": "TensorMemoryLayout::WIDTH_SHARDED",
                            "buffer_type": "BufferType::L1",
                        }
                    elif "HEIGHT_SHARDED" in output_mem_cfg:
                        output_mem_cfg = {
                            "memory_layout": "TensorMemoryLayout::HEIGHT_SHARDED",
                            "buffer_type": "BufferType::L1",
                        }
                    else:
                        output_mem_cfg = {
                            "memory_layout": "TensorMemoryLayout::INTERLEAVED",
                            "buffer_type": "BufferType::DRAM",
                        }

                transformed_config = {
                    "input_shape": processed_input_shape,
                    "weight_shape": processed_weight_shape,
                    "bias_shape": bias_shape,
                    "input_a_dtype": config.get("input_a_dtype", "DataType.BFLOAT16"),
                    "input_b_dtype": config.get("input_b_dtype", "DataType.BFLOAT16"),
                    "input_a_layout": config.get("input_a_layout", "Layout.TILE"),
                    "input_b_layout": config.get("input_b_layout", "Layout.TILE"),
                    "input_a_memory_config": input_a_mem_cfg,
                    "input_b_memory_config": input_b_mem_cfg,
                    "output_memory_config": output_mem_cfg,
                    "transpose_a": transpose_a,
                    "transpose_b": transpose_b,
                    "has_bias": has_bias,
                }

                # Apply parsers if provided
                if parse_dtype:
                    transformed_config["input_a_dtype"] = parse_dtype(transformed_config["input_a_dtype"])
                    transformed_config["input_b_dtype"] = parse_dtype(transformed_config["input_b_dtype"])
                if parse_layout:
                    transformed_config["input_a_layout"] = parse_layout(transformed_config["input_a_layout"])
                    transformed_config["input_b_layout"] = parse_layout(transformed_config["input_b_layout"])
                if parse_memory_config:
                    transformed_config["input_a_memory_config"] = parse_memory_config(input_a_mem_cfg, input_shape)
                    transformed_config["input_b_memory_config"] = parse_memory_config(input_b_mem_cfg, weight_shape)
                    transformed_config["output_memory_config"] = parse_memory_config(output_mem_cfg, input_shape)

                transformed_configs.append(transformed_config)

            except Exception as e:
                print(f"Error transforming linear config: {e}")
                continue

        return transformed_configs

    @staticmethod
    def _extract_conv2d_parameters(config: List) -> Optional[Dict]:
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
                # Extract parameters from the config - this is a simplified version
                # The full implementation would parse all the conv2d parameters
                pass

            return params if params else None
        except Exception:
            return None

    # Helper methods for parameter extraction
    @staticmethod
    def _parse_list_from_string(value) -> Optional[List]:
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
                    return json.loads(value.replace("'", '"'))
            return None
        except Exception:
            return None

    @staticmethod
    def _parse_numeric_value(value):
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
                    parsed = OperationParameterExtractors._parse_list_from_string(value)
                    if parsed is not None:
                        return parsed
                # Try as float
                if "." in value:
                    return float(value)
                # Try as int
                return int(value)
            return None
        except Exception:
            return None

    @staticmethod
    def _extract_permute_dims(config: List) -> Optional[List[int]]:
        """Extract dimensions for permute operation"""
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
                    elif isinstance(dims_str, list):
                        return dims_str
            return None
        except Exception:
            return None

    @staticmethod
    def _extract_shape_parameter(config: List, arg_name: str = "arg1") -> Optional[List[int]]:
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
                        parsed = OperationParameterExtractors._parse_list_from_string(shape_data)
                        if parsed is not None and isinstance(parsed, list):
                            return parsed
                    # Handle direct list
                    elif isinstance(shape_data, list):
                        return shape_data
            return None
        except Exception:
            return None

    @staticmethod
    def _extract_int_parameter(config: List, arg_name: str) -> Optional[int]:
        """Extract integer parameter from config (e.g., for transpose dim0, dim1)"""
        try:
            for arg in config:
                if isinstance(arg, dict) and arg_name in arg:
                    value = arg[arg_name]
                    if isinstance(value, (int, str)):
                        return int(value)
            return None
        except Exception:
            return None

    @staticmethod
    def extract_tensor_vector_from_unparsed(unparsed_data: Dict) -> Optional[List[Dict]]:
        """Extract vector of tensors from UnparsedElement for concat operations

        Args:
            unparsed_data: The UnparsedElement dictionary containing element_info

        Returns:
            List of tensor dictionaries, or None if extraction fails
        """
        element_info = unparsed_data.get("element_info", "")
        if not element_info:
            return None

        try:
            # Extract the JSON array string from element_info
            # Format: {"arg0": "[{tensor1}, {tensor2}, ...]"}
            array_match = re.search(r'"arg0"\s*:\s*"(\[.*\])"', element_info, re.DOTALL)
            if array_match:
                array_str = array_match.group(1)
                # Parse the JSON array
                tensor_array = json.loads(array_str)

                # Extract tensor information from each tensor in the array
                tensor_configs = []
                for tensor_obj in tensor_array:
                    if "tensor_spec" in tensor_obj:
                        tensor_configs.append(tensor_obj)

                return tensor_configs if tensor_configs else None
        except Exception as e:
            # If parsing fails, return None
            return None
        return None

    @staticmethod
    def extract_tensor_config(arg_data: Dict) -> Optional[TensorConfig]:
        """Extract tensor configuration from argument data"""
        if not isinstance(arg_data, dict):
            return None

        # Handle UnparsedElement by parsing its element_info string
        if "UnparsedElement" in arg_data:
            unparsed_data = arg_data["UnparsedElement"]
            element_info = unparsed_data.get("element_info", "")

            if element_info and element_info.startswith("{"):
                try:
                    # Check if this is a vector of tensors (for concat)
                    # Format: {"arg0": "[{tensor1}, {tensor2}, ...]"}
                    if '"arg0"' in element_info and element_info.find("[") != -1:
                        # This might be a vector - try extracting it
                        tensor_vector = OperationParameterExtractors.extract_tensor_vector_from_unparsed(unparsed_data)
                        if tensor_vector and len(tensor_vector) > 0:
                            # Return the first tensor's config (for single tensor extraction)
                            # For vector extraction, use extract_tensor_vector_from_unparsed directly
                            tensor_obj = tensor_vector[0]
                            tensor_spec = tensor_obj.get("tensor_spec", {})
                            tensor_layout = tensor_spec.get("tensor_layout", {})
                            shape = tensor_spec.get("logical_shape", [])
                            dtype = tensor_layout.get("dtype", "")
                            layout = tensor_layout.get("layout", "")
                            memory_config = tensor_layout.get("memory_config", {})
                            if not layout:
                                layout = "Layout::TILE"
                            if shape and dtype and layout and memory_config:
                                return TensorConfig(shape, dtype, layout, memory_config)

                    # Apply regex fixes for C++ style formats
                    fixed_json_str = element_info
                    # Fix C++ style braces in values like "{32, 32}" -> "[32, 32]"
                    fixed_json_str = re.sub(r':\s*"{\s*([^}]+)\s*}"', r': "[\1]"', fixed_json_str)
                    # Fix grid format: "grid":{[...], [...]} -> "grid":[[...], [...]]
                    fixed_json_str = re.sub(
                        r'"grid"\s*:\s*\{(\[.*?\](?:\s*,\s*\[.*?\])*)\}', r'"grid":[\1]', fixed_json_str
                    )
                    # Fix grid ranges like {"x":0,"y":0} - {"x":7,"y":7} -> {"x":0,"y":0}, {"x":7,"y":7}
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
                except Exception:
                    return None

        # Handle nested structure like {arg0: {Tensor: ...}} or {arg1: {Tensor: ...}}
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

        # Extract basic information
        shape = tensor_spec.get("logical_shape", [])
        # Handle both 'data_type' and 'dtype' fields
        dtype = tensor_layout.get("data_type", tensor_layout.get("dtype", ""))
        layout = tensor_layout.get("layout", "")
        memory_config = tensor_layout.get("memory_config", {})

        # If layout is missing, default to TILE for linear operations
        if not layout:
            layout = "Layout::TILE"

        if shape and dtype and layout and memory_config:
            return TensorConfig(shape, dtype, layout, memory_config)

        return None

    # Unary operation extractors
    @staticmethod
    def _extract_permute_parameters(config: List) -> Optional[Dict]:
        """Extract parameters for permute operation"""
        try:
            dims = OperationParameterExtractors._extract_permute_dims(config)
            if dims:
                return {"dims": dims}
            # Fallback to default if extraction fails
            return {"dims": [0, 1, 3, 2]}  # N, C, W, H -> N, C, H, W
        except Exception:
            return None

    @staticmethod
    def _extract_untilize_with_unpadding_parameters(config: List) -> Optional[Dict]:
        """Extract parameters for untilize_with_unpadding operation"""
        try:
            end_shape = OperationParameterExtractors._extract_shape_parameter(config, arg_name="arg1")
            if end_shape:
                return {"end_shape": end_shape}
            return None
        except Exception:
            return None

    @staticmethod
    def _extract_nlp_concat_heads_decode_parameters(config: List) -> Optional[Dict]:
        """Extract parameters for nlp_concat_heads_decode operation"""
        try:
            num_heads = None
            for arg in config:
                if isinstance(arg, dict) and "arg1" in arg:
                    num_heads_val = arg["arg1"]
                    if isinstance(num_heads_val, (int, str)) and num_heads_val != "nullopt":
                        try:
                            num_heads = int(num_heads_val)
                        except:
                            pass
                    break

            # Try to infer from tensor shape if available
            if num_heads is None:
                # Extract tensor config from arg0
                tensor_config = None
                for arg in config:
                    if isinstance(arg, dict) and "arg0" in arg:
                        tensor_config = OperationParameterExtractors.extract_tensor_config(arg["arg0"])
                        break

                if tensor_config:
                    # Try to infer from shape [B, 1, H, D] -> H might be num_heads
                    if len(tensor_config.shape) == 4 and tensor_config.shape[1] == 1:
                        num_heads = tensor_config.shape[2]  # Use shape[2] as num_heads
                    else:
                        num_heads = 16  # Default fallback
                else:
                    num_heads = 16  # Default fallback

            return {"num_heads": num_heads}
        except Exception:
            return {"num_heads": 16}  # Default fallback

    @staticmethod
    def _extract_transpose_parameters(config: List) -> Optional[Dict]:
        """Extract parameters for transpose operation"""
        try:
            dim0 = OperationParameterExtractors._extract_int_parameter(config, "arg1")
            dim1 = OperationParameterExtractors._extract_int_parameter(config, "arg2")
            if dim0 is not None and dim1 is not None:
                return {"dim0": dim0, "dim1": dim1}
            return None
        except Exception:
            return None

    @staticmethod
    def _extract_reshape_parameters(config: List) -> Optional[Dict]:
        """Extract parameters for reshape operation"""
        try:
            target_shape = OperationParameterExtractors._extract_shape_parameter(config, arg_name="arg1")
            if target_shape:
                # Extract tensor config to validate reshape
                tensor_config = None
                for arg in config:
                    if isinstance(arg, dict) and "arg0" in arg:
                        tensor_config = OperationParameterExtractors.extract_tensor_config(arg["arg0"])
                        break

                if tensor_config:
                    import math

                    input_elements = math.prod(tensor_config.shape) if tensor_config.shape else 0
                    # Handle -1 in target_shape (means infer from other dimensions)
                    if -1 in target_shape:
                        # Calculate what -1 should be
                        known_product = math.prod([d for d in target_shape if d != -1])
                        if known_product == 0:
                            # Invalid: cannot infer -1 with zero in other dimensions
                            return None
                        inferred_dim = input_elements // known_product
                        target_shape = [inferred_dim if d == -1 else d for d in target_shape]

                    target_elements = math.prod(target_shape) if target_shape else 0
                    if input_elements != target_elements:
                        # Invalid reshape config - skip it
                        return None

                return {"target_shape": target_shape}
            return None
        except Exception:
            return None

    @staticmethod
    def _extract_pad_parameters(config: List) -> Optional[Dict]:
        """Extract parameters for pad operation"""
        try:
            padding = None
            value = None
            for arg in config:
                if isinstance(arg, dict):
                    if "arg1" in arg:
                        padding = OperationParameterExtractors._parse_list_from_string(arg["arg1"])
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
                        value = OperationParameterExtractors._parse_numeric_value(arg["arg2"])
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
                return {"padding": padding, "value": value}
            return None
        except Exception:
            return None

    @staticmethod
    def _extract_paged_scaled_dot_product_attention_decode_parameters(config: List) -> Optional[Dict]:
        """Extract parameters for paged_scaled_dot_product_attention_decode operation

        Extracts:
        - Input tensor configs from arg0, arg1, arg2, arg3, arg6
        - Output memory config from arg10
        """
        try:
            params = {}

            # Extract input tensor configs
            input_configs = []
            for arg_idx in [0, 1, 2, 3, 6]:
                if len(config) > arg_idx:
                    arg = config[arg_idx]
                    if isinstance(arg, dict):
                        arg_key = f"arg{arg_idx}"
                        if arg_key in arg:
                            tensor_config = OperationParameterExtractors.extract_tensor_config(arg[arg_key])
                            if tensor_config:
                                input_configs.append(tensor_config)
                        elif "UnparsedElement" in arg:
                            # Handle UnparsedElement case (e.g., arg0)
                            tensor_config = OperationParameterExtractors.extract_tensor_config(arg)
                            if tensor_config:
                                input_configs.append(tensor_config)

            # Extract output memory config from arg10
            output_memory_config = None
            if len(config) > 10:
                arg10 = config[10]
                if isinstance(arg10, dict) and "arg10" in arg10:
                    arg10_data = arg10["arg10"]
                    if isinstance(arg10_data, dict) and "MemoryConfig" in arg10_data:
                        output_memory_config = arg10_data["MemoryConfig"]

            # Build params dict
            if input_configs:
                # Use first input's shape as primary input_shape (arg0 is the query tensor)
                params["input_shape"] = input_configs[0].shape if len(input_configs) > 0 else None
                params["input_a_dtype"] = input_configs[0].dtype.replace("DataType::", "")
                params["input_a_layout"] = input_configs[0].layout.replace("Layout::", "")
                params["input_a_memory_config"] = input_configs[0].memory_config

                # Extract other inputs if available, including their shapes
                if len(input_configs) > 1:
                    params["input_b_shape"] = input_configs[1].shape
                    params["input_b_dtype"] = input_configs[1].dtype.replace("DataType::", "")
                    params["input_b_layout"] = input_configs[1].layout.replace("Layout::", "")
                    params["input_b_memory_config"] = input_configs[1].memory_config
                if len(input_configs) > 2:
                    params["input_c_shape"] = input_configs[2].shape
                    params["input_c_dtype"] = input_configs[2].dtype.replace("DataType::", "")
                    params["input_c_layout"] = input_configs[2].layout.replace("Layout::", "")
                    params["input_c_memory_config"] = input_configs[2].memory_config
                if len(input_configs) > 3:
                    params["input_d_shape"] = input_configs[3].shape
                    params["input_d_dtype"] = input_configs[3].dtype.replace("DataType::", "")
                    params["input_d_layout"] = input_configs[3].layout.replace("Layout::", "")
                    params["input_d_memory_config"] = input_configs[3].memory_config
                if len(input_configs) > 4:
                    params["input_e_shape"] = input_configs[4].shape
                    params["input_e_dtype"] = input_configs[4].dtype.replace("DataType::", "")
                    params["input_e_layout"] = input_configs[4].layout.replace("Layout::", "")
                    params["input_e_memory_config"] = input_configs[4].memory_config

            if output_memory_config:
                params["output_memory_config"] = output_memory_config

            return params if params else None
        except Exception as e:
            import traceback

            print(f"Error extracting paged_scaled_dot_product_attention_decode parameters: {e}")
            traceback.print_exc()
            return None

    @staticmethod
    def _transform_paged_scaled_dot_product_attention_decode_parameters(
        configs: List[Dict],
        parse_dtype=None,
        parse_layout=None,
        parse_memory_config=None,
    ) -> List[Dict]:
        """Transform extracted paged_scaled_dot_product_attention_decode parameters to TTNN types"""
        transformed_configs = []

        for config in configs:
            try:
                transformed_config = {}

                # Transform input_shape (use dict format for multi-input)
                # Build input_shape dict from all available input shapes
                input_shape_dict = {}
                if "input_shape" in config:
                    # input_shape is the first input's shape
                    input_shape_dict["input_a"] = config["input_shape"]
                # Try to get other input shapes from the config if available
                # Note: The extractor extracts shapes from input_configs, but we need to reconstruct them
                # For now, we'll use the first input's shape for all inputs if others aren't available
                # This is a limitation - ideally we'd extract all shapes separately
                if "input_b_shape" in config:
                    input_shape_dict["input_b"] = config["input_b_shape"]
                if "input_c_shape" in config:
                    input_shape_dict["input_c"] = config["input_c_shape"]
                if "input_d_shape" in config:
                    input_shape_dict["input_d"] = config["input_d_shape"]
                if "input_e_shape" in config:
                    input_shape_dict["input_e"] = config["input_e_shape"]
                transformed_config["input_shape"] = input_shape_dict

                # Transform dtypes
                if parse_dtype:
                    transformed_config["input_a_dtype"] = parse_dtype(
                        f"DataType::{config.get('input_a_dtype', 'BFLOAT16')}"
                    )
                    if "input_b_dtype" in config:
                        transformed_config["input_b_dtype"] = parse_dtype(f"DataType::{config['input_b_dtype']}")
                    if "input_c_dtype" in config:
                        transformed_config["input_c_dtype"] = parse_dtype(f"DataType::{config['input_c_dtype']}")
                    if "input_d_dtype" in config:
                        transformed_config["input_d_dtype"] = parse_dtype(f"DataType::{config['input_d_dtype']}")
                    if "input_e_dtype" in config:
                        transformed_config["input_e_dtype"] = parse_dtype(f"DataType::{config['input_e_dtype']}")
                else:
                    transformed_config["input_a_dtype"] = config.get("input_a_dtype", "BFLOAT16")
                    if "input_b_dtype" in config:
                        transformed_config["input_b_dtype"] = config["input_b_dtype"]
                    if "input_c_dtype" in config:
                        transformed_config["input_c_dtype"] = config["input_c_dtype"]
                    if "input_d_dtype" in config:
                        transformed_config["input_d_dtype"] = config["input_d_dtype"]
                    if "input_e_dtype" in config:
                        transformed_config["input_e_dtype"] = config["input_e_dtype"]

                # Transform layouts
                if parse_layout:
                    transformed_config["input_a_layout"] = parse_layout(config.get("input_a_layout", "TILE"))
                    if "input_b_layout" in config:
                        transformed_config["input_b_layout"] = parse_layout(config["input_b_layout"])
                    if "input_c_layout" in config:
                        transformed_config["input_c_layout"] = parse_layout(config["input_c_layout"])
                    if "input_d_layout" in config:
                        transformed_config["input_d_layout"] = parse_layout(config["input_d_layout"])
                    if "input_e_layout" in config:
                        transformed_config["input_e_layout"] = parse_layout(config["input_e_layout"])
                else:
                    transformed_config["input_a_layout"] = config.get("input_a_layout", "TILE")
                    if "input_b_layout" in config:
                        transformed_config["input_b_layout"] = config["input_b_layout"]
                    if "input_c_layout" in config:
                        transformed_config["input_c_layout"] = config["input_c_layout"]
                    if "input_d_layout" in config:
                        transformed_config["input_d_layout"] = config["input_d_layout"]
                    if "input_e_layout" in config:
                        transformed_config["input_e_layout"] = config["input_e_layout"]

                # Transform memory configs
                if parse_memory_config:
                    input_shape = config.get("input_shape", [])
                    transformed_config["input_a_memory_config"] = parse_memory_config(
                        config.get("input_a_memory_config", {}), input_shape
                    )
                    if "input_b_memory_config" in config:
                        # Get shape from config if available, otherwise use input_shape
                        input_b_shape = config.get("input_b_shape", input_shape)
                        transformed_config["input_b_memory_config"] = parse_memory_config(
                            config["input_b_memory_config"], input_b_shape
                        )
                    if "input_c_memory_config" in config:
                        input_c_shape = config.get("input_c_shape", input_shape)
                        transformed_config["input_c_memory_config"] = parse_memory_config(
                            config["input_c_memory_config"], input_c_shape
                        )
                    if "input_d_memory_config" in config:
                        input_d_shape = config.get("input_d_shape", input_shape)
                        transformed_config["input_d_memory_config"] = parse_memory_config(
                            config["input_d_memory_config"], input_d_shape
                        )
                    if "input_e_memory_config" in config:
                        input_e_shape = config.get("input_e_shape", input_shape)
                        transformed_config["input_e_memory_config"] = parse_memory_config(
                            config["input_e_memory_config"], input_e_shape
                        )

                    # Transform output_memory_config
                    output_mem_config_dict = config.get("output_memory_config", {})
                    if output_mem_config_dict:
                        # Use input shape for output shape (approximation)
                        transformed_config["output_memory_config"] = parse_memory_config(
                            output_mem_config_dict, input_shape
                        )
                else:
                    transformed_config["input_a_memory_config"] = config.get("input_a_memory_config", {})
                    if "input_b_memory_config" in config:
                        transformed_config["input_b_memory_config"] = config["input_b_memory_config"]
                    if "input_c_memory_config" in config:
                        transformed_config["input_c_memory_config"] = config["input_c_memory_config"]
                    if "input_d_memory_config" in config:
                        transformed_config["input_d_memory_config"] = config["input_d_memory_config"]
                    if "input_e_memory_config" in config:
                        transformed_config["input_e_memory_config"] = config["input_e_memory_config"]
                    transformed_config["output_memory_config"] = config.get("output_memory_config", {})

                transformed_configs.append(transformed_config)
            except Exception as e:
                print(f"Error transforming paged_scaled_dot_product_attention_decode config: {e}")
                import traceback

                traceback.print_exc()
                continue

        return transformed_configs

    @staticmethod
    def _extract_tilize_with_val_padding_parameters(config: List) -> Optional[Dict]:
        """Extract parameters for tilize_with_val_padding operation"""
        try:
            padded_shape = OperationParameterExtractors._extract_shape_parameter(config, arg_name="arg1")
            pad_value = None
            for arg in config:
                if isinstance(arg, dict) and "arg2" in arg:
                    pad_value = OperationParameterExtractors._parse_numeric_value(arg["arg2"])
                    break

            if padded_shape and pad_value is not None:
                return {"padded_shape": padded_shape, "pad_value": pad_value}
            return None
        except Exception:
            return None

    @staticmethod
    def _extract_all_gather_async_parameters(config: List) -> Optional[Dict]:
        """Extract parameters for all_gather_async operation

        Handles UnparsedElement errors by extracting from element_info using regex.
        Extracts:
        - Input tensor config from arg0
        - Output memory config from arg5
        - dim from arg2
        - num_links from arg4
        """
        import json
        import re

        try:
            params = {}

            # Extract input tensor config from arg0 (handles UnparsedElement)
            input_shape = None
            input_dtype = None
            input_memory_config = None

            if len(config) > 0:
                arg0 = config[0]
                if isinstance(arg0, dict):
                    if "arg0" in arg0:
                        # Normal case - use extract_tensor_config
                        tensor_config = OperationParameterExtractors.extract_tensor_config(arg0["arg0"])
                        if tensor_config:
                            input_shape = tensor_config.shape
                            input_dtype = tensor_config.dtype.replace("DataType::", "")
                            input_memory_config = tensor_config.memory_config
                    elif "UnparsedElement" in arg0:
                        # UnparsedElement case - extract from element_info using regex
                        unparsed = arg0["UnparsedElement"]
                        element_info = unparsed.get("element_info", "")

                        # Try to use extract_tensor_config first (it handles UnparsedElement)
                        tensor_config = OperationParameterExtractors.extract_tensor_config(arg0)
                        if tensor_config:
                            input_shape = tensor_config.shape
                            input_dtype = tensor_config.dtype.replace("DataType::", "")
                            input_memory_config = tensor_config.memory_config
                        else:
                            # Fallback to regex extraction
                            shape_match = re.search(r'"logical_shape":\[([^\]]+)\]', element_info)
                            if shape_match:
                                try:
                                    input_shape = json.loads("[" + shape_match.group(1) + "]")
                                except:
                                    pass

                            dtype_match = re.search(r'"dtype":"DataType::([^"]+)"', element_info)
                            if dtype_match:
                                input_dtype = dtype_match.group(1)

                            # Extract memory config
                            if "memory_config" in element_info:
                                mem_layout_match = re.search(
                                    r'"memory_layout":"TensorMemoryLayout::([^"]+)"', element_info
                                )
                                buffer_type_match = re.search(r'"buffer_type":"BufferType::([^"]+)"', element_info)

                                input_memory_config = {}
                                if mem_layout_match:
                                    input_memory_config["memory_layout"] = mem_layout_match.group(1)
                                if buffer_type_match:
                                    input_memory_config["buffer_type"] = buffer_type_match.group(1)

                                # Extract shard_spec if present
                                if "shard_spec" in element_info and "nullopt" not in element_info:
                                    shard_match = re.search(r'"shard_spec":\{([^}]+)\}', element_info)
                                    if shard_match:
                                        shard_info = shard_match.group(1)
                                        input_memory_config["shard_spec"] = shard_info

            # Extract output memory config from arg5 (handles UnparsedElement)
            output_memory_config = None

            if len(config) > 5:
                arg5 = config[5]
                if isinstance(arg5, dict):
                    if "arg5" in arg5:
                        # Normal case - arg5 might be nested
                        mem_arg = arg5["arg5"]
                        if isinstance(mem_arg, dict):
                            if "MemoryConfig" in mem_arg:
                                output_memory_config = mem_arg["MemoryConfig"]
                            elif "memory_layout" in mem_arg or "buffer_type" in mem_arg:
                                # Already extracted format
                                output_memory_config = mem_arg
                            else:
                                # Try to find MemoryConfig deeper in the structure
                                import json

                                mem_arg_str = json.dumps(mem_arg)
                                if "MemoryConfig" in mem_arg_str:
                                    # Try to extract using regex
                                    mem_layout_match = re.search(
                                        r'"memory_layout":"TensorMemoryLayout::([^"]+)"', mem_arg_str
                                    )
                                    buffer_type_match = re.search(r'"buffer_type":"BufferType::([^"]+)"', mem_arg_str)
                                    if mem_layout_match or buffer_type_match:
                                        output_memory_config = {}
                                        if mem_layout_match:
                                            output_memory_config["memory_layout"] = mem_layout_match.group(1)
                                        if buffer_type_match:
                                            output_memory_config["buffer_type"] = buffer_type_match.group(1)
                    elif "UnparsedElement" in arg5:
                        # UnparsedElement case - extract from element_info
                        unparsed = arg5["UnparsedElement"]
                        element_info = unparsed.get("element_info", "")

                        if "MemoryConfig" in element_info:
                            mem_layout_match = re.search(r'"memory_layout":"TensorMemoryLayout::([^"]+)"', element_info)
                            buffer_type_match = re.search(r'"buffer_type":"BufferType::([^"]+)"', element_info)

                            output_memory_config = {}
                            if mem_layout_match:
                                output_memory_config["memory_layout"] = mem_layout_match.group(1)
                            if buffer_type_match:
                                output_memory_config["buffer_type"] = buffer_type_match.group(1)

                            # Extract shard_spec if present - handle nested braces
                            # Check for shard_spec specifically, not just absence of nullopt
                            shard_spec_start = element_info.find('"shard_spec":{')
                            if shard_spec_start != -1 and element_info.find('"shard_spec":"std::nullopt"') == -1:
                                # Find shard_spec start
                                shard_start = element_info.find('"shard_spec":{')
                                if shard_start != -1:
                                    # Find matching closing brace
                                    brace_count = 0
                                    start_pos = shard_start + len('"shard_spec":{')
                                    shard_spec_str = None
                                    for i in range(start_pos, len(element_info)):
                                        if element_info[i] == "{":
                                            brace_count += 1
                                        elif element_info[i] == "}":
                                            if brace_count == 0:
                                                shard_spec_str = element_info[
                                                    shard_start + len('"shard_spec":') : i + 1
                                                ]
                                                break
                                            brace_count -= 1

                                    if shard_spec_str:
                                        try:
                                            # Fix the " - " syntax in grid coordinates
                                            fixed_shard = re.sub(
                                                r'(\{"x":\d+,"y":\d+\})\s*-\s*(\{"x":\d+,"y":\d+\})',
                                                r"[\1, \2]",
                                                shard_spec_str,
                                            )
                                            # Fix shape format "{32, 64}" -> "[32, 64]"
                                            fixed_shard = re.sub(
                                                r'"shape":"\{(\d+),\s*(\d+)\}"', r'"shape":[\1, \2]', fixed_shard
                                            )
                                            # Parse as JSON
                                            shard_spec_dict = json.loads(fixed_shard)
                                            output_memory_config["shard_spec"] = shard_spec_dict
                                        except Exception as e:
                                            # Fallback: store as string for parse_memory_config to handle
                                            output_memory_config["shard_spec"] = shard_spec_str

            # Extract dim from arg2
            dim = OperationParameterExtractors._extract_int_parameter(config, "arg2")

            # Extract num_links from arg4
            num_links = OperationParameterExtractors._extract_int_parameter(config, "arg4")

            # Build params dict
            if input_shape:
                params["input_shape"] = input_shape
            if input_dtype:
                params["input_dtype"] = input_dtype
            if input_memory_config:
                params["input_memory_config"] = input_memory_config
            if output_memory_config:
                params["output_memory_config"] = output_memory_config
            if dim is not None:
                params["dim"] = dim
            if num_links is not None:
                params["num_links"] = num_links

            return params if params else None
        except Exception as e:
            import traceback

            print(f"Error extracting all_gather_async parameters: {e}")
            traceback.print_exc()
            return None

    @staticmethod
    def _transform_all_gather_async_parameters(
        configs: List[Dict],
        parse_dtype=None,
        parse_layout=None,
        parse_memory_config=None,
    ) -> List[Dict]:
        """Transform extracted all_gather_async parameters to TTNN types"""
        transformed_configs = []

        for config in configs:
            try:
                transformed_config = {}

                # Transform input_shape
                if "input_shape" in config:
                    transformed_config["input_shape"] = config["input_shape"]

                # Transform input_dtype
                input_dtype_str = config.get("input_dtype", "BFLOAT16")
                if parse_dtype:
                    transformed_config["input_dtype"] = parse_dtype(f"DataType::{input_dtype_str}")
                else:
                    transformed_config["input_dtype"] = input_dtype_str

                # Transform input_layout (default to TILE_LAYOUT)
                transformed_config["input_layout"] = config.get("input_layout", "TILE")

                # Transform input_memory_config
                input_mem_config_dict = config.get("input_memory_config", {})
                if input_mem_config_dict and parse_memory_config:
                    input_shape = config.get("input_shape")
                    transformed_config["input_memory_config"] = parse_memory_config(input_mem_config_dict, input_shape)
                else:
                    transformed_config["input_memory_config"] = input_mem_config_dict

                # Transform output_memory_config
                output_mem_config_dict = config.get("output_memory_config", {})
                if output_mem_config_dict and parse_memory_config:
                    # Output shape is input shape with width doubled (after gather)
                    input_shape = config.get("input_shape", [])
                    output_shape = input_shape.copy() if input_shape else []
                    if len(output_shape) >= 4:
                        output_shape[3] = output_shape[3] * 2  # Width doubles after gather

                    # Ensure memory_layout has full format if it's missing the prefix
                    if isinstance(output_mem_config_dict, dict):
                        if "memory_layout" in output_mem_config_dict:
                            mem_layout = output_mem_config_dict["memory_layout"]
                            if not mem_layout.startswith("TensorMemoryLayout::"):
                                output_mem_config_dict = output_mem_config_dict.copy()
                                output_mem_config_dict["memory_layout"] = f"TensorMemoryLayout::{mem_layout}"
                        if "buffer_type" in output_mem_config_dict:
                            buf_type = output_mem_config_dict["buffer_type"]
                            if not buf_type.startswith("BufferType::"):
                                output_mem_config_dict = output_mem_config_dict.copy()
                                output_mem_config_dict["buffer_type"] = f"BufferType::{buf_type}"

                    try:
                        transformed_config["output_memory_config"] = parse_memory_config(
                            output_mem_config_dict, output_shape
                        )
                    except Exception as e:
                        print(f"Warning: Failed to parse output_memory_config: {e}")
                        # Fallback to DRAM interleaved
                        transformed_config["output_memory_config"] = ttnn.DRAM_MEMORY_CONFIG
                else:
                    # Fallback to DRAM interleaved if no config provided
                    transformed_config["output_memory_config"] = (
                        ttnn.DRAM_MEMORY_CONFIG if parse_memory_config else output_mem_config_dict
                    )

                # Copy dim and num_links as-is
                if "dim" in config:
                    transformed_config["dim"] = config["dim"]
                if "num_links" in config:
                    transformed_config["num_links"] = config["num_links"]

                transformed_configs.append(transformed_config)
            except Exception as e:
                print(f"Error transforming all_gather_async config: {e}")
                continue

        return transformed_configs


# Register the built-in extractors
OperationParameterExtractors.register_extractor(
    "embedding",
    extract_func=OperationParameterExtractors._extract_embedding_parameters,
    transform_func=OperationParameterExtractors._transform_embedding_parameters,
)

OperationParameterExtractors.register_extractor(
    "linear",
    extract_func=OperationParameterExtractors._extract_linear_parameters,
    transform_func=OperationParameterExtractors._transform_linear_parameters,
)

OperationParameterExtractors.register_extractor(
    "conv2d", extract_func=OperationParameterExtractors._extract_conv2d_parameters
)

# Register unary operation extractors
OperationParameterExtractors.register_extractor(
    "permute",
    extract_func=OperationParameterExtractors._extract_permute_parameters,
)
OperationParameterExtractors.register_extractor(
    "ttnn::permute",
    extract_func=OperationParameterExtractors._extract_permute_parameters,
)

OperationParameterExtractors.register_extractor(
    "untilize_with_unpadding",
    extract_func=OperationParameterExtractors._extract_untilize_with_unpadding_parameters,
)

OperationParameterExtractors.register_extractor(
    "nlp_concat_heads_decode",
    extract_func=OperationParameterExtractors._extract_nlp_concat_heads_decode_parameters,
)
OperationParameterExtractors.register_extractor(
    "experimental::nlp_concat_heads_decode",
    extract_func=OperationParameterExtractors._extract_nlp_concat_heads_decode_parameters,
)
OperationParameterExtractors.register_extractor(
    "ttnn::experimental::nlp_concat_heads_decode",
    extract_func=OperationParameterExtractors._extract_nlp_concat_heads_decode_parameters,
)

OperationParameterExtractors.register_extractor(
    "transpose",
    extract_func=OperationParameterExtractors._extract_transpose_parameters,
)

OperationParameterExtractors.register_extractor(
    "reshape",
    extract_func=OperationParameterExtractors._extract_reshape_parameters,
)

OperationParameterExtractors.register_extractor(
    "pad",
    extract_func=OperationParameterExtractors._extract_pad_parameters,
)

OperationParameterExtractors.register_extractor(
    "tilize_with_val_padding",
    extract_func=OperationParameterExtractors._extract_tilize_with_val_padding_parameters,
)

# Register all_gather_async extractor
OperationParameterExtractors.register_extractor(
    "all_gather_async",
    extract_func=OperationParameterExtractors._extract_all_gather_async_parameters,
)
OperationParameterExtractors.register_extractor(
    "experimental::all_gather_async",
    extract_func=OperationParameterExtractors._extract_all_gather_async_parameters,
)
OperationParameterExtractors.register_extractor(
    "ttnn::experimental::all_gather_async",
    extract_func=OperationParameterExtractors._extract_all_gather_async_parameters,
)

# Register paged_scaled_dot_product_attention_decode extractor
OperationParameterExtractors.register_extractor(
    "paged_scaled_dot_product_attention_decode",
    extract_func=OperationParameterExtractors._extract_paged_scaled_dot_product_attention_decode_parameters,
    transform_func=OperationParameterExtractors._transform_paged_scaled_dot_product_attention_decode_parameters,
)
OperationParameterExtractors.register_extractor(
    "transformer::paged_scaled_dot_product_attention_decode",
    extract_func=OperationParameterExtractors._extract_paged_scaled_dot_product_attention_decode_parameters,
    transform_func=OperationParameterExtractors._transform_paged_scaled_dot_product_attention_decode_parameters,
)
OperationParameterExtractors.register_extractor(
    "ttnn::transformer::paged_scaled_dot_product_attention_decode",
    extract_func=OperationParameterExtractors._extract_paged_scaled_dot_product_attention_decode_parameters,
    transform_func=OperationParameterExtractors._transform_paged_scaled_dot_product_attention_decode_parameters,
)


# Example: How users can easily add their own operation extractors
def example_custom_operation_setup():
    """
    Example showing how users can easily add their own operation extractors.

    To add a new operation extractor, users just need to:
    1. Define extraction and transformation functions
    2. Register them with the OperationParameterExtractors

    Example:
    ```python
    from operation_parameter_extractors import OperationParameterExtractors

    def extract_my_operation_params(config):
        # Extract parameters specific to your operation
        params = {}
        # ... your extraction logic ...
        return params

    def transform_my_operation_configs(configs):
        # Transform raw configs to sweep format
        transformed = []
        # ... your transformation logic ...
        return transformed

    # Register the extractor
    OperationParameterExtractors.register_extractor(
        "my_operation",
        extract_func=extract_my_operation_params,
        transform_func=transform_my_operation_configs
    )

    # Now the operation will automatically use your custom extractors
    ```
    """
    pass


if __name__ == "__main__":
    # Demo: List registered operations
    print("Registered Operations:")
    for op in OperationParameterExtractors.list_registered_operations():
        print(f"  - {op}")

    print("\nTo add your own operation extractor, see the example_custom_operation_setup() function above.")
