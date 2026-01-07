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

            # Extract from arg1 (weight tensor)
            # In the traced config, arg1 might be in a dict with "arg1" key
            for arg in config:
                if isinstance(arg, dict):
                    # Case 1: {"arg1": {...}}
                    if "arg1" in arg:
                        tensor_config = OperationParameterExtractors.extract_tensor_config(arg["arg1"])
                        if tensor_config:
                            tensor_shapes.append(tensor_config.shape)
                            break
                    # Case 2: String-encoded tensor vector (e.g., concat operation)
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

            # Helper to extract value from arg dict
            def get_arg_value(arg_key: str, default=None):
                for arg in config:
                    if isinstance(arg, dict) and arg_key in arg:
                        return arg[arg_key]
                return default

            # Extract scalar arguments
            arg3 = get_arg_value("arg3")  # input_channels
            arg4 = get_arg_value("arg4")  # output_channels
            arg5 = get_arg_value("arg5")  # batch_size
            arg6 = get_arg_value("arg6")  # input_height
            arg7 = get_arg_value("arg7")  # input_width
            arg8 = get_arg_value("arg8")  # [kernel_h, kernel_w]
            arg9 = get_arg_value("arg9")  # [stride_h, stride_w]
            arg10 = get_arg_value("arg10")  # [pad_h1, pad_h2, pad_w1, pad_w2]
            arg11 = get_arg_value("arg11")  # [dilation_h, dilation_w]
            arg12 = get_arg_value("arg12")  # groups

            # Check if any required args are missing
            if None in [arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, arg12]:
                return None

            # Parse list strings (e.g., "[3, 3]" -> [3, 3])
            def parse_list_string(value):
                if isinstance(value, list):
                    return value
                if isinstance(value, str):
                    # Remove brackets and split
                    value = value.strip()
                    if value.startswith("[") and value.endswith("]"):
                        value = value[1:-1]
                    parts = [p.strip() for p in value.split(",") if p.strip()]
                    try:
                        return [int(p) for p in parts]
                    except ValueError:
                        # Try float if int fails
                        try:
                            return [float(p) for p in parts]
                        except ValueError:
                            return None
                return None

            kernel_list = parse_list_string(arg8)
            stride_list = parse_list_string(arg9)
            pad_list = parse_list_string(arg10)
            dilation_list = parse_list_string(arg11)

            if not all([kernel_list, stride_list, pad_list, dilation_list]):
                return None

            # Extract padding values - pad_list is [pad_h1, pad_h2, pad_w1, pad_w2]
            # Use pad_h1 and pad_w1 (or max of both sides)
            pad_h = max(pad_list[0], pad_list[1]) if len(pad_list) >= 2 else pad_list[0]
            pad_w = max(pad_list[2], pad_list[3]) if len(pad_list) >= 4 else pad_list[0]

            # Check for bias (arg14)
            has_bias = get_arg_value("arg14") is not None

            # Build params dict
            params = {
                "batch_size": int(arg5),
                "output_channels": int(arg4),
                "input_channels": int(arg3),
                "input_height": int(arg6),
                "input_width": int(arg7),
                "kernel_height": kernel_list[0],
                "kernel_width": kernel_list[1] if len(kernel_list) > 1 else kernel_list[0],
                "stride_h": stride_list[0],
                "stride_w": stride_list[1] if len(stride_list) > 1 else stride_list[0],
                "pad_h": pad_h,
                "pad_w": pad_w,
                "groups": int(arg12),
                "dilation_h": dilation_list[0],
                "dilation_w": dilation_list[1] if len(dilation_list) > 1 else dilation_list[0],
                "has_bias": has_bias,
            }

            return params
        except Exception as e:
            print(f"Error extracting conv2d parameters: {e}")
            import traceback

            traceback.print_exc()
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
                    dims_data = arg["arg1"]

                    # Handle direct list
                    if isinstance(dims_data, list):
                        # Check if list contains non-integer elements (unparsable)
                        if any(isinstance(d, str) and ("unsupported" in d or "std::" in d) for d in dims_data):
                            return None
                        # Validate all elements are integers
                        if all(isinstance(d, int) for d in dims_data):
                            return dims_data

                    # Handle dict with 'SmallVector' key
                    elif isinstance(dims_data, dict) and "SmallVector" in dims_data:
                        dims_list = dims_data["SmallVector"]
                        if isinstance(dims_list, list) and all(isinstance(d, int) for d in dims_list):
                            return dims_list

                    # Handle string format '[0, 2, 3, 1]'
                    elif isinstance(dims_data, str):
                        # Skip corrupted strings containing 'unsupported type' or 'std::reference_wrapper'
                        if "unsupported" in dims_data or "std::" in dims_data:
                            return None

                        if dims_data.startswith("[") and dims_data.endswith("]"):
                            # Parse the list string
                            dims_str = dims_data.strip("[]").strip()
                            if dims_str:
                                dims = [int(x.strip()) for x in dims_str.split(",")]
                                return dims
            return None
        except Exception:
            return None

    @staticmethod
    def _extract_permute_parameters(config: List) -> Optional[Dict]:
        """Extract parameters for permute operation"""
        dims = OperationParameterExtractors._extract_permute_dims(config)
        # Always return dims (None is fine, will be inferred by loader based on shape)
        return {"dims": dims}

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
        """Extract tensor configuration from argument data

        Note: Most UnparsedElements are now fixed by the tracer's post-processing.
        This method only handles string-encoded tensor vectors (e.g., concat operations).
        """
        if not isinstance(arg_data, dict):
            return None

        # Handle string-encoded tensor vectors (e.g., concat operation's arg0)
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
                except Exception:
                    return None

            # If it's an UnparsedElement but not a tensor vector, return None
            # (should not happen with post-processed data)
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
        # Layout is at top level of Tensor in new format, or inside tensor_layout in old format
        layout = tensor_data.get("layout", tensor_layout.get("layout", ""))
        memory_config = tensor_layout.get("memory_config", {})

        # If layout is missing, default to TILE for linear operations
        if not layout:
            layout = "Layout::TILE"

        if shape and dtype and layout and memory_config:
            return TensorConfig(shape, dtype, layout, memory_config)

        return None

    # Unary operation extractors
    # (permute extractor moved to earlier in file near _extract_permute_dims)

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
                        # Pass the whole arg dict so extract_tensor_config can handle nested structure
                        tensor_config = OperationParameterExtractors.extract_tensor_config(arg)
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
        """Extract parameters for pad operation

        ttnn.pad has two overloads:
        1. pad(input, padding, value) - padding is list[Tuple[int,int]]
        2. pad(input, output_padded_shape, input_tensor_start, value) - output_padded_shape is Array4D

        We need to detect which format is used based on arg1:
        - If arg1 is nested list (e.g., [[0,0], [0,13], [0,0], [0,0]]), it's padding format
        - If arg1 is flat 4-element list (e.g., [32, 16, 3, 3]), it's output_padded_shape format
        """
        try:
            padding = None
            output_padded_shape = None
            input_tensor_start = None
            value = None

            for arg in config:
                if isinstance(arg, dict):
                    if "arg1" in arg:
                        arg1_parsed = OperationParameterExtractors._parse_list_from_string(arg["arg1"])
                        if arg1_parsed and isinstance(arg1_parsed, list):
                            # Check if it's nested list (padding format) or flat list (output_padded_shape format)
                            if len(arg1_parsed) > 0 and isinstance(arg1_parsed[0], list):
                                # Nested list - this is padding format
                                padding = arg1_parsed
                            elif len(arg1_parsed) == 4 and all(isinstance(x, int) for x in arg1_parsed):
                                # Flat 4-element list - this is output_padded_shape format
                                output_padded_shape = arg1_parsed

                    if "arg2" in arg:
                        arg2_parsed = OperationParameterExtractors._parse_list_from_string(arg["arg2"])
                        if padding is not None:
                            # Padding format: arg2 is value
                            value = OperationParameterExtractors._parse_numeric_value(arg["arg2"])
                            if isinstance(value, list):
                                if len(set(value)) == 1:
                                    value = float(value[0])
                                else:
                                    value = float(value[0])
                            elif value is not None:
                                value = float(value)
                        elif output_padded_shape is not None:
                            # Output shape format: arg2 is input_tensor_start
                            if arg2_parsed and isinstance(arg2_parsed, list) and len(arg2_parsed) == 4:
                                input_tensor_start = arg2_parsed

                    if "arg3" in arg and output_padded_shape is not None:
                        # Output shape format: arg3 is value
                        value = OperationParameterExtractors._parse_numeric_value(arg["arg3"])
                        if isinstance(value, list):
                            if len(set(value)) == 1:
                                value = float(value[0])
                            else:
                                value = float(value[0])
                        elif value is not None:
                            value = float(value)

            # ALWAYS return output_padded_shape format for consistency
            # (The loader can't handle mixed formats in the same operation)
            if output_padded_shape is not None and input_tensor_start is not None and value is not None:
                # Already in output_padded_shape format
                return {
                    "output_padded_shape": output_padded_shape,
                    "input_tensor_start": input_tensor_start,
                    "value": value,
                }
            elif padding is not None and value is not None:
                # Convert padding format to output_padded_shape format
                # This is a LOSSY conversion but necessary for consistency
                # padding is [[front_0, back_0], [front_1, back_1], ...]
                # We'll use front padding as input_tensor_start and calculate output shape
                # This only works if we have the input shape, which we don't have here
                # So we'll just return the padding format and let the loader handle it
                return {"padding": padding, "value": value}
            return None
        except Exception as e:
            return None

    @staticmethod
    def _extract_max_pool2d_parameters(config: List) -> Optional[Dict]:
        """Extract parameters for max_pool2d operation

        Extracts from JSON:
        - arg1: batch_size
        - arg2: input_h
        - arg3: input_w
        - arg4: channels
        - arg5: kernel_size [H, W]
        - arg6: stride [H, W]
        - arg7: padding [H, W]
        - arg8: dilation [H, W]
        - arg11: applied_shard_scheme (TensorMemoryLayout enum)
        """
        try:
            params = {}
            for arg in config:
                if not isinstance(arg, dict):
                    continue
                # Extract batch_size (arg1)
                if "arg1" in arg:
                    batch_size = arg["arg1"]
                    if isinstance(batch_size, (int, str)) and batch_size != "nullopt":
                        try:
                            params["batch_size"] = int(batch_size)
                        except:
                            pass
                # Extract input_h (arg2)
                if "arg2" in arg:
                    input_h = arg["arg2"]
                    if isinstance(input_h, (int, str)) and input_h != "nullopt":
                        try:
                            params["input_h"] = int(input_h)
                        except:
                            pass
                # Extract input_w (arg3)
                if "arg3" in arg:
                    input_w = arg["arg3"]
                    if isinstance(input_w, (int, str)) and input_w != "nullopt":
                        try:
                            params["input_w"] = int(input_w)
                        except:
                            pass
                # Extract channels (arg4)
                if "arg4" in arg:
                    channels = arg["arg4"]
                    if isinstance(channels, (int, str)) and channels != "nullopt":
                        try:
                            params["channels"] = int(channels)
                        except:
                            pass
                # Extract kernel_size (arg5) - list format [H, W]
                if "arg5" in arg:
                    kernel_size = arg["arg5"]
                    if isinstance(kernel_size, list) and len(kernel_size) == 2:
                        params["kernel_size"] = kernel_size
                    elif isinstance(kernel_size, str):
                        # Try to parse string like "[5, 5]"
                        parsed = OperationParameterExtractors._parse_list_from_string(kernel_size)
                        if parsed and isinstance(parsed, list) and len(parsed) == 2:
                            params["kernel_size"] = parsed
                # Extract stride (arg6) - list format [H, W]
                if "arg6" in arg:
                    stride = arg["arg6"]
                    if isinstance(stride, list) and len(stride) == 2:
                        params["stride"] = stride
                    elif isinstance(stride, str):
                        parsed = OperationParameterExtractors._parse_list_from_string(stride)
                        if parsed and isinstance(parsed, list) and len(parsed) == 2:
                            params["stride"] = parsed
                # Extract padding (arg7) - list format [H, W]
                if "arg7" in arg:
                    padding = arg["arg7"]
                    if isinstance(padding, list) and len(padding) == 2:
                        params["padding"] = padding
                    elif isinstance(padding, str):
                        parsed = OperationParameterExtractors._parse_list_from_string(padding)
                        if parsed and isinstance(parsed, list) and len(parsed) == 2:
                            params["padding"] = parsed
                # Extract dilation (arg8) - list format [H, W]
                if "arg8" in arg:
                    dilation = arg["arg8"]
                    if isinstance(dilation, list) and len(dilation) == 2:
                        params["dilation"] = dilation
                    elif isinstance(dilation, str):
                        parsed = OperationParameterExtractors._parse_list_from_string(dilation)
                        if parsed and isinstance(parsed, list) and len(parsed) == 2:
                            params["dilation"] = parsed
                # Extract applied_shard_scheme (arg11) - TensorMemoryLayout enum
                if "arg11" in arg:
                    applied_shard_scheme = arg["arg11"]
                    if isinstance(applied_shard_scheme, str):
                        # Parse enum string like "TensorMemoryLayout::BLOCK_SHARDED"
                        if "BLOCK_SHARDED" in applied_shard_scheme:
                            params["applied_shard_scheme"] = "BLOCK_SHARDED"
                        elif "HEIGHT_SHARDED" in applied_shard_scheme:
                            params["applied_shard_scheme"] = "HEIGHT_SHARDED"
                        elif "WIDTH_SHARDED" in applied_shard_scheme:
                            params["applied_shard_scheme"] = "WIDTH_SHARDED"
                        elif "INTERLEAVED" in applied_shard_scheme:
                            params["applied_shard_scheme"] = "INTERLEAVED"
                        elif applied_shard_scheme == "nullopt":
                            params["applied_shard_scheme"] = None

            if params:
                return params
            return None
        except Exception as e:
            return None

    @staticmethod
    def _extract_upsample_parameters(config: List) -> Optional[Dict]:
        """Extract parameters for upsample operation

        Extracts from JSON:
        - arg1: scale_factor (int or [int, int] array)
        - arg2: mode (e.g., "nearest")
        """
        try:
            params = {}
            for arg in config:
                if not isinstance(arg, dict):
                    continue
                # Extract scale_factor (arg1) - can be int or [int, int] array
                if "arg1" in arg:
                    scale_factor = arg["arg1"]
                    if isinstance(scale_factor, int):
                        params["scale_factor"] = scale_factor
                    elif isinstance(scale_factor, list) and len(scale_factor) == 2:
                        # If array, use first element (or could use both)
                        params["scale_factor"] = scale_factor[0] if scale_factor[0] == scale_factor[1] else scale_factor
                    elif isinstance(scale_factor, str):
                        # Try to parse string - could be int like "2" or array like "[2, 2]"
                        if scale_factor.isdigit():
                            params["scale_factor"] = int(scale_factor)
                        else:
                            # Try parsing as array
                            parsed = OperationParameterExtractors._parse_list_from_string(scale_factor)
                            if parsed and isinstance(parsed, list) and len(parsed) == 2:
                                # If both elements are same, use single value
                                if parsed[0] == parsed[1]:
                                    params["scale_factor"] = parsed[0]
                                else:
                                    params["scale_factor"] = parsed
                # Extract mode (arg2)
                if "arg2" in arg:
                    mode = arg["arg2"]
                    if isinstance(mode, str) and mode != "nullopt":
                        params["mode"] = mode

            if params:
                return params
            return None
        except Exception as e:
            return None

    @staticmethod
    def _extract_update_cache_parameters(config: List) -> Optional[Dict]:
        """Extract parameters for update_cache operation

        Extracts:
        - arg0: cache_tensor (input_a)
        - arg1: input_tensor (input_b)
        - arg2: cache_idx (scalar)
        - arg3: batch_offset (scalar_1)
        - arg4: memory_config (optional, often nullopt)
        """
        try:
            params = {}

            # Extract first tensor (cache)
            if len(config) > 0 and isinstance(config[0], dict) and "arg0" in config[0]:
                tensor_config_a = OperationParameterExtractors.extract_tensor_config(config[0]["arg0"])
                if tensor_config_a:
                    cache_shape = tensor_config_a.shape
                    params["input_a_dtype"] = tensor_config_a.dtype.replace("DataType::", "")
                    params["input_a_layout"] = tensor_config_a.layout.replace("Layout::", "")
                    params["input_a_memory_config"] = tensor_config_a.memory_config
                else:
                    return None

            # Extract second tensor (input)
            if len(config) > 1 and isinstance(config[1], dict) and "arg1" in config[1]:
                tensor_config_b = OperationParameterExtractors.extract_tensor_config(config[1]["arg1"])
                if tensor_config_b:
                    input_shape = tensor_config_b.shape
                    params["input_b_dtype"] = tensor_config_b.dtype.replace("DataType::", "")
                    params["input_b_layout"] = tensor_config_b.layout.replace("Layout::", "")
                    params["input_b_memory_config"] = tensor_config_b.memory_config
                else:
                    return None

            # Store shapes separately for transform step
            params["input_shape"] = cache_shape  # Primary input (cache)
            params["input_b_shape"] = input_shape  # Secondary input

            # Extract cache_idx (arg2)
            if len(config) > 2 and isinstance(config[2], dict) and "arg2" in config[2]:
                cache_idx = config[2]["arg2"]
                # Extract batch_offset (arg3)
                batch_offset = None
                if len(config) > 3 and isinstance(config[3], dict) and "arg3" in config[3]:
                    batch_offset = config[3]["arg3"]

                # Store both scalars in a dict
                params["scalar"] = {
                    "update_index": str(cache_idx),
                    "batch_offset": str(batch_offset) if batch_offset is not None else "0",
                }

            # Use cache tensor's memory config as output
            params["output_memory_config"] = tensor_config_a.memory_config

            return params

        except Exception as e:
            import traceback

            print(f"Error extracting update_cache parameters: {e}")
            traceback.print_exc()
            return None

    @staticmethod
    def _extract_scale_mask_softmax_in_place_parameters(config: List) -> Optional[Dict]:
        """Extract parameters for scale_mask_softmax_in_place operation

        Extracts:
        - arg0: input_tensor (input_a)
        - arg1: scale (scalar)
        - arg2: mask_tensor (input_b, optional - can be nullopt)
        """
        try:
            params = {}

            # Extract input tensor (arg0)
            if len(config) > 0 and isinstance(config[0], dict) and "arg0" in config[0]:
                tensor_config_a = OperationParameterExtractors.extract_tensor_config(config[0]["arg0"])
                if tensor_config_a:
                    params["input_shape"] = tensor_config_a.shape
                    params["input_a_dtype"] = tensor_config_a.dtype.replace("DataType::", "")
                    params["input_a_layout"] = tensor_config_a.layout.replace("Layout::", "")
                    params["input_a_memory_config"] = tensor_config_a.memory_config
                else:
                    return None

            # Extract scale (arg1)
            if len(config) > 1 and isinstance(config[1], dict) and "arg1" in config[1]:
                scale_value = config[1]["arg1"]
                if scale_value != "nullopt":
                    params["scalar"] = str(scale_value)

            # Extract mask tensor (arg2, optional)
            if len(config) > 2 and isinstance(config[2], dict) and "arg2" in config[2]:
                arg2_value = config[2]["arg2"]
                if arg2_value != "nullopt":
                    tensor_config_b = OperationParameterExtractors.extract_tensor_config(arg2_value)
                    if tensor_config_b:
                        params["input_b_shape"] = tensor_config_b.shape
                        params["input_b_dtype"] = tensor_config_b.dtype.replace("DataType::", "")
                        params["input_b_layout"] = tensor_config_b.layout.replace("Layout::", "")
                        params["input_b_memory_config"] = tensor_config_b.memory_config

            # Use input tensor's memory config as output
            params["output_memory_config"] = tensor_config_a.memory_config

            return params

        except Exception as e:
            import traceback

            print(f"Error extracting scale_mask_softmax_in_place parameters: {e}")
            traceback.print_exc()
            return None

    @staticmethod
    def _transform_scale_mask_softmax_in_place_parameters(
        extracted_params: List[Dict],
        parse_dtype=None,
        parse_layout=None,
        parse_memory_config=None,
    ) -> List[Dict]:
        """Transform extracted scale_mask_softmax_in_place parameters to match test function signature"""
        transformed = []

        for params in extracted_params:
            transformed_config = {}

            # Transform input_shape
            if "input_shape" in params:
                transformed_config["input_shape"] = tuple(params["input_shape"])

            # Transform input tensor properties
            if "input_a_dtype" in params:
                transformed_config["input_a_dtype"] = (
                    parse_dtype(f"DataType::{params['input_a_dtype']}") if parse_dtype else params["input_a_dtype"]
                )
            if "input_a_layout" in params:
                transformed_config["input_a_layout"] = (
                    parse_layout(f"Layout::{params['input_a_layout']}") if parse_layout else params["input_a_layout"]
                )
            if "input_a_memory_config" in params:
                transformed_config["input_a_memory_config"] = (
                    parse_memory_config(params["input_a_memory_config"], [])
                    if parse_memory_config
                    else params["input_a_memory_config"]
                )

            # Transform mask tensor properties (optional)
            if "input_b_shape" in params:
                transformed_config["mask_shape"] = tuple(params["input_b_shape"])
            if "input_b_dtype" in params:
                transformed_config["input_b_dtype"] = (
                    parse_dtype(f"DataType::{params['input_b_dtype']}") if parse_dtype else params["input_b_dtype"]
                )
            if "input_b_layout" in params:
                transformed_config["input_b_layout"] = (
                    parse_layout(f"Layout::{params['input_b_layout']}") if parse_layout else params["input_b_layout"]
                )
            if "input_b_memory_config" in params:
                transformed_config["input_b_memory_config"] = (
                    parse_memory_config(params["input_b_memory_config"], [])
                    if parse_memory_config
                    else params["input_b_memory_config"]
                )

            # Output memory config
            if "output_memory_config" in params:
                transformed_config["output_memory_config"] = (
                    parse_memory_config(params["output_memory_config"], [])
                    if parse_memory_config
                    else params["output_memory_config"]
                )

            # Pass through scale value
            if "scalar" in params:
                transformed_config["scalar"] = float(params["scalar"])

            transformed.append(transformed_config)

        return transformed

    @staticmethod
    def _transform_update_cache_parameters(
        extracted_params: List[Dict],
        parse_dtype=None,
        parse_layout=None,
        parse_memory_config=None,
    ) -> List[Dict]:
        """Transform extracted update_cache parameters to match test function signature"""
        transformed = []

        for params in extracted_params:
            transformed_config = {}

            # Transform input_shape (build dict with 'self' and 'other' keys from separate shape fields)
            input_shape_dict = {}
            if "input_shape" in params:
                # input_shape is the cache tensor shape (self)
                input_shape_dict["self"] = tuple(params["input_shape"])
            if "input_b_shape" in params:
                # input_b_shape is the input tensor shape (other)
                input_shape_dict["other"] = tuple(params["input_b_shape"])
            transformed_config["input_shape"] = input_shape_dict

            # Transform dtypes, layouts, memory configs for both tensors
            if "input_a_dtype" in params:
                transformed_config["input_a_dtype"] = (
                    parse_dtype(f"DataType::{params['input_a_dtype']}") if parse_dtype else params["input_a_dtype"]
                )
            if "input_b_dtype" in params:
                transformed_config["input_b_dtype"] = (
                    parse_dtype(f"DataType::{params['input_b_dtype']}") if parse_dtype else params["input_b_dtype"]
                )

            if "input_a_layout" in params:
                transformed_config["input_a_layout"] = (
                    parse_layout(f"Layout::{params['input_a_layout']}") if parse_layout else params["input_a_layout"]
                )
            if "input_b_layout" in params:
                transformed_config["input_b_layout"] = (
                    parse_layout(f"Layout::{params['input_b_layout']}") if parse_layout else params["input_b_layout"]
                )

            if "input_a_memory_config" in params:
                transformed_config["input_a_memory_config"] = (
                    parse_memory_config(params["input_a_memory_config"], [])
                    if parse_memory_config
                    else params["input_a_memory_config"]
                )
            if "input_b_memory_config" in params:
                transformed_config["input_b_memory_config"] = (
                    parse_memory_config(params["input_b_memory_config"], [])
                    if parse_memory_config
                    else params["input_b_memory_config"]
                )
            if "output_memory_config" in params:
                transformed_config["output_memory_config"] = (
                    parse_memory_config(params["output_memory_config"], [])
                    if parse_memory_config
                    else params["output_memory_config"]
                )

            # Pass through scalar (cache_idx and batch_offset)
            if "scalar" in params:
                transformed_config["scalar"] = params["scalar"]

            transformed.append(transformed_config)

        return transformed

    @staticmethod
    def _extract_paged_update_cache_parameters(config: List) -> Optional[Dict]:
        """Extract parameters for paged_update_cache operation

        Extracts:
        - arg0: cache_tensor (input_a)
        - arg1: input_tensor (input_b)
        - arg2: update_idxs_tensor (optional, often empty) - skip
        - arg3: page_table_indices (input_c, INT32)
        - arg4: nullopt - skip
        - arg5: page_table (input_d, INT32)
        """
        try:
            params = {}

            # Extract input tensor configs from specific positions
            input_configs = []
            tensor_positions = [0, 1, 3, 5]  # cache, input, page_table_indices, page_table

            for arg_idx in tensor_positions:
                if len(config) > arg_idx:
                    arg = config[arg_idx]
                    if isinstance(arg, dict):
                        arg_key = f"arg{arg_idx}"
                        if arg_key in arg:
                            tensor_config = OperationParameterExtractors.extract_tensor_config(arg[arg_key])
                            if tensor_config:
                                input_configs.append(tensor_config)

            # Build params dict
            if len(input_configs) >= 3:  # Need at least cache, input, and page_table_indices
                # arg0: cache tensor
                params["input_shape"] = input_configs[0].shape
                params["input_a_dtype"] = input_configs[0].dtype.replace("DataType::", "")
                params["input_a_layout"] = input_configs[0].layout.replace("Layout::", "")
                params["input_a_memory_config"] = input_configs[0].memory_config

                # arg1: input tensor
                params["input_b_shape"] = input_configs[1].shape
                params["input_b_dtype"] = input_configs[1].dtype.replace("DataType::", "")
                params["input_b_layout"] = input_configs[1].layout.replace("Layout::", "")
                params["input_b_memory_config"] = input_configs[1].memory_config

                # arg3: page_table_indices
                params["input_c_shape"] = input_configs[2].shape
                params["input_c_dtype"] = input_configs[2].dtype.replace("DataType::", "")
                params["input_c_layout"] = input_configs[2].layout.replace("Layout::", "")
                params["input_c_memory_config"] = input_configs[2].memory_config

                # arg5: page_table (if present)
                if len(input_configs) > 3:
                    params["input_d_shape"] = input_configs[3].shape
                    params["input_d_dtype"] = input_configs[3].dtype.replace("DataType::", "")
                    params["input_d_layout"] = input_configs[3].layout.replace("Layout::", "")
                    params["input_d_memory_config"] = input_configs[3].memory_config

                # Use cache tensor's memory config as output
                params["output_memory_config"] = input_configs[0].memory_config

                return params

            return None
        except Exception as e:
            import traceback

            print(f"Error extracting paged_update_cache parameters: {e}")
            traceback.print_exc()
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
                            # Handle string-encoded tensor vectors
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

            # Extract input tensor config from arg0
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
                        # String-encoded tensor vector (should be rare, mostly for concat)
                        tensor_config = OperationParameterExtractors.extract_tensor_config(arg0)
                        if tensor_config:
                            input_shape = tensor_config.shape
                            input_dtype = tensor_config.dtype.replace("DataType::", "")
                            input_memory_config = tensor_config.memory_config

            # Extract output memory config from arg5
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
                        # String-encoded data (should not happen with post-processed data)
                        # Skip it as the data should be clean
                        pass

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

# Register scale_mask_softmax_in_place extractor (input, scale, optional mask)
OperationParameterExtractors.register_extractor(
    "scale_mask_softmax_in_place",
    extract_func=OperationParameterExtractors._extract_scale_mask_softmax_in_place_parameters,
    transform_func=OperationParameterExtractors._transform_scale_mask_softmax_in_place_parameters,
)
OperationParameterExtractors.register_extractor(
    "ttnn::scale_mask_softmax_in_place",
    extract_func=OperationParameterExtractors._extract_scale_mask_softmax_in_place_parameters,
    transform_func=OperationParameterExtractors._transform_scale_mask_softmax_in_place_parameters,
)

# Register update_cache extractor (custom extractor for cache, input, cache_idx, batch_offset)
OperationParameterExtractors.register_extractor(
    "update_cache",
    extract_func=OperationParameterExtractors._extract_update_cache_parameters,
    transform_func=OperationParameterExtractors._transform_update_cache_parameters,
)
OperationParameterExtractors.register_extractor(
    "ttnn::update_cache",
    extract_func=OperationParameterExtractors._extract_update_cache_parameters,
    transform_func=OperationParameterExtractors._transform_update_cache_parameters,
)

# Register paged_update_cache extractor (custom extractor for arg positions 0,1,3,5)
OperationParameterExtractors.register_extractor(
    "paged_update_cache",
    extract_func=OperationParameterExtractors._extract_paged_update_cache_parameters,
    transform_func=OperationParameterExtractors._transform_paged_scaled_dot_product_attention_decode_parameters,
)
OperationParameterExtractors.register_extractor(
    "experimental::paged_update_cache",
    extract_func=OperationParameterExtractors._extract_paged_update_cache_parameters,
    transform_func=OperationParameterExtractors._transform_paged_scaled_dot_product_attention_decode_parameters,
)
OperationParameterExtractors.register_extractor(
    "ttnn::experimental::paged_update_cache",
    extract_func=OperationParameterExtractors._extract_paged_update_cache_parameters,
    transform_func=OperationParameterExtractors._transform_paged_scaled_dot_product_attention_decode_parameters,
)

# Register max_pool2d extractor
OperationParameterExtractors.register_extractor(
    "max_pool2d",
    extract_func=OperationParameterExtractors._extract_max_pool2d_parameters,
)
OperationParameterExtractors.register_extractor(
    "ttnn::max_pool2d",
    extract_func=OperationParameterExtractors._extract_max_pool2d_parameters,
)

# Register upsample extractor
OperationParameterExtractors.register_extractor(
    "upsample",
    extract_func=OperationParameterExtractors._extract_upsample_parameters,
)
OperationParameterExtractors.register_extractor(
    "ttnn::upsample",
    extract_func=OperationParameterExtractors._extract_upsample_parameters,
)


# Add gt extractor method to the class
def _extract_gt_parameters(config: List) -> Optional[Dict]:
    """Extract parameters for gt (greater than) operation

    Extracts from JSON:
    - arg1: scalar value for comparison (e.g., 0)
    """
    try:
        params = {}
        for arg in config:
            if not isinstance(arg, dict):
                continue
            # Extract scalar value (arg1)
            if "arg1" in arg:
                scalar_value = arg["arg1"]
                if scalar_value != "nullopt" and scalar_value is not None:
                    # Convert to numeric if possible
                    if isinstance(scalar_value, (int, float)):
                        params["scalar"] = float(scalar_value)
                    elif isinstance(scalar_value, str):
                        try:
                            params["scalar"] = float(scalar_value)
                        except ValueError:
                            # If not numeric, keep as is
                            params["scalar"] = scalar_value

        return params if params else None
    except Exception as e:
        import traceback

        print(f"Error extracting gt parameters: {e}")
        traceback.print_exc()
        return None


# Add typecast extractor method to the class
def _extract_typecast_parameters(config: List) -> Optional[Dict]:
    """Extract parameters for typecast operation

    Extracts from JSON:
    - arg1: output_dtype (e.g., "DataType::BFLOAT8_B")
    """
    try:
        params = {}
        for arg in config:
            if not isinstance(arg, dict):
                continue
            # Extract output_dtype (arg1)
            if "arg1" in arg:
                output_dtype_str = arg["arg1"]
                if isinstance(output_dtype_str, str) and "DataType::" in output_dtype_str:
                    # Extract dtype name (e.g., "BFLOAT8_B" from "DataType::BFLOAT8_B")
                    dtype_name = output_dtype_str.replace("DataType::", "").strip()
                    params["output_dtype"] = dtype_name
                elif output_dtype_str and output_dtype_str != "nullopt":
                    params["output_dtype"] = output_dtype_str

        return params if params else None
    except Exception as e:
        import traceback

        print(f"Error extracting typecast parameters: {e}")
        traceback.print_exc()
        return None


# Add where extractor method to the class
def _extract_where_parameters(config: List) -> Optional[Dict]:
    """Extract parameters for where operation

    Extracts from JSON:
    - arg1: scalar_if_true value (e.g., 1.0)
    - arg2: scalar_if_false value (e.g., 0.0)
    """
    try:
        params = {}
        for arg in config:
            if not isinstance(arg, dict):
                continue
            # Extract scalar_if_true (arg1)
            if "arg1" in arg:
                scalar_value = arg["arg1"]
                if scalar_value != "nullopt" and scalar_value is not None:
                    # Convert to numeric if possible
                    if isinstance(scalar_value, (int, float)):
                        params["scalar_if_true"] = float(scalar_value)
                    elif isinstance(scalar_value, str):
                        try:
                            params["scalar_if_true"] = float(scalar_value)
                        except ValueError:
                            # If not numeric, keep as is
                            params["scalar_if_true"] = scalar_value
            # Extract scalar_if_false (arg2)
            if "arg2" in arg:
                scalar_value = arg["arg2"]
                if scalar_value != "nullopt" and scalar_value is not None:
                    # Convert to numeric if possible
                    if isinstance(scalar_value, (int, float)):
                        params["scalar_if_false"] = float(scalar_value)
                    elif isinstance(scalar_value, str):
                        try:
                            params["scalar_if_false"] = float(scalar_value)
                        except ValueError:
                            # If not numeric, keep as is
                            params["scalar_if_false"] = scalar_value

        return params if params else None
    except Exception as e:
        import traceback

        print(f"Error extracting where parameters: {e}")
        traceback.print_exc()
        return None


# Add div extractor method
def _extract_div_parameters(config: List) -> Optional[Dict]:
    """Extract parameters for div operation

    Extracts from JSON:
    - arg1: scalar value for division (e.g., 2.0)
    """
    try:
        params = {}
        for arg in config:
            if not isinstance(arg, dict):
                continue
            # Extract scalar value (arg1)
            if "arg1" in arg:
                scalar_value = arg["arg1"]
                if scalar_value != "nullopt" and scalar_value is not None:
                    # Convert to numeric if possible
                    if isinstance(scalar_value, (int, float)):
                        params["scalar"] = float(scalar_value)
                    elif isinstance(scalar_value, str):
                        try:
                            params["scalar"] = float(scalar_value)
                        except ValueError:
                            params["scalar"] = scalar_value
        return params if params else None
    except Exception as e:
        return None


# Add rms_norm_pre_all_gather extractor method
def _extract_rms_norm_pre_all_gather_parameters(config: List) -> Optional[Dict]:
    """Extract parameters for rms_norm_pre_all_gather operation"""
    try:
        params = {}
        # Extract tensor config from arg0 (input tensor)
        tensor_config = None
        for arg in config:
            if isinstance(arg, dict) and "arg0" in arg:
                tensor_config = OperationParameterExtractors.extract_tensor_config(arg["arg0"])
                break

        if tensor_config:
            params["input_shape"] = {"input_a": tensor_config.shape}
            params["input_a_dtype"] = tensor_config.dtype
            params["input_a_layout"] = tensor_config.layout
            params["input_a_memory_config"] = tensor_config.memory_config

            # Weight shape: typically [last_dim] for RMS norm
            weight_shape = [tensor_config.shape[-1]] if tensor_config.shape else [32]
            params["input_shape"]["input_b"] = weight_shape

            # Extract output memory config if present
            output_memory_config = None
            for arg in config:
                if isinstance(arg, dict) and "output_memory_config" in arg:
                    output_memory_config = arg["output_memory_config"]
                    break
            params["output_memory_config"] = output_memory_config or tensor_config.memory_config

        return params if params else None
    except Exception:
        return None


def _transform_rms_norm_pre_all_gather_parameters(
    configs: List, parse_dtype=None, parse_layout=None, parse_memory_config=None
) -> List[Dict]:
    """Transform rms_norm_pre_all_gather traced configs to run function format"""
    transformed_configs = []

    for config in configs:
        try:
            if not isinstance(config, dict):
                continue

            input_shape_dict = config.get("input_shape", {})
            if not input_shape_dict or "input_a" not in input_shape_dict:
                continue

            input_a_shape = input_shape_dict["input_a"]
            input_b_shape = input_shape_dict.get("input_b", [input_a_shape[-1]])

            # Parse dtypes
            input_a_dtype_str = config.get("input_a_dtype", "DataType::BFLOAT16")
            input_b_dtype_str = config.get("input_b_dtype", "DataType::BFLOAT16")
            input_a_layout_str = config.get("input_a_layout", "Layout::TILE")
            input_b_layout_str = config.get("input_b_layout", "Layout::ROW_MAJOR")

            # Parse memory configs
            input_a_mem_config = config.get("input_a_memory_config", {})
            input_b_mem_config = config.get("input_b_memory_config", {})
            output_mem_config = config.get("output_memory_config", input_a_mem_config)

            transformed_config = {
                "input_shape": input_shape_dict,
                "input_a_dtype": input_a_dtype_str,
                "input_b_dtype": input_b_dtype_str,
                "input_a_layout": input_a_layout_str,
                "input_b_layout": input_b_layout_str,
                "input_a_memory_config": input_a_mem_config,
                "input_b_memory_config": input_b_mem_config,
                "output_memory_config": output_mem_config,
            }

            # Apply parsers if provided
            if parse_dtype:
                transformed_config["input_a_dtype"] = parse_dtype(input_a_dtype_str)
                transformed_config["input_b_dtype"] = parse_dtype(input_b_dtype_str)
            if parse_layout:
                transformed_config["input_a_layout"] = parse_layout(input_a_layout_str)
                transformed_config["input_b_layout"] = parse_layout(input_b_layout_str)
            if parse_memory_config:
                transformed_config["input_a_memory_config"] = parse_memory_config(input_a_mem_config, input_a_shape)
                transformed_config["input_b_memory_config"] = parse_memory_config(input_b_mem_config, input_b_shape)
                transformed_config["output_memory_config"] = parse_memory_config(output_mem_config, input_a_shape)

            transformed_configs.append(transformed_config)

        except Exception as e:
            print(f"Error transforming rms_norm_pre_all_gather config: {e}")
            continue

    return transformed_configs


# Add methods to class
OperationParameterExtractors._extract_gt_parameters = staticmethod(_extract_gt_parameters)
OperationParameterExtractors._extract_typecast_parameters = staticmethod(_extract_typecast_parameters)
OperationParameterExtractors._extract_where_parameters = staticmethod(_extract_where_parameters)
OperationParameterExtractors._extract_div_parameters = staticmethod(_extract_div_parameters)
OperationParameterExtractors._extract_rms_norm_pre_all_gather_parameters = staticmethod(
    _extract_rms_norm_pre_all_gather_parameters
)
OperationParameterExtractors._transform_rms_norm_pre_all_gather_parameters = staticmethod(
    _transform_rms_norm_pre_all_gather_parameters
)

# Register gt extractor
OperationParameterExtractors.register_extractor(
    "gt",
    extract_func=OperationParameterExtractors._extract_gt_parameters,
)
OperationParameterExtractors.register_extractor(
    "ttnn::gt",
    extract_func=OperationParameterExtractors._extract_gt_parameters,
)

# Register typecast extractor
OperationParameterExtractors.register_extractor(
    "typecast",
    extract_func=OperationParameterExtractors._extract_typecast_parameters,
)
OperationParameterExtractors.register_extractor(
    "ttnn::typecast",
    extract_func=OperationParameterExtractors._extract_typecast_parameters,
)

# Register where extractor
OperationParameterExtractors.register_extractor(
    "where",
    extract_func=OperationParameterExtractors._extract_where_parameters,
)
OperationParameterExtractors.register_extractor(
    "ttnn::where",
    extract_func=OperationParameterExtractors._extract_where_parameters,
)

# Register div extractor
OperationParameterExtractors.register_extractor(
    "div",
    extract_func=OperationParameterExtractors._extract_div_parameters,
)
OperationParameterExtractors.register_extractor(
    "ttnn::div",
    extract_func=OperationParameterExtractors._extract_div_parameters,
)

# Register rms_norm_pre_all_gather extractor
OperationParameterExtractors.register_extractor(
    "rms_norm_pre_all_gather",
    extract_func=OperationParameterExtractors._extract_rms_norm_pre_all_gather_parameters,
    transform_func=OperationParameterExtractors._transform_rms_norm_pre_all_gather_parameters,
)
OperationParameterExtractors.register_extractor(
    "ttnn::rms_norm_pre_all_gather",
    extract_func=OperationParameterExtractors._extract_rms_norm_pre_all_gather_parameters,
    transform_func=OperationParameterExtractors._transform_rms_norm_pre_all_gather_parameters,
)

# Register rms_norm_post_all_gather extractor (reuse pre_all_gather)
OperationParameterExtractors.register_extractor(
    "rms_norm_post_all_gather",
    extract_func=OperationParameterExtractors._extract_rms_norm_pre_all_gather_parameters,
    transform_func=OperationParameterExtractors._transform_rms_norm_pre_all_gather_parameters,
)
OperationParameterExtractors.register_extractor(
    "ttnn::rms_norm_post_all_gather",
    extract_func=OperationParameterExtractors._extract_rms_norm_pre_all_gather_parameters,
    transform_func=OperationParameterExtractors._transform_rms_norm_pre_all_gather_parameters,
)

# Register fill_cache, reshard extractors (generic, no special extraction needed)
OperationParameterExtractors.register_extractor("fill_cache", extract_func=None, transform_func=None)
OperationParameterExtractors.register_extractor("reshard", extract_func=None, transform_func=None)


# Add custom extractor for scaled_dot_product_attention_decode (define before registration)
def _extract_sdpa_decode_params(config: List) -> Optional[Dict]:
    """Extract parameters for scaled_dot_product_attention_decode operation

    Config is the arg_list directly: [{'arg0': {...}}, {'arg1': {...}}, ...]

    Extracts:
    - Input tensor configs from arg0 (Q), arg1 (K), arg2 (V), arg6 (cur_pos)
    - Scalar parameters: arg3 (is_causal), arg8 (scale), arg9 (k_chunk_size)
    - Output memory config from arg10
    """
    try:
        params = {}

        # Extract input tensor configs (arg0=Q, arg1=K, arg2=V, arg6=cur_pos)
        input_configs = []
        for arg_idx in [0, 1, 2, 6]:
            if arg_idx < len(config):
                arg_elem = config[arg_idx]
                if isinstance(arg_elem, dict):
                    if f"arg{arg_idx}" in arg_elem:
                        tensor_config = OperationParameterExtractors.extract_tensor_config(arg_elem[f"arg{arg_idx}"])
                        if tensor_config:
                            input_configs.append(tensor_config)
                    elif "UnparsedElement" in arg_elem:
                        # Try to extract from string-encoded tensor vector
                        tensor_config = OperationParameterExtractors.extract_tensor_config(arg_elem)
                        if tensor_config:
                            input_configs.append(tensor_config)

        # Build input_shape dict for multi-input operation
        if input_configs:
            input_shape_dict = {}
            tensor_keys = ["input_a", "input_b", "input_c", "input_d"]
            for idx, tensor_config in enumerate(input_configs):
                if idx < len(tensor_keys) and tensor_config and tensor_config.shape:
                    key = tensor_keys[idx]
                    input_shape_dict[key] = tensor_config.shape
                    # Store dtype, layout, memory_config
                    params[f"{key}_dtype"] = tensor_config.dtype.replace("DataType::", "")
                    params[f"{key}_layout"] = tensor_config.layout.replace("Layout::", "")
                    params[f"{key}_memory_config"] = tensor_config.memory_config

            if input_shape_dict:
                params["input_shape"] = input_shape_dict

        # Extract scalar parameters
        # arg3: is_causal
        if len(config) > 3 and isinstance(config[3], dict) and "arg3" in config[3]:
            arg3_val = config[3]["arg3"]
            if isinstance(arg3_val, str):
                params["is_causal"] = arg3_val

        # arg8: scale
        if len(config) > 8 and isinstance(config[8], dict) and "arg8" in config[8]:
            arg8_val = config[8]["arg8"]
            if isinstance(arg8_val, str):
                try:
                    params["scale"] = float(arg8_val)
                except (ValueError, TypeError):
                    params["scale"] = arg8_val

        # arg9: k_chunk_size
        if len(config) > 9 and isinstance(config[9], dict) and "arg9" in config[9]:
            arg9_val = config[9]["arg9"]
            if isinstance(arg9_val, str):
                try:
                    params["k_chunk_size"] = int(arg9_val)
                except (ValueError, TypeError):
                    params["k_chunk_size"] = arg9_val

        # arg10: output_memory_config
        if len(config) > 10 and isinstance(config[10], dict) and "arg10" in config[10]:
            arg10_data = config[10]["arg10"]
            if isinstance(arg10_data, dict) and "MemoryConfig" in arg10_data:
                params["output_memory_config"] = arg10_data["MemoryConfig"]

        return params if params else None
    except Exception as e:
        import traceback

        print(f"Error extracting scaled_dot_product_attention_decode parameters: {e}")
        traceback.print_exc()
        return None


# Define custom transformer for scaled_dot_product_attention_decode
def _transform_sdpa_decode_params(
    configs: List[Dict],
    parse_dtype=None,
    parse_layout=None,
    parse_memory_config=None,
) -> List[Dict]:
    """Transform extracted scaled_dot_product_attention_decode parameters to TTNN types"""
    transformed_configs = []

    for config in configs:
        try:
            transformed_config = {}

            # Handle input_shape (dict format for multi-input)
            if "input_shape" in config and isinstance(config["input_shape"], dict):
                transformed_config["input_shape"] = config["input_shape"]

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
            else:
                transformed_config["input_a_dtype"] = config.get("input_a_dtype", "BFLOAT16")
                for key in ["input_b_dtype", "input_c_dtype", "input_d_dtype"]:
                    if key in config:
                        transformed_config[key] = config[key]

            # Transform layouts
            if parse_layout:
                transformed_config["input_a_layout"] = parse_layout(config.get("input_a_layout", "TILE"))
                for key in ["input_b_layout", "input_c_layout", "input_d_layout"]:
                    if key in config:
                        transformed_config[key] = parse_layout(config[key])
            else:
                transformed_config["input_a_layout"] = config.get("input_a_layout", "TILE")
                for key in ["input_b_layout", "input_c_layout", "input_d_layout"]:
                    if key in config:
                        transformed_config[key] = config[key]

            # Transform memory configs
            if parse_memory_config:
                input_shape_dict = config.get("input_shape", {})
                input_a_shape = input_shape_dict.get("input_a", []) if isinstance(input_shape_dict, dict) else []
                transformed_config["input_a_memory_config"] = parse_memory_config(
                    config.get("input_a_memory_config", {}), input_a_shape
                )
                for key in ["input_b_memory_config", "input_c_memory_config", "input_d_memory_config"]:
                    if key in config:
                        shape_key = key.replace("_memory_config", "_shape")
                        shape = (
                            input_shape_dict.get(shape_key.replace("input_", "input_"), [])
                            if isinstance(input_shape_dict, dict)
                            else []
                        )
                        transformed_config[key] = parse_memory_config(config[key], shape)

                # Transform output_memory_config
                if "output_memory_config" in config:
                    transformed_config["output_memory_config"] = parse_memory_config(
                        config["output_memory_config"], input_a_shape
                    )
            else:
                transformed_config["input_a_memory_config"] = config.get("input_a_memory_config", {})
                for key in [
                    "input_b_memory_config",
                    "input_c_memory_config",
                    "input_d_memory_config",
                    "output_memory_config",
                ]:
                    if key in config:
                        transformed_config[key] = config[key]

            # *** IMPORTANT: Pass through scalar parameters ***
            for scalar_param in ["scale", "k_chunk_size", "is_causal"]:
                if scalar_param in config:
                    transformed_config[scalar_param] = config[scalar_param]

            transformed_configs.append(transformed_config)
        except Exception as e:
            print(f"Error transforming scaled_dot_product_attention_decode config: {e}")
            import traceback

            traceback.print_exc()
            continue

    return transformed_configs


# Store as static methods
OperationParameterExtractors._extract_scaled_dot_product_attention_decode_parameters = staticmethod(
    _extract_sdpa_decode_params
)
OperationParameterExtractors._transform_scaled_dot_product_attention_decode_parameters = staticmethod(
    _transform_sdpa_decode_params
)

# Register attention operation extractors
OperationParameterExtractors.register_extractor(
    "transformer::chunked_scaled_dot_product_attention", extract_func=None, transform_func=None
)
OperationParameterExtractors.register_extractor(
    "transformer::scaled_dot_product_attention_decode",
    extract_func=_extract_sdpa_decode_params,
    transform_func=_transform_sdpa_decode_params,
)
OperationParameterExtractors.register_extractor(
    "ttnn::transformer::scaled_dot_product_attention_decode",
    extract_func=_extract_sdpa_decode_params,
    transform_func=_transform_sdpa_decode_params,
)


# Add rms_norm extractor method
def _extract_rms_norm_parameters(config: List) -> Optional[Dict]:
    """Extract parameters for rms_norm operation

    For RMS norm, the weight should match the input's last dimension.
    The traced "other" shape might be padded, so we recalculate it.
    """
    try:
        params = {}

        # Extract first 2 tensor configs (input and weight)
        tensor_configs = []
        for arg in config:
            if isinstance(arg, dict):
                for key in sorted(arg.keys()):
                    if key.startswith("arg"):
                        tensor_config = OperationParameterExtractors.extract_tensor_config(arg[key])
                        if tensor_config:
                            tensor_configs.append(tensor_config)
                            if len(tensor_configs) >= 2:
                                break
                if len(tensor_configs) >= 2:
                    break

        if len(tensor_configs) >= 2:
            input_shape = tensor_configs[0].shape
            # Calculate correct weight shape from input's last dimension
            weight_size = input_shape[-1]
            # Pad to 32-byte alignment for TTNN
            padded_weight_size = ((weight_size + 31) // 32) * 32
            # Weight shape in [1, 1, -1, 32] format
            weight_shape = [1, 1, padded_weight_size // 32, 32]

            params["input_shape"] = {"self": input_shape, "other": weight_shape}
            params["input_a_dtype"] = tensor_configs[0].dtype
            params["input_b_dtype"] = tensor_configs[1].dtype
            params["input_a_layout"] = tensor_configs[0].layout
            params["input_b_layout"] = tensor_configs[1].layout
            params["input_a_memory_config"] = tensor_configs[0].memory_config
            params["input_b_memory_config"] = tensor_configs[1].memory_config

            # Extract output memory config if present
            output_memory_config = None
            for arg in config:
                if isinstance(arg, dict):
                    for key, val in arg.items():
                        if "output" in key.lower() or (
                            key.startswith("arg") and isinstance(val, dict) and "MemoryConfig" in str(val)
                        ):
                            if isinstance(val, dict) and "MemoryConfig" in val:
                                output_memory_config = val["MemoryConfig"]
                                break

            if output_memory_config is None:
                output_memory_config = tensor_configs[0].memory_config

            params["output_memory_config"] = output_memory_config

            return params
        return None
    except Exception as e:
        return None


def _transform_rms_norm_parameters(
    configs: List, parse_dtype=None, parse_layout=None, parse_memory_config=None
) -> List[Dict]:
    """Transform rms_norm traced configs to run function format

    Handles layout-specific weight shape adjustment:
    - TILE layout: weight shape becomes [1, 1, 1, input_last_dim]
    - ROW_MAJOR layout: keep traced shape [1, 1, H, 32]
    """
    transformed_configs = []

    for config in configs:
        try:
            if not isinstance(config, dict):
                continue

            input_shape_dict = config.get("input_shape", {})
            if not input_shape_dict or "self" not in input_shape_dict or "other" not in input_shape_dict:
                continue

            shape_a = input_shape_dict["self"]
            shape_b = input_shape_dict["other"]

            # Parse dtypes and layouts
            input_a_dtype_str = config.get("input_a_dtype", "DataType::BFLOAT16")
            input_b_dtype_str = config.get("input_b_dtype", "DataType::BFLOAT16")
            input_a_layout_str = config.get("input_a_layout", "Layout::TILE")
            input_b_layout_str = config.get("input_b_layout", "Layout::ROW_MAJOR")

            # Adjust weight shape based on layout BEFORE parsing
            # Check if layout is TILE (handle both parsed and string formats)
            is_tile_layout = "TILE" in str(input_b_layout_str)

            if is_tile_layout and isinstance(shape_b, list) and len(shape_b) == 4:
                # For TILE layout, adjust weight shape to [1, 1, 1, input_last_dim]
                input_last_dim = shape_a[-1]
                adjusted_shape_b = [1, 1, 1, input_last_dim]
                # Update the input_shape dict with adjusted weight shape
                input_shape_dict = {"self": shape_a, "other": adjusted_shape_b}
                shape_b = adjusted_shape_b

            # Parse memory configs
            input_a_mem_config = config.get("input_a_memory_config", {})
            input_b_mem_config = config.get("input_b_memory_config", {})
            output_mem_config = config.get("output_memory_config", input_a_mem_config)

            transformed_config = {
                "input_shape": input_shape_dict,  # Use adjusted shape
                "input_a_dtype": input_a_dtype_str,
                "input_b_dtype": input_b_dtype_str,
                "input_a_layout": input_a_layout_str,
                "input_b_layout": input_b_layout_str,
                "input_a_memory_config": input_a_mem_config,
                "input_b_memory_config": input_b_mem_config,
                "output_memory_config": output_mem_config,
            }

            # Apply parsers if provided
            if parse_dtype:
                transformed_config["input_a_dtype"] = parse_dtype(input_a_dtype_str)
                transformed_config["input_b_dtype"] = parse_dtype(input_b_dtype_str)
            if parse_layout:
                transformed_config["input_a_layout"] = parse_layout(input_a_layout_str)
                transformed_config["input_b_layout"] = parse_layout(input_b_layout_str)
            if parse_memory_config:
                transformed_config["input_a_memory_config"] = parse_memory_config(input_a_mem_config, shape_a)
                transformed_config["input_b_memory_config"] = parse_memory_config(input_b_mem_config, shape_b)
                transformed_config["output_memory_config"] = parse_memory_config(output_mem_config, shape_a)

            transformed_configs.append(transformed_config)

        except Exception as e:
            print(f"Error transforming rms_norm config: {e}")
            continue

    return transformed_configs


# Add methods to class
OperationParameterExtractors._extract_rms_norm_parameters = staticmethod(_extract_rms_norm_parameters)
OperationParameterExtractors._transform_rms_norm_parameters = staticmethod(_transform_rms_norm_parameters)

# Register rms_norm extractor - even though it has no master data, we need it to transform traced configs
OperationParameterExtractors.register_extractor(
    "rms_norm",
    extract_func=None,  # No extraction needed for traced-only data
    transform_func=OperationParameterExtractors._transform_rms_norm_parameters,
)
OperationParameterExtractors.register_extractor(
    "ttnn::rms_norm",
    extract_func=None,  # No extraction needed for traced-only data
    transform_func=OperationParameterExtractors._transform_rms_norm_parameters,
)


# Add subtract extractor method
def _extract_subtract_parameters(config: List) -> Optional[Dict]:
    """Extract parameters for subtract operation

    Subtract has 2 tensor inputs but may be traced with 3 in some configs.
    This extractor ensures we only extract the 2 needed inputs.
    """
    try:
        params = {}
        # Extract first 2 tensor configs only (input_a and input_b)
        tensor_configs = []
        for arg in config:
            if isinstance(arg, dict):
                for key in sorted(arg.keys()):  # Process in order: arg0, arg1, arg2...
                    if key.startswith("arg"):
                        tensor_config = OperationParameterExtractors.extract_tensor_config(arg[key])
                        if tensor_config:
                            tensor_configs.append(tensor_config)
                            if len(tensor_configs) >= 2:  # Only need first 2
                                break
                if len(tensor_configs) >= 2:
                    break

        if len(tensor_configs) >= 2:
            params["input_shape"] = {"self": tensor_configs[0].shape, "other": tensor_configs[1].shape}
            params["input_a_dtype"] = tensor_configs[0].dtype
            params["input_b_dtype"] = tensor_configs[1].dtype
            params["input_a_layout"] = tensor_configs[0].layout
            params["input_b_layout"] = tensor_configs[1].layout
            params["input_a_memory_config"] = tensor_configs[0].memory_config
            params["input_b_memory_config"] = tensor_configs[1].memory_config

            # Extract output memory config if present
            output_memory_config = None
            for arg in config:
                if isinstance(arg, dict):
                    for key, val in arg.items():
                        if "output" in key.lower() or (
                            key.startswith("arg") and isinstance(val, dict) and "MemoryConfig" in str(val)
                        ):
                            if isinstance(val, dict) and "MemoryConfig" in val:
                                output_memory_config = val["MemoryConfig"]
                                break

            # Default to input_a memory config if no output specified
            if output_memory_config is None:
                output_memory_config = tensor_configs[0].memory_config

            params["output_memory_config"] = output_memory_config

            return params
        return None
    except Exception as e:
        return None


def _transform_subtract_parameters(
    configs: List, parse_dtype=None, parse_layout=None, parse_memory_config=None
) -> List[Dict]:
    """Transform subtract traced configs to run function format"""
    transformed_configs = []

    for config in configs:
        try:
            if not isinstance(config, dict):
                continue

            input_shape_dict = config.get("input_shape", {})
            if not input_shape_dict or "self" not in input_shape_dict or "other" not in input_shape_dict:
                continue

            shape_a = input_shape_dict["self"]
            shape_b = input_shape_dict["other"]

            # Parse dtypes and layouts
            input_a_dtype_str = config.get("input_a_dtype", "DataType::BFLOAT16")
            input_b_dtype_str = config.get("input_b_dtype", "DataType::BFLOAT16")
            input_a_layout_str = config.get("input_a_layout", "Layout::TILE")
            input_b_layout_str = config.get("input_b_layout", "Layout::TILE")

            # Parse memory configs
            input_a_mem_config = config.get("input_a_memory_config", {})
            input_b_mem_config = config.get("input_b_memory_config", {})
            output_mem_config = config.get("output_memory_config", input_a_mem_config)

            transformed_config = {
                "input_shape": input_shape_dict,  # Keep as dict with 'self' and 'other'
                "input_a_dtype": input_a_dtype_str,
                "input_b_dtype": input_b_dtype_str,
                "input_a_layout": input_a_layout_str,
                "input_b_layout": input_b_layout_str,
                "input_a_memory_config": input_a_mem_config,
                "input_b_memory_config": input_b_mem_config,
                "output_memory_config": output_mem_config,
            }

            # Apply parsers if provided
            if parse_dtype:
                transformed_config["input_a_dtype"] = parse_dtype(input_a_dtype_str)
                transformed_config["input_b_dtype"] = parse_dtype(input_b_dtype_str)
            if parse_layout:
                transformed_config["input_a_layout"] = parse_layout(input_a_layout_str)
                transformed_config["input_b_layout"] = parse_layout(input_b_layout_str)
            if parse_memory_config:
                transformed_config["input_a_memory_config"] = parse_memory_config(input_a_mem_config, shape_a)
                transformed_config["input_b_memory_config"] = parse_memory_config(input_b_mem_config, shape_b)
                transformed_config["output_memory_config"] = parse_memory_config(output_mem_config, shape_a)

            transformed_configs.append(transformed_config)

        except Exception as e:
            print(f"Error transforming subtract config: {e}")
            continue

    return transformed_configs


# Add methods to class
OperationParameterExtractors._extract_subtract_parameters = staticmethod(_extract_subtract_parameters)
OperationParameterExtractors._transform_subtract_parameters = staticmethod(_transform_subtract_parameters)

# Register subtract extractor
OperationParameterExtractors.register_extractor(
    "subtract",
    extract_func=OperationParameterExtractors._extract_subtract_parameters,
    transform_func=OperationParameterExtractors._transform_subtract_parameters,
)
OperationParameterExtractors.register_extractor(
    "ttnn::subtract",
    extract_func=OperationParameterExtractors._extract_subtract_parameters,
    transform_func=OperationParameterExtractors._transform_subtract_parameters,
)


# Add multiply_ extractor method (scalar multiply - in-place operation)
def _extract_multiply__parameters(config: List) -> Optional[Dict]:
    """Extract parameters for multiply_ operation (scalar multiply)

    multiply_ is an in-place operation that multiplies a tensor by a scalar.
    Args: tensor (arg0), scalar (arg1)
    """
    try:
        params = {}
        tensor_config = None
        scalar_value = None

        # Extract tensor and scalar
        for arg in config:
            if isinstance(arg, dict):
                for key, val in arg.items():
                    if key == "arg0":
                        tensor_config = OperationParameterExtractors.extract_tensor_config(val)
                    elif key == "arg1":
                        # Extract scalar value
                        if isinstance(val, (int, float)):
                            scalar_value = float(val)
                        elif isinstance(val, str):
                            try:
                                scalar_value = float(val)
                            except ValueError:
                                scalar_value = 1.0

        if tensor_config:
            params["input_shape"] = tensor_config.shape
            params["input_a_dtype"] = tensor_config.dtype
            params["input_a_layout"] = tensor_config.layout
            params["input_a_memory_config"] = tensor_config.memory_config
            params["scalar_value"] = scalar_value if scalar_value is not None else 1.0
            params["output_memory_config"] = tensor_config.memory_config

            return params
        return None
    except Exception as e:
        return None


def _transform_multiply__parameters(
    configs: List, parse_dtype=None, parse_layout=None, parse_memory_config=None
) -> List[Dict]:
    """Transform multiply_ traced configs to run function format"""
    transformed_configs = []

    for config in configs:
        try:
            if not isinstance(config, dict):
                continue

            input_shape = config.get("input_shape")
            if not input_shape:
                continue

            # Parse dtype and layout
            input_a_dtype_str = config.get("input_a_dtype", "DataType::BFLOAT16")
            input_a_layout_str = config.get("input_a_layout", "Layout::TILE")

            # Parse memory config
            input_a_mem_config = config.get("input_a_memory_config", {})
            output_mem_config = config.get("output_memory_config", input_a_mem_config)

            # Get scalar value
            scalar_value = config.get("scalar_value", 1.0)

            transformed_config = {
                "input_shape": input_shape,
                "input_a_dtype": input_a_dtype_str,
                "input_a_layout": input_a_layout_str,
                "input_a_memory_config": input_a_mem_config,
                "output_memory_config": output_mem_config,
                "scalar_value": scalar_value,
            }

            # Apply parsers if provided
            if parse_dtype:
                transformed_config["input_a_dtype"] = parse_dtype(input_a_dtype_str)
            if parse_layout:
                transformed_config["input_a_layout"] = parse_layout(input_a_layout_str)
            if parse_memory_config:
                transformed_config["input_a_memory_config"] = parse_memory_config(input_a_mem_config, input_shape)
                transformed_config["output_memory_config"] = parse_memory_config(output_mem_config, input_shape)

            transformed_configs.append(transformed_config)

        except Exception as e:
            print(f"Error transforming multiply_ config: {e}")
            continue

    return transformed_configs


# Add methods to class
OperationParameterExtractors._extract_multiply__parameters = staticmethod(_extract_multiply__parameters)
OperationParameterExtractors._transform_multiply__parameters = staticmethod(_transform_multiply__parameters)

# Register multiply_ extractor
OperationParameterExtractors.register_extractor(
    "multiply_",
    extract_func=OperationParameterExtractors._extract_multiply__parameters,
    transform_func=OperationParameterExtractors._transform_multiply__parameters,
)
OperationParameterExtractors.register_extractor(
    "ttnn::multiply_",
    extract_func=OperationParameterExtractors._extract_multiply__parameters,
    transform_func=OperationParameterExtractors._transform_multiply__parameters,
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


# ============================================================================
# Extractors for newly added operations
# ============================================================================


def _extract_pow_parameters(config: List) -> Optional[Dict]:
    """Extract parameters for pow operation"""
    try:
        params = {}
        tensor_config = None
        exponent = 2.0  # default
        output_memory_config = None

        for arg in config:
            if isinstance(arg, dict):
                # Extract tensor config from arg0
                if "arg0" in arg and isinstance(arg["arg0"], dict) and "Tensor" in arg["arg0"]:
                    tensor_config = OperationParameterExtractors.extract_tensor_config(arg["arg0"])

                # Extract exponent from arg1
                elif "arg1" in arg:
                    exponent_str = str(arg["arg1"])
                    try:
                        exponent = float(exponent_str)
                    except ValueError:
                        exponent = 2.0

                # Extract output memory_config from arg2
                elif "arg2" in arg and isinstance(arg["arg2"], dict):
                    if "MemoryConfig" in arg["arg2"]:
                        output_memory_config = arg["arg2"]["MemoryConfig"]

        if tensor_config:
            params["input_shape"] = tensor_config.shape
            params["input_a_dtype"] = tensor_config.dtype
            params["input_a_layout"] = tensor_config.layout
            params["input_a_memory_config"] = tensor_config.memory_config
            params["exponent"] = exponent
            params["output_memory_config"] = (
                output_memory_config if output_memory_config else tensor_config.memory_config
            )
            return params
        return None
    except Exception:
        return None


def _extract_clamp_parameters(config: List) -> Optional[Dict]:
    """Extract parameters for clamp operation"""
    try:
        params = {}
        tensor_config = None
        min_val = -10.0
        max_val = 10.0
        output_memory_config = None

        for arg in config:
            if isinstance(arg, dict):
                # Extract tensor config from arg0
                if "arg0" in arg and isinstance(arg["arg0"], dict) and "Tensor" in arg["arg0"]:
                    tensor_config = OperationParameterExtractors.extract_tensor_config(arg["arg0"])

                # Extract min from arg1
                elif "arg1" in arg:
                    try:
                        min_val = float(str(arg["arg1"]))
                    except ValueError:
                        pass

                # Extract max from arg2
                elif "arg2" in arg:
                    try:
                        max_val = float(str(arg["arg2"]))
                    except ValueError:
                        pass

                # Extract output memory_config from arg3
                elif "arg3" in arg and isinstance(arg["arg3"], dict):
                    if "MemoryConfig" in arg["arg3"]:
                        output_memory_config = arg["arg3"]["MemoryConfig"]

        if tensor_config:
            params["input_shape"] = tensor_config.shape
            params["input_a_dtype"] = tensor_config.dtype
            params["input_a_layout"] = tensor_config.layout
            params["input_a_memory_config"] = tensor_config.memory_config
            params["min"] = min_val
            params["max"] = max_val
            params["output_memory_config"] = (
                output_memory_config if output_memory_config else tensor_config.memory_config
            )
            return params
        return None
    except Exception:
        return None


def _extract_argmax_parameters(config: List) -> Optional[Dict]:
    """Extract parameters for argmax operation"""
    try:
        params = {}
        tensor_config = None
        dim = -1
        output_memory_config = None

        for arg in config:
            if isinstance(arg, dict):
                # Extract tensor config from arg0
                if "arg0" in arg and isinstance(arg["arg0"], dict) and "Tensor" in arg["arg0"]:
                    tensor_config = OperationParameterExtractors.extract_tensor_config(arg["arg0"])

                # Extract dim from arg1
                elif "arg1" in arg:
                    try:
                        dim = int(str(arg["arg1"]))
                    except ValueError:
                        dim = -1

                # Extract output memory_config from arg2 or arg3
                elif ("arg2" in arg or "arg3" in arg) and isinstance(list(arg.values())[0], dict):
                    arg_value = list(arg.values())[0]
                    if "MemoryConfig" in arg_value:
                        output_memory_config = arg_value["MemoryConfig"]

        if tensor_config:
            params["input_shape"] = tensor_config.shape
            params["input_a_dtype"] = tensor_config.dtype
            params["input_a_layout"] = tensor_config.layout
            params["input_a_memory_config"] = tensor_config.memory_config
            params["dim"] = dim
            params["output_memory_config"] = (
                output_memory_config if output_memory_config else tensor_config.memory_config
            )
            return params
        return None
    except Exception:
        return None


def _extract_sum_parameters(config: List) -> Optional[Dict]:
    """Extract parameters for sum operation"""
    try:
        params = {}
        tensor_config = None
        dim = None
        output_memory_config = None

        for arg in config:
            if isinstance(arg, dict):
                # Extract tensor config from arg0
                if "arg0" in arg and isinstance(arg["arg0"], dict) and "Tensor" in arg["arg0"]:
                    tensor_config = OperationParameterExtractors.extract_tensor_config(arg["arg0"])

                # Extract dim from arg1 (may be optional)
                elif "arg1" in arg:
                    dim_str = str(arg["arg1"])
                    if dim_str and dim_str != "None" and dim_str != "nullopt":
                        try:
                            dim = int(dim_str)
                        except ValueError:
                            dim = None

                # Extract output memory_config from later args
                elif any(k.startswith("arg") for k in arg.keys()):
                    arg_value = list(arg.values())[0]
                    if isinstance(arg_value, dict) and "MemoryConfig" in arg_value:
                        output_memory_config = arg_value["MemoryConfig"]

        if tensor_config:
            params["input_shape"] = tensor_config.shape
            params["input_a_dtype"] = tensor_config.dtype
            params["input_a_layout"] = tensor_config.layout
            params["input_a_memory_config"] = tensor_config.memory_config
            params["dim"] = dim
            params["output_memory_config"] = (
                output_memory_config if output_memory_config else tensor_config.memory_config
            )
            return params
        return None
    except Exception:
        return None


def _extract_std_parameters(config: List) -> Optional[Dict]:
    """Extract parameters for std operation"""
    try:
        params = {}
        tensor_config = None
        dim = None
        output_memory_config = None

        for arg in config:
            if isinstance(arg, dict):
                # Extract tensor config from arg0
                if "arg0" in arg and isinstance(arg["arg0"], dict) and "Tensor" in arg["arg0"]:
                    tensor_config = OperationParameterExtractors.extract_tensor_config(arg["arg0"])

                # Extract dim from arg1 (may be optional)
                elif "arg1" in arg:
                    dim_str = str(arg["arg1"])
                    if dim_str and dim_str != "None" and dim_str != "nullopt":
                        try:
                            dim = int(dim_str)
                        except ValueError:
                            dim = None

                # Extract output memory_config from later args
                elif any(k.startswith("arg") for k in arg.keys()):
                    arg_value = list(arg.values())[0]
                    if isinstance(arg_value, dict) and "MemoryConfig" in arg_value:
                        output_memory_config = arg_value["MemoryConfig"]

        if tensor_config:
            params["input_shape"] = tensor_config.shape
            params["input_a_dtype"] = tensor_config.dtype
            params["input_a_layout"] = tensor_config.layout
            params["input_a_memory_config"] = tensor_config.memory_config
            params["dim"] = dim
            params["output_memory_config"] = (
                output_memory_config if output_memory_config else tensor_config.memory_config
            )
            return params
        return None
    except Exception:
        return None


def _extract_softmax_parameters(config: List) -> Optional[Dict]:
    """Extract parameters for softmax operation"""
    try:
        params = {}
        tensor_config = None
        dim = -1
        output_memory_config = None

        for arg in config:
            if isinstance(arg, dict):
                # Extract tensor config from arg0
                if "arg0" in arg and isinstance(arg["arg0"], dict) and "Tensor" in arg["arg0"]:
                    tensor_config = OperationParameterExtractors.extract_tensor_config(arg["arg0"])

                # Extract dim from arg1
                elif "arg1" in arg:
                    try:
                        dim = int(str(arg["arg1"]))
                    except ValueError:
                        dim = -1

                # Extract output memory_config from later args
                elif any(k.startswith("arg") for k in arg.keys()):
                    arg_value = list(arg.values())[0]
                    if isinstance(arg_value, dict) and "MemoryConfig" in arg_value:
                        output_memory_config = arg_value["MemoryConfig"]

        if tensor_config:
            params["input_shape"] = tensor_config.shape
            params["input_a_dtype"] = tensor_config.dtype
            params["input_a_layout"] = tensor_config.layout
            params["input_a_memory_config"] = tensor_config.memory_config
            params["dim"] = dim
            params["output_memory_config"] = (
                output_memory_config if output_memory_config else tensor_config.memory_config
            )
            return params
        return None
    except Exception:
        return None


def _extract_repeat_parameters(config: List) -> Optional[Dict]:
    """Extract parameters for repeat operation"""
    try:
        params = {}
        tensor_config = None
        shape = [1, 1, 2, 1]  # default repetition vector
        output_memory_config = None

        for arg in config:
            if isinstance(arg, dict):
                # Extract tensor config from arg0
                if "arg0" in arg and isinstance(arg["arg0"], dict) and "Tensor" in arg["arg0"]:
                    tensor_config = OperationParameterExtractors.extract_tensor_config(arg["arg0"])

                # Extract repetition vector from arg1
                elif "arg1" in arg:
                    shape_str = str(arg["arg1"])
                    # Try to parse as list
                    if "[" in shape_str:
                        try:
                            shape = eval(shape_str)
                        except:
                            pass

                # Extract output memory_config from arg2
                elif "arg2" in arg and isinstance(arg["arg2"], dict):
                    if "MemoryConfig" in arg["arg2"]:
                        output_memory_config = arg["arg2"]["MemoryConfig"]

        if tensor_config:
            params["input_shape"] = tensor_config.shape
            params["input_a_dtype"] = tensor_config.dtype
            params["input_a_layout"] = tensor_config.layout
            params["input_a_memory_config"] = tensor_config.memory_config
            params["shape"] = shape
            params["output_memory_config"] = (
                output_memory_config if output_memory_config else tensor_config.memory_config
            )
            return params
        return None
    except Exception:
        return None


# Register the new extractors
OperationParameterExtractors.register_extractor("pow", extract_func=_extract_pow_parameters)
OperationParameterExtractors.register_extractor("ttnn::pow", extract_func=_extract_pow_parameters)

OperationParameterExtractors.register_extractor("clamp", extract_func=_extract_clamp_parameters)
OperationParameterExtractors.register_extractor("ttnn::clamp", extract_func=_extract_clamp_parameters)

OperationParameterExtractors.register_extractor("argmax", extract_func=_extract_argmax_parameters)
OperationParameterExtractors.register_extractor("ttnn::argmax", extract_func=_extract_argmax_parameters)

OperationParameterExtractors.register_extractor("sum", extract_func=_extract_sum_parameters)
OperationParameterExtractors.register_extractor("ttnn::sum", extract_func=_extract_sum_parameters)

OperationParameterExtractors.register_extractor("std", extract_func=_extract_std_parameters)
OperationParameterExtractors.register_extractor("ttnn::std", extract_func=_extract_std_parameters)

OperationParameterExtractors.register_extractor("softmax", extract_func=_extract_softmax_parameters)
OperationParameterExtractors.register_extractor("ttnn::softmax", extract_func=_extract_softmax_parameters)

OperationParameterExtractors.register_extractor("repeat", extract_func=_extract_repeat_parameters)
OperationParameterExtractors.register_extractor("ttnn::repeat", extract_func=_extract_repeat_parameters)


def _extract_group_norm_parameters(config: List) -> Optional[Dict]:
    """Extract parameters for group_norm operation (all 16 arguments)"""
    try:
        params = {}
        tensor_config = None
        num_groups = 32  # default
        epsilon = 1e-6  # default
        input_mask_config = None
        weight_config = None
        bias_config = None
        reciprocals_config = None
        output_memory_config = None
        inplace = False
        num_out_blocks = None
        use_welford = False

        for arg in config:
            if isinstance(arg, dict):
                # arg0: Extract input tensor config
                if "arg0" in arg and isinstance(arg["arg0"], dict) and "Tensor" in arg["arg0"]:
                    tensor_config = OperationParameterExtractors.extract_tensor_config(arg["arg0"])

                # arg1: num_groups
                elif "arg1" in arg:
                    try:
                        num_groups = int(str(arg["arg1"]))
                    except ValueError:
                        num_groups = 32

                # arg2: epsilon
                elif "arg2" in arg:
                    try:
                        epsilon = float(str(arg["arg2"]))
                    except ValueError:
                        epsilon = 1e-6

                # arg3: input_mask tensor
                elif "arg3" in arg and isinstance(arg["arg3"], dict) and "Tensor" in arg["arg3"]:
                    input_mask_config = OperationParameterExtractors.extract_tensor_config(arg["arg3"])

                # arg4: weight tensor
                elif "arg4" in arg and isinstance(arg["arg4"], dict) and "Tensor" in arg["arg4"]:
                    weight_config = OperationParameterExtractors.extract_tensor_config(arg["arg4"])

                # arg5: bias tensor
                elif "arg5" in arg and isinstance(arg["arg5"], dict) and "Tensor" in arg["arg5"]:
                    bias_config = OperationParameterExtractors.extract_tensor_config(arg["arg5"])

                # arg6: reciprocals tensor
                elif "arg6" in arg and isinstance(arg["arg6"], dict) and "Tensor" in arg["arg6"]:
                    reciprocals_config = OperationParameterExtractors.extract_tensor_config(arg["arg6"])

                # arg7: output memory_config
                elif "arg7" in arg:
                    arg_value = arg["arg7"]
                    if isinstance(arg_value, dict) and "MemoryConfig" in arg_value:
                        output_memory_config = arg_value["MemoryConfig"]

                # arg10: inplace
                elif "arg10" in arg:
                    try:
                        inplace = bool(int(str(arg["arg10"])))
                    except (ValueError, TypeError):
                        inplace = False

                # arg12: num_out_blocks
                elif "arg12" in arg:
                    try:
                        val = str(arg["arg12"])
                        if val != "nullopt":
                            num_out_blocks = int(val)
                    except (ValueError, TypeError):
                        pass

                # arg15: use_welford
                elif "arg15" in arg:
                    try:
                        use_welford = bool(int(str(arg["arg15"])))
                    except (ValueError, TypeError):
                        use_welford = False

        if tensor_config:
            params["input_shape"] = tensor_config.shape
            params["input_a_dtype"] = tensor_config.dtype
            params["input_a_layout"] = tensor_config.layout
            params["input_a_memory_config"] = tensor_config.memory_config
            params["num_groups"] = num_groups
            params["epsilon"] = epsilon
            params["output_memory_config"] = (
                output_memory_config if output_memory_config else tensor_config.memory_config
            )
            params["inplace"] = inplace
            params["use_welford"] = use_welford

            # Add optional tensor configs
            if input_mask_config:
                params["input_mask_shape"] = input_mask_config.shape
                params["input_mask_dtype"] = input_mask_config.dtype
                params["input_mask_layout"] = input_mask_config.layout
                params["input_mask_memory_config"] = input_mask_config.memory_config

            if weight_config:
                params["weight_shape"] = weight_config.shape
                params["weight_dtype"] = weight_config.dtype
                params["weight_layout"] = weight_config.layout
                params["weight_memory_config"] = weight_config.memory_config

            if bias_config:
                params["bias_shape"] = bias_config.shape
                params["bias_dtype"] = bias_config.dtype
                params["bias_layout"] = bias_config.layout
                params["bias_memory_config"] = bias_config.memory_config

            if reciprocals_config:
                params["reciprocals_shape"] = reciprocals_config.shape
                params["reciprocals_dtype"] = reciprocals_config.dtype
                params["reciprocals_layout"] = reciprocals_config.layout
                params["reciprocals_memory_config"] = reciprocals_config.memory_config

            if num_out_blocks is not None:
                params["num_out_blocks"] = num_out_blocks

            return params
        return None
    except Exception:
        return None


# Register group_norm extractor
OperationParameterExtractors.register_extractor("group_norm", extract_func=_extract_group_norm_parameters)
OperationParameterExtractors.register_extractor("ttnn::group_norm", extract_func=_extract_group_norm_parameters)

# Register permute extractor
OperationParameterExtractors.register_extractor(
    "permute", extract_func=OperationParameterExtractors._extract_permute_parameters
)
OperationParameterExtractors.register_extractor(
    "ttnn::permute", extract_func=OperationParameterExtractors._extract_permute_parameters
)


if __name__ == "__main__":
    # Demo: List registered operations
    print("Registered Operations:")
    for op in OperationParameterExtractors.list_registered_operations():
        print(f"  - {op}")

    print("\nTo add your own operation extractor, see the example_custom_operation_setup() function above.")
