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
    def transform_parameters(cls, operation_name: str, configs: List) -> List[Dict]:
        """Transform parameters for a specific operation"""
        if operation_name in cls._transformers:
            return cls._transformers[operation_name](configs)
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

            # Find the input_shape argument which contains {'self': [...], 'other': [...]}
            for arg in config:
                if isinstance(arg, dict) and "input_shape" in arg:
                    input_shape_data = arg["input_shape"]
                    if (
                        isinstance(input_shape_data, dict)
                        and "self" in input_shape_data
                        and "other" in input_shape_data
                    ):
                        params["input_shape"] = input_shape_data
                        return params

            return None
        except Exception:
            return None

    @staticmethod
    def _transform_embedding_parameters(configs: List) -> List[Dict]:
        """Transform embedding traced configs to run function format"""
        transformed_configs = []

        for config in configs:
            try:
                # config is a tuple: (input_shape_dict, input_a_dtype, input_b_dtype, ...)
                input_shape_dict = config[0]  # {'self': [...], 'other': [...]}
                indices_shape = input_shape_dict["self"]
                weights_shape = input_shape_dict["other"]

                # Extract vocab_size and embedding_dim from weights shape
                # weights_shape is [1, 1, vocab_size, embedding_dim]
                vocab_size = weights_shape[-2]
                embedding_dim = weights_shape[-1]

                # Convert indices shape to batch_size, seq_length format
                if len(indices_shape) == 1:
                    batch_size, seq_length = 1, indices_shape[0]
                elif len(indices_shape) == 2:
                    batch_size, seq_length = indices_shape[0], indices_shape[1]
                else:
                    # For higher dimensions, flatten all but last dimension
                    batch_size = 1
                    seq_length = 1
                    for dim in indices_shape[:-1]:
                        seq_length *= dim

                embedding_args = (batch_size, seq_length, embedding_dim, vocab_size)

                transformed_config = {
                    "embedding_args": embedding_args,
                    "input_dtype": config[1],  # input_a_dtype (uint32)
                    "weight_dtype": config[2],  # input_b_dtype (bfloat16)
                    "output_dtype": config[2],  # Use same as weight dtype for output
                    "input_layout": config[3],  # input_a_layout
                    "weight_layout": config[4],  # input_b_layout (will be forced to ROW_MAJOR)
                    "input_memory_config": config[5],  # input_a_memory_config
                    "weight_memory_config": config[6],  # input_b_memory_config
                    "output_memory_config": config[7],  # output_memory_config
                }

                transformed_configs.append(transformed_config)

            except Exception:
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
    def _transform_linear_parameters(configs: List) -> List[Dict]:
        """Transform linear traced configs to run function format"""
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

                transformed_config = {
                    "input_shape": processed_input_shape,
                    "weight_shape": processed_weight_shape,
                    "bias_shape": bias_shape,
                    "input_a_dtype": "DataType.BFLOAT16",  # Default, could be extracted from config
                    "input_b_dtype": "DataType.BFLOAT16",
                    "input_a_layout": "Layout.TILE",  # Default
                    "input_b_layout": "Layout.TILE",
                    "input_a_memory_config": "MemoryConfig.INTERLEAVED",  # Default
                    "input_b_memory_config": "MemoryConfig.INTERLEAVED",
                    "output_memory_config": "MemoryConfig.INTERLEAVED",
                    "transpose_a": transpose_a,
                    "transpose_b": transpose_b,
                    "has_bias": has_bias,
                }

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

    @staticmethod
    def _extract_permute_dims(config: List) -> Optional[List[int]]:
        """Extract dimensions for permute operation"""
        try:
            for arg in config:
                if isinstance(arg, dict) and "arg1" in arg:
                    dims = arg["arg1"]
                    if isinstance(dims, list):
                        return dims
            return None
        except Exception:
            return None

    @staticmethod
    def _extract_shape_parameter(config: List, arg_name: str = "arg1") -> Optional[List[int]]:
        """Extract shape parameter from config"""
        try:
            for arg in config:
                if isinstance(arg, dict) and arg_name in arg:
                    shape = arg[arg_name]
                    if isinstance(shape, list):
                        return shape
            return None
        except Exception:
            return None

    @staticmethod
    def _extract_int_parameter(config: List, arg_name: str) -> Optional[int]:
        """Extract integer parameter from config"""
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
