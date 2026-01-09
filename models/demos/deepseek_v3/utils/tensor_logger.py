# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from loguru import logger

import ttnn


class TensorLogger:
    """
    Utility class for logging detailed tensor metadata during model execution.
    Captures tensor properties for analysis and debugging purposes.
    """

    def __init__(self, log_file: Optional[str] = None, enable_logging: bool = True):
        self.enable_logging = enable_logging
        if not enable_logging:
            return

        if log_file is None:
            log_file = f"tensor_log_{os.getpid()}.jsonl"

        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        # Initialize log file
        with open(self.log_file, "w") as f:
            f.write("")  # Clear the file

        logger.info(f"TensorLogger initialized, logging to: {self.log_file}")

    def log_tensor(
        self, tensor: Any, step_name: str, tensor_name: str, additional_info: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log detailed metadata for a tensor.

        Args:
            tensor: The tensor to log (ttnn.Tensor or torch.Tensor)
            step_name: Name of the operation/step
            tensor_name: Name/description of the tensor
            additional_info: Optional additional information to log
        """
        if not self.enable_logging:
            return

        try:
            metadata = self._extract_tensor_metadata(tensor, step_name, tensor_name, additional_info)
            self._write_log_entry(metadata)
        except Exception as e:
            logger.warning(f"Failed to log tensor {tensor_name} at step {step_name}: {e}")

    def _extract_tensor_metadata(
        self, tensor: Any, step_name: str, tensor_name: str, additional_info: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Extract metadata from tensor based on its type."""
        metadata = {
            "step_name": step_name,
            "tensor_name": tensor_name,
            "timestamp": str(torch.cuda.current_stream().cuda_time() if torch.cuda.is_available() else 0),
        }

        if additional_info:
            metadata.update(additional_info)

        if isinstance(tensor, ttnn.Tensor):
            metadata.update(self._extract_ttnn_metadata(tensor))
        elif isinstance(tensor, torch.Tensor):
            metadata.update(self._extract_torch_metadata(tensor))
        else:
            metadata["tensor_type"] = str(type(tensor))
            metadata["error"] = f"Unknown tensor type: {type(tensor)}"

        return metadata

    def _extract_ttnn_metadata(self, tensor: ttnn.Tensor) -> Dict[str, Any]:
        """Extract metadata from ttnn tensor."""
        try:
            metadata = {
                "tensor_type": "ttnn",
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "layout": str(tensor.layout),
                "rank": len(tensor.shape),
                "storage_type": str(tensor.storage_type()),
            }

            # Memory configuration
            try:
                memory_config = tensor.memory_config()
                metadata["memory_config"] = {
                    "memory_layout": str(memory_config.memory_layout),
                    "buffer_type": str(memory_config.buffer_type),
                }

                # Shard specification if available
                if hasattr(memory_config, "shard_spec") and memory_config.shard_spec is not None:
                    shard_spec = memory_config.shard_spec
                    metadata["shard_spec"] = {
                        "shape": list(shard_spec.shape) if hasattr(shard_spec, "shape") else None,
                        "orientation": str(shard_spec.orientation) if hasattr(shard_spec, "orientation") else None,
                        "halo": str(shard_spec.halo) if hasattr(shard_spec, "halo") else None,
                    }
                else:
                    metadata["shard_spec"] = None

            except Exception as e:
                metadata["memory_config_error"] = str(e)

            # Device information
            try:
                if hasattr(tensor, "device"):
                    device = tensor.device()
                    if hasattr(device, "shape"):
                        metadata["device_mesh_shape"] = list(device.shape)
                    metadata["device_type"] = str(type(device))
                else:
                    metadata["device"] = "No device information"
            except Exception as e:
                metadata["device_error"] = str(e)

        except Exception as e:
            metadata = {
                "tensor_type": "ttnn",
                "error": f"Failed to extract ttnn metadata: {e}",
                "fallback_shape": list(tensor.shape) if hasattr(tensor, "shape") else "unknown",
            }

        return metadata

    def _extract_torch_metadata(self, tensor: torch.Tensor) -> Dict[str, Any]:
        """Extract metadata from torch tensor."""
        try:
            metadata = {
                "tensor_type": "torch",
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "device": str(tensor.device),
                "rank": len(tensor.shape),
                "requires_grad": tensor.requires_grad,
                "is_contiguous": tensor.is_contiguous(),
                "stride": list(tensor.stride()) if tensor.stride else None,
            }

            # Memory usage
            try:
                metadata["element_count"] = tensor.numel()
                metadata["memory_bytes"] = tensor.element_size() * tensor.numel()
            except Exception as e:
                metadata["memory_error"] = str(e)

        except Exception as e:
            metadata = {
                "tensor_type": "torch",
                "error": f"Failed to extract torch metadata: {e}",
                "fallback_shape": list(tensor.shape) if hasattr(tensor, "shape") else "unknown",
            }

        return metadata

    def _write_log_entry(self, metadata: Dict[str, Any]) -> None:
        """Write metadata entry to log file."""
        try:
            # Convert non-serializable objects to strings
            serializable_metadata = self._make_serializable(metadata)
            with open(self.log_file, "a") as f:
                f.write(json.dumps(serializable_metadata) + "\n")
        except Exception as e:
            logger.warning(f"Failed to write tensor log entry: {e}")

    def _make_serializable(self, obj: Any) -> Any:
        """Convert non-JSON serializable objects to serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_serializable(item) for item in obj]
        elif hasattr(obj, "__dict__"):
            # Convert objects with attributes to string representation
            return str(obj)
        elif not isinstance(obj, (str, int, float, bool, type(None))):
            # Convert other non-serializable types to string
            return str(obj)
        return obj

    def log_operation(
        self,
        operation_name: str,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log an entire operation with its inputs and outputs.

        Args:
            operation_name: Name of the operation
            inputs: Dictionary of input tensors {name: tensor}
            outputs: Dictionary of output tensors {name: tensor}
            config: Optional configuration parameters
        """
        if not self.enable_logging:
            return

        # Log operation start
        metadata = {"event_type": "operation_start", "operation_name": operation_name, "config": config or {}}
        self._write_log_entry(metadata)

        # Log input tensors
        for input_name, tensor in inputs.items():
            self.log_tensor(tensor, f"{operation_name}_input", input_name)

        # Log output tensors
        for output_name, tensor in outputs.items():
            self.log_tensor(tensor, f"{operation_name}_output", output_name)

        # Log operation end
        metadata = {"event_type": "operation_end", "operation_name": operation_name}
        self._write_log_entry(metadata)

    def close(self) -> None:
        """Close the logger and finalize log file."""
        if not self.enable_logging:
            return
        logger.info(f"TensorLogger closed. Log saved to: {self.log_file}")


# Global logger instance - can be enabled/disabled via environment variable
ENABLE_TENSOR_LOGGING = os.getenv("ENABLE_TENSOR_LOGGING", "false").lower() in ("true", "1", "yes")
LOG_FILE = os.getenv("TENSOR_LOG_FILE")

# Global instance
_global_logger: Optional[TensorLogger] = None


def get_tensor_logger() -> TensorLogger:
    """Get or create the global tensor logger instance."""
    global _global_logger
    if _global_logger is None:
        _global_logger = TensorLogger(log_file=LOG_FILE, enable_logging=ENABLE_TENSOR_LOGGING)
    return _global_logger


def log_tensor(tensor: Any, step_name: str, tensor_name: str, additional_info: Optional[Dict[str, Any]] = None) -> None:
    """Convenience function to log a tensor using the global logger."""
    get_tensor_logger().log_tensor(tensor, step_name, tensor_name, additional_info)


def log_operation(
    operation_name: str, inputs: Dict[str, Any], outputs: Dict[str, Any], config: Optional[Dict[str, Any]] = None
) -> None:
    """Convenience function to log an operation using the global logger."""
    get_tensor_logger().log_operation(operation_name, inputs, outputs, config)
