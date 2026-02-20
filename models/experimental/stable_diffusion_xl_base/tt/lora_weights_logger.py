#!/usr/bin/env python3

"""
Utility for logging LoRA-impacted weights in SDXL UNet implementation.
"""

import logging
import os
from datetime import datetime


class LoRAWeightsLogger:
    """Logger specifically for tracking LoRA-impacted weights during model initialization."""

    _instance = None
    _logger = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LoRAWeightsLogger, cls).__new__(cls)
            cls._setup_logger()
        return cls._instance

    @classmethod
    def _setup_logger(cls):
        """Set up the logger for LoRA weights tracking."""
        # Create logs directory if it doesn't exist
        log_dir = "lora_weights_logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # Create log file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"lora_weights_{timestamp}.log")

        # Set up logger
        cls._logger = logging.getLogger("lora_weights")
        cls._logger.setLevel(logging.INFO)

        # Remove existing handlers to avoid duplicates
        cls._logger.handlers.clear()

        # File handler
        file_handler = logging.FileHandler(log_file)
        file_formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        file_handler.setFormatter(file_formatter)
        cls._logger.addHandler(file_handler)

        # Console handler for immediate feedback
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter("LORA_WEIGHT: %(message)s")
        console_handler.setFormatter(console_formatter)
        cls._logger.addHandler(console_handler)

        print(f"LoRA weights logging initialized. Log file: {log_file}")

    def _get_bytes_per_element(self, weight_dtype):
        """
        Get the number of bytes per element for a given ttnn data type.

        Args:
            weight_dtype: The ttnn data type (can be string or ttnn.DataType)

        Returns:
            int: Number of bytes per element
        """
        dtype_str = str(weight_dtype).lower()

        # ttnn data type mappings to bytes
        dtype_mappings = {
            # 8-bit types
            "bfloat8_b": 1,
            "int8": 1,
            "uint8": 1,
            # 16-bit types
            "bfloat16": 2,
            "float16": 2,
            "int16": 2,
            "uint16": 2,
            # 32-bit types
            "float32": 4,
            "int32": 4,
            "uint32": 4,
            # 64-bit types
            "float64": 8,
            "int64": 8,
            "uint64": 8,
        }

        # Try to match the data type
        for dtype_key, byte_size in dtype_mappings.items():
            if dtype_key in dtype_str:
                return byte_size

        # Special handling for common patterns
        if "8" in dtype_str and ("bfloat" in dtype_str or "int" in dtype_str):
            return 1
        elif "16" in dtype_str:
            return 2
        elif "32" in dtype_str:
            return 4
        elif "64" in dtype_str:
            return 8

        # Default fallback (log warning for unknown types)
        print(f"Warning: Unknown data type '{weight_dtype}', assuming 4 bytes per element")
        return 4

    def _get_memory_address(self, tensor_obj):
        """
        Extract memory address information from a tensor object.

        Args:
            tensor_obj: The tensor object (ttnn tensor, torch tensor, etc.)

        Returns:
            str: Memory address information
        """
        if tensor_obj is None:
            return "N/A"

        try:
            # For ttnn tensors on device, use buffer_address()
            if hasattr(tensor_obj, "buffer_address"):
                address = tensor_obj.buffer_address()
                return f"0x{address:08x}"

            # For ttnn tensors, try alternative buffer methods
            if hasattr(tensor_obj, "buffer"):
                buffer = tensor_obj.buffer()
                if hasattr(buffer, "address"):
                    return f"0x{buffer.address():08x}"
                elif hasattr(buffer, "data_ptr"):
                    return f"0x{buffer.data_ptr():08x}"

            # For tensors with data_ptr method (torch tensors)
            if hasattr(tensor_obj, "data_ptr"):
                return f"0x{tensor_obj.data_ptr():08x}"

            # For objects with storage
            if hasattr(tensor_obj, "storage"):
                storage = tensor_obj.storage()
                if hasattr(storage, "data_ptr"):
                    return f"0x{storage.data_ptr():08x}"

            # Try to get Python object id as fallback
            obj_id = id(tensor_obj)
            return f"obj_id:0x{obj_id:08x}"

        except Exception as e:
            return f"error:{str(e)[:20]}"

    def _get_tensor_location(self, tensor_obj):
        """
        Determine if tensor is on host or device.

        Args:
            tensor_obj: The tensor object

        Returns:
            str: Location information ("host", "device", "unknown")
        """
        if tensor_obj is None:
            return "unknown"

        try:
            # For ttnn tensors, check if they have buffer_address (device tensors)
            if hasattr(tensor_obj, "buffer_address"):
                return "device"

            # For ttnn tensors, check storage location
            if hasattr(tensor_obj, "storage_type"):
                storage_type = str(tensor_obj.storage_type())
                if "device" in storage_type.lower():
                    return "device"
                elif "host" in storage_type.lower():
                    return "host"

            # For PyTorch tensors, check device
            if hasattr(tensor_obj, "device"):
                device = str(tensor_obj.device)
                if device == "cpu":
                    return "host"
                elif "cuda" in device or "mps" in device:
                    return "device"

            return "unknown"

        except Exception:
            return "unknown"

    def _detect_weight_sharing(self, address, module_path, weight_name):
        """
        Track and detect weight sharing between layers.

        Args:
            address: Memory address of the tensor
            module_path: Current module path
            weight_name: Current weight name

        Returns:
            str: Weight sharing information
        """
        if not hasattr(self, "_address_registry"):
            self._address_registry = {}

        if address == "N/A" or "error:" in address or "obj_id:" in address:
            return ""

        # Check if this address is already used
        if address in self._address_registry:
            existing_location = self._address_registry[address]
            sharing_info = f" [SHARED with {existing_location}]"
        else:
            # Register this address
            self._address_registry[address] = f"{module_path}.{weight_name}"
            sharing_info = ""

        return sharing_info

    def log_weight_creation(
        self,
        module_path,
        weight_name,
        weight_shape,
        weight_dtype,
        weight_device,
        description="",
        tensor_obj=None,
        host_creation_time_ms=None,
        host_to_device_time_ms=None,
    ):
        """
        Log creation of a LoRA-impacted weight.

        Args:
            module_path: Path to the module (e.g., "down_blocks.1.transformer_blocks.0.attn1")
            weight_name: Name of the weight variable (e.g., "tt_q_weights")
            weight_shape: Shape of the weight tensor
            weight_dtype: Data type of the weight
            weight_device: Device where weight is allocated
            description: Optional description of the weight's role
            tensor_obj: The actual tensor object (for memory address extraction)
            host_creation_time_ms: Time to create host tensor (ms)
            host_to_device_time_ms: Time to copy from host to device (ms)
        """
        # Calculate memory usage
        if hasattr(weight_shape, "__iter__"):
            total_elements = 1
            for dim in weight_shape:
                total_elements *= dim
        else:
            total_elements = weight_shape

        # Calculate bytes per element based on actual data type
        bytes_per_element = self._get_bytes_per_element(weight_dtype)
        total_bytes = total_elements * bytes_per_element
        mb_size = total_bytes / (1024 * 1024)

        # Get memory address information
        memory_info = self._get_memory_address(tensor_obj)

        # Get tensor location info (host vs device)
        location_info = self._get_tensor_location(tensor_obj)

        # Detect weight sharing
        sharing_info = self._detect_weight_sharing(memory_info, module_path, weight_name)

        log_message = (
            f"{module_path} | {weight_name} | "
            f"Shape: {weight_shape} | Dtype: {weight_dtype} | "
            f"Device: {weight_device} | Location: {location_info} | Size: {mb_size:.2f} MB | "
            f"Elements: {total_elements:,} | Bytes/elem: {bytes_per_element} | "
            f"Address: {memory_info}{sharing_info}"
        )

        # Add timing information if provided
        if host_creation_time_ms is not None and host_to_device_time_ms is not None:
            total_time_ms = host_creation_time_ms + host_to_device_time_ms
            log_message += f" | Host_Creation: {host_creation_time_ms:.3f} ms | Host_To_Device: {host_to_device_time_ms:.3f} ms | Total_Time: {total_time_ms:.3f} ms"

        if description:
            log_message += f" | {description}"

        self._logger.info(log_message)

    def log_module_start(self, module_path, module_type):
        """Log the start of a module initialization."""
        self._logger.info(f"=== STARTING {module_type}: {module_path} ===")

    def log_module_end(self, module_path, module_type):
        """Log the end of a module initialization."""
        self._logger.info(f"=== FINISHED {module_type}: {module_path} ===")

    def log_summary_stats(self, total_weights, total_memory_mb):
        """Log summary statistics."""
        self._logger.info(
            f"SUMMARY: {total_weights} LoRA-impacted weights created, Total memory: {total_memory_mb:.2f} MB"
        )


# Global instance
lora_logger = LoRAWeightsLogger()
