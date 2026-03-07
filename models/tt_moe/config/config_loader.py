"""
Configuration loader for MoE models.
Loads JSON configuration files for DeepSeek and GPT-OSS backends.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional

import ttnn


def load_json_config(filepath: str) -> Dict[str, Any]:
    """
    Load configuration from a JSON file.

    Args:
        filepath: Path to the JSON configuration file

    Returns:
        Dictionary containing the configuration

    Raises:
        FileNotFoundError: If the configuration file doesn't exist
        json.JSONDecodeError: If the JSON is invalid
    """
    config_path = Path(filepath)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {filepath}")

    with open(config_path, "r") as f:
        config = json.load(f)

    return config


def load_config(backend: str, mode: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration for a specific backend and mode.

    Args:
        backend: Backend type ("deepseek" or "gptoss")
        mode: Optional mode ("decode" or "prefill")

    Returns:
        Configuration dictionary

    Raises:
        ValueError: If backend is not supported
    """
    # Get the directory where this file is located
    config_dir = Path(__file__).parent

    if backend.lower() == "deepseek":
        config_file = config_dir / "deepseek_config.json"
    elif backend.lower() in ["gptoss", "gpt_oss", "gpt-oss"]:
        config_file = config_dir / "gptoss_config.json"
    else:
        raise ValueError(f"Unsupported backend: {backend}. Supported backends: deepseek, gptoss")

    # Load the base configuration
    config = load_json_config(str(config_file))

    # If mode is specified and exists in config, merge mode-specific settings
    if mode and mode in config:
        # Merge mode-specific configuration
        mode_config = config[mode]
        # This would be where we merge mode-specific settings
        # For now, just return the full config

    return config


def get_architecture_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract architecture configuration from the full config.

    Args:
        config: Full configuration dictionary

    Returns:
        Architecture configuration dictionary
    """
    return config.get("architecture", {})


def get_common_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract common configuration from the full config.

    Args:
        config: Full configuration dictionary

    Returns:
        Common configuration dictionary
    """
    return config.get("common", {})


def get_model_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract model configuration from the full config.

    Args:
        config: Full configuration dictionary

    Returns:
        Model configuration dictionary
    """
    return config.get("model", {})


def convert_memory_config(value: Any) -> Any:
    """
    Convert string memory configuration to ttnn memory config object.

    Args:
        value: Memory config value (string or dict)

    Returns:
        ttnn memory config object or original value if not a memory config string
    """
    if isinstance(value, str):
        # Convert string memory configs to ttnn objects
        if value == "L1":
            return ttnn.L1_MEMORY_CONFIG
        elif value == "DRAM":
            return ttnn.DRAM_MEMORY_CONFIG
        elif value == "L1_WIDTH_SHARDED":
            return ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG
        elif value == "L1_HEIGHT_SHARDED":
            return ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG
        elif value == "L1_BLOCK_SHARDED":
            return ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG
        elif value == "DRAM_INTERLEAVED":
            return ttnn.DRAM_MEMORY_CONFIG  # Default DRAM is interleaved
    elif isinstance(value, dict) and value.get("type") == "SHARDED":
        # Handle sharded memory configuration
        # This is a placeholder - the actual sharded config needs more complex handling
        # that will be implemented in the model_config method
        return value  # Return as-is for now, will be processed later

    return value


def get_mode_config(config: Dict[str, Any], mode: str) -> Dict[str, Any]:
    """
    Get mode-specific configuration (decode or prefill).

    Args:
        config: Full configuration dictionary
        mode: Mode name ("decode" or "prefill")

    Returns:
        Mode-specific configuration with memory configs converted
    """
    mode_config = config.get(mode, {})

    # Convert memory configurations in mode config
    if "memory_config" in mode_config:
        mode_config["memory_config"] = convert_memory_config(mode_config["memory_config"])

    if "input_output_memory_config" in mode_config:
        # For sharded configs, keep as dict for later processing
        if isinstance(mode_config["input_output_memory_config"], dict):
            mode_config["input_output_memory_config"] = mode_config["input_output_memory_config"]
        else:
            mode_config["input_output_memory_config"] = convert_memory_config(mode_config["input_output_memory_config"])

    # Convert metadata memory configs if present
    if "all_to_all_dispatch_metadata_memory" in mode_config:
        mode_config["all_to_all_dispatch_metadata_memory"] = convert_memory_config(
            mode_config["all_to_all_dispatch_metadata_memory"]
        )

    return mode_config


def get_cluster_configurations(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get cluster configurations from common config.

    Args:
        config: Full configuration dictionary

    Returns:
        Cluster configurations dictionary
    """
    common_config = get_common_config(config)
    return common_config.get("cluster_configurations", {})


def get_throughput_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get throughput-specific configuration for GPT-OSS.

    Args:
        config: Full configuration dictionary

    Returns:
        Throughput configuration dictionary
    """
    return config.get("throughput_config", {})


def get_weight_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get weight configuration from model config.

    Args:
        config: Full configuration dictionary

    Returns:
        Weight configuration dictionary
    """
    model_config = get_model_config(config)
    return model_config.get("weights", {})
