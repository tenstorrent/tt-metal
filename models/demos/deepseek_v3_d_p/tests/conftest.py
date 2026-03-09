# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""
Shared fixtures for deepseek_v3_d_p tests.
Reuses fixtures from parent conftest and provides pretrained weights checking.
Automatically downloads weights from HuggingFace if not available locally.
"""

import os
from pathlib import Path

import pytest
from loguru import logger
from transformers import AutoConfig

from models.demos.deepseek_v3.utils.config_helpers import sub_state_dict
from models.demos.deepseek_v3.utils.test_utils import dequantize_state_dict, load_state_dict


def download_model_config_only(cache_dir: Path) -> Path:
    """
    Download only DeepSeek-R1-0528 config files (without weight shards).
    This is fast and only downloads ~few MB for config files.

    Args:
        cache_dir: Directory to cache downloaded config

    Returns:
        Path to the downloaded model directory with config
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        logger.error("huggingface_hub is not installed. Install it with: pip install huggingface_hub")
        raise

    model_id = "deepseek-ai/DeepSeek-R1-0528"
    logger.info(f"Downloading DeepSeek-R1-0528 config only (no weights) from HuggingFace")
    logger.info(f"Cache directory: {cache_dir}")

    # Create cache directory
    cache_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Download only config files, no weight shards
        allow_patterns = [
            "config.json",
            "*.safetensors.index.json",
            "generation_config.json",
            "tokenizer*",
        ]

        # Add custom model code files (needed for trust_remote_code=True)
        allow_patterns.extend(
            [
                "configuration_deepseek.py",
                "modeling_deepseek.py",
                "*.py",  # Include all Python files for custom model code
            ]
        )

        model_dir = snapshot_download(
            repo_id=model_id,
            cache_dir=str(cache_dir),
            allow_patterns=allow_patterns,
            ignore_patterns=["*.safetensors"],  # Don't download weight files
        )

        logger.success(f"✓ Config files downloaded to: {model_dir}")
        return Path(model_dir)

    except Exception as e:
        logger.error(f"Failed to download config: {e}")
        raise


def download_model_weights(cache_dir: Path, layer_idx: int = 0) -> Path:
    """
    Download DeepSeek-R1-0528 model weights from HuggingFace.

    Args:
        cache_dir: Directory to cache downloaded weights
        layer_idx: Which layer to download weights for (default: 0)

    Returns:
        Path to the downloaded model directory
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        logger.error("huggingface_hub is not installed. Install it with: pip install huggingface_hub")
        raise

    model_id = "deepseek-ai/DeepSeek-R1-0528"
    logger.info(f"Downloading DeepSeek-R1-0528 weights from HuggingFace (model: {model_id})")
    logger.info(f"Cache directory: {cache_dir}")
    logger.info(f"Note: Only downloading files needed for layer {layer_idx} to minimize download size")

    # Create cache directory
    cache_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Download essential files + index
        logger.info("Step 1/2: Downloading configuration and index files...")
        allow_patterns = [
            "config.json",
            "*.safetensors.index.json",
            "generation_config.json",
            "tokenizer*",
        ]

        # Add custom model code files (needed for trust_remote_code=True)
        allow_patterns.extend(
            [
                "configuration_deepseek.py",
                "modeling_deepseek.py",
                "*.py",  # Include all Python files for custom model code
            ]
        )

        # First download just the index to figure out which shards we need
        index_dir = snapshot_download(
            repo_id=model_id,
            cache_dir=str(cache_dir),
            allow_patterns=allow_patterns,
            ignore_patterns=["*.safetensors"],  # Don't download weight files yet
        )

        logger.info(f"✓ Configuration downloaded to: {index_dir}")

        # Now download the first few weight shards (layer 0 is usually in first 1-3 shards)
        logger.info("Step 2/2: Downloading weight files for first layer...")
        logger.info("This will download ~3-5GB (first few shards containing layer 0 weights)")

        # Download first 3 shards which should contain layer 0
        shard_patterns = [
            "*-00001-of-*.safetensors",
            "*-00002-of-*.safetensors",
            "*-00003-of-*.safetensors",
        ]

        model_dir = snapshot_download(
            repo_id=model_id,
            cache_dir=str(cache_dir),
            allow_patterns=allow_patterns + shard_patterns,
        )

        logger.success(f"✓ Model weights downloaded successfully!")
        logger.info(f"Model location: {model_dir}")
        return Path(model_dir)

    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        logger.info("You can also manually set DEEPSEEK_V3_HF_MODEL to point to existing weights")
        raise


def get_or_download_model(layer_idx: int = 0) -> Path:
    """
    Get model path, downloading from HuggingFace if necessary.

    Args:
        layer_idx: Which layer weights to ensure are available

    Returns:
        Path to model directory with weights
    """
    # Check environment variable first
    env_path = os.getenv("DEEPSEEK_V3_HF_MODEL")
    if env_path:
        model_path = Path(env_path)
        if model_path.exists():
            index_file = model_path / "model.safetensors.index.json"
            if index_file.exists():
                logger.info(f"Using existing model from DEEPSEEK_V3_HF_MODEL: {model_path}")
                return model_path.resolve()
            else:
                logger.warning(f"DEEPSEEK_V3_HF_MODEL set but missing index file: {index_file}")

    # Check default location
    default_path = Path("models/demos/deepseek_v3/reference")
    if default_path.exists():
        index_file = default_path / "model.safetensors.index.json"
        if index_file.exists():
            logger.info(f"Using model from default location: {default_path}")
            return default_path.resolve()

    # Download from HuggingFace
    logger.info("Model not found locally. Downloading DeepSeek-R1-0528 from HuggingFace...")

    # Determine cache directory
    cache_dir = Path(os.getenv("HF_HOME", Path.home() / ".cache" / "huggingface"))
    logger.info(f"Will cache to: {cache_dir}")
    logger.warning("⚠️  This will download ~3-5GB for the first layer weights")
    logger.info("The full DeepSeek-R1-0528 model is large, but we only download what's needed for testing")

    return download_model_weights(cache_dir, layer_idx)


@pytest.fixture(scope="session")
def model_path():
    """
    Get model path and resolve symlinks to ensure all operations can find files.
    Automatically downloads weights from HuggingFace if not available locally.

    Checks in order:
    1. DEEPSEEK_V3_HF_MODEL environment variable
    2. models/demos/deepseek_v3/reference/ (default location)
    3. Downloads from HuggingFace to HF cache if not found
    """
    return get_or_download_model(layer_idx=0)


@pytest.fixture(scope="session")
def hf_config(model_path):
    """
    Load DeepSeek config for testing.
    Returns None if model path doesn't exist (weights not available).
    """
    # Check if model path exists
    if not model_path.exists():
        return None

    # Check if config.json exists
    config_file = model_path / "config.json"
    if not config_file.exists():
        return None

    try:
        config = AutoConfig.from_pretrained(str(model_path), trust_remote_code=True)
        logger.info(f"Loaded HF config from {model_path}")
        return config
    except Exception as e:
        logger.warning(f"Failed to load config from {model_path}: {e}")
        return None


@pytest.fixture(scope="session")
def config_only():
    """
    Load DeepSeek config for random weight tests (downloads only config, not weights).
    This is fast and only downloads ~few MB.
    """
    # Check environment variable first
    env_path = os.getenv("DEEPSEEK_V3_HF_MODEL")
    if env_path:
        model_path = Path(env_path)
        if model_path.exists():
            config_file = model_path / "config.json"
            if config_file.exists():
                logger.info(f"Using existing config from DEEPSEEK_V3_HF_MODEL: {model_path}")
                config = AutoConfig.from_pretrained(str(model_path), trust_remote_code=True)
                return config

    # Check default location
    default_path = Path("models/demos/deepseek_v3/reference")
    if default_path.exists():
        config_file = default_path / "config.json"
        if config_file.exists():
            logger.info(f"Using config from default location: {default_path}")
            config = AutoConfig.from_pretrained(str(default_path), trust_remote_code=True)
            return config

    # Download only config files from HuggingFace (not weight shards)
    logger.info("Config not found locally. Downloading config only (no weights) from HuggingFace...")

    cache_dir = Path(os.getenv("HF_HOME", Path.home() / ".cache" / "huggingface"))
    logger.info(f"Will cache config to: {cache_dir}")

    config_path = download_model_config_only(cache_dir)
    config = AutoConfig.from_pretrained(str(config_path), trust_remote_code=True)
    logger.info(f"Loaded config from {config_path}")
    return config


@pytest.fixture(scope="session")
def state_dict(model_path):
    """
    Load state dict for testing.
    Returns None if model path doesn't exist (weights not available).
    """
    # Check if model path exists
    if not model_path.exists():
        return None

    # Check if model.safetensors.index.json exists
    index_file = model_path / "model.safetensors.index.json"
    if not index_file.exists():
        return None

    try:
        state_dict = load_state_dict(model_path, "")
        logger.info(f"Loaded state dict from {model_path}")
        return state_dict
    except Exception as e:
        logger.warning(f"Failed to load state dict from {model_path}: {e}")
        return None


def _check_pretrained_available(model_path: Path = None) -> bool:
    """
    Check if pretrained weights are available.

    Args:
        model_path: Optional model path to check. If None, uses default from env or fallback.

    Returns:
        True if pretrained weights are available, False otherwise.
    """
    if model_path is None:
        model_path = Path(os.getenv("DEEPSEEK_V3_HF_MODEL", "models/demos/deepseek_v3/reference"))

    index_file = model_path / "model.safetensors.index.json"
    config_file = model_path / "config.json"

    available = index_file.exists() and config_file.exists()

    if available:
        logger.info(f"✓ Pretrained weights found at {model_path}")
    else:
        logger.info(f"✗ Pretrained weights not found at {model_path}")

    return available


@pytest.fixture
def pretrained_weights(model_path, hf_config, state_dict):
    """
    Load pretrained weights from DeepSeek model (layer 0 only).

    This fixture reuses the shared model_path, hf_config, and state_dict fixtures
    to ensure consistent weight loading across all tests.

    Returns:
        Tuple of (config, weights_dict) or skips if not available
    """
    # Check if pretrained weights are available
    if not _check_pretrained_available(model_path):
        pytest.skip("Pretrained weights not available. Set DEEPSEEK_V3_HF_MODEL or download model.")

    # Check if fixtures loaded successfully
    if hf_config is None:
        pytest.skip("Failed to load HF config. Check model path.")

    if state_dict is None:
        pytest.skip("Failed to load state dict. Check model path and weights.")

    logger.info(f"Loading pretrained weights from: {model_path}")

    # Extract layer 0 attention weights
    layer_idx = 0
    module_path = f"model.layers.{layer_idx}.self_attn"

    layer_state_dict = sub_state_dict(state_dict, module_path + ".")
    dequantized_weights = dequantize_state_dict(layer_state_dict, hf_config)

    logger.info(f"Loaded {len(dequantized_weights)} pretrained weight tensors")

    return hf_config, dequantized_weights
