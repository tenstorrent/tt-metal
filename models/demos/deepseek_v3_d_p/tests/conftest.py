# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Shared fixtures for deepseek_v3_d_p tests.
Provides mesh topology markers and pretrained weights checking.
Automatically downloads weights from HuggingFace if not available locally.
"""

import json
import os
from pathlib import Path

import pytest
import torch
from loguru import logger
from transformers import AutoConfig, AutoTokenizer

import ttnn
from models.common.utility_functions import is_blackhole, is_wormhole_b0
from models.demos.deepseek_v3.utils.config_helpers import sub_state_dict
from models.demos.deepseek_v3.utils.test_utils import dequantize_state_dict, load_state_dict
from models.demos.deepseek_v3_d_p.utils.transformer_helpers import download_infinitebench_subset


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "requires_mesh_topology(mesh_shape, topology): mark test to run only on compatible "
        "device/topology combinations. mesh_shape is (rows, cols) tuple, topology is 'ring' or 'linear'. "
        "Skips automatically based on available devices and arch constraints.",
    )


def pytest_collection_modifyitems(config, items):
    """
    Skip tests based on mesh/topology requirements at collection time.

    Hardware constraints:
    - Blackhole: Only supports 4-device configs (linear-4, ring-4)
    - Wormhole: Ring topology only works with 8 devices (ring-8)
    """

    for item in items:
        marker = item.get_closest_marker("requires_mesh_topology")
        if not marker:
            continue

        # this opens a device
        num_devices = ttnn.get_num_devices()

        # Extract marker arguments
        mesh_shape = marker.kwargs.get("mesh_shape") or (marker.args[0] if marker.args else None)
        topology = marker.kwargs.get("topology") or (marker.args[1] if len(marker.args) > 1 else None)

        if mesh_shape is None or topology is None:
            continue

        devices_needed = mesh_shape[0] * mesh_shape[1]
        is_ring = topology == "ring"

        skip_reason = None

        # Check device count first
        if devices_needed > num_devices:
            skip_reason = f"Requires {devices_needed} devices, only {num_devices} available"

        # Architecture-specific constraints
        elif is_blackhole():
            # BH: only supports all available devices configs
            if devices_needed != num_devices:
                skip_reason = f"Blackhole only supports {num_devices}-device mesh configs (requested {devices_needed})"

        elif is_wormhole_b0():
            # WH: ring topology only works with 8 devices
            if is_ring and devices_needed != 8:
                skip_reason = f"Wormhole ring topology only works with 8 devices (requested ring-{devices_needed})"

        if skip_reason:
            item.add_marker(pytest.mark.skip(reason=skip_reason))


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


def download_model_weights(cache_dir: Path, layer_idx: int = 0, num_layers: int = 1) -> Path:
    """
    Download DeepSeek-R1-0528 model weights from HuggingFace.

    Args:
        cache_dir: Directory to cache downloaded weights
        layer_idx: Which layer to download weights for (default: 0)
        num_layers: Number of layers to download weights for (default: 1).
            When >1, downloads additional shards for layers 0..num_layers-1.

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

        # Systematically determine which shards are needed based on the index
        index_path = Path(index_dir) / "model.safetensors.index.json"
        with open(index_path, "r") as f:
            index_data = json.load(f)

        weight_map = index_data.get("weight_map", {})
        required_shards = set()

        # Find shards for embeddings
        for key, shard_file in weight_map.items():
            if "embed_tokens" in key:
                required_shards.add(shard_file)

        # Find shards for the requested layers
        for layer_id in range(layer_idx, layer_idx + num_layers):
            for key, shard_file in weight_map.items():
                if f"model.layers.{layer_id}." in key:
                    required_shards.add(shard_file)

        # Find shard for model.norm (always needed by pretrained_transformer_weights fixture)
        for key, shard_file in weight_map.items():
            if "model.norm.weight" in key:
                required_shards.add(shard_file)
                break

        # Convert shard filenames to patterns
        shard_patterns = []
        for shard_file in sorted(required_shards):
            # Extract shard number from filename like "model-00001-of-000163.safetensors"
            shard_num = shard_file.split("-")[1]
            shard_patterns.append(f"*-{shard_num}-of-*.safetensors")

        logger.info(
            f"Step 2/2: Downloading weight shards for layers {layer_idx}..{layer_idx + num_layers - 1} + embeddings + norm..."
        )
        logger.info(
            f"Required shards: {len(required_shards)} files ({', '.join(sorted(required_shards)[:5])}{'...' if len(required_shards) > 5 else ''})"
        )
        estimated_size_gb = len(required_shards) * 0.28  # Approximate 280MB per shard
        logger.info(f"Estimated download size: ~{estimated_size_gb:.1f}GB")

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


def get_or_download_model(layer_idx: int = 0, num_layers: int = 6) -> Path:
    """
    Get model path, downloading from HuggingFace if necessary.

    Args:
        layer_idx: Which layer weights to ensure are available
        num_layers: Number of layers to download (default: 6).
                    When >1, downloads additional shards including shard 160 for model.norm.

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

    # Check shared weights location
    shared_path = Path("/proj_sw/user_dev/deepseek-ai/DeepSeek-R1-0528")
    if shared_path.exists():
        index_file = shared_path / "model.safetensors.index.json"
        if index_file.exists():
            logger.info(f"Using model from shared location: {shared_path}")
            return shared_path.resolve()

    # Download from HuggingFace
    logger.info("Model not found locally. Downloading DeepSeek-R1-0528 from HuggingFace...")

    # Determine cache directory
    cache_dir = Path(os.getenv("HF_HOME", Path.home() / ".cache" / "huggingface"))
    logger.info(f"Will cache to: {cache_dir}")
    # Note: Detailed download size is logged by download_model_weights() after analyzing the index

    return download_model_weights(cache_dir, layer_idx, num_layers)


@pytest.fixture(scope="session")
def model_path():
    """
    Get model path and resolve symlinks to ensure all operations can find files.
    Automatically downloads weights from HuggingFace if not available locally.
    Downloads weights for layers 0-11 (12 layers total) to support all test cases.

    Checks in order:
    1. DEEPSEEK_V3_HF_MODEL environment variable
    2. models/demos/deepseek_v3/reference/ (default location)
    3. Downloads from HuggingFace to HF cache if not found
    """
    return get_or_download_model(layer_idx=0, num_layers=24)


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

    # Check shared weights location
    shared_path = Path("/proj_sw/user_dev/deepseek-ai/DeepSeek-R1-0528")
    if shared_path.exists():
        config_file = shared_path / "config.json"
        if config_file.exists():
            logger.info(f"Using config from shared location: {shared_path}")
            config = AutoConfig.from_pretrained(str(shared_path), trust_remote_code=True)
            return config

    # Download only config files from HuggingFace (not weight shards)
    logger.info("Config not found locally. Downloading config only (no weights) from HuggingFace...")

    cache_dir = Path(os.getenv("HF_HOME", Path.home() / ".cache" / "huggingface"))
    logger.info(f"Will cache config to: {cache_dir}")

    config_path = download_model_config_only(cache_dir)
    config = AutoConfig.from_pretrained(str(config_path), trust_remote_code=True)
    logger.info(f"Loaded config from {config_path}")
    return config


@pytest.fixture(scope="session", params=["right"])
def tokenizer(request):
    """Load DeepSeek tokenizer, searching known model locations.

    Default padding_side is "right" (back-padding). To test with left padding,
    override in your test: @pytest.mark.parametrize("tokenizer", ["left"], indirect=True)
    """
    padding_side = request.param
    candidates = [
        os.getenv("DEEPSEEK_V3_HF_MODEL"),
        "models/demos/deepseek_v3/reference",
        "/proj_sw/user_dev/deepseek-ai/DeepSeek-R1-0528",
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        p = Path(candidate)
        if p.exists() and any(p.glob("tokenizer*")):
            logger.info(f"Loading tokenizer from: {p}")
            tok = AutoTokenizer.from_pretrained(str(p), use_fast=True, trust_remote_code=True)
            tok.padding_side = padding_side
            return tok

    # Fall back to downloading config-only (includes tokenizer files)
    cache_dir = Path(os.getenv("HF_HOME", Path.home() / ".cache" / "huggingface"))
    config_path = download_model_config_only(cache_dir)
    logger.info(f"Loading tokenizer from downloaded config: {config_path}")
    tok = AutoTokenizer.from_pretrained(str(config_path), use_fast=True, trust_remote_code=True)
    tok.padding_side = padding_side
    return tok


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


@pytest.fixture(scope="session")
def weight_cache_path(model_path):
    """
    Return a directory for caching TTNN weight tensors (.tensorbin files).

    First run: ttnn.as_tensor() dumps converted weights here.
    Subsequent runs: weights are loaded directly, bypassing torch conversion.

    The path encodes architecture + device count to prevent cross-config clashes.
    Returns None if pretrained weights are unavailable (random-weight tests skip caching).
    """
    if not _check_pretrained_available(model_path):
        return None
    arch = "bh" if is_blackhole() else "wh"
    num_devices = ttnn.get_num_devices()
    env_cache = os.getenv("TT_DS_PREFILL_TTNN_CACHE")
    if env_cache:
        cache_dir = Path(env_cache) / f"deepseek_v3_d_p_{arch}_{num_devices}dev"
    else:
        cache_dir = model_path / f"tensor_cache_{arch}_{num_devices}dev"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


@pytest.fixture
def random_weights(config_only):
    """
    Generate random weights for testing using the config.

    Args:
        config_only: HuggingFace config (only downloads config files, not weight shards)

    Returns:
        Tuple of (config, weights_dict) in bfloat16
    """
    config = config_only

    torch.manual_seed(42)  # this is tied to already cached reference results, so keep it consistent for now

    # Use proper initialization scale from config (typically 0.02)
    std = config.initializer_range

    # Generate random weights matching MLA architecture using actual config
    # Generate in float32 first, then convert to bfloat16 for better numerical properties
    weights = {
        "q_a_proj.weight": (torch.randn(config.q_lora_rank, config.hidden_size) * std).to(torch.bfloat16),
        "q_a_layernorm.weight": torch.ones(config.q_lora_rank, dtype=torch.bfloat16),
        "q_b_proj.weight": (
            torch.randn(
                config.num_attention_heads * (config.qk_nope_head_dim + config.qk_rope_head_dim),
                config.q_lora_rank,
            )
            * std
        ).to(torch.bfloat16),
        "kv_a_proj_with_mqa.weight": (
            torch.randn(
                config.kv_lora_rank + config.qk_rope_head_dim,
                config.hidden_size,
            )
            * std
        ).to(torch.bfloat16),
        "kv_a_layernorm.weight": torch.ones(config.kv_lora_rank, dtype=torch.bfloat16),
        "kv_b_proj.weight": (
            torch.randn(
                config.num_attention_heads * (config.qk_nope_head_dim + config.v_head_dim),
                config.kv_lora_rank,
            )
            * std
        ).to(torch.bfloat16),
        "o_proj.weight": (
            torch.randn(
                config.hidden_size,
                config.num_attention_heads * config.v_head_dim,
            )
            * std
        ).to(torch.bfloat16),
    }

    logger.info(f"Generated {len(weights)} random weight tensors using config dimensions")
    return config, weights


@pytest.fixture
def pretrained_transformer_weights(model_path, hf_config, state_dict, request):
    """
    Dequantized pretrained weights for N-layer transformer in TT state_dict format.

    Extracts embed, norm, and per-layer weights (attention, FFN/MoE) using
    sub_state_dict() + dequantize_state_dict(), matching the format produced
    by extract_tt_state_dict() in transformer_helpers.py.

    Parametrize with num_layers (default 6) via indirect fixture or marker:
        @pytest.mark.parametrize("pretrained_transformer_weights", [4], indirect=True)

    Returns:
        Tuple of (hf_config, tt_state_dict) or skips if not available
    """
    if not _check_pretrained_available(model_path):
        pytest.skip("Pretrained weights not available. Set DEEPSEEK_V3_HF_MODEL or download model.")
    if hf_config is None:
        pytest.skip("Failed to load HF config. Check model path.")
    if state_dict is None:
        pytest.skip("Failed to load state dict. Check model path and weights.")

    num_layers = request.node.callspec.params.get("num_layers", 1)
    first_k_dense = hf_config.first_k_dense_replace  # 3
    n_routed = hf_config.n_routed_experts  # 256

    logger.info(f"Loading pretrained transformer weights for {num_layers} layers from: {model_path}")

    # Embed tokens
    embed_sd = sub_state_dict(state_dict, "model.embed_tokens.")
    embed_dequant = dequantize_state_dict(embed_sd, hf_config)
    result = {
        "embed_weight": embed_dequant["weight"].float(),
    }

    # Final norm
    norm_sd = sub_state_dict(state_dict, "model.norm.")
    norm_dequant = dequantize_state_dict(norm_sd, hf_config)
    result["norm_weight"] = norm_dequant["weight"]

    # Per-layer weights
    result["layers"] = []
    for i in range(num_layers):
        logger.info(f"Loading layer {i} weights...")
        layer_sd = sub_state_dict(state_dict, f"model.layers.{i}.")
        layer_dequant = dequantize_state_dict(layer_sd, hf_config)

        layer_dict = {
            "attn_norm_weight": layer_dequant["input_layernorm.weight"],
            "mla_weights": {
                "q_a_proj.weight": layer_dequant["self_attn.q_a_proj.weight"],
                "q_a_layernorm.weight": layer_dequant["self_attn.q_a_layernorm.weight"],
                "q_b_proj.weight": layer_dequant["self_attn.q_b_proj.weight"],
                "kv_a_proj_with_mqa.weight": layer_dequant["self_attn.kv_a_proj_with_mqa.weight"],
                "kv_a_layernorm.weight": layer_dequant["self_attn.kv_a_layernorm.weight"],
                "kv_b_proj.weight": layer_dequant["self_attn.kv_b_proj.weight"],
                "o_proj.weight": layer_dequant["self_attn.o_proj.weight"],
            },
            "ffn_norm_weight": layer_dequant["post_attention_layernorm.weight"],
        }

        is_dense = i < first_k_dense
        if is_dense:
            layer_dict["ffn_weights"] = {
                "gate_proj": layer_dequant["mlp.gate_proj.weight"],
                "up_proj": layer_dequant["mlp.up_proj.weight"],
                "down_proj": layer_dequant["mlp.down_proj.weight"],
            }
        else:
            layer_dict["gate_weights"] = {
                "weight": layer_dequant["mlp.gate.weight"],
                "e_score_correction_bias": layer_dequant["mlp.gate.e_score_correction_bias"],
            }
            layer_dict["routed_expert_weights"] = [
                {
                    "gate_proj": layer_dequant[f"mlp.experts.{j}.gate_proj.weight"],
                    "up_proj": layer_dequant[f"mlp.experts.{j}.up_proj.weight"],
                    "down_proj": layer_dequant[f"mlp.experts.{j}.down_proj.weight"],
                }
                for j in range(n_routed)
            ]
            layer_dict["shared_expert_weights"] = {
                "gate_proj": layer_dequant["mlp.shared_experts.gate_proj.weight"],
                "up_proj": layer_dequant["mlp.shared_experts.up_proj.weight"],
                "down_proj": layer_dequant["mlp.shared_experts.down_proj.weight"],
            }

        result["layers"].append(layer_dict)
        logger.info(f"Layer {i} loaded ({'dense' if is_dense else 'MoE'})")

    logger.info(f"Loaded pretrained transformer weights for {num_layers} layers")
    return hf_config, result


# ---------------------------------------------------------------------------
# InfiniteBench prompt fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def infinitebench_prompt(request):
    """
    Pytest fixture that provides a long prompt from InfiniteBench.

    Parametrize with the subset name to select which category:

        @pytest.mark.parametrize("infinitebench_prompt",
            ["passkey", "kv_retrieval", "longdialogue_qa_eng", "longbook_qa_eng"],
            indirect=True,
        )
        def test_prefill(infinitebench_prompt):
            subset, prompt_text = infinitebench_prompt
            ...

    Downloads from HuggingFace on first use, then caches locally.

    Returns:
        Tuple of (subset_name, prompt_text).
    """
    subset = request.param
    cached_path = download_infinitebench_subset(subset)

    with open(cached_path) as f:
        data = json.load(f)

    return data["subset"], data["prompt"]
