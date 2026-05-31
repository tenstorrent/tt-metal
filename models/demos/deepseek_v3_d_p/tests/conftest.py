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
from functools import lru_cache
from pathlib import Path

import pytest
import torch
from loguru import logger
from transformers import AutoConfig, AutoTokenizer

import ttnn
from models.common.utility_functions import is_blackhole, is_wormhole_b0
from models.demos.deepseek_v3.utils.config_helpers import sub_state_dict
from models.demos.deepseek_v3.utils.test_utils import dequantize_state_dict, load_state_dict
from models.demos.deepseek_v3_d_p.tests.model_variants import DSV3, TEST_VARIANTS, TestVariant
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


@pytest.fixture
def variant(request) -> TestVariant:
    """The active test variant. Driven by indirect parametrize:
        @pytest.mark.parametrize("variant", ["deepseek_v3"], indirect=True)
    Tests that don't parametrize fall back to DSv3.
    """
    param = getattr(request, "param", None)
    if param is None:
        return DSV3
    return TEST_VARIANTS[param] if isinstance(param, str) else param


def download_model_config_only(variant: TestVariant, cache_dir: Path) -> Path:
    """Download only config files (no weight shards) for the variant's HF repo."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        logger.error("huggingface_hub is not installed. Install it with: pip install huggingface_hub")
        raise

    logger.info(f"Downloading {variant.hf_repo_id} config-only from HuggingFace")
    logger.info(f"Cache directory: {cache_dir}")

    cache_dir.mkdir(parents=True, exist_ok=True)

    allow_patterns = [
        "config.json",
        "*.safetensors.index.json",
        "generation_config.json",
        "tokenizer*",
        "vocab*",
        "merges*",
        "special_tokens_map.json",
        "*.model",
        "*.tiktoken",
        "*.py",
    ]

    try:
        model_dir = snapshot_download(
            repo_id=variant.hf_repo_id,
            cache_dir=str(cache_dir),
            allow_patterns=allow_patterns,
            ignore_patterns=["*.safetensors"],
        )
        logger.success(f"✓ Config files downloaded to: {model_dir}")
        return Path(model_dir)
    except Exception as e:
        logger.error(f"Failed to download {variant.hf_repo_id} config: {e}")
        raise


def download_model_weights(variant: TestVariant, cache_dir: Path, layer_idx: int = 0, num_layers: int = 1) -> Path:
    """Download model weights for the variant's HF repo.

    Pulls shards covering layers [layer_idx, layer_idx+num_layers) plus embeddings
    and `model.norm.weight`. Other shards are skipped to bound download size.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        logger.error("huggingface_hub is not installed. Install it with: pip install huggingface_hub")
        raise

    logger.info(f"Downloading {variant.hf_repo_id} weights from HuggingFace")
    logger.info(f"Cache directory: {cache_dir}")
    logger.info(f"Pulling shards for layers {layer_idx}..{layer_idx + num_layers - 1} + embed + norm")

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
            "vocab*",
            "merges*",
            "special_tokens_map.json",
            "*.model",
            "*.tiktoken",
            "*.py",  # custom model code for trust_remote_code=True
        ]

        # First download just the index to figure out which shards we need
        index_dir = snapshot_download(
            repo_id=variant.hf_repo_id,
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
            repo_id=variant.hf_repo_id,
            cache_dir=str(cache_dir),
            allow_patterns=allow_patterns + shard_patterns,
        )

        logger.success(f"✓ Model weights downloaded successfully!")
        logger.info(f"Model location: {model_dir}")
        return Path(model_dir)

    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        logger.info(f"You can also manually set {variant.env_var} to point to existing weights")
        raise


def get_or_download_model(variant: TestVariant, layer_idx: int = 0, num_layers: int = 6) -> Path:
    """
    Get model path, downloading from HuggingFace if necessary.

    Args:
        layer_idx: Which layer weights to ensure are available
        num_layers: Number of layers to download (default: 6).
                    When >1, downloads additional shards including shard 160 for model.norm.

    Returns:
        Path to model directory with weights
    """
    env_path = os.getenv(variant.env_var)
    if env_path:
        candidate = Path(env_path)
        if (candidate / "model.safetensors.index.json").exists():
            logger.info(f"Using existing model from {variant.env_var}: {candidate}")
            return candidate.resolve()
        logger.warning(f"{variant.env_var} set but missing index file in {candidate}")

    for label, p in (("default", variant.default_local_path), ("shared", variant.shared_path)):
        if p is not None and (p / "model.safetensors.index.json").exists():
            logger.info(f"Using model from {label} location: {p}")
            return p.resolve()

    logger.info(f"Model not found locally. Downloading {variant.hf_repo_id} from HuggingFace...")
    cache_dir = Path(os.getenv("HF_HOME", Path.home() / ".cache" / "huggingface"))
    return download_model_weights(variant, cache_dir, layer_idx, num_layers)


@lru_cache(maxsize=None)
def _resolve_model_path(variant_name: str) -> Path:
    v = TEST_VARIANTS[variant_name]
    return get_or_download_model(v, layer_idx=0, num_layers=v.num_layers_to_download)


@lru_cache(maxsize=None)
def _resolve_config_only(variant_name: str):
    v = TEST_VARIANTS[variant_name]
    env_path = os.getenv(v.env_var)
    candidates = []
    if env_path:
        candidates.append(Path(env_path))
    if v.default_local_path is not None:
        candidates.append(v.default_local_path)
    if v.shared_path is not None:
        candidates.append(v.shared_path)
    for p in candidates:
        if (p / "config.json").exists():
            logger.info(f"Using config from {p}")
            return AutoConfig.from_pretrained(str(p), trust_remote_code=True)
    logger.info(f"Config not found locally. Downloading {v.hf_repo_id} config-only from HuggingFace...")
    cache_dir = Path(os.getenv("HF_HOME", Path.home() / ".cache" / "huggingface"))
    config_path = download_model_config_only(v, cache_dir)
    return AutoConfig.from_pretrained(str(config_path), trust_remote_code=True)


@lru_cache(maxsize=None)
def _resolve_hf_config(model_path_str: str):
    p = Path(model_path_str)
    if not (p / "config.json").exists():
        return None
    try:
        cfg = AutoConfig.from_pretrained(str(p), trust_remote_code=True)
        logger.info(f"Loaded HF config from {p}")
        return cfg
    except Exception as e:
        logger.warning(f"Failed to load HF config from {p}: {e}")
        return None


@lru_cache(maxsize=None)
def _resolve_state_dict(model_path_str: str):
    p = Path(model_path_str)
    if not (p / "model.safetensors.index.json").exists():
        return None
    try:
        sd = load_state_dict(p, "")
        logger.info(f"Loaded state dict from {p}")
        return sd
    except Exception as e:
        logger.warning(f"Failed to load state dict from {p}: {e}")
        return None


@lru_cache(maxsize=None)
def _resolve_tokenizer(variant_name: str, padding_side: str):
    v = TEST_VARIANTS[variant_name]
    env_path = os.getenv(v.env_var)
    candidates = []
    if env_path:
        candidates.append(Path(env_path))
    if v.default_local_path is not None:
        candidates.append(v.default_local_path)
    if v.shared_path is not None:
        candidates.append(v.shared_path)
    for p in candidates:
        if p.exists() and any(p.glob("tokenizer*")):
            logger.info(f"Loading tokenizer from: {p}")
            tok = AutoTokenizer.from_pretrained(str(p), use_fast=True, trust_remote_code=True)
            tok.padding_side = padding_side
            return tok
    cache_dir = Path(os.getenv("HF_HOME", Path.home() / ".cache" / "huggingface"))
    config_path = download_model_config_only(v, cache_dir)
    logger.info(f"Loading tokenizer from downloaded config: {config_path}")
    tok = AutoTokenizer.from_pretrained(str(config_path), use_fast=True, trust_remote_code=True)
    tok.padding_side = padding_side
    return tok


@pytest.fixture
def model_path(variant) -> Path:
    return _resolve_model_path(variant.name)


@pytest.fixture
def hf_config(variant, model_path):
    return _resolve_hf_config(str(model_path))


@pytest.fixture
def config_only(variant):
    return _resolve_config_only(variant.name)


@pytest.fixture(params=["right"])
def tokenizer(request, variant):
    """Padding side via indirect parametrize. Defaults to right-padding."""
    return _resolve_tokenizer(variant.name, request.param)


@pytest.fixture
def state_dict(variant, model_path):
    return _resolve_state_dict(str(model_path))


def _check_pretrained_available(model_path: Path) -> bool:
    index_file = model_path / "model.safetensors.index.json"
    config_file = model_path / "config.json"

    available = index_file.exists() and config_file.exists()

    if available:
        logger.info(f"✓ Pretrained weights found at {model_path}")
    else:
        logger.info(f"✗ Pretrained weights not found at {model_path}")

    return available


@pytest.fixture
def weight_cache_path(variant, model_path):
    """Directory for caching TTNN weight tensors. None if pretrained weights aren't on disk."""
    if not _check_pretrained_available(model_path):
        return None
    arch = "bh" if is_blackhole() else "wh"
    num_devices = ttnn.get_num_devices()
    env_cache = os.getenv("TT_DS_PREFILL_TTNN_CACHE")
    if env_cache:
        cache_dir = Path(env_cache) / f"{variant.weight_cache_prefix}_{arch}_{num_devices}dev"
    else:
        cache_dir = model_path / f"tensor_cache_{arch}_{num_devices}dev"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


@pytest.fixture
def random_weights(variant, config_only):
    """Random MLA weights sized from the variant's config."""
    config = config_only
    torch.manual_seed(42)  # tied to cached reference results; keep stable

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
def pretrained_transformer_weights(variant, model_path, hf_config, state_dict, request):
    """Dequantized pretrained weights for N-layer transformer in TT state_dict format.

    Skips if the variant doesn't support pretrained weights, or if config/state_dict
    couldn't be loaded.
    """
    if not variant.supports_pretrained:
        pytest.skip(f"{variant.name}: pretrained weights not wired")
    if not _check_pretrained_available(model_path):
        pytest.skip(f"{variant.name}: pretrained weights not available. Set {variant.env_var} or download.")
    if hf_config is None:
        pytest.skip(f"{variant.name}: failed to load HF config")
    if state_dict is None:
        pytest.skip(f"{variant.name}: failed to load state dict")

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
