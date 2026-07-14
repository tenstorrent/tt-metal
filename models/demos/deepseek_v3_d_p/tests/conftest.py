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
import shutil
import tempfile
from functools import lru_cache
from pathlib import Path

import pytest
import torch
from loguru import logger
from transformers import AutoConfig, AutoTokenizer

import ttnn
from models.common.utility_functions import is_blackhole, is_wormhole_b0
from models.demos.common.prefill.adapter import ADAPTER_PATHS, PrefillModelAdapter, get_adapter
from models.demos.deepseek_v3.utils.config_helpers import sub_state_dict
from models.demos.deepseek_v3.utils.test_utils import load_state_dict

# The per-model registry now lives in models/demos/common/prefill/adapter.py and is shared by the
# runner and the tests. These aliases keep the existing fixture/test references (TestVariant /
# TEST_VARIANTS / DSV3) working.
TestVariant = PrefillModelAdapter
TEST_VARIANTS = {name: get_adapter(name) for name in ADAPTER_PATHS}
DSV3 = get_adapter("deepseek_v3_d_p")

# glm_5_2 is a TEST-ONLY variant here: its adapter is intentionally kept out of the shared common
# ADAPTER_PATHS (prefill serving is not wired), so register it locally for the `variant` fixture
# without modifying the common prefill registry.
from models.demos.deepseek_v3_d_p.tt.runners.adapters.sparse_mla import GLM52Adapter

TEST_VARIANTS["glm_5_2"] = GLM52Adapter()
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import create_fabric_router_config, get_max_payload_size
from models.demos.deepseek_v3_d_p.utils.test_utils import dequantize_state_dict, detect_language_model_prefix
from models.demos.deepseek_v3_d_p.utils.transformer_helpers import download_infinitebench_subset

# Shared FABRIC_2D parametrize entries for the prefill block + transformer tests.
# Minimum CI-gated coverage: (4,2) on BH LoudBox, (8,4) on BH Galaxy. (2,4) included
# for asymmetry coverage. RELAXED_INIT matches the canonical pattern in test_prefill_block.py
# and is required on BH Galaxy for FABRIC_2D bring-up.
FABRIC_2D_PREFILL_BLOCK_MESH_PARAMS = [
    pytest.param(
        (4, 2),
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_2D,
            "fabric_router_config": create_fabric_router_config(max_payload_size=get_max_payload_size()),
            "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
        },
        1,
        ttnn.Topology.Linear,
        marks=pytest.mark.requires_mesh_topology(mesh_shape=(4, 2), topology="mesh-4x2"),
        id="fabric2d-mesh-4x2",
    ),
    pytest.param(
        (2, 4),
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_2D,
            "fabric_router_config": create_fabric_router_config(max_payload_size=get_max_payload_size()),
            "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
        },
        1,
        ttnn.Topology.Linear,
        marks=pytest.mark.requires_mesh_topology(mesh_shape=(2, 4), topology="mesh-2x4"),
        id="fabric2d-mesh-2x4",
    ),
    pytest.param(
        (8, 4),
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_2D,
            "fabric_router_config": create_fabric_router_config(max_payload_size=get_max_payload_size()),
            "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
        },
        2,
        ttnn.Topology.Linear,
        marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
        id="fabric2d-mesh-8x4",
    ),
    # FABRIC_2D_TORUS_Y: single-galaxy 8x4 with the SP axis (mesh dim 0, 8-long) closed into a
    # ring; the TP axis (dim 1, 4-wide) stays a line. Per-axis topology (SP, TP) = (Ring, Linear):
    # Ring drives the SP-axis MoE dispatch/combine, Linear the TP-axis collectives (RMS-norm, MLA,
    # shared-expert, gate). A scalar Ring here would deadlock the TP-axis all-gathers on a column
    # wrap link that has no physical fabric edge. 2-link mirrors the fabric2d-mesh-8x4 sibling.
    pytest.param(
        (8, 4),
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_Y,
            "fabric_router_config": create_fabric_router_config(max_payload_size=get_max_payload_size()),
            "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
        },
        2,
        (ttnn.Topology.Ring, ttnn.Topology.Linear),
        marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
        # Id omits `fabric2d-`/`mesh-` so it stays out of the count-guarded `-k "8x4 and fabric2d"`
        # (EXPECT_NUM_TESTS=1) CI selectors that target the fabric2d-mesh siblings; select via `-k torus-y`.
        id="torus-y-8x4",
    ),
    # FABRIC_2D_TORUS_Y on a 4x4 sub-torus (16 of the galaxy's 32 chips). Same [RING, LINE]
    # shape as the 8x4 torus but with a Ring-4 on the SP axis (dim 0). Requires carving the
    # sub-torus at runtime via TT_VISIBLE_DEVICES (16 chips) + TT_MESH_GRAPH_DESC_PATH pointing at
    # models/demos/deepseek_v3_d_p/experimental_descriptors/single_bh_galaxy_subtorus_y4_graph_descriptor.textproto
    # (channels count: 2 -> 2 links).
    # Per-axis topology (SP, TP) = (Ring, Linear), as for the 8x4 torus.
    pytest.param(
        (4, 4),
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_Y,
            "fabric_router_config": create_fabric_router_config(max_payload_size=get_max_payload_size()),
            "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
        },
        2,
        (ttnn.Topology.Ring, ttnn.Topology.Linear),
        marks=pytest.mark.requires_mesh_topology(mesh_shape=(4, 4), topology="mesh-4x4"),
        id="torus-y-4x4",
    ),
]


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

    Galaxy ring/subtorus guard (CI): ring and torus fabrics need physical wrap cabling. On a
    GALAXY board that wrap is subtorus wiring that CI galaxy runners do not have — the native
    galaxy mesh is 32x4 across 4 hosts, so an 8-ring or 4-ring on a single galaxy is a sub-torus
    (and the 4-ring exists ONLY on a subtorus-wired galaxy, not on a LoudBox). Opening such a
    fabric on a CI galaxy just hangs. So on CI + galaxy we skip EVERY torus
    (FABRIC_2D_TORUS_{X,Y,XY}) and FABRIC_1D_RING config. LoudBox is untouched: its ring is
    native there, and galaxy-only torus configs can't be collected on it anyway (device count).
    None of this fires off CI, so a subtorus-wired host (CI unset) still runs everything.
    """
    on_ci = os.getenv("CI") == "true" or "TT_GH_CI_INFRA" in os.environ
    ring_or_torus_fabrics = {
        ttnn.FabricConfig.FABRIC_2D_TORUS_X,
        ttnn.FabricConfig.FABRIC_2D_TORUS_Y,
        ttnn.FabricConfig.FABRIC_2D_TORUS_XY,
        ttnn.FabricConfig.FABRIC_1D_RING,
    }

    def _is_ring_or_torus(item):
        params = getattr(getattr(item, "callspec", None), "params", {})
        dp = params.get("device_params")
        return isinstance(dp, dict) and dp.get("fabric_config") in ring_or_torus_fabrics

    # Only skip ring/torus on a galaxy; LoudBox rings are native. Galaxy detection via
    # get_cluster_type() OPENS the chip cluster as a side effect, so only call it when this session
    # actually collects a ring/torus config to decide on. Otherwise a device-free session — notably the
    # tracy-based perf tests, which spawn a child pytest — would open the device here in the PARENT and
    # hold CHIP_IN_USE, deadlocking the child (get_num_devices() below is likewise gated by a marker).
    # On detection failure default to skipping (a missed skip on a galaxy hangs, worse than over-skip on LB).
    skip_rings = False
    if on_ci and any(_is_ring_or_torus(item) for item in items):
        try:
            skip_rings = ttnn.cluster.get_cluster_type() in (
                ttnn.cluster.ClusterType.GALAXY,
                ttnn.cluster.ClusterType.TG,
                ttnn.cluster.ClusterType.BLACKHOLE_GALAXY,
            )
        except Exception:
            skip_rings = True

    for item in items:
        # Galaxy ring/subtorus guard — runs before the marker check so it catches configs whether or
        # not they carry a requires_mesh_topology mark.
        if skip_rings:
            if _is_ring_or_torus(item):
                item.add_marker(
                    pytest.mark.skip(
                        reason="ring/subtorus config on a CI galaxy runner: an 8-/4-ring on one galaxy "
                        "needs subtorus wrap cabling that CI galaxies lack -> would hang. Runs only on a "
                        "subtorus-wired galaxy (CI unset), or — for a native ring — a LoudBox."
                    )
                )
                continue

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
    param = getattr(request, "param", None)
    if param is None:
        return DSV3
    return TEST_VARIANTS[param] if isinstance(param, str) else param


def download_model_config_only(variant: TestVariant, cache_dir: Path) -> Path:
    """
    Download only config files (without weight shards) for the variant's HF repo.
    This is fast and only downloads ~few MB for config files.

    Args:
        variant: The TestVariant whose HF repo to download from.
        cache_dir: Directory to cache downloaded config.

    Returns:
        Path to the downloaded model directory with config.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        logger.error("huggingface_hub is not installed. Install it with: pip install huggingface_hub")
        raise

    logger.info(f"Downloading {variant.hf_repo_id} config only (no weights) from HuggingFace")
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
            "tiktoken*",  # Kimi K2.6 ships its BBPE tokenizer as tiktoken.model
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
            repo_id=variant.hf_repo_id,
            cache_dir=str(cache_dir),
            allow_patterns=allow_patterns,
            ignore_patterns=["*.safetensors"],  # Don't download weight files
        )

        # Variants that load config/tokenizer without trust_remote_code (e.g. DeepSeek-V3, stock fast
        # tokenizer) can use the HF snapshot dir directly — no flat copy needed. Skipping it also avoids
        # writing into a possibly read-only HF cache mount.
        if not variant.needs_flat_config_dir:
            logger.success(f"✓ Config files downloaded to: {model_dir}")
            return Path(model_dir)

        # The HF cache stores files as symlinks into blobs/ (content-hash names). With
        # trust_remote_code=True, transformers resolves the remote module to its blob realpath
        # and then looks for its relative-import siblings (e.g. tool_declaration_ts.py) by name in
        # that same dir, which fails in blobs/. Copy into a flat dir of real files so relative
        # imports resolve by name. The dir name is made dot-free (trust_remote_code can't import a
        # dynamic module whose dir contains '.'). Weight shards are excluded so we never materialize
        # the hundreds of GB the snapshot may hold from a prior weight download, and the flat dir lives
        # in a writable temp location so a read-only HF cache mount doesn't break the copy.
        flat_dir = (
            Path(tempfile.gettempdir())
            / "ttnn_flat_config"
            / variant.hf_repo_id.replace("/", "__").replace(".", "_").replace("-", "_").replace("_", "-")
        )

        shutil.copytree(
            model_dir,
            flat_dir,
            symlinks=False,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns("*.safetensors"),
        )

        logger.success(f"✓ Config files downloaded to: {model_dir} (flattened to: {flat_dir})")
        return flat_dir

    except Exception as e:
        logger.error(f"Failed to download {variant.hf_repo_id} config: {e}")
        raise


def download_model_weights(variant: TestVariant, cache_dir: Path, layer_idx: int = 0, num_layers: int = 1) -> Path:
    """
    Download model weights from HuggingFace for the variant's HF repo.

    Args:
        variant: The TestVariant whose HF repo to download from.
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

    logger.info(f"Downloading {variant.hf_repo_id} weights from HuggingFace")
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
            "tiktoken*",  # Kimi K2.6 ships its BBPE tokenizer as tiktoken.model
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


def _resolve_hf_snapshot_dir(path: Path) -> Path:
    """If `path` is an HF hub-cache repo root (``models--org--name/`` with a ``snapshots/`` subdir),
    return the active snapshot dir (the ``refs/main`` commit, else the newest snapshot that has the
    safetensors index) so callers see the real config.json + shards. Otherwise return `path` as-is.

    Lets ``*_HF_MODEL`` point at either the hub root (``.../hub/models--zai-org--GLM-5.1``) or a plain
    checkout dir. The hash snapshot dir also sidesteps the trust_remote_code dot-in-path import issue.
    """
    if (path / "model.safetensors.index.json").exists():
        return path
    snaps = path / "snapshots"
    if snaps.is_dir():
        ref = path / "refs" / "main"
        if ref.is_file():
            cand = snaps / ref.read_text().strip()
            if (cand / "model.safetensors.index.json").exists():
                return cand
        cands = [d for d in snaps.iterdir() if d.is_dir() and (d / "model.safetensors.index.json").exists()]
        if cands:
            return max(cands, key=lambda d: d.stat().st_mtime)
    return path


def get_or_download_model(variant: TestVariant, layer_idx: int = 0, num_layers: int = 6) -> Path:
    """
    Get model path, downloading from HuggingFace if necessary.

    Args:
        variant: The TestVariant to resolve weights for.
        layer_idx: Which layer weights to ensure are available.
        num_layers: Number of layers to download (default: 6).
                    When >1, downloads additional shards including shard 160 for model.norm.

    Returns:
        Path to model directory with weights.
    """
    # Check environment variable first
    env_path = os.getenv(variant.env_var)
    if env_path:
        model_path = Path(env_path)
        if model_path.exists():
            # Accept an HF hub-cache root (e.g. /mnt/MLPerf/huggingface/hub/models--zai-org--GLM-5.1)
            # by descending into its current snapshot, where config.json + the safetensors index live.
            model_path = _resolve_hf_snapshot_dir(model_path)
            index_file = model_path / "model.safetensors.index.json"
            if index_file.exists():
                logger.info(f"Using existing model from {variant.env_var}: {model_path}")
                # Keep the user path absolute but do NOT symlink-resolve it: resolve() would follow a
                # dot-free symlink (e.g. Kimi-K2_6) back to a dotted real dir (Kimi-K2.6), and HF
                # trust_remote_code cannot import a dynamic module whose name contains a '.'. The
                # safetensors load works through the symlink either way; only the config import cares.
                # This matches _resolve_config_only, which already loads config from the raw env path.
                return model_path.absolute()
            else:
                logger.warning(f"{variant.env_var} set but missing index file: {index_file}")

    # Check default location
    if variant.default_local_path is not None and variant.default_local_path.exists():
        index_file = variant.default_local_path / "model.safetensors.index.json"
        if index_file.exists():
            logger.info(f"Using model from default location: {variant.default_local_path}")
            return variant.default_local_path.resolve()

    # Check shared weights location
    if variant.shared_path is not None and variant.shared_path.exists():
        index_file = variant.shared_path / "model.safetensors.index.json"
        if index_file.exists():
            logger.info(f"Using model from shared location: {variant.shared_path}")
            return variant.shared_path.resolve()

    # Download from HuggingFace
    logger.info(f"Model not found locally. Downloading {variant.hf_repo_id} from HuggingFace...")

    # Determine cache directory
    cache_dir = Path(os.getenv("HF_HOME", Path.home() / ".cache" / "huggingface"))
    logger.info(f"Will cache to: {cache_dir}")
    # Note: Detailed download size is logged by download_model_weights() after analyzing the index

    return download_model_weights(variant, cache_dir, layer_idx, num_layers)


def _unwrap_multimodal_config(cfg):
    """Unwrap Kimi K2.5/K2.6's multimodal wrapper config to the inner text_config.

    The LM fields the rest of the code reads (hidden_size, n_routed_experts, etc.) live
    under `text_config`.
    """
    if hasattr(cfg, "text_config") and hasattr(cfg.text_config, "hidden_size"):
        logger.info(f"Unwrapping multimodal wrapper config (inner model_type={cfg.text_config.model_type})")
        cfg = cfg.text_config
    return cfg


# --- Cached resolvers ---
# Session-scoped fixtures don't compose with the function-scoped `variant` fixture, so the
# expensive resolution work is cached at the function level keyed on variant.name instead.


@lru_cache(maxsize=None)
def _resolve_model_path(variant_name: str) -> Path:
    v = TEST_VARIANTS[variant_name]
    return get_or_download_model(v, layer_idx=0, num_layers=v.num_layers_to_download)


@lru_cache(maxsize=None)
def _resolve_hf_config(model_path_str: str):
    p = Path(model_path_str)
    if not (p / "config.json").exists():
        return None
    try:
        cfg = AutoConfig.from_pretrained(str(p), trust_remote_code=True)
        logger.info(f"Loaded HF config from {p}")
        return _unwrap_multimodal_config(cfg)
    except Exception as e:
        logger.warning(f"Failed to load HF config from {p}: {e}")
        return None


@lru_cache(maxsize=None)
def _resolve_config_only(variant_name: str):
    v = TEST_VARIANTS[variant_name]
    # Hand-built config takes precedence: some models (e.g. GLM-5.1 `glm_moe_dsa`, DeepSeek-V3.2
    # `deepseek_v32`) are not registered with transformers, so AutoConfig cannot load them. The builder
    # returns a ready HF-attribute config. (Result is lru_cached like the AutoConfig path; tests that
    # mutate config.max_seq_len already rely on this shared/cached object.)
    if v.config_builder is not None:
        return v.config_builder()
    # Check environment variable first
    env_path = os.getenv(v.env_var)
    if env_path:
        model_path = Path(env_path)
        if (model_path / "config.json").exists():
            logger.info(f"Using existing config from {v.env_var}: {model_path}")
            return _unwrap_multimodal_config(AutoConfig.from_pretrained(str(model_path), trust_remote_code=True))

    # Check default location
    if v.default_local_path is not None and (v.default_local_path / "config.json").exists():
        logger.info(f"Using config from default location: {v.default_local_path}")
        return _unwrap_multimodal_config(AutoConfig.from_pretrained(str(v.default_local_path), trust_remote_code=True))

    # Check shared weights location
    if v.shared_path is not None and (v.shared_path / "config.json").exists():
        logger.info(f"Using config from shared location: {v.shared_path}")
        return _unwrap_multimodal_config(AutoConfig.from_pretrained(str(v.shared_path), trust_remote_code=True))

    # Download only config files from HuggingFace (not weight shards)
    logger.info(f"Config not found locally. Downloading {v.hf_repo_id} config only from HuggingFace...")
    cache_dir = Path(os.getenv("HF_HOME", Path.home() / ".cache" / "huggingface"))
    config_path = download_model_config_only(v, cache_dir)
    return _unwrap_multimodal_config(AutoConfig.from_pretrained(str(config_path), trust_remote_code=True))


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
    # Only variants that ship custom tokenizer code (e.g. Kimi) need trust_remote_code; DeepSeek-V3
    # uses a stock fast tokenizer and turns it off to avoid the flat-config custom-import path.
    trust_remote_code = v.tokenizer_trust_remote_code
    candidates = [
        os.getenv(v.env_var),
        str(v.default_local_path) if v.default_local_path is not None else None,
        str(v.shared_path) if v.shared_path is not None else None,
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        p = Path(candidate)
        if p.exists() and any(p.glob("tokenizer*")):
            logger.info(f"Loading tokenizer from: {p}")
            tok = AutoTokenizer.from_pretrained(str(p), use_fast=True, trust_remote_code=trust_remote_code)
            tok.padding_side = padding_side
            return tok

    # Fall back to downloading config-only (includes tokenizer files)
    cache_dir = Path(os.getenv("HF_HOME", Path.home() / ".cache" / "huggingface"))
    config_path = download_model_config_only(v, cache_dir)
    logger.info(f"Loading tokenizer from downloaded config: {config_path}")
    tok = AutoTokenizer.from_pretrained(str(config_path), use_fast=True, trust_remote_code=trust_remote_code)
    tok.padding_side = padding_side
    return tok


@pytest.fixture
def model_path(variant) -> Path:
    """
    Get model path and resolve symlinks to ensure all operations can find files.
    Automatically downloads weights from HuggingFace if not available locally.
    Downloads weights for layers 0-23 (24 layers total) by default to support test cases.

    Checks in order:
    1. variant.env_var environment variable
    2. variant.default_local_path (default location)
    3. variant.shared_path
    4. Downloads from HuggingFace to HF cache if not found
    """
    return _resolve_model_path(variant.name)


@pytest.fixture
def hf_config(model_path):
    """
    Load HF config for testing.
    Returns None if model path doesn't exist (weights not available).
    """
    return _resolve_hf_config(str(model_path))


@pytest.fixture
def config_only(variant):
    """
    Load HF config for random weight tests (downloads only config, not weights).
    This is fast and only downloads ~few MB.
    """
    return _resolve_config_only(variant.name)


@pytest.fixture(params=["right"])
def tokenizer(request, variant):
    """Load the variant's tokenizer, searching known model locations.

    Default padding_side is "right" (back-padding). To test with left padding,
    override in your test: @pytest.mark.parametrize("tokenizer", ["left"], indirect=True)
    """
    return _resolve_tokenizer(variant.name, request.param)


@pytest.fixture
def state_dict(model_path):
    """
    Load state dict for testing.
    Returns None if model path doesn't exist (weights not available).
    """
    return _resolve_state_dict(str(model_path))


def _check_pretrained_available(model_path: Path) -> bool:
    """
    Check if pretrained weights are available at the given path.

    Returns:
        True if pretrained weights are available, False otherwise.
    """
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
    """
    Return a directory for caching TTNN weight tensors (.tensorbin files).

    First run: ttnn.as_tensor() dumps converted weights here.
    Subsequent runs: weights are loaded directly, bypassing torch conversion.

    The path encodes variant + architecture + device count to prevent cross-config clashes.
    Returns None if pretrained weights are unavailable (random-weight tests skip caching).
    """
    if not _check_pretrained_available(model_path):
        return None
    arch = "bh" if is_blackhole() else "wh"
    num_devices = ttnn.get_num_devices()
    env_name = variant.ttnn_cache_env or "TT_DS_PREFILL_TTNN_CACHE"
    env_cache = os.getenv(env_name)
    if env_cache:
        cache_dir = Path(env_cache) / f"{variant.name}_{arch}_{num_devices}dev"
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
def pretrained_transformer_weights(variant, model_path, hf_config, state_dict, request):
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
    if not variant.supports_pretrained:
        pytest.skip(f"{variant.name}: pretrained weights not wired")
    if not _check_pretrained_available(model_path):
        pytest.skip(f"{variant.name}: pretrained weights not available. Set {variant.env_var} or download model.")
    if hf_config is None:
        pytest.skip(f"{variant.name}: failed to load HF config. Check model path.")
    if state_dict is None:
        pytest.skip(f"{variant.name}: failed to load state dict. Check model path and weights.")

    num_layers = request.node.callspec.params.get("num_layers", 1)
    first_k_dense = hf_config.first_k_dense_replace
    n_routed = hf_config.n_routed_experts

    # Kimi's raw multimodal checkpoint nests the LM under a `language_model.` prefix; the
    # dequantized/stripped checkpoint and DeepSeek use bare `model.` keys. Detect it from the
    # actual keys so the same variant works for either, then `sub_state_dict` strips it.
    prefix = detect_language_model_prefix(state_dict)

    logger.info(f"Loading pretrained transformer weights for {num_layers} layers from: {model_path}")

    # Embed tokens
    embed_sd = sub_state_dict(state_dict, f"{prefix}model.embed_tokens.")
    embed_dequant = dequantize_state_dict(embed_sd, hf_config)
    result = {
        "embed_weight": embed_dequant["weight"].float(),
    }

    # Final norm
    norm_sd = sub_state_dict(state_dict, f"{prefix}model.norm.")
    norm_dequant = dequantize_state_dict(norm_sd, hf_config)
    result["norm_weight"] = norm_dequant["weight"]

    # Per-layer weights
    result["layers"] = []
    for i in range(num_layers):
        logger.info(f"Loading layer {i} weights...")
        layer_sd = sub_state_dict(state_dict, f"{prefix}model.layers.{i}.")
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


def pytest_collection_finish(session):
    """Optional CI guardrail: warn (do NOT fail) when the number of selected
    deepseek_v3_d_p tests differs from EXPECT_NUM_TESTS.

    Inert unless EXPECT_NUM_TESTS is set, so it has zero effect on normal runs.
    Intended for pipeline commands whose ``-k`` filter must resolve to a known
    count — e.g. topology-gated tests that can silently collect 0 on the wrong
    mesh. Emits a GitHub Actions ``::warning::`` annotation but never changes the
    exit code, so the job still passes."""
    expected_raw = os.getenv("EXPECT_NUM_TESTS")
    if not expected_raw:
        return
    try:
        expected = int(expected_raw)
    except ValueError:
        print(f"::warning title=Test count check::EXPECT_NUM_TESTS={expected_raw!r} is not an integer; skipping check")
        return
    actual = len(session.items)
    if actual == expected:
        return
    invocation = " ".join(session.config.invocation_params.args)
    msg = f"expected {expected} test(s) to be collected but got {actual} (pytest {invocation})"
    annotation = f"::warning title=Unexpected test count::{msg}"

    # The annotation must reach the step's live log stream for GitHub to parse it,
    # so emit it with pytest's output capture suspended (a plain print() here can be
    # swallowed by capturing and never appear in the runner log).
    capman = session.config.pluginmanager.get_plugin("capturemanager")
    if capman is not None:
        with capman.global_and_fixture_disabled():
            print(annotation, flush=True)
    else:
        print(annotation, flush=True)

    # Also surface it on the GitHub job-summary page when available.
    summary_path = os.getenv("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a") as fh:
            fh.write(f"⚠️ **Unexpected test count** — {msg}\n")
