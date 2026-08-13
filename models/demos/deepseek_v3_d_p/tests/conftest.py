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

from models.demos.deepseek_v3_d_p.tests.fabric_profiles import (
    assert_torus_xy_descriptor,
    fabric2d_device_params,
    torus_x_device_params,
    torus_xy_device_params,
    torus_y_device_params,
)

# glm_5_2 is a TEST-ONLY variant here: its adapter is intentionally kept out of the shared common
# ADAPTER_PATHS (prefill serving is not wired), so register it locally for the `variant` fixture
# without modifying the common prefill registry.
from models.demos.deepseek_v3_d_p.tt.runners.adapters.glm_5_2 import GLM52Adapter

TEST_VARIANTS["glm_5_2"] = GLM52Adapter()

# kimi_k3 is TEST-ONLY for the same reason, more strongly: 69 of its 93 layers are KDA
# linear-attention layers with no TT implementation, so only its MLA layer is testable.
from models.demos.deepseek_v3_d_p.tt.runners.adapters.kimi_k3 import KimiK3Adapter

TEST_VARIANTS["kimi_k3"] = KimiK3Adapter()
from models.demos.deepseek_v3_d_p.utils.test_utils import convert_state_dict, detect_language_model_prefix
from models.demos.deepseek_v3_d_p.utils.transformer_helpers import download_infinitebench_subset

# Shared production-policy params for prefill block + transformer tests. LoudBox executes canonical
# 2x4 Fabric2D and one 4x2 axis-order diagnostic; Galaxy production executes only 8x4 TorusXY.
FABRIC_2D_PREFILL_BLOCK_MESH_PARAMS = [
    pytest.param(
        (4, 2),
        fabric2d_device_params(),
        1,
        marks=pytest.mark.requires_mesh_topology(mesh_shape=(4, 2), topology="mesh-4x2"),
        id="fabric2d-mesh-4x2",
    ),
    pytest.param(
        (2, 4),
        fabric2d_device_params(),
        1,
        marks=pytest.mark.requires_mesh_topology(mesh_shape=(2, 4), topology="mesh-2x4"),
        id="fabric2d-mesh-2x4",
    ),
    # FABRIC_2D_TORUS_XY on the full 8x4 galaxy: Ring on BOTH axes (SP dim 0 = Ring-8, TP dim 1 =
    # Ring-4). SP-axis MoE dispatch/combine ride #48225's ring-aware kernels; TP-axis collectives ring.
    pytest.param(
        (8, 4),
        torus_xy_device_params(),
        2,
        marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
        id="torus-xy-8x4",
    ),
    # Existing 16-chip subtorus diagnostics. These run only with an explicit 4x4 carve descriptor;
    # they are a distinct workload and are not substitutes for production 8x4 TorusXY coverage.
    pytest.param(
        (4, 4),
        torus_y_device_params(),
        2,
        marks=pytest.mark.requires_mesh_topology(mesh_shape=(4, 4), topology="mesh-4x4"),
        id="torus-y-4x4",
    ),
    pytest.param(
        (4, 4),
        torus_x_device_params(),
        2,
        marks=pytest.mark.requires_mesh_topology(mesh_shape=(4, 4), topology="mesh-4x4"),
        id="torus-x-4x4",
    ),
    pytest.param(
        (4, 4),
        torus_xy_device_params(),
        2,
        marks=pytest.mark.requires_mesh_topology(mesh_shape=(4, 4), topology="mesh-4x4"),
        id="torus-xy-4x4",
    ),
]


def pytest_addoption(parser):
    try:
        parser.addoption(
            "--wrapper-invocation",
            action="store_true",
            default=False,
            help="Set by wrapper tests on the child pytest they spawn: every uncollect_if trim "
            "is bypassed, so the wrapper's -k filter fully owns the selection.",
        )
    except ValueError:
        pass


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "requires_mesh_topology(mesh_shape, topology): mark test to run only on compatible "
        "device/topology combinations. mesh_shape is (rows, cols) tuple, topology is 'ring' or 'linear'. "
        "Skips automatically based on available devices and arch constraints.",
    )
    config.addinivalue_line(
        "markers",
        "uncollect_if(pred): deselect parametrized cases for which pred(**params) returns True. "
        "pred receives the test's collection-time param values as keyword args, plus is_ci_env / is_ci_v2_env.",
    )


def _test_defined_uncollection(items, is_ci_env, is_ci_v2_env):
    kept = []
    is_bh = is_blackhole()
    for item in items:
        marker = item.get_closest_marker("uncollect_if")
        if marker is None:
            kept.append(item)
            continue
        params = dict(getattr(getattr(item, "callspec", None), "params", {}))
        # Values the predicate wants that come from fixtures, not parametrization.
        params.setdefault("is_ci_env", is_ci_env)
        params.setdefault("is_ci_v2_env", is_ci_v2_env)
        params.setdefault("is_bh", is_bh)
        if not marker.kwargs["pred"](**params):
            kept.append(item)

    return kept


def pytest_collection_modifyitems(config, items):
    """
    Skip tests based on mesh/topology requirements at collection time.

    Hardware constraints:
    - Blackhole: multi-device test shapes must consume the complete local box
    - Wormhole: wrapped topology only works with 8 devices

    Galaxy TorusXY guard (CI): the production ring/ring fabric requires an explicit descriptor and
    a cabling-certified allocation. Generic Galaxy jobs skip it before device open. Certified jobs
    set PREFILL_TORUS_XY_CERTIFIED=1 and TT_MESH_GRAPH_DESC_PATH. Native Nx1 ring proxies use
    Fabric2D TorusY and 1xN ring proxies use TorusX; non-ring local shapes remain unwrapped Fabric2D.
    """
    is_ci_env = os.getenv("CI") == "true"
    is_ci_v2_env = "TT_GH_CI_INFRA" in os.environ
    on_ci = is_ci_env or is_ci_v2_env

    # A wrapper's child pytest owns its own selection through -k; trimming it here would
    # hide exactly the cases the wrapper exists to measure.
    if not config.getoption("--wrapper-invocation"):
        items[:] = _test_defined_uncollection(items, is_ci_env, is_ci_v2_env)

    torus_xy_certified = os.getenv("PREFILL_TORUS_XY_CERTIFIED") == "1"
    torus_xy_fabric = ttnn.FabricConfig.FABRIC_2D_TORUS_XY
    ring_or_torus_fabrics = {
        ttnn.FabricConfig.FABRIC_2D_TORUS_X,
        ttnn.FabricConfig.FABRIC_2D_TORUS_Y,
        ttnn.FabricConfig.FABRIC_2D_TORUS_XY,
    }

    CT = ttnn.cluster.ClusterType
    FC = ttnn.FabricConfig
    DEFAULT_ALLOWED_FABRICS = frozenset({FC.DISABLED, FC.FABRIC_1D, FC.FABRIC_2D})
    # DISABLED is listed only on shapes that already own fabric-irrelevant diagnostics
    # (currently single-chip and the P300 1x2 masked-bincount row). Do not expand it into
    # communicating-test matrices merely to make this table visually symmetric.
    CI_ALLOWED_FABRICS = {
        CT.P150: {(1, 1): [FC.DISABLED, FC.FABRIC_2D]},  # single chip
        CT.P300: {
            (2, 1): [FC.FABRIC_2D],
            (1, 2): [FC.DISABLED, FC.FABRIC_2D],
        },  # 2 chips
        CT.P300_X2: {  # 4-chip QuietBox
            (4, 1): [FC.FABRIC_1D, FC.FABRIC_2D_TORUS_Y],
            (2, 2): [FC.FABRIC_2D],
            (1, 4): [FC.FABRIC_2D_TORUS_X],
        },
        CT.P150_X8: {
            (8, 1): [FC.FABRIC_1D, FC.FABRIC_2D_TORUS_Y],
            (4, 2): [FC.FABRIC_2D],
            (2, 4): [FC.FABRIC_2D],
            (1, 8): [FC.FABRIC_2D_TORUS_X],
        },
        CT.T3K: {
            (8, 1): [FC.FABRIC_1D, FC.FABRIC_2D_TORUS_Y],
            (4, 2): [FC.FABRIC_2D],
            (2, 4): [FC.FABRIC_2D],
            (1, 8): [FC.FABRIC_2D_TORUS_X],
        },
        CT.BLACKHOLE_GALAXY: {
            (32, 1): [FC.FABRIC_2D],
            (16, 2): [FC.FABRIC_2D],
            (8, 4): [FC.FABRIC_1D, FC.FABRIC_2D, FC.FABRIC_2D_TORUS_XY],
            (4, 4): [FC.FABRIC_2D_TORUS_X, FC.FABRIC_2D_TORUS_Y, FC.FABRIC_2D_TORUS_XY],
            (4, 8): [FC.FABRIC_1D, FC.FABRIC_2D],
            (2, 16): [FC.FABRIC_2D],
            (1, 32): [FC.FABRIC_2D],
        },
        CT.GALAXY: {
            (32, 1): [FC.FABRIC_2D],
            (16, 2): [FC.FABRIC_2D],
            (8, 4): [FC.FABRIC_1D, FC.FABRIC_2D, FC.FABRIC_2D_TORUS_XY],
            (4, 4): [FC.FABRIC_2D_TORUS_X, FC.FABRIC_2D_TORUS_Y, FC.FABRIC_2D_TORUS_XY],
            (4, 8): [FC.FABRIC_1D, FC.FABRIC_2D],
            (2, 16): [FC.FABRIC_2D],
            (1, 32): [FC.FABRIC_2D],
        },
        CT.TG: {
            (32, 1): [FC.FABRIC_2D],
            (16, 2): [FC.FABRIC_2D],
            (8, 4): [FC.FABRIC_1D, FC.FABRIC_2D, FC.FABRIC_2D_TORUS_XY],
            (4, 4): [FC.FABRIC_2D_TORUS_X, FC.FABRIC_2D_TORUS_Y, FC.FABRIC_2D_TORUS_XY],
            (4, 8): [FC.FABRIC_1D, FC.FABRIC_2D],
            (2, 16): [FC.FABRIC_2D],
            (1, 32): [FC.FABRIC_2D],
        },
    }

    def _get_requested_fabric_cfg(item):  # returns fabric cfg that a particular test case requested for the test
        params = getattr(getattr(item, "callspec", None), "params", {})
        dp = params.get("device_params")
        if isinstance(dp, dict):
            return dp.get("fabric_config")
        else:
            return None

    def _is_torus_xy(item):
        return _get_requested_fabric_cfg(item) == torus_xy_fabric

    torus_xy_items_collected = any(_is_torus_xy(item) for item in items)
    if torus_xy_items_collected and torus_xy_certified and not os.getenv("TT_MESH_GRAPH_DESC_PATH"):
        pytest.exit("PREFILL_TORUS_XY_CERTIFIED=1 requires explicit TT_MESH_GRAPH_DESC_PATH", returncode=2)
    if torus_xy_items_collected and torus_xy_certified:
        assert_torus_xy_descriptor(os.environ["TT_MESH_GRAPH_DESC_PATH"])

    # Generic CI galaxies are not wrap-cabling certified. get_cluster_type() opens the chip cluster as a
    # side effect, so call it only when this session collects a wrapped device configuration. This keeps
    # device-free tracy perf wrappers from opening devices in the parent; their parametrized child session
    # performs the detection instead. On detection failure default to skipping (a missed skip can hang).
    skip_rings = False
    cluster_type = None
    if any((_get_requested_fabric_cfg(item) in ring_or_torus_fabrics) for item in items):
        try:
            cluster_type = ttnn.cluster.get_cluster_type()
            skip_rings = on_ci and cluster_type in [CT.GALAXY, CT.BLACKHOLE_GALAXY, CT.TG]
        except Exception:
            skip_rings = True

    for item in items:
        # Galaxy TorusXY guard — runs before the marker check so it catches configs whether or
        # not they carry a requires_mesh_topology mark.
        requested_fabric_cfg = _get_requested_fabric_cfg(item)
        if skip_rings:
            certified_production_torus = requested_fabric_cfg == torus_xy_fabric and torus_xy_certified
            if requested_fabric_cfg in ring_or_torus_fabrics and not certified_production_torus:
                item.add_marker(
                    pytest.mark.skip(
                        reason="Wrapped fabric requires a compatible physical ring; Galaxy TorusXY additionally "
                        "requires an explicit ring/ring descriptor and a cabling-certified allocation"
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

        if on_ci:
            # Unsupported fabric rings on QB/LB meshes
            allowed_fabric_cfgs = DEFAULT_ALLOWED_FABRICS
            if cluster_type in CI_ALLOWED_FABRICS.keys():
                allowed_fabric_dct = CI_ALLOWED_FABRICS[cluster_type]
                if mesh_shape in allowed_fabric_dct.keys():
                    allowed_fabric_cfgs = allowed_fabric_dct[mesh_shape]

            # A case with no device_params fabric never opens a fabric, so it cannot request an
            # unfeasible mesh/fabric combination — only device-count matching below applies to it.
            if requested_fabric_cfg is not None and requested_fabric_cfg not in allowed_fabric_cfgs:
                item.add_marker(
                    pytest.mark.skip(
                        reason="requested combination of fabric config and mesh, unfeasible on the given hardware"
                    )
                )
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


@pytest.fixture(autouse=True)
def _assert_certified_torus_profile(request):
    """Fail closed after mesh open for every certified TorusXY parametrized case."""
    params = getattr(getattr(request.node, "callspec", None), "params", {})
    device_params = params.get("device_params")
    if not isinstance(device_params, dict):
        return
    if device_params.get("fabric_config") != ttnn.FabricConfig.FABRIC_2D_TORUS_XY:
        return
    if os.getenv("PREFILL_TORUS_XY_CERTIFIED") != "1":
        return
    request.getfixturevalue("mesh_device")
    assert ttnn.get_fabric_config() == ttnn.FabricConfig.FABRIC_2D_TORUS_XY
    from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology

    assert per_axis_topology() == (ttnn.Topology.Ring, ttnn.Topology.Ring)


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

    # Kimi-K3 output gate. Appended AFTER the block above so the manual_seed(42) draw order for every
    # non-gated variant is unchanged (the cached reference results depend on it — see above).
    if getattr(config, "mla_use_output_gate", False):
        weights["g_proj.weight"] = (
            torch.randn(
                config.num_attention_heads * config.v_head_dim,
                config.hidden_size,
            )
            * std
        ).to(torch.bfloat16)

    logger.info(f"Generated {len(weights)} random weight tensors using config dimensions")
    return config, weights


def _load_mla_weights(state_dict, hf_config, prefix: str, layer_idx: int) -> dict:
    """One layer's MLA weights, read without the rest of the layer: a whole-layer view drags in the
    MoE experts, which Kimi-K3 packs as MXFP4 and convert_state_dict raises on."""
    sd = convert_state_dict(sub_state_dict(state_dict, f"{prefix}model.layers.{layer_idx}.self_attn."), hf_config)
    names = [
        "q_a_proj.weight",
        "q_a_layernorm.weight",
        "q_b_proj.weight",
        "kv_a_proj_with_mqa.weight",
        "kv_a_layernorm.weight",
        "kv_b_proj.weight",
        "o_proj.weight",
    ]
    if getattr(hf_config, "mla_use_output_gate", False):
        names.append("g_proj.weight")  # Kimi-K3; ttMLA reads it with no default
    return {name: sd[name] for name in names}


@pytest.fixture
def pretrained_transformer_weights(variant, model_path, hf_config, state_dict, request):
    """
    Dequantized pretrained weights for N-layer transformer in TT state_dict format.

    Extracts embed, norm, and per-layer weights (attention, FFN/MoE) using
    sub_state_dict() + convert_state_dict(), matching the format produced
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
    embed_dequant = convert_state_dict(embed_sd, hf_config)
    result = {
        "embed_weight": embed_dequant["weight"].float(),
    }

    # Final norm
    norm_sd = sub_state_dict(state_dict, f"{prefix}model.norm.")
    norm_dequant = convert_state_dict(norm_sd, hf_config)
    result["norm_weight"] = norm_dequant["weight"]

    # Per-layer weights
    result["layers"] = []
    for i in range(num_layers):
        logger.info(f"Loading layer {i} weights...")
        layer_sd = sub_state_dict(state_dict, f"{prefix}model.layers.{i}.")
        layer_dequant = convert_state_dict(layer_sd, hf_config)

        layer_dict = {
            "attn_norm_weight": layer_dequant["input_layernorm.weight"],
            "mla_weights": _load_mla_weights(state_dict, hf_config, prefix, i),
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


@pytest.fixture
def pretrained_mla_layer_weights(variant, model_path, hf_config, state_dict):
    """Pretrained MLA weights from ``variant.pretrained_mla_layer``, as ``(hf_config, weights)``.

    Same shape ``random_weights`` returns, so an MLA test swaps one fixture for the other. Separate
    from ``pretrained_transformer_weights`` because that one also loads the embedding, the norms and
    the full MoE side, which Kimi-K3 cannot do.
    """
    if variant.pretrained_mla_layer is None:
        pytest.skip(f"{variant.name}: no reachable checkpoint, so no MLA weights to load")
    if not _check_pretrained_available(model_path):
        pytest.skip(f"{variant.name}: pretrained weights not available. Set {variant.env_var} or download model.")
    if hf_config is None:
        pytest.skip(f"{variant.name}: failed to load HF config. Check model path.")
    if state_dict is None:
        pytest.skip(f"{variant.name}: failed to load state dict. Check model path and weights.")

    # The torch MLA reference reads all three with no defaults. Kimi-K3's checkpoint config omits the
    # first two, and for the third transformers synthesizes {'rope_type': 'default'}, on which
    # _init_rope KeyErrors -- a NoPE model has no scaling, so None is the value it wants.
    for field, default in (("attention_bias", False), ("attention_dropout", 0.0)):
        if not hasattr(hf_config, field):
            setattr(hf_config, field, default)
    if getattr(hf_config, "mla_use_nope", False):
        hf_config.rope_scaling = None

    layer_idx = variant.pretrained_mla_layer
    prefix = detect_language_model_prefix(state_dict)
    logger.info(f"Loading pretrained MLA weights from layer {layer_idx} of {model_path}")
    weights = _load_mla_weights(state_dict, hf_config, prefix, layer_idx)
    logger.info(f"Loaded {len(weights)} MLA weight tensors (layer {layer_idx})")
    return hf_config, weights


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
