# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path

import pytest
from loguru import logger

from models.common.utility_functions import is_blackhole, is_wormhole_b0
from models.demos.stable_diffusion_xl_base.lora.config import TEST_LORA_FILENAME, TEST_LORA_REPO_ID
from models.demos.stable_diffusion_xl_base.tests.test_common import SDXL_L1_SMALL_SIZE, SDXL_L1_SMALL_SIZE_BH


@pytest.fixture(autouse=True)
def set_mm_throttle(request):
    if is_blackhole() or "clip_encoder" in request.node.nodeid or "test_sdxl_op_unit_test_perf" in request.node.nodeid:
        os.environ["TT_MM_THROTTLE_PERF"] = "0"
    else:
        os.environ["TT_MM_THROTTLE_PERF"] = "5"


# =============================================================================
# SDXL Model Location Fixtures (Session-scoped for CIv2 download efficiency)
# =============================================================================
# These fixtures download models once per pytest session and cache the location.
# This prevents redundant downloads when running multiple SDXL tests.
# =============================================================================

# --- Base Model Locations ---


@pytest.fixture(scope="session")
def sdxl_base_unet_location(model_location_generator, is_ci_v2_env):
    """
    Returns the location for SDXL base UNet model weights.
    In CIv2: Downloads once per session from large file cache.
    In CIv1/local: Returns HF repo ID (resolved via HF_HUB_CACHE to local cache or MLPerf mount).
    """
    if is_ci_v2_env:
        return model_location_generator(
            "stable-diffusion-xl-base-1.0/unet",
            download_if_ci_v2=True,
            ci_v2_timeout_in_s=1800,
        )
    else:
        return "stabilityai/stable-diffusion-xl-base-1.0"


@pytest.fixture(scope="session")
def sdxl_base_pipeline_location(model_location_generator, is_ci_v2_env):
    """
    Returns the location for full SDXL base pipeline.
    In CIv2: Downloads once per session from large file cache.
    In CIv1/local: Returns HF repo ID (resolved via HF_HUB_CACHE to local cache or MLPerf mount).
    """
    if is_ci_v2_env:
        return model_location_generator(
            "stable-diffusion-xl-base-1.0",
            download_if_ci_v2=True,
            ci_v2_timeout_in_s=1800,
        )
    else:
        return "stabilityai/stable-diffusion-xl-base-1.0"


@pytest.fixture(scope="session")
def sdxl_base_vae_location(model_location_generator, is_ci_v2_env):
    """
    Returns the location for SDXL base VAE model weights.
    In CIv2: Downloads once per session from large file cache.
    In CIv1/local: Returns HF repo ID (resolved via HF_HUB_CACHE to local cache or MLPerf mount).
    """
    if is_ci_v2_env:
        return model_location_generator(
            "stable-diffusion-xl-base-1.0/vae",
            download_if_ci_v2=True,
            ci_v2_timeout_in_s=1800,
        )
    else:
        return "stabilityai/stable-diffusion-xl-base-1.0"


@pytest.fixture(scope="session")
def sdxl_base_text_encoder_location(model_location_generator, is_ci_v2_env):
    """
    Returns the location for SDXL base text_encoder (CLIP) model weights.
    In CIv2: Downloads once per session from large file cache.
    In CIv1/local: Returns HF repo ID (resolved via HF_HUB_CACHE to local cache or MLPerf mount).
    """
    if is_ci_v2_env:
        return model_location_generator(
            "stable-diffusion-xl-base-1.0/text_encoder",
            download_if_ci_v2=True,
            ci_v2_timeout_in_s=1800,
        )
    else:
        return "stabilityai/stable-diffusion-xl-base-1.0"


@pytest.fixture(scope="session")
def sdxl_base_text_encoder_2_location(model_location_generator, is_ci_v2_env):
    """
    Returns the location for SDXL base text_encoder_2 (CLIP with projection) model weights.
    In CIv2: Downloads once per session from large file cache.
    In CIv1/local: Returns HF repo ID (resolved via HF_HUB_CACHE to local cache or MLPerf mount).
    """
    if is_ci_v2_env:
        return model_location_generator(
            "stable-diffusion-xl-base-1.0/text_encoder_2",
            download_if_ci_v2=True,
            ci_v2_timeout_in_s=1800,
        )
    else:
        return "stabilityai/stable-diffusion-xl-base-1.0"


@pytest.fixture(scope="session")
def sdxl_base_tokenizer_location(model_location_generator, is_ci_v2_env):
    """
    Returns the location for SDXL base tokenizer.
    In CIv2: Downloads once per session from large file cache.
    In CIv1/local: Returns HF repo ID (resolved via HF_HUB_CACHE to local cache or MLPerf mount).
    """
    if is_ci_v2_env:
        return model_location_generator(
            "stable-diffusion-xl-base-1.0/tokenizer",
            download_if_ci_v2=True,
            ci_v2_timeout_in_s=1800,
        )
    else:
        return "stabilityai/stable-diffusion-xl-base-1.0"


@pytest.fixture(scope="session")
def sdxl_base_tokenizer_2_location(model_location_generator, is_ci_v2_env):
    """
    Returns the location for SDXL base tokenizer_2.
    In CIv2: Downloads once per session from large file cache.
    In CIv1/local: Returns HF repo ID (resolved via HF_HUB_CACHE to local cache or MLPerf mount).
    """
    if is_ci_v2_env:
        return model_location_generator(
            "stable-diffusion-xl-base-1.0/tokenizer_2",
            download_if_ci_v2=True,
            ci_v2_timeout_in_s=1800,
        )
    else:
        return "stabilityai/stable-diffusion-xl-base-1.0"


# --- Inpainting Model Locations ---


@pytest.fixture(scope="session")
def sdxl_inpainting_unet_location(model_location_generator, is_ci_v2_env):
    """
    Returns the location for SDXL inpainting UNet model weights.
    In CIv2: Downloads once per session from large file cache.
    In CIv1/local: Returns HF repo ID.
    """
    if is_ci_v2_env:
        return model_location_generator(
            "stable-diffusion-xl-1.0-inpainting-0.1/unet",
            download_if_ci_v2=True,
            ci_v2_timeout_in_s=1800,
        )
    else:
        return "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"


@pytest.fixture(scope="session")
def sdxl_inpainting_pipeline_location(model_location_generator, is_ci_v2_env):
    """
    Returns the location for full SDXL inpainting pipeline.
    In CIv2: Downloads once per session from large file cache.
    In CIv1/local: Returns HF repo ID.
    """
    if is_ci_v2_env:
        return model_location_generator(
            "stable-diffusion-xl-1.0-inpainting-0.1",
            download_if_ci_v2=True,
            ci_v2_timeout_in_s=1800,
        )
    else:
        return "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"


# --- Refiner Model Locations ---


@pytest.fixture(scope="session")
def sdxl_refiner_unet_location(model_location_generator, is_ci_v2_env):
    """
    Returns the location for SDXL refiner UNet model weights.
    In CIv2: Downloads once per session from large file cache.
    In CIv1/local: Returns HF repo ID (resolved via HF_HUB_CACHE to local cache or MLPerf mount).
    """
    if is_ci_v2_env:
        return model_location_generator(
            "stable-diffusion-xl-refiner-1.0/unet",
            download_if_ci_v2=True,
            ci_v2_timeout_in_s=1800,
        )
    else:
        return "stabilityai/stable-diffusion-xl-refiner-1.0"


@pytest.fixture(scope="session")
def sdxl_refiner_pipeline_location(model_location_generator, is_ci_v2_env):
    """
    Returns the location for full SDXL refiner pipeline.
    In CIv2: Downloads once per session from large file cache.
    In CIv1/local: Returns HF repo ID (resolved via HF_HUB_CACHE to local cache or MLPerf mount).
    """
    if is_ci_v2_env:
        return model_location_generator(
            "stable-diffusion-xl-refiner-1.0",
            download_if_ci_v2=True,
            ci_v2_timeout_in_s=1800,
        )
    else:
        return "stabilityai/stable-diffusion-xl-refiner-1.0"


def pytest_configure(config):
    """Override global timeout setting for SDXL tests"""
    config.option.timeout = 0


def pytest_addoption(parser):
    parser.addoption(
        "--start-from",
        action="store",
        default=0,
        help="Start from prompt number (0-4999)",
    )
    parser.addoption(
        "--num-prompts",
        action="store",
        default=5000,
        help="Number of prompts to process (default: 5000)",
    )
    parser.addoption(
        "--reset-bool",
        action="store",
        type=int,
        default=1,
        help="Whether to reset periodically (1 or 0), default: 1",
    )
    parser.addoption(
        "--reset-period",
        action="store",
        default=200,
        type=int,
        help="How often to reset (default: 200 (images))",
    )
    parser.addoption(
        "--loop-iter-num",
        action="store",
        default=10,
        help="Number of iterations of denoising loop (default: 10)",
    )
    parser.addoption(
        "--debug-mode",
        action="store_true",
        default=False,
        help="Run SDXL in debug mode (default: False)",
    )
    parser.addoption(
        "--lora-weights",
        action="store",
        default=None,
        help="Full path to a local .safetensors file with LoRA weights. Overrides --lora-hf-repo and --lora-hf-filename",
    )
    parser.addoption(
        "--lora-hf-repo",
        action="store",
        default=None,
        help="Hugging Face repo id for LoRA (e.g. 'user/repo'). Required together with --lora-hf-filename",
    )
    parser.addoption(
        "--lora-hf-filename",
        action="store",
        default=None,
        help="Filename in the Hugging Face repo (e.g. 'lora.safetensors'). Required together with --lora-hf-repo",
    )


@pytest.fixture
def evaluation_range(request):
    start_from = request.config.getoption("--start-from")
    num_prompts = request.config.getoption("--num-prompts")
    if start_from is not None:
        start_from = int(start_from)
    else:
        start_from = 0
    if num_prompts is not None:
        num_prompts = int(num_prompts)
    else:
        num_prompts = 5000
    return start_from, num_prompts


@pytest.fixture
def reset_config(request):
    reset_bool_val = request.config.getoption("--reset-bool")
    reset_period = request.config.getoption("--reset-period")
    if reset_bool_val is not None:
        reset_bool = bool(reset_bool_val)
    else:
        reset_bool = True
    if reset_period is not None:
        reset_period = int(reset_period)
    else:
        reset_period = 200
    return reset_bool, reset_period


def is_galaxy():
    import ttnn

    return (
        ttnn.cluster.get_cluster_type() == ttnn.cluster.ClusterType.GALAXY
        or ttnn.cluster.get_cluster_type() == ttnn.cluster.ClusterType.BLACKHOLE_GALAXY
    )


def get_device_name():
    import ttnn

    cluster_type = ttnn.cluster.get_cluster_type()
    cluster_type_to_name = {
        ttnn.cluster.ClusterType.N150: "n150",
        ttnn.cluster.ClusterType.N300: "n300",
        ttnn.cluster.ClusterType.N300_2x2: "n300_2x2",
        ttnn.cluster.ClusterType.T3K: "t3k",
        ttnn.cluster.ClusterType.GALAXY: "galaxy",
        ttnn.cluster.ClusterType.BLACKHOLE_GALAXY: "bh_galaxy",
        ttnn.cluster.ClusterType.P100: "p100",
        ttnn.cluster.ClusterType.P150: "p150",
        ttnn.cluster.ClusterType.P150_X2: "p150x2",
        ttnn.cluster.ClusterType.P150_X4: "p150x4",
        ttnn.cluster.ClusterType.P150_X8: "p150x8",
        ttnn.cluster.ClusterType.P300: "p300",
        ttnn.cluster.ClusterType.P300_X2: "p300x2",
    }
    return cluster_type_to_name.get(cluster_type, "unknown")


@pytest.fixture
def loop_iter_num(request):
    return int(request.config.getoption("--loop-iter-num"))


@pytest.fixture
def debug_mode(request):
    return request.config.getoption("--debug-mode")


@pytest.fixture(scope="function")
def validate_fabric_compatibility(request):
    """
    Validate that fabric configuration is compatible with the requested mesh device configuration.
    This fixture runs before mesh_device creation to catch incompatibilities early.
    It is needed to be able to gracefully fail if the configuration is not possible.
    """
    import ttnn

    params = getattr(request.node, "callspec", {}).params
    use_cfg_parallel = params.get("use_cfg_parallel", None)
    mesh_device_param = params.get("mesh_device", None)

    if not use_cfg_parallel:
        return

    if mesh_device_param is not None:
        total_devices = ttnn.GetNumAvailableDevices()

        if isinstance(mesh_device_param, int):
            requested_devices = mesh_device_param
        elif isinstance(mesh_device_param, tuple):
            requested_devices = mesh_device_param[0] * mesh_device_param[1]
        else:
            requested_devices = total_devices

        assert requested_devices == total_devices, "Requested devices must be equal to total devices"


@pytest.fixture
def sdxl_l1_small_size():
    """
    Returns the appropriate L1_SMALL_SIZE value based on device architecture.
    """
    return SDXL_L1_SMALL_SIZE if is_wormhole_b0() else SDXL_L1_SMALL_SIZE_BH


@pytest.fixture(scope="function")
def device_params(request, sdxl_l1_small_size):
    """
    Override the global device_params fixture to automatically inject SDXL L1_SMALL_SIZE.

    If the parametrized device_params dict doesn't contain 'l1_small_size',
    it will be automatically added based on the device architecture (Wormhole vs Blackhole).

    NOTE: `device_params` fixture exists in conftest.py in root but this one will take precedence.
    Fixture in conftest.py closest to the test in hierarchy will take precedence.
    We can still set L1_SMALL_SIZE inside tests for some specific cases if needed.
    Otherwise, when we do not specify it inside parentheses in test params, it will get set
    inside this device_params fixture.
    """
    params = getattr(request, "param", {})

    # Auto-inject l1_small_size if not already specified
    if "l1_small_size" not in params:
        params = {**params, "l1_small_size": sdxl_l1_small_size}

    return params


def _resolve_local_lora_file_path(path_input):
    if not path_input or not path_input.strip():
        return None
    resolved_path = Path(path_input).expanduser().resolve()
    if not resolved_path.exists() or not resolved_path.is_file():
        return None
    return str(resolved_path)


# Same large-file cache endpoint the rest of the SDXL suite fetches from in CIv2
# (see test_lora_perf.py and the root conftest's CIv2ModelDownloadUtils_). The env
# override exists so the fetch path can be exercised outside the CI cluster.
_LORA_CI_V2_CACHE_ENDPOINT = os.environ.get(
    "SDXL_LORA_CACHE_ENDPOINT",
    "http://large-file-cache.large-file-cache.svc.cluster.local//mldata/model_checkpoints/pytorch/huggingface",
)


def _endpoint_is_permitted(endpoint):
    """Whether the CIv2 cache endpoint may be fetched from.

    https anywhere; http only for the cluster-internal cache service. Everything else,
    including file:// and plaintext to an external host, is refused.
    """
    import urllib.parse

    parsed = urllib.parse.urlparse(endpoint)
    if parsed.scheme == "https":
        return True
    if parsed.scheme != "http":
        return False
    host = parsed.hostname or ""
    return host.endswith(".svc.cluster.local") or host in ("localhost", "127.0.0.1")


def _fetch_lora_from_ci_v2_cache(cache_dir, filename):
    """Fetch a single adapter file from the CIv2 large-file cache.

    Returns the local path on success, None on any failure (caller falls back to
    HF). CIv2 runners have no HF egress, so for adapters that are not already in
    the baked HF cache this is the only route that works there.
    """
    import shutil
    import urllib.request

    target_dir = Path("/tmp/ttnn_model_cache/lora") / cache_dir
    target = target_dir / filename
    if target.is_file() and target.stat().st_size > 0:
        return str(target)
    target_dir.mkdir(parents=True, exist_ok=True)
    endpoint = f"{_LORA_CI_V2_CACHE_ENDPOINT}/{cache_dir}/{filename}"
    # Fetched in-process rather than by shelling out to wget: no OS command means
    # nothing for the endpoint override to be injected into, and no dependency on
    # wget being installed on the runner.
    #
    # The endpoint is checked rather than trusted. urlopen would otherwise honour
    # file://, turning the override into a local file read, and plaintext to an
    # arbitrary host would be a real exposure. Cleartext is allowed only for the
    # in-cluster cache service, whose traffic never leaves the cluster and which is
    # the reason this fetch path exists at all; anything else must be TLS.
    if not _endpoint_is_permitted(endpoint):
        logger.warning(f"Refusing CIv2 LoRA cache endpoint {endpoint}: use https, or the in-cluster cache over http")
        return None
    try:
        with urllib.request.urlopen(endpoint, timeout=300) as response, open(target, "wb") as out:  # noqa: S310
            shutil.copyfileobj(response, out)
    except Exception as e:
        logger.warning(f"CIv2 LoRA cache fetch failed for {endpoint}: {e}")
        target.unlink(missing_ok=True)
        return None
    if not (target.is_file() and target.stat().st_size > 0):
        target.unlink(missing_ok=True)
        return None
    return str(target)


def _resolve_lora_weights_path(
    request, is_ci_env, is_ci_v2_env, default_repo_id, default_filename, default_revision=None, ci_v2_cache_dir=None
):
    """Resolve a LoRA weights path.

    1) --lora-weights: full path to a local .safetensors file.
    2) --lora-hf-repo and --lora-hf-filename: download from Hugging Face.
    3) If nothing provided: use the supplied default weights, pinned to
       `default_revision` so an upstream re-upload cannot silently change what
       the PCC references are compared against. The pin is best-effort: where it
       cannot be resolved (a baked HF cache holding a different snapshot) the
       fetch falls back to the default branch with a warning, rather than
       skipping the test and reporting green on no coverage. In CI (v1 or v2),
       adapters with a `ci_v2_cache_dir` are fetched from the large-file cache
       first: CIv2 runners have no HF egress, and CIv1 runners call
       hf_hub_download with local_files_only=True, so an adapter absent from
       the baked HF cache is unreachable on both without this. A runner outside
       the cluster fails the fetch instantly on DNS and falls through to HF, so
       the attempt is harmless where the cache is unreachable. Elsewhere, and
       as the CI fallback, adapters come from HF.

    Shared by the LoRA fixtures below, which differ only in which adapter they
    default to; resolution order and download behaviour are identical.
    """
    lora_weights_cli_path = request.config.getoption("--lora-weights", default=None)
    hf_repo_id = request.config.getoption("--lora-hf-repo", default=None)
    hf_filename = request.config.getoption("--lora-hf-filename", default=None)

    # Local file path via --lora-weights
    if lora_weights_cli_path is not None and str(lora_weights_cli_path).strip():
        resolved_lora_path = _resolve_local_lora_file_path(lora_weights_cli_path)
        if resolved_lora_path:
            return resolved_lora_path
        pytest.skip(
            f"LoRA path must be an existing .safetensors file: {lora_weights_cli_path}. "
            f"Provide a full path to the file (not a directory)."
        )
        return

    tried_ci_cache = False
    hf_revision = None
    if not (hf_repo_id and hf_filename):
        logger.warning(
            f"No LoRA weights provided. Using default weights. Repo: {default_repo_id}, File: {default_filename}"
        )
        hf_repo_id = default_repo_id
        hf_filename = default_filename
        # Only the built-in defaults are pinned. A caller-supplied repo/filename
        # keeps HF's default branch, since this revision does not describe it.
        hf_revision = default_revision

        if (is_ci_env or is_ci_v2_env) and ci_v2_cache_dir:
            tried_ci_cache = True
            cached = _fetch_lora_from_ci_v2_cache(ci_v2_cache_dir, default_filename)
            if cached:
                return cached

    def _download(revision):
        from huggingface_hub import hf_hub_download

        return hf_hub_download(
            repo_id=hf_repo_id,
            filename=hf_filename,
            revision=revision,
            local_files_only=is_ci_env and not is_ci_v2_env,
        )

    if hf_revision:
        try:
            return _download(hf_revision)
        except Exception as e:
            # The pin guards against an upstream re-upload; it is not worth losing
            # coverage over. A runner whose baked HF cache holds this adapter under a
            # different snapshot cannot resolve the pinned revision offline, and giving
            # up here would drop the whole suite to skips while still reporting green.
            # Fall back to whatever is reachable, but say so: the bytes under test are
            # then not the ones the pin describes, which is worth seeing in the log.
            logger.warning(
                f"Pinned revision {hf_revision} for {hf_repo_id}/{hf_filename} could not be resolved ({e}). "
                f"Falling back to the default branch: the adapter under test may differ from the pinned one, "
                f"so treat a PCC change here as a possible fixture change rather than a model regression."
            )

    try:
        return _download(None)
    except Exception as _:
        ci_cache_note = (
            f" Also tried the CIv2 large-file cache ({ci_v2_cache_dir}/{hf_filename}) without success."
            if tried_ci_cache
            else ""
        )
        revision_note = f" (pinned revision {hf_revision} was unresolvable too)" if hf_revision else ""
        pytest.skip(
            f"LoRA weights not available from HF ({hf_repo_id}, {hf_filename}){revision_note}.{ci_cache_note} "
            f"Use --lora-weights for a local file path, or ensure network/cache for HF."
        )
        return


@pytest.fixture(scope="function")
def lora_path(request, is_ci_env, is_ci_v2_env):
    """LoRA weights path, defaulting to the UNet-only test adapter."""
    from models.demos.stable_diffusion_xl_base.lora.config import TEST_LORA_REVISION

    return _resolve_lora_weights_path(
        request,
        is_ci_env,
        is_ci_v2_env,
        TEST_LORA_REPO_ID,
        TEST_LORA_FILENAME,
        default_revision=TEST_LORA_REVISION,
    )


@pytest.fixture(scope="function")
def te_lora_path(request, is_ci_env, is_ci_v2_env):
    """LoRA weights path, defaulting to a text-encoder-impacting adapter.

    That default trains both CLIP encoders plus the UNet, so it is the adapter used
    to exercise the text-encoder fuse/rollback path. Resolution is identical to
    `lora_path` except that in CI the adapter comes from the large-file cache,
    where it was staged on 2026-07-21 (this adapter is not in the runners' baked
    HF cache, and hf_hub_download cannot fetch it there on either CI flavour).
    """
    from models.demos.stable_diffusion_xl_base.lora.config import (
        TE_TEST_LORA_CI_CACHE_DIR,
        TE_TEST_LORA_FILENAME,
        TE_TEST_LORA_REPO_ID,
        TE_TEST_LORA_REVISION,
    )

    return _resolve_lora_weights_path(
        request,
        is_ci_env,
        is_ci_v2_env,
        TE_TEST_LORA_REPO_ID,
        TE_TEST_LORA_FILENAME,
        default_revision=TE_TEST_LORA_REVISION,
        ci_v2_cache_dir=TE_TEST_LORA_CI_CACHE_DIR,
    )
