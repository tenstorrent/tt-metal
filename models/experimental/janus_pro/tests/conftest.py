# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import glob
import os
import shutil

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--dummy_weights",
        action="store",
        default=False,
        type=bool,
        help="Use dummy/random weights instead of loading checkpoints in tests that support it.",
    )


@pytest.fixture
def dummy_weights(request):
    return request.config.getoption("--dummy_weights") or False


@pytest.fixture(autouse=True)
def _clear_stale_dummy_weight_cache(dummy_weights):
    """Remove tilized dummy-weight caches before every --dummy_weights test.

    The dummy cache dirs (tensor_cache_dummy_bf16 / _bfp8) are keyed only by
    tensor name+dtype+layout, never by weight content. Dummy weights are
    regenerated deterministically each run (reset_seeds), but a cache written
    by an EARLIER run with different weights (a JANUS_DEBUG_INIT_SCALE
    experiment, older code, a different transformers version) is silently
    reused and mismatches this run's fresh reference -> spurious PCC failures
    that look like device/precision bugs. Clearing forces a fresh tilize from
    the current weights. No-op for real-weight runs (they use a separate cache
    namespace and are not affected).
    """
    if dummy_weights:
        for cache_dir in _dummy_weight_cache_dirs():
            shutil.rmtree(cache_dir, ignore_errors=True)
    yield


def _dummy_weight_cache_dirs():
    """Locate tensor_cache_dummy_* dirs for the active HF_MODEL, across devices.

    Mirrors how ModelArgs resolves the cache root without opening a device:
    the tilized cache lives under either TT_CACHE_PATH or the resolved HF
    snapshot, in a per-device subdir (e.g. .../P150/tensor_cache_dummy_bfp8).
    """
    from models.experimental.janus_pro.tt.model_config import ModelArgs

    roots = []
    tt_cache = os.environ.get("TT_CACHE_PATH")
    if tt_cache:
        roots.append(tt_cache)
    hf_model = os.environ.get("HF_MODEL", "")
    if hf_model:
        if os.path.isabs(hf_model) and os.path.isdir(hf_model):
            roots.append(hf_model)
        else:
            snapshot = ModelArgs._resolve_hf_snapshot(hf_model)
            if snapshot:
                roots.append(snapshot)

    dirs = []
    for root in roots:
        # Cache sits one (device) or two (device/hf_rope) levels below the root.
        for depth in ("*", os.path.join("*", "*")):
            dirs.extend(glob.glob(os.path.join(root, depth, "tensor_cache_dummy_*")))
    return sorted(set(d for d in dirs if os.path.isdir(d)))
