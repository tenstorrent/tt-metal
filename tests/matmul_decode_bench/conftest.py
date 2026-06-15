# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Fixtures for the pi05_chunked_mmdecode profiling harness.

HARD fork-resolution gate (module import time): ttnn MUST resolve under the
matmul_decode fork, else HARD-ABORT -- ``ttnn.matmul_decode`` is the REAL device
op only on the fork. Run with the fork python_env + PYTHONPATH=<repo>.

``dev`` -- a single Blackhole (P150) device opened directly (no ttnn pytest-plugin
fixtures in this repo). ``_tracy_signpost`` -- per-test signpost (gated on
PI05_TRACY_SIGNPOST=1).
"""
from __future__ import annotations

import os
import subprocess

import pytest

import ttnn

# ---- HARD fork-resolution gate (BLOCKING, import time) ----
_FORK_TAG = "tt-metal-matmul_decode"
# deep-plan_5 working-tree edits were COMMITTED on top of the e4500c1f M-fix anchor
# as b5a4cf05d9f ("increased to full width") -- the wide-N cap + K-B stream collapse.
# Accept EITHER the original anchor (working-tree edits) OR the committed-fix HEAD.
_FORK_HEAD_ANCHOR = "e4500c1fae97c103b16fc24fc7010b852992a9e6"
# The plan5 "increased to full width" commit has been amended a couple of times
# (b5a4cf0 -> b56f57e); all are children of the e4500c1f anchor with the same
# (collapse) tree. Accept the anchor, the known plan5 HEADs, OR any commit whose
# PARENT is the anchor (an amend of the plan5 commit) so further amends don't break.
_FORK_HEAD_PLAN5 = "b5a4cf05d9fba3d828c392017fe1b520c544d42c"
_FORK_HEAD_PLAN5_AMEND = "b56f57e5ae421eebb2ad65177053db53c3be6f13"
# iter-7 evolution: "Added L1 matmul with frozen weights" (L1 matmul, frozen
# weights), committed as a child of b56f57e5. This is the CURRENT target HEAD.
_FORK_HEAD_FROZEN_L1 = "e7023ed97545c88b646b7ac73f314e54f9dec33b"
_FORK_HEADS_OK = (
    _FORK_HEAD_ANCHOR,
    _FORK_HEAD_PLAN5,
    _FORK_HEAD_PLAN5_AMEND,
    _FORK_HEAD_FROZEN_L1,
)
_FORK_BRANCH = "alnah005/matmul_decode_M_fix"
_ttnn_path = os.path.realpath(ttnn.__file__)
if _FORK_TAG not in _ttnn_path:
    raise RuntimeError(
        f"FORK REPOINT FAILED: ttnn.__file__={_ttnn_path} does not resolve under the fork "
        f"'{_FORK_TAG}'. Remedy: run with the fork python_env "
        f"($FORK/python_env/bin/python) + PYTHONPATH=<repo>. "
        f"`export TT_METAL_HOME` ALONE is INSUFFICIENT."
    )
if not hasattr(ttnn, "matmul_decode"):
    raise RuntimeError(
        "FORK REPOINT FAILED: ttnn.matmul_decode missing -- not the matmul_decode fork."
    )


def _fork_dir():
    return os.path.dirname(os.path.dirname(os.path.dirname(_ttnn_path)))


def _git(fork, *args):
    try:
        return subprocess.check_output(["git", "-C", fork, *args], text=True).strip()
    except Exception:
        return None


def _fork_head():
    return _git(_fork_dir(), "rev-parse", "HEAD")


@pytest.fixture(scope="session", autouse=True)
def _fork_resolution_gate():
    p = os.path.realpath(ttnn.__file__)
    assert _FORK_TAG in p, f"ttnn not from fork: {p}"
    assert hasattr(ttnn, "matmul_decode"), "ttnn.matmul_decode missing"
    head = _fork_head()
    # The plan5 collapse commit gets amended; accept the known HEADs OR any commit
    # whose parent is the e4500c1f anchor (an amend of the collapse commit).
    parent = _git(_fork_dir(), "rev-parse", "HEAD^")
    ok = head in _FORK_HEADS_OK or parent == _FORK_HEAD_ANCHOR
    assert ok, (
        f"fork HEAD {head} (parent {parent}) not the M-fix anchor / plan5 collapse "
        f"commit ({_FORK_HEADS_OK}, or child-of-{_FORK_HEAD_ANCHOR}). "
        f"Rebuild + install the fork ttnn target.")
    yield


# ---- test-only fill_cache shim (so the unchunked pi0.5 baseline + routed e2e
#      run on the fork; mmd src untouched) ----
try:
    from tests.models.pi05_blocked.pi05_blocked_helpers import (
        install_fill_cache_shim,
        restore_fill_cache_shim,
    )
except Exception:  # pragma: no cover
    install_fill_cache_shim = None
    restore_fill_cache_shim = None


@pytest.fixture(scope="session", autouse=True)
def _fill_cache_shim():
    if install_fill_cache_shim is not None:
        install_fill_cache_shim()
    _wrapped = ttnn.fill_cache

    def _dtype_coercing_fill_cache(cache_tensor, input_tensor, batch_idx, update_idx=None, **kwargs):
        src = input_tensor
        if (update_idx is None or update_idx == 0) and src.dtype != cache_tensor.dtype:
            src = ttnn.typecast(src, cache_tensor.dtype)
        return _wrapped(cache_tensor, src, batch_idx, update_idx=update_idx, **kwargs)

    ttnn.fill_cache = _dtype_coercing_fill_cache
    yield
    ttnn.fill_cache = _wrapped
    if restore_fill_cache_shim is not None:
        restore_fill_cache_shim()


@pytest.fixture(autouse=True)
def _tracy_signpost(request):
    if os.environ.get("PI05_TRACY_SIGNPOST"):
        try:
            from tracy import signpost

            signpost(header="T_" + request.node.name.replace("[", "_").replace("]", ""))
        except Exception:
            pass
    yield


@pytest.fixture(scope="module")
def dev():
    trace_region = int(os.environ.get("PI05_MMD_TRACE_REGION", 256 * 1024 * 1024))
    device = ttnn.open_device(device_id=0, l1_small_size=24576, trace_region_size=trace_region)
    try:
        yield device
    finally:
        ttnn.close_device(device)
