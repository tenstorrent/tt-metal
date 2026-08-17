# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Fixtures for the Qwen3.6-35B-A3B functional-decoder tests.

Building a layer materialises ~1.5 GiB of expert weights, so pairs are cached per
(kind, batch, context, weights, extra config) for the whole session and the device is
opened once.
"""

import gc

import pytest
import torch

import ttnn
from models.autoports.qwen_qwen3_6_35b_a3b.tests import harness

#: Logs that only a *whole* run of ``test_functional_decoder.py`` may replace.
_OWNED_LOGS = ("pcc", "pcc_real_weights")
#: Set during collection; see ``pytest_collection_modifyitems``.
_MAY_REPLACE_OWNED_LOGS = False


def pytest_collection_modifyitems(session, config, items):
    """Decide whether this session may replace the suite's provenance logs.

    There are three ways to run a subset here and all three have to be caught, because each one has
    destroyed evidence at least once during this stage:

    * ``-k`` -- ``run_watcher.sh`` and the contract check; collect tests that record nothing;
    * ``-m`` -- ``run_long_context.sh``;
    * a **node id** -- ``run_perf.sh`` passes ``test_perf.py::test_perf_prefill[linear]``, which sets
      neither option. That is how the perf runs came to delete ``pcc.jsonl``.

    A session may replace the owned logs only if it selected nothing (no ``-k``, no ``-m``, no node
    id) *and* actually collected the file that owns them. Everything else writes ``*_partial.jsonl``
    (gitignored) and leaves the committed evidence alone. ``long_context.jsonl`` is never diverted:
    it is *meant* to be written by six separate filtered runs accumulating into one file.
    """
    global _MAY_REPLACE_OWNED_LOGS
    selected = bool(
        getattr(config.option, "keyword", "")
        or getattr(config.option, "markexpr", "")
        or any("::" in str(arg) for arg in config.args)
    )
    owns = any("test_functional_decoder.py" in str(item.fspath) for item in items)
    _MAY_REPLACE_OWNED_LOGS = owns and not selected


@pytest.fixture(scope="session", autouse=True)
def _fresh_pcc_logs():
    """One run == one provenance file for the logs this suite owns.

    See ``pytest_collection_modifyitems`` for which sessions are allowed to replace them and why the
    check is about what was selected rather than about a single flag.
    """
    harness.PARTIAL_LOGS = set() if _MAY_REPLACE_OWNED_LOGS else set(_OWNED_LOGS)
    for name in _OWNED_LOGS:
        harness.reset_log(name)
    yield
    harness.PARTIAL_LOGS = set()


@pytest.fixture(scope="session")
def qwen_device():
    torch.set_num_threads(16)
    device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    yield device
    ttnn.close_mesh_device(device)


@pytest.fixture(scope="session")
def layer_pairs(qwen_device):
    cache: dict = {}

    def get(kind, *, max_batch_size=1, supported_context=4096, real_weights=False, **cfg_kwargs):
        key = (kind, max_batch_size, supported_context, real_weights, tuple(sorted(cfg_kwargs.items())))
        if key not in cache:
            # Each distinct key costs ~1.5 GiB of expert weights for the rest of the session, and
            # nothing evicts. Adding a test with a fresh (kind, batch, context) combination is
            # therefore a memory decision: prefer reusing an existing key. Logged so an OOM in a
            # later test is traceable to whoever added a key.
            print(f"[layer_pairs] building {key} (live pairs: {len(cache)})")
            cache[key] = harness.build_layer_pair(
                qwen_device,
                kind=kind,
                max_batch_size=max_batch_size,
                supported_context=supported_context,
                real_weights=real_weights,
                **cfg_kwargs,
            )
        pair = cache[key]
        pair.tt.reset_state()
        return pair

    yield get
    cache.clear()
    gc.collect()
