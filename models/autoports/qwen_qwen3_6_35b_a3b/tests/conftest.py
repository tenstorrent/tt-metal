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


@pytest.fixture(scope="session", autouse=True)
def _fresh_pcc_logs(request):
    """One run == one provenance file for the logs this suite owns.

    In a **filtered** session (``-k`` / ``-m``) *these* logs go to ``*_partial.jsonl`` instead.
    Without that, ``pytest -k context_contract`` — which collects two tests that record nothing —
    would delete the ``pcc.jsonl`` the full run had just produced. Only the names listed here are
    diverted: ``long_context.jsonl`` is written by five deliberately-filtered runs
    (``tests/run_long_context.sh``) and must keep accumulating into the real file.
    """
    owned = ("pcc", "pcc_real_weights")
    filtered = bool(getattr(request.config.option, "keyword", "") or getattr(request.config.option, "markexpr", ""))
    harness.PARTIAL_LOGS = set(owned) if filtered else set()
    for name in owned:
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
