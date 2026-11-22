# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""V4.1's device-resident prefix cache: ``capture_device_prefix`` keeps a session's state in device
DRAM by sharing its whole compressed blocks and copying the part-filled ones, and
``import_device_prefix`` starts another session from it the same way.

A session is filled with recognisable bytes (as in ``test_kv_migration_profile``) at a length that
leaves every compressor group's last block part-filled, and its host export is the reference. The
prefix is captured, the owner is reset and refilled with different bytes (so a block freed back to the
pool while the prefix still held it would show up as corruption), and then:

* a second session imported from the prefix must export bit-exactly the reference;
* the prefix's own host export (what the server does when the pools fill) must too;
* once everything is released, every pool must be back where it started.

Run (32-chip Galaxy, ttnn venv, from the repo root)::

    pytest models/experimental/deepseek_v41_flash/tests/decode/test_device_prefix_cache.py -s
"""

from __future__ import annotations

import contextlib
import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.deepseek_v41_flash.tests.decode.test_full_model_decode_demo import (
    _PAGE_BLOCK_SIZE,
    _build_rope,
    _checkpoint_available,
    _construct_model,
    _traced_max_seq,
)
from models.experimental.deepseek_v41_flash.tests.decode.test_kv_migration_profile import _assert_round_trip, _fill
from models.experimental.deepseek_v41_flash.tt.config import DEFAULT_MODEL_DIR, load_config
from models.experimental.deepseek_v4_flash.tt.weight_loader import DeepseekV4WeightLoader

# A whole number of sliding windows (so the ring slot is whole) that is not a whole number of blocks for
# any compressor group: ratio 4 has 5152 entries (80.5 blocks of 64), ratio 128 has 161 (2.5 blocks).
_LENGTH = int(os.environ.get("DEEPSEEK_V41_DEVICE_PREFIX_LEN", "20608"))


def _place(model, sid: int, length: int, seed: int) -> None:
    model.reset_session(sid)
    model.ensure_session_capacity(length - 1, sid)
    model._session_pos[sid] = length
    _fill(model, sid, torch.Generator().manual_seed(seed))


@pytest.mark.skipif(not _checkpoint_available(), reason=f"V4.1-Flash checkpoint not found under {DEFAULT_MODEL_DIR}")
@pytest.mark.timeout(3600)
@torch.no_grad()
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_XY,
            "num_command_queues": 2,
            "l1_small_size": int(os.environ.get("DEEPSEEK_V4_L1_SMALL_SIZE", "8192")),
        }
    ],
    indirect=["device_params"],
    ids=["fabric_1d"],
)
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True, ids=["tp4_8stage"])
def test_device_prefix_round_trip(mesh_device, reset_seeds) -> None:
    loader = DeepseekV4WeightLoader(DEFAULT_MODEL_DIR)
    config = load_config(loader.snapshot_dir)
    with contextlib.ExitStack() as prefetcher:
        model, lm_head, loader, config = _construct_model(mesh_device, prefetcher, loader=loader, config=config)
        max_seq = _traced_max_seq(config, _LENGTH + 1, model.num_layers)
        # Two sessions plus the prefix they share.
        model.prepare_static_decode(
            _build_rope(config, loader.snapshot_dir, max_seq),
            max_seq,
            lm_head=lm_head,
            num_sessions=2,
            total_tokens=3 * max_seq,
            block_size=_PAGE_BLOCK_SIZE,
        )
        owner, other = model.open_session(), model.open_session()
        empty = model._paged.usage()

        _place(model, owner, _LENGTH, seed=1)
        want = model.export_session(owner)

        start = time.perf_counter()
        prefix = model.capture_device_prefix(owner)
        capture_ms = (time.perf_counter() - start) * 1e3
        assert prefix.pos == _LENGTH

        # The owner moves on: its blocks go back to the pools, and new bytes land in whatever it gets next.
        _place(model, owner, _LENGTH, seed=2)
        moved_on = model.export_session(owner)
        assert not all(torch.equal(w, g) for w, g in zip(want.entries[min(want.entries)], moved_on.entries[min(want.entries)]))

        start = time.perf_counter()
        model.import_device_prefix(other, prefix)
        import_ms = (time.perf_counter() - start) * 1e3
        _assert_round_trip(want, model.export_session(other), _LENGTH)

        start = time.perf_counter()
        exported = model.export_device_park(prefix)
        export_ms = (time.perf_counter() - start) * 1e3
        _assert_round_trip(want, exported, _LENGTH)

        logger.info(
            f"device prefix at {_LENGTH} tokens ({want.nbytes / 1e6:.1f} MB): capture {capture_ms:.1f} ms, "
            f"import {import_ms:.1f} ms, export to host {export_ms:.1f} ms"
        )

        model.reset_session(owner)
        model.reset_session(other)
        assert model._paged.usage() == empty, f"blocks leaked: {model._paged.usage()} vs {empty}"
