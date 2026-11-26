# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""A one-row decode step against a B=4 batch step (``decode_batch_block``), timed and under the device profiler.

The instance is ``test_batch_decode.py``'s: one persistent TP4 stage on row 4, every MoE on the TP12 server, built
with ``batch_block=True``. Four sessions decode the same GSM8K-style text, filled by batch steps up to each of
``DEEPSEEK_V41_BATCH_PROFILE_POSITIONS`` (default 256: every select dense; 2048: every index layer's select scored).
At each position ``DEEPSEEK_V41_BATCH_PROFILE_STEPS`` (8) one-row steps of session 0 run (``decode_traced``,
synchronous), then as many batch steps of all four sessions, each phase between tracy signposts
(``ONE_ROW_<pos>`` / ``BATCH_<pos>`` and ``..._END``), the stage's device profiler buffers read after every step. The
log reports each phase's median host wall time per step. The fill steps overflow the profiler's buffers, which fails
tracy's merged ops report (``-r``), but its per-op device report is complete for the profiled phases, and
``tools/batch_profile_report.py`` reads it by trace id and replay session::

    python -m tracy -r -p -v --no-web-server -t 8160 --op-support-count 2000 -o <dir> -m pytest <this file> -s
    python models/experimental/deepseek_v41_flash/tools/batch_profile_report.py <dir>/.logs/cpp_device_perf_report.csv --list

Nov 25 (Galaxy, no profiler): one-row step 7.15 / 7.93 ms, batch step 10.72 / 13.50 ms at 256 / 2048. Under the
profiler the batch step's growth is the MoE requests (1.5x), flash_mqa_decode (2.1x), the all-reduces (1.8x), and from
2048 the select, which runs once per row: fused_lightning_select_kv 8 -> 32 calls a step, 4.05x, plus its per-row
head replication, slices and reshards (+1.0 ms).
"""

from __future__ import annotations

import contextlib
import os
import statistics
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.deepseek_v41_flash.tests.decode.test_full_model_decode_demo import (
    _L1_SMALL_SIZE,
    _PAGE_BLOCK_SIZE,
    _SEED,
    _build_rope,
    _checkpoint_available,
    _construct_model,
    _stage_engram_or_skip,
    _traced_max_seq,
)
from models.experimental.deepseek_v41_flash.tests.decode.test_multi_session_decode import _TEXTS
from models.experimental.deepseek_v41_flash.tt.config import DEFAULT_MODEL_DIR, load_config
from models.experimental.deepseek_v4_flash.tt.weight_loader import DeepseekV4WeightLoader

try:
    from tracy import signpost
except ImportError:  # pragma: no cover - tracy is optional

    def signpost(*args, **kwargs):
        pass


# DEEPSEEK_V41_BATCH_ROWS=2: B=2 batch steps (2-row blocks) instead of B=4.
_ROWS = int(os.environ.get("DEEPSEEK_V41_BATCH_ROWS", "4"))
assert _ROWS in (2, 4), f"DEEPSEEK_V41_BATCH_ROWS takes 2 or 4, got {_ROWS}"
_POSITIONS = tuple(int(p) for p in os.environ.get("DEEPSEEK_V41_BATCH_PROFILE_POSITIONS", "256,2048").split(","))
_STEPS = int(os.environ.get("DEEPSEEK_V41_BATCH_PROFILE_STEPS", "8"))
_PROFILING = os.environ.get("TT_METAL_DEVICE_PROFILER") == "1"
_BATCH_L1_SMALL_SIZE = max(_L1_SMALL_SIZE, 16384)  # as demo/server.py under --persistent-moe (B=2 needs past 12 KB)


def _flush(device) -> None:
    """Every step done, then the stage's profiler buffers read (so a phase never overflows them)."""
    ttnn.synchronize_device(device)
    if _PROFILING:
        ttnn.ReadDeviceProfiler(device)


@pytest.mark.skipif(not _checkpoint_available(), reason=f"V4.1-Flash checkpoint not found under {DEFAULT_MODEL_DIR}")
@pytest.mark.timeout(7200)
@torch.no_grad()
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_XY,
            "num_command_queues": 2,
            "l1_small_size": _BATCH_L1_SMALL_SIZE,
        }
    ],
    indirect=["device_params"],
    ids=["fabric_1d"],
)
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True, ids=["tp4_1stage_persistent_moe12"])
def test_batch_decode_profile(mesh_device, reset_seeds) -> None:
    from transformers import AutoTokenizer

    loader = DeepseekV4WeightLoader(DEFAULT_MODEL_DIR)
    config = load_config(loader.snapshot_dir)
    tokenizer = AutoTokenizer.from_pretrained(loader.snapshot_dir)
    num_layers = min(
        int(os.environ.get("DEEPSEEK_V4_DECODE_LAYERS", config.num_hidden_layers)), config.num_hidden_layers
    )
    _stage_engram_or_skip(loader, config)
    last = max(_POSITIONS) + 2 * _STEPS
    text = tokenizer.encode(" ".join(_TEXTS) + " ", add_special_tokens=False)
    tokens = (text * (last // len(text) + 2))[: last + 1]
    max_seq = _traced_max_seq(config, last + 1, num_layers)

    with contextlib.ExitStack() as stack:
        model, lm_head, _, _ = _construct_model(
            mesh_device,
            stack,
            loader=loader,
            config=config,
            num_stages=1,
            persistent_moe=True,
            verify_rows=_ROWS,
            batch_block=True,
        )
        rope = _build_rope(config, loader.snapshot_dir, max_seq)
        model.prepare_static_decode(
            rope,
            max_seq,
            lm_head=lm_head,
            num_sessions=_ROWS + 1,
            total_tokens=(_ROWS + 1) * max_seq,
            block_size=_PAGE_BLOCK_SIZE,
            device_sampling=True,
        )
        model.device_sampler.set_sampling(0.0, _SEED)
        model.prepare_batch_decode()
        sids = [model.open_session() for _ in range(_ROWS)]
        model.capture_batch_traces(tokens[0])
        for sid in sids:
            model.ensure_session_capacity(max_seq - 1, sid)
        device = model.submeshes_io[0]["device"]
        _flush(device)

        pos = [0] * _ROWS
        results = {}
        for target in _POSITIONS:
            # Every session to the target, in batch steps (the profiler drops what overflows its buffers here).
            while pos[0] < target:
                model.decode_batch_block([tokens[p] for p in pos], sids, positions=list(pos))
                pos = [p + 1 for p in pos]
            _flush(device)

            times = []
            model.activate_session(sids[0])
            signpost(f"ONE_ROW_{target}")
            for _ in range(_STEPS):
                t0 = time.perf_counter()
                model.decode_traced(tokens[pos[0]], pos[0])
                times.append(time.perf_counter() - t0)
                pos[0] += 1
                if _PROFILING:
                    _flush(device)  # a step's ~1000 programs fill the profiler's default buffers
            signpost(f"ONE_ROW_{target}_END")
            _flush(device)
            results[("one_row", target)] = times

            times = []
            signpost(f"BATCH_{target}")
            for _ in range(_STEPS):
                t0 = time.perf_counter()
                model.decode_batch_block([tokens[p] for p in pos], sids, positions=list(pos))
                times.append(time.perf_counter() - t0)
                pos = [p + 1 for p in pos]
                if _PROFILING:
                    _flush(device)
            signpost(f"BATCH_{target}_END")
            _flush(device)
            results[("batch", target)] = times
            # Session 0 ran ahead in the one-row phase: bring the others level for the next position.
            while min(pos) < max(pos):
                behind = [j for j in range(_ROWS) if pos[j] < max(pos)]
                model.decode_batch_block(
                    [tokens[pos[j]] for j in behind], [sids[j] for j in behind], [pos[j] for j in behind]
                )
                for j in behind:
                    pos[j] += 1

        for (kind, target), times in sorted(results.items(), key=lambda kv: (kv[0][1], kv[0][0])):
            ms = 1e3 * statistics.median(times)
            rows = 1 if kind == "one_row" else _ROWS
            logger.info(
                f"position {target}: {kind} step {ms:.2f} ms median of {len(times)} (host wall, synchronous), "
                f"{rows} token(s) -> {1e3 * rows / ms:.1f} tok/s"
                + (" [under the device profiler]" if _PROFILING else "")
            )
