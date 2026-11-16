# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""The traced DSpark draft on its own 1x4 row: replay time, host overhead and a golden of its outputs.

The drafter is built as ``DeepSeekV41Model.prepare_spec_decode`` builds it (row 3 of the 8x4 Galaxy, TP4, the
decode ``lm_head`` copy at bfloat4_b, its own prefetcher rings), with no decode model: random target-layer means
stand in for the stash. The rings are seeded position by position, ``write_main`` ``main_rows`` rows at a time, and
the traced draft runs at several frontiers (first lap, wrapped window, wrapped ring).

* Timing: the median of ``DSPARK_PERF_REPLAYS`` replays (dispatch + device, synchronized), the packet upload
  (``set_block``) and the readback of the ids and confidences, as ``spec_read_draft`` reads them.
* Exactness: ``DSPARK_PERF_GOLDEN=<file>`` saves the draft outputs (ids, confidences, the rows' logits) at each
  frontier when the file is missing, and otherwise requires the same ids and reports the logits' max difference
  (``DSPARK_PERF_EXACT=1`` also requires bit-equal logits).
* Profiling: each frontier's replay and the timed replays sit behind tracy signposts, e.g.

      python -m tracy -r -p -v --no-web-server -t 8150 -o <dir> -m pytest <this file> -s
"""

import contextlib
import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.deepseek_v41_flash.tests.decode.test_full_model_decode_demo import _build_rope
from models.experimental.deepseek_v41_flash.tt.config import CACHE_DIR, DEFAULT_MODEL_DIR, dequantized, load_config
from models.experimental.deepseek_v41_flash.tt.decode.dspark import DeepSeekV41DSpark, host_to_device
from models.experimental.deepseek_v41_flash.tt.lm_head import DeepSeekV41LMHead
from models.experimental.deepseek_v4_flash.tt.common import _trace_capture_guard
from models.experimental.deepseek_v4_flash.tt.weight_cache import WeightCache
from models.experimental.deepseek_v4_flash.tt.weight_loader import DeepseekV4WeightLoader, resolve_snapshot_dir
from tests.ttnn.unit_tests.operations.prefetcher_common import tensor_prefetcher_session

try:
    from tracy import signpost
except ImportError:  # pragma: no cover - tracy is optional

    def signpost(*args, **kwargs):
        pass


TP_SIZE = 4
DRAFTER_ROW = 3
REPLAYS = int(os.environ.get("DSPARK_PERF_REPLAYS", "20"))
GOLDEN = os.environ.get("DSPARK_PERF_GOLDEN", "")
EXACT = os.environ.get("DSPARK_PERF_EXACT", "0") == "1"
# The frontiers drafted at: first lap, past the window, past the ring (160 rows). The timing runs at the last.
FRONTIERS = tuple(
    int(f)
    for f in os.environ.get("DSPARK_PERF_FRONTIERS", "9,23,40,61,88,117,139,150,166,181,197,212,229,203").split(",")
)
# draft(greedy=True): the ids (and confidences) must match the general draft's golden; its rows are not written.
GREEDY = os.environ.get("DSPARK_PERF_GREEDY", "0") == "1"
PREFETCH = os.environ.get("DSPARK_PERF_PREFETCH", "1") == "1"
# Under tracy: flush the device profiler's buffers between phases, so a long seed does not overflow them.
PROFILE = os.environ.get("DSPARK_PERF_PROFILE", "0") == "1"
MAX_SEQ = 256


def _snapshot():
    try:
        return resolve_snapshot_dir(DEFAULT_MODEL_DIR)
    except FileNotFoundError:
        return None


def _flush(device) -> None:
    if PROFILE:
        ttnn.synchronize_device(device)
        ttnn.ReadDeviceProfiler(device)


def _to_torch(t: ttnn.Tensor) -> torch.Tensor:
    return ttnn.to_torch(ttnn.get_device_tensors(t)[0])


@pytest.mark.skipif(_snapshot() is None, reason=f"V4.1-Flash checkpoint not found under {DEFAULT_MODEL_DIR}")
@pytest.mark.timeout(3600)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_XY,
            "trace_region_size": 256 << 20,
            "l1_small_size": 16384,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True, ids=["8x4"])
@torch.no_grad()
def test_dspark_draft_perf(mesh_device):
    device = mesh_device.create_submesh(ttnn.MeshShape(1, TP_SIZE), ttnn.MeshCoordinate(DRAFTER_ROW, 0))
    snapshot = _snapshot()
    config = load_config(snapshot)
    loader = DeepseekV4WeightLoader(snapshot)
    cache = WeightCache(CACHE_DIR)
    lm_head = DeepSeekV41LMHead(
        dequantized(loader, "lm_head.weight", translate=True), device, cache.file("lm_head"), dtype=ttnn.bfloat4_b
    )
    rope = _build_rope(config, snapshot, MAX_SEQ)
    t0 = time.perf_counter()
    drafter = DeepSeekV41DSpark(
        config, loader, device, lm_head, rope["main"], cache=cache, tp_size=TP_SIZE, use_prefetcher=PREFETCH
    )
    logger.info(f"drafter built in {time.perf_counter() - t0:.0f}s")
    with tensor_prefetcher_session(device) if PREFETCH else contextlib.nullcontext():
        torch.manual_seed(0)
        hidden, n_targets = config.hidden_size, len(drafter.targets)
        main_hidden = torch.randn(MAX_SEQ, n_targets * hidden).bfloat16().float()
        tokens = torch.randint(config.vocab_size, (MAX_SEQ,))
        stash = torch.zeros(n_targets, 1, 1, drafter.stash_rows, hidden)
        for p in range(MAX_SEQ):
            stash[:, 0, 0, p % drafter.stash_rows] = main_hidden[p].reshape(n_targets, hidden)
        for k, pieces in enumerate(drafter.stash):
            for j, dst in enumerate(pieces):
                w = dst.shape[-1]
                host_to_device(stash[k][..., j * w : (j + 1) * w], dst)

        written = 0

        def write_up_to(frontier: int) -> None:
            nonlocal written
            while written <= frontier:
                n = min(drafter.main_rows, frontier + 1 - written)
                drafter.set_main_rows(range(written, written + n))
                drafter.write_main()
                written += n
                _flush(device)

        # Compile (every program before the capture), then capture.
        write_up_to(FRONTIERS[0])
        drafter.set_block(int(tokens[FRONTIERS[0] + 1]), FRONTIERS[0])
        drafter.draft(greedy=GREEDY)
        _flush(device)
        ttnn.synchronize_device(device)
        tid = ttnn.begin_trace_capture(device, cq_id=0)
        with _trace_capture_guard():
            out = drafter.draft(greedy=GREEDY)
        ttnn.end_trace_capture(device, tid, cq_id=0)
        ids_t, conf_t, rows_t = out

        results = {}
        for f in FRONTIERS:
            if f < written:  # rewind: the stash holds every position, so rewrite the window up to f
                written = max(0, f - drafter.window + 1)
            write_up_to(f)
            drafter.set_block(int(tokens[f + 1]), f)
            ttnn.synchronize_device(device)
            signpost(f"draft_f{f}")
            ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
            _flush(device)
            ttnn.synchronize_device(device)
            results[f] = (
                _to_torch(ids_t).flatten().long(),
                _to_torch(conf_t).float().flatten(),
                _to_torch(rows_t).float().reshape(drafter.block, -1),
            )
            logger.info(
                f"frontier {f}: ids {results[f][0].tolist()} conf {[round(c, 3) for c in results[f][1].tolist()]}"
            )

        # Timing at the last frontier: replay alone, and as a round does it (upload, replay, read).
        f = FRONTIERS[-1]
        token = int(tokens[f + 1])
        signpost("timed")
        replay, upload, read, total = [], [], [], []
        for _ in range(REPLAYS):
            ttnn.synchronize_device(device)
            a = time.perf_counter()
            drafter.set_block(token, f)
            b = time.perf_counter()
            ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(device)
            c = time.perf_counter()
            _to_torch(ids_t)
            _to_torch(conf_t)
            d = time.perf_counter()
            upload.append(b - a)
            replay.append(c - b)
            read.append(d - c)
            total.append(d - a)
        signpost("timed_end")
        # As spec_draft_async + spec_read_draft: upload, replay and read with no synchronize in between.
        rounds = []
        for _ in range(REPLAYS):
            ttnn.synchronize_device(device)
            a = time.perf_counter()
            drafter.set_block(token, f)
            ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
            _to_torch(ids_t)
            _to_torch(conf_t)
            rounds.append(time.perf_counter() - a)

        def ms(v):
            return f"{1e3 * sorted(v)[len(v) // 2]:.3f} ms"

        logger.info(
            f"[dspark perf] replay {ms(replay)}, set_block {ms(upload)}, read ids+conf {ms(read)}, "
            f"sum {ms(total)}; as a round (no sync) {ms(rounds)}"
        )
        ttnn.release_trace(device, tid)

        # fused_experts' experts per block, swept at the last frontier: each is compiled, captured and timed, and its
        # ids and logits rows must equal the run's above (the block size does not change the arithmetic).
        for block in [int(b) for b in os.environ.get("DSPARK_PERF_SWEEP_BLOCKS", "").split(",") if b]:
            for stage in drafter.stages:
                stage.experts_block = block
            drafter.set_block(token, f)
            drafter.draft(greedy=GREEDY)
            ttnn.synchronize_device(device)
            sweep_tid = ttnn.begin_trace_capture(device, cq_id=0)
            with _trace_capture_guard():
                ids_s, conf_s, rows_s = drafter.draft(greedy=GREEDY)
            ttnn.end_trace_capture(device, sweep_tid, cq_id=0)
            times = []
            for _ in range(REPLAYS):
                ttnn.synchronize_device(device)
                a = time.perf_counter()
                ttnn.execute_trace(device, sweep_tid, cq_id=0, blocking=False)
                ttnn.synchronize_device(device)
                times.append(time.perf_counter() - a)
            same_ids = torch.equal(_to_torch(ids_s).flatten().long(), results[f][0])
            rdiff = float((_to_torch(rows_s).float().reshape(drafter.block, -1) - results[f][2]).abs().max())
            logger.info(
                f"[dspark sweep] experts_block {block}: replay {ms(times)}, ids same {same_ids}, rows diff {rdiff}"
            )
            ttnn.release_trace(device, sweep_tid)

        if GOLDEN:
            if not os.path.exists(GOLDEN):
                torch.save(results, GOLDEN)
                logger.info(f"golden saved to {GOLDEN}")
            else:
                golden = torch.load(GOLDEN)
                bad = []
                for f, (ids, conf, rows) in results.items():
                    g_ids, g_conf, g_rows = golden[f]
                    diff = float((rows - g_rows).abs().max())
                    cdiff = float((conf - g_conf).abs().max())
                    logger.info(
                        f"[dspark golden] frontier {f}: ids {'equal' if torch.equal(ids, g_ids) else 'DIFFER'} "
                        f"({ids.tolist()} vs {g_ids.tolist()}), rows max diff {diff}, conf max diff {cdiff}"
                    )
                    if not torch.equal(ids, g_ids):
                        bad.append(f"frontier {f}: ids {ids.tolist()} vs golden {g_ids.tolist()}")
                    if EXACT and not GREEDY and diff != 0.0:
                        bad.append(f"frontier {f}: rows differ by {diff}")
                assert not bad, "; ".join(bad)
