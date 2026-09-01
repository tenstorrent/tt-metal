# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Before/after performance for the fused decoder: warmed prefill + traced
warmed decode, functional (baseline) and fused, both layer kinds.

Run standalone for wall-clock numbers, or under Tracy for the ops CSV:

    python -m tracy -r -p -v -m pytest \
        models/autoports/zai_org_glm_4_7_flash/tests/test_fused_perf.py -q -s

Signpost pairs (one per measured window):
    PERF_{PREFILL,DECODE}_{MOE,DENSE}_{FUNC,FUSED} / ..._END

Wall-clock JSON evidence is written to
doc/fused_decoder/perf_wallclock_<impl>_<mode>_<kind>.json.
"""

import json
import time
from pathlib import Path

import pytest
import torch
from tracy import signpost

import ttnn
from models.autoports.zai_org_glm_4_7_flash.tests import utils
from models.autoports.zai_org_glm_4_7_flash.tests.test_functional_decoder import Harness
from models.autoports.zai_org_glm_4_7_flash.tt.functional_decoder import FunctionalDecoder
from models.autoports.zai_org_glm_4_7_flash.tt.fused_decoder import FusedDecoder

DOC_DIR = Path(__file__).resolve().parents[1] / "doc" / "fused_decoder"

PREFILL_S = 2048
DECODE_CONTEXT = 1024
DECODE_ITERS = 32

IMPLS = {"func": FunctionalDecoder, "fused": FusedDecoder}


@pytest.fixture(scope="module")
def device():
    dev = ttnn.open_device(device_id=0, l1_small_size=32768, trace_region_size=0)
    yield dev
    ttnn.close_device(dev)


@pytest.fixture(scope="module")
def cfg():
    return utils.hf_config()


def _write_json(name, payload):
    DOC_DIR.mkdir(parents=True, exist_ok=True)
    (DOC_DIR / name).write_text(json.dumps(payload, indent=1))
    print(f"wrote {DOC_DIR / name}: {payload}")


@pytest.mark.parametrize("impl", ["func", "fused"])
@pytest.mark.parametrize("kind", ["moe", "dense"])
def test_prefill_perf(device, cfg, kind, impl):
    h = Harness(device, cfg, kind, decoder_cls=IMPLS[impl])
    x = utils.synth_activations(cfg, h.layer_idx, PREFILL_S, seed=7)
    x_tt = ttnn.from_torch(x.unsqueeze(0), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    cache, pt, _ = h.fresh_cache(seed=3)

    # compile/warm run
    out = h.dec.prefill_forward(x_tt, kv_cache=cache, page_table=pt, user_id=0, seq_len=PREFILL_S)
    ttnn.deallocate(out)
    ttnn.synchronize_device(device)
    ttnn.ReadDeviceProfiler(device)  # flush profiler buffers between phases

    tag = f"PERF_PREFILL_{kind.upper()}_{impl.upper()}"
    signpost(tag)
    t0 = time.perf_counter()
    out = h.dec.prefill_forward(x_tt, kv_cache=cache, page_table=pt, user_id=0, seq_len=PREFILL_S)
    ttnn.synchronize_device(device)
    t1 = time.perf_counter()
    signpost(f"{tag}_END")
    ttnn.ReadDeviceProfiler(device)
    ttnn.deallocate(out)
    ttnn.deallocate(cache)

    wall = t1 - t0
    _write_json(
        f"perf_wallclock_{impl}_prefill_{kind}.json",
        {
            "impl": impl,
            "kind": kind,
            "mode": "prefill (warmed, non-traced)",
            "seq_len": PREFILL_S,
            "prefill_chunk_size": h.dec.prefill_chunk_size,
            "wall_s": round(wall, 4),
            "tokens_per_s": round(PREFILL_S / wall, 1),
            "signposts": [tag, f"{tag}_END"],
        },
    )


@pytest.mark.parametrize("impl", ["func", "fused"])
@pytest.mark.parametrize("kind", ["moe", "dense"])
def test_decode_perf_traced(device, cfg, kind, impl):
    h = Harness(device, cfg, kind, decoder_cls=IMPLS[impl])
    S = DECODE_CONTEXT - 1
    x = utils.synth_activations(cfg, h.layer_idx, S + 2, seed=7)
    cache, pt, _ = h.fresh_cache(seed=3)
    h.prefill(x, cache, pt, seq_len=S)

    pos = S  # decode at context depth DECODE_CONTEXT
    x_dev = ttnn.from_torch(
        x[:, pos : pos + 1].unsqueeze(0).permute(0, 2, 1, 3),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    pos_dev = ttnn.from_torch(torch.tensor([pos], dtype=torch.int32), device=device)
    rot_dev = ttnn.from_torch(torch.tensor([[pos]], dtype=torch.uint32), device=device)

    out_c = h.dec.decode_forward(x_dev, kv_cache=cache, page_table=pt, cur_pos_tensor=pos_dev, rot_idxs=rot_dev)
    ttnn.deallocate(out_c)
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    out_t = h.dec.decode_forward(x_dev, kv_cache=cache, page_table=pt, cur_pos_tensor=pos_dev, rot_idxs=rot_dev)
    ttnn.end_trace_capture(device, tid, cq_id=0)
    for _ in range(3):  # warm replays
        ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(device)
    ttnn.ReadDeviceProfiler(device)  # flush profiler buffers before the window

    tag = f"PERF_DECODE_{kind.upper()}_{impl.upper()}"
    signpost(tag)
    t0 = time.perf_counter()
    for _ in range(DECODE_ITERS):
        ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(device)
    t1 = time.perf_counter()
    signpost(f"{tag}_END")

    ttnn.ReadDeviceProfiler(device)
    assert not torch.isnan(ttnn.to_torch(out_t)).any()
    ttnn.release_trace(device, tid)
    ttnn.deallocate(cache)

    wall = t1 - t0
    _write_json(
        f"perf_wallclock_{impl}_decode_{kind}.json",
        {
            "impl": impl,
            "kind": kind,
            "mode": f"decode (traced, warmed, batch 1, fixed position {pos})",
            "context_depth": DECODE_CONTEXT,
            "iterations": DECODE_ITERS,
            "wall_s": round(wall, 4),
            "ms_per_token": round(wall / DECODE_ITERS * 1000, 3),
            "tokens_per_s": round(DECODE_ITERS / wall, 1),
            "signposts": [tag, f"{tag}_END"],
        },
    )
