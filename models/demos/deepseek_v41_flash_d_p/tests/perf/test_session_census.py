# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""P.85 (owner 2026-11-19): the routed-expert load of whole sessions, each a REAL prefill from position 0 through its
end (chunk by chunk, real history: compressed KV, index keys, windows, engram tail), the routing read after every
chunk. Diagnostic only, no model change; under DECODER_SKIP=decode (layers 0-20 loaded, every chunk not the last).

CENSUS_SESSIONS = a JSON file: [{"text": <reference/texts name>, "start": <token offset in the text>, "n": <tokens>,
"id": <session id>, ...}, ...]. Each session's tokens are text[start : start + n], prefilled at positions 0, C, 2C, ...
(full chunks only: the last partial chunk is dropped). Per (session, chunk, MoE layer): per chip the local experts'
routed rows and row tiles, the busiest chip's tiles and DS41-299's fit (205 us + 6.77 us x tiles). One record per
session: <art>/census_sessions/<text>_s<start>.json; metrics: census_session_chunks, census_session_fit_ms (mean over
every chunk), census_session_wall_s.

    CENSUS_SESSIONS=/path/sessions.json DS41_DECODER_SKIP=decode PYTHONPATH=$PWD python -m \\
        models.demos.common.bringup.resident submit --spec $BRINGUP_SPEC --cmd "... test_session_census.py"
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from models.demos.common.bringup.core import metrics
from models.demos.common.bringup.testing import batch
from models.demos.common.bringup.testing.harness import mesh_parametrize, spec
from models.demos.deepseek_v41_flash_d_p.tt.settings import settings

S = spec()
BATCH_SHARED = True  # FIX.29
BATCH_ITEM_VARS = ("CENSUS_SESSIONS",)  # F279: per-run, not the model's configuration
BATCH_LAYERS = "executed"  # F285: a performance measurement: the layers it executes, in the resident runner
FIT_US = (205.0, 6.77)  # DS41-299


@mesh_parametrize(timeout_s=7200)
def test_session_census(mesh_device, monkeypatch):
    if mesh_device is None:
        pytest.skip("device test")
    import torch

    import ttnn
    from models.demos.common.bringup.reference import texts

    assert settings.get("DECODER_SKIP") == "decode", "run with DS41_DECODER_SKIP=decode (the measurement's mode)"
    if os.environ.get(settings.env_name("ENGRAM_TABLE")) is None:
        monkeypatch.setenv(settings.env_name("ENGRAM_TABLE"), str(settings.table["ENGRAM_TABLE"].default))
    chunk = int(S.get("target.chunk"))
    sessions = json.loads(Path(os.environ["CENSUS_SESSIONS"]).read_text())
    longest = max(s["n"] // chunk * chunk for s in sessions)
    monkeypatch.setitem(S.data["target"], "seq", max(longest, int(S.data["target"]["seq"])))
    layers = batch.run_layers(S, list(S.layers()), last_chunk=False)  # F282: decode: layers 0-20
    model = batch.device_model(S, mesh_device, layers, lm_head=False)
    loaded = model.skip_mode
    if loaded != "decode":
        model.use_decoder_skip("decode")

    mb = ttnn.bringup.moe_block
    real = mb.routed_experts_combine
    n_chips = len(mesh_device.get_device_ids())
    cap = {"layer": None, "calls": []}

    def wrapped(*args, **kw):
        out = real(*args, **kw)
        comp = ttnn.ConcatMeshToTensor(mesh_device, dim=0)
        counts = ttnn.to_torch(args[1], mesh_composer=comp).reshape(n_chips, -1).to(torch.int64)
        ids = ttnn.to_torch(args[6], mesh_composer=comp).reshape(n_chips, -1).to(torch.int64)
        cap["calls"].append((cap["layer"], counts, ids))
        return out

    monkeypatch.setattr(mb, "routed_experts_combine", wrapped)
    cache: dict = {}
    out_dir = Path(S.art) / "census_sessions"
    out_dir.mkdir(parents=True, exist_ok=True)
    fits, n_chunks, t0 = [], 0, time.time()
    try:
        for sess in sessions:
            if sess["text"] not in cache:
                cache[sess["text"]] = torch.tensor(texts.build(S, sess["text"])["token_ids"], dtype=torch.int64)
            ids = cache[sess["text"]][sess["start"] : sess["start"] + sess["n"]]
            m = len(ids) // chunk
            if m == 0:
                continue
            state = model.new_state(m * chunk)
            rec = {**sess, "chunk": chunk, "n_chips": n_chips, "fit_us": FIT_US, "chunks": []}
            ts = time.time()
            try:
                for j in range(m):
                    start = j * chunk
                    tokens = ids[start : start + chunk]
                    cap["calls"] = []
                    model.begin_chunk(tokens, start, last=False)
                    h = model.embed(tokens)
                    for i in layers:
                        cap["layer"] = i
                        h2 = model.layer(i, h, start, state)
                        model.free(h)
                        h = h2
                    model.free(h)
                    model.sync()
                    ch = {"start": start, "layers": []}
                    fit = 0.0
                    for layer, counts, eids in cap["calls"]:
                        chips = []
                        for c in range(n_chips):
                            loc = [int(counts[c, e]) for e in eids[c].tolist()]
                            chips.append({"experts": [int(e) for e in eids[c].tolist()], "counts": loc})
                        tiles_max = max(sum(-(-v // 32) for v in x["counts"]) for x in chips)
                        fit += FIT_US[0] + FIT_US[1] * tiles_max
                        ch["layers"].append({"layer": layer, "tiles_max": tiles_max, "chips": chips})
                    rec["chunks"].append(ch)
                    fits.append(fit / 1e3)
            finally:
                state.free()
            n_chunks += m
            rec["wall_s"] = round(time.time() - ts, 2)
            p = out_dir / f"{sess['text']}_s{sess['start']}.json"
            tmp = p.with_suffix(f".{os.getpid()}.tmp")
            tmp.write_text(json.dumps(rec))
            os.replace(tmp, p)
            print(
                f"SESSION {sess['text']} @{sess['start']} ({sess.get('id', '')}): {m} chunks in {rec['wall_s']} s, "
                f"fit {sum(fits[-m:]) / m:.2f} ms a chunk -> {p}",
                flush=True,
            )
    finally:
        if loaded != "decode":
            model.use_decoder_skip(loaded)
    metrics.record("census_session_chunks", n_chunks)
    metrics.record("census_session_fit_ms", round(sum(fits) / max(len(fits), 1), 2))
    metrics.record("census_session_wall_s", round(time.time() - t0, 1))
    print(f"SESSIONS: {len(sessions)} sessions, {n_chunks} chunks, {time.time() - t0:.0f} s")
