# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""PB.3A diagnostic (not gated): where the host time of a packed chunk goes, PACKED_ATTN_ONECALL as set. The walls'
4 x 2k @ 139264 chunk (test_packed_walls' construction: agentic text, each lane a session stretch, slots rotated a
lane every chunk, the real previous chunk first), warmed up, then: the wall (synced), the host enqueue time of
run_chunk alone (no sync inside), a cProfile of one chunk (the functions with the most own / cumulative time) and the
device time of the ops in the attention sections (real-time profiler, op rows). PB3A_PROBE_POS: the lanes' positions
(default 4 x 139264; e.g. 61440,139264,307200,546816 = the walls' mixed case, each slot sized to its position).
Runs on the walls' load (layers 0-20, DECODER_SKIP=decode)."""

import cProfile
import io
import pstats
import time

import pytest

from models.demos.common.bringup.testing import batch
from models.demos.common.bringup.testing.harness import mesh_parametrize, spec
from models.demos.common.bringup.testing.profile import PositionGolden

S = spec()
BATCH_SHARED = True
BATCH_LAYERS = "executed"
BATCH_ITEM_VARS = ("PB3A_PROBE_POS",)
CHUNK, LANE, SHIFT = 8192, 2048, 4096


@mesh_parametrize(timeout_s=3600)
def test_onecall_host(mesh_device, monkeypatch):
    if mesh_device is None:
        pytest.skip("device test")
    import os

    from models.demos.deepseek_v41_flash_d_p.tests.packed.packed_common import packed
    from models.demos.deepseek_v41_flash_d_p.tt.packed import ChunkDescriptor, Segment
    from models.demos.deepseek_v41_flash_d_p.tt.packed_device import ids_from_input, make_input, run_chunk
    from models.demos.deepseek_v41_flash_d_p.tt.settings import settings

    assert settings.get("DECODER_SKIP") == "decode"
    pos = [int(v) for v in os.environ.get("PB3A_PROBE_POS", "139264,139264,139264,139264").split(",")]
    top = max(pos)
    layers = batch.run_layers(S, list(S.layers()), last_chunk=False)
    monkeypatch.setitem(S.data["target"], "seq", max(top + CHUNK, int(S.data["target"]["seq"])))
    model = batch.device_model(S, mesh_device, layers, lm_head=False)
    if model.skip_mode != "decode":
        model.use_decoder_skip("decode")
    text = PositionGolden(S, top + 3 * SHIFT, CHUNK).tokens()
    states = {k: model.new_state(pos[k] + CHUNK) for k in range(4)}
    toks = {k: text[k * SHIFT : k * SHIFT + pos[k] + LANE].long() for k in range(4)}
    it = [0]

    def run(prev: bool, sync: bool = True):
        lanes = [None] * 4
        for k in range(4):
            lanes[(k + it[0]) % 4] = k
        s = {k: pos[k] - (LANE if prev else 0) for k in lanes}
        desc = ChunkDescriptor.packed(CHUNK, [Segment(10 + k, s[k], s[k] + LANE) for k in lanes])
        inp = make_input(mesh_device, desc, [toks[k][s[k] : s[k] + LANE] for k in lanes])
        model.sync()
        t0 = time.time()
        nat = ids_from_input(mesh_device, inp, desc)
        run_chunk(model, desc, nat, [states[k] for k in lanes], layers, last=False)
        t1 = time.time()
        if sync:
            model.sync()
        t2 = time.time()
        import ttnn

        ttnn.deallocate(inp)
        return (t1 - t0) * 1e3, (t2 - t0) * 1e3

    try:
        with packed("on"):
            for _ in range(4):
                run(True)
                run(False)
                it[0] += 1
            enq, walls = [], []
            for _ in range(3):
                run(True)
                e, w = run(False)
                enq.append(round(e, 1))
                walls.append(round(w, 1))
                it[0] += 1
            print(
                f"PB.3A host probe ONECALL={settings.get('PACKED_ATTN_ONECALL')}: walls {walls} ms, "
                f"run_chunk enqueue (host, no sync) {enq} ms",
                flush=True,
            )
            run(True)
            pr = cProfile.Profile()
            pr.enable()
            run(False)
            pr.disable()
            for key in ("tottime", "cumulative"):
                buf = io.StringIO()
                pstats.Stats(pr, stream=buf).sort_stats(key).print_stats(45)
                print(f"PB.3A host probe cProfile by {key}:\n" + buf.getvalue(), flush=True)
            # the attention sections' ops (device ns, the slowest chip), summed over the layers
            from collections import defaultdict

            from models.demos.common.bringup.testing import profiler

            run(True)
            profiler.enable(mesh_device, ops=True, rt=True)
            try:
                profiler.set_layer(None)
                run(False)
                profiler.signpost("end")
                profiler.finish_rt()
                res = profiler.result()
            finally:
                profiler.disable()
            agg = defaultdict(lambda: [0, 0.0])
            for sec, rows in res.get("op_ns", {}).items():
                name = sec.split(".", 1)[-1]
                if not name.startswith(("attn_", "indexer", "attention", "compressor")):
                    continue
                for row in rows:
                    nd = row.get("ns_dev")
                    ns = max(nd.values()) if isinstance(nd, dict) and nd else row.get("ns", 0)
                    a = agg[(name, row.get("op"), str(row.get("shape", ""))[:60])]
                    a[0] += row.get("calls", 1)
                    a[1] += ns / 1e6
            print(f"PB.3A probe {pos}: attention-section ops (calls, ms, slowest chip), top 40:", flush=True)
            for (name, op, shape), (calls, ms) in sorted(agg.items(), key=lambda x: -x[1][1])[:40]:
                print(f"  {name:12s} {ms:8.3f} ms {calls:4d} x {op} {shape}", flush=True)
    finally:
        for st in states.values():
            st.free()
