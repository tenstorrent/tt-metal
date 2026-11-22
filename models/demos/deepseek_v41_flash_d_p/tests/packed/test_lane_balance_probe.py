# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""PB.3M diagnostic (not gated): where the mixed packed chunk's device time is bounded by its longest lane. The walls'
mixed case (61k / 139k / 308k / 547k, agentic text, slots at the one capacity, OV-62), warmed up; then, for each value
of PACKED_LANE_BALANCE (off, on; the same chunk, the same lanes): one real-time profiled run, the device ns of every
indexer-section op per mesh row (a lane = 2 SP rows), the slowest chip's section totals, and the slots' served-cache
digests (on must equal off byte for byte). Runs on the walls' load (layers 0-20, DECODER_SKIP=decode).
PB3M_PROBE_SECTIONS: the section-name prefixes to itemize (default indexer)."""

import os
from collections import defaultdict

import pytest

from models.demos.common.bringup.testing import batch, profiler
from models.demos.common.bringup.testing.harness import mesh_parametrize, spec
from models.demos.common.bringup.testing.profile import PositionGolden

S = spec()
BATCH_SHARED = True
BATCH_LAYERS = "executed"
BATCH_ITEM_VARS = ("PB3M_PROBE_SECTIONS", "PACKED_WALL_CAP")
CHUNK, LANE, SHIFT = 8192, 2048, 4096
POS = [61440, 139264, 307200, 546816]


@mesh_parametrize(timeout_s=3600)
def test_lane_balance_probe(mesh_device, monkeypatch):
    if mesh_device is None:
        pytest.skip("device test")
    from models.demos.deepseek_v41_flash_d_p.tests.packed.packed_common import packed, served_digests, switch
    from models.demos.deepseek_v41_flash_d_p.tests.packed.test_packed_walls import _cap
    from models.demos.deepseek_v41_flash_d_p.tt.packed import ChunkDescriptor, Segment
    from models.demos.deepseek_v41_flash_d_p.tt.packed_device import ids_from_input, make_input, run_chunk
    from models.demos.deepseek_v41_flash_d_p.tt.settings import settings

    assert int(S.data["target"]["chunk"]) == CHUNK, "run with BRINGUP_CHUNK=8192"
    assert settings.get("DECODER_SKIP") == "decode"
    layers = batch.run_layers(S, list(S.layers()), last_chunk=False)
    deploy = int(S.get("deploy.seq") or S.get("target.seq"))
    cap = _cap(deploy)
    monkeypatch.setitem(S.data["target"], "seq", max(cap, int(S.data["target"]["seq"])))
    model = batch.device_model(S, mesh_device, layers, lm_head=False)
    if model.skip_mode != "decode":
        model.use_decoder_skip("decode")
    top = max(POS)
    text = PositionGolden(S, top + 3 * SHIFT, CHUNK).tokens()
    states = {k: model.new_state(cap) for k in range(4)}
    toks = {k: text[k * SHIFT : k * SHIFT + POS[k] + LANE].long() for k in range(4)}
    it = [0]
    cols = int(mesh_device.shape[1])
    prefixes = tuple(os.environ.get("PB3M_PROBE_SECTIONS", "indexer").split(","))

    def run(prev: bool):
        lanes = [None] * 4
        for k in range(4):
            lanes[(k + it[0]) % 4] = k
        s = {k: POS[k] - (LANE if prev else 0) for k in lanes}
        desc = ChunkDescriptor.packed(CHUNK, [Segment(10 + k, s[k], s[k] + LANE) for k in lanes])
        inp = make_input(mesh_device, desc, [toks[k][s[k] : s[k] + LANE] for k in lanes])
        model.sync()
        nat = ids_from_input(mesh_device, inp, desc)
        run_chunk(model, desc, nat, [states[k] for k in lanes], layers, last=False)
        model.sync()
        import ttnn

        ttnn.deallocate(inp)
        return lanes

    has_switch = "PACKED_LANE_BALANCE" in settings.table
    sides = ("off", "on") if has_switch else ("off",)
    try:
        with packed("on"):
            for _ in range(3):
                for v in sides:
                    with switch("PACKED_LANE_BALANCE", v) if has_switch else _null():
                        run(True)
                        run(False)
                it[0] += 1
            digs = {}
            for v in sides:
                with switch("PACKED_LANE_BALANCE", v) if has_switch else _null():
                    run(True)
                    profiler.enable(mesh_device, ops=True, rt=True)
                    try:
                        profiler.set_layer(None)
                        lanes = run(False)
                        profiler.signpost("end")
                        profiler.finish_rt()
                        res = profiler.result()
                    finally:
                        profiler.disable()
                print(f"PB.3M probe LANE_BALANCE={v}: lanes (segment of lane) {lanes}, positions {POS}", flush=True)
                _report(res, prefixes, cols)
                digs[v] = {k: served_digests(states[k], layers, POS[k] + LANE) for k in range(4)}
            if len(sides) == 2:
                a, b = digs["off"], digs["on"]
                bad = [(k, key) for k in range(4) for key in a[k] if a[k][key] != b[k].get(key)]
                n = sum(len(a[k]) for k in range(4))
                print(f"PB.3M probe digests on vs off: {len(bad)} of {n} differ {bad[:8]}", flush=True)
                assert not bad, bad[:8]
    finally:
        for st in states.values():
            st.free()


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _report(res: dict, prefixes: tuple, cols: int) -> None:
    per_chip = defaultdict(float)
    for d in res["kernel_ns_dev"].values():
        for c, ns in d.items():
            per_chip[c] += ns
    secs = defaultdict(float)
    for sec, ns in res["kernel_ns"].items():
        secs[sec.split(".", 1)[-1]] += ns
    print(
        f"  device (slowest chip) {max(per_chip.values()) / 1e6:.2f} ms; sections (slowest chip each, summed): "
        + ", ".join(f"{k} {v / 1e6:.2f}" for k, v in sorted(secs.items(), key=lambda x: -x[1])[:14]),
        flush=True,
    )
    agg = defaultdict(lambda: [0, defaultdict(float)])
    for sec, rows in res.get("op_ns", {}).items():
        name = sec.split(".", 1)[-1]
        if not name.startswith(prefixes):
            continue
        for row in rows:
            nd = row.get("ns_dev") or {}
            a = agg[(sec, row.get("op"), str(row.get("shape", ""))[:50])]
            a[0] += row.get("calls", 1)
            for c, ns in nd.items():
                r = int(c) // cols
                a[1][r] = max(a[1][r], ns)
    print("  ops (section, op, calls: one call's max ns over the chips of each mesh row 0..7, ms):", flush=True)

    def skew(item):  # (the longest lane's mark: the spread over the mesh rows)
        v = sorted(item[1][1].values()) or [0]
        return -(v[-1] - v[len(v) // 2])

    for (sec, op, shape), (calls, rows) in sorted(agg.items(), key=skew)[:40]:
        if not rows or max(rows.values()) < 2e4:
            continue
        print(
            f"  {sec:22s} {op:40s} x{calls:3d} " + " ".join(f"{rows[r] / 1e6:6.3f}" for r in sorted(rows)),
            flush=True,
        )
