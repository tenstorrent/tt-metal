# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""PB.9X diagnostic (not gated): which op builds a program for a glued chunk after warm-up. Glued chunks (lane-shaped,
misaligned, tiny / idle segments, six segments) at extreme histories (0, exact multiples of the chunk and of 4 chunks,
near the slot capacity, odd 64-multiples) in serving slots (tests/fresh_common.Slots: the contract's caches), each
chunk under the build probe (fresh_common.BuildProbe: the op, its call site and the argument that changed). Two
warm-up chunks first. GLUED_PROBE_ROUTE_LANES=1 also runs the lane headers (ChunkDescriptor) the walls use.

    <the walls env> scripts/run_safe_pytest.sh --run-all --no-precompile \\
        models/demos/deepseek_v41_flash_d_p/tests/packed/test_glued_build_probe.py
"""

from __future__ import annotations

import pytest

from models.demos.common.bringup.testing import batch
from models.demos.common.bringup.testing.harness import mesh_parametrize, spec
from models.demos.common.bringup.testing.profile import PositionGolden, cache_entries

S = spec()
BATCH_SHARED = True
BATCH_LAYERS = "executed"
CHUNK = 8192


def _chunks(cap: int) -> list:
    from models.demos.deepseek_v41_flash_d_p.tt.packed import ChunkDescriptor, GluedChunk, Segment

    top = cap - 2 * CHUNK
    out = []
    hist = [0, 8192, 32768, 65536, 131072, 262144, 524288, top // 64 * 64, 999936, 4160, 300096, 700032]
    for i, h in enumerate(hist):
        out.append(
            ("lane4x2k h=%d" % h, GluedChunk(CHUNK, tuple(Segment(k, h + 64 * k, h + 64 * k + 2048) for k in range(4))))
        )
        out.append(
            (
                "mis3 h=%d" % h,
                GluedChunk(
                    CHUNK, (Segment(4, h, h + 1600), Segment(5, h + 64, h + 64 + 3000), Segment(6, 64, 64 + 900))
                ),
            )
        )
    out.append(("tiny", GluedChunk(CHUNK, (Segment(7, 128, 129), Segment(8, 64, 64 + 700), Segment(9, 0, 1)))))
    out.append(("six", GluedChunk(CHUNK, tuple(Segment(10 + k, 64 * k * 37, 64 * k * 37 + 1300) for k in range(6)))))
    out.append(
        (
            "six long",
            GluedChunk(CHUNK, tuple(Segment(10 + k, top - 64 * 300 * k, top - 64 * 300 * k + 1344) for k in range(6))),
        )
    )
    out.append(("full", GluedChunk(CHUNK, (Segment(16, 139264, 139264 + CHUNK),))))
    out.append(
        (
            "lanes padded",
            ChunkDescriptor.packed(
                CHUNK,
                [
                    Segment(17 + k, 20000 * (k + 1) // 64 * 64, 20000 * (k + 1) // 64 * 64 + 1500 + 50 * k)
                    for k in range(4)
                ],
            ),
        )
    )
    out.append(
        (
            "lanes empty",
            ChunkDescriptor.packed(CHUNK, [Segment(21, 4096, 6144), None, Segment(22, 900032, 900032 + 777), None]),
        )
    )
    return out


@mesh_parametrize(timeout_s=3600)
def test_glued_build_probe(mesh_device, monkeypatch):
    if mesh_device is None:
        pytest.skip("device test")
    import ttnn
    from models.demos.deepseek_v41_flash_d_p.tests import fresh_common as F
    from models.demos.deepseek_v41_flash_d_p.tests.packed.packed_common import switch
    from models.demos.deepseek_v41_flash_d_p.tt.packed_device import ids_from_input, make_input, run_chunk

    full = list(S.layers())
    layers = batch.run_layers(S, full, last_chunk=False)
    deploy = int(S.get("deploy.seq") or S.get("target.seq"))
    cap = -(-deploy // CHUNK) * CHUNK
    monkeypatch.setitem(S.data["target"], "seq", max(cap, int(S.data["target"]["seq"])))
    model = batch.device_model(S, mesh_device, layers, lm_head=False)
    loaded = model.skip_mode
    if loaded != "decode":
        model.use_decoder_skip("decode")
    text = PositionGolden(S, cap, CHUNK).tokens()
    slots = F.Slots(model, mesh_device, cap, n=30, warm=6)
    try:
        with switch("PACKED", "on"), switch("PACKED_GLUED", "on"):

            def run(desc):
                segs = list(desc.segments)
                ids = [None if s is None else text[s.start : s.end].long() for s in segs]
                inp = make_input(mesh_device, desc, ids)
                sts = [None if s is None else slots.state(s.slot) for s in segs]
                nat = ids_from_input(mesh_device, inp, desc)
                run_chunk(model, desc, nat, sts, layers, last=False, inp=inp)
                model.sync()
                ttnn.deallocate(inp)

            from models.demos.deepseek_v41_flash_d_p.tt.packed_device import warm_glued_chunks

            for desc in warm_glued_chunks(CHUNK, list(range(6)), cap):  # serving's warm-up (adapter.compile's set)
                run(desc)
            import dataclasses

            def moved(desc, off=6):  # (the probe's requests in slots never warmed: 6..)
                segs = tuple(None if g is None else dataclasses.replace(g, slot=g.slot + off) for g in desc.segments)
                return dataclasses.replace(desc, segments=segs)

            for name, desc in _chunks(cap):
                desc = moved(desc)
                e0 = cache_entries(mesh_device)
                with F.BuildProbe(mesh_device) as pb:
                    run(desc)
                e1 = cache_entries(mesh_device)
                print(f"PROBE {name}: +{e1 - e0} cache entries", flush=True)
                pb.print(f"probe {name}")
    finally:
        slots.free()
        if loaded != "decode":
            model.use_decoder_skip(loaded)
