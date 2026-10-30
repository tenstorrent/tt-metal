# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""P.14 (PREPARED_ALL): the all-device model's chunk output with the fused tail shape-keyed (on) and address-keyed
(off), in one process, on the profile rung's last chunk (the golden prefix loaded before each run, as
testing/profile.py does). Not a gate: evidence that the switch does not change the model's numbers (bit-identical
final hidden state), and that the shape key survives an allocation shift (DS41-146) that makes the address-keyed tail
rebuild its programs.

    PYTHONPATH=$PWD DS41_ENGRAM_TABLE=resident scripts/run_safe_pytest.sh --run-all --no-precompile \\
        models/demos/deepseek_v41_flash_d_p/tt/perf_prepared_all.py -s

The side is switched on the block's TailA2A instance (its class: tt/prepared_tail.ShapeKeyedTailA2A or TailA2A). A
"shifted" run allocates a DRAM buffer in the middle of the forward (after the middle layer) and keeps it to the end,
so the later layers' per-call buffers land at other addresses (as P.4's lazily allocated CCL outputs did).
"""

from __future__ import annotations

import time

import pytest
import torch

import ttnn
from models.demos.common.bringup.core import metrics
from models.demos.common.bringup.reference.golden import Golden
from models.demos.common.bringup.testing import harness
from models.demos.common.bringup.testing.harness import begin_chunk, spec
from models.demos.common.bringup.testing.profile import profile_rung
from models.demos.deepseek_v41_flash_d_p.tt import moe_block_adapter
from models.demos.deepseek_v41_flash_d_p.tt.prepared_tail import ShapeKeyedTailA2A
from models.demos.moe_block.tt.tail_a2a import TailA2A


@pytest.mark.timeout(1500)  # (loading the 40-layer model takes ~5 min)
@pytest.mark.parametrize("device_params", [harness.device_params(spec())], indirect=True, ids=["box"])
@pytest.mark.parametrize("mesh_device", [tuple(spec().mesh)], indirect=True, ids=["x".join(map(str, spec().mesh))])
def test_prepared_all(mesh_device):
    s = spec()
    name = profile_rung(s)
    rung = s.rung(name)
    g = Golden.for_rung(s, name)
    layers = [i for i in s.layers() if i in g.layers]
    chunk = rung["chunk"]
    start = rung["seq"] - chunk
    model = s.hooks().device_model(mesh_device, s, layers, lm_head=False)
    state = model.new_state(rung["seq"])
    prefix = {i: g.state(i, at=start) for i in layers}
    tokens = g.tokens()[start:]
    starts = {i for k, i in enumerate(layers) if k == 0 or layers[k - 1] != i - 1}
    want = g.layer(g.n_chunks - 1, layers[-1])["out"].float()

    def tails():
        return [b.tail.a2a for _, b in moe_block_adapter._BLOCKS.values() if getattr(b, "tail", None) is not None]

    def run(side: str, shift: int = 0):
        for i in layers:
            state.load_prefix(i, prefix[i], start)
        for a in tails():
            a.__class__ = ShapeKeyedTailA2A if side == "on" else TailA2A
        m0 = sum(a.misses for a in tails())
        s0 = sum(getattr(a, "shape_misses", 0) for a in tails())
        pad, mid = None, layers[len(layers) // 2]
        h = None
        ttnn.synchronize_device(mesh_device)
        t0 = time.time()
        begin_chunk(model, tokens, start)
        for i in layers:
            if i in starts:
                if h is not None:
                    model.free(h)
                h = model.embed(tokens) if i == 0 else model.from_host(g.layer(g.n_chunks - 1, i)["in"].float())
            h2 = model.layer(i, h, start, state)
            model.free(h)
            h = h2
            if shift and i == mid:  # (kept to the end of the run: the later layers' allocations move)
                pad = ttnn.allocate_tensor_on_device(
                    ttnn.Shape([1, 1, 640 * shift, 1024]),
                    ttnn.bfloat16,
                    ttnn.TILE_LAYOUT,
                    mesh_device,
                    ttnn.DRAM_MEMORY_CONFIG,
                )
        model.sync()
        wall = (time.time() - t0) * 1e3
        out = model.to_host(h).float()
        model.free(h)
        if pad is not None:
            ttnn.deallocate(pad)
        misses = sum(a.misses for a in tails()) - m0
        smiss = sum(getattr(a, "shape_misses", 0) for a in tails()) - s0
        print(
            f"PREPARED_ALL {side} shift={shift}: wall {wall:.1f} ms, tail address-keyed builds {misses}, "
            f"shape-keyed builds {smiss}, pcc vs golden {metrics.pcc(out, want):.6f}",
            flush=True,
        )
        return out, wall

    plan = [("on", 0), ("off", 0)] * 2 + [("on", 1), ("on", 2), ("on", 3), ("off", 1), ("off", 2), ("off", 3)]
    plan += [("on", 0), ("off", 0)] * 3
    outs = [(side, sh, *run(side, sh)) for side, sh in plan]
    ref = outs[0][2]
    for side, sh, o, _ in outs:
        assert torch.equal(o, ref), f"PREPARED_ALL {side} shift={sh} differs from on (max |d| {(o - ref).abs().max()})"
    for side in ("on", "off"):
        ws = sorted(w for sd, sh, _, w in outs[-6:] if sd == side)
        print(f"PREPARED_ALL {side}: last-block walls {ws} median {ws[len(ws) // 2]:.1f} ms", flush=True)
    print(f"PREPARED_ALL fallbacks {sum(getattr(a, 'fallbacks', 0) for a in tails())}", flush=True)
    print("PREPARED_ALL every run bit-identical (on == off, shifted or not)", flush=True)
