# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""P.15 (HOST_ALLOC): the all-device model's chunk output with the load-time outputs used (on) and not (off), in one
process, on the profile rung's last chunk (the golden prefix loaded before each run, as testing/profile.py does). Not
a gate: evidence that the switch does not change the model's numbers (bit-identical final hidden state, every run).

    PYTHONPATH=$PWD DS41_ENGRAM_TABLE=resident DS41_HOST_ALLOC=on scripts/run_safe_pytest.sh --run-all \\
        --no-precompile models/demos/deepseek_v41_flash_d_p/tt/perf_host_alloc_identity.py -s

The model is built with HOST_ALLOC=on (its load-time buffers exist on both sides); "off" hides them for the run: the
gather sites' table (tt/ccl._SITES), the fused tail's output (tt/host_alloc._TAIL_OUT) and the exchange's kept arena
probe (the instance methods host_alloc.install_moe sets), so every call allocates as before.
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
from models.demos.deepseek_v41_flash_d_p.tt import ccl, host_alloc, moe_block_adapter


@pytest.mark.timeout(1500)  # (loading the 40-layer model takes ~5 min)
@pytest.mark.parametrize("device_params", [harness.device_params(spec())], indirect=True, ids=["box"])
@pytest.mark.parametrize("mesh_device", [tuple(spec().mesh)], indirect=True, ids=["x".join(map(str, spec().mesh))])
def test_host_alloc_identity(mesh_device):
    assert host_alloc.on(), "run with DS41_HOST_ALLOC=on (the off side is switched in-process)"
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
    sites, tail = dict(ccl._SITES), dict(host_alloc._TAIL_OUT)

    def exchanges():
        return [b.exchange.block for _, b in moe_block_adapter._BLOCKS.values() if b.exchange.block is not None]

    def side(on: bool):
        ccl._SITES.clear()
        host_alloc._TAIL_OUT.clear()
        if on:
            ccl._SITES.update(sites)
            host_alloc._TAIL_OUT.update(tail)
        for _, b in moe_block_adapter._BLOCKS.values():
            k3 = b.exchange.block
            for m in ("arena_fits", "alloc_arena"):
                k3.__dict__.pop(m, None)
            if on:
                host_alloc.install_moe(b)

    def run(on: bool):
        for i in layers:
            state.load_prefix(i, prefix[i], start)
        side(on)
        ccl.SITE_MISSES.clear()
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
        model.sync()
        wall = (time.time() - t0) * 1e3
        out = model.to_host(h).float()
        model.free(h)
        tag = "on" if on else "off"
        print(
            f"HOST_ALLOC {tag}: wall {wall:.1f} ms, site misses {dict(ccl.SITE_MISSES)}, "
            f"pcc vs golden {metrics.pcc(out, want):.6f}",
            flush=True,
        )
        return out, wall

    plan = [True, False] * 5
    outs = [(on, *run(on)) for on in plan]
    side(True)
    ref = outs[0][1]
    for on, o, _ in outs:
        assert torch.equal(
            o, ref
        ), f"HOST_ALLOC {'on' if on else 'off'} differs from on (max |d| {(o - ref).abs().max()})"
    for on in (True, False):
        ws = sorted(w for sd, _, w in outs[2:] if sd == on)
        print(f"HOST_ALLOC {'on' if on else 'off'}: walls {ws} median {ws[len(ws) // 2]:.1f} ms", flush=True)
    print("HOST_ALLOC every run bit-identical (on == off)", flush=True)
