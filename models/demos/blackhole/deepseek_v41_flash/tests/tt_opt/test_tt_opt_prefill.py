# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""tt-opt harness of the PREFILL window: ONE chunk of the single chunk trace (DSV41_PF_SINGLE=1, 512 tokens x 4 users = 4 mesh rows x 1 slot, S_pad = TT_OPT_WMAX = 1,048,576) on a LAYER SUBSET,
built exactly like tests/test_chunk_probe.py (``create_tt_model`` + ``Model.prefill_interleaved``).

    test_prefill_tracy   trace OFF: the prompt TT_OPT_PF_E tokens (default 32768) is prefilled through the trace, then ONE eager chunk at the last position (the traced chunk is position-independent:
                         ``begin_chunk(s0)`` writes the device-side positions / valid lengths, ``forward_device(bufs, C, 0, C, dyn=True)`` is the body that was captured) between PREFILL_START /
                         PREFILL_END signposts (profile command of tt-opt)
    test_prefill_latency trace ON: ``PROBE``-style replay mean over a prefill of the needle prompt TT_OPT_PF_LAT_IDS (65000 tokens); prints ``TRACE_STAGE_MS[prefill]=<ms per chunk replay>``
    test_prefill_gate    correctness: last-token logits of the 4 users for the 8031-token needle (TT_OPT_PF_GATE_IDS) and the 65000-token needle against a golden written by the unmodified tree
                         (TT_OPT_GOLDEN=write); fails below TT_OPT_PCC (0.9995) or when the top-1 token changes

Env: TT_OPT_LAYER_IDS (default 0,1,2,3,14,20,21,24), TT_OPT_WMAX, TT_OPT_PF_E, TT_OPT_GOLDEN_DIR, TT_OPT_GOLDEN, TT_OPT_PCC. The shipped environment (DSV41_PF_SINGLE=1, DSV41_PF_PAGED=read, DSV41_POOL_DTYPE ...)
comes from the spec. Host-side overheads of the prefill (export, snapshots, async replay) are NOT in any window.
"""

import os
import statistics
import sys
import time

import pytest
import torch

# the framework checkout this file lives in comes FIRST on the path: the tool's hook run starts a python that finds `models` through the editable install of ANOTHER checkout otherwise
_FW_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), *([".."] * 6)))
if _FW_ROOT not in sys.path:
    sys.path.insert(0, _FW_ROOT)

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt.common import create_tt_model, default_page_params
from models.tt_transformers.tt.common import PagedAttentionConfig

LAYERS = [int(x) for x in os.environ.get("TT_OPT_LAYER_IDS", "0,1,2,3,14,20,21,24").split(",")]
TRACY_LAYERS = [int(x) for x in os.environ.get("TT_OPT_TRACY_LAYER_IDS", ",".join(map(str, LAYERS))).split(",")]
WMAX = int(os.environ.get("TT_OPT_WMAX", "1048576"))
CHUNK = 512
B = 4
E_PROF = int(os.environ.get("TT_OPT_PF_E", "32768"))
IDS_LAT = os.environ.get("TT_OPT_PF_LAT_IDS", "/mnt/tt-data/ssinghal/dsv4-logs/ttopt/needle_65000.pt")
IDS_GATE = os.environ.get("TT_OPT_PF_GATE_IDS", "/mnt/tt-data/ssinghal/tmp/needle_8000.pt")
GOLDEN_DIR = os.environ.get("TT_OPT_GOLDEN_DIR", "/mnt/tt-data/ssinghal/dsv4-logs/ttopt/golden")

PARAMS = [
    pytest.param(
        {
            "l1_small_size": 16384,
            "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
            "trace_region_size": int(os.environ.get("DSV41_TRACE_REGION", "650000000")),
        },
        id="ring",
    )
]


def _log(m):
    print(m, flush=True)


def _setup(md, layers=None):
    pp = default_page_params(WMAX, 1)
    args, m, _, _ = create_tt_model(
        md,
        B,
        WMAX,
        PagedAttentionConfig(block_size=pp["page_block_size"], max_num_blocks=pp["page_max_num_blocks_per_dp"]),
        layer_ids=layers or LAYERS,
        log=_log,
    )
    assert m.Up == 1 and m.U == 1, (m.Up, m.U)
    return m


def _prefill(m, ids, E, want_logits):
    toks = ids.long().unsqueeze(0).repeat(B, 1)
    items = [(b, toks[b], 0, E) for b in range(B)]
    return m.prefill_interleaved(items, CHUNK, s_pad_max=WMAX, want_logits=want_logits)


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize("device_params", PARAMS, indirect=True)
@pytest.mark.timeout(14400)
@torch.no_grad()
def test_prefill_tracy(mesh_device):
    from tracy import signpost

    md = mesh_device
    m = _setup(md, TRACY_LAYERS)
    pm = m.prefill_model
    ids = torch.load(IDS_LAT)
    C = CHUNK
    s0 = E_PROF // C * C - C
    # Minimal setup (the device profiler records every program of 32 chips x all cores, ~12 MB per program; the host trace needs a device row for every op): the window is the EAGER COMPILE PASS of the
    # single chunk trace (the body that is captured), run at the chunk start s0 instead of 0 (device kernel times do not depend on the pool contents; the pages of E_PROF tokens are admitted so every
    # read is a valid address). No trace is captured / replayed.
    for b in range(B):
        m.pool.release(b)
        m.pool.admit(b, E_PROF + 1)
    m.pool.sync_page_table()
    orig_bc, orig_fd = pm.begin_chunk, pm.forward_device
    pm.begin_chunk = lambda s0_, C_: orig_bc(s0, C_)

    def fd(*a, **k):
        ttnn.synchronize_device(md)
        ttnn.ReadDeviceProfiler(md)
        signpost("PREFILL_START")
        out = orig_fd(*a, **k)
        ttnn.synchronize_device(md)
        signpost("PREFILL_END")
        ttnn.ReadDeviceProfiler(md)
        return out

    pm.forward_device = fd

    class _StopAfterCompile(Exception):
        pass

    def _stop(C_):
        raise _StopAfterCompile

    pm.capture_dyn_trace = _stop
    try:
        _prefill(m, ids, C, False)
    except _StopAfterCompile:
        pass
    ttnn.synchronize_device(md)
    _log(f"TT_OPT_WORKLOAD batch={B} chunk={C} s0={s0} S_pad={pm.dyn.S_pad} layers={TRACY_LAYERS}")


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize("device_params", PARAMS, indirect=True)
@pytest.mark.timeout(14400)
@torch.no_grad()
def test_prefill_latency(mesh_device):
    md = mesh_device
    m = _setup(md)
    pm = m.prefill_model
    ids = torch.load(IDS_LAT)
    E = int(ids.numel())
    rec = []
    orig_rw = pm.replay_window

    def rw(tokens, hashes, s0, C, active, vis=None, **kw):
        th, td = orig_rw(tokens, hashes, s0, C, active, vis=vis, **kw)
        rec.append((s0, th, td))
        return th, td

    pm.replay_window = rw
    means = []
    for rep in range(int(os.environ.get("TT_OPT_REPS", "2"))):
        rec.clear()
        t0 = time.perf_counter()
        _prefill(m, ids, E, False)
        wall = time.perf_counter() - t0
        td = [r[2] for r in rec]
        means.append(sum(td) / len(td) * 1e3)
        _log(
            f"PROBE rep={rep} E={E} replays={len(rec)} wall={wall:.2f}s replay_mean={means[-1]:.2f}ms host_mean={sum(r[1] for r in rec) / len(rec) * 1e3:.1f}ms"
        )
    print(f"TRACE_STAGE_MS[prefill]={statistics.median(means[1:] or means):.4f}", flush=True)


def _pcc(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    if float(a.std()) == 0.0 or float(b.std()) == 0.0:
        return 1.0 if torch.equal(a, b) else 0.0
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize("device_params", PARAMS, indirect=True)
@pytest.mark.timeout(14400)
@torch.no_grad()
def test_prefill_gate(mesh_device):
    md = mesh_device
    m = _setup(md)
    out = {}
    for name, path in (("needle8k", IDS_GATE), ("needle65k", IDS_LAT)):
        ids = torch.load(path)
        res = _prefill(m, ids, int(ids.numel()), True)
        out[name] = {b: (int(res[b][0]), res[b][1].float().clone()) for b in range(B)}
    mode = os.environ.get("TT_OPT_GOLDEN", "check")
    path = os.path.join(GOLDEN_DIR, f"prefill_L{'-'.join(map(str, LAYERS))}.pt")
    if mode == "write":
        os.makedirs(GOLDEN_DIR, exist_ok=True)
        torch.save(out, path)
        _log(f"GATE golden written: {path}")
        return
    gold = torch.load(path)
    thr = float(os.environ.get("TT_OPT_PCC", "0.9995"))
    worst, bad, bit = 1.0, [], True
    for name in gold:
        for b in range(B):
            gt, gl = gold[name][b]
            ct, cl = out[name][b]
            p = _pcc(gl, cl)
            worst = min(worst, p)
            bit = bit and bool(torch.equal(gl, cl))
            if p < thr or gt != ct:
                bad.append((name, b, p, gt, ct))
    _log(f"GATE prefill logits worst PCC {worst:.6f} bit_equal {bit} threshold {thr}")
    print(f"GATE_SCORES pcc_logits_min={worst:.6f} bit_equal={int(bit)}", flush=True)
    assert not bad, f"logits / top-1 changed: {bad[:5]}"
