# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""tt-opt harness of the DECODE window: ONE speculative verify round of the k=3 runner of the adaptive set ( 4 users x (1 + k) rows, the shipped B=4 / 1M-context build) on a LAYER SUBSET.

The model is built exactly like the demo (``create_tt_model`` at the 1,048,576 context so the decode indexer pays its O(context) cost), the 4 GSM8K prompts are prefilled and the spec runner
is seeded from them (``SpecRunner.seed``). The window is the state after that seed: block X [4, 1 + k] at positions base.

    test_decode_tracy   trace OFF: one eager round (``SpecDecoder.forward``: verify + accept + commit + draft) between DECODE_START / DECODE_END signposts (profile command of tt-opt)
    test_decode_latency trace ON: the captured round trace replayed back to back; prints ``TRACE_STAGE_MS[decode]=<device ms per round>`` (latency command of tt-opt)
    test_decode_gate    correctness: the eager round with the per-layer stream capture on; PCC of every layer's stream / the full logits / accepted ids against a golden file written by the
                        unmodified tree (TT_OPT_GOLDEN=write) -- fails below TT_OPT_PCC (0.9995)

Env: TT_OPT_LAYER_IDS (default 0,1,2,3,14,20,21,24: window / ratio-4 + indexer / ratio-128 layers and both Engram layers, with their kv sources), TT_OPT_MAX_CTX (1048576),
TT_OPT_GOLDEN_DIR, TT_OPT_GOLDEN=write|check, TT_OPT_PCC, TT_OPT_REPS. The shipped environment (DSV41_SPEC, DSV41_POOL_DTYPE, DSV41_PF_PAGED, ...) comes from the spec.
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
from models.common.sampling import SamplingParams
from models.demos.blackhole.deepseek_v41_flash.tt.common import create_tt_model, default_page_params
from models.demos.blackhole.deepseek_v41_flash.tt.generator import Generator
from models.tt_transformers.demo.simple_text_demo import load_inputs
from models.tt_transformers.tt.common import PagedAttentionConfig, preprocess_inputs_prefill

LAYERS = [int(x) for x in os.environ.get("TT_OPT_LAYER_IDS", "0,1,2,3,14,20,21,24").split(",")]
TRACY_LAYERS = [int(x) for x in os.environ.get("TT_OPT_TRACY_LAYER_IDS", ",".join(map(str, LAYERS))).split(",")]
MAX_CTX = int(os.environ.get("TT_OPT_MAX_CTX", "1048576"))
GSM = "models/demos/blackhole/deepseek_v41_flash/demo/sample_prompts/input_data_gsm8k_128.json"
GOLDEN_DIR = os.environ.get("TT_OPT_GOLDEN_DIR", "/mnt/tt-data/ssinghal/dsv4-logs/ttopt/golden")
B = 4
KSEL = int(os.environ.get("TT_OPT_K", "3"))  # lever harness: the runner of the adaptive set to time (3 or 5)
BASE_POS = int(
    os.environ.get("TT_OPT_BASE_POS", "0")
)  # minimal setup / gate: the block positions (0: the short GSM8K-like positions, else BASE_POS + (0, 7, 13, 21): a long-context window with the decode indexer ON)
GATE_MINIMAL = os.environ.get("TT_OPT_GATE_MINIMAL", "0") == "1"  # gate without prefill / seed (needed at BASE_POS)

PARAMS = [
    pytest.param(
        {
            "l1_small_size": 16384,
            "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
            "trace_region_size": int(os.environ.get("DSV41_TRACE_REGION", "1200000000")),
        },
        id="ring",
    )
]


def _log(m):
    print(m, flush=True)


def _setup(md, minimal=False, tracy_layers=True):
    """-> (generator, spec runner, X [B, n], base [B], snapshot of the attention states at that point).
    minimal (Tracy): no prefill / seed / warm-up round (the device profiler records every program of 32 chips x all cores: the setup of a full run is tens of GB); the window is the FIRST
    eager round on fixed blocks (device kernel times do not depend on the host compile)."""
    pp = default_page_params(MAX_CTX, 1)
    args, m, _, _ = create_tt_model(
        md,
        B,
        MAX_CTX,
        PagedAttentionConfig(block_size=pp["page_block_size"], max_num_blocks=pp["page_max_num_blocks_per_dp"]),
        layer_ids=TRACY_LAYERS if (minimal and tracy_layers) else LAYERS,
        log=_log,
    )
    gen = Generator([m], [args], md, tokenizer=args.tokenizer)
    from models.demos.blackhole.deepseek_v41_flash.tt import spec_policy

    gen.spec_choice = spec_policy.resolve_apply(B, md.shape[0], m.max_ctx)
    assert gen.spec_choice.on, f"the harness needs speculative decoding: {gen.spec_choice.describe()}"
    if minimal:
        gen.enable_spec()
        spec = (
            gen.spec
        )  # adaptive set (DSV41_SPEC=<k> only switches speculation on): the window is the k=3 runner of the set
        runner = spec.runners[KSEL] if hasattr(spec, "runners") else spec
        X = torch.tensor(
            [[30744, 9237, 4, 362], [80365, 6306, 29, 270], [66117, 223, 21, 38], [37969, 7812, 33330, 14]]
        )
        base = torch.tensor([68, 30, 53, 38]) if not BASE_POS else BASE_POS + torch.tensor([0, 7, 13, 21])
        for b in range(B):  # pages of the users (a prefill admits them)
            m.pool.release(b)
            m.pool.admit(b, int(base[b]) + 64)
        m.pool.sync_page_table()
        return gen, runner, X, base, None
    prompts, _ = load_inputs(GSM, B, instruct=True)
    toks, _, pos, _ = preprocess_inputs_prefill(
        prompts[:B], args.tokenizer, [args], instruct=True, max_generated_tokens=64, max_prefill_len=MAX_CTX
    )
    toks = torch.stack(toks).view(B, -1)
    samp = SamplingParams(temperature=0, top_k=1, top_p=1.0)
    kw = dict(prompt_lens=pos, sampling_params=samp, enable_trace=True, chunk=512)
    gen.prefill_forward_text(toks, **kw)  # compile / trace capture
    first, _ = gen.prefill_forward_text(toks, **kw)
    first = first.view(-1)
    gen.enable_spec()
    spec = gen.spec  # adaptive set: the window is the k=3 runner of the set
    runner = spec.runners[KSEL] if hasattr(spec, "runners") else spec
    lens = torch.as_tensor(pos).long()
    X, base = runner.seed(toks, lens, first)
    ttnn.synchronize_device(md)
    snaps = runner.dec.snapshot_states()
    _log(f"HARNESS ready: layers {LAYERS} k={runner.k} n={runner.n} X {X.tolist()} base {base.tolist()}")
    return gen, runner, X, base, snaps


def _feed_eager(runner, X, base, force):
    runner._ensure(base)
    if runner.tr:  # (no traces captured in the minimal Tracy setup)
        runner._use(base)
    # the position bucket of the shipped build (positions < 512: dense, no indexer): the eager round runs the same views as the bucket trace that serves these positions
    runner.set_bucket(runner.bucket_for(int(base.max()) + runner.n + 1))
    runner.dec.set_force(force)
    runner._feed(X, base)


def _eager_round(runner, X, base, snaps, force=-1):
    """one eager round on the seeded state; the attention states are restored afterwards (the pool slots it wrote are rewritten by the next identical round)."""
    _feed_eager(runner, X, base, force)
    runner.dec.forward()
    ttnn.synchronize_device(runner.md)
    runner.dec.restore_states(snaps)


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize("device_params", PARAMS, indirect=True)
@pytest.mark.timeout(7200)
@torch.no_grad()
def test_decode_tracy(mesh_device):
    from tracy import signpost

    md = mesh_device
    _, runner, X, base, _ = _setup(md, minimal=True)
    _feed_eager(runner, X, base, -1)
    ttnn.synchronize_device(md)
    ttnn.ReadDeviceProfiler(md)
    signpost("DECODE_START")
    runner.dec.forward()
    ttnn.synchronize_device(md)
    signpost("DECODE_END")
    ttnn.ReadDeviceProfiler(md)
    _log(f"TT_OPT_WORKLOAD batch={B} k={runner.k} rows_per_user={runner.n} layers={TRACY_LAYERS}")


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize("device_params", PARAMS, indirect=True)
@pytest.mark.timeout(7200)
@torch.no_grad()
def test_decode_latency(mesh_device):
    md = mesh_device
    # no prefill / seed: the traced round does not depend on the cache contents, only on the (fixed) token blocks; compile pass + trace capture of the runner as the server does it
    gen, runner, X, base, _ = _setup(md, minimal=True, tracy_layers=False)
    Xw = X
    master = gen.spec.runners[3] if hasattr(gen.spec, "runners") else runner
    if master.siblings:  # adaptive set: every runner is prepared before the FIRST trace is captured
        Xw = torch.cat([X, X[:, :2]], dim=1)  # the k=5 runner of the set verifies 6 rows per user
        master._capture_all(lambda nn: Xw[:, :nn], base)
    else:
        runner.capture(X, base)
    X = Xw[:, : runner.n]
    snaps = runner.dec.snapshot_states()
    runner._ensure(base)
    runner._use(base)
    runner.dec.set_force(runner.n - 1)  # all drafts accepted: a fixed, repeatable round
    runner._feed(X, base)
    tid = runner.tid
    for _ in range(5):
        ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(md)
    reps = int(os.environ.get("TT_OPT_REPS", "30"))
    ms = []
    for _ in range(5):
        t = time.perf_counter()
        for _ in range(reps):
            ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(md)
        ms.append((time.perf_counter() - t) / reps * 1e3)
    wall = []
    for _ in range(5):  # a full round as the loop runs it (feed + replay + readback)
        t = time.perf_counter()
        runner._round(X, base, torch.full((B,), runner.n - 1))
        wall.append((time.perf_counter() - t) * 1e3)
    runner.dec.restore_states(snaps)
    _log(f"TT_OPT_ROUND_WALL_MS={statistics.median(wall):.3f} device_batches={[round(x, 3) for x in ms]}")
    print(f"TRACE_STAGE_MS[decode]={statistics.median(ms):.4f}", flush=True)


def _pcc(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    if a.numel() < 2 or float(a.std()) == 0.0 or float(b.std()) == 0.0:
        return 1.0 if torch.equal(a, b) else 0.0
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize("device_params", PARAMS, indirect=True)
@pytest.mark.timeout(7200)
@torch.no_grad()
def test_decode_gate(mesh_device):
    md = mesh_device
    _, runner, X, base, snaps = _setup(md, minimal=True, tracy_layers=False) if GATE_MINIMAL else _setup(md)
    dec = runner.dec
    cols = md.shape[1]
    dec.dbg_store = []
    _feed_eager(runner, X, base, -1)
    dec.forward()
    ttnn.synchronize_device(md)
    out = {"X": X.clone(), "base": base.clone(), "streams": {}}
    for tag, rows in dec.dbg_store:
        out["streams"][tag] = [r.clone() for r in rows]
    dec.dbg_store = None
    lg = dec.dbg_logits  # [1, 1, T, V / cols] per device: the vocabulary is sharded over the mesh columns
    out["logits"] = torch.cat(
        [
            torch.cat(
                [
                    ttnn.to_torch(ttnn.get_device_tensors(lg)[r * cols + c]).float().reshape(dec.U * dec.n, -1)
                    for c in range(cols)
                ],
                dim=1,
            )
            for r in range(md.shape[0])
        ]
    )
    pk = runner._readback(dec.pack)
    out["a"], out["m"], out["d"] = pk[0].clone(), pk[1].clone(), pk[2].clone()
    if snaps is not None:
        dec.restore_states(snaps)
    mode = os.environ.get("TT_OPT_GOLDEN", "check")
    path = os.path.join(GOLDEN_DIR, f"decode_k{runner.k}_L{'-'.join(map(str, LAYERS))}_P{BASE_POS}.pt")
    if mode == "write":
        os.makedirs(GOLDEN_DIR, exist_ok=True)
        torch.save(out, path)
        _log(f"GATE golden written: {path} ({len(out['streams'])} streams)")
        return
    gold = torch.load(path)
    thr = float(os.environ.get("TT_OPT_PCC", "0.9995"))
    worst, worst_tag, bad = 1.0, "", []
    for tag, rows in gold["streams"].items():
        assert tag in out["streams"], f"stream {tag} missing"
        for r, (g, c) in enumerate(zip(rows, out["streams"][tag])):
            p = _pcc(g, c)
            if p < worst:
                worst, worst_tag = p, f"{tag} row {r}"
            if p < thr:
                bad.append((tag, r, p))
    t2 = _pcc(gold["logits"], out["logits"])
    same_a = bool(torch.equal(gold["a"], out["a"]))
    same_d = bool(torch.equal(gold["d"], out["d"]))
    _log(
        f"GATE streams worst PCC {worst:.6f} ({worst_tag}) over {len(gold['streams'])} streams; logits PCC {t2:.6f}; "
        f"accepted ids equal {same_a}; drafts equal {same_d}; threshold {thr}"
    )
    print(
        f"GATE_SCORES pcc_streams_min={worst:.6f} pcc_logits={t2:.6f} ids_equal={int(same_a)} drafts_equal={int(same_d)}",
        flush=True,
    )
    assert not bad, f"streams below {thr}: {bad[:5]}"
    assert t2 >= thr, f"logits PCC {t2} < {thr}"
