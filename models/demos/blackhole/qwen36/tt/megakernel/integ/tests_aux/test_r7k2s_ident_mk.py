# Pod-local identity harness for r7 K2 + K2s (NOT committed; copied into the tree's tests/ for one device session and
# deleted after). Based on the K2 session's test_k2_ident.py, plus a sampled mode.
# 8 users with real-text prompts (test_spec_continuous PROMPT_LENS 130..600) decode R7_IDENT_TOKENS tokens each through the
# PRODUCTION spec loop: deferred fused steps, depth-1 chained (vLLM async order), the chain + verify (+ topk) + accept +
# drafter traces of every step. Chain audit and fused audit (greedy S1 and sampled S2: device commit vs the host reference)
# on EVERY step; no program may compile after capture.
#   R7_IDENT_MODE=greedy  : every step greedy (S1 accept trace).
#   R7_IDENT_MODE=sampled : every step sampled, temperature 0.6 / top_k 20 / top_p 0.95 on every row, uniforms from a seeded
#                           torch Generator (SampledDriver._spec_sampling: the same draws in every run as long as the
#                           streams agree), so OFF vs ON compares the S2 accept op graph with the K2s kernel draw for draw.
# Writes per-user token md5s, the md5 of all streams, the acceptance line (committed tokens per live row-step), the kernels
# attached and the audit / boundary / overflow counters to R7_IDENT_OUT (one JSON line per run). K2 / K2s flags are read
# from the environment (QWEN36_SPEC_CHAIN_KERNEL, QWEN36_SPEC_ACCEPT_KERNEL, QWEN36_SPEC_ACCEPT_KERNEL_SAMPLED).
import hashlib
import json
import os

import torch
from loguru import logger

from models.demos.blackhole.qwen36.tests import test_spec_continuous as tc
from models.demos.blackhole.qwen36.tests.test_spec_async_engine import AsyncDriver
from models.demos.blackhole.qwen36.tests.test_spec_chain_engine import _env, _marks, _setup, _streams
from models.demos.blackhole.qwen36.tests.test_spec_fused_engine import _admit_all

NEW = int(os.environ.get("R7_IDENT_TOKENS", "400"))
OUT = os.environ.get("R7_IDENT_OUT", "/tmp/r7k2s/ident.jsonl")
MODE = os.environ.get("R7_IDENT_MODE", "greedy")
TAG = os.environ.get("R7_IDENT_TAG", "")
SEED = int(os.environ.get("R7_IDENT_SEED", "1"))  # aclab: sampling-uniform seed (default 1 = original harness)
FLAGS = ("QWEN36_SPEC_CHAIN_KERNEL", "QWEN36_SPEC_ACCEPT_KERNEL", "QWEN36_SPEC_ACCEPT_KERNEL_SAMPLED")


def _md5(xs):
    return hashlib.md5(torch.tensor(xs, dtype=torch.int64).numpy().tobytes()).hexdigest()[:16]


def _kname(dev):
    return type(getattr(dev, "_kernel", None)).__name__ if dev is not None else "absent"


@_marks
def test_r7k2s_ident(mesh_device, monkeypatch):
    assert MODE in ("greedy", "sampled"), MODE
    _env(monkeypatch)  # FUSED=1 CHAIN=1, chain audit every step, B=8, 48-block tables
    monkeypatch.setenv("QWEN36_SPEC_CHAIN_SAMPLED", "1")
    monkeypatch.setenv("QWEN36_SPEC_FUSED_AUDIT", "1")
    monkeypatch.setenv("QWEN36_SPEC_FUSED_AUDIT_EVERY", "1")
    flags = {k: os.environ.get(k, "0") for k in FLAGS}
    model, eng, tokenizer, prompts = _setup(mesh_device)
    assert eng._fs_trace is not None, "sampled fused accept trace missing"
    kern = {
        "chain_g": _kname(getattr(eng, "_ch_g", None)),
        "chain_s": _kname(getattr(eng, "_ch_s", None)),
        "accept": _kname(getattr(eng, "_fa", None)),
        "accept_sampled": _kname(getattr(eng, "_fs", None)),
    }
    n_prog = mesh_device.num_program_cache_entries()
    mesh_device.set_program_cache_misses_allowed(False)
    sampled = MODE == "sampled"
    try:
        drv = AsyncDriver(eng, seed=SEED, sampled=(lambda s: True) if sampled else (lambda s: False))
        _admit_all(drv, prompts, 8, NEW)
        drv.pipeline(8 * NEW)
        streams = _streams(drv)
        assert mesh_device.num_program_cache_entries() == n_prog, "a program compiled after capture"
    finally:
        mesh_device.set_program_cache_misses_allowed(True)
    # QWEN36_R16_MEGAKERNEL=stage0: prove the probe ran inside every verify replay (device counter == eager + replays)
    mk = getattr(model, "_vfy_mk", None)
    mk_report = {str(b): r for b, r in mk.report().items()} if mk is not None else None
    if mk_report is not None:
        logger.info(f"[mk-stage0] {mk_report}")
        assert any(r["replays"] > 0 for r in mk_report.values()), f"no verify replay counted: {mk_report}"
        assert all(r["ok"] for r in mk_report.values()), f"megakernel probe mismatch: {mk_report}"
    lens = {str(rid): len(s) for rid, s in sorted(streams.items())}
    committed = sum(m + 1 for s in drv.stats for m in s["m"].values())
    row_steps = sum(len(s["m"]) for s in drv.stats)
    accepted = sum(m for s in drv.stats for m in s["m"].values())
    fb = dict(getattr(eng, "fused_fallbacks", {}))
    rec = dict(
        tag=TAG,
        seed=SEED,
        weights=os.environ.get("HF_MODEL", ""),
        gptq_bfp4=os.environ.get("QWEN36_GPTQ_BFP4", "0"),
        cache=os.environ.get("TT_CACHE_PATH", ""),
        accepted=accepted,
        committed=committed,
        mode=MODE,
        flags=flags,
        kernels=kern,
        tokens=NEW,
        users=len(streams),
        lens=lens,
        md5={str(rid): _md5(s) for rid, s in sorted(streams.items())},
        md5_all=_md5([t for _, s in sorted(streams.items()) for t in s]),
        streams={str(rid): list(map(int, s)) for rid, s in sorted(streams.items())},
        steps=drv.steps,
        row_steps=row_steps,
        accept_per_row_step=round(accepted / max(1, row_steps), 4),
        committed_per_row_step=round(committed / max(1, row_steps), 4),
        chain_steps=int(eng.chain_steps),
        chain_audits=int(eng.chain_audits),
        chain_breaks=dict(eng.chain_breaks),
        fused_fallbacks=fb,
        overflow_redo=int(fb.get("overflow_redo", 0)),
        fused_greedy_checked=int(getattr(eng, "fused_greedy_checked", 0)),
        fused_sampled_steps=int(getattr(eng, "fused_sampled_steps", 0)),
        fused_audited_steps=int(getattr(eng, "fused_audited_steps", 0)),
        fused_boundary_diffs=int(getattr(eng, "fused_boundary_diffs", 0)),
        text0=tokenizer.decode(streams[0][:64]),
        mk_stage0=mk_report,
    )
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "a") as f:
        f.write(json.dumps(rec) + "\n")
    logger.info(
        f"[r7ident] tag={TAG} seed={SEED} gptq_bfp4={os.environ.get('QWEN36_GPTQ_BFP4', '0')} mode={MODE} flags={flags} kernels={kern} md5_all={rec['md5_all']} "
        f"accept={rec['accept_per_row_step']:.3f}/{tc.K} -> {rec['committed_per_row_step']:.3f} committed/iter over "
        f"{row_steps} row-steps ({drv.steps} steps); lens={sorted(set(lens.values()))} chain_steps={rec['chain_steps']} "
        f"audits={rec['chain_audits']} breaks={rec['chain_breaks']} fallbacks={fb} greedy_checked={rec['fused_greedy_checked']} "
        f"sampled_steps={rec['fused_sampled_steps']} audited={rec['fused_audited_steps']} "
        f"boundary_diffs={rec['fused_boundary_diffs']}"
    )
    assert all(n == NEW for n in lens.values()), lens
