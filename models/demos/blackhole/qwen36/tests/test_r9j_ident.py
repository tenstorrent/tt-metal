# r9 joint: copy of test_r7k2s_ident.py that also records every QWEN36_VERIFY_OUT_RES40* / QWEN36_BFP4_LAYOUT* /
# QWEN36_GPTQ_BFP4 env value, the RES40 sites that really ran (tp_common.res40_engaged) and the BFP4 layout targets that
# really took a config, and FAILS when R9J_EXPECT_RES40 lists a site that did not engage (or when the flag is off and any
# site engaged). Pod-local identity harness for r7 K2 + K2s (NOT committed; copied into the tree's tests/ for one device session and
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
FLAGS = ("QWEN36_SPEC_CHAIN_KERNEL", "QWEN36_SPEC_ACCEPT_KERNEL", "QWEN36_SPEC_ACCEPT_KERNEL_SAMPLED")


def _md5(xs):
    return hashlib.md5(torch.tensor(xs, dtype=torch.int64).numpy().tobytes()).hexdigest()[:16]


def _kname(dev):
    return type(getattr(dev, "_kernel", None)).__name__ if dev is not None else "absent"


@_marks
def test_r9j_ident(mesh_device, monkeypatch):
    assert MODE in ("greedy", "sampled"), MODE
    _env(monkeypatch)  # FUSED=1 CHAIN=1, chain audit every step, B=8, 48-block tables
    monkeypatch.setenv("QWEN36_SPEC_CHAIN_SAMPLED", "1")
    monkeypatch.setenv("QWEN36_SPEC_FUSED_AUDIT", "1")
    monkeypatch.setenv("QWEN36_SPEC_FUSED_AUDIT_EVERY", "1")
    flags = {k: os.environ.get(k, "0") for k in FLAGS}
    flags.update({k: v for k, v in sorted(os.environ.items())
                  if k.startswith(("QWEN36_VERIFY_OUT_RES40", "QWEN36_BFP4_LAYOUT", "QWEN36_GPTQ_BFP4", "QWEN36_ATTN_KV_SPLIT", "QWEN36_SPEC_DR_", "QWEN36_SPEC_VFY_", "QWEN36_MTP_REPL", "QWEN36_DRAFT_", "QWEN36_MLP_GU", "QWEN36_LMHEAD"))})
    flags["HF_MODEL"] = os.environ.get("HF_MODEL", "")
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
        drv = AsyncDriver(eng, seed=1, sampled=(lambda s: True) if sampled else (lambda s: False))
        _admit_all(drv, prompts, 8, NEW)
        drv.pipeline(8 * NEW)
        streams = _streams(drv)
        assert mesh_device.num_program_cache_entries() == n_prog, "a program compiled after capture"
    finally:
        mesh_device.set_program_cache_misses_allowed(True)
    from models.demos.blackhole.qwen36.tt import tp_common as tpc

    engaged = tpc.res40_engaged()
    layout = {t: tpc.bfp4_layout(t, __import__("ttnn").bfloat4_b) for t in sorted(tpc._BFP4_LAYOUT_LOGGED)}
    expect = [x for x in os.environ.get("R9J_EXPECT_RES40", "").split(",") if x]
    lens = {str(rid): len(s) for rid, s in sorted(streams.items())}
    committed = sum(m + 1 for s in drv.stats for m in s["m"].values())
    row_steps = sum(len(s["m"]) for s in drv.stats)
    accepted = sum(m for s in drv.stats for m in s["m"].values())
    fb = dict(getattr(eng, "fused_fallbacks", {}))
    rec = dict(
        tag=TAG,
        mode=MODE,
        flags=flags,
        res40_engaged=engaged,
        bfp4_layout=layout,
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
    )
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "a") as f:
        f.write(json.dumps(rec) + "\n")
    logger.info(
        f"[r7ident] tag={TAG} mode={MODE} flags={flags} kernels={kern} md5_all={rec['md5_all']} "
        f"accept={rec['accept_per_row_step']:.3f}/{tc.K} -> {rec['committed_per_row_step']:.3f} committed/iter over "
        f"{row_steps} row-steps ({drv.steps} steps); lens={sorted(set(lens.values()))} chain_steps={rec['chain_steps']} "
        f"audits={rec['chain_audits']} breaks={rec['chain_breaks']} fallbacks={fb} greedy_checked={rec['fused_greedy_checked']} "
        f"sampled_steps={rec['fused_sampled_steps']} audited={rec['fused_audited_steps']} "
        f"boundary_diffs={rec['fused_boundary_diffs']} res40_engaged={engaged} bfp4_layout={layout}"
    )
    if os.environ.get(tpc.VERIFY_OUT_RES40_ENV, "0") != "1":
        assert not engaged, f"RES40 flag off but sites engaged: {engaged}"
    missing = [x for x in expect if not engaged.get(x)]
    assert not missing, f"R9J_EXPECT_RES40 sites that did NOT engage: {missing} (engaged={engaged})"
    assert all(n == NEW for n in lens.values()), lens
