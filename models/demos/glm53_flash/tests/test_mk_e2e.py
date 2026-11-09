# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end run of the persistent-stack port through the reference's own generator (Glm53Generator over
StackBackend): B users with the same chat prompt of ISL tokens, prefill + OSL greedy tokens; reports TTFT (the batch's
prefill wall time up to the first token), the decode step and tok/s per user, the per-user text agreement and user 0's
text. Prefill runs through the backend's decode-program chunks (GLM53_STACK_PREFILL=decode, the default).

    GLM53_MESH_SHAPE=8,4 GLM53_FABRIC=1d GLM53_MK_EP=1 GLM53_STACK_DSA=1 GLM53_MK_B=4 GLM53_MK_ISL=128 GLM53_MK_OSL=128 \
        pytest .../test_mk_e2e.py -s
"""

from __future__ import annotations

import os
import time

import pytest
from loguru import logger

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tt import galaxy_defaults as GD
from models.autoports.zai_org_glm_5_3_flash.tt import optimized_decoder as OD
from models.autoports.zai_org_glm_5_3_flash.tt.config import load_hf_text_config
from models.autoports.zai_org_glm_5_3_flash.tt.mk_sources import MkTileCacheSource, TileCache, open_weight_source
from models.demos.glm53_flash.reference.checkpoint import default_snapshot_dir
from models.demos.glm53_flash.tt.mesh import mesh_device_params

MESH = tuple(int(v) for v in os.environ.get("GLM53_MESH_SHAPE", "8,4").split(","))
B = int(os.environ.get("GLM53_MK_B", "4"))
ISL = int(os.environ.get("GLM53_MK_ISL", "128"))
OSL = int(os.environ.get("GLM53_MK_OSL", "128"))
SEQ = int(os.environ.get("GLM53_MK_SEQ", str(max(2048, 1 << (ISL + OSL + 64).bit_length()))))
GALAXY = tuple(MESH) == GD.GALAXY
# the validated 8x4 configuration (tt/galaxy_defaults.py: TP32 x EP32, the sequence-sharded DSA pool, the TP prefill with
# the indexer, the pooled intermediates; the trace region / buckets / sparse ladder from SEQ) is applied INSIDE the test
# (never at collection: it would set every later-imported test's environment); the mesh parameters below take the
# trace region and the fabric directly
TRACE_MB = int(os.environ.get("GLM53_MODEL_TRACE_MB", str(GD.trace_mb(SEQ)) if GALAXY else "512"))
# the reference's SERVED worker L1 (pmk_layer WORKER_L1_SERVED): 16 KB more allocatable L1 than the mk_dense note's
# 1,264,512 -- the 45-layer stack sits at that floor (the sharded pool / the attached TP prefill clashed by 1 KB,
# 2026-11-05: "static circular buffers clash with L1 buffers ... 1269248 ... 1270272")
WORKER_L1 = int(os.environ.get("GLM53_MK_WORKER_L1", "1280896"))
FILLER = (
    "The Blackhole Galaxy is a 32-chip system; its chips talk over an Ethernet fabric and each one holds its own DRAM. "
    "A decoder layer of a mixture-of-experts model routes every token to a few experts, and the shared expert sees "
    "every token. Hyper-connections keep four residual streams per token and mix them before each sublayer. "
)
QUESTION = "Summarize the text above in two sentences."


def _layers() -> list[int]:
    spec = os.environ.get("GLM53_MK_LAYERS", "0-44")
    out: list[int] = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def chat_prompt(tok, isl: int, user: int = 0) -> list[int]:
    """a chat-rendered prompt of (about) isl tokens: filler text cut to the token budget + a question; or the text of
    GLM53_MK_PROMPT (=@file) as the whole user message (the HF greedy reference's prompt: ISL = its length).
    GLM53_MK_DISTINCT=1: user u's filler starts 7 u tokens into the text (the users route to different experts:
    every user of a batch with the same prompt picks the same experts, which flatters the EP expert phase)."""
    fixed = os.environ.get("GLM53_MK_PROMPT", "")
    if fixed.startswith("@"):
        with open(fixed[1:]) as f:
            fixed = f.read().strip()
    if fixed:
        return tok.encode_chat([{"role": "user", "content": fixed}])
    overhead = len(tok.encode_chat([{"role": "user", "content": QUESTION}]))
    budget = max(8, isl - overhead - 2)
    skip = 7 * user if os.environ.get("GLM53_MK_DISTINCT", "0") == "1" else 0
    filler_ids = tok.encode(FILLER * ((budget + skip) // 16 + 2))[skip : skip + budget]
    content = tok.decode(filler_ids) + "\n\n" + QUESTION
    ids = tok.encode_chat([{"role": "user", "content": content}])
    return ids


@pytest.mark.timeout(7000)
@pytest.mark.parametrize(
    "device_params",
    [
        mesh_device_params(
            *MESH,
            trace_region_size=TRACE_MB << 20,
            worker_l1_size=WORKER_L1,
            fabric=GD.KNOBS["GLM53_FABRIC"] if GALAXY else None,
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [MESH], indirect=True)
def test_mk_e2e(mesh_device, monkeypatch):
    from models.autoports.zai_org_glm_5_3_flash.tt.generator import Glm53Generator
    from models.autoports.zai_org_glm_5_3_flash.tt.stack_model import StackBackend
    from models.autoports.zai_org_glm_5_3_flash.tt.tokenizer_chat import load_tokenizer

    DEFAULTS = GD.apply(MESH, SEQ, ISL)  # every knob an explicit environment value does not already set
    snapshot = default_snapshot_dir()
    cfg = load_hf_text_config(str(snapshot))
    tok = load_tokenizer(snapshot)
    if DEFAULTS:
        logger.info(f"galaxy defaults applied: {DEFAULTS}")
    # the weight source (tt/mk_sources.py): GLM53_MK_TILE_CACHE_V3 = the reference-format tile cache (MkTileCacheSource:
    # READ when COMPLETE, else BUILT by this run -- every tensor the stack asks for is dumped from the HF source as it
    # loads, ~1 h on a cold runner -- and marked COMPLETE below), else the HF source (GLM53_MK_TILE_CACHE: TileCache)
    source = open_weight_source(snapshot, cfg)
    monkeypatch.setattr(OD, "FdTileCacheSource", lambda cache_dir, manifest: source)
    layers = _layers()
    ep = os.environ.get("GLM53_MK_EP", "1") == "1"
    native = os.environ.get("GLM53_MK_NATIVE_T", "1" if B > 4 else "0") == "1"
    t0 = time.perf_counter()
    # GLM53_MK_WEIGHT_CACHE=1: the served construction's weight cache (an empty dict this run fills): the sparse_dsa
    # layers then carry fabric phases for every t <= 4 besides the chunk (the speculative step's MTP layer shares
    # them at ts = (1, K + 1)) and the narrow widths (GLM53_STACK_NARROW_WIDTHS) are honoured
    wcache = {} if os.environ.get("GLM53_MK_WEIGHT_CACHE") == "1" else None
    backend = StackBackend(mesh_device, max_batch=B, layers=layers, hf_config=cfg, topk=1, sample=False, ep_weights=ep,
                           native_t=native, weight_cache=wcache)  # fmt: skip
    logger.info(f"StackBackend built in {time.perf_counter() - t0:.0f} s: mesh {MESH}, {len(layers)} layers, B {B}")
    if isinstance(source, MkTileCacheSource) and not source.ready:
        source.complete()
    gen = Glm53Generator(backend, tokenizer=tok, max_batch=B, max_seq_len=SEQ)
    t0 = time.perf_counter()
    gen.warmup(capture=True)
    ttnn.synchronize_device(mesh_device)
    logger.info(f"generator warm-up (prefill buckets, decode, traces) {time.perf_counter() - t0:.1f} s")
    ids = chat_prompt(tok, ISL)
    # GLM53_MK_LIVE=n (default B): the live users of the run on a backend built for B rows (the speculative step,
    # GLM53_SPEC_K, serves ONE live user in slot 0 of a backend whose max_batch >= K + 1)
    live = int(os.environ.get("GLM53_MK_LIVE", str(B)))
    prompts = [list(chat_prompt(tok, ISL, user=u)) for u in range(live)]
    logger.info(f"prompt {len(ids)} tokens (ISL target {ISL}; per user {[len(p) for p in prompts]}); OSL {OSL}; "
                f"cache {B} x {SEQ}; distinct {os.environ.get('GLM53_MK_DISTINCT', '0')}")  # fmt: skip
    # TTFT: prefill of every user + the first token (the generator's prefill chunks + one sampler step)
    t0 = time.perf_counter()
    first = gen.generate_batch(prompts, 1, enable_trace=True)
    ttnn.synchronize_device(mesh_device)
    ttft_s = time.perf_counter() - t0
    gen.reset()
    t0 = time.perf_counter()
    outs = gen.generate_batch(prompts, OSL, enable_trace=True)
    ttnn.synchronize_device(mesh_device)
    total_s = time.perf_counter() - t0
    n_gen = max(len(o) for o in outs)
    step_ms = (total_s - ttft_s) / max(1, n_gen - 1) * 1e3
    same = all(o == outs[0] for o in outs)
    text = tok.decode([t for t in outs[0] if 0 <= t < tok.vocab_size])
    logger.info(
        f"MK_E2E: mesh {MESH} {len(layers)} layers B {B} live {live} spec_k {os.environ.get('GLM53_SPEC_K', '0')} "
        f"ISL {len(ids)} OSL {OSL}: TTFT {ttft_s * 1e3:.0f} ms "
        f"(all {live} users), decode {step_ms:.1f} ms/step = {1e3 / step_ms:.1f} tok/s/u, generated {n_gen}, "
        f"users agree {same}, first {first[0][:1]} ids {outs[0][:24]}"
    )
    logger.info(f"MK_E2E_TEXT: {text!r}")
    logger.info(f"MK_E2E counters: {gen.counters_report()}")
    assert n_gen >= 1
    # the gates (a PASS means the numerics, not only the run): identical prompts decode identically across the users;
    # GLM53_MK_EXPECT_IDS="785,1196,702,3267,..." = user 0's first ids of the validated row for this configuration;
    # GLM53_MK_MIN_TOKS = the decode floor in tok/s per user (a regression below it fails the row)
    if live > 1 and os.environ.get("GLM53_MK_DISTINCT", "0") != "1":
        assert same, f"identical prompts, different outputs: {[o[:12] for o in outs]}"
    expect = [int(v) for v in os.environ.get("GLM53_MK_EXPECT_IDS", "").split(",") if v.strip()]
    if expect:
        assert outs[0][: len(expect)] == expect, f"ids {outs[0][: len(expect)]} != the validated {expect}"
    floor = float(os.environ.get("GLM53_MK_MIN_TOKS", "0") or 0)
    if floor:
        assert 1e3 / step_ms >= floor, f"decode {1e3 / step_ms:.1f} tok/s/u < the floor {floor}"
    if isinstance(source, TileCache) and not source.ready:
        source.complete()  # the e2e passed on the tiles it dumped: READ mode from the next run
    if isinstance(source, MkTileCacheSource) and source._added:
        source.save_manifest()
        logger.info(f"MkTileCacheSource: {source._added} tensors added to the manifest")
