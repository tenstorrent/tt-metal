# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end run of the persistent-stack port through the reference's own generator (Glm53Generator over
StackBackend): B users with the same chat prompt of ISL tokens, prefill + OSL greedy tokens; reports TTFT (the batch's
prefill wall time up to the first token), the decode step and tok/s per user, the per-user text agreement and user 0's
text. Prefill runs through the backend's decode-program chunks (GLM53_STACK_PREFILL=decode, the default).

    GLM53_MESH_SHAPE=8,4 GLM53_FABRIC=1d GLM53_MK_EP=1 GLM53_STACK_DSA=1 GLM53_MK_B=4 GLM53_MK_ISL=128 GLM53_MK_OSL=128 \
        pytest .../test_mk_e2e.py -s

GLM53_MK_PREFIX=1 (needs GLM53_MK_B >= 3): the device prefix cache through the generator's serving API instead (run_prefix:
a request A whose state at GLM53_MK_PREFIX_LEN = 32,768 tokens is snapshotted into a pool entry, a request B with the
same document and another question attaching to it with A's pages shared, and B prefilled from scratch); B's greedy ids
must equal the cold run's. Logs MK_PREFIX: lines (TTFT of the attach vs the cold prefill, decode, ids, counters).
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
# GLM53_MK_PREFIX=1: the prefix-cache run (run_prefix) instead of the batch; the shared boundary PREFIX_LEN is a multiple of
# the prefill chunk (8,192 with the 8x4 defaults' buckets), so the attach and the cold prefill run the same chunks past it
PREFIX = os.environ.get("GLM53_MK_PREFIX", "0") == "1"
PREFIX_LEN = int(os.environ.get("GLM53_MK_PREFIX_LEN", "32768"))
PREFIX_TAIL = 256  # the prompts' document runs about this many tokens past the boundary (then the question)
_ISL_SEQ = PREFIX_LEN + PREFIX_TAIL if PREFIX else ISL  # the longest prompt the cache must hold
SEQ = int(os.environ.get("GLM53_MK_SEQ", str(max(2048, 1 << (_ISL_SEQ + OSL + 64).bit_length()))))
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
QUESTION_B = "Which parts of the system does the text above describe? Answer in one sentence."


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


def chat_prompt(tok, isl: int, user: int = 0, question: str = QUESTION) -> list[int]:
    """a chat-rendered prompt of (about) isl tokens: filler text cut to the token budget + a question; or the text of
    GLM53_MK_PROMPT (=@file) as the whole user message (the HF greedy reference's prompt: ISL = its length).
    GLM53_MK_DISTINCT=1: user u's filler starts 7 u tokens into the text (the users route to different experts:
    every user of a batch with the same prompt picks the same experts, which flatters the EP expert phase).
    question: the question after the filler (the filler's budget is QUESTION's, so prompts that differ only in the
    question share the same filler)."""
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
    content = tok.decode(filler_ids) + "\n\n" + question
    ids = tok.encode_chat([{"role": "user", "content": content}])
    return ids


def prefix_prompts(tok, prefix_len: int) -> tuple[list[int], list[int]]:
    """GLM53_MK_PREFIX: two chat prompts that share their first prefix_len tokens (the filler document, which runs about
    PREFIX_TAIL tokens past the boundary) and end in different questions (QUESTION, QUESTION_B)"""
    if os.environ.get("GLM53_MK_PROMPT"):
        raise ValueError("GLM53_MK_PREFIX builds its own prompts: unset GLM53_MK_PROMPT")
    isl = prefix_len + PREFIX_TAIL
    a, b = chat_prompt(tok, isl, question=QUESTION), chat_prompt(tok, isl, question=QUESTION_B)
    common = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
    assert prefix_len < common < min(len(a), len(b)), f"prompts of {len(a)} / {len(b)} tokens share {common}"
    return a, b


def run_prefix(gen, kv, ids_a: list[int], ids_b: list[int], prefix_len: int, osl: int, sync) -> dict:
    """GLM53_MK_PREFIX=1: the device prefix cache (tt/generator.py prefill_forward's prefix_attach / snapshot_after) on a
    caller-owned cache (allocate_kv_cache(null_block=0), vLLM's null block 0; gen.shared_pages) with explicit page-table
    rows, P = prefix_len:
      A      ids_a in slot 0: [0, P) head-less with snapshot_after -> pool entry 0 (the KDA state + the DSA pooled keys /
             ring at P), then [P, end) -> A's first token; A is then released (its blocks stay mapped: cached);
      B      ids_b in slot 1: a NEW request attaching to entry 0 at start P (prefix_attach), A's first P / 64 blocks at
             the head of its row (shared pages, read-only), only [P, end) prefilled;
      B_cold ids_b in slot 2 from position 0 on its own blocks: P is on the chunk grid, so its chunks past P are B's.
    B and B_cold then decode osl - 1 more greedy tokens together. sync() waits for the device (the TTFT bounds).
    Returns the first tokens, the ids, the TTFTs (s), the decode wall (s) and steps."""
    from models.autoports.zai_org_glm_5_3_flash.tt.sampling import GREEDY

    bs, entry = gen.backend.block_size, 0
    assert prefix_len % gen.backend.prefill_chunk_tokens == 0, (prefix_len, gen.backend.prefill_chunk_tokens)
    gen.check_context(len(ids_b), osl)
    nxt = [1]  # block 0 is the null block

    def fresh(n_tok: int) -> list[int]:
        n = -(-n_tok // bs)
        nxt[0] += n
        return list(range(nxt[0] - n, nxt[0]))

    row_a = fresh(len(ids_a))
    row_b = row_a[: prefix_len // bs] + fresh(len(ids_b) + osl - prefix_len)  # the decode's blocks too
    row_c = fresh(len(ids_b) + osl)
    pt = lambda row: torch.tensor([row], dtype=torch.int64)  # noqa: E731
    ids = lambda v: torch.tensor([v], dtype=torch.int64)  # noqa: E731
    gen.shared_pages = True
    out = {}
    sync()
    t0 = time.perf_counter()
    gen.prefill_forward(ids(ids_a[:prefix_len]), page_table=pt(row_a), kv_cache=kv, prompt_lens=[prefix_len],
                        empty_slots=[0], start_pos=[0], sampling_params=[GREEDY], sample_rows=[False],
                        snapshot_after={0: entry})  # fmt: skip
    first = gen.prefill_forward(ids(ids_a[prefix_len:]), page_table=pt(row_a), kv_cache=kv,
                                prompt_lens=[len(ids_a) - prefix_len], empty_slots=[0], start_pos=[prefix_len],
                                sampling_params=[GREEDY])  # fmt: skip
    out["ttft_a"], out["first_a"] = time.perf_counter() - t0, int(first[0])
    gen.release_slot(kv, 0)  # A finished: its blocks are cached, the entry holds its state at P
    sync()
    t0 = time.perf_counter()
    first = gen.prefill_forward(ids(ids_b[prefix_len:]), page_table=pt(row_b), kv_cache=kv,
                                prompt_lens=[len(ids_b) - prefix_len], empty_slots=[1], start_pos=[prefix_len],
                                sampling_params=[GREEDY], prefix_attach={0: entry})  # fmt: skip
    out["ttft_b"], first_b = time.perf_counter() - t0, int(first[0])
    sync()
    t0 = time.perf_counter()
    first = gen.prefill_forward(ids(ids_b), page_table=pt(row_c), kv_cache=kv, prompt_lens=[len(ids_b)],
                                empty_slots=[2], start_pos=[0], sampling_params=[GREEDY])  # fmt: skip
    out["ttft_cold"], first_c = time.perf_counter() - t0, int(first[0])
    toks_b, toks_c = [first_b], [first_c]
    pos = len(ids_b)
    t0 = time.perf_counter()
    for _ in range(osl - 1):
        tok = gen.decode_forward(torch.tensor([-1, -1]), torch.tensor([pos, pos]), page_table=None, kv_cache=kv,
                                 slots=[1, 2], sampling_params=[GREEDY, GREEDY])  # fmt: skip
        toks_b.append(int(tok[0]))
        toks_c.append(int(tok[1]))
        pos += 1
    out["decode_s"], out["decode_steps"] = time.perf_counter() - t0, osl - 1
    out["ids_b"], out["ids_cold"] = toks_b, toks_c
    return out


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
    from models.autoports.zai_org_glm_5_3_flash.tt.sampling import SamplingParams
    from models.autoports.zai_org_glm_5_3_flash.tt.stack_model import StackBackend
    from models.autoports.zai_org_glm_5_3_flash.tt.tokenizer_chat import load_tokenizer

    # every knob an explicit environment value does not already set (the prefix run's prompts take the 8,192 buckets)
    DEFAULTS = GD.apply(MESH, SEQ, _ISL_SEQ)
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
    # GLM53_MK_TOPK=32 GLM53_MK_SAMPLE=1: the head's device sampler (the served construction, generator_vllm_stack
    # DEVICE_TOP_K); default topk 1 = greedy only
    topk = int(os.environ.get("GLM53_MK_TOPK", "1"))
    backend = StackBackend(mesh_device, max_batch=B, layers=layers, hf_config=cfg, topk=topk,
                           sample=os.environ.get("GLM53_MK_SAMPLE", "0") == "1", ep_weights=ep, native_t=native,
                           weight_cache=wcache)  # fmt: skip
    logger.info(f"StackBackend built in {time.perf_counter() - t0:.0f} s: mesh {MESH}, {len(layers)} layers, B {B}")
    if isinstance(source, MkTileCacheSource) and not source.ready:
        source.complete()
    gen = Glm53Generator(backend, tokenizer=tok, max_batch=B, max_seq_len=SEQ)
    if PREFIX:
        _prefix_e2e(gen, tok, mesh_device, source)
        return
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
    # GLM53_MK_TEMP=1.0 (GLM53_MK_TOP_P, GLM53_MK_TOP_K, GLM53_MK_SEED): sampled requests (GLM's generation config is
    # temperature 1.0 / top-p 0.95, no top-k); one seed for every user, so identical prompts still decode identically;
    # a sampled run decodes the whole OSL (no stop on EOS) so the step time covers OSL tokens
    temp = float(os.environ.get("GLM53_MK_TEMP", "0") or 0)
    sampling, stop = None, True
    if temp > 0:
        top_k = int(os.environ.get("GLM53_MK_TOP_K", "0") or 0)
        sp = SamplingParams(temperature=temp, top_k=top_k or None, top_p=float(os.environ.get("GLM53_MK_TOP_P", "0.95")),
                            seed=int(os.environ.get("GLM53_MK_SEED", "1234")))  # fmt: skip
        sampling, stop = [sp] * live, False
    # one call: TTFT = prefill of every user + its first token, decode = the rest (the generator splits its wall at the
    # first predictions: last_batch_timing). A separate TTFT call and decode = (second call - TTFT) / (OSL - 1) put the
    # difference of two prefills into every step: at 1 M context ~0.05 % of a 46 min prefill is ~20 ms per step
    outs = gen.generate_batch(prompts, OSL, sampling=sampling, stop_on_eos=stop, enable_trace=True)
    ttnn.synchronize_device(mesh_device)
    timing = gen.last_batch_timing
    n_gen = max(len(o) for o in outs)
    ttft_s = timing["prefill_s"]
    step_ms = timing["decode_s"] / max(1, timing["decode_steps"]) * 1e3
    same = all(o == outs[0] for o in outs)
    text = tok.decode([t for t in outs[0] if 0 <= t < tok.vocab_size])
    logger.info(
        f"MK_E2E: mesh {MESH} {len(layers)} layers B {B} live {live} spec_k {os.environ.get('GLM53_SPEC_K', '0')} "
        f"ISL {len(ids)} OSL {OSL}: TTFT {ttft_s * 1e3:.0f} ms "
        f"(all {live} users), decode {step_ms:.1f} ms/step = {1e3 / step_ms:.1f} tok/s/u, generated {n_gen}, "
        f"users agree {same}, first {outs[0][:1]} ids {outs[0][:24]}"
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
    if expect and sampling is None:  # the validated ids are greedy ones
        assert outs[0][: len(expect)] == expect, f"ids {outs[0][: len(expect)]} != the validated {expect}"
    floor = float(os.environ.get("GLM53_MK_MIN_TOKS", "0") or 0)
    if floor:
        assert 1e3 / step_ms >= floor, f"decode {1e3 / step_ms:.1f} tok/s/u < the floor {floor}"
    if isinstance(source, TileCache) and not source.ready:
        source.complete()  # the e2e passed on the tiles it dumped: READ mode from the next run
    if isinstance(source, MkTileCacheSource) and source._added:
        source.save_manifest()
        logger.info(f"MkTileCacheSource: {source._added} tensors added to the manifest")


def _prefix_e2e(gen, tok, mesh_device, source) -> None:
    """GLM53_MK_PREFIX=1: run_prefix on a serving-style cache (the plugin's pool: B x SEQ / 64 blocks + one per user,
    block 0 = vLLM's null block) warmed on that cache as the adapter does; B's greedy ids must equal B_cold's"""
    if B < 3:
        raise ValueError(f"GLM53_MK_PREFIX needs GLM53_MK_B >= 3 (slots A, B, B_cold), got {B}")
    kv = gen.allocate_kv_cache(B, SEQ, num_blocks=B * SEQ // 64 + B, null_block=0)
    entries = int(getattr(gen.backend, "snapshot_entries", 0))
    if not entries:
        raise ValueError("GLM53_MK_PREFIX needs the device snapshot pool (GLM53_PREFIX_DEVICE_ENTRIES > 0)")
    t0 = time.perf_counter()
    gen.warmup(kv, capture=True)
    ttnn.synchronize_device(mesh_device)
    logger.info(f"generator warm-up on the serving cache {time.perf_counter() - t0:.1f} s")
    ids_a, ids_b = prefix_prompts(tok, PREFIX_LEN)
    n_sfx = len(ids_b) - PREFIX_LEN
    logger.info(f"MK_PREFIX: mesh {MESH} B {B} cache {SEQ}: A {len(ids_a)} / B {len(ids_b)} tokens sharing {PREFIX_LEN} "
                f"(pool entry 0 of {entries}), B's suffix {n_sfx}; prefill chunk {gen.backend.prefill_chunk_tokens}, "
                f"continuation {gen.backend.continuation_align}, in-trace state I/O {gen.backend.prefill_state_io}; "
                f"OSL {OSL}")  # fmt: skip
    r = run_prefix(gen, kv, ids_a, ids_b, PREFIX_LEN, OSL, lambda: ttnn.synchronize_device(mesh_device))
    ttnn.synchronize_device(mesh_device)
    c = gen.counters
    step_ms = r["decode_s"] / max(1, r["decode_steps"]) * 1e3
    same = r["ids_b"] == r["ids_cold"]
    diverge = next((i for i, (x, y) in enumerate(zip(r["ids_b"], r["ids_cold"])) if x != y), None)
    text = lambda v: tok.decode([t for t in v if 0 <= t < tok.vocab_size])  # noqa: E731
    logger.info(f"MK_PREFIX: TTFT A {r['ttft_a'] * 1e3:.0f} ms (cold, snapshot at {PREFIX_LEN}), B {r['ttft_b'] * 1e3:.0f} "
                f"ms (attach + {n_sfx} tokens), B_cold {r['ttft_cold'] * 1e3:.0f} ms ({len(ids_b)} tokens): "
                f"x{r['ttft_cold'] / max(r['ttft_b'], 1e-9):.1f}")  # fmt: skip
    logger.info(f"MK_PREFIX: decode {step_ms:.1f} ms/step = {1e3 / step_ms if step_ms else 0:.1f} tok/s/u "
                f"(B + B_cold, {r['decode_steps']} steps)")  # fmt: skip
    logger.info(f"MK_PREFIX: ids B == B_cold {same} (first divergence {diverge}; first tokens A {r['first_a']} B "
                f"{r['ids_b'][0]} B_cold {r['ids_cold'][0]}); B {r['ids_b'][:24]} B_cold {r['ids_cold'][:24]}")  # fmt: skip
    logger.info(f"MK_PREFIX_TEXT B: {text(r['ids_b'])!r}")
    if not same:
        logger.info(f"MK_PREFIX_TEXT B_cold: {text(r['ids_cold'])!r}")
    logger.info(f"MK_PREFIX counters: prefix_attaches {c['prefix_attaches']} prefix_snapshots {c['prefix_snapshots']} "
                f"snapshot_copies {c['snapshot_copies']}; {gen.counters_report()}")  # fmt: skip
    assert same, f"B attached at {PREFIX_LEN} != B_cold from token {diverge}: {r['ids_b'][:12]} vs {r['ids_cold'][:12]}"
    assert c["prefix_attaches"] == 1 and c["prefix_snapshots"] >= 1, (c["prefix_attaches"], c["prefix_snapshots"])
    if isinstance(source, TileCache) and not source.ready:
        source.complete()  # the e2e passed on the tiles it dumped: READ mode from the next run
    if isinstance(source, MkTileCacheSource) and source._added:
        source.save_manifest()
        logger.info(f"MkTileCacheSource: {source._added} tensors added to the manifest")
