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

Image input (GLM53_MK_IMAGE_FEATS=<.pt> and / or GLM53_MK_IMAGE=<png>): every user's prompt is the image prompt (the
HF processor's ids: chat template + image placeholder run; from the file, else the processor on the png) and its
image positions take bf16 feature rows through the stack's extra embedding rows (GLM53_STACK_MM=1, set here).
GLM53_MK_IMAGE_TOWER: "feats" (the default with a file) = the file's HF CPU tower rows; "device" (the default without
one) = the stack's own tower (generator_vllm_stack.DeviceVisionEncoder: built and warmed by the backend's pre-capture
hook) on the file's / processor's pixel patches, its PCC vs the file's HF rows logged and gated (GLM53_MK_TOWER_PCC_MIN,
default 0.96), and a text user's ids generated before and after the tower call compared (GLM53_MK_TOWER_TEXT_CHECK,
default 1). Gate = the HF greedy reference ids (GLM53_MK_IMAGE_EXPECT, else the file's expect_ids): first token and
every id, the stack's top-5 per step (teacher-forced on the reference) logged when they differ. GLM53_MK_ISL = the
prompt's length.

Speculative A/B (Stage 17 M0, GLM53_SPEC_K=K with GLM53_MK_LIVE=1 and greedy): after the plain run the same prompt
decodes through the backend's MTP speculative step on the same build (Glm53Generator.generate_batch(spec_k=K)); the
ids must equal the plain ids up to the first near-tie (a mismatch fails only where the plain top-2 logit gap at that
position is >= GLM53_MK_SPEC_TIE, default 1.0); MK_SPEC logs tau (committed tokens per spec step), the spec step, the
effective tok/s/u and the plain step. On the 8x4 shard (B 4):

    GLM53_MESH_SHAPE=8,4 GLM53_MK_B=4 GLM53_MK_LIVE=1 GLM53_SPEC_K=3 GLM53_SPEC_ON_SHARD=1 GLM53_MK_ISL=128 \
        GLM53_MK_OSL=128 pytest .../test_mk_e2e.py -s

Multi-user A/B (GLM53_SPEC_MULTI=1, doc/perf/mtp_multi_user_design.md section 6: the backend's step serves G =
GLM53_SPEC_MULTI_USERS users on the group bases; GLM53_MK_LIVE = G, greedy, text): the G prompts decode through it after
the plain run; EVERY user's ids must equal its plain ids up to its first near-tie (_spec_ab with live > 1); MK_SPEC_MULTI
logs tau per user, the step and the effective tok/s/u. On the B 16 slot build at 1 M (G 4: the verify on all 16 rows):

    GLM53_MESH_SHAPE=8,4 GLM53_MK_B=16 GLM53_MK_SEQ=1048576 GLM53_MK_WEIGHT_CACHE=1 GLM53_STACK_NARROW_WIDTHS=1,2,4 \
        GLM53_STACK_NATIVE_WIDTHS=8 GLM53_SPEC_MULTI=1 GLM53_SPEC_MULTI_USERS=4 GLM53_MK_LIVE=4 GLM53_MK_DISTINCT=1 \
        GLM53_SPEC_ON_SHARD=1 GLM53_SPEC_K=3 GLM53_MK_ISL=128 GLM53_MK_OSL=128 pytest .../test_mk_e2e.py -s
"""

from __future__ import annotations

import hashlib
import os
import time

import pytest
import torch
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
IMAGE = os.environ.get("GLM53_MK_IMAGE", "")
IMAGE_FEATS = os.environ.get("GLM53_MK_IMAGE_FEATS", "")
IMAGE_TOWER = os.environ.get("GLM53_MK_IMAGE_TOWER", "") or ("feats" if IMAGE_FEATS else "device")
IMAGE_QUESTION = os.environ.get(
    "GLM53_MK_IMAGE_QUESTION", "What shapes and what word are in this image? Answer in one sentence."
)
# the HF greedy reference of demo/sample_prompts/shapes_hello.png + "What shapes and what word are in this image? Answer
# in one sentence." (doc/vision/multimodal.md): 'The image contains a red circle, a blue square, a green triangle, and'
SHAPES_HELLO_IDS = [785, 2168, 5610, 264, 2518, 12665, 11, 264, 6303, 9331, 11, 264, 6176, 21445, 11, 323]


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


def image_input(snapshot) -> dict:
    """{ids, rows, feats, pixel_values, grid, expect} of one image prompt: from GLM53_MK_IMAGE_FEATS -- the file's
    prompt ids, its image positions (every image-token id, in order), their [rows, 4096] bf16 HF feature rows and the
    processor's pixel patches / grid (GLM53_MK_IMAGE, when set, must be the file's image: its sha256) -- else from the
    HF processor on GLM53_MK_IMAGE + GLM53_MK_IMAGE_QUESTION (needs a transformers with glm5_next; no HF rows)"""
    if IMAGE_FEATS:
        d = torch.load(IMAGE_FEATS, weights_only=False)
        if IMAGE:
            with open(IMAGE, "rb") as f:
                sha = hashlib.sha256(f.read()).hexdigest()
            assert d.get("image_sha256") in (None, sha), f"{IMAGE} is not the image of {IMAGE_FEATS}"
    else:
        from transformers import AutoProcessor

        from models.demos.glm53_flash.tt.multimodal import encode_prompt

        try:
            processor = AutoProcessor.from_pretrained(str(snapshot))
        except Exception as e:  # noqa: BLE001
            pytest.fail(f"GLM53_MK_IMAGE without GLM53_MK_IMAGE_FEATS needs the GLM-5.3-Flash HF processor: {e}")
        enc = encode_prompt(processor, [{"type": "image", "image": IMAGE}, {"type": "text", "text": IMAGE_QUESTION}])
        d = dict(input_ids=enc.input_ids, rows=torch.nonzero(enc.input_ids[0] == 154854).reshape(-1),
                 pixel_values=enc.pixel_values, image_grid_thw=enc.image_grid_thw)  # fmt: skip
    ids = [int(v) for v in d["input_ids"].reshape(-1).tolist()]
    rows = d["rows"].reshape(-1).to(torch.int64)
    feats = d["feats"].to(torch.bfloat16) if d.get("feats") is not None else None
    tok_id = int(d.get("image_token_id", 154854))
    assert feats is None or tuple(feats.shape) == (int(rows.numel()), 4096), (tuple(feats.shape), int(rows.numel()))
    assert [i for i, t in enumerate(ids) if t == tok_id] == rows.tolist(), "the rows are not the prompt's image tokens"
    assert IMAGE_TOWER in ("feats", "device"), f"GLM53_MK_IMAGE_TOWER={IMAGE_TOWER}: feats | device"
    assert IMAGE_TOWER == "device" or feats is not None, "GLM53_MK_IMAGE_TOWER=feats needs the HF rows of a FEATS file"
    pv, grid = d.get("pixel_values"), d.get("image_grid_thw")
    assert IMAGE_TOWER == "feats" or pv is not None, "the device tower needs the pixel patches (pixel_values)"
    env = os.environ.get("GLM53_MK_IMAGE_EXPECT", "")
    expect = [int(v) for v in env.split(",") if v.strip()] or [int(v) for v in d.get("expect_ids", SHAPES_HELLO_IDS)]
    return dict(ids=ids, rows=rows, feats=feats, expect=expect, tower=IMAGE_TOWER,
                pixel_values=None if pv is None else torch.as_tensor(pv).float(),
                grid=None if grid is None else torch.as_tensor(grid).reshape(-1, 3).long())  # fmt: skip


def device_tower_rows(gen, enc, tok, mm_in: dict) -> torch.Tensor:
    """the stack's own tower on the prompt's pixel patches (after the warm-up captures): its PCC vs the HF rows and,
    GLM53_MK_TOWER_TEXT_CHECK=1, a text user's greedy ids generated before and after the tower call (equal: the tower
    leaves the text path's buffers and traces alone), both kept in mm_in["tower"] for the gates at the end of the test
    (tower_gates) -> its bf16 rows"""
    from models.demos.glm53_flash.tt.mm_inputs import MMItem

    assert enc.runtime is not None, "the device tower was not built by the backend's pre-capture hook"
    check = os.environ.get("GLM53_MK_TOWER_TEXT_CHECK", "1") == "1"
    text, n_chk = chat_prompt(tok, ISL), min(OSL, 16)
    before = gen.generate_batch([text], n_chk, stop_on_eos=False, enable_trace=True)[0] if check else None
    item = MMItem("image", mm_in["pixel_values"], mm_in["grid"][0])
    t0 = time.perf_counter()
    (rows,) = enc([item])
    tower_s = time.perf_counter() - t0
    assert tuple(rows.shape) == (int(mm_in["rows"].numel()), 4096), tuple(rows.shape)
    after = gen.generate_batch([text], n_chk, stop_on_eos=False, enable_trace=True)[0] if check else None
    pcc = None
    if mm_in["feats"] is not None:
        a, b = rows.float().flatten(), mm_in["feats"].float().flatten()
        pcc = float(torch.corrcoef(torch.stack([a, b]))[0, 1])
    logger.info(f"MK_E2E_TOWER: device tower {tuple(rows.shape)} in {tower_s * 1e3:.0f} ms (warm-up {enc.warm_s:.1f} s, "
                f"{enc.fenced} fenced launches); PCC vs the HF rows {pcc}; text user {len(text)} tokens: ids before "
                f"{before} after {after} equal {before == after}")  # fmt: skip
    mm_in["tower_check"] = dict(pcc=pcc, before=before, after=after, tower_s=tower_s)
    return rows


def tower_gates(mm_in: dict) -> None:
    """the device tower's gates (after the ids are logged): PCC vs the HF rows >= GLM53_MK_TOWER_PCC_MIN (0.96; the
    default-L1 tower on one chip scored 0.988 on this image) and the text user's ids unchanged across the call"""
    t = mm_in.get("tower_check")
    if t is None:
        return
    floor = float(os.environ.get("GLM53_MK_TOWER_PCC_MIN", "0.96"))
    assert t["pcc"] is None or t["pcc"] >= floor, f"device tower PCC {t['pcc']:.5f} vs the HF features < {floor}"
    assert t["before"] == t["after"], f"a text user's ids changed across the tower call: {t['before']} -> {t['after']}"


def image_topk_report(gen, mm_in: dict, k: int = 5) -> list:
    """the stack's top-k at the image prompt's last position and after each reference token (teacher-forced on the
    reference ids), host logits route (prefill_forward / decode_forward without sampling params) in slot 0 of the
    generator's own cache -> [(step, reference id, [(id, logit)] * k)]"""
    b = gen._own_binding()
    kv, ids, exp = b.kv_cache, mm_in["ids"], mm_in["expect"]
    n = len(ids)
    row = b.identity_row(0).reshape(1, -1)
    top = lambda lg: [(int(i), round(float(v), 3)) for v, i in zip(*torch.topk(lg.reshape(-1).float(), k))]  # noqa
    out = []
    lg = gen.prefill_forward(torch.tensor([ids], dtype=torch.int64), page_table=row, kv_cache=kv, prompt_lens=[n],
                             empty_slots=[0], mm_rows={0: (mm_in["rows"], mm_in["feats"])})  # fmt: skip
    out.append((0, exp[0], top(lg)))
    for j in range(len(exp) - 1):
        lg = gen.decode_forward(torch.tensor([[exp[j]]]), torch.tensor([n + j]), page_table=row, kv_cache=kv,
                                slots=[0], sampling_params=None, read_from_device=True)  # fmt: skip
        out.append((j + 1, exp[j + 1], top(lg)))
    gen.release_slot(kv, 0)
    return out


def _last_logits(gen, ids: list[int]):
    """the plain model's logits of the last position of `ids` ([V] fp32): the generator-owned slot 0 through the
    backend's per-chunk prefill (the TP prefill's head on the last row) -- the top-2 gap at a spec / plain mismatch"""
    import torch

    from models.autoports.zai_org_glm_5_3_flash.tt.generator import SlotState

    b = gen._own_binding()
    gen._ensure_identity_rows(b, [0], {0: len(ids)})
    b.slots[0] = SlotState(active=True)
    try:
        return gen._prefill_user(b, 0, torch.as_tensor(list(ids), dtype=torch.int64)).reshape(-1).float()
    finally:
        b.slots[0].active = False
        b.slots[0].finished = True


def _first_mismatch(gen, tok, prompt, plain, sids):
    """(first index where the spec ids leave the plain ids or None, the plain model's top-2 gap there or None)"""
    n = min(len(plain), len(sids))
    first = next((j for j in range(n) if plain[j] != sids[j]), None)
    if first is None and len(plain) != len(sids):
        first = n
    gap = None
    if first is not None:  # the plain model's top-2 gap at the first mismatch (its context: prompt + plain[:first])
        lg = _last_logits(gen, list(prompt) + list(plain[:first]))
        top = lg[: tok.vocab_size].topk(2)
        gap = float(top.values[0] - top.values[1])
        logger.info(f"MK_SPEC first mismatch at {first}: plain {plain[first: first + 4]} spec {sids[first: first + 4]} "
                    f"plain top-2 {top.indices.tolist()} gap {gap:.3f}")  # fmt: skip
    return first, gap


def _spec_ab_multi(gen, backend, mesh_device, tok, prompts, plains, plain_timing, plain_ms, K, stop) -> None:
    """GLM53_SPEC_MULTI: the G prompts through the backend's multi-user step (generate_batch(spec_k=K) places them on
    the group bases) after the plain run on the same build; per user the ids == its plain ids up to its first near-tie
    (the gate), tau (its committed tokens per step), the plain top-2 gap at a mismatch; MK_SPEC_MULTI lines"""
    G = len(prompts)
    souts = gen.generate_batch([list(p) for p in prompts], OSL, stop_on_eos=stop, enable_trace=True, spec_k=K)
    ttnn.synchronize_device(mesh_device)
    st = gen.last_batch_timing
    steps = int(st["spec_steps"])
    spec_ms = st["decode_s"] / max(1, steps) * 1e3
    eff = st["decode_steps"] / st["decode_s"] if st["decode_s"] > 0 else 0.0
    res = [_first_mismatch(gen, tok, prompts[u], plains[u], souts[u]) for u in range(G)]
    taus = [c / max(1, steps) for c in st.get("spec_committed_u", [0] * G)]
    for u in range(G):
        logger.info(f"MK_SPEC_MULTI user {u}: tau {taus[u]:.2f} (committed {st['spec_committed_u'][u]}, accepts "
                    f"n=0..K {st['spec_accepts_u'][u]}), ids equal {res[u][0] is None} (first mismatch {res[u][0]}, "
                    f"plain top-2 gap {res[u][1]}); spec ids {souts[u][:16]} plain {plains[u][:16]}")  # fmt: skip
    logger.info(
        f"MK_SPEC_MULTI: G {G} K {K} tau mean {sum(taus) / max(1, G):.2f} per user {[round(t, 2) for t in taus]} "
        f"({steps} steps), spec step {spec_ms:.1f} ms, effective {eff:.1f} tok/s/u (decode {st['decode_steps']} tokens "
        f"in {st['decode_s'] * 1e3:.0f} ms), plain {plain_ms:.1f} ms/step = {1e3 / plain_ms:.1f} tok/s/u, speedup "
        f"{eff * plain_ms / 1e3:.2f}x; TTFT spec {st['prefill_s'] * 1e3:.0f} ms vs plain "
        f"{plain_timing['prefill_s'] * 1e3:.0f} ms (MTP prefill rows {st.get('spec_mtp_prefill_rows')} of "
        f"{sum(len(p) - 1 for p in prompts)}); ids equal per user {[r[0] is None for r in res]}"
    )
    logger.info(f"MK_SPEC_MULTI_TEXT user 0: {tok.decode([t for t in souts[0] if 0 <= t < tok.vocab_size])!r}")
    logger.info(f"MK_SPEC backend spec counters: {backend.spec.counters}")
    tie = float(os.environ.get("GLM53_MK_SPEC_TIE", "1.0"))
    assert steps > 0, "the multi-user speculative path ran no step"
    min_tau = float(os.environ.get("GLM53_MK_SPEC_MIN_TAU", "0") or 0)
    if min_tau:
        assert min(taus) >= min_tau, f"tau per user {taus} below the floor {min_tau}"
    bad = [(u, f, g) for u, (f, g) in enumerate(res) if f is not None and g >= tie]
    assert not bad, (f"spec ids differ from plain where the plain top-2 gap >= {tie}: (user, index, gap) {bad}; "
                     f"{[(souts[u][: f + 4], plains[u][: f + 4]) for u, f, _ in bad]}")  # fmt: skip


def _spec_ab(gen, backend, mesh_device, tok, prompt, plain, plain_timing, plain_ms, K, stop) -> None:
    """Stage 17 M0: the same prompt through the speculative step (generate_batch(spec_k=K)) after the plain greedy run
    on the same build; MK_SPEC line + the gate (ids == plain up to the first near-tie)"""
    souts = gen.generate_batch([list(prompt)], OSL, stop_on_eos=stop, enable_trace=True, spec_k=K)
    ttnn.synchronize_device(mesh_device)
    st = gen.last_batch_timing
    sids = souts[0]
    steps, committed = int(st["spec_steps"]), int(st["spec_committed"])
    tau = committed / max(1, steps)
    spec_ms = st["decode_s"] / max(1, steps) * 1e3
    eff = st["decode_steps"] / st["decode_s"] if st["decode_s"] > 0 else 0.0
    n = min(len(plain), len(sids))
    first = next((j for j in range(n) if plain[j] != sids[j]), None)
    if first is None and len(plain) != len(sids):
        first = n
    gap = None
    if first is not None:  # the plain model's top-2 gap at the first mismatch (its context: prompt + plain[:first])
        lg = _last_logits(gen, list(prompt) + list(plain[:first]))
        top = lg[: tok.vocab_size].topk(2)
        gap = float(top.values[0] - top.values[1])
        logger.info(f"MK_SPEC first mismatch at {first}: plain {plain[first: first + 4]} spec {sids[first: first + 4]} "
                    f"plain top-2 {top.indices.tolist()} gap {gap:.3f}")  # fmt: skip
    text = tok.decode([t for t in sids if 0 <= t < tok.vocab_size])
    logger.info(
        f"MK_SPEC: K {K} tau {tau:.2f} (committed {committed} / {steps} steps, accepts n=0..K {st['spec_accepts']}), "
        f"spec step {spec_ms:.1f} ms, effective {eff:.1f} tok/s/u (decode {st['decode_steps']} tokens in "
        f"{st['decode_s'] * 1e3:.0f} ms), plain {plain_ms:.1f} ms/step = {1e3 / plain_ms:.1f} tok/s/u, speedup "
        f"{eff * plain_ms / 1e3:.2f}x; TTFT spec {st['prefill_s'] * 1e3:.0f} ms vs plain {plain_timing['prefill_s'] * 1e3:.0f} ms "
        f"(MTP prefill rows {st.get('spec_mtp_prefill_rows')} of {len(prompt) - 1}); ids equal "
        f"{first is None} (first mismatch {first}, plain top-2 gap {gap}); spec ids {sids[:24]}"
    )
    logger.info(f"MK_SPEC_TEXT: {text!r}")
    logger.info(f"MK_SPEC backend spec counters: {backend.spec.counters}")
    tie = float(os.environ.get("GLM53_MK_SPEC_TIE", "1.0"))
    assert steps > 0, "the speculative path ran no step"
    # GLM53_MK_SPEC_TTFT_RATIO (e.g. 1.10): the spec request's TTFT (its prefill fills the MTP cache) within that ratio
    # of the plain request's on the same build; GLM53_MK_SPEC_MIN_TAU: a floor on tau
    ratio = float(os.environ.get("GLM53_MK_SPEC_TTFT_RATIO", "0") or 0)
    if ratio:
        assert (
            st["prefill_s"] <= ratio * plain_timing["prefill_s"]
        ), f"spec TTFT {st['prefill_s'] * 1e3:.0f} ms > {ratio} x plain {plain_timing['prefill_s'] * 1e3:.0f} ms"
    min_tau = float(os.environ.get("GLM53_MK_SPEC_MIN_TAU", "0") or 0)
    if min_tau:
        assert tau >= min_tau, f"tau {tau:.2f} < the floor {min_tau}"
    assert first is None or gap < tie, (
        f"spec ids differ from plain at {first} where the plain top-2 gap is {gap:.3f} >= {tie}: {sids[:first + 4]} vs "
        f"{plain[:first + 4]}"
    )


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
    mm_in = image_input(snapshot) if (IMAGE_FEATS or IMAGE) else None
    if mm_in is not None:  # the embedding table's extra rows (tt/mk_head.py mm_extra_rows), before the backend is built
        monkeypatch.setenv("GLM53_STACK_MM", os.environ.get("GLM53_STACK_MM", "1"))
        assert os.environ["GLM53_STACK_MM"] == "1", "an image prompt needs GLM53_STACK_MM=1"
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
    encoder = None
    if mm_in is not None and mm_in["tower"] == "device":  # built + warmed by the pre-capture hook (gen.warmup below)
        from models.demos.glm53_flash.tt.generator_vllm_stack import DeviceVisionEncoder

        encoder = DeviceVisionEncoder(backend, snapshot)
    gen = Glm53Generator(backend, tokenizer=tok, max_batch=B, max_seq_len=SEQ)
    if PREFIX:
        _prefix_e2e(gen, tok, mesh_device, source)
        return
    t0 = time.perf_counter()
    gen.warmup(capture=True)
    ttnn.synchronize_device(mesh_device)
    logger.info(f"generator warm-up (prefill buckets, decode, traces) {time.perf_counter() - t0:.1f} s")
    if encoder is not None:  # the stack's own tower rows replace the file's in the prompt
        mm_in["feats"] = device_tower_rows(gen, encoder, tok, mm_in)
    ids = chat_prompt(tok, ISL) if mm_in is None else mm_in["ids"]
    # GLM53_MK_LIVE=n (default B): the live users of the run on a backend built for B rows (the speculative step,
    # GLM53_SPEC_K, serves ONE live user in slot 0 of a backend whose max_batch >= K + 1)
    live = int(os.environ.get("GLM53_MK_LIVE", str(B)))
    prompts = [list(chat_prompt(tok, ISL, user=u)) if mm_in is None else list(ids) for u in range(live)]
    mm = None if mm_in is None else [(mm_in["rows"], mm_in["feats"])] * live
    if mm_in is not None:
        r_ = mm_in["rows"]
        logger.info(f"image prompt: {len(ids)} tokens, {int(r_.numel())} image rows {int(r_[0])}..{int(r_[-1])}, "
                    f"extra embedding rows {backend.mm_extra_rows}")  # fmt: skip
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
    outs = gen.generate_batch(prompts, OSL, sampling=sampling, stop_on_eos=stop, enable_trace=True, mm=mm)
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
    import hashlib as _hl  # every user's generated ids, whole (the MK_E2E line shows user 0's first 24 only)

    logger.info(
        f"MK_E2E_ALL_IDS: sha256 {_hl.sha256(repr([list(o) for o in outs]).encode()).hexdigest()[:16]} "
        f"({len(outs)} users, {sum(len(o) for o in outs)} ids)"
    )
    logger.info(f"MK_E2E counters: {gen.counters_report()}")
    assert n_gen >= 1
    if mm_in is not None and sampling is None:  # the image gate: the HF greedy reference's ids
        exp = mm_in["expect"]
        got = outs[0][: len(exp)]
        tw = mm_in.get("tower_check")
        logger.info(f"MK_E2E_IMAGE: ids {got} vs HF {exp}: first {got[:1] == exp[:1]}, all {got == exp} "
                    f"({sum(a == b_ for a, b_ in zip(got, exp))}/{len(exp)} equal); rows from the "
                    f"{'device tower (' + format(tw['tower_s'] * 1e3, '.0f') + ' ms)' if tw else 'HF CPU file'}")  # fmt: skip
        if got != exp:
            try:
                for step, ref, tk in image_topk_report(gen, mm_in):
                    logger.info(f"MK_E2E_IMAGE top-5 step {step}: HF {ref} -> {tk}")
            except Exception as e:  # noqa: BLE001  (a diagnostic never hides the gate below)
                logger.warning(f"MK_E2E_IMAGE top-5 report failed: {type(e).__name__}: {e}")
        tower_gates(mm_in)
        assert got[:1] == exp[:1], f"first token {got[:1]} != the HF reference's {exp[:1]}"
        assert got == exp, f"ids {got} != the HF reference {exp}"
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
    spec_k = int(os.environ.get("GLM53_SPEC_K", "0") or 0)
    if spec_k > 0 and getattr(backend, "spec", None) is not None and backend.spec.drafter == "mtp":
        G = gen.spec_groups()  # GLM53_SPEC_MULTI: the backend's step serves G users (0: one user)
        if G and live == G and sampling is None and mm_in is None:
            _spec_ab_multi(gen, backend, mesh_device, tok, prompts, outs, timing, step_ms, spec_k, stop)
        elif not G and live == 1 and sampling is None and mm_in is None:
            _spec_ab(gen, backend, mesh_device, tok, prompts[0], outs[0], timing, step_ms, spec_k, stop)
        else:  # (an image prompt never speculates: the MTP-cache prefill embeds token ids, not the feature rows)
            logger.info(
                f"MK_SPEC: skipped (greedy text users: GLM53_MK_LIVE=1, or = GLM53_SPEC_MULTI_USERS {G} with "
                f"GLM53_SPEC_MULTI=1; no GLM53_MK_TEMP, no image; live {live})"
            )
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
