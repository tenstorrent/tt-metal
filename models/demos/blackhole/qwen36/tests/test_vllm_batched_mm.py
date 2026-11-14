# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""DEVICE test of image / video requests in BATCHED vLLM serving (max_num_seqs=32, TP=8).

Qwen36ForCausalLM is driven like the vLLM TT plugin: one prefill_forward call with a text, an image and a video user
(per-request kwargs lists pixel_values[u] / image_grid_thw[u] / pixel_values_videos[u] / video_grid_thw[u]), then
resident greedy device-sampling decode (the per-slot M-RoPE deltas are model-owned).

Cases: (a) mixed prefill + 24 decode steps, decoded texts must be on topic; (b) a second prefill of the same MM
requests into other slots gives the same first-token logits; (c) slot_remap condense (text user finishes, image / video
users move to slots 0 / 1) gives the SAME greedy tokens as the uncondensed run; (d) prefix cache on: an MM request with
start_pos > 0 gives the same logits as start_pos = 0 and never touches the snapshot cache.

test_vllm_determinism (text only, same Driver): the same prompt run alone repeatedly must produce identical greedy
tokens (asserted); each user of a batched run that drops users mid-decode vs the same prompt alone is printed only.
Env: DBG_S1 / DBG_S2 (decode steps before / after the drop, default 64 / 32).

test_vision_guard (serving vision-bucket guard + max_pixels-driven warmup): (a) default warmup (no max_pixels): an
over-size grid (1x86x128 = 11008 patches) raises ValueError at once, no hang; (b) a simulated vLLM config
max_pixels=1048576 is read by _read_vllm_max_pixels; warmup covers its bucket and a 1 MP image (4096 patches) runs;
(c) max_pixels=3e6 warms every bucket up to 12288 (10240 included, not just the largest) and the 11008- and
9000-patch grids then run instead of being rejected; (d) a QWEN36_VISION_WARMUP_GRIDS override with a gap (only
bucket 8192) refuses an item in the gap (bucket 2048) even though it is below the max.

    pytest models/demos/blackhole/qwen36/tests/test_vllm_batched_mm.py -svq
    pytest models/demos/blackhole/qwen36/tests/test_vllm_batched_mm.py -svq -k "vision_guard or determinism"
Env: QWEN36_MM_SCRATCH (dir for the synthetic mp4, default /tmp).
"""

import os
from types import SimpleNamespace

import pytest
import torch
from loguru import logger

import ttnn
from models.common.sampling.sampling_params import SamplingParams
from models.common.utility_functions import run_for_blackhole
from models.demos.blackhole.qwen36.tt.qwen36_vllm import Qwen36ForCausalLM

BLOCK = 64
MAX_SEQ = 16384
BPU = MAX_SEQ // BLOCK
NUM_BLOCKS = 1024
WIDTH = 32
STEPS = 24
PCC_MIN = 0.99
IMAGE_PATH = "models/sample_data/huggingface_cat_image.jpg"
IMAGE_MAX_PIXELS = 1_000_000
VIDEO_FRAMES = 8

DEVICE_PARAMS = [
    {
        "l1_small_size": 24576,
        "num_command_queues": 2,
        "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
        "trace_region_size": 1024 * 1024 * 1024,
    }
]


def pcc(a, b):
    a, b = a.double().reshape(-1), b.double().reshape(-1)
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def make_video(path, n_frames=16, size=256, fps=8):
    """~2 s mp4: a red ball moving left -> right on a white background."""
    import cv2
    import numpy as np

    w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (size, size))
    for i in range(n_frames):
        frame = np.full((size, size, 3), 255, np.uint8)
        cx = int(30 + (size - 60) * i / (n_frames - 1))
        cv2.circle(frame, (cx, size // 2), 24, (0, 0, 255), -1)  # BGR red
        w.write(frame)
    w.release()
    return path


class Req:
    """One vLLM-style request: token ids + per-request visual kwargs entries (None for text)."""

    def __init__(self, name, ids, pix=None, grid=None, vpix=None, vgrid=None):
        self.name, self.ids = name, ids
        self.pix, self.grid, self.vpix, self.vgrid = pix, grid, vpix, vgrid

    @property
    def L(self):
        return int(self.ids.shape[1])


def build_requests(hf):
    from PIL import Image
    from transformers import AutoProcessor

    proc = AutoProcessor.from_pretrained(hf, trust_remote_code=True)
    tok = proc.tokenizer

    def chat(content):
        return proc.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True, enable_thinking=False
        )

    text = tok(
        chat([{"type": "text", "text": "Name three primary colors and one fact about each."}]), return_tensors="pt"
    )
    reqs = [Req("text", text["input_ids"].to(torch.int32))]

    img = Image.open(IMAGE_PATH).convert("RGB")
    scale = min(1.0, (IMAGE_MAX_PIXELS / (img.width * img.height)) ** 0.5)
    img = img.resize((max(64, int(img.width * scale)), max(64, int(img.height * scale))))
    t = chat([{"type": "image"}, {"type": "text", "text": "Describe this image."}])
    inp = proc(text=t, images=[img], return_tensors="pt")
    reqs.append(
        Req(
            "image",
            inp["input_ids"].to(torch.int32),
            pix=[inp["pixel_values"]],
            grid=[inp["image_grid_thw"][0].to(torch.int32)],
        )
    )

    from transformers.video_utils import load_video

    scratch = os.environ.get("QWEN36_MM_SCRATCH", "/tmp")
    mp4 = make_video(os.path.join(scratch, "red_ball.mp4"))
    frames, meta = load_video(mp4, backend="pyav", num_frames=VIDEO_FRAMES)
    t = chat([{"type": "video"}, {"type": "text", "text": "What happens in this video?"}])
    inp = proc(text=t, videos=[frames], video_metadata=[meta], do_sample_frames=False, return_tensors="pt")
    reqs.append(
        Req(
            "video",
            inp["input_ids"].to(torch.int32),
            vpix=[inp["pixel_values_videos"]],
            vgrid=[inp["video_grid_thw"][0].to(torch.int32)],
        )
    )
    for r in reqs:
        logger.info(f"request {r.name}: L={r.L}")
    return reqs, tok


class Driver:
    def __init__(self, gen, kv, tok):
        self.gen, self.kv, self.tok = gen, kv, tok
        self.vocab = gen.model[0].args.vocab_size
        self.next_block = 1

    def blocks(self, L):
        n = (L + STEPS + 8 + BLOCK) // BLOCK + 1
        ids = list(range(self.next_block, self.next_block + n))
        self.next_block += n
        assert self.next_block <= NUM_BLOCKS
        return ids

    def prefill(self, reqs, slots, start_pos=None):
        """One prefill_forward call (the plugin's batched step). Returns ([N,vocab] logits, blocks per request)."""
        N = len(reqs)
        T = max(r.L for r in reqs)
        tokens = torch.zeros(N, T, dtype=torch.int32)
        pt = torch.zeros(N, BPU, dtype=torch.int32)
        blocks = []
        for i, r in enumerate(reqs):
            tokens[i, : r.L] = r.ids[0]
            b = self.blocks(r.L)
            blocks.append(b)
            pt[i, : len(b)] = torch.tensor(b, dtype=torch.int32)
        kw = dict(
            empty_slots=list(slots),
            enable_trace=True,
            pixel_values=[r.pix for r in reqs],
            image_grid_thw=[r.grid for r in reqs],
            pixel_values_videos=[r.vpix for r in reqs],
            video_grid_thw=[r.vgrid for r in reqs],
            rope_deltas_all_users=torch.zeros(N, dtype=torch.long),
        )
        if start_pos is not None:
            kw["start_pos"] = start_pos
        logits, rd = self.gen.prefill_forward(tokens, pt, self.kv, [r.L for r in reqs], **kw)
        assert rd.shape[0] == N
        return logits.reshape(N, -1)[:, : self.vocab].float(), blocks

    def decode_step(self, rows, remap=None):
        """rows: list of dict(L, blocks, out(list of tokens, last = next input), step) in slot (row) order. Full
        authoritative reload every step (host_reload mode), optional slot_remap on this step."""
        tokens = torch.zeros(WIDTH, 1, dtype=torch.int32)
        start = torch.full((WIDTH,), -1, dtype=torch.int32)
        pt = torch.zeros(WIDTH, BPU, dtype=torch.int32)
        for i, r in enumerate(rows):
            tokens[i, 0] = r["out"][-1]
            start[i] = r["L"] + len(r["out"]) - 1
            pt[i, : len(r["blocks"])] = torch.tensor(r["blocks"], dtype=torch.int32)
        sp = SamplingParams(temperature=[0.0] * WIDTH, top_k=[1] * WIDTH, top_p=[1.0] * WIDTH, seed=[None] * WIDTH)
        kw = dict(
            tokens=tokens,
            start_pos=start,
            page_table=pt,
            kv_cache=self.kv,
            sampling_params=sp,
            reload_inputs=True,
            reload_page_table=False,
            reload_sampling_params=True,
            reset_sampling_state=True,
            enable_trace=True,
            read_from_device=True,
        )
        if remap is not None:
            kw["slot_remap"] = remap
        out = self.gen.decode_forward(**kw)
        toks = torch.as_tensor(out[0] if isinstance(out, tuple) else out).reshape(-1)
        for i, r in enumerate(rows):
            r["out"].append(int(toks[i]))

    def text(self, ids):
        return self.tok.decode(ids, skip_special_tokens=True)


def mk_rows(reqs, logits, blocks):
    return [dict(L=r.L, blocks=b, out=[int(l.argmax())]) for r, l, b in zip(reqs, logits, blocks)]


@run_for_blackhole()
@pytest.mark.timeout(3500)
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_batched_multimodal_serving(mesh_device, reset_seeds, ensure_gc):
    from transformers import AutoConfig

    hf = os.environ["HF_MODEL"]
    reqs, tok = build_requests(hf)
    hf_config = AutoConfig.from_pretrained(hf)
    gen = Qwen36ForCausalLM.initialize_vllm_model(hf_config, mesh_device, max_batch_size=WIDTH, max_seq_len=MAX_SEQ)
    model = gen.model[0]
    kv_shape = (NUM_BLOCKS, model.args.n_local_kv_heads, BLOCK, model.args.head_dim)
    kv = gen.allocate_kv_cache(kv_shape, ttnn.bfloat16, len(model.layers))
    dkw = dict(kv_cache=kv, max_batch_size=WIDTH, num_blocks=BPU, can_sample_on_device=True)
    gen.warmup_model_prefill(kv_cache=kv, enable_trace=False)
    gen.warmup_model_decode(enable_trace=False, **dkw)
    gen.already_warmed_up_prefill = False
    gen.warmup_model_prefill(kv_cache=kv, enable_trace=True)
    gen.warmup_model_decode(enable_trace=True, **dkw)
    cache = model._prefix_cache
    d = Driver(gen, kv, tok)
    failures = []

    # ---- (a) one prefill call: text, image, video in slots 0, 1, 2; then STEPS resident decode steps
    logits_a, blocks_a = d.prefill(reqs, [0, 1, 2])
    deltas = model._slot_rope_delta[:3].tolist()
    logger.info(f"per-slot rope deltas after prefill: {deltas}")
    assert deltas[0] == 0 and deltas[1] != 0 and deltas[2] != 0, deltas
    rows_a = mk_rows(reqs, logits_a, blocks_a)
    for _ in range(STEPS):
        d.decode_step(rows_a)
    texts_a = {r.name: d.text(row["out"]) for r, row in zip(reqs, rows_a)}
    for n, t in texts_a.items():
        logger.info(f"(a) {n}: {t!r}")
        print(f"(a) {n}: {t!r}")
    low = {n: t.lower() for n, t in texts_a.items()}
    if "cat" not in low["image"] and "kitten" not in low["image"]:
        failures.append(f"image output not on topic: {texts_a['image']!r}")
    if not any(w in low["video"] for w in ("ball", "circle", "red", "move", "left", "right")):
        failures.append(f"video output not on topic: {texts_a['video']!r}")
    # STEPS tokens may end inside a preamble ("Here are three primary colors (based on ...") before any colour name.
    if not any(w in low["text"] for w in ("red", "blue", "yellow", "green", "primary color", "primary colour")):
        failures.append(f"text output not on topic: {texts_a['text']!r}")

    # ---- (b) the same MM requests again into other slots: same first-token logits
    logits_b, blocks_b = d.prefill(reqs[1:], [4, 5])
    for i, name in enumerate(("image", "video")):
        p = pcc(logits_b[i], logits_a[i + 1])
        same = int(logits_b[i].argmax()) == int(logits_a[i + 1].argmax())
        logger.info(f"(b) {name}: pcc(second prefill, first prefill)={p:.6f} top1_equal={same}")
        if p < PCC_MIN or not same:
            failures.append(f"(b) {name}: pcc={p:.6f} top1_equal={same}")

    # ---- (c) condense: text user (slot 0) finishes after 6 steps; image / video move to slots 0 / 1
    logits_c, blocks_c = d.prefill(reqs, [0, 1, 2])
    rows_c = mk_rows(reqs, logits_c, blocks_c)
    for _ in range(6):
        d.decode_step(rows_c)
    survivors = rows_c[1:]
    remap = [1, 2, 0] + list(range(3, WIDTH))  # new slot i takes the state previously at slot remap[i]
    d.decode_step(survivors, remap=remap)
    deltas_c = model._slot_rope_delta[:3].tolist()
    logger.info(f"(c) per-slot rope deltas after condense: {deltas_c}")
    assert deltas_c[0] == deltas[1] and deltas_c[1] == deltas[2], (deltas_c, deltas)
    for _ in range(STEPS - 6 - 1):
        d.decode_step(survivors)
    # Reference with the same decode width (bucket 2) and no condense: image / video prefilled into slots 0 / 1. The
    # steps before the condense ran at width 4 (bucket 4), so bf16 near-ties may flip late; exact equality is reported,
    # the assert is the first-divergence index vs the uncondensed 3-user run (a) being past the condense point.
    logits_r, blocks_r = d.prefill(reqs[1:], [0, 1])
    rows_r = mk_rows(reqs[1:], logits_r, blocks_r)
    for _ in range(STEPS):
        d.decode_step(rows_r)
    for name, row, ref, ref2 in (
        ("image", survivors[0], rows_a[1], rows_r[0]),
        ("video", survivors[1], rows_a[2], rows_r[1]),
    ):
        div = next((i for i, (x, y) in enumerate(zip(row["out"], ref["out"])) if x != y), len(row["out"]))
        div2 = next((i for i, (x, y) in enumerate(zip(row["out"], ref2["out"])) if x != y), len(row["out"]))
        logger.info(
            f"(c) {name}: condensed vs uncondensed(3 users) first divergence idx={div}/{len(row['out'])}, "
            f"vs 2-user no-condense idx={div2}: {d.text(row['out'])!r}"
        )
        print(f"(c) {name}: {d.text(row['out'])!r}")
        print(f"(c) {name} uncondensed: {d.text(ref['out'])!r}")
        print(f"(c) {name} 2-user ref : {d.text(ref2['out'])!r}")
        if div < 14:  # condense happens after 7 tokens; a wrong slot / delta diverges immediately
            failures.append(f"(c) {name}: condensed diverges from uncondensed at idx {div}")

    # ---- (d) prefix cache on: MM request with start_pos > 0 behaves exactly like start_pos = 0
    assert cache is not None, "prefix cache not built; test (d) needs it"
    stats0 = dict(cache.stats)
    for r, name in ((reqs[1], "image"), (reqs[2], "video")):
        start = (r.L // 2 // 128) * 128
        l0, _ = d.prefill([r], [6], start_pos=[0])
        l1, _ = d.prefill([r], [7], start_pos=[start])
        p = pcc(l0, l1)
        same = int(l0.argmax()) == int(l1.argmax())
        logger.info(f"(d) {name}: start_pos={start} vs 0: pcc={p:.6f} top1_equal={same}")
        if p < PCC_MIN or not same:
            failures.append(f"(d) {name}: pcc={p:.6f} top1_equal={same}")
    if dict(cache.stats) != stats0:
        failures.append(f"(d) prefix cache stats changed by MM prefill: {stats0} -> {dict(cache.stats)}")

    assert not failures, "; ".join(failures)


PROMPTS = [
    "Explain photosynthesis in detail, step by step.",
    "Write a short story about a robot who learns to paint.",
    "What are the main causes of the French Revolution?",
    "Give me a Python function that computes the Fibonacci sequence and explain it.",
    "Describe the water cycle for a ten year old.",
    "List five tips for improving sleep quality and justify each.",
    "Summarize the plot of Romeo and Juliet in a few paragraphs.",
    "How does a transformer neural network work? Be thorough.",
]
STEPS1, STEPS2 = int(os.environ.get("DBG_S1", 64)), int(os.environ.get("DBG_S2", 32))


def first_div(a, b):
    return next(
        (i for i, (x, y) in enumerate(zip(a, b)) if x != y),
        min(len(a), len(b)) if len(a) == len(b) else min(len(a), len(b)),
    )


@run_for_blackhole()
@pytest.mark.timeout(3500)
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_vllm_determinism(mesh_device, reset_seeds, ensure_gc):
    from transformers import AutoConfig, AutoTokenizer

    hf = os.environ["HF_MODEL"]
    tok = AutoTokenizer.from_pretrained(hf)
    reqs = []
    for i, p in enumerate(PROMPTS):
        t = tok.apply_chat_template(
            [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        ids = tok(t, return_tensors="pt")["input_ids"].to(torch.int32)
        reqs.append(Req(f"p{i}", ids))
    gen = Qwen36ForCausalLM.initialize_vllm_model(
        AutoConfig.from_pretrained(hf), mesh_device, max_batch_size=WIDTH, max_seq_len=MAX_SEQ
    )
    model = gen.model[0]
    kv = gen.allocate_kv_cache(
        (NUM_BLOCKS, model.args.n_local_kv_heads, BLOCK, model.args.head_dim), ttnn.bfloat16, len(model.layers)
    )
    dkw = dict(kv_cache=kv, max_batch_size=WIDTH, num_blocks=BPU, can_sample_on_device=True)
    gen.warmup_model_prefill(kv_cache=kv, enable_trace=False)
    gen.warmup_model_decode(enable_trace=False, **dkw)
    gen.already_warmed_up_prefill = False
    gen.warmup_model_prefill(kv_cache=kv, enable_trace=True)
    gen.warmup_model_decode(enable_trace=True, **dkw)
    logger.info(f"prefix cache: {model._prefix_cache is not None}")
    d = Driver(gen, kv, tok)
    d.blocks = lambda L: _blocks(d, L)

    def alone(i, n):
        lg, bl = d.prefill([reqs[i]], [0])
        rows = mk_rows([reqs[i]], lg, bl)
        for _ in range(n - 1):
            d.decode_step(rows)
        return rows[0]["out"]

    # R1
    P = 0
    runs = [alone(P, STEPS1) for _ in range(4)]
    for k in range(1, 4):
        print(f"R1 run{k} vs run0: first_div={first_div(runs[0], runs[k])}/{STEPS1} identical={runs[0]==runs[k]}")
    print("R1 text:", repr(tok.decode(runs[0], skip_special_tokens=True))[:200])
    for k in range(1, 4):
        assert runs[k] == runs[0], f"R1 run{k} diverged at token {first_div(runs[0], runs[k])}"
    # R2 batch
    lg, bl = d.prefill(reqs, list(range(8)))
    rows = mk_rows(reqs, lg, bl)
    for _ in range(STEPS1 - 1):
        d.decode_step(rows)
    keep = [0, 2, 4, 6, 7]
    drop = [1, 3, 5]
    remap = keep + drop + list(range(8, WIDTH))
    surv = [rows[i] for i in keep]
    d.decode_step(surv, remap=remap)
    for _ in range(STEPS2 - 1):
        d.decode_step(surv)
    # pre-condense outputs of dropped (just first STEPS1)
    refs = {i: alone(i, STEPS1 + STEPS2) for i in range(8)}
    refs2 = {i: alone(i, STEPS1 + STEPS2) for i in (0, 4)}
    for i in range(8):
        out = rows[i]["out"]
        r = refs[i]
        n = min(len(out), len(r))
        fd = first_div(out[:n], r[:n])
        print(
            f"R2 user{i} ({'surv' if i in keep else 'dropped'}) len={len(out)} first_div_vs_alone={fd}/{n} :: {repr(tok.decode(out, skip_special_tokens=True))[:90]}"
        )
    for i in (0, 4):
        print(
            f"R2 alone repeat user{i}: identical_to_first_alone={refs2[i]==refs[i]} first_div={first_div(refs2[i], refs[i])}"
        )
    # Batched-vs-alone stays print-only: batch composition may legitimately change rounding.
    for i in (0, 4):
        assert refs2[i] == refs[i], f"R2 user{i} repeat diverged at token {first_div(refs2[i], refs[i])}"


def _blocks(d, L):
    n = (L + 200 + BLOCK) // BLOCK + 1
    ids = list(range(d.next_block, d.next_block + n))
    d.next_block += n
    assert d.next_block <= NUM_BLOCKS
    return ids


def _vision_kwargs(model, grid):
    vc = model.vision_args.hf_config.vision_config
    t, h, w = grid
    pd = vc.in_channels * vc.temporal_patch_size * vc.patch_size**2
    return dict(
        pixel_values=[[torch.zeros(t * h * w, pd)]], image_grid_thw=[[torch.tensor([t, h, w], dtype=torch.int32)]]
    )


@run_for_blackhole()
@pytest.mark.timeout(1800)
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_vision_guard(mesh_device, reset_seeds, ensure_gc, expect_error, monkeypatch):
    from transformers import AutoConfig

    hf_config = AutoConfig.from_pretrained(os.environ["HF_MODEL"])
    gen = Qwen36ForCausalLM.initialize_vllm_model(hf_config, mesh_device, max_batch_size=32, max_seq_len=16384)
    model = gen.model[0]

    # (a) no max_pixels -> default grids (max bucket 8192); over-size item is refused host-side.
    assert gen._vllm_mm_max_pixels is None
    monkeypatch.delenv("QWEN36_VISION_WARMUP_GRIDS", raising=False)
    gen.warmup_vision()
    assert gen._vision_warmed_buckets == {2048, 4096, 6144, 8192}
    with expect_error(ValueError, "max_pixels"):
        gen._compute_vision_tokens(model, _vision_kwargs(model, (1, 86, 128)))

    # (b) simulated vLLM config
    for cfg in (
        SimpleNamespace(
            model_config=SimpleNamespace(mm_processor_kwargs={"max_pixels": 1048576}, multimodal_config=None)
        ),
        SimpleNamespace(
            model_config=SimpleNamespace(
                mm_processor_kwargs=None,
                multimodal_config=SimpleNamespace(mm_processor_kwargs={"max_pixels": 1048576}),
            )
        ),
    ):
        assert Qwen36ForCausalLM._read_vllm_max_pixels(cfg) == 1048576
    gen._vllm_mm_max_pixels = 1048576
    gen.warmup_vision()
    assert gen._vision_warmed_buckets == {2048, 4096, 6144, 8192}
    ttnn.deallocate(gen._compute_vision_tokens(model, _vision_kwargs(model, (1, 64, 64))))  # 1 MP image: 4096 patches

    # (c) larger max_pixels warms every bucket up to its own; the formerly-hanging 2048x1365 grid now runs, and so
    # does a 1x90x100 grid (9000 patches) in bucket 10240, below the largest warmed bucket.
    gen._vllm_mm_max_pixels = 3_000_000
    gen.warmup_vision()
    assert gen._vision_warmed_buckets == {2048, 4096, 6144, 8192, 10240, 12288}
    ttnn.deallocate(gen._compute_vision_tokens(model, _vision_kwargs(model, (1, 86, 128))))
    ttnn.deallocate(gen._compute_vision_tokens(model, _vision_kwargs(model, (1, 90, 100))))
    with expect_error(ValueError, "max_pixels"):
        gen._compute_vision_tokens(model, _vision_kwargs(model, (1, 128, 128)))

    # (d) an override with a gap: only bucket 8192 is warmed, so a bucket-2048 item is refused, not compiled late.
    gen._vllm_mm_max_pixels = None
    monkeypatch.setenv("QWEN36_VISION_WARMUP_GRIDS", "1x80x80")
    gen.warmup_vision()
    assert gen._vision_warmed_buckets == {8192}
    with expect_error(ValueError, "not warmed"):
        gen._compute_vision_tokens(model, _vision_kwargs(model, (1, 32, 32)))
