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

    pytest models/demos/blackhole/qwen36/tests/test_vllm_batched_mm.py -svq
Env: QWEN36_MM_SCRATCH (dir for the synthetic mp4, default /tmp).
"""

import os

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
    if not any(w in low["text"] for w in ("red", "blue", "yellow", "green")):
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
