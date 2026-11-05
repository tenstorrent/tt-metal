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
from models.autoports.zai_org_glm_5_3_flash.tt import optimized_decoder as OD
from models.autoports.zai_org_glm_5_3_flash.tt.config import load_hf_text_config
from models.demos.glm53_flash.reference.checkpoint import default_snapshot_dir
from models.demos.glm53_flash.tests.test_mk_backend_probe import HFSource
from models.demos.glm53_flash.tt.mesh import mesh_device_params

MESH = tuple(int(v) for v in os.environ.get("GLM53_MESH_SHAPE", "8,4").split(","))
B = int(os.environ.get("GLM53_MK_B", "4"))
ISL = int(os.environ.get("GLM53_MK_ISL", "128"))
OSL = int(os.environ.get("GLM53_MK_OSL", "128"))
SEQ = int(os.environ.get("GLM53_MK_SEQ", str(max(2048, 1 << (ISL + OSL + 64).bit_length()))))
TRACE_MB = int(os.environ.get("GLM53_MODEL_TRACE_MB", "512"))
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
    [mesh_device_params(*MESH, trace_region_size=TRACE_MB << 20, worker_l1_size=WORKER_L1)],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [MESH], indirect=True)
def test_mk_e2e(mesh_device, monkeypatch):
    from models.autoports.zai_org_glm_5_3_flash.tt.generator import Glm53Generator
    from models.autoports.zai_org_glm_5_3_flash.tt.stack_model import StackBackend
    from models.autoports.zai_org_glm_5_3_flash.tt.tokenizer_chat import load_tokenizer

    snapshot = default_snapshot_dir()
    cfg = load_hf_text_config(str(snapshot))
    tok = load_tokenizer(snapshot)
    source = HFSource(snapshot, cfg)
    monkeypatch.setattr(OD, "FdTileCacheSource", lambda cache_dir, manifest: source)
    layers = _layers()
    ep = os.environ.get("GLM53_MK_EP", "1") == "1"
    native = os.environ.get("GLM53_MK_NATIVE_T", "1" if B > 4 else "0") == "1"
    t0 = time.perf_counter()
    backend = StackBackend(mesh_device, max_batch=B, layers=layers, hf_config=cfg, topk=1, sample=False, ep_weights=ep,
                           native_t=native)  # fmt: skip
    logger.info(f"StackBackend built in {time.perf_counter() - t0:.0f} s: mesh {MESH}, {len(layers)} layers, B {B}")
    gen = Glm53Generator(backend, tokenizer=tok, max_batch=B, max_seq_len=SEQ)
    t0 = time.perf_counter()
    gen.warmup(capture=True)
    ttnn.synchronize_device(mesh_device)
    logger.info(f"generator warm-up (prefill buckets, decode, traces) {time.perf_counter() - t0:.1f} s")
    ids = chat_prompt(tok, ISL)
    prompts = [list(chat_prompt(tok, ISL, user=u)) for u in range(B)]
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
        f"MK_E2E: mesh {MESH} {len(layers)} layers B {B} ISL {len(ids)} OSL {OSL}: TTFT {ttft_s * 1e3:.0f} ms "
        f"(all {B} users), decode {step_ms:.1f} ms/step = {1e3 / step_ms:.1f} tok/s/u, generated {n_gen}, "
        f"users agree {same}, first {first[0][:1]} ids {outs[0][:24]}"
    )
    logger.info(f"MK_E2E_TEXT: {text!r}")
    logger.info(f"MK_E2E counters: {gen.counters_report()}")
    assert n_gen >= 1
