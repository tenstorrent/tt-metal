# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""DEVICE test of image / video requests on the MTP spec-decode class (Qwen36MTPContractForCausalLM), batched.

The class is driven like the vLLM TT plugin runner (no server): initialize_vllm_model -> allocate_kv_cache -> 2-phase
warmup -> prefill_forward (per-request visual kwargs lists pixel_values[u] / image_grid_thw[u] /
pixel_values_videos[u] / video_grid_thw[u], exactly as tests/test_vllm_batched_mm.py) -> decode_forward as a verify
(spec_mode argmax_ids, or fused_sample when QWEN36_SPEC_FUSED=1) -> propose_draft_tokens -> release_request.
Same assets as the non-spec test (cat image, synthetic red-ball mp4, text prompt); greedy only.

Scenarios (one pytest function each, so each runs as one node id):
  test_spec_mm_batch3     (a) B=3: text + image + video prefilled in slots 0,1,2, N=24 tokens each through the spec
                          contract; image / video outputs must be on topic (the text keyword check is skipped, it is
                          known to fail; the text output is only logged).
  test_nonspec_mm_reference  (b, part 1) the same three requests through the NON-spec class (Qwen36ForCausalLM,
                          resident greedy decode, as test_vllm_batched_mm.py); writes the token streams to
                          $QWEN36_MM_REF (default /tmp/qwen36_mm_nonspec_ref.json).
  test_spec_mm_lossless   (b, part 2) the MTP class on the same requests; token streams compared with that file: first
                          divergence index per user is reported, >= 16 identical tokens per user are required.
                          Two 27B models do not fit one mesh, so (b) is two processes: run part 1 first, then part 2.
                          Near-tie handling: the non-spec device-sampled decode returns no logits, so there is no
                          top-2 gap (test_spec_lossless.NEAR_TIE_GAP is used only to label the report); a flip after
                          >= 16 tokens is reported, not failed.
  test_spec_mm_join       (c) the text user decodes alone, the image user joins mid-decode (prefill while the text
                          user is live), then the video user joins; outputs must stay on topic (image / video) and the
                          text user must keep decoding (same tokens as when it ran alone is NOT asserted: bf16 ties).

Run (one node id per process; needs a P150x8 box, MESH_DEVICE=P150x8):
    source work/env.sh      # profile + tt-metal python_env; sets TT_METAL_RUNTIME_ROOT, PYTHONPATH
    export HF_MODEL=<Qwen3.8-27B checkpoint dir> TT_CACHE_PATH=<tensor cache dir> HF_HUB_OFFLINE=1 MESH_DEVICE=P150x8
    export QWEN36_SPEC_K=3 QWEN36_SPEC_FAST_PREFILL=1 QWEN36_SPEC_FUSED=1 QWEN36_SPEC_CHAIN=1 \\
           QWEN36_SPEC_JOIN_TRACE=1 QWEN36_SPEC_BUCKETS=1,2,4,8
    pytest models/demos/blackhole/qwen36/tests/test_spec_vllm_mm.py::test_spec_mm_batch3 -svq
    pytest models/demos/blackhole/qwen36/tests/test_spec_vllm_mm.py::test_nonspec_mm_reference -svq   # then:
    pytest models/demos/blackhole/qwen36/tests/test_spec_vllm_mm.py::test_spec_mm_lossless -svq
    pytest models/demos/blackhole/qwen36/tests/test_spec_vllm_mm.py::test_spec_mm_join -svq
Flags are read from the process ENVIRONMENT (not test args): QWEN36_SPEC_K at import of qwen36_vllm (class attr
_SPEC_K); SPEC_FUSED / SPEC_CHAIN (tp_common.spec_*_enabled, also the model capabilities), SPEC_FAST_PREFILL, SPEC_BUCKETS,
SPEC_JOIN_TRACE by the engine, as in the server. With QWEN36_SPEC_FUSED=1 the verify step uses spec_mode
"fused_sample" (non-deferred, host walk of the device commit); otherwise "argmax_ids" with the host greedy walk.
Other env: QWEN36_MM_SCRATCH (dir for the synthetic mp4, default /tmp), QWEN36_MM_REF (reference file),
QWEN36_MM_WIDTH (max_num_seqs, default 8). Only one device test per run (a P300 cache build can fill the disk).
"""

import json
import math
import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.blackhole.qwen36.demo.text_demo import _MESH_SHAPE, _MULTI, DEVICE_PARAMS
from models.demos.blackhole.qwen36.tests.test_spec_lossless import NEAR_TIE_GAP
from models.demos.blackhole.qwen36.tests.test_vllm_batched_mm import DEVICE_PARAMS as NONSPEC_DEVICE_PARAMS
from models.demos.blackhole.qwen36.tests.test_vllm_batched_mm import Driver as NonSpecDriver
from models.demos.blackhole.qwen36.tests.test_vllm_batched_mm import build_requests, mk_rows
from models.demos.blackhole.qwen36.tt.qwen36_vllm import Qwen36ForCausalLM, Qwen36MTPContractForCausalLM
from models.demos.blackhole.qwen36.tt.tp_common import spec_fused_enabled

PAD = -1
BLOCK = 64
K = int(os.environ.get("QWEN36_SPEC_K", "3"))
MAX_SEQ = 8192
BPU = math.ceil(MAX_SEQ / BLOCK)
NUM_BLOCKS = 640
WIDTH = int(os.environ.get("QWEN36_MM_WIDTH", "8"))
N_TOK = 24  # tokens per user, scenarios (a) and (c)
N_LOSSLESS = 32
MIN_IDENTICAL = 16
REF_PATH = os.environ.get("QWEN36_MM_REF", "/tmp/qwen36_mm_nonspec_ref.json")
IMAGE_WORDS = ("cat", "kitten")
VIDEO_WORDS = ("ball", "circle", "red", "move", "left", "right")


class SpecDriver:
    """Host mirror of the plugin runner on the MTP class: row == state slot, rows past the live ones are padding."""

    def __init__(self, cls, kv, tok):
        self.cls, self.kv, self.tok = cls, kv, tok
        self.rows = [None] * WIDTH
        self.next_block = 1
        self.fused = spec_fused_enabled()
        self.steps = 0
        self.accepted = 0  # drafts accepted
        self.drafted = 0  # drafts verified

    def _blocks_upto(self, row, last_pos):
        """Grow the row's blocks to cover last_pos + (1+K) + (K+1) lookahead; returns the [BPU] int32 table row."""
        need = min(BPU, math.ceil((last_pos + (1 + K) + (K + 1)) / BLOCK))
        while len(row["blocks"]) < need:
            row["blocks"].append(self.next_block)
            self.next_block += 1
            assert self.next_block < NUM_BLOCKS - 1, "KV block pool exhausted"
        return torch.tensor(row["blocks"] + [0] * (BPU - len(row["blocks"])), dtype=torch.int32)

    def live(self):
        return [s for s in range(WIDTH) if self.rows[s] is not None]

    def admit(self, reqs, slots, max_new):
        """One prefill_forward call (the plugin's batched step) for reqs into slots; first token = argmax of logits."""
        N = len(reqs)
        T = max(r.L for r in reqs)
        tokens = torch.zeros(N, T, dtype=torch.int32)
        pt = torch.zeros(N, BPU, dtype=torch.int32)
        new_rows = []
        for i, r in enumerate(reqs):
            tokens[i, : r.L] = r.ids[0]
            row = {"name": r.name, "P": r.L, "blocks": [], "out": [], "proposal": [], "acc": 1, "max_new": max_new}
            pt[i] = self._blocks_upto(row, r.L - 1)
            new_rows.append(row)
        logits, rd = self.cls.prefill_forward(
            tokens,
            pt,
            self.kv,
            torch.tensor([r.L for r in reqs]),
            empty_slots=list(slots),
            enable_trace=True,
            pixel_values=[r.pix for r in reqs],
            image_grid_thw=[r.grid for r in reqs],
            pixel_values_videos=[r.vpix for r in reqs],
            video_grid_thw=[r.vgrid for r in reqs],
            rope_deltas_all_users=torch.zeros(N, dtype=torch.long),
            start_pos=torch.zeros(N, dtype=torch.long),
        )
        assert tuple(logits.shape[:2]) == (N, 1) and rd.shape[0] == N, (tuple(logits.shape), tuple(rd.shape))
        for i, (s, row) in enumerate(zip(slots, new_rows)):
            assert self.rows[s] is None, f"slot {s} is live"
            row["out"] = [int(logits[i, -1].argmax())]
            self.rows[s] = row

    def _finish(self, s):
        self.cls.release_request(s)
        self.rows[s] = None

    def step(self):
        """One verify step over every live row, then the drafter's proposal for the next one."""
        cls = self.cls
        live = self.live()
        tokens = torch.full((WIDTH, 1 + K), PAD, dtype=torch.int32)
        start = torch.full((WIDTH, 1 + K), PAD, dtype=torch.int32)
        nvd = torch.zeros(WIDTH, dtype=torch.int32)
        acc = torch.ones(WIDTH, dtype=torch.int32)
        pt = torch.zeros((WIDTH, BPU), dtype=torch.int32)
        pos0, ns = {}, {}
        for s in live:
            q = self.rows[s]
            pos0[s] = q["P"] + len(q["out"]) - 1
            n = ns[s] = len(q["proposal"])
            tokens[s, : 1 + n] = torch.tensor([q["out"][-1]] + q["proposal"], dtype=torch.int32)
            start[s, : 1 + n] = torch.arange(pos0[s], pos0[s] + 1 + n, dtype=torch.int32)
            nvd[s], acc[s] = n, q["acc"]
            pt[s] = self._blocks_upto(q, pos0[s])
        res = cls.decode_forward(
            tokens=tokens,
            start_pos=start,
            page_table=pt,
            kv_cache=self.kv,
            enable_trace=True,
            read_from_device=False,
            num_valid_drafts=nvd,
            accepted_counts=acc,
            spec_mode="fused_sample" if self.fused else "argmax_ids",
            reload_inputs=True,
            reload_page_table=False,
            reload_sampling_params=False,
            reset_sampling_state=False,
        )
        committed = torch.full((WIDTH, 1 + K), PAD, dtype=torch.int32)
        cpos = torch.full((WIDTH, 1 + K), PAD, dtype=torch.int32)
        acc_vec = torch.ones(WIDTH, dtype=torch.int32)
        for s in live:
            q, n = self.rows[s], ns[s]
            if self.fused:
                full = res.accepted_token_ids[s, : int(res.accepted_counts[s])].tolist()
            else:
                ids = res.argmax_ids[s].tolist()
                assert all(i != PAD for i in ids[: n + 1]), f"slot {s}: -1 inside valid ids {ids} (n={n})"
                m = 0
                while m < n and q["proposal"][m] == ids[m]:
                    m += 1
                full = q["proposal"][:m] + [ids[m]]
            self.drafted += n
            self.accepted += len(full) - 1
            new = full[: q["max_new"] - len(q["out"])]
            q["out"].extend(new)
            q["acc"] = len(new)
            committed[s, : len(new)] = torch.tensor(new, dtype=torch.int32)
            cpos[s, : len(new)] = torch.arange(pos0[s] + 1, pos0[s] + 1 + len(new), dtype=torch.int32)
            acc_vec[s] = len(new)
        d = cls.propose_draft_tokens(K, committed, cpos, acc_vec, hidden=None)
        assert tuple(d.draft_token_ids.shape) == (WIDTH, K)
        for s in range(WIDTH):
            nv = int(d.num_valid[s])
            if s in pos0:
                assert 0 <= nv <= K
                self.rows[s]["proposal"] = [int(t) for t in d.draft_token_ids[s, :nv]]
            else:
                assert nv == 0, f"padding slot {s} proposed {nv} drafts"
        self.steps += 1

    def done(self, s):
        q = self.rows[s]
        return len(q["out"]) >= q["max_new"]

    def finish_all_done(self):
        """Release finished rows; returns {name: tokens} of the released ones."""
        out = {}
        for s in self.live():
            if self.done(s):
                out[self.rows[s]["name"]] = list(self.rows[s]["out"])
                self._finish(s)
        return out

    def text(self, ids):
        return self.tok.decode(ids, skip_special_tokens=True)


def _build_spec(mesh_device):
    from transformers import AutoConfig

    mesh_device.enable_program_cache()
    hf_config = AutoConfig.from_pretrained(os.environ["HF_MODEL"])
    cls = Qwen36MTPContractForCausalLM.initialize_vllm_model(
        hf_config, mesh_device, max_batch_size=WIDTH, max_seq_len=MAX_SEQ, tt_data_parallel=1, optimizations=None
    )
    model = cls.model[0]
    tc = model.args.hf_config.get_text_config()
    n_kv = tc.num_key_value_heads
    kv_shape = (NUM_BLOCKS, n_kv // min(model.num_devices, n_kv), BLOCK, model.args.head_dim)
    kv = cls.allocate_kv_cache(kv_shape, ttnn.bfloat16, len(model._attention_layer_indices))
    cls.warmup_model_prefill(enable_trace=False, kv_cache=kv, can_sample_on_device=False)
    cls.warmup_model_decode(
        enable_trace=False, kv_cache=kv, max_batch_size=WIDTH, num_blocks=BPU, can_sample_on_device=False
    )
    cls.warmup_model_prefill(enable_trace=True, kv_cache=kv, can_sample_on_device=False)
    cls.warmup_model_decode(
        enable_trace=True, kv_cache=kv, max_batch_size=WIDTH, num_blocks=BPU, can_sample_on_device=False
    )
    return cls, kv


def _check_topic(texts, failures, tag, check_text=False):
    low = {n: t.lower() for n, t in texts.items()}
    if "image" in low and not any(w in low["image"] for w in IMAGE_WORDS):
        failures.append(f"{tag}: image output not on topic: {texts['image']!r}")
    if "video" in low and not any(w in low["video"] for w in VIDEO_WORDS):
        failures.append(f"{tag}: video output not on topic: {texts['video']!r}")
    if check_text and "text" in low and not any(w in low["text"] for w in ("red", "blue", "yellow", "green")):
        failures.append(f"{tag}: text output not on topic: {texts['text']!r}")


def _run_three(drv, reqs, max_new):
    drv.admit(reqs, [0, 1, 2], max_new)
    outs = {}
    while drv.live():
        drv.step()
        outs.update(drv.finish_all_done())
    return outs


def _log_accept(drv, tag):
    logger.info(
        f"[spec_mm] {tag}: steps={drv.steps} drafts verified={drv.drafted} accepted={drv.accepted} "
        f"({drv.accepted / max(drv.drafted, 1):.2f}) fused={drv.fused}"
    )


def _finish(cls):
    try:
        cls.release_persistent_capture()
    except Exception as e:  # teardown only
        logger.warning(f"release_persistent_capture: {e}")


@run_for_blackhole()
@pytest.mark.timeout(7200)
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_spec_mm_batch3(mesh_device, reset_seeds, ensure_gc):
    """(a) text + image + video in one B=3 batch on the MTP class, N_TOK greedy tokens each, on topic."""
    if not _MULTI:
        pytest.skip("spec decode is the TP path; run with MESH_DEVICE=P150x8")
    reqs, tok = build_requests(os.environ["HF_MODEL"])
    cls, kv = _build_spec(mesh_device)
    drv = SpecDriver(cls, kv, tok)
    outs = _run_three(drv, reqs, N_TOK)
    _log_accept(drv, "batch3")
    texts = {n: drv.text(o) for n, o in outs.items()}
    for n, t in texts.items():
        logger.info(f"(a) {n}: {t!r}")
        print(f"(a) {n}: {t!r}")
    failures = []
    for n, o in outs.items():
        if len(o) != N_TOK:
            failures.append(f"{n}: {len(o)} tokens, wanted {N_TOK}")
    _check_topic(texts, failures, "(a)")  # text keyword check skipped (known to fail)
    _finish(cls)
    assert not failures, "; ".join(failures)


@run_for_blackhole()
@pytest.mark.timeout(3500)
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", NONSPEC_DEVICE_PARAMS, indirect=True)
def test_nonspec_mm_reference(mesh_device, reset_seeds, ensure_gc):
    """(b) part 1: the same three requests through the NON-spec class, greedy; token streams saved to REF_PATH."""
    from transformers import AutoConfig

    hf = os.environ["HF_MODEL"]
    reqs, tok = build_requests(hf)
    gen = Qwen36ForCausalLM.initialize_vllm_model(
        AutoConfig.from_pretrained(hf), mesh_device, max_batch_size=32, max_seq_len=16384
    )
    model = gen.model[0]
    nb = 1024
    kv_shape = (nb, model.args.n_local_kv_heads, BLOCK, model.args.head_dim)
    kv = gen.allocate_kv_cache(kv_shape, ttnn.bfloat16, len(model.layers))
    dkw = dict(kv_cache=kv, max_batch_size=32, num_blocks=16384 // BLOCK, can_sample_on_device=True)
    gen.warmup_model_prefill(kv_cache=kv, enable_trace=False)
    gen.warmup_model_decode(enable_trace=False, **dkw)
    gen.already_warmed_up_prefill = False
    gen.warmup_model_prefill(kv_cache=kv, enable_trace=True)
    gen.warmup_model_decode(enable_trace=True, **dkw)
    d = NonSpecDriver(gen, kv, tok)
    logits, blocks = d.prefill(reqs, [0, 1, 2])
    rows = mk_rows(reqs, logits, blocks)
    for _ in range(N_LOSSLESS - 1):
        d.decode_step(rows)
    ref = {r.name: row["out"][:N_LOSSLESS] for r, row in zip(reqs, rows)}
    for n, o in ref.items():
        logger.info(f"(b-ref) {n}: {d.text(o)!r}")
    with open(REF_PATH, "w") as f:
        json.dump({"n": N_LOSSLESS, "tokens": ref}, f)
    logger.info(f"non-spec reference written to {REF_PATH}")
    assert all(len(o) == N_LOSSLESS for o in ref.values())


@run_for_blackhole()
@pytest.mark.timeout(7200)
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_spec_mm_lossless(mesh_device, reset_seeds, ensure_gc):
    """(b) part 2: MTP streams vs the non-spec reference file: >= MIN_IDENTICAL identical tokens per user."""
    if not _MULTI:
        pytest.skip("spec decode is the TP path; run with MESH_DEVICE=P150x8")
    if not os.path.exists(REF_PATH):
        pytest.fail(f"{REF_PATH} missing: run test_nonspec_mm_reference first (separate process)")
    ref = json.load(open(REF_PATH))["tokens"]
    reqs, tok = build_requests(os.environ["HF_MODEL"])
    cls, kv = _build_spec(mesh_device)
    drv = SpecDriver(cls, kv, tok)
    outs = _run_three(drv, reqs, N_LOSSLESS)
    _log_accept(drv, "lossless")
    failures = []
    for n in ("text", "image", "video"):
        a, b = outs[n], ref[n]
        div = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
        logger.info(
            f"(b) {n}: first divergence idx={div}/{min(len(a), len(b))} (near-tie gap threshold {NEAR_TIE_GAP}, "
            f"gap unavailable: non-spec device sampling returns tokens only)"
        )
        print(f"(b) {n}: first divergence idx={div}; spec={drv.text(a)!r}\n    nonspec={drv.text(b)!r}")
        if div < MIN_IDENTICAL:
            failures.append(f"{n}: diverges at idx {div} < {MIN_IDENTICAL}")
    _finish(cls)
    assert not failures, "; ".join(failures)


@run_for_blackhole()
@pytest.mark.timeout(7200)
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_spec_mm_join(mesh_device, reset_seeds, ensure_gc):
    """(c) text user decodes alone; image user joins mid-decode (prefill while decoding); then the video user."""
    if not _MULTI:
        pytest.skip("spec decode is the TP path; run with MESH_DEVICE=P150x8")
    reqs, tok = build_requests(os.environ["HF_MODEL"])
    by_name = {r.name: r for r in reqs}
    cls, kv = _build_spec(mesh_device)
    drv = SpecDriver(cls, kv, tok)
    max_new = 40  # long enough that the text user is still live when the video user joins
    outs = {}

    def run(n):
        for _ in range(n):
            if drv.live():
                drv.step()
                outs.update(drv.finish_all_done())

    drv.admit([by_name["text"]], [0], max_new)
    run(3)
    assert 0 in drv.live(), "text user finished before the image user joined"
    drv.admit([by_name["image"]], [1], max_new)  # joins while the text user is mid-decode
    run(3)
    assert 0 in drv.live(), "text user finished before the video user joined"
    drv.admit([by_name["video"]], [2], max_new)
    while drv.live():
        run(1)
    _log_accept(drv, "join")
    texts = {n: drv.text(o) for n, o in outs.items()}
    for n, t in texts.items():
        logger.info(f"(c) {n}: {t!r}")
        print(f"(c) {n}: {t!r}")
    failures = [f"{n}: {len(o)} tokens" for n, o in outs.items() if len(o) != max_new]
    if set(outs) != {"text", "image", "video"}:
        failures.append(f"finished users {sorted(outs)}")
    _check_topic(texts, failures, "(c)")
    _finish(cls)
    assert not failures, "; ".join(failures)
