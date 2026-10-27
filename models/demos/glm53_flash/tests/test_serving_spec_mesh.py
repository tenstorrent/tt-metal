# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The vLLM adapter's speculative path on a (2, 4) block, driven the way the vllm-tt-plugin runner drives it
(``tests/serving_spec_driver.py``: candidate blocks, the greedy accept walk, the commit, ``propose_draft_tokens``
after every step): ``GLM53FlashForCausalLM.initialize_vllm_model`` with ``GLM53_VLLM_SPEC=1`` builds the MTP drafter
and the P / V / C / D / D0 graphs; the plugin's boot order warms and captures them; four prompts of different lengths
(one crossing the dense bound mid-run) are prefilled through ``prefill_forward`` (the MTP cache filled per chunk);
the served speculative loop's committed tokens are compared with the adapter's own ordinary decode (the plain
reference, host logits) token by token (identical up to the plain path's first near-tie, top-2 logit gap <
``GLM53_SPEC_NEAR_TIE_GAP``); a verify the step cannot run (a prompt beyond the dense range arriving while a
proposal is outstanding) is answered one column wide and stays lossless. The acceptance and the tok/s per user of
both loops are logged (a reduced model: a plumbing number).

    GLM53_MESH_SHAPE=2,4 GLM53_MODEL_LAYERS=0-3 GLM53_MODEL_B=4 GLM53_MODEL_TRACE_MB=512 \\
        pytest models/demos/glm53_flash/tests/test_serving_spec_mesh.py -s --timeout 7200
    (GLM53_VLLM_SPEC_K=3 and GLM53_SPEC_COMMIT=fused are the defaults; GLM53_SPEC_PROMPT="<text>" for a natural
    prompt instead of the dump's ids)
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from loguru import logger

ttnn = pytest.importorskip("ttnn")

from models.demos.glm53_flash.reference.checkpoint import default_snapshot_dir  # noqa: E402
from models.demos.glm53_flash.tests.serving_spec_driver import SpecRunnerSim, decode_kw, greedy_params  # noqa: E402
from models.demos.glm53_flash.tt import vllm_spec as vs  # noqa: E402
from models.demos.glm53_flash.tt.config import GLM53FlashConfig  # noqa: E402
from models.demos.glm53_flash.tt.generator_vllm import GLM53FlashForCausalLM  # noqa: E402
from models.demos.glm53_flash.tt.mesh import mesh_device_params  # noqa: E402

DUMP = Path("bringup/references/glm53_flash/L45_S128")
MESH = tuple(int(v) for v in os.environ.get("GLM53_MESH_SHAPE", "2,4").split(","))
B = int(os.environ.get("GLM53_MODEL_B", "4"))  # users per row = max_num_seqs (replicated rows)
MAX_SEQ_LEN = 256
N_TOKENS = int(os.environ.get("GLM53_SPEC_TOKENS", "24"))
TRACE_MB = int(os.environ.get("GLM53_MODEL_TRACE_MB", "512"))
NEAR_TIE_GAP = float(os.environ.get("GLM53_SPEC_NEAR_TIE_GAP", "1.0"))
# prompt lengths per slot: the last one crosses the dense bound (anchor + K > max_seq_len - 1) during the run
PROMPT_LENS = [32, 40, 48, 230]
DEVICE_PARAMS = mesh_device_params(*MESH, trace_region_size=TRACE_MB * 1_000_000)


def _prompt_ids(cfg: GLM53FlashConfig, snapshot: Path, n: int) -> torch.Tensor:
    text = os.environ.get("GLM53_SPEC_PROMPT")
    if text:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(snapshot)
        ids = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, tokenize=True)
        base = torch.tensor(ids, dtype=torch.long)
    elif (DUMP / "prefill_input.pt").exists():
        base = torch.load(DUMP / "prefill_input.pt")["prompt_ids"].reshape(-1).long()
    else:
        pytest.skip(f"host dump {DUMP} missing and no GLM53_SPEC_PROMPT")
    reps = -(-n // int(base.numel()))
    return base.repeat(reps)[:n].contiguous()


def _build(mesh_device, monkeypatch):
    try:
        snapshot = default_snapshot_dir()
    except FileNotFoundError as e:
        pytest.skip(str(e))
    monkeypatch.setenv(vs.SPEC_ENV, "1")
    monkeypatch.setenv("GLM53_PARTITION_USERS", "0")  # the speculative verify is wired for replicated caches only
    monkeypatch.setenv("GLM53_VLLM_WARMUP_PREFILL", "128")
    monkeypatch.setenv("GLM53_MESH_SHAPE", ",".join(str(v) for v in MESH))
    cfg = GLM53FlashConfig.from_hf_config_json(snapshot / "config.json")
    hf_config = SimpleNamespace(_name_or_path=str(snapshot))
    t0 = time.perf_counter()
    a = GLM53FlashForCausalLM.initialize_vllm_model(hf_config, mesh_device, max_batch_size=B, max_seq_len=MAX_SEQ_LEN)
    assert a.spec is not None and a.spec_policy is not None, "GLM53_VLLM_SPEC=1 must build the drafter"
    logger.info(
        f"adapter built in {time.perf_counter() - t0:.0f} s: K={a.spec.K} commit {a.spec.commit_mode} "
        f"dense bound {a.spec.max_dense_pos} (anchors <= {a.spec_policy.max_dense_pos - a.spec_policy.draft_len})"
    )
    t0 = time.perf_counter()
    n = a.geometry.request_capacity
    a.warmup_model_prefill(enable_trace=False, can_sample_on_device=True)
    a.warmup_model_decode(enable_trace=False, max_batch_size=n, num_blocks=1, can_sample_on_device=True)
    a.warmup_model_prefill(enable_trace=True, can_sample_on_device=True)
    a.warmup_model_decode(enable_trace=True, max_batch_size=n, num_blocks=1, can_sample_on_device=True)
    logger.info(f"warm-up + captures (prefill traces, P, V, C, D, D0) in {time.perf_counter() - t0:.0f} s")
    return a, cfg, snapshot


def _prefill(a, slot: int, ids: torch.Tensor) -> tuple[int, int]:
    n = int(ids.numel())
    out, _ = a.prefill_forward(
        ids.reshape(1, n).to(torch.int32),
        prompt_lens=[n],
        start_pos=torch.tensor([0]),
        empty_slots=[slot],
        sampling_params=greedy_params(1),
    )
    return int(out[0, 0]), n


def _reference(a, prompts: dict[int, torch.Tensor], n: int) -> tuple[dict[int, list[int]], dict[int, list[float]]]:
    """The adapter's ordinary decode (host logits) for ``n`` tokens per slot: the tokens and the top-2 logit gaps."""
    a.model.reset_kda_states()
    anchors, toks, gaps = {}, {}, {}
    for slot, ids in prompts.items():
        first, p = _prefill(a, slot, ids)
        anchors[slot] = (first, p)
        toks[slot], gaps[slot] = [], []
    wire = a.geometry.request_capacity
    t0 = time.perf_counter()
    for _ in range(n):
        tokens = torch.zeros(wire, 1, dtype=torch.int32)
        positions = torch.full((wire,), -1, dtype=torch.int32)
        for slot, (tok, p) in anchors.items():
            tokens[slot, 0], positions[slot] = tok, p
        logits, _ = a.decode_forward(tokens, positions, **decode_kw())
        for slot in anchors:
            row = logits[slot, 0].float()
            top = row.topk(2).values
            nxt = int(row.argmax())
            toks[slot].append(nxt)
            gaps[slot].append(float(top[0] - top[1]))
            anchors[slot] = (nxt, anchors[slot][1] + 1)
    plain_s = time.perf_counter() - t0
    for slot in prompts:
        a.release_request(slot)
    return toks, gaps, plain_s


def _compare(name: str, got: dict[int, list[int]], ref: dict[int, list[int]], gaps: dict[int, list[float]]):
    matched = {}
    for slot, seq in got.items():
        r = ref[slot]
        n = min(len(seq), len(r))
        i = next((i for i in range(n) if seq[i] != r[i]), n)
        matched[slot] = i
        if i < n:
            gap = gaps[slot][i]
            logger.info(
                f"{name}: slot {slot} first difference at token {i} (plain top-2 gap {gap:.3f}): {seq[i]} vs {r[i]}"
            )
            assert (
                gap < NEAR_TIE_GAP
            ), f"{name}: slot {slot} diverges from the plain decode at token {i} (gap {gap:.3f})"
    logger.info(f"{name}: tokens identical to the plain decode per slot: {matched}")


@pytest.mark.timeout(7200)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH], indirect=True)
def test_served_spec_loop_matches_the_plain_decode(mesh_device, monkeypatch):
    a, cfg, snapshot = _build(mesh_device, monkeypatch)
    K = a.spec.K
    assert K >= 1
    wire = a.geometry.request_capacity
    assert wire == B and len(PROMPT_LENS) <= wire
    prompts = {slot: _prompt_ids(cfg, snapshot, n) for slot, n in enumerate(PROMPT_LENS)}
    assert max(PROMPT_LENS) + N_TOKENS < MAX_SEQ_LEN

    # ---- the plain reference: the adapter's own ordinary decode (P: the plain step + MTP level 1), host logits
    ref, gaps, plain_s = _reference(a, prompts, N_TOKENS)
    logger.info(f"plain reference: {N_TOKENS} steps x {len(prompts)} slots in {plain_s:.2f} s")

    # ---- the served speculative loop (device greedy sampling), the runner's calls
    a.model.reset_kda_states()
    runner = SpecRunnerSim(a, K, wire=wire)
    for slot, ids in prompts.items():
        first, p = _prefill(a, slot, ids)
        runner.add(slot, first, p)
    t0 = time.perf_counter()
    runner.step()  # the first step after the prefills: an ordinary decode, every row proposes
    runner.drop_drafts(2)  # a dropped proposal: slot 2 rides the next verify with num_valid 0 (a mixed block)
    log = runner.step()
    assert log.kind == "verify" and log.num_valid.get(2) == 0 and log.counts.get(2) == 1
    got = runner.run(N_TOKENS)
    spec_s = time.perf_counter() - t0
    _compare("served spec loop", got, ref, gaps)
    verifies = [s for s in runner.log if s.kind == "verify"]
    full = [s for s in verifies if s.width > 1]
    assert full, "no full-width verify ran"
    for s in full:
        assert all(1 <= c <= s.num_valid[slot] + 1 for slot, c in s.counts.items()), s
        assert all(a.spec_policy.anchor_ok(p) for p in s.anchors.values()), s
    st = a._spec_stats
    assert st.verifies == len(full) and a._spec_pending is None
    logger.info(
        f"SPEC-SERVED mesh {MESH} layers {len(a.model.layers)} users {wire} K {K} commit {a.spec.commit_mode}: "
        f"{len(runner.log)} steps ({len(full)} full-width verifies, {len(verifies) - len(full)} one-column, "
        f"{len(runner.log) - len(verifies)} plain); {st.summary()}"
    )
    plain_tps = N_TOKENS / plain_s
    spec_tps = N_TOKENS / spec_s
    logger.info(
        f"PERF-SPEC-SERVED mesh {MESH} layers {len(a.model.layers)} users {wire} K {K}: plain {plain_tps:.1f} tok/s per "
        f"user ({1000 * plain_s / N_TOKENS:.1f} ms per token, host logits); spec {spec_tps:.1f} tok/s per user "
        f"({1000 * spec_s / N_TOKENS:.1f} ms per token, device greedy), x{spec_tps / plain_tps:.2f} (a reduced model: a "
        "plumbing number)"
    )

    # ---- a verify the step cannot run: a prompt beyond the dense bound arrives while slot 0's proposal is outstanding
    a.model.reset_kda_states()
    bound = a.spec_policy.max_dense_pos - K  # the last anchor the verify may start at
    long_n = bound + 1
    assert long_n + 2 < MAX_SEQ_LEN
    prompts2 = {0: prompts[0], 1: _prompt_ids(cfg, snapshot, long_n)}
    ref2, gaps2, _ = _reference(a, prompts2, 2)
    a.model.reset_kda_states()
    runner = SpecRunnerSim(a, K, wire=wire)
    first0, p0 = _prefill(a, 0, prompts2[0])
    runner.add(0, first0, p0)
    runner.step()  # plain: slot 0 proposes K drafts
    assert len(runner.slots[0].drafts) == K
    first1, p1 = _prefill(a, 1, prompts2[1])
    runner.add(1, first1, p1)
    fallbacks = a._spec_stats.fallbacks
    log = runner.step()  # the runner sends a verify for slot 0's drafts; the adapter answers one column wide
    assert log.kind == "verify" and log.width == 1 and log.counts == {0: 1, 1: 1}
    assert a._spec_stats.fallbacks == fallbacks + 1 and runner.slots[0].drafts == [] and runner.slots[1].drafts == []
    got2 = runner.run(2)
    _compare("one-column verify", got2, ref2, gaps2)

    # ---- the eager chunk form: the same prompts prefilled with enable_trace=False (the eager prefill pads them to
    # the 256 bucket: 128-row slabs, 96-224 rows past seq_len) and the loop against the traced-form reference
    a.model.reset_kda_states()
    runner = SpecRunnerSim(a, K, wire=wire)
    for slot in (0, 1):
        ids = prompts[slot]
        n = int(ids.numel())
        out, _ = a.prefill_forward(
            ids.reshape(1, n).to(torch.int32),
            prompt_lens=[n],
            start_pos=torch.tensor([0]),
            empty_slots=[slot],
            sampling_params=greedy_params(1),
            enable_trace=False,
        )
        runner.add(slot, int(out[0, 0]), n)
    got3 = runner.run(8)
    _compare("eager-form prefill", got3, ref, gaps)
    logger.info(f"MTP prompt fill warmed at {a.spec.warmed_fill_lengths} before the captures")
    assert set(a.spec.warmed_fill_lengths) >= set(a.model.trace_prefill_lengths()), "every traced length must be warmed"
    a.release_persistent_capture()
