# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Speculative decoding with the MTP drafter on the resident mesh model (``tt/spec_decode.py``, ``tt/mtp.py``,
``GLM53MeshModel.decode_step_with_draft``), real weights, a (2,4) block by default:

1. ``test_draft_step_trace`` (plan step 3): the traced plain step + MTP level 1 (``capture_with_draft``): the
   trace's drafts equal an eager ``TtMTPBlock.decode_forward`` on the trace's own hidden (bit-equal), the greedy
   ids equal the host argmax of the logits; the draft-vs-next-greedy acceptance over the run is logged.
2. ``test_verify_and_commit_vs_sequential`` (steps 4 + 5): one verify step over ``B (K + 1)`` rows with the plain
   path's own greedy continuation as the drafts: per-row greedy ids == the plain traced steps' ids (near-ties
   tolerated by logit gap), the KDA state after the verify == the plain state after ``K + 1`` steps (PCC), the
   commit with ``m = K`` reproduces the post-verify state bit-exactly, with ``m = 0`` the plain state after one
   step, with a per-user mix the per-user plain states; the pooled-key ring after the commit holds the committed
   rows only.
3. ``test_spec_decode_lossless`` (step 6): ``SpecDecoder.generate`` vs the plain traced greedy decode over 64
   tokens: token-identical up to the first near-tie of the plain path (top-2 logit gap < 1), acceptance logged.

    GLM53_MESH_SHAPE=2,4 GLM53_MODEL_LAYERS=0-7 GLM53_MODEL_B=2 GLM53_SPEC_K=3 \\
        pytest models/demos/glm53_flash/tests/test_mtp_spec.py -s --timeout 10800
    GLM53_MESH_SHAPE=8,4 GLM53_MODEL_LAYERS=0-44 GLM53_MODEL_B=4 GLM53_SPEC_K=3 ...   (the whole model, meaningful
        acceptance; GLM53_SPEC_PROMPT="<text>" tokenizes a natural prompt instead of the dump's random ids)

A reduced layer set is a nonsense model (the drafter then accepts ~nothing), but every property above holds for
any weights: that is what the block runs check; the Galaxy run adds the acceptance rate.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest
import torch
from loguru import logger

ttnn = pytest.importorskip("ttnn")

from models.demos.glm53_flash.reference.checkpoint import default_snapshot_dir, shard_headers  # noqa: E402
from models.demos.glm53_flash.tests.utils import pcc  # noqa: E402
from models.demos.glm53_flash.tt.config import GLM53FlashConfig  # noqa: E402
from models.demos.glm53_flash.tt.mesh import mesh_device_params  # noqa: E402
from models.demos.glm53_flash.tt.mesh_model import GLM53MeshModel  # noqa: E402
from models.demos.glm53_flash.tt.mtp import TtMTPBlock  # noqa: E402
from models.demos.glm53_flash.tt.spec_decode import SpecDecoder, prefill_with_mtp, spec_draft_len  # noqa: E402

DUMP = Path("bringup/references/glm53_flash/L45_S128")
MESH = tuple(int(v) for v in os.environ.get("GLM53_MESH_SHAPE", "2,4").split(","))
B = int(os.environ.get("GLM53_MODEL_B", "2"))  # users per row
S = int(os.environ.get("GLM53_SPEC_S", "32"))  # prompt tokens (<= 128 from the dump)
K_ENV = int(os.environ.get("GLM53_SPEC_K", "3"))
N_TOKENS = int(os.environ.get("GLM53_SPEC_TOKENS", "64"))
TRACE_MB = int(os.environ.get("GLM53_MODEL_TRACE_MB", "512"))
NEAR_TIE_GAP = float(os.environ.get("GLM53_SPEC_NEAR_TIE_GAP", "1.0"))
MAX_SEQ_LEN = 256
_DT = {"bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b}
EXPERT_DTYPE = _DT[os.environ.get("GLM53_MODEL_EXPERT_DTYPE", "bfp8")]
EXPERT_DOWN_DTYPE = _DT[os.environ.get("GLM53_MODEL_EXPERT_DOWN_DTYPE", "bfp8")]
_cache_env = os.environ.get("GLM53_MESH_CACHE", "/home/ttuser/ashai/glm53/_cache/mesh")
CACHE_ROOT = None if _cache_env.lower() == "none" else Path(_cache_env)
STATE_PCC = 0.9999
DEVICE_PARAMS = mesh_device_params(*MESH, trace_region_size=TRACE_MB * 1_000_000)


def _layer_range() -> list[int]:
    spec = os.environ.get("GLM53_MODEL_LAYERS", "0-7")
    if "-" in spec:
        a, b = spec.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(v) for v in spec.split(",")]


def _prompt(cfg: GLM53FlashConfig, snapshot: Path) -> torch.Tensor:
    text = os.environ.get("GLM53_SPEC_PROMPT")
    if text:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(snapshot)
        ids = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, tokenize=True)
        return torch.tensor(ids, dtype=torch.long)
    if not (DUMP / "prefill_input.pt").exists():
        pytest.skip(f"host dump {DUMP} missing and no GLM53_SPEC_PROMPT")
    return torch.load(DUMP / "prefill_input.pt")["prompt_ids"].reshape(-1)[:S].long()


def _build(mesh_device):
    try:
        snapshot = default_snapshot_dir()
    except FileNotFoundError as e:
        pytest.skip(str(e))
    layers = _layer_range()
    cfg = GLM53FlashConfig.from_hf_config_json(snapshot / "config.json")
    headers = shard_headers(snapshot)
    t0 = time.perf_counter()
    model = GLM53MeshModel(
        mesh_device,
        cfg,
        layers,
        max_batch=B,
        max_seq_len=MAX_SEQ_LEN,
        weight_cache_path=CACHE_ROOT,
        expert_dtype=EXPERT_DTYPE,
        expert_down_dtype=EXPERT_DOWN_DTYPE,
    )
    mtp = TtMTPBlock.from_checkpoint(
        mesh_device,
        cfg,
        snapshot,
        headers,
        embedding=model.embedding,
        head=model.head,
        weight_cache_root=CACHE_ROOT,
        mesh=model.mesh,
        expert_dtype=EXPERT_DTYPE,
        expert_down_dtype=EXPERT_DOWN_DTYPE,
        max_batch=B,
        max_seq_len=MAX_SEQ_LEN + 32,
    )
    model.attach_mtp(mtp)
    ttnn.synchronize_device(mesh_device)
    logger.info(
        f"model layers {layers[0]}..{layers[-1]} + MTP built in {time.perf_counter() - t0:.0f} s on mesh {MESH}"
    )
    return model, mtp, cfg, _prompt(cfg, snapshot)


def _prefill_all(model, mtp, prompt) -> tuple[int, int]:
    """Every user slot (of every row) gets the same prompt; returns (first token, S). Logs the prefill time as the
    run's TTFT (eager prefill with the MTP hidden capture, one call per user slot of a row, each serving its 8
    replicas): the README's row rule wants TTFT and decode from the same run."""
    model.reset_kda_states()
    tok = pos = None
    t0 = time.perf_counter()
    for u in range(model.max_batch):
        tok, pos = prefill_with_mtp(model, mtp, prompt, user=u)
    ttnn.synchronize_device(model.device)
    dt = time.perf_counter() - t0
    logger.info(
        f"PERF-SPEC-TTFT mesh {MESH} users/row {model.max_batch}: prompt {pos} tokens, eager prefill + MTP capture "
        f"{dt:.2f} s for {model.max_batch} prefills = {1000 * dt / model.max_batch:.0f} ms per prefill (TTFT)"
    )
    return tok, pos


def _hidden_rows_to_device(model, hn: torch.Tensor) -> ttnn.Tensor:
    """Host rows ``[U, H]`` bf16 -> ``[1, 1, B, H]`` bf16 TILE per row chip (row r's users), the MTP's hidden input."""
    t = hn.to(torch.bfloat16).reshape(1, 1, -1, hn.shape[-1]).contiguous()
    if model.mesh.rows > 1:
        return model.mesh.upload(t, ttnn.bfloat16, rows=2)
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=model.device)


def _top2_gap(logits_host: torch.Tensor) -> torch.Tensor:
    """[U, V] -> [U] the top-1 minus top-2 logit of every user."""
    top = logits_host.float().topk(2, dim=-1).values
    return top[:, 0] - top[:, 1]


def _kda_state_host(model) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    """{layer: (recurrent [rows*B, H_loc, K, V] fp32, conv [taps, rows*B, W_loc] bf16)} of column 0's chips."""
    out = {}
    rows = model.mesh.rows
    for dec in model.layers:
        if dec.kda_batched is None:
            continue
        rec = model.mesh.to_host(dec.kda_batched.recurrent, rows=0 if rows > 1 else None)
        conv = model.mesh.to_host(dec.kda_batched.convolution, rows=1 if rows > 1 else None)
        out[dec.layer_idx] = (rec.float().clone(), conv.clone())
    return out


def _ring_host(model) -> dict[int, torch.Tensor]:
    """{DSA layer: ring_k [rows*slots, 1, 32, D]} (the raw indexer key ring of every user slot)."""
    rows = model.mesh.rows
    return {
        dec.layer_idx: model.mesh.to_host(dec.cache.index.ring_k, rows=0 if rows > 1 else None).float().clone()
        for dec in model.layers
        if not dec.is_kda
    }


def _state_close(a, b, what: str, bits: bool = False):
    """Per-layer comparison of two ``_kda_state_host`` snapshots: bit-equal (``bits``) or PCC >= STATE_PCC."""
    worst = 1.0
    for layer, (rec_a, conv_a) in a.items():
        rec_b, conv_b = b[layer]
        if bits:
            assert torch.equal(rec_a, rec_b), f"{what}: layer {layer} recurrent state differs in bits"
            assert torch.equal(conv_a, conv_b), f"{what}: layer {layer} conv history differs in bits"
            continue
        p_rec, p_conv = pcc(rec_a, rec_b), pcc(conv_a.float(), conv_b.float())
        worst = min(worst, p_rec, p_conv)
        assert p_rec > STATE_PCC and p_conv > STATE_PCC, f"{what}: layer {layer} pcc rec {p_rec:.6f} conv {p_conv:.6f}"
    logger.info(f"{what}: {'bit-equal' if bits else f'worst pcc {worst:.6f}'} over {len(a)} KDA layers")


def _plain_steps(model, trace_id, outs, first_tok: int, pos0: int, n: int, *, snapshots: bool = False):
    """``n`` plain traced steps feeding the greedy token back, every user its own chain. Returns the tokens
    ``[n + 1, U]`` (row 0 = the anchor), the top-2 gaps ``[n, U]`` and (optionally) the KDA / ring snapshots after
    every step."""
    _, logits_tr, ids_tr = outs
    U = model.total_users
    toks = [torch.full((U,), first_tok, dtype=torch.long)]
    gaps, kda, rings = [], [], []
    for t in range(n):
        model.load_ids(toks[-1])
        model.set_positions([pos0 + t] * U)
        model.replay(trace_id)
        ids = model.ids_to_host(ids_tr)
        logits = model.mesh.to_host(logits_tr, rows=2 if model.mesh.rows > 1 else None).float()[0, 0]  # [U, V]
        assert torch.equal(ids, logits.argmax(-1)), "device argmax != host argmax"
        gaps.append(_top2_gap(logits))
        toks.append(ids)
        if snapshots:
            kda.append(_kda_state_host(model))
            rings.append(_ring_host(model))
    return torch.stack(toks), torch.stack(gaps), kda, rings


# ============================================================================================== step 3
@pytest.mark.timeout(10800)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH], indirect=True)
def test_draft_step_trace(mesh_device):
    model, mtp, cfg, prompt = _build(mesh_device)
    U = model.total_users
    tok, pos0 = _prefill_all(model, mtp, prompt)  # compiles the prefill paths (incl. the MTP row collectives)
    trace_id, (logits_tr, ids_tr, draft_tr, hn_tr) = model.capture_with_draft(torch.full((U,), tok), [pos0] * U)
    tok, pos0 = _prefill_all(model, mtp, prompt)  # the capture consumed two tokens: start over
    n = N_TOKENS
    cur = torch.full((U,), tok, dtype=torch.long)
    ids_all, drafts_all = [], []
    checks = {0, n // 2, n - 1}
    for t in range(n):
        model.load_ids(cur)
        model.set_spec_positions([pos0 + t] * U)
        model.replay(trace_id)
        ids = model.ids_to_host(ids_tr)
        drafts = model.ids_to_host(draft_tr)
        if t in checks:
            logits = model.mesh.to_host(logits_tr, rows=2 if model.mesh.rows > 1 else None).float()[0, 0]
            assert torch.equal(ids, logits.argmax(-1)), f"step {t}: device argmax != host argmax"
            # the drafter eagerly on the trace's own hidden and greedy ids (same cache row, same inputs)
            hn = model.mesh.to_host(hn_tr, rows=2 if model.mesh.rows > 1 else None)[0, 0]  # [U, H] bf16
            hid = _hidden_rows_to_device(model, hn)
            ids_dev = model.ids_to_device(ids)
            out, normed, mlog, dr = mtp.decode_forward(ids_dev, hid, model.mtp_positions)
            eager = model.ids_to_host(dr)
            for x in (out, normed, mlog, dr, hid, ids_dev):
                ttnn.deallocate(x)
            logger.info(f"step {t}: traced drafts {drafts.tolist()} eager {eager.tolist()} (ids {ids.tolist()})")
            assert torch.equal(eager, drafts), f"step {t}: traced draft ids differ from the eager MTP step"
        ids_all.append(ids)
        drafts_all.append(drafts)
        cur = ids
    ids_all, drafts_all = torch.stack(ids_all), torch.stack(drafts_all)  # [n, U]
    acc = (drafts_all[:-1] == ids_all[1:]).float().mean(0)  # draft at t == greedy at t + 1
    logger.info(
        f"draft step: {n} traced steps, level-1 acceptance per user {[f'{a:.2f}' for a in acc.tolist()]} "
        f"(layers {_layer_range()[0]}..{_layer_range()[-1]}, random-prompt dump: a plumbing number on a reduced model)"
    )
    ttnn.release_trace(mesh_device, trace_id)


# =========================================================================================== steps 4 + 5
@pytest.mark.timeout(10800)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH], indirect=True)
def test_verify_and_commit_vs_sequential(mesh_device):
    model, mtp, cfg, prompt = _build(mesh_device)
    U = model.total_users
    K = spec_draft_len(B, K_ENV)
    assert K >= 1, f"B={B} leaves no room for drafts"
    spec = SpecDecoder(model, mtp, K)
    tok, pos0 = _prefill_all(model, mtp, prompt)
    # Every program is compiled BEFORE the first trace capture: a collective of a new shape compiled after a capture
    # puts its semaphores where the captured trace's L1 intermediates live and the replay hangs (2026-10-26 on the
    # block: the plain trace captured first, the spec warm-up compiled the 8-row collectives, the replay spun for
    # 21 min and needed a reset; the 2026-10-16 Gemma 4 case is the same mechanism). The warm-up writes placeholder
    # rows, so the prefill is repeated before the capture.
    spec.warmup([pos0] * U)
    tok, pos0 = _prefill_all(model, mtp, prompt)
    trace_plain, outs = model.capture(torch.full((U,), tok), [pos0] * U)
    spec.capture()

    # ---- the plain reference: K + 1 traced steps, states after every step
    tok, pos0 = _prefill_all(model, mtp, prompt)
    toks, gaps, kda_plain, rings_plain = _plain_steps(model, trace_plain, outs, tok, pos0, K + 1, snapshots=True)
    # ---- the verify on the same tokens: candidates = the plain continuation (every draft is right)
    tok, pos0 = _prefill_all(model, mtp, prompt)
    tokens = toks[: K + 1].t().contiguous()  # [U, K + 1]: anchor, d_1 .. d_K
    ids_v, d1_v = spec.verify_forward(tokens, [pos0] * U)
    flips = []
    for u in range(U):
        for j in range(K + 1):
            if int(ids_v[u, j]) != int(toks[j + 1, u]):
                flips.append((u, j, float(gaps[j, u])))
    logger.info(
        f"verify K={K}: rows == plain ids for {U * (K + 1) - len(flips)}/{U * (K + 1)} rows; flips (user, cand, plain "
        f"top-2 gap): {flips}; level-1 drafts of the rows {d1_v.tolist()}"
    )
    assert all(
        g < NEAR_TIE_GAP for _, _, g in flips
    ), f"verify rows differ from the plain path beyond near-ties: {flips}"
    post_verify = _kda_state_host(model)
    _state_close(post_verify, kda_plain[K], "KDA state after the verify vs plain after K + 1 steps")

    # ---- the real protocol: ONE commit after the verify, here a per-user mix m_u = u mod (K + 1)
    m = torch.tensor([u % (K + 1) for u in range(U)])
    spec.commit(m, [pos0] * U)
    mixed = _kda_state_host(model)
    for layer, (rec, conv) in mixed.items():
        for u in range(U):
            rec_ref, conv_ref = kda_plain[int(m[u])][layer]
            p = min(pcc(rec[u], rec_ref[u]), pcc(conv[:, u].float(), conv_ref[:, u].float()))
            assert p > STATE_PCC, f"commit(mixed m) user {u} (m={int(m[u])}) layer {layer}: pcc {p:.6f}"
    logger.info(f"commit(mixed m={m.tolist()}): every user's state matches the plain state after m_u + 1 steps")
    # the pooled-key ring now holds the committed rows only: rows 0..3 of user u == the plain ring after m_u + 1
    # steps (the rejected candidates' rows went to the dead rows 4..31 of the ring block)
    rings = _ring_host(model)
    P = cfg.index_kpool
    for layer, ring in rings.items():
        for u in range(U):
            ref = rings_plain[int(m[u])][layer][u, 0, :P]  # [P, D] the plain ring rows of user u
            p = pcc(ring[u, 0, :P], ref)
            assert p > 0.999, f"layer {layer} user {u}: ring rows after commit(m={int(m[u])}) pcc {p:.6f} vs plain"
    logger.info("index commit: ring rows 0..3 of every user equal the plain path's after m_u + 1 steps")
    # ---- one commit per verify (the commit spends the backup): a fresh prefill + verify before each further check.
    # commit(m = K) = the K + 1 launches of the verify replayed on the backup from the same cached projections:
    # the post-verify state, bit for bit
    tok, pos0 = _prefill_all(model, mtp, prompt)
    ids_v2, _ = spec.verify_forward(tokens, [pos0] * U)
    assert torch.equal(ids_v2, ids_v), "a repeated verify from the same prefill is not deterministic"
    post_verify2 = _kda_state_host(model)
    _state_close(post_verify2, post_verify, "post-verify state, repeated verify", bits=True)
    spec.commit(torch.full((U,), K, dtype=torch.long), [pos0] * U)
    _state_close(_kda_state_host(model), post_verify2, "commit(m = K) vs the post-verify state", bits=True)
    # commit(m = 0): the plain state after the anchor only
    tok, pos0 = _prefill_all(model, mtp, prompt)
    spec.verify_forward(tokens, [pos0] * U)
    spec.commit(torch.zeros(U, dtype=torch.long), [pos0] * U)
    _state_close(_kda_state_host(model), kda_plain[0], "commit(m = 0) vs plain after 1 step")
    spec.release()
    ttnn.release_trace(mesh_device, trace_plain)


# ============================================================================================== step 6
@pytest.mark.timeout(10800)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH], indirect=True)
def test_spec_decode_lossless(mesh_device):
    model, mtp, cfg, prompt = _build(mesh_device)
    U = model.total_users
    K = spec_draft_len(B, K_ENV)
    assert K >= 1
    spec = SpecDecoder(model, mtp, K)
    tok, pos0 = _prefill_all(model, mtp, prompt)
    # Every program is compiled BEFORE the first trace capture: a collective of a new shape compiled after a capture
    # puts its semaphores where the captured trace's L1 intermediates live and the replay hangs (2026-10-26 on the
    # block: the plain trace captured first, the spec warm-up compiled the 8-row collectives, the replay spun for
    # 21 min and needed a reset; the 2026-10-16 Gemma 4 case is the same mechanism). The warm-up writes placeholder
    # rows, so the prefill is repeated before the capture.
    spec.warmup([pos0] * U)
    tok, pos0 = _prefill_all(model, mtp, prompt)
    trace_plain, outs = model.capture(torch.full((U,), tok), [pos0] * U)
    spec.capture()
    n = N_TOKENS
    assert pos0 + n + K + 1 < MAX_SEQ_LEN

    tok, pos0 = _prefill_all(model, mtp, prompt)
    t0 = time.perf_counter()
    toks, gaps, _, _ = _plain_steps(model, trace_plain, outs, tok, pos0, n)
    plain_s = time.perf_counter() - t0
    tok, pos0 = _prefill_all(model, mtp, prompt)
    outs_spec, stats = spec.generate([tok] * U, [pos0] * U, n)
    # the comparison first, the perf line next, the verdict last: a sweep point reports its numbers either way
    matched, failures = [], []
    for u in range(U):
        ref = toks[1:, u].tolist()  # the plain continuation after the anchor
        got = outs_spec[u]
        assert len(got) == n, f"user {u}: {len(got)} tokens"
        i = next((i for i in range(n) if got[i] != ref[i]), n)
        matched.append(i)
        if i < n:
            gap = float(gaps[i, u])
            # every slot decodes the same prompt: the plain path's own choices across the users at this step are the
            # near-tie's two sides (the 8 rows reduce in different orders and the 45-layer chain amplifies the ulp:
            # the 2026-10-29 sweep anchor saw rows 2 and 5 of the PLAIN path disagree at token 4, one with a 2.06
            # margin); a spec token the plain path itself picked on another user is that tie, not a divergence
            plain_choices = set(int(v) for v in toks[1 + i].tolist())
            tie = gap < NEAR_TIE_GAP or int(got[i]) in plain_choices
            logger.info(
                f"user {u}: first difference at token {i} (plain top-2 gap {gap:.3f}): {got[i]} vs {ref[i]}; the plain "
                f"path's choices across the users at this step {sorted(plain_choices)} -> {'near-tie' if tie else 'DIVERGENCE'}"
            )
            if not tie:
                failures.append(f"user {u}: spec diverges from plain greedy at token {i} with gap {gap:.3f}")
    logger.info(
        f"SPEC mesh {MESH} layers {len(model.layers)} users {U} K {K}: {stats.summary()}; plain {n} steps {plain_s:.2f} s; "
        f"tokens identical to plain greedy per user: {matched} of {n} (n = all)"
    )
    plain_tps, spec_tps = n / plain_s, n / stats.seconds if stats.seconds else float("nan")
    logger.info(
        f"PERF-SPEC mesh {MESH} layers {len(model.layers)} users {U} K {K}: plain {plain_tps:.1f} tok/s per user "
        f"({plain_tps * U:.0f} aggregate, {1000 * plain_s / n:.1f} ms per token); spec {spec_tps:.1f} tok/s per user "
        f"({spec_tps * U:.0f} aggregate, {1000 * stats.seconds / n:.1f} ms per token), x{spec_tps / plain_tps:.2f}, "
        f"{stats.tokens_per_iteration / U:.2f} tokens per user per iteration"
    )
    spec.release()
    ttnn.release_trace(mesh_device, trace_plain)
    assert not failures, failures
