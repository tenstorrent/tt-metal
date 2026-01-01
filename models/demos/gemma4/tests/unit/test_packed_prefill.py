# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Packed multi-prompt prefill (``GEMMA4_PACKED_PREFILL=1``) against the per-user prefill loop.

``PACKED_PREFILL_USERS`` (32) distinct ``PACKED_PREFILL_LEN`` (128)-token prompts are prefilled three ways, eagerly:
the per-user loop (one ``prefill_forward_text`` call per user, the served sequential path), the stock batched
forward (flag off: one call, per-slot lm_head slices) and the packed forward (flag on: one call, one gathered
lm_head). Per user the first-token id and the last-token logits PCC are compared against the loop, then one eager
decode step at position L reads the K/V each mode wrote (per-slot page-table placement check). Wall ms of every
mode is logged (``PACKED_PREFILL_SUMMARY``). Never ``enable_trace=True`` here.

Pass bars (``PACKED_PREFILL_MIN_PCC`` 0.75, ``PACKED_PREFILL_MEAN_PCC`` 0.95, ``PACKED_PREFILL_MIN_AGREE`` 0.75 of
the users, ``PACKED_PREFILL_MIN_DEC_PCC`` 0.70 for the decode check): this MoE stack's numerics are shape-dependent -- the per-user loop run at the 1024 bucket instead of
the 128 bucket agrees with itself at PCC min 0.867 / mean 0.970 (29/32 first tokens), and against the HF CPU
reference the loop scores PCC min 0.773 / mean 0.953 (27/32) while the packed forward scores 0.814 / 0.961 (24/32)
(26B-A4B, 1x4, 2026-10-23). A packing bug shows as PCC far below these bars from layer 0 on (``PACKED_PREFILL_TAPS``).

    TT_VISIBLE_DEVICES=23,22,21,20 MESH_DEVICE=P150x4 HF_MODEL=google/gemma-4-26B-A4B-it GEMMA4_BOUNDED_SLIDING=0 \\
    pytest models/demos/gemma4/tests/unit/test_packed_prefill.py -k 1x4 -s
"""

import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.gemma4.demo.text_demo import (
    _batch_page_params,
    _create_tt_page_table,
    _install_hybrid_page_tables,
    _resolve_bounded_sliding,
)
from models.demos.gemma4.tests.test_factory import parametrize_mesh_with_fabric, skip_if_config_only_checkpoint
from models.demos.gemma4.tt.generator import Gemma4Generator, plan_packed_prefill_groups
from models.tt_transformers.tt.common import PagedAttentionConfig, get_padded_prefill_len

B = int(os.environ.get("PACKED_PREFILL_USERS", "32"))
L = int(os.environ.get("PACKED_PREFILL_LEN", "128"))
REPEATS = int(os.environ.get("PACKED_PREFILL_REPEATS", "2"))
# PACKED_PREFILL_TAPS=1: after the timed section, capture every decoder layer's attention / shared-MLP / MoE /
# layer output in both modes and report, per user, the first (layer, module) where the packed forward diverges.
TAPS = os.environ.get("PACKED_PREFILL_TAPS", "0") == "1"
# PACKED_PREFILL_BUCKET_PROBE=1: rerun the per-user loop with the same prompts forced into the 1024 bucket (M=1024
# rows per forward instead of 128, everything else equal) and compare it with the 128-bucket loop: the
# shape-only numerical drift of this stack, the yardstick for the packed-vs-loop PCC.
BUCKET_PROBE = os.environ.get("PACKED_PREFILL_BUCKET_PROBE", "0") == "1"
# PACKED_PREFILL_HF_REF=<pt>: HF last-position logits for the same prompts (scratchpad hf_ref_packed.py) -> per-user
# PCC / first-token agreement of the loop, the stock batched and the packed forward against HF.
HF_REF = os.environ.get("PACKED_PREFILL_HF_REF")
# PACKED_PREFILL_OUT_DIR=<dir>: save the loop / batched / packed logits and the prompts for offline comparison.
OUT_DIR = os.environ.get("PACKED_PREFILL_OUT_DIR")
MIN_PCC = float(os.environ.get("PACKED_PREFILL_MIN_PCC", "0.75"))
MEAN_PCC = float(os.environ.get("PACKED_PREFILL_MEAN_PCC", "0.95"))
MIN_AGREE = float(os.environ.get("PACKED_PREFILL_MIN_AGREE", "0.75"))
# The decode check guards per-slot K/V placement (a misplaced user reads another user's K/V: PCC near 0 and a
# wrong token); the shape-dependent drift carried into the K/V leaves one user in 32 at ~0.72 (2026-10-23).
MIN_DEC_PCC = float(os.environ.get("PACKED_PREFILL_MIN_DEC_PCC", "0.70"))
TAP_PCC = float(os.environ.get("PACKED_PREFILL_TAP_PCC", "0.995"))
# Structural bar on the first PACKED_PREFILL_EARLY_LAYERS (3) layers: a per-segment attention mask, RoPE position or
# K/V placement fault shows at layer 0 as a PCC far below 0.99 for the affected users, while the shape-dependent MoE
# routing drift (the loop at the 1024 bucket vs the 128 bucket drifts the same way) only accumulates deeper: measured
# 2026-10-23 on (1,4), every user's first tap below 0.995 was at layers 4-8 and always the MoE / attention output,
# layers 0-3 >= 0.995 for all 32 users.
EARLY_LAYERS = int(os.environ.get("PACKED_PREFILL_EARLY_LAYERS", "3"))
EARLY_PCC = float(os.environ.get("PACKED_PREFILL_EARLY_PCC", "0.99"))
_HERE = os.path.dirname(os.path.abspath(__file__))


def _install_taps(monkeypatch, state):
    """Wrap the layer and its three sub-blocks so their prefill outputs land in ``state["rec"]`` as
    ``(layer, module) -> fp16 [rows, H]`` (device shard 0) while ``state["on"]``."""
    from models.demos.gemma4.tt.attention import Gemma4Attention
    from models.demos.gemma4.tt.layer import Gemma4DecoderLayer
    from models.demos.gemma4.tt.moe import MoEBlock
    from models.demos.gemma4.tt.shared_mlp import SharedMLP

    def record(name, out):
        t = out[0] if isinstance(out, tuple) else out
        if not isinstance(t, ttnn.Tensor):
            return
        host = ttnn.to_torch(ttnn.get_device_tensors(t)[0])
        state["rec"][(state["layer"], name)] = host.reshape(-1, host.shape[-1]).to(torch.float16)

    def wrap(cls, name, is_layer=False):
        orig = cls.__call__

        def wrapped(self, *a, **k):
            if not state["on"]:
                return orig(self, *a, **k)
            if is_layer:
                state["layer"] += 1
            out = orig(self, *a, **k)
            record(name, out)
            return out

        monkeypatch.setattr(cls, "__call__", wrapped)

    wrap(Gemma4DecoderLayer, "layer", is_layer=True)
    wrap(Gemma4Attention, "attention")
    wrap(SharedMLP, "shared_mlp")
    wrap(MoEBlock, "moe")


def _prompts(tokenizer, users, length):
    """``[users, length]`` distinct natural-text prompts: BOS + consecutive README windows (realistic routing)."""
    readme = os.path.join(_HERE, "..", "..", "README.md")
    text = open(readme, encoding="utf-8", errors="ignore").read() if os.path.exists(readme) else "The quick brown fox. "
    ids = tokenizer.encode(text, add_special_tokens=False)
    while len(ids) < users * (length - 1):
        ids = ids + ids
    bos = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else ids[0]
    rows = [[bos] + ids[u * (length - 1) : (u + 1) * (length - 1)] for u in range(users)]
    return torch.tensor(rows, dtype=torch.long)


def _pcc(a, b):
    a = a.flatten().double()
    b = b.flatten().double()
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-12))


@parametrize_mesh_with_fabric(
    mesh_shapes=[(1, 4), (1, 8), (2, 4)], device_params_extra={"trace_region_size": 256_000_000}
)
def test_packed_prefill(mesh_device, monkeypatch):
    skip_if_config_only_checkpoint()
    monkeypatch.setenv("GEMMA4_PACKED_PREFILL", "0")
    model_path = os.environ.get("HF_MODEL", "google/gemma-4-26B-A4B-it")
    max_seq_len = 4096
    # The decode check runs a 32-row eager step (the decode path's tile batch), so the generator, the page pool
    # and the page table are sized for 32 slots; the prefills use the first B rows.
    MAXB = max(B, 32)
    page_params = _batch_page_params(MAXB, L, 8)
    paged = PagedAttentionConfig(
        block_size=int(page_params["page_block_size"]), max_num_blocks=int(page_params["page_max_num_blocks"])
    )
    page_table_all = _create_tt_page_table(MAXB, paged)
    page_table = page_table_all[:B]
    bounded = _resolve_bounded_sliding(max_seq_len, mesh_device, model_path)
    t0 = time.perf_counter()
    generator, kv, tokenizer = Gemma4Generator.from_pretrained(
        mesh_device=mesh_device,
        model_path=model_path,
        max_batch_size=MAXB,
        max_seq_len=max_seq_len,
        paged_attention_config=paged,
        bounded_sliding_kv_cache=bounded,
    )
    if bounded:
        _install_hybrid_page_tables(
            generator.model[0],
            generator.model_args[0],
            batch_size=MAXB,
            block_size=int(page_params["page_block_size"]),
            max_seq_len=max_seq_len,
        )
    logger.info(
        f"model built in {time.perf_counter() - t0:.1f}s: bounded_sliding={bounded} "
        f"max_prefill_chunk_size={generator.model_args[0].max_prefill_chunk_size} mesh={tuple(mesh_device.shape)}"
    )
    vocab = int(generator.model[0].vocab_size)
    tokens = _prompts(tokenizer, B, L)
    assert len({tuple(r.tolist()) for r in tokens}) == B, "prompts must be distinct"

    def sync():
        ttnn.synchronize_device(mesh_device)

    def loop_prefill():
        out = torch.zeros(B, vocab)
        for u in range(B):
            o = generator.prefill_forward_text(
                tokens[u : u + 1],
                page_table=page_table[u : u + 1],
                kv_cache=kv,
                prompt_lens=[L],
                empty_slots=[u],
                enable_trace=False,
                warmup_prefill=False,
            )
            out[u] = o.reshape(-1)[:vocab].float()
        return out

    def one_call(packed: bool):
        monkeypatch.setenv("GEMMA4_PACKED_PREFILL", "1" if packed else "0")
        try:
            o = generator.prefill_forward_text(
                tokens,
                page_table=page_table,
                kv_cache=kv,
                prompt_lens=[L] * B,
                empty_slots=list(range(B)),
                enable_trace=False,
                warmup_prefill=False,
            )
        finally:
            monkeypatch.setenv("GEMMA4_PACKED_PREFILL", "0")
        return o.reshape(B, -1)[:, :vocab].float()

    def decode_logits(tok):
        """One eager decode step at position L for every user: reads the K/V the last prefill wrote.
        Rows B..MAXB-1 are padding (position -1)."""
        tok_all = torch.zeros(MAXB, 1, dtype=tok.dtype)
        tok_all[:B] = tok
        pos_all = torch.full((MAXB,), -1, dtype=torch.int32)
        pos_all[:B] = L
        o = generator.decode_forward(
            tok_all,
            pos_all,
            page_table=page_table_all,
            kv_cache=kv,
            enable_trace=False,
            reload_inputs=True,
            reload_page_table=True,
            reload_sampling_params=False,
            reset_sampling_state=False,
        )
        logits = o[0] if isinstance(o, tuple) else o
        return logits.reshape(MAXB, -1)[:B, :vocab].float()

    def timed(fn, *a):
        best, out = None, None
        for _ in range(REPEATS):
            sync()
            t = time.perf_counter()
            out = fn(*a)
            sync()
            ms = (time.perf_counter() - t) * 1e3
            best = ms if best is None else min(best, ms)
        return out, best

    # Compile pass for every mode (programs, caches), then timed runs.
    t0 = time.perf_counter()
    loop_prefill()
    one_call(packed=False)
    one_call(packed=True)
    sync()
    logger.info(f"compile pass (loop + batched + packed): {time.perf_counter() - t0:.1f}s")

    ref, loop_ms = timed(loop_prefill)
    ref_tok = ref.argmax(-1)
    tok_in = ref_tok.reshape(B, 1)
    dec_ref, _ = timed(decode_logits, tok_in)
    # Determinism baseline: the loop against itself (the timed() runs above already ran it REPEATS times).
    ref2 = loop_prefill()
    loop_self = [_pcc(ref2[u], ref[u]) for u in range(B)]
    logger.info(
        f"loop vs loop: first-token agreement {int((ref2.argmax(-1) == ref_tok).sum())}/{B}, PCC min {min(loop_self):.5f}"
    )

    batched, batched_ms = timed(one_call, False)
    dec_batched, _ = timed(decode_logits, tok_in)

    packed, packed_ms = timed(one_call, True)
    dec_packed, _ = timed(decode_logits, tok_in)

    def compare(name, out, dec):
        tok_agree = int((out.argmax(-1) == ref_tok).sum())
        pccs = [_pcc(out[u], ref[u]) for u in range(B)]
        dec_agree = int((dec.argmax(-1) == dec_ref.argmax(-1)).sum())
        dec_pccs = [_pcc(dec[u], dec_ref[u]) for u in range(B)]
        logger.info(
            f"{name}: first-token agreement {tok_agree}/{B}, logits PCC min {min(pccs):.5f} mean "
            f"{sum(pccs) / B:.5f}; decode@{L} token agreement {dec_agree}/{B}, PCC min {min(dec_pccs):.5f}"
        )
        out_tok = out.argmax(-1)
        dec_tok = dec.argmax(-1)
        dec_ref_tok = dec_ref.argmax(-1)
        rows = [
            f"u{u:02d} pcc={pccs[u]:.4f} tok={int(out_tok[u])}{'' if int(out_tok[u]) == int(ref_tok[u]) else '!=' + str(int(ref_tok[u]))}"
            f" dec_pcc={dec_pccs[u]:.4f} dtok={int(dec_tok[u])}{'' if int(dec_tok[u]) == int(dec_ref_tok[u]) else '!=' + str(int(dec_ref_tok[u]))}"
            for u in range(B)
        ]
        logger.info(f"{name} per user:\n  " + "\n  ".join(rows))
        return tok_agree, min(pccs), dec_agree, min(dec_pccs), sum(pccs) / B, sum(dec_pccs) / B

    b_tok, b_pcc, b_dec, b_dec_pcc, _, _ = compare("batched(stock)", batched, dec_batched)
    p_tok, p_pcc, p_dec, p_dec_pcc, p_mean, p_dec_mean = compare("packed", packed, dec_packed)
    logger.info(
        f"PACKED_PREFILL_SUMMARY users={B} len={L} mesh={tuple(mesh_device.shape)} | loop {loop_ms:.0f} ms "
        f"({loop_ms / B:.0f} ms/user) | batched(stock) {batched_ms:.0f} ms | packed {packed_ms:.0f} ms "
        f"(x{loop_ms / packed_ms:.2f} vs loop) | packed first-token {p_tok}/{B} pcc_min {p_pcc:.5f} "
        f"decode-token {p_dec}/{B} decode_pcc_min {p_dec_pcc:.5f} | batched first-token {b_tok}/{B} pcc_min {b_pcc:.5f}"
    )
    if OUT_DIR:
        os.makedirs(OUT_DIR, exist_ok=True)
        torch.save(
            {
                "tokens": tokens,
                "loop": ref,
                "batched": batched,
                "packed": packed,
                "dec_loop": dec_ref,
                "dec_packed": dec_packed,
            },
            os.path.join(OUT_DIR, f"packed_prefill_b{B}_l{L}.pt"),
        )
    if HF_REF:
        hf = torch.load(HF_REF)
        assert torch.equal(hf["tokens"], tokens), "HF reference prompts differ from the harness prompts"
        hf_logits = hf["hf_last_logits"][:, :vocab].float()
        hf_tok = hf_logits.argmax(-1)
        for name, out in (("loop", ref), ("batched(stock)", batched), ("packed", packed)):
            pc = [_pcc(out[u], hf_logits[u]) for u in range(B)]
            logger.info(
                f"HF_REF {name} vs HF: first-token agreement {int((out.argmax(-1) == hf_tok).sum())}/{B}, PCC min "
                f"{min(pc):.5f} mean {sum(pc) / B:.5f}; per user " + " ".join(f"{v:.3f}" for v in pc)
            )
    if BUCKET_PROBE:
        import models.demos.gemma4.tt.generator as g4_generator_mod
        import models.tt_transformers.tt.generator as tt_generator_mod

        orig_pad = tt_generator_mod.get_padded_prefill_len
        forced = lambda n: 1024 if int(n) <= 1024 else orig_pad(n)  # noqa: E731
        monkeypatch.setattr(tt_generator_mod, "get_padded_prefill_len", forced)
        monkeypatch.setattr(g4_generator_mod, "get_padded_prefill_len", forced)
        try:
            loop_prefill()  # compile the 1024-row programs
            probe, probe_ms = timed(loop_prefill)
        finally:
            monkeypatch.setattr(tt_generator_mod, "get_padded_prefill_len", orig_pad)
            monkeypatch.setattr(g4_generator_mod, "get_padded_prefill_len", orig_pad)
        pr = [_pcc(probe[u], ref[u]) for u in range(B)]
        logger.info(
            f"BUCKET_PROBE loop@1024-bucket vs loop@128-bucket: first-token agreement "
            f"{int((probe.argmax(-1) == ref_tok).sum())}/{B}, PCC min {min(pr):.5f} mean {sum(pr) / B:.5f} "
            f"({probe_ms:.0f} ms); per user " + " ".join(f"{v:.3f}" for v in pr)
        )
        pp = [_pcc(probe[u], packed[u]) for u in range(B)]
        logger.info(f"BUCKET_PROBE loop@1024-bucket vs packed: PCC min {min(pp):.5f} mean {sum(pp) / B:.5f}")
    if TAPS:
        state = {"on": False, "layer": -1, "rec": {}}
        _install_taps(monkeypatch, state)
        loop_taps = {}  # user -> {(layer, module): [S, H]}
        for u in range(B):
            state.update(on=True, layer=-1, rec={})
            generator.prefill_forward_text(
                tokens[u : u + 1],
                page_table=page_table[u : u + 1],
                kv_cache=kv,
                prompt_lens=[L],
                empty_slots=[u],
                enable_trace=False,
                warmup_prefill=False,
            )
            state["on"] = False
            loop_taps[u] = state["rec"]
        state.update(on=True, layer=-1, rec={})
        one_call(packed=True)
        state["on"] = False
        packed_taps = state["rec"]
        n_layers = state["layer"] + 1
        logger.info(f"taps: {n_layers} layers, {len(packed_taps)} packed records, {len(loop_taps[0])} per loop user")
        first_div = {}
        worst = {}
        tap_pcc = {}  # (user, layer, module) -> PCC packed vs loop
        for u in range(B):
            for layer in range(n_layers):
                for mod in ("attention", "shared_mlp", "moe", "layer"):
                    key = (layer, mod)
                    if key not in packed_taps or key not in loop_taps[u]:
                        continue
                    rows = packed_taps[key].shape[0] // B
                    pk = packed_taps[key][u * rows : (u + 1) * rows].float()
                    lp = loop_taps[u][key].float()
                    if pk.shape != lp.shape:
                        logger.warning(
                            f"tap shape mismatch u{u} {key}: packed {tuple(pk.shape)} loop {tuple(lp.shape)}"
                        )
                        continue
                    pcc = _pcc(pk, lp)
                    tap_pcc[(u, layer, mod)] = pcc
                    worst[(u, mod)] = min(worst.get((u, mod), 1.0), pcc)
                    if pcc < TAP_PCC and u not in first_div:
                        first_div[u] = (layer, mod, pcc)
        for u in range(B):
            if u in first_div:
                layer, mod, pcc = first_div[u]
                logger.info(f"TAPS u{u:02d}: first divergence layer {layer} {mod} pcc={pcc:.4f}")
        logger.info(
            "TAPS users with no divergence below %.3f: %s" % (TAP_PCC, [u for u in range(B) if u not in first_div])
        )
        for mod in ("attention", "shared_mlp", "moe", "layer"):
            vals = [worst.get((u, mod), 1.0) for u in range(B)]
            logger.info(f"TAPS {mod}: min PCC over layers per user: " + " ".join(f"{v:.3f}" for v in vals))
        early = {(u, layer, mod): v for (u, layer, mod), v in tap_pcc.items() if layer < EARLY_LAYERS}
        if early:
            for mod in ("attention", "shared_mlp", "moe", "layer"):
                per_layer = [
                    min((v for (u, l, m), v in early.items() if l == layer and m == mod), default=1.0)
                    for layer in range(EARLY_LAYERS)
                ]
                logger.info(f"TAPS_EARLY {mod}: min PCC over users at layers 0..{EARLY_LAYERS - 1}: {per_layer}")
            (u_w, l_w, m_w), v_w = min(early.items(), key=lambda kv: kv[1])
            logger.info(f"TAPS_EARLY worst: u{u_w:02d} layer {l_w} {m_w} pcc={v_w:.4f} (bar {EARLY_PCC})")
            assert v_w >= EARLY_PCC, (
                f"packed segment structure: u{u_w} layer {l_w} {m_w} pcc={v_w:.4f} < {EARLY_PCC} "
                "(a per-segment mask / position / K-V placement fault, not routing drift)"
            )
    need = int(MIN_AGREE * B + 0.999)
    assert (
        p_pcc >= MIN_PCC and p_mean >= MEAN_PCC
    ), f"packed last-token logits vs loop: PCC min {p_pcc:.4f} mean {p_mean:.4f}"
    assert (
        p_dec_pcc >= MIN_DEC_PCC and p_dec_mean >= MEAN_PCC
    ), f"decode logits after the packed prefill vs after the loop (KV placement): min {p_dec_pcc:.4f} mean {p_dec_mean:.4f}"
    assert p_tok >= need, f"packed first tokens agree on {p_tok}/{B} users (need {need})"
    assert p_dec >= need, f"decode tokens after the packed prefill agree on {p_dec}/{B} users (need {need})"


# ===== merged from tests/unit/test_packed_prefill_plan.py: Host tests for the packed-prefill planner (``plan_packed_prefill_groups``).
@pytest.fixture()
def _blackhole_user_cap(monkeypatch):
    """The planner inherits the batched-prefill user cap (4 on Wormhole, 32 on Blackhole); pin the
    Blackhole value so the groups below do not depend on the host running the test."""
    monkeypatch.setenv("GEMMA4_MAX_BATCHED_PREFILL_USERS", "32")


@pytest.mark.usefixtures("_blackhole_user_cap")
def test_one_bucket_of_short_prompts_is_one_call():
    groups = plan_packed_prefill_groups([128] * 32, [0] * 32, max_prefill_chunk_size=4096, max_batch_size=32)
    assert groups == [(128, list(range(32)))]


@pytest.mark.usefixtures("_blackhole_user_cap")
def test_hetero_actual_lengths_share_the_padded_bucket():
    groups = plan_packed_prefill_groups([100, 128, 65, 127], [0] * 4, max_prefill_chunk_size=4096, max_batch_size=32)
    assert groups == [(128, [0, 1, 2, 3])]


@pytest.mark.usefixtures("_blackhole_user_cap")
def test_buckets_split_and_singles_stay_on_the_stock_path():
    lens = [128, 2048, 128, 300, 128, 300]
    groups = plan_packed_prefill_groups(lens, [0] * 6, max_prefill_chunk_size=4096, max_batch_size=32)
    assert groups == [(128, [0, 2, 4]), (get_padded_prefill_len(300), [3, 5])]  # 2048 alone: stock path


@pytest.mark.usefixtures("_blackhole_user_cap")
def test_cached_continuations_and_long_prompts_are_excluded():
    lens = [128, 128, 128, 8192, 8192]
    cached = [0, 64, 0, 0, 0]
    groups = plan_packed_prefill_groups(lens, cached, max_prefill_chunk_size=4096, max_batch_size=32)
    assert groups == [(128, [0, 2])]


@pytest.mark.usefixtures("_blackhole_user_cap")
def test_token_budget_splits_a_wide_bucket():
    # Default budget = the single-pass prefill budget (4096): 32 x 128 is one call, 32 x 1024 is eight calls of 4,
    # 32 x 4096 packs nothing; a 2048 budget splits 32 x 128 into two calls of 16.
    groups = plan_packed_prefill_groups([1024] * 32, [0] * 32, max_prefill_chunk_size=4096, max_batch_size=32)
    assert [len(g[1]) for g in groups] == [4] * 8
    assert groups[0][1] == [0, 1, 2, 3] and groups[-1][1] == [28, 29, 30, 31]
    groups = plan_packed_prefill_groups(
        [128] * 32, [0] * 32, max_prefill_chunk_size=4096, max_batch_size=32, max_tokens=2048
    )
    assert [len(g[1]) for g in groups] == [16, 16]
    assert groups[0][1] == list(range(16)) and groups[1][1] == list(range(16, 32))
    assert plan_packed_prefill_groups([4096] * 32, [0] * 32, max_prefill_chunk_size=4096, max_batch_size=32) == []
    # A raised budget stops at the batched kernel cap (128Ki tokens): 32 x 4096 is two calls of 16.
    groups = plan_packed_prefill_groups(
        [4096] * 32, [0] * 32, max_prefill_chunk_size=4096, max_batch_size=32, max_tokens=1 << 20
    )
    assert [len(g[1]) for g in groups] == [16, 16]


@pytest.mark.usefixtures("_blackhole_user_cap")
@pytest.mark.parametrize("slots", [[0, 1, 2, 3], [20, 21, 22, 23], [31, 0, 5, 9]])
def test_physical_slots_keep_the_call_within_max_batch(slots):
    groups = plan_packed_prefill_groups(
        [128] * 4, [0] * 4, max_prefill_chunk_size=4096, empty_slots=slots, max_batch_size=32
    )
    assert groups == [(128, [0, 1, 2, 3])]


@pytest.mark.usefixtures("_blackhole_user_cap")
def test_nothing_to_pack():
    assert plan_packed_prefill_groups([128], [0], max_prefill_chunk_size=4096, max_batch_size=32) == []
    assert plan_packed_prefill_groups([128, 256], [0, 0], max_prefill_chunk_size=4096, max_batch_size=32) == []
