# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Device unit tests for ``tt/functional_decoder.py``: one full-width decoder layer plus its per-user cache.

``FunctionalDecoder`` is hard-wired to the model's hidden size (packed streams ``[1, 1, T, 4 * 4096]`` fp32), so the
layers here are full width -- 64 x 128 KDA heads, 64-head MLA with the 1536 / 512 low-rank widths -- with only the
FFN shrunk (dense I = 1024; 16 experts top-2 of I = 256). Each one is built with ``from_state_dict`` from the state
dict of a transformers ``Glm5NextTextDecoderLayer`` and graded against that HF layer's forward over the whole
sequence (both attention kinds are causal, so decode step ``t`` must equal position ``t``). MLA runs in the DSA
dense-exact regime (kv_len <= 2051), where the indexer selects every visible token.

KDA prefill needs ``prepare_chunk_recurrence``, which does not fit Wormhole (``test_kda.py::PREP_BLOCKER``).
"""

from __future__ import annotations

import pytest
import torch

import ttnn
from models.experimental.glm53_flash.tests.test_kda import PREP_BLOCKER
from models.experimental.glm53_flash.tests.test_kda_block import (
    assert_bf4_experts,
    bf4_round_experts,
    bf16_round,
    init_like_checkpoint,
    pcc,
)
from models.experimental.glm53_flash.tt.config import Glm5NextTextConfig as TtConfig
from models.experimental.glm53_flash.tt.functional_decoder import (
    HC_WIDTH,
    PREFILL_CHUNK,
    FunctionalDecoder,
    bucket_for,
    plan_prefill_chunks,
)

transformers_glm = pytest.importorskip("transformers.models.glm5_next.modeling_glm5_next")
from transformers.models.glm5_next.configuration_glm5_next import Glm5NextTextConfig as HfConfig  # noqa: E402

pytestmark = pytest.mark.use_module_device({"l1_small_size": 24576, "trace_region_size": 64 << 20})

HIDDEN, HC = 4096, 4
KDA_DENSE, MLA_MOE, KDA_MOE = 0, 3, 4
NUM_LAYERS = 5
LAYER_TYPES = ["linear_attention"] * 3 + ["indexed_attention", "linear_attention"]
MLP_TYPES = ["dense"] * 3 + ["sparse"] * 2
SMALL_FFN = dict(intermediate_size=1024, moe_intermediate_size=256, n_routed_experts=16, num_experts_per_tok=2)
MAX_SEQ = 128


def hf_config() -> HfConfig:
    return HfConfig(num_hidden_layers=NUM_LAYERS, layer_types=LAYER_TYPES, mlp_layer_types=MLP_TYPES, **SMALL_FFN)


def tt_config() -> TtConfig:
    return TtConfig(num_hidden_layers=NUM_LAYERS, layer_types=LAYER_TYPES, mlp_layer_types=MLP_TYPES, **SMALL_FFN)


# ---------------------------------------------------------------------------------------------
# Host helpers
# ---------------------------------------------------------------------------------------------


def packed(streams: torch.Tensor) -> torch.Tensor:
    """HF ``[B, S, hc, D]`` -> the decoder's packed rows ``[B * S, hc * D]`` (stream i in columns [D i, D (i + 1)))."""
    return streams.reshape(-1, HC_WIDTH)


def to_dev_packed(rows: torch.Tensor, device) -> ttnn.Tensor:
    return ttnn.from_torch(
        rows.reshape(1, 1, rows.shape[0], HC_WIDTH).float(),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def positions(device, pos: list[int]) -> ttnn.Tensor:
    return ttnn.from_torch(
        torch.tensor(pos, dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )


def hf_forward(layer, streams: torch.Tensor) -> torch.Tensor:
    """The HF layer over whole sequences ``[B, S, hc, D]``, called the way ``Glm5NextTextModel.forward`` calls it
    (NoPE: no position embeddings; the all-true mask; no cache)."""
    b, s = streams.shape[:2]
    with torch.no_grad():
        return layer(
            streams,
            attention_mask=torch.ones(b, s, dtype=torch.bool),
            position_ids=torch.arange(s)[None].expand(b, -1),
            position_embeddings=None,
            input_ids=torch.zeros(b, s, dtype=torch.long),
        )[0]


ROUTER_TIE_MARGIN = 4e-3


def routing_ties(layer, streams: torch.Tensor) -> torch.Tensor:
    """``[B, S]`` bool: tokens whose HF router is a near-tie at the top-k boundary (k-th vs (k+1)-th corrected score
    within ``ROUTER_TIE_MARGIN``). bf16 rounding of the router input can legitimately pick the other expert there,
    which changes that token's whole FFN output."""
    b, s = streams.shape[:2]
    if not hasattr(layer.mlp, "experts"):
        return torch.zeros(b, s, dtype=torch.bool)
    cap = {}
    hook = layer.mlp.register_forward_hook(lambda m, i, o: cap.setdefault("x", i[0].detach()))
    hf_forward(layer, streams)
    hook.remove()
    x = cap["x"].reshape(b * s, -1).float()
    scores = (x @ layer.mlp.gate.weight.float().t()).sigmoid() + layer.mlp.gate.e_score_correction_bias.float()
    k = layer.mlp.config.num_experts_per_tok
    top = scores.topk(k + 1, dim=-1).values
    return (top[:, k - 1] - top[:, k] < ROUTER_TIE_MARGIN).reshape(b, s)


def check_rows(got: torch.Tensor, ref: torch.Tensor, inp: torch.Tensor, what: str):
    """Full output PCC and, separately, PCC of what the layer added to the residual (the stream mix alone would score
    high on the output even with a broken sublayer)."""
    assert pcc(got, ref) > 0.999, f"{what}: output PCC {pcc(got, ref):.6f}"
    assert pcc(got - inp, ref - inp) > 0.98, f"{what}: residual-update PCC {pcc(got - inp, ref - inp):.5f}"


# ---------------------------------------------------------------------------------------------
# Fixtures: one HF layer + its FunctionalDecoder per layer kind (built once per module)
# ---------------------------------------------------------------------------------------------


def hf_layer(idx: int, *, bf4_experts: bool = True):
    ref = init_like_checkpoint(transformers_glm.Glm5NextTextDecoderLayer(hf_config(), idx), seed=idx)
    return bf4_round_experts(ref) if bf4_experts else ref


@pytest.fixture(scope="module")
def layers(_device_module_impl):
    tcfg = tt_config()
    out = {}
    for idx in (KDA_DENSE, MLA_MOE, KDA_MOE):
        ref = hf_layer(idx)
        dec = FunctionalDecoder.from_state_dict(
            ref.state_dict(), hf_config=tcfg, layer_idx=idx, mesh_device=_device_module_impl
        )
        if dec.layer.spec.mlp == "moe":
            assert_bf4_experts(dec.layer.mlp)
        out[idx] = (ref, dec)
    return out


def _decode_steps(dec, cache, streams: torch.Tensor, page_table=None, active_pos=None):
    """Decode ``streams`` [B, S, hc, D] one position per step for all B slots -> [B, S, hc, D] (host)."""
    b, s = streams.shape[:2]
    outs = []
    for t in range(s):
        pos = [t] * b if active_pos is None else active_pos(t)
        y = dec.decode_forward(
            to_dev_packed(packed(streams[:, t : t + 1]), dec.device),
            cache=cache,
            page_table=page_table,
            cur_pos=positions(dec.device, pos),
        )
        outs.append(ttnn.to_torch(y).float().reshape(b, 1, HC, HIDDEN))
    return torch.cat(outs, dim=1)


# ---------------------------------------------------------------------------------------------
# Decode, every layer kind
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("idx", [KDA_DENSE, KDA_MOE, MLA_MOE], ids=["kda_dense", "kda_moe", "mla_moe"])
@pytest.mark.parametrize("users", [1, 2])
def test_decode_matches_hf(layers, idx, users):
    ref, dec = layers[idx]
    torch.manual_seed(100 + idx + users)
    steps = 4
    streams = bf16_round(torch.randn(users, steps, HC, HIDDEN))
    expected = hf_forward(ref, streams)

    cache = dec.allocate_cache(max_batch=users, max_seq_len=MAX_SEQ)
    page_table = None if dec.is_kda else dec.make_page_table(cache)
    got = _decode_steps(dec, cache, streams, page_table)
    for t in range(steps):
        check_rows(got[:, t], expected[:, t], streams[:, t], f"layer {idx} step {t}")


def test_bf4_expert_storage_budget(layers):
    """Against the unrounded HF weights (same seed; bf4 rounding is idempotent, so the device holds the same values)
    the MoE layer stays within the bfloat4_b expert-storage budget."""
    _, dec = layers[KDA_MOE]
    ref = hf_layer(KDA_MOE, bf4_experts=False)
    torch.manual_seed(105)
    streams = bf16_round(torch.randn(1, 2, HC, HIDDEN))
    expected = hf_forward(ref, streams)
    got = _decode_steps(dec, dec.allocate_cache(max_batch=1, max_seq_len=MAX_SEQ), streams)
    assert pcc(got, expected) > 0.98, f"PCC {pcc(got, expected):.5f}"


@pytest.mark.parametrize("idx", [KDA_MOE, MLA_MOE], ids=["kda_moe", "mla_moe"])
def test_decode_inactive_slot(layers, idx):
    """cur_pos == -1 leaves a slot untouched: user 1 idles for step 1 (its row holds junk) and then resumes with its
    own step 1, which must equal its uninterrupted sequence. User 0 is unaffected throughout."""
    ref, dec = layers[idx]
    torch.manual_seed(110 + idx)
    streams = bf16_round(torch.randn(2, 2, HC, HIDDEN))
    junk = bf16_round(torch.randn(1, 1, HC, HIDDEN))
    expected = hf_forward(ref, streams)
    cache = dec.allocate_cache(max_batch=2, max_seq_len=MAX_SEQ)
    pt = None if dec.is_kda else dec.make_page_table(cache)

    def step(rows, pos):
        y = dec.decode_forward(
            to_dev_packed(packed(rows), dec.device), cache=cache, page_table=pt, cur_pos=positions(dec.device, pos)
        )
        return ttnn.to_torch(y).float().reshape(2, HC, HIDDEN)

    step(streams[:, 0:1], [0, 0])
    y1 = step(torch.cat([streams[0:1, 1:2], junk]), [1, -1])
    check_rows(y1[0], expected[0, 1], streams[0, 1], "user 0 step 1")
    y2 = step(torch.cat([junk, streams[1:2, 1:2]]), [-1, 1])
    check_rows(y2[1], expected[1, 1], streams[1, 1], "user 1 step 1 after idling")


def test_reset_slot_starts_a_fresh_request(layers):
    """After reset_slot a KDA slot decodes as if new: same output as a fresh cache for the same first token."""
    ref, dec = layers[KDA_DENSE]
    torch.manual_seed(120)
    streams = bf16_round(torch.randn(1, 3, HC, HIDDEN))
    expected = hf_forward(ref, streams[:, :1])
    cache = dec.allocate_cache(max_batch=1, max_seq_len=MAX_SEQ)
    _decode_steps(dec, cache, streams)  # dirty the slot
    dec.reset_slot(cache, 0)
    assert float(ttnn.to_torch(cache.kda.recurrent[0]).abs().max()) == 0.0
    assert float(ttnn.to_torch(cache.kda.conv[0]).abs().max()) == 0.0
    got = _decode_steps(dec, cache, streams[:, :1])
    check_rows(got[:, 0], expected[:, 0], streams[:, 0], "first token after reset")


def test_snapshot_restore_roundtrip(layers):
    """restore_state(snapshot_state()) puts the KDA state back bit-exactly, so a replayed step reproduces itself."""
    _, dec = layers[KDA_MOE]
    torch.manual_seed(121)
    streams = bf16_round(torch.randn(1, 2, HC, HIDDEN))
    cache = dec.allocate_cache(max_batch=1, max_seq_len=MAX_SEQ)
    _decode_steps(dec, cache, streams[:, :1])
    snap = dec.snapshot_state(cache)
    s_before = ttnn.to_torch(cache.kda.recurrent[0]).clone()
    first = _decode_steps(dec, cache, streams[:, 1:2], active_pos=lambda t: [1])
    assert not torch.equal(ttnn.to_torch(cache.kda.recurrent[0]), s_before)
    dec.restore_state(cache, snap)
    assert torch.equal(ttnn.to_torch(cache.kda.recurrent[0]), s_before)
    again = _decode_steps(dec, cache, streams[:, 1:2], active_pos=lambda t: [1])
    assert torch.equal(first, again)


def test_decode_trace_replays_eager(layers):
    """capture_decode_trace + replay over persistent inputs gives the eager result for the same state."""
    _, dec = layers[KDA_DENSE]
    torch.manual_seed(122)
    rows = packed(bf16_round(torch.randn(1, 1, HC, HIDDEN)))
    cache = dec.allocate_cache(max_batch=1, max_seq_len=MAX_SEQ)
    streams = to_dev_packed(rows, dec.device)
    cur_pos = positions(dec.device, [0])
    snap = dec.snapshot_state(cache)
    eager = ttnn.to_torch(dec.decode_forward(streams, cache=cache, cur_pos=cur_pos)).float()
    dec.restore_state(cache, snap)
    trace = dec.capture_decode_trace(streams, cache=cache, cur_pos=cur_pos)
    try:
        dec.restore_state(cache, snap)
        replayed = ttnn.to_torch(trace.replay(blocking=True)).float()
    finally:
        trace.release()
    assert torch.equal(replayed, eager)


# ---------------------------------------------------------------------------------------------
# Prefill
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("seq_len", [5, 64, 70], ids=["partial_chunk", "one_page", "two_pages"])
def test_mla_prefill_then_decode_matches_hf(layers, seq_len):
    """prefill_forward over a prompt (padded to the 64-token page), then decode continuing from the latent cache."""
    ref, dec = layers[MLA_MOE]
    torch.manual_seed(130 + seq_len)
    dec_steps = 2
    streams = bf16_round(torch.randn(1, seq_len + dec_steps, HC, HIDDEN))
    expected = hf_forward(ref, streams)
    cache = dec.allocate_cache(max_batch=1, max_seq_len=MAX_SEQ)
    pt = dec.make_page_table(cache)
    got = dec.prefill_forward(streams[:, :seq_len], cache=cache, user_slot=0).reshape(1, seq_len, HC, HIDDEN)
    keep = ~routing_ties(ref, streams[:, :seq_len])
    assert keep.float().mean() > 0.75, "too many router near-ties for a meaningful check"
    check_rows(got[keep], expected[:, :seq_len][keep], streams[:, :seq_len][keep], f"prefill of {seq_len}")
    tail = _decode_steps(dec, cache, streams[:, seq_len:], pt, active_pos=lambda t: [seq_len + t])
    check_rows(tail, expected[:, seq_len:], streams[:, seq_len:], "decode after prefill")


def test_kda_prefill_then_decode_matches_hf(request, layers):
    ref, dec = layers[KDA_DENSE]
    if dec.device.arch() == ttnn.Arch.WORMHOLE_B0:
        request.applymarker(pytest.mark.xfail(strict=True, raises=RuntimeError, reason=PREP_BLOCKER))
    torch.manual_seed(131)
    streams = bf16_round(torch.randn(1, 34, HC, HIDDEN))
    expected = hf_forward(ref, streams)
    cache = dec.allocate_cache(max_batch=1, max_seq_len=MAX_SEQ)
    got = dec.prefill_forward(streams[:, :32], cache=cache).reshape(1, 32, HC, HIDDEN)
    check_rows(got, expected[:, :32], streams[:, :32], "kda prefill")
    tail = _decode_steps(dec, cache, streams[:, 32:], active_pos=lambda t: [32 + t])
    check_rows(tail, expected[:, 32:], streams[:, 32:], "decode after kda prefill")


# ---------------------------------------------------------------------------------------------
# Cache allocation and host-side validation
# ---------------------------------------------------------------------------------------------


def test_allocate_cache_shapes(layers):
    _, kda = layers[KDA_DENSE]
    c = kda.allocate_cache(max_batch=3, max_seq_len=MAX_SEQ)
    assert c.kda.slots == 3 and c.latent is None
    assert tuple(c.kda.recurrent[0].shape) == (1, 64, 128, 128)
    assert tuple(c.kda.conv[0].shape) == (1, 3, 3 * 64 * 128)

    _, mla = layers[MLA_MOE]
    c = mla.allocate_cache(max_batch=2, max_seq_len=MAX_SEQ)
    assert c.kda is None
    # MAX_SEQ + the padded tail of a chunk (< 64 rows) in 64-token pages, rounded to 8 table entries
    assert c.blocks_per_seq == 8
    assert c.latent.shape[0] == 2 * c.blocks_per_seq
    pt = mla.make_page_table(c)
    assert tuple(pt.shape) == (2, c.blocks_per_seq)
    assert torch.equal(c.page_table_host.flatten(), torch.arange(2 * c.blocks_per_seq, dtype=torch.int32))


def test_mla_cache_refuses_sparse_regime(layers, expect_error):
    _, mla = layers[MLA_MOE]
    with expect_error(NotImplementedError, "DSA indexer"):
        mla.allocate_cache(max_batch=1, max_seq_len=4096)


def test_update_page_table_validates_permutation(layers, expect_error):
    _, mla = layers[MLA_MOE]
    c = mla.allocate_cache(max_batch=1, max_seq_len=MAX_SEQ)
    pt = mla.make_page_table(c)
    perm = torch.arange(c.blocks_per_seq).flip(0)
    mla.update_page_table(c, pt, perm)
    assert torch.equal(ttnn.to_torch(pt).flatten(), perm.to(torch.int32))
    with expect_error(ValueError, "permutation"):
        mla.update_page_table(c, pt, torch.zeros(c.blocks_per_seq, dtype=torch.int32))


def test_check_decode_positions(layers, expect_error):
    _, kda = layers[KDA_DENSE]
    _, mla = layers[MLA_MOE]
    c = kda.allocate_cache(max_batch=2, max_seq_len=MAX_SEQ)
    kda.check_decode_positions([0, -1], c)
    with expect_error(ValueError, "decode slots"):
        kda.check_decode_positions([0], c)
    with expect_error(ValueError, "outside"):
        kda.check_decode_positions([0, MAX_SEQ], c)
    with expect_error(ValueError, "outside"):
        kda.check_decode_positions([-2, 0], c)
    kda.check_decode_positions([1_000_000])  # KDA has no positional limit
    with expect_error(NotImplementedError, ""):
        mla.check_decode_positions([4096])


def test_begin_prefill_chunk_validation(layers, expect_error):
    _, mla = layers[MLA_MOE]
    c = mla.allocate_cache(max_batch=1, max_seq_len=MAX_SEQ)
    with expect_error(ValueError, "page table"):
        mla.begin_prefill_chunk(c, user_slot=0, chunk_start=0, n_valid=8)
    mla.make_page_table(c)
    with expect_error(ValueError, "user_slot"):
        mla.begin_prefill_chunk(c, user_slot=1, chunk_start=0, n_valid=8)
    with expect_error(ValueError, "multiple of 64"):
        mla.begin_prefill_chunk(c, user_slot=0, chunk_start=10, n_valid=8)
    with expect_error(ValueError, "invalid"):
        mla.begin_prefill_chunk(c, user_slot=0, chunk_start=64, n_valid=MAX_SEQ)
    with expect_error(ValueError, "decode_forward needs cur_pos"):
        mla.decode_forward(None, cache=c)


def test_chunk_planning(expect_error):
    assert bucket_for(1, 32) == 32 and bucket_for(33, 32) == 64 and bucket_for(64, 64) == 64
    with expect_error(ValueError, ""):
        bucket_for(0, 32)
    with expect_error(ValueError, ""):
        bucket_for(PREFILL_CHUNK + 1, 32)
    assert plan_prefill_chunks(5, 64) == [(0, 5, 64)]
    assert plan_prefill_chunks(PREFILL_CHUNK + 40, 32) == [(0, PREFILL_CHUNK, PREFILL_CHUNK), (PREFILL_CHUNK, 40, 64)]
    with expect_error(ValueError, ""):
        plan_prefill_chunks(0, 32)
