# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Correctness tests for the Qwen3.6-35B-A3B TTNN functional decoder.

Every test uses the **real** ``text_config`` shapes (hidden 2048, head_dim 256, 16/2 GQA,
32 v-heads x 128 for the DeltaNet, 256 experts / top-8). Synthetic weights are generated
deterministically from real-checkpoint per-tensor statistics
(``doc/functional_decoder/weight_stats/``) so CI needs no checkpoint download; the
``real_weights`` tests read the actual checkpoint.

Both decoder layer kinds are covered everywhere:
  * ``linear`` -> layer 0, ``Qwen3_5MoeGatedDeltaNet`` + MoE
  * ``full``   -> layer 3, ``Qwen3_5MoeAttention`` (output-gated, partial RoPE) + MoE
"""

import inspect
import json
import re

import pytest
import torch

import ttnn
from models.autoports.qwen_qwen3_6_35b_a3b.tests import harness
from models.autoports.qwen_qwen3_6_35b_a3b.tests.harness import (
    TracedDecode,
    compare,
    decode_and_compare,
    from_tt,
    prefill_and_compare,
    read_kv_cache,
    read_tt_linear_state,
    record,
    restore_state,
    seed_tt_linear_state,
    snapshot_state,
    to_tt_decode,
    to_tt_positions,
    to_tt_prefill,
    tt_conv_state_to_hf,
)
from models.autoports.qwen_qwen3_6_35b_a3b.tt import functional_decoder as fd
from models.autoports.qwen_qwen3_6_35b_a3b.tt import reference as ref

#: Functional-decoder acceptance bar (skill default; no model-specific exception needed —
#: measured margin is ~3 orders of magnitude, see doc/functional_decoder/README.md).
PCC_BAR = 0.995

KINDS = ["linear", "full"]


def _assert_pcc(result, bar=PCC_BAR, log="pcc"):
    record(result, log)
    assert result.pcc >= bar, f"{result} (bar {bar})"


# =======================================================================================
# capability contract
# =======================================================================================
def test_layer_kinds_cover_the_whole_model():
    """The two tested layer indices really are the only distinct decoder kinds."""
    cfg = ref.load_hf_text_config()
    kinds = set(cfg.layer_types)
    assert kinds == {"linear_attention", "full_attention"}, kinds
    assert len(cfg.layer_types) == cfg.num_hidden_layers == 40
    assert cfg.layer_types[ref.LINEAR_ATTENTION_LAYER_IDX] == "linear_attention"
    assert cfg.layer_types[ref.FULL_ATTENTION_LAYER_IDX] == "full_attention"
    # every full_attention layer is structurally identical to layer 3, and likewise for
    # linear layers, so one index per kind is complete coverage
    assert [i for i, t in enumerate(cfg.layer_types) if t == "full_attention"] == list(range(3, 40, 4))


def test_config_matches_hf():
    """DecoderConfig derives, not guesses, every shape from the HF config."""
    hf = ref.load_hf_text_config()
    for kind, idx in (("linear", 0), ("full", 3)):
        cfg = fd.DecoderConfig.from_hf(hf, idx, mesh_device=None, supported_context=262144)
        assert cfg.layer_kind == hf.layer_types[idx]
        assert cfg.hidden_size == 2048
        assert cfg.head_dim == 256 and cfg.num_attention_heads == 16 and cfg.num_key_value_heads == 2
        assert cfg.rotary_dim == 64  # head_dim * partial_rotary_factor(0.25)
        assert cfg.num_experts == 256 and cfg.num_experts_per_tok == 8
        assert cfg.moe_intermediate_size == 512 and cfg.shared_expert_intermediate_size == 512
        assert cfg.hf_max_position_embeddings == 262144
        # DeltaNet: q/k duplicated 16 -> 32 heads, plus the z block riding the conv
        assert cfg.delta_qkv_width == 3 * 32 * 128 == 12288
        assert cfg.conv_dim == 4 * 32 * 128 == 16384
        assert cfg.num_v_head_groups == 2
        del kind


# =======================================================================================
# prefill PCC
# =======================================================================================
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize(
    "seq_len",
    [
        1,  # single token through the prefill path
        32,  # exactly one tile
        33,  # just past a tile
        64,  # exactly one paged block / one delta chunk
        65,  # just past a block boundary
        128,  # exactly PREFILL_ALIGN
        129,  # just past PREFILL_ALIGN
        1024,
        1025,
        2048,  # exactly the internal prefill chunk
        2049,  # just past the internal prefill chunk (forces chunk continuation)
        3000,  # long, divisible by no boundary in play
        4096,
    ],
)
@pytest.mark.timeout(1800)
def test_prefill_pcc_sequence_lengths(layer_pairs, kind, seq_len):
    """Any logical length is accepted; padding/masking is the layer's job, not the caller's.

    Boundaries in play: tile 32, paged block 64, delta-rule chunk 64, PREFILL_ALIGN 128,
    internal prefill chunk 2048, MoE tile group 32.
    """
    pair = layer_pairs(kind, supported_context=8192)
    result = prefill_and_compare(pair, seq_len=seq_len, seed=seq_len)
    _assert_pcc(result)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.timeout(1800)
def test_prefill_chunk_continuation(layer_pairs, kind):
    """Streaming a prompt as several prefill calls == one long call.

    Exercises the carried KV cache (page-table block offset per chunk) for full attention
    and the carried conv + recurrent state for linear attention.
    """
    pair = layer_pairs(kind, supported_context=8192)
    pair.tt.reset_state()
    total, step = 1024, 512
    x = ref.synthetic_hidden_states(pair.hf_config, 1, total, seed=41)

    cache = ref.make_cache(pair.hf_config)
    outs = []
    for start in range(0, total, step):
        piece = x[:, start : start + step]
        want = ref.hf_prefill(pair.hf, pair.hf_config, piece, start_pos=start, cache=cache).output
        tt_x = to_tt_prefill(pair.device, piece)
        tt_out = pair.tt.prefill_forward(tt_x, user_id=0, page_table=pair.page_table, start_pos=start)
        outs.append(compare(f"prefill-cont[{kind}] start={start}", from_tt(tt_out), want, start_pos=start))
        ttnn.deallocate(tt_x)
        ttnn.deallocate(tt_out)
    for result in outs:
        _assert_pcc(result)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.timeout(2400)
def test_prefill_per_user_slots(layer_pairs, kind):
    """32 sequences prefilled into 32 distinct slots stay independent.

    For ``full`` this is 32 page-table rows into a shuffled physical block pool; for
    ``linear`` it is 32 conv/recurrent state slots. Each slot is re-compared against its own
    HF run *after* all 32 have been written, so a slot that leaked into another fails.
    """
    batch = 32
    pair = layer_pairs(kind, max_batch_size=batch, supported_context=1024)
    pair.tt.reset_state()
    seq = 256
    inputs = [ref.synthetic_hidden_states(pair.hf_config, 1, seq, seed=500 + i) for i in range(batch)]
    outs = []
    for user_id, x in enumerate(inputs):
        tt_x = to_tt_prefill(pair.device, x)
        tt_out = pair.tt.prefill_forward(tt_x, user_id=user_id, page_table=pair.page_table)
        outs.append(from_tt(tt_out))
        ttnn.deallocate(tt_x)
        ttnn.deallocate(tt_out)

    for user_id, (x, got) in enumerate(zip(inputs, outs)):
        want = ref.hf_prefill(pair.hf, pair.hf_config, x).output
        _assert_pcc(compare(f"prefill-slot[{kind}] user={user_id}", got, want, user_id=user_id))


def test_prefill_covering_whole_context_does_not_free_weights(layer_pairs):
    """Regression: a prefill chunk that spans the entire RoPE table must not free it.

    ``ttnn.slice`` returns an *alias* when the slice covers the whole tensor, so slicing the
    persistent cos/sin tables for a chunk at ``abs_pos == 0`` whose length equals
    ``supported_context`` and then deallocating the result would free the layer's own weights.
    Reachable whenever ``supported_context <= prefill_chunk_size``. The second forward is the
    assertion: it only works if the tables survived the first.
    """
    context = 1024  # <= prefill_chunk_size (2048), so the whole context is one chunk
    pair = layer_pairs("full", max_batch_size=1, supported_context=context)
    pair.tt.reset_state()
    # padded_len == context exactly, so the rope slice covers the whole table
    x = ref.synthetic_hidden_states(pair.hf_config, 1, context, seed=61)
    first = prefill_and_compare(pair, seq_len=context, hidden_states=x, seed=61)
    _assert_pcc(first)

    # same layer, second call: fails with "Tensor is not allocated" if the tables were freed
    second = prefill_and_compare(pair, seq_len=context - 24, seed=62)
    _assert_pcc(second)
    third = decode_and_compare(pair, prefill_len=context - 24, steps=1, seed=63)
    for result in third:
        _assert_pcc(result)


@pytest.mark.parametrize("kind", KINDS)
def test_prefill_resets_linear_state_for_new_sequence(layer_pairs, kind):
    """A slot reused for a new sequence must not inherit the previous sequence's state.

    ``start_pos == 0`` means "this slot starts a new sequence". Full attention self-heals — the
    prefill rewrites every paged block it will later read — but the DeltaNet conv left-context and
    recurrent state are a running summary that would silently continue the previous occupant's
    sequence, which is exactly the multi-request case a serving stage hits.

    Deliberately does **not** go through ``prefill_and_compare``: that helper calls
    ``reset_state()`` for ``start_pos == 0``, which is what hid this. Here slot 0 is dirtied by a
    first sequence and the second prefill is compared against a *fresh* HF layer state, with no
    reset in between.
    """
    # (kind, 2, 1024) on purpose: every distinct layer_pairs key materialises another ~1.5 GiB of
    # expert weights for the whole session, so tests reuse keys rather than inventing them.
    pair = layer_pairs(kind, max_batch_size=2, supported_context=1024)
    pair.tt.reset_state()

    # 1. dirty the slot with a first sequence
    dirty = ref.synthetic_hidden_states(pair.hf_config, 1, 1024, seed=81)
    tt_dirty = to_tt_prefill(pair.device, dirty)
    ttnn.deallocate(pair.tt.prefill_forward(tt_dirty, user_id=0, page_table=pair.page_table))
    ttnn.deallocate(tt_dirty)

    # 2. a new sequence in the same slot, no reset_state() anywhere
    x = ref.synthetic_hidden_states(pair.hf_config, 1, 640, seed=82)
    tt_x = to_tt_prefill(pair.device, x)
    tt_out = pair.tt.prefill_forward(tt_x, user_id=0, page_table=pair.page_table)
    got = from_tt(tt_out)
    ttnn.deallocate(tt_out)
    ttnn.deallocate(tt_x)

    # 3. HF, from scratch: this is what "new sequence" has to mean
    want = ref.hf_prefill(pair.hf, pair.hf_config, x, start_pos=0, cache=ref.make_cache(pair.hf_config)).output
    _assert_pcc(compare(f"prefill-fresh-slot[{kind}] seq=640", got, want))


def test_decode_forward_rejects_out_of_contract_inputs(layer_pairs, expect_error):
    """The documented API constraints are enforced, not just written down."""
    pair = layer_pairs("full", max_batch_size=2, supported_context=1024)
    cfg = pair.cfg
    x = to_tt_decode(pair.device, ref.synthetic_hidden_states(pair.hf_config, 2, 1, seed=71).reshape(2, 1, -1))
    pos = to_tt_positions(pair.device, torch.tensor([0, 0]))
    try:
        with expect_error(ValueError, "page_table"):
            pair.tt.decode_forward(x, current_pos=pos, page_table=None)
        with expect_error(ValueError, "current_pos"):
            pair.tt.decode_forward(x, current_pos=None, page_table=pair.page_table)
        wrong_batch = to_tt_decode(
            pair.device, ref.synthetic_hidden_states(pair.hf_config, 1, 1, seed=72).reshape(1, 1, -1)
        )
        with expect_error(ValueError, "max_batch_size"):
            pair.tt.decode_forward(wrong_batch, current_pos=pos, page_table=pair.page_table)
        ttnn.deallocate(wrong_batch)

        prefill_x = to_tt_prefill(pair.device, ref.synthetic_hidden_states(pair.hf_config, 1, 128, seed=73))
        with expect_error(ValueError, "multiple of"):
            pair.tt.prefill_forward(prefill_x, user_id=0, page_table=pair.page_table, start_pos=64)
        with expect_error(ValueError, "exceeds supported context"):
            pair.tt.prefill_forward(prefill_x, user_id=0, page_table=pair.page_table, start_pos=cfg.supported_context)
        with expect_error(ValueError, "user_id"):
            pair.tt.prefill_forward(prefill_x, user_id=cfg.max_batch_size, page_table=pair.page_table)
        with expect_error(ValueError, "page_table"):
            pair.tt.prefill_forward(prefill_x, user_id=0, page_table=None)
        ttnn.deallocate(prefill_x)
    finally:
        ttnn.deallocate(x)
        ttnn.deallocate(pos)


# =======================================================================================
# decode PCC
# =======================================================================================
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("batch", [1, 8, 32])
@pytest.mark.timeout(2400)
def test_decode_pcc_batches(layer_pairs, kind, batch):
    """Batched paged decode after a per-slot prefill, several steps."""
    pair = layer_pairs(kind, max_batch_size=batch, supported_context=2048)
    results = decode_and_compare(pair, prefill_len=256, steps=3, seed=batch)
    for result in results:
        _assert_pcc(result)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.timeout(1800)
def test_decode_many_steps_no_drift(layer_pairs, kind):
    """64 consecutive decode steps: the recurrent state / KV cache must not drift."""
    pair = layer_pairs(kind, max_batch_size=1, supported_context=2048)
    results = decode_and_compare(pair, prefill_len=128, steps=64, seed=17)
    worst = min(results, key=lambda r: r.pcc)
    record(results, "pcc")
    assert worst.pcc >= PCC_BAR, f"worst of 64 steps: {worst}"
    assert results[-1].pcc >= PCC_BAR, f"last step: {results[-1]}"


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("prefill_len", [65, 129, 320, 1000])
@pytest.mark.timeout(1800)
def test_decode_after_non_aligned_prefill(layer_pairs, kind, prefill_len):
    """Decode must continue from the last *logical* token of a non-aligned prefill.

    Regression test for padded prefill tokens leaking into the carried state: prefill pads
    up to PREFILL_ALIGN, and for the DeltaNet layer the padded rows would otherwise advance
    the conv and recurrent state past the end of the real sequence. Output PCC alone does not
    catch it (the padded outputs are sliced off), only the next decode step does.
    """
    pair = layer_pairs(kind, max_batch_size=1, supported_context=2048)
    results = decode_and_compare(pair, prefill_len=prefill_len, steps=2, seed=prefill_len)
    for result in results:
        _assert_pcc(result)


@pytest.mark.timeout(1800)
def test_decode_ragged_current_positions(layer_pairs):
    """Per-slot ``current_pos`` really is honoured (full attention).

    Each slot is prefilled to a different length, so the KV write index, the SDPA context
    length and the RoPE lookup all differ per row within one batched call. A layer that
    used a single shared position (or the batch max) fails here.
    """
    batch = 4
    pair = layer_pairs("full", max_batch_size=batch, supported_context=2048)
    pair.tt.reset_state()
    lengths = [128, 256, 384, 640]
    caches, tokens = [], []
    for user_id, length in enumerate(lengths):
        x = ref.synthetic_hidden_states(pair.hf_config, 1, length, seed=800 + user_id)
        tt_x = to_tt_prefill(pair.device, x)
        ttnn.deallocate(pair.tt.prefill_forward(tt_x, user_id=user_id, page_table=pair.page_table))
        ttnn.deallocate(tt_x)
        caches.append(ref.hf_prefill(pair.hf, pair.hf_config, x).cache)
        tokens.append(ref.synthetic_hidden_states(pair.hf_config, 1, 1, seed=900 + user_id).reshape(-1))

    token_batch = torch.stack(tokens).reshape(batch, 1, pair.cfg.hidden_size)
    positions = torch.tensor(lengths, dtype=torch.int32)
    tt_x = to_tt_decode(pair.device, token_batch)
    tt_pos = to_tt_positions(pair.device, positions)
    tt_out = pair.tt.decode_forward(tt_x, current_pos=tt_pos, page_table=pair.page_table)
    got = from_tt(tt_out).reshape(batch, 1, pair.cfg.hidden_size)
    for t in (tt_x, tt_pos, tt_out):
        ttnn.deallocate(t)

    for user_id, length in enumerate(lengths):
        want = ref.hf_decode(
            pair.hf,
            pair.hf_config,
            token_batch[user_id : user_id + 1],
            positions=torch.tensor([length]),
            cache=caches[user_id],
        )
        _assert_pcc(compare(f"decode-ragged user={user_id} pos={length}", got[user_id : user_id + 1], want))


def test_decode_skips_inactive_slots_with_negative_position(layer_pairs):
    """``current_pos = -1`` on a slot must not disturb the active slots or their cache.

    ``paged_scaled_dot_product_attention_decode`` documents -1 as "skip this batch index".
    This pins what the *whole layer* does with it: active slots still match HF, and the paged
    K/V of an inactive slot is left exactly as it was (i.e. the cache update does not scribble
    through a negative index).
    """
    batch = 4
    pair = layer_pairs("full", max_batch_size=batch, supported_context=1024)
    pair.tt.reset_state()
    prefill_len = 128
    caches = []
    for user_id in range(batch):
        x = ref.synthetic_hidden_states(pair.hf_config, 1, prefill_len, seed=2000 + user_id)
        tt_x = to_tt_prefill(pair.device, x)
        ttnn.deallocate(pair.tt.prefill_forward(tt_x, user_id=user_id, page_table=pair.page_table))
        ttnn.deallocate(tt_x)
        caches.append(ref.hf_prefill(pair.hf, pair.hf_config, x).cache)

    inactive = [1, 3]
    before = {u: read_kv_cache(pair, user_id=u, seq=prefill_len) for u in inactive}

    tokens = torch.stack(
        [ref.synthetic_hidden_states(pair.hf_config, 1, 1, seed=2100 + i).reshape(-1) for i in range(batch)]
    ).reshape(batch, 1, pair.cfg.hidden_size)
    positions = torch.tensor([prefill_len, -1, prefill_len, -1], dtype=torch.int32)
    tt_x = to_tt_decode(pair.device, tokens)
    tt_pos = to_tt_positions(pair.device, positions)
    tt_out = pair.tt.decode_forward(tt_x, current_pos=tt_pos, page_table=pair.page_table)
    got = from_tt(tt_out).reshape(batch, 1, pair.cfg.hidden_size)
    for t in (tt_x, tt_pos, tt_out):
        ttnn.deallocate(t)

    for user_id in (0, 2):
        want = ref.hf_decode(
            pair.hf,
            pair.hf_config,
            tokens[user_id : user_id + 1],
            positions=torch.tensor([prefill_len]),
            cache=caches[user_id],
        )
        _assert_pcc(compare(f"decode-active-slot user={user_id}", got[user_id : user_id + 1], want))

    for user_id in inactive:
        assert torch.isfinite(got[user_id]).all(), (
            f"inactive slot {user_id} produced non-finite output; current_pos=-1 must stay in "
            "bounds for the RoPE table lookup"
        )
        after = read_kv_cache(pair, user_id=user_id, seq=prefill_len)
        for name, old, new in zip(("keys", "values"), before[user_id], after):
            assert (
                float((old - new).abs().max()) == 0.0
            ), f"inactive slot {user_id} {name} cache changed under current_pos=-1"


# =======================================================================================
# paged cache / state behaviour
# =======================================================================================
def test_paged_kv_cache_contents_match_hf(layer_pairs):
    """The paged K/V written through a shuffled page table equals HF's cache."""
    pair = layer_pairs("full", max_batch_size=4, supported_context=1024)
    pair.tt.reset_state()
    seq = 512
    user_id = 2
    x = ref.synthetic_hidden_states(pair.hf_config, 1, seq, seed=61)
    tt_x = to_tt_prefill(pair.device, x)
    ttnn.deallocate(pair.tt.prefill_forward(tt_x, user_id=user_id, page_table=pair.page_table))
    ttnn.deallocate(tt_x)

    golden = ref.hf_prefill(pair.hf, pair.hf_config, x)
    want_k, want_v = golden.cache.layers[pair.layer_idx].keys, golden.cache.layers[pair.layer_idx].values
    got_k, got_v = read_kv_cache(pair, user_id=user_id, seq=seq)
    _assert_pcc(compare("paged-kv keys", got_k, want_k))
    _assert_pcc(compare("paged-kv values", got_v, want_v))

    # other slots must be untouched by this user's prefill
    for other in (0, 1, 3):
        other_k, _ = read_kv_cache(pair, user_id=other, seq=seq)
        assert float(other_k.abs().max()) == 0.0, f"slot {other} was written by user {user_id}"


def test_linear_state_matches_hf(layer_pairs):
    """Conv + recurrent state after prefill equals HF's, in HF's own layout."""
    pair = layer_pairs("linear", max_batch_size=2, supported_context=1024)
    pair.tt.reset_state()
    seq = 320
    x = ref.synthetic_hidden_states(pair.hf_config, 1, seq, seed=71)
    tt_x = to_tt_prefill(pair.device, x)
    ttnn.deallocate(pair.tt.prefill_forward(tt_x, user_id=1, page_table=None))
    ttnn.deallocate(tt_x)

    golden = ref.hf_prefill(pair.hf, pair.hf_config, x)
    want_conv, want_recurrent = ref.hf_linear_attention_state(golden.cache, pair.layer_idx)
    taps, recurrent = read_tt_linear_state(pair)

    got_conv = tt_conv_state_to_hf(pair, [tap[1:2] for tap in taps])
    # HF's oldest column is dead by construction; compare the 3 live ones
    _assert_pcc(compare("linear conv_state", got_conv[..., 1:], want_conv[..., 1:]))
    _assert_pcc(compare("linear recurrent_state", recurrent[1:2], want_recurrent))
    # unused slot 0 stays zero
    assert float(recurrent[0].abs().max()) == 0.0
    assert max(float(tap[0].abs().max()) for tap in taps) == 0.0


def test_decode_from_seeded_random_linear_state(layer_pairs):
    """Decode from an arbitrary (non-zero, non-prefill-derived) recurrent+conv state.

    Catches state-indexing/zero-init assumptions that a prefill-then-decode test cannot:
    the state here is random, so every term of the recurrence contributes.
    """
    pair = layer_pairs("linear", max_batch_size=2, supported_context=1024)
    pair.tt.reset_state()
    cfg = pair.cfg
    gen = torch.Generator().manual_seed(99)
    conv_hf = torch.randn(
        2,
        2 * cfg.linear_num_key_heads * cfg.linear_key_head_dim + cfg.linear_num_value_heads * cfg.linear_value_head_dim,
        cfg.conv_kernel,
        generator=gen,
    )
    recurrent = 0.1 * torch.randn(
        2, cfg.linear_num_value_heads, cfg.linear_key_head_dim, cfg.linear_value_head_dim, generator=gen
    )
    seed_tt_linear_state(pair, conv_hf, recurrent)

    tokens = torch.randn(2, 1, cfg.hidden_size, generator=gen)
    tt_x = to_tt_decode(pair.device, tokens)
    tt_pos = to_tt_positions(pair.device, torch.tensor([64, 64]))
    tt_out = pair.tt.decode_forward(tt_x, current_pos=tt_pos, page_table=None)
    got = from_tt(tt_out).reshape(2, 1, cfg.hidden_size)
    for t in (tt_x, tt_pos, tt_out):
        ttnn.deallocate(t)

    wants = []
    for i in range(2):
        cache = ref.make_cache(pair.hf_config)
        ref.seed_hf_linear_attention_state(cache, pair.layer_idx, conv_hf[i : i + 1, :, 1:], recurrent[i : i + 1])
        wants.append(
            ref.hf_decode(pair.hf, pair.hf_config, tokens[i : i + 1], positions=torch.tensor([64]), cache=cache)
        )
    _assert_pcc(compare("decode-seeded-state", got, torch.cat(wants, dim=0)))


def test_page_table_permutation_is_respected(layer_pairs):
    """Rewriting the page table to a different permutation changes what decode reads.

    Guards against silently treating the paged cache as contiguous.
    """
    pair = layer_pairs("full", max_batch_size=1, supported_context=1024)
    pair.tt.reset_state()
    seq = 512
    x = ref.synthetic_hidden_states(pair.hf_config, 1, seq, seed=81)
    tt_x = to_tt_prefill(pair.device, x)
    ttnn.deallocate(pair.tt.prefill_forward(tt_x, user_id=0, page_table=pair.page_table))
    ttnn.deallocate(tt_x)

    token = ref.synthetic_hidden_states(pair.hf_config, 1, 1, seed=82)
    tt_pos = to_tt_positions(pair.device, torch.tensor([seq]))
    tt_tok = to_tt_decode(pair.device, token)
    tt_out = pair.tt.decode_forward(tt_tok, current_pos=tt_pos, page_table=pair.page_table)
    baseline = from_tt(tt_out)
    ttnn.deallocate(tt_out)

    shuffled = pair.page_table_torch.clone()
    shuffled[0, :8] = shuffled[0, torch.tensor([3, 1, 0, 2, 6, 4, 7, 5])]
    bad_pt = to_tt_positions(pair.device, shuffled)
    tt_out2 = pair.tt.decode_forward(tt_tok, current_pos=tt_pos, page_table=bad_pt)
    perturbed = from_tt(tt_out2)
    for t in (tt_tok, tt_pos, tt_out2, bad_pt):
        ttnn.deallocate(t)

    assert float((baseline - perturbed).abs().max()) > 1e-3, (
        "decode output is unchanged after permuting the page table, so the page table is " "being ignored"
    )


# =======================================================================================
# traced decode
# =======================================================================================
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.timeout(1800)
def test_traced_decode_pcc(layer_pairs, kind):
    """PCC measured from **trace replay** output, not from an uncaptured forward.

    The layer is prefilled, the state is snapshotted, the trace is captured (which runs a
    warmup forward and so perturbs the state), the state is rewound, and only then is the
    replay compared against HF. Two different positions are replayed through the same
    captured trace to prove ``current_pos`` is read from device memory.
    """
    pair = layer_pairs(kind, max_batch_size=8, supported_context=2048)
    pair.tt.reset_state()
    cfg = pair.cfg
    batch = cfg.max_batch_size
    prefill_len = 256

    hf_caches = []
    for user_id in range(batch):
        x = ref.synthetic_hidden_states(pair.hf_config, 1, prefill_len, seed=1200 + user_id)
        tt_x = to_tt_prefill(pair.device, x)
        ttnn.deallocate(pair.tt.prefill_forward(tt_x, user_id=user_id, page_table=pair.page_table))
        ttnn.deallocate(tt_x)
        hf_caches.append(ref.hf_prefill(pair.hf, pair.hf_config, x).cache)

    snap = snapshot_state(pair)
    traced = TracedDecode(pair)
    try:
        # TracedDecode's warmup + capture ran the forward, which advanced the persistent
        # state; rewind so the replayed steps line up with the HF caches.
        restore_state(pair, snap)
        for step in range(2):
            pos = prefill_len + step
            tokens = torch.stack(
                [
                    ref.synthetic_hidden_states(pair.hf_config, 1, 1, seed=1300 + 7 * step + i).reshape(-1)
                    for i in range(batch)
                ]
            ).reshape(batch, 1, cfg.hidden_size)
            got = traced.run(tokens, torch.full((batch,), pos, dtype=torch.int32))
            wants = [
                ref.hf_decode(
                    pair.hf,
                    pair.hf_config,
                    tokens[i : i + 1],
                    positions=torch.tensor([pos]),
                    cache=hf_caches[i],
                )
                for i in range(batch)
            ]
            _assert_pcc(compare(f"traced-decode[{kind}] pos={pos} batch={batch}", got, torch.cat(wants, 0)))
    finally:
        traced.release()


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.timeout(1800)
def test_traced_decode_matches_eager(layer_pairs, kind):
    """Trace replay and eager decode produce bit-comparable output from the same state."""
    pair = layer_pairs(kind, max_batch_size=4, supported_context=1024)
    pair.tt.reset_state()
    cfg = pair.cfg
    batch = cfg.max_batch_size
    for user_id in range(batch):
        x = ref.synthetic_hidden_states(pair.hf_config, 1, 128, seed=1400 + user_id)
        tt_x = to_tt_prefill(pair.device, x)
        ttnn.deallocate(pair.tt.prefill_forward(tt_x, user_id=user_id, page_table=pair.page_table))
        ttnn.deallocate(tt_x)

    snap = snapshot_state(pair)
    tokens = torch.stack(
        [ref.synthetic_hidden_states(pair.hf_config, 1, 1, seed=1500 + i).reshape(-1) for i in range(batch)]
    ).reshape(batch, 1, cfg.hidden_size)
    positions = torch.full((batch,), 128, dtype=torch.int32)

    tt_x = to_tt_decode(pair.device, tokens)
    tt_pos = to_tt_positions(pair.device, positions)
    tt_out = pair.tt.decode_forward(tt_x, current_pos=tt_pos, page_table=pair.page_table)
    eager = from_tt(tt_out).reshape(batch, 1, cfg.hidden_size)
    for t in (tt_x, tt_pos, tt_out):
        ttnn.deallocate(t)

    restore_state(pair, snap)
    traced = TracedDecode(pair)
    try:
        restore_state(pair, snap)
        replayed = traced.run(tokens, positions)
    finally:
        traced.release()
    assert (
        float((eager - replayed).abs().max()) == 0.0
    ), f"trace replay differs from eager: maxabs {float((eager - replayed).abs().max()):.3e}"


# =======================================================================================
# determinism
# =======================================================================================
@pytest.mark.parametrize("kind", KINDS)
def test_prefill_determinism(layer_pairs, kind):
    """Identical prefill inputs from identical state give bit-identical outputs."""
    pair = layer_pairs(kind, max_batch_size=1, supported_context=2048)
    x = ref.synthetic_hidden_states(pair.hf_config, 1, 384, seed=21)
    outs = []
    for _ in range(3):
        pair.tt.reset_state()
        tt_x = to_tt_prefill(pair.device, x)
        tt_out = pair.tt.prefill_forward(tt_x, user_id=0, page_table=pair.page_table)
        outs.append(from_tt(tt_out))
        ttnn.deallocate(tt_x)
        ttnn.deallocate(tt_out)
    for i in (1, 2):
        assert float((outs[0] - outs[i]).abs().max()) == 0.0, f"prefill run {i} differs"


@pytest.mark.parametrize("kind", KINDS)
def test_decode_determinism(layer_pairs, kind):
    """Identical decode inputs replayed from the same state give bit-identical outputs."""
    pair = layer_pairs(kind, max_batch_size=4, supported_context=1024)
    pair.tt.reset_state()
    cfg = pair.cfg
    for user_id in range(cfg.max_batch_size):
        x = ref.synthetic_hidden_states(pair.hf_config, 1, 128, seed=1600 + user_id)
        tt_x = to_tt_prefill(pair.device, x)
        ttnn.deallocate(pair.tt.prefill_forward(tt_x, user_id=user_id, page_table=pair.page_table))
        ttnn.deallocate(tt_x)
    snap = snapshot_state(pair)

    tokens = torch.stack(
        [
            ref.synthetic_hidden_states(pair.hf_config, 1, 1, seed=1700 + i).reshape(-1)
            for i in range(cfg.max_batch_size)
        ]
    ).reshape(cfg.max_batch_size, 1, cfg.hidden_size)
    positions = torch.full((cfg.max_batch_size,), 128, dtype=torch.int32)

    outs = []
    for _ in range(3):
        restore_state(pair, snap)
        tt_x = to_tt_decode(pair.device, tokens)
        tt_pos = to_tt_positions(pair.device, positions)
        tt_out = pair.tt.decode_forward(tt_x, current_pos=tt_pos, page_table=pair.page_table)
        outs.append(from_tt(tt_out))
        for t in (tt_x, tt_pos, tt_out):
            ttnn.deallocate(t)
    for i in (1, 2):
        assert float((outs[0] - outs[i]).abs().max()) == 0.0, f"decode run {i} differs"


# =======================================================================================
# runtime fallback audit
# =======================================================================================
_FORBIDDEN = (
    r"\btorch\b",
    r"ttnn\.from_torch",
    r"ttnn\.to_torch",
    r"\.cpu\(",
    r"\.item\(\)",
    r"\.tolist\(",
    r"ttnn\.to_device",
    r"ttnn\.from_device",
    r"copy_host_to_device_tensor",
)

#: Runtime call graph of the two forward entry points. Every helper reachable from
#: prefill_forward / decode_forward is audited; setup-only helpers are not.
_RUNTIME_METHODS = [
    "prefill_forward",
    "_prefill_chunk",
    "_full_attention_prefill",
    "_norm_heads",
    "_partial_rope_prefill",
    "_load_linear_carry",
    "_store_linear_carry",
    "_linear_attention_prefill",
    "_valid_mask",
    "_masked",
    "_delta_gates",
    "_cast",
    "_l2_normalize",
    "_ut_transform",
    "_gated_delta_rule_prefill",
    "decode_forward",
    "_full_attention_decode",
    "_decode_norm_interleaved",
    "_decode_rope",
    "_partial_rope_decode",
    "_linear_attention_decode",
    "_router",
    "_shared_expert",
    "_experts",
    "_moe_prefill",
    "_moe_decode",
]


def _code_only(source: str) -> str:
    """Drop docstrings and comments so prose about torch does not trip the audit."""
    import io
    import tokenize

    kept = []
    prev = tokenize.INDENT
    for tok in tokenize.generate_tokens(io.StringIO(source).readline):
        if tok.type == tokenize.COMMENT:
            continue
        if tok.type == tokenize.STRING and prev in (tokenize.INDENT, tokenize.NEWLINE, tokenize.NL, tokenize.DEDENT):
            continue  # docstring position
        if tok.type not in (tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT):
            prev = tok.type
        else:
            prev = tok.type
        kept.append(tok.string)
    return "\n".join(kept)


#: Module-level helpers that only ever run at weight-load time. Everything else at module level
#: is on the runtime path and gets audited; ``_hifi_config`` builds the compute-kernel config in
#: ``from_state_dict``, and the other three are the torch->ttnn conversion itself.
_SETUP_HELPERS = frozenset({"_hifi_config", "_prepare_weights", "_build_rope_tables", "_to_device"})


def test_no_runtime_host_fallback():
    """No torch / host round-trip anywhere in a prefill or decode pass.

    Static audit of the whole runtime call graph (not just the two entry points), plus the
    module-level helpers they call. Setup-only code (``from_state_dict``,
    ``_prepare_weights``, ``_build_rope_tables``, ``_init_state``, ``_to_device``) is
    exempt — that *is* the weight-loading boundary.
    """
    offenders = []
    for name in _RUNTIME_METHODS:
        source = _code_only(inspect.getsource(getattr(fd.FunctionalDecoder, name)))
        for pattern in _FORBIDDEN:
            if re.search(pattern, source):
                offenders.append(f"FunctionalDecoder.{name}: {pattern}")
    # Every module-level callable except the setup-time converter is audited, derived from the
    # module rather than listed by hand: a hand-list silently misses new helpers (``_view`` was
    # missing from it), and the per-method self-check below only sees ``self._*`` calls.
    helpers = tuple(
        obj
        for name, obj in vars(fd).items()
        if inspect.isfunction(obj)
        and name.startswith("_")
        and obj.__module__ == fd.__name__
        and name not in _SETUP_HELPERS
    )
    assert {h.__name__ for h in helpers} >= {
        "_move",
        "_subview",
        "_owned_slice",
        "_view",
        "_dealloc",
        "_sparse_program_config",
        "_sdpa_program_config",
    }, sorted(h.__name__ for h in helpers)
    for helper in helpers:
        source = _code_only(inspect.getsource(helper))
        for pattern in _FORBIDDEN:
            if re.search(pattern, source):
                offenders.append(f"{helper.__name__}: {pattern}")
    assert not offenders, "host fallback in the runtime path:\n" + "\n".join(offenders)

    # the audited list really does cover everything reachable from the entry points
    entry_sources = "\n".join(inspect.getsource(getattr(fd.FunctionalDecoder, n)) for n in _RUNTIME_METHODS)
    called = set(re.findall(r"self\.(_[a-z_0-9]+)\(", entry_sources))
    missing = sorted(called - set(_RUNTIME_METHODS))
    assert not missing, f"runtime helpers not covered by the audit: {missing}"


@pytest.mark.parametrize("kind", KINDS)
def test_no_host_ops_during_forward(layer_pairs, kind):
    """Dynamic check: patch out the host bridges and run a real prefill + decode.

    Complements the static audit — catches a host round-trip hidden inside a ttnn python
    wrapper the regex cannot see.
    """
    pair = layer_pairs(kind, max_batch_size=2, supported_context=1024)
    pair.tt.reset_state()
    x = to_tt_prefill(pair.device, ref.synthetic_hidden_states(pair.hf_config, 1, 256, seed=31))
    tok = to_tt_decode(pair.device, ref.synthetic_hidden_states(pair.hf_config, 2, 1, seed=32).reshape(2, 1, -1))
    pos = to_tt_positions(pair.device, torch.tensor([256, 256]))

    tripped = []
    originals = {}
    for name in ("from_torch", "to_torch", "copy_host_to_device_tensor", "to_device", "from_device"):
        originals[name] = getattr(ttnn, name)

        def guard(*a, _n=name, **kw):
            tripped.append(_n)
            raise AssertionError(f"ttnn.{_n} called inside a measured forward pass")

        setattr(ttnn, name, guard)
    try:
        out = pair.tt.prefill_forward(x, user_id=0, page_table=pair.page_table)
        ttnn.deallocate(out)
        out = pair.tt.decode_forward(tok, current_pos=pos, page_table=pair.page_table)
        ttnn.deallocate(out)
    finally:
        for name, fn in originals.items():
            setattr(ttnn, name, fn)
    for t in (x, tok, pos):
        ttnn.deallocate(t)
    assert not tripped, tripped


# =======================================================================================
# real weights
# =======================================================================================
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.timeout(3600)
def test_real_weights_prefill_and_decode(layer_pairs, kind):
    """Real checkpoint weights, both layer kinds: prefill plus two eager decode steps.

    Eager, not traced, on purpose: trace capture is covered by ``test_traced_decode_pcc`` /
    ``test_traced_decode_matches_eager`` (which prove replay is bit-identical to eager), and
    keeping this case eager makes it a direct check of the real-weight *numerics*.
    """
    pair = layer_pairs(kind, max_batch_size=2, supported_context=2048, real_weights=True)
    assert pair.weights_source == "real"
    result = prefill_and_compare(pair, seq_len=1024, seed=55)
    _assert_pcc(result, log="pcc_real_weights")
    for decode_result in decode_and_compare(pair, prefill_len=512, steps=2, seed=56):
        _assert_pcc(decode_result, log="pcc_real_weights")


def test_weight_stats_match_real_checkpoint():
    """The recorded stats used to synthesise CI weights still match the checkpoint.

    Also pins the state-dict key/shape contract: ``from_state_dict`` must accept exactly the
    real HF layer keys.
    """
    for layer_idx in (ref.LINEAR_ATTENTION_LAYER_IDX, ref.FULL_ATTENTION_LAYER_IDX):
        recorded = ref.load_weight_stats(layer_idx)
        actual = ref.weight_stats(ref.real_layer_state_dict(layer_idx, dtype=torch.float32))
        assert set(recorded) == set(actual), set(recorded) ^ set(actual)
        for name, meta in actual.items():
            assert recorded[name]["shape"] == meta["shape"], name
            for field in ("mean", "std", "min", "max"):
                assert abs(recorded[name][field] - meta[field]) <= 1e-6 + 1e-4 * abs(meta[field]), (
                    name,
                    field,
                )


# =======================================================================================
# contract artifacts
# =======================================================================================
def test_context_contract_file_is_consistent():
    """``doc/context_contract.json`` agrees with the HF config *and* with the recorded evidence.

    The cross-check against ``long_context.jsonl`` is the anti-staleness guard: the contract
    cannot claim a prefill/decode length that no recorded run actually reached, and cannot claim
    both layer kinds unless both appear in the evidence.
    """
    path = ref.AUTOPORT_DIR / "doc" / "context_contract.json"
    assert path.exists(), f"{path} missing"
    contract = json.loads(path.read_text())
    hf = ref.load_hf_text_config()
    assert contract["hf_advertised_context"] == hf.max_position_embeddings == 262144
    assert (
        contract["supported_context"] == contract["hf_advertised_context"]
    ), "advertised capability must not be reduced without a hard physical-limit reason"
    assert contract["capability_reduction"] is None

    evidence_path = harness.ARTIFACT_DIR / "long_context.jsonl"
    assert evidence_path.exists(), f"{evidence_path} missing — run tests/test_long_context.py"
    rows = [json.loads(line) for line in evidence_path.read_text().splitlines() if line.strip()]
    prefill = {r["label"]: r for r in rows if r["label"].startswith("longest-prefill[")}
    decode = {r["label"]: r for r in rows if r["label"].startswith("longest-decode[")}
    for kind in ("linear", "full"):
        assert any(f"[{kind}]" in label for label in prefill), f"no longest-prefill evidence for {kind}"
        assert any(f"[{kind}]" in label for label in decode), f"no longest-decode evidence for {kind}"

    assert contract["largest_prefill_tested"] == max(r["seq_len"] for r in prefill.values())
    assert contract["largest_decode_context_tested"] == max(r["position"] for r in decode.values()) + 1
    for row in list(prefill.values()) + list(decode.values()):
        assert row["pcc"] >= PCC_BAR, row
