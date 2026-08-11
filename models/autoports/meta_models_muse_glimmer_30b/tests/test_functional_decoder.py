# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Correctness tests for the Muse-Glimmer-30B TTNN functional decoder layer.

Every test covers both decoder-layer kinds of ``text_config``
(``sliding_attention`` + RoPE and ``full_attention`` + NoPE) at the real config shapes
(hidden 6656, 32 Q heads / 2 KV heads, head_dim 128, SwiGLU 19968, window 2048).

Run the fast suite:

    pytest models/autoports/meta_models_muse_glimmer_30b/tests/test_functional_decoder.py \
        -m "not long_context"
"""

from __future__ import annotations

import math

import pytest
import torch

import ttnn
from models.autoports.meta_models_muse_glimmer_30b.reference import hf_reference as R
from models.autoports.meta_models_muse_glimmer_30b.tests import decoder_test_utils as U
from models.autoports.meta_models_muse_glimmer_30b.tt.functional_decoder import MuseGlimmerDecoderConfig, _round_up

PCC_BAR = 0.995
KIND_IDS = ["sliding_rope", "full_nope"]


# --------------------------------------------------------------------------- helpers


def _blocks_for(total_tokens: int, block_size: int) -> int:
    return _round_up(total_tokens, block_size) // block_size


def _alloc_paged(decoder, mesh_device, *, batch, total_tokens, block_size, page_seed, pool_multiplier=3):
    """Allocate a paged cache from a *pool* larger than needed, with a shuffled page table.

    The shuffle plus the larger pool means logical block ``i`` almost never maps to
    physical block ``i``, so an address/indexing bug cannot pass by accident. The width comes
    from ``decoder.blocks_per_seq`` because the decode SDPA rounds its K extent up to the K
    chunk and reads the page table over the *rounded* extent.
    """
    blocks_per_seq = decoder.blocks_per_seq(total_tokens)
    total_blocks = max(blocks_per_seq * batch * pool_multiplier, blocks_per_seq * batch + 3)
    page_table = U.build_page_table(
        batch=batch, blocks_per_seq=blocks_per_seq, total_blocks=total_blocks, seed=page_seed
    )
    assert page_table.min() >= 0
    assert len(set(page_table.reshape(-1).tolist())) == page_table.numel(), "page table must not alias blocks"
    kv_cache = decoder.allocate_kv_cache(
        batch_size=batch, max_seq_len=blocks_per_seq * block_size, num_blocks=total_blocks
    )
    return kv_cache, page_table, U.page_table_to_device(page_table, mesh_device)


def _ref_prefill_batch(ref, hidden, *, cache_len, backend="eager"):
    """Reference prefill plus zero-padded K/V buffers long enough for later decode steps."""
    ref_out, ref_k, ref_v = ref.prefill(hidden, backend=backend)
    batch, kv_heads, seq, head_dim = ref_k.shape
    k_full = torch.zeros(batch, kv_heads, cache_len, head_dim, dtype=ref_k.dtype)
    v_full = torch.zeros_like(k_full)
    k_full[:, :, :seq] = ref_k
    v_full[:, :, :seq] = ref_v
    return ref_out, k_full, v_full


# ------------------------------------------------------------------- host-only tests


@pytest.mark.parametrize("kind_id", KIND_IDS)
@pytest.mark.parametrize("seq_len", [512, 3000])
def test_reference_matches_hf(kind_id, seq_len, kinds, text_config, synthetic_state_dicts, build_reference):
    """The chunked host reference is numerically the untouched HF layer forward.

    Everything downstream compares TTNN against the chunked reference, so this pins the
    chunked/streaming re-drive (and the hand-built causal / sliding-window masks) to
    ``MuseGlimmerTextDecoderLayer.forward`` with HF's own mask builders. ``seq_len=3000``
    is past the 2048 sliding window, so the window boundary is exercised.
    """
    kind = kinds[kind_id]
    ref = build_reference(kind.layer_idx, synthetic_state_dicts[kind.layer_idx])
    hidden = R.unit_rms_hidden_states((1, seq_len, text_config.hidden_size), seed=1)

    hf_out = ref.forward_hf(hidden)
    eager_out, _, _ = ref.prefill(hidden, backend="eager", q_chunk=512, keep_kv=False)
    sdpa_out, _, _ = ref.prefill(hidden, backend="sdpa", q_chunk=512, keep_kv=False)

    assert U.pcc(hf_out, eager_out) > 1 - 1e-12
    assert U.pcc(hf_out, sdpa_out) > 1 - 1e-10
    scale = hf_out.abs().max()
    assert (hf_out - eager_out).abs().max() < 1e-4 * scale
    assert (hf_out - sdpa_out).abs().max() < 1e-4 * scale


@pytest.mark.parametrize(
    "length,cap,expected",
    [
        (8192, 256, 256),  # the measured-fast configuration is kept when it divides
        (131072, 256, 256),
        (2080, 512, 32),  # 2080 = 4*512 + 32: a non-dividing q_chunk HANGS the SDPA op
        (3008, 256, 64),
        (128, 256, 128),  # short chunk: falls back to a divisor, never a padded 256
        (10240, 128, 128),
        (10240, 256, 256),
    ],
)
def test_sdpa_chunk_divides_length(length, cap, expected):
    """SDPA chunk sizes must divide the sequence length.

    Regression for the hang in bug 7 of ``work_log.md``:
    ``scaled_dot_product_attention`` with a ``q_chunk_size`` that does not divide the Q
    length does not error, it hangs the device.
    """
    from models.autoports.meta_models_muse_glimmer_30b.tt.functional_decoder import FunctionalDecoder

    chunk = FunctionalDecoder._sdpa_chunk_for_length(length, cap)
    assert chunk == expected
    assert length % chunk == 0
    assert chunk <= cap


@pytest.mark.parametrize(
    "chunk_len,chunk_start,kv_length,expected",
    [
        (8192, 0, 131072, (256, 256)),  # both call sites must use the swept-optimal geometry
        (8192, 8192, 131072, (256, 256)),
        # k=256/128 would round the K extent up past the 12352-token page table, so the
        # capacity rule (see test_chunked_sdpa_k_chunk_stays_inside_the_page_table) picks 64.
        (4160, 8192, 12352, (64, 64)),
        (32, 0, 64, (32, 64)),
    ],
)
def test_chunked_sdpa_chunk_sizes_respect_the_configured_cap(
    chunk_len, chunk_start, kv_length, expected, kinds, text_config
):
    """The chunked (full-attention) prefill SDPA must honour ``prefill_sdpa_q_chunk``.

    Regression for a review finding: an uncapped ``max(candidates)`` silently ran the
    full-attention layers at q_chunk 512 — the size the Blackhole sweep measured as *slower*
    than 256 — while the sliding layers used the configured 256, so the recorded geometry and
    the executed geometry disagreed.
    """
    from models.autoports.meta_models_muse_glimmer_30b.tt.functional_decoder import (
        PREFILL_SDPA_K_CHUNK,
        PREFILL_SDPA_Q_CHUNK,
        FunctionalDecoder,
    )

    decoder = FunctionalDecoder.__new__(FunctionalDecoder)
    decoder.prefill_sdpa_q_chunk = PREFILL_SDPA_Q_CHUNK
    decoder.prefill_sdpa_k_chunk = PREFILL_SDPA_K_CHUNK
    q_chunk, k_chunk = decoder._chunked_sdpa_chunk_sizes(chunk_len, chunk_start, kv_length)
    assert (q_chunk, k_chunk) == expected
    assert q_chunk <= PREFILL_SDPA_Q_CHUNK and k_chunk <= PREFILL_SDPA_K_CHUNK
    assert chunk_len % q_chunk == 0 and chunk_start % q_chunk == 0 and chunk_start % k_chunk == 0


@pytest.mark.parametrize("block_size", [32, 64, 128])
@pytest.mark.parametrize("seq_len", [3000, 8256, 12345, 16448, 131072])
def test_chunked_sdpa_k_chunk_stays_inside_the_page_table(seq_len, block_size, text_config):
    """The k-chunk-padded K extent must fit the page table, for every chunk of every length.

    Regression for the bug `$autofix` found (work_log section 9):
    ``chunked_scaled_dot_product_attention`` validates only
    ``kv_length >= q_len + chunk_start_idx``, but its program factory rounds the K extent up
    to ``k_chunk_size`` and the reader consumes one page-table entry per ``block_size`` of
    that *rounded* extent with no bound check
    (``sdpa/device/kernels/dataflow/dataflow_common.hpp``). Overrunning reads the page-table
    stick's 32-byte alignment padding as block ids and then reads K/V from outside the cache
    buffer — measured as layer PCC 0.7345 instead of 0.9998, and separately as a device hang.

    This walks the real chunking for a range of lengths and page block sizes and asserts the
    selected ``k_chunk`` never overruns.
    """
    from models.autoports.meta_models_muse_glimmer_30b.tt.functional_decoder import (
        PREFILL_CHUNK_SIZE,
        PREFILL_SDPA_K_CHUNK,
        PREFILL_SDPA_Q_CHUNK,
        FunctionalDecoder,
        _round_up,
    )

    decoder = FunctionalDecoder.__new__(FunctionalDecoder)
    decoder.prefill_sdpa_q_chunk = PREFILL_SDPA_Q_CHUNK
    decoder.prefill_sdpa_k_chunk = PREFILL_SDPA_K_CHUNK

    padded_seq = _round_up(seq_len, 32)
    # What a caller allocates: enough blocks for the prompt plus one decode step.
    blocks_per_seq = _blocks_for(seq_len + 1, block_size)
    kv_length = blocks_per_seq * block_size

    chunk_start = 0
    while chunk_start < padded_seq:
        chunk_len = min(PREFILL_CHUNK_SIZE, padded_seq - chunk_start)
        q_chunk, k_chunk = decoder._chunked_sdpa_chunk_sizes(chunk_len, chunk_start, kv_length)
        padded_k_extent = _round_up(chunk_start + chunk_len, k_chunk)
        assert padded_k_extent <= kv_length, (
            f"seq_len={seq_len} block_size={block_size} chunk[{chunk_start}, "
            f"{chunk_start + chunk_len}): k_chunk={k_chunk} rounds the K extent up to "
            f"{padded_k_extent}, past the {kv_length}-token page table"
        )
        assert chunk_start % q_chunk == 0 and chunk_start % k_chunk == 0
        assert chunk_len % q_chunk == 0
        chunk_start += chunk_len


@pytest.mark.parametrize("block_size", [32, 64, 128])
@pytest.mark.parametrize("seq_len", [777, 1000, 3000, 8256, 131072])
def test_decode_page_table_capacity_covers_the_k_chunk_rounding(seq_len, block_size, expect_error):
    """The decode SDPA's k-chunk rounding must not walk past the page table either.

    ``paged_scaled_dot_product_attention_decode`` rounds its K extent up to ``k_chunk_size``
    (``rt_args_common.hpp``: ``valid_seq_len = nearest_n(cur_pos + 1, k_chunk_size)``) and the
    reader resolves the rounded extent through the unbounded
    ``page_table_ptr[virtual_block]``, exactly like the chunked prefill op. ``cur_pos`` is a
    device tensor, so the guard has to hold for *every* position the page table can address —
    which is what ``blocks_per_seq`` rounding up to a whole number of K chunks buys.
    """
    from models.autoports.meta_models_muse_glimmer_30b.tt.functional_decoder import (
        DECODE_SDPA_K_CHUNK,
        FunctionalDecoder,
        _round_up,
    )

    decoder = FunctionalDecoder.__new__(FunctionalDecoder)
    decoder.decode_sdpa_k_chunk = DECODE_SDPA_K_CHUNK
    decoder.block_size = block_size

    blocks = decoder.blocks_per_seq(seq_len + 1)
    capacity = blocks * block_size
    assert capacity >= seq_len + 1
    # Every addressable position, not just the ones the tests happen to decode at.
    for position in (0, 1, seq_len - 1, seq_len, capacity - 1):
        assert _round_up(position + 1, DECODE_SDPA_K_CHUNK) <= capacity, (
            f"seq_len={seq_len} block_size={block_size} capacity={capacity}: decoding at "
            f"position {position} rounds the K extent to "
            f"{_round_up(position + 1, DECODE_SDPA_K_CHUNK)}, past the page table"
        )
    decoder._check_decode_page_table_capacity(capacity)

    # An under-sized page table must be refused, not silently read out of bounds.
    unaligned = _blocks_for(seq_len + 1, block_size) * block_size
    if unaligned % DECODE_SDPA_K_CHUNK:
        with expect_error(ValueError, "whole number of decode K chunks"):
            decoder._check_decode_page_table_capacity(unaligned)


@pytest.mark.parametrize("batch,expected_cores", [(1, 32), (16, 32), (17, 110), (32, 110)])
def test_decode_sdpa_grid_covers_kv_heads(batch, expected_cores, kinds, text_config, expect_error):
    """The decode SDPA grid must give every (user, KV head) pair its own core.

    Regression for bug 8 of ``work_log.md``: a grid with
    ``cores < batch * num_key_value_heads`` passes the op's own validation and silently
    returns wrong results (batch 32 on 8x4 measured PCC 0.7176). Exercised without a device
    by driving the grid selection directly.
    """
    from models.autoports.meta_models_muse_glimmer_30b.tt.functional_decoder import (
        FunctionalDecoder,
        MuseGlimmerDecoderConfig,
    )

    decoder = FunctionalDecoder.__new__(FunctionalDecoder)
    decoder.config = MuseGlimmerDecoderConfig.from_hf_config(text_config, kinds["sliding_rope"].layer_idx)
    decoder.device_grid = (11, 10)
    decoder.decode_sdpa_core_grid = (8, 4)
    decoder.decode_sdpa_k_chunk = 256
    decoder.sdpa_core_grid = (11, 10)

    program_config = decoder._decode_sdpa_program_config(batch)
    grid = program_config.compute_with_storage_grid_size
    cores = grid.x * grid.y
    assert cores == expected_cores
    assert cores >= batch * text_config.num_key_value_heads

    # Beyond the grid's capacity the layer must fail loudly rather than mis-compute.
    with expect_error(ValueError, "needs"):
        decoder._decode_sdpa_program_config(56)


@pytest.mark.parametrize("kind_id", KIND_IDS)
def test_reference_decode_matches_hf(kind_id, kinds, text_config, synthetic_state_dicts, build_reference):
    """The host reference's *decode* step is the untouched HF layer with HF's own cache.

    ``test_reference_matches_hf`` pins the prefill re-drive; this pins the decode one, which is
    the golden for every decode PCC in the suite. HF drives the step through a real
    ``DynamicCache`` and its own mask builder, so cache handling and the single-query mask are
    both compared, not just the arithmetic.
    """
    kind = kinds[kind_id]
    ref = build_reference(kind.layer_idx, synthetic_state_dicts[kind.layer_idx])
    prompt_len = 128  # below the 2048 window so HF's own cache does not truncate
    hidden = R.unit_rms_hidden_states((1, prompt_len, text_config.hidden_size), seed=61)
    new_token = R.unit_rms_hidden_states((1, 1, text_config.hidden_size), seed=62)

    hf_out = ref.forward_hf_decode_step(hidden, new_token)

    _, ref_k, ref_v = ref.prefill(hidden, backend="eager")
    k_full = torch.zeros(1, text_config.num_key_value_heads, prompt_len + 1, text_config.head_dim)
    v_full = torch.zeros_like(k_full)
    k_full[:, :, :prompt_len] = ref_k
    v_full[:, :, :prompt_len] = ref_v
    ours = ref.decode(new_token, k_full, v_full, torch.tensor([prompt_len]))

    assert U.pcc(hf_out, ours) > 1 - 1e-10
    assert (hf_out - ours).abs().max() < 1e-4 * hf_out.abs().max()


def test_streaming_pcc_matches_comp_pcc():
    """``StreamingPCC`` (used for long-context evidence) matches the repo PCC helper."""
    from models.common.utility_functions import comp_pcc

    torch.manual_seed(0)
    golden = torch.randn(4, 997, 33)
    calculated = golden + 0.01 * torch.randn_like(golden)

    _, reference_pcc = comp_pcc(golden, calculated, pcc=0.0)
    streaming = R.StreamingPCC()
    for start in range(0, golden.shape[1], 101):
        streaming.update(golden[:, start : start + 101], calculated[:, start : start + 101])
    assert abs(streaming.pcc - reference_pcc) < 1e-9
    assert abs(U.pcc(golden, calculated) - reference_pcc) < 1e-9


@pytest.mark.parametrize("kind_id", KIND_IDS)
def test_layer_config_matches_hf(kind_id, kinds, text_config):
    """``MuseGlimmerDecoderConfig`` reads the HF contract, including the NoPE marker."""
    kind = kinds[kind_id]
    cfg = MuseGlimmerDecoderConfig.from_hf_config(text_config, kind.layer_idx)
    assert cfg.layer_type == kind.layer_type
    assert cfg.is_sliding == (kind.layer_type == "sliding_attention")
    assert cfg.sliding_window == kind.sliding_window
    assert cfg.uses_rope == (kind.rope_theta is not None)
    assert cfg.rope_theta == kind.rope_theta
    assert cfg.max_position_embeddings == text_config.max_position_embeddings == 131072
    assert cfg.qk_scale_factor == text_config.qk_scale_factor
    assert cfg.post_norm_eps == text_config.post_norm_eps
    assert math.isclose(cfg.sdpa_scale, text_config.head_dim**-0.5)
    # The model interleaves [sliding, sliding, sliding, full]; both kinds must exist.
    assert set(kinds) == {"sliding_rope", "full_nope"}


# ------------------------------------------------------------------- device tests


@pytest.mark.parametrize("kind_id", KIND_IDS)
@pytest.mark.parametrize(
    "seq_len",
    [
        32,  # smallest tile-aligned smoke length
        100,  # non-aligned, below one tile row of pages
        1000,  # non-aligned, below the sliding window
        2048,  # exactly the sliding-window boundary
        2080,  # just across the sliding-window boundary
        3000,  # non-aligned, past the window
        8192,  # exactly the prefill chunk boundary
        8256,  # just across the prefill chunk boundary
        12345,  # long, divisible by neither tile, page, window nor chunk
    ],
)
def test_paged_prefill_decode_pcc(
    kind_id,
    seq_len,
    kinds,
    text_config,
    synthetic_state_dicts,
    build_reference,
    build_cached_decoder,
    mg_mesh_device,
    record_pcc,
):
    """Paged prefill + paged decode PCC across tile/page/window/chunk boundaries.

    Also checks the paged K/V cache contents (un-paged through the shuffled page table)
    and that decode reads the cache the prefill wrote.
    """
    kind = kinds[kind_id]
    state_dict = synthetic_state_dicts[kind.layer_idx]
    ref = build_reference(kind.layer_idx, state_dict)
    block_size = 64
    decoder = build_cached_decoder(kind.layer_idx, state_dict, block_size=block_size)

    decode_steps = 2
    hidden = R.unit_rms_hidden_states((1, seq_len, text_config.hidden_size), seed=13)
    ref_out, ref_k, ref_v = _ref_prefill_batch(ref, hidden, cache_len=seq_len + decode_steps)

    kv_cache, page_table, page_table_tt = _alloc_paged(
        decoder, mg_mesh_device, batch=1, total_tokens=seq_len + decode_steps, block_size=block_size, page_seed=101
    )
    hidden_tt = U.prefill_input(hidden, mg_mesh_device)
    out_tt = decoder.prefill_forward(hidden_tt, kv_cache=kv_cache, page_table=page_table_tt)
    out = U.prefill_output(out_tt)
    assert out.shape == (1, seq_len, text_config.hidden_size)
    prefill_pcc = U.pcc(ref_out, out)
    record_pcc("prefill", prefill_pcc, seq_len=seq_len, kind=kind_id)
    assert prefill_pcc >= PCC_BAR, f"prefill PCC {prefill_pcc}"

    k_dev = U.read_paged_cache(kv_cache[0], page_table, block_size=block_size, seq_len=seq_len)
    v_dev = U.read_paged_cache(kv_cache[1], page_table, block_size=block_size, seq_len=seq_len)
    k_pcc = U.pcc(ref_k[:, :, :seq_len], k_dev)
    v_pcc = U.pcc(ref_v[:, :, :seq_len], v_dev)
    record_pcc("prefill_k_cache", k_pcc, seq_len=seq_len, kind=kind_id)
    record_pcc("prefill_v_cache", v_pcc, seq_len=seq_len, kind=kind_id)
    assert k_pcc >= PCC_BAR and v_pcc >= PCC_BAR, f"cache PCC k={k_pcc} v={v_pcc}"

    for step in range(decode_steps):
        position = seq_len + step
        hidden_d = R.unit_rms_hidden_states((1, 1, text_config.hidden_size), seed=200 + step)
        ref_d = ref.decode(hidden_d, ref_k, ref_v, torch.tensor([position]))
        current_pos, rope_idxs = U.position_tensors([position], mg_mesh_device)
        out_d = decoder.decode_forward(
            U.decode_input(hidden_d, mg_mesh_device),
            kv_cache=kv_cache,
            page_table=page_table_tt,
            current_pos=current_pos,
            rope_idxs=rope_idxs,
        )
        decode_pcc = U.pcc(ref_d, U.decode_output(out_d))
        record_pcc(f"decode_step{step}", decode_pcc, seq_len=seq_len, position=position, kind=kind_id)
        assert decode_pcc >= PCC_BAR, f"decode PCC {decode_pcc} at position {position}"


@pytest.mark.parametrize("kind_id", KIND_IDS)
@pytest.mark.parametrize("block_size", [32, 64, 128])
def test_page_block_sizes(
    kind_id,
    block_size,
    kinds,
    text_config,
    synthetic_state_dicts,
    build_reference,
    build_cached_decoder,
    mg_mesh_device,
    record_pcc,
):
    """Page-table handling is independent of the block size (32/64/128 tokens per page)."""
    kind = kinds[kind_id]
    state_dict = synthetic_state_dicts[kind.layer_idx]
    ref = build_reference(kind.layer_idx, state_dict)
    decoder = build_cached_decoder(kind.layer_idx, state_dict, block_size=block_size)

    seq_len = 777
    hidden = R.unit_rms_hidden_states((1, seq_len, text_config.hidden_size), seed=17)
    ref_out, ref_k, ref_v = _ref_prefill_batch(ref, hidden, cache_len=seq_len + 1)
    kv_cache, page_table, page_table_tt = _alloc_paged(
        decoder, mg_mesh_device, batch=1, total_tokens=seq_len + 1, block_size=block_size, page_seed=block_size
    )
    out = U.prefill_output(
        decoder.prefill_forward(U.prefill_input(hidden, mg_mesh_device), kv_cache=kv_cache, page_table=page_table_tt)
    )
    prefill_pcc = U.pcc(ref_out, out)
    record_pcc("prefill", prefill_pcc, block_size=block_size, kind=kind_id)
    assert prefill_pcc >= PCC_BAR

    hidden_d = R.unit_rms_hidden_states((1, 1, text_config.hidden_size), seed=18)
    ref_d = ref.decode(hidden_d, ref_k, ref_v, torch.tensor([seq_len]))
    current_pos, rope_idxs = U.position_tensors([seq_len], mg_mesh_device)
    out_d = U.decode_output(
        decoder.decode_forward(
            U.decode_input(hidden_d, mg_mesh_device),
            kv_cache=kv_cache,
            page_table=page_table_tt,
            current_pos=current_pos,
            rope_idxs=rope_idxs,
        )
    )
    decode_pcc = U.pcc(ref_d, out_d)
    record_pcc("decode", decode_pcc, block_size=block_size, kind=kind_id)
    assert decode_pcc >= PCC_BAR


@pytest.mark.parametrize("kind_id", KIND_IDS)
@pytest.mark.parametrize("batch", [4, 32])
def test_batched_prefill_and_decode(
    kind_id,
    batch,
    kinds,
    text_config,
    synthetic_state_dicts,
    build_reference,
    build_cached_decoder,
    mg_mesh_device,
    record_pcc,
):
    """Batched paged prefill (one ``paged_fill_cache`` per user slot) and batched decode."""
    kind = kinds[kind_id]
    state_dict = synthetic_state_dicts[kind.layer_idx]
    ref = build_reference(kind.layer_idx, state_dict)
    block_size = 64
    decoder = build_cached_decoder(kind.layer_idx, state_dict, block_size=block_size)

    seq_len = 300
    hidden = R.unit_rms_hidden_states((batch, seq_len, text_config.hidden_size), seed=23)
    ref_out, ref_k, ref_v = _ref_prefill_batch(ref, hidden, cache_len=seq_len + 1)

    kv_cache, page_table, page_table_tt = _alloc_paged(
        decoder, mg_mesh_device, batch=batch, total_tokens=seq_len + 1, block_size=block_size, page_seed=7 + batch
    )
    out = U.prefill_output(
        decoder.prefill_forward(
            U.prefill_input(hidden, mg_mesh_device),
            kv_cache=kv_cache,
            page_table=page_table_tt,
            user_ids=list(range(batch)),
        )
    )
    prefill_pcc = U.pcc(ref_out, out)
    record_pcc("prefill", prefill_pcc, batch=batch, kind=kind_id)
    assert prefill_pcc >= PCC_BAR
    for b in range(batch):
        per_user = U.pcc(ref_out[b], out[b])
        assert per_user >= PCC_BAR, f"user {b} prefill PCC {per_user}"

    hidden_d = R.unit_rms_hidden_states((batch, 1, text_config.hidden_size), seed=24)
    positions = torch.full((batch,), seq_len, dtype=torch.long)
    ref_d = ref.decode(hidden_d, ref_k, ref_v, positions)
    current_pos, rope_idxs = U.position_tensors(positions.tolist(), mg_mesh_device)
    out_d = U.decode_output(
        decoder.decode_forward(
            U.decode_input(hidden_d, mg_mesh_device),
            kv_cache=kv_cache,
            page_table=page_table_tt,
            current_pos=current_pos,
            rope_idxs=rope_idxs,
        )
    )
    decode_pcc = U.pcc(ref_d, out_d)
    record_pcc("decode", decode_pcc, batch=batch, kind=kind_id)
    assert decode_pcc >= PCC_BAR
    for b in range(batch):
        per_user = U.pcc(ref_d[b], out_d[b])
        assert per_user >= PCC_BAR, f"user {b} decode PCC {per_user}"


@pytest.mark.parametrize("kind_id", KIND_IDS)
def test_ragged_slots_and_current_positions(
    kind_id,
    kinds,
    text_config,
    synthetic_state_dicts,
    build_reference,
    build_cached_decoder,
    mg_mesh_device,
    record_pcc,
):
    """Per-user prefill into permuted cache slots, then batched decode at ragged positions.

    Each user gets a different (non-aligned) prompt length and is written to a *shuffled*
    cache slot, so a wrong ``batch_idx`` / page-table row or a wrong ``current_pos`` entry
    shows up as a per-user PCC failure. The sliding kind spans lengths on both sides of
    the 2048 window.
    """
    kind = kinds[kind_id]
    state_dict = synthetic_state_dicts[kind.layer_idx]
    ref = build_reference(kind.layer_idx, state_dict)
    block_size = 64
    decoder = build_cached_decoder(kind.layer_idx, state_dict, block_size=block_size)

    prompt_lens = [37, 1000, 2049, 3111]
    slots = [3, 0, 2, 1]  # user u prefills into cache slot slots[u]
    batch = len(prompt_lens)
    max_tokens = max(prompt_lens) + 1

    kv_cache, page_table, page_table_tt = _alloc_paged(
        decoder, mg_mesh_device, batch=batch, total_tokens=max_tokens, block_size=block_size, page_seed=55
    )

    ref_k = torch.zeros(batch, text_config.num_key_value_heads, max_tokens, text_config.head_dim)
    ref_v = torch.zeros_like(ref_k)
    ref_prefill_out = {}
    for user, (prompt_len, slot) in enumerate(zip(prompt_lens, slots)):
        hidden = R.unit_rms_hidden_states((1, prompt_len, text_config.hidden_size), seed=300 + user)
        out_ref, k_ref, v_ref = ref.prefill(hidden, backend="eager")
        ref_prefill_out[slot] = out_ref
        ref_k[slot, :, :prompt_len] = k_ref[0]
        ref_v[slot, :, :prompt_len] = v_ref[0]
        out = U.prefill_output(
            decoder.prefill_forward(
                U.prefill_input(hidden, mg_mesh_device),
                kv_cache=kv_cache,
                page_table=page_table_tt,
                user_ids=[slot],
            )
        )
        per_user = U.pcc(out_ref, out)
        record_pcc(f"prefill_user{user}_slot{slot}", per_user, seq_len=prompt_len, kind=kind_id)
        assert per_user >= PCC_BAR, f"user {user} (slot {slot}) prefill PCC {per_user}"

    # Batched decode: slot s continues the prompt that was written into slot s.
    slot_lens = {slot: prompt_len for prompt_len, slot in zip(prompt_lens, slots)}
    positions = torch.tensor([slot_lens[s] for s in range(batch)], dtype=torch.long)
    hidden_d = torch.cat(
        [R.unit_rms_hidden_states((1, 1, text_config.hidden_size), seed=400 + s) for s in range(batch)], dim=0
    )
    ref_d = ref.decode(hidden_d, ref_k, ref_v, positions)
    current_pos, rope_idxs = U.position_tensors(positions.tolist(), mg_mesh_device)
    out_d = U.decode_output(
        decoder.decode_forward(
            U.decode_input(hidden_d, mg_mesh_device),
            kv_cache=kv_cache,
            page_table=page_table_tt,
            current_pos=current_pos,
            rope_idxs=rope_idxs,
        )
    )
    for slot in range(batch):
        per_user = U.pcc(ref_d[slot], out_d[slot])
        record_pcc(f"decode_slot{slot}", per_user, position=int(positions[slot]), kind=kind_id)
        assert per_user >= PCC_BAR, f"slot {slot} decode PCC {per_user} at position {int(positions[slot])}"


@pytest.mark.parametrize("kind_id", KIND_IDS)
def test_short_prefill_lengths(
    kind_id,
    kinds,
    text_config,
    synthetic_state_dicts,
    build_reference,
    build_cached_decoder,
    mg_mesh_device,
    record_pcc,
):
    """Sub-tile prompt lengths (1, 7, 31) — the bottom of the advertised ``[1, 131072]`` range."""
    kind = kinds[kind_id]
    state_dict = synthetic_state_dicts[kind.layer_idx]
    ref = build_reference(kind.layer_idx, state_dict)
    block_size = 64
    decoder = build_cached_decoder(kind.layer_idx, state_dict, block_size=block_size)

    for seq_len in (1, 7, 31):
        hidden = R.unit_rms_hidden_states((1, seq_len, text_config.hidden_size), seed=800 + seq_len)
        ref_out, ref_k, ref_v = _ref_prefill_batch(ref, hidden, cache_len=seq_len + 1)
        kv_cache, page_table, page_table_tt = _alloc_paged(
            decoder,
            mg_mesh_device,
            batch=1,
            total_tokens=seq_len + 1,
            block_size=block_size,
            page_seed=800 + seq_len,
        )
        out = U.prefill_output(
            decoder.prefill_forward(
                U.prefill_input(hidden, mg_mesh_device), kv_cache=kv_cache, page_table=page_table_tt
            )
        )
        assert out.shape == (1, seq_len, text_config.hidden_size)
        prefill_pcc = U.pcc(ref_out, out)
        record_pcc(f"prefill_len{seq_len}", prefill_pcc, seq_len=seq_len, kind=kind_id)
        assert prefill_pcc >= PCC_BAR, f"prefill PCC {prefill_pcc} at seq_len {seq_len}"

        hidden_d = R.unit_rms_hidden_states((1, 1, text_config.hidden_size), seed=900 + seq_len)
        ref_d = ref.decode(hidden_d, ref_k, ref_v, torch.tensor([seq_len]))
        current_pos, rope_idxs = U.position_tensors([seq_len], mg_mesh_device)
        decode_pcc = U.pcc(
            ref_d,
            U.decode_output(
                decoder.decode_forward(
                    U.decode_input(hidden_d, mg_mesh_device),
                    kv_cache=kv_cache,
                    page_table=page_table_tt,
                    current_pos=current_pos,
                    rope_idxs=rope_idxs,
                )
            ),
        )
        record_pcc(f"decode_after_len{seq_len}", decode_pcc, position=seq_len, kind=kind_id)
        assert decode_pcc >= PCC_BAR, f"decode PCC {decode_pcc} after seq_len {seq_len}"
        for cache in kv_cache:
            cache.deallocate(True)


@pytest.mark.parametrize("kind_id", KIND_IDS)
def test_batched_multichunk_prefill_shared_pool(
    kind_id,
    kinds,
    text_config,
    synthetic_state_dicts,
    build_reference,
    build_cached_decoder,
    mg_mesh_device,
    record_pcc,
):
    """Batched prefill that crosses the internal chunk boundary, out of a shared slot pool.

    This is the combination the single-user and short-batched tests miss:

    * ``seq_len`` > ``prefill_chunk_size``, so the second chunk fills the cache at
      ``first_block > 0`` **and** the sliding overlap-trim path runs, both with batch > 1;
    * the page table has more rows than the batch and ``user_ids`` is not the identity, so
      ``_rows_page_table``'s multi-row gather + concat branch is exercised (the chunked SDPA
      needs exactly the batch's rows, in batch order);
    * a non-default page block size with batch > 1.
    """
    kind = kinds[kind_id]
    state_dict = synthetic_state_dicts[kind.layer_idx]
    ref = build_reference(kind.layer_idx, state_dict)
    block_size = 128
    decoder = build_cached_decoder(kind.layer_idx, state_dict, block_size=block_size)

    batch, seq_len = 2, 8256  # 8256 = 8192 + 64: two chunks, second one only 64 tokens long
    slots = [3, 1]  # non-identity rows out of a 4-row pool
    pool_rows = 4

    hidden = R.unit_rms_hidden_states((batch, seq_len, text_config.hidden_size), seed=1717)
    ref_out, ref_k, ref_v = _ref_prefill_batch(ref, hidden, cache_len=seq_len + 1, backend="sdpa")

    blocks_per_seq = decoder.blocks_per_seq(seq_len + 1)
    page_table = U.build_page_table(
        batch=pool_rows, blocks_per_seq=blocks_per_seq, total_blocks=blocks_per_seq * pool_rows + 5, seed=1717
    )
    kv_cache = decoder.allocate_kv_cache(
        batch_size=pool_rows,
        max_seq_len=blocks_per_seq * block_size,
        num_blocks=blocks_per_seq * pool_rows + 5,
    )
    page_table_tt = U.page_table_to_device(page_table, mg_mesh_device)
    assert page_table.shape[0] > batch, "the point of this test is a pool wider than the batch"

    out = U.prefill_output(
        decoder.prefill_forward(
            U.prefill_input(hidden, mg_mesh_device),
            kv_cache=kv_cache,
            page_table=page_table_tt,
            user_ids=slots,
        )
    )
    # A *tighter* bar than PCC_BAR on purpose: this configuration is how the
    # chunked-SDPA page-table overrun was found, and at 0.9971 it still cleared 0.995. Every
    # other prefill measurement in the suite sits at >=0.9997, so anything below 0.9995 here
    # means the overrun (or something like it) is back.
    multichunk_bar = 0.9995
    for index in range(batch):
        per_user = U.pcc(ref_out[index], out[index])
        record_pcc(
            f"prefill_batch{index}_slot{slots[index]}",
            per_user,
            seq_len=seq_len,
            kind=kind_id,
            block_size=block_size,
            threshold=multichunk_bar,
        )
        assert per_user >= multichunk_bar, (
            f"batch row {index} (slot {slots[index]}) prefill PCC {per_user} — below the "
            f"tightened {multichunk_bar} bar for this configuration"
        )

    # Each row must have landed in *its* slot: compare the un-paged cache slot-by-slot.
    k_dev = U.read_paged_cache(kv_cache[0], page_table, block_size=block_size, seq_len=seq_len)
    for index, slot in enumerate(slots):
        cache_pcc = U.pcc(ref_k[index, :, :seq_len], k_dev[slot])
        record_pcc(f"k_cache_slot{slot}", cache_pcc, seq_len=seq_len, kind=kind_id)
        assert cache_pcc >= PCC_BAR, f"slot {slot} K cache PCC {cache_pcc}"


@pytest.mark.parametrize("kind_id", KIND_IDS)
def test_continued_prefill_contract(
    kind_id,
    kinds,
    text_config,
    synthetic_state_dicts,
    build_reference,
    build_cached_decoder,
    mg_mesh_device,
    record_pcc,
    expect_error,
):
    """``start_pos > 0``: correct on full-attention layers, refused on sliding layers.

    A continued segment's first ``sliding_window`` positions need K/V that live in the paged
    cache, and the windowed prefill SDPA reads its prefix from the input tensor instead, so a
    sliding layer must refuse rather than return a truncated-window answer. Full-attention
    layers read the whole prefix from the cache via the chunked SDPA, so they are exact — and
    that is checked against a single-shot prefill of the same prompt.
    """
    kind = kinds[kind_id]
    state_dict = synthetic_state_dicts[kind.layer_idx]
    ref = build_reference(kind.layer_idx, state_dict)
    block_size = 64
    decoder = build_cached_decoder(kind.layer_idx, state_dict, block_size=block_size)

    first, second = 1024, 512
    total = first + second
    hidden = R.unit_rms_hidden_states((1, total, text_config.hidden_size), seed=1919)
    kv_cache, page_table, page_table_tt = _alloc_paged(
        decoder, mg_mesh_device, batch=1, total_tokens=total + 1, block_size=block_size, page_seed=1919
    )
    decoder.prefill_forward(
        U.prefill_input(hidden[:, :first], mg_mesh_device), kv_cache=kv_cache, page_table=page_table_tt
    ).deallocate(True)

    if kind.sliding_window:
        with expect_error(ValueError, "start_pos"):
            decoder.prefill_forward(
                U.prefill_input(hidden[:, first:], mg_mesh_device),
                kv_cache=kv_cache,
                page_table=page_table_tt,
                start_pos=first,
            )
        return

    ref_out, _, _ = ref.prefill(hidden, backend="eager", keep_kv=False)
    out = U.prefill_output(
        decoder.prefill_forward(
            U.prefill_input(hidden[:, first:], mg_mesh_device),
            kv_cache=kv_cache,
            page_table=page_table_tt,
            start_pos=first,
        )
    )
    continued_pcc = U.pcc(ref_out[:, first:], out)
    record_pcc("continued_prefill", continued_pcc, seq_len=second, start_pos=first, kind=kind_id)
    assert continued_pcc >= PCC_BAR, f"continued prefill PCC {continued_pcc}"


@pytest.mark.parametrize("kind_id", KIND_IDS)
def test_sliding_window_is_enforced(
    kind_id, kinds, text_config, synthetic_state_dicts, build_cached_decoder, mg_mesh_device
):
    """A sliding layer must ignore tokens outside its window; a full layer must not.

    Perturbing the cache at a position that is ``> sliding_window`` behind the decode
    position changes the full-attention output and leaves the sliding output untouched.
    This distinguishes a correct window from "window silently ignored".
    """
    kind = kinds[kind_id]
    state_dict = synthetic_state_dicts[kind.layer_idx]
    block_size = 64
    decoder = build_cached_decoder(kind.layer_idx, state_dict, block_size=block_size)

    seq_len = 3072  # > 2048 window, so early positions fall outside it
    position = seq_len
    hidden = R.unit_rms_hidden_states((1, seq_len, text_config.hidden_size), seed=31)
    hidden_far = hidden.clone()
    hidden_far[:, :512] += 4.0  # positions 0..511 are >2048 behind `position`

    outputs = []
    for prompt in (hidden, hidden_far):
        kv_cache, page_table, page_table_tt = _alloc_paged(
            decoder, mg_mesh_device, batch=1, total_tokens=seq_len + 1, block_size=block_size, page_seed=71
        )
        decoder.prefill_forward(U.prefill_input(prompt, mg_mesh_device), kv_cache=kv_cache, page_table=page_table_tt)
        hidden_d = R.unit_rms_hidden_states((1, 1, text_config.hidden_size), seed=32)
        current_pos, rope_idxs = U.position_tensors([position], mg_mesh_device)
        outputs.append(
            U.decode_output(
                decoder.decode_forward(
                    U.decode_input(hidden_d, mg_mesh_device),
                    kv_cache=kv_cache,
                    page_table=page_table_tt,
                    current_pos=current_pos,
                    rope_idxs=rope_idxs,
                )
            )
        )
        for cache in kv_cache:
            cache.deallocate(True)

    delta = (outputs[0] - outputs[1]).abs().max().item()
    scale = outputs[0].abs().max().item()
    if kind.sliding_window:
        assert delta <= 1e-3 * scale, f"sliding layer reacted to out-of-window tokens (delta {delta}, scale {scale})"
    else:
        assert delta > 1e-2 * scale, f"full-attention layer ignored distant tokens (delta {delta}, scale {scale})"


@pytest.mark.parametrize("kind_id", KIND_IDS)
def test_determinism_repeated_inputs(
    kind_id, kinds, text_config, synthetic_state_dicts, build_cached_decoder, mg_mesh_device
):
    """Identical inputs produce bit-identical prefill and decode outputs."""
    kind = kinds[kind_id]
    state_dict = synthetic_state_dicts[kind.layer_idx]
    block_size = 64
    decoder = build_cached_decoder(kind.layer_idx, state_dict, block_size=block_size)

    seq_len = 1000
    hidden = R.unit_rms_hidden_states((1, seq_len, text_config.hidden_size), seed=41)
    hidden_d = R.unit_rms_hidden_states((1, 1, text_config.hidden_size), seed=42)

    prefill_runs, decode_runs = [], []
    for repeat in range(3):
        kv_cache, page_table, page_table_tt = _alloc_paged(
            decoder, mg_mesh_device, batch=1, total_tokens=seq_len + 1, block_size=block_size, page_seed=91
        )
        prefill_runs.append(
            U.prefill_output(
                decoder.prefill_forward(
                    U.prefill_input(hidden, mg_mesh_device), kv_cache=kv_cache, page_table=page_table_tt
                )
            )
        )
        current_pos, rope_idxs = U.position_tensors([seq_len], mg_mesh_device)
        decode_runs.append(
            U.decode_output(
                decoder.decode_forward(
                    U.decode_input(hidden_d, mg_mesh_device),
                    kv_cache=kv_cache,
                    page_table=page_table_tt,
                    current_pos=current_pos,
                    rope_idxs=rope_idxs,
                )
            )
        )
        for cache in kv_cache:
            cache.deallocate(True)

    for repeat in range(1, 3):
        assert torch.equal(prefill_runs[0], prefill_runs[repeat]), "prefill is not deterministic"
        assert torch.equal(decode_runs[0], decode_runs[repeat]), "decode is not deterministic"


@pytest.mark.parametrize("kind_id", KIND_IDS)
def test_traced_decode_pcc(
    kind_id,
    kinds,
    text_config,
    synthetic_state_dicts,
    build_reference,
    build_cached_decoder,
    mg_mesh_device,
    record_pcc,
):
    """Decode runs under ``ttnn`` traced execution and PCC is measured from the replay.

    Trace inputs (hidden state, ``current_pos``, ``rope_idxs``) are pre-allocated device
    tensors; replay only copies new contents into them, so no tensor is allocated after
    capture and the cache/page-table buffer addresses stay baked into the trace.
    """
    kind = kinds[kind_id]
    state_dict = synthetic_state_dicts[kind.layer_idx]
    ref = build_reference(kind.layer_idx, state_dict)
    block_size = 64
    decoder = build_cached_decoder(kind.layer_idx, state_dict, block_size=block_size)

    seq_len = 1000
    steps = 3
    hidden = R.unit_rms_hidden_states((1, seq_len, text_config.hidden_size), seed=51)
    ref_out, ref_k, ref_v = _ref_prefill_batch(ref, hidden, cache_len=seq_len + steps)
    kv_cache, page_table, page_table_tt = _alloc_paged(
        decoder, mg_mesh_device, batch=1, total_tokens=seq_len + steps, block_size=block_size, page_seed=61
    )
    decoder.prefill_forward(U.prefill_input(hidden, mg_mesh_device), kv_cache=kv_cache, page_table=page_table_tt)

    hidden_steps = [R.unit_rms_hidden_states((1, 1, text_config.hidden_size), seed=500 + s) for s in range(steps)]
    positions = [seq_len + s for s in range(steps)]

    # Persistent trace inputs, written in place before every replay.
    x_dev = U.decode_input(hidden_steps[0], mg_mesh_device)
    pos_dev, rope_dev = U.position_tensors([positions[0]], mg_mesh_device)

    # Warmup (compiles kernels and fills the program cache) and capture both execute the
    # step, and a decode step is idempotent in the cache: it writes the same K/V to the
    # same slot for the same input and position. So the replayed step 0 still sees exactly
    # the post-prefill cache state plus its own (identical) write.
    decoder.decode_forward(x_dev, kv_cache=kv_cache, page_table=page_table_tt, current_pos=pos_dev, rope_idxs=rope_dev)
    ttnn.synchronize_device(mg_mesh_device)

    trace_id = ttnn.begin_trace_capture(mg_mesh_device, cq_id=0)
    out_dev = decoder.decode_forward(
        x_dev, kv_cache=kv_cache, page_table=page_table_tt, current_pos=pos_dev, rope_idxs=rope_dev
    )
    ttnn.end_trace_capture(mg_mesh_device, trace_id, cq_id=0)
    ttnn.synchronize_device(mg_mesh_device)

    for step in range(steps):
        host_x = U.host_decode_input(hidden_steps[step], mg_mesh_device)
        host_pos, host_rope = U.position_tensors([positions[step]], mg_mesh_device, device=False)
        ttnn.copy_host_to_device_tensor(host_x, x_dev)
        ttnn.copy_host_to_device_tensor(host_pos, pos_dev)
        ttnn.copy_host_to_device_tensor(host_rope, rope_dev)
        ttnn.execute_trace(mg_mesh_device, trace_id, cq_id=0, blocking=True)
        traced_out = U.decode_output(out_dev)

        ref_d = ref.decode(hidden_steps[step], ref_k, ref_v, torch.tensor([positions[step]]))
        traced_pcc = U.pcc(ref_d, traced_out)
        record_pcc(f"traced_decode_step{step}", traced_pcc, position=positions[step], kind=kind_id)
        assert traced_pcc >= PCC_BAR, f"traced decode PCC {traced_pcc} at position {positions[step]}"

    ttnn.release_trace(mg_mesh_device, trace_id)


@pytest.mark.parametrize("kind_id", KIND_IDS)
def test_no_runtime_host_fallback(
    kind_id, kinds, text_config, synthetic_state_dicts, build_cached_decoder, mg_mesh_device
):
    """A single prefill and decode pass must not touch torch or host<->device transfer.

    ``TorchFunctionMode`` catches *any* torch op, and the ttnn conversion entry points are
    replaced with tripwires, so a hidden host fallback in the layer or in a helper it
    calls fails the test instead of quietly costing latency.
    """
    kind = kinds[kind_id]
    state_dict = synthetic_state_dicts[kind.layer_idx]
    block_size = 64
    decoder = build_cached_decoder(kind.layer_idx, state_dict, block_size=block_size)

    seq_len = 1024
    hidden = R.unit_rms_hidden_states((1, seq_len, text_config.hidden_size), seed=71)
    kv_cache, page_table, page_table_tt = _alloc_paged(
        decoder, mg_mesh_device, batch=1, total_tokens=seq_len + 1, block_size=block_size, page_seed=81
    )
    hidden_tt = U.prefill_input(hidden, mg_mesh_device)
    hidden_d_tt = U.decode_input(R.unit_rms_hidden_states((1, 1, text_config.hidden_size), seed=72), mg_mesh_device)
    current_pos, rope_idxs = U.position_tensors([seq_len], mg_mesh_device)

    class NoTorchOps(torch.overrides.TorchFunctionMode):
        def __torch_function__(self, func, types, args=(), kwargs=None):
            raise AssertionError(f"torch op {getattr(func, '__name__', func)} called inside a measured pass")

    tripwires = {}
    for name in ("from_torch", "to_torch", "as_tensor"):
        tripwires[name] = getattr(ttnn, name)

        def _tripwire(*args, _name=name, **kwargs):
            raise AssertionError(f"ttnn.{_name} called inside a measured pass")

        setattr(ttnn, name, _tripwire)
    try:
        with NoTorchOps():
            out = decoder.prefill_forward(hidden_tt, kv_cache=kv_cache, page_table=page_table_tt)
            out_d = decoder.decode_forward(
                hidden_d_tt,
                kv_cache=kv_cache,
                page_table=page_table_tt,
                current_pos=current_pos,
                rope_idxs=rope_idxs,
            )
            ttnn.synchronize_device(mg_mesh_device)
    finally:
        for name, original in tripwires.items():
            setattr(ttnn, name, original)

    assert out.shape[-2] == seq_len
    assert out_d.shape[-2] == 1


@pytest.mark.real_weights
@pytest.mark.parametrize("kind_id", KIND_IDS)
def test_real_weights_prefill_decode(
    kind_id, kinds, text_config, build_reference, build_cached_decoder, mg_mesh_device, record_pcc
):
    """Real checkpoint weights and real activations, end to end.

    The layer input is the checkpoint's own embedding rows for real tokenizer output, run
    through the *real* preceding layers (``stacked_layer_input``), so both the weights and
    the activation distribution are the model's own — not synthetic.
    """
    kind = kinds[kind_id]
    state_dict = R.load_real_layer_state_dict(kind.layer_idx)
    assert set(state_dict) == set(R.LAYER_PARAM_NAMES)
    ref = build_reference(kind.layer_idx, state_dict, tag="real")
    block_size = 64
    decoder = build_cached_decoder(kind.layer_idx, state_dict, tag="real", block_size=block_size)

    seq_len = 512
    token_ids = R.real_token_ids(seq_len)
    hidden = R.stacked_layer_input(text_config, kind.layer_idx, token_ids)
    assert hidden.shape == (1, seq_len, text_config.hidden_size)
    ref_out, ref_k, ref_v = _ref_prefill_batch(ref, hidden, cache_len=seq_len + 1)

    kv_cache, page_table, page_table_tt = _alloc_paged(
        decoder, mg_mesh_device, batch=1, total_tokens=seq_len + 1, block_size=block_size, page_seed=909
    )
    out = U.prefill_output(
        decoder.prefill_forward(U.prefill_input(hidden, mg_mesh_device), kv_cache=kv_cache, page_table=page_table_tt)
    )
    prefill_pcc = U.pcc(ref_out, out)
    record_pcc("prefill_real_weights", prefill_pcc, seq_len=seq_len, kind=kind_id, weights="real")
    assert prefill_pcc >= PCC_BAR, f"real-weight prefill PCC {prefill_pcc}"

    k_dev = U.read_paged_cache(kv_cache[0], page_table, block_size=block_size, seq_len=seq_len)
    cache_pcc = U.pcc(ref_k[:, :, :seq_len], k_dev)
    record_pcc("prefill_k_cache_real_weights", cache_pcc, seq_len=seq_len, kind=kind_id, weights="real")
    assert cache_pcc >= PCC_BAR

    hidden_d = R.stacked_layer_input(text_config, kind.layer_idx, R.real_token_ids(1, offset=seq_len))
    ref_d = ref.decode(hidden_d, ref_k, ref_v, torch.tensor([seq_len]))
    current_pos, rope_idxs = U.position_tensors([seq_len], mg_mesh_device)
    out_d = U.decode_output(
        decoder.decode_forward(
            U.decode_input(hidden_d, mg_mesh_device),
            kv_cache=kv_cache,
            page_table=page_table_tt,
            current_pos=current_pos,
            rope_idxs=rope_idxs,
        )
    )
    decode_pcc = U.pcc(ref_d, out_d)
    record_pcc("decode_real_weights", decode_pcc, position=seq_len, kind=kind_id, weights="real")
    assert decode_pcc >= PCC_BAR, f"real-weight decode PCC {decode_pcc}"


@pytest.mark.long_context
@pytest.mark.timeout(0)  # pytest.ini caps tests at 300 s; the 131072-token host reference needs far longer
@pytest.mark.parametrize("kind_id", KIND_IDS)
def test_full_context_prefill_and_decode(
    kind_id,
    kinds,
    text_config,
    synthetic_state_dicts,
    build_reference,
    build_cached_decoder,
    mg_mesh_device,
    record_pcc,
):
    """The full advertised context: prefill 131072 and 131071, decode at position 131071.

    * prefill at exactly ``max_position_embeddings`` (131072) — the advertised maximum;
    * prefill at 131071 — the same maximum minus one, i.e. a logical length divisible by
      neither the tile height, the page size, the 8192 prefill chunk nor the window;
    * decode at position 131071, the last addressable position, on the cache that the
      131071-token prefill wrote.

    PCC is accumulated with :class:`StreamingPCC` over every reference query block, so the
    comparison covers the whole 131072-token output without materialising two 3.5 GB
    golden tensors at once. Because attention is causal, the reference output at position
    ``p`` is the same for a 131071- and a 131072-token prompt, and K/V at position ``p``
    depend only on the layer input at ``p``, so one reference pass serves all three checks.
    """
    kind = kinds[kind_id]
    state_dict = synthetic_state_dicts[kind.layer_idx]
    ref = build_reference(kind.layer_idx, state_dict)
    block_size = 64
    decoder = build_cached_decoder(kind.layer_idx, state_dict, block_size=block_size)

    max_context = text_config.max_position_embeddings
    hidden_size = text_config.hidden_size
    hidden = R.unit_rms_hidden_states((1, max_context, hidden_size), seed=97)

    results = {}
    for prefill_len in (max_context, max_context - 1):
        kv_cache, page_table, page_table_tt = _alloc_paged(
            decoder,
            mg_mesh_device,
            batch=1,
            total_tokens=max_context,
            block_size=block_size,
            page_seed=1000 + prefill_len % 7,
            pool_multiplier=1,
        )
        hidden_tt = U.prefill_input(hidden[:, :prefill_len], mg_mesh_device)
        out_dev = decoder.prefill_forward(hidden_tt, kv_cache=kv_cache, page_table=page_table_tt)
        assert out_dev.shape[-2] == prefill_len
        hidden_tt.deallocate(True)

        stats = R.StreamingPCC()
        blocks = []

        def on_chunk(start, end, out_chunk, _stats=stats, _blocks=blocks, _limit=prefill_len):
            if start >= _limit:
                return
            end = min(end, _limit)
            device_slice = ttnn.slice(out_dev, [0, 0, start, 0], [1, 1, end, hidden_size])
            _stats.update(out_chunk[:, : end - start], U.prefill_output(device_slice))
            device_slice.deallocate(True)
            _blocks.append((start, end))

        _, ref_k, ref_v = ref.prefill(hidden, on_chunk=on_chunk, backend="sdpa")
        assert blocks[0][0] == 0 and blocks[-1][1] == prefill_len, f"incomplete coverage: {blocks[0]}..{blocks[-1]}"
        assert sum(e - s for s, e in blocks) == prefill_len, "streaming PCC did not cover every position"
        pcc_value = stats.pcc
        record_pcc(
            f"prefill_len{prefill_len}",
            pcc_value,
            seq_len=prefill_len,
            kind=kind_id,
            coverage="full",
            blocks=len(blocks),
        )
        assert pcc_value >= PCC_BAR, f"prefill PCC {pcc_value} at seq_len {prefill_len}"
        results[prefill_len] = pcc_value
        out_dev.deallocate(True)

        if prefill_len == max_context - 1:
            # Decode at the last addressable position on this cache.
            position = max_context - 1
            ref_k[:, :, position] = 0
            ref_v[:, :, position] = 0
            hidden_d = R.unit_rms_hidden_states((1, 1, hidden_size), seed=98)
            ref_d = ref.decode(hidden_d, ref_k, ref_v, torch.tensor([position]))
            current_pos, rope_idxs = U.position_tensors([position], mg_mesh_device)
            out_d = U.decode_output(
                decoder.decode_forward(
                    U.decode_input(hidden_d, mg_mesh_device),
                    kv_cache=kv_cache,
                    page_table=page_table_tt,
                    current_pos=current_pos,
                    rope_idxs=rope_idxs,
                )
            )
            decode_pcc = U.pcc(ref_d, out_d)
            record_pcc("decode_max_position", decode_pcc, position=position, kind=kind_id)
            assert decode_pcc >= PCC_BAR, f"decode PCC {decode_pcc} at position {position}"

        del ref_k, ref_v
        for cache in kv_cache:
            cache.deallocate(True)
        page_table_tt.deallocate(True)


def test_source_has_no_runtime_torch():
    """Static audit: torch / host-transfer calls only exist inside the setup boundary.

    ``from_state_dict`` (weight conversion) and ``allocate_kv_cache`` (empty cache
    allocation) are the two documented setup entry points; every other function in
    ``functional_decoder.py`` must be pure device code.
    """
    import ast
    import inspect

    from models.autoports.meta_models_muse_glimmer_30b.tt import functional_decoder as module

    setup_functions = {"from_state_dict", "allocate_kv_cache"}
    forbidden_ttnn = {"from_torch", "to_torch", "as_tensor", "to_torch_and_close"}
    tree = ast.parse(inspect.getsource(module))
    offenders: list[str] = []

    def visit(node, in_setup: bool):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            in_setup = in_setup or node.name in setup_functions
        if not in_setup:
            if isinstance(node, ast.Name) and node.id == "torch":
                offenders.append(f"line {node.lineno}: reference to torch")
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                if node.value.id == "torch":
                    offenders.append(f"line {node.lineno}: torch.{node.attr}")
                if node.value.id == "ttnn" and node.attr in forbidden_ttnn:
                    offenders.append(f"line {node.lineno}: ttnn.{node.attr}")
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] == "torch":
                        offenders.append(f"line {node.lineno}: import torch")
        for child in ast.iter_child_nodes(node):
            visit(child, in_setup)

    visit(tree, False)
    assert not offenders, "runtime host fallback in functional_decoder.py:\n" + "\n".join(offenders)
