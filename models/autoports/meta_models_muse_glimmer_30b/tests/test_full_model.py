# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Acceptance tests for the full TTNN model and its generator.

Most tests run on the **reduced** stack -- real weights for one sliding and one
full-attention layer, and the real terminal path (real embedding table, real final
norm, real LM head, real padded vocab, real KV-cache and page-table shapes, real
traces, real sampler).  That is deliberate: what these tests pin is the wrapper's
*contract* -- shapes, layouts, padding, page tables, cache ownership, trace
identity, token feedback, reset semantics -- none of which depends on how many
layers are stacked, and a 52-layer build costs ~160 s of host weight packing.

Accuracy and performance are **not** tested here.  They come from the readiness
runners over the all-layer model; see ``doc/full_model/README.md``.  The one
all-layer test is marked ``slow`` and asserts the capability contract that only
the full stack can show: the advertised context fits in DRAM.

Run::

    pytest models/autoports/meta_models_muse_glimmer_30b/tests/test_full_model.py
    pytest models/autoports/meta_models_muse_glimmer_30b/tests/test_full_model.py -m slow
"""

from __future__ import annotations

import inspect
import pathlib

import pytest
import torch

import ttnn
from models.autoports.meta_models_muse_glimmer_30b.tt import model as model_mod
from models.autoports.meta_models_muse_glimmer_30b.tt import optimized_decoder as dec_mod
from models.autoports.meta_models_muse_glimmer_30b.tt.generator import (
    DEFAULT_TRACE_REGION_SIZE,
    MuseGlimmerGenerator,
    build_generator,
    clear_generator_cache,
)
from models.autoports.meta_models_muse_glimmer_30b.tt.model import DECODE_ROWS, dram_capacity_bytes, padded_vocab_size
from models.autoports.meta_models_muse_glimmer_30b.tt.multichip_decoder import close_multichip_mesh, open_multichip_mesh
from models.common.readiness_check.contract import BUILD_GENERATOR_FUNCTION_NAME, Generator

MODEL_DIR = pathlib.Path(__file__).resolve().parents[1]
#: One layer of each kind.  Layer 0 is sliding-window + RoPE, layer 3 is full
#: attention + NoPE, which is the whole space of layer kinds in this checkpoint.
REDUCED_LAYERS = (0, 3)
REDUCED_MAX_SEQ = 4096
VOCAB = 202048
HIDDEN = 6656
HF_CONTEXT = 131072


@pytest.fixture(scope="module")
def mesh():
    device = open_multichip_mesh(trace_region_size=DEFAULT_TRACE_REGION_SIZE)
    yield device
    clear_generator_cache()
    close_multichip_mesh(device)


@pytest.fixture(scope="module")
def generator(mesh) -> MuseGlimmerGenerator:
    """Reduced generator, shared by the module: one build, many contracts."""
    return build_generator(
        MODEL_DIR,
        mesh,
        max_seq_len=REDUCED_MAX_SEQ,
        max_batch_size=1,
        layer_indices=REDUCED_LAYERS,
    )


@pytest.fixture
def clean(generator) -> MuseGlimmerGenerator:
    generator.reset()
    generator.reset_counters()
    return generator


def _prompt(length: int, *, seed: int = 3) -> list[int]:
    return [int(t) for t in torch.randint(0, VOCAB, (length,), generator=torch.Generator().manual_seed(seed)).tolist()]


# ------------------------------------------------------------------ contract


@pytest.mark.timeout(900)
def test_generator_satisfies_the_readiness_contract(generator):
    assert isinstance(generator, Generator)
    module = __import__(
        "models.autoports.meta_models_muse_glimmer_30b.tt.generator", fromlist=[BUILD_GENERATOR_FUNCTION_NAME]
    )
    factory = getattr(module, BUILD_GENERATOR_FUNCTION_NAME)
    parameters = list(inspect.signature(factory).parameters)
    assert parameters[:2] == ["model_dir", "mesh_device"]
    assert generator.tokenizer is not None


def test_generate_declares_enable_trace_explicitly(generator):
    """The teacher-forcing runner rejects a catch-all ``**kwargs``."""
    parameter = inspect.signature(generator.generate).parameters.get("enable_trace")
    assert parameter is not None
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is True


def test_padded_vocab_keeps_every_device_shard_tile_aligned():
    # 202048 / 4 = 50512, which is *not* a multiple of 32: without padding the
    # gather that reconstructs the vocab leaves a 16-column gap per device and
    # every token id on device d is shifted by d*16.
    assert VOCAB % 4 == 0
    assert (VOCAB // 4) % ttnn.TILE_SIZE != 0
    tile_only = padded_vocab_size(VOCAB, 4)
    assert tile_only == 202112 and (tile_only // 4) % ttnn.TILE_SIZE == 0
    dram_sharded = padded_vocab_size(VOCAB, 4, cores=8)
    assert dram_sharded == 202752 and (dram_sharded // 4) % (ttnn.TILE_SIZE * 8) == 0
    assert dram_sharded >= VOCAB


def test_config_reports_the_carried_forward_contract(generator):
    config = generator.model.config
    assert config.hidden_size == HIDDEN
    assert config.vocab_size == VOCAB
    assert config.padded_vocab_size % (config.tp * ttnn.TILE_SIZE) == 0
    assert config.decode_rows == DECODE_ROWS == 32
    assert config.page_block_size == 64
    assert config.tp == 4
    assert set(config.layer_kinds) == {"sliding", "full"}
    # Two widths, doing two different jobs. The *padded* width drives the per-device
    # index offsets -- an unpadded value there shifts every token id on devices 1..3.
    # The *real* width drives the invalid-vocab mask, and `vocab_padding.py` builds no
    # mask at all when the two are equal, which is how this shipped with the 704 padded
    # ids drawable. Both are asserted, and so is the mask they produce.
    sampler = generator.sampling.tt_sampling
    assert sampler.padded_vocab_size == config.padded_vocab_size
    assert sampler.vocab_size == config.vocab_size
    assert sampler.vocab_size != sampler.padded_vocab_size, "or no invalid-vocab mask is built"
    assert (
        sampler.tt_invalid_vocab_mask is not None or sampler.tt_invalid_vocab_tail_mask is not None
    ), "the padded vocab tail must be masked or its ids are sampling candidates"
    assert sampler._invalid_vocab_tail_width == config.padded_vocab_size - config.vocab_size == 704


def test_lm_head_is_column_parallel_and_softcapped(generator):
    head = generator.model.lm_head
    assert tuple(head.weight.shape)[-1] == generator.model.config.local_vocab_size
    assert head.softcap == 20.0
    # The real head is loaded, not the tied embedding: this checkpoint stores both
    # and they differ, so a silent fall back to the embedding would still produce
    # plausible text.
    assert head.tied_to_embedding is False


def test_embedding_is_hidden_fractured_with_a_zero_pad_row(generator):
    model = generator.model
    shards = ttnn.get_device_tensors(model.embed_weight)
    assert len(shards) == model.config.tp
    assert tuple(model.embed_weight.shape) == (VOCAB + 1, HIDDEN // model.config.tp)
    row = ttnn.to_torch(shards[0]).reshape(VOCAB + 1, -1)[model.embed_pad_id]
    assert torch.count_nonzero(row) == 0


def test_rope_tables_are_shared_across_the_sliding_layers(generator):
    model = generator.model
    sliding = [layer for layer in model.layers if layer.config.uses_rope]
    assert sliding, "the reduced stack must contain a sliding layer"
    for layer in sliding:
        assert layer.cos_cache is model.rope_cache["cos"]
        assert layer.sin_cache is model.rope_cache["sin"]
        assert layer.cos_cache_tile is model.rope_cache["cos_tile"]
        assert layer.sin_cache_tile is model.rope_cache["sin_tile"]
    for layer in model.layers:
        if not layer.config.uses_rope:
            assert layer.cos_cache is None


# -------------------------------------------------------------- prompt shapes


@pytest.mark.timeout(900)
@pytest.mark.parametrize("prompt_len", [1, 31, 32, 37, 63, 64, 127, 129, 2049])
def test_prefill_accepts_any_logical_prompt_length(clean, prompt_len):
    """Tile, page (64) and prefill-chunk (4096) alignment are all internal."""
    logits = clean.prefill_forward(
        tokens=torch.tensor([_prompt(prompt_len)], dtype=torch.long),
        page_table=None,
        kv_cache=None,
        prompt_lens=[prompt_len],
    )
    assert logits.shape == (1, 1, VOCAB)
    assert torch.isfinite(logits).all()


def test_prefill_all_logits_covers_every_prompt_position(clean):
    prompt_len = 37
    logits = clean.prefill_forward(
        tokens=torch.tensor([_prompt(prompt_len)], dtype=torch.long),
        page_table=None,
        kv_cache=None,
        prompt_lens=[prompt_len],
        return_all_logits=True,
    )
    assert logits.shape == (1, prompt_len, VOCAB)
    assert torch.isfinite(logits).all()


@pytest.mark.parametrize("prompt_len", [37, 128, 200, 1024])
def test_prefill_is_reproducible(clean, prompt_len):
    """The same prompt prefilled repeatedly must give bit-identical logits.

    Two distinct defects have hidden here.  The first was uninitialised tile padding
    in the embedded prompt, caught at 37 tokens and fixed with the zero-embedding pad
    id.  The second was the embedding all-gather (:data:`EMBED_GATHER_CHUNK_ROWS`),
    which this test could not see for a long time because it only ever ran 37 tokens
    -- one tile row past padding, and *below* the 64-row payload where the gather
    starts to move.  Hence the lengths: one under the threshold and three over it.

    Six repeats, not two.  The gather failed roughly one run in three, so a
    two-sample test would have passed most of the time it was broken
    (``doc/full_model/batch_slot_probe_len_b1.json``).
    """
    prompt = _prompt(prompt_len)
    runs = []
    for _ in range(6):
        clean.reset()
        logits = clean.prefill_forward(
            tokens=torch.tensor([prompt], dtype=torch.long),
            page_table=None,
            kv_cache=None,
            prompt_lens=[prompt_len],
        )
        runs.append(logits[0, 0].clone())
    for index in range(1, len(runs)):
        assert torch.equal(runs[0], runs[index]), (
            f"prefill of {prompt_len} tokens is not reproducible: run {index} differs from run 0 by "
            f"max abs {float((runs[0] - runs[index]).abs().max())}"
        )


def test_generate_rejects_a_prompt_past_the_supported_context(clean, expect_error):
    with expect_error(ValueError, "exceeds the supported context"):
        clean.generate(prompt_token_ids=_prompt(8), max_new_tokens=REDUCED_MAX_SEQ)


# --------------------------------------------------------------- page tables


def test_page_table_is_normalised_to_the_decode_row_width(generator):
    model = generator.model
    rows = model.normalize_page_table(None)
    assert rows.shape == (DECODE_ROWS, model.config.blocks_per_seq)
    assert rows.dtype == torch.int32
    assert int(rows.max()) < model.config.max_num_blocks
    # A caller's narrower table (the readiness prefill check hands in [1, 1024]) is
    # extended, not rejected, and the extension uses blocks the real row does not.
    narrow = torch.arange(16, dtype=torch.int32).reshape(1, 16)
    widened = model.normalize_page_table(narrow)
    assert widened.shape == (DECODE_ROWS, model.config.blocks_per_seq)
    assert torch.equal(widened[0, :16], narrow[0])
    # The supplied row is extended with blocks it does not already use, so the
    # active row's mapping stays injective.
    assert len(set(widened[0].tolist())) == model.config.blocks_per_seq
    # The rows past the caller's are the inactive ones and alias the last real row.
    assert torch.equal(widened[1], widened[0])


def test_page_table_out_of_range_is_rejected(generator, expect_error):
    model = generator.model
    bad = torch.full((1, model.config.blocks_per_seq), model.config.max_num_blocks, dtype=torch.int32)
    with expect_error(ValueError, "references block"):
        model.normalize_page_table(bad)


def test_repeated_decode_copies_the_page_table_once(clean):
    prompt = _prompt(64)
    identity = clean.model.normalize_page_table(None)
    clean.prefill_forward(
        tokens=torch.tensor([prompt], dtype=torch.long),
        page_table=identity,
        kv_cache=None,
        prompt_lens=[len(prompt)],
    )
    clean.reset_counters()
    for step in range(4):
        clean.decode_forward(
            tokens=torch.tensor([[prompt[-1]]], dtype=torch.long),
            start_pos=torch.tensor([len(prompt) + step], dtype=torch.int32),
            page_table=identity,
            kv_cache=None,
            sample_on_device=True,
        )
    assert clean.counters["trace_replays"] == 4
    assert clean.counters["page_table_refreshes"] == 1, clean.counters
    permuted = identity[:, torch.randperm(identity.shape[1], generator=torch.Generator().manual_seed(5))]
    clean.decode_forward(
        tokens=torch.tensor([[prompt[-1]]], dtype=torch.long),
        start_pos=torch.tensor([len(prompt) + 4], dtype=torch.int32),
        page_table=permuted,
        kv_cache=None,
        sample_on_device=True,
    )
    assert clean.counters["page_table_refreshes"] == 2, clean.counters


# ------------------------------------------------------------ cache ownership


def test_kv_cache_is_exposed_and_bindable(generator, expect_error):
    cache = generator.model.kv_cache
    assert len(cache) == len(generator.model.layers)
    assert all(len(pair) == 2 for pair in cache)
    # Rebinding the same handles is a no-op; a wrong shape must fail loudly rather
    # than silently read zeros.
    generator.model.set_kv_cache(cache)
    with expect_error(ValueError, "entries for"):
        generator.model.set_kv_cache(cache[:1])


def test_reset_zeroes_the_cache_without_dropping_traces(clean):
    prompt = _prompt(64)
    clean.generate(prompt_token_ids=prompt, max_new_tokens=3, enable_trace=True)
    trace_id = clean._trace_id
    assert trace_id is not None
    clean.reset()
    for layer in clean.model.layers:
        for cache in (layer.k_cache, layer.v_cache):
            shard = ttnn.to_torch(ttnn.get_device_tensors(cache)[0])
            assert torch.count_nonzero(shard) == 0
    assert clean._trace_id is trace_id, "reset() must not release the decode trace"
    assert clean._prev_page_table is None


# ---------------------------------------------------------- split sampling


def test_split_sampling_feeds_the_sampled_token_back_on_device(clean):
    """Two replays, different tokens and positions, no host staging between them."""
    prompt = _prompt(64)
    # ``max_new_tokens=2``, not 1, and the difference matters: the traces are captured
    # inside the *decode* loop, which ``max_new_tokens=1`` never enters -- that call is
    # prefill only.  With 1 this test still passed in a full-file run, because a sibling
    # test had already captured the traces on the module-scoped generator, and failed
    # with ``StopIteration`` the moment it was run on its own or in a subset.  Asking for
    # one decode step makes the test own the state it asserts on.
    clean.generate(prompt_token_ids=prompt, max_new_tokens=2, enable_trace=True)
    token_input = clean._device_inputs["tokens"]
    slot = next(iter(clean.sampling._trace_states.values()))
    assert slot["id"] is not None, "the sampling path must be traced, not eager"
    assert slot["input"] is clean._trace_logits, "the sampling trace must consume the decode trace's logits"
    assert slot["output"][0] is token_input, "tt_out_tok must be the persistent decode token input"

    def read(tensor):
        return int(ttnn.to_torch(ttnn.get_device_tensors(tensor)[0]).reshape(-1)[0])

    clean._stage(tokens=[prompt[-1]] * DECODE_ROWS, positions=torch.full((DECODE_ROWS,), len(prompt)))
    pos_before, tok_before = read(clean._device_inputs["current_pos"]), read(token_input)
    baseline = dict(clean.counters)

    first = int(clean._decode_step_traced(host_sampling=False)[0])
    assert read(token_input) == first, "the sampled token must land in the decode token input"
    assert read(clean._device_inputs["current_pos"]) == pos_before + 1, "position must advance on device"

    second = int(clean._decode_step_traced(host_sampling=False)[0])
    assert read(token_input) == second
    assert read(clean._device_inputs["current_pos"]) == pos_before + 2
    assert read(clean._device_inputs["rope_pos_ids"]) == pos_before + 2

    for key in ("token_refreshes", "position_refreshes", "page_table_refreshes", "synchronizations"):
        assert clean.counters[key] == baseline[key], f"{key} changed between replays: {clean.counters}"
    assert tok_before == prompt[-1], "the staged token is what the first replay must have consumed"


def test_greedy_is_the_top_k_path_not_force_argmax(generator):
    """Force-argmax would need a full-vocab all-gather and cannot feed ``tt_out_tok``."""
    assert generator.sampling.tt_sampling.force_argmax_sampling is False
    params = generator._sampling_params or None
    if params is None:
        generator._apply_sampling_params(None)
        params = generator._sampling_params
    assert params.top_k[0] == 1
    assert params.top_p[0] == 0.0
    assert params.temperature[0] == 1.0


def test_greedy_decode_is_deterministic_across_calls(clean):
    prompt = _prompt(64)
    first = clean.generate(prompt_token_ids=prompt, max_new_tokens=6, enable_trace=True)
    clean.reset()
    second = clean.generate(prompt_token_ids=prompt, max_new_tokens=6, enable_trace=True)
    assert first == second


def test_steady_state_decode_does_no_per_token_host_work(clean):
    prompt = _prompt(64)
    clean.generate(prompt_token_ids=prompt, max_new_tokens=4, enable_trace=True)  # capture
    clean.reset()
    clean.reset_counters()
    gen_len = 17
    clean.generate(prompt_token_ids=prompt, max_new_tokens=gen_len, enable_trace=True)
    counters = clean.counters
    assert counters["trace_replays"] == gen_len - 1
    # One token/position stage for the post-prefill reseed, one page-table copy per
    # request, and one 32-uint32 readback per token because the caller wants the
    # tokens.  Nothing else.
    assert counters["token_refreshes"] == 1, counters
    assert counters["position_refreshes"] == 1, counters
    assert counters["page_table_refreshes"] == 1, counters
    assert counters["synchronizations"] == 0, counters
    assert counters["readbacks"] == gen_len, counters


def test_teacher_forcing_returns_predictions_not_forced_inputs(clean):
    prompt = _prompt(64)
    forced = _prompt(6, seed=99)
    seen: list[int] = []

    def next_input(step: int, predicted: int) -> int:
        seen.append(predicted)
        return forced[step]

    predictions = clean.generate(prompt_token_ids=prompt, max_new_tokens=6, next_input=next_input, enable_trace=True)
    assert predictions == seen
    assert len(predictions) == 6
    assert predictions != forced or all(p == f for p, f in zip(predictions, forced))
    # Teacher forcing still runs through the traced decode and the traced sampler.
    assert clean._trace_id is not None
    assert clean._sampling_captured


def test_sampling_never_returns_a_padded_vocab_id(clean):
    """No sampled id may land in the padded tail, under greedy *or* top-k/top-p.

    The LM head zero-fills the 704 padded columns, so each carries logit exactly
    `20*tanh(0) = 0.0` — which beats every real token whose logit is negative. On the
    real stack that happens: `evidence_misses_bfp4.json` has positions where only three
    or four of 202048 real logits exceed 0. Without the invalid-vocab mask those columns
    enter the local top-k, and a temperature-raised request can draw one; the id is then
    outside the tokenizer and outside the 202049-row embedding table it is fed back into.

    Greedy is checked too, but the sampled path is the one that can actually reach them.
    """
    from models.common.sampling.generator import SamplingParams

    prompt = _prompt(64)
    greedy = clean.generate(prompt_token_ids=prompt, max_new_tokens=8, enable_trace=True)
    assert all(0 <= token < VOCAB for token in greedy), greedy

    for params in (
        SamplingParams(temperature=1.0, top_k=32, top_p=1.0),
        SamplingParams(temperature=2.0, top_k=32, top_p=1.0),
        SamplingParams(temperature=2.0, top_k=32, top_p=0.95),
    ):
        clean.reset()
        sampled = clean.generate(prompt_token_ids=prompt, max_new_tokens=16, enable_trace=True, sampling_params=params)
        assert all(0 <= token < VOCAB for token in sampled), (params, sampled)


def test_the_api_guards_refuse_what_they_cannot_do(generator, expect_error):
    """Each guard exists because the failure they replace was silent.

    ``generate(user_id=k)`` used to prefill cache slot *k* and then decode row 0, which
    at ``max_batch_size=1`` is invisible (every page-table row aliases slot 0) and on a
    multi-slot generator would silently decode an unfilled slot.

    ``prefill_forward(continuation=True)`` used to be swallowed by ``**kwargs`` and give
    an ordinary non-continuation prefill with no error — while README limitation 2
    advertised the sliding-tail hand-off as "implemented and exposed".

    ``start_pos`` is the exception: the serving prefill signature in
    ``models/common/readiness_check/contract_vllm.py`` makes it *required*, and 0 is what
    a single-chunk caller passes, so 0 must be accepted rather than refused.
    """
    prompt = _prompt(32)
    with expect_error(ValueError, "cache slot 0 only"):
        generator.generate(prompt_token_ids=prompt, max_new_tokens=2, user_id=1)
    with expect_error(NotImplementedError, "continuation"):
        generator.prefill_forward(
            tokens=torch.tensor([prompt], dtype=torch.long),
            page_table=None,
            kv_cache=None,
            prompt_lens=[len(prompt)],
            continuation=True,
        )
    with expect_error(NotImplementedError, "start_pos"):
        generator.prefill_forward(
            tokens=torch.tensor([prompt], dtype=torch.long),
            page_table=None,
            kv_cache=None,
            prompt_lens=[len(prompt)],
            start_pos=64,
        )
    # start_pos=0 is the serving contract's ordinary case and must work.
    generator.reset()
    logits = generator.prefill_forward(
        tokens=torch.tensor([prompt], dtype=torch.long),
        page_table=None,
        kv_cache=None,
        prompt_lens=[len(prompt)],
        start_pos=0,
    )
    assert logits.shape == (1, 1, VOCAB)


def test_topk_runs_through_the_multi_core_factory(generator):
    """The sampler's dominant op must be in the multi-core regime, and stay there.

    A silent revert to the single-core `ttnn.topk` costs **9.10 ms/token** — a third of
    the decode step — and changes no output at all, so every other test in this file
    passes either way. This is the only gate on it.

    The conditions are `topk_device_operation.cpp::select_program_factory`'s: each
    reduced width a power of two, below the uint16 index bound, at least
    `multi_core_min_width`, with `k <= 64`.
    """
    sampler = generator.sampling.tt_sampling
    shard = generator.model.config.padded_vocab_size // generator.model.config.tp
    assert sampler.topk_split_to_power_of_2, "the multi-core topk split is the shipped configuration"
    assert sampler.pad_to_power_of_2, "the split needs the pad: 50688 must become 65536 before it is halved"
    assert sampler.topk_pieces == 2, sampler.topk_pieces
    assert sampler.candidates_per_device == sampler.max_top_k * sampler.topk_pieces == 64

    padded = 1 << (shard - 1).bit_length()
    per_piece = padded // sampler.topk_pieces
    assert padded == 65536 and per_piece == 32768
    assert per_piece & (per_piece - 1) == 0, "power of two, for the bitonic sort"
    assert per_piece < 65535, "under the uint16 index bound the multi-core factory requires"
    assert per_piece >= 8192, "at least multi_core_min_width"
    assert sampler.max_top_k <= 64, "the multi-core factory's k limit"
    # The gathered width must still give ttnn.sampling a power-of-two tile count.
    gathered_tiles = sampler.candidates_per_device * generator.model.config.tp // ttnn.TILE_SIZE
    assert gathered_tiles == 8 and gathered_tiles & (gathered_tiles - 1) == 0


def test_caller_driven_decode_restages_only_the_token(clean):
    """Teacher forcing changes the token, not the position.

    The in-trace ``plus_one`` advances ``current_pos`` and the RoPE index after every
    read of them, so a caller-driven step only has to resupply the token it is forcing.
    Restaging positions as well is redundant host work per token: the values written are
    the values already there. No millisecond figure is claimed for removing it -- every
    teacher-forcing rate in the evidence tree also spans the topk split, which moved that
    number by ~9 ms/token, so nothing isolates this change. This test exists because the
    regression is invisible in output and cheap to reintroduce.

    One position refresh is expected and required: the first decode step after prefill,
    whose position is the prompt length rather than a +1.
    """
    prompt = _prompt(64)
    forced = _prompt(8, seed=77)
    clean.generate(prompt_token_ids=prompt, max_new_tokens=2, enable_trace=True)  # capture
    clean.reset()
    clean.reset_counters()
    clean.generate(
        prompt_token_ids=prompt,
        max_new_tokens=8,
        next_input=lambda step, predicted: forced[step],
        enable_trace=True,
    )
    counters = clean.counters
    assert counters["position_refreshes"] == 1, counters
    # 8 requested tokens is *7* decode steps: the first prediction comes out of prefill.
    # Each step resupplies its forced token, and exactly one of them (the reseed) also
    # writes positions -- which is the single position refresh asserted above.
    assert counters["token_refreshes"] == 7, counters
    assert counters["synchronizations"] == 0, counters


def test_device_sampling_keeps_each_batch_row_token_in_its_own_row(mesh):
    """Row *i*'s sampled token must be row *i*'s argmax, for **distinct** prompts.

    Nothing else pins this. The mixed-length batch test asserts only shapes and id
    ranges, and the cross-slot reproducibility test puts the *same* prompt in both
    slots -- so a permutation through ``tt_out_tok`` would be invisible to both. This
    is the contract the vLLM stage depends on, so it is checked against host argmax on
    the same state: prefill, read the per-row logits, then re-prefill the identical
    state and sample on device.
    """
    batch = 4
    generator = build_generator(MODEL_DIR, mesh, max_seq_len=1024, max_batch_size=batch, layer_indices=REDUCED_LAYERS)
    try:
        lengths = [64, 96, 128, 160]
        width = max(lengths)
        tokens = torch.zeros(batch, width, dtype=torch.long)
        for user, length in enumerate(lengths):
            tokens[user, :length] = torch.tensor(_prompt(length, seed=200 + user))
        step_tokens = torch.tensor([[int(tokens[u, lengths[u] - 1])] for u in range(batch)], dtype=torch.long)
        start = torch.tensor(lengths, dtype=torch.int32)

        generator.reset()
        generator.prefill_forward(tokens=tokens, page_table=None, kv_cache=None, prompt_lens=lengths)
        logits = generator.decode_forward(
            tokens=step_tokens, start_pos=start, page_table=None, kv_cache=None, sample_on_device=False
        )
        expected = [int(logits[user].argmax()) for user in range(batch)]

        generator.reset()
        generator.prefill_forward(tokens=tokens, page_table=None, kv_cache=None, prompt_lens=lengths)
        sampled = generator.decode_forward(
            tokens=step_tokens, start_pos=start, page_table=None, kv_cache=None, sample_on_device=True
        )
        got = [int(sampled[user]) for user in range(batch)]
        # Distinct prompts, so a row permutation shows up as a mismatch rather than as
        # four copies of the same token.
        assert len(set(expected)) > 1, f"prompts too degenerate to detect a permutation: {expected}"
        assert got == expected, f"device sampling permuted rows: got {got}, host argmax {expected}"
    finally:
        generator.teardown()


def test_host_sampling_agrees_with_the_device_sampler_on_the_same_logits(clean):
    """The device sampler and a host argmax must pick the same token, or tie.

    Asserted on **one shared logits tensor** rather than on two independent
    generations.  Two generations can legitimately diverge at a near-tie: the
    device sampler's top-k path and a host argmax break an exact bf16 tie
    differently, and on the reduced stack with random-id prompts near-ties are
    common (the same sensitivity the README records for the permuted page table).
    Comparing on shared logits makes the claim precise: they agree unless the
    top-2 margin is inside the bf16 quantum at that magnitude, in which case
    either answer is correct.
    """
    prompt = _prompt(64)
    clean.generate(prompt_token_ids=prompt, max_new_tokens=2, enable_trace=True)  # capture
    clean._stage(tokens=[prompt[-1]] * DECODE_ROWS, positions=torch.full((DECODE_ROWS,), len(prompt)))
    device_token = int(clean._decode_step_traced(host_sampling=False)[0])

    logits = clean.model.logits_to_torch(clean._trace_logits)[0].float()
    top2 = torch.topk(logits, k=2)
    host_token = int(top2.indices[0])
    margin = float(top2.values[0] - top2.values[1])
    quantum = abs(float(top2.values[0])) * 2**-8  # bf16 has an 8-bit mantissa

    assert device_token == host_token or margin <= quantum, (
        f"device sampler picked {device_token}, host argmax picked {host_token}, "
        f"top-2 margin {margin:.4f} exceeds the bf16 quantum {quantum:.4f} at this magnitude"
    )


def test_host_sampling_mode_runs_end_to_end(clean):
    """The compatibility mode itself works: same length, real ids, no exceptions."""
    prompt = _prompt(64)
    host_tokens = clean.generate(prompt_token_ids=prompt, max_new_tokens=6, enable_trace=True, host_sampling=True)
    assert len(host_tokens) == 6
    assert all(0 <= token < VOCAB for token in host_tokens)


def test_top_k_top_p_runs_through_the_same_path_and_greedy_survives_it(clean):
    from models.common.sampling.generator import SamplingParams

    prompt = _prompt(64)
    greedy = clean.generate(prompt_token_ids=prompt, max_new_tokens=6, enable_trace=True)
    clean.reset()
    clean.generate(
        prompt_token_ids=prompt,
        max_new_tokens=6,
        enable_trace=True,
        sampling_params=SamplingParams(temperature=0.8, top_k=32, top_p=0.95),
    )
    clean.reset()
    again = clean.generate(prompt_token_ids=prompt, max_new_tokens=6, enable_trace=True)
    assert again == greedy, "a sampled request must not leave greedy decode changed"


# ------------------------------------------------------------- low-level API


def test_decode_forward_returns_logits_or_tokens_as_asked(clean):
    prompt = _prompt(64)
    clean.prefill_forward(
        tokens=torch.tensor([prompt], dtype=torch.long),
        page_table=None,
        kv_cache=clean.model.kv_cache,
        prompt_lens=[len(prompt)],
    )
    logits = clean.decode_forward(
        tokens=torch.tensor([[prompt[-1]]], dtype=torch.long),
        start_pos=torch.tensor([len(prompt)], dtype=torch.int32),
        page_table=None,
        kv_cache=clean.model.kv_cache,
    )
    assert logits.shape == (1, VOCAB)
    tokens = clean.decode_forward(
        tokens=torch.tensor([[prompt[-1]]], dtype=torch.long),
        start_pos=torch.tensor([len(prompt) + 1], dtype=torch.int32),
        page_table=None,
        kv_cache=clean.model.kv_cache,
        sample_on_device=True,
    )
    assert tokens.shape == (1,)
    assert 0 <= int(tokens[0]) < VOCAB


def test_inactive_rows_carry_the_minus_one_sentinel(generator):
    current_pos, rope = generator.model.positions_to_device(torch.tensor([5]), device=False)
    host_pos = ttnn.to_torch(ttnn.get_device_tensors(current_pos)[0]).reshape(-1)
    host_rope = ttnn.to_torch(ttnn.get_device_tensors(rope)[0]).reshape(-1)
    assert host_pos.numel() == DECODE_ROWS
    assert int(host_pos[0]) == 5
    assert all(int(v) == -1 for v in host_pos[1:])
    # The RoPE index is unsigned, so its inactive rows are clamped to 0 instead.
    assert int(host_rope[0]) == 5
    assert all(int(v) == 0 for v in host_rope[1:])


# ------------------------------------------------------------------- batching


@pytest.mark.slow
@pytest.mark.timeout(1800)
@pytest.mark.parametrize("batch", [4, 32])
def test_batched_prefill_and_decode_with_mixed_lengths(mesh, batch):
    """Cache slots and per-user prompt lengths, through the low-level API."""
    seq = 1024
    generator = build_generator(MODEL_DIR, mesh, max_seq_len=seq, max_batch_size=batch, layer_indices=REDUCED_LAYERS)
    try:
        # The ladder is derived from the context, not a fixed stride: a fixed 37
        # tokens per user runs off the end at batch 32 (200 + 37*31 = 1347 > 1024).
        # ``seq - 128`` leaves room for the decode step below to stay in range.
        base, top = 200, seq - 128
        step = max(1, (top - base) // max(1, batch - 1))
        lengths = [base + step * user for user in range(batch)]
        assert max(lengths) + 1 <= seq
        width = max(lengths)
        tokens = torch.zeros(batch, width, dtype=torch.long)
        for user, length in enumerate(lengths):
            tokens[user, :length] = torch.tensor(_prompt(length, seed=user))
        logits = generator.prefill_forward(tokens=tokens, page_table=None, kv_cache=None, prompt_lens=lengths)
        assert logits.shape == (batch, 1, VOCAB)
        assert torch.isfinite(logits).all()
        sampled = generator.decode_forward(
            tokens=tokens[:, :1],
            start_pos=torch.tensor(lengths, dtype=torch.int32),
            page_table=None,
            kv_cache=None,
            sample_on_device=True,
        )
        assert sampled.shape == (batch,)
        assert all(0 <= int(t) < VOCAB for t in sampled)
    finally:
        generator.teardown()


# --------------------------------------------------------------- all layers


@pytest.mark.slow
@pytest.mark.timeout(1800)
def test_logits_are_reproducible_across_batch_positions(mesh):
    """The same prompt in two cache slots must produce the same logits and token.

    A per-slot difference would mean the batch dimension leaks -- a page-table row
    mixed up, a position tensor row misaligned, or a sampler parameter row shifted --
    and none of those show up at batch 1.
    """
    generator = build_generator(MODEL_DIR, mesh, max_seq_len=1024, max_batch_size=4, layer_indices=REDUCED_LAYERS)
    try:
        prompt = _prompt(200, seed=41)
        tokens = torch.tensor([prompt, prompt], dtype=torch.long)
        logits = generator.prefill_forward(
            tokens=tokens, page_table=None, kv_cache=None, prompt_lens=[len(prompt), len(prompt)]
        )
        assert logits.shape == (2, 1, VOCAB)
        # Two slots, same prompt, different page-table rows and different cache
        # blocks: the logits must still agree exactly.
        assert torch.equal(logits[0], logits[1]), (
            "the same prompt in two cache slots produced different logits: "
            f"max abs diff {float((logits[0] - logits[1]).abs().max())}"
        )
        sampled = generator.decode_forward(
            tokens=torch.tensor([[prompt[-1]], [prompt[-1]]], dtype=torch.long),
            start_pos=torch.tensor([len(prompt), len(prompt)], dtype=torch.int32),
            page_table=None,
            kv_cache=None,
            sample_on_device=True,
        )
        assert int(sampled[0]) == int(sampled[1]), (int(sampled[0]), int(sampled[1]))
    finally:
        generator.teardown()


@pytest.mark.slow
@pytest.mark.timeout(3600)
def test_all_layers_fit_the_advertised_context(mesh):
    """The capability contract: 52 layers plus a full-context cache, in DRAM."""
    generator = build_generator(MODEL_DIR, mesh, max_seq_len=HF_CONTEXT, max_batch_size=1)
    try:
        report = generator.model.dram_report()
        capacity = dram_capacity_bytes(mesh)
        assert generator.model.config.max_seq_len == HF_CONTEXT
        assert generator.model.config.num_layers == 52
        assert report["per_device_total_bytes"] < capacity
        # The RoPE tables are shared, not per layer: 39 sliding layers x 134 MB
        # would be 5.2 GB, more than the whole weight footprint.
        assert report["per_device_rope_table_bytes"] == 4 * HF_CONTEXT * 128 * 2
        prompt = _prompt(128)
        tokens = generator.generate(prompt_token_ids=prompt, max_new_tokens=4, enable_trace=True)
        assert len(tokens) == 4
        assert all(0 <= t < VOCAB for t in tokens)
        # On the real stack the logits are not degenerate, so the host-sampling
        # compatibility mode must agree token for token with the device sampler.
        generator.reset()
        host = generator.generate(prompt_token_ids=prompt, max_new_tokens=4, enable_trace=True, host_sampling=True)
        assert host == tokens, (host, tokens)
    finally:
        generator.teardown()


# ------------------------------------------- optimized-full-model stage contracts
#
# The three decode-path changes the optimized-full-model stage ships. Each one is
# a *layout* change with no arithmetic in it, so what needs pinning is that the
# layout is the one claimed and that the numbers did not move.


def test_lm_head_softcap_runs_in_l1_and_matches_the_dram_form(generator):
    """The softcap runs on the matmul's own shard, and is the same tensor either way.

    ``LM_HEAD_SOFTCAP_IN_L1`` moves ``tanh`` and the scalar multiply off a
    DRAM-interleaved ``[1, 1, 32, 50688]`` tensor and onto the width-sharded L1
    output the matmul already produced.  The shard is padded (975 columns per core
    is not a tile multiple), so this also pins that no padded lane survives into
    the logits the sampler sees.
    """
    model = generator.model
    head = model.lm_head
    assert head.softcap_in_l1 is True
    assert model_mod.LM_HEAD_SOFTCAP_IN_L1 is True

    hidden = ttnn.from_torch(
        torch.randn(1, 1, DECODE_ROWS, HIDDEN, generator=torch.Generator().manual_seed(5)).to(torch.bfloat16),
        device=generator.mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=model.boundary_memcfg(DECODE_ROWS),
        mesh_mapper=ttnn.ReplicateTensorToMesh(generator.mesh_device),
    )
    try:
        shipped = model.logits_to_torch(head.forward(hidden))
        head.softcap_in_l1 = False
        dram = model.logits_to_torch(head.forward(hidden))
    finally:
        head.softcap_in_l1 = True
        ttnn.deallocate(hidden)
    assert shipped.shape == dram.shape == (DECODE_ROWS, VOCAB)
    # bf16 in, bf16 out, same two ops in the same order: bit-identical.
    assert torch.equal(shipped, dram), (shipped - dram).abs().max()
    assert torch.isfinite(shipped).all()
    assert shipped.abs().max() <= head.softcap + 1e-3


@pytest.mark.parametrize("token", [7, 0, VOCAB - 1])
def test_decode_embedding_gathers_straight_into_the_boundary_layout(generator, token):
    """The sharded-output gather returns the boundary layout *and* the same values.

    The layout assertion is the cheap half.  The half that matters is that an async
    all-gather writing width-sharded L1 returns what the DRAM-interleaved gather plus
    ``interleaved_to_sharded`` returned -- the decoder stage rejected persistent CCL
    buffers on an intermittent *first-use* fault in this same op family, so a new
    output-layout contract on it gets a value comparison, repeated, not just a shape
    check.  Three token ids because the gather's payload is the embedding row, and
    row 0 and the last real row exercise the two ends of the table.
    """
    model = generator.model
    assert model_mod.EMBED_DECODE_GATHER_SHARDED is True
    boundary = model.boundary_memcfg(DECODE_ROWS)
    tokens = model.tokens_to_device([token] * DECODE_ROWS)

    def to_host(tensor):
        source = tensor if not tensor.is_sharded() else ttnn.sharded_to_interleaved(tensor, ttnn.DRAM_MEMORY_CONFIG)
        host = ttnn.to_torch(ttnn.get_device_tensors(source)[0])
        if source is not tensor:
            ttnn.deallocate(source)
        return host.reshape(DECODE_ROWS, HIDDEN)

    try:
        # The interleaved reference, gathered the way the full-model stage did it.
        reference = model._embed(tokens)
        try:
            assert reference.memory_config() == ttnn.DRAM_MEMORY_CONFIG
            want = to_host(reference)
        finally:
            ttnn.deallocate(reference)
        # Repeated, because the failure mode this guards against is intermittent.
        for attempt in range(4):
            gathered = model._embed(tokens, memory_config=boundary)
            try:
                assert gathered.is_sharded()
                assert gathered.memory_config() == boundary
                assert tuple(gathered.shape)[-1] == HIDDEN
                got = to_host(gathered)
            finally:
                ttnn.deallocate(gathered)
            assert torch.equal(got, want), (attempt, (got - want).abs().max())
    finally:
        ttnn.deallocate(tokens)


def test_swiglu_multiply_runs_on_the_wide_grid_and_returns_the_narrow_one(generator):
    """The SwiGLU reshard is on, and the MLP's public output layout is unchanged.

    ``DECODE_SWIGLU_MUL_CORES`` spreads the SFPU SiLU over 80 cores and reshards
    the product back, so ``mlp_down``'s input is still the gate/up grid. What that
    must not do is change the layer's boundary contract, which is what this checks
    on the real decode step.
    """
    assert dec_mod.DECODE_SWIGLU_MUL_CORES == 80
    assert 5120 // 32 % dec_mod.DECODE_SWIGLU_MUL_CORES == 0, "the wide grid must divide the intermediate width"
    model = generator.model
    boundary = model.boundary_memcfg(DECODE_ROWS)
    hidden = ttnn.from_torch(
        torch.randn(1, 1, DECODE_ROWS, HIDDEN, generator=torch.Generator().manual_seed(11)).to(torch.bfloat16) * 0.05,
        device=generator.mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=boundary,
        mesh_mapper=ttnn.ReplicateTensorToMesh(generator.mesh_device),
    )
    layer = model.layers[0]
    try:
        out = layer.mlp.decode_forward(hidden, DECODE_ROWS)
        try:
            assert out.memory_config() == boundary
            assert tuple(out.shape) == (1, 1, DECODE_ROWS, HIDDEN)
            values = ttnn.to_torch(
                ttnn.get_device_tensors(ttnn.sharded_to_interleaved(out, ttnn.DRAM_MEMORY_CONFIG))[0]
            )
            assert torch.isfinite(values.to(torch.float32)).all()
        finally:
            ttnn.deallocate(out)
    finally:
        ttnn.deallocate(hidden)


def test_prefill_trace_is_opt_in_and_matches_the_eager_path(mesh):
    """The opt-in prefill trace: same tokens, one bucket, eager fallback beyond it.

    Prefill on this mesh is host-issue bound (see
    ``doc/optimized_full_model/README.md``), and a trace is the only thing that removes
    host issue.  It is off by default because one trace serves one 32-row bucket and
    capture costs ~98 ms, so this pins three things a caller turning it on depends on:
    the traced prompt returns exactly what the eager path returned, a *different*
    padded length past ``prefill_trace_max_entries`` still works (through the eager
    path), and a non-tile-aligned prompt inside the traced bucket is served by it.
    """
    clear_generator_cache()
    eager = build_generator(MODEL_DIR, mesh, max_seq_len=REDUCED_MAX_SEQ, layer_indices=REDUCED_LAYERS, reuse=False)
    prompt = _prompt(128, seed=41)
    unaligned = _prompt(120, seed=41)[:120]
    other = _prompt(256, seed=41)
    try:
        want = eager.generate(prompt_token_ids=prompt, max_new_tokens=4, enable_trace=True)
        eager.reset()
        want_unaligned = eager.generate(prompt_token_ids=unaligned, max_new_tokens=3, enable_trace=True)
        eager.reset()
        want_other = eager.generate(prompt_token_ids=other, max_new_tokens=3, enable_trace=True)
        assert eager._prefill_traces == {}, "the prefill trace must be off by default"
    finally:
        eager.teardown()
        eager.model.deallocate()
        clear_generator_cache()

    traced = build_generator(
        MODEL_DIR,
        mesh,
        max_seq_len=REDUCED_MAX_SEQ,
        layer_indices=REDUCED_LAYERS,
        reuse=False,
        prefill_trace=True,
    )
    try:
        got = traced.generate(prompt_token_ids=prompt, max_new_tokens=4, enable_trace=True)
        assert list(traced._prefill_traces) == [128], traced._prefill_traces
        assert got == want, (got, want)

        # A second call on the same bucket replays rather than recaptures.
        traced.reset()
        again = traced.generate(prompt_token_ids=prompt, max_new_tokens=4, enable_trace=True)
        assert again == want, (again, want)
        assert list(traced._prefill_traces) == [128]

        # 120 tokens pad to the same 128-row bucket, so the trace serves them and only
        # the row within the tile differs -- which is host arithmetic.
        traced.reset()
        assert traced.generate(prompt_token_ids=unaligned, max_new_tokens=3, enable_trace=True) == want_unaligned
        assert list(traced._prefill_traces) == [128]

        # A different bucket past the cache bound falls back to eager rather than evicting.
        traced.reset()
        assert traced.generate(prompt_token_ids=other, max_new_tokens=3, enable_trace=True) == want_other
        assert list(traced._prefill_traces) == [128]
    finally:
        traced.teardown()
        assert traced._prefill_traces == {}, "teardown must release every prefill trace"
        traced.model.deallocate()
        clear_generator_cache()


def test_prefill_trace_survives_rebinding_the_same_external_cache(mesh):
    """The serving path: the same external KV cache every request must not recapture.

    ``prefill_forward(kv_cache=...)`` is the API a vLLM adapter drives, and it is the
    caller the prefill trace is advertised for.  Invalidating on ``kv_cache is not None``
    rather than on the cache *moving* would release and recapture the trace every
    request -- 98 ms of capture plus a 45 ms replay against a 60 ms eager prefill, i.e.
    the flag would make that caller ~83 ms/request slower.  This pins the identity
    comparison: same handles -> one capture and then replays; different handles -> the
    traces go, because their baked addresses no longer point at the caller's cache, and
    the next prefill recaptures against the new ones.
    """
    clear_generator_cache()
    generator = build_generator(
        MODEL_DIR,
        mesh,
        max_seq_len=REDUCED_MAX_SEQ,
        layer_indices=REDUCED_LAYERS,
        reuse=False,
        prefill_trace=True,
    )
    prompt = _prompt(96, seed=53)
    try:
        own = generator.model.kv_cache  # the model's own pairs, threaded back in as a caller would
        first = generator.prefill_forward(torch.tensor([prompt]), kv_cache=own, prompt_lens=[len(prompt)])
        assert list(generator._prefill_traces) == [96], generator._prefill_traces
        captured = dict(generator._prefill_traces)
        signature = generator._prefill_trace_cache_sig
        assert signature, "the capture must record the cache it was captured over"

        # Same handles, three more requests: the trace ids must not change.
        for _ in range(3):
            generator.reset()
            again = generator.prefill_forward(torch.tensor([prompt]), kv_cache=own, prompt_lens=[len(prompt)])
            assert {k: v["id"] for k, v in generator._prefill_traces.items()} == {
                k: v["id"] for k, v in captured.items()
            }, "rebinding the same cache must not release the prefill trace"
            assert torch.equal(again.argmax(dim=-1), first.argmax(dim=-1))
        assert generator._prefill_trace_cache_sig == signature

        # A cache bound to *different* buffers must release the old traces -- the baked
        # addresses would otherwise write the old buffers and the caller would read zeros
        # -- and then recapture against the new ones, so the caller keeps the flag it
        # asked for instead of being silently downgraded to eager prefill forever.
        moved = [
            [ttnn.clone(k, memory_config=k.memory_config()), ttnn.clone(v, memory_config=v.memory_config())]
            for k, v in own
        ]
        try:
            generator.reset()
            out = generator.prefill_forward(torch.tensor([prompt]), kv_cache=moved, prompt_lens=[len(prompt)])
            assert generator._prefill_trace_releases == 1, "a moved cache must release the prefill traces"
            assert list(generator._prefill_traces) == [96], "and then recapture against the new buffers"
            recaptured = {k: v["id"] for k, v in generator._prefill_traces.items()}
            assert recaptured != {k: v["id"] for k, v in captured.items()}, "with new trace ids"
            assert generator._prefill_trace_cache_sig not in ((), signature)
            assert torch.equal(out.argmax(dim=-1), first.argmax(dim=-1))
            # ...and a later request on the same moved cache reuses the recapture rather
            # than releasing again.
            generator.reset()
            generator.prefill_forward(torch.tensor([prompt]), kv_cache=moved, prompt_lens=[len(prompt)])
            assert {k: v["id"] for k, v in generator._prefill_traces.items()} == recaptured
            assert generator._prefill_trace_releases == 1
        finally:
            # Order matters, and getting it wrong is what the watcher caught: the
            # recaptured trace holds the *moved* cache's buffer addresses, so freeing
            # those buffers while it still exists is a use-after-free.  It shows up as
            # ``subordinate_erisc detected invalid NOC command buffer state ...
            # fabric_erisc_router.cpp`` on acteth core 29-25, not as a wrong number
            # (``doc/optimized_full_model/logs/watcher_bisect_rebind.log``).  This is a
            # live hazard in this test, not a historical note: the assertions above
            # require a trace over ``moved`` to exist at this point.
            generator._release_prefill_traces()
            generator.model.set_kv_cache(own)
            for pair in moved:
                for tensor in pair:
                    ttnn.deallocate(tensor)
    finally:
        generator.teardown()
        generator.model.deallocate()
        clear_generator_cache()
