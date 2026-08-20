# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Full-model and generator gates for Qwen3-Coder-30B-A3B on the 4-die mesh.

Two tiers, selected by ``QWEN3_FULL_MODEL_LAYERS`` (default 2):

* the **reduced** tier, one real layer of each kind (there is only one kind
  here) with every other shape, memory config, cache/page-table layout, terminal
  norm/LM head and sampler call identical to the shipped path. Two layers load
  in ~10 s, which is what makes these runnable as a normal test suite;
* the **all-layer** tier, ``QWEN3_FULL_MODEL_LAYERS=48``, which is the final
  evidence and takes several minutes to load.

Accuracy against HuggingFace is *not* asserted here. A 48-layer torch reference
is a 61 GB CPU forward, so the accuracy gate is
``models.common.readiness_check.run_prefill_check`` /
``run_teacher_forcing`` against the AIME24 chat reference, reported in
``doc/full_model/README.md``. What these tests own is everything that can be
wrong *without* moving PCC: the trace/feedback contract, position coherence,
page-table refresh policy, non-aligned prompt lengths, batch handling, cache
ownership, reset semantics and the runtime fallback audit.
"""

from __future__ import annotations

import math
import os

import pytest
import torch

import ttnn

from ..tt import multichip_decoder as MC
from ..tt.generator import Qwen3CoderGenerator, build_generator
from ..tt.model import DEFAULT_TRACE_REGION_SIZE

MODEL_DIR = "models/demos/blackhole/qwen3_coder_30b_a3b"
CONTEXT = 8192


@pytest.fixture(scope="module")
def mesh_device():
    """A **module-scoped** 4-die ring mesh, opened here rather than by conftest.

    The repository's `mesh_device` fixture is function-scoped, so a module-scoped
    generator cannot depend on it (`ScopeMismatch`). Reopening the mesh per test
    would also mean reloading the model per test -- ten seconds at two layers and
    over three minutes at forty-eight -- which would make the all-layer tier
    unrunnable. `FABRIC_1D_RING` must be set before the open, exactly as the
    stage-03/04 tests and every probe in `doc/full_model/probes/` do it.
    """
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(*MC.MESH_SHAPE), trace_region_size=DEFAULT_TRACE_REGION_SIZE)
    yield mesh
    ttnn.close_mesh_device(mesh)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


#: Layer count for this run. An environment variable rather than a pytest
#: option so that the choice survives being run as part of the whole model
#: suite from the repository root, where a subdirectory ``conftest.py`` would
#: not have been loaded in time to register an option.
NUM_LAYERS = int(os.environ.get("QWEN3_FULL_MODEL_LAYERS", "2"))


@pytest.fixture(scope="module")
def num_layers():
    return NUM_LAYERS


@pytest.fixture(scope="module")
def generator(mesh_device, num_layers):
    gen = build_generator(
        MODEL_DIR,
        mesh_device,
        override_num_layers=num_layers,
        max_context_len=CONTEXT,
        max_batch_size=1,
    )
    yield gen
    gen.teardown()


@pytest.fixture(scope="module")
def batch_generator(mesh_device, num_layers):
    gen = build_generator(
        MODEL_DIR,
        mesh_device,
        override_num_layers=num_layers,
        max_context_len=1024,
        max_batch_size=4,
    )
    yield gen
    gen.teardown()


@pytest.fixture(scope="module")
def small_rope_generator(mesh_device, num_layers):
    """A generator whose cos/sin tables are far shorter than its context.

    ``rope_cache_len`` defaults to 8192 against a 262144-token contract, so the
    gap between "the table is sized" and "the context is advertised" is real on
    the shipped configuration; 64 makes it reachable in a handful of tokens.
    """
    gen = build_generator(
        MODEL_DIR,
        mesh_device,
        override_num_layers=num_layers,
        max_context_len=CONTEXT,
        max_batch_size=1,
        rope_cache_len=64,
    )
    yield gen
    gen.teardown()


def _prompt_ids(gen, text: str) -> list[int]:
    rendered = gen.tokenizer.apply_chat_template(
        [{"role": "user", "content": text}], add_generation_prompt=True, tokenize=False
    )
    return gen.tokenizer(rendered, add_special_tokens=False)["input_ids"]


# --- the generator contract ---------------------------------------------------


def test_generator_implements_the_contract(generator):
    from models.common.readiness_check.contract import Generator

    assert isinstance(generator, Generator)
    assert isinstance(generator, Qwen3CoderGenerator)
    assert generator.tokenizer is not None
    import inspect

    # The teacher-forcing runner requires an explicit keyword, not **kwargs.
    assert "enable_trace" in inspect.signature(generator.generate).parameters


# --- split sampling, token feedback, position coherence -----------------------


def test_split_sampling_feeds_its_own_token_back_on_device(generator):
    """Step N's sampled token *is* step N+1's token input, with no host copy."""
    generator.reset()
    prompt = _prompt_ids(generator, "List three prime numbers.")
    kv_cache = generator._ensure_kv_cache()
    page_table = generator.make_page_table([len(prompt) + 8])
    sampled = generator.prefill_forward(
        torch.tensor([prompt]),
        page_table=page_table,
        kv_cache=kv_cache,
        prompt_lens=[len(prompt)],
        sampling_mode="device",
    )
    first = int(generator._sampled_to_torch(sampled)[0].item())

    def read(tensor):
        return int(ttnn.to_torch(ttnn.get_device_tensors(tensor)[0]).reshape(-1)[0].item())

    # Installing the trace also performs the first replay, so the prefill token
    # has already been consumed by the time anything can be read back.
    host_copies_before = generator.trace_stats["token_host_copies"]
    generator.decode_forward(
        None,
        torch.tensor([len(prompt)]),
        page_table=page_table,
        kv_cache=kv_cache,
        sampling_mode="device",
        enable_trace=True,
        active_batch=1,
    )
    token_in, current_pos, rotary_pos, _ = generator._trace_inputs

    observed = [read(token_in)]
    positions = [(read(current_pos), read(rotary_pos))]
    # The sampler wrote through tt_out_tok, so the persistent decode token input
    # already holds what the sampling trace produced -- same value, same tensor.
    assert read(generator._trace_sampled) == observed[0]

    for _ in range(3):
        generator.decode_forward(
            None, None, page_table=None, kv_cache=kv_cache, sampling_mode="device", enable_trace=True
        )
        observed.append(read(token_in))
        positions.append((read(current_pos), read(rotary_pos)))
        assert read(generator._trace_sampled) == observed[-1]

    # Positions were advanced on device by the trace itself, one per replay,
    # starting from the prompt length, and cache position and rotary position
    # stayed in lockstep.
    assert positions[0] == (len(prompt) + 1, len(prompt) + 1), positions
    for step in range(len(positions) - 1):
        assert positions[step + 1][0] == positions[step][0] + 1, positions
        assert positions[step + 1][1] == positions[step][1] + 1, positions

    # Nothing was written to the token input from the host across any of it.
    assert generator.trace_stats["token_host_copies"] == host_copies_before

    # And the tokens observed on device are exactly what the public generator
    # returns for the same prompt: [prefill sample, then one per replay].
    generator.reset()
    produced = generator.generate(prompt, 1 + len(observed), enable_trace=True, sampling_mode="device")
    assert produced == [first] + observed, (produced, first, observed)


def test_steady_state_decode_does_no_host_work(generator):
    """Only ``replays`` may move between two steady-state tokens."""
    generator.reset()
    prompt = _prompt_ids(generator, "Say hello.")
    generator.generate(prompt, 4, enable_trace=True, sampling_mode="device")
    before = dict(generator.trace_stats)
    generator.decode_forward(
        None, None, page_table=None, kv_cache=generator._kv_cache, sampling_mode="device", enable_trace=True
    )
    after = dict(generator.trace_stats)
    moved = {k: (before[k], after[k]) for k in before if before[k] != after[k]}
    assert moved == {"replays": (before["replays"], before["replays"] + 1)}, moved


def test_unchanged_page_table_costs_no_host_copy(generator):
    """A page table that has not changed must not be re-uploaded."""
    generator.reset()
    prompt = _prompt_ids(generator, "Say hello.")
    generator.generate(prompt, 3, enable_trace=True, sampling_mode="device")
    page_table = generator._trace_page_table_snapshot.clone()
    before = generator.trace_stats["page_table_host_copies"]
    generator._refresh_persistent_page_table(page_table, generator._trace_kv_cache, active_batch=1)
    assert generator.trace_stats["page_table_host_copies"] == before

    changed = page_table.clone()
    changed[0, -1] = 0 if changed[0, -1] != 0 else 1
    generator._refresh_persistent_page_table(changed, generator._trace_kv_cache, active_batch=1)
    assert generator.trace_stats["page_table_host_copies"] == before + 1


def test_device_split_sampling_matches_host_argmax(generator):
    """Greedy split sampling is semantically greedy, not merely close."""
    generator.reset()
    prompt = _prompt_ids(generator, "The capital of France is")
    device_tokens = generator.generate(prompt, 6, enable_trace=True, sampling_mode="device")
    generator.reset()
    host_tokens = generator.generate(prompt, 6, sampling_mode="host")
    assert device_tokens == host_tokens, (device_tokens, host_tokens)


def test_force_argmax_matches_split_sampling(generator):
    """The rejected alternative gives the same token as the shipped one."""
    generator.reset()
    prompt = _prompt_ids(generator, "2 + 2 =")
    kv_cache = generator._ensure_kv_cache()
    page_table = generator.make_page_table([len(prompt) + 1])
    split = generator.prefill_forward(
        torch.tensor([prompt]),
        page_table=page_table,
        kv_cache=kv_cache,
        prompt_lens=[len(prompt)],
        sampling_mode="device",
    )
    split_token = int(generator._sampled_to_torch(split)[0].item())

    generator.reset()
    logits = generator.prefill_forward(
        torch.tensor([prompt]),
        page_table=page_table,
        kv_cache=kv_cache,
        prompt_lens=[len(prompt)],
        sampling_mode="host",
    )
    assert split_token == int(logits[0, 0].argmax().item())


# --- rotary capacity on the low-level API -------------------------------------


def _low_level_decode(gen, prompt, steps, *, decode_horizon=None):
    """Drive ``prefill_forward``/``decode_forward`` exactly as the docstring says.

    This is the surface a serving adapter drives, and the only one that can
    decode past the rotary table: ``generate`` sizes for its own horizon.
    """
    gen.reset()
    kv_cache = gen._ensure_kv_cache()
    page_table = gen.make_page_table([len(prompt) + steps + 1])
    sampled = gen.prefill_forward(
        torch.tensor([prompt]),
        page_table=page_table,
        kv_cache=kv_cache,
        prompt_lens=[len(prompt)],
        sampling_mode="device",
    )
    tokens = [int(gen._sampled_to_torch(sampled)[0].item())]
    for step in range(steps):
        initial = step == 0
        sampled = gen.decode_forward(
            None,
            torch.tensor([len(prompt)]) if initial else None,
            page_table=page_table if initial else None,
            kv_cache=kv_cache,
            sampling_mode="device",
            enable_trace=True,
            active_batch=1,
            **({"decode_horizon": decode_horizon} if initial and decode_horizon is not None else {}),
        )
        tokens.append(int(gen._sampled_to_torch(sampled)[0].item()))
    return tokens


def test_decode_past_the_rope_cache_length_through_the_low_level_api(small_rope_generator, expect_error):
    """Walking off the cos/sin table must raise, and must be preventable.

    The traced loop advances ``rotary_position`` with ``ttnn.plus_one`` and
    nothing on device clamps it, so an out-of-range ``ttnn.embedding`` gather
    would rotate at a wrong position and return a plausible-looking token. The
    contract advertises 262144; the tables default to 8192.
    """
    gen = small_rope_generator
    rope_len = gen.model.rope_cache_len
    prompt = _prompt_ids(gen, "Count upwards.")
    assert len(prompt) < rope_len, (len(prompt), rope_len)
    steps = rope_len - len(prompt) + 6  # comfortably past the table

    # 1. Undeclared horizon: the run must stop rather than silently gather out
    #    of range. The message has to name the fix.
    with expect_error(RuntimeError, "rotary table"):
        _low_level_decode(gen, prompt, steps)
    assert gen.model.rope_cache_len == rope_len, "the failing path must not have grown the tables"

    # 2. Declared horizon: the same run completes, and the tables grew.
    horizon = len(prompt) + steps
    declared = _low_level_decode(gen, prompt, steps, decode_horizon=horizon)
    assert len(declared) == steps + 1
    assert gen.model.rope_cache_len > rope_len
    assert gen.model.rope_cache_len >= horizon

    # 3. And it is the *right* answer: identical to what the high-level
    #    ``generate``, which sizes its own rotary horizon, produces.
    gen.reset()
    reference = gen.generate(prompt, steps + 1, enable_trace=True, sampling_mode="device")
    assert declared == reference, (declared, reference)


def test_eager_decode_grows_the_rope_tables_for_its_position(small_rope_generator):
    """The eager/host branch gathers cos/sin too, and holds no trace to protect."""
    gen = small_rope_generator
    gen.reset()
    prompt_len = 8
    position = gen.model.rope_cache_len + 5
    kv_cache = gen._ensure_kv_cache()
    page_table = gen.make_page_table([position + 1])
    gen.prefill_forward(
        torch.arange(1000, 1000 + prompt_len, dtype=torch.long).unsqueeze(0),
        page_table=page_table,
        kv_cache=kv_cache,
        prompt_lens=[prompt_len],
        sampling_mode="host",
    )
    logits = gen.decode_forward(
        torch.tensor([[42]]),
        torch.tensor([position]),
        page_table=page_table,
        kv_cache=kv_cache,
        sampling_mode="host",
        enable_trace=False,
    )
    assert torch.isfinite(logits).all()
    assert gen.model.rope_cache_len > position


def test_decode_beyond_the_advertised_context_is_refused(generator, expect_error):
    with expect_error(ValueError, "exceeds the supported context"):
        generator.decode_forward(
            None,
            torch.tensor([generator.model.max_cache_len + 1]),
            page_table=generator.make_page_table([8]),
            kv_cache=generator._ensure_kv_cache(),
            sampling_mode="device",
            enable_trace=True,
            active_batch=1,
        )


# --- top-k / top-p sampling ---------------------------------------------------


def test_top_k_top_p_sampling_runs_through_a_traced_generate(generator):
    """The stochastic route is exercised, not merely reachable.

    ``README.md`` calls the top-k/top-p route "a live code path, not a promise";
    this is the assertion behind that sentence. It drives ``sample_split``
    through a captured trace and checks the generator really switched strategy.
    """
    gen = generator
    gen.reset()
    prompt = _prompt_ids(gen, "Name a colour.")
    calls = {"split": 0, "argmax": 0}
    real_split = gen.model.sample_split
    real_argmax = gen.model.sample_greedy_argmax

    def counting_split(*a, **k):
        calls["split"] += 1
        return real_split(*a, **k)

    def counting_argmax(*a, **k):
        calls["argmax"] += 1
        return real_argmax(*a, **k)

    gen.model.sample_split = counting_split
    gen.model.sample_greedy_argmax = counting_argmax
    try:
        tokens = gen.generate(prompt, 5, enable_trace=True, sampling_mode="device", top_k=8, top_p=0.9, temperature=0.8)
    finally:
        gen.model.sample_split = real_split
        gen.model.sample_greedy_argmax = real_argmax

    assert len(tokens) == 5
    assert all(0 <= t < gen.model.vocab_size for t in tokens), tokens
    assert gen._sampling_stochastic is True
    # Every sampler dispatch on this run went to the split path, and none to
    # force-argmax -- warm-up, capture and prefill included.
    assert calls["split"] > 0 and calls["argmax"] == 0, calls
    assert gen._trace_model_id is not None and gen._trace_sampling_id is not None


def test_alternating_sampling_modes_recapture_the_traces(generator):
    """The trace-id cache is keyed by sampling mode; prove the key is honoured.

    A stale trace served across a greedy/stochastic flip would silently sample
    with the wrong strategy, which no accuracy gate on this stage would catch.
    """
    gen = generator
    gen.reset()
    prompt = _prompt_ids(gen, "The capital of France is")

    greedy_first = gen.generate(prompt, 4, enable_trace=True, sampling_mode="device", top_k=1)
    assert gen._sampling_stochastic is False
    releases_before = gen.trace_stats["releases"]
    captures_before = gen.trace_stats["captures"]

    gen.reset()
    stochastic = gen.generate(
        prompt, 4, enable_trace=True, sampling_mode="device", top_k=16, top_p=0.95, temperature=1.0
    )
    assert gen._sampling_stochastic is True
    assert len(stochastic) == 4
    assert gen.trace_stats["releases"] > releases_before, gen.trace_stats
    assert gen.trace_stats["captures"] > captures_before, gen.trace_stats

    gen.reset()
    greedy_again = gen.generate(prompt, 4, enable_trace=True, sampling_mode="device", top_k=1)
    assert gen._sampling_stochastic is False
    # Flipping back must restore the greedy strategy exactly, not leave the
    # stochastic trace installed.
    assert greedy_again == greedy_first, (greedy_first, greedy_again)


def test_temperature_zero_is_spelled_as_greedy(generator):
    """A serving stack spells greedy ``temperature=0``; it must not go stochastic."""
    gen = generator
    gen.reset()
    prompt = _prompt_ids(gen, "The capital of France is")
    greedy = gen.generate(prompt, 4, enable_trace=True, sampling_mode="device", top_k=1)
    gen.reset()
    as_temp_zero = gen.generate(prompt, 4, enable_trace=True, sampling_mode="device", top_k=0, temperature=0.0)
    assert gen._sampling_stochastic is False
    assert as_temp_zero == greedy, (greedy, as_temp_zero)


def test_set_sampling_params_releases_traces_only_on_a_mode_flip(generator):
    """Changing k/p within one mode must not cost a recapture."""
    gen = generator
    gen.reset()
    prompt = _prompt_ids(gen, "Say hello.")
    gen.generate(prompt, 3, enable_trace=True, sampling_mode="device", top_k=8, top_p=0.9)
    assert gen._sampling_stochastic is True
    releases = gen.trace_stats["releases"]

    gen.set_sampling_params(top_k=16, top_p=0.5, temperature=0.7, active_batch=1)
    assert gen._sampling_stochastic is True
    assert gen.trace_stats["releases"] == releases, "a k/p change inside one mode recaptured"
    assert gen._trace_model_id is not None

    gen.set_sampling_params(top_k=1, top_p=0.0, temperature=1.0, active_batch=1)
    assert gen._sampling_stochastic is False
    assert gen.trace_stats["releases"] == releases + 1, "the greedy flip did not recapture"


# --- prompt lengths -----------------------------------------------------------


@pytest.mark.parametrize("prompt_len", [1, 31, 33, 100, 127, 128, 129, 257, 1000])
def test_non_aligned_prompt_lengths(generator, prompt_len):
    """Every length up to the context, aligned or not, through the public API."""
    generator.reset()
    tokens = torch.arange(1000, 1000 + prompt_len, dtype=torch.long).unsqueeze(0)
    kv_cache = generator._ensure_kv_cache()
    page_table = generator.make_page_table([prompt_len + 1])
    logits = generator.prefill_forward(
        tokens,
        page_table=page_table,
        kv_cache=kv_cache,
        prompt_lens=[prompt_len],
        sampling_mode="host",
    )
    assert tuple(logits.shape) == (1, 1, generator.model.vocab_size)
    assert torch.isfinite(logits).all()


def test_return_all_logits_is_sliced_to_the_logical_length(generator):
    generator.reset()
    prompt_len = 37
    tokens = torch.arange(500, 500 + prompt_len, dtype=torch.long).unsqueeze(0)
    kv_cache = generator._ensure_kv_cache()
    logits = generator.prefill_forward(
        tokens,
        page_table=generator.make_page_table([prompt_len]),
        kv_cache=kv_cache,
        prompt_lens=[prompt_len],
        return_all_logits=True,
        sampling_mode="host",
    )
    assert tuple(logits.shape) == (1, prompt_len, generator.model.vocab_size)
    assert torch.isfinite(logits).all()


# --- batch, fixed slots, inactive rows ---------------------------------------


def test_mixed_length_batch_prefill_and_decode(batch_generator):
    """Four users, four different prompt lengths, disjoint physical pages."""
    gen = batch_generator
    gen.reset()
    lengths = [7, 33, 64, 129]
    width = max(lengths)
    tokens = torch.zeros(len(lengths), width, dtype=torch.long)
    for user, length in enumerate(lengths):
        tokens[user, :length] = torch.arange(100 + user * 50, 100 + user * 50 + length)
    kv_cache = gen._ensure_kv_cache()
    page_table = gen.make_page_table([length + 4 for length in lengths])
    logits = gen.prefill_forward(
        tokens, page_table=page_table, kv_cache=kv_cache, prompt_lens=lengths, sampling_mode="host"
    )
    assert tuple(logits.shape) == (len(lengths), 1, gen.model.vocab_size)
    assert torch.isfinite(logits).all()

    predicted = logits[:, 0].argmax(dim=-1)
    decoded = gen.decode_forward(
        predicted.reshape(-1, 1),
        torch.tensor(lengths),
        page_table=page_table,
        kv_cache=kv_cache,
        sampling_mode="host",
        enable_trace=False,
    )
    assert tuple(decoded.shape) == (len(lengths), gen.model.vocab_size)
    assert torch.isfinite(decoded).all()


def test_inactive_rows_are_expressible(batch_generator):
    """A negative position marks an inactive slot and must not be validated."""
    gen = batch_generator
    gen.reset()
    lengths = [16, 16]
    tokens = torch.arange(2 * 16, dtype=torch.long).reshape(2, 16)
    kv_cache = gen._ensure_kv_cache()
    page_table = gen.make_page_table([20, 20])
    gen.prefill_forward(tokens, page_table=page_table, kv_cache=kv_cache, prompt_lens=lengths, sampling_mode="host")
    positions = torch.tensor([16, -1])
    gen._validate_page_coverage(gen._normalise_page_table(page_table, 2), positions, 2)


def test_page_table_must_map_disjoint_pages(batch_generator, expect_error):
    gen = batch_generator
    table = gen.make_page_table([64, 64])
    table[1, :] = table[0, :]
    with expect_error(ValueError, "must map disjoint physical cache pages"):
        gen._validate_page_coverage(gen._normalise_page_table(table, 2), torch.tensor([63, 63]), 2)


def test_sdpa_rounded_page_count_covers_the_read_window(generator):
    """The allocation must match the kernel's rounded read, not the token count."""
    for tokens, expected in ((1, 1), (32, 1), (33, 2), (96, 4), (256, 8), (257, 16), (320, 16), (513, 24)):
        assert generator._sdpa_rounded_page_count(tokens) == expected, tokens
    for tokens in (1, 33, 100, 257, 1000, 4095):
        assert generator._sdpa_rounded_page_count(tokens) >= math.ceil(tokens / generator.page_block_size)


def test_distributed_argmax_is_exact_at_batch_above_one(batch_generator):
    """The live-row slice must be right at every batch, not only at batch 1.

    Stage 06 made ``_WatcherCleanSampling1D._sample_argmax`` **batch-dependent**:
    it slices the 32-slot logit tile down to ``_dist_active_rows`` before the
    per-die ``ttnn.argmax`` and pads the result back. Every other device-sampling
    test in this file uses the ``max_batch_size=1`` fixture and the
    ``max_batch_size=4`` fixture only ever samples on the host, so the branch
    that the slice introduced was uncovered at batch > 1 -- which is exactly
    where an off-by-one in the slice or the pad would live.

    This drives the sampler directly with crafted logits, because the property
    is about the reduction and not about what the model predicts:

    * every **live** row returns the host argmax of the same bf16 logits;
    * every **padding** row returns token 0, which is the value the shipped
      32-row reduction produces for a zero-logit row and the value the pad
      writes back, so the 32-slot buffer is unchanged slot for slot;
    * the caller's ``tt_out_tok`` object survives -- the traced decode loop
      feeds that exact tensor back, so a new tensor would break feedback
      silently.

    The all-negative leg matters on its own: it is the case where "the padding
    rows are zero" stops being harmless, because a zero padding row would beat
    every live row if the slice were not there.
    """
    gen = batch_generator
    model = gen.model
    sampler = model.sampler
    mesh = gen.mesh_device
    dies = mesh.get_num_devices()
    local_vocab = model.vocab_size // dies

    assert sampler._dist_active_rows == model.max_batch_size > 1, (
        f"this test is only meaningful when the sampler is batched: "
        f"_dist_active_rows={sampler._dist_active_rows}, max_batch_size={model.max_batch_size}"
    )
    sampler.load_device_buffers()
    assert getattr(sampler, "_dist_die_offset", None) is not None, "the distributed path is not active"
    assert sampler._dist_local_vocab == local_vocab

    slots = 32
    active = sampler._dist_active_rows
    torch.manual_seed(0)
    legs = {
        # random logits: the ordinary case, and the winner lands on a different
        # die for different rows
        "random": torch.randn(1, 1, slots, model.vocab_size),
        # every live logit strictly negative: a padding row's exact 0.0 would win
        # every row if the live-row slice were not doing its job
        "all_negative": -1.0 - torch.rand(1, 1, slots, model.vocab_size),
    }
    for name, logits in legs.items():
        # bf16 on the way in, so the host reference sees the same values the
        # device compares -- otherwise near-ties round differently.
        logits = logits.to(torch.bfloat16).to(torch.float32)
        expected = logits[0, 0, :active].argmax(dim=-1).tolist()

        device_logits = ttnn.from_torch(
            logits,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-1),
        )
        out_tok = ttnn.from_torch(
            torch.full((1, 1, 1, slots), 12345, dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        returned, logprobs = sampler._sample_argmax(device_logits, out_tok)
        assert logprobs is None
        assert returned is out_tok, f"{name}: the sampler returned a new tensor, breaking token feedback"

        tokens = [int(v) for v in ttnn.to_torch(ttnn.get_device_tensors(returned)[0]).reshape(-1)[:slots].tolist()]
        assert tokens[:active] == expected, f"{name}: live rows {tokens[:active]} != host argmax {expected}"
        assert tokens[active:] == [0] * (slots - active), f"{name}: padding rows are {tokens[active:]}, not 0"
        ttnn.deallocate(device_logits)
        ttnn.deallocate(out_tok)


# --- cache ownership, reset, determinism -------------------------------------


def test_caller_owned_cache_is_used_verbatim(generator):
    """A caller-allocated cache must be honoured, not silently replaced."""
    generator.reset()
    caller_cache = generator.model.allocate_kv_cache(max_cache_len=1024, num_blocks=64)
    prompt = _prompt_ids(generator, "Hi.")
    generator.prefill_forward(
        torch.tensor([prompt]),
        page_table=generator.make_page_table([len(prompt)]),
        kv_cache=caller_cache,
        prompt_lens=[len(prompt)],
        sampling_mode="host",
    )
    written = ttnn.to_torch(ttnn.get_device_tensors(caller_cache[0].k)[0])
    assert written.abs().sum() > 0, "prefill did not write into the caller's cache"
    for cache in caller_cache:
        ttnn.deallocate(cache.k, True)
        ttnn.deallocate(cache.v, True)


def test_reset_makes_generation_reproducible(generator):
    prompt = _prompt_ids(generator, "Count to five.")
    generator.reset()
    first = generator.generate(prompt, 6, enable_trace=True, sampling_mode="device")
    generator.reset()
    second = generator.generate(prompt, 6, enable_trace=True, sampling_mode="device")
    assert first == second, (first, second)


def test_reset_zeroes_the_cache(generator):
    generator.reset()
    prompt = _prompt_ids(generator, "Hello there.")
    generator.generate(prompt, 2, enable_trace=True, sampling_mode="device")
    generator.reset()
    for cache in generator._kv_cache:
        assert ttnn.to_torch(ttnn.get_device_tensors(cache.k)[0]).abs().sum() == 0
        assert ttnn.to_torch(ttnn.get_device_tensors(cache.v)[0]).abs().sum() == 0


def test_prefill_logits_are_deterministic_across_runs(generator):
    generator.reset()
    prompt = _prompt_ids(generator, "Deterministic?")
    args = dict(
        page_table=generator.make_page_table([len(prompt)]),
        kv_cache=generator._ensure_kv_cache(),
        prompt_lens=[len(prompt)],
        sampling_mode="host",
    )
    first = generator.prefill_forward(torch.tensor([prompt]), **args)
    generator.reset()
    second = generator.prefill_forward(torch.tensor([prompt]), **args)
    assert torch.equal(first, second)


# --- the carried-forward decoder contract ------------------------------------


def test_runtime_fallback_audit_is_clean(generator):
    audit = generator.model.runtime_fallback_audit()
    assert audit["dram_sharded_taken"] is True
    # Stage 07 retuned both to their full-K ceilings (+2.83% decode, no
    # accuracy change); see doc/datatype_sweep/README.md.
    assert audit["gate_up_in0_block_w"] == 64
    assert audit["down_in0_block_w"] == 24
    assert audit["expert_intermediate_buffer"] == "L1"
    assert audit["local_heads"] == (8, 1)
    assert audit["local_experts"] == 32
    assert audit["norm_shard_feeds_qkv_directly"] is True
    assert audit["decode_ccl_buffers_persistent"] is True
    assert audit["host_logit_readback_on_token_out_path"] is False
    assert audit["host_argmax_on_token_out_path"] is False
    assert audit["vocab_padding"] == 0
    assert audit["kv_cache_dtype"] == "bfloat16"
    assert audit["collective_topology"] == "Topology.Ring"
    assert (audit["prefill_num_links"], audit["decode_num_links"]) == (2, 1)


def test_inter_layer_residual_contract_is_preserved(generator):
    """A layer's output must be indistinguishable from its input as a tensor.

    This is the stage-04 contract restated at the full-model boundary: if it
    holds, 48 layers stack with no conversion, which is what
    ``decode_hidden``'s bare ``for`` loop assumes.
    """
    model = generator.model
    generator.reset()
    tokens = ttnn.from_torch(
        torch.zeros((1, 1, 1, 32), dtype=torch.int32),
        device=model.mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(model.mesh_device),
    )
    hidden = model.embed_decode(tokens)
    assert tuple(hidden.shape) == (1, 1, model.max_batch_size, model.hidden_size)
    assert hidden.dtype == ttnn.bfloat16
    assert hidden.layout == ttnn.TILE_LAYOUT
    assert hidden.memory_config() == ttnn.DRAM_MEMORY_CONFIG

    current_pos = ttnn.from_torch(
        torch.zeros(model.max_batch_size, dtype=torch.int32),
        device=model.mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(model.mesh_device),
    )
    rotary = ttnn.from_torch(
        torch.zeros((1, model.max_batch_size), dtype=torch.int32),
        device=model.mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(model.mesh_device),
    )
    caches = generator._ensure_kv_cache()
    model.bind_page_table(caches, generator._prefill_page_table)
    cos, sin = model.rope_decode_tables(rotary)
    out = MC.decoder_layer_decode_multichip(
        hidden,
        model.layers[0],
        model.config,
        model.ctx,
        cos,
        sin,
        caches[0],
        current_pos,
        0,
        rope=model._rope_decode,
    )
    assert tuple(out.shape) == tuple(hidden.shape)
    assert out.dtype == hidden.dtype
    assert out.layout == hidden.layout
    assert out.memory_config() == hidden.memory_config()
