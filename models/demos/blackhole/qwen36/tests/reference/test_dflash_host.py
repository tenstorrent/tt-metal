# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Host tests for the Qwen3.6-27B-DFlash reference. No device, no ttnn.

    # structural correctness — seconds, no checkpoint needed
    pytest models/demos/blackhole/qwen36/tests/reference/test_dflash_host.py -sv

    # + the real 3.5 GB drafter checkpoint
    DFLASH_HF_MODEL=z-lab/Qwen3.6-27B-DFlash \
      pytest models/demos/blackhole/qwen36/tests/reference/test_dflash_host.py -sv

    # + the real 27B target end-to-end (minutes on CPU)
    HF_MODEL=Qwen/Qwen3.6-27B DFLASH_RUN_TARGET=1 \
      pytest models/demos/blackhole/qwen36/tests/reference/test_dflash_host.py -sv

The load-bearing test is :func:`test_speculation_matches_autoregressive_greedy`: with greedy
decoding, speculation must reproduce the target's own token stream exactly, whatever the drafter
proposes. Everything else is scaffolding around that claim.
"""

from __future__ import annotations

import pytest
import torch
from loguru import logger
from transformers import DynamicCache

from models.demos.blackhole.qwen36.reference.dflash.dflash import extract_context_feature
from models.demos.blackhole.qwen36.reference.dflash.generate import dflash_generate
from models.demos.blackhole.qwen36.reference.dflash.loader import check_drafter_matches_target
from models.demos.blackhole.qwen36.reference.dflash.targets import HFTarget

from .conftest import TINY_BLOCK, TINY_HIDDEN, TINY_TAPS, TINY_VOCAB


def _prompt(length: int = 12, *, vocab: int = TINY_VOCAB, seed: int = 7) -> torch.Tensor:
    # Stay clear of the mask token (vocab - 1) so the prompt is never confused with a masked slot.
    return torch.randint(0, vocab - 1, (1, length), generator=torch.Generator().manual_seed(seed))


# ------------------------------------------------------------------------------------------------
# Drafter shape / plumbing
# ------------------------------------------------------------------------------------------------


def test_drafter_forward_shapes(tiny_drafter):
    """One drafter forward turns a whole block of mask slots into a block of hidden states."""
    ctx_len, block = 10, TINY_BLOCK
    target_hidden = torch.randn(1, ctx_len, len(TINY_TAPS) * TINY_HIDDEN)
    noise = torch.randn(1, block, TINY_HIDDEN)

    out = tiny_drafter(
        target_hidden=target_hidden,
        noise_embedding=noise,
        # position_ids spans the K/V axis (context + block), not just the queried block.
        position_ids=torch.arange(ctx_len + block)[None],
    )
    assert out.shape == (1, block, TINY_HIDDEN)


def test_drafter_reads_taps_through_fc(tiny_drafter):
    """``fc`` consumes the concatenated taps, so its input width pins the number of taps."""
    assert tiny_drafter.fc.in_features == len(TINY_TAPS) * TINY_HIDDEN
    assert tiny_drafter.fc.out_features == TINY_HIDDEN
    assert list(tiny_drafter.target_layer_ids) == list(TINY_TAPS)


def test_drafter_block_is_bidirectional(tiny_drafter):
    """The block is drafted in parallel: a later slot's content changes an earlier slot's output.

    This is what separates DFlash from an autoregressive drafter, and it is carried by the single
    ``full_attention`` layer being non-causal. If that layer ever became causal this test fails.
    """
    ctx_len, block = 6, TINY_BLOCK
    target_hidden = torch.randn(1, ctx_len, len(TINY_TAPS) * TINY_HIDDEN)
    noise = torch.randn(1, block, TINY_HIDDEN)
    position_ids = torch.arange(ctx_len + block)[None]

    base = tiny_drafter(target_hidden=target_hidden, noise_embedding=noise, position_ids=position_ids)
    perturbed = noise.clone()
    perturbed[:, -1] += 10.0  # only the LAST slot changes
    other = tiny_drafter(target_hidden=target_hidden, noise_embedding=perturbed, position_ids=position_ids)

    assert not torch.allclose(base[:, 0], other[:, 0]), "first slot ignored the last slot — block is causal"


def test_drafter_cache_keeps_context_and_drops_the_block(tiny_drafter):
    """A step must keep the context rows it appended and drop only the noise block.

    This is the drafter's whole state management. Keeping too much (the block) feeds masked guesses
    back as context; keeping too little (an earlier version restored to the pre-forward length)
    leaves the drafter permanently context-starved, which fails nothing and just drafts worse.
    """
    from models.demos.blackhole.qwen36.reference.dflash.generate import truncate_kv

    ctx_len, block = 6, TINY_BLOCK
    cache = DynamicCache(config=tiny_drafter.config)
    start = ctx_len  # the block sits at [ctx_len, ctx_len + block)
    tiny_drafter(
        target_hidden=torch.randn(1, ctx_len, len(TINY_TAPS) * TINY_HIDDEN),
        noise_embedding=torch.randn(1, block, TINY_HIDDEN),
        position_ids=torch.arange(ctx_len + block)[None],
        past_key_values=cache,
        use_cache=True,
    )
    assert cache.get_seq_length() == ctx_len + block

    truncate_kv(cache, start)
    assert cache.get_seq_length() == start, "context rows must survive, block rows must not"

    # A second step appends its own context on top, so the history accumulates.
    tiny_drafter(
        target_hidden=torch.randn(1, 2, len(TINY_TAPS) * TINY_HIDDEN),
        noise_embedding=torch.randn(1, block, TINY_HIDDEN),
        position_ids=torch.arange(start, start + 2 + block)[None],
        past_key_values=cache,
        use_cache=True,
    )
    truncate_kv(cache, start + 2)
    assert cache.get_seq_length() == start + 2


def test_truncate_kv_refuses_past_the_window(tiny_drafter, expect_error):
    """Past a sliding layer's window the older rows are evicted, so tail truncation is not exact.

    ``truncate_kv`` must say so rather than silently leave the layer short with a
    ``cumulative_length`` that still claims the evicted rows. This is the documented ceiling on how
    long a context the drafter can carry through this helper.
    """
    from models.demos.blackhole.qwen36.reference.dflash.generate import truncate_kv

    window = tiny_drafter.config.sliding_window
    cache = DynamicCache(config=tiny_drafter.config)
    over = window + TINY_BLOCK
    tiny_drafter(
        target_hidden=torch.randn(1, over, len(TINY_TAPS) * TINY_HIDDEN),
        noise_embedding=torch.randn(1, TINY_BLOCK, TINY_HIDDEN),
        position_ids=torch.arange(over + TINY_BLOCK)[None],
        past_key_values=cache,
        use_cache=True,
    )
    with expect_error(AssertionError, "past its .* window"):
        truncate_kv(cache, over)


# ------------------------------------------------------------------------------------------------
# Gated DeltaNet rollback — the hazard this reference exists to get right
# ------------------------------------------------------------------------------------------------


def test_linear_attention_crop_is_a_noop_tripwire(tiny_target):
    """Tripwire: transformers' ``crop`` does not roll back Gated DeltaNet state.

    ``LinearAttentionCacheLayerMixin.crop`` is documented as a no-op, so after cropping a rejected
    block the 6 GDN layers here still carry the rejected tokens even though ``get_seq_length()``
    reports the cropped length. The generate loop compensates with snapshot/restore + replay.

    If a transformers upgrade ever makes ``crop`` roll the recurrent state back properly, this test
    fails — and that is the signal to delete the replay path in ``generate.py``.
    """
    prompt = _prompt(16)
    junk = torch.randint(0, TINY_VOCAB - 1, (1, 8))
    tail = _prompt(4, seed=99)

    def run(model, seq, cache, pos0):
        return model(
            seq, position_ids=torch.arange(pos0, pos0 + seq.shape[1])[None], past_key_values=cache, use_cache=True
        )

    with torch.inference_mode():
        clean = DynamicCache(config=tiny_target.config)
        run(tiny_target, prompt, clean, 0)
        golden = run(tiny_target, tail, clean, 16).logits

        polluted = DynamicCache(config=tiny_target.config)
        run(tiny_target, prompt, polluted, 0)
        run(tiny_target, junk, polluted, 16)
        polluted.crop(16)
        assert polluted.get_seq_length() == 16, "attention layers should have been truncated"
        after_crop = run(tiny_target, tail, polluted, 16).logits

    assert not torch.allclose(golden, after_crop, atol=1e-3), (
        "`crop` now appears to roll back the GDN recurrent state — re-check whether generate.py "
        "still needs its snapshot/restore + replay path"
    )


def test_snapshot_restore_is_exact(tiny_target):
    """Snapshot/restore + replay must reproduce the un-speculated cache bit for bit."""
    prompt = _prompt(16)
    junk = torch.randint(0, TINY_VOCAB - 1, (1, 8))
    tail = _prompt(4, seed=99)

    def run(seq, cache, pos0):
        return tiny_target(
            seq, position_ids=torch.arange(pos0, pos0 + seq.shape[1])[None], past_key_values=cache, use_cache=True
        )

    with torch.inference_mode():
        clean = DynamicCache(config=tiny_target.config)
        run(prompt, clean, 0)
        golden = run(tail, clean, 16).logits

        # The same rollback the loop performs, through the target abstraction.
        host_target = HFTarget(tiny_target, TINY_TAPS)
        host_target.reset()
        rolled_back = host_target.cache
        run(prompt, rolled_back, 0)
        saved = host_target.snapshot()
        run(junk, rolled_back, 16)  # the rejected block
        host_target.restore(saved, 16)
        recovered = run(tail, rolled_back, 16).logits

    assert torch.equal(golden, recovered), "rollback did not restore the pre-block state exactly"


# ------------------------------------------------------------------------------------------------
# End-to-end equivalence
# ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("block_size", [2, TINY_BLOCK, 8], ids=lambda b: f"block{b}")
def test_speculation_matches_autoregressive_greedy(tiny_target, tiny_drafter, block_size):
    """Greedy speculation must emit exactly the tokens plain decoding would.

    The drafter here is random, so almost every draft is rejected and almost every step takes the
    rollback path — which is precisely the path that must not corrupt the GDN state. A mismatch
    here is the signature of a botched rollback, not of a bad drafter.
    """
    prompt = _prompt(12)
    baseline = dflash_generate(tiny_drafter, tiny_target, prompt, max_new_tokens=24, block_size=1)
    speculative = dflash_generate(tiny_drafter, tiny_target, prompt, max_new_tokens=24, block_size=block_size)

    assert torch.equal(baseline, speculative), (
        f"block_size={block_size} diverged from autoregressive decoding:\n"
        f"  baseline    {baseline.tolist()}\n  speculative {speculative.tolist()}"
    )


def test_stats_and_stop_tokens(tiny_target, tiny_drafter):
    """Stats bookkeeping adds up, and a stop token truncates the output at its first occurrence."""
    prompt = _prompt(12)
    stats = dflash_generate(
        tiny_drafter, tiny_target, prompt, max_new_tokens=24, block_size=TINY_BLOCK, return_stats=True
    )
    assert stats.output_ids.shape[1] == stats.num_input_tokens + stats.num_output_tokens
    # The +1 is prefill's own token, emitted before the first speculative step.
    assert 1 + sum(stats.acceptance_lengths) == stats.num_output_tokens
    assert all(1 <= a <= TINY_BLOCK for a in stats.acceptance_lengths)
    assert 1.0 <= stats.mean_acceptance_length <= TINY_BLOCK

    # Stop on whatever the unconstrained run produced first, and the run must end right there.
    first_generated = int(stats.output_ids[0, stats.num_input_tokens])
    stopped = dflash_generate(
        tiny_drafter, tiny_target, prompt, max_new_tokens=24, block_size=TINY_BLOCK, stop_token_ids=[first_generated]
    )
    assert stopped[0, -1].item() == first_generated
    assert stopped.shape[1] == prompt.shape[1] + 1


def test_sampling_runs_and_stays_in_vocab(tiny_target, tiny_drafter):
    """The rejection-sampling path runs end to end and never emits an out-of-range id."""
    torch.manual_seed(0)
    out = dflash_generate(
        tiny_drafter,
        tiny_target,
        _prompt(12),
        max_new_tokens=24,
        block_size=TINY_BLOCK,
        temperature=0.8,
        top_p=0.95,
    )
    assert out.shape[1] > 12
    assert int(out.max()) < TINY_VOCAB and int(out.min()) >= 0
    assert (out != tiny_drafter.mask_token_id).all(), "a mask slot leaked into the output"


# ------------------------------------------------------------------------------------------------
# Real checkpoints
# ------------------------------------------------------------------------------------------------


def test_real_drafter_config(real_drafter_config):
    """The parsed config matches z-lab/Qwen3.6-27B-DFlash's published architecture."""
    cfg = real_drafter_config
    assert (cfg.num_hidden_layers, cfg.hidden_size, cfg.head_dim) == (5, 5120, 128)
    assert (cfg.num_attention_heads, cfg.num_key_value_heads) == (32, 8)
    assert cfg.block_size == 16 and cfg.num_speculative_tokens == 15
    assert cfg.target_layer_ids == (1, 16, 31, 46, 61)
    assert cfg.num_target_layers == 64, "drafter must declare the 64-layer Qwen3.6-27B target"
    assert cfg.target_feature_size == 5 * 5120
    assert cfg.layer_types == ("sliding_attention",) * 4 + ("full_attention",)
    assert cfg.vocab_size == 248320 and cfg.mask_token_id == 248070


def test_real_drafter_loads_and_runs(real_drafter, real_drafter_config):
    """The real weights load with no missing/unexpected tensors and produce a finite block."""
    cfg = real_drafter_config
    assert len(real_drafter.layers) == cfg.num_hidden_layers
    assert real_drafter.fc.in_features == cfg.target_feature_size

    ctx_len, block = 8, cfg.block_size
    torch.manual_seed(0)
    out = real_drafter(
        target_hidden=torch.randn(1, ctx_len, cfg.target_feature_size),
        noise_embedding=torch.randn(1, block, cfg.hidden_size),
        position_ids=torch.arange(ctx_len + block)[None],
    )
    assert out.shape == (1, block, cfg.hidden_size)
    assert torch.isfinite(out).all()


def test_real_drafter_matches_real_target(real_drafter_config, real_target):
    """The drafter/target cross-check the generate loop depends on."""
    check_drafter_matches_target(real_drafter_config, real_target)


def test_real_end_to_end_speculation(real_drafter, real_target, real_drafter_config):
    """The whole point: real drafter + real 27B target, greedy, on host.

    Two claims at once — speculation is lossless (identical tokens to plain decoding) and it is
    useful (a trained drafter accepts more than the 1 token/step a rejected draft would give).
    """
    from transformers import AutoTokenizer

    from models.demos.blackhole.qwen36.reference.dflash.loader import resolve_target_path

    tokenizer = AutoTokenizer.from_pretrained(resolve_target_path())
    prompt = tokenizer("The capital of France is", return_tensors="pt").input_ids
    max_new_tokens = 32

    stats = dflash_generate(real_drafter, real_target, prompt, max_new_tokens=max_new_tokens, return_stats=True)
    baseline = dflash_generate(real_drafter, real_target, prompt, max_new_tokens=max_new_tokens, block_size=1)

    logger.info(f"generated: {tokenizer.decode(stats.output_ids[0, stats.num_input_tokens:])!r}")
    logger.info(
        f"acceptance: mean {stats.mean_acceptance_length:.2f} tok/step over {len(stats.acceptance_lengths)} steps "
        f"(per-step {stats.acceptance_lengths}, rollbacks {stats.num_rollbacks}, block_size {stats.block_size})"
    )

    assert torch.equal(stats.output_ids, baseline), "speculative output diverged from autoregressive decoding"
    assert stats.mean_acceptance_length > 1.0, (
        f"drafter accepted {stats.mean_acceptance_length:.2f} tok/step — no better than no speculation; "
        "check the tap layer ids and that the drafter weights actually loaded"
    )


def test_real_tap_feature_width(real_drafter, real_target, real_drafter_config):
    """The target's tapped hidden states concatenate to exactly what the drafter's ``fc`` expects."""
    prompt = torch.tensor([[1, 2, 3, 4]])
    with torch.inference_mode():
        out = real_target(prompt, output_hidden_states=True, use_cache=False)
    feature = extract_context_feature(out.hidden_states, list(real_drafter.target_layer_ids))
    assert feature.shape == (1, 4, real_drafter_config.target_feature_size)
    assert feature.shape[-1] == real_drafter.fc.in_features


def test_host_throughput(real_drafter, real_target, real_drafter_config):
    """What does speculation actually buy on host, in tokens/second?

    Same three-row shape as the device benchmark (``test_dflash_throughput.py``) so the two are
    directly comparable:

    * **HF generate** — the target's own native decode path. The host analogue of ``decode_tp``.
    * **block_size=1** — speculation disabled but still going through our loop. Isolates whatever
      the harness itself costs, which on host should be near zero (unlike the device, where this row
      still pays a 128-token bucket).
    * **block_size=16** — real speculation.

    Every variant is warmed first: the first forward pulls 54 GB of weights off disk into page
    cache, which would otherwise be charged to whichever ran first.
    """
    import time

    from transformers import AutoTokenizer

    from models.demos.blackhole.qwen36.reference.dflash.drafters import HostDrafter
    from models.demos.blackhole.qwen36.reference.dflash.loader import resolve_target_path
    from models.demos.blackhole.qwen36.reference.dflash.targets import HFTarget

    cfg = real_drafter_config
    tokenizer = AutoTokenizer.from_pretrained(resolve_target_path())
    prompt = tokenizer("The capital of France is", return_tensors="pt").input_ids
    max_new = 24

    target = HFTarget(real_target, cfg.target_layer_ids)
    drafter = HostDrafter(real_drafter, target)

    timers = {"target.forward": 0.0, "drafter.propose": 0.0}

    def _timed(name, fn):
        def wrapper(*a, **kw):
            t = time.perf_counter()
            out = fn(*a, **kw)
            timers[name] += time.perf_counter() - t
            return out

        return wrapper

    target.forward = _timed("target.forward", type(target).forward.__get__(target))
    drafter.propose = _timed("drafter.propose", type(drafter).propose.__get__(drafter))

    # ---- row 1: the target's own native decode path (the host decode_tp analogue) ----
    gen_kw = dict(max_new_tokens=max_new, do_sample=False, use_cache=True)
    real_target.generate(prompt, max_new_tokens=4, do_sample=False, use_cache=True)  # warm
    t0 = time.perf_counter()
    native = real_target.generate(prompt, **gen_kw)
    native_s = time.perf_counter() - t0
    native_new = native.shape[1] - prompt.shape[1]

    results, splits = {}, {}
    for label, block_size in (("our loop (block=1)", 1), (f"speculative (block={cfg.block_size})", None)):
        dflash_generate(drafter, target, prompt, max_new_tokens=4, block_size=block_size)  # warm
        for k in timers:
            timers[k] = 0.0
        t0 = time.perf_counter()
        stats = dflash_generate(
            drafter, target, prompt, max_new_tokens=max_new, block_size=block_size, return_stats=True
        )
        results[label] = (time.perf_counter() - t0, stats)
        splits[label] = dict(timers)

    logger.info("")
    logger.info(f"{'path':<32} {'tok/s':>8} {'s/tok':>8}   detail")
    logger.info(f"{'-' * 74}")
    logger.info(f"{'HF generate (native decode)':<32} {native_new / native_s:>8.3f} {native_s / native_new:>8.2f}")
    for label, (elapsed, stats) in results.items():
        n = stats.num_output_tokens
        detail = f"{stats.mean_acceptance_length:.2f} tok/step over {len(stats.acceptance_lengths)} steps"
        logger.info(f"{label:<32} {n / elapsed:>8.3f} {elapsed / n:>8.2f}   {detail}")
    logger.info("")
    for label, (elapsed, _) in results.items():
        parts = splits[label]
        rest = elapsed - sum(parts.values())
        detail = "  ".join(f"{k} {v:.1f}s ({100 * v / elapsed:.0f}%)" for k, v in parts.items())
        logger.info(f"  {label:<32} total {elapsed:.1f}s = {detail}  loop {rest:.1f}s ({100 * rest / elapsed:.0f}%)")

    def tps(label):
        elapsed, stats = results[label]
        return stats.num_output_tokens / elapsed

    spec_tps = tps(f"speculative (block={cfg.block_size})")
    logger.info("")
    logger.info(f"speculation vs our block=1 loop:   {spec_tps / tps('our loop (block=1)'):.2f}x")
    logger.info(f"speculation vs HF native generate: {spec_tps / (native_new / native_s):.2f}x")
    logger.info(f"our block=1 loop vs HF generate:   {tps('our loop (block=1)') / (native_new / native_s):.2f}x")

    base = results["our loop (block=1)"][1]
    spec = results[f"speculative (block={cfg.block_size})"][1]
    assert torch.equal(base.output_ids, spec.output_ids), "host speculation changed the tokens"
    # Our block=1 loop must agree with HF's own greedy decode — that is what makes it a fair baseline.
    n = min(base.output_ids.shape[1], native.shape[1])
    assert torch.equal(base.output_ids[:, :n], native[:, :n].cpu()), (
        f"our block=1 loop diverged from HF generate:\n  ours {base.output_ids[0, :n].tolist()}\n"
        f"  HF   {native[0, :n].tolist()}"
    )
