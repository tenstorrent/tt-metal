# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""DFlash speculative-decode loop, driven against either a host or a device Qwen3.6-27B target.

Per step: draft a 16-slot block in ONE drafter forward (1 confirmed anchor + 15 masked slots),
verify all 16 in ONE target forward, accept the longest matching prefix, and emit a bonus token
from the target's own distribution. Output is distributionally identical to plain autoregressive
decoding — greedy accepts only exact argmax matches, sampling uses standard rejection sampling —
so acceptance length is a pure speedup metric, never a quality knob.

The drafter always runs on host. The target is anything satisfying
:class:`~.targets.SpeculativeTarget`: :class:`~.targets.HFTarget` (host ``Qwen3_5ForCausalLM``, the
golden reference) or :class:`~.targets.TtTarget` (the ttnn ``Qwen36Model`` on a mesh). The loop
below is identical for both — that is what makes "do the device's tokens match the host's?" a
single-line assertion.

Rolling back a rejected block
-----------------------------
This is the part that is NOT a port of upstream's loop, and the part a device implementation has
to get right. Speculation must undo a rejected block: after verifying 16 slots and accepting *k*,
the target's state must go back to what it was after *k* tokens. For attention that is a KV
truncation. For Qwen3.6-27B's **48 Gated DeltaNet layers** it is not — they carry a recurrent state
and a conv state, not a per-token history.

Upstream (``z-lab/dflash``) does this with ``cache.crop()`` after ``cache.activate_past_recording()``.
Neither works against Qwen3.6-27B on transformers 5.12.1:

1. ``activate_past_recording`` does not exist in 5.12.1; and even on transformers ``main`` it only
   preserves the linear-attention **conv** states — ``recurrent_states`` are overwritten in place
   and never rolled back.
2. ``crop()`` on a ``linear_attention`` layer is an unconditional no-op
   (``LinearAttentionCacheLayerMixin.crop``: *"We don't crop the linear attention cache"*).

So a naive port silently leaves 48 recurrent states advanced past the accepted prefix while
``get_seq_length()`` reports the cropped length. No error, just wrong tokens — measured on a shrunk
Qwen3.6 config, it flips ~25% of subsequent argmaxes.

The loop therefore rolls back explicitly, through the target's ``snapshot`` / ``restore``: on a
partial acceptance it restores the pre-block state and **replays the accepted tokens**. The replay
re-advances the recurrent state correctly and also produces the taps the next draft needs, so it is
not wasted work — but it does cost a second target forward on any step that is not fully accepted.

Porting note: that extra forward is an expedient, not a design. It is affordable on host and it is
what :class:`~.targets.TtTarget` does today, but on device it eats most of the speedup speculation
buys. The real fix is for the Gated DeltaNet kernel to checkpoint its recurrent state at each
candidate accept position inside the block, making rollback a state select instead of a replay.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from transformers import DynamicCache


@dataclass
class DFlashStats:
    """Result of :func:`dflash_generate` when ``return_stats=True``."""

    output_ids: torch.Tensor
    num_input_tokens: int
    num_output_tokens: int
    block_size: int
    # Tokens committed per speculative step, each in [1, block_size]. 1 means every draft was
    # rejected and only the target's own bonus token survived. Excludes prefill's own token, so
    # `num_output_tokens == 1 + sum(acceptance_lengths)`.
    acceptance_lengths: list[int] = field(default_factory=list)
    # Steps that needed the restore-and-replay path (a partial acceptance).
    num_rollbacks: int = 0

    @property
    def mean_acceptance_length(self) -> float:
        """Tokens per target forward. 1.0 means speculation bought nothing."""
        return sum(self.acceptance_lengths) / max(len(self.acceptance_lengths), 1)


# --------------------------------------------------------------------------------------------
# Sampling (upstream z-lab semantics, so acceptance behaviour matches the reference implementation)
# --------------------------------------------------------------------------------------------


def _validate_sampling(temperature: float, top_p: float, top_k: int) -> None:
    if temperature < 0 or not 0 < top_p <= 1 or top_k < 0:
        raise ValueError("temperature and top_k must be non-negative, and top_p in (0, 1].")


def _sampling_probs(logits: torch.Tensor, temperature: float, top_p: float = 1.0, top_k: int = 0) -> torch.Tensor:
    scores = logits.float() / temperature
    vocab_size = scores.shape[-1]
    indices = None
    if 0 < top_k < vocab_size:
        scores, indices = torch.topk(scores, top_k, dim=-1)

    probs = torch.softmax(scores, dim=-1)
    if top_p < 1.0:
        sorted_probs, order = probs.sort(dim=-1, descending=True)
        keep = sorted_probs.cumsum(dim=-1) - sorted_probs < top_p
        sorted_probs = sorted_probs * keep
        probs = torch.zeros_like(probs).scatter(-1, order, sorted_probs)
        probs = probs / probs.sum(dim=-1, keepdim=True)

    if indices is not None:
        probs = torch.zeros_like(logits, dtype=probs.dtype).scatter(-1, indices, probs)
    return probs


def _sample_probs(probs: torch.Tensor) -> torch.Tensor:
    shape = probs.shape[:-1]
    return torch.multinomial(probs.view(-1, probs.shape[-1]), 1).view(shape)


def sample(logits: torch.Tensor, temperature: float = 0.0, top_p: float = 1.0, top_k: int = 0) -> torch.Tensor:
    _validate_sampling(temperature, top_p, top_k)
    if temperature <= 0:
        return torch.argmax(logits, dim=-1)
    return _sample_probs(_sampling_probs(logits, temperature, top_p, top_k))


def _rejection_sample(draft_tokens, target_probs, draft_probs) -> tuple[int, torch.Tensor]:
    """Standard speculative rejection sampling: accept count, plus the replacement/bonus token."""
    gamma = draft_tokens.shape[1]
    p = target_probs[:, :gamma].gather(-1, draft_tokens[..., None])[..., 0]
    q = draft_probs.gather(-1, draft_tokens[..., None])[..., 0]
    accepted = (torch.rand_like(q) * q < p).to(torch.int32).cumprod(-1).sum(-1)[0].item()
    if accepted == gamma:
        return accepted, _sample_probs(target_probs[:, -1])[0]

    residual = target_probs[0, accepted].clone()
    residual.sub_(draft_probs[0, accepted])
    residual.clamp_min_(0)
    total = residual.sum()
    residual = torch.where(
        total > 0,
        residual / total.clamp_min(torch.finfo(residual.dtype).tiny),
        target_probs[0, accepted],
    )
    return accepted, _sample_probs(residual[None])[0]


# --------------------------------------------------------------------------------------------
# Drafter cache
# --------------------------------------------------------------------------------------------


def truncate_kv(cache: DynamicCache, length: int) -> None:
    """Drop everything past ``length`` tokens from the drafter's cache.

    This is the drafter's whole state management, and getting it wrong is silent. Each step appends
    K/V for two things: the newly accepted **context** rows (positions ``[length - n_new, length)``)
    and the **noise block** (``[length, length + q_len)``). The context must persist — it is how the
    drafter sees anything older than the last accepted block — and the block must not, because those
    slots were masked guesses. Truncating to exactly ``length`` keeps the first and drops the second.

    An earlier version of this snapshotted before the forward and restored after, which looked like
    the same thing but returned the cache to its *pre-forward* length — i.e. permanently empty. The
    drafter still drafted (``target_hidden`` alone carries a lot) so nothing failed, it just ran
    context-starved. Hence the length assertions below.
    """
    for layer in cache.layers:
        if not getattr(layer, "is_initialized", False):
            continue
        held = layer.keys.shape[-2]
        cumulative = getattr(layer, "cumulative_length", held)
        window = getattr(layer, "sliding_window", None)
        if window is not None:
            # Past the window the older rows are already evicted, so a tail truncation would leave
            # the layer short while `cumulative_length` still claimed them — wrong mask offsets, no
            # error. Fail loudly instead; a longer context needs a real cache, not this helper.
            assert cumulative < window, (
                f"drafter sliding layer has seen {cumulative} tokens, past its {window}-token window; "
                "truncation is no longer exact — the drafter needs a paged/rolling cache to go further"
            )
        drop = cumulative - length
        if drop <= 0:
            continue
        assert drop <= held, f"cannot drop {drop} of {held} cached rows"
        layer.keys = layer.keys[..., : held - drop, :]
        layer.values = layer.values[..., : held - drop, :]
        if hasattr(layer, "cumulative_length"):
            layer.cumulative_length = length


# --------------------------------------------------------------------------------------------
# The loop
# --------------------------------------------------------------------------------------------


def _as_target(target, drafter):
    """Accept either a :class:`~.targets.SpeculativeTarget` or a bare HF model (wrapped for you)."""
    from models.demos.blackhole.qwen36.reference.dflash.targets import HFTarget

    if hasattr(target, "snapshot") and hasattr(target, "forward") and hasattr(target, "lm_head"):
        return target
    return HFTarget(target, drafter.target_layer_ids)


def _as_drafter(drafter, target):
    """Accept either a :class:`~.drafters.SpeculativeDrafter` or a bare host ``DFlashDraftModel``."""
    from models.demos.blackhole.qwen36.reference.dflash.drafters import HostDrafter

    if hasattr(drafter, "propose"):
        return drafter
    return HostDrafter(drafter, target)


def _taps_head(target, taps, rows):
    """The FIRST ``rows`` rows of a block's taps — the tokens that were actually accepted.

    (Not the last: a block's taps cover positions ``[start, start + S)`` and acceptance always takes
    a prefix of them. ``TtTarget.taps_tail`` is a different operation, on a different axis.)
    """
    if hasattr(target, "taps_head"):
        return target.taps_head(taps, rows)
    return taps[:, :rows]


@torch.inference_mode()
def dflash_generate(
    drafter,
    target,
    input_ids: torch.LongTensor,
    max_new_tokens: int,
    *,
    stop_token_ids: list[int] | None = None,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 0,
    block_size: int | None = None,
    return_stats: bool = False,
):
    """Generate with DFlash speculation.

    Args:
        drafter: a :class:`~.dflash.DFlashDraftModel`, always on host.
        target: a :class:`~.targets.SpeculativeTarget` — :class:`~.targets.HFTarget` or
            :class:`~.targets.TtTarget`. A bare HF model is wrapped in ``HFTarget`` for you.
        input_ids: ``[1, prompt_len]``. Batch size 1 only — the rejection sampler and the
            acceptance bookkeeping are both written for a single sequence.
        block_size: slots per speculative step, defaulting to the drafter's own ``block_size``
            (16 for Qwen3.6-27B-DFlash: 1 anchor + 15 drafted). ``1`` disables speculation and
            gives a plain autoregressive baseline through this same code path — the control every
            equivalence check is read against.

    Returns:
        ``[1, prompt_len + generated]`` token ids, or a :class:`DFlashStats` when ``return_stats``.
    """
    _validate_sampling(temperature, top_p, top_k)
    assert input_ids.shape[0] == 1, f"batch size must be 1, got {input_ids.shape[0]}"
    target = _as_target(target, drafter)
    drafter = _as_drafter(drafter, target)

    input_ids = input_ids.cpu()
    num_input_tokens = input_ids.shape[1]
    max_length = num_input_tokens + max_new_tokens
    block_size = drafter.block_size if block_size is None else block_size
    speculating = block_size > 1

    # Unwritten slots hold the mask token, so a block's un-drafted tail is already the drafter's
    # "noise" input with no extra bookkeeping. One slot of slack for the final bonus token.
    output_ids = torch.full((1, max_length + 1), drafter.mask_token_id, dtype=torch.long)

    # ---- prefill ----
    target.reset()
    drafter.reset()
    logits, target_hidden = target.forward(input_ids, 0, all_logits=False)
    output_ids[:, :num_input_tokens] = input_ids
    output_ids[:, num_input_tokens] = sample(logits.cpu(), temperature, top_p, top_k)[0, -1]

    stop_tokens = torch.tensor(stop_token_ids, dtype=output_ids.dtype) if stop_token_ids else None
    stopped = stop_tokens is not None and bool(torch.isin(output_ids[:, num_input_tokens], stop_tokens).any())

    acceptance_lengths: list[int] = []
    num_rollbacks = 0
    start = num_input_tokens

    # ---- speculative decode ----
    while start + 1 < max_length and not stopped:
        # A target may cap the block: TtTarget's device bucket must not be crossed mid-block.
        verify_size = min(block_size, max_length - start, getattr(target, "max_block", lambda _: block_size)(start))
        block_ids = output_ids[:, start : start + verify_size].clone()
        draft_probs = None

        if verify_size > 1:
            # One drafter forward fills slots 1..verify_size-1. Everything about HOW — the noise
            # embedding, the KV history, the LM head, the sampling — sits behind `propose`, which
            # is what lets the ttnn drafter keep the entire draft on the mesh.
            drafted, draft_probs = drafter.propose(
                target_hidden, block_ids, start, temperature=temperature, top_p=top_p, top_k=top_k
            )
            block_ids[:, 1:] = drafted.cpu()

        # ---- verify: one target forward over the whole block ----
        snap = target.snapshot() if verify_size > 1 else None
        logits, taps = target.forward(block_ids, start)
        logits = logits.cpu()

        if temperature > 0:
            target_probs = _sampling_probs(logits, temperature, top_p, top_k)
            if verify_size > 1:
                acceptance_length, bonus = _rejection_sample(block_ids[:, 1:], target_probs, draft_probs)
            else:
                acceptance_length, bonus = 0, _sample_probs(target_probs[:, -1])[0]
        else:
            posterior = torch.argmax(logits, dim=-1)
            acceptance_length = (block_ids[:, 1:] == posterior[:, :-1]).cumprod(dim=1).sum(dim=1)[0].item()
            bonus = posterior[:, acceptance_length][0]

        output_ids[:, start : start + acceptance_length + 1] = block_ids[:, : acceptance_length + 1]
        output_ids[:, start + acceptance_length + 1] = bonus
        produced = min(acceptance_length + 1, max_length - start - 1)

        if stop_tokens is not None:
            hits = torch.isin(output_ids[0, start + 1 : start + produced + 1], stop_tokens).nonzero(as_tuple=True)[0]
            if hits.numel() > 0:
                produced = int(hits[0].item()) + 1
                stopped = True

        # ---- roll back the rejected tail ----
        if produced < verify_size:
            num_rollbacks += 1
            target.restore(snap, start)
            if getattr(target, "replays_after_rollback", True):
                # `restore` only rewound: replay the accepted prefix to re-advance the GDN state.
                # It also yields the next draft's taps, so the second forward is not pure overhead.
                _, taps = target.forward(output_ids[:, start : start + produced], start)
                target_hidden = taps
            else:
                # The target recomputes from its own anchor, so the rejected tail never happened
                # and the taps already in hand are correct. No replay.
                target_hidden = _taps_head(target, taps, produced)
        else:
            target_hidden = _taps_head(target, taps, produced)

        start += produced
        acceptance_lengths.append(produced)

    output_ids = output_ids[:, : min(start + 1, max_length)]
    if not return_stats:
        return output_ids
    return DFlashStats(
        output_ids=output_ids,
        num_input_tokens=num_input_tokens,
        num_output_tokens=output_ids.shape[1] - num_input_tokens,
        block_size=block_size,
        acceptance_lengths=acceptance_lengths,
        num_rollbacks=num_rollbacks,
    )
