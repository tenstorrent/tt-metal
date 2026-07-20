# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""ACE-Step subclasses of ``tt_transformers`` post-processing modules.

ACE-Step's 5 Hz LM handler runs HF's ``generate`` loop on the host, with a per-step
FSM constrained logits processor that needs to inspect logits on CPU. That precludes
the canonical end-to-end on-device sampling loop that
:class:`~models.common.modules.sampling.penalties_1d.Penalties1D` /
:class:`~models.common.modules.sampling.sampling_1d.Sampling1D` are designed for.

To still use the upstream ``tt_transformers`` post-processing primitives without
rewriting the constrained-decoding architecture, this module exposes:

- :class:`AceStepPenalties1D` — subclasses :class:`Penalties1D` and adds a stateless
  :meth:`AceStepPenalties1D.apply_hf_repetition` that emulates the HF
  ``RepetitionPenaltyLogitsProcessor`` contract (per-call: scatter ``input_ids`` into
  ``params.prompt_mask`` via the parent's ``init_prompt_penalties``; run the parent's
  ``decode_forward`` with ``presence=0``, ``frequency=0``, ``repetition=penalty``).
- :class:`AceStepSampling1D` — subclasses :class:`Sampling1D` and adds a stateless
  :meth:`AceStepSampling1D.sample_topk_topp_temp` that builds per-call ``k``/``p``/``temp``
  TTNN tensors and invokes the parent's ``decode_forward`` (fused top-k + top-p + temp
  + sample on device).
- :func:`apply_penalty_filter_sample` — the fused entry point used by
  ``five_hz_llm_inference._postprocess_and_sample_ttnn``. Runs repetition penalty
  AND top-k/top-p/temperature sampling in a **single** on-device pipeline (logits never
  leave the chip between penalty and sample), so per-token latency drops from
  3 host↔device round-trips to 1.

Both subclasses cache one instance per ``(device_id, vocab_size, max_batch_size)``
tuple to amortise the heavy buffer allocation across decode steps; callers should
treat the module-level helpers as the public API.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch

import ttnn
from models.common.modules.sampling.penalties_1d import Penalties1D, PenaltyAccumulator, PenaltyParams
from models.common.modules.sampling.sampling_1d import Sampling1D

# ---------------------------------------------------------------------------
# Penalties subclass
# ---------------------------------------------------------------------------


class AceStepPenalties1D(Penalties1D):
    """Penalties1D + stateless ``apply_hf_repetition`` for ACE-Step's host-driven loop.

    All on-device buffer work is inherited from ``Penalties1D``; this class only adds
    the per-call façade.
    """

    def apply_hf_repetition(
        self,
        scores: torch.Tensor,
        input_ids: torch.Tensor,
        penalty: float,
    ) -> torch.Tensor:
        """Match :class:`transformers.RepetitionPenaltyLogitsProcessor` semantics.

        For every token id appearing in ``input_ids[b]``: if ``scores[b, id] > 0`` divide
        by ``penalty``, else multiply by ``penalty``. Stateless from the caller's POV;
        internally we re-initialise ``params.prompt_mask`` from ``input_ids`` each call
        and leave ``accum.output_mask`` / ``output_counts`` at zero so the
        presence/frequency terms cancel exactly.

        Args:
            scores: ``[B, V]`` host torch logits (any float dtype).
            input_ids: ``[B, seq]`` host torch ids (any int dtype).
            penalty: ``> 0``.

        Returns:
            ``[B, V]`` host torch logits with the same dtype as ``scores``.
        """
        if penalty == 1.0:
            return scores
        if penalty <= 0:
            raise ValueError(f"penalty must be > 0, got {penalty}")
        if scores.dim() != 2:
            raise ValueError(f"expected scores [B, V], got {tuple(scores.shape)}")

        device = self.config.mesh_device
        params, accum = self._get_or_build_state()

        # Write the per-call penalty value into the cached params.repetition_penalties /
        # inverse_repetition_penalties buffers (host → device copy of 2 small [B,1] tensors).
        self._set_repetition_penalty(params, float(penalty))

        # Scatter input_ids into params.prompt_mask (overwrites previous content because
        # the parent's _token_bin_counts_and_mask scatters from self._zeros every call).
        self.init_prompt_penalties(params, accum, input_ids)

        # Upload scores → device, run penalty, read back.
        scores_tt = ttnn.from_torch(
            scores.detach().to(dtype=torch.bfloat16).contiguous(),
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=self._replicate_mapper,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        out_tt = self.decode_forward(scores_tt, params, accum)
        out = ttnn.to_torch(out_tt, dtype=torch.float32).contiguous()
        ttnn.deallocate(out_tt)
        return out.to(dtype=scores.dtype)

    # -- Internals ---------------------------------------------------------

    def _get_or_build_state(self) -> Tuple[PenaltyParams, PenaltyAccumulator]:
        """Lazily build + cache one ``(PenaltyParams, PenaltyAccumulator)`` pair.

        ACE-Step's batch is always 1 and the device + vocab are fixed for the lifetime
        of an :class:`AceStepPenalties1D` instance — one cached pair is sufficient.
        """
        cached = getattr(self, "_acestep_state", None)
        if cached is not None:
            return cached

        # Materialise the parent's lazy buffers (idempotent).
        self.load_device_buffers()
        cfg = self.config

        # All buffers are pre-allocated zero tensors by ``_resolve_penalties1d_config``;
        # we just materialise the LazyBuffer handles into ttnn tensors here.
        params = PenaltyParams(
            prompt_mask=_materialise_buf(cfg.prompt_mask),
            presence_penalties=_materialise_buf(cfg.presence_penalties),
            frequency_penalties=_materialise_buf(cfg.frequency_penalties),
            repetition_penalties=_materialise_buf(cfg.repetition_penalties),
            inverse_repetition_penalties=_materialise_buf(cfg.inverse_repetition_penalties),
        )
        accum = PenaltyAccumulator(
            output_mask=_materialise_buf(cfg.output_mask),
            output_counts=_materialise_buf(cfg.output_counts),
            output_counts_gathered=_materialise_buf(cfg.output_counts_gathered),
        )
        self._acestep_state = (params, accum)
        return self._acestep_state

    def _set_repetition_penalty(self, params: PenaltyParams, penalty: float) -> None:
        """Refresh ``params.repetition_penalties`` and ``inverse_repetition_penalties``.

        Both tensors are ``[max_batch_size, 1]`` bf16; for ACE-Step we only use slot 0
        but we fill the whole tensor so ``decode_forward`` works uniformly across the
        padded batch (other rows are no-ops since their prompt_mask stays zero).

        We deallocate the previous TTNN tensor before swapping in the new one so the
        per-call rebuilds don't leak device memory across the (typically thousands of)
        decode steps in an ACE-Step generate loop.
        """
        cfg = self.config
        rep_host = torch.full((cfg.max_batch_size, 1), float(penalty), dtype=torch.float32)
        inv_host = torch.full((cfg.max_batch_size, 1), 1.0 / float(penalty), dtype=torch.float32)

        rep_tt = ttnn.from_torch(
            rep_host,
            device=cfg.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=self._replicate_mapper,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        inv_tt = ttnn.from_torch(
            inv_host,
            device=cfg.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=self._replicate_mapper,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        # Free the previous device tensors before swapping the dataclass refs so we don't
        # accumulate stale [max_batch_size, 1] buffers across decode steps.
        prev_rep = params.repetition_penalties
        prev_inv = params.inverse_repetition_penalties
        params.repetition_penalties = rep_tt
        params.inverse_repetition_penalties = inv_tt
        for prev in (prev_rep, prev_inv):
            if isinstance(prev, ttnn.Tensor):
                try:
                    ttnn.deallocate(prev)
                except Exception:
                    pass


def _materialise_buf(buf) -> Any:
    """LazyBuffer → ttnn.Tensor; passthrough if already a Tensor."""
    if isinstance(buf, ttnn.Tensor):
        return buf
    return buf.get_device_buffer()


# ---------------------------------------------------------------------------
# Sampling subclass
# ---------------------------------------------------------------------------


class AceStepSampling1D(Sampling1D):
    """Sampling1D + stateless ``sample_topk_topp_temp`` for ACE-Step.

    The parent's ``decode_forward(logits, *, k, p, temp, seeds)`` takes per-batch TTNN
    tensors; this façade accepts Python scalars (the values ACE-Step actually has) and
    builds the small ``[max_batch_size]`` tensors per call.
    """

    def sample_topk_topp_temp(
        self,
        scores_tt: ttnn.Tensor,
        top_k: int,
        top_p: float,
        temperature: float,
        seed: int,
    ) -> ttnn.Tensor:
        """Fused top-k + top-p + temperature sampling on device.

        Args:
            scores_tt: ``[1, 1, B, V]`` device logits (TILE layout, bfloat16). Caller-owned.
            top_k: ``>= 1`` (clamped to ``max_top_k``). ``None``/``<=0`` → no top-k filter
                (top_k = vocab_size).
            top_p: ``0 < p <= 1`` (``p=1`` ⇒ no nucleus filter).
            temperature: ``> 0`` (``temperature=0`` ⇒ argmax via greedy path).
            seed: monotonic per-call seed (caller increments).

        Returns:
            ``[max_batch_size]`` int32 device tensor of sampled ids; caller reads row 0.
        """
        device = self.config.mesh_device
        cfg = self.config

        # Greedy path: temperature == 0 ⇒ argmax. Sampling1D requires either all of
        # (k, p, temp) or none + ``allow_force_argmax=True``. We didn't enable the
        # argmax-force flag (it requires extra all_gather config) — emulate greedy as
        # top-1 sampling with temp=1.0, which is exact.
        eff_top_k = int(top_k) if top_k is not None and int(top_k) > 0 else int(cfg.max_top_k)
        eff_top_k = min(eff_top_k, int(cfg.max_top_k))
        eff_top_p = float(top_p) if top_p is not None and 0.0 < float(top_p) <= 1.0 else 1.0
        if temperature is None or float(temperature) <= 0.0:
            eff_top_k = 1
            eff_temp = 1.0
        else:
            eff_temp = float(temperature)

        # Per-call sampling-param tensors: [max_batch_size], replicated across mesh.
        k_host = torch.full((cfg.max_batch_size,), int(eff_top_k), dtype=torch.int64).to(torch.int32)
        p_host = torch.full((cfg.max_batch_size,), float(eff_top_p), dtype=torch.float32)
        temp_host = torch.full((cfg.max_batch_size,), float(eff_temp), dtype=torch.float32)
        seed_host = torch.full((cfg.max_batch_size,), int(seed) & 0x7FFFFFFF, dtype=torch.int64).to(torch.int32)

        replicate_mapper = ttnn.ShardTensor2dMesh(device, dims=(None, None), mesh_shape=device.shape)
        k_tt = ttnn.from_torch(
            k_host,
            device=device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=replicate_mapper,
        )
        p_tt = ttnn.from_torch(
            p_host,
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=replicate_mapper,
        )
        temp_tt = ttnn.from_torch(
            temp_host,
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=replicate_mapper,
        )
        seeds_tt = ttnn.from_torch(
            seed_host,
            device=device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=replicate_mapper,
        )

        tokens_tt, _log_probs = self.decode_forward(scores_tt, k=k_tt, p=p_tt, temp=temp_tt, seeds=seeds_tt)
        ttnn.deallocate(k_tt)
        ttnn.deallocate(p_tt)
        ttnn.deallocate(temp_tt)
        ttnn.deallocate(seeds_tt)
        return tokens_tt


# ---------------------------------------------------------------------------
# Per-(device, vocab, max_batch_size) caches
# ---------------------------------------------------------------------------

_PENALTIES_CACHE: Dict[Tuple[int, int, int], AceStepPenalties1D] = {}
_SAMPLING_CACHE: Dict[Tuple[int, int, int], AceStepSampling1D] = {}
_DEFAULT_MAX_BATCH = 32


def _penalties_max_batch_size(batch: int) -> int:
    """``Penalties1D`` penalty tensors are ``[max_batch_size, …]``; logits must match on dim 0."""
    return max(_DEFAULT_MAX_BATCH, int(batch))


def _pad_logits_batch(scores: torch.Tensor, max_batch: int) -> tuple[torch.Tensor, int]:
    """Pad ``[B, V]`` logits to ``[max_batch, V]``; return (padded, original_B)."""
    batch = int(scores.shape[0])
    max_batch = int(max_batch)
    if batch > max_batch:
        raise ValueError(f"logits batch {batch} exceeds max_batch_size {max_batch}")
    if batch == max_batch:
        return scores, batch
    pad_rows = max_batch - batch
    filler = torch.zeros((pad_rows, int(scores.shape[-1])), dtype=scores.dtype, device=scores.device)
    return torch.cat([scores, filler], dim=0).contiguous(), batch


def _device_key(device: Any) -> int:
    """Stable per-device identity for cache keys (mesh_device or single device)."""
    for attr in ("get_device_id", "id", "device_id"):
        fn = getattr(device, attr, None)
        try:
            if callable(fn):
                return int(fn())
            if fn is not None:
                return int(fn)
        except Exception:
            continue
    return id(device)


def _get_penalties(device: Any, vocab_size: int, max_batch_size: int = 32) -> AceStepPenalties1D:
    key = (_device_key(device), int(vocab_size), int(max_batch_size))
    cached = _PENALTIES_CACHE.get(key)
    if cached is not None:
        return cached
    inst = AceStepPenalties1D(vocab_size=int(vocab_size), mesh_device=device, max_batch_size=int(max_batch_size))
    _PENALTIES_CACHE[key] = inst
    return inst


def _get_sampling(device: Any, vocab_size: int, max_batch_size: int = 32, max_top_k: int = 256) -> AceStepSampling1D:
    key = (_device_key(device), int(vocab_size), int(max_batch_size))
    cached = _SAMPLING_CACHE.get(key)
    if cached is not None:
        return cached
    inst = AceStepSampling1D(
        vocab_size=int(vocab_size),
        mesh_device=device,
        max_batch_size=int(max_batch_size),
        max_top_k=int(max_top_k),
    )
    _SAMPLING_CACHE[key] = inst
    return inst


# ---------------------------------------------------------------------------
# Public stateless helpers (drop-in replacements for the deleted *_bf16 helpers)
# ---------------------------------------------------------------------------


def _pad_batch_for_sampling(
    scores_2d: torch.Tensor,
    input_ids: torch.Tensor,
    max_batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Pad ``[B, V]`` logits (and matching ``input_ids``) to ``max_batch_size`` rows."""
    batch = int(scores_2d.shape[0])
    vocab = int(scores_2d.shape[1])
    if batch > int(max_batch_size):
        raise ValueError(f"batch={batch} exceeds Sampling1D max_batch_size={max_batch_size}")
    if batch >= int(max_batch_size):
        return scores_2d, input_ids, batch
    scores_pad = torch.full(
        (int(max_batch_size), vocab),
        float("-inf"),
        dtype=scores_2d.dtype,
        device=scores_2d.device,
    )
    scores_pad[:batch] = scores_2d
    seq = int(input_ids.shape[1])
    ids_pad = torch.zeros((int(max_batch_size), seq), dtype=input_ids.dtype, device=input_ids.device)
    ids_pad[:batch] = input_ids
    return scores_pad, ids_pad, batch


def _upload_logits_tile_4d(scores_2d: torch.Tensor, *, device: Any) -> ttnn.Tensor:
    """Upload ``[B, V]`` host logits as ``[1, 1, B, V]`` TILE (``Sampling1D`` layout)."""
    if scores_2d.dim() != 2:
        raise ValueError(f"expected [B, V] scores, got {tuple(scores_2d.shape)}")
    host_4d = scores_2d.detach().to(dtype=torch.bfloat16).unsqueeze(0).unsqueeze(0).contiguous()
    replicate_mapper = ttnn.ShardTensor2dMesh(device, dims=(None, None), mesh_shape=device.shape)
    return ttnn.from_torch(
        host_4d,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=replicate_mapper,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def repetition_penalty_apply(
    scores: torch.Tensor,
    input_ids: torch.Tensor,
    penalty: float,
    *,
    device: Any,
) -> torch.Tensor:
    """Stateless façade matching the old ``repetition_penalty_apply_bf16`` signature.

    Routes through :class:`AceStepPenalties1D` (cached per device+vocab).
    """
    if penalty == 1.0:
        return scores
    batch = int(scores.shape[0])
    vocab = int(scores.shape[-1])
    max_batch = _penalties_max_batch_size(batch)
    # ``Penalties1D`` expects vocab to be divisible by num_devices. Pad to
    # ``TILE_SIZE * num_devices`` (TP-4 → 128) so multi-device LM sampling stays even;
    # callers see the original ``vocab`` columns sliced back.
    from models.experimental.ace_step_v1_5.utils.ace_step_tp import ace_step_padded_vocab_size
    from models.experimental.ace_step_v1_5.utils.tt_device import ace_step_device_num_chips

    n_dev = max(1, ace_step_device_num_chips(device))
    pad_to = ace_step_padded_vocab_size(vocab, n_dev)
    if pad_to != vocab:
        padded = torch.full((batch, pad_to), float("-inf"), dtype=scores.dtype, device=scores.device)
        padded[:, :vocab] = scores
        scores_in = padded
    else:
        scores_in = scores
    scores_in, batch = _pad_logits_batch(scores_in, max_batch)
    inst = _get_penalties(device, pad_to, max_batch_size=max_batch)
    out = inst.apply_hf_repetition(scores_in, input_ids, float(penalty))
    out = out[:batch]
    if pad_to != vocab:
        out = out[:, :vocab].contiguous()
    return out


def apply_penalty_filter_sample(
    scores: torch.Tensor,
    input_ids: torch.Tensor,
    *,
    repetition_penalty: float,
    top_k: Optional[int],
    top_p: Optional[float],
    temperature: float,
    seed: int,
    device: Any,
) -> torch.Tensor:
    """Fused on-device: repetition penalty → top-k + top-p + temperature → sample.

    Logits never leave the device between the penalty step and the sample step (one
    host→device upload, one device→host token download). Returns ``[B]`` host int64
    token ids.
    """
    batch = int(scores.shape[0])
    vocab = int(scores.shape[-1])
    max_batch = _penalties_max_batch_size(batch)
    from models.experimental.ace_step_v1_5.utils.ace_step_tp import ace_step_padded_vocab_size
    from models.experimental.ace_step_v1_5.utils.tt_device import ace_step_device_num_chips

    n_dev = max(1, ace_step_device_num_chips(device))
    pad_to = ace_step_padded_vocab_size(vocab, n_dev)

    if pad_to != vocab:
        padded = torch.full((batch, pad_to), float("-inf"), dtype=scores.dtype, device=scores.device)
        padded[:, :vocab] = scores
        scores_in = padded
    else:
        scores_in = scores

    penalties = _get_penalties(device, pad_to, max_batch_size=max_batch)
    sampler = _get_sampling(device, pad_to, max_batch_size=max_batch, max_top_k=max(32, int(top_k or 32)))
    penalties.load_device_buffers()

    scores_in, input_ids_pad, batch = _pad_batch_for_sampling(scores_in, input_ids, max_batch)

    # Sampling1D top-k splits on dim=3 → logits must be [1, 1, max_batch_size, V].
    scores_tt = _upload_logits_tile_4d(scores_in, device=device)

    # Step 1: repetition penalty (if enabled). Reuses cached PenaltyParams/Accumulator.
    if repetition_penalty != 1.0:
        params, accum = penalties._get_or_build_state()
        penalties._set_repetition_penalty(params, float(repetition_penalty))
        penalties.init_prompt_penalties(params, accum, input_ids_pad)
        scores_post_penalty = penalties.decode_forward(scores_tt, params, accum)
        ttnn.deallocate(scores_tt)
        scores_tt = scores_post_penalty

    # Step 2: fused top-k + top-p + temperature sampling.
    tokens_tt = sampler.sample_topk_topp_temp(
        scores_tt,
        top_k=top_k if top_k is not None else 0,
        top_p=top_p if top_p is not None else 1.0,
        temperature=temperature,
        seed=int(seed),
    )
    ttnn.deallocate(scores_tt)

    tokens_host = ttnn.to_torch(tokens_tt).reshape(-1).to(dtype=torch.int64)
    ttnn.deallocate(tokens_tt)
    return tokens_host[:batch].contiguous()


__all__ = [
    "AceStepPenalties1D",
    "AceStepSampling1D",
    "apply_penalty_filter_sample",
    "repetition_penalty_apply",
]
