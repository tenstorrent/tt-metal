# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""On-device greedy/sampling for the Devstral-2 text demo (single command queue).

Mirrors the sibling Devstral Small PR (#44834): a ``SamplingGenerator`` runs on the
column-parallel ``lm_head`` logits so token selection happens **on device** and gets
folded into the captured decode trace. Each decode step then transfers only one
``int32`` token to host (``tt_read_token``) instead of the full sharded vocab row
plus a host ``torch.argmax``.

Shape note
----------
The Devstral-2 decode forward produces logits with a *logical* batch of 1 (physically
padded to a 32-row tile). ``TTSampling`` is locked to ``max_batch_size`` ≥ 32 — its
``ttnn.topk`` indices buffer is ``[1, 1, 32, vocab/num_devices]`` — so the 1-row logits
are padded up to 32 logical rows (:func:`pad_logits_to_sampling_batch`) before sampling.
Only row 0 (the single active user) is read back; the padded rows produce throw-away
tokens.

Gating
------
On-device sampling is **on by default** when the mesh supports it (multi-device, and
``vocab_size / num_devices <= 64K`` per shard). Set ``DEVSTRAL2_ONDEVICE_SAMPLING=0`` to
use host ``torch.argmax`` instead. ``TTSampling`` corrupts on a single-device mesh and the
top-k path caps each shard at 64K.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator, Optional

import torch
from loguru import logger

import ttnn
from models.common.sampling import SamplingGenerator, SamplingParams, format_sampling_params
from models.experimental.devstral2_123B_instruct.tt.model_args import Devstral2Args


_SAMPLING_LOG_PREFIX = "models.common.sampling"


def verbose_sampling_logs() -> bool:
    """``True`` when ``DEVSTRAL2_VERBOSE_SAMPLING`` is set (show per-token TTSampling logs)."""
    return os.environ.get("DEVSTRAL2_VERBOSE_SAMPLING", "").strip().lower() in ("1", "true", "yes", "on")


@contextmanager
def _suppress_sampling_module_logs() -> Iterator[None]:
    """Temporarily mute ``models.common.sampling`` loguru output (no library edits)."""
    if verbose_sampling_logs():
        yield
        return
    logger.disable(_SAMPLING_LOG_PREFIX)
    try:
        yield
    finally:
        logger.enable(_SAMPLING_LOG_PREFIX)


def quiet_per_token_sampling_logs() -> None:
    """Hide per-decode-step ``TTSampling`` INFO/DEBUG spam (default for Devstral demos).

    ``text_demo`` / ``tt_demo_agent`` call this at startup. Each ``sample`` call is also
    wrapped via :func:`_suppress_sampling_module_logs` so logs stay quiet even if something
    re-enables the module. Opt back in with ``DEVSTRAL2_VERBOSE_SAMPLING=1``.
    """
    if verbose_sampling_logs():
        logger.enable(_SAMPLING_LOG_PREFIX)
    else:
        logger.disable(_SAMPLING_LOG_PREFIX)


def _wrap_generator_sample(generator: SamplingGenerator) -> None:
    """Patch ``SamplingGenerator.sample`` so direct/test callers stay quiet too."""
    if getattr(generator, "_devstral_quiet_sample", False):
        return
    original_sample = generator.sample

    def quiet_sample(*args, **kwargs):
        with _suppress_sampling_module_logs():
            return original_sample(*args, **kwargs)

    generator.sample = quiet_sample  # type: ignore[method-assign]
    generator._devstral_quiet_sample = True  # type: ignore[attr-defined]


def on_device_sampling_enabled() -> bool:
    """``True`` by default; set ``DEVSTRAL2_ONDEVICE_SAMPLING=0`` to use host argmax."""
    raw = os.environ.get("DEVSTRAL2_ONDEVICE_SAMPLING")
    if raw is None:
        return True
    return raw.strip().lower() not in ("0", "false", "no", "off")


def supports_on_device_sampling(args: Devstral2Args, mesh_device) -> bool:
    """``True`` if the mesh can run ``TTSampling`` (multi-device, vocab shard ≤ 64K).

    ``TTSampling`` corrupts on a ``[1, 1]`` mesh, and its per-device top-k path requires
    each vocab shard to fit the 64K limit.
    """
    if list(mesh_device.shape) == [1, 1]:
        return False
    return args.vocab_size // args.num_devices <= 64 * 1024


def sample_in_decode_trace(mesh_device) -> bool:
    """Whether to fold sampling ops into the captured decode trace. **Default: no.**

    Sampling runs as its own program after ``execute_trace`` so its ``ttnn.topk`` /
    ``ttnn.sampling`` circular buffers never share L1 with the full 123B decode graph
    (folding them in clashes on Blackhole and risks OOM/L1 exhaustion on large meshes).

    Set ``DEVSTRAL2_SAMPLE_IN_TRACE=1`` to opt into in-trace sampling for lower per-step
    latency where L1 headroom allows.
    """
    env = os.environ.get("DEVSTRAL2_SAMPLE_IN_TRACE")
    if env is not None:
        return env.strip().lower() in ("1", "true", "yes", "on")
    return False


def _sampling_force_argmax_config(mesh_device) -> dict:
    """``SAMPLING_AG_CONFIG`` for greedy (k=1, p=1, temp=1) — all_gather + argmax, no top-k."""
    num_devices = mesh_device.get_num_devices()
    num_links = min(8, max(1, num_devices))
    topology = ttnn.Topology.Ring if num_devices >= 8 else ttnn.Topology.Linear
    return {
        "allow_force_argmax": True,
        "num_links": num_links,
        "chunks_per_sync": 10,
        "num_workers_per_link": 1,
        "topology": topology,
    }


class _SamplingArgs:
    """Adapter exposing only the attributes ``TTSampling`` / ``TTPenalties`` read from ``args``.

    ``Devstral2Args`` lacks ``cluster_shape`` (the sampling modules' one required extra field);
    every other field falls back to a ``getattr`` default inside ``TTSampling``.
    """

    def __init__(self, args: Devstral2Args, mesh_device):
        self.vocab_size = int(args.vocab_size)
        self.cluster_shape = tuple(mesh_device.shape)
        self.max_batch_size = int(args.max_batch_size)
        self.model_config = {"SAMPLING_AG_CONFIG": _sampling_force_argmax_config(mesh_device)}


class OnDeviceSampler:
    """Owns a ``SamplingGenerator`` plus the persistent ``int32`` output-token buffer.

    The token buffer is allocated once and bound into the decode trace, so trace replay
    writes the sampled tokens to a stable address that ``tt_read_token`` reads back.
    """

    def __init__(self, generator: SamplingGenerator, token_buffer: ttnn.Tensor, mesh_device):
        self.generator = generator
        self.token_buffer = token_buffer
        self.mesh_device = mesh_device
        self.sampling_batch = int(generator.tt_sampling.max_batch_size)

    def advance_seed(self) -> None:
        """Push fresh RNG state for the next step (no-op in the steady greedy/unseeded state)."""
        self.generator.seed_manager.get_new_values()

    def sample_into_buffer(self, logits: ttnn.Tensor) -> None:
        """Run sampling on ``logits`` (padded to the sampling batch) writing to ``token_buffer``.

        Called *inside* the decode trace capture region with ``enable_trace=False`` so the
        sampling ops are recorded into the surrounding decode trace rather than a separate one.
        """
        logits = pad_logits_to_sampling_batch(logits, self.sampling_batch)
        if logits.memory_config().buffer_type != ttnn.BufferType.DRAM:
            logits = ttnn.to_memory_config(logits, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        with _suppress_sampling_module_logs():
            self.generator.sample(logits, enable_trace=False, tt_out_tok=self.token_buffer)

    def tt_read_token(self, batch_slot: int = 0) -> int:
        """Read the sampled global token id for ``batch_slot`` (D2H of one ``int32``)."""
        flat = ttnn.to_torch(ttnn.get_device_tensors(self.token_buffer)[0]).reshape(-1)
        return int(flat[batch_slot].item())

    def deallocate(self) -> None:
        ttnn.deallocate(self.token_buffer)


def sample_first_token_from_prefill_logits(
    sampler: OnDeviceSampler,
    prefill_logits: ttnn.Tensor,
    *,
    local_pos: int,
) -> int:
    """Sample the token at ``local_pos`` in the last prefill chunk via on-device argmax.

    Prefill logits are ``[1, 1, chunk_len, vocab/num_devices]`` per shard; only the row at
    ``local_pos`` (last prompt index within the chunk) is sliced and passed to
    :meth:`OnDeviceSampler.sample_into_buffer` (same ``ttnn.argmax`` path as decode).
    """
    row_logits = ttnn.slice(
        prefill_logits,
        (0, 0, local_pos, 0),
        (1, 1, local_pos + 1, prefill_logits.shape[-1]),
    )
    sampler.advance_seed()
    sampler.sample_into_buffer(row_logits)
    token = sampler.tt_read_token(0)
    if row_logits is not prefill_logits:
        row_logits.deallocate(True)
    return token


def pad_logits_to_sampling_batch(logits: ttnn.Tensor, sampling_batch: int) -> ttnn.Tensor:
    """Pad decode logits from a logical batch of 1 up to ``sampling_batch`` (32) rows.

    No-op when the logits already have ≥ ``sampling_batch`` rows (e.g. a prefill block).
    """
    rows = int(logits.shape[-2])
    if rows >= sampling_batch:
        return logits
    return ttnn.pad(
        logits,
        [(0, 0), (0, 0), (0, sampling_batch - rows), (0, 0)],
        value=0.0,
    )


def build_sampler(
    args: Devstral2Args,
    mesh_device,
    tt_ccl,
    *,
    temperature: float = 0.0,
    top_k: int = 1,
    top_p: float = 1.0,
    seed: Optional[int] = None,
) -> OnDeviceSampler:
    """Construct a greedy-by-default on-device sampler over the column-parallel lm_head logits.

    ``temperature == 0.0`` selects greedy decoding (``top_k`` is forced to 1 by
    ``format_sampling_params``, matching the host ``argmax`` path). A positive temperature
    enables stochastic top-k / top-p sampling.
    """
    with _suppress_sampling_module_logs():
        generator = SamplingGenerator(
            args=_SamplingArgs(args, mesh_device),
            mesh_device=mesh_device,
            tt_ccl=tt_ccl,
            enable_internal_trace=False,  # sampling ops are captured in the demo's decode trace
        )

    _wrap_generator_sample(generator)

    empty_slots = list(range(generator.tt_sampling.max_batch_size))
    if temperature == 0.0:
        params = SamplingParams(temperature=0.0, top_k=1, top_p=1.0, seed=seed)
    else:
        params = SamplingParams(temperature=temperature, top_k=top_k, top_p=top_p, seed=seed)
    formatted = format_sampling_params(params, len(empty_slots))
    generator.reset_sampling_params(formatted)
    generator.seed_manager.reset_seed(formatted.seed, empty_slots)

    sampling_batch = int(generator.tt_sampling.max_batch_size)
    token_buffer = ttnn.from_torch(
        torch.zeros((1, 1, 1, sampling_batch), dtype=torch.int32),
        device=mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    mode = "greedy (argmax)" if temperature == 0.0 else f"top-k={top_k} top-p={top_p} temp={temperature}"
    logger.info(f"On-device sampling enabled: {mode}, sampling batch {sampling_batch}, vocab {args.vocab_size}.")
    return OnDeviceSampler(generator, token_buffer, mesh_device)
