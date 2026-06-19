# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""On-device greedy / stochastic sampling for Seamless M4T v2 (Devstral-style).

Uses ``models.common.sampling.SamplingGenerator`` / ``TTSampling`` on width-sharded
``lm_head`` logits. When active this replaces the chunked ``ttnn.argmax`` path for both
greedy (``temperature=0``) and ``do_sample=True`` decode. Decode Metal trace replay runs
decoder + ``lm_head`` only; token selection happens via ``sample_into_buffer`` after trace
replay (default) or inside the trace when ``SEAMLESS_SAMPLE_IN_TRACE=1``.

Gating
------
On by default on multi-device meshes with local vocab shard ≤ 64K.
Set ``SEAMLESS_ONDEVICE_SAMPLING=0`` to fall back to chunked argmax / host sampling.
"""

from __future__ import annotations

import os
import random
from contextlib import contextmanager
from typing import Any, Iterator, Optional

import torch
from loguru import logger

import ttnn
from models.common.sampling import SamplingGenerator, SamplingParams, format_sampling_params

_SAMPLING_LOG_PREFIX = "models.common.sampling"
_MAX_UINT32 = 2**32 - 1


def on_device_sampling_enabled() -> bool:
    """``True`` by default; set ``SEAMLESS_ONDEVICE_SAMPLING=0`` for host fallback."""
    raw = os.environ.get("SEAMLESS_ONDEVICE_SAMPLING")
    if raw is None:
        return True
    return raw.strip().lower() not in ("0", "false", "no", "off")


def sample_in_decode_trace(mesh_device: ttnn.Device) -> bool:
    """Fold sampling into the decode trace. Default: no (post-trace sampling)."""
    env = os.environ.get("SEAMLESS_SAMPLE_IN_TRACE")
    if env is not None:
        return env.strip().lower() in ("1", "true", "yes", "on")
    return False


def supports_on_device_sampling(
    mesh_device: ttnn.Device,
    *,
    local_vocab_size: int,
) -> bool:
    if list(getattr(mesh_device, "shape", (1, 1))) == [1, 1]:
        return False
    if local_vocab_size <= 0 or local_vocab_size > 64 * 1024:
        return False
    return mesh_device.get_num_devices() > 1


def _sampling_force_argmax_config(mesh_device: ttnn.Device) -> dict:
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


class _SeamlessSamplingArgs:
    """Minimal ``args`` adapter for ``TTSampling`` using Seamless ``lm_head`` padding."""

    def __init__(
        self,
        mesh_device: ttnn.Device,
        *,
        vocab_size: int,
        padded_vocab_size: int,
        tt_ccl: Any,
    ):
        self.vocab_size = int(vocab_size)
        self.padded_vocab_size = int(padded_vocab_size)
        self.cluster_shape = tuple(mesh_device.shape)
        self.max_batch_size = 32
        self.sampling_all_gather_axis = 1 if int(mesh_device.shape[0]) == 1 else 0
        self.num_devices = mesh_device.get_num_devices()
        self.is_galaxy = int(mesh_device.shape[0]) > 1
        self.sampling_dp = 1
        if tt_ccl is not None:
            self.model_config = {"SAMPLING_AG_CONFIG": _sampling_force_argmax_config(mesh_device)}
        else:
            # Without tt_ccl, force-argmax all_gather_async is unavailable; top-k k=1 path
            # still works via ttnn.all_gather in TTSampling._perform_all_gather.
            self.model_config = {}


@contextmanager
def _suppress_sampling_module_logs() -> Iterator[None]:
    logger.disable(_SAMPLING_LOG_PREFIX)
    try:
        yield
    finally:
        logger.enable(_SAMPLING_LOG_PREFIX)


def pad_logits_to_sampling_batch(logits: ttnn.Tensor, sampling_batch: int) -> ttnn.Tensor:
    rows = int(logits.shape[-2])
    if rows >= sampling_batch:
        return logits
    return ttnn.pad(
        logits,
        [(0, 0), (0, 0), (0, sampling_batch - rows), (0, 0)],
        value=0.0,
    )


def prepare_row_logits_for_sampling(logits: ttnn.Tensor, *, dec_len: int) -> ttnn.Tensor:
    """Select one decode row and reshape to ``[1, 1, 1, V/tp]`` for ``TTSampling``."""
    rank = len(logits.shape)
    idx = max(0, int(dec_len) - 1)
    if rank == 3:
        batch, seq, vocab = (int(logits.shape[0]), int(logits.shape[1]), int(logits.shape[2]))
        row = logits
        if seq > 1:
            row = ttnn.slice(logits, (0, idx, 0), (batch, idx + 1, vocab))
        return ttnn.reshape(row, [1, 1, 1, vocab])
    if rank == 2:
        vocab = int(logits.shape[-1])
        return ttnn.reshape(logits, [1, 1, 1, vocab])
    if rank == 4:
        rows = int(logits.shape[-2])
        vocab = int(logits.shape[-1])
        if rows > 1:
            return ttnn.slice(logits, (0, 0, idx, 0), (1, 1, idx + 1, vocab))
        return logits
    raise ValueError(f"Unexpected sharded logits rank {rank} (shape={logits.shape})")


class OnDeviceSampler:
    """Wraps ``SamplingGenerator`` and a persistent output-token buffer."""

    def __init__(self, generator: SamplingGenerator, token_buffer: ttnn.Tensor, mesh_device: ttnn.Device):
        self.generator = generator
        self.token_buffer = token_buffer
        self.mesh_device = mesh_device
        self.sampling_batch = int(generator.tt_sampling.max_batch_size)
        self._configured_key: Optional[tuple] = None

    @property
    def max_batch_size(self) -> int:
        return self.sampling_batch

    def configure(
        self,
        *,
        do_sample: bool,
        temperature: float,
        top_k: int,
        top_p: float,
        repetition_penalty: float = 1.0,
        seed: Optional[int] = None,
    ) -> None:
        key = (bool(do_sample), float(temperature), int(top_k), float(top_p), float(repetition_penalty), seed)
        if key == self._configured_key:
            return
        self._configured_key = key

        if not do_sample or temperature <= 0.0:
            params = SamplingParams(
                temperature=0.0,
                top_k=1,
                top_p=1.0,
                repetition_penalty=repetition_penalty,
                seed=seed,
            )
        else:
            params = SamplingParams(
                temperature=temperature,
                top_k=top_k if top_k > 0 else 32,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                seed=seed,
            )
        empty_slots = list(range(self.sampling_batch))
        formatted = format_sampling_params(params, self.sampling_batch)
        if formatted.seed is None or (isinstance(formatted.seed, list) and formatted.seed[0] is None):
            rng_seed = random.randint(0, _MAX_UINT32)
            formatted = format_sampling_params(
                SamplingParams(
                    temperature=params.temperature,
                    top_k=params.top_k,
                    top_p=params.top_p,
                    repetition_penalty=repetition_penalty,
                    seed=rng_seed,
                ),
                self.sampling_batch,
            )
        with _suppress_sampling_module_logs():
            self.generator.reset_sampling_params(formatted)
            self.generator.seed_manager.reset_seed(formatted.seed, empty_slots)

    def advance_seed(self) -> None:
        with _suppress_sampling_module_logs():
            self.generator.seed_manager.get_new_values()

    def sample_into_buffer(self, logits: ttnn.Tensor) -> None:
        logits = pad_logits_to_sampling_batch(logits, self.sampling_batch)
        if logits.memory_config().buffer_type != ttnn.BufferType.DRAM:
            logits = ttnn.to_memory_config(logits, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        with _suppress_sampling_module_logs():
            self.generator.sample(logits, enable_trace=False, tt_out_tok=self.token_buffer)

    def sample_and_read(self, logits: ttnn.Tensor, *, dec_len: int = 1) -> int:
        row_logits = prepare_row_logits_for_sampling(logits, dec_len=dec_len)
        self.advance_seed()
        self.sample_into_buffer(row_logits)
        if row_logits is not logits:
            ttnn.deallocate(row_logits)
        return self.tt_read_token(0)

    def tt_read_token(self, batch_slot: int = 0) -> int:
        flat = ttnn.to_torch(ttnn.get_device_tensors(self.token_buffer)[0]).reshape(-1)
        return int(flat[batch_slot].item())

    def deallocate(self) -> None:
        ttnn.deallocate(self.token_buffer)


def sample_first_token_from_prefill_logits(
    sampler: OnDeviceSampler,
    prefill_logits: ttnn.Tensor,
    *,
    dec_len: int,
) -> int:
    """Sample the token at ``dec_len-1`` in sharded prefill logits."""
    row_logits = prepare_row_logits_for_sampling(prefill_logits, dec_len=dec_len)
    sampler.advance_seed()
    sampler.sample_into_buffer(row_logits)
    token = sampler.tt_read_token(0)
    if row_logits is not prefill_logits:
        ttnn.deallocate(row_logits)
    return token


def try_create_on_device_sampler(
    mesh_device: ttnn.Device,
    *,
    vocab_size: int,
    lm_head: Any,
    tt_ccl: Any = None,
) -> Optional[OnDeviceSampler]:
    if not on_device_sampling_enabled():
        return None

    local_vocab = int(getattr(lm_head, "local_vocab_size", 0) or 0)
    padded_vocab = int(getattr(lm_head, "padded_vocab_size", 0) or 0)
    if padded_vocab <= 0:
        padded_vocab = int(getattr(lm_head, "vocab_size", vocab_size) or vocab_size)

    if not supports_on_device_sampling(mesh_device, local_vocab_size=local_vocab):
        return None

    args = _SeamlessSamplingArgs(
        mesh_device,
        vocab_size=int(vocab_size),
        padded_vocab_size=padded_vocab,
        tt_ccl=tt_ccl,
    )
    with _suppress_sampling_module_logs():
        generator = SamplingGenerator(
            args=args,
            mesh_device=mesh_device,
            tt_ccl=tt_ccl,
            enable_internal_trace=False,
        )

    sampling_batch = int(generator.tt_sampling.max_batch_size)
    token_buffer = ttnn.from_torch(
        torch.zeros((1, 1, 1, sampling_batch), dtype=torch.int32),
        device=mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    sampler = OnDeviceSampler(generator, token_buffer, mesh_device)
    ccl_note = "with tt_ccl force-argmax" if tt_ccl is not None else "tt_ccl=None (top-k k=1 path)"
    logger.info(
        f"Seamless on-device sampling enabled ({ccl_note}, vocab={vocab_size}, "
        f"padded_vocab={padded_vocab}, local_shard={local_vocab})."
    )
    return sampler
