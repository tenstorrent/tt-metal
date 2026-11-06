# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The standard ``models.common.sampling`` stack (``SamplingGenerator`` over ``TTSampling``) on the vocab-sharded
LM-head logits, built to run inside the caller's own trace capture.

``sample`` records the top-k, all-gather, top-p and draw ops without the generator's internal trace, into a
preallocated token tensor.  Per-slot parameters and request seeds are host writes into persistent device tensors,
so they take effect on the next replay.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

import ttnn
from models.common.sampling.generator import SamplingGenerator, SamplingParams
from models.demos.blackhole.qwen38_flash_next.tp import TP_SIZE
from models.demos.blackhole.qwen38_flash_next.ttnn.embedding import TOKEN_ROW_SHAPE

SAMPLER_VOCAB_SIZE = 248_077
SAMPLER_PADDED_VOCAB_SIZE = 248_320
SAMPLER_BATCH = 32
SAMPLER_MAX_TOP_K = 32


def sampler_args() -> SimpleNamespace:
    """The fields ``TTSampling`` reads.  The per-device vocabulary shard (31040 wide) is padded to a power of two
    for the top-k; no ``SAMPLING_AG_CONFIG``, so the candidate path always runs (a greedy lane takes top-1)."""

    return SimpleNamespace(
        vocab_size=SAMPLER_VOCAB_SIZE,
        padded_vocab_size=SAMPLER_PADDED_VOCAB_SIZE,
        max_batch_size=SAMPLER_BATCH,
        max_top_k=SAMPLER_MAX_TOP_K,
        cluster_shape=[1, TP_SIZE],
        pad_logits_to_power_of_2=True,
        sub_core_grids=None,
        sampling_dp=1,
        model_config={},
    )


class Qwen38TTSampler:
    """Slot parameters, seeds and a trace-capturable ``sample`` for the 32 sampler lanes."""

    def __init__(self, mesh, tt_ccl):
        self.mesh = mesh
        self.generator = SamplingGenerator(args=sampler_args(), mesh_device=mesh, tt_ccl=tt_ccl)
        # Lanes never configured sample greedily (temperature 0).
        self._temperature = [0.0] * SAMPLER_BATCH
        self._top_k = [1] * SAMPLER_BATCH
        self._top_p = [1.0] * SAMPLER_BATCH
        self._seed: list[int | None] = [None] * SAMPLER_BATCH
        self._upload()

    @property
    def seeds_active(self) -> bool:
        return self.generator.seed_manager._seed_active

    def make_out_tok(self):
        """The preallocated replicated UINT32 ROW_MAJOR [1,1,1,32] token tensor ``sample`` writes."""

        return ttnn.from_torch(
            torch.zeros(1, 1, 1, SAMPLER_BATCH, dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
        )

    def set_slot_params(self, slot=0, temperature=0.0, top_k=1, top_p=1.0, seed=None):
        """Set one slot (an int) or several (an iterable) to the same parameters and upload all 32 lanes.

        ``temperature`` 0 is greedy.  ``seed`` None draws from the device PRNG; an int makes the slot's draws a
        function of the seed and the number of ``write_step_seeds`` calls since this call.
        """

        slots = [slot] if isinstance(slot, int) else list(slot)
        for lane in slots:
            self._temperature[lane] = float(temperature)
            self._top_k[lane] = int(top_k)
            self._top_p[lane] = float(top_p)
            self._seed[lane] = None if seed is None else int(seed)
        self._upload()

    def _upload(self):
        params = SamplingParams(
            temperature=list(self._temperature),
            top_k=list(self._top_k),
            top_p=list(self._top_p),
            seed=list(self._seed),
        )
        self.generator.apply_decode_state([params], reload_sampling_params=True, reset_sampling_state=False)
        manager = self.generator.seed_manager
        manager.reset_seed(self._seed, range(SAMPLER_BATCH))
        # Unseeded lanes: this pushes the entropy seeds; the next write_step_seeds pushes the skip that lets the
        # device PRNG advance on its own.
        manager.get_new_values()

    def write_step_seeds(self):
        """Host write of the next per-lane device seeds; call before every replay (a no-op once the unseeded state
        is steady)."""

        self.generator.seed_manager.get_new_values()

    def sample(self, logits, out_tok):
        """Record the sampling ops on ``logits`` (bf16 TILE [1,1,R,V/TP] per device, vocab sharded on dim 3) into
        ``out_tok``.  Valid eagerly and inside a trace capture."""

        tokens, _ = self.generator.sample(logits, enable_trace=False, tt_out_tok=out_tok)
        return tokens

    def token_row_from_out(self, out_tok):
        """Lane 0 of ``out_tok`` as a replicated FP32 TILE [1,1,1,32] token row (id at column 0, zeros elsewhere).
        Only data-movement and SFPU typecast ops touch the id, so every id below 2**24 is exact."""

        lane0 = ttnn.slice(out_tok, (0, 0, 0, 0), (1, 1, 1, 1))
        row = ttnn.pad(lane0, [(0, 0), (0, 0), (0, 0), (0, TOKEN_ROW_SHAPE[-1] - 1)], value=0)
        row = ttnn.to_layout(row, ttnn.TILE_LAYOUT)
        return ttnn.typecast(row, ttnn.float32)
