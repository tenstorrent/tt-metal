# SPDX-License-Identifier: Apache-2.0
"""Chunked (FLA) GatedDeltaNet recurrence for the serial prefill view.

Replaces the token-serial resident kernel with ttnn.transformer.chunk_gated_delta_rule
(flat token-major q/k/v, in-kernel L2 norm, chunk 32, fp32 state, round_fp32_operands: the FPU
truncates fp32 operands to 9 mantissa bits toward zero, which unrounded biases the carried state low
chunk after chunk). The initial state is
the view's recurrent_state (gathered from the slot bank by _route_state); the final state
is copied back into it, so the bank scatter and the decode hand-off are unchanged.
Causal-conv state handling is untouched. Toggle: QWEN_PREFILL_CHUNK_GDN=0 restores the
serial kernel exactly (default on).
"""

import os

import torch

import ttnn

from .gdn_prefill_resident import PrefillResidentGDN

CHUNK = 32


def chunk_prefill_enabled():
    return os.environ.get("QWEN_PREFILL_CHUNK_GDN", "1") != "0"


_CONSTS = {}


def const_tiles(ops):
    """eye/tril/ones/quadrant masks, uploaded once per mesh before any trace capture."""
    key = id(ops.mesh)
    if key not in _CONSTS:
        ii, jj = torch.arange(32).unsqueeze(1), torch.arange(32).unsqueeze(0)
        lo_i, lo_j = ii < 16, jj < 16
        masks = torch.cat([(lo_i & lo_j).float(), (~lo_i & ~lo_j).float(), (~lo_i & lo_j).float()], dim=1)
        _CONSTS[key] = tuple(
            ops.upload(t.reshape(1, 1, *t.shape), ttnn.float32)
            for t in (torch.eye(CHUNK), torch.tril(torch.ones(CHUNK, CHUNK)), torch.ones(CHUNK, CHUNK), masks)
        )
    return _CONSTS[key]


def chunk_recurrence(q, k, v, g, beta, state, consts, **options):
    """q/k [1,S,4*128] raw (pre-norm), v [1,S,12*128], g log-decay / beta [1,S,12].

    Returns head-major o [1,12,S,128] fp32 and the final state [1,12,128,128] fp32.
    Padded tail tokens (S not a multiple of 32) carry g=0, beta=0: identity state updates.
    """
    b, s, _ = q.shape
    pad = (-s) % CHUNK
    dram = ttnn.DRAM_MEMORY_CONFIG
    if pad:
        q, k, v, g, beta = (
            ttnn.pad(t, [(0, 0), (0, pad), (0, 0)], 0.0, memory_config=dram) for t in (q, k, v, g, beta)
        )
    eye, tril, ones, masks = consts
    o, final = ttnn.transformer.chunk_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        scale=128**-0.5,
        initial_state=state,
        output_final_state=True,
        chunk_size=CHUNK,
        output_head_major=True,
        memory_config=dram,
        eye=eye,
        tril=tril,
        ones=ones,
        masks=masks,
        round_fp32_operands=True,
        **options,
    )
    o = ttnn.reshape(o, (b, 12, s + pad, 128))
    if pad:
        o = ttnn.slice(o, [0, 0, 0, 0], [b, 12, s, 128])
    return o, final


class PrefillChunkGDN(PrefillResidentGDN):
    prefill_chunk_enabled = False

    @property
    def prefill_log_decay(self):
        return self.prefill_chunk_enabled

    def _recurrence(self, q, k, v, g, beta):
        if not self.prefill_chunk_enabled:
            return super()._recurrence(q, k, v, g, beta)
        if q.shape[0] != 1:
            raise ValueError("Chunked GDN prefill runs one serial row")
        o, final = chunk_recurrence(q, k, v, g, beta, self.recurrent_state, self.prefill_chunk_consts)
        ttnn.copy(final, self.recurrent_state)
        return o, self.recurrent_state
