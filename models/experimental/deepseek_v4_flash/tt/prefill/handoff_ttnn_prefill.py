# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Hand the pure-ttnn chunked prefill (``models/demos/deepseek_v3_d_p/tt/v4``, ``TtV4PrefillRuntime``) over to the traced
decode model (:class:`~..model.DeepSeekV4Model`). The counterpart of :mod:`.handoff`, which hands over *this* directory's
prefill (``DeepSeekV4PrefillModel``); the decode buffers written and the conventions are the same (see its module doc).

Source: the prefill's WORKING state per layer (``runtime.model.layers[i].states[slot]``), which is replicated on every
chip of the prefill mesh and kept in bf16 / fp32 -- not its export caches, which are laid out for the tt-blaze decode ring
(bfp8 tiles, row-major, Hadamard-rotated index keys, gates with the position bias added).

For a prefix of ``T`` tokens, ``T`` a multiple of 128 (the prefill is run on exactly ``T`` tokens; the prompt's ragged tail
is replayed through ``decode_traced`` afterwards, as in ``tests/prefill/test_prefill_decode_demo.py``):

================  ===================================================  ==============================================
layer kind        prefill working state                                decode buffer
================  ===================================================  ==============================================
every layer       ``sliding_carry`` ``[1,1,128,512]``: the last 128    window ring rows ``[0, 128)``: slot ``pos % 128``
                  real K rows in TOKEN order (row j = token T-128+j)   = row j because ``T % 128 == 0``
CSA / HCA         ``compressed_kv`` rows ``[0, entry_count)``          rows ``[128, 128 + T/rate)`` (dense CSA buffer /
                  (entry w RoPE'd at position ``w * rate``)            HCA paged pool through the session page table)
CSA               ``prior_c`` = (kv_a, gate_a + position_bias) fp32    ``prev_kv`` / ``prev_gate`` Ca half: kv_a, and
                  ``[1,1,32,512]``; rows 28..31 = tokens T-4..T-1      gate_a - position_bias[:, :512] (decode adds the
                  (slots 0..3)                                         bias itself inside ``csa_pool_window``)
================  ===================================================  ==============================================

Not handed over (as in :mod:`.handoff`): the lightning indexer's state, so the whole conversation must stay below
``index_topk * 4 = 2048`` tokens until the indexer hand-off lands.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch

import ttnn

from ..common import _MASK_NEG
from .handoff import _assign, _pool_rows, scache_pool_blocks

ALIGNMENT = 128
_SLIDING = "sliding_attention"
_CSA = "compressed_sparse_attention"
_HCA = "heavily_compressed_attention"


@dataclass
class LayerHandoff:
    """One layer's decode-side state on the host, fp32."""

    kind: str
    ring: torch.Tensor  # [128, head_dim] in ring order (slot = pos % 128)
    entries: Optional[torch.Tensor]  # [T / rate, head_dim] (CSA / HCA)
    prev_kv: Optional[torch.Tensor]  # [rate, head_dim] Ca kv of the last closed window (CSA)
    prev_gate: Optional[torch.Tensor]  # [rate, head_dim] Ca gate WITHOUT the position bias (CSA)


def _one_replica(t: ttnn.Tensor) -> torch.Tensor:
    """A replicated mesh tensor -> host torch, from its first device (every chip holds the same rows)."""
    return ttnn.to_torch(ttnn.get_device_tensors(t)[0]).float()


def extract_prefill_handoff(runtime, slot: int, seq_len: int) -> list[LayerHandoff]:
    """Read every layer's working state of ``slot`` after the prefill ran exactly ``seq_len`` tokens into it."""
    if seq_len <= 0 or seq_len % ALIGNMENT:
        raise ValueError(f"prefilled length {seq_len} must be a positive multiple of {ALIGNMENT}")
    out: list[LayerHandoff] = []
    for blk in runtime.model.layers:
        st = blk.states[slot]
        if int(st.kv_actual) != seq_len:
            raise ValueError(f"layer {blk.layer_idx}: the prefill state holds {st.kv_actual} tokens, not {seq_len}")
        kind = blk.kind
        carry = _one_replica(st.sliding_carry)[0, 0]  # [128, Dh], token order
        if carry.shape[0] != ALIGNMENT:
            raise ValueError(f"layer {blk.layer_idx}: sliding carry has {carry.shape[0]} rows, expected {ALIGNMENT}")
        entries = prev_kv = prev_gate = None
        if kind in (_CSA, _HCA):
            n = int(st.entry_count)
            rate = int(blk.attn.compressor.compress_rate)
            if n != seq_len // rate:
                raise ValueError(f"layer {blk.layer_idx} ({kind}): {n} entries for {seq_len} tokens at rate {rate}")
            entries = _one_replica(st.compressed_kv)[0, 0, :n]
        if kind == _CSA:
            comp = blk.attn.compressor
            rate, width = int(comp.compress_rate), int(comp.head_dim)
            pk, pg = st.prior_c
            kv_a = _one_replica(pk)[0, 0, -rate:, :width]  # tokens T-4 .. T-1 = slots 0..3 (T % 4 == 0)
            gate_a = _one_replica(pg)[0, 0, -rate:, :width]  # + position_bias (the a-series half)
            bias_a = comp._bias_host[:, :width].float()  # [rate, 2W] -> the a-series (first) half, row = slot
            prev_kv, prev_gate = kv_a, gate_a - bias_a
        out.append(LayerHandoff(kind=kind, ring=carry, entries=entries, prev_kv=prev_kv, prev_gate=prev_gate))
    return out


def load_ttnn_prefill_into_decode(
    decode_model,
    layers: list[LayerHandoff],
    seq_len: int,
    session_id: int,
    progress: Optional[Callable[[str], None]] = None,
) -> int:
    """Write ``layers`` (from :func:`extract_prefill_handoff`) into ``decode_model``'s buffers, in place; returns ``T``.

    Same preconditions as :func:`.handoff.load_prefill_state_into_decode`: ``prepare_static_decode`` has run, the session is
    active, the traces exist (the throw-away first step). The next ``decode_traced`` continues at position ``T``."""
    note = progress or (lambda message: None)
    config = decode_model.config
    if len(layers) < decode_model.num_layers:
        raise ValueError(f"need {decode_model.num_layers} layers, got {len(layers)}")
    if seq_len <= 0 or seq_len % ALIGNMENT:
        raise ValueError(f"prefilled length {seq_len} must be a positive multiple of {ALIGNMENT}")
    if not decode_model.paged:
        raise RuntimeError("call prepare_static_decode() first")
    if decode_model._indexer_active() and decode_model._index_sparse_step(seq_len):
        raise ValueError(
            f"a prefix of {seq_len} tokens already needs the CSA lightning indexer in decode, whose key cache this hand-off "
            "does not fill yet; hand over at most index_topk * compress_rate - 1 tokens"
        )
    decode_model.ensure_session_capacity(seq_len - 1)
    window = config.sliding_window
    head_dim = config.head_dim
    for sm in decode_model.submeshes_io:
        for li in sm["layers"]:
            lh = layers[li]
            layer_type = config.layer_types[li]
            if lh.kind != layer_type:
                raise ValueError(f"layer {li}: prefill kind {lh.kind} != decode kind {layer_type}")
            scache = sm["scaches"][li]
            note(f"hand-off layer {li + 1}/{decode_model.num_layers} ({layer_type}) -> decode submesh {sm['index']}")
            if layer_type in (_SLIDING, _CSA):
                dense = torch.zeros(tuple(scache.kv.shape))
                rows = window + (0 if lh.entries is None else lh.entries.shape[0])
                if rows > dense.shape[2]:
                    raise ValueError(f"layer {li}: {rows} KV rows do not fit the decode buffer's {dense.shape[2]}")
                dense[0, 0, :window] = lh.ring
                if lh.entries is not None:
                    dense[0, 0, window:rows] = lh.entries
                _assign(scache.kv, dense)
            else:
                assert layer_type == _HCA, layer_type
                pool = _pool_rows(decode_model, layer_type, session_id, scache_pool_blocks(sm, li), lh.ring, lh.entries)
                _assign(sm["pools"][li], pool)
            if layer_type == _CSA:
                rate = lh.prev_kv.shape[0]
                kv_window = torch.zeros(rate, 1, 1, 2 * head_dim)
                gate_window = torch.full((rate, 1, 1, 2 * head_dim), _MASK_NEG)
                kv_window[:, 0, 0, :head_dim] = lh.prev_kv
                gate_window[:, 0, 0, :head_dim] = lh.prev_gate
                _assign(scache.prev_kv, kv_window)
                _assign(scache.prev_gate, gate_window)
    for sm in decode_model.submeshes_io:
        ttnn.synchronize_device(sm["device"])
    note("hand-off done")
    return seq_len
