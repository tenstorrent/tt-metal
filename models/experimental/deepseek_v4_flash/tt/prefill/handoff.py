# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Hand a finished prefill over to the traced decode model.

After :meth:`DeepSeekV4PrefillModel.prefill` the attention state of every layer
(:class:`~.attention.PrefillAttentionState`) lives on the prefill submeshes. Decode keeps the *same
information* in different buffers (``DeepSeekV4Model.prepare_static_decode``): this module rewrites the
prefill state into those buffers, in place, so the next traced decode step continues the prompt as if decode
itself had consumed it. Nothing in the decode model is modified or subclassed; only its persistent device
buffers are overwritten (the traces address them, so they are written *in place*, never re-allocated).

What maps to what, for a prompt of ``T`` tokens (``T`` a multiple of 128 = ``sliding_window``, so no
compressor window is half full and no ring slot is rotated):

================  ==========================================  ==========================================
layer type        prefill state                               decode buffer
================  ==========================================  ==========================================
sliding           ``kv_tail`` ``[1,1,W,Dh]``                  ``scache.kv`` rows ``[0, W)`` (slot ``pos % W``,
                                                              and ``T % W == 0`` so slot ``j`` = token ``T-W+j``)
CSA               + ``compressed_kv`` ``[1,1,T/4,Dh]``        ``scache.kv`` rows ``[W, W + T/4)``
                  + ``csa_prev_kv`` / ``csa_prev_gate``       ``scache.prev_kv`` / ``prev_gate`` ``[cr,1,1,2*Dh]``
                                                              (Ca half; see below)
HCA               ``kv_tail`` + ``compressed_kv`` ``[T/128]`` the layer's paged pool, rows ``[0, W)`` (ring) and
                                                              ``[W, W + T/128)`` through the session's page table
================  ==========================================  ==========================================

The one representation difference is CSA's overlap gate. Prefill keeps the previous window's gate with the
compressor ``position_bias`` already added (it pools in fp32 and splits Ca/Cb after the add); decode keeps the
raw projection and adds the bias inside ``csa_pool_window``. So ``prev_gate`` is written as
``prefill_gate - position_bias[:, :Dh]``. Only the Ca half of the previous window is ever pooled, so the Cb
half of ``prev_*`` is left neutral (zero kv, ``_MASK_NEG`` gate). The half-open current windows
(``win_kv`` / ``win_gate``) need nothing: at an aligned ``T`` the next token starts a fresh window and decode
writes every slot of it before pooling.

The prompt's ragged tail (``len % 128`` tokens) is not part of a prefill; feed it to ``decode_traced`` one token
at a time afterwards.

Not handed over: the CSA lightning indexer's key cache (prefill skips the indexer). That is only ever read once
the sequence reaches ``index_topk * compress_rate`` tokens (2048), so decode must stay below that: the
function refuses a prefill that already reaches it, and the caller is responsible for the generation length.
"""

import math
from typing import Callable, Optional

import torch

import ttnn

from ..common import _MASK_NEG
from .attention import ALIGNMENT, PrefillAttentionState
from .model import DeepSeekV4PrefillModel

_SLIDING = "sliding_attention"
_CSA = "compressed_sparse_attention"
_HCA = "heavily_compressed_attention"


def _replicate(device):
    return ttnn.ReplicateTensorToMesh(device) if device.get_num_devices() > 1 else None


def _assign(dst: ttnn.Tensor, host: torch.Tensor) -> None:
    """Overwrite the persistent device tensor ``dst`` with ``host`` (same shape), keeping its address.

    Staged through a DRAM-interleaved temporary that is freed right away. A destination in another memory
    config (the CSA windows are L1 width-sharded) is filled with ``to_memory_config(..., output_tensor=)``,
    exactly as the decode model swaps those windows itself.
    """
    if tuple(host.shape) != tuple(dst.shape):
        raise ValueError(f"cannot write a {tuple(host.shape)} tensor into a {tuple(dst.shape)} buffer")
    device = dst.device()
    staged = ttnn.from_torch(
        host.to(torch.float32),
        dtype=dst.dtype,
        layout=dst.layout,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=_replicate(device),
    )
    if dst.memory_config() == staged.memory_config():
        ttnn.copy(staged, dst)
    else:
        ttnn.to_memory_config(staged, dst.memory_config(), output_tensor=dst)
    ttnn.deallocate(staged)


def _pool_rows(
    decode_model, group_name: str, session_id: int, num_blocks: int, ring: torch.Tensor, entries: torch.Tensor
) -> torch.Tensor:
    """A paged layer's whole pool ``[num_blocks, 1, block, Dh]`` holding ring + entries at the session's blocks."""
    group = decode_model._paged_groups[group_name]
    page_row = decode_model._require_paged().page_row(session_id, group_name)[0].tolist()
    window, block = group.sliding_window, group.block_size
    rows = window + entries.shape[0]
    blocks = math.ceil(rows / block)
    axis = torch.zeros(blocks * block, ring.shape[-1])
    axis[:window] = ring
    axis[window:rows] = entries
    pool = torch.zeros(num_blocks, 1, block, ring.shape[-1])
    for logical in range(blocks):
        physical = page_row[logical]
        if physical == 0:
            raise RuntimeError(f"logical block {logical} of {group_name} is unmapped: capacity was not ensured")
        pool[physical, 0] = axis[logical * block : (logical + 1) * block]
    return pool


def load_prefill_state_into_decode(
    decode_model,
    prefill_model: DeepSeekV4PrefillModel,
    states: list[PrefillAttentionState],
    session_id: int,
    progress: Optional[Callable[[str], None]] = None,
) -> int:
    """Write ``states`` (from ``prefill_model.prefill``) into ``decode_model``'s buffers; returns the prompt length ``T``.

    ``decode_model`` must have run :meth:`prepare_static_decode` and have ``session_id`` active (see
    ``activate_session``), ideally after its first step so its traces exist (the buffers are then written
    in place, and the compile runs' scratch writes are overwritten rather than the other way round). The next
    ``decode_traced`` continues at position ``T``.
    """
    note = progress or (lambda message: None)
    config = decode_model.config
    if len(states) < decode_model.num_layers:
        raise ValueError(f"need {decode_model.num_layers} layer states, got {len(states)}")
    lengths = {state.seq_len for state in states[: decode_model.num_layers]}
    if len(lengths) != 1:
        raise ValueError(f"layer states disagree on the prompt length: {sorted(lengths)}")
    total = lengths.pop()
    if total <= 0 or total % ALIGNMENT:
        raise ValueError(f"prefilled length {total} must be a positive multiple of {ALIGNMENT}")
    if not decode_model.paged:
        raise RuntimeError("call prepare_static_decode() first")
    if decode_model._indexer_active() and decode_model._index_sparse_step(total):
        raise ValueError(
            f"a prompt of {total} tokens already needs the CSA lightning indexer in decode, whose key cache "
            "prefill does not fill; hand over at most index_topk * compress_rate - 1 tokens"
        )

    # Blocks for every row the prompt occupies in the paged (HCA) pools, and their page tables.
    decode_model.ensure_session_capacity(total - 1)

    window = config.sliding_window
    head_dim = config.head_dim
    for sm in decode_model.submeshes_io:
        for li in sm["layers"]:
            state = states[li]
            layer_type = config.layer_types[li]
            attn = prefill_model.layers[li].self_attn
            pdev = prefill_model.layer_devices[li]
            host = prefill_model.to_host
            ring = host(state.kv_tail, pdev)[0, 0]  # [W, Dh]
            entries = host(state.compressed_kv, pdev)[0, 0] if state.compressed_kv is not None else None
            scache = sm["scaches"][li]
            note(f"hand-off layer {li + 1}/{decode_model.num_layers} ({layer_type}) -> decode submesh {sm['index']}")

            if layer_type in (_SLIDING, _CSA):
                dense = torch.zeros(tuple(scache.kv.shape))
                rows = window + (0 if entries is None else entries.shape[0])
                if rows > dense.shape[2]:
                    raise ValueError(
                        f"layer {li}: {rows} KV rows do not fit the decode buffer's {dense.shape[2]} "
                        f"(the dense CSA buffer holds {dense.shape[2] - window} compressed entries)"
                    )
                dense[0, 0, :window] = ring
                if entries is not None:
                    dense[0, 0, window:rows] = entries
                _assign(scache.kv, dense)
            else:
                assert layer_type == _HCA, layer_type
                pool = _pool_rows(decode_model, layer_type, session_id, scache_pool_blocks(sm, li), ring, entries)
                _assign(sm["pools"][li], pool)

            if layer_type == _CSA:
                rate = attn.rate
                prev_kv = host(state.csa_prev_kv, pdev).reshape(rate, head_dim)  # slot-major Ca kv
                prev_gate = host(state.csa_prev_gate, pdev).reshape(rate, head_dim)  # + position_bias
                bias_ca = torch.stack([host(b, pdev).reshape(-1)[:head_dim] for b in attn.c_bias_slots])  # [rate, Dh]
                kv_window = torch.zeros(rate, 1, 1, 2 * head_dim)
                gate_window = torch.full((rate, 1, 1, 2 * head_dim), _MASK_NEG)
                kv_window[:, 0, 0, :head_dim] = prev_kv
                gate_window[:, 0, 0, :head_dim] = prev_gate - bias_ca
                _assign(scache.prev_kv, kv_window)
                _assign(scache.prev_gate, gate_window)
    for sm in decode_model.submeshes_io:
        ttnn.synchronize_device(sm["device"])
    note("hand-off done")
    return total


def scache_pool_blocks(sm: dict, li: int) -> int:
    """Blocks in layer ``li``'s paged pool ``[num_blocks, 1, block, Dh]``."""
    return sm["pools"][li].shape[0]
