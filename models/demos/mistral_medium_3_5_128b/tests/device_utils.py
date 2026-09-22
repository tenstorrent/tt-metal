# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host<->mesh conversion and PCC reporting shared by the device tests.

The sharding conventions here are the package's, stated once so no test re-invents them:

* **activations** ``[1, 1, tokens, hidden]`` — sequence over the SP rows, hidden **replicated**
  across the TP cols: ``dims=[-2, None]``.
* **heads** ``[1, heads, tokens, head_dim]`` — heads over the TP cols, sequence over the SP rows:
  ``dims=[-2, -3]``.
* **column-parallel weights** ``[in, out]`` — ``dims=[None, -1]``; **row-parallel** ``dims=[None, -2]``.

SP sharding is plain contiguous ``torch.chunk``-style splitting. That is not an approximation of
the block-cyclic KV layout, it is exactly equal to it at every chunk-aligned offset, which is the
only kind this model produces (see ``tt/rope.py``).
"""

import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc

MESH_SHAPE = (8, 4)  # SP=8 rows, TP=4 cols
L1_SMALL_SIZE = 1152
PCC_LOWER_BOUND = 0.85  # spec acceptance.pcc_lower_bound — asserted
PCC_TARGET = 0.99  # spec acceptance.pcc_target — aimed at, reported when missed


def to_mesh(mesh, tensor, dims, *, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    """Push a torch tensor onto the mesh with a 2D shard spec (``None`` = replicate on that axis)."""
    return ttnn.from_torch(
        tensor,
        dtype=dtype,
        layout=layout,
        device=mesh,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=MESH_SHAPE, dims=dims),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def replicate(mesh, tensor, *, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    """Push the same torch tensor onto every device."""
    return ttnn.from_torch(
        tensor,
        dtype=dtype,
        layout=layout,
        device=mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def from_mesh_replicated(mesh, tt):
    """Read back a tensor that is identical on every device (takes device (0, 0))."""
    out = ttnn.to_torch(tt, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh, mesh_shape=MESH_SHAPE, dims=(0, 1)))
    return out[:1, :1]


def from_mesh_sp(mesh, tt, seq_dim=2):
    """Read back an SP-sharded tensor, concatenating the rows back into one sequence.

    The TP cols hold replicas, so the col axis is folded onto dim 1 and sliced away.
    """
    out = ttnn.to_torch(tt, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh, mesh_shape=MESH_SHAPE, dims=(seq_dim, 1)))
    return out[:, :1]


def from_mesh_2d(mesh, tt, dims):
    """Read back with an explicit ``(row_dim, col_dim)`` concat — for heads-over-TP tensors."""
    return ttnn.to_torch(tt, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh, mesh_shape=MESH_SHAPE, dims=dims))


def assert_pcc(name, ref, out, *, lower=PCC_LOWER_BOUND, target=PCC_TARGET):
    """Compare in fp32, log the measured PCC, assert the spec's lower bound.

    A value that clears ``lower`` but misses ``target`` is logged at WARNING with both numbers —
    that is the record §4 of the recipe asks for, and the README's PCC table is built from these
    lines rather than from a second measurement.
    """
    assert ref.shape == out.shape, f"{name}: shape {tuple(out.shape)} != reference {tuple(ref.shape)}"
    passing, pcc = comp_pcc(ref.float(), out.float(), lower)
    if passing and _pcc_value(pcc) < target:
        logger.warning(f"[pcc] {name}: {pcc} — above lower bound {lower}, below target {target}")
    else:
        logger.info(f"[pcc] {name}: {pcc}")
    assert passing, f"{name}: PCC {pcc} < lower bound {lower}"
    return pcc


def _pcc_value(pcc):
    """``comp_pcc`` returns either a float or a message string; pull the number out of either."""
    if isinstance(pcc, (int, float)):
        return float(pcc)
    try:
        return float(str(pcc).strip().split()[-1])
    except (ValueError, IndexError):
        return float("nan")


def shard_heads_seq(mesh, tensor, *, dtype=ttnn.bfloat16):
    """Push ``[1, heads, tokens, head_dim]`` with heads over TP cols and the sequence over SP rows."""
    return to_mesh(mesh, tensor, dims=[2, 1], dtype=dtype)


def gather_heads_seq(mesh, tt):
    """Read back a heads-over-TP / sequence-over-SP tensor into ``[1, heads, tokens, head_dim]``."""
    return from_mesh_2d(mesh, tt, dims=(2, 1))


def torch_gqa_causal(q, k, v, scale=None):
    """fp32 GQA causal SDPA reference. ``q`` ``[1, nq, s, d]``, ``k``/``v`` ``[1, nkv, s_kv, d]``.

    The queries are treated as the **last** ``s`` positions of the ``s_kv``-long context, which is
    what a chunked call needs and reduces to a plain causal mask when the two are equal.
    """
    nq, nkv, d = q.shape[1], k.shape[1], q.shape[-1]
    rep = nq // nkv
    kf = k.repeat_interleave(rep, dim=1).float()
    vf = v.repeat_interleave(rep, dim=1).float()
    scores = (q.float() @ kf.transpose(-1, -2)) * (d**-0.5 if scale is None else scale)
    s, s_kv = q.shape[2], k.shape[2]
    offset = s_kv - s
    q_pos = torch.arange(s)[:, None] + offset
    k_pos = torch.arange(s_kv)[None, :]
    scores = scores.masked_fill(k_pos > q_pos, float("-inf"))
    return torch.softmax(scores, dim=-1) @ vf


def read_kv_cache(mesh, cache, *, cache_global, chunk_size, upto=None):
    """Read a whole KV cache back into ``[slots, num_kv_heads, upto, head_dim]``, in sequence order.

    The cache is block-cyclic over the SP rows; ``cache_row_index`` is the inverse permutation and
    is the single place that layout is described. Every read-back in the package goes through here,
    so there is exactly one implementation that can be wrong.

    **All slots at once, deliberately.** Composing the mesh tensor is a full device-to-host
    transfer of the cache, and the acceptance run wants all 88 layers: reading slot by slot would
    repeat that transfer 88 times for the same bytes. :func:`read_kv_slot` is the single-slot view
    on top of this, for the block-level tests where one slot is all there is.
    """
    from models.demos.mistral_medium_3_5_128b.tt.attention.kv_cache import cache_row_index

    sp = MESH_SHAPE[0]
    full = from_mesh_2d(mesh, cache, dims=(2, 1))  # [slots, num_kv_heads, sp*seq_local, head_dim]
    rows = cache_row_index(cache_global, sp, chunk_size)[: cache_global if upto is None else upto]
    return full[:, :, rows, :]


def read_kv_slot(mesh, cache, *, slot, cache_global, chunk_size, upto=None):
    """One (user, layer) slot of a KV cache as ``[1, num_kv_heads, upto, head_dim]``."""
    return read_kv_cache(mesh, cache, cache_global=cache_global, chunk_size=chunk_size, upto=upto)[slot : slot + 1]


def sp_chunk(tensor, sp, dim=-2):
    """The per-row contiguous SP shards of ``tensor`` as a list — the host-side mirror of the split
    ``to_mesh(..., dims=[-2, ...])`` performs."""
    assert tensor.shape[dim] % sp == 0, f"dim {dim} = {tensor.shape[dim]} is not divisible by sp={sp}"
    return list(torch.chunk(tensor, sp, dim=dim))
