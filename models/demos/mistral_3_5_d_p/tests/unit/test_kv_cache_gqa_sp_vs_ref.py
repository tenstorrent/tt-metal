# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Write AND read-back for the model's OWN cache shape on the chunked-KV substrate, at target SP x TP
and the spec's real chunk size. Pattern: ``minimax_m3/tests/unit/test_kv_cache_gqa_sp_vs_ref.py``.

``test_kv_cache_write_vs_ref.py`` proves the layout and the production write seam at small,
convenient sizes. This file re-runs the round trip at the geometry the model actually serves —
``chunk_size`` 5120 from the BINDING spec, 8 KV heads on 8 TP columns, 4 SP rows, ``head_dim`` 128,
bfloat8_b — because the things that break only at production scale are exactly the ones the small
case cannot show: the 32-token DRAM bank walk wrapping many times per slot, and the block-cyclic
period being a real chunk rather than the whole cache.

It also pins the four §5.2 decisions as assertions, so a change to the cache shape has to change
this test: two tensors (k, v) and no auxiliary cache, head_dim 128, and the spec's cache dtype.
"""

from __future__ import annotations

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3_d_p.tt.mla.utils import blockcyclic_positions
from models.demos.mistral_3_5_d_p.reference.mistral_config import MistralMedium35Config as C
from models.demos.mistral_3_5_d_p.spec import SPEC
from models.demos.mistral_3_5_d_p.tt.attention import MistralKVCache, allocate_kv_cache, write_kv_chunk
from models.demos.mistral_3_5_d_p.tt.attention.kv_cache import NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK

from ..test_factory import parametrize_mesh, sp_tp_shard_mapper
from .test_kv_cache_write_vs_ref import read_cache_natural

HEAD_DIM, NKV = C.HEAD_DIM, C.NUM_KEY_VALUE_HEADS


@parametrize_mesh()
def test_cache_shape_matches_the_layout_decision(mesh_device, device_params):
    """The four per-model KV decisions (recipe §5.2), as assertions.

    1. two cache tensors — dense GQA, so ``k`` and ``v`` and nothing else;
    2. ``head_dim`` 128, the model's head dim as-is;
    3. ``cache_dtype`` from the spec (bfloat8_b);
    4. no auxiliary cache (no MLA latent, no MSA ``index_k``).
    """
    sp = mesh_device.shape[SPEC.sp_axis]
    num_layers, num_users = 3, 2
    capacity = ttnn.TILE_SIZE * sp * 4
    kv = allocate_kv_cache(
        mesh_device,
        num_layers=num_layers,
        max_seq_len=capacity,
        sp_axis=SPEC.sp_axis,
        num_users=num_users,
        head_dim=HEAD_DIM,
    )
    tensor_fields = [f for f in MistralKVCache.__dataclass_fields__ if isinstance(getattr(kv, f), ttnn.Tensor)]
    assert tensor_fields == ["k", "v"], f"dense GQA allocates exactly k and v; got {tensor_fields}"
    for name in ("k", "v"):
        tensor = getattr(kv, name)
        assert tensor.dtype == SPEC.kv_cache_dtype, f"{name} must use the spec's kv dataformat"
        # Per-chip shape [num_users*num_layers, 1, seq_local, head_dim] — the canonical layout.
        assert tuple(tensor.shape) == (num_users * num_layers, 1, capacity // sp, HEAD_DIM), tuple(tensor.shape)
    assert NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK == 32, "the address-table bank walk assumes 32-token banks"
    logger.info(
        f"KV layout OK: 2 tensors, head_dim={HEAD_DIM}, dtype={SPEC.kv_cache_dtype}, seq_local={capacity // sp}"
    )


@parametrize_mesh()
@pytest.mark.parametrize("n_chunks", [2], ids=["c2"])
def test_kv_cache_gqa_sp_production_chunk(mesh_device, device_params, n_chunks, reset_seeds):
    """Round-trip at the spec's real chunk_size (5120) and a multi-chunk cache, through the
    production ``write_kv_chunk`` seam."""
    rows, cols = tuple(mesh_device.shape)
    sp = rows
    assert cols == NKV, "this layout maps KV head c -> TP column c"
    chunk = SPEC.chunk_size
    chunk_local = chunk // sp
    capacity = chunk * n_chunks
    assert chunk_local % NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK == 0, (
        f"chunk_size {chunk} / sp {sp} = {chunk_local} must be a multiple of "
        f"{NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK} for the DRAM bank walk"
    )

    sent_k = torch.randn(NKV, capacity, HEAD_DIM)
    sent_v = torch.randn(NKV, capacity, HEAD_DIM)

    kv_cache = allocate_kv_cache(
        mesh_device,
        num_layers=1,
        max_seq_len=capacity,
        sp_axis=SPEC.sp_axis,
        num_users=1,
        head_dim=HEAD_DIM,
    )
    mapper = sp_tp_shard_mapper(mesh_device, seq_dim=2, head_dim=1)
    positions = blockcyclic_positions(sp, chunk, capacity)

    for i in range(n_chunks):
        # The cache rows this chunk owns, in per-chip row order — the order the writer expects.
        idx = torch.tensor([int(p) for p in positions if i * chunk <= int(p) < (i + 1) * chunk], dtype=torch.long)
        tt_k = ttnn.from_torch(
            sent_k[:, idx, :].reshape(1, NKV, chunk, HEAD_DIM),
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mapper,
        )
        tt_v = ttnn.from_torch(
            sent_v[:, idx, :].reshape(1, NKV, chunk, HEAD_DIM),
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mapper,
        )
        write_kv_chunk(kv_cache, tt_k, tt_v, slot_idx=0, layer_idx=0, kv_actual=i * chunk, sp_axis=SPEC.sp_axis)
        tt_k.deallocate(True)
        tt_v.deallocate(True)
    ttnn.synchronize_device(mesh_device)

    got_k = read_cache_natural(
        kv_cache.k, mesh_device, slot=0, chunk_size=chunk, capacity=capacity, n_kv=NKV, n_tokens=capacity
    )
    got_v = read_cache_natural(
        kv_cache.v, mesh_device, slot=0, chunk_size=chunk, capacity=capacity, n_kv=NKV, n_tokens=capacity
    )
    ok_k, pcc_k = comp_pcc(sent_k, got_k, SPEC.pcc)
    ok_v, pcc_v = comp_pcc(sent_v, got_v, SPEC.pcc)
    logger.info(
        f"GQA/SP cache round trip at chunk_size={chunk} ({n_chunks} chunks, {capacity} tokens): "
        f"K pcc={pcc_k} V pcc={pcc_v}"
    )
    assert ok_k, f"K mismatch at production chunk size: {pcc_k}"
    assert ok_v, f"V mismatch at production chunk size: {pcc_v}"
