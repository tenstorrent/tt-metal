# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""A 2-chunk sequence pushed through the SAME ``Attention`` module twice; the second chunk's output
must match a reference that saw the whole sequence at once.
Pattern: ``minimax_m3/tests/unit/test_attention_chunked_vs_ref.py``.

This is the row that proves the cache-read path is WIRED, not merely callable. The pieces below it
all pass on their own — the cache write (``test_kv_cache_write_vs_ref``), the ring cache-read op
(``test_ring_joint_cache_read_sp_vs_ref``), the indexed rope's offset dispatch
(``test_indexed_rope_vs_ref``) — and this is where the module has to compose them: chunk 0 writes
its K/V and attends itself, chunk 1 is delivered at ``cached_len = chunk`` and must attend chunk 0's
K/V out of the cache with the rope positions continuing from where chunk 0 stopped.

Any one of three mistakes fails here and passes everywhere else: forgetting to pass ``cached_len``
into the rope (chunk 1 rotated as if it were at position 0), folding the layer into the cache batch
index wrongly, or an ``actual_start`` that is not the write offset.

Both chunks go through the production ``Attention.__call__``, at the spec's SP=4 x TP=8 and the
whole-cache indexed rope.
"""

from __future__ import annotations

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.mistral_3_5_d_p.reference import model as reference
from models.demos.mistral_3_5_d_p.reference.mistral_config import MistralMedium35Config as C
from models.demos.mistral_3_5_d_p.reference.mistral_config import reduced_text_config
from models.demos.mistral_3_5_d_p.spec import SPEC
from models.demos.mistral_3_5_d_p.tt.attention import allocate_kv_cache
from models.demos.mistral_3_5_d_p.tt.rope import build_indexed_rope

from ..test_factory import build_mesh_and_ccl, parametrize_mesh, sp_tp_shard_mapper
from .test_attention_vs_ref import HEAD_DIM, HIDDEN, build_attention, build_cos_sin, random_attention_weights

YARN = dict(
    rope_theta=C.ROPE_THETA,
    yarn_factor=C.YARN_FACTOR,
    yarn_orig_max_pos=C.YARN_ORIG_MAX_POS,
    yarn_beta_fast=C.YARN_BETA_FAST,
    yarn_beta_slow=C.YARN_BETA_SLOW,
    truncate=C.YARN_TRUNCATE,
)


@parametrize_mesh()
@pytest.mark.parametrize("chunk", [256, 512], ids=["c256", "c512"])
def test_attention_chunked_vs_ref(mesh_device, device_params, chunk, reset_seeds):
    """Two chunks through one Attention module; the second chunk's output vs a whole-sequence golden.

    The SP layout makes each chunk's tokens arrive block-cyclic over the SP rows, and at a single
    chunk period that reduces to a contiguous split — so a chunk's rows are ``[c*chunk + r*chunk/sp,
    ...)`` on row ``r``, and the golden is compared per SP row rather than after a re-gather, which
    keeps the block-cyclic ordering out of the comparison.
    """
    rows, cols = tuple(mesh_device.shape)
    sp = rows
    assert (rows, cols) == SPEC.mesh_shape
    n_chunks = 2
    total = chunk * n_chunks
    chunk_local = chunk // sp

    hf_state = random_attention_weights()
    x = torch.randn(1, total, HIDDEN) * 0.1

    # Golden: the WHOLE sequence at once, HF-convention rope over positions [0, total).
    (cos_hf, sin_hf), _ = build_cos_sin(total)
    ref = reference.attention_reference(
        x, hf_state, reduced_text_config(), cos_sin=(cos_hf.unsqueeze(0), sin_hf.unsqueeze(0))
    ).output  # [1, total, HIDDEN]

    mesh_config, ccl = build_mesh_and_ccl(mesh_device)
    kv_cache = allocate_kv_cache(
        mesh_device,
        num_layers=1,
        max_seq_len=total,
        sp_axis=SPEC.sp_axis,
        num_users=1,
        head_dim=HEAD_DIM,
    )
    attn = build_attention(mesh_device, mesh_config, ccl, hf_state, max_seq_len=total, sequence_parallel=True)
    rope_mats = build_indexed_rope(
        mesh_device, head_dim=HEAD_DIM, max_seq_len=total, chunk_size=chunk, sp_axis=SPEC.sp_axis, **YARN
    )
    seq_mapper = sp_tp_shard_mapper(mesh_device, seq_dim=2)

    outputs = []
    for c in range(n_chunks):
        lo = c * chunk
        # This chunk's tokens in the block-cyclic row order the runtime delivers (identity within a
        # single chunk period), SP-sharded on the sequence dim and replicated across TP.
        chunk_x = x[:, lo : lo + chunk, :].reshape(1, 1, chunk, HIDDEN)
        x_tt = ttnn.from_torch(
            chunk_x,
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=seq_mapper,
        )
        out = attn(x_tt, rope_mats=rope_mats, kv_cache=kv_cache, user_id=0, cached_len=lo, indexed_rope=True)
        ttnn.synchronize_device(mesh_device)
        outputs.append(out)

    # Compare the SECOND chunk (the one that had to read the cache) per SP row. Row r of chunk 1 holds
    # global positions [chunk + r*chunk_local, chunk + (r+1)*chunk_local).
    shards = ttnn.get_device_tensors(outputs[1])
    worst = 1.0
    for r in range(rows):
        got = ttnn.to_torch(shards[r * cols]).float().reshape(1, chunk_local, HIDDEN)
        lo = chunk + r * chunk_local
        want = ref[:, lo : lo + chunk_local, :]
        ok, pcc = comp_pcc(want, got, SPEC.pcc)
        worst = min(worst, float(pcc))
        assert ok, f"chunk 1 output disagrees with the whole-sequence golden on SP row {r}: {pcc}"
    logger.info(f"chunked attention (2 x {chunk}) second-chunk output: worst SP-row pcc={worst}")

    # Chunk 0 is compared too: it takes the cache-backed ring path here (the cache is larger than the
    # first chunk), so passing it is not implied by the one-shot test.
    shards0 = ttnn.get_device_tensors(outputs[0])
    for r in range(rows):
        got = ttnn.to_torch(shards0[r * cols]).float().reshape(1, chunk_local, HIDDEN)
        lo = r * chunk_local
        ok, pcc = comp_pcc(ref[:, lo : lo + chunk_local, :], got, SPEC.pcc)
        assert ok, f"chunk 0 output disagrees with the golden on SP row {r}: {pcc}"
