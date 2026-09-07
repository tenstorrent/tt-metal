# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The on-device INDEXED RoPE: whole-cache block-cyclic SP-sharded YaRN cos/sin + the offset
dispatch. Pattern: ``gpt_oss_d_p/tests/unit/test_indexed_rope_vs_ref.py``.

Two independent claims, because they fail in different ways:

  1. **The table math.** ``build_indexed_rope`` block-cyclic-reorders the whole-cache cos/sin and
     SP-shards them, so device ``(r, *)``'s contiguous shard must hold — in local-cache-row order —
     the rope for exactly the global positions it will carry. Checked on host against
     ``blockcyclic_positions``, with no device involved, because an error here is a silent
     per-position phase error rather than a crash.

  2. **The offset dispatch.** ``apply_rope(..., kv_actual_global=offset)`` must apply the same
     rotation the non-indexed op applies with a cos/sin table sliced to ``[offset, offset+chunk)``.
     That is what makes chunk N of a real prefill see positions ``N*chunk ...`` without any host
     reshard.

The raw ``rotary_embedding_indexed`` op has its own op-level tests in ``deepseek_v3_d_p``; what is
Mistral-specific and covered here is the YaRN table (theta 1e6, factor 64, truncated correction
dims) plus the block-cyclic assembly at the spec's SP.
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
from models.demos.mistral_3_5_d_p.tt.attention.operations import apply_rope
from models.demos.mistral_3_5_d_p.tt.rope import (
    build_indexed_rope,
    build_transformation_mat,
    build_yarn_cos_sin,
    yarn_params_from_config,
)

from ..test_factory import parametrize_mesh, sp_tp_shard_mapper

HEAD_DIM = C.HEAD_DIM
YARN = dict(
    rope_theta=C.ROPE_THETA,
    yarn_factor=C.YARN_FACTOR,
    yarn_orig_max_pos=C.YARN_ORIG_MAX_POS,
    yarn_beta_fast=C.YARN_BETA_FAST,
    yarn_beta_slow=C.YARN_BETA_SLOW,
    truncate=C.YARN_TRUNCATE,
)


def test_indexed_rope_table_layout_host():
    """Host-only: each SP shard's rows carry the rope of the global positions that shard will hold.

    ``build_indexed_rope`` = ``build_yarn_cos_sin`` + ``block_cyclic_reorder`` + an SP shard, and
    ``blockcyclic_positions`` is the documented inverse of the cache writer. So shard row ``r`` of
    the reordered table must equal the un-reordered table at row ``blockcyclic_positions[r]``. A
    mismatch is exactly the "scattered specific positions" failure mode that costs long-context K
    PCC and raises nothing.
    """
    from models.demos.deepseek_v3_d_p.tt.mla.utils import block_cyclic_reorder

    sp, chunk_size = SPEC.sp, 512
    capacity = chunk_size * 4
    cos, _sin = build_yarn_cos_sin(capacity, HEAD_DIM, **YARN)
    reordered = block_cyclic_reorder(cos, chunk_size // sp, sp, seq_dim=2)
    positions = blockcyclic_positions(sp, chunk_size, capacity)

    assert torch.equal(torch.sort(positions).values, torch.arange(capacity)), "not a permutation"
    assert not torch.equal(positions, torch.arange(capacity)), f"expected a non-identity reorder at sp={sp}"
    assert torch.equal(reordered[0, 0], cos[0, 0][positions]), "block-cyclic table rows are not at their positions"
    logger.info(f"indexed-rope table layout OK (sp={sp}, chunk={chunk_size}, capacity={capacity})")


def test_yarn_params_come_from_the_config():
    """The table must be built from ``rope_parameters``, not from module defaults.

    Reading the config is what keeps a config edit from silently diverging from the device; this
    pins the wiring, while ``test_reference_config.py`` pins the values against transformers.
    """
    from models.demos.mistral_3_5_d_p.reference.mistral_config import load_text_config

    params = yarn_params_from_config(load_text_config())
    assert params == YARN, f"rope params read from the config {params} != the transcribed constants {YARN}"


@parametrize_mesh()
@pytest.mark.parametrize("n_heads", [8], ids=["h8"])
@pytest.mark.parametrize("chunk, offset", [(512, 512), (5120, 5120)], ids=["c512-off512", "c5120-off5120"])
def test_indexed_rope_matches_nonindexed_at_offset(mesh_device, device_params, n_heads, chunk, offset, reset_seeds):
    """Indexed RoPE at ``kv_actual_global=offset`` == non-indexed RoPE over ``[offset, offset+chunk)``.

    Both sides run the same Q on device, so the comparison isolates the table build + the offset
    dispatch from everything else. ``c5120`` is the spec's real chunk size.
    """
    rows, cols = tuple(mesh_device.shape)
    sp = rows
    assert offset % (ttnn.TILE_SIZE * sp) == 0 and chunk % (ttnn.TILE_SIZE * sp) == 0
    capacity = offset + chunk  # a cache holding the prefix plus this chunk

    q = torch.randn(1, n_heads, chunk, HEAD_DIM)
    xform = build_transformation_mat(mesh_device)
    q_mapper = sp_tp_shard_mapper(mesh_device, seq_dim=2, head_dim=1)

    def q_to_device():
        return ttnn.from_torch(
            q,
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=q_mapper,
        )

    # --- indexed: the whole-cache block-cyclic table + rotary_embedding_indexed at the offset ---
    indexed = build_indexed_rope(
        mesh_device, head_dim=HEAD_DIM, max_seq_len=capacity, chunk_size=chunk, sp_axis=SPEC.sp_axis, **YARN
    )
    out_indexed = apply_rope(q_to_device(), indexed, xform, kv_actual_global=offset, cluster_axis=SPEC.sp_axis)

    # --- non-indexed reference: the same YaRN table, sliced to [offset, offset+chunk) ---
    cos, sin = build_yarn_cos_sin(capacity, HEAD_DIM, **YARN)
    seq_mapper = sp_tp_shard_mapper(mesh_device, seq_dim=2)

    def slice_to_device(t):
        return ttnn.from_torch(
            t[:, :, offset : offset + chunk, :],
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=seq_mapper,
        )

    out_ref = apply_rope(q_to_device(), [slice_to_device(cos), slice_to_device(sin)], xform)

    # Compare every device's shard: the indexed op derives a per-row start from the SP coordinate, so
    # a mistake there shows up on some rows and not others.
    indexed_shards = ttnn.get_device_tensors(out_indexed)
    ref_shards = ttnn.get_device_tensors(out_ref)
    worst = 1.0
    for r in range(rows):
        got = ttnn.to_torch(indexed_shards[r * cols]).float()
        want = ttnn.to_torch(ref_shards[r * cols]).float()
        ok, pcc = comp_pcc(want, got, SPEC.pcc)
        worst = min(worst, float(pcc))
        assert ok, f"indexed RoPE disagrees with the non-indexed reference on SP row {r}: {pcc}"
    logger.info(f"indexed vs non-indexed RoPE @ offset={offset} chunk={chunk}: worst row pcc={worst}")
