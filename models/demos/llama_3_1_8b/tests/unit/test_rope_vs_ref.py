# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""RoPE on device vs the torch reference, across the HF/Meta layout seam.

The device rotates in the **Meta interleaved** column order and the reference in **HF half-split**.
These are the same rotation on differently ordered columns, and the whole package depends on that
being true, so the test drives the full chain explicitly:

    HF-ordered host K --(hf_to_meta_perm)--> device --(rotary_embedding_llama)-->
    read back --(meta_to_hf_perm)--> compare against ref.apply_rope(HF K, HF cos/sin)

Both device paths are covered: the per-chunk ``rotary_embedding_llama`` and the whole-cache
``rotary_embedding_indexed`` that the runtime actually uses. The second is checked at a NON-zero
chunk start, because a table built once for the whole cache and indexed on device is exactly where a
block-cyclic off-by-one hides — at ``start=0`` the block-cyclic permutation is the identity on
device 0 and a broken index still looks right.
"""

from __future__ import annotations

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.llama_3_1_8b.reference import model as ref
from models.demos.llama_3_1_8b.tests.common import (
    assert_pcc,
    cfg_full,
    galaxy_mesh,
    pcc,
    spec_mesh_config,
)
from models.demos.llama_3_1_8b.tt.attention.operations import apply_rope
from models.demos.llama_3_1_8b.tt.rope import RopeSetup
from models.demos.llama_3_1_8b.utils.rope_layout import hf_to_meta_perm, meta_to_hf_perm

N_HEADS_LOCAL = 2  # what one chip carries at num_key_value_heads=8, tp=4
CHUNK = 5120


def _to_device_meta(host_hf, mesh_device, mc, perm):
    """Permute an HF-ordered ``[1, heads, s, d]`` host tensor into Meta order and SP-shard it."""
    return ttnn.from_torch(
        host_hf[..., perm],
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mc.sequence_parallel(mesh_device, seq_dim=2),
    )


def _from_device_hf(tt, mesh_device, mc, inv_perm):
    dims = [None, None]
    dims[mc.sp_axis] = 2
    dims[mc.tp_axis] = 0
    composer = ttnn.ConcatMesh2dToTensor(mesh_device, dims=tuple(dims), mesh_shape=mesh_device.shape)
    out = ttnn.to_torch(tt, mesh_composer=composer).float()[:1]
    return out[..., inv_perm]


@galaxy_mesh()
@pytest.mark.parametrize("start_pos", [0, CHUNK], ids=["chunk0", "chunk1"])
def test_rope_per_chunk_vs_ref(mesh_device, device_params, start_pos, topology_name):
    """``rotary_embedding_llama`` with per-chunk SP-sharded cos/sin."""
    cfg = cfg_full()
    mc = spec_mesh_config(mesh_device)
    perm, inv = hf_to_meta_perm(cfg.head_dim), meta_to_hf_perm(cfg.head_dim)
    rope = RopeSetup(mesh_device, cfg, mc)

    torch.manual_seed(0)
    k_hf = torch.randn(1, N_HEADS_LOCAL, CHUNK, cfg.head_dim)
    tt_k = _to_device_meta(k_hf, mesh_device, mc, perm)
    rotated = apply_rope(tt_k, rope.chunk_cos_sin(CHUNK, start_pos), rope.transformation_mat)
    got = _from_device_hf(rotated, mesh_device, mc, inv)

    cos, sin = ref.rope_cos_sin(cfg, CHUNK, start_pos=start_pos, dtype=torch.float32)
    expected = ref.apply_rope(k_hf, cos, sin)
    assert_pcc(f"rope_per_chunk[{topology_name}] start={start_pos}", pcc(expected, got))


@galaxy_mesh()
@pytest.mark.parametrize("start_pos", [0, CHUNK], ids=["chunk0", "chunk1"])
def test_rope_indexed_vs_ref(mesh_device, device_params, start_pos, topology_name):
    """``rotary_embedding_indexed`` over the whole-cache block-cyclic table.

    The input is the chunk laid out exactly as the model presents it: each SP row holds its own
    contiguous ``chunk/sp`` slice of the chunk's positions, and the op is told only the global offset.
    """
    cfg = cfg_full()
    mc = spec_mesh_config(mesh_device)
    cache_seq = 2 * CHUNK
    perm, inv = hf_to_meta_perm(cfg.head_dim), meta_to_hf_perm(cfg.head_dim)
    rope = RopeSetup(mesh_device, cfg, mc)
    indexed = rope.build_indexed_rope(cache_seq, CHUNK)

    torch.manual_seed(0)
    k_hf = torch.randn(1, N_HEADS_LOCAL, CHUNK, cfg.head_dim)
    tt_k = _to_device_meta(k_hf, mesh_device, mc, perm)
    rotated = apply_rope(
        tt_k, indexed, rope.transformation_mat, kv_actual_global=start_pos, cluster_axis=mc.sp_axis
    )
    got = _from_device_hf(rotated, mesh_device, mc, inv)

    cos, sin = ref.rope_cos_sin(cfg, CHUNK, start_pos=start_pos, dtype=torch.float32)
    expected = ref.apply_rope(k_hf, cos, sin)
    assert_pcc(f"rope_indexed[{topology_name}] start={start_pos}", pcc(expected, got))


@galaxy_mesh()
def test_rope_indexed_matches_per_chunk(mesh_device, device_params):
    """The two device paths must agree — they are used interchangeably across chunk boundaries."""
    cfg = cfg_full()
    mc = spec_mesh_config(mesh_device)
    perm = hf_to_meta_perm(cfg.head_dim)
    rope = RopeSetup(mesh_device, cfg, mc)
    indexed = rope.build_indexed_rope(2 * CHUNK, CHUNK)

    torch.manual_seed(3)
    k_hf = torch.randn(1, N_HEADS_LOCAL, CHUNK, cfg.head_dim)
    for start in (0, CHUNK):
        a = apply_rope(_to_device_meta(k_hf, mesh_device, mc, perm), rope.chunk_cos_sin(CHUNK, start), rope.transformation_mat)
        b = apply_rope(
            _to_device_meta(k_hf, mesh_device, mc, perm),
            indexed,
            rope.transformation_mat,
            kv_actual_global=start,
            cluster_axis=mc.sp_axis,
        )
        inv = meta_to_hf_perm(cfg.head_dim)
        ta = _from_device_hf(a, mesh_device, mc, inv)
        tb = _from_device_hf(b, mesh_device, mc, inv)
        p = pcc(ta, tb)
        logger.info(f"indexed vs per-chunk rope at start={start}: PCC {p:.6f}")
        assert p > 0.9999, f"the two rope paths disagree at start={start}"


@galaxy_mesh()
def test_rope_position_shift_is_detectable(mesh_device, device_params):
    """Negative control: rotating chunk 1 with chunk 0's positions must NOT match.

    Without this, an indexed-rope path that silently ignored ``kv_actual_global`` would pass every
    other test in this file — chunk 0 is the case where the two agree.
    """
    cfg = cfg_full()
    mc = spec_mesh_config(mesh_device)
    perm, inv = hf_to_meta_perm(cfg.head_dim), meta_to_hf_perm(cfg.head_dim)
    rope = RopeSetup(mesh_device, cfg, mc)

    torch.manual_seed(0)
    k_hf = torch.randn(1, N_HEADS_LOCAL, CHUNK, cfg.head_dim)
    rotated = apply_rope(
        _to_device_meta(k_hf, mesh_device, mc, perm), rope.chunk_cos_sin(CHUNK, CHUNK), rope.transformation_mat
    )
    got = _from_device_hf(rotated, mesh_device, mc, inv)

    cos0, sin0 = ref.rope_cos_sin(cfg, CHUNK, start_pos=0, dtype=torch.float32)
    wrong = ref.apply_rope(k_hf, cos0, sin0)
    p_wrong = pcc(wrong, got)
    logger.info(f"chunk-1 output vs chunk-0 positions: PCC {p_wrong:.6f}")
    assert p_wrong < 0.9, "rope output does not depend on the chunk's absolute positions"
