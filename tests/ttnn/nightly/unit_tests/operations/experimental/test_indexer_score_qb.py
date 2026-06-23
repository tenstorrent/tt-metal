# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Multi-device (QuietBox, 4 BH) functional test for ``indexer_score`` with a PER-DEVICE chunk_start.

This exercises the deployment shape that motivated making chunk_start a runtime value: a SINGLE mesh
dispatch where each device is a different SP ring position and therefore needs its own chunk_start.
The per-device value is derived host-side from each device's mesh COORDINATE (ring_joint_sdpa style):
device r uses ``chunk_start_idx + r * Sq`` (r = linearized index, Sq = per-device query count, derived
from the q seq len -- not passed). One compiled program serves all four devices and all chunk steps --
chunk_start is excluded from the program hash.

Scaled-down GLX chunked prefill (SP=4 instead of the deployed SP=8) so 640 queries/device is preserved:

    history = 25600 (25k)   chunk = 2560 (2.5k = 4 * 640)   T = history + chunk = 28160 keys
    device r (r in 0..3) handles queries [r*640, (r+1)*640) with chunk_start = history + r*640

Functional only (exact -inf map + PCC >= 0.999 per device); no perf assertions.
"""

import pytest
import torch

import ttnn

# Reuse the single-device reference, the match assertion, and the GLX knobs.
from tests.ttnn.nightly.unit_tests.operations.experimental.test_indexer_score import (
    indexer_score_ref,
    assert_indexer_match,
    glx_config,
)

pytestmark = pytest.mark.skipif(not ttnn.device.is_blackhole(), reason="indexer_score is Blackhole-only")

QB_DIM = 128  # indexer head dim
QB_SQ = 640  # queries per device (preserved from the SP=8 deployment)
QB_SP = 4  # devices (QuietBox); SP ring positions 0..3
QB_CHUNK = QB_SP * QB_SQ  # 2560 chunk queries (2.5k), sharded SP=4 -> 640/device
QB_HISTORY = 25600  # 25k history, tile-aligned (800 tiles)
QB_T = QB_HISTORY + QB_CHUNK  # 28160 all-gathered keys (880 tiles)

# GLM5 (8 heads) and DSv32 (16 heads), as in the single-device deployment cases.
QB_CASES = [("glm5", 8), ("dsv32", 16)]
QB_IDS = [c[0] for c in QB_CASES]


def _shard_inputs(mesh_device, heads, seed):
    """Global GLX inputs sharded across the SP=4 mesh: q/w along seq (each device its own 640 rows),
    k replicated (the all-gathered keys are identical on every device). Deployed dtypes (bf16 q, bfp8 k)."""
    g = torch.Generator().manual_seed(seed)
    q_g = torch.randn(1, heads, QB_CHUNK, QB_DIM, generator=g, dtype=torch.bfloat16)
    k_g = torch.randn(1, 1, QB_T, QB_DIM, generator=g, dtype=torch.bfloat16)
    w_g = torch.randn(1, heads, QB_CHUNK, 1, generator=g, dtype=torch.bfloat16)
    q_dev = ttnn.from_torch(
        q_g,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=2),
    )
    k_dev = ttnn.from_torch(
        k_g,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat8_b,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    w_dev = ttnn.from_torch(
        w_g,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=2),
    )
    return q_g, k_g, w_g, q_dev, k_dev, w_dev


@pytest.mark.parametrize("mesh_device", [QB_SP], indirect=True)
@pytest.mark.parametrize("case_id, heads", QB_CASES, ids=QB_IDS)
def test_indexer_score_qb_per_device_chunk_start(mesh_device, case_id, heads):
    """One mesh dispatch over 4 BH devices, each deriving its own chunk_start from its coordinate via
    chunk_start_stride. Validate each device's output against its own chunk_start reference."""
    if mesh_device.get_num_devices() < QB_SP:
        pytest.skip(f"requires {QB_SP} devices, have {mesh_device.get_num_devices()}")

    q_g, k_g, w_g, q_dev, k_dev, w_dev = _shard_inputs(mesh_device, heads, seed=42)

    # Multichip: chunk_start_idx is OMITTED -> the op deduces base = T - num_devices*Sq = QB_HISTORY,
    # then device r (linear order, cluster_axis=None) gets base + r*Sq. No chunk_start passed at all.
    out = ttnn.experimental.indexer_score(
        q_dev,
        k_dev,
        w_dev,
        program_config=glx_config(heads),
    )
    # Reassemble the per-device [1,1,640,T] outputs back into the global [1,1,2560,T].
    out_t = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=2))

    # Per-device reference, each with that device's own chunk_start, concatenated along seq.
    refs = []
    for r in range(QB_SP):
        q_r = q_g[:, :, r * QB_SQ : (r + 1) * QB_SQ, :]
        w_r = w_g[:, :, r * QB_SQ : (r + 1) * QB_SQ, :]
        refs.append(indexer_score_ref(q_r, k_g, w_r, QB_HISTORY + r * QB_SQ))
    ref = torch.cat(refs, dim=2)

    assert_indexer_match(out_t, ref, QB_CHUNK, QB_T, check_neg=True)


@pytest.mark.parametrize("mesh_device", [QB_SP], indirect=True)
@pytest.mark.parametrize("case_id, heads", QB_CASES, ids=QB_IDS)
def test_indexer_score_qb_one_compile_all_chunk_starts(mesh_device, case_id, heads):
    """chunk_start is excluded from the program hash: running several different (base, stride) pairs must
    add exactly ONE program-cache entry (the first compile), proving no per-value recompile."""
    if mesh_device.get_num_devices() < QB_SP:
        pytest.skip(f"requires {QB_SP} devices, have {mesh_device.get_num_devices()}")

    _, _, _, q_dev, k_dev, w_dev = _shard_inputs(mesh_device, heads, seed=7)

    # Three distinct chunk-start bases (all within the causal window). Only the first should compile.
    bases = [QB_HISTORY, QB_HISTORY - QB_SQ, QB_HISTORY - 2 * QB_SQ]

    entries_before = mesh_device.num_program_cache_entries()
    for base in bases:
        ttnn.experimental.indexer_score(
            q_dev,
            k_dev,
            w_dev,
            chunk_start_idx=base,
            program_config=glx_config(heads),
        ).deallocate()

    added = mesh_device.num_program_cache_entries() - entries_before
    assert added == 1, f"expected 1 program-cache entry across 3 distinct chunk_start bases, got {added}"
