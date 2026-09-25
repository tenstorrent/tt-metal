# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in (TEST_SDPA_RECIPE_BLOCKING_PERF=1): op-selected recipe blocking vs the models' tuned chunks.

Two connected Blackholes as a 1x2 FABRIC_1D_RING mesh (the ring/exp-ring perf fixture). Each case runs
one recipe variant twice on the same inputs, once with the chunks left to the op and once with the
chunks the tt_dit model used before op-selected blocking (its tuned table mapped through the old
recipe helper), and records the median trace wall time of each. Shapes are per-device DiT lengths:
dense/joint run on one chip's full sequence, ring/exp ring on a two-chip ring with the per-device
length a larger mesh would give each chip. A tuned configuration the op rejects (e.g. L1) is recorded
as rejected; the auto run must not be rejected.
"""

import math
import os

import pytest
import torch
import ttnn

from .sdpa_recipe_test_utils import VARIANTS, prepare
from .test_sdpa_recipe_blocking import blocking_mesh, precision_of, randn, recipe_kwargs  # noqa: F401
from .test_sdpa_recipe_ring_perf import time_trace

pytestmark = pytest.mark.skipif(os.getenv("TEST_SDPA_RECIPE_BLOCKING_PERF") != "1", reason="Opt-in blocking perf")
T = ttnn._ttnn.operations.transformer
SELECTED = os.getenv("PERF_VARIANTS", ",".join(VARIANTS)).split(",")


def old_recipe_chunks(q, k, *, exp_ring=False):
    """models/tt_dit/utils/sdpa_recipe.py before op-selected blocking."""
    q = q if q % 32 == 0 and 128 <= q <= 320 else 256
    k = 512 if exp_ring else (k if k in (256, 384, 512) else 512)
    return q, k


# name, op, heads, q rows, k rows, joint rows, head dim, model-tuned (q, k) before auto blocking
DENSE_CASES = [
    # FLUX.1 1024x1024 (4096 image + 512 text), 24 heads, tp=1: blocks/attention_opt default (128, 512).
    ("flux1_1024_joint", "joint", 24, 4096, 4096, 512, 128, (128, 512)),
    # Wan 2.2 480p cross-attention per chip on a 1x2 mesh (sp=2): Q256/K512 recipe config, 512 text keys.
    ("wan480_cross_sp2", "dense", 40, 16384, 512, 0, 128, (256, 512)),
    # Ideogram4 D256: tuned (128, 256), L1-limited.
    ("ideogram4_d256", "dense", 8, 4096, 4096, 0, 256, (128, 256)),
    # LTX-2 audio D64 self-attention.
    ("ltx_audio_d64", "dense", 32, 1024, 1024, 0, 64, (256, 512)),
    # Short-K cross attention (one K block per Q chunk): LTX-2 text (32 keys) and A2V (256 keys) cross at
    # 4864 video rows (legacy tuned (192, 128) / (192, 256)), Wan 2.2 720p cross (512 text keys).
    ("ltx_text_cross", "dense", 8, 4864, 32, 0, 128, (192, 128)),
    ("ltx_a2v_cross", "dense", 8, 4864, 256, 0, 128, (192, 256)),
    ("wan720_cross", "dense", 10, 9472, 512, 0, 128, (256, 512)),
]
# Cases whose tuned chunks are the model's legacy SDPA config, run as-is (recipes take any tile-aligned
# chunk; the old recipe helper would have mapped K128 to K512).
LEGACY_TUNED = {"ltx_text_cross", "ltx_a2v_cross"}
# name, heads, per-device rows, model-tuned (q, k)
RING_CASES = [
    ("wan480_8x4", 10, 4096, (288, 512)),  # Wan (True, 8, 4) -> (288, 512)
    ("wan720_8x4", 10, 9472, (288, 512)),
    ("wan480_2x2", 20, 8192, (128, 512)),  # Wan (True, 2, 2) -> (128, 512)
    ("h3_5s", 14, 4768, (320, 384)),  # MiniMax H3 measured_sdpa_chunk_sizes
    ("h3_10s", 14, 9216, (256, 512)),
]
EXP_CASES = [
    ("wan720_4x32", 10, 2368),  # Wan (True, 32, 4) tuned 224 does not fill 10 SDPA columns here
    ("h3_4x32", 14, 1216),
]


def h3_exp_search(seq_local, heads, grid):
    """MiniMax H3's recipe exp-ring search before op-selected blocking (the tuned exp baseline)."""
    best = None
    for cols in range(grid.x - 1, 1, -1):
        for segs in (1, 2, 3):
            chunks = cols * segs
            q = math.ceil(math.ceil(seq_local / chunks) / 32) * 32
            if math.ceil(seq_local / q) != chunks or not 128 <= q <= 320:
                continue
            passes = math.ceil(heads * segs / grid.y)
            if passes > 3:
                continue
            score = (passes * q, -cols)
            if best is None or score < best[0]:
                best = (score, cols, q)
    return None if best is None else (best[1] + 1, best[2])


def measure(record_property, label, config, invoke):
    try:
        median, _minimum = time_trace(invoke.mesh, lambda: invoke(config))
    except RuntimeError as error:
        record_property(f"{label}_rejected", str(error).split("\n")[0][:200])
        return None
    record_property(f"{label}_ms", round(median, 4))
    return median


def compare(record_property, invoke, auto_config, tuned_config, resolved):
    record_property("auto_chunks", f"Q{resolved.q_chunk_size}/K{resolved.k_chunk_size}")
    record_property("auto_grid", str((resolved.compute_with_storage_grid_size.x, resolved.compute_with_storage_grid_size.y)))
    if tuned_config is not None:
        record_property("tuned_chunks", f"Q{tuned_config.q_chunk_size}/K{tuned_config.k_chunk_size}")
    auto = measure(record_property, "auto", auto_config, invoke)
    assert auto is not None, "op-selected blocking was rejected"
    tuned = None if tuned_config is None else measure(record_property, "tuned", tuned_config, invoke)
    if tuned is not None:
        record_property("auto_over_tuned", round(auto / tuned, 4))


class Invoke:
    def __init__(self, mesh, fn):
        self.mesh, self.fn = mesh, fn

    def __call__(self, config):
        return self.fn(config)


@pytest.mark.parametrize("variant", SELECTED)
@pytest.mark.parametrize("case", DENSE_CASES, ids=[c[0] for c in DENSE_CASES])
def test_dense_blocking_perf(blocking_mesh, case, variant, record_property):
    mesh, *_ = blocking_mesh
    name, op, heads, q_rows, k_rows, joint, head_dim, tuned = case
    shapes = [(1, heads, n, head_dim) for n in (q_rows, k_rows, k_rows)] + ([(1, heads, joint, head_dim)] * 3 if joint else [])
    inputs = [ttnn.from_torch(randn(s, i), device=mesh, layout=ttnn.TILE_LAYOUT, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh)) for i, s in enumerate(shapes)]
    if variant.startswith("E_"):
        inputs = [*prepare(inputs[:3], variant), *(prepare(inputs[3:], variant) if joint else [])]
    options = recipe_kwargs(variant)
    grid = mesh.compute_with_storage_grid_size()
    auto_config = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=grid)
    q, k = tuned if name in LEGACY_TUNED else old_recipe_chunks(*tuned)
    tuned_config = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=grid, q_chunk_size=q, k_chunk_size=k)
    if joint:
        resolved = T._sdpa_recipe_resolved_program_config(
            "joint", options["precision"], inputs[0], inputs[1], joint_q=inputs[3], joint_k=inputs[4], program_config=auto_config
        )
        fn = lambda config: ttnn.transformer.joint_scaled_dot_product_attention(
            *inputs, joint_strategy="rear", program_config=config, **options
        )
    else:
        resolved = T._sdpa_recipe_resolved_program_config("dense", options["precision"], inputs[0], inputs[1], program_config=auto_config)
        fn = lambda config: ttnn.transformer.scaled_dot_product_attention(*inputs, is_causal=False, program_config=config, **options)
    compare(record_property, Invoke(mesh, fn), auto_config, tuned_config, resolved)


def ring_inputs(mesh, heads, local, variant):
    shard = ttnn.ShardTensorToMesh(mesh, dim=2)
    inputs = [ttnn.from_torch(randn((1, heads, 2 * local, 128), i), device=mesh, layout=ttnn.TILE_LAYOUT, mesh_mapper=shard) for i in range(3)]
    inputs = prepare(inputs, variant) if variant.startswith("E_") else inputs
    backing = [
        ttnn.allocate_tensor_on_device([1, heads, 2 * local, 128], x.dtype, ttnn.TILE_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG)
        for x in inputs[1:]
    ]
    return inputs, backing


@pytest.mark.parametrize("variant", SELECTED)
@pytest.mark.parametrize("case", RING_CASES, ids=[c[0] for c in RING_CASES])
def test_ring_blocking_perf(blocking_mesh, case, variant, record_property):
    mesh, subdevice, semaphores, full = blocking_mesh
    name, heads, local, tuned = case
    inputs, backing = ring_inputs(mesh, heads, local, variant)
    options = recipe_kwargs(variant)
    grid = (full.x - 1, full.y)
    auto_config = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=grid)
    q, k = old_recipe_chunks(*tuned)
    tuned_config = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=grid, q_chunk_size=q, k_chunk_size=k)
    resolved = T._sdpa_recipe_resolved_program_config(
        "ring", options["precision"], inputs[0], inputs[1], program_config=auto_config, ring_size=2
    )

    def fn(config):
        return ttnn.transformer.ring_joint_scaled_dot_product_attention(
            *inputs,
            None,
            None,
            None,
            persistent_output_buffer_k=backing[0],
            persistent_output_buffer_v=backing[1],
            joint_strategy="rear",
            logical_n=2 * local,
            logical_l=0,
            is_causal=False,
            dim=2,
            multi_device_global_semaphore=semaphores,
            num_links=1,
            cluster_axis=1,
            mesh_device=mesh,
            topology=ttnn.Topology.Linear,
            subdevice_id=subdevice,
            ccl_core_grid_offset=(full.x - 1, 0),
            use_column_major_ccl=True,
            program_config=config,
            **options,
        )

    compare(record_property, Invoke(mesh, fn), auto_config, tuned_config, resolved)


@pytest.mark.parametrize("variant", SELECTED)
@pytest.mark.parametrize("case", EXP_CASES, ids=[c[0] for c in EXP_CASES])
def test_exp_ring_blocking_perf(blocking_mesh, case, variant, record_property):
    mesh, subdevice, semaphores, full = blocking_mesh
    name, heads, local = case
    inputs, backing = ring_inputs(mesh, heads, local, variant)
    options = recipe_kwargs(variant)
    auto_config = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=full)
    searched = h3_exp_search(local, heads, full)
    tuned_config = (
        None
        if searched is None
        else ttnn.SDPAProgramConfig(compute_with_storage_grid_size=(searched[0], full.y), q_chunk_size=searched[1], k_chunk_size=512)
    )
    resolved = T._sdpa_recipe_resolved_program_config(
        "exp_ring", options["precision"], inputs[0], inputs[1], program_config=auto_config, ring_size=2
    )

    def fn(config):
        return ttnn.transformer.exp_ring_joint_scaled_dot_product_attention(
            *inputs,
            persistent_output_buffer_k=backing[0],
            persistent_output_buffer_v=backing[1],
            joint_strategy="rear",
            logical_n=2 * local,
            program_config=config,
            dim=2,
            multi_device_global_semaphore=semaphores[:2],
            num_links=2,
            cluster_axis=1,
            mesh_device=mesh,
            topology=ttnn.Topology.Ring,
            subdevice_id=subdevice,
            num_workers_per_link=full.y // 2,
            num_buffers_per_channel=16,
            **options,
        )

    compare(record_property, Invoke(mesh, fn), auto_config, tuned_config, resolved)
