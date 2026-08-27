# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Host-only contracts for the shared WH Galaxy geometry and placements."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

import ttnn
from models.common.models.galaxy.recipes import (
    GALAXY_MESH_SHAPE,
    RING_ALIGNMENT,
    GalaxyDenseGeometry,
    distributed_norm_decode_memory_config,
    dram_sharded_weight_memory_config,
    galaxy_padded_vocab_size,
    pad_ring_width,
    prefetch_sender_cores,
    resolve_galaxy_decode_placements,
    resolve_galaxy_prefill_placements,
    rope_core_grids,
    sampling_core_grids,
    validate_galaxy_mesh,
    worker_cores,
)
from models.common.modules.lazy_weight import LazyWeight
from models.common.modules.rmsnorm import rmsnorm_2d
from models.common.modules.rmsnorm.rmsnorm_2d import RMSNorm2DConfig, RMSNorm2DGeometry, _resolve_2d_config

LLAMA = dict(dim=8192, hidden_dim=28672, n_heads=64, n_kv_heads=8, head_dim=128, vocab_size=128256)
QWEN = dict(dim=5120, hidden_dim=25600, n_heads=64, n_kv_heads=8, head_dim=128, vocab_size=151936)


def _mesh(shape=GALAXY_MESH_SHAPE, *, devices=32, arch=ttnn.device.Arch.WORMHOLE_B0, dram_width=12, grid=(7, 10)):
    mesh = MagicMock(spec=ttnn.MeshDevice)
    mesh.shape = shape
    mesh.get_num_devices.return_value = devices
    mesh.arch.return_value = arch
    mesh.dram_grid_size.return_value = SimpleNamespace(x=dram_width, y=1)
    # A real CoreCoord: ttnn.num_cores_to_corerangeset is a pybind11 binding
    # and will not accept a duck-typed stand-in.
    mesh.compute_with_storage_grid_size.return_value = ttnn.CoreCoord(grid[0], grid[1])
    return mesh


def _geometry(model: dict, **overrides) -> GalaxyDenseGeometry:
    values = dict(model, max_seq_len=2048, prefill_sequence_lengths=(128, 2048))
    values.update(overrides)
    return GalaxyDenseGeometry(**values)


@pytest.mark.parametrize("model", [LLAMA, QWEN], ids=["llama-3.3-70b", "qwen3-32b"])
def test_geometry_partitions_the_mesh_consistently(model):
    geometry = _geometry(model)

    assert geometry.local_dim == model["dim"] // 4
    assert geometry.row_dim == model["dim"] // 8
    assert geometry.local_hidden_dim == model["hidden_dim"] // 8
    assert geometry.attention_dim == model["n_heads"] * model["head_dim"]
    assert geometry.local_attention_dim == geometry.attention_dim // 8
    assert geometry.qkv_size == model["head_dim"] * (model["n_heads"] + 2 * model["n_kv_heads"])
    assert geometry.local_qkv_size == geometry.qkv_size // 8
    assert geometry.local_heads == model["n_heads"] // 8
    assert geometry.local_kv_heads == model["n_kv_heads"] // 8
    assert geometry.users_per_column == 8


def test_geometry_reports_galaxy_aligned_vocabularies():
    # Ring-exact, not minimal: see `galaxy_padded_vocab_size` and D-B19. A width
    # that is only tile-aligned leaves the LM head's column all-reduce waiting on a
    # shard that is never full, forever and without an abort.
    assert galaxy_padded_vocab_size(128256) == 129024
    assert galaxy_padded_vocab_size(151936) == 153600
    assert _geometry(LLAMA).padded_vocab_size == 129024
    assert _geometry(QWEN).padded_vocab_size == 153600
    assert _geometry(QWEN).local_padded_vocab_size == 153600 // 8
    # Ring-exact for both: the per-device width is a whole number of 24-core ring
    # rows, which is what makes `lm_head_reduce_core_count`'s divisor search exact
    # and what stops `all_reduce_async` waiting on a shard that is never full.
    for vocab in (128256, 151936):
        assert (galaxy_padded_vocab_size(vocab) // 8) % (24 * 32) == 0


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"max_batch_size": 16}, "physical batch 32"),
        ({"dim": 8200}, "must shard over 4 columns"),
        ({"hidden_dim": 28680}, "must shard over 8 rows"),
        ({"n_heads": 60}, "must shard over the 8 mesh rows"),
        ({"n_kv_heads": 4}, "must shard over the 8 mesh rows"),
        ({"head_dim": 48}, "tile aligned"),
        ({"dim": 8320}, "distributed norm grid"),
        ({"prefill_sequence_lengths": (100,)}, "multiple of 128"),
        ({"prefill_sequence_lengths": (4096,)}, "exceeds max_seq_len"),
        ({"max_seq_len": 2000}, "positive multiple of 128"),
    ],
)
def test_geometry_fails_closed_on_unsupported_shapes(overrides, message):
    with pytest.raises(ValueError, match=message):
        _geometry(LLAMA, **overrides)


def test_mesh_validation_requires_wormhole_galaxy():
    validate_galaxy_mesh("test", _mesh())
    with pytest.raises(ValueError, match=r"logical mesh shape \(8, 4\)"):
        validate_galaxy_mesh("test", _mesh(shape=(4, 8)))
    with pytest.raises(ValueError, match="exactly 32 devices"):
        validate_galaxy_mesh("test", _mesh(devices=31))
    with pytest.raises(ValueError, match="Wormhole only"):
        validate_galaxy_mesh("test", _mesh(arch=ttnn.device.Arch.BLACKHOLE))


@pytest.mark.parametrize("model", [LLAMA, QWEN], ids=["llama-3.3-70b", "qwen3-32b"])
def test_residual_placement_matches_the_rmsnorm2d_default(monkeypatch, model):
    """The residual stream must land exactly where the fused norm expects it."""

    monkeypatch.setattr(rmsnorm_2d, "resolve_lazy_weight", lambda weight, **_: weight)
    mesh = _mesh()
    geometry = _geometry(model)
    weight = LazyWeight(source=torch.zeros(model["dim"], dtype=torch.bfloat16), device=mesh)
    resolved = _resolve_2d_config(
        RMSNorm2DConfig(
            weight=weight,
            mesh_device=mesh,
            geometry=RMSNorm2DGeometry.DISTRIBUTED,
            tt_ccl=_ccl(mesh),
            collective_resource_selector=lambda *_args, **_kwargs: None,
        )
    )

    assert distributed_norm_decode_memory_config(geometry) == resolved.decode_input_memcfg
    assert distributed_norm_decode_memory_config(geometry) == resolved.decode_output_memcfg


def _ccl(mesh):
    def context(mode):
        return SimpleNamespace(
            mesh_device=mesh,
            mode=mode,
            worker_sub_device_id=f"{mode}-worker",
            resources=lambda *_args, **_kwargs: None,
            next_semaphore_handles=lambda *_args, **_kwargs: None,
            next_barrier_semaphore_handle=lambda *_args, **_kwargs: None,
        )

    return SimpleNamespace(mesh_device=mesh, context=context)


@pytest.mark.parametrize("model", [LLAMA, QWEN], ids=["llama-3.3-70b", "qwen3-32b"])
def test_decode_placements_derive_every_shard_from_the_geometry(model):
    mesh = _mesh()
    geometry = _geometry(model)
    decode = resolve_galaxy_decode_placements(geometry, mesh)

    residual = decode.residual_memcfg.shard_spec
    assert tuple(residual.shape) == (32, 128)
    assert residual.grid.num_cores() == geometry.local_dim // 128

    # The fused create-QKV-heads scratch keeps four head columns per head core,
    # and its reduced output keeps one.
    scratch = decode.attention_qkv_scratch_memcfg.shard_spec
    assert tuple(scratch.shape) == (32, 4 * geometry.head_dim)
    assert scratch.grid.num_cores() == geometry.local_qkv_size // geometry.head_dim
    reduced = decode.attention_qkv_reduced_memcfg.shard_spec
    assert tuple(reduced.shape) == (32, geometry.head_dim)
    assert reduced.grid.num_cores() == geometry.local_qkv_size // geometry.head_dim

    # The scattered W1/W3 placement is padded for the 24-core ring, while the
    # resource key uses the logical width TTNN reports.
    scattered = decode.mlp_reduce_scatter_memcfg.shard_spec
    assert tuple(scattered.shape) == (32, 32)
    assert scattered.grid.num_cores() == geometry.decode_reduce_scatter_padded_width // 32
    assert geometry.decode_reduce_scatter_width <= geometry.decode_reduce_scatter_padded_width

    assert decode.mlp_input_memcfg.shard_spec.grid.num_cores() == 24
    assert decode.mlp_w2_input_memcfg.shard_spec.grid.num_cores() == 24
    assert decode.all_reduce_buffer_memcfg.shard_spec.grid.num_cores() == worker_cores().num_cores()
    assert decode.attention_input_memcfg == ttnn.DRAM_MEMORY_CONFIG


def test_ring_padding_is_tile_and_core_aligned():
    assert pad_ring_width(1) == RING_ALIGNMENT
    assert pad_ring_width(RING_ALIGNMENT) == RING_ALIGNMENT
    assert pad_ring_width(3584) == 3840
    assert pad_ring_width(2048) == 2304
    assert pad_ring_width(1280) == 1536


def test_llama_decode_reduce_scatter_width_matches_the_qualified_recipe():
    # 30 cores of 32 columns is the width the qualified MLP2D hardware recipe
    # uses for both models; the logical widths differ.
    assert _geometry(LLAMA).decode_reduce_scatter_width == 960
    assert _geometry(QWEN).decode_reduce_scatter_width == 800
    assert _geometry(LLAMA).decode_reduce_scatter_padded_width == 960
    assert _geometry(QWEN).decode_reduce_scatter_padded_width == 960


@pytest.mark.parametrize("model", [LLAMA, QWEN], ids=["llama-3.3-70b", "qwen3-32b"])
def test_prefill_placements_cover_every_qualified_sequence_length(model):
    mesh = _mesh()
    geometry = _geometry(model)
    prefill = resolve_galaxy_prefill_placements(geometry, mesh)

    assert set(prefill.attention_program_configs) == set(geometry.prefill_sequence_lengths)
    assert set(prefill.attention_wo_program_configs) == set(geometry.prefill_sequence_lengths)
    assert set(prefill.attention_sdpa_program_configs) == set(geometry.prefill_sequence_lengths)
    assert prefill.residual_memcfg == ttnn.DRAM_MEMORY_CONFIG


@pytest.mark.parametrize("model", [LLAMA, QWEN], ids=["llama-3.3-70b", "qwen3-32b"])
def test_batched_prefill_placements_are_resolved_per_row_length(model):
    """Concatenated prefill projects 32 rows at once but attends per row."""

    mesh = _mesh()
    geometry = _geometry(model, batched_prefill_sequence_lengths=(128,))
    prefill = resolve_galaxy_prefill_placements(geometry, mesh)

    assert set(prefill.batched_attention_program_configs) == {128}
    assert set(prefill.batched_attention_wo_program_configs) == {128}
    # SDPA still runs one causal sequence per row, so it reuses the single-row
    # geometry; the projections do not.
    assert prefill.batched_attention_sdpa_program_configs[128] != prefill.attention_program_configs[128]
    assert prefill.batched_attention_program_configs[128] != prefill.attention_program_configs[128]
    assert prefill.chunked_sdpa_program_config is not None


def test_batched_prefill_token_counts_drive_the_collective_geometry():
    geometry = _geometry(LLAMA, prefill_sequence_lengths=(128,), batched_prefill_sequence_lengths=(128, 256))

    assert geometry.batched_prefill_token_counts == (32 * 128, 32 * 256)
    assert geometry.collective_prefill_token_counts == (128, 4096, 8192)
    # The whole batched stream still reshapes on the MLP's 1024-token cutoff.
    assert geometry.prefill_leading_shape(4096) == (1, 4, 1024)


def test_batched_prefill_lengths_are_validated_like_single_row_lengths():
    with pytest.raises(ValueError, match="batched prefill length 100 must be a positive multiple of 128"):
        _geometry(LLAMA, batched_prefill_sequence_lengths=(100,))
    with pytest.raises(ValueError, match="batched prefill length 4096 exceeds max_seq_len"):
        _geometry(LLAMA, batched_prefill_sequence_lengths=(4096,))


def test_default_geometry_resolves_no_batched_recipe():
    geometry = _geometry(LLAMA)

    assert geometry.batched_prefill_sequence_lengths == ()
    assert geometry.collective_prefill_token_counts == geometry.prefill_sequence_lengths
    assert resolve_galaxy_prefill_placements(geometry, _mesh()).batched_attention_program_configs == {}


def test_weight_placement_pads_the_ring_output_width():
    mesh = _mesh(dram_width=12)
    memory_config = dram_sharded_weight_memory_config(mesh, 2048, 3584)

    assert tuple(memory_config.shard_spec.shape) == (2048, 3840 // 12)
    assert memory_config.shard_spec.grid.num_cores() == 12


def test_sampling_and_rope_core_grids_are_explicit():
    sub_core_grids, topk_grid, start_core = sampling_core_grids()
    assert sub_core_grids.num_cores() == worker_cores().num_cores()
    assert topk_grid.num_cores() == 30
    assert (start_core.x, start_core.y) == (1, 0)

    mesh = _mesh()
    _, batch_grid = rope_core_grids(mesh, use_qk_fused=False)
    assert batch_grid.num_cores() == 8
    _, fused_batch_grid = rope_core_grids(mesh, use_qk_fused=True)
    assert fused_batch_grid.num_cores() == 16


def test_geometry_prefill_reshape_matches_the_mlp_cutoff():
    geometry = _geometry(LLAMA, prefill_sequence_lengths=(128, 1024, 2048))

    assert geometry.prefill_leading_shape(128) == (1, 1, 128)
    assert geometry.prefill_leading_shape(1024) == (1, 1, 1024)
    assert geometry.prefill_leading_shape(2048) == (1, 2, 1024)


def test_geometry_is_frozen():
    geometry = _geometry(LLAMA)
    with pytest.raises(FrozenInstanceError):
        geometry.dim = 4096
    assert replace(geometry, max_seq_len=4096).max_seq_len == 4096


# =============================================================================
# Decode placements must live inside the worker sub-device
# =============================================================================


@pytest.mark.parametrize("use_qk_fused", [False, True], ids=["plain", "qk-fused"])
def test_rope_batch_grid_lies_inside_the_worker_sub_device(use_qk_fused):
    """The decode cos/sin shards may not land on a prefetch sender column.

    A program may only touch cores owned by the loaded sub-device manager. The
    Galaxy decode manager partitions the grid into prefetch senders (``x=0`` and
    ``x=4``) and workers, so a decode placement built from the *whole* compute
    grid puts shards on sender cores and on a core outside every sub-device.
    ``ttnn.embedding`` then aborts with ``Kernel group cores do not match sub
    device cores for programmable core type TENSIX``, and because the abort
    happens inside a multi-sub-device program it leaves the mesh un-drainable -
    teardown blocks forever in ``FDMeshCommandQueue``'s destructor.

    This is the same failure shape as Milestone A D1/C1 - a grid named
    independently of the partition that must contain it - so it is checked on
    host rather than left to a device run.
    """

    _, batch_grid = rope_core_grids(_mesh(), use_qk_fused=use_qk_fused)
    rows = 8 * (2 if use_qk_fused else 1)
    workers = worker_cores()

    assert batch_grid.num_cores() == rows, (batch_grid.num_cores(), rows)
    outside = batch_grid.subtract(workers)
    assert outside.num_cores() == 0, f"{outside.num_cores()} rope core(s) outside worker_cores(): {outside}"

    senders = ttnn.CoreRangeSet([ttnn.CoreRange(core, core) for core in prefetch_sender_cores()])
    overlap = batch_grid.subtract(batch_grid.subtract(senders))
    assert overlap.num_cores() == 0, f"rope shards overlap prefetch senders: {overlap}"
