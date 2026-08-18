# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Shared helpers for the per-op test suite (``tests/ops/``).

Goal
----
The end-to-end token-accuracy demo
(``tests/demos/llama32_1b/demo.py -k token-accuracy``) is extremely slow on the
Quasar emulator. These op-level tests isolate *each individual ttnn op* that the
model executes, with the *exact shapes / dtypes / memory configs* the model uses,
so a single op can be exercised on the emulator in isolation.

Every op file should:
  * import dims/helpers from here (do not re-hardcode Llama-3.2-1B dims),
  * ground each parametrization in a real call site in the model source
    (cite ``file:line`` in a comment),
  * use the ``ttnn_mesh_device`` fixture (provided by ``tests/conftest.py``),
    parametrized ``indirect`` — default ``[(1, 1)]`` for the emulator,
  * compare against a torch reference with ``assert_pcc`` where a clear reference
    exists; otherwise assert output shape / dtype / layout and finiteness with
    ``assert_shape_dtype``.

These tests deliberately keep no dependency on the full model build — they
allocate fresh random tensors so each op runs in <1s of device time.
"""

from __future__ import annotations

import pytest
import torch

import ttnn
from models.experimental.llama32_1b_quasar.auto_compose import to_torch_auto_compose
from models.experimental.llama32_1b_quasar.utility_functions import comp_allclose, comp_pcc

# =============================================================================
# Llama-3.2-1B-Instruct architecture constants
# (config.json: hidden 2048, 16 layers, 32 heads / 8 kv heads, head_dim 64,
#  intermediate 8192, vocab 128256, rms_norm_eps 1e-5, rope_theta 5e5)
# =============================================================================

DIM = 2048  # hidden_size
N_LAYERS = 16
N_HEADS = 32
N_KV_HEADS = 8
HEAD_DIM = 64
Q_DIM = N_HEADS * HEAD_DIM  # 2048
KV_DIM = N_KV_HEADS * HEAD_DIM  # 512
QKV_DIM = Q_DIM + 2 * KV_DIM  # 3072 (fused qkv projection width)
INTERMEDIATE = 8192  # MLP hidden
VOCAB = 128256
NORM_EPS = 1e-5
ROPE_THETA = 500000.0
MAX_BATCH = 32  # decode users
TILE = 32

# Representative sequence lengths / batches the demo drives through the model.
# Prefill chunks are tile-aligned; decode runs 1 token for `batch` users.
PREFILL_SEQ_LENS = [128, 512, 1024]
DECODE_BATCHES = [1, 32]

# Default mesh for the emulator (single logical device). Multi-device shapes are
# added per-op only where the model actually shards across devices.
DEFAULT_MESH = (1, 1)


def with_default_mesh(*shapes):
    """Parametrize the ``ttnn_mesh_device`` fixture (indirect). Default: [(1, 1)]."""
    return pytest.mark.parametrize("ttnn_mesh_device", list(shapes) or [DEFAULT_MESH], indirect=True)


# =============================================================================
# Tensor helpers
# =============================================================================


def torch_rand(shape, dtype=torch.bfloat16, mean=0.0, std=1.0):
    """Reproducible random torch tensor. Seed is set per-test via reset_seeds."""
    return (torch.randn(*shape) * std + mean).to(dtype)


def to_tt(
    torch_tensor: torch.Tensor,
    mesh_device,
    *,
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
    mesh_mapper="replicate",
    shard_dim=None,
) -> ttnn.Tensor:
    """torch -> ttnn on ``mesh_device`` with the given layout / memory config.

    ``mesh_mapper``: ``"replicate"`` (default) replicates across the mesh;
    an int shards along that torch dim; ``None`` uses no mapper (single device).
    """
    if mesh_mapper == "replicate":
        mapper = ttnn.replicate_tensor_to_mesh_mapper(mesh_device)
    elif isinstance(mesh_mapper, int):
        mapper = ttnn.shard_tensor_to_mesh_mapper(mesh_device, dim=mesh_mapper)
    elif shard_dim is not None:
        mapper = ttnn.shard_tensor_to_mesh_mapper(mesh_device, dim=shard_dim)
    else:
        mapper = None
    return ttnn.from_torch(
        torch_tensor,
        dtype=dtype,
        layout=layout,
        device=mesh_device,
        memory_config=memory_config,
        mesh_mapper=mapper,
    )


def from_tt(tt_tensor: ttnn.Tensor, mesh_device=None) -> torch.Tensor:
    """ttnn -> torch (host), composing across the mesh. Returns float32."""
    return to_torch_auto_compose(tt_tensor, mesh_device).float()


# =============================================================================
# Sharded memory-config helpers (decode batch-core grids)
# =============================================================================


def height_sharded_batch_memcfg(mesh_device, num_cores, shard_shape, *, start_core=None):
    """HEIGHT_SHARDED L1 memory config over ``num_cores`` cores (one shard/core, row-major).

    The decode attention ops (nlp_concat_heads_decode, paged_update_cache,
    rotary_embedding_llama_fused_qk) require HEIGHT_SHARDED inputs laid out one
    *user* per core over a batch-core grid. This mirrors the model's config
    resolver — e.g. ``decode_scores_memcfg``
    (modules/attention/attention_1d.py:1727-1734) — and the canonical op
    reference tests/ttnn/nightly/.../test_rotary_embedding_llama.py:233-243.

    Args:
        num_cores: number of cores == number of users/batch (one shard per core).
        shard_shape: ``(shard_height, shard_width)`` per-core shard.
        start_core: optional ``ttnn.CoreCoord`` origin so a second tensor can be
            placed on a NON-OVERLAPPING grid (fused-QK Q/K requirement). Cores are
            laid out in rows of 8 (matches attention_1d.py:_num_to_corerange).

    Emulator gating: one shard is placed per core, so a decode config with
    ``batch`` users needs ``batch`` cores (fused Q/K needs ``2*batch``). If that
    exceeds the device's compute grid — as ``batch=32`` does on the 2-node Quasar
    emulator — the test is SKIPPED (not failed), so the batch=1 case still passes.
    Larger devices (e.g. N150's 8x8 grid) run every batch.
    """
    core_grid = mesh_device.compute_with_storage_grid_size()
    available = core_grid.x * core_grid.y
    # rows of 8 mirror _num_to_corerange; start offset consumes leading cores.
    start_idx = 0 if start_core is None else (start_core.y * 8 + start_core.x)
    needed = start_idx + num_cores
    if needed > available:
        pytest.skip(
            f"height-sharded op needs {needed} cores "
            f"({num_cores} shards" + (f" at offset {start_idx}" if start_core else "") + f"); "
            f"device has {available} — batch too large for this device (fits batch=1)"
        )
    if start_core is None:
        grid = ttnn.num_cores_to_corerangeset(num_cores, core_grid, row_wise=True)
    else:
        grid = ttnn.CoreRangeSet({_num_to_corerange(num_cores, start_core)})
    return ttnn.create_sharded_memory_config(
        shape=tuple(shard_shape),
        core_grid=grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def _num_to_corerange(num_cores: int, start_core: "ttnn.CoreCoord | None" = None) -> "ttnn.CoreRange":
    """Contiguous CoreRange of ``num_cores`` cores in rows of 8 from ``start_core``.

    Copied from modules/attention/attention_1d.py:2234 (_num_to_corerange) so the
    fused-QK test can place Q and K on adjacent non-overlapping grids exactly as
    _reshard_k_for_fused does (attention_1d.py:1242-1243)."""
    if start_core is None:
        start_core = ttnn.CoreCoord(0, 0)
    if num_cores == 1:
        return ttnn.CoreRange(start_core, start_core)
    row_size = 8
    start_x, start_y = start_core.x, start_core.y
    total = start_x + num_cores
    end_y = start_y + (total - 1) // row_size
    end_x = (total - 1) % row_size
    return ttnn.CoreRange(start_core, ttnn.CoreCoord(end_x, end_y))


# =============================================================================
# Assertions
# =============================================================================


def assert_pcc(torch_ref: torch.Tensor, tt_out: ttnn.Tensor, *, pcc=0.99, mesh_device=None):
    """Compare a ttnn output against a torch reference by PCC."""
    got = from_tt(tt_out, mesh_device)
    ref = torch_ref.float()
    # Align on the reference's element count when the op pads to tiles.
    if got.shape != ref.shape and got.numel() >= ref.numel():
        got = got.reshape(-1)[: ref.numel()].reshape(ref.shape)
    passing, msg = comp_pcc(ref, got, pcc)
    _, all_msg = comp_allclose(ref, got)
    assert passing, f"PCC below {pcc}: {msg} | {all_msg}"


def assert_shape_dtype(tt_out: ttnn.Tensor, *, shape=None, dtype=None, finite=True, mesh_device=None):
    """For ops without a simple torch reference: check shape/dtype and finiteness."""
    if shape is not None:
        assert tuple(tt_out.shape) == tuple(shape), f"expected shape {tuple(shape)}, got {tuple(tt_out.shape)}"
    if dtype is not None:
        assert tt_out.dtype == dtype, f"expected dtype {dtype}, got {tt_out.dtype}"
    if finite:
        host = from_tt(tt_out, mesh_device)
        assert torch.isfinite(host).all(), "output contains non-finite values"
