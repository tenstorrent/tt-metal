# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""indexer_score_dsa in page-table mode on the held one-by-eight mesh: the op on a block pool plus a page table equals
the op on the contiguous logical k, bitwise, for the QSA call forms (generic decode window, 32-row chunk tile,
per-lane cache_batch_idx, wide flat cache with kv_len), block_rows 32 and 512."""

import os

import pytest
import torch

import ttnn
from models.demos.blackhole.qwen38_flash_next.tests import qsa_paged_reference as ref
from models.demos.blackhole.qwen38_flash_next.tests.tp_harness import DEVICE_PARAMS
from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import MESH_SHAPE

pytestmark = pytest.mark.skipif(
    os.environ.get("QWEN38_FUSED_DEVICE_TEST") != "1", reason="requires held one-by-eight mesh"
)
MESH = pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
PARAMS = pytest.mark.parametrize("device_params", [{**DEVICE_PARAMS, "trace_region_size": 2_000_000}], indirect=True)
DRAM = ttnn.DRAM_MEMORY_CONFIG
HEAD_DIM = 128  # qsa.INDEX_HEAD_DIM
STAGING_AXIS = 0  # qsa.STAGING_AXIS
# (block_rows, W): the logical user is W * block_rows rows = blocks + one 32-row window tile, as the model's cache.
GEOMETRY = pytest.mark.parametrize("block_rows,width", [(32, 8), (512, 2)])
# Model-size users: 8448 rows (bs 128, C/4 + 32) and 12288 rows (bs 2048): several k bands, mcast, window straddle.
GEOMETRY_LARGE = pytest.mark.parametrize("block_rows,width", [(32, 8), (512, 2), (32, 264), (512, 24)])
GEOMETRY_WIDE = pytest.mark.parametrize("block_rows,width", [(32, 8), (512, 2), (32, 264)])


def up(mesh, tensor, dtype, layout):
    return ttnn.from_torch(
        tensor,
        dtype=dtype,
        layout=layout,
        device=mesh,
        memory_config=DRAM,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


def host(tensor):
    replicas = [ttnn.to_torch(local) for local in ttnn.get_device_tensors(tensor)]
    assert all(torch.equal(replicas[0], other) for other in replicas[1:])
    return replicas[0]


def compute_config(mesh):
    # qsa.py:3569 (the indexer's LLK rejects FP32 DEST)
    return ttnn.init_device_compute_kernel_config(
        mesh.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )


class Case:
    """Contiguous logical k, its pool and table, the query tile and the gate, all on the mesh."""

    def __init__(self, mesh, users, width, block_rows, seed, table_shape=None, user_rows=None, wide=False):
        generator = torch.Generator().manual_seed(seed)
        self.rows = user_rows or width * block_rows
        self.users, self.block_rows, self.user_rows, self.wide = users, block_rows, user_rows, wide
        num_blocks = users * width + 8
        self.logical = torch.randn(users, 1, self.rows, HEAD_DIM, generator=generator).to(torch.bfloat16)
        used = width if user_rows is None else -(-user_rows // block_rows)  # entries a user's rows touch
        ids = 1 + torch.randperm(num_blocks - 1, generator=generator)[: users * used]  # distinct, never block 0
        table = torch.zeros(users, width, dtype=torch.int32)  # padding entries stay block 0
        table[:, :used] = ids.reshape(users, used).to(torch.int32)
        pool = ref.scatter_index_pool(self.logical, table, block_rows, num_blocks, generator, user_rows=user_rows)
        self.mesh = mesh
        self.config = compute_config(mesh)
        # qsa.py:5405: the query's own tile (1, heads_per_device=1, 32, 128); the gate (1, 1, 32, 1) as qsa.py:3610.
        query = torch.randn(1, 1, ttnn.TILE_SIZE, HEAD_DIM, generator=generator).to(torch.bfloat16)
        gate = torch.full((1, 1, ttnn.TILE_SIZE, 1), HEAD_DIM**-0.5, dtype=torch.bfloat16)
        self.query = up(mesh, query, ttnn.bfloat16, ttnn.TILE_LAYOUT)
        self.gate = up(mesh, gate, ttnn.bfloat16, ttnn.TILE_LAYOUT)
        self.pool = up(mesh, pool, ttnn.bfloat16, ttnn.TILE_LAYOUT)
        shape = table_shape or (1, 1, users, width)
        self.table = up(mesh, table.reshape(shape), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)

    def flat_logical(self):
        return self.logical.reshape(1, 1, self.users * self.rows, HEAD_DIM)

    def score(self, k_dense, k_pool, paged, **kwargs):
        kw = dict(compute_kernel_config=self.config, seq_shard_axes=[STAGING_AXIS], **kwargs)
        if paged:
            kw.update(page_table=self.table, paged_block_rows=self.block_rows)
            if self.user_rows is not None:
                kw.update(paged_user_rows=self.user_rows, paged_wide=self.wide)
            k = k_pool
        else:
            k = k_dense
        return ttnn.experimental.indexer_score_dsa(self.query, k, self.gate, **kw)


def compare(case, k_dense, chunk_start, width_limit=None, **kwargs):
    dense = case.score(k_dense, None, False, chunk_start_idx=chunk_start, **kwargs)
    paged = case.score(None, case.pool, True, chunk_start_idx=chunk_start, **kwargs)
    a, b = host(dense), host(paged)
    assert a.shape == b.shape
    if width_limit is not None:  # only columns [0, kv_len) are written
        a, b = a[..., :width_limit], b[..., :width_limit]
    assert torch.equal(a, b)


@MESH
@PARAMS
@GEOMETRY_LARGE
def test_indexer_paged_generic_window(mesh_device, block_rows, width):
    """B = 1 decode window: qsa._score_blocks_generic (qsa.py:5414-5422); chunk_start = blocks = rows - 32
    (self.indexer_chunk_start, qsa.py:3542), k = compressed_index_cache [1, 1, blocks + 32, 128]."""

    case = Case(mesh_device, 1, width, block_rows, seed=1)
    k = up(mesh_device, case.logical, ttnn.bfloat16, ttnn.TILE_LAYOUT)
    compare(case, k, case.rows - ttnn.TILE_SIZE)


@MESH
@PARAMS
@GEOMETRY
def test_indexer_paged_chunk_tile(mesh_device, block_rows, width):
    """32-row prefill chunk tile: qsa._score_blocks_chunk (qsa.py:6048-6052), same kwargs and window as the
    generic call, run on a second random query tile."""

    case = Case(mesh_device, 1, width, block_rows, seed=2)
    k = up(mesh_device, case.logical, ttnn.bfloat16, ttnn.TILE_LAYOUT)
    compare(case, k, case.rows - ttnn.TILE_SIZE)


@MESH
@PARAMS
@GEOMETRY
@pytest.mark.parametrize("lane", [0, 3])
def test_indexer_paged_lanes_per_lane(mesh_device, block_rows, width, lane):
    """Per-lane form: qsa._score_blocks_lanes_fused else branch (qsa.py:8664-8672), k = [B=4, 1, rows, 128],
    cache_batch_idx = lane, chunk_start = indexer_chunk_start = rows - 32."""

    case = Case(mesh_device, 4, width, block_rows, seed=3 + lane)
    k = up(mesh_device, case.logical, ttnn.bfloat16, ttnn.TILE_LAYOUT)
    compare(case, k, case.rows - ttnn.TILE_SIZE, cache_batch_idx=lane)


@MESH
@PARAMS
@GEOMETRY_WIDE
@pytest.mark.parametrize("count", [4, 3])
def test_indexer_paged_lanes_wide(mesh_device, block_rows, width, count):
    """Wide form: qsa._score_blocks_lanes_fused wide branch (qsa.py:8648-8658): table reshaped to [1, R * W]
    against the flat logical k [1, 1, R * rows, 128]; chunk_start = count * rows - 32; count < R is the bucket view
    and passes kv_len = count * rows (qsa.py:8650-8651)."""

    users = 4
    case = Case(mesh_device, users, width, block_rows, seed=7 + count, table_shape=(1, 1, 1, users * width))
    k = up(mesh_device, case.flat_logical(), ttnn.bfloat16, ttnn.TILE_LAYOUT)
    kwargs = {"kv_len": count * case.rows} if count < users else {}
    compare(case, k, count * case.rows - ttnn.TILE_SIZE, width_limit=count * case.rows, **kwargs)


# The model's cache: user rows S = C/4 + 32 = 8224 (C = 32768), a prefix of the W * block_rows table span.
USER_ROWS = 8224
GEOMETRY_USER_ROWS = pytest.mark.parametrize("block_rows,width", [(32, 264), (512, 24)])


@MESH
@PARAMS
@GEOMETRY_USER_ROWS
def test_indexer_paged_user_rows_window(mesh_device, block_rows, width):
    """paged_user_rows = S < W * block_rows, R = 1: the generic decode window, chunk_start = S - 32."""

    case = Case(mesh_device, 1, width, block_rows, seed=21, user_rows=USER_ROWS)
    k = up(mesh_device, case.logical, ttnn.bfloat16, ttnn.TILE_LAYOUT)
    compare(case, k, USER_ROWS - ttnn.TILE_SIZE)


@MESH
@PARAMS
@GEOMETRY_USER_ROWS
@pytest.mark.parametrize("lane", [0, 3])
def test_indexer_paged_user_rows_lanes_per_lane(mesh_device, block_rows, width, lane):
    """paged_user_rows = S, R = 4, cache_batch_idx = lane: the logical batch stride is S/32 tiles, the table's W."""

    case = Case(mesh_device, 4, width, block_rows, seed=23 + lane, user_rows=USER_ROWS)
    k = up(mesh_device, case.logical, ttnn.bfloat16, ttnn.TILE_LAYOUT)
    compare(case, k, USER_ROWS - ttnn.TILE_SIZE, cache_batch_idx=lane)


@MESH
@PARAMS
@GEOMETRY_USER_ROWS
@pytest.mark.parametrize("count", [4, 3])
def test_indexer_paged_user_rows_wide(mesh_device, block_rows, width, count):
    """paged_wide with paged_user_rows = S: the logical k is the flat [1, 1, R * S, 128]; bands straddle users
    (S/32 = 257 tiles); count < R is the bucket view (kv_len = count * S)."""

    users = 4
    case = Case(mesh_device, users, width, block_rows, seed=27 + count, user_rows=USER_ROWS, wide=True)
    k = up(mesh_device, case.flat_logical(), ttnn.bfloat16, ttnn.TILE_LAYOUT)
    kwargs = {"kv_len": count * USER_ROWS} if count < users else {}
    compare(case, k, count * USER_ROWS - ttnn.TILE_SIZE, width_limit=count * USER_ROWS, **kwargs)


def sparse_case(mesh, users, reals, block_rows, width, seed, wide=False):
    """User u keeps only its first reals[u] table entries (shuffled non-zero ids); every other entry is 0 (the null
    block). The pool's block 0 holds large values, the unpaged logical k is ZERO beyond each user's real blocks."""

    case = Case(mesh, users, width, block_rows, seed=seed, user_rows=USER_ROWS, wide=wide)
    generator = torch.Generator().manual_seed(seed + 1000)
    num_blocks = users * width + 8
    ids = 1 + torch.randperm(num_blocks - 1, generator=generator)[: sum(reals)]
    table = torch.zeros(users, width, dtype=torch.int32)
    start = 0
    for user, real in enumerate(reals):
        table[user, :real] = ids[start : start + real].to(torch.int32)
        case.logical[user, :, real * block_rows :] = 0
        start += real
    pool = ref.scatter_index_pool(case.logical, table, block_rows, num_blocks, generator, user_rows=USER_ROWS)
    pool[0] = (torch.randn(pool[0].shape, generator=generator) * 1000.0 + 500.0).to(torch.bfloat16)
    case.pool = up(mesh, pool, ttnn.bfloat16, ttnn.TILE_LAYOUT)
    case.table = up(mesh, table.reshape(1, 1, users, width), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
    return case


@MESH
@PARAMS
def test_indexer_paged_zero_entries_read_as_zeros(mesh_device):
    """Null-block entries (0) are not read from DRAM; their k tiles are zero-filled. Only the first 3 table entries
    are real, block 0 holds large values: the paged output equals the unpaged op on a logical k that is zero beyond
    those blocks, bitwise on all columns (R = 1 window)."""

    case = sparse_case(mesh_device, 1, [3], 32, 264, seed=31)
    k = up(mesh_device, case.logical, ttnn.bfloat16, ttnn.TILE_LAYOUT)
    compare(case, k, USER_ROWS - ttnn.TILE_SIZE)


@MESH
@PARAMS
def test_indexer_paged_zero_entries_read_as_zeros_wide(mesh_device):
    """Wide form, R = 4, a different number of real blocks per lane (including 0 and a user with all but the window
    real); bands straddle users, so zero tiles also come through the per-tile straddle path."""

    users = 4
    case = sparse_case(mesh_device, users, [3, 0, 100, 257], 32, 264, seed=33, wide=True)
    k = up(mesh_device, case.flat_logical(), ttnn.bfloat16, ttnn.TILE_LAYOUT)
    compare(case, k, users * USER_ROWS - ttnn.TILE_SIZE)
