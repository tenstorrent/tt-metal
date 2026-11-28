# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Every row count a FLUX 3 Klein table is keyed by is one a served request reaches: per device, the padded length of a
stream or of the joined sequence, at every served size, reference bucket (none, a single reference, 1-10 references)
and text bucket, for SP8 x TP4, SP4 x TP2 (DP4) and SP2 x TP2 (DP8). A row no request reaches is dead config."""

import pytest

import ttnn

from ....models.transformers.flux3_klein import transformer_block_flux3_klein as block
from ....models.transformers.flux3_klein import transformer_flux3_klein as transformer
from ....models.transformers.flux3_klein.transformer_flux3_klein import Flux3KleinSequenceLayout
from ....pipelines.flux3_klein.packing import TEXT_BUCKETS, TEXT_BUCKETS_BY_LAYOUT
from ....pipelines.flux3_klein.pipeline_flux3_klein import MULTI_REFERENCE_TOKEN_BUCKETS, REFERENCE_TOKEN_BUCKETS, SIZES
from ....utils import matmul

REFERENCE_BUCKETS = sorted({0, *REFERENCE_TOKEN_BUCKETS, *MULTI_REFERENCE_TOKEN_BUCKETS})


def reachable_rows(sp: int) -> set[int]:
    rows = set()
    for size in SIZES:
        for ref in REFERENCE_BUCKETS:
            for text in TEXT_BUCKETS_BY_LAYOUT.get((size, ref), TEXT_BUCKETS):
                layout = Flux3KleinSequenceLayout(
                    sp, {"image": (size // 16) ** 2, "image_cond": ref, "text": text}, 1, ref
                )
                rows |= {layout.padded[m] // sp for m in layout.modes} | {layout.total_padded // sp}
    return rows - {0}


@pytest.mark.parametrize(
    ("sp", "tables"),
    [
        (
            8,
            [
                block.RING_SDPA_CHUNKS_BY_LOCAL_ROWS,
                {m for m, *_ in transformer.SP8_AGMM_SHAPE_CONFIGS},
                {m for m, *_ in transformer.SP8_MLP_BLOCKINGS},
            ],
        ),
        (
            4,
            [
                block.SP4_TP2_RING_SDPA_CHUNKS_BY_LOCAL_ROWS,
                block.SP4_TP2_QFAST_RING_SDPA_CHUNKS_BY_LOCAL_ROWS,
                block.SP4_TP2_K256_RING_SDPA_CHUNKS_BY_LOCAL_ROWS,
                {m for m, *_ in transformer.DP4_AGMM_SHAPE_CONFIGS},
                {m for m, *_ in transformer.DP4_MATMUL_SHAPE_CONFIGS},
                {m for m, *_ in transformer.DP4_MLP_BLOCKINGS},
            ],
        ),
    ],
    ids=["sp8", "sp4"],
)
def test_every_table_row_is_reachable(sp, tables):
    reachable = reachable_rows(sp)
    dead = sorted({m for table in tables for m in table} - reachable)
    assert not dead, f"rows no served request reaches on SP{sp}: {dead}"


@pytest.fixture
def register(monkeypatch):  # a layout's registrations, undone after the test
    grid = dict(matmul.grid_12_9_configs)
    monkeypatch.setattr(matmul, "agmm_shape_configs", {})
    monkeypatch.setattr(matmul, "matmul_shape_configs", {})
    matmul._clear_config_caches()
    yield transformer.register_layout_shape_configs
    matmul.grid_12_9_configs.clear()
    matmul.grid_12_9_configs.update(grid)
    monkeypatch.undo()
    matmul._clear_config_caches()


def mlp_blocking(m, n):
    c = matmul.get_matmul_config(m, 3072, n, ttnn.CoreCoord(12, 9), use_heuristic=True)
    return c.M_block_size, c.K_block_size, c.N_block_size


def test_sp8_rows_keep_their_blockings(register):
    """SP8 x TP4's MLP blockings at every SP8 row are what the import gives (no DP4 entry: DP4's QKV shares N 4608 with
    SP8 x TP4's MLP-in), and registering SP8 x TP4 changes none of them."""
    shapes = [(m, n) for m in reachable_rows(8) for n in (2304, 4608)]
    before = {s: mlp_blocking(*s) for s in shapes}
    register(8, 4)
    assert {s: mlp_blocking(*s) for s in shapes} == before
    installed = [k for k, v in transformer.DP4_MLP_BLOCKINGS.items() if matmul.grid_12_9_configs.get(k) == v]
    assert not installed, installed


def test_dp4_rows_take_the_dp4_mlp_blockings(register):
    register(4, 2)
    for (m, _, n), (mb, kb, nb, _) in transformer.DP4_MLP_BLOCKINGS.items():
        assert mlp_blocking(m, n) == (mb, kb, nb), (m, n)
