# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Layout-agnostic V4.1 mesh checks (bead 11.1, no device): LoudBox 2x4 / 4x2 and Galaxy 8x4 / 4x8 accepted at
real and small dims, unsupported meshes and chunks rejected with the requirement they break."""

from types import SimpleNamespace

import pytest

from models.demos.deepseek_v3_d_p.reference.deepseek_v41_flash_config import DeepSeekV41FlashConfig as Flash
from models.demos.deepseek_v3_d_p.tests.v41.small_config import SmallV41Config as Small
from models.demos.deepseek_v3_d_p.tt.v41.cache import V41CacheGeometry
from models.demos.deepseek_v3_d_p.tt.v41.layout import V41MeshLayout

LAYOUTS = [(2, 4), (4, 2), (8, 4), (4, 8)]


@pytest.mark.parametrize("config", [Flash, Small], ids=["flash", "small"])
@pytest.mark.parametrize("shape", LAYOUTS, ids=[f"{s}x{t}" for s, t in LAYOUTS])
def test_supported_layouts(shape, config):
    layout = V41MeshLayout.of(SimpleNamespace(shape=shape))
    assert (layout.sp, layout.tp, layout.chips) == (*shape, shape[0] * shape[1])
    layout.check_model(config)


# smallest chunk each layout accepts: lcm(SP rule 2*32*sp, query rule 32*sp*tp) -> 256, 256, 1024, 1024;
# the production chunk 5120 gives 640 query rows per chip on 8 chips and 160 on 32
@pytest.mark.parametrize(
    "shape, smallest", [((2, 4), 256), ((4, 2), 256), ((8, 4), 1024), ((4, 8), 1024)], ids=lambda v: str(v)
)
def test_chunk_alignment(shape, smallest, expect_error):
    layout = V41MeshLayout(*shape)
    for chunk in (smallest, 5120):
        V41CacheGeometry(Flash, max_seq_len=4 * 5120, chunk=chunk, sp=layout.sp)
        layout.check_chunk(chunk)
    below = smallest // 2
    with expect_error(AssertionError, "multiple of"):
        V41CacheGeometry(Flash, max_seq_len=4 * 5120, chunk=below, sp=layout.sp)
        layout.check_chunk(below)


def test_query_split_rejects_partial_tiles(expect_error):
    # 2x4, chunk 128: the SP rule holds (64 rows per SP rank) but each chip would get 16 query rows
    V41CacheGeometry(Flash, max_seq_len=1024, chunk=128, sp=2)
    with expect_error(AssertionError, "32\\*sp\\*tp = 256"):
        V41MeshLayout(2, 4).check_chunk(128)


class _Heads32Hidden2048(Flash):
    EMB_SIZE = 2048
    NUM_ATTENTION_HEADS = 32


class _Heads48(Flash):
    NUM_ATTENTION_HEADS = 48


@pytest.mark.parametrize(
    "shape, config, message",
    [
        ((1, 3), Flash, "hidden 5120 does not split over TP=3"),  # 5120 / 3 is not whole tiles
        ((1, 64), _Heads32Hidden2048, "32 attention heads do not split over TP=64"),  # 2048 / 64 = 1 tile
        ((1, 4), _Heads48, "multiple of 32 heads per chip, got 48"),  # all heads per chip after head->sequence
        ((4, 16), Flash, "8 output groups do not split over TP=16"),  # 5120 / 16 = 10 tiles, 4 heads per chip
        ((5, 4), Flash, "384 routed experts do not split over 20 chips"),
    ],
    ids=["hidden_tiles", "heads_over_tp", "sdpa_heads", "output_groups", "experts"],
)
def test_unsupported_layouts(shape, config, message, expect_error):
    with expect_error(AssertionError, message):
        V41MeshLayout(*shape).check_model(config)


def test_mesh_must_be_2d(expect_error):
    with expect_error(AssertionError, "2D"):
        V41MeshLayout.of(SimpleNamespace(shape=(32,)))
