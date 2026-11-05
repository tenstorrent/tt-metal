# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""tt-blaze G53F-0256: the GLM prefill export's host packers are byte-identical to the ring seed path's, and the table
geometry is the ring's (no device). The tt-blaze reference functions are lifted from its source file by name, so the
check needs neither tt-blaze's package import (which opens the chips) nor a copy kept in sync by hand:

    BLAZE_HELPERS=<tt-blaze>/blaze/kv_chunk_migration_helpers.py \\
    pytest --noconftest models/demos/glm53_flash/tests/test_runner_kv_export.py
"""

from __future__ import annotations

import ast
import os

import pytest
import torch

from models.demos.glm53_flash.tt.runners import kv_export as X

HELPERS = os.environ.get(
    "BLAZE_HELPERS", "/home/ttuser/sdawle/tt-blaze-glm53f-ttdgen/blaze/kv_chunk_migration_helpers.py"
)
_NAMES = (
    "GLM_KV_DIM",
    "GLM_KV_NOPE_DIM",
    "GLM_KV_ROPE_DIM",
    "GLM_SCALED_KV_TOKENS_PER_PAGE",
    "glm_scaled_kv_row_bytes",
    "glm_scaled_kv_physical_page_bytes",
    "glm_scaled_kv_rows_to_pages",
    "pack_glm_scaled_kv_cache",
)


def _blaze():
    if not os.path.exists(HELPERS):
        pytest.skip(f"tt-blaze helpers not found at {HELPERS} (BLAZE_HELPERS)")
    tree = ast.parse(open(HELPERS).read())
    keep = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in _NAMES:
            keep.append(node)
        elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in _NAMES for t in node.targets):
            keep.append(node)
    ns = {"torch": torch}
    exec(compile(ast.Module(body=keep, type_ignores=[]), HELPERS, "exec"), ns)
    missing = [n for n in _NAMES if n not in ns]
    assert not missing, f"{HELPERS} lacks {missing}"
    return ns


@pytest.mark.parametrize("n", [32, 256, 1024])
@pytest.mark.parametrize("pow2", [False, True])
def test_latent_pages_match_blaze(n, pow2):
    B = _blaze()
    torch.manual_seed(n)
    lat = (torch.randn(n, 512) * torch.logspace(-3, 2, n).unsqueeze(1)).to(torch.bfloat16)
    lat[0, :128] = 0  # an all-zero block takes the 1e-4 clamp
    rows, _ = B["pack_glm_scaled_kv_cache"](lat, rope_dim=0, pow2_scale=pow2, reconstruct=False)
    want = B["glm_scaled_kv_rows_to_pages"](rows, 0)
    got = X.pack_latent_pages(lat, pow2_scale=pow2)
    assert got.dtype == torch.uint8 and tuple(got.shape) == tuple(want.shape) == (n // 32, X.LATENT_PAGE_BYTES)
    assert torch.equal(got, want)


def test_pool_rows_match_seed_tiles():
    ttnn = pytest.importorskip("ttnn")
    pools = torch.randn(96, 128)
    x = pools.float()
    # prefill_seed_apply._bfp8_tiles (tt-blaze), joined per 32-row tile row: the ring's 4 consecutive device pages
    r, w = x.shape
    tiles = x.reshape(r // 32, 32, w // 32, 32).permute(0, 2, 1, 3).contiguous().reshape(-1).numpy()
    words = ttnn._ttnn.bfp_utils.pack_bfp8(tiles, row_major_input=True)
    per = len(words) // ((r // 32) * (w // 32))
    seed = [words[i * per : (i + 1) * per].tobytes() for i in range((r // 32) * (w // 32))]
    got = X.pack_pool_rows(pools)
    assert tuple(got.shape) == (3, X.POOL_ROW_BYTES)
    for row in range(3):
        assert got[row].numpy().tobytes() == b"".join(seed[row * 4 : (row + 1) * 4])


def test_ring_geometry():
    assert (X.LATENT_PAGE_BYTES, X.POOL_ROW_BYTES, X.KDA_S_SEGMENT, X.KDA_CONV_SEGMENT) == (18432, 4352, 16384, 768)
    assert (X.KDA_WINDOW, X.KDA_S_STRIDE, X.KDA_CONV_STRIDE) == (24576, 96, 128)
    assert X.DSA_LAYERS == (3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43) and len(X.KDA_LAYERS) == 34
    assert X.kda_window_range(1, 2) == (0, 24576) and X.kda_window_range(4096, 2) == (24576, 49152)
