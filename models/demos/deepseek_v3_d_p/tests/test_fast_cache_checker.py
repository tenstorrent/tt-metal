# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""FastCacheChecker answers exactly what `any(cache_path.glob(pattern))` answers, from one listing. Host only.

Every completeness check that moved from `Path.glob` to the checker relies on that equivalence: a looser match
reports a cache complete that is not, and the model then loads placeholders without raising.
"""

import pytest

from models.demos.deepseek_v3_d_p.utils.fast_cache_checker import FastCacheChecker

_KDA_FULL_NAME = "layer_1.kda.norm.v3.0123456789abcdef.mesh4x4.tpaxis0_dtype_BFLOAT16_layout_TILE.tensorbin"

_NAMES = (
    "a.tensorbin",
    "ab",
    "abab",
    "layer_1.kda.norm.v3.0123.mesh4x4.tpaxis0_dtype_BFLOAT16_layout_TILE.tensorbin",
    "layer_1.kda.norm.v2.0123.mesh4x4.tpaxis0_dtype_BFLOAT16_layout_TILE.tensorbin",
    "layer_10.kda.norm.v3.0123.mesh4x4.tpaxis0_dtype_BFLOAT16_layout_TILE.tensorbin",
    "layer_3.mla.indexer_wk_dtype_BFLOAT8_B_layout_TILE.tensorbin",
    "layer_3.mla.indexer_wq_b_repl_dtype_BFLOAT8_B_layout_TILE.tensorbin",
    ".layer_3.complete",
    # KDAWeights.check_cache_complete asks for the full name as_tensor writes (no '*'); a longer name sharing it as a
    # prefix must not stand in for it.
    _KDA_FULL_NAME,
    _KDA_FULL_NAME.replace("mesh4x4", "mesh8x4") + ".partial",
    "zz",
)

_PATTERNS = (
    # prefix and suffix overlap in a name shorter than both together: glob's '*' cannot match a negative length
    "ab*ab",
    "ab*b",
    "a*a.tensorbin",
    "*",
    "*.tensorbin",
    "a*",
    "ab",  # literal, present
    "abc",  # literal, absent
    "layer_1.kda.norm.v3.*_dtype_BFLOAT16_layout_TILE.tensorbin",
    "layer_1.kda.norm.v3.*_dtype_FLOAT32_layout_TILE.tensorbin",
    "layer_10.kda.norm.v3.*_dtype_BFLOAT16_layout_TILE.tensorbin",
    "layer_1.kda.norm.v4.*_dtype_BFLOAT16_layout_TILE.tensorbin",
    "layer_3.mla.indexer_wq_b_dtype_BFLOAT8_B_*.tensorbin",  # must not match wq_b_repl
    "layer_3.mla.indexer_wq_b_repl_dtype_BFLOAT8_B_*.tensorbin",
    "layer_3.mla.indexer_wk_dtype_BFLOAT16_*.tensorbin",
    ".layer_3.*",
    "zz*",
    "zzz*",
    _KDA_FULL_NAME,  # literal, present
    _KDA_FULL_NAME.replace("mesh4x4", "mesh8x4"),  # literal, only a longer name has it as a prefix
    _KDA_FULL_NAME.replace("BFLOAT16", "FLOAT32"),  # literal, absent
)


@pytest.fixture
def cache_dir(tmp_path):
    for name in _NAMES:
        (tmp_path / name).touch()
    return tmp_path


@pytest.mark.parametrize("pattern", _PATTERNS)
def test_pattern_exists_matches_path_glob(cache_dir, pattern):
    assert FastCacheChecker(cache_dir).pattern_exists(pattern) is any(cache_dir.glob(pattern))


def test_one_listing_serves_every_pattern(cache_dir):
    checker = FastCacheChecker(cache_dir)
    first = [checker.pattern_exists(pattern) for pattern in _PATTERNS]
    (cache_dir / "abc").touch()  # a write after the listing is not seen: the caller owns the listing's lifetime
    assert [checker.pattern_exists(pattern) for pattern in _PATTERNS] == first


@pytest.mark.parametrize("pattern", ["a*b*", "a?", "a[b]*", "sub/a*"])
def test_unsupported_glob_syntax_is_rejected(cache_dir, pattern):
    # allow-pytest.raises: host-only test, runs without the root conftest (it opens the device) that defines expect_error
    with pytest.raises(ValueError, match="one '\\*'"):  # allow-pytest.raises: see above
        FastCacheChecker(cache_dir).pattern_exists(pattern)
