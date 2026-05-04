# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only unit tests for formula-based shard distribution. No hardware required."""
import pytest
from ttnn.experimental.moe_compute_utils import shard_tiles, w2_shard_tiles, auto_output_width_shard_dim


def test_shard_tiles_deepseek_w0w1():
    """DS: Nt=64, n_cores=12 → big=6, small=5; big at positions where (i*4)%12 < 4 = {0,3,6,9}."""
    n_tiles, n_cores = 64, 12
    result = [shard_tiles(n_tiles, c, n_cores) for c in range(n_cores)]
    assert sum(result) == n_tiles
    assert result[0] == 6  # (0*4)%12=0 < 4 → big
    assert result[1] == 5  # (1*4)%12=4, not < 4 → small
    assert result[3] == 6  # (3*4)%12=0 < 4 → big
    assert result[6] == 6
    assert result[9] == 6
    assert all(v in (5, 6) for v in result)


def test_shard_tiles_gpt_w0w1():
    """GPT: Nt=90, n_cores=12 → big=8, small=7; exactly 6 big cores."""
    n_tiles, n_cores = 90, 12
    result = [shard_tiles(n_tiles, c, n_cores) for c in range(n_cores)]
    assert sum(result) == n_tiles
    assert result.count(8) == 6
    assert result.count(7) == 6


def test_shard_tiles_glm5_w0w1():
    """GLM-5: Nt=64 (same as DS), same distribution."""
    result = [shard_tiles(64, c, 12) for c in range(12)]
    assert sum(result) == 64
    assert result[0] == 6 and result[3] == 6  # same big positions as DS


def test_shard_tiles_exactly_divisible():
    """DS V4 Pro: Nt=96, 96%12=0 → all cores get 8."""
    result = [shard_tiles(96, c, 12) for c in range(12)]
    assert all(v == 8 for v in result)
    assert sum(result) == 96


def test_w2_shard_tiles_deepseek():
    """DS: Ht=224, Nt=64, n=12, n_big_nt+n_big_ht=4+8=12 → complement.
    W0W1-big cores {0,3,6,9} get small W2 (18 tiles);
    W0W1-small cores get big W2 (19 tiles)."""
    result = [w2_shard_tiles(224, c, 64, 12) for c in range(12)]
    assert sum(result) == 224
    assert result[0] == 18  # W0W1-big → small W2
    assert result[1] == 19  # W0W1-small → big W2
    assert result[3] == 18
    assert result[2] == 19


def test_w2_shard_tiles_glm5():
    """GLM-5: Ht=192, Nt=64, n=12 → Ht%12=0 so n_big_ht=0.
    n_big_nt+n_big_ht=4+0≠12 → fallback to shard_tiles(192,c,12)=16 uniform."""
    result = [w2_shard_tiles(192, c, 64, 12) for c in range(12)]
    assert all(v == 16 for v in result)
    assert sum(result) == 192


def test_w2_shard_tiles_dsv4_flash():
    """DS V4 Flash: Ht=128, Nt=64, n=12. n_big_nt=4, n_big_ht=8, 4+8=12 → complement."""
    result = [w2_shard_tiles(128, c, 64, 12) for c in range(12)]
    assert sum(result) == 128
    assert result[0] == 128 // 12  # W0W1-big → small W2
    assert result[1] == 128 // 12 + 1  # W0W1-small → big W2


def test_w2_shard_tiles_gpt():
    """GPT: Ht=90, Nt=90, n=12. n_big_nt=6, n_big_ht=6, 6+6=12 → complement."""
    result = [w2_shard_tiles(90, c, 90, 12) for c in range(12)]
    assert sum(result) == 90
    assert result.count(8) + result.count(7) == 12


def test_auto_output_width_shard_dim():
    assert auto_output_width_shard_dim(7168) == 4  # DS: Ht=224, 224%4=0
    assert auto_output_width_shard_dim(2880) == 3  # GPT: Ht=90, 90%4≠0, 90%3=0
    assert auto_output_width_shard_dim(6144) == 4  # GLM-5: Ht=192, 192%4=0
    assert auto_output_width_shard_dim(8192) == 4  # Ling-1T: Ht=256
    assert auto_output_width_shard_dim(5120) == 4  # GLM-4.7: Ht=160
    assert auto_output_width_shard_dim(4096) == 4  # DS V4 Flash: Ht=128
    assert auto_output_width_shard_dim(7168) == 4  # Kimi K2.5: same as DS


def test_shard_tiles_total_always_correct():
    """Property: sum of all shard_tiles == n_tiles, for all interesting model shapes."""
    shapes = [(64, 12), (90, 12), (96, 12), (192, 12), (48, 12), (128, 12), (224, 12), (256, 12)]
    for n_tiles, n_cores in shapes:
        result = [shard_tiles(n_tiles, c, n_cores) for c in range(n_cores)]
        assert sum(result) == n_tiles, f"Failed for n_tiles={n_tiles}, n_cores={n_cores}"
