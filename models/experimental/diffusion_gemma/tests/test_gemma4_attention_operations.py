from models.demos.gemma4.tt.attention.operations import _largest_tile_divisor


def test_largest_tile_divisor_never_returns_non_tile_multiple():
    assert _largest_tile_divisor(100, 100) == 32


def test_largest_tile_divisor_prefers_largest_aligned_divisor():
    assert _largest_tile_divisor(384, 256) == 192
    assert _largest_tile_divisor(512, 256) == 256
