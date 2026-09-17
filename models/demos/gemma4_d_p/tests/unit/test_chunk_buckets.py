# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host checks for per-request prefill chunk-width selection."""

from types import SimpleNamespace

import pytest

from models.demos.gemma4_d_p.config import MeshConfig
from models.demos.gemma4_d_p.tt.chunk_buckets import (
    PREFILL_CHUNK_COST_MS,
    bucket_switch_points,
    modelled_prefill_ms,
    select_chunk_size,
    validate_chunk_sizes,
)
from models.demos.gemma4_d_p.tt.model import normalize_prefill_chunk_sizes

BUCKETS = (4096, 32768)


def test_a_c_is_sublinear_up_to_16384_then_turns():
    """The premise of the design: a narrow chunk is not intrinsically cheaper per token.

    ``a(C)`` is sublinear up to 16384 -- the per-chunk floor (weight reads, norms, the constant
    sliding halo) is amortised over more tokens -- so for tokens that FILL the chunk, wider
    always wins there. It turns superlinear at 32768 because intra-chunk causal self-attention
    is O(C^2); that is what bounds the useful width, and why the widest bucket is 32768 and not
    the context length.
    """
    for narrow, wide in zip([2048, 4096, 8192], [4096, 8192, 16384]):
        a_narrow, a_wide = PREFILL_CHUNK_COST_MS[narrow][0], PREFILL_CHUNK_COST_MS[wide][0]
        assert a_wide < a_narrow * (wide / narrow), f"chunk {wide} is not sublinear in chunk {narrow}"
        assert modelled_prefill_ms(wide, wide) < modelled_prefill_ms(wide, narrow)
    assert PREFILL_CHUNK_COST_MS[32768][0] > 2 * PREFILL_CHUNK_COST_MS[16384][0]


def test_padding_is_what_makes_a_narrow_bucket_win():
    """A prompt that fills the narrow chunk but not the wide one picks the narrow one."""
    assert select_chunk_size(4096, BUCKETS) == 4096
    assert modelled_prefill_ms(4096, 32768) == modelled_prefill_ms(32768, 32768)


def test_long_prompts_pick_the_wide_bucket():
    assert select_chunk_size(262144, BUCKETS) == 32768


def test_selection_is_sawtooth_not_a_single_threshold():
    """Padding waste recurs at every wide-chunk boundary, so the policy is not one threshold.

    A prompt just past a multiple of 32768 pays for a whole extra wide chunk that is nearly all
    padding, and the narrow bucket wins again. Any admission policy written as
    ``narrow if prompt < T else wide`` is wrong in that band -- at 36864 tokens it would pick
    32768 and pay 1.18x.
    """
    plan = bucket_switch_points(BUCKETS, max_prompt_len=262144)
    assert [chunk for _, chunk in plan] == [4096, 32768, 4096, 32768], plan
    assert [prompt for prompt, _ in plan] == [4096, 24576, 36864, 45056], plan
    # The relapse band is real, not a rounding artefact.
    assert modelled_prefill_ms(36864, 4096) < modelled_prefill_ms(36864, 32768)
    # Past 2 wide chunks the prefix term settles it for good.
    for prompt_len in range(45056, 262145, 4096):
        assert select_chunk_size(prompt_len, BUCKETS) == 32768


def test_unmeasured_width_is_refused_rather_than_interpolated(expect_error):
    with expect_error(KeyError, "no measured cost"):
        modelled_prefill_ms(4096, 1024)


@pytest.mark.parametrize("prompt_len", [0, -1])
def test_non_positive_prompt_is_rejected(prompt_len, expect_error):
    with expect_error(ValueError, "must be positive"):
        modelled_prefill_ms(prompt_len, 4096)


def test_every_configured_width_is_geometry_checked(expect_error):
    mesh_config = MeshConfig(SimpleNamespace(shape=(8, 4)))
    assert validate_chunk_sizes((32768, 4096), mesh_config.cp_degree, 262144) == (4096, 32768)
    # 512 gives a 64-token Q slab at CP=8: 16 halo hops around an 8-rank ring.
    with expect_error(ValueError, "prefill chunk 512"):
        validate_chunk_sizes((4096, 512), mesh_config.cp_degree, 262144)
    # max_seq_len must be a whole number of chunks at EVERY width, not just the widest.
    with expect_error(ValueError, "prefill chunk 24576"):
        validate_chunk_sizes((24576,), mesh_config.cp_degree, 262144)


@pytest.mark.parametrize(
    "value,expected",
    [(8192, (8192,)), ([32768, 4096], (4096, 32768)), ((4096, 4096), (4096,))],
)
def test_widths_normalize_to_a_sorted_deduplicated_tuple(value, expected):
    assert normalize_prefill_chunk_sizes(value) == expected
