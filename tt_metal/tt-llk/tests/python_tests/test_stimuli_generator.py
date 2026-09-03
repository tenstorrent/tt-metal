# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""``generate_stimuli(generate_operand_B=False)`` must change only what it claims to.

The unary SFPU drivers pass this to skip generating, packing and DMA-ing a whole operand
B their kernel never reads. The saving is real, but it puts a second path through
``generate_stimuli``, and the only caller is the sweep itself -- so a regression in it
(the ``_clamp_mx_tensors`` stand-in for the dropped B, the ``None`` return, the tile
count that still sizes B's L1 region) would surface as a golden mismatch several stages
downstream rather than as a failure pointing here.

The contract, in one sentence: A comes out bit-identical to the two-operand call, B comes
back as ``None``, and ``tile_cnt_B`` is unchanged -- because B's L1 region is still
reserved and still declared in the generated header. Each clause is a test below.
"""

import pytest
import torch
from helpers.format_config import MX_FORMAT_MAX_NORMAL, DataFormat
from helpers.stimuli_generator import StimuliSpec, generate_stimuli

# Enough tiles that a wrong tile count is visible rather than coincidentally right.
DIMENSIONS_A = [32, 128]
DIMENSIONS_B = [32, 64]


# Reseeded before each call rather than relying on ``spec.seed``: a spec with no seed
# draws from the global RNG, which is the case the default per-format specs hit, and A is
# generated before B in both branches -- so an identical global seed makes A's draws
# identical and any difference attributable to the new code path.
_SEED = 1234


def _both_ways(**kwargs):
    """The same call with and without operand B, from the same RNG state."""
    torch.manual_seed(_SEED)
    with_b = generate_stimuli(**kwargs, generate_operand_B=True)
    torch.manual_seed(_SEED)
    without_b = generate_stimuli(**kwargs, generate_operand_B=False)
    return with_b, without_b


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param(
            {
                "stimuli_format_A": DataFormat.Float32,
                "stimuli_format_B": DataFormat.Float32,
                "input_dimensions_A": DIMENSIONS_A,
                "input_dimensions_B": DIMENSIONS_B,
                "spec_A": StimuliSpec.uniform(low=-5.0, high=5.0, seed=0),
                "spec_B": StimuliSpec.uniform(low=-5.0, high=5.0, seed=1),
            },
            id="standard_fp32",
        ),
        pytest.param(
            {
                "stimuli_format_A": DataFormat.Float16_b,
                "stimuli_format_B": DataFormat.Bfp8_b,
                "input_dimensions_A": DIMENSIONS_A,
                "input_dimensions_B": DIMENSIONS_B,
            },
            id="mixed_formats_default_specs",
        ),
        pytest.param(
            {
                "stimuli_format_A": DataFormat.Float16_b,
                "stimuli_format_B": DataFormat.Float16_b,
                "input_dimensions_A": DIMENSIONS_A,
                "input_dimensions_B": DIMENSIONS_B,
                "tile_dimensions": [16, 32],
            },
            id="dense_mode",
        ),
        pytest.param(
            {
                "stimuli_format_A": DataFormat.Float16_b,
                "stimuli_format_B": DataFormat.Float16_b,
                # face_r_dim < 16 is the partial single-tile case, so it has to
                # equal input_dimensions_A[0].
                "input_dimensions_A": [8, 32],
                "input_dimensions_B": [8, 32],
                "face_r_dim": 8,
                "num_faces": 2,
            },
            id="partial_faces",
        ),
    ],
)
def test_dropping_operand_b_leaves_a_and_the_tile_counts_alone(kwargs):
    """A is bit-identical, B is None, and both tile counts are unchanged."""
    (a_ref, cnt_a_ref, b_ref, cnt_b_ref), (a, cnt_a, b, cnt_b) = _both_ways(**kwargs)

    assert b_ref is not None, "the two-operand call should still produce a B"
    assert b is None, "generate_operand_B=False must return None in B's place"

    # B's L1 region is still reserved from this count -- the operands are laid out
    # contiguously, so a count that changed here would move buffer_Res.
    assert cnt_b == cnt_b_ref
    assert cnt_a == cnt_a_ref

    assert a.dtype == a_ref.dtype
    assert a.shape == a_ref.shape
    assert torch.equal(a, a_ref), "A must not depend on whether B was generated"


@pytest.mark.parametrize(
    "output_format", [DataFormat.MxFp8P, DataFormat.MxFp8R], ids=lambda f: f.name
)
def test_the_mx_output_clamp_still_applies_to_a(output_format):
    """The clamp is not purely pairwise, so dropping B must not drop it.

    ``_clamp_mx_tensors`` has two independent halves: one fires when A and B are
    *different* MX formats, the other when the **output** format is an MX one -- and the
    second clamps A on its own account. Skipping the call for a missing B would silently
    widen A's range for an MX output; the empty stand-in (``srcA_tensor[:0]``) is what
    keeps the A-side behaviour identical, and this is the test that says so.

    The population deliberately exceeds the format's max normal, so an unclamped A is a
    visible failure rather than a no-op.
    """
    limit = MX_FORMAT_MAX_NORMAL[output_format]
    kwargs = {
        "stimuli_format_A": DataFormat.Float32,
        "stimuli_format_B": DataFormat.Float32,
        "input_dimensions_A": DIMENSIONS_A,
        "input_dimensions_B": DIMENSIONS_B,
        "spec_A": StimuliSpec.uniform(low=-4 * limit, high=4 * limit, seed=0),
        "spec_B": StimuliSpec.uniform(low=-4 * limit, high=4 * limit, seed=1),
        "output_format": output_format,
    }
    (a_ref, _, _, _), (a, _, b, _) = _both_ways(**kwargs)

    assert b is None
    assert torch.equal(a, a_ref)
    assert a.abs().max().item() <= limit, (
        f"A was not clamped to the {output_format.name} max normal ({limit}); the "
        "output-format half of _clamp_mx_tensors was skipped along with operand B"
    )
    # And the clamp actually bit, so this is not passing on an already-in-range
    # population.
    assert (a.abs() == limit).any(), (
        "no element reached the clamp bound, so this population no longer exercises "
        "the output-format clamp"
    )


def test_mixed_mx_input_formats_still_clamp_a():
    """The other half of the clamp: A and B being different MX formats clamps A too."""
    limit = MX_FORMAT_MAX_NORMAL[DataFormat.MxFp8P]
    kwargs = {
        "stimuli_format_A": DataFormat.MxFp8R,
        "stimuli_format_B": DataFormat.MxFp8P,
        "input_dimensions_A": DIMENSIONS_A,
        "input_dimensions_B": DIMENSIONS_B,
        "spec_A": StimuliSpec.uniform(low=-4 * limit, high=4 * limit, seed=0),
        "spec_B": StimuliSpec.uniform(low=-4 * limit, high=4 * limit, seed=1),
    }
    (a_ref, _, _, _), (a, _, b, _) = _both_ways(**kwargs)

    assert b is None
    assert torch.equal(a, a_ref)
    assert a.abs().max().item() <= limit
    assert (a.abs() == limit).any(), (
        "no element reached the MxFp8P bound, so this population no longer exercises "
        "the mixed-MX-input clamp"
    )
