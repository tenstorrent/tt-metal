# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Regression test: the rank-stamped TopK merge must not leak the SFPU's shared -1.0f.

The merge programs a lo16 clear mask into a programmable constant register and never
restores it. If that register is LREG11 (which sfpi::vConstNeg1 aliases and firmware
loads once at boot), every later bare read of -1.0f in the same device session is
corrupted. The kernel poisons with one merge, then probes with x - 1 computed against
that constant.

``unstable`` keeps the merge but compiles the mask write out, so it isolates the leak to
the stamping branch rather than to "a merge ran". ``rank_stamped`` is the variant under
test. Both must pass. Observed on Blackhole p100a before the fix: 1024/1024 elements
came back +inf.
"""

import torch
from conftest import skip_for_quasar
from helpers.format_config import DataFormat
from helpers.llk_params import DestAccumulation, TopKSortDirection, format_dict
from helpers.param_config import input_output_formats, parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import StimuliSpec, generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import TOPK
from helpers.utils import passed_test

# Quasar has no topk SFPU implementation. Wormhole and Blackhole both carry the path.
pytestmark = [skip_for_quasar]

ELEMENTS_PER_TILE = 1024

# Float16_b only: no format axis makes a shared-register clobber more visible.
FORMATS = input_output_formats([DataFormat.Float16_b], same=True)

# x - 1 is exact in bfloat16 over [1.0, 2.0], so the comparison can be strict.
PROBE_STIMULI = StimuliSpec.uniform(low=1.0, high=2.0)

# Zero tolerance; passed_test's Float16_b default is atol/rtol 0.05.
STRICT_ATOL = 0.0
STRICT_RTOL = 0.0

# k=32 at width 128 gives TOPK_NUM_ITERATIONS == 1, which rank-stamped requires.
TOPK_K = 32
TOPK_MATRIX_WIDTH = 128


def _templates(poison: str) -> list:
    """Template parameters for one poison variant."""
    return [
        TOPK(
            topk_k=TOPK_K,
            topk_matrix_width=TOPK_MATRIX_WIDTH,
            topk_sort_direction=TopKSortDirection.Descending,
            topk_stable_sort=False,
            topk_fused_stable=False,
            topk_rank_stamped=(poison == "rank_stamped"),
        )
    ]


@parametrize(
    formats=FORMATS,
    poison=["unstable", "rank_stamped"],
)
def test_topk_const_leak(formats, poison):
    torch.manual_seed(0)

    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=[32, 32],
        stimuli_format_B=formats.input_format,
        input_dimensions_B=[32, 32],
        spec_A=PROBE_STIMULI,
        spec_B=PROBE_STIMULI,
    )

    configuration = TestConfig(
        "sources/topk_const_leak_test.cpp",
        formats,
        templates=_templates(poison),
        runtimes=[],
        variant_stimuli=StimuliConfig(
            src_A.flatten(),
            formats.input_format,
            src_B.flatten(),
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=1,
        ),
        unpack_to_dest=False,
        # Pinned: rank-stamped static_asserts on 32-bit DEST.
        dest_acc=DestAccumulation.Yes,
    )

    res_from_L1 = configuration.run().result[:ELEMENTS_PER_TILE]
    torch_format = format_dict[formats.output_format]

    # Golden: the probe's own arithmetic, in fp32 off the same stimuli.
    golden = (src_A.flatten().to(torch.float32) - 1.0)[:ELEMENTS_PER_TILE]

    device = torch.tensor(res_from_L1, dtype=torch_format).flatten()
    golden = golden.to(torch_format).flatten()

    # Non-finite first: it is this defect's signature and deserves its own message.
    nonfinite = ~torch.isfinite(device)
    nonfinite_count = int(nonfinite.sum().item())
    assert nonfinite_count == 0, (
        f"{nonfinite_count} of {ELEMENTS_PER_TILE} probe elements are non-finite "
        f"(poison={poison}): the shared -1.0f constant no longer holds -1.0f after "
        f"the merge. The 'unstable' variant runs the identical probe and passes."
    )

    assert passed_test(
        golden,
        device,
        formats.output_format,
        custom_atol=STRICT_ATOL,
        custom_rtol=STRICT_RTOL,
        print_errors=True,
    ), (
        f"the 'x - 1' probe returned wrong finite values with poison={poison}. "
        f"x - 1 is exact in bfloat16 here, so this is a bit-for-bit comparison."
    )
