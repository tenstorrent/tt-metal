# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Host-only regression tests for the MLA tuned-matmul config guard. No device, no weights.

Fails on the code as it stood before `perf(mla): guard tuned matmul configs against the wrong K`.
"""

from types import SimpleNamespace

import pytest

import ttnn
from models.demos.deepseek_v3_d_p.tt.mla.mla import ttMLA


@pytest.mark.parametrize(
    "in0_block_w, kt, fits",
    [
        # DeepSeek's tuning: Kt = 7168/4/32 = 56, and 56 % 14 == 0.
        (14, 56, True),
        # The same tuned row reached by a model whose Kt is 32 (hidden 4096 / tp 4 / 32). 32 % 14 != 0,
        # so the matmul TT_FATALs: "Kt (32) must be divisible by in0_block_w (14)".
        (14, 32, False),
        (8, 32, True),
        # Kt = 64 (tp 2) divides by 8, so the config is ACCEPTED -- the guard's known gap: another
        # model's tiling then applies silently. Pinned here so a future change to that behaviour is
        # a deliberate decision rather than a surprise.
        (8, 64, True),
    ],
)
def test_tuned_matmul_cfg_is_rejected_when_block_width_does_not_divide_k(in0_block_w, kt, fits):
    """Guards `perf(mla): guard tuned matmul configs against the wrong K`.

    `MLA_MATMUL_CONFIG` is keyed on (weight_name, seq_len_local) and knows nothing about the
    variant's dimensions, so a config tuned for one model is applied to another with a different K.
    Rejecting a non-dividing block width degrades to the generic program config -- untuned, so
    possibly slower -- instead of dying.
    """
    stub = SimpleNamespace(q_a_proj_weight=SimpleNamespace(shape=(kt * ttnn.TILE_SIZE, 1024)))
    cfg = {"program_config": SimpleNamespace(in0_block_w=in0_block_w)}
    assert (
        ttMLA._cfg_fits_weight(stub, cfg, "q_a_proj") is fits
    ), f"in0_block_w={in0_block_w} against Kt={kt}: expected fits={fits}"


def test_tuned_matmul_cfg_kept_when_there_is_nothing_to_check_against():
    """A config with no in0_block_w, or a weight that is absent, must not be rejected."""
    stub = SimpleNamespace(q_a_proj_weight=SimpleNamespace(shape=(1024, 1024)))
    assert (
        ttMLA._cfg_fits_weight(stub, {"program_config": SimpleNamespace()}, "q_a_proj") is True
    ), "a config with no in0_block_w has nothing to check against and must be kept"
    assert ttMLA._cfg_fits_weight(
        SimpleNamespace(), {"program_config": SimpleNamespace(in0_block_w=14)}, "q_a_proj"
    ), "an absent weight has nothing to check against and must be kept"
