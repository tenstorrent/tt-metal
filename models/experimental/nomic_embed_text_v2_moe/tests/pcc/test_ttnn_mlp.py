# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Module PCC for TtNomicBertMLP, the dense FFN on even-numbered layers. Bring-up gate 4."""

import pytest
import torch

import ttnn

from models.common.metrics import compute_max_abs_error, compute_pcc
from models.common.utility_functions import run_for_blackhole
from models.experimental.nomic_embed_text_v2_moe.reference.modeling_nomic_moe import NomicBertMLP
from models.experimental.nomic_embed_text_v2_moe.tests.pcc.module_common import (
    DENSE_LAYER,
    TOKEN_SHAPES,
    from_block_layout,
    hidden_states,
    load_reference,
    to_block_layout,
)
from models.experimental.nomic_embed_text_v2_moe.tt.common import to_device
from models.experimental.nomic_embed_text_v2_moe.tt.mlp import TtNomicBertMLP
from tests.ttnn.utils_for_testing import assert_with_pcc

pytestmark = [run_for_blackhole(), pytest.mark.use_module_device, pytest.mark.needs_weights]

MODULE_PCC = 0.999

PREFIX = f"encoder.layers.{DENSE_LAYER}.mlp."


@pytest.fixture
def reference(config, state_dict):
    return load_reference(lambda: NomicBertMLP(config), state_dict, PREFIX)


@pytest.fixture
def tt_mlp(device, config, tt_config, state_dict):
    return TtNomicBertMLP(device, config, tt_config, state_dict, PREFIX)


@pytest.mark.parametrize("batch, seqlen", TOKEN_SHAPES)
def test_mlp(device, config, reference, tt_mlp, batch, seqlen):
    """H -> F -> H with an exact-erf GELU between, against the reference module."""
    x = hidden_states(batch, seqlen, config.hidden_size)

    out = tt_mlp(to_device(to_block_layout(x), device))

    with torch.no_grad():
        ref = reference(x)
    assert tuple(out.shape) == (batch, 1, seqlen, config.hidden_size)
    assert_with_pcc(ref, from_block_layout(out), MODULE_PCC)


def test_gelu_stays_accurate(device, config, reference, tt_mlp, tt_config, state_dict):
    """The approximate GELU is a worse fit than bfloat16 itself, so the module must not use it.

    ttnn.gelu defaults to the accurate variant, which matches nn.GELU(approximate="none"). The
    LUT variant measures 2.34e-2 max-abs against exact erf, above the 1.58e-2 bfloat16 noise
    floor, so it is not swamped by the dtype. The repo's BERT idiom
    fused_activation=(ttnn.UnaryOpType.GELU, True) selects that LUT; this is the guard against
    reaching for it later.
    """
    x = hidden_states(2, 128, config.hidden_size)
    x_tt = to_device(to_block_layout(x), device)

    with torch.no_grad():
        ref = to_block_layout(reference(x))

    accurate = ttnn.to_torch(tt_mlp(x_tt)).float()
    intermediate = ttnn.linear(
        x_tt,
        tt_mlp.fc1_weight,
        bias=tt_mlp.fc1_bias,
        compute_kernel_config=tt_config.compute_kernel_config,
    )
    approximate = ttnn.to_torch(
        ttnn.linear(
            ttnn.gelu(intermediate, fast_and_approximate_mode=True),
            tt_mlp.fc2_weight,
            bias=tt_mlp.fc2_bias,
            compute_kernel_config=tt_config.compute_kernel_config,
        )
    ).float()

    assert compute_pcc(accurate, ref) > MODULE_PCC
    assert compute_max_abs_error(approximate, ref) > compute_max_abs_error(accurate, ref), (
        "the approximate GELU no longer costs accuracy here; re-check whether the accurate "
        "variant is still worth its cost"
    )
