# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
# Probe (not for the PR): every SFP_STOCH_RND int8 output, every tie and both saturation ends, checked bit for bit.

import torch
from conftest import skip_for_wormhole
from helpers.golden_generators import ELEMENTS_PER_TILE
from helpers.param_config import parametrize
from test_sfpu_quant_scalar import _reference, _run


@skip_for_wormhole
@parametrize(quant_op=["quant", "requant"], scale_form=["tile", "scalar"], shift=[0, 1, 2, 3])
def test_ra2_qprobe(quant_op, scale_form, shift):
    # x * 0.25 + 3 runs over [-128 + shift / 8, 127.75 + shift / 8] in quarter steps: every integer, every tie.
    k = torch.arange(ELEMENTS_PER_TILE, dtype=torch.int64) - 524
    if quant_op == "quant":
        src_A = k.to(torch.float32) + shift * 0.125 * 4 / 4
    else:
        src_A = (k + (shift if shift < 2 else -shift)).to(torch.int32)
    res = _run(quant_op, scale_form, src_A)
    golden = _reference(quant_op, src_A)
    bad = int((res.to(torch.int64) != golden.to(torch.int64)).sum())
    vals = set(res.tolist())
    print(f"QPROBE {quant_op} {scale_form} {shift} distinct={len(vals)} min={min(vals)} max={max(vals)} bad={bad}")
    assert bad == 0
    assert set(range(-127, 128)) <= vals
