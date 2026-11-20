# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host arithmetic only (run it under the mock Galaxy: ttnn's host conversions create the device context): what the
BF16 hop of a sharded block-format tensor (tests/test_stream_reference.py) does to a weight stored as two BFP4 terms
(fused/moe/layout.py two_terms: the stream is given w and w - BFP4(w)).

    as modelled      BFP4(w) + BFP4(w - BFP4(w))                           fused/moe/layout.py two_terms_value
    as stored        BFP4(BF16(w)) + BFP4(BF16(w - BFP4(w)))               what the two sharded terms hold
    consistent       q1 = BFP4(BF16(w)), then q1 + BFP4(BF16(w - q1))      the second term made for the first term
                                                                           the device really has
    pre-rounded      the stream is given BFP4(w) and BFP4(w - BFP4(w)) themselves (already representable, so the
                     BF16 hop and the second rounding change nothing)

Relative rms error against the float32 weights, random weights at a projection's scale and the checkpoint's q_a
columns of chip 13, layer 1 when $HY4_HF_MODEL is set. Log lines start with HY4_MK9."""

import os

import pytest
import torch
from loguru import logger

from models.experimental.hy4_preview.fused.fidelity import lab as LAB


def rel(ref, v):
    return ((v.double() - ref.double()).pow(2).sum().sqrt() / ref.double().pow(2).sum().sqrt()).item()


def forms(w):
    one, sharded = (lambda m: LAB.stored(m, "bfp4")), (lambda m: LAB.stored_sharded(m, "bfp4"))
    q1 = one(w)
    q1d = sharded(w)
    return {
        "one BFP4 term, one rounding": q1,
        "one BFP4 term, sharded": q1d,
        "two terms as modelled": q1 + one(w - q1),
        "two terms as stored": q1d + sharded(w - q1),
        "two terms, consistent": q1d + sharded(w - q1d),
        "two terms, pre-rounded": sharded(q1) + sharded(one(w - q1)),
        "BFP8, one rounding": LAB.stored(w, "bfp8"),
        "BFP8, sharded": LAB.stored_sharded(w, "bfp8"),
    }


@pytest.mark.parametrize("source", ["random", "checkpoint"])
def test_two_terms_host(source):
    if not LAB.MOCK:
        pytest.skip("host arithmetic: run under the mock Galaxy")
    if source == "random":
        g = torch.Generator().manual_seed(6)
        w = torch.randn(6144, 64, generator=g) * 6144**-0.5
    else:
        if not os.environ.get("HY4_HF_MODEL"):
            pytest.skip("HY4_HF_MODEL is not set")
        from models.experimental.hy4_preview.fused.projections import layout as L
        from models.experimental.hy4_preview.fused.projections import prepare

        w = L.chip_weights(prepare.layer_weights(1), 13)["q_a"].contiguous()
    values = forms(w)
    for name, v in values.items():
        logger.info(f"HY4_MK9 two terms ({source}) {name}: relative rms error to the float32 weights {rel(w, v):.3e}")
    modelled, stored = values["two terms as modelled"], values["two terms as stored"]
    logger.info(
        f"HY4_MK9 two terms ({source}): as stored differs from as modelled in "
        f"{float((modelled != stored).float().mean()):.4f} of the entries, relative rms {rel(modelled, stored):.3e}"
    )
    assert torch.equal(values["two terms, pre-rounded"], modelled)
