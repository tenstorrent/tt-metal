# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V4.1 prefill transformer (bead F9): embed -> layers -> final collapse -> norm -> head, chunked.

Compares the last token's logits with the oracle's single-shot prefill of the same layer subset (PCC and
top-1), for one chunk, two chunks, and a padded last chunk; repeats are bit-identical. Bar (G1 stacks): >= 0.98.
"""

import pytest
import torch
from loguru import logger

from models.demos.deepseek_v3_d_p.reference.deepseek_v41 import oracle as orc
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import fabric2d_device_params
from models.demos.deepseek_v3_d_p.tests.v41.prototype_oracle import device_weights
from models.demos.deepseek_v3_d_p.tests.v41.small_config import SmallV41Config, small_spec
from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology
from models.demos.deepseek_v3_d_p.tt.v41.transformer import TtV41Transformer
from tests.ttnn.utils_for_testing import comp_pcc

LAYERS = (0, 2, 3, 20, 21, 24)
SEQ = 512
STACK_PCC = 0.98


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("case", ["one_chunk", "two_chunks", "padded"])
@pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        pytest.param(
            (2, 4),
            fabric2d_device_params(),
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(2, 4), topology="mesh-2x4"),
            id="fabric2d-mesh-2x4",
        )
    ],
    indirect=True,
)
def test_v41_transformer_small(mesh_device, device_params, case):
    chunk, total = {"one_chunk": (SEQ, SEQ), "two_chunks": (SEQ // 2, SEQ), "padded": (SEQ // 2, SEQ - 12)}[case]
    spec = small_spec(LAYERS, SEQ)
    tokens = orc.random_tokens(spec, total)
    reference = orc.build_reference(spec)
    result = orc.oracle(spec, tokens, model=reference)
    model = TtV41Transformer(
        mesh_device,
        SmallV41Config,
        list(LAYERS),
        lambda layer, include_moe: device_weights(reference, LAYERS.index(layer)),
        reference.embed.weight.detach(),
        reference.norm.weight.detach(),
        reference.head.weight.detach(),
        max_seq_len=SEQ,
        chunk=chunk,
        topology=per_axis_topology(device_params["fabric_config"])[1],
    )
    logits, _, _ = model.prefill(tokens[0])
    logits2, _, _ = model.prefill(tokens[0])
    expected = result["logits"].float()
    pcc = comp_pcc(expected, logits, 0.0)[1]
    top1 = int(expected.argmax()) == int(logits.argmax())
    logger.info(f"transformer {case}: logits PCC {pcc:.5f}, top-1 {'match' if top1 else 'MISMATCH'}")
    assert torch.equal(logits, logits2), "prefill is not bit-identical across repeats"
    assert pcc >= STACK_PCC and top1, (pcc, top1)
