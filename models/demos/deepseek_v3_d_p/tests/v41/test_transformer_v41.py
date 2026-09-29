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
from models.demos.deepseek_v3_d_p.tt.v41.engram import TtV41Engram, V41EngramHash, V41EngramTable
from models.demos.deepseek_v3_d_p.tt.v41.transformer import TtV41Transformer
from models.demos.deepseek_v3_d_p.tt.v41.weights import dequant_fp8_block
from tests.ttnn.utils_for_testing import comp_pcc

SCHEDULES = {"sharing": (0, 2, 3, 20, 21, 24), "engram": (0, 1, 2, 3)}
SEQ = 512
STACK_PCC = 0.98


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("schedule", list(SCHEDULES))
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
def test_v41_transformer_small(mesh_device, device_params, case, schedule):
    layers = SCHEDULES[schedule]
    topology = per_axis_topology(device_params["fabric_config"])[1]
    chunk, total = {"one_chunk": (SEQ, SEQ), "two_chunks": (SEQ // 2, SEQ), "padded": (SEQ // 2, SEQ - 12)}[case]
    spec = small_spec(layers, SEQ)
    tokens = orc.random_tokens(spec, total)
    reference = orc.build_reference(spec)
    orc.load_engram_rows(reference, spec, tokens)  # synthetic Engram rows the prompt hashes to
    result = orc.oracle(spec, tokens, model=reference)
    engram, engram_hash = {}, None
    if reference.engram_hash is not None:
        engram_hash = V41EngramHash(SmallV41Config, reference.engram_hash.token_map)
        for pos, layer in enumerate(layers):
            e = reference.layers[pos].engram
            if e is None:
                continue
            weights = {
                "wkv": dequant_fp8_block(e.wkv.weight.detach(), e.wkv.scale.detach()),
                "q_weight": e.q_weight.detach(),
                "k_weight": e.k_weight.detach(),
            }
            table = V41EngramTable(e.embed.weight.detach(), e.embed.scale.detach(), e.embed.oracle_rows)
            engram[layer] = TtV41Engram(mesh_device, SmallV41Config, layer, weights, table, topology)
    model = TtV41Transformer(
        mesh_device,
        SmallV41Config,
        list(layers),
        lambda layer, include_moe: device_weights(reference, layers.index(layer)),
        reference.embed.weight.detach(),
        reference.norm.weight.detach(),
        reference.head.weight.detach(),
        max_seq_len=SEQ,
        chunk=chunk,
        engram=engram,
        engram_hash=engram_hash,
        topology=topology,
    )
    logits, _, _ = model.prefill(tokens[0])
    logits2, _, _ = model.prefill(tokens[0])
    expected = result["logits"].float()
    pcc = comp_pcc(expected, logits, 0.0)[1]
    top1 = int(expected.argmax()) == int(logits.argmax())
    # top-1 is only meaningful when the reference's own margin exceeds the device's logit error (G1 asks top-1
    # of the full model on Galaxy; stacks are gated on PCC)
    top2 = expected.topk(2).values
    margin, error = float(top2[0] - top2[1]), float((logits - expected).pow(2).mean().sqrt())
    decisive = margin > 3 * error
    logger.info(
        f"transformer {schedule} {case}: logits PCC {pcc:.5f}, top-1 {'match' if top1 else 'MISMATCH'} "
        f"(reference margin {margin:.4f}, device logit error {error:.4f}, {'decisive' if decisive else 'near tie'})"
    )
    assert torch.equal(logits, logits2), "prefill is not bit-identical across repeats"
    assert pcc >= STACK_PCC, pcc
    assert top1 or not decisive, (margin, error)
