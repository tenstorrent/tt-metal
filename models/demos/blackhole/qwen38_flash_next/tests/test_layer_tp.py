# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""One decoder layer (gated residual read, attention, gated residual write, MoE) on the 1x4 mesh against the torch
reference: layer 0 (GDN) and layer 3 (QSA), positions 0..2 with a fresh random residual at each and the layer state
carried on both sides.  Two PCCs per position: the output residual, and the injection (output minus input residual),
which is the layer's own contribution.  Layer 1 (the PLE layer) needs token ids and is not covered here.

test_layer_component.py runs ``Qwen38DecoderLayer`` on the CPU; this is the device side.

Run:  pytest models/demos/blackhole/qwen38_flash_next/tests/test_layer_tp.py -v -s
Env:  MODEL_WEIGHTS_DIR, MESH_DEVICE="(1, 4)", QWEN38_CACHE_ROOT (skipped without it); QWEN38_BF4_CORPUS with
      QWEN38_BF4_CORPUS_VERIFICATION when the experts come from the CPU-staged corpus
"""

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.blackhole.qwen38_flash_next.tests.tp_harness import DEVICE_PARAMS, HIDDEN_SIZE, RESIDUAL_BRANCHES, pcc
from models.demos.blackhole.qwen38_flash_next.tt.layer import Qwen38DecoderLayer
from models.demos.blackhole.qwen38_flash_next.tt.model import text_rope
from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import MESH_SHAPE
from models.demos.blackhole.qwen38_flash_next.ttnn.layer import Qwen38TTNNLayerType

pytestmark = pytest.mark.skipif(
    os.environ.get("QWEN38_FUSED_DEVICE_TEST") != "1",
    reason="a device test: set QWEN38_FUSED_DEVICE_TEST=1 on a held four-die line (the no-device sets are masked)",
)

POSITIONS = 3
RESIDUAL_WIDTH = RESIDUAL_BRANCHES * HIDDEN_SIZE


@run_for_blackhole()
@pytest.mark.timeout(900)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("layer_index", [pytest.param(0, id="layer0-gdn"), pytest.param(3, id="layer3-qsa")])
def test_layer_tp(tp_harness, layer_index, pcc_thresholds):
    harness = tp_harness
    _builder, layer = harness.build_layer(layer_index)
    reference = Qwen38DecoderLayer.from_checkpoint(harness.checkpoint, harness.placement, layer_index=layer_index)
    generator = torch.Generator().manual_seed(4000 + layer_index)
    residuals = (torch.randn((1, POSITIONS, RESIDUAL_WIDTH), generator=generator) * 0.02).to(torch.bfloat16)
    is_qsa = layer.layer_type is Qwen38TTNNLayerType.QSA
    cos_table, sin_table = text_rope(harness.config, batch=1, length=POSITIONS)

    state = layer.allocate_state()
    reference_state = None
    try:
        for position in range(POSITIONS):
            residual = residuals[:, position : position + 1]
            # [1,1,10240] is branch-major.  The layer's residual is [1,4,1,640] per device (ttnn/layer.py
            # RESIDUAL_LOCAL_SHAPE: dim 1 the four branches, the hidden axis sharded, as ttnn/model.py _embed_residual
            # lays it out), so shard [1,4,1,2560] on its last axis.
            device_residual = harness.upload_sharded(residual.reshape(1, RESIDUAL_BRANCHES, 1, HIDDEN_SIZE))
            keyword = {"return_routing": True}
            rope = None
            if is_qsa:
                # As ttnn/model.py forward_decode passes them: the production uploader's four tensors
                # (block_start_cos/sin None except at positions 3 mod 4, where the index block closes).
                rope = harness.rope.for_position(position)
                keyword.update(
                    cos=rope.cos,
                    sin=rope.sin,
                    block_start_cos=rope.block_start_cos,
                    block_start_sin=rope.block_start_sin,
                )
            result = layer.forward_decode(device_residual, state, **keyword)  # consumes the input residual
            state = result.state
            # [1,4,1,2560] back to the reference's branch-major [1,1,10240].
            got = harness.download_sharded(result.residual_sharded).reshape(1, 1, RESIDUAL_WIDTH)
            indices = harness.download_replicated(result.aux.routing.indices).to(torch.int64).flatten()
            for tensor in (result.residual_sharded, result.aux.routing.scores, result.aux.routing.indices):
                ttnn.deallocate(tensor)
            if rope is not None:
                rope.deallocate()

            reference_keyword = {}
            if is_qsa:
                reference_keyword = {
                    "position_embeddings": (cos_table[:, : position + 1], sin_table[:, : position + 1]),
                    "attention_mask": torch.zeros((1, 1, 1, position + 1), dtype=torch.float32),
                }
            expected, reference_state, aux = reference.forward(residual, state=reference_state, **reference_keyword)
            assert state.position == position + 1
            # Router logits are bf16: a near-tie at the top-10 boundary may swap one expert on random inputs.
            reference_experts = aux.routing.indices.flatten().tolist()
            common = set(indices.tolist()) & set(reference_experts)
            assert (
                len(common) >= len(reference_experts) - 1
            ), f"position {position}: experts {indices.tolist()} vs reference {reference_experts}"
            residual_in = residual.to(torch.float32)
            score = pcc(got, expected)
            injection_score = pcc(got - residual_in, expected.to(torch.float32) - residual_in)
            logger.info(
                f"layer {layer_index} position {position}: PCC {score:.6f}, injection PCC {injection_score:.6f}"
            )
            assert score >= pcc_thresholds["test_layer_tp"], f"position {position}: PCC {score:.6f}"
            assert (
                injection_score >= pcc_thresholds["test_layer_tp_injection"]
            ), f"position {position}: injection PCC {injection_score:.6f}"
    finally:
        layer.release_state(state)
