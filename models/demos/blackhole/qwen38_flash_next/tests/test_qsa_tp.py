# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""QSA decode on the 1x4 mesh against the torch reference: one checkpoint layer, positions 0..5 with the KV and index
caches carried on both sides, the checkpoint's RoPE at each position, PCC per position.  Position 3 closes the first
four-token index block (the device needs the block's first-token RoPE for that) and positions 4 and 5 attend across
the block boundary.  At this length the 2,048-token budget covers every causal token, so the device's sparse selection
and the reference's dense attention see the same keys.

test_qsa_component.py proves ``Qwen38QSA`` against the pinned Transformers module on the CPU; this is the device side.

Run:  pytest models/demos/blackhole/qwen38_flash_next/tests/test_qsa_tp.py -v -s
Env:  MODEL_WEIGHTS_DIR, MESH_DEVICE="(1, 4)"; QWEN38_CACHE_ROOT optional (component cache reuse)
"""

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.blackhole.qwen38_flash_next.tests.tp_harness import DEVICE_PARAMS, HIDDEN_SIZE, pcc
from models.demos.blackhole.qwen38_flash_next.tt.model import text_rope
from models.demos.blackhole.qwen38_flash_next.tt.qsa import Qwen38QSA, Qwen38QSAWeights
from models.demos.blackhole.qwen38_flash_next.ttnn.builder import RESIDENT_DEFAULT_QSA_CACHE_CAPACITY
from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import MESH_SHAPE
from models.demos.blackhole.qwen38_flash_next.ttnn.qsa import COMPRESS_RATIO, Qwen38TTNNQSA, Qwen38TTNNQSAWeights

pytestmark = pytest.mark.skipif(
    os.environ.get("QWEN38_FUSED_DEVICE_TEST") != "1",
    reason="a device test: set QWEN38_FUSED_DEVICE_TEST=1 on a held four-die line (the no-device sets are masked)",
)

POSITIONS = 6


@run_for_blackhole()
@pytest.mark.timeout(600)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("layer_index", [3, 47])  # the first and the last QSA layer
def test_qsa_tp(tp_harness, layer_index, pcc_thresholds):
    harness = tp_harness
    weights = Qwen38TTNNQSAWeights.from_checkpoint(
        harness.checkpoint,
        harness.placement,
        harness.mesh_device,
        harness.contract,
        harness.component_cache_root,
        layer_index=layer_index,
        tt_metal_sha=harness.tt_metal_sha,
    )
    module = Qwen38TTNNQSA(
        harness.mesh_device,
        harness.contract,
        weights,
        layer_index=layer_index,
        rms_norm_eps=harness.config.rms_norm_eps,
        max_context=harness.config.max_position_embeddings,
        allocated_context=RESIDENT_DEFAULT_QSA_CACHE_CAPACITY,
        collective_topology=harness.topology,
    )
    reference = Qwen38QSA(Qwen38QSAWeights.from_checkpoint(harness.checkpoint, layer_index))
    generator = torch.Generator().manual_seed(2000 + layer_index)
    hidden = torch.randn((1, POSITIONS, HIDDEN_SIZE), generator=generator).to(torch.bfloat16)
    cos_table, sin_table = text_rope(harness.config, batch=1, length=POSITIONS)

    state = module.allocate_state()
    reference_state = None
    try:
        for position in range(POSITIONS):
            token = hidden[:, position : position + 1]
            device_token = harness.upload_sharded(token.reshape(1, 1, 1, HIDDEN_SIZE))
            # The production uploader (ttnn/model.py Qwen38TTNNRoPE.for_position, passed through by ttnn/layer.py
            # forward_decode): block_start_cos/sin are the block's first-token RoPE at positions 3 mod 4, else None.
            rope = harness.rope.for_position(position)
            result = module.forward_decode(
                device_token,
                state,
                cos=rope.cos,
                sin=rope.sin,
                block_start_cos=rope.block_start_cos,
                block_start_sin=rope.block_start_sin,
                position=position,
            )
            state = result.state
            got = harness.download_sharded(result.hidden_sharded).reshape(1, 1, HIDDEN_SIZE)
            ttnn.deallocate(device_token)
            ttnn.deallocate(result.hidden_sharded)
            rope.deallocate()

            expected, reference_state, selected = reference.forward(
                token,
                (cos_table[:, : position + 1], sin_table[:, : position + 1]),
                torch.zeros((1, 1, 1, position + 1), dtype=torch.float32),
                state=reference_state,
            )
            assert bool(selected.all()), "the reference's token budget must cover every causal token at this length"
            assert state.next_position == position + 1
            assert (state.compressed_blocks, state.raw_tail_count) == divmod(position + 1, COMPRESS_RATIO)
            score = pcc(got, expected)
            logger.info(
                f"layer {layer_index} position {position}: PCC {score:.6f}, "
                f"device selected {result.selection.valid_token_count} tokens"
            )
            assert score >= pcc_thresholds["test_qsa_tp"], f"position {position}: PCC {score:.6f}"
    finally:
        module.release_state(state)
        module.deallocate()
        weights.deallocate()
