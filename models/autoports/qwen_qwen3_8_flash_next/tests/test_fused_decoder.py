# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Offline target-shape CI for the delivered fused default, including odd tails.

The shared correctness body checks HF prefill, three changing traced decode
inputs, reset determinism and remapped paged KV. Real checkpoint, full resident
PLE, capacity and long-context acceptance use the recorded stage probes.
"""

import gc

import pytest
import torch
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextDecoderLayer

from models.autoports.qwen_qwen3_8_flash_next.tests import test_synthetic_decoder as shared
from models.autoports.qwen_qwen3_8_flash_next.tests.synthetic_weights import layer_weights, target_config
from models.autoports.qwen_qwen3_8_flash_next.tt.fused_decoder import (
    FusedDecoder,
    FusedExactQKV,
    FusedOrderedIndexerProjection,
    FusedPreciseProjection,
    FusedQSA,
    FusedResidualWriter,
    FusedSparseMoe,
)

single_chip_mesh = shared.single_chip_mesh


@pytest.fixture(scope="module", params=(0, 3), ids=("gdn", "indexed_qsa"))
def fused_bundle(request, single_chip_mesh):
    config = target_config()
    weights = layer_weights(request.param)
    with torch.device("meta"):
        reference = Qwen4ExpTextDecoderLayer(config, request.param)
    reference.load_state_dict(weights, assign=True)
    reference.eval()
    model = FusedDecoder.from_state_dict(
        weights, hf_config=config, layer_idx=request.param, mesh_device=single_chip_mesh
    )
    assert isinstance(model.attn_mix, FusedResidualWriter)
    assert isinstance(model.mlp_mix, FusedResidualWriter)
    assert model.mlp.compact_prefill_limit == 2
    assert type(model) is FusedDecoder
    assert isinstance(model.mlp, FusedSparseMoe)
    assert model.mlp.compact_experts and hasattr(model.mlp, "expert_packed")
    assert tuple(model.mlp.program.decode_gate_up_cores) == (10, 4)
    assert model.mlp.fuse_packed_activation and model.mlp.early_packed_activation
    assert model.mlp.fuse_moe_cleanup and model.mlp.precise_router.widen_in_unpack
    if request.param == 0:
        assert isinstance(model.attn.qkv, FusedExactQKV)
        assert model.attn.compact_recurrence_io
    assert isinstance(model.mlp.precise_router, FusedPreciseProjection)
    assert hasattr(model.mlp, "reduce_indices"), "Default must exercise the fused expert reduction"
    if request.param == 3:
        assert isinstance(model.attn, FusedQSA)
        assert isinstance(model.attn.project_index_qk, FusedOrderedIndexerProjection)
        assert model.chunked_attn.fuse_selector
    yield config, request.param, reference, model
    del model, reference, weights
    gc.collect()


@pytest.mark.parametrize("sequence", (1, 2, 31, 33, 34, 65, 66))
def test_fused_prefill_and_traced_decode(fused_bundle, single_chip_mesh, sequence, record_property):
    shared.test_full_decoder_prefill_and_traced_decode(fused_bundle, single_chip_mesh, sequence, record_property)
