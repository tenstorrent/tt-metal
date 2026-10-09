# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Checkpoint-free LB transformer orchestration with real embedding, norms, KDA and dense FFN.

The explicit plain residual arm isolates KDA request state from AttnRes qualification.
Existing checkpoint-backed transformer depth tests retain full architecture coverage.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.deepseek_v3_d_p.reference.kda import KDAConfig, kda_forward_reference
from models.demos.deepseek_v3_d_p.tests.kda.utils import (
    assert_matches_reference,
    mla_row_permutation,
    random_weights,
    reconstruct_sp_tp_tensor,
)
from models.demos.deepseek_v3_d_p.tt.kda.state_adapter import KdaContractGeometry, KdaStates, allocate_native_state
from models.demos.deepseek_v3_d_p.tt.kimi_k3.residual import PlainResidualStream
from models.demos.deepseek_v3_d_p.tt.kimi_k3.transformer import TtKimiK3Transformer
from tests.ttnn.unit_tests.operations.experimental.kda.kda_test_utils import assert_accurate, assert_bit_identical

pytestmark = run_for_blackhole()


@pytest.mark.parametrize("mesh_device", [pytest.param((2, 4), id="LB-SP2xTP4")], indirect=True)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_synthetic_kda_transformer_requests(mesh_device, device_params, monkeypatch):
    config = SimpleNamespace(vocab_size=512, hidden_size=256, intermediate_size=512, rms_norm_eps=1e-5)
    kda_config = KDAConfig(
        hidden_size=256,
        num_heads=96,
        head_k_dim=128,
        head_v_dim=128,
        conv_kernel_size=4,
        norm_eps=1e-5,
    )
    # Inject geometry only: build_attention still constructs the real policy-enabled ttKDA and adapter.
    monkeypatch.setattr(
        "models.demos.deepseek_v3_d_p.reference.kimi_k3_config.kimi_k3_kda_config",
        lambda: kda_config,
    )
    model_config = SimpleNamespace(NUM_LAYERS=1, NUM_DENSE_LAYERS=1, mla_layer_ids=lambda: ())
    generator = torch.Generator().manual_seed(59988)
    embedding = torch.randn(512, 256, generator=generator).bfloat16()
    ff = {
        "gate_proj": (torch.randn(512, 256, generator=generator) * 0.03).bfloat16(),
        "up_proj": (torch.randn(512, 256, generator=generator) * 0.03).bfloat16(),
        "down_proj": (torch.randn(256, 512, generator=generator) * 0.03).bfloat16(),
    }
    kda_weights = random_weights(kda_config)
    weights = {
        "embed_weight": embedding,
        "layers": [
            {
                "kda_weights": kda_weights,
                "ffn_weights": ff,
                "attn_norm_weight": torch.ones(256),
                "ffn_norm_weight": torch.ones(256),
            }
        ],
    }
    models = {}
    slabs = {}
    geometry = KdaContractGeometry.from_kda_config(kda_config, mesh_shape=(2, 4), sp_axis=0, tp_axis=1)
    for capacity in (128, 256):
        model = TtKimiK3Transformer(
            mesh_device,
            config,
            model_config,
            weights,
            num_layers=1,
            seq_len=capacity,
            residual_factory=lambda hidden, inherited: PlainResidualStream(hidden),
            topology=(ttnn.Topology.Linear, ttnn.Topology.Linear),
            build_tail=False,
            is_chunked=True,
            shared_expert_weights_dtype=ttnn.bfloat16,
        )
        models[capacity] = model
        slabs[capacity] = KdaStates.allocate(mesh_device, geometry, layer_ids=(0,), num_slots=1)
        model.kda_states.bind_slabs(slabs[capacity])
    addresses = {
        length: (m.kda_states.read(0).recurrent.buffer_address(), m.kda_states.read(0).convolution.buffer_address())
        for length, m in models.items()
    }

    def rms(value):
        return (
            value.float() * torch.rsqrt(value.float().square().mean(-1, keepdim=True) + config.rms_norm_eps)
        ).bfloat16()

    def reference(tokens, state):
        hidden = embedding[tokens].unsqueeze(0)
        attended, state = kda_forward_reference(rms(hidden), kda_weights, kda_config, state)
        hidden = (hidden + attended).bfloat16()
        normalized = rms(hidden).float()
        feedforward = F.linear(
            F.silu(F.linear(normalized, ff["gate_proj"].float())) * F.linear(normalized, ff["up_proj"].float()),
            ff["down_proj"].float(),
        )
        return (hidden.float() + feedforward).bfloat16(), state

    try:
        # Each model is reused for a second, distinct request with its previous carries still dirty.
        for request in range(2):
            tokens = torch.randint(0, 512, (256,), generator=generator)
            results = {}
            for capacity, model in models.items():
                reference_state = None
                outputs = []
                for start in range(0, 256, capacity):
                    chunk = tokens[start : start + capacity]
                    permutation = mla_row_permutation(start, 2, capacity // 2)
                    input_tt = ttnn.from_torch(
                        chunk[permutation].reshape(1, 1, capacity),
                        dtype=ttnn.uint32,
                        layout=ttnn.ROW_MAJOR_LAYOUT,
                        device=mesh_device,
                        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(2, 4), dims=(2, None)),
                    )
                    output = model.forward(input_tt, actual_start=start, actual_end=start + capacity)
                    expected, reference_state = reference(chunk, reference_state)
                    state = model.kda_states.read(0)
                    assert_matches_reference(
                        output_tt=output,
                        state=state,
                        permutation=permutation,
                        expected_output=expected,
                        expected_state=reference_state,
                        mesh_device=mesh_device,
                        sp_axis=0,
                        tp_axis=1,
                        config=kda_config,
                        state_linf_threshold=None,
                        label=f"LB transformer request={request} start={start} capacity={capacity}",
                    )
                    assert (state.recurrent.buffer_address(), state.convolution.buffer_address()) == addresses[capacity]
                    imported = allocate_native_state(mesh_device, geometry)
                    slabs[capacity].import_layer(imported, 0, 0)
                    for live, restored in (
                        (state.recurrent, imported.recurrent),
                        (state.convolution, imported.convolution),
                    ):
                        for a, b in zip(ttnn.get_device_tensors(live), ttnn.get_device_tensors(restored), strict=True):
                            assert_bit_identical(ttnn.to_torch(a), ttnn.to_torch(b), name="transformer slab")
                        ttnn.deallocate(restored)
                    physical = reconstruct_sp_tp_tensor(output, mesh_device, 0, 1, tp_dim=2, sp_dim=1)
                    outputs.append(physical[:, torch.argsort(permutation)].clone())
                    ttnn.deallocate(output)
                    ttnn.deallocate(input_tt)
                results[capacity] = torch.cat(outputs, dim=1)
            assert_accurate(
                results[256], results[128], name="LB transformer one-shot versus chunked", pcc_threshold=0.999
            )
    finally:
        for model in models.values():
            model.kda_states.deallocate()
            model.release_sub_device_managers()
        for slab in slabs.values():
            ttnn.deallocate(slab.recurrent)
            ttnn.deallocate(slab.convolution)
