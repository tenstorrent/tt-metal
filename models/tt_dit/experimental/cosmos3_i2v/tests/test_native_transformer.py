# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""PCC test for the native Cosmos3OmniTransformer trunk.

Validates the per-layer wiring + the two final RMSNorms by stacking N
reference decoder layers + diffusers RMSNorm against the native
`Cosmos3OmniTransformer` with the same random weights. PCC bound is
loose (0.96) because errors compound across `num_hidden_layers` stages —
that's exactly what we're stress-testing.

Parametrized over WH LoudBox shapes (1,1)/(1,2)/(1,4)/(1,8) and
`num_hidden_layers` in {1, 2, 4} so we see how PCC scales with depth.
At 4 layers we can already extrapolate: if PCC drops fast across depth
here, the full 64-layer trunk will produce garbage and we need to
tighten per-layer precision (HiFi4, fp32_dest_acc_en, etc.) before the
Galaxy demo will work.
"""

from __future__ import annotations

import pytest
import torch

import ttnn
from models.tt_dit.parallel.config import DiTParallelConfig, ParallelFactor
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.utils.check import assert_quality
from models.tt_dit.utils.tensor import bf16_tensor
from models.tt_dit.utils.test import line_params


@pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        pytest.param((1, 1), line_params, id="1x1"),
        pytest.param((1, 2), line_params, id="1x2"),
        pytest.param((1, 4), line_params, id="1x4"),
        pytest.param((1, 8), line_params, id="1x8"),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("num_hidden_layers", [1, 2, 4], ids=["L1", "L2", "L4"])
@pytest.mark.timeout(600)
def test_native_transformer(mesh_device: ttnn.MeshDevice, num_hidden_layers: int) -> None:
    from diffusers.models.normalization import RMSNorm as RefRMSNorm

    from models.tt_dit.experimental.cosmos3_i2v.model.transformer import Cosmos3OmniTransformer as TTTransformer
    from models.tt_dit.experimental.cosmos3_i2v.reference.transformer_cosmos3 import (
        Cosmos3VLTextMoTDecoderLayer as RefDecoderLayer,
    )
    from models.tt_dit.experimental.cosmos3_i2v.reference.transformer_cosmos3 import Cosmos3VLTextRotaryEmbedding

    torch.manual_seed(42)

    mesh_shape = tuple(mesh_device.shape)
    tp_factor = mesh_shape[1]
    sp_factor = mesh_shape[0]

    hidden_size = 256
    head_dim = 64
    num_attention_heads = 32
    num_key_value_heads = 8
    intermediate_size = 512
    rms_norm_eps = 1e-6
    rope_theta = 5_000_000.0
    rope_axes_dim = (12, 10, 10)
    N_und = 128
    N_gen = 128
    N_total = N_und + N_gen

    # Build the reference: N decoder layers + 2 final RMSNorms.
    ref_layers = [
        RefDecoderLayer(
            hidden_size=hidden_size,
            head_dim=head_dim,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            intermediate_size=intermediate_size,
            attention_bias=False,
            rms_norm_eps=rms_norm_eps,
        )
        for _ in range(num_hidden_layers)
    ]
    ref_norm = RefRMSNorm(hidden_size, eps=rms_norm_eps, elementwise_affine=True, bias=False)
    ref_norm_moe_gen = RefRMSNorm(hidden_size, eps=rms_norm_eps, elementwise_affine=True, bias=False)
    for m in [*ref_layers, ref_norm, ref_norm_moe_gen]:
        m.eval()
        m.to(dtype=torch.bfloat16)

    # Build the native trunk on the device mesh.
    parallel_config = DiTParallelConfig(
        cfg_parallel=ParallelFactor(1, 0),
        sequence_parallel=ParallelFactor(sp_factor, 0),
        tensor_parallel=ParallelFactor(tp_factor, 1),
    )
    ccl_manager = (
        CCLManager(mesh_device=mesh_device, num_links=1, topology=ttnn.Topology.Linear) if tp_factor > 1 else None
    )
    tt_trunk = TTTransformer(
        hidden_size=hidden_size,
        head_dim=head_dim,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_hidden_layers,
        attention_bias=False,
        rms_norm_eps=rms_norm_eps,
        mesh_device=mesh_device,
        parallel_config=parallel_config,
        ccl_manager=ccl_manager,
    )

    # Compose the state dict the native trunk expects: layers.{i}.<...>, norm.<...>, norm_moe_gen.<...>
    state: dict[str, torch.Tensor] = {}
    for i, layer in enumerate(ref_layers):
        for k, v in layer.state_dict().items():
            state[f"layers.{i}.{k}"] = v
    for k, v in ref_norm.state_dict().items():
        state[f"norm.{k}"] = v
    for k, v in ref_norm_moe_gen.state_dict().items():
        state[f"norm_moe_gen.{k}"] = v
    tt_trunk.load_torch_state_dict(state)

    # cos/sin for the joint sequence, sliced into und + gen halves.
    rope = Cosmos3VLTextRotaryEmbedding(
        head_dim=head_dim,
        rope_theta=rope_theta,
        rope_axes_dim=list(rope_axes_dim),
    )
    position_ids = torch.arange(N_total).unsqueeze(0)
    cos_all, sin_all = rope(position_ids, device=torch.device("cpu"), dtype=torch.bfloat16)
    cos_all = cos_all.squeeze(0)
    sin_all = sin_all.squeeze(0)
    cos_und_NE = cos_all[:N_und]
    sin_und_NE = sin_all[:N_und]
    cos_gen_NE = cos_all[N_und:]
    sin_gen_NE = sin_all[N_und:]

    und_seq = torch.randn(N_und, hidden_size, dtype=torch.bfloat16)
    gen_seq = torch.randn(N_gen, hidden_size, dtype=torch.bfloat16)

    with torch.no_grad():
        ref_und, ref_gen = und_seq, gen_seq
        for layer in ref_layers:
            ref_und, ref_gen = layer(ref_und, ref_gen, (cos_und_NE, sin_und_NE, cos_gen_NE, sin_gen_NE))
        ref_und = ref_norm(ref_und)
        ref_gen = ref_norm_moe_gen(ref_gen)

    und_seq_tt = bf16_tensor(und_seq.reshape(1, 1, N_und, hidden_size), device=mesh_device)
    gen_seq_tt = bf16_tensor(gen_seq.reshape(1, 1, N_gen, hidden_size), device=mesh_device)
    cos_und_tt = bf16_tensor(cos_und_NE.reshape(1, 1, N_und, head_dim), device=mesh_device)
    sin_und_tt = bf16_tensor(sin_und_NE.reshape(1, 1, N_und, head_dim), device=mesh_device)
    cos_gen_tt = bf16_tensor(cos_gen_NE.reshape(1, 1, N_gen, head_dim), device=mesh_device)
    sin_gen_tt = bf16_tensor(sin_gen_NE.reshape(1, 1, N_gen, head_dim), device=mesh_device)

    tt_und, tt_gen = tt_trunk(und_seq_tt, gen_seq_tt, cos_und_tt, sin_und_tt, cos_gen_tt, sin_gen_tt)

    tt_und_view = ttnn.to_torch(ttnn.get_device_tensors(tt_und)[0]).reshape(N_und, hidden_size)
    tt_gen_view = ttnn.to_torch(ttnn.get_device_tensors(tt_gen)[0]).reshape(N_gen, hidden_size)

    # Loose PCC because errors compound across num_hidden_layers stages.
    assert_quality(ref_und, tt_und_view, pcc=0.96)
    assert_quality(ref_gen, tt_gen_view, pcc=0.96)
