# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Per-layer PCC against the HuggingFace reference, gated on the weight-quantization floor.

The precision policy stores every projection as bfloat4_b, which costs far more accuracy than
any absolute PCC threshold worth writing down: a full-attention layer lands near 0.98 against
fp32 when it is working correctly. So each layer is compared against a reference whose weights
have been round-tripped through the same dtype, and the gate is that the device tracks that
floor rather than any fixed number.
"""

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.qwen38_27b_t3k.tt.generator import configure_fabric
from models.demos.qwen38_27b_t3k.tt.model import Qwen38Model

# One layer of each kind: Qwen3.8-27B repeats a full_attention layer every fourth position.
LAYERS = [0, 3]
LENGTH = 128
# The device runs bfloat16 activations at LoFi as well, which the weight-only floor does not
# model, so it is allowed to sit slightly below it.
FLOOR_MARGIN = 0.01


def _quantized(state_dict):
    """The layer's weights as the device stores them, read back into torch."""
    out = {}
    for name, value in state_dict.items():
        tileable = value.dim() == 2 and not (value.shape[-1] % 32 or value.shape[-2] % 32)
        if not tileable:
            out[name] = value.float()
            continue
        dtype = ttnn.bfloat4_b if value.dim() == 2 else ttnn.bfloat16
        out[name] = (
            ttnn.to_torch(ttnn.from_torch(value.float(), dtype=dtype, layout=ttnn.TILE_LAYOUT))
            .reshape(value.shape)
            .float()
        )
    return out


def _pcc(actual, expected):
    return comp_pcc(expected.float().reshape(-1), actual.float().reshape(-1), 0.0)[1]


@pytest.fixture(scope="module")
def harness():
    from models.demos.qwen38_27b_t3k.tt.decoder_tp import native_mesh_shape

    configure_fabric()
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(*native_mesh_shape()), trace_region_size=200000000)
    try:
        yield mesh, Qwen38Model(mesh, layer_indices=LAYERS)
    finally:
        ttnn.close_mesh_device(mesh)


@pytest.mark.parametrize("layer_index", LAYERS)
def test_prefill_layer_tracks_the_quantization_floor(harness, layer_index):
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer, Qwen3_5TextRotaryEmbedding

    mesh, model = harness
    position = next(i for i, candidate in enumerate(model.layers) if candidate.layer_idx == layer_index)
    layer = model.layers[position]
    # Cache states are allocated per layer in model.layers order, and a full_attention layer's
    # state carries key/value where a linear_attention one carries conv/recurrent.
    state = model.allocate_cache(batch_size=1, capacity=LENGTH).layers[position]

    tokens = torch.arange(1000, 1000 + LENGTH, dtype=torch.int32).reshape(1, LENGTH)
    ids = model.upload(tokens, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
    positions = model.upload(
        torch.arange(LENGTH, dtype=torch.int32).reshape(1, LENGTH), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
    )
    page_table = model.upload(
        torch.arange(4, dtype=torch.int32).reshape(1, 4), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT
    )
    x = model.embed(ids, batch=1, length=LENGTH)
    # The residual is replicated, so every device holds the same shard of it.
    x_host = ttnn.to_torch(x, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=-1)).chunk(model.TP, dim=-1)[0].float()
    cos, sin = model.rope(positions, batch=1, length=LENGTH)

    rope = Qwen3_5TextRotaryEmbedding(model.config)
    cos_host, sin_host = rope(torch.zeros(1, LENGTH, 1), torch.arange(LENGTH)[None])
    reference_kwargs = dict(position_embeddings=(cos_host, sin_host), position_ids=torch.arange(LENGTH)[None])
    if layer.kind == "full_attention":
        reference_kwargs["attention_mask"] = torch.full((1, 1, LENGTH, LENGTH), float("-inf")).triu(1)

    def reference(weights):
        module = Qwen3_5DecoderLayer(model.config, layer_index).float().eval()
        module.load_state_dict(weights)
        with torch.no_grad():
            result = module(x_host.clone(), **reference_kwargs)
        return result[0] if isinstance(result, tuple) else result

    weights = model.checkpoint.layer(layer_index)
    golden = reference({name: value.float() for name, value in weights.items()})
    floor = _pcc(reference(_quantized(weights)), golden)

    output = layer.prefill_forward(
        x,
        state=state,
        start_pos=0,
        page_table=page_table,
        cos=cos,
        sin=sin,
        positions=ttnn.reshape(ttnn.typecast(positions, ttnn.int32), [LENGTH, 1]),
    )
    device = ttnn.to_torch(output, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=-1))
    measured = _pcc(device.chunk(model.TP, dim=-1)[0].float(), golden)

    assert measured >= floor - FLOOR_MARGIN, (
        f"layer {layer_index} ({layer.kind}) PCC {measured:.6f} is below the bfloat4_b weight "
        f"floor {floor:.6f} by more than {FLOOR_MARGIN}"
    )
