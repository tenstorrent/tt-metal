# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# One Qwen3-VL text decoder layer -- plus its attention and MLP in isolation -- at
# the MiniMax-H3 conditioner dimensions, sized so a Tracy capture measures the
# layer's arithmetic instead of a 32B whole-stack weight load.

import pytest
import torch
import transformers
from loguru import logger

import ttnn

from ....encoders.qwen3vl.model_qwen3vl import (
    Qwen3VlAttention,
    Qwen3VlContext,
    Qwen3VlDecoderLayer,
    Qwen3VlMlp,
    create_rope_tensors,
)
from ....parallel.manager import CCLManager
from ....utils import tensor
from ....utils.check import assert_quality
from ....utils.tensor import bf16_tensor
from .common import FABRIC, NO_FABRIC

# MiniMax-H3's `text_encoder/config.json` (text_config).
HIDDEN_SIZE = 5120
INTERMEDIATE_SIZE = 25600
NUM_HEADS = 64
NUM_KV_HEADS = 8
HEAD_DIM = 128  # explicit in the checkpoint; 5120 // 64 would give 80
ROPE_THETA = 5_000_000.0
MROPE_SECTION = [24, 20, 20]  # sums to HEAD_DIM // 2
NORM_EPS = 1e-6
HIDDEN_ACT = "silu"

VOCAB_SIZE = 256  # the layer never sees the embedding table; tiny keeps the fp32 reference small
SEQ_LEN = 128  # == SEQ_BUCKET_SIZE, so the encoder's prompt bucketing does not pad

_MESH = [
    pytest.param((1, 1), (1, 1), None, 1, NO_FABRIC, id="single"),
    pytest.param((4, 8), (4, 8), 1, 2, FABRIC, id="tp8_axis1"),
    pytest.param((8, 4), (8, 4), 0, 2, FABRIC, id="tp8_axis0"),
]
_MESH_REPRESENTATIVE = [_MESH[1]]  # tp8_axis1: the (4, 8) mesh the MiniMax-H3 pipeline actually deploys on

_PARAMS = pytest.mark.parametrize(
    ("mesh_device", "submesh_shape", "tp_axis", "num_links", "device_params"),
    _MESH,
    indirect=["mesh_device", "device_params"],
)
_PARAMS_REPRESENTATIVE = pytest.mark.parametrize(
    ("mesh_device", "submesh_shape", "tp_axis", "num_links", "device_params"),
    _MESH_REPRESENTATIVE,
    indirect=["mesh_device", "device_params"],
)


def _config():
    return transformers.Qwen3VLTextConfig(
        vocab_size=VOCAB_SIZE,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
        num_hidden_layers=1,
        num_attention_heads=NUM_HEADS,
        num_key_value_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        rms_norm_eps=NORM_EPS,
        hidden_act=HIDDEN_ACT,
        rope_theta=ROPE_THETA,
        rope_scaling={"rope_type": "default", "mrope_section": MROPE_SECTION, "mrope_interleaved": True},
    )


@pytest.fixture(scope="module")
def golden():
    # fp32 reference: casting it to bf16 would degrade the rotary's inv_freq and poison the golden
    torch.manual_seed(0)
    cfg = _config()
    assert sum(MROPE_SECTION) == HEAD_DIM // 2, f"mrope_section sums to {sum(MROPE_SECTION)}, want {HEAD_DIM // 2}"

    lm = transformers.models.qwen3_vl.modeling_qwen3_vl.Qwen3VLTextModel._from_config(cfg).eval()
    layer = lm.layers[0]

    cap: dict[str, torch.Tensor | None] = {}

    def grab(key):
        def hook(_m, args, kwargs, out):
            inp = args[0] if args else kwargs.get("hidden_states")
            assert inp is not None, f"{key}: could not find hidden states in args/kwargs"
            cap[f"{key}_in"] = inp.detach()
            cap[f"{key}_out"] = (out[0] if isinstance(out, tuple) else out).detach()
            if key == "layer":
                cap["mask"] = kwargs.get("attention_mask")
                cap["pos"] = kwargs.get("position_embeddings")

        return hook

    handles = [
        layer.register_forward_hook(grab("layer"), with_kwargs=True),
        layer.self_attn.register_forward_hook(grab("attn"), with_kwargs=True),
        layer.mlp.register_forward_hook(grab("mlp"), with_kwargs=True),
    ]
    ids = torch.randint(0, VOCAB_SIZE, (1, SEQ_LEN))
    with torch.no_grad():
        lm(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
    for h in handles:
        h.remove()

    # a future transformers materializing a mask here would silently break causal parity; fail loudly
    assert cap["mask"] is None, (
        f"reference passed an explicit attention_mask ({type(cap['mask'])}) to the decoder layer; "
        "the is_causal=True path on our side no longer provably matches it"
    )
    return {"state": layer.state_dict(), **cap}


def _ctx(mesh_device, submesh_shape, tp_axis, num_links):
    submesh = mesh_device.create_submesh(ttnn.MeshShape(*submesh_shape))
    ccl = CCLManager(submesh, num_links=num_links, topology=ttnn.Topology.Linear) if tp_axis is not None else None
    return submesh, Qwen3VlContext(device=submesh, tp_axis=tp_axis, ccl_manager=ccl, fsdp_mesh_axis=None)


def _rope(submesh):
    # interleaved stays False (as in production): the layouts coincide while all three MRoPE axes share the position
    cos, sin = create_rope_tensors(1, SEQ_LEN, None, HEAD_DIM, ROPE_THETA, MROPE_SECTION)
    return bf16_tensor(cos, device=submesh), bf16_tensor(sin, device=submesh)


@_PARAMS
def test_decoder_block_on_device(golden, mesh_device, submesh_shape, tp_axis, num_links):
    submesh, ctx = _ctx(mesh_device, submesh_shape, tp_axis, num_links)

    block = Qwen3VlDecoderLayer(
        hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
        hidden_act=HIDDEN_ACT,
        num_attention_heads=NUM_HEADS,
        num_key_value_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        rms_norm_eps=NORM_EPS,
        ctx=ctx,
    )
    block.load_torch_state_dict(golden["state"])

    out = block.forward(
        bf16_tensor(golden["layer_in"], device=submesh),
        attention_bias=None,
        pos_embeds=_rope(submesh),
    )

    tp_factor = tuple(submesh.shape)[tp_axis] if tp_axis is not None else 1
    logger.info(f"qwen3vl decoder layer TP={tp_factor} (axis {tp_axis}):")
    assert_quality(golden["layer_out"].float(), tensor.to_torch(out, mesh_axes=[None, None, None]), pcc=0.99)


@_PARAMS_REPRESENTATIVE
def test_decoder_attention_on_device(golden, mesh_device, submesh_shape, tp_axis, num_links):
    submesh, ctx = _ctx(mesh_device, submesh_shape, tp_axis, num_links)

    attn = Qwen3VlAttention(
        hidden_size=HIDDEN_SIZE,
        num_heads=NUM_HEADS,
        num_key_value_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        rms_norm_eps=NORM_EPS,
        ctx=ctx,
    )
    prefix = "self_attn."
    attn.load_torch_state_dict({k[len(prefix) :]: v for k, v in golden["state"].items() if k.startswith(prefix)})

    out = attn.forward(
        bf16_tensor(golden["attn_in"], device=submesh),
        attention_bias=None,
        pos_embeds=_rope(submesh),
    )
    assert_quality(golden["attn_out"].float(), tensor.to_torch(out, mesh_axes=[None, None, None]), pcc=0.99)


@_PARAMS_REPRESENTATIVE
def test_decoder_mlp_on_device(golden, mesh_device, submesh_shape, tp_axis, num_links):
    submesh, ctx = _ctx(mesh_device, submesh_shape, tp_axis, num_links)

    mlp = Qwen3VlMlp(hidden_size=HIDDEN_SIZE, intermediate_size=INTERMEDIATE_SIZE, hidden_act=HIDDEN_ACT, ctx=ctx)
    mlp.load_torch_state_dict({k[len("mlp.") :]: v for k, v in golden["state"].items() if k.startswith("mlp.")})

    out = mlp.forward(bf16_tensor(golden["mlp_in"], device=submesh))
    assert_quality(golden["mlp_out"].float(), tensor.to_torch(out, mesh_axes=[None, None, None]), pcc=0.99)
