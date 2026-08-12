# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# =============================================================================
# Qwen3-VL text decoder: one layer, plus its attention and MLP in isolation.
#
# The whole-stack tests (`test_qwen3vl.py`, `tests/models/minimax_h3/
# test_text_encoder_minimax_h3.py`) are the parity tests; neither is usable for
# profiling. The MiniMax-H3 one loads ~32B of weights onto the mesh, which
# dominates any Tracy capture by orders of magnitude over the arithmetic. This
# file runs a SINGLE layer at the production dimensions, so a capture contains
# one iteration of the thing being measured and ~1 GiB of weights instead of 62.
#
# Dimensions are MiniMax-H3's Qwen3-VL conditioner (also Qwen3-VL-32B): note
# `head_dim` 128 != hidden_size // num_heads (5120 // 64 == 80), so the q/k/v
# inner dimension (8192) is WIDER than the residual stream. That is a property of
# the checkpoint, not a typo -- see the note in `Qwen3VlTextEncoder.__init__`.
#
# `vocab_size` is deliberately tiny. The decoder layer never sees the embedding
# table, so shrinking it takes the fp32 reference from ~7 GiB to ~2 GiB and
# changes nothing that is measured.
#
# Under TP the layer takes and returns FULL-WIDTH REPLICATED tensors: `qkv_proj`
# is column-parallel (so its input must be replicated), and both `o_proj` and
# `down_proj` are followed by an all-gather. So no sharding is needed on either
# side of the comparison.
#
# The four matmuls per layer, at TP=8 -- these are exactly the shapes that warn
# as unknown in `utils/matmul.py::grid_11_10_configs`:
#     qkv_proj   (128, 5120, 1280)   5120 -> (64 + 2*8)*128 = 10240, /8
#     o_proj     (128, 8192, 640)    64*128 = 8192 -> 5120, /8
#     gate/up    (128, 5120, 3200)   5120 -> 25600, /8
#     down_proj  (128, 3200, 5120)   row-parallel: K sharded, N full
# =============================================================================

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

# The layer is independent of the vocabulary; see the header.
VOCAB_SIZE = 256
SEQ_LEN = 128  # == SEQ_BUCKET_SIZE, so the encoder's prompt bucketing does not pad

# `single` is the profiling target and runs anywhere. The TP=8 configs mirror
# `test_text_encoder_minimax_h3.py` and need a 32-chip mesh; they are what
# reproduce the sharded matmul shapes listed in the header. Per-config
# `device_params` (fabric only where there is CCL) is explained on the dicts
# in common.py.
_MESH = [
    pytest.param((1, 1), (1, 1), None, 1, NO_FABRIC, id="single"),
    pytest.param((4, 8), (4, 8), 1, 2, FABRIC, id="tp8_axis1"),
    pytest.param((8, 4), (8, 4), 0, 2, FABRIC, id="tp8_axis0"),
]
# The layer test runs the full matrix; the attention/MLP isolation tests keep debugging granularity
# but only one representative mesh config. `tp8_axis1` is the (4, 8) mesh the MiniMax-H3 pipeline
# actually deploys on (every `test_pipeline_*_minimax_h3.py` uses a (4, 8) mesh), and it reproduces
# the sharded matmul shapes listed in the header; `single` and `tp8_axis0` are still covered at the
# layer level above.
_MESH_REPRESENTATIVE = [_MESH[1]]

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
    """One reference forward, capturing the inputs and outputs of the layer and of its two halves.

    Hooks rather than direct calls: `Qwen3VLTextDecoderLayer.forward` takes `position_embeddings` and
    a pre-built mask, and letting the reference model construct both is what keeps this test honest
    about HF's own conventions instead of a reimplementation of them.

    The reference is kept in fp32. Casting the model to bf16 would also cast the rotary's `inv_freq`
    buffer, degrading `cos`/`sin` to bf16 precision and putting a ~1e-1 error into the golden's rotary
    -- far larger than anything being measured. Our side is bf16 on device regardless.
    """
    torch.manual_seed(0)
    cfg = _config()
    assert sum(MROPE_SECTION) == HEAD_DIM // 2, f"mrope_section sums to {sum(MROPE_SECTION)}, want {HEAD_DIM // 2}"

    lm = transformers.models.qwen3_vl.modeling_qwen3_vl.Qwen3VLTextModel._from_config(cfg).eval()
    layer = lm.layers[0]

    cap: dict[str, torch.Tensor | None] = {}

    def grab(key):
        def hook(_m, args, kwargs, out):
            # The reference calls the layer positionally but `self_attn` by keyword, so the hidden
            # states arrive in `args` for some of these hooks and in `kwargs` for others.
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

    # The reference reaches the layer with NO explicit mask, so its attention takes the internal
    # causal path -- which is what our `attention_bias=None -> is_causal=True` matches. If a future
    # transformers materializes a mask here, the two sides may no longer agree on causality, and that
    # must fail loudly rather than silently compare a causal golden against a non-causal actual.
    assert cap["mask"] is None, (
        f"reference passed an explicit attention_mask ({type(cap['mask'])}) to the decoder layer; "
        "the is_causal=True path on our side no longer provably matches it"
    )
    return {"state": layer.state_dict(), **cap}


def _ctx(mesh_device, submesh_shape, tp_axis, num_links):
    """`(submesh, Qwen3VlContext)`. `tp_axis=None` is the replicated single-device case."""
    submesh = mesh_device.create_submesh(ttnn.MeshShape(*submesh_shape))
    ccl = CCLManager(submesh, num_links=num_links, topology=ttnn.Topology.Linear) if tp_axis is not None else None
    # `fsdp_mesh_axis=None`: matching `test_text_encoder_minimax_h3.py`, which leaves `is_fsdp` at its
    # default because a Blackhole chip holds a TP=8 shard without it. FSDP is a weight-placement knob
    # and does not change these matmul shapes.
    return submesh, Qwen3VlContext(device=submesh, tp_axis=tp_axis, ccl_manager=ccl, fsdp_mesh_axis=None)


def _rope(submesh):
    # Production (`test_text_encoder_minimax_h3.py`) leaves `interleaved` at False even though this
    # checkpoint declares `mrope_interleaved`: the two layouts coincide exactly while all three MRoPE
    # axes carry the same position, which is the text-only case. Verified equal here to 7.6e-6, i.e.
    # the documented `theta ** -x` vs `1 / theta ** x` ulp difference and nothing else.
    #
    # Omitting `position_ids` must stay equivalent to passing the token index on all three axes -- the
    # call Ideogram 4.0 makes through the shared encoder. That equivalence is pinned by the host-only
    # `test_qwen3vl_mrope.py::test_position_ids_default_is_the_shared_token_index`, so it is not
    # re-asserted here.
    cos, sin = create_rope_tensors(1, SEQ_LEN, None, HEAD_DIM, ROPE_THETA, MROPE_SECTION)
    return bf16_tensor(cos, device=submesh), bf16_tensor(sin, device=submesh)


@_PARAMS
def test_decoder_block_on_device(golden, mesh_device, submesh_shape, tp_axis, num_links):
    """The whole pre-norm layer: RMSNorm + attention + RMSNorm + MLP, both residuals.

    This is the profiling entry point -- one iteration of the layer that the 64-layer conditioner
    repeats, and the four matmuls listed in the header.
    """
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
        attention_bias=None,  # internal causal path; see the assertion in the `golden` fixture
        pos_embeds=_rope(submesh),
    )

    tp_factor = tuple(submesh.shape)[tp_axis] if tp_axis is not None else 1
    logger.info(f"qwen3vl decoder layer TP={tp_factor} (axis {tp_axis}):")
    assert_quality(golden["layer_out"].float(), tensor.to_torch(out, mesh_axes=[None, None, None]), pcc=0.99)


@_PARAMS_REPRESENTATIVE
def test_decoder_attention_on_device(golden, mesh_device, submesh_shape, tp_axis, num_links):
    """Attention alone: fused qkv, per-head QK-RMSNorm, RoPE, SDPA, o_proj. Excludes the residual and
    the input norm, so it attributes the `qkv_proj` / `o_proj` half of the layer's time."""
    submesh, ctx = _ctx(mesh_device, submesh_shape, tp_axis, num_links)

    attn = Qwen3VlAttention(
        hidden_size=HIDDEN_SIZE,
        num_heads=NUM_HEADS,
        num_key_value_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        rms_norm_eps=NORM_EPS,
        ctx=ctx,
    )
    # `_prepare_torch_state` fuses q/k/v into `qkv_proj` (and reorders `o_proj`) as part of the load.
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
    """MLP alone: SwiGLU over `intermediate_size` 25600. Three of the layer's four matmuls by FLOPs,
    so this is where the layer's time is expected to sit."""
    submesh, ctx = _ctx(mesh_device, submesh_shape, tp_axis, num_links)

    mlp = Qwen3VlMlp(hidden_size=HIDDEN_SIZE, intermediate_size=INTERMEDIATE_SIZE, hidden_act=HIDDEN_ACT, ctx=ctx)
    mlp.load_torch_state_dict({k[len("mlp.") :]: v for k, v in golden["state"].items() if k.startswith("mlp.")})

    out = mlp.forward(bf16_tensor(golden["mlp_in"], device=submesh))
    assert_quality(golden["mlp_out"].float(), tensor.to_torch(out, mesh_axes=[None, None, None]), pcc=0.99)
