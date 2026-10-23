# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# One Qwen3-VL text decoder layer -- plus its attention and MLP in isolation -- at the MiniMax-H3
# conditioner dimensions, sized so a Tracy capture measures the layer's arithmetic instead of a 32B
# whole-stack weight load. Note `head_dim` 128 != hidden_size // num_heads: q/k/v (8192) is wider
# than the residual stream (5120), which is a property of the checkpoint.

import time

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

# The layer never sees the embedding table; tiny keeps the fp32 reference small.
VOCAB_SIZE = 256
SEQ_LEN = 128  # == SEQ_BUCKET_SIZE, so the encoder's prompt bucketing does not pad

# Decoder length of a two_refs vision request (9,536 image tokens); tile-aligned (298 * 32).
TWO_REFS_SEQ_LEN = 9536

# Per-config `device_params`: `FABRIC_1D` on a 1x1 mesh fails router init, so `single` runs without fabric.
_L1_SMALL = 32768
_FABRIC = {"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": _L1_SMALL}
_NO_FABRIC = {"l1_small_size": _L1_SMALL}

# `sp_axis` shards the sequence and `fsdp_axis` the weights, both on the non-TP axis; None = off.
_MESH = [
    pytest.param((1, 1), (1, 1), None, None, None, 1, _NO_FABRIC, id="single"),
    pytest.param((4, 8), (4, 8), 1, None, None, 2, _FABRIC, id="tp8_axis1"),
    pytest.param((8, 4), (8, 4), 0, None, None, 2, _FABRIC, id="tp8_axis0"),
    pytest.param((4, 8), (4, 8), 1, 0, None, 2, _FABRIC, id="tp8_sp4_axis1"),
    pytest.param((8, 4), (8, 4), 0, 1, None, 2, _FABRIC, id="tp8_sp4_axis0"),
    pytest.param((8, 4), (8, 4), 0, None, 1, 2, _FABRIC, id="tp8_fsdp4_axis0"),
    pytest.param((8, 4), (8, 4), 0, 1, 1, 2, _FABRIC, id="tp8_sp4_fsdp4_axis0"),
]
_PARAMS = pytest.mark.parametrize(
    ("mesh_device", "submesh_shape", "tp_axis", "sp_axis", "fsdp_axis", "num_links", "device_params"),
    _MESH,
    indirect=["mesh_device", "device_params"],
)

# The two_refs case (9,536 tokens) needs headroom over the global timeout.
pytestmark = pytest.mark.timeout(1800)


# Iter 1 compiles/caches kernels; iter 2 is the measured steady-state pass.
_PERF_ITERS = 2


def _timed(submesh, tag, prep, op):
    """Run `op(prep())` `_PERF_ITERS` times with device-synced prep/op/e2e timing; returns the last result."""
    result = None
    n = _PERF_ITERS
    for i in range(n):
        ttnn.synchronize_device(submesh)
        t0 = time.time()
        inputs = prep()
        ttnn.synchronize_device(submesh)
        t1 = time.time()
        result = op(inputs)
        ttnn.synchronize_device(submesh)
        t2 = time.time()
        logger.info(
            f"{tag} iter {i + 1}/{n}: prep {(t1 - t0) * 1000:8.1f} ms (host build + H2D) | "
            f"op {(t2 - t1) * 1000:8.1f} ms | e2e {(t2 - t0) * 1000:8.1f} ms"
        )
    return result


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


@pytest.fixture(
    scope="module",
    params=[
        pytest.param(SEQ_LEN, id="short_128"),
        pytest.param(TWO_REFS_SEQ_LEN, id="two_refs_9536"),
    ],
)
def seq_len(request):
    """Sequence length under test: the short profiling length, and the two_refs decoder scale."""
    return request.param


@pytest.fixture(scope="module")
def golden(seq_len):
    """One fp32 reference forward, hooking the inputs and outputs of the layer and of its two halves.

    Kept in fp32: casting to bf16 would also degrade the rotary's `inv_freq` and poison the golden.
    """
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
    ids = torch.randint(0, VOCAB_SIZE, (1, seq_len))
    with torch.no_grad():
        lm(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
    for h in handles:
        h.remove()

    assert cap["mask"] is None, (
        f"reference passed an explicit attention_mask ({type(cap['mask'])}) to the decoder layer; "
        "the is_causal=True path on our side no longer provably matches it"
    )
    return {"state": layer.state_dict(), **cap}


def _ctx(mesh_device, submesh_shape, tp_axis, sp_axis, fsdp_axis, num_links):
    """`(submesh, Qwen3VlContext)`; a None `tp_axis` / `sp_axis` / `fsdp_axis` disables TP / SP / FSDP."""
    submesh = mesh_device.create_submesh(ttnn.MeshShape(*submesh_shape))
    ccl = CCLManager(submesh, num_links=num_links, topology=ttnn.Topology.Linear) if tp_axis is not None else None
    return submesh, Qwen3VlContext(
        device=submesh, tp_axis=tp_axis, ccl_manager=ccl, fsdp_mesh_axis=fsdp_axis, sp_axis=sp_axis
    )


def _sp_factor(submesh, sp_axis):
    return tuple(submesh.shape)[sp_axis] if sp_axis is not None else 1


def _sp_seq_pad(seq_len, sp):
    """Sequence length padded up to a multiple of `sp * 32`, so each SP shard is tile-aligned."""
    mult = sp * 32
    return -(-seq_len // mult) * mult


def _shard_seq(x, submesh, sp_axis, seq_dim, seq_pad):
    """Pad `x`'s sequence (at `seq_dim`) up to `seq_pad` and shard it across the SP axis."""
    trailing = x.ndim - 1 - seq_dim
    x = torch.nn.functional.pad(x, [0, 0] * trailing + [0, seq_pad - x.shape[seq_dim]])
    return bf16_tensor(x, device=submesh, mesh_axis=sp_axis, shard_dim=seq_dim)


def _gather_seq(out, sp_axis, seq_len):
    """Gather a sequence-sharded `[1, seq, hidden]` output across `sp_axis` and drop the SP padding."""
    got = tensor.to_torch(out, mesh_axes=[None, sp_axis, None])
    return got[:, :seq_len, :]


def _rope(submesh, seq_len, sp_axis=None, seq_pad=None):
    cos, sin = create_rope_tensors(1, seq_len, None, HEAD_DIM, ROPE_THETA, MROPE_SECTION)

    explicit = create_rope_tensors(
        1,
        seq_len,
        None,
        HEAD_DIM,
        ROPE_THETA,
        MROPE_SECTION,
        position_ids=torch.arange(seq_len).view(1, 1, -1).expand(3, 1, -1),
    )
    for a, b, which in zip((cos, sin), explicit, ("cos", "sin")):
        assert torch.equal(a, b), f"{which}: omitting position_ids no longer matches the shared token index"

    if sp_axis is not None:
        return (
            _shard_seq(cos, submesh, sp_axis, 2, seq_pad),
            _shard_seq(sin, submesh, sp_axis, 2, seq_pad),
        )
    return bf16_tensor(cos, device=submesh), bf16_tensor(sin, device=submesh)


@_PARAMS
def test_decoder_block_on_device(golden, seq_len, mesh_device, submesh_shape, tp_axis, sp_axis, fsdp_axis, num_links):
    """The whole pre-norm layer: RMSNorm + attention + RMSNorm + MLP, both residuals."""
    submesh, ctx = _ctx(mesh_device, submesh_shape, tp_axis, sp_axis, fsdp_axis, num_links)

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
    seq_pad = _sp_seq_pad(seq_len, _sp_factor(submesh, sp_axis))

    def prep():
        x = (
            _shard_seq(golden["layer_in"], submesh, sp_axis, 1, seq_pad)
            if sp_axis is not None
            else bf16_tensor(golden["layer_in"], device=submesh)
        )
        return x, _rope(submesh, seq_len, sp_axis=sp_axis, seq_pad=seq_pad)

    out = _timed(
        submesh,
        f"decoder layer tp_axis={tp_axis} sp_axis={sp_axis} seq={seq_len}",
        prep,
        lambda inp: block.forward(inp[0], attention_bias=None, pos_embeds=inp[1]),
    )

    tp_factor = tuple(submesh.shape)[tp_axis] if tp_axis is not None else 1
    logger.info(f"qwen3vl decoder layer TP={tp_factor} (axis {tp_axis}) SP axis={sp_axis}:")
    got = (
        _gather_seq(out, sp_axis, seq_len)
        if sp_axis is not None
        else tensor.to_torch(out, mesh_axes=[None, None, None])
    )
    assert_quality(golden["layer_out"].float(), got, pcc=0.99)


@_PARAMS
def test_decoder_attention_on_device(
    golden, seq_len, mesh_device, submesh_shape, tp_axis, sp_axis, fsdp_axis, num_links
):
    """Attention alone: fused qkv, per-head QK-RMSNorm, RoPE, SDPA, o_proj (no residual or input norm)."""
    submesh, ctx = _ctx(mesh_device, submesh_shape, tp_axis, sp_axis, fsdp_axis, num_links)

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
    seq_pad = _sp_seq_pad(seq_len, _sp_factor(submesh, sp_axis))

    def prep():
        x = (
            _shard_seq(golden["attn_in"], submesh, sp_axis, 1, seq_pad)
            if sp_axis is not None
            else bf16_tensor(golden["attn_in"], device=submesh)
        )
        return x, _rope(submesh, seq_len, sp_axis=sp_axis, seq_pad=seq_pad)

    out = _timed(
        submesh,
        f"decoder attn tp_axis={tp_axis} sp_axis={sp_axis} seq={seq_len}",
        prep,
        lambda inp: attn.forward(inp[0], attention_bias=None, pos_embeds=inp[1]),
    )
    got = (
        _gather_seq(out, sp_axis, seq_len)
        if sp_axis is not None
        else tensor.to_torch(out, mesh_axes=[None, None, None])
    )
    assert_quality(golden["attn_out"].float(), got, pcc=0.99)


@_PARAMS
def test_decoder_mlp_on_device(golden, seq_len, mesh_device, submesh_shape, tp_axis, sp_axis, fsdp_axis, num_links):
    """MLP alone: SwiGLU over `intermediate_size` 25600, three of the layer's four matmuls."""
    submesh, ctx = _ctx(mesh_device, submesh_shape, tp_axis, sp_axis, fsdp_axis, num_links)

    mlp = Qwen3VlMlp(hidden_size=HIDDEN_SIZE, intermediate_size=INTERMEDIATE_SIZE, hidden_act=HIDDEN_ACT, ctx=ctx)
    mlp.load_torch_state_dict({k[len("mlp.") :]: v for k, v in golden["state"].items() if k.startswith("mlp.")})
    seq_pad = _sp_seq_pad(seq_len, _sp_factor(submesh, sp_axis))

    def prep():
        return (
            _shard_seq(golden["mlp_in"], submesh, sp_axis, 1, seq_pad)
            if sp_axis is not None
            else bf16_tensor(golden["mlp_in"], device=submesh)
        )

    out = _timed(
        submesh,
        f"decoder mlp tp_axis={tp_axis} sp_axis={sp_axis} seq={seq_len}",
        prep,
        lambda inp: mlp.forward(inp),
    )
    got = (
        _gather_seq(out, sp_axis, seq_len)
        if sp_axis is not None
        else tensor.to_torch(out, mesh_axes=[None, None, None])
    )
    assert_quality(golden["mlp_out"].float(), got, pcc=0.99)
