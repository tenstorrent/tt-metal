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

# The layer is independent of the vocabulary; see the header.
VOCAB_SIZE = 256
SEQ_LEN = 128  # == SEQ_BUCKET_SIZE, so the encoder's prompt bucketing does not pad

# The decoder sequence length a two_refs vision request produces: 38,144 patches (a 128x128 plus a
# 128x170 reference image) merged 2x2 -> 9,536 image tokens (4,096 + 5,440), which become the
# <|image_pad|> rows the decoder runs over. We emulate that scale here -- as if the tower produced
# these rows under tp8_sp4 + windowed SDPA -- and let the block run its normal causal TP=8 path (the
# block is length-agnostic to which rows are image). Tile-aligned: 9536 == 298 * 32.
TWO_REFS_SEQ_LEN = 9536

# `single` is the profiling target and runs anywhere. The TP=8 configs mirror
# `test_text_encoder_minimax_h3.py` and need a 32-chip mesh; they are what
# reproduce the sharded matmul shapes listed in the header.
#
# Each config carries its own `device_params`, because fabric is not universally
# safe to request: `FABRIC_1D` on a 1x1 mesh has no remote ethernet partner, so
# router init fails the handshake and times out ("Fabric Router Sync: Timeout").
# The TP configs need it for the CCL all-gathers; `single` has no CCL at all.
_L1_SMALL = 32768
_FABRIC = {"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": _L1_SMALL}
_NO_FABRIC = {"l1_small_size": _L1_SMALL}

# `sp_axis` is the sequence-parallel axis (the non-TP axis of the 32-chip mesh); None = TP-only.
# `fsdp_axis` shards the WEIGHTS on that axis (all-gathered to full immediately before use). SP and
# FSDP compose on the same axis -- one shards activation rows, the other weights, and the two never
# interact (see Qwen3VlContext) -- so `tp8_sp4_fsdp4` is the full production-shaped combination:
# TP=8 on one axis, sequence AND weights sharded over the other.
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

# The two_refs case (9,536 tokens) is far heavier than the 128 profiling case -- the fp32 reference's
# quadratic attention and a 27x-longer device run -- so it needs headroom over the global budget. The
# 128 cases finish in seconds regardless.
pytestmark = pytest.mark.timeout(1800)


# Warmup+measure iterations: iter 1 compiles/caches kernels, iter 2 is the measured steady-state pass
# (read iter 2's numbers). Matches the vision-tower test's perf loop.
_PERF_ITERS = 2


def _timed(submesh, tag, prep, op):
    """Run `op(prep())` under device-synced prep/op/e2e wall-clock timing, `_PERF_ITERS` times.

    `prep()` builds and uploads the inputs (host build + H2D transfer); `op(inputs)` runs the device
    op. A sync brackets each half so the numbers are honest wall clock. Returns the last op result (the
    steady-state one), so the PCC assertion after still runs on a real output.
    """
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
    ids = torch.randint(0, VOCAB_SIZE, (1, seq_len))
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


def _ctx(mesh_device, submesh_shape, tp_axis, sp_axis, fsdp_axis, num_links):
    """`(submesh, Qwen3VlContext)`. `tp_axis=None` is the replicated single-device case; `sp_axis`
    (non-None) shards the sequence on that axis and routes attention through the causal ring;
    `fsdp_axis` (non-None) shards the weights on that axis, gathered to full at use. FSDP is a
    weight-placement knob and does not change the matmul shapes; it may share the axis with SP."""
    submesh = mesh_device.create_submesh(ttnn.MeshShape(*submesh_shape))
    ccl = CCLManager(submesh, num_links=num_links, topology=ttnn.Topology.Linear) if tp_axis is not None else None
    return submesh, Qwen3VlContext(
        device=submesh, tp_axis=tp_axis, ccl_manager=ccl, fsdp_mesh_axis=fsdp_axis, sp_axis=sp_axis
    )


def _sp_factor(submesh, sp_axis):
    return tuple(submesh.shape)[sp_axis] if sp_axis is not None else 1


def _sp_seq_pad(seq_len, sp):
    """Sequence length padded up to a multiple of `sp * 32`, so each SP shard is tile-aligned. The
    two_refs length 9536 is 298 tiles globally but 9536/4 = 74.5 tiles/shard, so SP=4 pads it to 9600."""
    mult = sp * 32
    return -(-seq_len // mult) * mult


def _shard_seq(x, submesh, sp_axis, seq_dim, seq_pad):
    """Pad `x`'s sequence (at `seq_dim`) up to `seq_pad` and shard it across the SP axis. The trailing
    pad rows are harmless under causal attention (real rows never attend forward into them) and are
    sliced off after the gather."""
    trailing = x.ndim - 1 - seq_dim
    x = torch.nn.functional.pad(x, [0, 0] * trailing + [0, seq_pad - x.shape[seq_dim]])
    return bf16_tensor(x, device=submesh, mesh_axis=sp_axis, shard_dim=seq_dim)


def _gather_seq(out, sp_axis, seq_len):
    """Gather a `[1, seq, hidden]` output sharded on the sequence (dim 1) across `sp_axis`, then drop
    the SP alignment padding back to `seq_len`."""
    got = tensor.to_torch(out, mesh_axes=[None, sp_axis, None])
    return got[:, :seq_len, :]


def _rope(submesh, seq_len, sp_axis=None, seq_pad=None):
    # Production (`test_text_encoder_minimax_h3.py`) leaves `interleaved` at False even though this
    # checkpoint declares `mrope_interleaved`: the two layouts coincide exactly while all three MRoPE
    # axes carry the same position, which is the text-only case. Verified equal here to 7.6e-6, i.e.
    # the documented `theta ** -x` vs `1 / theta ** x` ulp difference and nothing else.
    cos, sin = create_rope_tensors(1, seq_len, None, HEAD_DIM, ROPE_THETA, MROPE_SECTION)

    # Omitting `position_ids` must stay equivalent to passing the token index on all three axes. This is
    # the call Ideogram 4.0 makes through the shared encoder, and `test_qwen3vl.py` -- the test that
    # would otherwise catch a regression -- is pinned to a (2, 4) mesh, so this is the only guard that
    # currently runs.
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
        # cos/sin are (batch, 1, seq, head_dim): shard the sequence (dim 2) so each device gets the
        # rotary tables for exactly the sequence rows it holds.
        return (
            _shard_seq(cos, submesh, sp_axis, 2, seq_pad),
            _shard_seq(sin, submesh, sp_axis, 2, seq_pad),
        )
    return bf16_tensor(cos, device=submesh), bf16_tensor(sin, device=submesh)


@_PARAMS
def test_decoder_block_on_device(golden, seq_len, mesh_device, submesh_shape, tp_axis, sp_axis, fsdp_axis, num_links):
    """The whole pre-norm layer: RMSNorm + attention + RMSNorm + MLP, both residuals.

    This is the profiling entry point -- one iteration of the layer that the 64-layer conditioner
    repeats, and the four matmuls listed in the header. Under `sp_axis` the sequence is sharded on the
    SP axis (FSDP off), so every matmul/norm runs `1/sp` of the rows and attention rings over the axis.
    """
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
        # internal causal path; see the assertion in the `golden` fixture
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
    """Attention alone: fused qkv, per-head QK-RMSNorm, RoPE, SDPA, o_proj. Excludes the residual and
    the input norm, so it attributes the `qkv_proj` / `o_proj` half of the layer's time. Under
    `sp_axis` the SDPA is the causal ring path (`_ring_attention`)."""
    submesh, ctx = _ctx(mesh_device, submesh_shape, tp_axis, sp_axis, fsdp_axis, num_links)

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
    """MLP alone: SwiGLU over `intermediate_size` 25600. Three of the layer's four matmuls by FLOPs,
    so this is where the layer's time is expected to sit. Purely row-wise, so under `sp_axis` it just
    runs `1/sp` of the rows -- no ring, no model change, the input shard is the whole story."""
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
