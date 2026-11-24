# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Device unit tests for the GLM-5.3-Flash decoder layer and the functional decoder wiring.

Two layers of coverage, cheapest first:

1. **Structure** -- ``LayerSpec``/``DecoderLayer`` pick the right attention and FFN for each of the
   three layer kinds, and the module set is exactly the eight a layer owns. No device work beyond
   what the constructors need.
2. **The residual contract** -- ``DecoderLayer.forward``'s wiring is
   ``attn_hc.pre -> norm -> attn -> attn_hc.post -> ffn_hc.pre -> norm -> ffn -> ffn_hc.post``, with
   the two hyper-connections doing the stream mixing. This is tested with *stub* attention/FFN
   blocks whose outputs are known constants, so the assertion is about the wiring (does the residual
   come back out, are the streams the right shape, is the order right) and not about numerics that
   module-level tests already cover.

A full numeric layer test needs real weights; the CPU side of that is
``reference/hf_layer_reference.py``, which already builds real ``Glm5NextTextDecoderLayer`` instances
from the checkpoint. Wiring it to this device layer is the next step, and is blocked for KDA layers by
``prepare_chunk_recurrence`` (see ``test_kda.py::PREP_BLOCKER``).
"""

from __future__ import annotations

import pytest
import torch

import ttnn
from models.experimental.glm53_flash.tt.config import Glm5NextTextConfig, LayerSpec
from models.experimental.glm53_flash.tt.decoder_layer import DecoderLayer
from models.experimental.glm53_flash.tt.hyperconnection import HyperConnection

pytestmark = pytest.mark.use_module_device({"l1_small_size": 24576})

DRAM = ttnn.DRAM_MEMORY_CONFIG
HC = 4
HIDDEN = 4096


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.flatten().float(), b.flatten().float()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


# ---------------------------------------------------------------------------------------------
# 1. Structure: the layer kind decides the modules
# ---------------------------------------------------------------------------------------------


def _cfg():
    c = Glm5NextTextConfig()
    c.__post_init__()
    return c


@pytest.mark.parametrize(
    ("layer_idx", "attention", "mlp"),
    [
        (0, "kda", "dense"),  # first_k_dense_replace == 3
        (2, "kda", "dense"),
        (4, "kda", "moe"),
        (3, "mla", "moe"),  # every 4th layer from 3 is DSA/MLA
        (43, "mla", "moe"),
    ],
)
def test_layer_spec_selects_modules(layer_idx, attention, mlp):
    """The layer index alone determines the attention and FFN kind, with no device needed."""
    spec = LayerSpec.from_config(_cfg(), layer_idx)
    assert spec.attention == attention, f"layer {layer_idx}: expected {attention}, got {spec.attention}"
    assert spec.mlp == mlp, f"layer {layer_idx}: expected {mlp}, got {spec.mlp}"


def test_dsa_layers_are_the_expected_set():
    """The 11 MLA layers are 3, 7, ... 43 (a checkpoint fact, not a tunable)."""
    cfg = _cfg()
    mla_layers = [i for i in range(cfg.num_hidden_layers) if LayerSpec.from_config(cfg, i).attention == "mla"]
    assert mla_layers == [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43]


def test_decoder_layer_owns_the_expected_modules():
    """``DecoderLayer.__init__`` wires exactly: 2 hyper-connections, 2 norms, 1 attention, 1 FFN.

    Checked on the class source rather than by construction, so it runs without a weight source (the
    real constructor needs a 300 GB checkpoint and, for a KDA layer, the blocked prefill op).
    """
    import inspect

    src = inspect.getsource(DecoderLayer.__init__)
    assert "self.attn_hc = HyperConnection(" in src
    assert "self.ffn_hc = HyperConnection(" in src
    assert "self.input_norm" in src
    assert "self.post_attn_norm" in src
    assert "self.attn = KDA(" in src and "self.attn = MLA(" in src
    assert "self.mlp = SwiGLUMLP(" in src and "self.mlp = MoE(" in src


def test_forward_order_is_attn_then_ffn():
    """The residual order is hc -> attn -> hc -> ffn -> hc; a swapped pair would still run."""
    import inspect

    body = inspect.getsource(DecoderLayer.forward)
    i_attn_pre = body.index("self.attn_hc.pre")
    i_attn_call = body.index("attention(self._norm")  # the call site, not the parameter name
    i_attn_post = body.index("self.attn_hc.post")
    i_ffn_pre = body.index("self.ffn_hc.pre")
    i_ffn_call = body.index("self._ffn(self._norm")
    i_ffn_post = body.index("self.ffn_hc.post")
    assert i_attn_pre < i_attn_call < i_attn_post < i_ffn_pre < i_ffn_call < i_ffn_post


# ---------------------------------------------------------------------------------------------
# 2. The residual contract, with stub sublayers
# ---------------------------------------------------------------------------------------------


class _StubLayer:
    """A ``DecoderLayer``-shaped object whose attention/FFN are known constants.

    Reuses the real ``DecoderLayer.forward`` (unbound) and the real hyper-connections, with
    ``_norm``/``_ffn`` replaced by identity/constant so the test isolates the stream wiring.
    ``_ffn`` adds a device-allocated constant of the right shape, so it stays ttnn-native.
    """

    forward = DecoderLayer.forward

    def __init__(self, attn_hc, ffn_hc, cfg, device, tokens, attn_scale, ffn_scale):
        self.attn_hc = attn_hc
        self.ffn_hc = ffn_hc
        self.cfg = cfg
        self.input_norm = None
        self.post_attn_norm = None
        self.device = device
        self.tokens = tokens
        torch.manual_seed(99)
        self._ffn_const = ttnn.from_torch(
            torch.randn(1, 1, tokens, HIDDEN) * ffn_scale,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=DRAM,
        )
        self._attn_scale = attn_scale

    def _norm(self, h, weight):
        return h  # identity: the norms are covered by their own tests

    def _ffn(self, x, decode: bool):
        return ttnn.add(x, self._ffn_const)

    def attention(self, h):
        torch.manual_seed(98)
        a = torch.randn(1, 1, self.tokens, HIDDEN) * self._attn_scale
        return ttnn.from_torch(a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device, memory_config=DRAM)


def _hc(device, seed, hc_eps=1e-6, norm_eps=1e-5, iters=20):
    torch.manual_seed(seed)
    fn = torch.randn((2 + HC) * HC, HC * HIDDEN) * 0.02
    return HyperConnection(
        device,
        ttnn.from_torch(
            fn.t().contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=DRAM
        ),
        torch.randn((2 + HC) * HC) * 0.1,
        torch.randn(3) * 0.1,
        hc_mult=HC,
        hidden_size=HIDDEN,
        sinkhorn_iters=iters,
        hc_eps=hc_eps,
        norm_eps=norm_eps,
    )


@pytest.mark.parametrize("tokens", [1, 32])
def test_forward_preserves_stream_shape(device, tokens):
    """One layer step maps ``[1,1,T,hc*D]`` fp32 to the same shape -- the pipeline's contract."""
    cfg = _cfg()
    layer = _StubLayer(_hc(device, 0), _hc(device, 1), cfg, device, tokens, attn_scale=0.1, ffn_scale=0.1)
    streams = torch.randn(1, 1, tokens, HC * HIDDEN) * 0.5
    st = ttnn.from_torch(streams, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=DRAM)

    out = layer.forward(st, layer.attention, decode=(tokens == 1))
    ttnn.synchronize_device(device)

    assert tuple(out.shape) == (1, 1, tokens, HC * HIDDEN), out.shape
    assert out.dtype == ttnn.float32, f"streams must stay fp32, got {out.dtype}"


def test_forward_is_not_identity(device):
    """Guards against a wiring bug that silently returns the input streams unchanged."""
    cfg = _cfg()
    tokens = 32
    layer = _StubLayer(_hc(device, 2), _hc(device, 3), cfg, device, tokens, attn_scale=0.5, ffn_scale=0.5)
    streams = torch.randn(1, 1, tokens, HC * HIDDEN) * 0.5
    st = ttnn.from_torch(streams, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=DRAM)
    out = layer.forward(st, layer.attention, decode=False)
    ttnn.synchronize_device(device)
    got = ttnn.to_torch(out).float().reshape(tokens, HC * HIDDEN)
    assert pcc(got, streams.reshape(tokens, HC * HIDDEN)) < 0.999, "output equals the input: mixing did not run"


def test_forward_does_not_mutate_input_streams(device):
    """Callers keep the streams tensor across the step (it is the residual), so it must survive."""
    cfg = _cfg()
    tokens = 32
    layer = _StubLayer(_hc(device, 4), _hc(device, 5), cfg, device, tokens, attn_scale=0.0, ffn_scale=0.0)
    streams = torch.randn(1, 1, tokens, HC * HIDDEN) * 0.5
    st = ttnn.from_torch(streams, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=DRAM)
    layer.forward(st, layer.attention, decode=False)
    ttnn.synchronize_device(device)
    assert torch.allclose(ttnn.to_torch(st).float().reshape(streams.shape), streams, rtol=0, atol=0)
