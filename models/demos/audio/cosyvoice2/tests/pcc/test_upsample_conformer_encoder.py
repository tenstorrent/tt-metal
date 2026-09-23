# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""`UpsampleConformerEncoder`: `PreLookaheadLayer` + a 6-block token-rate Conformer
stack + the encoder's own `Upsample1D` (token-rate -> mel-rate) + a 4-block mel-rate
Conformer stack + a final LayerNorm -- see tt/flow/encoder.py's module docstring for
the verified real-source wiring (`streaming=False` only; two independent
`LinearNoSubsampling` instances, not a shared one; `NUM_UP_BLOCKS=4` hardcoded in
real source, not a yaml parameter).

Same situation as every other component in this phase: `matcha-tts`/`diffusers`/
`conformer`/`wenet`/`espnet` are not installed here, so `*Ref` classes are a
line-by-line transcription of the real source. `test_conformer_encoder.py` already
covers the rel-pos-attention Conformer layer itself in isolation; this file covers
what wraps around it.
"""

from __future__ import annotations

import pytest
import torch

from models.common.utility_functions import comp_pcc

GATE_BF16 = 0.99


# --------------------------------------------------------------------------
# host tier -- no device
# --------------------------------------------------------------------------
def test_prelookahead_layer_has_bounded_receptive_field():
    """`PreLookaheadLayer` mixes a genuine look-ahead (`conv1`, right-padded by
    `pre_lookahead_len`) with a causal conv (`conv2`) -- neither direction is
    unbounded. A perturbation at one position must leave positions far away (not
    just the very next/previous one) completely untouched -- the real
    architectural fact that distinguishes a local conv stack from e.g. the
    Conformer layer's own attention, which mixes every valid position."""
    from models.demos.audio.cosyvoice2.tt.flow.encoder import PreLookaheadLayerRef

    torch.manual_seed(0)
    pl = PreLookaheadLayerRef(8, pre_lookahead_len=3)
    pl.eval()
    x = torch.randn(1, 20, 8)
    with torch.no_grad():
        y1 = pl(x)
        x2 = x.clone()
        x2[:, 10, :] += 999.0
        y2 = pl(x2)
    diff = (y1 - y2).abs().sum(dim=-1)[0]
    changed = set((diff > 1e-4).nonzero().flatten().tolist())
    assert changed, "the perturbed position's own neighbourhood must be affected"
    assert 0 not in changed and 19 not in changed, f"receptive field leaked to a far position: {sorted(changed)}"
    assert changed <= set(range(6, 14)), f"receptive field wider than expected: {sorted(changed)}"


def test_upsample1d_doubles_length():
    from models.demos.audio.cosyvoice2.tt.flow.encoder import Upsample1DRef

    torch.manual_seed(0)
    up = Upsample1DRef(8, stride=2)
    up.eval()
    x = torch.randn(1, 15, 8)
    with torch.no_grad():
        out = up(x)
    assert out.shape == (1, 30, 8)


def test_encoder_torch_reference_shape_and_range():
    from models.demos.audio.cosyvoice2.tt.flow.encoder import UpsampleConformerEncoderRef

    torch.manual_seed(1)
    enc = UpsampleConformerEncoderRef()
    enc.eval()
    b, t_len, d = 1, 20, 512
    x = torch.randn(b, t_len, d) * 0.1
    with torch.no_grad():
        out = enc(x)
    assert out.shape == (b, t_len * 2, d)
    assert torch.isfinite(out).all()


# --------------------------------------------------------------------------
# device tier -- needs silicon
# --------------------------------------------------------------------------
needs_l1_small = pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)


@needs_l1_small
def test_device_prelookahead_layer_matches_torch_reference(device):
    import ttnn
    from models.demos.audio.cosyvoice2.tt.flow.encoder import D_MODEL, PreLookaheadLayerRef, TtPreLookaheadLayer

    torch.manual_seed(0)
    b, t_len = 1, 32
    pl = PreLookaheadLayerRef(D_MODEL)
    pl.eval()
    x = torch.randn(b, t_len, D_MODEL) * 0.1
    with torch.no_grad():
        want = pl(x)

    tt_pl = TtPreLookaheadLayer(device, pl)
    x_dev = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    got = ttnn.to_torch(tt_pl(x_dev, t_len, 1)).float()

    assert got.shape == want.shape
    passed, pcc = comp_pcc(want, got, GATE_BF16)
    print(f"\n  device PreLookaheadLayer PCC {pcc}")
    assert passed, pcc


@needs_l1_small
def test_device_upsample1d_matches_torch_reference(device):
    import ttnn
    from models.demos.audio.cosyvoice2.tt.flow.encoder import D_MODEL, TtUpsample1D, Upsample1DRef

    torch.manual_seed(1)
    b, t_len = 1, 32
    up = Upsample1DRef(D_MODEL)
    up.eval()
    x = torch.randn(b, t_len, D_MODEL) * 0.1
    with torch.no_grad():
        want = up(x)

    tt_up = TtUpsample1D(device, up)
    x_dev = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    got = ttnn.to_torch(tt_up(x_dev, t_len, 1)).float()

    assert got.shape == want.shape
    passed, pcc = comp_pcc(want, got, GATE_BF16)
    print(f"\n  device Upsample1D (encoder's own) PCC {pcc}")
    assert passed, pcc


@needs_l1_small
def test_device_linear_no_subsampling_matches_torch_reference(device):
    import ttnn
    from models.demos.audio.cosyvoice2.tt.flow.encoder import D_MODEL, LinearNoSubsamplingRef, TtLinearNoSubsampling

    torch.manual_seed(2)
    b, t_len = 1, 32
    emb = LinearNoSubsamplingRef(D_MODEL, D_MODEL)
    emb.eval()
    x = torch.randn(b, t_len, D_MODEL) * 0.1
    with torch.no_grad():
        want = emb(x)

    tt_emb = TtLinearNoSubsampling(device, emb)
    x_dev = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    got = ttnn.to_torch(tt_emb(x_dev)).float()

    assert got.shape == want.shape
    passed, pcc = comp_pcc(want, got, GATE_BF16)
    print(f"\n  device LinearNoSubsampling PCC {pcc}")
    assert passed, pcc


@needs_l1_small
@pytest.mark.parametrize("t_len", [20, 33])
def test_device_upsample_conformer_encoder_matches_torch_reference(device, t_len):
    """The full encoder end to end: embed -> pre-lookahead -> 6 Conformer blocks
    -> upsample -> up_embed -> 4 more Conformer blocks -> final norm. Random-init
    (no CosyVoice2 checkpoint yet). Two lengths, including an odd one (33), since
    nothing in this wiring assumes an even/tile-round input length."""
    import ttnn
    from models.demos.audio.cosyvoice2.tt.flow.encoder import TtUpsampleConformerEncoder, UpsampleConformerEncoderRef

    torch.manual_seed(t_len)
    enc = UpsampleConformerEncoderRef()
    enc.eval()
    b, d = 1, 512
    x = torch.randn(b, t_len, d) * 0.1
    with torch.no_grad():
        want = enc(x)

    tt_enc = TtUpsampleConformerEncoder(device, enc)
    x_dev = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    got = ttnn.to_torch(tt_enc(x_dev, t_len, 1)).float()

    assert got.shape == want.shape == (b, t_len * 2, d)
    passed, pcc = comp_pcc(want, got, GATE_BF16)
    print(f"\n  device UpsampleConformerEncoder (T={t_len}) PCC {pcc}")
    assert passed, pcc


needs_l1_small_trace = pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 32768, "trace_region_size": 50_000_000}], indirect=True
)


@needs_l1_small_trace
@pytest.mark.parametrize("t_len", [20, 33])
def test_device_upsample_conformer_encoder_traced_matches_eager(device, t_len):
    """Cached/traced whole-encoder forward (`TtUpsampleConformerEncoder._capture`/
    `_call_traced`, added 2026-09-22) vs. the untraced eager path, at the SAME instance --
    two independent calls at the same `(t_len, batch_size)` with DIFFERENT random inputs,
    so the second call exercises `_reuse_trace`'s cache-hit path (not just a fresh
    capture), and both are checked against their own eager result rather than a shared
    golden answer."""
    import ttnn
    from models.demos.audio.cosyvoice2.tt.flow.encoder import TtUpsampleConformerEncoder, UpsampleConformerEncoderRef

    torch.manual_seed(t_len)
    enc = UpsampleConformerEncoderRef()
    enc.eval()
    b, d = 1, 512
    tt_enc = TtUpsampleConformerEncoder(device, enc)
    try:
        for rep in range(2):
            torch.manual_seed(t_len * 1000 + rep)
            x = torch.randn(b, t_len, d) * 0.1
            x_dev_eager = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
            eager = ttnn.to_torch(tt_enc(x_dev_eager, t_len, 1, use_trace=False)).float()
            x_dev_traced = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
            traced = ttnn.to_torch(tt_enc(x_dev_traced, t_len, 1, use_trace=True)).float()
            assert traced.shape == eager.shape
            passed, pcc = comp_pcc(eager, traced, GATE_BF16)
            print(f"\n  rep {rep} device UpsampleConformerEncoder traced vs eager (T={t_len}) PCC {pcc}")
            assert passed, f"rep {rep}: {pcc}"
    finally:
        tt_enc.release_encoder_trace()
