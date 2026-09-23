# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""`UpsampleConformerEncoder`: `PreLookaheadLayer` + a 6-block token-rate Conformer
stack + the encoder's own `Upsample1D` (token-rate -> mel-rate) + a 4-block mel-rate
Conformer stack + a final LayerNorm -- see tt/flow/encoder.py's module docstring for
the verified real-source wiring (two independent `LinearNoSubsampling` instances, not
a shared one; `NUM_UP_BLOCKS=4` hardcoded in real source, not a yaml parameter).

Streaming (`streaming=True`/`context=`, added 2026-09-23) tests are below the
non-streaming ones -- real chunk-causal masking (`subsequent_chunk_mask_torch`) and
real-lookahead `context` for `pre_lookahead_layer`, verified against a real checkpoint
in `scripts/perf_2026_09_23/bucket_padding_boundary_check_v2.py` before being ported
here as permanent regression tests (random-init weights here, matching this file's own
convention -- these check structural/masking correctness, not weight-specific accuracy,
which the real-checkpoint script already covers).

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


# --------------------------------------------------------------------------
# streaming -- chunk-causal masking, real-lookahead context, bucketing (2026-09-23)
# --------------------------------------------------------------------------


def test_subsequent_chunk_mask_matches_real_upstream_example():
    """`subsequent_chunk_mask_torch` against real upstream's own docstring example
    (`cosyvoice.utils.mask.subsequent_chunk_mask(4, 2) ==
    [[1,1,0,0],[1,1,0,0],[1,1,1,1],[1,1,1,1]]`), fetched directly from
    github.com/FunAudioLLM/CosyVoice 2026-09-23."""
    from models.demos.audio.cosyvoice2.tt.flow.encoder import subsequent_chunk_mask_torch

    want = torch.tensor(
        [
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
        ],
        dtype=torch.bool,
    )
    got = subsequent_chunk_mask_torch(4, 2)
    assert torch.equal(got, want), got


def test_bucket_length_rounds_up_to_step():
    from models.demos.audio.cosyvoice2.tt.flow.encoder import BUCKET_STEP, bucket_length

    assert bucket_length(1) == BUCKET_STEP
    assert bucket_length(BUCKET_STEP) == BUCKET_STEP
    assert bucket_length(BUCKET_STEP + 1) == 2 * BUCKET_STEP
    assert bucket_length(0) == 0


@needs_l1_small
def test_device_streaming_encoder_naive_lookahead_corrupts_real_lookahead_recovers(device):
    """The core streaming-correctness regression (ported from the real-checkpoint
    verification in `scripts/perf_2026_09_23/bucket_padding_boundary_check_v2.py`, random-
    init weights here). Ground truth: real chunk-causal-masked encoder run on the WHOLE
    eventual `F`-token sequence at once (matches upstream's `finalize=True`). Compared
    against a `T`-token chunk (`finalize=False`) two ways:

    - naive zero-lookahead (no `context`): MUST measurably corrupt the boundary --
      asserted explicitly (not just documented), so a future change that accidentally
      makes this pass would itself be caught as a surprise, not silently accepted.
    - real lookahead `context` (the next `pre_lookahead_len` real tokens): MUST clear the
      0.99 gate -- this is the actual correctness bar for a real bucketed implementation
      (see BRINGUP_STATUS.md's "Streaming design, round 1" conclusion: 0.99 is the right
      target, not a relaxed one -- the real checkpoint run measured 0.999919).
    """
    import ttnn
    from models.demos.audio.cosyvoice2.tt.flow.encoder import (
        CHUNK_SIZE,
        CHUNK_SIZE_UP,
        PRE_LOOKAHEAD_LEN,
        TtUpsampleConformerEncoder,
        UpsampleConformerEncoderRef,
        chunk_causal_bias_torch,
    )

    torch.manual_seed(0)
    d = 512
    F_LEN, T_TRUE = 96, 75  # T_TRUE a multiple of CHUNK_SIZE=25 (3 chunks)
    assert T_TRUE % CHUNK_SIZE == 0

    enc = UpsampleConformerEncoderRef()
    enc.eval()
    tokens_emb = torch.randn(1, F_LEN, d) * 0.1
    tt_enc = TtUpsampleConformerEncoder(device, enc)

    def run(xs_dev, length: int, context_dev=None):
        return ttnn.to_torch(tt_enc(xs_dev, length, 1, context=context_dev, streaming=True)).float()

    xs_full = ttnn.from_torch(tokens_emb[:, :F_LEN], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    full_out = run(xs_full, F_LEN)
    T2 = T_TRUE * 2
    ref_out = full_out[:, :T2, :]

    xs_naive = ttnn.from_torch(tokens_emb[:, :T_TRUE], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    naive_out = run(xs_naive, T_TRUE)

    xs_main = ttnn.from_torch(tokens_emb[:, :T_TRUE], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    context_dev = ttnn.from_torch(
        tokens_emb[:, T_TRUE : T_TRUE + PRE_LOOKAHEAD_LEN], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    lookahead_out = run(xs_main, T_TRUE, context_dev=context_dev)

    def boundary_pcc(out, n=6):
        _, pcc = comp_pcc(ref_out[:, T2 - n : T2, :], out[:, T2 - n : T2, :], GATE_BF16)
        return pcc

    naive_pcc = boundary_pcc(naive_out)
    lookahead_pcc = boundary_pcc(lookahead_out)
    print(f"\n  naive zero-lookahead boundary PCC {naive_pcc}  |  real-lookahead-context boundary PCC {lookahead_pcc}")

    # naive_pcc's ABSOLUTE value is weight-magnitude dependent -- this file's random-init
    # weights (small, torch.randn*0.1) sit close to a near-identity regime where zero-vs-
    # real lookahead content barely moves the output (measured ~0.9965, nowhere near as
    # dramatic as the real checkpoint's 0.832 -- see BRINGUP_STATUS.md), so an absolute
    # `< GATE_BF16` assertion here would be fragile. The RELATIVE claim (naive is
    # measurably worse than real lookahead) is robust regardless of weight scale, and is
    # what this permanent regression actually needs to catch; the real checkpoint script
    # is the source of truth for the absolute risk.
    assert naive_pcc < lookahead_pcc, f"expected naive zero-lookahead to be worse than real lookahead context: naive={naive_pcc} lookahead={lookahead_pcc}"
    assert lookahead_pcc >= GATE_BF16, f"real lookahead context should clear the {GATE_BF16} gate, got {lookahead_pcc}"


@needs_l1_small
def test_device_bucketed_encoder_matches_exact_length(device):
    """Bucketing (`valid_length < length`: the tensor's geometry is a fixed bucket size,
    the true content is shorter) must be invisible to the result. `valid_length` decouples
    `pre_lookahead_layer`'s own true-content computation (context concatenates right after
    `valid_length`, matching a real chunk boundary) from the REST of the pipeline's
    bucket-shaped geometry (`length`) -- see `TtUpsampleConformerEncoder.__call__`'s
    docstring for why this decoupling is necessary (naively passing the bucket size straight
    through to `pre_lookahead_layer` would place `context` at the wrong position, right
    after the bucket's padding instead of right after the true content). Compares a chunk
    run at its EXACT length against the SAME real content padded out to a larger bucket
    length via `valid_length`, both real chunk-causal-masked with the same real lookahead
    context."""
    import ttnn
    from models.demos.audio.cosyvoice2.tt.flow.encoder import (
        PRE_LOOKAHEAD_LEN,
        TtUpsampleConformerEncoder,
        UpsampleConformerEncoderRef,
        bucket_length,
    )

    torch.manual_seed(1)
    d = 512
    T_TRUE = 75  # multiple of CHUNK_SIZE=25
    B_BUCKET = bucket_length(T_TRUE)  # 128 at BUCKET_STEP=64
    assert B_BUCKET > T_TRUE

    enc = UpsampleConformerEncoderRef()
    enc.eval()
    tokens_emb = torch.randn(1, T_TRUE + PRE_LOOKAHEAD_LEN, d) * 0.1
    tt_enc = TtUpsampleConformerEncoder(device, enc)

    context_dev = ttnn.from_torch(
        tokens_emb[:, T_TRUE : T_TRUE + PRE_LOOKAHEAD_LEN], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )

    xs_exact = ttnn.from_torch(tokens_emb[:, :T_TRUE], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    exact_out = ttnn.to_torch(tt_enc(xs_exact, T_TRUE, 1, context=context_dev, streaming=True)).float()

    # Bucketed run: same real T_TRUE tokens, zero-padded out to B_BUCKET at the TENSOR
    # level (content beyond T_TRUE is never read -- valid_length tells the encoder where
    # the true content actually ends).
    pad = torch.zeros(1, B_BUCKET - T_TRUE, d)
    xs_bucketed = ttnn.from_torch(
        torch.cat([tokens_emb[:, :T_TRUE], pad], dim=1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    bucketed_out = ttnn.to_torch(
        tt_enc(xs_bucketed, B_BUCKET, 1, context=context_dev, streaming=True, valid_length=T_TRUE)
    ).float()

    T2 = T_TRUE * 2
    passed, pcc = comp_pcc(exact_out[:, :T2, :], bucketed_out[:, :T2, :], GATE_BF16)
    print(f"\n  exact length T={T_TRUE} vs bucketed B={B_BUCKET} (valid_length={T_TRUE}) PCC {pcc}")
    assert passed, pcc
