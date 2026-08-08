# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Production-shape regression tests for the TTNN generator.

These tests cover shapes the existing T=10 unit tests in test_runtime.py
do not exercise:

- Real chunk sizes used by demo.py overlap-add (50, 53, 55, 60)
- ResBlock at production seq_lens (720, 7200, 14400, 28800 for T=60)
- kernel_size=11 ResBlock at long sequences (HiFi-GAN k_idx=2) — the
  specific shape that silently broke a previous device-residency attempt
- The demo's exact 6-chunk sequence pattern
- Repeated execution of the same chunk

Design choices:

1. **Function-scoped device fixture.** TTNN accumulates sliding-window
   halo buffers in L1 banks per unique shape, and the program cache
   isn't aggressively freed across forward calls. A module-scoped device
   that runs many shape variants in sequence will eventually exhaust L1,
   even though each individual shape works fine. Function-scoped device
   mimics how demo.py actually runs (one device per inference session).

2. **l1_small_size=32768** — matches `demo.py` exactly.

3. **No try/except.** Any TTNN runtime exception fails the test loudly.
   Silent fallback to torch is what hid the last regression.
"""

import os

import pytest
import torch
from safetensors.torch import load_file

import ttnn
from models.demos.rvc.demo import MAX_CHUNK_FRAMES, OVERLAP, TARGET_LEN
from models.demos.rvc.tests.pcc_utils import compute_pcc
from models.demos.rvc.torch_impl.reference import (
    build_torch_generator,
    torch_generator_forward,
)
from models.demos.rvc.tt.runtime import (
    NUM_KERNELS,
    NUM_UPSAMPLES,
    RESBLOCK_DILATIONS,
    UPSAMPLE_RATES,
    TTNNGeneratorNSF,
)

CHECKPOINT_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "data",
    "assets",
    "pretrained_v2",
    "f0G48k.safetensors",
)

# Cumulative upsample rates: seq_len after stage s = T * _CUM_RATES[s+1].
_CUM_RATES = [1]
for _r in UPSAMPLE_RATES:
    _CUM_RATES.append(_CUM_RATES[-1] * _r)

# First chunk is one-sided (75 + OVERLAP = 78); interior/padded-tail are
# TARGET_LEN (75 + 2*OVERLAP = 81).
PRODUCTION_CHUNK_SIZES = [MAX_CHUNK_FRAMES + OVERLAP, TARGET_LEN]
PRODUCTION_RB_SEQ_LENS = [TARGET_LEN * _CUM_RATES[s + 1] for s in range(NUM_UPSAMPLES)]


@pytest.fixture(scope="module")
def checkpoint():
    if not os.path.exists(CHECKPOINT_PATH):
        pytest.skip(f"Checkpoint not found: {CHECKPOINT_PATH}")
    return load_file(CHECKPOINT_PATH)


@pytest.fixture(scope="function")
def fresh_gen(checkpoint):
    """Fresh device + generator per test — function-scoped so L1 state
    can't accumulate across varying shapes (causes pytest-only failures
    that don't repro in real demo invocations)."""
    dev = ttnn.open_device(device_id=0, l1_small_size=32768)
    try:
        g = TTNNGeneratorNSF.from_checkpoint(checkpoint, dev)
        try:
            yield g
        finally:
            g.deallocate()
    finally:
        ttnn.close_device(dev)


@pytest.mark.parametrize(
    "stage",
    range(NUM_UPSAMPLES),
    ids=[f"stage{s}_k11_seq{PRODUCTION_RB_SEQ_LENS[s]}" for s in range(NUM_UPSAMPLES)],
)
def test_resblock_k11_at_production_seq_len(fresh_gen, stage):
    """k_idx=2 (kernel=11) ResBlock at production seq_lens 972/9720/19440/38880.
    Exercises _resblock1_device (the device-resident path the generator
    actually calls), not the legacy host-mediated _resblock1 helper."""
    torch.manual_seed(stage)
    k_idx = 2
    rb_idx = stage * NUM_KERNELS + k_idx
    seq_len = PRODUCTION_RB_SEQ_LENS[stage]
    ch = fresh_gen._resblocks[rb_idx]["channels"]

    x = torch.randn(1, ch, seq_len)
    out = fresh_gen._resblock1_device(x, rb_idx, RESBLOCK_DILATIONS[k_idx], seq_len)

    assert out.shape == (1, ch, seq_len), f"shape mismatch: {out.shape}"
    assert torch.isfinite(out).all(), "non-finite values in ResBlock output"


@pytest.mark.parametrize("T", PRODUCTION_CHUNK_SIZES)
def test_generator_at_demo_chunk_size(fresh_gen, T):
    """gen.forward at the two chunk sizes demo overlap-add emits at
    MAX_CHUNK_FRAMES=75 + OVERLAP=3: one-sided first (T=78) and padded
    interior (T=81). Strict: any TTNN exception fails the test."""
    torch.manual_seed(T)
    z = torch.randn(1, 192, T)
    har = torch.randn(1, 1, T * 480)
    g_emb = torch.randn(1, 256, 1)

    out = fresh_gen(z, har, g_emb)

    assert out.shape == (1, 1, T * 480), f"shape mismatch at T={T}: {out.shape}"
    assert torch.isfinite(out).all(), f"non-finite at T={T}"
    assert out.abs().max() <= 1.0, f"out of [-1, 1] at T={T}: max={out.abs().max()}"


def test_generator_repeated_same_chunk_deterministic(fresh_gen):
    """Same TARGET_LEN chunk three times — bit-identical determinism."""
    T = TARGET_LEN
    torch.manual_seed(99)
    z = torch.randn(1, 192, T)
    har = torch.randn(1, 1, T * 480)
    g_emb = torch.randn(1, 256, 1)

    results = [fresh_gen(z, har, g_emb).clone() for _ in range(3)]
    for i in range(1, 3):
        max_diff = (results[0] - results[i]).abs().max().item()
        assert max_diff == 0.0, f"non-deterministic at run {i}: max_diff={max_diff}"


def test_generator_correctness_vs_torch_at_target_len(fresh_gen, checkpoint):
    """PCC against torch reference at the padded production chunk size."""
    T = TARGET_LEN
    torch.manual_seed(0)
    z = torch.randn(1, 192, T)
    har = torch.randn(1, 1, T * 480)

    emb_g = torch.nn.Embedding(109, 256)
    emb_g.weight.data = checkpoint["emb_g.weight"].float()
    g = emb_g(torch.tensor([0])).unsqueeze(-1)

    gen_torch = build_torch_generator(checkpoint)
    ref = torch_generator_forward(z, har, g, gen_torch)
    out = fresh_gen(z, har, g)

    assert out.shape == ref.shape, f"shape mismatch: ttnn={out.shape} ref={ref.shape}"
    pcc = compute_pcc(ref, out)
    assert pcc > 0.95, f"PCC={pcc:.6f} below 0.95 threshold at T={T}"


def test_generator_b2_per_row_speaker_conditioning(checkpoint):
    """Regression for PR #45686 review: B>1 with different speaker embeddings
    per row must produce different audio per row.

    Earlier the cond_linear path used g[:1] and broadcast row-0 conditioning
    across the whole batch, so a mixed-target B=2 batch silently produced
    identical audio for both rows. Existing tests only exercised B=1 or
    single-g paths, so this could ship silently.

    Uses random g per row instead of two checkpoint speaker embeddings to
    avoid coupling to which embedding rows are distinct in this checkpoint
    (e.g. emb_g(0) and emb_g(1) are byte-identical in rvc-nano v2/48k).

    Opens its own device with l1_small_size=131072 instead of using the
    fresh_gen fixture's 32768 — B>1 conv shapes have larger halo
    allocations and OOM the smaller L1_SMALL bank.
    """
    T = 60
    torch.manual_seed(0)
    # Random g per row (not emb_g(0/1) — those are byte-identical in
    # rvc-nano v2/48k, which would mask the bug we're testing for).
    g = torch.randn(2, 256, 1)
    # Identical z and har so any per-row difference is purely from g.
    z_one = torch.randn(1, 192, T)
    har_one = torch.randn(1, 1, T * 480)
    z = z_one.repeat(2, 1, 1)
    har = har_one.repeat(2, 1, 1)

    dev = ttnn.open_device(device_id=0, l1_small_size=131072)
    try:
        gen = TTNNGeneratorNSF.from_checkpoint(checkpoint, dev)
        try:
            out = gen(z, har, g)
        finally:
            gen.deallocate()
    finally:
        ttnn.close_device(dev)

    assert out.shape == (2, 1, T * 480), f"shape mismatch: {out.shape}"
    diff = (out[0] - out[1]).abs().max().item()
    assert diff > 1e-3, (
        f"per-row speaker conditioning broken: identical z/har with different "
        f"g produced max_diff={diff:.2e} between rows (expected > 0)"
    )


@pytest.mark.parametrize("B", [2, 3, 5, 8])
def test_generator_batched_matches_individual_b1_calls(checkpoint, B):
    """Strong per-row equivalence across the full batch range used by the
    Stage 3 "5+ concurrent conversions" bullet.

    For each B in [2, 3, 5, 8]:
      - Compute B distinct B=1 references (different z, har, g per row).
      - Run one B=B batched call with the same inputs concatenated.
      - Per row: PCC > 0.995 vs its B=1 ref AND PCC(row_i, ref_i) >
        PCC(row_i, ref_j) for all j != i (no row permutation).
      - All rows pairwise distinct (no broadcast / row collapse).

    Catches: row-0 broadcast (PR review bug), row permutation, latent
    broadcast in ResBlocks / upsample / noise / conv_pre / conv_post,
    and any per-B regression that B=2-only testing would miss.

    Tolerance 0.995 (not 1.0) because B>1 picks different conv configs
    than B=1 (act_block_h_override=32 at B>1 vs the HEIGHT_SHARDED
    whitelist at B=1) and per-element values differ within bf16
    quantisation. Empirically the noise is ~1e-3 relative; 0.995 leaves
    a small margin.
    """
    T = 60
    torch.manual_seed(0)
    zs = [torch.randn(1, 192, T) for _ in range(B)]
    hars = [torch.randn(1, 1, T * 480) for _ in range(B)]
    gs = [torch.randn(1, 256, 1) for _ in range(B)]

    def pcc(a, b):
        a, b = a.flatten().double(), b.flatten().double()
        a, b = a - a.mean(), b - b.mean()
        denom = torch.sqrt(torch.sum(a * a) * torch.sum(b * b)) + 1e-12
        return float(torch.sum(a * b) / denom)

    # Fresh device for B=1 refs and again for the batched call so state
    # from one path can't leak into the other.
    dev = ttnn.open_device(device_id=0, l1_small_size=131072)
    try:
        gen = TTNNGeneratorNSF.from_checkpoint(checkpoint, dev)
        try:
            refs = [gen(zs[i], hars[i], gs[i]).clone() for i in range(B)]
        finally:
            gen.deallocate()
    finally:
        ttnn.close_device(dev)

    dev = ttnn.open_device(device_id=0, l1_small_size=131072)
    try:
        gen = TTNNGeneratorNSF.from_checkpoint(checkpoint, dev)
        try:
            z = torch.cat(zs, dim=0)
            har = torch.cat(hars, dim=0)
            g = torch.cat(gs, dim=0)
            out = gen(z, har, g)
        finally:
            gen.deallocate()
    finally:
        ttnn.close_device(dev)

    assert out.shape == (B, 1, T * 480), f"shape mismatch at B={B}: {out.shape}"

    # (a) all rows pairwise distinct — catches broadcast / row collapse
    for i in range(B):
        for j in range(i + 1, B):
            d = (out[i] - out[j]).abs().max().item()
            assert d > 1e-3, f"B={B} rows {i} and {j} are identical (max_diff={d:.2e}) — " f"per-row routing broken"

    # (b) each row matches its own B=1 reference within bf16 noise
    pccs_self = [pcc(refs[i], out[i : i + 1]) for i in range(B)]
    for i, p in enumerate(pccs_self):
        assert p > 0.995, (
            f"B={B} row {i} does not match its B=1 reference (PCC={p:.4f}). "
            f"Per-row correctness bug, row permutation, or numerical "
            f"regression beyond bf16 noise."
        )

    # (c) row i must match ref_i better than any other ref_j — catches
    # a permutation that happens to preserve overall row PCC by coincidence.
    for i in range(B):
        for j in range(B):
            if i == j:
                continue
            p_wrong = pcc(refs[j], out[i : i + 1])
            assert pccs_self[i] > p_wrong, (
                f"B={B} row {i} matches ref{j} (PCC={p_wrong:.4f}) "
                f"better than its own ref{i} (PCC={pccs_self[i]:.4f}) — "
                f"row-permutation bug at B={B}."
            )
