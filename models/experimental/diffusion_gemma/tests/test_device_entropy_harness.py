# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""On-device entropy + Gumbel-max harness validation (#47468).

The #47468 accuracy harness must extend PCC **beyond logits** to the diffusion
*decisions*: per-position **entropy** (−Σ p·log p) and **Gumbel-max argmax
agreement**, validated device-vs-torch and **especially under bfp8** (small-
probability drift can flip accept/renoise). gemma4 has no entropy op, so these
ttnn primitives (`tt/sampling.py`) are net-new; here we diff them against the
pure-torch oracle (`reference/sampling.py`) at the op level on QB2.

Measured on QB2 (P150x4):
  * entropy (bf16): mean|Δ| ≈ 0.09 on values ~7.6 (≈1%) — accurate.
  * entropy (bfp8): mean|Δ| ≈ 2.6 — **materially degraded**. This is the harness's
    headline finding: under bfp8 the per-position entropy (hence accept/renoise
    decisions) is unreliable, so the loop must validate *decisions*, not trust
    bfp8 probabilities. The test asserts bf16-accurate AND bfp8-strictly-worse.
  * Gumbel-max argmax agreement under injected noise: ~0.98 (bf16) — the ~2% gap
    is near-max ties flipping under bf16 logit quantization, not an op error.

Determinism: Gumbel noise is generated in torch and **injected** into both paths
(on-device RNG won't reproduce torch's RNG bit-exactly), so argmax agreement is a
real token-for-token decision check, not a distributional one.

Run on QB2:
  DG_RUN_DEVICE=1 pytest models/experimental/diffusion_gemma/tests/test_device_entropy_harness.py -s
"""

import os

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_pcc
from models.experimental.diffusion_gemma.reference import sampling as S
from models.experimental.diffusion_gemma.tt import sampling as TS
from models.experimental.diffusion_gemma.tt import trace_gumbel
from models.experimental.diffusion_gemma.tt.traced_denoise import _trace_capture_guard

pytestmark = [
    pytest.mark.skipif(
        os.environ.get("DG_RUN_DEVICE") != "1",
        reason="set DG_RUN_DEVICE=1 to run on a Tenstorrent device (needs sfpi >= 7.60.0)",
    ),
    pytest.mark.use_module_device,  # one device open/teardown — avoid QB2 erisc cycling
]

_DTYPES = {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b, "fp32": ttnn.float32}
_VOCAB = 2048
_SEQ = 256


def _gen(seed):
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def _varied_logits(seed=1):
    """Logits whose per-position entropy genuinely VARIES (low scale -> high
    entropy, high scale -> low entropy). A flat-entropy input makes PCC/agreement
    ill-conditioned, so vary the per-row temperature deliberately."""
    base = torch.randn(1, _SEQ, _VOCAB, generator=_gen(seed))
    scales = torch.linspace(0.2, 6.0, _SEQ).view(1, _SEQ, 1)
    return base * scales


def _to(t, device, dtype):
    return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)


def test_trace_seeded_uniform_refresh_matches_rand(device):
    """A replay must read the refreshed device seed rather than block-0's captured seed."""
    # Production chunk geometry: 256 canvas positions × 1024 vocab entries.
    shape = (1, 1, 256, 1024)

    def seed_tensor(seed):
        host = torch.zeros((1, 1, 32, 32), dtype=torch.int64)
        host[0, 0, 0, 0] = seed
        return _to(host, device, ttnn.uint32)

    seed = seed_tensor(17)
    output = trace_gumbel.allocate_uniform_buffer(device, shape)
    trace_id = None
    expected_17 = expected_101 = None
    try:
        trace_gumbel.trace_seeded_uniform(seed, output, seed_offset=0)
        ttnn.synchronize_device(device)

        trace_id = ttnn.begin_trace_capture(device, cq_id=0)
        trace_gumbel.trace_seeded_uniform(seed, output, seed_offset=0)
        ttnn.end_trace_capture(device, trace_id, cq_id=0)

        ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
        actual_17 = ttnn.to_torch(output)
        expected_17 = ttnn.rand(
            shape,
            device=device,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            low=0.0,
            high=1.0,
            seed=17,
        )
        assert torch.equal(actual_17, ttnn.to_torch(expected_17))

        fresh = seed_tensor(101)
        ttnn.copy(fresh, seed)
        fresh.deallocate(True)
        ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
        actual_101 = ttnn.to_torch(output)
        expected_101 = ttnn.rand(
            shape,
            device=device,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            low=0.0,
            high=1.0,
            seed=101,
        )
        assert torch.equal(actual_101, ttnn.to_torch(expected_101))
        assert not torch.equal(actual_17, actual_101)
    finally:
        if trace_id is not None:
            ttnn.release_trace(device, trace_id)
        for tensor in (expected_17, expected_101, output, seed):
            if tensor is not None:
                tensor.deallocate(True)


def test_trace_capture_guard_recovers_after_injected_failure(device, expect_error):
    """An exception after begin_trace_capture must not poison the next device command."""
    with expect_error(RuntimeError, match="injected capture failure"):
        with _trace_capture_guard(device, cq_id=0):
            raise RuntimeError("injected capture failure")

    source = _to(torch.ones((1, 1, 32, 32)), device, ttnn.float32)
    destination = ttnn.clone(source)
    trace_id = None
    try:
        ttnn.copy(source, destination)
        ttnn.synchronize_device(device)
        with _trace_capture_guard(device, cq_id=0) as trace_id:
            ttnn.copy(source, destination)
        ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
        assert torch.equal(ttnn.to_torch(destination), torch.ones((1, 1, 32, 32)))
    finally:
        if trace_id is not None:
            ttnn.release_trace(device, trace_id)
        source.deallocate(True)
        destination.deallocate(True)


@pytest.mark.parametrize("temperature", [1.0, 0.6])
def test_token_entropy_bf16_accurate_and_bfp8_degrades(device, temperature):
    """ttnn −Σ p·log p matches torch in bf16; bfp8 is strictly worse (the #47468
    bfp8-drift finding). Uses varied-entropy logits so the metric is meaningful."""
    logits = _varied_logits()
    ref = S.token_entropy(logits, temperature=temperature)  # [1, 256]

    def err(dtype):
        out = ttnn.to_torch(TS.token_entropy(_to(logits, device, dtype), temperature=temperature)).squeeze(-1)
        assert torch.isfinite(out).all(), f"{dtype} entropy produced non-finite values (log(0) guard regressed?)"
        return (ref - out).abs(), comp_pcc(ref, out, 0.0)[1]

    bf16_d, bf16_pcc = err(ttnn.bfloat16)
    bfp8_d, bfp8_pcc = err(ttnn.bfloat8_b)
    print(
        f"\n[entropy T={temperature}] bf16: mean|Δ|={bf16_d.mean():.4f} max|Δ|={bf16_d.max():.3f} PCC={bf16_pcc:.5f}"
        f" | bfp8: mean|Δ|={bfp8_d.mean():.4f} max|Δ|={bfp8_d.max():.3f} PCC={bfp8_pcc:.5f}"
    )
    # bf16 path is accurate (mean abs err ~1% of the ~7.6 range; bound is generous vs measured ~0.09)
    assert bf16_d.mean() < 0.5, f"bf16 entropy mean|Δ|={bf16_d.mean():.4f} too high (expected ~0.09)"
    assert bf16_pcc >= 0.99, f"bf16 entropy PCC={bf16_pcc:.5f} < 0.99"
    # bfp8 is materially degraded — the harness's headline finding (decisions must be validated, not trusted)
    assert bfp8_d.mean() > 2.0 * bf16_d.mean(), "expected bfp8 entropy to be materially worse than bf16"


# Gumbel-max/argmax run on bf16 or fp32 — `ttnn.argmax` rejects bfp8 TILE inputs
# ("Only BFLOAT16, FLOAT32 are supported", assert.hpp). The canvas sampler therefore
# keeps logits at bf16+ for the argmax step (see test_gumbel_max_rejects_bfp8).
@pytest.mark.parametrize("dtype_name", ["bf16", "fp32"])
def test_gumbel_max_argmax_agreement(device, dtype_name):
    """ttnn argmax(logits/T + injected_gumbel) agrees with torch under the SAME noise."""
    dtype = _DTYPES[dtype_name]
    temperature = 0.6
    logits = _varied_logits(seed=2)
    noise = S.sample_gumbel_noise((1, _SEQ, _VOCAB), generator=_gen(3))

    golden = S.gumbel_max_sample(logits, temperature, noise=noise)  # [1, 256] token ids
    out = ttnn.to_torch(TS.gumbel_max(_to(logits, device, dtype), temperature, _to(noise, device, dtype)))
    out = out.squeeze(-1).to(torch.long)

    agreement = float((out == golden).float().mean())
    print(f"\n[gumbel-max {dtype_name}] argmax agreement={agreement:.4f}")
    # ~0.99 measured (bf16); the gap is near-max ties flipping under logit quantization, not an op error.
    assert agreement >= 0.95, f"gumbel-max agreement {agreement:.4f} < 0.95 ({dtype_name})"


@pytest.mark.parametrize("dtype_name", ["bf16", "fp32"])
def test_zero_noise_gumbel_is_argmax(device, dtype_name):
    """noise=0 -> argmax(logits) (temperature preserves argmax); a clean op-level check."""
    dtype = _DTYPES[dtype_name]
    logits = _varied_logits(seed=4)

    golden = logits.argmax(dim=-1)  # [1, 256]
    zero = torch.zeros(1, _SEQ, _VOCAB)
    out = ttnn.to_torch(TS.gumbel_max(_to(logits, device, dtype), 0.8, _to(zero, device, dtype)))
    out = out.squeeze(-1).to(torch.long)

    agreement = float((out == golden).float().mean())
    print(f"\n[zero-noise argmax {dtype_name}] agreement={agreement:.4f}")
    assert agreement >= 0.95, f"zero-noise argmax agreement {agreement:.4f} < 0.95 ({dtype_name})"


def test_chunked_gumbel_max_matches_materialized_chunked_noise(device):
    """Chunked Gumbel-max reduces per-chunk winners without a full noise tensor."""
    temperature = 0.6
    logits = _to(_varied_logits(seed=7), device, ttnn.bfloat16)
    full_noise = TS.sample_gumbel_noise_by_vocab_chunks(logits.shape, device=device, seed=11, vocab_chunk_size=1024)

    expected = TS.gumbel_max(logits, temperature, full_noise)
    out = TS.gumbel_max(logits, temperature, TS.ChunkedGumbelNoise(seed=11, vocab_chunk_size=1024))

    expected_torch = ttnn.to_torch(expected).squeeze(-1).to(torch.long)
    out_torch = ttnn.to_torch(out).squeeze(-1).to(torch.long)
    assert torch.equal(out_torch, expected_torch)

    full_noise.deallocate(True)
    expected.deallocate(True)
    out.deallocate(True)
    logits.deallocate(True)


def test_gumbel_max_rejects_bfp8(device, expect_error):
    """Document the op constraint: `ttnn.argmax` rejects bfp8 TILE inputs, so the
    Gumbel-max/argmax decision step must use bf16+ logits (entropy is fine in bfp8,
    but it shows large drift — see test_token_entropy_*)."""
    logits = _varied_logits(seed=5)
    with expect_error(RuntimeError, match="BFLOAT16, FLOAT32"):
        TS.gumbel_max(
            _to(logits, device, ttnn.bfloat8_b), 0.8, _to(torch.zeros(1, _SEQ, _VOCAB), device, ttnn.bfloat8_b)
        )
