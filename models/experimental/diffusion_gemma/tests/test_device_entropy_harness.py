# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""On-device entropy + Gumbel-max harness validation (#47468).

The #47468 accuracy harness must extend PCC **beyond logits** to the diffusion
*decisions*: per-position **entropy** (−Σ p·log p) and **Gumbel-max argmax
agreement**, validated device-vs-torch and **especially under bfp8** (small-
probability drift can flip accept/renoise). gemma4 has no entropy op, so these
ttnn primitives (`tt/sampling.py`) are net-new; here we diff them against the
pure-torch oracle (`reference/sampling.py`) at the op level on QB2.

Determinism: Gumbel noise is generated in torch and **injected** into both paths
(on-device RNG won't reproduce torch's RNG bit-exactly), so argmax agreement is a
real token-for-token decision check, not a distributional one.

Run on QB2:
  DG_RUN_DEVICE=1 pytest models/experimental/diffusion_gemma/tests/test_device_entropy_harness.py
"""

import os

import pytest
import torch

import ttnn
from models.experimental.diffusion_gemma.reference import sampling as S
from models.experimental.diffusion_gemma.tt import sampling as TS
from tests.ttnn.utils_for_testing import assert_with_pcc

pytestmark = [
    pytest.mark.skipif(
        os.environ.get("DG_RUN_DEVICE") != "1",
        reason="set DG_RUN_DEVICE=1 to run on a Tenstorrent device (needs sfpi >= 7.60.0)",
    ),
    pytest.mark.use_module_device,  # one device open/teardown — avoid QB2 erisc cycling
]

_DTYPES = {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b}


def _gen(seed):
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def _to(t, device, dtype):
    return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)


@pytest.mark.parametrize("dtype_name", ["bf16", "bfp8"])
@pytest.mark.parametrize("vocab", [2048])
@pytest.mark.parametrize("temperature", [1.0, 0.6])
def test_token_entropy_matches_reference(device, dtype_name, vocab, temperature):
    """ttnn −Σ p·log p == torch reference entropy. bfp8 gets a looser bar (the point)."""
    dtype = _DTYPES[dtype_name]
    logits = torch.randn(1, 256, vocab, generator=_gen(1))

    golden = S.token_entropy(logits, temperature=temperature)  # [1, 256]
    out = ttnn.to_torch(TS.token_entropy(_to(logits, device, dtype), temperature=temperature)).squeeze(-1)  # [1,256]

    pcc = 0.999 if dtype_name == "bf16" else 0.99  # bfp8 mantissa drift
    assert_with_pcc(golden, out, pcc)


@pytest.mark.parametrize("dtype_name", ["bf16", "bfp8"])
@pytest.mark.parametrize("vocab", [2048])
def test_gumbel_max_argmax_agreement(device, dtype_name, vocab):
    """ttnn argmax(logits/T + injected_gumbel) agrees with torch under the SAME noise."""
    dtype = _DTYPES[dtype_name]
    temperature = 0.6
    logits = torch.randn(1, 256, vocab, generator=_gen(2))
    noise = S.sample_gumbel_noise((1, 256, vocab), generator=_gen(3))

    golden = S.gumbel_max_sample(logits, temperature, noise=noise)  # [1, 256] token ids
    out = ttnn.to_torch(TS.gumbel_max(_to(logits, device, dtype), temperature, _to(noise, device, dtype)))
    out = out.squeeze(-1).to(torch.long)  # [1, 256]

    agreement = float((out == golden).float().mean())
    # bf16: near-exact under injected noise; bfp8: small-probability drift can flip a few.
    bar = 0.99 if dtype_name == "bf16" else 0.90
    assert agreement >= bar, f"gumbel-max argmax agreement {agreement:.4f} < {bar} ({dtype_name})"


@pytest.mark.parametrize("dtype_name", ["bf16", "bfp8"])
def test_zero_noise_gumbel_is_argmax(device, dtype_name):
    """noise=0 -> argmax(logits) (temperature preserves argmax); a clean op-level check."""
    dtype = _DTYPES[dtype_name]
    vocab = 2048
    logits = torch.randn(1, 256, vocab, generator=_gen(4))

    golden = logits.argmax(dim=-1)  # [1, 256]
    zero = torch.zeros(1, 256, vocab)
    out = ttnn.to_torch(TS.gumbel_max(_to(logits, device, dtype), 0.8, _to(zero, device, dtype)))
    out = out.squeeze(-1).to(torch.long)

    agreement = float((out == golden).float().mean())
    bar = 0.99 if dtype_name == "bf16" else 0.95
    assert agreement >= bar, f"zero-noise argmax agreement {agreement:.4f} < {bar} ({dtype_name})"
