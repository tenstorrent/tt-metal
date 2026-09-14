# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Does ttnn SDPA attend to a key sequence's TILE PADDING when the logical length is not a multiple of 32?

ANSWER, MEASURED 2026-09-14 (T3K, bf16, 8 heads, head_dim 128, 16 causal queries): **YES, it does.**

    kv_len  pad rows   pcc        |q| ratio row 0        |q| ratio row 15
        32         0   0.999847   0.9647               0.9845          (control: no padding)
        16        16   0.820283   0.0602  (pred 1/17)  0.4975  (pred 16/32)
        23         9   0.990719   0.4594  (pred 8/17)  0.7070  (pred 23/32)

The measured ratios track the DILUTION PREDICTION to within a percent at every point. A causal query
row that sees ``r`` real keys alongside ``p`` attended pad rows should retain ``r / (r + p)`` of its
magnitude, because the pad rows carry zero V (contributing nothing to the numerator) but still take
softmax weight (inflating the denominator). Row 0 of the kv_len=16 case sees ONE real key against 16
pad rows and retains 6 % of its magnitude. This is not a rounding difference; it is a different
function, and it is worst exactly where the causal mask is tightest.

The mechanism is that the mask cannot disable them: a ``[1, 1, 16, 23]`` additive mask is itself
tile-padded out to 32 columns with ZEROS, and zero in an additive mask means VISIBLE.

WHY THIS MATTERED
-----------------
Found while converting the DFlash drafter to fixed-capacity KV. The drafter attends over
``hist_len + new_ctx + q_len`` keys -- 16, 23, 39, 42, 53 in a normal run, almost never a multiple
of 32 -- so it has been running diluted attention on nearly every step. The fixed-capacity path is
immune by construction (``C + 32`` is always tile-aligned and every non-real column carries an
explicit mask entry), which is why the two paths disagreed and why the ONE step that agreed at PCC
1.0 was the one where the legacy length happened to land on 32.

No emitted token was ever wrong: greedy speculation lets the target decide every token, so a weaker
drafter costs acceptance and nothing else. The target itself is unaffected -- it runs 128-row
buckets, always tile-aligned.

WHAT IT COSTS, AND WHY THE DRAFTER WAS NOT FIXED
------------------------------------------------
Nothing, measured. tests/reference/test_dflash_acceptance_capacity.py ran the diluted path against
the immune one on the full 27B: **5.182 vs 5.182 tok/step, identical on every prompt.** The reason
is RMSNorm. The dilution is close to a per-row SCALE factor (a row keeps ``r / (r + p)`` of its
magnitude), RMSNorm is per-row scale-invariant, and the drafter re-norms after every sublayer and
once more before the LM head. PCC sees it because PCC compares magnitudes; argmax, which is all
acceptance depends on, does not.

And the cheap fix does not exist: widening the mask to the tile-aligned extent is REJECTED by the op
(``TT_FATAL ... mask_shape[3] == k_shape[...]``, test_widening_the_mask_fixes_it below). SDPA
validates the mask against the key tensor's LOGICAL width while reading its PADDED extent -- which
is precisely the inconsistency that creates the bug. Fixing the growing-history path would therefore
mean padding K and V themselves, a concat per layer per step, to buy zero acceptance. So it was left
alone, deliberately, and documented here instead.

The padded cases below are marked xfail because they document a defect in the primitive rather than
a regression in this repo. An xpass means SDPA started respecting logical shapes, and the drafter's
masking can then be revisited.

This is a question about the primitive, asked because of what it would mean for the drafter. The
DFlash drafter builds its attention over ``hist_len + new_ctx + q_len`` keys, a number that is
almost never a multiple of 32 -- 16, 23, 39, 42, 53 in a normal run. If SDPA reads the PADDED
extent rather than the logical one, then each such step attends to (32 - len % 32) extra key rows
whose K and V are zero and whose mask entries are the mask tensor's OWN tile padding, which is
zero, i.e. "visible" in an additive mask.

Those rows would contribute no value but would still take softmax weight, diluting the real weights
-- and diluting them UNEQUALLY, because a causal row that sees 1 real key is diluted far more than
one that sees 16. That is not a rounding difference; it changes the function.

The test is a direct comparison against a torch reference over the LOGICAL keys only:

* ``len 32`` is the control -- no padding exists, so agreement is expected whatever the answer is.
* ``len 16`` / ``len 23`` are the cases with padding. Agreement means SDPA respects logical shapes
  and padding is inert. Disagreement in the direction of DILUTION (device outputs systematically
  smaller than reference, worst for the earliest query rows) means it does not.

Run::

    MESH_DEVICE=T3K pytest -svq models/demos/blackhole/qwen36/tests/unit/test_sdpa_tile_padding.py
"""

from __future__ import annotations

import os

import pytest
import torch
from loguru import logger

import ttnn

NH, HD, Q_LEN = 8, 128, 16


def _mesh_shape():
    name = (os.environ.get("MESH_DEVICE") or "").upper()
    return {"P150": (1, 1), "N150": (1, 1), "N300": (1, 2), "T3K": (1, 8)}.get(name, (1, 8))


MESH_SHAPE = _mesh_shape()


def _reference(q, k, v, mask):
    """Plain torch SDPA over the logical keys."""
    logits = (q.float() @ k.float().transpose(-1, -2)) * (HD**-0.5) + mask.float()
    return torch.softmax(logits, dim=-1) @ v.float()


@pytest.mark.parametrize(
    "kv_len",
    [
        32,
        pytest.param(
            16, marks=pytest.mark.xfail(reason="SDPA attends tile padding; see module docstring", strict=False)
        ),
        pytest.param(
            23, marks=pytest.mark.xfail(reason="SDPA attends tile padding; see module docstring", strict=False)
        ),
    ],
    ids=lambda n: f"len{n}",
)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
def test_sdpa_ignores_tile_padding(mesh_device, kv_len, reset_seeds, ensure_gc):
    """Compare device SDPA against a torch reference computed over the logical keys only."""
    g = torch.Generator().manual_seed(4)
    q = (torch.randn(1, NH, Q_LEN, HD, generator=g) * 0.3).to(torch.bfloat16)
    k = (torch.randn(1, NH, kv_len, HD, generator=g) * 0.3).to(torch.bfloat16)
    v = (torch.randn(1, NH, kv_len, HD, generator=g) * 0.3).to(torch.bfloat16)

    # Causal over the trailing q_len positions, exactly as the drafter builds it.
    q_pos = torch.arange(kv_len - Q_LEN, kv_len).unsqueeze(1)
    k_pos = torch.arange(kv_len).unsqueeze(0)
    mask = torch.where(k_pos <= q_pos, 0.0, float("-inf")).reshape(1, 1, Q_LEN, kv_len).to(torch.bfloat16)

    def _dev(t, layout=ttnn.TILE_LAYOUT):
        return ttnn.from_torch(
            t,
            dtype=ttnn.bfloat16,
            layout=layout,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    out = ttnn.transformer.scaled_dot_product_attention(
        _dev(q),
        _dev(k),
        _dev(v),
        attn_mask=_dev(mask),
        is_causal=False,
        scale=HD**-0.5,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    got = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[:1].float()
    want = _reference(q, k, v, mask)

    # Dilution shows up as a systematic magnitude deficit, worst on the earliest query row (which
    # sees the fewest real keys and so is diluted hardest). Report it either way.
    ratio = (got.abs().mean(dim=-1) / want.abs().mean(dim=-1).clamp_min(1e-6))[0, 0]
    from models.common.utility_functions import comp_pcc

    _, pcc = comp_pcc(want, got, 0.99)
    value = float(str(pcc).split()[-1]) if not isinstance(pcc, float) else pcc
    pad = (-kv_len) % 32
    logger.info(
        f"kv_len={kv_len:3d} ({pad:2d} tile-pad rows): pcc {value:.6f}  "
        f"magnitude ratio row0 {ratio[0]:.4f} row{Q_LEN - 1} {ratio[-1]:.4f}"
    )
    assert value > 0.99, (
        f"ttnn SDPA disagrees with a logical-keys reference at kv_len={kv_len} ({pad} tile-pad "
        f"rows): pcc {value:.6f}, magnitude ratio row0 {ratio[0]:.4f}. A ratio well below 1 that is "
        "worst on row 0 means the tile padding is being attended and is diluting the softmax"
    )


@pytest.mark.parametrize("kv_len", [16, 23], ids=lambda n: f"len{n}")
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
def test_widening_the_mask_fixes_it(mesh_device, kv_len, reset_seeds, ensure_gc):
    """THE FIX: build the mask at the 32-ALIGNED width so the pad columns get a real mask entry.

    If SDPA is going to read the padded extent regardless, then the mask has to cover it. The
    question is whether ttnn accepts a mask WIDER than the key tensor's logical length -- if it
    does, this is a two-line fix for the drafter's sliding mask and costs nothing; if it rejects the
    shape, the fix has to pad K and V themselves, which costs a concat per layer.

    Either answer is useful, so a rejection is reported rather than swallowed.
    """
    from models.common.utility_functions import comp_pcc

    g = torch.Generator().manual_seed(4)
    q = (torch.randn(1, NH, Q_LEN, HD, generator=g) * 0.3).to(torch.bfloat16)
    k = (torch.randn(1, NH, kv_len, HD, generator=g) * 0.3).to(torch.bfloat16)
    v = (torch.randn(1, NH, kv_len, HD, generator=g) * 0.3).to(torch.bfloat16)

    q_pos = torch.arange(kv_len - Q_LEN, kv_len).unsqueeze(1)
    k_pos = torch.arange(kv_len).unsqueeze(0)
    visible = k_pos <= q_pos
    narrow = torch.where(visible, 0.0, float("-inf")).reshape(1, 1, Q_LEN, kv_len)

    # Same mask, widened to the tile-aligned extent, with every pad column explicitly invisible.
    kv_pad = -(-kv_len // 32) * 32
    wide = torch.full((1, 1, Q_LEN, kv_pad), float("-inf"))
    wide[..., :kv_len] = narrow

    def _dev(t):
        return ttnn.from_torch(
            t.to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    try:
        out = ttnn.transformer.scaled_dot_product_attention(
            _dev(q),
            _dev(k),
            _dev(v),
            attn_mask=_dev(wide),
            is_causal=False,
            scale=HD**-0.5,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    except Exception as e:  # noqa: BLE001 -- a shape rejection is a result, not a failure
        pytest.skip(
            f"ttnn rejected a mask wider than the key length: {type(e).__name__}: {str(e).splitlines()[0][:140]}"
        )

    got = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[:1].float()
    want = _reference(q, k, v, narrow)
    ratio = (got.abs().mean(dim=-1) / want.abs().mean(dim=-1).clamp_min(1e-6))[0, 0]
    _, pcc = comp_pcc(want, got, 0.99)
    value = float(str(pcc).split()[-1]) if not isinstance(pcc, float) else pcc
    logger.info(
        f"kv_len={kv_len:3d} with a {kv_pad}-wide mask: pcc {value:.6f}  "
        f"magnitude ratio row0 {ratio[0]:.4f} row{Q_LEN - 1} {ratio[-1]:.4f}"
    )
    assert value > 0.99, (
        f"widening the mask to {kv_pad} did not restore agreement (pcc {value:.6f}, row0 ratio "
        f"{ratio[0]:.4f}); the pad columns are being attended even with an explicit mask entry, so "
        "the fix has to pad K and V themselves"
    )
