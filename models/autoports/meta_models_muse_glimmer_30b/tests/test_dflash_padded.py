# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""``forward_padded`` must equal ``forward`` -- the path the 13/13 device PCC graded.

This is the gate for the shape-stability optimisation, and it is deliberately an
**old-vs-new** comparison rather than a comparison against HF.  Both sides run the
same weights through the same kernels in the same dtype, so the only thing that can
differ is the padding and the mask; that lets the threshold sit far tighter than the
0.99 used against HF, where bf16-vs-fp32 reduction-order noise sets the floor.

Two properties are checked separately, because they fail for different reasons:

* **Padding invariance** -- padding the context must not change the output.  This is
  what catches a mask that lets pad rows be attended to.  Drafter attention is
  bidirectional, so a leaked pad row corrupts the real slots instead of being
  ignored the way padding is on the target's causal path.
* **Window semantics past 2048** -- ``ctx_4096`` is the only case where the sliding
  window's lower bound does anything at all.  A window regression is *invisible*
  below 2048, and this project has already been burned by a golden that silently ran
  with no window: an unwindowed implementation scored 0.99997 against it while the
  correct one scored 0.9294.

Graded against ``forward``, not ``forward_cached``: only the uncached path is
directly PCC-validated against HF, so it is the sole trustworthy reference here.
"""

from __future__ import annotations

import pytest
import torch
from loguru import logger

import ttnn
from models.autoports.meta_models_muse_glimmer_30b.tests import dflash_checkpoint as R
from models.autoports.meta_models_muse_glimmer_30b.tt.dflash_drafter import (
    DFlashDrafter,
    bidirectional_sliding_mask,
    context_bucket,
)
from models.common.utility_functions import comp_pcc

#: Coarse tripwire for padded-vs-unpadded.  **Not** the real gate -- see
#: ``test_padded_scores_no_worse_against_hf_than_unpadded``, which is.
#:
#: A tighter bound is not available and 0.9999 was tried first: it fails at
#: 0.9978-0.9989 for every genuinely padded case.  The reason is not the padding.
#: ``kv_len = context + 16`` is never a multiple of the 32-row tile, so *both* paths
#: carry tile-padding columns into softmax, at different widths; the two device
#: paths therefore disagree by bf16-scale noise no matter how the mask behaves.
#: Measured against HF, padding costs 0.000000 to -0.000150 PCC -- i.e. nothing --
#: which is what actually establishes correctness.  This threshold only catches a
#: gross regression.
PCC_THRESHOLD = 0.997

#: (real context rows, padded width).  4096 is the only case exceeding the 2048
#: sliding window; 67 is a real prompt length from the device run.
CASES = ((1, 32), (16, 32), (67, 128), (128, 128), (2048, 2048), (4096, 4096))


@pytest.fixture(scope="session")
def drafter(mesh_device):
    """``mesh_device`` comes from ``conftest.py`` -- see there for why it is not local."""
    return DFlashDrafter.from_state_dict(
        R.draft_state_dict(),
        hf_config=R.draft_config(),
        mesh_device=mesh_device,
        weight_dtype=ttnn.bfloat16,
        activation_dtype=ttnn.bfloat16,
    )


def _upload(mesh_device, tensor: torch.Tensor) -> ttnn.Tensor:
    return ttnn.from_torch(
        tensor.reshape(1, 1, *tensor.shape[-2:]).to(torch.bfloat16),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _to_host(mesh_device, tensor: ttnn.Tensor) -> torch.Tensor:
    return ttnn.to_torch(tensor, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[0:1].float()


def _pcc_value(expected: torch.Tensor, actual: torch.Tensor) -> float:
    """The PCC as a float.

    Computed directly rather than scraped out of ``comp_pcc``'s message string,
    whose format is not part of its contract -- this test compares two PCCs
    numerically, so it needs the number rather than a report.
    """
    a = expected.flatten().to(torch.float64)
    b = actual.reshape(expected.shape).flatten().to(torch.float64)
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def test_context_bucket_rounds_up_and_refuses_overflow(expect_error):
    assert context_bucket(1) == 32
    assert context_bucket(32) == 32
    assert context_bucket(33) == 64
    assert context_bucket(2048) == 2048
    # Truncating instead of raising would silently drop the oldest accepted
    # context, which reads as an acceptance-rate collapse rather than as a bug.
    with expect_error(ValueError, "exceeds the largest bucket"):
        context_bucket(2049)


def test_mask_blocks_padding_the_window_bound_would_admit():
    """A pad row at position 0 satisfies ``kv > q - 2048``; only ``kv_valid`` stops it."""
    q_positions = torch.arange(100, 104)
    kv_positions = torch.zeros(8, dtype=torch.long)
    kv_positions[:3] = torch.arange(3)
    kv_valid = torch.zeros(8, dtype=torch.bool)
    kv_valid[:3] = True

    unguarded = bidirectional_sliding_mask(q_positions, kv_positions, 2048, torch.float32)[0, 0]
    guarded = bidirectional_sliding_mask(q_positions, kv_positions, 2048, torch.float32, kv_valid=kv_valid)[0, 0]
    blocked = torch.finfo(torch.float32).min

    assert (unguarded == blocked).sum() == 0, "the window bound alone admits every pad row"
    assert (guarded[:, 3:] == blocked).all(), "kv_valid must block every pad row"
    assert (guarded[:, :3] == 0.0).all(), "real rows must stay unmasked"


@pytest.fixture(scope="session")
def goldens():
    from models.autoports.meta_models_muse_glimmer_30b.tests import reference_dflash as HF

    path = HF.golden_path()
    if not path.exists():  # pragma: no cover - goldens not generated
        pytest.skip(f"missing {path}; run reference_dflash.py first")
    return torch.load(path, weights_only=False)


#: HF-vs-port threshold, matching ``test_dflash_drafter.py``.  bf16-vs-fp32
#: reduction order sets this floor, not the padding.
HF_PCC_THRESHOLD = 0.99

#: How much worse the padded path may score against HF than the unpadded path does.
#: The question that matters is not "is padded bit-identical to unpadded" -- it is
#: "does padding move the port further from the real model".  Both paths already
#: sit at 0.995-0.998 against HF, so a padding bug is only meaningful if it makes
#: that *worse*.
MAX_PCC_REGRESSION_VS_HF = 0.001


#: ``(context_len, padded_width)`` for the HF comparison, chosen so every case is
#: **actually padded** -- ``bucket == context_len`` would make the two paths the same
#: call and score a meaningless 1.0.  ``(2048, 4096)`` is the important one: padded
#: *and* past the sliding window, so it exercises the mask's lower bound and the pad
#: blocking together.  Widths come from the golden set's context lengths.
HF_CASES = ((1, 32), (16, 32), (128, 256), (2048, 4096))


@pytest.mark.parametrize("context_len,bucket", HF_CASES)
def test_padded_scores_no_worse_against_hf_than_unpadded(mesh_device, drafter, goldens, context_len, bucket):
    """The decisive gate: padding must not move the port away from HF.

    ``test_padded_matches_unpadded`` deliberately does not settle this.  The two
    device paths differ by ~0.998 at padded widths, but so does *each* of them
    against HF, because at any context length ``kv_len = context + 16`` is never a
    multiple of the 32-row tile -- both paths therefore carry tile-padding columns
    through softmax. Comparing them to each other cannot tell a padding bug apart
    from that shared pre-existing noise; comparing both to the real model can.
    """
    config = drafter.config
    block = config.block_size
    assert bucket > context_len, "a case where bucket == context_len compares a call with itself"
    inputs = R.synthetic_inputs(context_len=context_len)
    expected = goldens[f"ctx{context_len}"]["outputs"]["last_hidden_state"].float()

    noise = _upload(mesh_device, inputs["noise_embeds"])
    context = _upload(mesh_device, inputs["context_hidden_states"])
    unpadded = _to_host(mesh_device, drafter(noise, context, position_ids=torch.arange(context_len + block)))
    ttnn.deallocate(context)

    padded_host = torch.zeros(1, bucket, config.context_fan_in, dtype=torch.float32)
    padded_host[:, :context_len, :] = inputs["context_hidden_states"].float()
    padded_tt = _upload(mesh_device, padded_host)
    padded = _to_host(
        mesh_device,
        drafter.forward_padded(noise, padded_tt, context_valid=context_len, noise_start=context_len),
    )
    ttnn.deallocate(padded_tt)
    ttnn.deallocate(noise)

    unpadded_value = _pcc_value(expected, unpadded)
    padded_value = _pcc_value(expected, padded)
    logger.info(
        f"ctx={context_len} bucket={bucket}: vs HF unpadded={unpadded_value:.6f} padded={padded_value:.6f} "
        f"delta={padded_value - unpadded_value:+.6f}"
    )

    assert padded_value >= HF_PCC_THRESHOLD, f"padded path fell below the HF gate: {padded_value}"
    assert padded_value >= unpadded_value - MAX_PCC_REGRESSION_VS_HF, (
        f"padding cost {unpadded_value - padded_value:.6f} PCC against HF "
        f"(unpadded {unpadded_value:.6f} -> padded {padded_value:.6f}), "
        f"more than the {MAX_PCC_REGRESSION_VS_HF} allowance"
    )


@pytest.mark.parametrize("context_len,bucket", CASES)
def test_padded_matches_unpadded(mesh_device, drafter, context_len, bucket):
    config = drafter.config
    block = config.block_size
    inputs = R.synthetic_inputs(context_len=context_len)
    noise_host = inputs["noise_embeds"]
    context_host = inputs["context_hidden_states"]

    # Reference: the unpadded, PCC-validated path at the real context length.
    noise = _upload(mesh_device, noise_host)
    context = _upload(mesh_device, context_host)
    position_ids = torch.arange(context_len + block)
    expected = _to_host(mesh_device, drafter(noise, context, position_ids=position_ids))
    ttnn.deallocate(context)

    # Candidate: the same context zero-padded out to the bucket width.
    padded_host = torch.zeros(1, bucket, config.context_fan_in, dtype=torch.float32)
    padded_host[:, :context_len, :] = context_host.float()
    padded = _upload(mesh_device, padded_host)
    actual = _to_host(
        mesh_device,
        drafter.forward_padded(noise, padded, context_valid=context_len, noise_start=context_len),
    )
    ttnn.deallocate(padded)
    ttnn.deallocate(noise)

    passed, message = comp_pcc(expected, actual.reshape(expected.shape), PCC_THRESHOLD)
    logger.info(f"padded ctx={context_len} bucket={bucket}: {message}")
    assert passed, message
