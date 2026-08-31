# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Prefill input-sequence-length (ISL) sweep PCC for one Gemma4 decoder layer.

One decoder layer, real checkpoint weights, shipped dtypes, paged KV cache, run at
each ISL in ``PREFILL_ISL_SWEEP`` and compared against the HuggingFace
``Gemma4TextDecoderLayer`` token by token.

Why a sweep and not a single length: the prefill path changes shape with the ISL.

* ``32 … 1024`` sit inside the 1024 sliding window, so a sliding layer's SDPA is
  effectively dense there;
* ``2048+`` cross it, and the window masking has to match HF's mask exactly;
* above ``TT_PREFILL_CHUNK`` (the demo's 4096) the call becomes generator-style
  multi-chunk — ``chunk_start_idx``, a per-chunk page table, and on a sliding layer
  the ``_sliding_prefill_tail`` carried from the previous chunk. ``6144`` is the
  non-power-of-two case (3 x 2048, not 4096 + partial).

**This file opts out of ``--max-prefill``.** The shared ``_enforce_max_prefill``
fixture (``tests/conftest.py``) skips any node whose ``seq_len`` exceeds the flag,
default 8192; the module-local override below disables that here, so the sweep
always runs its declared range up to the checkpoints' ``max_position_embeddings``
(256K) no matter what the flag says. The long rows are bounded by the *HF
reference*, not by TT — see ``PREFILL_ISL_SWEEP`` below for the measured cost and
for the two things those rows do not cover (bounded sliding KV, and the WH T3K ISL
ceiling).

The full range is hours per model (~5 h on 31B, most of it 256K), so select
explicitly for a short run — ``-k`` is the only gate now::

    HF_MODEL=google/gemma-4-31B-it MESH_DEVICE=1x8 \\
      pytest models/demos/gemma4/tests/unit/test_prefill.py -k "1x8" -v   # ALL 15 lengths
    # short run: name the lengths you want
    ... -k "1x8 and (isl128 or isl1024 or isl4096)"
    # or the sanity test, which is two lengths by construction
    ... ::test_prefill_isl_sanity -k "1x8"
"""

from __future__ import annotations

import os

import pytest
import torch
from loguru import logger
from transformers.cache_utils import DynamicCache

from models.demos.gemma4.tt.generator_trace import GEMMA4_DEFAULT_PREFILL_CHUNK
from models.tt_transformers.tt.common import get_max_prefill_chunk_size

from ...tests.test_factory import get_pcc_threshold, parametrize_mesh_with_fabric, uses_ci_config_only_checkpoint
from .decoder_pcc_common import (
    KV_PCC_REQUIRED,
    PCC_BATCH_SIZE,
    build_decoder_pcc_context,
    check_pcc,
    compare_kv_cache,
    hf_forward_span,
    tt_prefill_chunk,
)

# Skipped at *collection* time, not inside the test body: the CI unit job points
# HF_MODEL at a checked-in config stub with no safetensors, and skipping per-test
# would still pay one mesh setup/teardown per parametrized node.
pytestmark = pytest.mark.skipif(
    uses_ci_config_only_checkpoint(),
    reason="Decoder PCC needs the checkpoint's real weights; HF_MODEL is a config-only stub",
)

# TT prefill chunk. The demo's default; a call below it is single-chunk, above it
# the sweep drives the generator's multi-chunk path (``chunk_start_idx`` +
# per-chunk page table).
TT_PREFILL_CHUNK = int(os.environ.get("GEMMA4_DECODER_PCC_TT_CHUNK", str(GEMMA4_DEFAULT_PREFILL_CHUNK)))

# HF reference chunk. Bounds the eager score tensor to
# ``num_heads * HF_PREFILL_CHUNK * isl`` elements (fp32): ~2 GB at 512 x 32K on
# 31B. Lower it if the host is tight on RAM.
HF_PREFILL_CHUNK = int(os.environ.get("GEMMA4_DECODER_PCC_HF_CHUNK", "512"))

# Input-sequence-length sweep, up to the checkpoints' ``max_position_embeddings``
# (256K on both 12B and 31B). Below 2048 the lengths bracket the 1024 sliding
# window (32 … 1024 fit inside it, 2048+ do not); 6144 is the non-power-of-two
# multi-chunk case (chunk 2048 x 3, not 4096).
#
# Every length here runs: the module-local ``_enforce_max_prefill`` override below
# disables the shared ``--max-prefill`` auto-skip for this file, so adding a length
# to this list adds it to the default run.
#
# **The long rows are bounded by the HF reference, not by TT.** Eager attention
# materializes a ``[heads, rows, keys]`` score tensor, so the reference forward
# grows superlinearly in the ISL while the device side stays linear. Measured on
# 12B / 1x8 (sliding layer, one node, ~3x per doubling): 50 s at 32K, 146 s at
# 64K, 485 s at 128K, ~32 min at 256K. The score tensor is
# ``heads * HF_PREFILL_CHUNK * isl * 4`` bytes (16 GB at 512 x 256K on 31B) and
# ``GEMMA4_DECODER_PCC_HF_CHUNK`` trades it for more Python overhead — but it is
# not the whole footprint: the fp32 hidden, its bf16 copy, the per-span output
# list plus its ``torch.cat``, and the fp32 ``DynamicCache`` add ~26 GB at 256K
# on 31B and do *not* shrink with the chunk.
#
# Two things the long rows do *not* cover: the demo auto-enables **bounded sliding
# KV** above its per-(model, device) cutover (``GEMMA4_LONG_CONTEXT_POLICY``) while
# this fixture always builds the unbounded paged cache; and on WH T3K the 31B
# ceiling is 16k unbounded / 32k bounded, so 64K+ there is past what the demo path
# itself is validated for.
PREFILL_ISL_SWEEP = [
    32,
    64,
    128,
    256,
    512,
    1024,
    2048,
    4096,
    6144,
    8192,
    16384,
    32768,
    65536,
    131072,
    262144,
]

# Short gate for CI / smoke runs: one length inside the sliding window, one at it.
PREFILL_ISL_SANITY = [128, 1024]


@pytest.fixture(autouse=True)
def _enforce_max_prefill():
    """Disable the shared ``--max-prefill`` auto-skip for this module.

    ``tests/conftest.py`` defines an autouse fixture of this name that skips any
    node whose ``seq_len`` param exceeds ``--max-prefill`` (default 8192). A
    same-named fixture defined in the module shadows it, so declaring this no-op
    opts the whole file out: the ISL sweep always runs its full declared range,
    32 … 262144, whatever the flag is set to.

    The point is that an ISL sweep's *coverage* should be a property of the list
    it declares, not of a flag someone forgot to pass — a silently skipped 256K
    row reads exactly like a passing one in a summary.
    """
    return


def tt_prefill_chunk_size(seq_len: int) -> int:
    """Chunk the TT prefill the way the generator would for this ISL.

    Single call at or below the demo chunk; above it, the largest multiple of 2048
    that divides ``seq_len`` (``get_max_prefill_chunk_size``), so every chunk is
    full and no partial-chunk padding enters the comparison.
    """
    if seq_len <= TT_PREFILL_CHUNK:
        return seq_len
    return get_max_prefill_chunk_size(seq_len, TT_PREFILL_CHUNK)


def _hf_reference_outputs(ctx, hidden: torch.Tensor, cache: DynamicCache) -> torch.Tensor:
    """HF layer output for the whole sequence, computed ``HF_PREFILL_CHUNK`` rows at a time.

    Chunking here is a host-memory bound, not a modelling choice: eager attention
    materializes ``[heads, rows, keys]`` scores, which a full-ISL forward cannot
    afford at 16K+. The ``DynamicCache`` plus the explicit causal/sliding mask make
    the pieces add up to the same forward, and the chunk size is independent of
    TT's.
    """
    seq_len = hidden.shape[1]
    outputs = []
    for start in range(0, seq_len, HF_PREFILL_CHUNK):
        end = min(start + HF_PREFILL_CHUNK, seq_len)
        outputs.append(hf_forward_span(ctx, hidden[:, start:end], start_pos=start, cache=cache))
    return torch.cat(outputs, dim=1)


def _run_prefill_isl(mesh_device, request, *, layer_type: str, seq_len: int) -> None:
    threshold = get_pcc_threshold(request)
    # Fresh context per length: the paged cache and the sliding tail must start empty.
    ctx = build_decoder_pcc_context(mesh_device, layer_type, max_seq_len=max(seq_len, 128))
    chunk = tt_prefill_chunk_size(seq_len)

    logger.info(
        "Prefill ISL sweep: layer={} ({}), seq_len={}, tt_chunk={} ({} call(s)), hf_chunk={}, pcc>={}",
        ctx.layer_idx,
        layer_type,
        seq_len,
        chunk,
        seq_len // chunk,
        HF_PREFILL_CHUNK,
        threshold,
    )

    hidden = torch.randn(PCC_BATCH_SIZE, seq_len, ctx.hidden_size, dtype=torch.float32)

    hf_cache = DynamicCache()
    hf_out = _hf_reference_outputs(ctx, hidden, hf_cache)

    hidden_bf16 = hidden.to(torch.bfloat16)
    failures = []
    min_pcc = 1.0
    for chunk_start in range(0, seq_len, chunk):
        tt_chunk_out = tt_prefill_chunk(
            ctx, mesh_device, hidden_bf16[:, chunk_start : chunk_start + chunk], chunk_start=chunk_start
        )
        label = f"prefill seq_len={seq_len} chunk=[{chunk_start}:{chunk_start + chunk})"
        passing, pcc = check_pcc(label, hf_out[:, chunk_start : chunk_start + chunk], tt_chunk_out, threshold)
        min_pcc = min(min_pcc, pcc)
        if not passing:
            failures.append(f"output chunk@{chunk_start} pcc={pcc:.6f}")

    # The KV the prefill wrote is what the following decode reads, so gate it here
    # too — sampled positions rather than the whole cache, including the last one
    # (the row a bounded/partial-block fill is most likely to miss).
    sampled = sorted({0, 1, seq_len // 2, seq_len - 1})
    (tt_k, hf_k), (tt_v, hf_v) = compare_kv_cache(ctx, hf_cache, sampled, written_through=seq_len)
    k_pass, k_pcc = check_pcc(f"kv K at positions {sampled}", hf_k, tt_k, KV_PCC_REQUIRED)
    v_pass, v_pcc = check_pcc(f"kv V at positions {sampled}", hf_v, tt_v, KV_PCC_REQUIRED)
    if not k_pass:
        failures.append(f"K cache pcc={k_pcc:.6f}")
    if not v_pass:
        failures.append(f"V cache pcc={v_pcc:.6f}")

    logger.info("Prefill seq_len={} min chunk PCC {:.6f}", seq_len, min_pcc)
    assert not failures, (
        f"Prefill ISL sweep failed for layer {ctx.layer_idx} ({layer_type}), "
        f"seq_len={seq_len}, tp={ctx.tp}: " + "; ".join(failures)
    )


@parametrize_mesh_with_fabric()
@pytest.mark.parametrize("layer_type", ["sliding_attention", "full_attention"], ids=["sliding", "global"])
@pytest.mark.parametrize("seq_len", PREFILL_ISL_SANITY, ids=lambda n: f"isl{n}")
def test_prefill_isl_sanity(layer_type, seq_len, mesh_device, reset_seeds, request):
    """Short ISL gate (128, 1024): one prefill per length, both layer types."""
    _run_prefill_isl(mesh_device, request, layer_type=layer_type, seq_len=seq_len)


@pytest.mark.slow
@parametrize_mesh_with_fabric()
@pytest.mark.parametrize("layer_type", ["sliding_attention", "full_attention"], ids=["sliding", "global"])
@pytest.mark.parametrize("seq_len", PREFILL_ISL_SWEEP, ids=lambda n: f"isl{n}")
@pytest.mark.timeout(0)
def test_prefill_isl_sweep(layer_type, seq_len, mesh_device, reset_seeds, request):
    """Full ISL sweep (32 … 262144, never ``--max-prefill``-gated), single- and multi-chunk."""
    _run_prefill_isl(mesh_device, request, layer_type=layer_type, seq_len=seq_len)
