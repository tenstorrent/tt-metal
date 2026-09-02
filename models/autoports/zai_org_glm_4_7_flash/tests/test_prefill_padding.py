# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Prefill-length bucketing must be invisible to the model output.

The full-model stage pads the *physical* prefill length to a small bucket set
so the number of compiled prefill shapes stays bounded (see
``doc/full_model/work_log.md`` FM-006). That is only legitimate if the padded
positions can never influence a real one, and if the cache rows they wrote can
never be attended later.

This runs on the reduced 2-layer probe (HF layer 0 dense + layer 1 moe, real
embedding / norm / LM head / paged cache), so two models with different
bucketing fit on the card at once and the suite runs in well under a minute.

    pytest models/autoports/zai_org_glm_4_7_flash/tests/test_prefill_padding.py -q -s
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

import ttnn
from models.autoports.zai_org_glm_4_7_flash.tt.generator import GLM47FlashGenerator
from models.autoports.zai_org_glm_4_7_flash.tt.model import GLM47FlashModel

MODEL_DIR = Path(__file__).resolve().parents[1]
PROBE_LAYERS = [0, 1]
PROBE_SEQ_LEN = 8192


@pytest.fixture(scope="module")
def device():
    dev = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1), l1_small_size=32768, trace_region_size=350_000_000)
    yield dev
    ttnn.close_mesh_device(dev)


def _build(dev, buckets):
    return GLM47FlashModel.from_pretrained(
        dev,
        max_batch_size=1,
        max_seq_len=PROBE_SEQ_LEN,
        layer_indices=PROBE_LAYERS,
        prefill_buckets=buckets,
    )


@pytest.fixture(scope="module")
def models(device):
    bucketed = _build(device, (128, 256, 512, 1024, 2048))
    unbucketed = _build(device, ())  # pad only to the 64-token paged block
    yield bucketed, unbucketed


@pytest.mark.parametrize("seq", [17, 65, 129, 154, 700, 2049, 2600])
def test_bucketed_prefill_matches_block_aligned_prefill(models, seq):
    """Padding to a bucket changes nothing a real position can see."""
    bucketed, unbucketed = models
    assert bucketed.prefill_physical_len(seq) >= unbucketed.prefill_physical_len(seq)
    ids = _ids(seq)

    out = []
    for model in (bucketed, unbucketed):
        cache = model.allocate_kv_cache()
        pt = model.page_table_to_device(model.default_page_table())
        out.append(model.prefill_forward(ids, kv_cache=cache, page_table=pt, seq_len=seq, return_all_logits=True))
        for tensor in cache:
            ttnn.deallocate(tensor)
        ttnn.deallocate(pt)

    a, b = out
    assert a.shape == b.shape == (1, seq, bucketed.vocab_size)
    disagree = (a[0].argmax(-1) != b[0].argmax(-1)).nonzero().flatten().tolist()
    pcc = _pcc(a, b)
    print(
        f"seq={seq} physical bucketed={bucketed.prefill_physical_len(seq)} "
        f"block-aligned={unbucketed.prefill_physical_len(seq)} "
        f"argmax disagreements={len(disagree)}/{seq} pcc={pcc:.8f}"
    )
    assert pcc > 0.9999, pcc
    # A different physical prefill length is a different matmul M and a
    # different flash-prefill K extent, so the two runs accumulate in a
    # different order. Every argmax disagreement must therefore be a bf16
    # near-tie between the same two candidates, not a real change of answer.
    # (The sharp non-leakage proof is test_pad_token_value_* below, which holds
    # the shape fixed and gets bit-identical logits.)
    for pos in disagree:
        ta, tb = int(a[0, pos].argmax()), int(b[0, pos].argmax())
        gap_a = float(a[0, pos, ta] - a[0, pos, tb])
        gap_b = float(b[0, pos, tb] - b[0, pos, ta])
        ulp = _bf16_ulp(max(abs(float(a[0, pos, ta])), abs(float(b[0, pos, tb]))))
        print(
            f"  pos {pos}: {ta} vs {tb}, gaps {gap_a:.4g} / {gap_b:.4g} = {gap_a / ulp:.2f} / {gap_b / ulp:.2f} bf16 ULP"
        )
        assert gap_a <= 4 * ulp and gap_b <= 4 * ulp, (
            f"position {pos} flipped between two candidates {gap_a / ulp:.1f}/{gap_b / ulp:.1f} bf16 ULP apart, "
            "which is not a tie"
        )


@pytest.mark.parametrize("seq", [154, 2049, 2600])
def test_pad_token_value_cannot_reach_a_real_position(models, seq):
    """The sharp leakage test: same prompt, same physical prefill length, two
    completely different pad token values. If a padded position could reach a
    real one, the logits would move; they must be bit-identical.

    Covers single-chunk (154 -> 256) and multi-chunk prefill (2049 -> 2048+128,
    2600 -> 2048+1024), where the bucketed tail meets the chunk-offset-dependent
    RoPE slice, ``chunked_flash_mla_prefill(chunk_start_idx=...)`` and the
    per-chunk ``paged_fill_cache`` page-table slice."""
    bucketed, _ = models
    ids = _ids(seq)
    assert bucketed.prefill_physical_len(seq) > seq
    original = bucketed.pad_token_id
    out = []
    try:
        for pad in (0, 12345):
            bucketed.pad_token_id = pad
            cache = bucketed.allocate_kv_cache()
            pt = bucketed.page_table_to_device(bucketed.default_page_table())
            out.append(
                bucketed.prefill_forward(ids, kv_cache=cache, page_table=pt, seq_len=seq, return_all_logits=True)
            )
            for tensor in cache:
                ttnn.deallocate(tensor)
            ttnn.deallocate(pt)
    finally:
        bucketed.pad_token_id = original
    a, b = out
    print(f"pad-token invariance: max |delta| = {float((a - b).abs().max()):.3e}")
    assert torch.equal(a, b), "the pad token value changed a real position's logits"


def test_padded_cache_rows_cannot_reach_decode(models):
    """The pad rows a bucketed prefill wrote into the KV cache can never be
    attended by a later decode step: each step writes its own row before
    reading, so decode past the end of the prompt is invariant to the pad
    token value at a fixed physical length."""
    bucketed, _ = models
    seq = 2600  # physical 3072: a multi-chunk prefill with 472 pad rows in the tail chunk
    ids = _ids(seq)
    original = bucketed.pad_token_id
    runs = []
    try:
        for pad in (0, 12345):
            bucketed.pad_token_id = pad
            gen = GLM47FlashGenerator(bucketed)
            gen._ensure_owned_state()
            gen.capture_decode_trace()
            gen.reset()
            runs.append(gen.generate(ids, 24, enable_trace=True, stop_on_eos=False))
            gen.teardown()
            for tensor in gen._kv_cache:
                ttnn.deallocate(tensor)
            gen._kv_cache = None
    finally:
        bucketed.pad_token_id = original
    print("pad=0     :", runs[0])
    print("pad=12345 :", runs[1])
    assert runs[0] == runs[1], "decode saw the pad rows the bucketed prefill wrote"


def _ids(seq):
    from transformers import AutoTokenizer

    from models.autoports.zai_org_glm_4_7_flash.tt.model import resolve_checkpoint_dir

    tok = AutoTokenizer.from_pretrained(str(resolve_checkpoint_dir()), local_files_only=True)
    text = (
        "Tenstorrent builds AI accelerators. The prefill padding test needs an ordinary "
        "in-distribution prompt of a controllable length. "
    ) * 400
    ids = tok.encode(text, add_special_tokens=True)
    while len(ids) < seq:
        ids = ids + ids
    return ids[:seq]


def _bf16_ulp(x):
    """Spacing between adjacent bfloat16 values at magnitude ``x`` (8-bit mantissa)."""
    import math

    if x == 0:
        return 2.0**-133
    return 2.0 ** (math.floor(math.log2(abs(x))) - 7)


def _pcc(a, b):
    x = a.flatten().double()
    y = b.flatten().double()
    x = x - x.mean()
    y = y - y.mean()
    return float((x @ y) / (x.norm() * y.norm()))
