# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""A resumed prefill must be indistinguishable from a single-pass one.

``GLM47FlashForCausalLM.prefill_forward`` used to refuse any non-zero
``start_pos``, citing an MLA-family limitation shared with the DeepSeek-V3 TT
adapter. That was accurate about the behaviour and wrong about the cause: the
per-chunk primitive was *already* position-aware.
``FusedDecoder._attn_prefill_chunk`` takes a ``chunk_start``, builds its RoPE
matrices for ``[chunk_start, chunk_start + S_c)``, slices the page table to that
chunk's blocks, writes with ``paged_fill_cache`` at those blocks, and passes
``chunk_start_idx`` into attention. The model already chunks every long prompt at
``prefill_chunk_size``. The only thing it could not do was *begin* anywhere but
zero, because the driver loop was hardcoded to:

    for chunk_idx, start in enumerate(range(0, S_pad, chunk)):

This file is the acceptance test for that gap, written before the parameter
existed so it reported a precise skip rather than a red error. It still skips
cleanly if ``chunk_start`` is ever removed. Passing it is the definition of done
for the model half of chunked prefill; the plugin's model-type allowlist is a
separate change in a repo this one does not own.

What it proves, and why each part is needed:

* **The cache.** Every position written by the resumed path must match the
  single-pass path. This is the real claim: RoPE offsets and page-table slicing
  are correct when the prefix was written by an earlier call. Asserted on the
  cache itself, not argued from the ops' documentation.
* **The output.** Logits at the final position must match, proving attention
  actually read the resumed prefix rather than an empty or misaligned one.
* **Sensitivity.** A deliberately wrong resume offset must NOT match. Without
  this an implementation that silently ignores ``chunk_start`` would pass the
  first two checks by re-prefilling from zero every time.
* **Unaligned resumption.** vLLM chunks by its own token budget, which has no
  reason to land on this model's ``prefill_chunk_size``. A split that is
  block-aligned but not chunk-aligned is where page-table slicing is most likely
  to be wrong, so it is parametrised rather than assumed.

Runs on the reduced 2-layer probe (HF layer 0 dense + layer 1 moe, real embedding
/ norm / LM head / paged cache), so the whole file is well under a minute on one
Blackhole chip.

    pytest models/autoports/zai_org_glm_4_7_flash/tests/test_prefill_resume.py -q -s
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import torch

import ttnn
from models.autoports.zai_org_glm_4_7_flash.tt.model import GLM47FlashModel

MODEL_DIR = Path(__file__).resolve().parents[1]
PROBE_LAYERS = [0, 1]

#: Long enough to need several chunks at the 2048 default, short enough to stay fast.
SEQ_LEN = 4096

#: Paged block size the model was built around. A resume offset that is not a
#: multiple of this cannot be expressed by the page table at all, so every split
#: below is block-aligned.
BLOCK = 64

#: Largest prefill bucket. ``prefill_physical_len`` rounds a chunk UP to a bucket
#: so the number of compiled prefill shapes stays bounded, which means a resumed
#: chunk's PHYSICAL length can exceed its logical one: a 1536-token tail resuming
#: at 2560 writes through position 4608, not 4096. The page table therefore needs
#: room for ``chunk_start + padded_length``, not just for the prompt. Found by
#: this test failing on the block-aligned-only split, and it is a real constraint
#: on any caller: vLLM satisfies it by allocating blocks for max_model_len, but a
#: test sized exactly to the prompt does not.
MAX_BUCKET = 2048


@pytest.fixture(scope="module")
def device():
    dev = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1), l1_small_size=32768, trace_region_size=350_000_000)
    yield dev
    ttnn.close_mesh_device(dev)


@pytest.fixture(scope="module")
def model(device):
    return GLM47FlashModel.from_pretrained(
        device,
        max_batch_size=1,
        # Headroom for bucket padding on a resumed tail; see MAX_BUCKET.
        max_seq_len=SEQ_LEN + MAX_BUCKET,
        layer_indices=PROBE_LAYERS,
    )


def _supports_resume() -> bool:
    """Whether the prefill driver accepts a resume offset yet."""
    from models.autoports.zai_org_glm_4_7_flash.tt.fused_decoder import FusedDecoder

    return "chunk_start" in inspect.signature(FusedDecoder.prefill_forward).parameters


requires_resume = pytest.mark.skipif(
    not _supports_resume(),
    reason=(
        "FusedDecoder.prefill_forward has no chunk_start parameter, so a prefill "
        "cannot begin anywhere but position 0. This test is the acceptance "
        "criterion for adding it; see the module docstring."
    ),
)


def _prompt(seq: int) -> torch.Tensor:
    """A deterministic prompt. Fixed seed so a failure is reproducible."""
    g = torch.Generator().manual_seed(20260907)
    return torch.randint(0, 32000, (1, seq), generator=g)


def _fresh_state(model):
    """A cache and page table with nothing in them."""
    return model.allocate_kv_cache(), model.page_table_to_device(model.default_page_table())


def _cache_to_host(model, cache, upto_tokens: int) -> torch.Tensor:
    """Every cache block backing positions ``[0, upto_tokens)``, as one host tensor.

    Read whole blocks rather than sampling: a resume bug that lands one block
    early or late writes real values to the wrong place, which sampling three
    blocks the way the padding suite does would miss.
    """
    n_blocks = -(-upto_tokens // BLOCK)
    layers = []
    for layer_cache in cache:
        rows = ttnn.slice(
            layer_cache,
            [0, 0, 0, 0],
            [n_blocks, int(layer_cache.shape[1]), int(layer_cache.shape[2]), int(layer_cache.shape[3])],
            [1, 1, 1, 1],
        )
        layers.append(ttnn.to_torch(rows).to(torch.float32))
        ttnn.deallocate(rows)
    return torch.cat([layer.flatten() for layer in layers])


def _prefill_single_pass(model, ids, seq):
    cache, pt = _fresh_state(model)
    logits = model.prefill_forward(ids, kv_cache=cache, page_table=pt, seq_len=seq, return_all_logits=True)
    return logits, _cache_to_host(model, cache, seq)


def _prefill_resumed(model, ids, seq, split):
    """Prefill ``[0, split)``, then resume from ``split`` to ``seq``.

    The second call passes only the tail and tells the driver where it starts,
    because at THIS layer ``seq_len`` is a length. That is not the shape vLLM
    uses: the plugin hands the adapter the whole prefix from 0 with
    ``prompt_lens`` as the chunk's END position, and ``Generator.prefill_forward``
    slices ``tokens[start:end]`` before reaching here. Conflating the two is a
    real bug that got as far as a passing test once; the adapter-level
    counterpart lives in ``test_generator_vllm_adapter.py`` and uses vLLM's
    convention deliberately.
    """
    cache, pt = _fresh_state(model)
    model.prefill_forward(ids[:, :split], kv_cache=cache, page_table=pt, seq_len=split, return_all_logits=True)
    logits = model.prefill_forward(
        ids[:, split:],
        kv_cache=cache,
        page_table=pt,
        seq_len=seq - split,
        chunk_start=split,
        return_all_logits=True,
    )
    return logits, _cache_to_host(model, cache, seq)


def _pcc(a, b):
    x = a.flatten().double()
    y = b.flatten().double()
    x = x - x.mean()
    y = y - y.mean()
    return float((x @ y) / (x.norm() * y.norm()))


# ---------------------------------------------------------------------------- the claim


@requires_resume
@pytest.mark.parametrize(
    "split",
    [
        pytest.param(2048, id="chunk-aligned"),
        # Block-aligned but NOT a multiple of prefill_chunk_size. vLLM's token
        # budget has no reason to match this model's chunk size, and this is where
        # the page-table slice in _attn_prefill_chunk is most likely to be wrong.
        pytest.param(2560, id="block-aligned-only"),
        # A resume that begins inside the model's first chunk, so the tail call
        # spans a chunk boundary it did not start on.
        pytest.param(1024, id="mid-first-chunk"),
    ],
)
def test_resumed_prefill_writes_the_same_cache(model, split):
    """The cache after a resumed prefill is the cache after a single pass.

    This is the whole question. If the paged writes and the rotary offsets are
    correct across a call boundary, every position matches; if the resume lands on
    the wrong blocks or re-bases positions to zero, they do not.
    """
    ids = _prompt(SEQ_LEN)
    _, single = _prefill_single_pass(model, ids, SEQ_LEN)
    _, resumed = _prefill_resumed(model, ids, SEQ_LEN, split)

    assert single.shape == resumed.shape, (single.shape, resumed.shape)
    pcc = _pcc(single, resumed)
    mismatched = int((single != resumed).sum())
    assert pcc > 0.9999, (
        f"resumed prefill wrote a different cache: pcc={pcc:.8f}, "
        f"{mismatched}/{single.numel()} elements differ, split={split}"
    )


@requires_resume
def test_resumed_prefill_produces_the_same_logits(model):
    """Attention read the resumed prefix, not an empty or misaligned one.

    The cache check alone would pass if the tail were written correctly but never
    attended over. The final position depends on every earlier one, so its logits
    are the cheapest end-to-end proof that the prefix was actually used.
    """
    ids = _prompt(SEQ_LEN)
    single, _ = _prefill_single_pass(model, ids, SEQ_LEN)
    resumed, _ = _prefill_resumed(model, ids, SEQ_LEN, 2048)

    a = single[..., -1, :].flatten().float()
    b = resumed[..., -1, :].flatten().float()
    pcc = _pcc(a, b)
    assert pcc > 0.9999, f"last-position logits diverged: pcc={pcc:.8f}"
    assert int(a.argmax()) == int(
        b.argmax()
    ), f"resumed prefill would emit a different first token: {int(a.argmax())} vs {int(b.argmax())}"


# ---------------------------------------------------------------- the test can still fail


@requires_resume
def test_resume_offset_actually_changes_the_result(model):
    """A wrong resume offset must NOT match the single pass.

    Without this, an implementation that accepts ``chunk_start`` and ignores it,
    re-prefilling from zero every time, would pass every check above. This asserts
    the tests are sensitive to the parameter they exist to validate.
    """
    ids = _prompt(SEQ_LEN)
    _, single = _prefill_single_pass(model, ids, SEQ_LEN)

    # Feed the correct tail but claim it starts a block too early. The positions
    # and the blocks are then both off by one block.
    cache, pt = _fresh_state(model)
    split = 2048
    model.prefill_forward(ids[:, :split], kv_cache=cache, page_table=pt, seq_len=split, return_all_logits=True)
    model.prefill_forward(
        ids[:, split:],
        kv_cache=cache,
        page_table=pt,
        seq_len=SEQ_LEN - split,
        chunk_start=split - BLOCK,
        return_all_logits=True,
    )
    wrong = _cache_to_host(model, cache, SEQ_LEN)

    pcc = _pcc(single, wrong)
    assert pcc < 0.9999, (
        f"a resume offset {BLOCK} tokens early produced the same cache (pcc={pcc:.8f}); "
        "chunk_start is being ignored, so the equivalence tests above prove nothing"
    )
