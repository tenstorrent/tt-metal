# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
CPU test harness for the DeepSeek V3.2 reference layers (Indexer + MLA).

  * ``test_indexer_layer`` — random indexer, one forward pass; asserts shapes.
  * ``test_mla_layer``     — random MLA, one-shot prefill vs. prefill(P-1)+decode(1);
    asserts the decoded position matches the prefill (cache-correctness invariant).
  * ``test_*_pretrained_layer0`` — load real layer-0 weights (skipped if the shard
    isn't cached) and run one forward.

Models and inputs are built once via module-scoped fixtures and reused: random
(`mla`, `indexer`) and pretrained layer-0 (`mla_pretrained`, `indexer_pretrained`)
instances, plus shared inputs (`x`, `qr`, `freqs_cis`). ``x`` is parametrizable
(one shape for now). Reuse is safe because each MLA forward fully rewrites the
causal cache range it reads, so tests are order-independent.

All non-pretrained tests are pytest-discoverable and also run from ``main()``
(which calls the creation helpers directly instead of using fixtures).
"""

import json
import os
import sys
from pathlib import Path

# Allow `python test_model.py` from this directory: ensure the repo root is on
# sys.path so the absolute `models.demos.deepseek_v3_d_p.reference.cpu_deepseek_v32.*` imports
# below resolve. Under `python -m pytest` from the repo root this is a no-op.
_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pytest
import torch
from loguru import logger

from models.demos.deepseek_v3_d_p.reference.cpu_deepseek_v32.model import MLACPU, IndexerCPU, ModelArgs
from models.demos.deepseek_v3_d_p.reference.cpu_deepseek_v32.utils import precompute_freqs_cis
from models.demos.deepseek_v3_d_p.reference.cpu_deepseek_v32.weights import initialize_weights

SEED = 42
BATCH_SIZE = 2
# Small seq_len -> index_topk collapses to min(2048, T) = T (all positions
# selected), so the index mask is all-zero and the MLA equivalence test isolates
# the attention math from the discrete top-k selection.
SEQ_LEN = 8
PRETRAINED_LAYER = 0
N_DETERMINISM_RUNS = 5
PCC = 0.9999
# Prefill (MHA) and decode (MQA-absorbed) are algebraically identical (fp32 agrees
# to ~1e-6); in bf16 they match to PCC ~0.9999, so assert a safe 0.999 floor —
# far above a broken path, well below the bf16 noise floor.
EQUIV_PCC = 0.999
# Saved tensors / metadata land under ~/.cache, outside the repo, so the artifacts need no
# .gitignore. Override with DEEPSEEK_V32_TEST_OUTPUT_DIR.
_DEFAULT_OUTPUT_DIR = Path.home() / ".cache" / "deepseek_v32" / "test_outputs"
OUTPUT_DIR = Path(os.environ.get("DEEPSEEK_V32_TEST_OUTPUT_DIR", _DEFAULT_OUTPUT_DIR))


# ===== Creation helpers (encapsulation) =====


def make_args() -> ModelArgs:
    return ModelArgs()


def make_input(args: ModelArgs, batch_size: int = BATCH_SIZE, seq_len: int = SEQ_LEN, seed: int = SEED + 1):
    """Random input features ``[B, S, dim]`` (seeded for reproducibility)."""
    torch.manual_seed(seed)
    return torch.randn(batch_size, seq_len, args.dim, dtype=torch.bfloat16)


def make_qr(args: ModelArgs, batch_size: int = BATCH_SIZE, seq_len: int = SEQ_LEN, seed: int = SEED + 2):
    """Random query latent ``[B, S, q_lora_rank]`` (the indexer's `qr` input)."""
    torch.manual_seed(seed)
    return torch.randn(batch_size, seq_len, args.q_lora_rank, dtype=torch.bfloat16)


def build_mla(args: ModelArgs, seed: int = SEED) -> MLACPU:
    """An ``MLACPU`` with random (seeded) weights, including the nested indexer."""
    torch.manual_seed(seed)
    mla = MLACPU(args)
    initialize_weights(mla)
    return mla


def build_indexer(args: ModelArgs, seed: int = SEED) -> IndexerCPU:
    """A standalone ``IndexerCPU`` with random (seeded) weights."""
    torch.manual_seed(seed)
    indexer = IndexerCPU(args)
    initialize_weights(indexer)
    return indexer


def _load_layer0_or_skip(module):
    """Load cached layer-0 pretrained weights into ``module``, or skip the test."""
    from huggingface_hub.utils import LocalEntryNotFoundError

    try:
        initialize_weights(module, layer=PRETRAINED_LAYER, local_files_only=True)
    except LocalEntryNotFoundError:
        pytest.skip("layer-0 shard not cached; run a pretrained load once to populate it (downloads ~5 GB)")
    return module


# ===== Shared fixtures (built once per module, reused across tests) =====


@pytest.fixture(scope="module")
def args() -> ModelArgs:
    return make_args()


@pytest.fixture(scope="module")
def freqs_cis(args) -> torch.Tensor:
    return precompute_freqs_cis(args)


# (batch, seq) shapes to test. seq stays <= index_topk (2048) so the index mask
# is all-zero and the MLA equivalence test isolates the attention math.
X_SHAPES = [(BATCH_SIZE, SEQ_LEN), (1, 100)]


@pytest.fixture(scope="module", params=X_SHAPES, ids=lambda p: f"b{p[0]}_s{p[1]}")
def x(args, request) -> torch.Tensor:
    """Input features. Parametrizable: add (batch, seq) tuples to `X_SHAPES`."""
    batch_size, seq_len = request.param
    return make_input(args, batch_size, seq_len)


@pytest.fixture(scope="module")
def qr(args, x) -> torch.Tensor:
    """Query latent matching ``x``'s shape (the indexer's `qr` input)."""
    batch_size, seq_len, _ = x.shape
    return make_qr(args, batch_size, seq_len)


@pytest.fixture(scope="module")
def mla(args) -> MLACPU:
    return build_mla(args)


@pytest.fixture(scope="module")
def indexer(args) -> IndexerCPU:
    return build_indexer(args)


@pytest.fixture(scope="module")
def mla_pretrained(args) -> MLACPU:
    return _load_layer0_or_skip(MLACPU(args))


@pytest.fixture(scope="module")
def indexer_pretrained(args) -> IndexerCPU:
    return _load_layer0_or_skip(IndexerCPU(args))


# ===== Indexer tests =====


def _check_indexer_forward(indexer: IndexerCPU, x: torch.Tensor, qr: torch.Tensor, freqs_cis: torch.Tensor):
    """Run one indexer forward and assert output shapes / finiteness."""
    bsz, seqlen, _ = x.size()
    topk_indices, index_scores = indexer.forward(x, qr, start_pos=0, freqs_cis=freqs_cis[:seqlen], mask=None)
    topk_k = min(indexer.index_topk, seqlen)
    assert index_scores.shape == (bsz, seqlen, seqlen)
    assert topk_indices.shape == (bsz, seqlen, topk_k)
    assert torch.isfinite(index_scores).all()
    return topk_indices, index_scores


@pytest.mark.slow
def test_indexer_layer(indexer, x, qr, freqs_cis):
    """Random indexer: one forward pass, check output shapes."""
    logger.info("DeepSeek V3.2 Indexer CPU Test (random)")
    topk_indices, index_scores = _check_indexer_forward(indexer, x, qr, freqs_cis)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(topk_indices, OUTPUT_DIR / "topk_indices.pt")
    torch.save(index_scores, OUTPUT_DIR / "index_scores.pt")
    logger.info("✓ Indexer test completed successfully")


@pytest.mark.slow
def test_indexer_pretrained_layer0(indexer_pretrained, x, qr, freqs_cis):
    """Pretrained layer-0 indexer: one forward pass, check output shapes."""
    logger.info("DeepSeek V3.2 Indexer CPU Test (pretrained layer 0)")
    _check_indexer_forward(indexer_pretrained, x, qr, freqs_cis)
    logger.info("✓ Pretrained layer-0 indexer forward OK")


# ===== MLA tests =====


def _run_prefill(mla: MLACPU, x: torch.Tensor, freqs_cis_full: torch.Tensor) -> torch.Tensor:
    """One-shot prefill over the whole chunk (mask is not None -> MHA path)."""
    _, seqlen, _ = x.size()
    mask = torch.full((seqlen, seqlen), float("-inf")).triu_(1)
    return mla.forward(x, start_pos=0, freqs_cis=freqs_cis_full[0:seqlen], mask=mask)


def _run_prefill_then_decode(mla: MLACPU, x: torch.Tensor, freqs_cis_full: torch.Tensor) -> torch.Tensor:
    """
    Prefill the first P-1 tokens, then decode token P-1 (mask is None -> MQA
    path). Returns the decoded output for the single last position [B, 1, dim].
    """
    _, seqlen, _ = x.size()
    p = seqlen - 1
    if p > 0:
        mask = torch.full((p, p), float("-inf")).triu_(1)
        _ = mla.forward(x[:, :p], start_pos=0, freqs_cis=freqs_cis_full[0:p], mask=mask)
    return mla.forward(x[:, p : p + 1], start_pos=p, freqs_cis=freqs_cis_full[p : p + 1], mask=None)


@pytest.mark.slow
def test_mla_layer(mla, x, freqs_cis):
    """
    Random MLA prefill vs. prefill+decode on the same module, asserting the
    cache-correctness invariant: prefilling P tokens then
    decoding token P gives the same output for position P as prefilling P+1 at
    once. Same weights (one module) + fresh causal cache writes each run.

    Compared via PCC: the paths are algebraically identical, so the residual is
    pure bf16 rounding (raw max-abs would surface a misleadingly large worst
    element; PCC reflects that the vectors are effectively the same).
    """
    from tests.ttnn.utils_for_testing import assert_with_pcc

    logger.info("DeepSeek V3.2 MLA CPU Test (random)")
    prefill_out = _run_prefill(mla, x, freqs_cis)
    decode_out = _run_prefill_then_decode(mla, x, freqs_cis)

    p = x.size(1) - 1
    ref_last = prefill_out[:, p : p + 1].float()
    dec_last = decode_out.float()
    assert torch.isfinite(prefill_out).all() and torch.isfinite(decode_out).all()

    _, pcc_msg = assert_with_pcc(ref_last, dec_last, pcc=EQUIV_PCC)
    max_abs = (ref_last - dec_last).abs().max().item()
    logger.info(f"Prefill-vs-decode equivalence (position P-1): {pcc_msg}; max|Δ|={max_abs:.6f}")
    logger.info("✓ Paths agree (PCC)")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(prefill_out, OUTPUT_DIR / "mla_prefill_out.pt")
    torch.save(decode_out, OUTPUT_DIR / "mla_decode_out.pt")
    metadata = {
        "batch_size": x.size(0),
        "seq_len": x.size(1),
        "seed": SEED,
        "prefill_out_shape": list(prefill_out.shape),
        "decode_out_shape": list(decode_out.shape),
        "equiv_pcc_threshold": EQUIV_PCC,
        "max_abs_diff": max_abs,
    }
    with open(OUTPUT_DIR / "mla_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("✓ MLA test completed successfully")


@pytest.mark.slow
def test_mla_pretrained_layer0(mla_pretrained, x, freqs_cis):
    """Pretrained layer-0 MLA: one prefill forward, check shape / finiteness."""
    logger.info("DeepSeek V3.2 MLA CPU Test (pretrained layer 0)")
    bsz, seqlen, _ = x.size()
    mask = torch.full((seqlen, seqlen), float("-inf")).triu_(1)
    with torch.no_grad():
        out = mla_pretrained.forward(x, start_pos=0, freqs_cis=freqs_cis[:seqlen], mask=mask)
    assert out.shape == (bsz, seqlen, mla_pretrained.dim)
    assert torch.isfinite(out).all()
    logger.info("✓ Pretrained layer-0 MLA forward OK")


@pytest.mark.slow
def test_mla_pretrained_determinism(mla_pretrained, x, freqs_cis, n_runs: int = N_DETERMINISM_RUNS):
    """
    Pretrained MLA forward must be deterministic and idempotent: the same input
    run ``n_runs`` times yields the same output, and the MLA + indexer caches are
    unchanged after each run. Compared against the first run via PCC.
    """
    from tests.ttnn.utils_for_testing import assert_with_pcc

    logger.info(f"DeepSeek V3.2 MLA determinism test ({n_runs} runs, pcc={PCC})")
    bsz, seqlen, _ = x.size()
    mask = torch.full((seqlen, seqlen), float("-inf")).triu_(1)

    def run_forward():
        with torch.no_grad():
            return mla_pretrained.forward(x, start_pos=0, freqs_cis=freqs_cis[:seqlen], mask=mask)

    def snapshot_caches():
        # Only the written slice [:bsz, :end_pos] is meaningful; the rest stays zero.
        m = mla_pretrained
        return {
            "kv_cache": m.kv_cache[:bsz, :seqlen].clone(),
            "pe_cache": m.pe_cache[:bsz, :seqlen].clone(),
            "indexer.k_cache": m.indexer.k_cache[:bsz, :seqlen].clone(),
            "indexer.k_scale_cache": m.indexer.k_scale_cache[:bsz, :seqlen].clone(),
        }

    out_ref = run_forward()
    caches_ref = snapshot_caches()

    for i in range(1, n_runs):
        out = run_forward()
        assert_with_pcc(out_ref.float(), out.float(), pcc=PCC)
        caches = snapshot_caches()
        for name, ref in caches_ref.items():
            assert_with_pcc(ref.float(), caches[name].float(), pcc=PCC)
        logger.info(f"  run {i + 1}/{n_runs}: output + caches match")

    logger.info(f"✓ MLA forward deterministic across {n_runs} runs (output + caches unchanged)")


def main():
    """Run the random-weight tests outside pytest, building model/data once."""
    logger.info(f"Random seed set to {SEED}")
    try:
        args = make_args()
        freqs_cis = precompute_freqs_cis(args)
        x = make_input(args)
        qr = make_qr(args)

        test_indexer_layer(build_indexer(args), x, qr, freqs_cis)
        test_mla_layer(build_mla(args), x, freqs_cis)
        logger.info("✓ All tests completed successfully")
    except Exception as e:
        logger.error(f"❌ Error during test: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
