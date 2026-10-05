# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Laguna-S-2.1 full-model accuracy on p150x4: top-1 / top-5 / top-100 and logits PCC against the original model.

The reference is Hugging Face's Laguna run in fp32 on the CPU, one layer at a time, over the AIME24 chat prompt
(235 tokens) plus a fixed 100-token answer: ``tests/reference_outputs/readiness_aime24_chat_s.refpt`` holds the
tokens, and the fp32 logits (100 positions x 100,352 vocabulary scores, 40 MB, not in the repository) are made once
with ``tests/gen_streamed_reference.py --save-logits`` (CPU only, about 5 minutes, about 30 GB of host memory):

    python -m models.demos.laguna.tests.gen_streamed_reference --dtype fp32 \\
      --output generated/laguna_reference/readiness_aime24_chat_s.refpt \\
      --save-logits generated/laguna_reference/Laguna-S-2.1-aime24-logits.pt

The test feeds the prompt (prefill) and then the 100 answer tokens one at a time (decode), and at each of the 100
positions compares Laguna's next-token scores with the reference's:

    top-1       Laguna's highest-scoring token is the reference's highest-scoring token
    top-5       the reference's highest-scoring token is among Laguna's 5 highest
    top-100     ... among Laguna's 100 highest
    PCC         Pearson correlation of all 100,352 scores with the reference's (mean and worst over positions)
    PCC top-100 the same over the reference's 100 highest-scoring tokens only
    traced top-1 top-1 of the traced decode path the server uses (on-device greedy sampling, teacher-forced)

Bars: top-1 >= 0.90, top-5 >= 0.98, top-100 = 1.00, traced top-1 >= 0.90, mean PCC >= 0.95. Experts are stored as
4-bit bfloat4_b, so whole-model scores carry rounding error (mean PCC about 0.97) while the chosen tokens agree.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

for _key, _value in {
    "TT_LAGUNA_MODEL": "poolside/Laguna-S-2.1",
    "LAGUNA_PROFILE": "p150x4",
    "TT_VISIBLE_DEVICES": "0,1,2,3",
    "LAGUNA_FABRIC_CONFIG": "FABRIC_1D_RING",
    "TT_LAGUNA_CCL_TOPOLOGY": "ring",
    "TT_LAGUNA_CCL_NUM_LINKS": "2",
    "TT_LAGUNA_DECODE_SDPA_PC": "1",
}.items():
    os.environ.setdefault(_key, _value)  # the serving profile's settings, before any Laguna import reads them

import pytest  # noqa: E402
import torch  # noqa: E402

import ttnn  # noqa: E402
from models.common.readiness_check.schema import load_reference  # noqa: E402
from models.demos.laguna.tests.laguna_test_utils import close_mesh, open_mesh, resolve_profile  # noqa: E402
from models.demos.laguna.tt.generator import LagunaGenerator  # noqa: E402

MODEL_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[4]
REFERENCE_TOKENS = MODEL_DIR / "tests" / "reference_outputs" / "readiness_aime24_chat_s.refpt"
REFERENCE_LOGITS = Path(
    os.environ.get("LAGUNA_REFERENCE_LOGITS")
    or REPO_ROOT / "generated" / "laguna_reference" / "Laguna-S-2.1-aime24-logits.pt"
)
TRACE_REGION = 200_000_000  # the 48-layer decode trace is 27.2 MB
BARS = {"top1": 0.90, "top5": 0.98, "top100": 1.00, "traced_top1": 0.90, "pcc_mean": 0.95}


def pcc(actual: torch.Tensor, expected: torch.Tensor) -> float:
    pair = torch.stack((actual.float().reshape(-1), expected.float().reshape(-1)))
    return float(torch.corrcoef(pair)[0, 1])


@pytest.mark.no_reset_default_device
@pytest.mark.timeout(3600)
def test_laguna_s_accuracy():
    if not REFERENCE_LOGITS.is_file():
        pytest.fail(
            f"reference logits not found at {REFERENCE_LOGITS}. Make them once (CPU only, ~5 min, ~30 GB RAM) from the "
            "repository root:\n  python -m models.demos.laguna.tests.gen_streamed_reference --dtype fp32 "
            "--output generated/laguna_reference/readiness_aime24_chat_s.refpt "
            "--save-logits generated/laguna_reference/Laguna-S-2.1-aime24-logits.pt\n"
            "or point LAGUNA_REFERENCE_LOGITS at an existing file."
        )
    entry = load_reference(REFERENCE_TOKENS).entries[0]
    prompt = entry.prompt_tokens[0].tolist()
    answer = entry.generated_tokens[0].tolist()
    reference = torch.load(REFERENCE_LOGITS)["logits"]  # [100, vocab]; row i scores the token at answer[i]
    positions = len(answer)
    assert reference.shape[0] == positions, f"reference logits have {reference.shape[0]} rows for {positions} tokens"

    mesh = generator = None
    try:
        mesh = open_mesh(ttnn, resolve_profile("p150x4", trace_region_size=TRACE_REGION))
        generator = LagunaGenerator.from_pretrained(mesh, max_seq_len=1024)
        generator._ensure_cache(1, len(prompt) + positions + 1)

        # Prefill predicts answer[0]; each decode step i feeds answer[i] and predicts answer[i + 1].
        start = time.perf_counter()
        logits = [generator.prefill_forward(torch.tensor([prompt]), prompt_lens=[len(prompt)]).reshape(-1)]
        for i in range(positions - 1):
            step = generator.decode_forward(
                torch.tensor([[answer[i]]]), torch.tensor([len(prompt) + i]), return_logits=True
            )
            logits.append(step.reshape(-1))
        eager_seconds = time.perf_counter() - start

        # The serving path: captured decode trace with on-device greedy sampling, fed the same answer tokens.
        generator.reset()
        traced = generator.generate(prompt, positions, next_input=lambda i, _previous: answer[i], enable_trace=True)
    finally:
        if generator is not None:
            generator.teardown()
        if mesh is not None:
            close_mesh(ttnn, mesh)

    target = reference.argmax(dim=-1)
    ranks = [(row.float().topk(100).indices == int(target[i])).nonzero() for i, row in enumerate(logits)]
    rank = [int(found[0, 0]) if found.numel() else 100 for found in ranks]  # 0 = Laguna's top token
    pccs = [pcc(row, reference[i]) for i, row in enumerate(logits)]
    top_ids = [reference[i].float().topk(100).indices for i in range(positions)]
    pccs_top = [pcc(row.reshape(-1)[ids], reference[i].reshape(-1)[ids]) for i, (row, ids) in enumerate(zip(logits, top_ids))]
    result = {
        "top1": sum(r == 0 for r in rank) / positions,
        "top5": sum(r < 5 for r in rank) / positions,
        "top100": sum(r < 100 for r in rank) / positions,
        "traced_top1": sum(int(a) == int(b) for a, b in zip(traced, target)) / positions,
        "pcc_mean": sum(pccs) / positions,
        "pcc_worst": min(pccs),
        "pcc_top100_mean": sum(pccs_top) / positions,
    }
    print(
        f"\nLaguna-S-2.1 accuracy vs fp32 Hugging Face, {positions} positions (prefill + {positions - 1} decode steps, "
        f"{eager_seconds:.1f} s)\n"
        f"  top-1 {result['top1']:.2f}   top-5 {result['top5']:.2f}   top-100 {result['top100']:.2f}   "
        f"traced top-1 {result['traced_top1']:.2f}\n"
        f"  PCC mean {result['pcc_mean']:.4f}   PCC worst {result['pcc_worst']:.4f}   "
        f"PCC over the reference's top-100 tokens {result['pcc_top100_mean']:.4f}",
        flush=True,
    )
    failures = [f"{name} {result[name]:.4f} < {bar}" for name, bar in BARS.items() if result[name] < bar]
    assert not failures, "; ".join(failures)
