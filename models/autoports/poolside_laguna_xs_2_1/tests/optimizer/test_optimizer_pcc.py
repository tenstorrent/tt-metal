# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Full-model correctness gate for tt_hw_planner optimizer runs on Laguna-S-2.1 (p150x4, all 48 layers).

The sequence is the AIME24 readiness sequence (tests/reference_outputs/readiness_aime24_chat_s.refpt): a
235-token chat prompt and a fixed 100-token continuation. The reference is HF fp32, computed layer by layer on
the host by tests/gen_streamed_reference.py --save-logits (S does not fit in host RAM in one piece) and kept
under the repo's gitignored generated/optimizer_reference/.

Checks, all over the same 100 positions (one prefill position + 99 teacher-forced eager decode steps):

1. The gate score the optimizer parses ("PCC: x") is min(eager top-1 agreement, traced top-1 agreement) with HF's
   argmax, as a fraction, held to the absolute floor PCC_THRESHOLD (the readiness top-1 bar Laguna-S was qualified
   on). Logits correlation is not used as the absolute floor: Laguna's bfloat4_b experts and bfloat16 logits never
   reproduce fp32 logits closely even when every token choice matches. Measured on the unmodified tree: top-1
   99/100 while the worst position's correlation is 0.74 over the full vocabulary and 0.86 over HF's top-100 ids.
   Both correlations are still reported and their means are held relative to the baseline (check 2).
2. Top-1 / top-5 agreement with the HF argmax, RELATIVE to a baseline pinned from the unmodified tree
   (generated/optimizer_accuracy_baseline_Laguna-S-2.1.json). One position is 1.0 point.
3. Top-1 of the TRACED token-out path (on-device Sampling1D greedy, the path the perf gate times), teacher
   forced over the same continuation, also relative to the pinned baseline.

The optimizer keeps or reverts a change on the parsed PCC alone (perf_automation/agent/pcc_runner.py takes
the WORST "pcc ... <float>" in the output), not on the pytest exit code. So when an agreement check fails this
test also prints "PCC: 0.000000".
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

# Serving knobs of the qualified p150x4 Laguna-S profile (serve_vllm.sh), set before any Laguna import so
# module-level reads see them, whoever launches the gate.
for _k, _v in {
    "TT_LAGUNA_MODEL": "poolside/Laguna-S-2.1",
    "LAGUNA_PROFILE": "p150x4",
    "TT_VISIBLE_DEVICES": "0,1,2,3",
    "LAGUNA_FABRIC_CONFIG": "FABRIC_1D_RING",
    "TT_LAGUNA_CCL_TOPOLOGY": "ring",
    "TT_LAGUNA_CCL_NUM_LINKS": "2",
    "TT_LAGUNA_DECODE_SDPA_PC": "1",
    "TT_LAGUNA_DECODE_K": "64",
    "TT_LAGUNA_DECODE_EXP": "0",
    "TT_LAGUNA_DECODE_MAXCORES": "16",
    "TT_LAGUNA_PREFILL_FAST": "1",
    "TT_LAGUNA_PIPE_CHUNK": "2048",
    # The optimizer's own converted-weight cache, so its experiments never touch the serving cache.
    "TT_LAGUNA_WEIGHT_CACHE": "/home/ttuser/benchmark-results/laguna-opt-weight-cache",
}.items():
    os.environ.setdefault(_k, _v)

import pytest  # noqa: E402
import torch  # noqa: E402

import ttnn  # noqa: E402
from models.autoports.poolside_laguna_xs_2_1.tests.laguna_test_utils import close_mesh, open_mesh, resolve_profile  # noqa: E402
from models.autoports.poolside_laguna_xs_2_1.tt.generator import LagunaGenerator  # noqa: E402
from models.common.readiness_check.schema import load_reference  # noqa: E402

# The checkpoint this gate runs; stated as a literal so the optimizer's roofline can resolve the model.
HF_MODEL_ID = os.environ.get("HF_MODEL_ID") or "poolside/Laguna-S-2.1"

# Absolute floor for the worst logits PCC over every position. The optimizer lifts this constant from the
# file text as its pass/fail threshold (model_files._extract_pcc_threshold).
PCC_THRESHOLD = 0.90
MAX_SEQ_LEN = 1024
# 27.2 MB decode trace on S; the optimizer may raise TT_PERF_TRACE_REGION.
TRACE_REGION = max(int(os.environ.get("TT_PERF_TRACE_REGION") or 0), 200_000_000)
FULL_DEPTH = 48

# Relative floors against the pinned baseline: one position is 1.0 point, so 1.0 allows one flip.
TOP1_DROP_PTS = float(os.environ.get("PCC_GATE_TOP1_DROP_PTS", "1.0"))
TOP5_DROP_PTS = float(os.environ.get("PCC_GATE_TOP5_DROP_PTS", "1.0"))
TRACED_TOP1_DROP_PTS = float(os.environ.get("PCC_GATE_TRACED_TOP1_DROP_PTS", "1.0"))
MEAN_PCC_DROP = float(os.environ.get("PCC_GATE_MEAN_PCC_DROP", "0.002"))
TOP100_MEAN_PCC_DROP = float(os.environ.get("PCC_GATE_TOP100_MEAN_PCC_DROP", "0.005"))

MODEL_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = Path(__file__).resolve().parents[5]
GENERATED = REPO_ROOT / "generated"
REFERENCE = MODEL_DIR / "tests" / "reference_outputs" / "readiness_aime24_chat_s.refpt"
REFERENCE_LOGITS = GENERATED / "optimizer_reference" / "Laguna-S-2.1-aime24-logits.pt"
BASELINE_PATH = GENERATED / f"optimizer_accuracy_baseline_{Path(os.environ.get('HF_MODEL') or HF_MODEL_ID).name}.json"


def _pcc(actual: torch.Tensor, expected: torch.Tensor) -> float:
    actual = actual.float().reshape(-1)
    expected = expected.float().reshape(-1)
    assert actual.shape == expected.shape
    assert torch.isfinite(actual).all()
    assert torch.isfinite(expected).all()
    return float(torch.corrcoef(torch.stack((actual, expected)))[0, 1])


def _pcc_top(actual: torch.Tensor, expected: torch.Tensor, k: int = 100) -> float:
    """Correlation over the reference's top-k token ids at this position (the tokens that matter)."""
    ids = expected.reshape(-1).float().topk(k).indices
    return _pcc(actual.reshape(-1)[ids], expected.reshape(-1)[ids])


def _agreement(actual: torch.Tensor, expected: torch.Tensor) -> tuple[bool, bool]:
    """(top-1 match, reference argmax within the model's top 5) for one position."""
    ref = int(expected.reshape(-1).argmax())
    top5 = actual.reshape(-1).float().topk(5).indices.tolist()
    return top5[0] == ref, ref in top5


def _git_head() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=str(MODEL_DIR), capture_output=True, text=True, timeout=30
        ).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


# One JSON line per gate run (accuracy over the optimizer's lifetime). The working-tree diff hash tells
# candidates on the same HEAD apart.
HISTORY_PATH = Path(
    os.environ.get("OPTIMIZER_ACCURACY_HISTORY")
    or "/home/ttuser/benchmark-results/ashwary-laguna/accuracy_history.jsonl"
)


def _tree_diff_hash() -> str:
    try:
        diff = subprocess.run(
            ["git", "diff", "HEAD"], cwd=str(MODEL_DIR), capture_output=True, timeout=60
        ).stdout
    except Exception:  # noqa: BLE001
        return ""
    if not diff:
        return "clean"
    import hashlib

    return hashlib.sha1(diff).hexdigest()[:12]


def _append_history(record: dict) -> None:
    try:
        HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        with HISTORY_PATH.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
    except OSError as error:
        print(f"ACCURACY_HISTORY not written ({error})", flush=True)


def _load_baseline() -> dict | None:
    try:
        doc = json.loads(BASELINE_PATH.read_text())
    except (OSError, ValueError):
        return None
    return doc if isinstance(doc, dict) and doc.get("top1_pct") is not None else None


@pytest.mark.no_reset_default_device
@pytest.mark.timeout(3600)
def test_optimizer_full_model_pcc():
    """Prefill + 99 teacher-forced eager decode steps against HF fp32, every layer; plus the traced path."""
    # Under `pytest -s` the first print lands on the node-id line, which contains "pcc"; the optimizer's
    # parser reads any "pcc ... <float>" on a line as a measurement. End that line first.
    print("", flush=True)
    assert REFERENCE_LOGITS.is_file(), (
        f"missing {REFERENCE_LOGITS}; produce it with tests/gen_streamed_reference.py --dtype fp32 --save-logits"
    )
    entry = load_reference(REFERENCE).entries[0]
    prompt = entry.prompt_tokens[0].tolist()
    cont = entry.generated_tokens[0].tolist()
    hf = torch.load(REFERENCE_LOGITS)["logits"]  # [100, vocab]: row i predicts cont[i]
    P, G = len(prompt), len(cont)
    assert hf.shape[0] == G

    mesh = gen = None
    try:
        mesh = open_mesh(ttnn, resolve_profile("p150x4", trace_region_size=TRACE_REGION))
        gen = LagunaGenerator.from_pretrained(mesh, max_seq_len=MAX_SEQ_LEN)
        assert len(gen.model.layers) == FULL_DEPTH
        assert mesh.get_num_devices() == 4

        # The generator owns the cache; size it for prompt + continuation before prefill (prefill alone would
        # size it for the prompt only, and the eager decode steps would run past it).
        gen._ensure_cache(1, P + G + 1)
        t0 = time.perf_counter()
        tt_prefill = gen.prefill_forward(torch.tensor([prompt]), prompt_lens=[P]).reshape(-1)
        pccs = [_pcc(tt_prefill, hf[0])]
        top_pccs = [_pcc_top(tt_prefill, hf[0])]
        hits = [_agreement(tt_prefill, hf[0])]
        eager_argmax = [int(tt_prefill.argmax())]
        for i in range(G - 1):
            logits = gen.decode_forward(torch.tensor([[cont[i]]]), torch.tensor([P + i]), return_logits=True).reshape(-1)
            pccs.append(_pcc(logits, hf[i + 1]))
            top_pccs.append(_pcc_top(logits, hf[i + 1]))
            hits.append(_agreement(logits, hf[i + 1]))
            eager_argmax.append(int(logits.argmax()))
        eager_seconds = time.perf_counter() - t0

        # Traced token-out path: the decode trace with on-device greedy sampling, teacher forced.
        gen.reset()
        t1 = time.perf_counter()
        traced = gen.generate(prompt, G, next_input=lambda i, p: cont[i], enable_trace=True)
        traced_seconds = time.perf_counter() - t1
        hf_argmax = hf.argmax(dim=-1).tolist()
        traced_top1 = 100.0 * sum(int(a == b) for a, b in zip(traced, hf_argmax)) / G
        traced_vs_eager = sum(int(a == b) for a, b in zip(traced, eager_argmax))

        positions = len(pccs)
        top1_pct = 100.0 * sum(h[0] for h in hits) / positions
        top5_pct = 100.0 * sum(h[1] for h in hits) / positions
        mean_corr = sum(pccs) / positions
        worst_corr = min(pccs)
        worst_pos = min(range(positions), key=pccs.__getitem__)
        worst_top = min(top_pccs)
        top100_mean = sum(top_pccs) / positions
        score = min(top1_pct, traced_top1) / 100.0
        worst_top_pos = min(range(positions), key=top_pccs.__getitem__)
        print(
            f"ACCURACY positions={positions} top1_pct={top1_pct:.2f} top5_pct={top5_pct:.2f} "
            f"mean_corr={mean_corr:.6f} worst_corr={worst_corr:.6f} worst_pos={worst_pos} "
            f"prefill_corr={pccs[0]:.6f} top100_mean_corr={top100_mean:.6f} top100_worst_corr={worst_top:.6f} "
            f"top100_worst_pos={worst_top_pos} traced_top1_pct={traced_top1:.2f} traced_vs_eager={traced_vs_eager}/{G} "
            f"eager_seconds={eager_seconds:.1f} traced_seconds={traced_seconds:.1f} tree={_git_head()}",
            flush=True,
        )

        baseline = _load_baseline()
        failures = []
        if os.environ.get("PCC_GATE_PIN_BASELINE") == "1" or baseline is None:
            BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
            BASELINE_PATH.write_text(
                json.dumps(
                    {
                        "top1_pct": top1_pct,
                        "top5_pct": top5_pct,
                        "traced_top1_pct": traced_top1,
                        "mean_corr": mean_corr,
                        "worst_corr": worst_corr,
                        "top100_worst_corr": worst_top,
                        "top100_mean_corr": top100_mean,
                        "positions": positions,
                        "tree": _git_head(),
                        "pinned_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    },
                    indent=1,
                )
            )
            print(
                f"ACCURACY_BASELINE pinned to {BASELINE_PATH} from tree {_git_head() or '?'}: top1={top1_pct:.2f} "
                f"top5={top5_pct:.2f} traced_top1={traced_top1:.2f} mean_corr={mean_corr:.6f}.",
                flush=True,
            )
        else:
            if top1_pct < baseline["top1_pct"] - TOP1_DROP_PTS:
                failures.append(f"top-1 {top1_pct:.2f}% < baseline {baseline['top1_pct']:.2f}% - {TOP1_DROP_PTS}")
            if top5_pct < baseline["top5_pct"] - TOP5_DROP_PTS:
                failures.append(f"top-5 {top5_pct:.2f}% < baseline {baseline['top5_pct']:.2f}% - {TOP5_DROP_PTS}")
            if traced_top1 < baseline.get("traced_top1_pct", traced_top1) - TRACED_TOP1_DROP_PTS:
                failures.append(
                    f"traced top-1 {traced_top1:.2f}% < baseline {baseline['traced_top1_pct']:.2f}% - {TRACED_TOP1_DROP_PTS}"
                )
            if top100_mean < baseline.get("top100_mean_corr", top100_mean) - TOP100_MEAN_PCC_DROP:
                failures.append(
                    f"top-100 mean correlation {top100_mean:.6f} < baseline {baseline['top100_mean_corr']:.6f} - {TOP100_MEAN_PCC_DROP}"
                )
            if mean_corr < baseline["mean_corr"] - MEAN_PCC_DROP:
                failures.append(f"mean logits correlation {mean_corr:.6f} < baseline {baseline['mean_corr']:.6f} - {MEAN_PCC_DROP}")
            print(
                f"ACCURACY_VS_BASELINE tree={baseline.get('tree') or '?'} "
                f"top1_delta_pts={top1_pct - baseline['top1_pct']:+.2f} "
                f"top5_delta_pts={top5_pct - baseline['top5_pct']:+.2f} "
                f"traced_top1_delta_pts={traced_top1 - baseline.get('traced_top1_pct', traced_top1):+.2f} "
                f"top100_mean_corr_delta={top100_mean - baseline.get('top100_mean_corr', top100_mean):+.6f} "
                f"mean_corr_delta={mean_corr - baseline['mean_corr']:+.6f}",
                flush=True,
            )
        _append_history(
            {
                "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "tree": _git_head(),
                "diff": _tree_diff_hash(),
                "top1_pct": top1_pct,
                "top5_pct": top5_pct,
                "traced_top1_pct": traced_top1,
                "mean_corr": mean_corr,
                "worst_corr": worst_corr,
                "top100_mean_corr": top100_mean,
                "top100_worst_corr": worst_top,
                "score": score if not failures else 0.0,
                "passed": not failures and score >= PCC_THRESHOLD,
                "failures": failures,
                "eager_seconds": round(eager_seconds, 1),
                "traced_seconds": round(traced_seconds, 1),
            }
        )
        if failures:
            print("ACCURACY GATE FAILED: " + "; ".join(failures), flush=True)
            print("PCC: 0.000000 (accuracy gate failed, see ACCURACY GATE FAILED above)", flush=True)
            assert not failures, "; ".join(failures)

        print(
            f"GATE_SCORE eager_top1={top1_pct / 100:.4f} traced_top1={traced_top1 / 100:.4f} "
            f"(top-100 corr worst {worst_top:.4f} mean {top100_mean:.4f}) | PCC: {score:.6f}",
            flush=True,
        )
        assert score >= PCC_THRESHOLD
    finally:
        if gen is not None:
            try:
                gen.teardown()
            except Exception:  # noqa: BLE001
                pass
        gen = None
        if mesh is not None:
            close_mesh(ttnn, mesh)
