# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""
Decode vs trusted prefill self-consistency (see design in task spec).

Phase 1: autoregressive decode on device.
Phase 2: one prefill forward on ``prompt_ids + generated_ids``.
Phase 3: each decode token should rank highly in prefill logits at the matching causal row.

Greedy sampling (temperature 0) is required for a meaningful top-1 rate; stochastic sampling can
pass via top-k / log-prob while top-1 vs prefill argmax is intentionally not asserted.

Thresholds default slightly below the ideal (>90% top-1, >99% top-5) to absorb decode vs
one-shot prefill numeric drift; override with environment variables if needed.
"""

import os
from pathlib import Path

import pytest

from models.demos.deepseek_v3.demo.demo import load_prompts_from_json, run_demo

MODEL_PATH = Path(
    os.getenv("DEEPSEEK_V3_HF_MODEL", "/mnt/MLPerf/tt_dnn-models/deepseek-ai/DeepSeek-R1-0528-dequantized")
)
CACHE_DIR = Path(os.getenv("DEEPSEEK_V3_CACHE", "/mnt/MLPerf/tt_dnn-models/deepseek-ai/DeepSeek-R1-0528-Cache/CI"))


def _threshold(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return float(raw)


@pytest.mark.timeout(2400)
@pytest.mark.requires_device(["DUAL"])
def test_demo_decode_prefill_agreement_dual_full_demo_8upr(force_recalculate_weight_config: bool):
    """
    Same shape as ``dual_full_demo_8upr`` in ``test_demo.py`` (24 prompts, 8 users/row, 129 new tokens),
    plus ``decode_prefill_verify``: after generation, re-score user 0 with a full-sequence prefill.

    Uses greedy decoding so top-1 agreement (decode token == prefill argmax at that row) is meaningful.
    """
    json_path = "models/demos/deepseek_v3/demo/test_prompts.json"
    max_prompts = 24
    max_users_per_row = 8
    max_new_tokens = 129

    prompts = load_prompts_from_json(json_path, max_prompts=max_prompts)

    results = run_demo(
        prompts=prompts,
        model_path=MODEL_PATH,
        cache_dir=CACHE_DIR,
        random_weights=False,
        max_new_tokens=max_new_tokens,
        max_users_per_row=max_users_per_row,
        repeat_batches=1,
        enable_trace=True,
        sample_on_device=True,
        profile_decode=False,
        force_recalculate=force_recalculate_weight_config,
        signpost=True,
        decode_prefill_verify=True,
        sampling_temperature=0.0,
        sampling_top_k=1,
        sampling_top_p=1.0,
    )

    requested_system_name = os.getenv("MESH_DEVICE")
    assert requested_system_name is not None, "MESH_DEVICE must be set for demo tests"

    gen0 = results["generations"][0]
    n_tok = int(gen0["decode_prefill_agreement_num_tokens"])
    assert n_tok > 0, "Expected at least one generated token for agreement metrics"
    assert n_tok == len(gen0["tokens"]), "Agreement should cover every decoded token for user 0"

    top1_min = _threshold("DEEPSEEK_DECODE_PREFILL_TOP1_MIN", 0.82)
    top5_min = _threshold("DEEPSEEK_DECODE_PREFILL_TOP5_MIN", 0.985)

    top1 = float(gen0["decode_prefill_agreement_top1_rate"])
    top5 = float(gen0["decode_prefill_agreement_topk_rate"])
    mean_lp = float(gen0["decode_prefill_agreement_mean_logprob"])

    assert top5 >= top5_min, (
        f"Prefill top-5 containment rate {top5:.4f} below {top5_min:.4f} "
        f"(decode tokens not in prefill top-5 — likely decode bug or misaligned logits rows)"
    )
    assert top1 >= top1_min, (
        f"Prefill top-1 agreement {top1:.4f} below {top1_min:.4f} "
        f"(raise DEEPSEEK_DECODE_PREFILL_TOP1_MIN if this is expected drift on your stack)"
    )

    mean_lp_min = os.getenv("DEEPSEEK_DECODE_PREFILL_MEAN_LOGPROB_MIN")
    if mean_lp_min is not None and mean_lp_min != "":
        assert mean_lp >= float(
            mean_lp_min
        ), f"Mean log P under prefill {mean_lp:.4f} below floor {float(mean_lp_min):.4f}"

    # Log for CI artifacts / debugging
    print(
        f"\nDecode/prefill agreement (user 0): top1={top1:.4f} top5={top5:.4f} "
        f"mean_logprob={mean_lp:.4f} max_rank={gen0['decode_prefill_agreement_max_rank']} "
        f"over {n_tok} tokens\n"
    )
