# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""
Autoregressive decode guard using an external judge LM (default: Qwen2.5 instruct).

Teacher forcing compares TT logits to an HF reference at each step while the
KV context is fed the HF token sequence. A broken decode stack can therefore
still pass token-accuracy tests while greedy autoregressive output is garbage.

This test runs the same prompt as the checked-in ``.refpt`` in normal greedy
decode (sampling on host, temperature 0), decodes TT tokens with the DeepSeek
tokenizer, and compares perplexity under an external judge LM (default: a recent
Qwen2.5 instruct checkpoint; override with ``DEEPSEEK_PPL_JUDGE_MODEL``) against
the HF reference continuation from the same file. A healthy TT run should land
within a modest ratio of the HF baseline; incoherent TT text typically drives
the ratio orders of magnitude higher.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3.demo.demo import run_demo
from models.demos.deepseek_v3.demo.perplexity_judge import (
    DEFAULT_JUDGE_MODEL,
    load_judge_lm,
    mean_nll_continuation_with_prompt,
    perplexity_from_mean_nll,
)
from models.demos.deepseek_v3.utils.config_helpers import (
    DEFAULT_MAX_SEQ_LEN,
    K_CHUNK_SIZE,
    align_prefill_padded_seq_len,
)
from models.demos.deepseek_v3.utils.hf_model_utils import load_tokenizer
from models.demos.deepseek_v3.utils.test_utils import system_name_to_mesh_shape

MODEL_PATH = Path(
    os.getenv(
        "DEEPSEEK_V3_HF_MODEL",
        "/mnt/MLPerf/tt_dnn-models/deepseek-ai/DeepSeek-R1-0528-dequantized",
    )
)
CACHE_DIR = Path(
    os.getenv(
        "DEEPSEEK_V3_CACHE",
        "/mnt/MLPerf/tt_dnn-models/deepseek-ai/DeepSeek-R1-0528-Cache/dev/",
    )
)
REFERENCE_FILE = Path(os.getenv("DEEPSEEK_V3_REFPT", str(Path(__file__).with_name("gpqa_diamond_racemic.refpt"))))


def _get_tile_aligned_max_seq_len(length: int) -> int:
    k_chunk_size = K_CHUNK_SIZE
    aligned_size = max(int(ttnn.TILE_SIZE), k_chunk_size)
    return ((int(length) + aligned_size - 1) // aligned_size) * aligned_size


def _min_judge_continuation_tokens(judge_tokenizer, continuation: str) -> int:
    return len(judge_tokenizer.encode(continuation, add_special_tokens=False))


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("reference_file", [REFERENCE_FILE])
@pytest.mark.parametrize("max_new_tokens", [128], ids=["128"])
@pytest.mark.parametrize("max_users_per_row", [8], ids=["8"])
def test_demo_autoreg_perplexity_vs_hf_reference(
    reference_file: Path,
    max_new_tokens: int,
    is_ci_env: bool,
    force_recalculate_weight_config: bool,
    max_users_per_row: int,
):
    if os.getenv("DEEPSEEK_PPL_GUARD", "1").strip().lower() in ("0", "false", "no"):
        pytest.skip("DEEPSEEK_PPL_GUARD disabled for this run.")

    if is_ci_env and max_new_tokens != 128:
        pytest.skip("CI runs only the 128-token autoregressive perplexity guard.")

    if not reference_file.exists():
        pytest.fail(
            f"Reference file not found at {reference_file}. "
            "Generate it first (see generate_teacher_forced_file.py) or point DEEPSEEK_V3_REFPT."
        )

    payload = torch.load(reference_file, weights_only=False)
    for key in ("prompt_tokens", "generated_tokens", "tf_prompt_len"):
        assert key in payload, f"Reference file missing '{key}'"

    prompt_text = str(payload.get("prompt", ""))
    if not prompt_text.strip():
        pytest.fail("Reference payload has no 'prompt' string; cannot build judge conditioning text.")

    generated_tokens = payload["generated_tokens"]
    saved_max_new_tokens = int(payload.get("max_new_tokens", max_new_tokens))
    if max_new_tokens > saved_max_new_tokens:
        pytest.fail(
            f"Requested max_new_tokens={max_new_tokens} exceeds reference max_new_tokens={saved_max_new_tokens}."
        )

    requested_system_name = os.getenv("MESH_DEVICE")
    if requested_system_name is None:
        pytest.fail("Environment variable $MESH_DEVICE is not set. Please set it to DUAL, QUAD, TG, or T3K.")

    mesh_shape = system_name_to_mesh_shape(requested_system_name.upper())
    num_users = max_users_per_row * mesh_shape[0]

    ds_tokenizer = load_tokenizer(MODEL_PATH)
    ref_gen_row = generated_tokens[0] if generated_tokens.dim() == 2 else generated_tokens
    hf_gen_ids = ref_gen_row.tolist()[:max_new_tokens]
    hf_continuation = ds_tokenizer.decode(hf_gen_ids, skip_special_tokens=True)

    tf_prompt_len = int(payload["tf_prompt_len"])
    tf_prompt_len_padded = align_prefill_padded_seq_len(tf_prompt_len, mesh_shape[0])
    configured_max_seq_len = _get_tile_aligned_max_seq_len(tf_prompt_len_padded + max_new_tokens)
    if configured_max_seq_len > DEFAULT_MAX_SEQ_LEN:
        pytest.skip(
            f"Autoregressive perplexity guard needs max_seq_len={configured_max_seq_len}, "
            f"which exceeds DEFAULT_MAX_SEQ_LEN={DEFAULT_MAX_SEQ_LEN}."
        )

    max_ratio = float(os.getenv("DEEPSEEK_PPL_MAX_RATIO", "40"))
    if max_ratio <= 1.0:
        pytest.fail("DEEPSEEK_PPL_MAX_RATIO must be > 1.")

    logger.info("=== Autoregressive decode (greedy, sample on host) for perplexity guard ===")
    results = run_demo(
        prompts=[prompt_text] * num_users,
        model_path=MODEL_PATH,
        cache_dir=CACHE_DIR,
        random_weights=False,
        max_new_tokens=max_new_tokens,
        max_seq_len=configured_max_seq_len,
        max_users_per_row=max_users_per_row,
        repeat_batches=1,
        token_accuracy=False,
        enable_trace=True,
        force_recalculate=force_recalculate_weight_config,
        stop_at_eos=False,
        sample_on_device=False,
        sampling_temperature=0.0,
        sampling_top_k=0,
        sampling_top_p=1.0,
    )

    assert results.get("generations"), "run_demo returned no generations"
    tt_tokens = results["generations"][0]["tokens"]
    if len(tt_tokens) != max_new_tokens:
        pytest.fail(
            f"Expected exactly {max_new_tokens} greedy tokens with stop_at_eos=False, got {len(tt_tokens)}. "
            "Partial output makes perplexity comparison ambiguous."
        )
    tt_continuation = ds_tokenizer.decode(tt_tokens[:max_new_tokens], skip_special_tokens=True)

    try:
        judge_model, judge_tokenizer, judge_device = load_judge_lm()
    except Exception as exc:  # pragma: no cover - environment specific
        pytest.skip(f"Could not load perplexity judge model: {exc}")

    min_judge_toks = int(os.getenv("DEEPSEEK_PPL_MIN_JUDGE_TOKENS", "48"))
    if _min_judge_continuation_tokens(judge_tokenizer, hf_continuation) < min_judge_toks:
        pytest.skip(
            f"HF continuation encodes to fewer than {min_judge_toks} judge tokens under "
            f"{os.getenv('DEEPSEEK_PPL_JUDGE_MODEL', DEFAULT_JUDGE_MODEL)}; perplexity would be too noisy."
        )

    nll_hf = mean_nll_continuation_with_prompt(
        judge_model,
        judge_tokenizer,
        judge_device,
        prompt=prompt_text,
        continuation=hf_continuation,
    )
    nll_tt = mean_nll_continuation_with_prompt(
        judge_model,
        judge_tokenizer,
        judge_device,
        prompt=prompt_text,
        continuation=tt_continuation,
    )
    ppl_hf = perplexity_from_mean_nll(nll_hf)
    ppl_tt = perplexity_from_mean_nll(nll_tt)

    logger.info(
        "Perplexity judge ({}): HF ref nll={:.4f} ppl={:.2f} | TT AR nll={:.4f} ppl={:.2f} | ratio={:.2f}",
        os.getenv("DEEPSEEK_PPL_JUDGE_MODEL", DEFAULT_JUDGE_MODEL),
        nll_hf,
        ppl_hf,
        nll_tt,
        ppl_tt,
        (ppl_tt / ppl_hf) if math.isfinite(ppl_hf) and ppl_hf > 0 else float("nan"),
    )

    assert math.isfinite(nll_hf) and math.isfinite(nll_tt), "Judge returned non-finite NLL (sequence too short?)."

    floor = float(os.getenv("DEEPSEEK_PPL_HF_FLOOR", "8"))
    effective_hf = max(ppl_hf, floor)
    ratio = ppl_tt / effective_hf
    assert ratio <= max_ratio, (
        f"TT autoregressive text is much less natural than the HF reference under the judge LM "
        f"(ppl_tt={ppl_tt:.2f} vs ppl_hf={ppl_hf:.2f}, ratio={ratio:.2f} > max_ratio={max_ratio}). "
        "Greedy decode may be broken even if teacher-forcing accuracy still passes. "
        f"Tune DEEPSEEK_PPL_JUDGE_MODEL / DEEPSEEK_PPL_MAX_RATIO if the baseline is ill-suited to your prompt language."
    )
