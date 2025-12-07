# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path

import pytest
import torch
from transformers import AutoTokenizer

from models.demos.deepseek_v3.demo.demo import run_demo

MODEL_PATH = Path(
    os.getenv(
        "DEEPSEEK_V3_HF_MODEL",
        "/mnt/MLPerf/tt_dnn-models/deepseek-ai/DeepSeek-R1-0528",
    )
)
CACHE_DIR = Path(
    os.getenv(
        "DEEPSEEK_V3_CACHE",
        "/mnt/MLPerf/tt_dnn-models/deepseek-ai/DeepSeek-R1-0528-Cache",
    )
)

# Reference file lives next to this script (and next to the test file)
REFERENCE_FILE = Path(__file__).with_name("deepseek_v3_teacher_forcing.refpt")

TEST_PROMPT = "What is the capital of France? Please provide a brief answer."


def generate_reference(
    max_new_tokens: int = 32,
    override_num_layers: int = 5,
    reference_file: Path = REFERENCE_FILE,
) -> Path:
    """
    Phase 1: Run DeepSeek v3 normally to generate reference tokens and
    store them in a file that will be used later by the pytest.

    The saved file includes:
      - reference_tokens: full prompt+generated sequence (1 x T tensor)
      - tf_prompt_len: length of the prompt in tokens
      - max_new_tokens: used when the reference was generated
      - prompt: the text prompt used for generation
    """

    print("\n=== Phase 1: Generate reference tokens with TT model ===")
    print(f"Using model_path={MODEL_PATH}")
    print(f"Using cache_dir={CACHE_DIR}")
    print(f"Prompt: {TEST_PROMPT!r}")

    results = run_demo(
        prompts=[TEST_PROMPT],
        model_path=MODEL_PATH,
        cache_dir=CACHE_DIR,
        random_weights=False,
        max_new_tokens=max_new_tokens,
        override_num_layers=override_num_layers,
        repeat_batches=1,
        token_accuracy=False,  # Normal generation, no teacher forcing
    )

    assert "generations" in results, "run_demo results must contain 'generations'"
    assert len(results["generations"]) > 0, "Expected at least one generation"

    generated_tokens = results["generations"][0]["tokens"]
    print(f"Phase 1: Generated {len(generated_tokens)} tokens")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    raw_prompt_tokens = tokenizer.encode(TEST_PROMPT, add_special_tokens=False)

    full_sequence = raw_prompt_tokens + generated_tokens
    reference_tokens_tensor = torch.tensor(full_sequence, dtype=torch.long).unsqueeze(0)

    reference_file.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "reference_tokens": reference_tokens_tensor,
        "tf_prompt_len": len(raw_prompt_tokens),
        "max_new_tokens": max_new_tokens,
        "prompt": TEST_PROMPT,
    }

    torch.save(payload, reference_file)

    print(
        f"Phase 1: Created reference file with {len(full_sequence)} tokens "
        f"(prompt={len(raw_prompt_tokens)}, generated={len(generated_tokens)})"
    )
    print(f"Reference file saved to: {reference_file}")

    return reference_file


@pytest.mark.parametrize(
    "max_new_tokens, override_num_layers",
    [
        pytest.param(32, 5, id="32tokens_5layers"),
    ],
)
def test_generate_reference_file(max_new_tokens, override_num_layers):
    """
    Test that generates the reference file for teacher forcing tests.

    This test generates reference tokens by running the model normally
    and saves them to a file that can be used by test_demo_teacher_forced.py.
    """
    path = generate_reference(
        max_new_tokens=max_new_tokens,
        override_num_layers=override_num_layers,
    )
    assert path.exists(), f"Reference file should exist at {path}"
    print(f"\nDone. Reference saved to: {path}")


if __name__ == "__main__":
    path = generate_reference()
    print(f"\nDone. Reference saved to: {path}")
