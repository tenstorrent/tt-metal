import json
from pathlib import Path

import pytest

from models.demos.deepseek_v3.demo.demo import run_demo

# Paths
MODEL_PATH = Path("/proj_sw/user_dev/deepseek-ai/DeepSeek-R1-0528")
CACHE_DIR = Path("/proj_sw/user_dev/deepseek-v3-cache")
REFERENCE_JSON = Path("models/demos/deepseek_v3/demo/deepseek_32_prompts_outputs.json")

# Limit of prompts to pass in one run (demo supports up to 32)
MAX_PROMPTS = 32


# Reference JSON loading
def load_reference_map(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError("Reference JSON must be a JSON object mapping prompts to expected texts.")
    ref_map: dict[str, str] = {}
    for k, v in raw.items():
        if isinstance(v, str):
            ref_map[k] = v
        elif isinstance(v, dict) and "text" in v and isinstance(v["text"], str):
            ref_map[k] = v["text"]
        else:
            raise ValueError(f"Invalid reference for prompt {k!r}: expected string or object with 'text' field.")
    if not ref_map:
        raise ValueError("Reference JSON is empty; nothing to test.")
    return ref_map


@pytest.mark.parametrize(
    "max_new_tokens",
    [200],
)
@pytest.mark.parametrize(
    "num_runs",
    [1],
)
def test_multi_prompt_generation_matches_reference(max_new_tokens, num_runs):
    """
    Loads a JSON dict {prompt: expected_text}, executes the demo ONCE with up to 32 prompts
    by calling run_demo(), and validates each returned generation['text'] against its reference.

    Defaults to case/punctuation-sensitive exact match (WER==0); see flags above.
    """
    ref_map = load_reference_map(REFERENCE_JSON)

    # Ensure demo's 32-prompt limit
    all_prompts = list(ref_map.keys())
    prompts = all_prompts[:MAX_PROMPTS]

    # Ensure at least one prompt
    assert len(prompts) > 0, "No prompts found in reference JSON."

    # Run the demo once with ALL prompts
    results = run_demo(
        prompts=prompts,
        model_path=str(MODEL_PATH),
        max_new_tokens=max_new_tokens,
        cache_dir=str(CACHE_DIR),
        random_weights=False,
        token_accuracy=False,
        early_print_first_user=False,
        num_runs=num_runs,
        validate_against_ref=True,
        reference_texts=ref_map,
    )
