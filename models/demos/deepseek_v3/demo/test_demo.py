# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

import pytest

from models.demos.deepseek_v3.demo.demo import run_demo


@pytest.mark.timeout(180)
def test_random_single_layer_generates_16_tokens(tmp_path):
    """Quick smoketest: run demo in random single-layer mode and expect 16 token IDs.

    - Uses the in-repo reference config (no tokenizer/safetensors required)
    - Writes cache to a temp directory to avoid polluting the workspace
    """

    model_path = "models/demos/deepseek_v3/reference"
    cache_dir = tmp_path / "cache"

    result = run_demo(
        None,
        model_path=model_path,
        cache_dir=str(cache_dir),
        random_weights=True,
        single_layer="mlp",
        max_new_tokens=16,
    )

    tokens = result.get("tokens")
    assert isinstance(tokens, list)
    assert len(tokens) == 16, f"Expected 16 tokens, got {len(tokens)}"
