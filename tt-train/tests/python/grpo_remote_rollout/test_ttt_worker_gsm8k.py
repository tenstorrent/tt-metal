# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Determinism check for the data-parallel TttGenerationWorker.

One GSM8K question, replicated 16x, generated greedily (temperature 0) on a [1, 2]
mesh (8 completions per submesh). All 16 completions must be identical -- otherwise the
two submeshes decoded differently for the same input. Single process, no MPI/bridge.

Run (needs >= 2 chips + HF_TOKEN):
    cd tt-train/tests/python/grpo_remote_rollout
    python3 -m pytest -s test_ttt_worker_gsm8k.py
"""

from __future__ import annotations

import gc

import pytest

MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
MESH_SHAPE = (1, 2)  # data_parallel = 2 submeshes
PER_SUBMESH_BATCH = 16  # global batch = 2 * 16 = 32
NUM_COMPLETIONS = 32
MAX_NEW_TOKENS = 256

# GSM8K train split, first example.
GSM8K_QUESTION = (
    "Natalia sold clips to 48 of her friends in April, and then she sold half as many "
    "clips in May. How many clips did Natalia sell altogether in April and May?"
)


@pytest.mark.timeout(0)
def test_16_greedy_completions_are_identical():
    import ttnn
    from transformers import AutoTokenizer

    from _completer_utils import _TRACE_REGION_SIZE, build_completer

    if len(ttnn.get_device_ids()) < MESH_SHAPE[0] * MESH_SHAPE[1]:
        pytest.skip(f"needs >= {MESH_SHAPE[0] * MESH_SHAPE[1]} chips")

    mesh_device = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(*MESH_SHAPE),
        trace_region_size=_TRACE_REGION_SIZE,
        dispatch_core_config=ttnn.DispatchCoreConfig(ttnn.device.DispatchCoreType.ETH),
    )
    completer = None
    try:
        completer = build_completer(mesh_device, dummy_weights=False, max_batch_size=PER_SUBMESH_BATCH)

        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        prompt_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": GSM8K_QUESTION}],
            tokenize=True,
            add_generation_prompt=True,
        )

        completions = completer.generate(
            [prompt_ids] * NUM_COMPLETIONS,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=0.0,
        )

        assert len(completions) == NUM_COMPLETIONS, f"expected {NUM_COMPLETIONS} completions, got {len(completions)}"
        print(f"\n[gsm8k] Q: {GSM8K_QUESTION}")
        print(
            f"[gsm8k] completion[0] ({len(completions[0])} tok): {tokenizer.decode(completions[0], skip_special_tokens=True)!r}\n"
        )

        for i in range(1, NUM_COMPLETIONS):
            assert completions[i] == completions[0], (
                f"completion {i} differs from completion 0 at temperature 0 "
                f"(lens {len(completions[i])} vs {len(completions[0])}) -> submeshes diverged"
            )
    finally:
        completer = None
        gc.collect()
        ttnn.close_mesh_device(mesh_device)
