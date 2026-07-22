# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""End-to-end regression test for the P100 QKV prefill matmul override.

Guards the P100-specific ``per_core_M=1`` override at
``models/tt_transformers/tt/model_config.py`` (search "P100 runs OOM in L1 with
8 per_core_M") for the ``MatmulMultiCoreReuseMultiCastProgramConfig`` used by
the QKV projection during prefill.

Without a divisor-of-Mt ``per_core_M`` on P100 the L1-sharded 2D-mcast compute
kernel writes stale L1 into the padded tail of the QKV output shard. The
corruption compounds across all model layers and shows up as garbled generated
tokens end-to-end (originally observed on BoolQ prompts <=128 tokens on P100
with Llama-3.2-1B).

Why the existing PCC-based tests do NOT catch this:

* ``test_decoder_prefill.py`` and ``test_model_prefill.py`` compare only the
  last valid tile row of the block/model output. Garbage padded rows are never
  read, so PCC stays high.
* ``simple_text_demo.py::ci-token-matching`` teacher-forces reference tokens
  each decode step, so per-step corruption never compounds into visible token
  bit-flips.

This test is the smallest artifact that reproduces the failure: greedy
generation on a fixed short prompt, ALL model layers, NO teacher forcing,
first ``NUM_MATCH_TOKENS`` generated tokens must match a checked-in golden
list captured on P100 with the working config.

Bootstrap / update the golden by running with ``PYTEST_UPDATE_GOLDEN=1`` set:
the test will (re)write the golden JSON next to itself and skip the assertion.
"""

import json
import os
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.tt_transformers.tt.common import PagedAttentionConfig, create_tt_model, sample_host
from models.tt_transformers.tt.generator import Generator
from models.tt_transformers.tt.model_config import DecodersPrecision

# Short deterministic prompt. Must tokenize to <=128 tokens so prefill lands in
# the 128 bucket (see get_padded_prefill_len in models/tt_transformers/tt/common.py).
PROMPT = "Question: Is the sky blue? Answer with yes or no and give one short reason."

# Number of tokens to compare against the golden. Kept small so bfloat8_b
# numerical drift beyond ~8 tokens does not turn this into a flake.
NUM_MATCH_TOKENS = 8

# Prefill bucket we expect to hit; asserted after tokenization.
EXPECTED_PREFILL_BUCKET = 128

GOLDEN_FILE = Path(__file__).parent / "reference_outputs" / "llama_3_2_1b_p100_qkv_prefill_regression_golden.json"


@torch.no_grad()
@pytest.mark.timeout(300)
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize("device_params", [{"fabric_config": True}], indirect=True)
def test_p100_qkv_prefill_regression(mesh_device, reset_seeds, ensure_gc):
    """Greedy end-to-end token match on P100 for the 128-bucket QKV prefill config."""

    # 1) Build the full model (all layers - the QKV corruption must compound).
    model_args, tt_model, tt_kv_cache, _ = create_tt_model(
        mesh_device,
        instruct=True,
        max_batch_size=1,
        optimizations=lambda m: DecodersPrecision.performance(m.n_layers, m.model_name),
        max_seq_len=1024,
        paged_attention_config=PagedAttentionConfig(block_size=32, max_num_blocks=1024),
        dtype=ttnn.bfloat8_b,
        num_layers=None,  # full model - corruption compounds across layers
    )

    # The golden below was captured on P100; bfloat8_b numerics differ enough
    # across BH SKUs (P100 vs P150 vs P300) that a shared golden would flake.
    # If someone runs this on another device by mistake we skip rather than
    # produce a false failure.
    if model_args.device_name != "P100":
        pytest.skip(
            f"P100-specific regression (device_name={model_args.device_name}). "
            "The golden token list was captured on P100 and is not portable to "
            "other Blackhole SKUs due to bfloat8_b numerical drift."
        )

    # 2) Tokenize prompt. Must land in the 128 prefill bucket to exercise the
    # QKV MatmulMultiCoreReuseMultiCastProgramConfig branch we are guarding.
    encoded = model_args.encode_prompt(PROMPT, instruct=True)
    prompt_len = len(encoded)
    assert prompt_len <= EXPECTED_PREFILL_BUCKET, (
        f"Prompt tokenized to {prompt_len} tokens; must be <= "
        f"{EXPECTED_PREFILL_BUCKET} to hit the 128 prefill bucket that the "
        "P100 QKV prefill config guards."
    )
    logger.info(f"Prompt: {PROMPT!r}  -> {prompt_len} tokens")

    generator = Generator([tt_model], [model_args], mesh_device, tokenizer=model_args.tokenizer)

    # Paged-attention page table (identity permutation is fine; we only need
    # any valid mapping).
    permutation = torch.randperm(1024)
    reverse_permutation = torch.argsort(permutation)
    page_table = reverse_permutation.reshape(model_args.max_batch_size, -1)

    # 3) Prefill (exercises the P100-guarded QKV matmul at seq_len=128).
    prefill_input = torch.tensor(encoded, dtype=torch.int32).unsqueeze(0)  # [1, prompt_len]
    prefill_logits = generator.prefill_forward_text(
        prefill_input,
        page_table=page_table,
        kv_cache=[tt_kv_cache],
        prompt_lens=[prompt_len],
    )
    # prefill_forward_text returns logits [batch, 1, vocab_size]; argmax -> [batch, 1]
    generated_tokens = [int(torch.argmax(prefill_logits, dim=-1).item())]
    logger.info(f"Prefill token: {generated_tokens[0]}")

    # 4) Greedy decode loop. NO teacher forcing -- corruption at any step
    # propagates into the KV cache and biases subsequent tokens, so token
    # drift becomes visible.
    out_tok = torch.tensor([[generated_tokens[0]]], dtype=torch.int32)  # [1, 1]
    current_pos = torch.tensor([prompt_len], dtype=torch.int64)
    for step in range(1, NUM_MATCH_TOKENS):
        logits, _ = generator.decode_forward(
            out_tok,
            current_pos,
            page_table=page_table,
            kv_cache=[tt_kv_cache],
            enable_trace=True,
            reset_batch=(step == 1),  # first decode step needs a batch reset
            sampling_params=None,  # host sampling -> returns (logits, log_probs)
        )
        # Greedy: argmax on the returned logits [batch, 1, vocab_size].
        _, out_tok = sample_host(logits, temperature=0, top_p=0.0, on_host=True)
        out_tok = out_tok.to(torch.int32)
        generated_tokens.append(int(out_tok[0, 0].item()))
        current_pos += 1

    decoded = model_args.tokenizer.decode(generated_tokens)
    logger.info(f"Generated tokens: {generated_tokens}")
    logger.info(f"Decoded text:     {decoded!r}")

    # 5) Golden comparison / bootstrap.
    if os.environ.get("PYTEST_UPDATE_GOLDEN") == "1":
        GOLDEN_FILE.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN_FILE.write_text(
            json.dumps(
                {
                    "prompt": PROMPT,
                    "prompt_len": prompt_len,
                    "tokens": generated_tokens,
                    "decoded": decoded,
                },
                indent=2,
            )
            + "\n"
        )
        logger.warning(
            f"PYTEST_UPDATE_GOLDEN=1: wrote {GOLDEN_FILE.name} and skipped "
            "assertion. Rerun without PYTEST_UPDATE_GOLDEN to enforce."
        )
        pytest.skip("Golden updated; rerun without PYTEST_UPDATE_GOLDEN.")

    assert GOLDEN_FILE.exists(), (
        f"Golden file missing: {GOLDEN_FILE}. Bootstrap it by running this "
        "test once with PYTEST_UPDATE_GOLDEN=1 on P100 with the working "
        "QKV prefill config (per_core_M=1)."
    )
    golden = json.loads(GOLDEN_FILE.read_text())
    assert generated_tokens == golden["tokens"], (
        f"Generated tokens do not match the P100 QKV prefill golden.\n"
        f"  got:    {generated_tokens}\n"
        f"  golden: {golden['tokens']}\n"
        f"  decoded:        {decoded!r}\n"
        f"  golden decoded: {golden.get('decoded')!r}\n"
        "This is the exact signature of the per_core_M-not-a-divisor-of-Mt "
        "corruption in the QKV prefill matmul. Confirm the P100 override in "
        "models/tt_transformers/tt/model_config.py (search 'P100 runs OOM in "
        "L1 with 8 per_core_M') is intact and unchanged."
    )
