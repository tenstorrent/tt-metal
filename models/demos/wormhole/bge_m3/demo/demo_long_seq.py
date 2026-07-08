# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""BGE-M3 long-sequence demo: batch 12, sequence length 8192.

This is the long-context serving shape optimized for a single Wormhole N300
chip (64 cores). It shows the minimal trace-capture flow: build the model,
run a SINGLE warmup forward (JIT compile), capture the trace, then a SINGLE
trace replay (one forward pass) that produces the encoder hidden states.

Usage:
    TT_VISIBLE_DEVICES=0 python models/demos/wormhole/bge_m3/demo/demo_long_seq.py
"""

import torch
import torch.nn.functional as F
from loguru import logger

import ttnn
from models.common.auto_compose import to_torch_auto_compose
from models.demos.wormhole.bge_m3.tt.common import create_tt_model

MODEL_NAME = "BAAI/bge-m3"
BATCH_SIZE = 12
SEQ_LEN = 8192

# Device launch parameters for the B12/S8192 config (must match the perf test
# ``test_embedding_perf_b12_s8192``). Single command queue + a trace region
# large enough to hold the captured 24-layer encoder program.
TRACE_REGION_SIZE = 50_000_000
NUM_COMMAND_QUEUES = 1

# 12 prompts -> one per batch row. Anything shorter is padded up to SEQ_LEN.
PROMPTS = [
    "Artificial intelligence is transforming how we interact with technology.",
    "The weather is sunny today with clear blue skies.",
    "Quantum computing promises to solve problems classical computers cannot.",
    "A cat sat on a warm windowsill watching birds outside.",
    "The stock market closed higher today after strong earnings reports.",
    "Machine learning algorithms are revolutionizing data analysis.",
    "Deep learning networks can process complex patterns in data.",
    "Neural networks mimic the human brain's structure and function.",
    "Natural language processing enables computers to understand text.",
    "Computer vision allows machines to interpret visual information.",
    "Renewable energy sources are becoming increasingly cost effective.",
    "The ancient library held thousands of handwritten manuscripts.",
]


def extract_embedding(hidden_states: torch.Tensor) -> torch.Tensor:
    """CLS pooling + L2 normalize (matches HF sentence-transformers BGE-M3)."""
    if hidden_states.dim() == 4 and hidden_states.shape[1] == 1:
        hidden_states = hidden_states.squeeze(1)
    cls = hidden_states[:, 0, :].to(torch.float32)
    return F.normalize(cls, p=2, dim=-1)


def main():
    assert len(PROMPTS) == BATCH_SIZE, f"expected {BATCH_SIZE} prompts, got {len(PROMPTS)}"

    device = ttnn.open_device(
        device_id=0,
        trace_region_size=TRACE_REGION_SIZE,
        num_command_queues=NUM_COMMAND_QUEUES,
    )

    try:
        # ── Create model (B12/S8192, bf8) ──
        logger.info(f"Creating BGE-M3 model (B{BATCH_SIZE} S{SEQ_LEN})...")
        model_args, model, _ = create_tt_model(
            mesh_device=device,
            max_batch_size=BATCH_SIZE,
            max_seq_len=SEQ_LEN,
            dtype=ttnn.bfloat8_b,
            hf_model_name=MODEL_NAME,
        )

        # ── Tokenize + stage device inputs via model_config.encode_prompts ──
        # prompt_length=SEQ_LEN pads/truncates all 12 prompts to the fixed 8192
        # trace shape. ``model_inputs`` are long-lived device tensors the trace
        # binds to; replay reads from these same addresses.
        logger.info(f"Encoding {BATCH_SIZE} prompts to fixed shape [B{BATCH_SIZE}, S{SEQ_LEN}]...")
        encoded = model_args.encode_prompts(PROMPTS, prompt_length=SEQ_LEN)
        staged = encoded["model_inputs"]

        # ── Single warmup forward (JIT compile) ──
        logger.info("Warmup forward (compile)...")
        warmup_out = model(**staged)
        ttnn.synchronize_device(device)
        ttnn.deallocate(warmup_out)

        # ── Capture trace (records the program at fixed device addresses) ──
        logger.info("Capturing trace...")
        output_dev = model.capture_trace(
            input_ids=staged["input_ids"],
            attention_mask=staged["attention_mask"],
            token_type_ids=staged["token_type_ids"],
            position_ids=staged["position_ids"],
            mesh_device=device,
            cq_id=0,
        )

        # ── Single trace replay: one forward pass ──
        logger.info("Executing trace (single forward pass)...")
        model.execute_trace(blocking=True)

        hidden_states = to_torch_auto_compose(output_dev, device=device)
        embeddings = extract_embedding(hidden_states)

        logger.info(f"Encoder hidden-state shape: {tuple(hidden_states.shape)}")
        logger.info(f"Pooled embedding shape: {tuple(embeddings.shape)}")
        for i, prompt in enumerate(PROMPTS):
            preview = f"{prompt[:48]}..." if len(prompt) > 48 else prompt
            logger.info(f"  [{i+1:>2}/{BATCH_SIZE}] {preview!r}")
            logger.info(f"          embedding[:5] = {[f'{v:.4f}' for v in embeddings[i, :5].tolist()]}")

        model.release_trace()
        logger.info("\nDone.")

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
