# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""BGE-M3 embedding demo: warmup, capture trace, sweep 5 prompts via trace replay.

Usage:
    TT_VISIBLE_DEVICES=0 python models/demos/wormhole/bge_m3/demo/demo_v2.py
"""

import torch
import torch.nn.functional as F
from loguru import logger

import ttnn
from models.common.auto_compose import to_torch_auto_compose
from models.demos.wormhole.bge_m3.tt.common import create_tt_model

MODEL_NAME = "BAAI/bge-m3"
BATCH_SIZE = 1
SEQ_LEN = 512

PROMPTS = [
    "Artificial intelligence is transforming how we interact with technology.",
    "The weather is sunny today with clear blue skies.",
    "Quantum computing promises to solve problems classical computers cannot.",
    "A cat sat on a warm windowsill watching birds outside.",
    "The stock market closed higher today after strong earnings reports.",
]


def extract_embedding(hidden_states: torch.Tensor) -> torch.Tensor:
    """CLS pooling + L2 normalize (matches HF sentence-transformers BGE-M3)."""
    if hidden_states.dim() == 4 and hidden_states.shape[1] == 1:
        hidden_states = hidden_states.squeeze(1)
    cls = hidden_states[:, 0, :].to(torch.float32)
    return F.normalize(cls, p=2, dim=-1)


def main():
    device = ttnn.open_device(
        device_id=0,
        trace_region_size=50_000_000,
        num_command_queues=1,
    )

    try:
        # ── Create model ──
        logger.info("Creating BGE-M3 model...")
        model_args, model, _ = create_tt_model(
            mesh_device=device,
            max_batch_size=BATCH_SIZE,
            max_seq_len=SEQ_LEN,
            dtype=ttnn.bfloat8_b,
            hf_model_name=MODEL_NAME,
        )

        # ── Warmup (JIT compile) ──
        logger.info("Warmup forward...")
        encoded = model_args.encode_prompts(["warmup"], prompt_length=SEQ_LEN)
        staged = encoded["model_inputs"]
        input_ids_dev = staged["input_ids"]
        attention_mask_dev = staged["attention_mask"]
        token_type_ids_dev = staged["token_type_ids"]
        position_ids_dev = staged["position_ids"]

        warmup_out = model(**staged)
        ttnn.synchronize_device(device)
        ttnn.deallocate(warmup_out)

        # ── Capture trace ──
        logger.info("Capturing trace...")
        output_dev = model.capture_trace(
            input_ids=input_ids_dev,
            attention_mask=attention_mask_dev,
            token_type_ids=token_type_ids_dev,
            position_ids=position_ids_dev,
            mesh_device=device,
            cq_id=0,
        )
        logger.info("Trace captured. Sweeping prompts via replay:\n")

        # ── Sweep prompts: same trace, new data each time ──
        # The trace is bound to the device tensor addresses from capture.
        # To feed new inputs, we overwrite those tensors in-place with
        # copy_host_to_device_tensor (host CPU → same device address).
        # This avoids reallocation and lets the trace replay read fresh data.
        for i, prompt in enumerate(PROMPTS):
            enc = model_args.encode_prompts([prompt], prompt_length=SEQ_LEN)

            ttnn.copy_host_to_device_tensor(
                ttnn.from_torch(enc["input_ids"].int(), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT),
                input_ids_dev,
            )
            ttnn.copy_host_to_device_tensor(
                ttnn.from_torch(
                    enc["attention_mask"].bfloat16(), dtype=model_args.attention_mask_dtype, layout=ttnn.TILE_LAYOUT
                ),
                attention_mask_dev,
            )
            ttnn.copy_host_to_device_tensor(
                ttnn.from_torch(enc["token_type_ids"].int(), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT),
                token_type_ids_dev,
            )
            ttnn.copy_host_to_device_tensor(
                ttnn.from_torch(enc["position_ids"].int(), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT),
                position_ids_dev,
            )

            model.execute_trace(blocking=True)

            hidden_states = to_torch_auto_compose(output_dev, device=device)
            emb = extract_embedding(hidden_states)
            logger.info(f"  [{i+1}/{len(PROMPTS)}] {prompt!r}")
            logger.info(f"         embedding[:5] = {[f'{v:.4f}' for v in emb[0, :5].tolist()]}")

        model.release_trace()
        logger.info("\nDone.")

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
