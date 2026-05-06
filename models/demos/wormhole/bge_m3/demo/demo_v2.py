# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Minimal BGE-M3 trace-capture demo for one preprocessed prompt."""

import argparse

from loguru import logger

import ttnn
from models.common.auto_compose import to_torch_auto_compose
from models.demos.wormhole.bge_m3.tt.common import create_tt_model

MODEL_NAME = "BAAI/bge-m3"
PROMPT = "The weather is sunny today with clear blue skies."
BATCH_SIZE = 1
SEQ_LEN = 512


def run_demo(device):
    logger.info("Creating BGE-M3 TT model...")
    model_args, model, _state_dict = create_tt_model(
        mesh_device=device,
        max_batch_size=BATCH_SIZE,
        max_seq_len=SEQ_LEN,
        dtype=ttnn.bfloat8_b,
        hf_model_name=MODEL_NAME,
    )

    logger.info(f"Preprocessing prompt: {PROMPT!r}")
    encoded_prompt = model_args.encode_prompts(PROMPT, prompt_length=SEQ_LEN)
    staged_inputs = encoded_prompt["model_inputs"]
    logger.info(
        f"Prepared input_ids={tuple(encoded_prompt['input_ids'].shape)}, "
        f"attention_mask={tuple(encoded_prompt['attention_mask'].shape)}, "
        f"position_ids={tuple(encoded_prompt['position_ids'].shape)}, "
        f"nonpad_tokens={int(encoded_prompt['tokenizer_attention_mask'].sum().item())}"
    )

    logger.info("Running warmup forward...")
    warmup_output = model(**staged_inputs)
    ttnn.synchronize_device(device)
    logger.info("Warmup forward complete.")
    ttnn.deallocate(warmup_output)

    logger.info("Capturing trace with preprocessed prompt tensors...")
    trace_output = model.capture_trace(
        input_ids=staged_inputs["input_ids"],
        attention_mask=staged_inputs["attention_mask"],
        token_type_ids=staged_inputs["token_type_ids"],
        position_ids=staged_inputs["position_ids"],
        mesh_device=device,
        cq_id=0,
    )
    logger.info("Trace captured successfully.")

    logger.info("Replaying captured trace once...")
    model.execute_trace(blocking=False, synchronize=True)

    hidden_states = to_torch_auto_compose(trace_output, device=device)
    if hidden_states.dim() == 4 and hidden_states.shape[1] == 1:
        hidden_states = hidden_states.squeeze(1)
    logger.info(f"Trace replay output hidden-state shape: {tuple(hidden_states.shape)}")

    return model


def main():
    parser = argparse.ArgumentParser(description="Minimal BGE-M3 preprocessed prompt trace demo")
    parser.add_argument("--device-id", type=int, default=0, help="TT device ID")
    args = parser.parse_args()

    logger.info(f"Opening device {args.device_id}...")
    device = ttnn.open_device(
        device_id=args.device_id,
        l1_small_size=32768,
        trace_region_size=50000000,
        num_command_queues=1,
    )
    try:
        model = run_demo(device)
    finally:
        if "model" in locals():
            model.release_trace()
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
