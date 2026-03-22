# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Simple BGE test with correct device parameters.

Usage:
    cd /home/ubuntu/agentic/tt-metal
    export TT_METAL_HOME=$(pwd) && export PYTHONPATH=$(pwd) && source python_env/bin/activate
    python models/demos/minimax_m2/agentic/tests/test_bge_simple.py
"""

import time

import torch
import transformers
from loguru import logger

import ttnn
from models.demos.bge_large_en.runner.performant_runner import BGEPerformantRunner
from models.demos.sentence_bert.reference.sentence_bert import custom_extended_mask
from models.demos.wormhole.bge_large_en.ttnn.common import BGE_L1_SMALL_SIZE, BGE_SEQ_LENGTH


def model_location_generator(model_version, model_subdir=""):
    from models.common.utility_functions import get_model_prefix

    model_prefix = get_model_prefix()
    return f"{model_prefix}/bge_large_en/{model_version}"


def main():
    logger.info("=" * 60)
    logger.info("Simple BGE Test")
    logger.info("=" * 60)

    # Open single device with BGE-specific parameters
    logger.info("\n[1/4] Opening device with BGE parameters...")
    device = ttnn.open_device(
        device_id=0,
        l1_small_size=BGE_L1_SMALL_SIZE,  # 0 for BGE
        trace_region_size=23887872,
        num_command_queues=2,
    )
    device.enable_program_cache()
    logger.info("Device opened")

    try:
        # Load tokenizer
        logger.info("\n[2/4] Loading tokenizer...")
        MODEL_NAME = "BAAI/bge-large-en-v1.5"
        tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)

        # Tokenize dummy inputs
        texts = ["Hello world"] * 8
        encoded = tokenizer(
            texts,
            padding="max_length",
            max_length=BGE_SEQ_LENGTH,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        token_type_ids = encoded["token_type_ids"]
        extended_mask = custom_extended_mask(attention_mask, dtype=torch.bfloat16)
        position_ids = torch.arange(0, input_ids.shape[-1], dtype=torch.int64).unsqueeze(dim=0)

        # Create runner
        logger.info("\n[3/4] Creating BGE runner...")
        runner = BGEPerformantRunner(
            device=device,
            model_location_generator=model_location_generator,
            input_ids=input_ids,
            extended_mask=extended_mask,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            model_name=MODEL_NAME,
        )

        # Capture trace
        logger.info("Capturing trace...")
        runner._capture_bge_trace_2cqs()
        logger.info("Trace captured")

        # Run inference
        logger.info("\n[4/4] Running inference...")
        start = time.time()
        output = runner.run(input_ids, token_type_ids, position_ids, extended_mask, attention_mask)
        elapsed = time.time() - start

        # Convert output
        embeddings = ttnn.to_torch(output, dtype=torch.float32)
        logger.info(f"Output shape: {embeddings.shape}")
        logger.info(f"Inference time: {elapsed:.3f}s")

        # Cleanup
        runner.release()

    finally:
        logger.info("\nClosing device...")
        ttnn.close_device(device)
        logger.info("Done!")


if __name__ == "__main__":
    main()
