# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Test Sentence BERT with our device parameters (l1_small_size=79104).
"""

import time

import torch
from loguru import logger

import ttnn
from models.demos.sentence_bert.runner.performant_runner import SentenceBERTPerformantRunner


def model_location_generator(model_version, model_subdir=""):
    from models.common.utility_functions import get_model_prefix

    model_prefix = get_model_prefix()
    return f"{model_prefix}/sentence_bert/{model_version}"


def main():
    logger.info("=" * 60)
    logger.info("Sentence BERT Test (l1_small_size=79104)")
    logger.info("=" * 60)

    # Open device with our parameters
    logger.info("\n[1/3] Opening device...")
    device = ttnn.open_device(
        device_id=0,
        l1_small_size=79104,
        trace_region_size=100_000_000,
        num_command_queues=2,
    )
    device.enable_program_cache()
    logger.info("Device opened")

    try:
        # Create runner
        logger.info("\n[2/3] Creating Sentence BERT runner...")
        runner = SentenceBERTPerformantRunner(
            device=device,
            device_batch_size=8,
            sequence_length=384,
            model_location_generator=model_location_generator,
        )

        # Capture trace
        logger.info("Capturing trace...")
        runner._capture_sentencebert_trace_2cqs()
        logger.info("Trace captured")

        # Run inference
        logger.info("\n[3/3] Running inference...")
        start = time.time()
        for i in range(5):
            output = runner.run()
        elapsed = time.time() - start

        # Convert output
        embeddings = ttnn.to_torch(output, dtype=torch.float32)
        logger.info(f"Output shape: {embeddings.shape}")
        logger.info(f"5 iterations in {elapsed:.3f}s ({elapsed/5*1000:.1f}ms per batch)")

        # Cleanup
        runner.release()
        logger.info("SUCCESS!")

    except Exception as e:
        logger.error(f"FAILED: {e}")
        raise

    finally:
        logger.info("\nClosing device...")
        ttnn.close_device(device)
        logger.info("Done!")


if __name__ == "__main__":
    main()
