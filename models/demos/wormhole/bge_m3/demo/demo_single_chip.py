# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""BGE-M3 single-chip embedding demo.

Shows the simplest possible customer flow: encode prompts, then call
``model.forward(**inputs, mode="trace")`` once per batch. The first call captures
a device trace; every later call replays it (much faster) -- all hidden behind
the same one-liner.

Usage:
    TT_VISIBLE_DEVICES=0 python models/demos/wormhole/bge_m3/demo/demo_single_chip.py
"""

import torch
import torch.nn.functional as F
from loguru import logger

import ttnn
from models.demos.wormhole.bge_m3.tt.common import create_tt_model

MODEL_NAME = "BAAI/bge-m3"
SEQ_LEN = 512
DTYPE = ttnn.bfloat8_b
HIDDEN = 1024  # BGE-M3 hidden dim

# Batch sizes to demonstrate. Each batch size builds its own model + trace,
# because a captured trace is bound to a fixed input shape.
BATCH_SIZES = [1, 8, 16, 32]

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


def make_batch(prompts: list[str], batch_size: int) -> list[str]:
    """Repeat/truncate the prompt list to exactly ``batch_size`` rows.

    A captured trace is fixed-shape, so each call must feed exactly the batch
    size the model was built with.
    """
    if not prompts:
        raise ValueError("need at least one prompt")
    repeated = (prompts * ((batch_size // len(prompts)) + 1))[:batch_size]
    return repeated


def allocate_d2h_stack(output_dev: ttnn.Tensor, hidden: int):
    """One-time setup for the optimized device->host readback.

    Returns (dram_staging, dest_torch). ``copy_device_to_torch`` later DMAs the
    device result straight into ``dest_torch`` (no intermediate host tensor).
    """
    b, _, s, _ = output_dev.shape
    sample_rm = ttnn.untilize_with_unpadding(
        output_dev, output_tensor_end=(b - 1, 0, s - 1, hidden - 1), use_multicore=True
    )
    dram_staging = ttnn.clone(sample_rm, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(sample_rm)
    dest_torch = torch.empty((b, 1, s, hidden), dtype=torch.bfloat16)
    return dram_staging, dest_torch


def read_output_optimized(output_dev: ttnn.Tensor, dram_staging: ttnn.Tensor, dest_torch: torch.Tensor) -> torch.Tensor:
    """Optimized D2H: untilize_with_unpadding -> DRAM staging copy -> copy_device_to_torch.

    Direct PCIe DMA into the pre-allocated torch tensor -- faster than a generic
    ``ttnn.to_torch`` because it avoids the intermediate host-side tensor.
    """
    b, _, s, _ = output_dev.shape
    out_rm = ttnn.untilize_with_unpadding(
        output_dev, output_tensor_end=(b - 1, 0, s - 1, dest_torch.shape[-1] - 1), use_multicore=True
    )
    ttnn.copy(out_rm, dram_staging)
    ttnn.deallocate(out_rm)
    ttnn.copy_device_to_torch(dram_staging, dest_torch)
    return dest_torch


def run_batch(device, batch_size: int) -> None:
    logger.info(f"=== batch_size={batch_size} ===")

    # Build a model sized for this batch.
    model_args, model, _ = create_tt_model(
        mesh_device=device,
        max_batch_size=batch_size,
        max_seq_len=SEQ_LEN,
        dtype=DTYPE,
        hf_model_name=MODEL_NAME,
    )

    # Replicate the prompt(s) to fill exactly ``batch_size`` rows.
    prompts = make_batch(PROMPTS, batch_size)

    # 1) Encode prompts -> device tensors (tokenize + torch->ttnn handled here).
    inputs = model_args.encode_prompts(prompts, prompt_length=SEQ_LEN)["model_inputs"]

    # 2) Single pass. The first ``mode="trace"`` call warms up (JIT-compiles
    #    kernels) and captures the trace internally, then runs the pass.
    output_dev = model.forward(**inputs, mode="trace")

    # 3) Read the result back to torch using the optimized D2H stack
    #    (untilize_with_unpadding -> DRAM staging -> copy_device_to_torch), then
    #    pool into sentence embeddings.
    dram_staging, dest_torch = allocate_d2h_stack(output_dev, HIDDEN)
    hidden_states = read_output_optimized(output_dev, dram_staging, dest_torch)
    embeddings = extract_embedding(hidden_states)

    preview = [f"{v:.4f}" for v in embeddings[0, :5].tolist()]
    logger.info(f"  output shape: {tuple(embeddings.shape)}  (batch x embedding_dim)")
    logger.info(f"  embedding[:5] = {preview}")

    model.release_trace()


def main():
    device = ttnn.open_device(
        device_id=0,
        trace_region_size=50_000_000,
        num_command_queues=1,
    )
    try:
        for batch_size in BATCH_SIZES:
            run_batch(device, batch_size)
        logger.info("\nDone.")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
