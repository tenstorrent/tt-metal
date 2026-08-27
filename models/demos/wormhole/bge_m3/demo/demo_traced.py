# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""BGE-M3 trace-capture demo.

The demo builds the model, runs one warmup forward to compile the kernels,
captures the trace, and then replays the trace to embed the prompts.

Two shapes are available:
  * batch 1, sequence length 512  - replays the trace once per prompt
  * batch 12, sequence length 8192 - the long-context serving shape

Add --data-parallel to run batch 12 across the two chips of an N300. Each chip
then embeds 6 prompts and uses no inter-chip collectives.

Usage:
    TT_VISIBLE_DEVICES=0 python models/demos/wormhole/bge_m3/demo/demo_traced.py
    TT_VISIBLE_DEVICES=0 python models/demos/wormhole/bge_m3/demo/demo_traced.py --batch 12 --seq-len 8192
    TT_VISIBLE_DEVICES=0 python models/demos/wormhole/bge_m3/demo/demo_traced.py --batch 12 --seq-len 8192 --data-parallel
"""

import argparse

import torch
import torch.nn.functional as F
from loguru import logger

import ttnn
from models.common.auto_compose import to_torch_auto_compose
from models.demos.wormhole.bge_m3.tt.common import create_tt_model

MODEL_NAME = "BAAI/bge-m3"
TRACE_REGION_SIZE = 50_000_000

# Each shape allows one sequence length.
SUPPORTED_SHAPES = {1: 512, 12: 8192}

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
    """Take the CLS token and normalize it. This matches HF sentence-transformers."""
    if hidden_states.dim() == 4 and hidden_states.shape[1] == 1:
        hidden_states = hidden_states.squeeze(1)
    cls = hidden_states[:, 0, :].to(torch.float32)
    return F.normalize(cls, p=2, dim=-1)


def log_embedding(index: int, total: int, prompt: str, embedding: torch.Tensor) -> None:
    preview = f"{prompt[:48]}..." if len(prompt) > 48 else prompt
    logger.info(f"  [{index + 1:>2}/{total}] {preview!r}")
    logger.info(f"          embedding[:5] = {[f'{v:.4f}' for v in embedding[:5].tolist()]}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batch", type=int, default=1, choices=sorted(SUPPORTED_SHAPES))
    parser.add_argument("--seq-len", type=int, default=None, help="Defaults to the value the batch allows.")
    parser.add_argument("--data-parallel", action="store_true", help="Shard batch 12 across the 2 chips of an N300.")
    args = parser.parse_args()

    allowed_seq_len = SUPPORTED_SHAPES[args.batch]
    if args.seq_len is None:
        args.seq_len = allowed_seq_len
    if args.seq_len != allowed_seq_len:
        parser.error(f"--batch {args.batch} allows --seq-len {allowed_seq_len}, got {args.seq_len}")
    if args.data_parallel and args.batch != 12:
        parser.error("--data-parallel requires --batch 12")
    return args


def open_device(data_parallel: bool):
    """Open one chip, or a (2, 1) mesh for the data-parallel run."""
    if not data_parallel:
        return ttnn.open_device(device_id=0, trace_region_size=TRACE_REGION_SIZE, num_command_queues=1)

    device = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(2, 1),
        trace_region_size=TRACE_REGION_SIZE,
        num_command_queues=1,
    )
    # Data-parallel needs both chips of an N300. Report the shortfall here
    # rather than let the model raise a less direct error later.
    if device.get_num_devices() != 2:
        num_devices = device.get_num_devices()
        ttnn.close_mesh_device(device)
        raise RuntimeError(
            f"--data-parallel needs 2 chips, but the mesh opened {num_devices}. "
            "Use an N300 and set TT_VISIBLE_DEVICES to one card."
        )
    return device


def main() -> None:
    args = parse_args()
    prompts = PROMPTS[: args.batch]
    assert len(prompts) == args.batch, f"need {args.batch} prompts, got {len(prompts)}"

    device = open_device(args.data_parallel)
    close_device = ttnn.close_mesh_device if args.data_parallel else ttnn.close_device

    try:
        mode = "data-parallel, 2 chips" if args.data_parallel else "single chip"
        logger.info(f"Building BGE-M3: batch {args.batch}, sequence length {args.seq_len} ({mode})")
        model_args, model, _ = create_tt_model(
            mesh_device=device,
            max_batch_size=args.batch,
            max_seq_len=args.seq_len,
            dtype=ttnn.bfloat8_b,
            hf_model_name=MODEL_NAME,
            data_parallel=args.data_parallel,
        )

        # Data-parallel shards the batch along dim 0, so each chip receives
        # batch/2 prompts. The single-chip run keeps the ttnn default.
        mesh_mapper = ttnn.ShardTensorToMesh(device, dim=0) if args.data_parallel else None

        logger.info(f"Encoding {len(prompts)} prompts to [{args.batch}, {args.seq_len}]")
        encoded = model_args.encode_prompts(prompts, prompt_length=args.seq_len, inputs_mesh_mapper=mesh_mapper)
        staged = encoded["model_inputs"]
        if args.data_parallel:
            # The data-parallel path takes compact [B, 1] valid lengths, not the
            # dense mask that encode_prompts stages. The kernels then build only
            # the boundary mask tiles.
            valid_lengths = encoded["tokenizer_attention_mask"].sum(dim=1, keepdim=True)
            staged["attention_mask"] = ttnn.from_torch(
                valid_lengths.int(),
                device=device,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=mesh_mapper,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        logger.info("Running the warmup forward to compile the kernels")
        warmup_output = model(**staged)
        ttnn.synchronize_device(device)
        ttnn.deallocate(warmup_output)

        logger.info("Capturing the trace")
        output_device = model.capture_trace(**staged, mesh_device=device, cq_id=0)

        if args.batch == 1:
            # Replay the trace once per prompt. Each iteration overwrites the
            # input tensors that the trace reads, so the trace stays valid.
            logger.info(f"Replaying the trace for {len(PROMPTS)} prompts")
            for index, prompt in enumerate(PROMPTS):
                encoded = model_args.encode_prompts([prompt], prompt_length=args.seq_len)
                for key, tensor in staged.items():
                    layout = ttnn.TILE_LAYOUT if key == "attention_mask" else ttnn.ROW_MAJOR_LAYOUT
                    dtype = model_args.attention_mask_dtype if key == "attention_mask" else ttnn.uint32
                    source = encoded["attention_mask"] if key == "attention_mask" else encoded[key].int()
                    ttnn.copy_host_to_device_tensor(ttnn.from_torch(source, dtype=dtype, layout=layout), tensor)
                model.execute_trace(blocking=True)
                embeddings = extract_embedding(to_torch_auto_compose(output_device, device=device))
                log_embedding(index, len(PROMPTS), prompt, embeddings[0])
        else:
            logger.info("Replaying the trace once")
            model.execute_trace(blocking=True)
            hidden_states = to_torch_auto_compose(output_device, device=device)
            embeddings = extract_embedding(hidden_states)
            logger.info(f"Hidden states: {tuple(hidden_states.shape)}")
            logger.info(f"Embeddings:    {tuple(embeddings.shape)}")
            for index, prompt in enumerate(prompts):
                log_embedding(index, len(prompts), prompt, embeddings[index])

        model.release_trace()
        logger.info("Done.")
    finally:
        close_device(device)


if __name__ == "__main__":
    main()
