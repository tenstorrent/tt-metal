# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Store GPU KV heads in the channel order used by Gemma4 validation."""

import argparse
import json
import shutil
import time
from pathlib import Path

import torch
from loguru import logger
from safetensors.torch import save_file

from models.demos.gemma4_d_p.tt.runners.adapters.gemma4 import Gemma4PrefillAdapter, Gemma4ServiceConfig
from models.demos.gemma4_d_p.tt.runners.kv_chunk_table import CONFIG_NAMES
from models.demos.gemma4_d_p.tt.runners.kv_validation import PREPARED_GPU_TRACE_LAYOUT, load_gpu_cache_heads


def prepare_layer(source, destination, layer, tokens):
    started = time.perf_counter()
    expected = load_gpu_cache_heads(source, layer, tokens)
    source_seconds = time.perf_counter() - started
    path = destination / "kv_cache" / f"layer_{layer}.safetensors"
    started = time.perf_counter()
    save_file(
        {CONFIG_NAMES[config]: tensor.bfloat16().contiguous() for config, tensor in expected.items()},
        str(path),
        metadata={"layout": PREPARED_GPU_TRACE_LAYOUT},
    )
    write_seconds = time.perf_counter() - started
    started = time.perf_counter()
    actual = load_gpu_cache_heads(destination, layer, tokens)
    prepared_seconds = time.perf_counter() - started
    for config in expected:
        if not torch.equal(expected[config], actual[config]):
            raise ValueError(f"Layer {layer}: prepared GPU head {CONFIG_NAMES[config]} differs from source")
    return dict(
        layer=layer,
        bytes=path.stat().st_size,
        source_seconds=source_seconds,
        write_seconds=write_seconds,
        prepared_seconds=prepared_seconds,
    )


def prepare_gpu_reference(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    metadata = json.loads((source / "metadata.json").read_text())
    if metadata["model_id"] != Gemma4PrefillAdapter().hf_model_id or metadata["layout"] != "chunked_group_a_v1":
        raise ValueError("Expected a Gemma4-31B-it chunked_group_a_v1 GPU capture")
    if (
        metadata["n_layers"] != Gemma4ServiceConfig.NUM_LAYERS
        or len(metadata["token_ids"]) != Gemma4ServiceConfig.MAX_SEQ_LEN
    ):
        raise ValueError("Expected all 60 layers and 262144 token IDs")
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "kv_cache").mkdir()
    measurements = []
    for layer in range(Gemma4ServiceConfig.NUM_LAYERS):
        result = prepare_layer(source, destination, layer, Gemma4ServiceConfig.MAX_SEQ_LEN)
        measurements.append(result)
        logger.info(f"GPU reference layer {layer + 1}/60 verified: {result}")
    metadata.update(
        layout=PREPARED_GPU_TRACE_LAYOUT,
        source_trace_dir=str(source),
        kv_cache_layout={
            "path": "kv_cache/layer_{layer}.safetensors",
            "tensor_names": list(CONFIG_NAMES),
            "axes": ["token", "channel"],
            "sliding_k": "Adjacent-pair RoPE channel order",
            "sliding_v": "HF channel order",
            "global_kv": "K rotary (128) | V non-rotary (384) | V rotary (128)",
        },
    )
    metadata.pop("chunk_rows", None)
    metadata.pop("decoder_input_chain", None)
    if (source / "input.txt").is_file():
        shutil.copyfile(source / "input.txt", destination / "input.txt")
    (destination / "preparation.json").write_text(json.dumps(measurements, indent=2) + "\n")
    (destination / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    logger.info(f"Prepared GPU reference: {destination}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    prepare_gpu_reference(args.source, args.destination)


if __name__ == "__main__":
    main()
