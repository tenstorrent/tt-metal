# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Record target shapes and storage from safetensors headers without loading weights."""

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import struct
from pathlib import Path

from transformers import AutoConfig
from transformers.models.qwen4_exp import modeling_qwen4_exp


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = AutoConfig.from_pretrained(args.snapshot, local_files_only=True)
    index = json.loads((args.snapshot / "model.safetensors.index.json").read_text())["weight_map"]
    headers = {}
    layers = {}
    for layer in (0, 1, 3):
        records = {}
        prefix = f"model.language_model.layers.{layer}."
        for key, filename in index.items():
            if not key.startswith(prefix):
                continue
            if filename not in headers:
                with (args.snapshot / filename).open("rb") as stream:
                    size = struct.unpack("<Q", stream.read(8))[0]
                    headers[filename] = json.loads(stream.read(size))
            item = headers[filename][key]
            records[key.removeprefix(prefix)] = {
                "shape": item["shape"],
                "dtype": item["dtype"],
                "bytes": item["data_offsets"][1] - item["data_offsets"][0],
                "shard": filename,
            }
        layers[str(layer)] = {
            "kind": config.text_config.layer_types[layer],
            "has_ple": layer + 1 in config.text_config.ple_layer_ids,
            "total_weight_bytes": sum(item["bytes"] for item in records.values()),
            "weights": records,
        }
    model_source = Path(inspect.getfile(modeling_qwen4_exp))
    report = {
        "model_id": "Qwen/Qwen3.8-Flash-Next",
        "revision": args.snapshot.name,
        "snapshot": str(args.snapshot),
        "config_sha256": hashlib.sha256((args.snapshot / "config.json").read_bytes()).hexdigest(),
        "reference_source": str(model_source),
        "reference_sha256": hashlib.sha256(model_source.read_bytes()).hexdigest(),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("transformers", "torch", "huggingface-hub", "ttnn", "tt-perf-report")
        },
        "context": config.text_config.max_position_embeddings,
        "hidden_size": config.text_config.hidden_size,
        "residual_width": config.text_config.hidden_size * config.text_config.hc_count,
        "layers": layers,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "context": report["context"],
                "layer_weight_bytes": {k: v["total_weight_bytes"] for k, v in layers.items()},
            }
        )
    )


if __name__ == "__main__":
    main()
