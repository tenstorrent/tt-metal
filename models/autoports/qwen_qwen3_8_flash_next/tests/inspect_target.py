"""Inspect the pinned checkpoint without constructing its 95 GiB PLE table.

Run from the checkout with ``python models/autoports/qwen_qwen3_8_flash_next/
tests/inspect_target.py``. This is contract evidence, not a decoder test.
"""

import hashlib
import inspect
import json
import math
from pathlib import Path

import transformers
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoConfig
from transformers.models.qwen4_exp import modeling_qwen4_exp

MODEL = "Qwen/Qwen3.8-Flash-Next"
REVISION = "de4b8e4d43b917e7706784d8bb445c9af86a3540"
ROOT = Path(__file__).resolve().parents[1]


def main():
    checkpoint = Path(snapshot_download(MODEL, revision=REVISION, local_files_only=True))
    config = AutoConfig.from_pretrained(checkpoint).text_config
    weight_map = json.loads((checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    prefixes = [f"model.language_model.layers.{i}." for i in (0, 1, 3)]
    keys = [k for k in weight_map if any(k.startswith(p) for p in prefixes)]
    tensors = {}
    for shard in sorted({weight_map[k] for k in keys}):
        with safe_open(checkpoint / shard, framework="pt", device="cpu") as f:
            for key in keys:
                if weight_map[key] == shard:
                    tensor = f.get_slice(key)
                    tensors[key] = {"shape": tensor.get_shape(), "dtype": tensor.get_dtype(), "shard": shard}
    ple = [v for k, v in tensors.items() if "ngram_embedding.shard_" in k]
    elements = sum(math.prod(t["shape"]) for t in ple)
    source = Path(inspect.getfile(modeling_qwen4_exp))
    evidence = {
        "model": MODEL,
        "revision": REVISION,
        "checkpoint": str(checkpoint),
        "transformers_version": transformers.__version__,
        "reference_source": str(source),
        "reference_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "config_sha256": hashlib.sha256((checkpoint / "config.json").read_bytes()).hexdigest(),
        "hf_advertised_context": config.max_position_embeddings,
        "decoder_input_output_width": config.hidden_size * config.hc_count,
        "layer_kinds": {"0": "linear_attention", "1": "linear_attention_with_ple", "3": "indexed_attention"},
        "ple_table_elements": elements,
        "ple_table_bf16_bytes": elements * 2,
        "ple_table_tile_storage_bytes": {
            name: sum(math.ceil(t["shape"][0] / 32) * math.ceil(t["shape"][1] / 32) * size for t in ple)
            for name, size in (("bfloat16", 2048), ("bfloat8_b", 1088), ("bfloat4_b", 576), ("bfloat2_b", 320))
        },
        "layer_1_non_table_bf16_bytes": sum(
            math.prod(v["shape"]) * 2
            for k, v in tensors.items()
            if k.startswith(prefixes[1]) and "ngram_embedding.shard_" not in k and v["dtype"] == "BF16"
        ),
        "tensors": tensors,
    }
    dest = ROOT / "doc/functional_decoder/target_inspection.json"
    dest.write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps({k: v for k, v in evidence.items() if k != "tensors"}, indent=2))


if __name__ == "__main__":
    main()
