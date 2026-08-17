"""Generate auditable statistics for every real checkpoint tensor consumed by layers 0 and 3."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import snapshot_download
from safetensors import safe_open

MODEL_ID = "Qwen/Qwen3.8-27B"
LAYER_KINDS = {0: "linear_attention", 3: "full_attention"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    root = Path(
        snapshot_download(
            MODEL_ID,
            allow_patterns=["config.json", "model.safetensors.index.json", "model-*.safetensors"],
            local_files_only=True,
        )
    ).resolve()
    index_path = root / "model.safetensors.index.json"
    weight_map = json.loads(index_path.read_text())["weight_map"]
    records = []
    source_shards = set()
    for layer_idx, layer_kind in LAYER_KINDS.items():
        prefix = f"model.language_model.layers.{layer_idx}."
        keys = sorted(key for key in weight_map if key.startswith(prefix))
        by_shard: dict[str, list[str]] = {}
        for key in keys:
            by_shard.setdefault(weight_map[key], []).append(key)
        for shard_name, shard_keys in sorted(by_shard.items()):
            source_shards.add(shard_name)
            with safe_open(root / shard_name, framework="pt", device="cpu") as shard:
                for key in shard_keys:
                    tensor = shard.get_tensor(key)
                    values = tensor.float()
                    records.append(
                        {
                            "layer_idx": layer_idx,
                            "layer_kind": layer_kind,
                            "name": key,
                            "shape": list(tensor.shape),
                            "dtype": str(tensor.dtype),
                            "mean": values.mean().item(),
                            "std": values.std(unbiased=False).item(),
                            "source_shard": shard_name,
                        }
                    )
                    del values, tensor
    output = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": MODEL_ID,
        "checkpoint_revision": root.name,
        "checkpoint_snapshot_path": str(root),
        "index_file": index_path.name,
        "index_sha256": sha256(index_path),
        "source_shards": sorted(source_shards),
        "statistics": {
            "mean": "arithmetic mean after float32 conversion",
            "std": "population standard deviation (unbiased=False) after float32 conversion",
        },
        "scope": "Every canonical checkpoint tensor consumed by FunctionalDecoder.from_state_dict for representative layer kinds 0 and 3.",
        "tensor_count": len(records),
        "tensors": records,
    }
    output_path = Path(__file__).with_name("real_weight_stats.json")
    output_path.write_text(json.dumps(output, indent=2) + "\n")
    print(f"wrote {len(records)} tensor records to {output_path}")


if __name__ == "__main__":
    main()
