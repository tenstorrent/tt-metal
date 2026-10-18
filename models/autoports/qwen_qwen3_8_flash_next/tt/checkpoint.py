"""Setup-only lazy checkpoint access; no model weight payloads are kept in git."""
import json
from collections.abc import Mapping
from pathlib import Path


class CheckpointState(Mapping):
    def __init__(self, path, prefix=""):
        self.path = Path(path)
        self.prefix = prefix
        index = json.loads((self.path / "model.safetensors.index.json").read_text())["weight_map"]
        self.index = {k[len(prefix) :]: value for k, value in index.items() if k.startswith(prefix)}

    def __getitem__(self, key):
        from safetensors import safe_open

        with safe_open(self.path / self.index[key], framework="pt", device="cpu") as f:
            return f.get_tensor(self.prefix + key)

    def __iter__(self):
        return iter(self.index)

    def __len__(self):
        return len(self.index)


def layer_state(layer_idx):
    from huggingface_hub import snapshot_download

    path = snapshot_download(
        "Qwen/Qwen3.8-Flash-Next", revision="de4b8e4d43b917e7706784d8bb445c9af86a3540", local_files_only=True
    )
    return CheckpointState(path, f"model.language_model.layers.{layer_idx}.")
