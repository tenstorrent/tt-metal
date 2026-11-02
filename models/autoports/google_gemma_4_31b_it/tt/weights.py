# SPDX-License-Identifier: Apache-2.0
"""Host-only pinned checkpoint reader shared by TT construction and HF controls."""

import json
from pathlib import Path

from huggingface_hub import hf_hub_download
from safetensors import safe_open

HF_MODEL = "google/gemma-4-31B-it"
HF_REVISION = "842da3794eaa0b77d5f08bae87a17459d91ff475"


class PinnedWeights:
    def __init__(self):
        self.index = json.loads(
            Path(hf_hub_download(HF_MODEL, "model.safetensors.index.json", revision=HF_REVISION)).read_text()
        )["weight_map"]

    def load(self, names):
        result = {}
        for shard in sorted({self.index[name] for name in names}):
            with safe_open(hf_hub_download(HF_MODEL, shard, revision=HF_REVISION), framework="pt") as f:
                for name in names:
                    if self.index[name] == shard:
                        result[name] = f.get_tensor(name)
        return result

    def layer(self, index):
        prefix = f"model.language_model.layers.{index}."
        return {
            k.removeprefix(prefix): v for k, v in self.load([k for k in self.index if k.startswith(prefix)]).items()
        }
