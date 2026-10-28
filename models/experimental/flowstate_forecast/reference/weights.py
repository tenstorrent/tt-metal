# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""The r1.1 checkpoint as nested namespaces with the HF attribute paths (model.model.encoder.layers[i].ssm...),
for environments without granite-tsfm (python_env). The weights come from the shared HF cache, never a copy."""

import os
from types import SimpleNamespace

from safetensors.torch import load_file

REVISION = "b80e0f1b1c2d90edebe3d4f243ad046a3f935daa"
PATH = os.path.expanduser(
    f"~/.cache/huggingface/hub/models--ibm-granite--granite-timeseries-flowstate-r1/snapshots/{REVISION}/model.safetensors"
)


def load(path=PATH):
    root = {}
    for k, v in load_file(path).items():
        node = root
        *parts, leaf = k.split(".")
        for p in parts:
            node = node.setdefault(p, {})
        node[leaf] = v

    def ns(d):
        if all(k.isdigit() for k in d):
            return [ns(d[str(i)]) for i in range(len(d))]
        return SimpleNamespace(**{k: ns(v) if isinstance(v, dict) else v for k, v in d.items()})

    return SimpleNamespace(model=ns(root))
