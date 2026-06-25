# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Model-name → adapter registry.

Maps a `PREFILL_MODEL_NAME` to a `"dotted.module.path:adapter_class"` string and imports it LAZILY in
`get_adapter`. This is the only place that knows a model module path — and even that is a string
resolved on demand, so importing this module (and the runner core) never pulls in a model package.
New models register by adding one line here (or via PREFILL_ADAPTER_FACTORY, below).
"""

import importlib
import os

from models.common.prefill_runner.adapter import PrefillModelAdapter

# name -> "module.path:factory". The factory is called with no args and returns a PrefillModelAdapter
# (an adapter class works directly).
_ADAPTER_FACTORIES = {
    "deepseek_v3_d_p": "models.demos.deepseek_v3_d_p.tt.runners.ds_prefill_adapter:DeepSeekPrefillAdapter",
    "kimi_k2_6": "models.demos.deepseek_v3_d_p.tt.runners.ds_prefill_adapter:KimiPrefillAdapter",
}


def _resolve(spec: str):
    """Import `module.path:factory` and return the factory callable."""
    module_path, _, attr = spec.partition(":")
    if not attr:
        raise ValueError(f"adapter factory spec must be 'module.path:factory', got {spec!r}")
    return getattr(importlib.import_module(module_path), attr)


def get_adapter(name: str) -> PrefillModelAdapter:
    """Resolve a model name to a constructed adapter (lazy import of the model package).

    PREFILL_ADAPTER_FACTORY ("module.path:factory") overrides the table — lets an out-of-tree model
    plug in without editing this file."""
    spec = os.environ.get("PREFILL_ADAPTER_FACTORY")
    if not spec:
        try:
            spec = _ADAPTER_FACTORIES[name]
        except KeyError:
            raise KeyError(
                f"Unknown PREFILL_MODEL_NAME={name!r}; valid: {sorted(_ADAPTER_FACTORIES)} "
                f"(or set PREFILL_ADAPTER_FACTORY=module.path:factory)"
            )
    return _resolve(spec)()
