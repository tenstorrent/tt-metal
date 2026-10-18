"""Local checkpoint policy for this model's optional real-weight diagnostics.

The main decoder suite defaults to committed-statistics synthetic weights and
must remain runnable without HF cache/downloads. Real-only diagnostics skip
explicitly when their local checkpoint metadata is absent. A present checkpoint
with missing/corrupt payloads fails normally; this hook does not conceal those
failures or satisfy the stage's mandatory real-weight evidence gates.
"""
import os
from pathlib import Path

import pytest

_DEFAULT_CHECKPOINT = Path(
    "/mnt/models/huggingface/hub/models--Qwen--Qwen3.8-Flash-Next/snapshots/" "de4b8e4d43b917e7706784d8bb445c9af86a3540"
)
_TEST_ROOT = Path(__file__).resolve().parent
_REAL_ONLY_MODULES = {
    "test_components.py",
    "test_gdn.py",
    "gdn_long_context.py",
    "test_qsa.py",
    "test_qsa_positions.py",
}


def _requires_real_checkpoint(item):
    if not Path(item.path).resolve().is_relative_to(_TEST_ROOT):
        return False
    name = Path(item.path).name
    if name in _REAL_ONLY_MODULES:
        # test_qsa's optional synthetic tensors still derive moments by reading
        # actual checkpoint tensors. The independent synthetic CI route is the
        # complete test_functional_decoder suite with committed statistics.
        return True
    if name == "test_ple.py":
        return item.originalname == "test_real_weights_full_table_ple"
    if name == "test_functional_decoder.py":
        return os.environ.get("QWEN38_WEIGHT_SOURCE", "synthetic") == "real"
    return False


def pytest_collection_modifyitems(items):
    for item in items:
        if not _requires_real_checkpoint(item):
            continue
        checkpoint = getattr(item.module, "CHECKPOINT", None)
        if checkpoint is None:
            variable = "TT_QSA_CHECKPOINT" if Path(item.path).name.startswith("test_qsa") else "QWEN38_CHECKPOINT"
            checkpoint = os.environ.get(variable, _DEFAULT_CHECKPOINT)
        checkpoint = Path(checkpoint)
        missing = [
            name for name in ("config.json", "model.safetensors.index.json") if not (checkpoint / name).is_file()
        ]
        if missing:
            item.add_marker(
                pytest.mark.skip(
                    reason=(
                        f"Real Qwen3.8-Flash-Next diagnostic requires local checkpoint {checkpoint}; "
                        f"missing {', '.join(missing)}. No download is attempted. "
                        "The default complete-decoder synthetic suite remains enabled."
                    )
                )
            )
