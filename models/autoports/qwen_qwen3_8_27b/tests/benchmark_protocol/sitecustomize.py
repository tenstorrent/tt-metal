"""Load hash-verified local GPQA bytes while retaining the upstream task recipe.

Only active when this directory is explicitly added to the benchmark client's
PYTHONPATH. No server/model code is imported or changed.
"""
import hashlib
import json
from pathlib import Path

import datasets

_original_load_dataset = datasets.load_dataset
_identity_path = Path(__file__).resolve().parents[2] / "doc/benchmark/subset-source/mirror.json"


def _load_dataset(path, name=None, *args, **kwargs):
    if path != "Idavidrein/gpqa" or name != "gpqa_diamond":
        return _original_load_dataset(path, name, *args, **kwargs)
    identity = json.loads(_identity_path.read_text())
    local = Path(identity["local_path"])
    if hashlib.sha256(local.read_bytes()).hexdigest() != identity["sha256"]:
        raise ValueError("GPQA mirror CSV differs from the verified setup artifact")
    # lm-eval still owns all prompts, deterministic processing, splits and scores.
    # Its benchmark wrapper rechecks the complete processed population hash.
    kwargs.pop("trust_remote_code", None)
    return _original_load_dataset("csv", *args, data_files={"train": str(local)}, **kwargs)


datasets.load_dataset = _load_dataset
