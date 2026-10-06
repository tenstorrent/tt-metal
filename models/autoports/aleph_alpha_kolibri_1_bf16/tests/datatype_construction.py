# SPDX-License-Identifier: Apache-2.0
"""Host-only proof that the selected policy is required and overrides stay explicit."""

import copy
import hashlib
import json
import os
import tempfile
from pathlib import Path

from ..tt import precision


def main():
    assert not os.environ.get("KOLIBRI_PRECISION_CONFIG"), "Check the ordinary default path"
    path = precision.DEFAULT_PATH
    selected = json.loads(path.read_text())
    assert precision.load_precision_config() == selected
    baseline = precision.baseline_config()
    assert precision.load_precision_config(baseline) == baseline
    with tempfile.TemporaryDirectory(prefix="kolibri-policy-") as temporary:
        missing = Path(temporary) / "missing.json"
        precision.DEFAULT_PATH = missing
        try:
            try:
                precision.load_precision_config()
            except FileNotFoundError:
                pass
            else:
                raise AssertionError("A missing selected artifact must not silently fall back")
            override = Path(temporary) / "baseline.json"
            override.write_text(json.dumps(baseline))
            os.environ["KOLIBRI_PRECISION_CONFIG"] = str(override)
            assert precision.load_precision_config() == json.loads(json.dumps(baseline))
            assert precision.load_precision_config(selected) == selected
        finally:
            precision.DEFAULT_PATH = path
            os.environ.pop("KOLIBRI_PRECISION_CONFIG", None)
    rejected = []
    for key, value in (
        ("activation_dtype", "bfloat8_b"),
        ("embedding_dtype", "bfloat8_b"),
        ("norm_dtype", "bfloat8_b"),
        ("norm_fidelity", "LoFi"),
        ("sampling_parameter_dtype", "float32"),
        ("sampling_index_dtype", "int32"),
        ("max_context", 8192),
    ):
        invalid = copy.deepcopy(selected)
        invalid["runtime"][key] = value
        try:
            precision.load_precision_config(invalid)
        except ValueError:
            rejected.append(key)
        else:
            raise AssertionError(f"Unsupported operation contract accepted: {key}")
    result = dict(
        status="pass",
        selected_config_id=selected["config_id"],
        selected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        default_equals_selected=True,
        missing_default_raises=True,
        explicit_baseline_override=True,
        precedence="explicit argument > environment override > required selected artifact",
        rejected_fixed_contract_changes=rejected,
        scope="Host resolver contract; actual native consumption is separately asserted in selected_default/result.json",
    )
    (path.parent / "construction_contract.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
