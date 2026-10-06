# SPDX-License-Identifier: Apache-2.0
"""Versioned construction policy shared by model and generator entrypoints."""

import copy
import json
import os
from dataclasses import asdict
from pathlib import Path

from .multichip_decoder import MeshPolicy

DEFAULT_PATH = Path(__file__).resolve().parents[1] / "doc/datatype_sweep/selected_precision_config.json"


def baseline_config():
    return dict(
        schema_version=1,
        config_id="baseline_bfp4_lofi",
        mesh_policy=asdict(MeshPolicy()),
        layer_exceptions={},
        runtime=dict(
            kv_cache_dtype="bfloat8_b",
            residual_dtype="bfloat16",
            activation_dtype="bfloat16",
            embedding_dtype="bfloat16",
            norm_dtype="bfloat16",
            head_weight_dtype="float32",
            head_output_dtype="float32",
            head_fidelity="HiFi4",
            head_fp32=True,
            norm_fidelity="HiFi4",
            sampling_logits_dtype="float32",
            sampling_parameter_dtype="bfloat16",
            sampling_index_dtype="uint32",
            max_context=262144,
        ),
    )


def load_precision_config(value=None):
    if value is None:
        value = os.environ.get("KOLIBRI_PRECISION_CONFIG") or DEFAULT_PATH
    config = copy.deepcopy(value) if isinstance(value, dict) else json.loads(Path(value).read_text())
    baseline = baseline_config()
    if set(config) != set(baseline) or config["schema_version"] != 1:
        raise ValueError("Unknown precision configuration schema")
    if set(config["runtime"]) != set(baseline["runtime"]):
        raise ValueError("Runtime precision configuration must be complete")
    if set(config["mesh_policy"]) != set(baseline["mesh_policy"]):
        raise ValueError("Mesh policy must specify every field")
    MeshPolicy(**config["mesh_policy"])
    if config["mesh_policy"]["safe_compute_fidelity"] != "HiFi4":
        raise ValueError("Exact prefix offsets require HiFi4; shared norm/RoPE/large-router config is fixed")
    for layer, overrides in config["layer_exceptions"].items():
        if not 0 <= int(layer) < 50:
            raise ValueError(f"Invalid layer exception {layer}")
        resolved = config["mesh_policy"] | overrides
        MeshPolicy(**resolved)
        if resolved["safe_compute_fidelity"] != "HiFi4":
            raise ValueError("Layer exception violates the exact prefix-offset contract")
    # These are operation contracts, explicitly validated rather than silently
    # ignored knobs. BF16 intermediates feed norm, RoPE and paged_update_cache;
    # reduced residual storage is converted back at the next layer boundary.
    fixed = (
        "activation_dtype",
        "embedding_dtype",
        "norm_dtype",
        "norm_fidelity",
        "sampling_parameter_dtype",
        "sampling_index_dtype",
        "max_context",
    )
    for key in fixed:
        if config["runtime"][key] != baseline["runtime"][key]:
            raise ValueError(f"Unsupported {key}: {config['runtime'][key]}")
    if config["runtime"]["sampling_logits_dtype"] not in ("float32", "bfloat16"):
        raise ValueError("Sampler logits must be FP32 or BF16")
    if config["runtime"]["head_output_dtype"] != config["runtime"]["sampling_logits_dtype"]:
        raise ValueError("Head output must match the persistent sampler logits")
    return config


def layer_policy(config, layer):
    return MeshPolicy(**(config["mesh_policy"] | config["layer_exceptions"].get(str(layer), {})))
