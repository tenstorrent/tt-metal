# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The default construction path consumes the selected precision config (stage 8).

Builds the model with ``PplxDeciderModel.from_snapshot(device)`` (no policy argument, so
``PrecisionPolicy.default()`` reads ``doc/datatype_sweep/selected_precision_config.json`` or
``$PPLX_DECIDER_PRECISION_CONFIG``) on a reduced stack: layer 0 (linear_attention) and layer 3
(full_attention) plus the embedding and the decision head. Then it checks, per matmul role:

- the dtype of every loaded device weight tensor (``tensor.dtype``) equals the JSON dtype;
- during one forward, every ``minimal_matmul`` / ``linear`` call that reads that weight runs with the
  JSON compute fidelity (``compute_kernel_config.math_fidelity``) and the JSON activation dtype.

The per-role table is printed and written to ``$PPLX_DECIDER_STAGE6_DIR/precision_config_check.json``.

Run::

    pytest models/demos/pplx_decider_v1_27b/tests/e2e/test_precision_config.py -q -s
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

import pytest
import torch

import ttnn
from models.demos.pplx_decider_v1_27b.tt.model import PplxDeciderModel
from models.demos.pplx_decider_v1_27b.tt.optimizations import (
    PRECISION_CONFIG_ENV,
    SELECTED_CONFIG_PATH,
    PrecisionPolicy,
)

OUT_DIR = Path(os.environ.get("PPLX_DECIDER_STAGE6_DIR", "/local/ttuser/gtobar/artifacts/pplx_decider/stage6"))

pytestmark = pytest.mark.use_module_device({"l1_small_size": 24576})


def role_weights(model) -> dict[str, list]:
    """role -> the device weight tensors of that role, from the loaded modules."""
    roles = defaultdict(list)
    for layer in model.layers:
        m = layer.mixer
        if layer.kind == "full_attention":
            roles["attention_qkvg"] += [m.qkv, m.gate]
            roles["attention_out"].append(m.o_proj)
        else:
            roles["delta_in"] += [m.in_qkv, m.in_zba]
            roles["delta_out"].append(m.out_proj)
        roles["mlp_gate_up"].append(layer.mlp.gate_up)
        roles["mlp_down"].append(layer.mlp.down)
    roles["readout"].append(model.head.readout_weight)
    return roles


@pytest.mark.timeout(1800)
def test_default_path_uses_selected_precision(_device_module_impl, monkeypatch):
    source = os.environ.get(PRECISION_CONFIG_ENV) or str(SELECTED_CONFIG_PATH)
    expected = PrecisionPolicy.from_json(source)
    model = PplxDeciderModel.from_snapshot(_device_module_impl, layer_ids=(0, 3))
    policy = model.config.optimizations.policy
    assert policy == expected, "from_snapshot() did not build the policy from the default JSON"

    roles = role_weights(model)
    by_id = {id(t): role for role, tensors in roles.items() for t in tensors}
    calls = defaultdict(set)

    def spy(fn):
        def wrapped(x, weight, *args, **kwargs):
            role = by_id.get(id(weight))
            if role is not None:
                cfg = kwargs["compute_kernel_config"]
                calls[role].add((str(weight.dtype), str(cfg.math_fidelity), str(kwargs.get("dtype"))))
            return fn(x, weight, *args, **kwargs)

        return wrapped

    monkeypatch.setattr(ttnn.experimental, "minimal_matmul", spy(ttnn.experimental.minimal_matmul))
    monkeypatch.setattr(ttnn, "linear", spy(ttnn.linear))
    tokens, last_index = model.upload_tokens(list(range(1000, 1600)))  # 600 tokens: minimal_matmul rows
    probs, logits, _ = model(tokens, last_index, 4)
    assert torch.isfinite(ttnn.to_torch(probs).float()).all()
    monkeypatch.undo()

    table = {}
    for role, tensors in sorted(roles.items()):
        table[role] = {
            "json_dtype": policy.weight_dtypes[role],
            "json_fidelity": policy.fidelities[role],
            "device_tensor_dtypes": sorted({str(t.dtype) for t in tensors}),
            "matmul_calls_weight_dtype_fidelity_output": sorted(map(list, calls[role])),
        }
        print(
            f"PRECISION {role:<15} json {policy.weight_dtypes[role]:<10} {policy.fidelities[role]:<6} | device tensors "
            f"{table[role]['device_tensor_dtypes']} | matmul calls {table[role]['matmul_calls_weight_dtype_fidelity_output']}"
        )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "precision_config_check.json").write_text(
        json.dumps({"config": source, "policy": policy.name, "roles": table}, indent=2) + "\n"
    )

    for role, row in table.items():
        want_dtype = str(getattr(ttnn, row["json_dtype"]))
        want_fid = str(getattr(ttnn.MathFidelity, row["json_fidelity"]))
        assert row["device_tensor_dtypes"] == [want_dtype], (role, row)
        assert calls[role], f"no matmul read the {role} weights"
        for dtype, fidelity, out_dtype in calls[role]:
            assert (dtype, fidelity) == (want_dtype, want_fid), (role, row)
            assert out_dtype == str(getattr(ttnn, policy.activation_dtype)), (role, out_dtype)
