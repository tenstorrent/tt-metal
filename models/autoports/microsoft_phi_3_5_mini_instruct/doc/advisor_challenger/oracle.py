"""Real-weight differential oracle: advisor candidate versus frozen incumbent."""
from __future__ import annotations

import json
import os

import torch

import ttnn
from models.autoports.microsoft_phi_3_5_mini_instruct.doc.advisor_challenger.harness import build, decode
from models.autoports.microsoft_phi_3_5_mini_instruct.tests.test_functional_decoder import _to_torch_decode
from models.common.utility_functions import comp_pcc


def main():
    root = "models/autoports/microsoft_phi_3_5_mini_instruct/doc/advisor_challenger"
    with open(f"{root}/policy.json") as handle:
        incumbent_policy = json.load(handle)["shipped_policy"]
    with open(f"{root}/policy_rope_l1.json") as handle:
        candidate_policy = json.load(handle)["shipped_policy"]
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    try:
        incumbent = build(mesh, incumbent_policy)
        incumbent_output = _to_torch_decode(decode(incumbent))
        ttnn.synchronize_device(mesh)
        candidate = build(mesh, candidate_policy)
        candidate_output = _to_torch_decode(decode(candidate))
        ttnn.synchronize_device(mesh)
        passed, message = comp_pcc(incumbent_output.float(), candidate_output.float(), 0.995)
        diff = (incumbent_output.float() - candidate_output.float()).abs()
        record = {
            "oracle_weights": "real",
            "reference": "frozen incumbent implementation with identical real layer-0 checkpoint weights and inputs",
            "threshold": 0.995,
            "oracle_passed": bool(passed),
            "oracle_pcc_message": message,
            "max_abs_diff": float(diff.max()),
            "mean_abs_diff": float(diff.mean()),
            "decode_batch": 32,
            "requested_decode_batch": 32,
        }
        with open(f"{root}/measurements/rope_l1_rect32_oracle.json", "w") as handle:
            json.dump(record, handle, indent=2)
        print(json.dumps(record, indent=2))
        if not passed:
            raise SystemExit(1)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    os.environ.setdefault("CHALLENGER_DECODE_BATCH", "32")
    main()
