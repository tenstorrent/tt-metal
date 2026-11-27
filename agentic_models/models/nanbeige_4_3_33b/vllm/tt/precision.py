"""One precision policy for the production server, toolkit adapter and sweeps."""

import hashlib
import json
import os
from pathlib import Path

GROUPS = ("attention", "mlp_gate_up", "mlp_down", "lm_head")
DEFAULT_PATH = Path(__file__).resolve().parents[1] / "config/precision.json"


def load_policy():
    path = Path(os.environ.get("NANBEIGE_PRECISION_CONFIG", DEFAULT_PATH)).resolve()
    policy = json.loads(path.read_text())
    for group in GROUPS:
        if policy["weights"][group] not in ("bf16", "bfp8", "bfp4"):
            raise ValueError(f"Unsupported weight format for {group}")
        for mode in ("prefill", "decode"):
            c = policy["compute"][mode][group]
            if c["fidelity"] not in ("HiFi4", "HiFi2", "LoFi"):
                raise ValueError(f"Unsupported fidelity for {mode}/{group}")
            if c["rounding"] != "nearest":
                raise ValueError("These dense kernels expose deterministic nearest rounding only")
    if policy["kv_dtype"] not in ("bf16", "bfp8"):
        raise ValueError("Unsupported KV format")
    if (
        not isinstance(policy["kv_pool_tokens"], int)
        or policy["kv_pool_tokens"] < 131072
        or policy["kv_pool_tokens"] % 32
    ):
        raise ValueError("KV request pool must cover 131072 tokens in whole 32-token pages")
    if policy["recurrent_state_dtype"] != "float32":
        raise ValueError("Recurrent state precision must remain FP32 (this model has no recurrent layers)")
    policy["path"] = str(path)
    policy["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return policy


def dtype(name):
    import ttnn

    return {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b}[name]


def compute_config(spec):
    import ttnn

    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=getattr(ttnn.MathFidelity, spec["fidelity"]),
        math_approx_mode=False,
        fp32_dest_acc_en=spec["fp32_dest_acc"],
        packer_l1_acc=spec["packer_l1_acc"],
    )
