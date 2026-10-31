"""Serialized stage05 candidates; stop at the first failure for diagnosis."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--layer", type=int, default=0)
p.add_argument("--batch", type=int, default=1)
p.add_argument("--names", nargs="+", required=True)
a = p.parse_args()
base = {"stage05": True, "ring_all": True, "row_direct": True, "no_identity_cast": True}
q = "self_attn.qkv_proj" if a.layer == 0 else "self_attn.qk_proj"
cases = {
    "batch_l1_chunks": {"row_direct": False, "batch_l1_chunks": True},
    "input_l1_chunks": {
        "row_direct": False,
        "batch_l1_chunks": True,
        "prefill_chunk_rows": 2048,
        "prefill_input_l1": [q, "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"],
        "prefill_shared_mlp_input_l1": True,
    },
    **{
        f"input_l1_{role}{suffix}": {
            "row_direct": False,
            "prefill_input_l1": names,
            "prefill_mblock": m,
            "prefill_kblock": k,
            "prefill_nblock": n,
            "prefill_shared_mlp_input_l1": role == "mlp",
        }
        for role, names in (
            ("q", [q]),
            ("o", ["self_attn.o_proj"]),
            ("mlp", ["mlp.gate_proj", "mlp.up_proj"]),
            ("down", ["mlp.down_proj"]),
        )
        for suffix, m, k, n in (
            ("", 4, 8, 16),
            ("_m1", 1, 8, 16),
            ("_m1n8", 1, 8, 8),
            ("_small", 1, 4, 4),
            ("_tiny", 1, 2, 2),
        )
    },
    "direct": {},
    "pooled_slice": {"row_l1_slice": True, "ccl_l1": True, "pool_ccl": True},
    "direct_repeat": {},
    "row_l1_slice": {"row_l1_slice": True},
    "attn8": {"activation": "attention"},
    "mlp8": {"activation": "mlp"},
    "attn_hifi2": {"attention_precision": "bfp8"},
    "attn_lofi": {"attention_precision": "bfp8_lofi"},
    "mlp_hifi2": {"precision": "bfp8", "down_precision": "bfp4"},
    "mlp_lofi": {"precision": "bfp8_lofi", "down_precision": "bfp4"},
    "down_hifi2": {"down_precision": "bfp8"},
    "down_lofi": {"down_precision": "bfp8_lofi"},
    "ccl_pool": {"ccl_l1": True, "pool_ccl": True},
    "ccl_l1": {"ccl_l1": True},
    "norm8": {"prefill_norm": True},
    "norm8poolq24": {
        "prefill_norm": True,
        "row_l1_slice": True,
        "ccl_l1": True,
        "pool_ccl": True,
        "geometry": {q: {"cores": 7, "block": 24, "readers": 2}},
    },
    "norm8poolccl8": {
        "prefill_norm": True,
        "row_l1_slice": True,
        "ccl_l1": True,
        "pool_ccl": True,
        "decode_ccl8": True,
    },
    "norm8pool": {"prefill_norm": True, "row_l1_slice": True, "ccl_l1": True, "pool_ccl": True},
    "norm3": {"prefill_norm": True},
    "norm2": {"prefill_norm": True},
    "norm": {"prefill_norm": True},
    "output_ag": {"output_ag": True, "ag_bf16_input": True},
    "output_ag_sharded": {"output_ag": True, "ag_bf16_input": True, "sharded": True},
    "q12": {"geometry": {q: {"cores": 14, "block": 12, "readers": 2}}},
    "q24": {"geometry": {q: {"cores": 7, "block": 24, "readers": 2}}},
    "gate12": {"geometry": {n: {"cores": 14, "block": 12, "readers": 2} for n in ["mlp.gate_proj", "mlp.up_proj"]}},
    "gate24": {"geometry": {n: {"cores": 7, "block": 24, "readers": 2} for n in ["mlp.gate_proj", "mlp.up_proj"]}},
    "down22": {"geometry": {"mlp.down_proj": {"cores": 8, "block": 22, "readers": 2}}},
    "o16": {"geometry": {"self_attn.o_proj": {"cores": 4, "block": 16 if a.layer == 0 else 32, "readers": 2}}},
}
root = Path("models/autoports/google_gemma_4_31b_it/doc/optimized_multichip_decoder")
for name in a.names:
    options = {**base, **cases[name]}
    path = root / f"{name}_layer{a.layer}_batch{a.batch}"
    cmd = [
        sys.executable,
        "-m",
        "models.autoports.google_gemma_4_31b_it.tests.run_multichip_decoder",
        "--layer",
        str(a.layer),
        "--batch",
        str(a.batch),
        "--seq",
        "4096",
        "--cyclic",
        "--inputs",
        f"bringup/artifacts/optimized_decoder_layer{a.layer}_inputs262144.pt",
        "--candidate",
        json.dumps(options),
        "--result",
        str(path.with_suffix(".json")),
    ]
    if not options.get("sharded"):
        cmd.append("--replicated")
    print("RUN", path.name, flush=True)
    with path.with_suffix(".log").open("w") as f:
        r = subprocess.run(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            env=dict(os.environ, TT_METAL_TRACE_ALLOC_TRACKING="1"),
            timeout=300,
        )
    path.with_suffix(".exit.json").write_text(json.dumps({"command": cmd, "exit_code": r.returncode}) + "\n")
    if r.returncode:
        raise SystemExit(f"Failed {path}; inspect before continuing")
    print("PASS", path.name, flush=True)
