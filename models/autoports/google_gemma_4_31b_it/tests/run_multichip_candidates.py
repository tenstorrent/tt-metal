"""Serialized, precision-locked decoder geometry/topology experiments."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--layer", type=int, default=0)
p.add_argument("--names", nargs="*")
p.add_argument("--batch", type=int, default=1)
p.add_argument("--cyclic", action="store_true")
a = p.parse_args()
base = {"carry_reduced": True, "pad_mlp": True, "ccl8": True, "ccl8_prefill_only": True}
cases = {
    "finalq1": {
        "ring_all": True,
        "persistent_ccl": True,
        "geometry": {
            "self_attn.qkv_proj"
            if a.layer == 0
            else "self_attn.qk_proj": {
                "cores": 28 if a.layer == 0 else 8,
                "block": 6 if a.layer == 0 else 21,
                "readers": 1,
            }
        },
    },
    "finalo1": {
        "ring_all": True,
        "persistent_ccl": True,
        "geometry": {
            "self_attn.o_proj": {"cores": 8, "block": 8 if a.layer == 0 else 16, "readers": 1},
            "self_attn.qk_proj": {"cores": 8, "block": 21, "readers": 2},
        },
    },
    "finaldown1": {
        "ring_all": True,
        "persistent_ccl": True,
        "geometry": {
            "mlp.down_proj": {"cores": 8, "block": 11, "readers": 1},
            "self_attn.qk_proj": {"cores": 8, "block": 21, "readers": 2},
        },
    },
    "finalgate1": {
        "ring_all": True,
        "persistent_ccl": True,
        "geometry": {
            "mlp.gate_proj": {"cores": 28, "block": 6, "readers": 1},
            "mlp.up_proj": {"cores": 28, "block": 6, "readers": 1},
            "self_attn.qk_proj": {"cores": 8, "block": 21, "readers": 2},
        },
    },
    "finalsplit": {
        "ring_all": True,
        "persistent_ccl": True,
        "geometry": {"self_attn.qk_proj": {"cores": 8, "block": 21, "readers": 2}},
    },
    "finalpacked8": {
        "ring_all": True,
        "persistent_ccl": True,
        "packed_mlp": True,
        "packed_direct_bfp8": True,
        "geometry": {"self_attn.qk_proj": {"cores": 8, "block": 21, "readers": 2}},
    },
    "finalpacked": {
        "ring_all": True,
        "persistent_ccl": True,
        "packed_mlp": True,
        "geometry": {"self_attn.qk_proj": {"cores": 8, "block": 21, "readers": 2}},
    },
    "ringpersistent": {"ring_all": True, "persistent_ccl": True},
    "ringpacked": {"ring_all": True, "packed_mlp": True},
    "ringpackedpersistent": {"ring_all": True, "packed_mlp": True, "persistent_ccl": True},
    "ring1reader": {
        "ring_all": True,
        "geometry": {n: {"cores": 28, "block": 6, "readers": 1} for n in ["mlp.gate_proj", "mlp.up_proj"]},
    },
    "mixed": {},
    "pref14": {"prefill_kblock": 14},
    "pref21": {"prefill_kblock": 21},
    "pref14n8": {"prefill_kblock": 14, "prefill_nblock": 8, "prefill_grid": [11, 8]},
    "mlp6144": {"mlp_width": 6144, "prefill_kblock": 14, "prefill_nblock": 8, "prefill_grid": [8, 8]},
    "ring": {"ring_all": True},
    "ringq21": {"ring_all": True, "geometry": {"self_attn.qk_proj": {"cores": 8, "block": 21, "readers": 2}}},
    "ringq3readers": {"ring_all": True, "geometry": {"self_attn.qk_proj": {"cores": 8, "block": 21, "readers": 3}}},
    "ringpref14n8": {"ring_all": True, "prefill_kblock": 14, "prefill_nblock": 8, "prefill_grid": [11, 8]},
    "ring6144": {
        "ring_all": True,
        "mlp_width": 6144,
        "prefill_kblock": 14,
        "prefill_nblock": 8,
        "prefill_grid": [8, 8],
    },
    "q21": {
        "geometry": {
            "self_attn.qkv_proj" if a.layer == 0 else "self_attn.qk_proj": {"cores": 8, "block": 21, "readers": 2}
        }
    },
    "q24": {
        "geometry": {
            "self_attn.qkv_proj" if a.layer == 0 else "self_attn.qk_proj": {"cores": 7, "block": 24, "readers": 2}
        }
    },
    "q4readers": {"geometry": {"self_attn.qkv_proj": {"cores": 28, "block": 6, "readers": 4}}},
    "gate12": {"geometry": {n: {"cores": 14, "block": 12, "readers": 2} for n in ["mlp.gate_proj", "mlp.up_proj"]}},
    "gate24": {"geometry": {n: {"cores": 7, "block": 24, "readers": 2} for n in ["mlp.gate_proj", "mlp.up_proj"]}},
    "down22": {"geometry": {"mlp.down_proj": {"cores": 8, "block": 22, "readers": 2}}},
    "attn8": {"ccl8": "attention"},
    "mlp8": {"ccl8": "mlp"},
}
out = Path("models/autoports/google_gemma_4_31b_it/doc/multichip_decoder")
for name in a.names or cases:
    candidate = {**base, **cases[name]}
    path = out / (
        f"candidate_layer{a.layer}_{name}"
        + (f"_batch{a.batch}" if a.batch != 1 else "")
        + ("_cyclic" if a.cyclic else "")
    )
    cmd = [
        sys.executable,
        "-m",
        "models.autoports.google_gemma_4_31b_it.tests.run_multichip_decoder",
        "--layer",
        str(a.layer),
        "--seq",
        "4096",
        "--inputs",
        f"bringup/artifacts/optimized_decoder_layer{a.layer}_inputs262144.pt",
        "--replicated",
        "--candidate",
        json.dumps(candidate),
        "--result",
        str(path.with_suffix(".json")),
    ]
    cmd += ["--batch", str(a.batch)]
    if a.cyclic:
        cmd.append("--cyclic")
    print("RUN", name, flush=True)
    with path.with_suffix(".log").open("w") as f:
        result = subprocess.run(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            env=dict(os.environ, TT_METAL_TRACE_ALLOC_TRACKING="1"),
            timeout=180,
        )
    if result.returncode:
        raise SystemExit(f"Failed {name}; inspect log and recover before next device job")
    print("PASS", name, flush=True)
