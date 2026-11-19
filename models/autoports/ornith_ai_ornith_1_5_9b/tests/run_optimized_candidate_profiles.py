# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Serial real-weight BFP4 diagnostic profiler evidence, never a selected policy."""
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for layer in (0, 3):
    policy = dict(
        mlp_dtype="bfloat4_b",
        mlp_fidelity="LoFi",
        topology="sharded_split_mlp",
        prefill_family="minimal",
        prefill_block=16,
        prefill_subblock=2,
        carry_mlp=True,
        residual_cores=64,
        alias_gdn=True,
        gdn_receivers=1,
    )
    if layer == 3:
        policy.update(kv_dtype="bfloat8_b", sdpa_k_chunk=256)
    subprocess.run(
        [
            "timeout",
            "2000",
            "bash",
            str(ROOT / "tests/profile_optimized_candidate.sh"),
            "bfp4_profile",
            json.dumps(policy),
            str(layer),
        ],
        check=True,
    )
print("BFP4_PROFILES_DONE")
