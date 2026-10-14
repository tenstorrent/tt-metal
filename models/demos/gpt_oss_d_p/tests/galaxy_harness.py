# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Setup shared by the standalone GPT-OSS Galaxy harnesses (not pytest)."""

from __future__ import annotations

import os
import resource
import sys
from typing import Optional

import ttnn

ROWS, COLS = 4, 8  # SP=4 (rows), TP=8 (cols), EP=32
GALAXY_NUM_DEVICES = ROWS * COLS


def raise_nproc_limit(tag: str) -> None:
    """Raise RLIMIT_NPROC to the hard limit so tt-metal's parallel kernel JIT (a burst of g++/make
    procs) doesn't starve with EAGAIN mid-build."""
    soft, hard = resource.getrlimit(resource.RLIMIT_NPROC)
    if soft != resource.RLIM_INFINITY and (hard == resource.RLIM_INFINITY or soft < hard):
        try:
            resource.setrlimit(resource.RLIMIT_NPROC, (hard, hard))
            print(f"[{tag}] raised RLIMIT_NPROC soft {soft} -> {hard}", flush=True)
        except (ValueError, OSError) as e:
            print(f"[{tag}] WARNING: could not raise RLIMIT_NPROC (soft={soft}): {e}", file=sys.stderr)


def galaxy_exit_code(tag: str) -> Optional[int]:
    """None on a 4x8 Galaxy. Elsewhere the harness skips with rc 0, unless PREFILL_REQUIRE_GALAXY=1
    (CI), where a missing chip must fail instead of reporting PASS."""
    have = ttnn.get_num_devices()
    if have >= GALAXY_NUM_DEVICES:
        return None
    if os.getenv("PREFILL_REQUIRE_GALAXY") == "1":
        print(f"[{tag}] FAIL: needs the (4,8) galaxy ({GALAXY_NUM_DEVICES} devices); have {have}", flush=True)
        return 1
    print(f"[{tag}] SKIP: needs the (4,8) galaxy ({GALAXY_NUM_DEVICES} devices) for TP=8 + EP=32; have {have}")
    return 0


def linear_topology() -> bool:
    return os.getenv("PREFILL_TOPOLOGY", "ring") == "linear"


def open_galaxy_mesh():
    """Open the 4x8 mesh. Ring (default) needs the torus descriptor; PREFILL_TOPOLOGY=linear does not."""
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D if linear_topology() else ttnn.FabricConfig.FABRIC_1D_RING)
    return ttnn.open_mesh_device(ttnn.MeshShape(ROWS, COLS))


def runtime_topology():
    return ttnn.Topology.Linear if linear_topology() else ttnn.Topology.Ring


def expert_weight_dtype():
    """EXPERT_DTYPE=bf8|bf4 (default bf4); must match the tilized cache being loaded."""
    return ttnn.bfloat8_b if os.getenv("EXPERT_DTYPE", "bf4") == "bf8" else ttnn.bfloat4_b


def load_state_dict(model_args, tag: str) -> dict:
    """GPT_OSS_WEIGHTS_FROM_CACHE=1 passes an empty state_dict so weights come from the tilized cache."""
    from models.demos.gpt_oss_d_p.tt.model_config import ModelArgs

    if os.getenv("GPT_OSS_WEIGHTS_FROM_CACHE") == "1":
        print(f"[{tag}] GPT_OSS_WEIGHTS_FROM_CACHE=1 -> empty state_dict (load tilized cache)", flush=True)
        return {}
    print(f"[{tag}] loading real bf16 weights (slow: safetensors read) ...", flush=True)
    return ModelArgs.load_state_dict(model_args.weights_path)
