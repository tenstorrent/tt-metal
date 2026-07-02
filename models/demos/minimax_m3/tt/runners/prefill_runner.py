# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
MiniMax-M2 prefill runner — service entry point.

SCAFFOLD — see PREFILL_PROPOSAL.md §8. Two modes:
  * standalone: JSON token_ids in, first_token out (no C++ server) — for bring-up/bench
  * SHM:        request loop over shared memory from the C++ inference server

Both call MiniMaxPrefillPipeline.prefill(), which is itself Tier-1 scaffold. Env vars
mirror the DeepSeek-V3-D/P runner so the team can reuse its ops tooling.

Reference: models/demos/deepseek_v3_d_p/tt/runners/prefill_runner.py
"""

import os

PREFILL_SP = int(os.getenv("PREFILL_SP", "4"))
PREFILL_TP = int(os.getenv("PREFILL_TP", "8"))
PREFILL_NUM_LAYERS = int(os.getenv("PREFILL_NUM_LAYERS", "62"))
PREFILL_MAX_SEQ_LEN = int(os.getenv("PREFILL_MAX_SEQ_LEN", str(128 * 1024)))
PREFILL_CHUNK_SIZE = int(os.getenv("PREFILL_CHUNK_SIZE", "5120"))
PREFILL_ENABLE_MIGRATION = int(os.getenv("PREFILL_ENABLE_MIGRATION", "0"))


def build_pipeline(mesh_device):
    """Construct model + pipeline. OWNER: runner work (Tier 1/2)."""
    raise NotImplementedError(
        "build_pipeline: open mesh (SP x TP), create_tt_model, wrap in MiniMaxPrefillPipeline. "
        "Needs the multi-card Galaxy (full model does not fit one Wormhole). See PREFILL_PROPOSAL.md §8."
    )


def run_standalone_loop(pipeline):
    """Read token_ids from a JSON file, run prefill, print first_token. OWNER: runner work (Tier 1)."""
    raise NotImplementedError("run_standalone_loop: Tier 1 — JSON in, pipeline.prefill(), first_token out.")


def run_request_loop(pipeline):
    """SHM request loop from the C++ server. OWNER: serving/runner team (Tier 2)."""
    raise NotImplementedError("run_request_loop: owner=serving team; blocked on SHM protocol + C++ server.")


def main():
    raise NotImplementedError(
        "prefill_runner.main: scaffold. Wire build_pipeline -> (standalone | SHM) loop. "
        "See PREFILL_PROPOSAL.md §8 and SKELETON.md for the work split."
    )


if __name__ == "__main__":
    main()
