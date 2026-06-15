# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
GPT-OSS prefill runner — service entry point.

SCAFFOLD — see PREFILL_PROPOSAL.md §8. Two modes:
  * standalone: JSON token_ids in, first_token out (no C++ server) — for bring-up/bench
  * SHM:        request loop over shared memory from the C++ inference server

Both call GptOssPrefillPipeline.prefill(), which is itself Tier-1 scaffold. Env vars
mirror the DeepSeek-V3-D/P and MiniMax-M2 runners so the team can reuse its ops tooling.

Key differences from MiniMax-M2 runner:
  * PREFILL_NUM_LAYERS=36 (GPT-OSS 120B has 36 decoder layers, not 62)
  * No chunking by default (§8.4 of proposal: full-sequence until per-layer migration
    rate proves a bottleneck; then tune PREFILL_CHUNK_SIZE)
  * PREFILL_MAX_SEQ_LEN=131072 (max_position_embeddings from config.json)

Reference: models/demos/deepseek_v3_d_p/tt/runners/prefill_runner.py
"""

import os

PREFILL_SP = int(os.getenv("PREFILL_SP", "4"))
PREFILL_TP = int(os.getenv("PREFILL_TP", "8"))
PREFILL_NUM_LAYERS = int(os.getenv("PREFILL_NUM_LAYERS", "36"))
PREFILL_MAX_SEQ_LEN = int(os.getenv("PREFILL_MAX_SEQ_LEN", "131072"))
PREFILL_ENABLE_MIGRATION = int(os.getenv("PREFILL_ENABLE_MIGRATION", "0"))
PREFILL_STANDALONE = int(os.getenv("PREFILL_STANDALONE", "0"))


def build_pipeline(mesh_device):
    """Construct model + pipeline. OWNER: runner work (Tier 1)."""
    raise NotImplementedError(
        "build_pipeline: open mesh (SP x TP = 4x8), create_tt_model, wrap in GptOssPrefillPipeline. "
        "Needs the full BH Galaxy (32 chips). See PREFILL_PROPOSAL.md §8."
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
        "See PREFILL_PROPOSAL.md §8 for the work split."
    )


if __name__ == "__main__":
    main()
