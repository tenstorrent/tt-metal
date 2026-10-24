# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Environment knobs of the lane-parallel prefill shared by the generator and the vLLM bridge.

A leaf module on purpose: ``generator_vllm`` imports from ``generator`` at module scope and the unit conftest's
host-only import stubs ``models.demos.gemma4.tt.generator`` with a fixed name list, so a knob defined in either of
those modules is not importable from the other under the stubs (the b6e3bd3e4a8 CI unit failures).
"""

import os


def _env_flag(name, default):
    return os.environ.get(name, default).lower() in ("1", "true", "yes")


def lane_prefill_table_upload_enabled():
    """Every lane-parallel prefill call uploads its per-layer page tables into the persistent device buffers
    (GEMMA4_LANE_PREFILL_TABLE_UPLOAD, default 1). ``ttnn_prefill_forward`` resolves the stash through
    ``_page_tables_to_ttnn``, which hands back a key's existing buffers without refreshing them, so without the upload
    every call after a key's first prefills through that first call's block ids. 0 reproduces that pre-fix path (the
    before/after coherence evidence of the 2026-10-24 Galaxy window 6: 20/32 coherent at conc 32 against 32/32);
    never serve with it."""
    return _env_flag("GEMMA4_LANE_PREFILL_TABLE_UPLOAD", "1")


def lane_prefill_batched_enabled():
    """The rounds of a gathered step prefill as one lane-batched walk (GEMMA4_LANE_PREFILL_BATCHED, default 1): k
    users per lane as one batched stream per column. Off, every round walks on its own in the one-user-per-lane
    layout, which isolates the plugin's admission change from the model's batched walk on the device."""
    return _env_flag("GEMMA4_LANE_PREFILL_BATCHED", "1")
