# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Experiment-only pytest plugin: force chunked VAE decode for the BH 4x8 Wan preset.

The stock BH (4, 8) preset in models/tt_dit/pipelines/wan/pipeline_wan.py sets
vae_t_chunk_size=None (all latent frames decoded in one pass). On 7-bank (Bin6) Blackhole
that runs out of DRAM at 720p (issue #59504). This plugin sets it to 7 (the BH 2x4 value)
for one pytest process only; nothing on disk changes.

Usage (from the repo root):
    PYTHONPATH=$PWD:$PWD/handoff/wan-bin6 pytest -p vae_chunk_patch ...
"""


def pytest_configure(config):
    from models.tt_dit.pipelines.wan import pipeline_wan

    pipeline_wan._PRESETS_BH[(4, 8)]["vae_t_chunk_size"] = 7
    print("PATCHED BH 4x8 vae_t_chunk_size ->", pipeline_wan._PRESETS_BH[(4, 8)]["vae_t_chunk_size"])

    # SKIP_VBENCH=1: skip the VBench check, which scores 720p video on CPU for 30+ minutes.
    # The test module imports assert_vbench_quality at collection, after this hook runs.
    import os

    if os.environ.get("SKIP_VBENCH") == "1":
        from models.tt_dit.utils import vbench

        vbench.assert_vbench_quality = lambda *args, **kwargs: print("SKIP_VBENCH=1: VBench check skipped")
