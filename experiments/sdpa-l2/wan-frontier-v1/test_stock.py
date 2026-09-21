# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Run the existing Wan performance test against an explicitly pinned snapshot.

Only checkpoint resolution and host thread count change. The imported test
retains its original device/configuration parameters and performance gates.
"""

import os
from pathlib import Path

import pytest
import torch

from models.tt_dit.pipelines.wan.pipeline_wan import WanPipelineConfig
from models.tt_dit.tests.models.wan2_2.test_performance_wan import test_pipeline_performance  # noqa: F401


@pytest.fixture(autouse=True)
def pinned_stock_checkpoint(monkeypatch):
    checkpoint = Path(os.environ["WAN_CHECKPOINT"])
    assert (checkpoint / "model_index.json").is_file()
    original = WanPipelineConfig.default

    def default(cls, **kwargs):
        assert kwargs["checkpoint_name"] == "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
        kwargs["checkpoint_name"] = str(checkpoint)
        return original(**kwargs)

    monkeypatch.setattr(WanPipelineConfig, "default", classmethod(default))
    previous = torch.get_num_threads()
    torch.set_num_threads(16)
    print(f"Pinned Wan checkpoint: {checkpoint}", flush=True)
    try:
        yield
    finally:
        torch.set_num_threads(previous)
