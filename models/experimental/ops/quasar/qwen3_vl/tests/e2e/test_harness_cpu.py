# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests of the e2e harness; no device needed."""
from pathlib import Path

import pytest

from models.experimental.ops.quasar.qwen3_vl.tests.e2e.config import HF_MODEL_ID, RunConfig, parse_grid
from models.experimental.ops.quasar.qwen3_vl.tests.e2e.presets import PRESETS, build_inputs


def _opts(**over):
    base = {
        "--qwen-size": "tiny",
        "--qwen-vision-layers": 2,
        "--qwen-text-layers": 2,
        "--qwen-decode-steps": 1,
        "--qwen-deepstack-at": None,
        "--qwen-kv-blocks": None,
        "--qwen-host-ops": "",
        "--qwen-disable-wa": "",
        "--qwen-quasar-config": False,
        "--qwen-expect-grid": None,
        "--qwen-run-dir": "/tmp/x",
    }
    base.update(over)
    return base.__getitem__


def test_run_config_defaults():
    cfg = RunConfig.from_options(_opts())
    assert (cfg.vision_layers, cfg.text_layers, cfg.decode_steps) == (2, 2, 1)
    assert cfg.deepstack_at is None and cfg.host_ops == () and cfg.run_dir == Path("/tmp/x")


def test_run_config_lists_and_grid():
    cfg = RunConfig.from_options(
        _opts(**{"--qwen-host-ops": "linear, rms_norm", "--qwen-expect-grid": "3x2", "--qwen-deepstack-at": 0})
    )
    assert cfg.host_ops == ("linear", "rms_norm")
    assert cfg.expect_grid == (3, 2) and cfg.deepstack_at == 0


def test_parse_grid_rejects_garbage(expect_error):
    with expect_error(ValueError, "grid must look like"):
        parse_grid("3by2")


def test_cache_key_covers_everything():
    a = RunConfig.from_options(_opts())
    keys = {
        a.cache_key((3, 2)),
        a.cache_key((8, 4)),
        RunConfig.from_options(_opts(**{"--qwen-text-layers": 3})).cache_key((3, 2)),
        RunConfig.from_options(_opts(**{"--qwen-deepstack-at": 0})).cache_key((3, 2)),
        RunConfig.from_options(_opts(**{"--qwen-quasar-config": True})).cache_key((3, 2)),
    }
    assert len(keys) == 5


@pytest.mark.parametrize(
    "name, grid, image_tokens, seq",
    [("tiny", [1, 16, 16], 64, 78), ("demo", [1, 86, 128], 2752, 2766)],
)
def test_preset_token_counts(name, grid, image_tokens, seq):
    from transformers import AutoProcessor

    inputs = build_inputs(PRESETS[name], AutoProcessor.from_pretrained(HF_MODEL_ID))
    assert inputs["image_grid_thw"][0].tolist() == grid
    assert int(inputs["image_grid_thw"][0].prod()) // 4 == image_tokens
    assert inputs["input_ids"].shape[1] == seq


def test_preset_kv_capacity_covers_seq():
    for p in PRESETS.values():
        assert p.kv_blocks * p.block_size >= p.max_seq_len
