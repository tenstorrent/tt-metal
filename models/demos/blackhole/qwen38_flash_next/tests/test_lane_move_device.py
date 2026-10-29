# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""W5 device test: ``Qwen38PlainLaneDecoder.move_lane`` (compaction) against the same run without the move.

a. ``test_lane_move_matches_unmoved``: 32 lanes, buckets (8, 16, 32), the bucket forced to 16 for the whole run; user A
   in lane 0 and user B in lane 9 (two prompts); 8 lock-step steps; mode ``move`` then moves lane 9 to lane 1 (and syncs
   the page table and parks as the adapter does), mode ``stay`` does nothing; 12 more steps.  Each run saves, per step,
   the tokens and the fp32 logits rows of A's lane and B's lane to ``$QWEN38_BUCKET_TEST_OUT/<mode>.pt`` (the move file
   also holds ``last_move["ms"]``).  Run it once per mode (the two runs do not share a device session).
b. ``test_lane_move_compare``: the two files must agree: equal tokens, bit-equal logits for A and for B.

Run:  QWEN38_BUCKET_TEST_OUT=<dir> pytest models/demos/blackhole/qwen38_flash_next/tests/test_lane_move_device.py -v -s -k "matches_unmoved and stay"
Env:  as test_batch_buckets_device.py (HF_MODEL, TT_CACHE_PATH, TT_METAL_TRACE_ALLOC_TRACKING=1, MESH_DEVICE).
"""

import pytest
import torch
from loguru import logger

from models.common.utility_functions import run_for_blackhole
from models.demos.blackhole.qwen38_flash_next.checkpoint import checkpoint_directory
from models.demos.blackhole.qwen38_flash_next.demo.text_demo import (  # noqa: F401  (device_params is a fixture)
    DEVICE_PARAMS,
    device_params,
)
from models.demos.blackhole.qwen38_flash_next.tests.test_batch_buckets_device import (
    lock_step,
    needle_prompts,
    open_decoder,
    out_dir,
)
from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import MESH_SHAPE

BUCKETS = (8, 16, 32)
FORCED_BUCKET = 16
LANE_A, LANE_B, LANE_B_MOVED = 0, 9, 1
BEFORE_MOVE, AFTER_MOVE = 8, 12


@run_for_blackhole()
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mode", ["stay", "move"])
@pytest.mark.timeout(5400)
def test_lane_move_matches_unmoved(mesh_device, mode):
    out = out_dir()
    prompts, _tokenizer, _config = needle_prompts(checkpoint_directory(), 2, None)
    with open_decoder(mesh_device, 32, prompts, BEFORE_MOVE + AFTER_MOVE + 1, BUCKETS) as (decoder, _vm, _ckpt):
        decoder.force_bucket = FORCED_BUCKET
        current = [0] * 32
        current[LANE_A] = decoder.admit(LANE_A, prompts[0])
        current[LANE_B] = decoder.admit(LANE_B, prompts[1])
        lane_b = LANE_B
        saved = {"mode": mode, "first": [current[LANE_A], current[LANE_B]], "tokens": [], "logits": [], "move_ms": None}
        for step in range(BEFORE_MOVE + AFTER_MOVE):
            if step == BEFORE_MOVE and mode == "move":
                decoder.move_lane(
                    LANE_B, LANE_B_MOVED
                )  # as the adapter after a compaction: table upload, then the parking
                decoder.sync_page_table()
                current[LANE_B_MOVED], current[LANE_B] = current[LANE_B], 0
                lane_b = LANE_B_MOVED
                saved["move_ms"] = decoder.last_move["ms"]
                logger.info(f"moved lane {LANE_B} -> {LANE_B_MOVED}: {decoder.last_move}")
            live = [LANE_A, lane_b]
            current = lock_step(decoder, current, live)
            saved["tokens"].append([current[LANE_A], current[lane_b]])
            saved["logits"].append(decoder.read_full_logits()[live].to(torch.float32).clone())
        saved["logits"] = torch.stack(saved["logits"])  # [steps, 2 (A, B), vocab]
        assert decoder.active.lanes == FORCED_BUCKET and decoder.bucket_steps[FORCED_BUCKET] == BEFORE_MOVE + AFTER_MOVE
        assert decoder.lane_moves == (1 if mode == "move" else 0)
        torch.save(saved, out / f"{mode}.pt")
        logger.info(f"{mode}: bucket steps {decoder.bucket_steps}, move ms {saved['move_ms']}")


def test_lane_move_compare():
    out = out_dir()
    stay_path, move_path = out / "stay.pt", out / "move.pt"
    if not stay_path.exists() or not move_path.exists():
        pytest.skip(f"{stay_path.name} and {move_path.name} are not both in {out}: run the stay and move cases first")
    stay, move = torch.load(stay_path), torch.load(move_path)
    assert stay["first"] == move["first"]
    logger.info(f"move took {move['move_ms']} ms")
    for step, (a, b) in enumerate(zip(stay["tokens"], move["tokens"])):
        assert a == b, f"step {step}: tokens differ stay {a} move {b} (user A, user B)"
    assert stay["logits"].shape == move["logits"].shape
    if not torch.equal(stay["logits"], move["logits"]):
        diff = (stay["logits"] - move["logits"]).abs()
        per_step_user = diff.reshape(diff.shape[0], diff.shape[1], -1).amax(-1)
        step, user = [int(i) for i in (per_step_user > 0).nonzero()[0]]
        pytest.fail(
            f"logits differ: first at step {step} user {'AB'[user]}, max abs diff there "
            f"{per_step_user[step, user]:.6g}, overall {diff.max():.6g}"
        )
