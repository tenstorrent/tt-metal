# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""The qwen3 example's tensor-parallel model through ``ttml.distributed_checkpointing``, saved on one TP size and
loaded on another: each world in its own process (``dcp_qwen3_driver.py``), as in ``test_dcp_cross_world.py``.

Every case saves on a ``[1, 2]`` TP mesh after two training steps and loads onto a fresh model: on the same world,
where everything must come back exactly, and on TP 4 (a ``[1, 4]`` line, which a QuietBox and a loudbox open) or TP 8
(a loudbox's eight chips), where the parameters, the AdamW moments and the train state must come back exactly and the
logits and the step after the load within tolerance. The variants cover what the qwen3 example declares for
checkpoints: the fused ``kv_proj``, the tied LM head padded to ``32 * tp`` rows (vocab 300: 320 rows at TP 2, 384 at
TP 4, 512 at TP 8), the untied embedding padded the same way while the untied LM head keeps the logical vocab (vocab
320: shards that end inside a tile at TP 4 and 8), and LoRA adapters on every projection, ``lora_B`` following
``kv_proj``'s fused rows. A host with too few chips for a world skips that case, as a QuietBox skips TP 8.
"""

import os
import sys

import pytest
from dcp_harness import MGD_DIR, run_checked

# As in test_dcp_cross_world: a hung child fails after 10 minutes, so one hang leaves a 25-minute CI leg the time to
# report the tests after it.
CHILD_TIMEOUT_S = 600
pytestmark = [pytest.mark.requires_device, pytest.mark.timeout(2 * CHILD_TIMEOUT_S + 60)]

DRIVER = [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "dcp_qwen3_driver.py")]

WORLDS = {  # name: (mesh dp x tp, descriptor, devices needed)
    "tp2": ("1x2", "bh_1_2_line_line.textproto", 2),
    "tp4": ("1x4", "bh_1_4_line_line.textproto", 4),
    "tp8": ("1x8", "bh_1_8_line_line.textproto", 8),
}
MODELS = {  # name: the save world's model flags
    "tied": ("--tie", "--vocab", "300"),
    "untied": ("--vocab", "320"),
    "lora": ("--tie", "--vocab", "300", "--lora"),
}
CASES = [
    ("tied", "tp2", ("--same-layout",)),
    ("tied", "tp4", ()),
    ("tied", "tp8", ()),
    ("untied", "tp4", ()),
    ("untied", "tp8", ()),
    ("lora", "tp4", ()),
    ("lora", "tp8", ()),
]


def _world_args(name: str) -> list[str]:
    mesh, mgd, _ = WORLDS[name]
    return ["--mesh", mesh, "--mgd", os.path.join(MGD_DIR, mgd)]


def _run(args: list[str], what: str) -> str:
    lines = run_checked(DRIVER + args, what, CHILD_TIMEOUT_S, needs_tt_run=False, save=args[0] == "save")
    return "\n".join(lines)


_SAVED: dict[str, str] = {}


def _saved_checkpoint(model: str, tmp_path_factory) -> str:
    """Save once per model: the cases that load the same model share its TP 2 checkpoint."""
    if model not in _SAVED:
        ckpt = str(tmp_path_factory.mktemp(f"qwen3_{model}") / "step_0000002")
        _run(["save", "--dir", ckpt, *_world_args("tp2"), *MODELS[model]], f"{model} save on tp2")
        _SAVED[model] = ckpt
    return _SAVED[model]


@pytest.mark.parametrize("model,load_world,flags", CASES, ids=[f"{m}_tp2->{w}" for m, w, _ in CASES])
def test_qwen3_checkpoint_loads_onto_another_tp_size(
    tmp_path_factory, release_metal_env, skip_if_host_too_small, model, load_world, flags
):
    skip_if_host_too_small((WORLDS[load_world][2],), f"{model} on {load_world}")
    release_metal_env()  # the children open the chips
    ckpt = _saved_checkpoint(model, tmp_path_factory)
    out = _run(["load", "--dir", ckpt, *_world_args(load_world), *flags], f"{model} load on {load_world}")
    assert "RESULT: OK" in out, out.splitlines()[-8:]
