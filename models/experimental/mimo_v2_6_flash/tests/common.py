# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import contextlib
import math
import os
import threading
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.mimo_v2_6_flash.reference.checkpoint import MiMoCheckpoint, default_model_path
from models.experimental.mimo_v2_6_flash.reference.config import MiMoConfig
from models.experimental.mimo_v2_6_flash.tt.generator import create_dp_submeshes

# CCL ops keep their semaphores for the program's lifetime. Without an L1_SMALL region they go in the
# main L1 pool, and after enough distinct prefill shapes they fragment it badly enough to clash with
# later ops' static circular buffers. all_gather places them in L1_SMALL when one exists.
L1_SMALL_SIZE = 65536
GALAXY_DEVICE_PARAMS = {
    "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
    "trace_region_size": 0,
    "l1_small_size": L1_SMALL_SIZE,
}
DP = 4
TP = 8


def model_path():
    path = default_model_path()
    if not os.path.exists(os.path.join(path, "config.json")):
        pytest.skip(f"MiMo checkpoint not found at {path} (set MIMO_MODEL_PATH)")
    return path


def load_config_and_ckpt():
    path = model_path()
    cfg = MiMoConfig.from_model_path(path)
    return cfg, MiMoCheckpoint(path, cfg)


def cache_dir(tag):
    """Converted TTNN weights for ``tag``, kept per checkpoint: MOPD and RL share tensor names, not values."""
    root = os.environ.get("MIMO_TT_CACHE", "/home/zni/models/mimo-tt-cache")
    return os.path.join(root, os.path.basename(os.path.normpath(model_path())), tag)


def golden_path():
    """Golden hidden states / logits of reference/make_golden.py, kept per checkpoint (MIMO_GOLDEN overrides)."""
    ckpt_name = os.path.basename(os.path.normpath(model_path()))
    path = os.environ.get("MIMO_GOLDEN", os.path.join("/home/zni/models/mimo-golden", ckpt_name, "golden_48l.pt"))
    if not os.path.exists(path):
        pytest.skip(f"golden file {path} not found; run reference/make_golden.py")
    return path


@contextlib.contextmanager
def heartbeat(what, every_sec=60):
    """Log a line every ``every_sec`` seconds while the block runs.

    The tt-device-mcp broker reaps a job that prints nothing for 300 s as hung
    (TT_DEVICE_MCP_HUNG_SILENCE_SEC). A long prefill waits on the device in silence, so long
    tests wrap those waits in this."""
    stop = threading.Event()
    t0 = time.time()

    def beat():
        while not stop.wait(every_sec):
            logger.info(f"{what}: still running after {time.time() - t0:.0f} s")

    thread = threading.Thread(target=beat, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join()


def pcc(a, b):
    """Pearson correlation. Raises on NaN: callers take min() over scores, and min() would skip a NaN score."""
    a = a.flatten().double()
    b = b.flatten().double()
    score = torch.corrcoef(torch.stack([a, b]))[0, 1].item()
    assert math.isfinite(score), "PCC is not finite: an input holds NaN/Inf or is constant"
    return score


# Two runs of the same prompt (one-shot vs chunked prefill, say) round their logits differently, so where the top two
# logits tie to within bf16 rounding (a step is 2**-7 of the logit's power of two) either run may pick either token.
TIE_STEPS = 2


def tie_margin(logits, token):
    """How far ``token`` scores below the best logit, in bf16 steps at the best logit's magnitude."""
    best = float(logits.max())
    step = 2.0 ** (math.floor(math.log2(abs(best))) - 7)
    return (best - float(logits[token])) / step


def near_top(logits, token):
    """``token`` scores within TIE_STEPS bf16 steps of the best logit (ties from rounding the logits to bf16)."""
    return tie_margin(logits, token) <= TIE_STEPS


def first_divergence(tokens, logits, ref_tokens, ref_logits):
    """Two greedy runs compared up to their first different token (``logits[k]`` is what picked ``tokens[k]``).

    None if they agree throughout, else (k, near_tie) at the first difference k: near_tie when each run's pick is
    near_top in the other run's logits there. Later tokens are not compared: the runs' histories fork at k.
    """
    for k, (t, ref) in enumerate(zip(tokens, ref_tokens)):
        if t != ref:
            return k, near_top(ref_logits[k], t) and near_top(logits[k], ref)
    return None


def dp_submeshes(mesh_device):
    """The 4x8 Galaxy split into its four 1x8 rows: one TP=8 replica per row."""
    assert tuple(mesh_device.shape) == (DP, TP), f"expected a {DP}x{TP} mesh, got {tuple(mesh_device.shape)}"
    subs = create_dp_submeshes(mesh_device, DP)  # also registers the streaming kernels' bank cores per row
    assert len(subs) == DP
    return subs


def block_page_table(max_batch, blocks_per_user, seed=0):
    """A shuffled slot -> block mapping, so the paged path is exercised with non-contiguous blocks."""
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(max_batch * blocks_per_user, generator=g)
    return perm.reshape(max_batch, blocks_per_user).to(torch.int32)
