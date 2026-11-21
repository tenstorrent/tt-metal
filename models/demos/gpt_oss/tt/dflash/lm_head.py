# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The DFlash multi-row LM head's weight stream: the fused decode terminal's per-device vocab split (device d owns ids
[d * vd, (d + 1) * vd), fused_decode/terminal.py) with each device's columns padded so they split evenly over `readers`
cores per DRAM bank (vd = 50272 is 1571 tiles = 197 columns per bank, a prime: one core per bank). The 8-row argmax
LM head (LinearStream out_mode 6) is compute-bound on one core per bank; padded columns carry a -1e30 bias and never
win the argmax."""

import json
import os
from pathlib import Path

import torch

import ttnn

from ..fused_decode.config import LM_HEAD_DECODE_WEIGHT_DTYPE
from ..fused_decode.stream import as_stream_tensor, linear_stream_rows, mark_cache_built, stream_linear_layout
from ..fused_decode.terminal import MASK

TILE = ttnn.TILE_SIZE


def _hf_lm_head():
    """The checkpoint's LM-head weight (HF_MODEL: a local checkpoint directory or a HuggingFace id, as
    tt/model_config.py)."""
    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    model = os.getenv("HF_MODEL", "openai/gpt-oss-20b")

    def fetch(patterns):
        return Path(model) if os.path.isdir(model) else Path(snapshot_download(model, allow_patterns=patterns))

    idx = json.loads((fetch(["*.json"]) / "model.safetensors.index.json").read_text())["weight_map"]
    name = "lm_head.weight"
    with safe_open(str(fetch([idx[name]]) / idx[name]), "pt") as f:
        return f.get_tensor(name)


def padded_lm_head(
    mesh_device, terminal, readers, cache_dir=None, dtype=LM_HEAD_DECODE_WEIGHT_DTYPE, round_robin=False
):
    """-> (weights, n): the per-device [hidden, n] LM-head stream (n = vd padded to banks * readers columns).
    round_robin: column tile n on bank n % banks, slot n / banks (the rescore's random column reads spread over the
    banks; contiguous ranges put neighbouring vocabulary tiles on one bank)."""
    banks = mesh_device.dram_grid_size().x
    vd, hidden, tp, vocab = terminal.vd, terminal.hidden, terminal.tp, terminal.vocab_size
    step = banks * readers
    n_tiles = -(-(vd // TILE) // step) * step
    n = n_tiles * TILE
    rows = linear_stream_rows(mesh_device, hidden, n)
    dt = {ttnn.bfloat8_b: "bfp8", ttnn.bfloat4_b: "bfp4"}[dtype]
    cache = (
        None
        if cache_dir is None
        else f"{cache_dir}/dflash_lm_head_{dt}_vd{vd}_n{n}_b{banks}" + ("_rr" if round_robin else "")
    )
    C = n_tiles // banks
    order = [i * banks + b for b in range(banks) for i in range(C)] if round_robin else None
    layouts = None
    marker = None if cache is None else Path(cache + ".built")  # as_tensor's cache file name carries extra suffixes
    if marker is None or not marker.exists():
        w = _hf_lm_head().transpose(0, 1).float()  # [hidden, vocab]
        layouts = []
        for d in range(tp):
            lo, hi = d * vd, min((d + 1) * vd, vocab)
            wd = torch.zeros(hidden, n)
            wd[:, : hi - lo] = w[:, lo:hi]
            bias = torch.zeros(n)
            bias[hi - lo :] = MASK
            layouts.append(stream_linear_layout(wd, bias, banks, order))
        assert layouts[0].shape[0] == rows
    t = as_stream_tensor(mesh_device, layouts, rows, dtype, cache)
    mark_cache_built(marker)
    return t, n
