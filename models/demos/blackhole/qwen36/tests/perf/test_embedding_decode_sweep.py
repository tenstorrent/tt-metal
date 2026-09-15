# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Sweep the DECODE token-embedding over its layout / batch-shape choices.

WHY
---
MEASURED (T3K TP=8, 27B, tests/perf/test_profile_model_tail_decode.py), decode embedding costs:

    B=1    EmbeddingsDeviceOperation  1.5 us  +  TilizeWithValPaddingDeviceOperation  55.4 us  (1 core)
    B=32   EmbeddingsDeviceOperation  7.9 us  +  (none)

i.e. B=1 pays 55 us -- 2.7% of the whole B=1 model tail -- for a tilize that B=32 does not pay at
all, on a tensor of 20 tiles. 55 us to write ~40 KB is ~0.7 GB/s, so it is kernel overhead, not
bandwidth.

The cause is exact and is in ttnn: ``embedding.cpp`` sets ``fused_tilized`` only when
``input.padded_shape()[-1] % TILE_HEIGHT == 0`` -- the TOKEN tensor's last dim. Decode ids arrive
``[B, 1]`` and ``tp_common.decode_ids_for_embed`` flattens them to ``[1, B]``, so the last dim IS B:
32 at serving batch (fuses), 1 at B=1 (does not). When it does not fuse, the op emits ROW_MAJOR and
``ttnn::to_layout`` appends the separate tilize.

Two independent levers, hence four variants:

``cur``          what ships at TP=8: ``[1, B]`` ids, no memory_config. (``args.emb_decode_memcfg``
                 is gated to ``wh_9b_n300`` in model_config.py, so the 27B never passes one.)
``shard``        same ids, but pass the WIDTH-SHARDED L1 output config. That config exists for the
                 9B/N300 decode path (21 -> 3 us there) and is about the OUTPUT placement, not the
                 fusion, so it may well parallelize the tilize at B=1 too -- the existing gate skips
                 B=1 only because B=1 fails the FUSION precondition, which is a different thing.
``pad32``        widen the id row to 32 so the fusion fires, then slice row 0 back out. The catch is
                 that ``embedding`` reshapes its output to ``{batch, sentence, dim}``, so a 32-wide
                 id row yields ``[1, 32, dim]`` and the slice is a SUB-TILE row slice -- which may
                 cost as much as the tilize it removes. That is the question this variant answers.
``pad32_shard``  both.

Every variant is checked against ``cur``'s output on the real weights before it is timed, so a fast
wrong answer cannot win.

Run (needs a device; skipped unless the env var is set)::

    QWEN_EMB_SWEEP=1 MESH_DEVICE=T3K HF_MODEL=Qwen/Qwen3.6-27B \\
      pytest models/demos/blackhole/qwen36/tests/perf/test_embedding_decode_sweep.py -v -s

Under tracy, each variant is signposted; read DEVICE KERNEL DURATION. Wall clock is reported too but
the variants differ in op count, and the served decode path runs inside a captured trace where host
dispatch is free -- so device time is the number that decides this.
"""

from __future__ import annotations

import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_wormhole_b0_or_blackhole

NUM_LAYERS = 1
BATCHES = [1, 8, 32]
ITERS = 3
_SKIP = os.environ.get("QWEN_EMB_SWEEP") != "1"

try:
    from tracy import signpost as _SP
except ImportError:  # pragma: no cover
    _SP = None


def _mesh_shape():
    name = (os.environ.get("MESH_DEVICE") or "").upper()
    explicit = {"P150": (1, 1), "N150": (1, 1), "P150X4": (1, 4), "N150X4": (1, 4), "N300": (1, 2), "T3K": (1, 8)}
    return explicit.get(name, (1, max(1, min(ttnn.get_num_devices(), 2))))


MESH_SHAPE = _mesh_shape()
_MULTI = MESH_SHAPE != (1, 1)
DEVICE_PARAMS = [
    {
        "l1_small_size": 24576,
        "num_command_queues": 2,
        **({"fabric_config": ttnn.FabricConfig.FABRIC_1D} if _MULTI else {}),
    }
]


@pytest.mark.skipif(_SKIP, reason="set QWEN_EMB_SWEEP=1 to run the decode embedding sweep")
@pytest.mark.timeout(1800)
@run_for_wormhole_b0_or_blackhole()
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("B", BATCHES, ids=[f"batch{b}" for b in BATCHES])
def test_embedding_decode_sweep(mesh_device, device_params, B):
    """Correctness-checked sweep of the decode embedding's layout choices."""
    del device_params
    from models.demos.blackhole.qwen36.tt import tp_common as tpc
    from models.demos.blackhole.qwen36.tt.model import Qwen36Model

    mesh_device.enable_program_cache()
    model = Qwen36Model.from_pretrained(mesh_device, max_batch_size=B, max_seq_len=256, n_layers=NUM_LAYERS)
    emb = model.embd
    nd = max(1, model.num_devices)
    dim_frac = model.args.dim // nd
    shard_cfg = tpc.create_activation_shard_config(dim_frac)
    logger.info(f"emb sweep: B={B} dim_frac={dim_frac} mesh={MESH_SHAPE} shard={shard_cfg is not None}")

    torch.manual_seed(0)
    ids = torch.randint(0, 2000, (B, 1), dtype=torch.int32)
    rep = ttnn.ReplicateTensorToMesh(mesh_device) if _MULTI else None

    def _mk(t):
        return ttnn.from_torch(t, dtype=ttnn.uint32, device=mesh_device, **({"mesh_mapper": rep} if rep else {}))

    tok_flat = _mk(tpc.decode_ids_for_embed(ids))  # [1, B] -- what ships
    # 32-wide id row: real ids in the first B slots, the rest are don't-care rows we slice away.
    W = 32
    padded_ids = torch.zeros(1, W, dtype=torch.int32)
    padded_ids[0, :B] = ids.reshape(-1)[:B]
    tok_pad = _mk(padded_ids) if B < W else tok_flat

    def _cur(mc=None):
        return emb(tok_flat, memory_config=mc) if mc is not None else emb(tok_flat)

    def _pad(mc=None):
        out = emb(tok_pad, memory_config=mc) if mc is not None else emb(tok_pad)
        if B == W:
            return out
        sliced = ttnn.slice(out, (0, 0, 0), (1, B, out.shape[-1]))
        ttnn.deallocate(out)
        return sliced

    def _devpad(mc=None):
        """The IMPLEMENTABLE form: widen the id row on DEVICE, inside decode_embed.

        pad32/pad32_shard widen the id tensor on the host, which is not available to the shipped
        path: the decode id buffer is persistent and its address is baked into the decode trace, so
        changing its width means changing every call site that allocates it. Padding on device
        instead keeps the change inside tp_common.decode_embed -- at the cost of one extra op, which
        is what this variant prices.
        """
        if B >= W:
            return _cur(mc)
        wide = ttnn.pad(tok_flat, [(0, 0), (0, W - B)], value=0)
        out = emb(wide, memory_config=mc) if mc is not None else emb(wide)
        ttnn.deallocate(wide)
        sliced = ttnn.slice(out, (0, 0, 0), (1, B, out.shape[-1]))
        ttnn.deallocate(out)
        return sliced

    variants = {
        "cur": lambda: _cur(None),
        "shard": lambda: _cur(shard_cfg),
        "pad32": lambda: _pad(None),
        "pad32_shard": lambda: _pad(shard_cfg),
        "devpad_shard": lambda: _devpad(shard_cfg),
    }

    comp = ttnn.ConcatMeshToTensor(mesh_device, dim=0) if _MULTI else None

    def _to_host(t):
        r = ttnn.to_torch(t, mesh_composer=comp) if comp else ttnn.to_torch(t)
        return r.float().reshape(-1, r.shape[-1])[:B]

    ref = None
    results = []
    for name, fn in variants.items():
        try:
            out = fn()
            ttnn.synchronize_device(mesh_device)
            host = _to_host(out)
            ttnn.deallocate(out)
            if ref is None:
                ref = host
                ok = True
            else:
                ok = bool(torch.allclose(ref, host, atol=1e-3, rtol=1e-3))
        except Exception as e:  # a rejected combination is a result, not a failure
            logger.warning(f"  {name:12} UNSUPPORTED: {type(e).__name__}: {str(e)[:150]}")
            results.append((name, None, "unsupported"))
            continue

        if _SP is not None:
            _SP(f"{name}_start")
        t0 = time.time()
        for _ in range(ITERS):
            o = fn()
            ttnn.deallocate(o)
        ttnn.synchronize_device(mesh_device)
        us = (time.time() - t0) / ITERS * 1e6
        if _SP is not None:
            _SP(f"{name}_stop")
        results.append((name, us, "OK" if ok else "WRONG"))
        logger.info(f"  {name:12} {us:9.1f} us/call (wall)   {results[-1][2]}")

    logger.info(f"=== emb sweep B={B} summary ===")
    for name, us, ok in results:
        logger.info(f"  {name:12} {'n/a' if us is None else f'{us:9.1f} us':>12}   {ok}")
    ttnn.deallocate(tok_flat)
    if tok_pad is not tok_flat:
        ttnn.deallocate(tok_pad)
    # Reports; only the shipped variant is gated.
    assert results[0][2] == "OK", f"reference variant failed: {results[0]}"
