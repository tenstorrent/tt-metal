# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Why nlp_create_qkv_heads costs ~47 us in every Talker prefill bucket.

The interleaved program factory's unit of work is a TILE-ROW of the sequence
(``nlp_create_qkv_heads_program_factory.cpp:61``):

    num_blocks = shape[0] * shape[1] * shape[2] / TILE_HEIGHT

so a ``[1, 1, 64, 4096]`` QKV tensor is **2 blocks -> 2 cores** and ``[1,1,32,4096]`` is
1 block -> 1 core. The 4096-wide axis contributes no parallelism at all, which is why the
op measures the same ~47-48 us at seq=32, 64 and 128.

The sharded factory instead parallelises over the output shard grid
(``:402  num_cores = max(q_cores.num_cores(), k_cores.num_cores())``). Attention builds
specs for it in ``_build_sharded_nlp_memcfgs`` — but for the Talker (num_heads=16)
``q_grid`` is one ROW, ``CoreRange((0,0), (15,0))``, bounding box ``[0-0 - 15-0]``: x=8..15
are off an 8x8 compute grid. This probe measures each arm and prints why any is refused.

Writes ``generated/qkv_split_manifest.json`` (arms in run order, REPS launches each);
score it with ``qkv_split_probe_report.py`` after the tracy run, because the profiler
CSV is only written at process exit.

    python -m tracy -p -v -r --op-support-count 100000 \
        -m pytest -s -q models/demos/qwen3_tts/tests/test_qwen3_tts_qkv_split_probe.py
    python models/demos/qwen3_tts/tests/qkv_split_probe_report.py
"""

from __future__ import annotations

import json
import os

import pytest
import torch

import ttnn

TILE = 32
HEADS, KV_HEADS, HEAD_DIM = 16, 8, 128
FUSED_QKV = (HEADS + 2 * KV_HEADS) * HEAD_DIM  # 4096
REPS = 4
MANIFEST = "generated/qkv_split_manifest.json"
_MANIFEST_ARMS: list = []


def _why(e: Exception) -> str:
    """First TT_FATAL / 'Error' line, which is the part that says what was rejected."""
    for line in str(e).splitlines():
        s = line.strip()
        if "TT_FATAL" in s or s.startswith("RuntimeError") or "must be" in s or "not supported" in s:
            return s[:150]
    return str(e).splitlines()[0][:150] if str(e).splitlines() else repr(e)[:150]


@pytest.fixture(scope="module")
def device():
    d = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1), l1_small_size=32768)
    d.enable_program_cache()
    yield d
    ttnn.close_mesh_device(d)
    os.makedirs("generated", exist_ok=True)
    with open(MANIFEST, "w") as f:
        json.dump({"reps": REPS, "arms": _MANIFEST_ARMS}, f, indent=2)
    print(f"\nwrote {MANIFEST}: {len(_MANIFEST_ARMS)} arms x {REPS} reps")


@pytest.mark.parametrize("m", [32, 64, 128], ids=["seq32", "seq64", "seq128"])
def test_qkv_split_configs(device, m):
    cg = device.compute_with_storage_grid_size()
    torch.manual_seed(0)
    x = torch.randn(1, 1, m, FUSED_QKV, dtype=torch.bfloat16)

    qkv_shard_w = (HEADS // KV_HEADS + 2) * HEAD_DIM  # 512
    qkv_grid = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(FUSED_QKV // qkv_shard_w - 1, 0))}
    )
    in_sharded = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(qkv_grid, (m, qkv_shard_w), ttnn.ShardOrientation.ROW_MAJOR),
    )
    row_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(HEADS - 1, 0))})
    legal_grid = ttnn.num_cores_to_corerangeset(HEADS, cg, True)
    print(
        f"\n### m={m}  q_grid today bbox={row_grid.bounding_box()} | "
        f"legal bbox={legal_grid.bounding_box()}  (compute grid {cg.x}x{cg.y})"
    )

    def _hs(grid):
        return ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(grid, (m, HEAD_DIM), ttnn.ShardOrientation.ROW_MAJOR),
        )

    arms = [
        ("interleaved (what prefill runs today)", ttnn.L1_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG),
        ("sharded, q_grid = 16-wide row (today's spec)", in_sharded, _hs(row_grid)),
        ("sharded, q_grid = num_cores_to_corerangeset", in_sharded, _hs(legal_grid)),
    ]

    for label, in_mc, out_mc in arms:
        tag = f"m{m}:{label}"
        try:
            xt = ttnn.from_torch(x, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=in_mc)
        except Exception as e:
            print(f"  {label:46s} -> INPUT REFUSED: {_why(e)}")
            continue
        try:
            for _ in range(REPS + 1):  # first launch is compile/cold; report drops it
                q, k, v = ttnn.experimental.nlp_create_qkv_heads(
                    xt, num_heads=HEADS, num_kv_heads=KV_HEADS, transpose_k_heads=False, memory_config=out_mc
                )
                ttnn.synchronize_device(device)
                for t in (q, k, v):
                    ttnn.deallocate(t)
            _MANIFEST_ARMS.append({"tag": tag, "m": m, "label": label, "reps": REPS})
            print(f"  {label:46s} -> ran, {REPS} timed launches")
        except Exception as e:
            print(f"  {label:46s} -> REFUSED: {_why(e)}")
        ttnn.deallocate(xt)
