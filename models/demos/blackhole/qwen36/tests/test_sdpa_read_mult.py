# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""TT_SDPA_DECODE_READ_MULT (sdpa_decode program factory + reader_decode_all.cpp): K/V read pacing for the decode SDPA.

The env multiplies the reader's outstanding one-tile read budget (get_barrier_read_threshold) via the
SDPA_READ_BARRIER_MULT reader define. It changes only when the reader drains, never which tiles land where, so the
fused spec-verify SDPA output must be bit-identical with it on.

  test_read_mult_source_guards   host: the define is guarded (unset env = today's kernel source) and the factory only
                                 adds it for MULT > 1.
  test_read_mult_bit_identical   device (chip 0, no fabric): the KV-split verify shape (Tg=4, 6 packed q heads, bf8
                                 paged KV) at 4K / 32K, OFF vs MULT=4 in separate processes (the env is read when the
                                 program is built and is not part of the program hash), md5 of the valid rows equal.

Run:  pytest models/demos/blackhole/qwen36/tests/test_sdpa_read_mult.py -v -s   (device test needs TT_VISIBLE_DEVICES)
"""
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[5]
D = ROOT / "ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device"

_BENCH = r'''
import hashlib, math, os, torch, ttnn
BLOCK, TG, NH, HD = 64, 4, 6, 256
dev = ttnn.CreateDevice(device_id=0)
grid = dev.compute_with_storage_grid_size()
for ctx, G in ((4096, 1), (32768, 2)):
    nb = math.ceil((ctx + TG) / BLOCK); n = G * nb
    g = torch.Generator().manual_seed(11)
    k = torch.randn(n, 1, BLOCK, HD, generator=g).to(torch.bfloat16)
    v = torch.randn(n, 1, BLOCK, HD, generator=g).to(torch.bfloat16)
    pt = torch.randperm(n, generator=g).reshape(G, nb).to(torch.int32)
    pos = torch.tensor([ctx - TG + j for _ in range(G) for j in range(TG)], dtype=torch.int32)
    q = torch.randn(1, G, TG * 32, HD, generator=g).to(torch.bfloat16)
    f = lambda t, dt, lay: ttnn.from_torch(t, dtype=dt, layout=lay, device=dev)
    cfg = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=(grid.x, grid.y), exp_approx_mode=False,
                                 q_chunk_size=0, k_chunk_size=0,
                                 max_cores_per_head_batch=min(64, max(1, (grid.x * grid.y) // G)))
    o = ttnn.transformer.paged_scaled_dot_product_attention_decode(
        f(q, ttnn.bfloat16, ttnn.TILE_LAYOUT), f(k, ttnn.bfloat8_b, ttnn.TILE_LAYOUT), f(v, ttnn.bfloat8_b, ttnn.TILE_LAYOUT),
        page_table_tensor=f(pt, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT), cur_pos_tensor=f(pos, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT),
        scale=HD ** -0.5, program_config=cfg, memory_config=ttnn.L1_MEMORY_CONFIG, spec_multi_pos_tiles=TG,
        spec_pack_heads=NH)
    out = ttnn.to_torch(o).float().contiguous()
    print(f"MD5 {ctx} {G} {hashlib.md5(out.numpy().tobytes()).hexdigest()}", flush=True)
ttnn.close_device(dev)
'''


def test_read_mult_source_guards():
    reader = (D / "kernels/dataflow/reader_decode_all.cpp").read_text()
    fac = (D / "sdpa_decode_program_factory.cpp").read_text()
    m = re.search(r"#ifdef SDPA_READ_BARRIER_MULT(.*?)#else(.*?)#endif", reader, re.S)
    assert m, "reader: SDPA_READ_BARRIER_MULT must be an #ifdef with today's line in the #else"
    assert "* SDPA_READ_BARRIER_MULT" in m.group(1)
    assert "get_barrier_read_threshold<q_tile_bytes, num_cores>();" in m.group(2)
    assert 'getenv("TT_SDPA_DECODE_READ_MULT")' in fac
    assert re.search(r"if \(mult > 1\)\s*\{\s*reader_desc\.defines\.emplace_back\(\"SDPA_READ_BARRIER_MULT\"", fac)
    # the pack defines are assigned before the env define is appended (an assignment after it would drop it)
    assert fac.index("reader_desc.defines = pack_defines;") < fac.index('emplace_back("SDPA_READ_BARRIER_MULT"')


def _run(mult):
    env = dict(os.environ)
    env.pop("TT_SDPA_DECODE_READ_MULT", None)
    if mult:
        env["TT_SDPA_DECODE_READ_MULT"] = str(mult)
    r = subprocess.run([sys.executable, "-c", _BENCH], env=env, capture_output=True, text=True, timeout=1200)
    assert r.returncode == 0, r.stderr[-3000:]
    return [l for l in r.stdout.splitlines() if l.startswith("MD5")]


@pytest.mark.skipif(os.environ.get("TT_VISIBLE_DEVICES") is None, reason="device test: set TT_VISIBLE_DEVICES=0")
def test_read_mult_bit_identical():
    off, on = _run(0), _run(4)
    assert len(off) == 2 and off == on, f"OFF {off} vs MULT=4 {on}"
