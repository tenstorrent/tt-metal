# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""PERF EXPERIMENT I2 — "colvalid_payload" isolated bake-off.

IDEA. Every tile the combine collective moves over the NoC (level-1 gather,
level-2 gather, rstd multicast) is a `reduce<SUM, REDUCE_ROW>` result, i.e.
COLUMN-0-VALID: only column 0 of each of the 32 rows carries information, and
column 0 lives in FACES 0 and 2 of the tile. So the payload could be smaller
than the whole tile.

VARIANTS (writer-only; `RMSN_PAYLOAD_MODE` -D on the writer kernel):
    0  baseline == today's op: whole tile, T bytes,   1 transaction / tile
    1  faces 0+2 only:         T/2 bytes,             2 transactions / tile
    2  contiguous prefix 0..2: 3T/4 bytes,            1 transaction / tile
  modes 1/2 also trim the multicast by the last tile's trailing garbage face.

HONESTY. The BASELINE variant is the op's real writer run through this same
harness (mode 0 compiles to `noc_async_write(src, dst, stat_tile_bytes)` — the
production expression), so the only difference between variants is the payload.
The user's precision contract is fixed for every variant: bf16 in/out, TILE,
gamma bf16 TILE, fp32_dest_acc_en=False, MathFidelity.HiFi2 (the perf loose
cases' config). Nothing here touches a dtype or a compute-config field.

Each run reports BOTH the device kernel ns and the PCC against torch, from one
dispatch — correctness is the pass/fail, perf is only measured.

    TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_MID_RUN_DUMP=1 \
    TT_METAL_PROFILER_CPP_POST_PROCESS=1 scripts/tt-probe.sh rms_norm <<'EOF'
    import importlib.util
    p = ".../perf_experiments/colvalid_payload/bench.py"
    s = importlib.util.spec_from_file_location("cb", p); m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    m.sweep([(0, "bshard1024", 0), (1, "bshard1024", 0), (2, "bshard1024", 0)])
    EOF

RESULTS — blackhole_p150b, 1350 MHz, in-process device profiler ON (so every
number here carries the same zone overhead; compare only within this table).
DEVICE KERNEL DURATION [ns], median of the samples taken; PCC vs torch bf16.

  geometry                       G1  S2  R   blocks | mode0   mode1    mode2
  decode7168 (FOCUS, 110 cores)  11  10  1   1      |  8928    8953     8929   (n=4 each)
  decode1024 (110 cores)         11   2  1   1      |  6055    6133     6146   (n=3-4)
  wtail 32x2047                  11   5  1   1      |  6506    6313       -    (n=3)
  rm_interleaved 128x512 RM      11   1  1   1      |  9917   10035       -    (n=3)
  prefill7168 8192x7168           2   1  1   5      | 590566 591001       -    (n=1)
  hshard512 (no collective, G=1)  1   1  1   1      |  5396    5405       -    (n=1)
  wshard1024 [32,128] on (8,1)    8   1  1   1      |  4564    4421     4500   (n=4-5)
  wshard7168 [32,256] on (7,4)    7   4  1   1      |  5844    5698     5767   (n=4)
  bshard1024 [1024,128] on (8,8)  8   1  16  2      | 50453   47935    49330   (n=2)
  decode7168 @ fp32_dest_acc_en=True (stat tile 4096 B) |  9344    9192      -  (n=1)

  PCC is BIT-IDENTICAL across modes on every geometry (0.99994 - 0.99999), and
  UNCHANGED with RMSN_POISON_ODD_FACES=1 (bf16 NaN in face 1, -Inf in face 3 of
  every slot a short payload never writes) — so no zero-fill is needed. Priced
  anyway: the one-time fill costs +20.3 us on decode7168, +9.0 us on wshard1024,
  +139 us on bshard1024, i.e. an order of magnitude more than the win.

  ZONE ATTRIBUTION, bshard1024 (per-core sums over 2 blocks, critical-path core):
    wr_gather_issue    6291 -> 3839 (mode1) / 4799 (mode2)
    wr_gather_sem_wait 5367 -> 2655           / 4186
    cp_gather_wait     7270 -> 4684           / 5776
  Mode 1 HALVES the bytes and DOUBLES the transactions, and still beats mode 2 —
  the gather leg is destination-L1/NoC-BYTE bound here, not RISC-issue bound
  (128 stat tiles land on one leader per block: 256 KB -> 128 KB).
"""

from __future__ import annotations

import os
import shutil
import time

import torch
import ttnn

from ttnn.operations.rms_norm.rms_norm_program_descriptor import create_program_descriptor

_ML = ttnn.TensorMemoryLayout

# name -> (rows, hidden, memory_layout, shard_shape, core_grid). Same table as
# tests/.../rms_norm/perf_zone_harness.py (the op's own pinned perf geometries).
CASES = {
    # THE FOCUS SHAPE (perf-flagged loose case).
    "decode7168": (32, 7168, _ML.INTERLEAVED, None, None),
    "decode1024": (32, 1024, _ML.INTERLEAVED, None, None),
    "prefill7168": (8192, 7168, _ML.INTERLEAVED, None, None),
    # The pinned sharded perf geometries.
    "bshard1024": (8192, 1024, _ML.BLOCK_SHARDED, [1024, 128], (8, 8)),
    "wshard7168": (32, 7168, _ML.WIDTH_SHARDED, [32, 256], (7, 4)),
    "wshard1024": (32, 1024, _ML.WIDTH_SHARDED, [32, 128], (8, 1)),
    # Guard-set representatives: distinct kernel legs.
    "hshard512": (256, 512, _ML.HEIGHT_SHARDED, [32, 512], (1, 8)),  # w_group_size == 1: no collective
    "wtail": (32, 2047, _ML.INTERLEAVED, None, None),  # ragged hidden tile
    "rm_interleaved": (128, 512, _ML.INTERLEAVED, None, None, True),  # ROW_MAJOR in/out legs
}

WRITER_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels", "writer_colvalid.cpp")


def _tensors(device, rows, hidden, memory_layout, shard_shape, core_grid, row_major=False):
    torch.manual_seed(0)
    shape = (1, 1, rows, hidden)
    layout = ttnn.ROW_MAJOR_LAYOUT if row_major else ttnn.TILE_LAYOUT
    torch_input = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
    torch_gamma = torch.randn((1, 1, 1, hidden), dtype=torch.float32).to(torch.bfloat16)
    memory_config = None
    if memory_layout != _ML.INTERLEAVED:
        from eval.sharding import shard_config

        memory_config = shard_config(
            shard_shape, core_grid, memory_layout, layout=layout, dtype=ttnn.bfloat16, device=device
        )
    kw = {} if memory_config is None else {"memory_config": memory_config}
    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=layout, device=device, **kw)
    tt_gamma = ttnn.from_torch(torch_gamma, dtype=ttnn.bfloat16, layout=layout, device=device)
    return torch_input, torch_gamma, tt_input, tt_gamma


def _torch_ref(x, gamma, eps):
    xf = x.to(torch.float32)
    rstd = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (xf * rstd * gamma.to(torch.float32)).to(torch.bfloat16)


def _pcc(a, b):
    a = a.to(torch.float32).flatten()
    b = b.to(torch.float32).flatten()
    if torch.equal(a, b):
        return 1.0
    if not (torch.isfinite(a).all() and torch.isfinite(b).all()):
        return float("nan")
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def _writer_facts(pd):
    """(W_GROUP_SIZE, STAGE2_SPAN, num_blocks, block_row_tiles) off the real descriptor."""
    k = pd.kernels[1]
    ct = list(k.compile_time_args)
    w_group, stage2 = ct[3], ct[7]
    nb = brt = None
    try:
        rt = k.runtime_args
        # Find the first core with a non-empty arg vector (num_blocks != 0).
        for xi in range(16):
            for yi in range(16):
                try:
                    v = list(rt[xi][yi])
                except Exception:
                    continue
                if len(v) > 4 and v[2] != 0:
                    nb, brt = v[2], v[3]
                    raise StopIteration
    except StopIteration:
        pass
    except Exception:
        pass
    return w_group, stage2, nb, brt


def _run_once(device, name, mode, poison, epsilon=1e-6, fp32_dest=False):
    spec = CASES[name]
    rows, hidden, memory_layout, shard_shape, core_grid = spec[:5]
    row_major = len(spec) > 5 and spec[5]
    torch_input, torch_gamma, tt_input, tt_gamma = _tensors(
        device, rows, hidden, memory_layout, shard_shape, core_grid, row_major=row_major
    )
    # The user's precision contract, IDENTICAL for every variant. `fp32_dest` is a
    # DOMAIN dimension (does the pattern still hold when the stat tile is fp32, i.e.
    # 4096 B / 1024 B faces?), never a lever: it is compared mode-0 vs mode-1 at the
    # same setting.
    cfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=fp32_dest)
    out_mc = tt_input.memory_config()

    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(tt_input.shape)), tt_input.dtype, tt_input.layout, device, out_mc
    )
    pd = create_program_descriptor(tt_input, tt_gamma, out, epsilon=epsilon, compute_kernel_config=cfg)

    # ---- the ONLY difference between variants: the writer's payload mode ----
    ks = list(pd.kernels)
    assert str(ks[1].kernel_source).endswith("rms_norm_writer.cpp"), ks[1].kernel_source
    ks[1].kernel_source = WRITER_SRC
    ks[1].defines = [("RMSN_PAYLOAD_MODE", str(mode)), ("RMSN_POISON_ODD_FACES", str(poison))]
    pd.kernels = ks
    facts = _writer_facts(pd)

    ttnn.ReadDeviceProfiler(device)  # flush the prep programs (from_torch / to_sharded)
    res = ttnn.generic_op([tt_input, tt_gamma, out], pd)
    ttnn.synchronize_device(device)
    ttnn.ReadDeviceProfiler(device)

    ns = None
    for programs in (ttnn.get_latest_programs_perf_data() or {}).values():
        for program in programs:
            entry = (getattr(program, "program_analyses_results", None) or {}).get("DEVICE KERNEL DURATION [ns]")
            if entry is not None:
                d = float(entry.duration)
                ns = d if ns is None else max(ns, d)

    got = ttnn.to_torch(res)
    pcc = _pcc(got, _torch_ref(torch_input, torch_gamma, epsilon))
    return ns, pcc, facts


def main(mode=0, names=("decode7168",), poison=0, keep_log=True):
    sweep([(mode, n, poison) for n in names], keep_log=keep_log)


def sweep(triples, keep_log=True):
    """triples = [(mode, case_name, poison[, fp32_dest]), ...]; one dispatch each."""
    device = ttnn.open_device(device_id=0)
    logdir = os.path.join(os.environ.get("TT_METAL_HOME", "."), "generated", "profiler", ".logs")
    try:
        for t in triples:
            mode, name, poison = t[0], t[1], t[2]
            fp32_dest = bool(t[3]) if len(t) > 3 else False
            tag = f"m{mode}" + (f"p{poison}" if poison else "") + ("f32" if fp32_dest else "")
            ns, pcc, facts = _run_once(device, name, mode, poison, fp32_dest=fp32_dest)
            w_group, stage2, nb, brt = facts
            print(
                f"COLVALID mode={mode} poison={poison} fp32d={int(fp32_dest)} {name}: ns={ns} pcc={pcc:.6f} "
                f"G1={w_group} S2={stage2} blocks={nb} R={brt}"
            )
            if keep_log:
                src = os.path.join(logdir, "profile_log_device.csv")
                if os.path.exists(src):
                    dst = os.path.join(logdir, f"colvalid_{tag}_{name}.csv")
                    shutil.copyfile(src, dst)
                    print(f"COLVALID {name}: zones -> {dst}")
            time.sleep(0.2)
    finally:
        ttnn.close_device(device)
