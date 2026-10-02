# Source: umalesTT's matmul sweep for #57884 (https://gist.github.com/umalesTT/cdf9d3cbf9ce4c2d9a6d7dc638566645), copied below this line; only reformatted by black.
"""1D-mcast ttnn.matmul sweep over Llama-3.3-70B (TP8) and Llama-3.2-1B shapes (8B opt-in).

Shapes: every projection (q, k/v, fused qkv, o, gate/up, fused gate_up, down,
lm_head) x training pass (fwd, dgrad; wgrad opt-in) x tokens per step,
deduplicated.

For every shape that tt-metal routes to the 1D multicast kernel
(MatmulMultiCoreReuseMultiCast1DProgramConfig: mcast_in0 for wide shapes,
mcast_in1 for tall ones) this compares

  metal default  ttnn.matmul without a program config (tt-metal chooses)
  oracle         fastest legal (out_block_h, out_block_w, in0_block_w) found by
                 timing every candidate that fits L1
  our strategy   tt-mlir TTNNSetMatmulProgramConfig 1D rule (mcast_in0; mcast_in1
                 mirrors it): out_block_w = per_core_N, out_block_h = per_core_M
                 or per_core_M / 2 when M >= --split-min-dim, largest in0_block_w
                 that fits, tie -> larger out_block_h

and prints markdown tables to the log file (and stdout).

All operands are bf16, tiled, DRAM interleaved; compute config HiFi4 with
fp32_dest_acc_en = true (what tt-mlir sets by default). Forward shapes use
transpose_b (torch Linear weight [N, K]); dgrad does not. wgrad reads dY stored
[T, N] with transpose_a.

Usage:
  source env/activate
  python matmul_sweep_1d.py                              # 70B + 1B, fwd + dgrad
  python matmul_sweep_1d.py --models 70B 8B 1B --passes fwd dgrad wgrad --tokens 1024
  python matmul_sweep_1d.py --list                       # print shapes, no device
"""

import argparse
import math
import sys
import time
from dataclasses import dataclass

import torch

KERNELS = ("1d_in0", "1d_in1")
DEFAULT_LOG = "matmul_sweep_1d.log"

TILE = 32
BF16_TILE_BYTES = 2048
FP32_TILE_BYTES = 4096
NARROW_SHAPE_RATIO_THRESHOLD = 8
MAX_SUBBLOCK_AREA = 4  # fp32 dest accumulation halves the 8-tile dest register
# tt-metal's SUBBLOCK_HW_CHOICES as (out_subblock_h, out_subblock_w).
SUBBLOCK_HW_CHOICES = [
    (4, 2),
    (2, 4),
    (8, 1),
    (1, 8),
    (7, 1),
    (1, 7),
    (3, 2),
    (2, 3),
    (6, 1),
    (1, 6),
    (5, 1),
    (1, 5),
    (2, 2),
    (4, 1),
    (1, 4),
    (3, 1),
    (1, 3),
    (2, 1),
    (1, 2),
    (1, 1),
]

# Per-device projection sizes. 70B runs 1x8 tensor parallel: column-parallel
# q/kv/gate/up/lm_head split N, row-parallel o/down split K. 8B and 1B run on
# one device.
MODELS = {
    "70B": dict(hidden=8192, q=1024, kv=128, ffn=3584, vocab=16032),  # Llama-3.3-70B, TP8
    "8B": dict(hidden=4096, q=4096, kv=1024, ffn=14336, vocab=128256),  # Llama-3.1-8B
    "1B": dict(hidden=2048, q=2048, kv=512, ffn=8192, vocab=128256),  # Llama-3.2-1B
}
DEFAULT_MODELS = ("70B", "1B")
# Tokens per step (batch * sequence) = M of the forward matmuls.
TOKENS = (128, 256, 512, 1024, 2048, 4096)
# (name, K, N) of the forward matmul, as functions of the model dims.
PROJECTIONS = [
    ("q_proj", lambda m: (m["hidden"], m["q"])),
    ("k_proj/v_proj", lambda m: (m["hidden"], m["kv"])),
    ("qkv_proj", lambda m: (m["hidden"], m["q"] + 2 * m["kv"])),
    ("o_proj", lambda m: (m["q"], m["hidden"])),
    ("gate_proj/up_proj", lambda m: (m["hidden"], m["ffn"])),
    ("gate_up_proj", lambda m: (m["hidden"], 2 * m["ffn"])),
    ("down_proj", lambda m: (m["ffn"], m["hidden"])),
    ("lm_head", lambda m: (m["hidden"], m["vocab"])),
]
# Training matmuls of Y[T, N] = X[T, K] @ W^T with W stored [N, K]:
#   fwd    Y  = X  @ W^T   T x K x N, transpose_b
#   dgrad  dX = dY @ W     T x N x K
#   wgrad  dW = dY^T @ X   N x T x K, transpose_a (dY stored [T, N])
PASSES = {
    "fwd": lambda T, K, N: (T, K, N, True, False),
    "dgrad": lambda T, K, N: (T, N, K, False, False),
    "wgrad": lambda T, K, N: (N, T, K, False, True),
}
DEFAULT_PASSES = ("fwd", "dgrad")


@dataclass(frozen=True)
class Shape:
    name: str
    M: int
    K: int
    N: int
    transpose_b: bool
    transpose_a: bool

    Mt = property(lambda s: math.ceil(s.M / TILE))
    Kt = property(lambda s: math.ceil(s.K / TILE))
    Nt = property(lambda s: math.ceil(s.N / TILE))
    flops = property(lambda s: 2 * s.M * s.K * s.N)

    @property
    def operands(self):
        return (
            f"[{self.M}x{self.K}, {self.K}x{self.N}]"
            + (" A^T" if self.transpose_a else "")
            + (" B^T" if self.transpose_b else "")
        )


@dataclass(frozen=True)
class Config:
    mcast_in0: bool
    per_core_M: int
    per_core_N: int
    out_block_h: int
    out_block_w: int
    in0_block_w: int
    subblock_h: int
    subblock_w: int

    def __str__(self):
        return (
            f"blk {self.out_block_h}x{self.out_block_w} bw {self.in0_block_w} "
            f"sb {self.subblock_h}x{self.subblock_w}"
        )


def llama_shapes(models, tokens, passes):
    """Unique (M, K, N, transpose_b, transpose_a) over models x tokens x projections x passes, named by every use."""
    names = {}
    for key in models:
        for T in tokens:
            for proj, dims in PROJECTIONS:
                for kind in passes:
                    names.setdefault(PASSES[kind](T, *dims(MODELS[key])), []).append(f"{key} {proj} {kind} T={T}")
    return [Shape(", ".join(v), *mknt) for mknt, v in names.items()]


def route(shape):
    """create_simple_matmul_program_config routing for rank-2, all-DRAM-interleaved operands."""
    height, width = shape.Mt * TILE, shape.Nt * TILE
    ratio = max(height, width) // min(height, width)
    if ratio > NARROW_SHAPE_RATIO_THRESHOLD or height <= TILE or width <= TILE:
        return "1d_in0" if width > height else "1d_in1"
    return "2d"


def per_core_shape(shape, grid_x, grid_y):
    num_cores = grid_x * grid_y
    if route(shape) == "1d_in0":
        return shape.Mt, math.ceil(shape.Nt / num_cores)
    return math.ceil(shape.Mt / num_cores), shape.Nt


def divisors(n, upper=None):
    upper = n if upper is None else min(n, upper)
    return [d for d in range(1, upper + 1) if n % d == 0]


def cb_bytes(shape, h, w, bw):
    """L1 circular buffers of the mcast factory, same model as the tt-mlir pass."""
    depth = 2 if shape.Kt // bw > 1 else 1
    in0_copies = 2 if shape.transpose_a else 1  # transpose_a adds an in0-sized CB for the transposed tiles
    return depth * (in0_copies * h + w) * bw * BF16_TILE_BYTES + h * w * (BF16_TILE_BYTES + FP32_TILE_BYTES)


def subblock(h, w):
    for sh, sw in SUBBLOCK_HW_CHOICES:
        if sh * sw <= MAX_SUBBLOCK_AREA and h % sh == 0 and w % sw == 0:
            return sh, sw
    return 1, 1


def candidates(shape, grid_x, grid_y, l1_budget, max_in0_block_w):
    """Every (h, w, bw) the pass could emit: bw | Kt with >= 2 K-blocks, h | per_core_M, w | per_core_N."""
    mcast_in0 = route(shape) == "1d_in0"
    pM, pN = per_core_shape(shape, grid_x, grid_y)
    out = []
    for bw in divisors(shape.Kt, max_in0_block_w):
        if shape.Kt > 1 and shape.Kt // bw < 2:
            continue
        for h in divisors(pM):
            for w in divisors(pN):
                if cb_bytes(shape, h, w, bw) <= l1_budget:
                    out.append(Config(mcast_in0, pM, pN, h, w, bw, *subblock(h, w)))
    return out


def our_pick(shape, configs, split_min_dim):
    """TTNNSetMatmulProgramConfig pick1D."""
    best, best_key = None, None
    for c in configs:
        if c.mcast_in0:
            pinned, pinned_full, free, full, dim = c.out_block_w, c.per_core_N, c.out_block_h, c.per_core_M, shape.M
        else:
            pinned, pinned_full, free, full, dim = c.out_block_h, c.per_core_M, c.out_block_w, c.per_core_N, shape.N
        allowed = {full, full // 2} if dim >= split_min_dim and full % 2 == 0 else {full}
        if pinned != pinned_full or free not in allowed:
            continue
        key = (c.in0_block_w, free)
        if best_key is None or key > best_key:
            best, best_key = c, key
    return best


def to_program_config(ttnn, c, grid_x, grid_y):
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(grid_x, grid_y),
        in0_block_w=c.in0_block_w,
        out_subblock_h=c.subblock_h,
        out_subblock_w=c.subblock_w,
        out_block_h=c.out_block_h,
        out_block_w=c.out_block_w,
        per_core_M=c.per_core_M,
        per_core_N=c.per_core_N,
        fuse_batch=False,
        fused_activation=None,
        mcast_in0=c.mcast_in0,
    )


class Log:
    def __init__(self, path):
        self.file = open(path, "w")

    def __call__(self, text=""):
        print(text, flush=True)
        self.file.write(text + "\n")
        self.file.flush()


class Runner:
    def __init__(self, ttnn, device, grid, args):
        self.ttnn, self.device, self.grid, self.args = ttnn, device, grid, args
        self.compute_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False
        )

    def load(self, shape):
        ttnn = self.ttnn
        torch.manual_seed(0)
        a = torch.randn(*((shape.K, shape.M) if shape.transpose_a else (shape.M, shape.K)), dtype=torch.bfloat16)
        b = torch.randn(*((shape.N, shape.K) if shape.transpose_b else (shape.K, shape.N)), dtype=torch.bfloat16)
        b = b / math.sqrt(shape.K)
        put = lambda t: ttnn.from_torch(
            t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        self.a, self.b, self.shape = put(a), put(b), shape

    def unload(self):
        self.ttnn.deallocate(self.a)
        self.ttnn.deallocate(self.b)

    def run(self, config):
        program_config = None if config is None else to_program_config(self.ttnn, config, *self.grid)
        return self.ttnn.matmul(
            self.a,
            self.b,
            transpose_a=self.shape.transpose_a,
            transpose_b=self.shape.transpose_b,
            program_config=program_config,
            compute_kernel_config=self.compute_config,
            memory_config=self.ttnn.DRAM_MEMORY_CONFIG,
            dtype=self.ttnn.bfloat16,
        )

    def time_ms(self, config, iters):
        """Mean device-synchronised wall time per op; None when tt-metal rejects the config."""
        try:
            for _ in range(self.args.warmup):
                self.ttnn.deallocate(self.run(config))
            self.ttnn.synchronize_device(self.device)
            start = time.perf_counter()
            for _ in range(iters):
                self.ttnn.deallocate(self.run(config))
            self.ttnn.synchronize_device(self.device)
        except Exception:
            return None
        return (time.perf_counter() - start) * 1e3 / iters

    def output(self, config):
        out = self.run(config)
        host = self.ttnn.to_torch(out).float()
        self.ttnn.deallocate(out)
        return host


def pcc(x, y):
    return torch.corrcoef(torch.stack([x.flatten(), y.flatten()]))[0, 1].item()


def sweep_shape(runner, shape, args, log):
    configs = candidates(shape, *runner.grid, args.l1_budget, args.max_in0_block_w)
    ours = our_pick(shape, configs, args.split_min_dim)
    kernel = route(shape)
    pM, pN = per_core_shape(shape, *runner.grid)
    log(
        f"=== {shape.name}  {shape.operands}  {kernel}  per_core={pM}x{pN}  Kt={shape.Kt}  "
        f"{len(configs)} candidates"
    )
    if ours is None:
        log("    skipped: no candidate fits L1")
        return None
    runner.load(shape)
    try:
        # Search pass: short timing of every candidate.
        searched = []
        for c in configs:
            t = runner.time_ms(c, args.search_iters)
            if t is not None:
                searched.append((t, c))
        searched.sort(key=lambda tc: tc[0])
        # Final pass: re-time the search leaders, ours and the default with more iterations.
        finalists = list(dict.fromkeys([c for _, c in searched[: args.retime_top]] + [ours]))
        final = {c: runner.time_ms(c, args.iters) for c in finalists}
        final = {c: t for c, t in final.items() if t is not None}
        metal_ms = runner.time_ms(None, args.iters)
        if not final or metal_ms is None:
            log("    skipped: tt-metal rejected every candidate or the default")
            return None
        oracle = min(final, key=final.get)
        ref = runner.output(None)
        ours_pcc = pcc(runner.output(ours), ref) if ours in final else float("nan")
    finally:
        runner.unload()

    row = dict(
        shape=shape,
        kernel=kernel,
        per_core=f"{pM}x{pN}",
        n=len(searched),
        metal=metal_ms,
        oracle=final[oracle],
        oracle_cfg=oracle,
        ours=final.get(ours),
        ours_cfg=ours,
        pcc=ours_pcc,
    )
    tflops = lambda ms: shape.flops / (ms * 1e-3) / 1e12
    log(f"    metal default {metal_ms:8.3f} ms  {tflops(metal_ms):6.1f} TFLOP/s")
    log(f"    oracle        {final[oracle]:8.3f} ms  {tflops(final[oracle]):6.1f} TFLOP/s  {oracle}")
    if row["ours"] is not None:
        log(f"    our strategy  {row['ours']:8.3f} ms  {tflops(row['ours']):6.1f} TFLOP/s  {ours}  pcc {ours_pcc:.5f}")
    else:
        log(f"    our strategy  rejected by tt-metal  {ours}")
    return row


def print_tables(rows, grid, log):
    speedup = lambda r, k: r["metal"] / r[k] if r[k] else float("nan")
    geo = lambda vals: math.exp(sum(math.log(v) for v in vals) / len(vals)) if vals else float("nan")
    log()
    log(f"## 1D mcast matmul, Blackhole {grid[0]}x{grid[1]}, bf16, HiFi4, fp32 dest acc")
    log(
        "Time per op; parentheses = speedup over metal default. A^T = A stored [K, M] (transpose_a), "
        "B^T = B stored [N, K] (transpose_b)."
    )
    log()
    log("| [MxK, KxN] | metal default | oracle | our strategy |")
    log("|---|---:|---:|---:|")
    for r in rows:
        ours = f"{r['ours']:.3f} ms ({speedup(r, 'ours'):.2f}x)" if r["ours"] else "rejected"
        log(
            f"| {r['shape'].operands} | {r['metal']:.3f} ms | {r['oracle']:.3f} ms ({speedup(r, 'oracle'):.2f}x) | {ours} |"
        )
    ok = [r for r in rows if r["ours"]]
    log(
        f"| **geomean speedup** | 1.00x | {geo([speedup(r, 'oracle') for r in rows]):.2f}x | "
        f"{geo([speedup(r, 'ours') for r in ok]):.2f}x |"
    )
    log()
    log("| op | kernel | per_core | oracle config | our config | ours / oracle | pcc vs default |")
    log("|---|---|---|---|---|---:|---:|")
    for r in rows:
        frac = f"{r['oracle'] / r['ours']:.3f}" if r["ours"] else "-"
        log(
            f"| {r['shape'].name} | {r['kernel']} | {r['per_core']} | {r['oracle_cfg']} | {r['ours_cfg']} | "
            f"{frac} | {r['pcc']:.5f} |"
        )
    log(f"| **geomean** | | | | | {geo([r['oracle'] / r['ours'] for r in ok]):.3f} | |")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="*", choices=list(MODELS), default=list(DEFAULT_MODELS))
    parser.add_argument("--tokens", nargs="*", type=int, default=list(TOKENS), help="tokens per step (batch * seq)")
    parser.add_argument("--passes", nargs="*", choices=list(PASSES), default=list(DEFAULT_PASSES))
    parser.add_argument("--only", nargs="*", help="keep shapes whose name contains one of these")
    parser.add_argument("--grid", help="XxY compute grid; default = device compute_with_storage_grid_size")
    parser.add_argument("--l1-budget", type=int, default=1_400_000, help="CB bytes allowed (pass default)")
    parser.add_argument("--max-in0-block-w", type=int, default=32, help="pass default")
    parser.add_argument("--split-min-dim", type=int, default=1024, help="pass default")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--search-iters", type=int, default=5, help="iterations per candidate in the search pass")
    parser.add_argument("--retime-top", type=int, default=3, help="search leaders re-timed for the oracle")
    parser.add_argument("--iters", type=int, default=20, help="iterations for reported times")
    parser.add_argument("--log", default=DEFAULT_LOG)
    parser.add_argument("--list", action="store_true", help="print the shapes and exit (no device)")
    parser.add_argument("--device-id", type=int, default=0)
    args = parser.parse_args()

    shapes = [s for s in llama_shapes(args.models, args.tokens, args.passes) if route(s) in KERNELS]
    if args.only:
        shapes = [s for s in shapes if any(o in s.name for o in args.only)]
    if args.list:
        grid = tuple(int(v) for v in (args.grid or "12x10").lower().split("x"))
        for s in shapes:
            n = len(candidates(s, *grid, args.l1_budget, args.max_in0_block_w))
            pc = "x".join(map(str, per_core_shape(s, *grid)))
            print(f"{s.operands:34s} {route(s):6s} per_core={pc:7s} cands={n:4d}  {s.name}")
        print(f"{len(shapes)} shapes")
        return

    import ttnn

    device = ttnn.open_device(device_id=args.device_id)
    device_grid = device.compute_with_storage_grid_size()
    grid = tuple(int(v) for v in args.grid.lower().split("x")) if args.grid else (device_grid.x, device_grid.y)
    if grid != (device_grid.x, device_grid.y):
        ttnn.close_device(device)
        sys.exit(
            f"--grid {grid} must equal the device grid {device_grid.x}x{device_grid.y}: "
            "metal default always uses the full device grid"
        )
    device.enable_program_cache()
    log = Log(args.log)
    log(
        f"1D sweep: {len(shapes)} shapes, grid {grid[0]}x{grid[1]}, models {' '.join(args.models)}, "
        f"tokens {' '.join(map(str, args.tokens))}, passes {' '.join(args.passes)}"
    )
    runner = Runner(ttnn, device, grid, args)
    rows = []
    try:
        for s in shapes:
            row = sweep_shape(runner, s, args, log)
            if row is not None:
                rows.append(row)
    finally:
        ttnn.close_device(device)
        print_tables(rows, grid, log)


if __name__ == "__main__":
    main()
