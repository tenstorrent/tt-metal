# 11 · Custom kernels with tt-lang (the kernel band)

When the TTNN op library can't express what an op needs — a fusion the stock op won't do, an
odd shape it handles poorly, a dataflow that wastes bandwidth — the remaining headroom lives
*below* the op API, in the kernel itself. **tt-lang (`ttl`)** is the Python kernel-authoring DSL
for that band: you write a tile-level kernel (compute + data movement) that compiles and runs on
the device, with no C++. This is the deepest lever — reach for it only after the TTNN-API levers
are spent.

### Toolchain (version-match with the repo's ttnn)
tt-lang executes kernels through `ttnn.generic_op`, so its package must coexist with the repo's
**source-built ttnn**, not replace it. The pip `tt-lang` wheel hard-pins a specific ttnn from
v1.0.2 on (`==0.69.0`, later `==0.72.0`), and installing those uninstalls a source ttnn. **Install
`tt-lang==1.0.1`** — the only release whose ttnn dependency is unpinned and optional
(`ttnn; extra == "device"`), so `pip install tt-lang==1.0.1` (without the `[device]` extra) rides on
the existing ttnn. Verified on-device: a `ttnn 0.65.1` build exposes the full ABI tt-lang needs
(`generic_op`, `ProgramDescriptor`, `SemaphoreDescriptor`, `CoreRangeSet`, `KernelDescriptor`,
`CBDescriptor`), and a fused `a*b+c` kernel JIT-built and ran on Blackhole with bit-accurate output.
The optimizer's availability gate (`route.py::_tt_lang_available`) only offers this lever once `ttl`
imports, so an env without it simply skips the lever.

## Write a custom tt-lang kernel for the hot op {#tt-lang-kernel}
<!-- route
op_class: matmul,attention,eltwise,reduction,conv_pool
rank: time
lever_type: structural
-->

**Fires when** the bucket's bottleneck is **kernel-level, not knob-level** — i.e. the route brief's
`regime_verdict` is `kernel`. That verdict is reached *directly from the roofline* (no need to
exhaust knobs first) when any of:
- the dominant op is **dispatch/launch-bound** (`bound_by == dispatch`) — many tiny ops; a knob
  cannot remove launch overhead, but fusing them into one kernel can;
- the dominant op is **at its single-op TTNN floor** yet the bucket is still slow — the remaining
  cost is *between* ops (DRAM round-trips), which only fusion removes;
- the **TTNN-API knobs for the bucket are spent** (fidelity walked, sharded, program-config tuned —
  all tried, no PCC-safe gain).

Conversely, when `regime_verdict` is `knob` (compute-bound with untried grid/fidelity, or
memory-bound with untried dtype/shard), take the cheaper knob first — don't reach for a kernel.

**Hard constraint — preserve the op's I/O contract.** The kernel must consume and produce the
**same dtype / layout / memory_config** the surrounding graph already passes (the emit-e2e
stitching and the next op depend on it). Optimize *inside* the kernel; do not change what it hands
downstream. The e2e PCC gate is the final authority — it's the only thing that sees the stitched
pipeline.

### The tt-lang model (what the kernel is made of)
- `@ttl.operation(grid=(R, C))` — declares the op and the core grid it runs on (tile coordinates
  = element count // 32).
- `@ttl.compute` — the compute kernel: tile-level math (matmul, eltwise, reduce) on 32×32 tiles.
- `@ttl.datamovement` — DM kernels: move tiles between DRAM and L1.
- `ttl.make_dataflow_buffer_like(t, shape=(m, n), block_count=2)` — an L1 dataflow buffer (DFB)
  shared between kernels; `block_count=2` double-buffers so DM fills one block while compute reads
  the other.
- `ttl.copy(dst, src)` / `tx.wait()` — start and await a transfer.
- `ttl.block.fill(shape, value)` — a block filled with a scalar (e.g. zero the accumulator before
  the k-reduction).

### PROVEN matmul template — ADAPT this, do NOT write from a blank page
This is the canonical tt-lang fused `relu(a@b + c)` kernel (k-reduction with an accumulator
ping-pong). It is CORRECT (PCC>0.99). Start from it and change only what the op needs (drop the
bias/relu if absent, match the dtype/layout contract). Writing a kernel from scratch is how you
get the naive 47×-slower or crashing kernels — adapt this instead.
```python
import ttnn, ttl
TILE = 32

@ttl.operation(grid=(1, 1))            # SPEED LEVER: see note below — (1,1) is correct but SLOW
def fused_matmul(a: ttnn.Tensor, b: ttnn.Tensor, c: ttnn.Tensor, y: ttnn.Tensor) -> None:
    m_tiles, n_tiles, k_tiles = a.shape[0] // TILE, b.shape[1] // TILE, a.shape[1] // TILE
    a_dfb   = ttl.make_dataflow_buffer_like(a, shape=(1, 1), block_count=2)
    b_dfb   = ttl.make_dataflow_buffer_like(b, shape=(1, 1), block_count=2)
    c_dfb   = ttl.make_dataflow_buffer_like(c, shape=(1, 1), block_count=2)
    acc_dfb = ttl.make_dataflow_buffer_like(y, shape=(1, 1), block_count=2)  # k-reduction accumulator
    y_dfb   = ttl.make_dataflow_buffer_like(y, shape=(1, 1), block_count=2)

    @ttl.datamovement()                # reader: stream c[m,n], then a[m,k]/b[k,n] for each k
    def read():
        for mt in range(m_tiles):
            for nt in range(n_tiles):
                with c_dfb.reserve() as c_blk:
                    ttl.copy(c[mt, nt], c_blk).wait()
                for kt in range(k_tiles):
                    with a_dfb.reserve() as a_blk, b_dfb.reserve() as b_blk:
                        ta = ttl.copy(a[mt, kt], a_blk); tb = ttl.copy(b[kt, nt], b_blk)
                        ta.wait(); tb.wait()

    @ttl.compute()                     # accumulate a@b over k (ping-pong acc), then +c, relu
    def compute():
        for _ in range(m_tiles):
            for _ in range(n_tiles):
                with acc_dfb.reserve() as acc_blk:
                    acc_blk.store(ttl.block.fill(0, shape=acc_blk.shape))   # zero the accumulator
                for _ in range(k_tiles):
                    with a_dfb.wait() as a_blk, b_dfb.wait() as b_blk, acc_dfb.wait() as pre:
                        with acc_dfb.reserve() as acc_blk:
                            acc_blk.store(pre + a_blk @ b_blk)              # partial product
                with c_dfb.wait() as c_blk, acc_dfb.wait() as acc_blk:
                    with y_dfb.reserve() as y_blk:
                        y_blk.store(ttl.math.relu(c_blk + acc_blk))        # drop relu/c if op has none

    @ttl.datamovement()                # writer: completed output tiles L1 -> DRAM
    def write():
        for mt in range(m_tiles):
            for nt in range(n_tiles):
                with y_dfb.wait() as y_blk:
                    ttl.copy(y_blk, y[mt, nt]).wait()
```

**SPEED LEVER — occupy the grid.** `grid=(1, 1)` runs the whole m×n×k loop on ONE Tensix core: correct
but SLOW (this is why a naive single-core kernel loses to the stock multi-core op — the 47× case).
For a real win, set `grid=(R, C)` to the device compute grid and distribute the output tiles across
cores (each core owns a slice of the m×n tiles). Step up the package tutorials
(`tt-lang-setup-tutorials` → `tutorials/matmul/` step_1 single-tile → step_2 multi-tile → step_3/4
grid/multi-device) to scale it. A kernel that isn't faster than the stock op is NOT a win — the
e2e measurement will reject it.

### Highest-value fusion — back-to-back matmuls (the FFN), intermediate stays in L1
A single matmul is usually NOT a kernel win: the stock TTNN matmul is already near its FLOP/bandwidth
floor (an adapted single-matmul kernel just matches it — NO-GAIN). The win is a fusion the op library
**cannot express**: two sequential linears with an activation between them —
`y = W2 @ act(W1 @ x + b1) + b2` (an FFN / MLP block). TTNN runs this as three ops and MUST
materialize the large `[m, hidden]` intermediate in DRAM between them (write it after linear-1, read
it back for linear-2). A fused kernel keeps that intermediate in **L1** and never round-trips it —
saving `2 × m × hidden × dtype_bytes` of DRAM traffic, which is the actual bottleneck when the
intermediate is wide (e.g. seq×4096). `ttnn.linear(activation=...)` only fuses the activation into
ONE matmul; it cannot fuse across the two matmuls — so this is genuinely kernel-only.

Structure (adapt the template above): one `@ttl.operation` taking `x, W1, b1, W2, b2, y`. For each
m-tile row: (1) compute `h = act(W1 @ x + b1)` into an L1 DFB (k-reduction over `in`); (2) immediately
compute `W2 @ h + b2` (k-reduction over `hidden`, reading `h` straight from that L1 DFB — never write
it to DRAM); (3) write only the final `[m, out]` tile to DRAM. Tile `m` so each row's `[1, hidden]`
intermediate fits L1; occupy the grid by distributing m-tiles across cores. Match the I/O dtype
contract (bf16 in/out, fp32 accumulate inside). Use the actual activation the FFN uses (silu/relu/gelu
via `ttl.math.*`) — check the op→source, do not assume relu.

### Recipe
1. From the op→source attribution, find the hot op's exact call site.
2. Author a `ttl` kernel that reproduces it (fuse adjacent ops where the gap is dispatch-bound;
   occupy the full grid where it's compute-bound; shard tiles into L1 where it's memory-bound).
3. Replace the call site so the model executes the kernel — keep the I/O contract identical.
4. The harness compiles + runs it under the e2e gate. A compile/lower failure routes to
   REPAIR_CODE with the compiler message (fix the kernel, don't delete it); a PCC drop routes to
   REPAIR_PCC; only a faster + PCC-clean result is kept.

A kept kernel **graduates** into a learned lever, so the recipe is reused on the next model with
the same op (it stops being a from-scratch authoring cost).
