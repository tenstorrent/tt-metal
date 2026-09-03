# mhc_post dtype support — handoff (NOT FOR MERGE)

Local experiment branch. Not pushed, no PR. Delete this file before any of it ships.

## Provenance

```
nmilicevic/ds-mhc-explore      PR #49039, the mHC parametrization kernel. Do not touch.
  └─ nmilicevic/ds-mhc-fused-post   the fused mhc_post kernel + perf test + HANDOFF.md
       └─ nmilicevic/ds-mhc-dtype   this branch: bf16 / bfloat8_b support for that kernel
```

This branch is **stacked, not independent**: it edits files that only exist on
`ds-mhc-fused-post`. It cannot be rebased onto main on its own.

## What changed

`mhc_post` was fp32-only. It now accepts FLOAT32, BFLOAT16 or BFLOAT8_B, with the
constraint that **all five inputs share one dtype**. The output takes the residual's dtype.

The uniform-dtype rule is not conservatism, it is forced by the reader. `reader_mhc_post.cpp`
derives a single page size from `CB_IN` and uses it for every NoC read — consts, post/comb,
y and residual alike:

```cpp
const uint32_t page = get_local_cb_interface(cb_in).fifo_page_size;
```

A narrower coefficient tensor would therefore be read at the wrong stride into `CB_PC` and
`CB_CONSTS`. Mixed precision needs per-CB page sizes there first.

Compute config is unchanged: `MathFidelity::HiFi4`, `fp32_dest_acc_en = true`. DEST stays
fp32 in every arm, so the five-term mix still accumulates at full width regardless of how
the operands are stored.

## Measured — T=640, C=7168, n=4, single Blackhole p150b, wall-clock, 10 iters

Traffic is `(1 + 2n) * T * C * bytes` = y in, n residual streams in, n streams out.

| dtype | ms/call | traffic | GB/s | vs fp32 | % of the machine's 445 GB/s |
|---|---|---|---|---|---|
| fp32 | 0.463 | 165.2 MB | 357 | 1.00x | 80% |
| bfloat16 | 0.236 | 82.6 MB | 349 | 1.96x | 79% |
| bfloat8_b | 0.147 | 43.9 MB | 299 | 3.15x | 67% |

fp32 and bf16 both hold ~350 GB/s, so the op is purely DRAM-bound and time tracks bytes
exactly. bfp8 falls off that line — at 43.9 MB the fixed unpack and dispatch cost stops
being negligible, which is why it returns 3.15x and not the 3.76x the byte count implies.

For scale: the composite ttnn `hc_post` chain this kernel replaces runs 3.764 ms in fp32,
so bf16 fused is 15.9x that and bfp8 fused is 25.6x.

## Precision — relative RMS against an fp64 reference

| arm | rel RMS | max abs err | pcc |
|---|---|---|---|
| device fp32 | 1.14e-03 | 1.41e-02 | 0.99999989 |
| device bfloat16 | 2.85e-03 | 4.68e-02 | 0.99999594 |
| device bfloat8_b | 1.18e-02 | 1.19e-01 | 0.99993381 |
| torch all-bf16, no device | 2.32e-03 | 4.02e-02 | 0.99999732 |

bf16 is 2.5x the error of fp32; bfp8 is 10x. Neither is free — read the next section
before deciding that 2.85e-03 is cheap.

## The non-obvious finding: fp32 storage does not buy fp32 arithmetic

The fp32 arm sits at 1.14e-03 relative RMS. That is roughly 2^-10, not the ~1e-07 an fp32
multiply-accumulate would give. Attribution:

- With power-of-two coefficients, so the broadcast matmul is bit-exact, the fp32 arm still
  shows **4.15e-04**. The loss is therefore in the `mul_tiles` operands, not in the
  coefficient broadcast.
- A torch simulation that truncates every operand to 10 explicit mantissa bits lands at
  5.78e-04 on the same inputs. The device sits at 1.14e-03 — about one extra truncation's
  worth, which is the coefficients passing through `matmul_tiles` and then through
  `mul_tiles`.
- A 23-bit-mantissa torch simulation is 1.03e-07 from the reference and **1.14e-03 from the
  device output**. The device is not doing fp32 arithmetic at any point.

So despite HiFi4 and fp32 DEST, srcA/srcB deliver ~10-11 mantissa bits. DEST accumulation
is genuinely fp32 — that is why summing five terms adds no further error — but the operands
are TF32-class going in.

The consequence for this op: fp32 pays 2x the bandwidth of bf16 to buy 2.5x the accuracy,
and that 2.5x comes entirely from operand *storage* width, not from any fp32 compute that
is actually happening. **bf16 is the right default here.**

Not yet checked: whether MathFidelity actually moves this. HiFi4's multi-pass reconstruction
is a matmul behaviour; whether the eltwise binary FPU path honours it is untested. An
A/B of LoFi against HiFi4 on the fp32 arm would settle it and needs only a rebuild — if
LoFi measures the same error, HiFi4 is buying nothing and can be dropped for whatever
throughput it costs.

## Two things that block turning this into a plan

1. **Mixed precision is worth less on device than in torch.** In torch, bf16 bulk alone
   (1.66e-03) and bf16 coefficients alone (1.61e-03) contribute about equally, which
   suggests fp32 coefficients plus bf16 bulk for free — post and comb are ~0.02% of traffic.
   But fp32 coefficients get truncated to ~11 bits in the FPU anyway, so the real gain is
   smaller than torch predicts. It also needs the per-CB page sizes described above.

2. **A bf16 mHC block is not a bf16 `mhc_post`.** `mhc_split_sinkhorn` is fp32-only, so the
   pre half would have to emit bf16 for this to pay end to end. If the surrounding sublayer
   runs fp32 activations, converting before `mhc_post` costs a full 165 MB read plus write,
   ~0.75 ms, which more than erases the 0.23 ms saved. This only pays where the sublayer is
   already bf16.

## Reproducing

Environment, from the worktree root:

```bash
cd /localdev/nmilicevic/tt-metal-ds-mhc-explore
source python_env/bin/activate
export TT_METAL_HOME=/localdev/nmilicevic/tt-metal-ds-mhc-explore
export PYTHONPATH=$TT_METAL_HOME
```

`/localdev/nmilicevic/tt-metal` is a different worktree with another session live in it.
Nothing here may read, write, build or run git against that path. The two share a `.git`
object store, so no bare `git stash` and no checkout of branches owned by that worktree.

Correctness suite (fp32 only; it does not cover the new dtypes yet) — 39 passed, 1 skipped
on this commit. Name the files rather than filtering the whole `tests/` tree with `-k mhc`:
collecting that tree trips a pre-existing error in `tests/sparse_mla/conftest.py`, which
declares `pytest_plugins` in a non-top-level conftest.

```bash
pytest models/demos/deepseek_v3_d_p/tests/pcc/test_mhc.py \
       models/demos/deepseek_v3_d_p/tests/pcc/test_mhc_post_op.py \
       models/demos/deepseek_v3_d_p/tests/pcc/test_mhc_reference_vs_canonical.py
```

The numbers in this document come from the script below. It needs a rebuild first
(`bash build_metal.sh`) because the dtype changes are host-side, not kernel-side.

```python
"""mhc_post across dtypes: throughput and error against an fp64 reference."""
import time, torch, ttnn
from models.demos.deepseek_v3_d_p.tt.mhc.tt_mhc import build_bcast_consts

T, C, n, ITERS = 640, 7168, 4, 10
g = torch.Generator().manual_seed(1)
y = torch.randn(1, 1, T, C, generator=g)
res = torch.randn(1, 1, T, n * C, generator=g)
post = 2.0 * torch.sigmoid(torch.randn(1, 1, T, n, generator=g))
comb = torch.rand(1, 1, T, n * n, generator=g)
comb = comb / comb.reshape(1, 1, T, n, n).sum(-1, keepdim=True).reshape(1, 1, T, n).repeat_interleave(n, -1)

def mix(y_, res_, post_, comb_):
    return torch.cat([sum([comb_[..., i*n+j:i*n+j+1] * res_[..., i*C:(i+1)*C] for i in range(n)],
                          post_[..., j:j+1] * y_) for j in range(n)], dim=-1).flatten()

ref = mix(y.double(), res.double(), post.double(), comb.double())
rms = ref.pow(2).mean().sqrt().item()

# bfp8_b has no element_size(): 1 sign+mantissa byte per datum plus a shared exponent per block.
BPE = {ttnn.float32: 4.0, ttnn.bfloat16: 2.0, ttnn.bfloat8_b: 1.0625}
dev = ttnn.open_device(device_id=0)
for name, dt in (("fp32", ttnn.float32), ("bfloat16", ttnn.bfloat16), ("bfloat8_b", ttnn.bfloat8_b)):
    up = lambda t: ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=dt)
    ts = (up(y), up(res), up(post), up(comb), up(build_bcast_consts(n)))
    f = lambda: ttnn.experimental.deepseek_prefill.mhc_post(*ts, n)
    out = ttnn.to_torch(f()).float().flatten()
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    for _ in range(ITERS):
        f()
    ttnn.synchronize_device(dev)
    ms = (time.perf_counter() - t0) / ITERS * 1e3
    nb = (1 + 2 * n) * T * C * BPE[dt]
    d = (out.double() - ref).abs()
    print(f"{name:10s} {ms:6.3f} ms  {nb/1e6:6.1f} MB  {nb/ms*1e3/1e9:6.1f} GB/s  "
          f"rel_rms={d.pow(2).mean().sqrt().item()/rms:.3e}  max|d|={d.max().item():.3e}")
ttnn.close_device(dev)
```

Timings here are host wall-clock. Do not compare them against Tracy device-FW numbers —
Tracy's per-op markers inflate small ops badly, and the two bases differ by ~1.5x on this op.

## Next

1. LoFi vs HiFi4 A/B on the fp32 arm, to find out whether MathFidelity does anything here.
2. Extend the PCC suite to bf16 and bfp8 with per-dtype thresholds; today it is fp32 only.
3. Per-CB page sizes in `reader_mhc_post.cpp` if mixed precision turns out to be wanted.
