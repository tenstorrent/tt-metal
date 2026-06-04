# Where the SDXL conv unit-test time actually goes (warm)

Question: for the nightly SDXL conv suite (`test_conv2d.py -k sdxl`), what dominates
wall time — the conv factory / auto-sharding path, random input creation, the PCC
check, or the torch reference output? And is precompile even attacking the bottleneck?

## Method (reproducible)

`cprofile_plugin.py` (repo root) wraps the whole pytest session in `cProfile` and
dumps `tottime`/`cumulative` tables to `CPROFILE_OUT`. Run over a **warm** on-disk
cache so JIT **compile is excluded** — this isolates the *non-compile* phases that
remain in steady state.

```bash
CPROFILE=1 CPROFILE_OUT=/tmp/p.txt TT_METAL_CACHE=/tmp/sdxlconv PYTHONPATH="$PWD" \
  scripts/run_safe_pytest.sh tests/ttnn/nightly/unit_tests/operations/conv/test_conv2d.py \
    -k "<selector>" -p cprofile_plugin -s
```

Two representative slices (warm):
- **Medium**: `test_conv2d_sdxl` — 38 cases, 16.9 s total (~0.44 s/case).
- **Giant**:  `vae_sdxl and 1024` — 16 VAE cases at 1024×1024 spatial, 124 s (~7.75 s/case).

## Breakdown — medium shapes (38 cases, 16.9 s)

| Phase | Time | % |
|---|---|---|
| ttnn conv2d build + auto-shard + dispatch/exec (device, warm) | ~4.9 s | 29% |
| `torch.randn` random input/weight creation | 4.06 s | 24% |
| **PCC check** (`comp_pcc` → `np.ma.corrcoef`) | 3.12 s | 18% |
| device CreateDevice/close (per-test fixture) | 1.76 s | 10% |
| `torch.conv2d` reference output (CPU) | 1.18 s | 7% |
| `from_torch` host→device | 0.76 s | 4.5% |
| collection + misc | ~0.6 s | 3.5% |

## Breakdown — giant shapes (16 VAE 1024×1024 cases, 124 s)

| Phase | Time | % |
|---|---|---|
| **PCC check** (`comp_pcc` → `np.ma.corrcoef`) | **86.4 s** | **70%** |
| `torch.randn` random input/weight creation | 11.9 s | 9.6% |
| `from_torch` host→device (tilize/convert) | 10.6 s | 8.6% |
| `torch.conv2d` reference output (CPU) | 6.6 s | 5.3% |
| ttnn conv2d build + auto-shard + dispatch/exec (device, warm) | ~2.2 s | 1.8% |
| `to_torch` readback | 1.9 s | 1.5% |
| device create + collection + misc | ~4.4 s | 3.5% |

## Findings

1. **The PCC check is the bottleneck, and it explodes with tensor size** — 18% at
   medium shapes, **70% at 1024×1024**. Root cause (`comp_pcc`,
   `models/common/utility_functions.py:542`):
   ```python
   np.min(np.ma.corrcoef(
       np.ma.masked_invalid(torch.squeeze(golden).numpy()).flatten(),
       np.ma.masked_invalid(torch.squeeze(calculated).numpy()).flatten()))
   ```
   `np.ma.corrcoef` on a 256×1024×1024 ≈ 268 M-element output `concatenate`s the two
   flattened **masked** arrays into a 2×N array (~GB), mean-subtracts, and computes
   covariance via a `dot`. Masked-array ops are Python-dispatched numpy with per-element
   mask bookkeeping → catastrophically slow and memory-heavy. (`corrcoef` 58 s,
   `_covhelper` 41 s, masked `concatenate` 14 s, `dot` 7.5 s, `numpy.array` 15 s …)

2. **The conv factory / auto-sharding path is NOT a bottleneck.** Warm, the entire
   `ttnn.conv2d` call (Python prep + auto-shard + program-cache lookup + enqueue) is a
   roughly **fixed ~0.13–0.14 s/case regardless of shape** — 29% of a tiny 16.9 s run,
   but only **1.8%** of the 124 s big-shape run. Device compute is efficient and async.

3. **Random input gen + host↔device transfer + reference conv** are real but secondary:
   together ~35% (medium) / ~23% (giant), all CPU/host.

## Implication for precompile

Precompile only attacks **JIT compile**, which is *excluded* from every number above
(warm cache). In steady state the SDXL conv suite is **host-bound**: PCC + randn +
transfer + reference dominate; the device conv (factory/auto-shard/exec) is ≤2% at the
big shapes. So even an infinitely fast compiler cannot speed this suite up much — the
lever for *this* suite is the **PCC check** (`np.ma.corrcoef` → `torch.corrcoef` /
plain `np.corrcoef`, or a chunked/torch implementation), not compilation. Precompile's
win remains on **compile-dominated** workloads (cold model bring-up; large suites of
many cheap ops), not execution/host-bound ones.
