# MoE block test — timings & ccache experiment

Standalone `moe_test.py` runs only the layer-1 MoE block (the code between the
`moe_start` / `moe_end` signposts in `main.py`), on captured e2e inputs, and
PCC-checks against captured e2e outputs. PCC = 1.000000 (max|Δ| = 0) in every
configuration below.

Hardware: 32x Blackhole (4x8 mesh). The `moe_block` phase includes JIT kernel
**compilation** (on a cold kernel cache) plus execution, so it is the phase that
`TT_METAL_CCACHE_KERNEL_SUPPORT` accelerates. The tt-metal JIT kernel cache
(`~/.cache/tt-metal-cache`) was cleared before every run so kernels recompile.

`TT_METAL_CCACHE_KERNEL_SUPPORT` (tt_metal/jit_build/build.cpp): when the env var
is present, tt-metal prepends `ccache` to the `riscv-tt-elf-g++` kernel-compile
command. ccache 4.9.1, cache_dir `~/.cache/ccache`.

| run | TT_METAL_CCACHE_KERNEL_SUPPORT | ccache state | device_open | ce_cache_load | moe_block | TOTAL |
|-----|--------------------------------|--------------|-------------|---------------|-----------|-------|
| baseline (cold JIT)      | unset | n/a              | 14.633 | 0.661 | **55.203** | 70.510 |
| ccache populate (cold JIT) | set | cold (0% hit)    | 15.346 | 0.657 | 64.539 | 80.553 |
| ccache measured (cold JIT) | set | warm (100% hit)  |  9.613 | 0.653 | **23.346** | 33.624 |

(times in seconds; `input_load` ≈ 0.01s elided)

## Takeaways
- With a warm ccache, kernel-compile-bound `moe_block` dropped **55.2s -> 23.3s
  (~2.4x, ~32s saved)**; total **70.5s -> 33.6s**. ccache reported 687/687
  hits (100%).
- The first ccache run (cold ccache) is slightly *slower* than baseline
  (64.5s vs 55.2s) due to ccache store overhead on misses — the win shows on
  subsequent cold-JIT runs.
- The const-eval cache (`moe_io/ce_cache/`, 12 tensors) keeps `ce_cache_load`
  at ~0.66s, vs the one-time full weight-load + const-eval build (~5 min).
- If the tt-metal JIT kernel cache itself is warm, `moe_block` is fast
  regardless of ccache (no compilation happens); ccache only matters when
  kernels must be recompiled (cleared cache, different build, CI, etc.).

## How to reproduce
```bash
cd deepseek_codegen/graph_0
export TT_METAL_HOME=/home/ubuntu/tt-metal PYTHONPATH=/home/ubuntu/tt-metal ARCH_NAME=blackhole
# baseline:
unset TT_METAL_CCACHE_KERNEL_SUPPORT; rm -rf ~/.cache/tt-metal-cache; python3 moe_test.py
# with ccache (run twice; 2nd run is the accelerated one):
export TT_METAL_CCACHE_KERNEL_SUPPORT=1
rm -rf ~/.cache/tt-metal-cache; python3 moe_test.py   # populate ccache
rm -rf ~/.cache/tt-metal-cache; python3 moe_test.py   # ccache hits -> fast
```
