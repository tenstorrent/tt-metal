# sparse_sdpa_msa packed-group tooling (kagent/m3-prefill-sdpa)

Report: tt-kernel-agent `docs/tenstorrent/m3/31-prefill-sparse-sdpa.md`. Dev root: `/mnt/data/kernel-agent/dev/prefill-sdpa`.

| File | What |
|---|---|
| `build-sdpa.sh` | Surgical host rebuild (~25 s): recompile only the transformer unity blob holding `sparse_sdpa_msa_*`, with the canonical build's flags and PCH, relink `_ttnncpp.so` into `<worktree>/build_sdpa/lib` (all other libs symlinked). Point the worktree's `build` symlink at `build_sdpa`. |
| `kcheck.py` | Offline JIT compile check (no device): replays the JIT's riscv compile commands for the legacy kernels (from a run log) on the packed kernels with packed CT args. `KROOT=<variant root>` checks a variant. |
| `packed_model.py` | Host model of the packed reader's grouping / union / lead / diagonal logic and of compute's mask modes, checked against per-token reference attention (fp64) on captured indices and adversarial cases. |
| `cb_sim.py` | CB-protocol simulator (reader / writer / compute as coroutines over the factory's CB capacities): deadlock and push/pop balance per core. |
| `union_test.cpp` | Host test of `sparse_sdpa_msa_packed_union.hpp` (hashed union) vs the linear-scan union: `g++ -std=c++17 -I<kernels/dataflow> union_test.cpp`. |
| `core_stats.py` | Per-core union / row-step statistics for the measured cases (cost model). |
| `mkvariant.py` | JIT variant root (symlink farm with a real copy of the sdpa kernels dir) for probes. Run the job with **cwd = the variant root**: the JIT resolves a kernel's relative source path against the cwd before `TT_METAL_HOME`. |
| `runjob.sh`, `run-cases.sh` | `tt-partition-run prefill` wrappers for the op bench (`G:case[:iters[:variant]]`, compares against a legacy baseline dump). |

Never run ttnn host conversions (e.g. `ttnn.from_torch(..., bfloat8_b)`) outside `tt-partition-run`: it opens the
UMD cluster (all chips) unless `TT_VISIBLE_DEVICES=""` (now set by `canonical-env.sh`).
