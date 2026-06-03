# Chunked-Prefill 50K+5K Perf Sweep — scaffolding & findings (TEMP)

This commit adds **temporary sweep scaffolding** and **perf findings** for the ring-joint SDPA
chunked-prefill path on **P150_X8**. None of it is meant to merge as-is — the test-file hooks and the
two helper scripts must be reverted before this branch is finalized.

## What's here

### Findings (results)
- **`chunked_prefill_latent_v_sweep_findings.md`** — earlier latent-V vs separate-V sweep for the
  single shape seq=1280/q=64/sp=4.
- **`chunked_prefill_dm_latent_sweep_findings.md`** — the main 50K+5K sweep across four shapes
  (C1 sp8/q32, C2 sp4/q64, C3 sp4/q96, C4 sp8/q128) × k∈{256,384,512,640,768} ×
  {latent / non-latent V} × {DM-on / DM-off}. Headlines: sp8 is DM-bound at small q (C1 ~31% DM) and
  compute-bound at sp4 (~2% DM); latent V always wins (largest on sp8, ~30%); larger q_chunk helps sp8
  a lot. Raw numbers in `sweep_runs/results.tsv`.

### Scaffolding (TEMP — revert before merge)
- **`tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py`** — env-driven sweep knobs:
  `CHUNKED_PER_DEVICE_CHUNK`, `CHUNKED_Q_CHUNK`, `CHUNKED_LATENT_V`, `CHUNKED_SP_SIZE`,
  `CHUNKED_ONLY_LAST_CHUNK` (profile just the last/largest chunk), and `CHUNKED_SKIP_PCC` (perf-only;
  skips the torch oracle entirely and relaxes the util-bound assert for the compute-ceiling runs).
- **`dm_toggle.py`** — comments out / restores the bulk NoC data-movement primitives in the four
  active ring-joint dataflow kernels (no macros; a source edit guarantees a fresh JIT). `on`/`off`/
  `status`. Drop-set mirrors `noc_dm_gate.hpp`; all barriers / flushes / semaphores / CB handshakes are
  kept so the pipeline never deadlocks (compute runs on stale L1 — outputs garbage, timing valid).
- **`sweep_driver.sh`** — runs the full matrix one cell per `run_safe_pytest` invocation (clean device +
  auto-reset each), resumable (skips combos already in `results.tsv`), restores DM-on at the end.

### Kernel changes already on the branch (pre-existing, kept)
- `noc_dm_gate.hpp` + includes in `dataflow_common.hpp` / `chain_link.hpp`, the two `#ifndef
  RING_JOINT_DISABLE_NOC_DM` guards in `ring_joint_reader.cpp`, and the program-factory env gate — the
  original macro-based compute-ceiling path. The sweep above used the comment-out path instead, but
  these are left in place.

## Reproduce
```
CHUNKED_SP_SIZE=<4|8> CHUNKED_PER_DEVICE_CHUNK=<n> CHUNKED_Q_CHUNK=<n> CHUNKED_LATENT_V=<0|1> \
CHUNKED_ONLY_LAST_CHUNK=1 CHUNKED_SKIP_PCC=1 \
scripts/run_safe_pytest.sh \
 "tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py::test_ring_joint_attention_create_chunked_perf_table[kimi50k-q<Q>-k<K>-chunk<CS>]"

python dm_toggle.py off   # disable data movement   (restore: python dm_toggle.py on)
bash sweep_driver.sh      # full matrix -> sweep_runs/results.tsv
```
