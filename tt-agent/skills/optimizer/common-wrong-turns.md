# Common wrong turns in TT optimization

Accumulated anti-patterns and debug techniques from prior optimizer sessions.
Read before starting a new session; keep in mind during the Iterate phase.
Each entry names the symptom, the correct form, and the reason.

## L1 memory traps

### Don't use `ttnn.L1_MEMORY_CONFIG` for matmul activations

`L1_MEMORY_CONFIG` is **interleaved** L1, not sharded. Interleaved L1 forces
every core to fetch its tiles via NOC from other cores' L1 — on medium-sized
matmul activations this is slower than DRAM interleaved (observed 6.5×
regression on a Gemma3 c_fc). The correct form for matmul inputs is
**sharded L1 matching the matmul's grid** (HEIGHT, WIDTH, or BLOCK
depending on the matmul's parallelization strategy).

Sharded shard-spec construction is delicate — use a known-working sibling
op (Llama MLP, SDPA) as a template. A wrong spec either OOMs L1 or fails
correctness silently.

### L1 usable compute budget ≈ 1.2 MB (not 1.5 MB)

Per-core L1 is 1.5 MB total, but firmware, kernels, debug regions, and
circular-buffer reservations consume ~300 KB. Budget compute data at 1.2 MB
per core when sizing `per_core_M × per_core_N` output tiles, `in0_block_w`
reader buffers, and intermediates.

Running the sizing math against 1.5 MB and being "just under" will OOM at
runtime. The PSE Matmul Configuration Guide documents this — it's the
authoritative source.

## Compute kernel config traps

### `fp32_dest_acc_en=True` is often unnecessary for short-K matmuls

Default `matmul_config()` sets `fp32_dest_acc_en=True`. Cost: DST register
count drops from 8 to 4, which in turn caps `out_subblock_h × out_subblock_w
≤ 4` (vs ≤ 8). That cap alone can pin `out_subblock_w=1` on narrow N dims,
leaving math throughput on the table.

For short reductions (K ≤ ~100 tiles) with PCC target ≥ 0.99, BF16
accumulation in DST is plenty accurate. Try `fp32_dest_acc_en=False` as a
standalone iteration — typical wins 2-5%, and the freed DST enables bigger
subblocks in a follow-up iteration.

### Fidelity changes are often neutral; test rather than assume

The assumed ranking (LoFi < HiFi2 < HiFi4 on speed-vs-quality) is real but
not load-bearing for most non-compute-bound matmuls. On a default kernel
path with low DRAM/FLOPs utilization, HiFi4 → HiFi3 → HiFi2 can be
near-neutral because the bound isn't math. On an explicit progcfg path,
HiFi3 → HiFi2 is typically a small (~2%) win.

Test fidelity lowering once before assuming it matters. The runtime warning
about `HiFi4 + fp32_dest_acc_en` on Wormhole is real but its perf impact
isn't automatically decisive.

### `packer_l1_acc=True` is almost always the right default

Packer L1 accumulation folds partial-sum accumulation into the packer, which
removes the BF16 round-trip between matmul blocks. Leave it on unless
evidence says otherwise.

## Debugging matmul crashes

### Print the realized program config

A Python-constructed `MatmulMultiCoreReuseMultiCastProgramConfig` exposes
all its fields via `__repr__`. When a matmul fails with a kernel-level
assertion (`num_blocks_x` mismatch, CB overflow, dim assertion), the
fastest root-cause is:

```python
pc = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(...)
print(f"pc_<name>: {pc}")
out = ttnn.linear(..., program_config=pc)
```

The actual `per_core_M`, `per_core_N`, `out_subblock_h/w`, `out_block_h/w`
values are always there. Integer-division truncation, defaulted fields, and
padding are the usual surprises. One print saves multiple guess-and-check
iterations.

### Shared progcfg lambdas can be stale for a model variant

In tt-metal, many models reuse Llama progcfg lambdas (e.g.,
`IMAGE_MLP_FC_PROGCFG`) that close over `args.<dim>` values. Those dims
can be wrong for a different model — notably when computed as
`intermediate_size // hidden_dim` the int-div truncates. If a shared lambda
crashes on a new model, suspect the cached dim before blaming the progcfg
shape. Use `self.<weight>.shape[-1]` in the new model's progcfg instead of
`self.args.<cached_dim>`.

### `num_blocks_x > num_cores_x` is a dim problem, not a tuning problem

When you see `num_blocks_x=10, num_cores_x=8` or similar, the derived
`per_core_N` doesn't match the N dim. Fix the progcfg inputs (usually N or
`per_core_N`), not the grid. Rule of thumb: `per_core_N × num_cores_x` must
cover `N_tiles` with integer division — pick `per_core_N` such that
`ceil(N_tiles / per_core_N) ≤ num_cores_x`.

## Hypothesis discipline

### One change per iteration; coordinated pairs only when physically necessary

Single-variable iterations produce clean attribution. Coordinated pairs
(e.g., `MAX_MM_SEQ_LEN` bump + `in0_block_w` bump that requires the smaller
`per_core_M`) should be used only when the change is physically impossible
otherwise (the second variable becomes invalid on its own). Note the
coordination explicitly in the commit message so the attribution trail is
traceable.

### Keep forensic commits — don't revert failures

Forensic commits (kept, not reverted) for failed trials are worth their
weight. In the first gemma3 session, three forensic commits (L1-interleaved
regression, L1 OOM on big `in0_block_w`, stale-dim crash) were each useful
for pivoting and none needed reverting. They make the branch history a
faithful record of what was tried.

## Profiling workflow

### `tt-perf-report` 1.2.3 crashes on HiFi3 rows

`Unknown math fidelity` on any row with `MATH FIDELITY=HiFi3`. `--no-stacked`
does not help. Fall back to reading the CSV with pandas — see
`knowledge/recipes/tt-metal/profiler.md` for the snippet.

### Don't chase sub-percent noise

Trial-to-trial noise on a 5ms matmul is typically ±0.5%. A "1% improvement"
is inside the noise band — re-run the baseline before committing. The
`improvement_threshold` in `convergence.md` (default 2%) exists for this
reason.
