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

## CCL / multi-device traps

### `num_links` on CCL ops must match the physical link count

`ttnn.experimental.all_gather_async(..., num_links=N, ...)` and siblings
(`reduce_scatter_*`, `all_reduce_*`) require `N` to match the physical
inter-chip ethernet link count for the device topology. Wrong values
**deadlock the device** (hang until watchdog reset) — not a soft failure.

Counts observed in production code:
- N150 single-chip: no CCL (N/A)
- N300 (2 chips, Ring): `num_links=1`
- Galaxy (32 chips, Ring): `num_links=4`
- TG (Tensor Galaxy): per-axis, see the specific model's `tt_ccl.py`

If unsure, copy from a sibling model's `tt_ccl` module (e.g.
`tt_transformers/tt/ccl.py`) which auto-detects. Hardcoding the galaxy value
on N300 was observed to deadlock for 10+ minutes before timing out, requiring
`tt_device_reset`. Keep `tt_device_reset` preloaded when testing CCL tuning
parameters (see SKILL.md Preflight).

### Cross-device bias fusion needs `÷num_devices` pre-divide

When a matmul is followed by AllGather + FastReduceNC (or any sum-reduce
across devices), fusing the bias into the matmul via `ttnn.linear(bias=...)`
adds the bias on each device, and the cross-device sum multiplies it by
`num_devices`. Correct form:

```python
# In __init__, before creating the ttnn tensor:
if args.num_devices > 1:
    bias_torch = state_dict[...] / args.num_devices
else:
    bias_torch = state_dict[...]
# Use a distinct cache filename (e.g. "bias_div") so the old cache is not reloaded.
```

The post-reduce sum then yields the original bias exactly. Same trick applies
to any reduced intermediate followed by a bias/scale add.

### AG/RS tuning parameters have local optima — sweep, don't assume monotonic

`ttnn.experimental.all_gather_async` knobs `chunks_per_sync`,
`num_workers_per_link`, `num_buffers_per_channel` do **not** improve
monotonically with any direction of change. Observed on Gemma3 N300:

- `chunks_per_sync`: 10 → 4 improved -309μs; 4 → 2 regressed +976μs
- `num_workers_per_link`: 2 → 4 improved; 4 → 8 neutral (wastes cores)
- `num_buffers_per_channel`: 2 → 4 regressed +47μs

Do a small targeted sweep (3-4 values per knob) rather than a directional
tune. Record the full sweep in a mini-table in the trend file — future
sessions reading it can avoid re-walking the same regressions.

## Numerical correctness traps

### Fused activations have a different numerical path than standalone

`matmul_config(..., fused_activation=ttnn.UnaryOpType.GELU)` (fused into
the matmul's packer) and `ttnn.gelu(...)` as a separate op do **not** use
identical approximations. Observed PCC shift: 0.9999596 → 0.9991 on a
Gemma3 MLP when fusing GELU. This is expected, not a bug — the fused path
is still within the 0.99 quality bar and the 0.999 abort threshold.

Before concluding a fusion "broke correctness", re-check against the user's
actual PCC bar (not the default 0.999 `pcc_abort`). Session-specific overrides
may allow continuing.

Corollary: the first GELU/SiLU/RELU-fusion iteration typically drops PCC
by 2-5× versus pre-fusion. Don't revert on that alone.

### `in0_block_w` divisor spacing can block the obvious next step

`in0_block_w` must divide `K_tiles` exactly. The *spacing* between valid
divisors matters — a K with sparse factorization leaves no middle ground.

Example: `K_tiles=68 = 4 × 17`. Valid divisors: {1, 2, 4, 17, 34, 68}. Going
from `in0_block_w=4` (17 K-iterations) to `=17` (4 K-iterations, 3× fewer
dispatches) requires a 1820 KB CB footprint, which OOMs L1 (max 1499 KB).
There is no `=8` or `=10` option.

Before planning a K-blocking jump, factorize `K_tiles` and enumerate valid
divisors. If the next valid value is too big for L1, the only ways forward
are (a) reshape `per_core_M` smaller to free L1 for the bigger
`in0_block_w`, or (b) accept the current block size and move the lever
elsewhere.

## Structural pivot traps

### Check the sibling implementation's *mode* before committing to a structural change

tt-metal models often share kernels across PREFILL and DECODE modes with
different progcfgs per mode. If a target file says "reused from X", verify
X's progcfg path used in the *same mode* as the target — not the other mode.

Observed: planned L1-block-sharded activations for Gemma3 vision MLP (a
PREFILL-like path at seq=16512), based on Llama MLP's DRAM-sharded config.
Llama MLP uses DRAM-sharded only in DECODE; its PREFILL uses
DRAM-interleaved. The L1-sharding plan would have burned iterations
pressuring the 1.2 MB per-core budget for no reason.

Prepare-phase check: `grep` the sibling module for the mode keyword
(`prefill`, `decode`, `mode==`) and confirm the progcfg branch the target
would inherit.

## Code hygiene

### Comments explain the current invariant — never the iteration history

The optimizer loop naturally produces commit-narrative comments because each
iteration is framed as a Δ from the previous best. Examples observed in a
single MLP file after 19 iterations:

```python
# c_proj: use full 8-column grid. With per_core_N=5, all 64 cores engage
# (vs 48 with per_core_N=6 which left 16 cores idle).
# ...
# MAX=4128 matches num_chunks=4, gives per_core_M=17 (fewer dispatch
# barriers than 1376).
# ...
# HiFi2 + BF16 accumulation. Drops fp32_dest_acc_en=True — lifts DST
# register pressure (8 regs instead of 4).
```

None of `48`, `1376`, `fp32_dest_acc_en=True`, or `4 regs` appear anywhere
in the current code. They are facts about the *iteration journey*, not the
*current code*. They rot the moment a future iteration touches a different
variable, and they duplicate `git log` and the commit message.

**Rule before leaving a comment:** would this comment still be true and
useful if this iteration were the first commit on the branch? If no, drop
it — the commit message captures the Δ.

**Keep** comments about non-obvious current-state invariants — a subtle
correctness argument, a hardware constraint the code depends on, a
workaround for a specific bug that would be surprising to a future reader.
Example that is good:

```python
# Pre-divide bias by num_devices so the post-AllReduce sum yields the
# original bias exactly.
```

This explains *why the current code is shaped this way*, independent of
history.

**Drop** anything that reads as a changelog: `vs`, `was`, `previously`,
`instead of`, `dropped`, `removed`, `now`, parenthesized numbers that no
longer appear in the code. Those belong in the commit message of the
iteration that introduced the change.

**Drop** reasoning-by-elimination of rejected alternatives ("we chose A
because B OOMs and C leaves cores idle"). By the time someone reads the
code, only A exists — the reader has no way to verify the rejection
claims are still true. If A has a non-obvious constraint (e.g. "A fits
L1 only when fp32_dest_acc_en=False"), state the constraint on A, not
the history of why B and C were rejected.

This rule applies to **every edit in every iteration**. At the end of a
productive iteration, re-read the edited region as if it were the first
commit on the branch — delete any comment that wouldn't make sense in
that framing. The commit subject already captures the Δ; don't duplicate.

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

### Forensic commits: label intent in the first word of the subject

`opt(<scope>): forensic — <what was tried>` distinguishes kept-but-failed
iterations from productive ones at a glance. When the trend file's
"forensic failures" table is regenerated from `git log`, the prefix filters
cleanly. Consider `opt(<scope>): revert — ...` for explicit reverts of
failed productive-commit branches.
