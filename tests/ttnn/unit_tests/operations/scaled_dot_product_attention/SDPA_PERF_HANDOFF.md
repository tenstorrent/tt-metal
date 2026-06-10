# SDPA perf-juicing handoff

## Objective

Make the generic **scaled_dot_product_attention** op as fast as possible on a
**small, favorable subset of shapes** — maximize on-device matmul utilization.
You do **not** need to preserve generality or precision elsewhere. The only bars are:

- **bfloat16** inputs, self-attention (MHA), `mask=none`, auto scale.
- `fp32_dest_acc_en` is **not** required (default compute config is fine).
- **PCC ≥ 0.99** on the four cases in `test_sdpa_perf_juice.py` — that's the *entire*
  correctness contract. You may trade away precision (lower fidelity, bf16/bf8b
  intermediates, approximations) as long as those four stay ≥ 0.99.
- Every case must keep **filling the full 8×8 grid** (the test asserts `cores == 64`).

You may freely break other dtypes, masks, alignments, cross-attention, GQA/MQA —
nothing outside these four cases matters for this task.

## Where things are

- Op: `ttnn/ttnn/operations/scaled_dot_product_attention/`
  - `scaled_dot_product_attention.py` — host entry + registry (`SUPPORTED`/`validate`).
  - `scaled_dot_product_attention_program_descriptor.py` — **work distribution + CBs** (the main lever).
  - `kernels/scaled_dot_product_attention_{reader,compute,writer}.cpp` — device kernels.
- Tests: `tests/ttnn/unit_tests/operations/scaled_dot_product_attention/test_sdpa_perf_juice.py`

## How to run / measure

```bash
scripts/run_safe_pytest.sh --run-all -s \
  tests/ttnn/unit_tests/operations/scaled_dot_product_attention/test_sdpa_perf_juice.py
```

Each case prints (and appends to `generated/sdpa_juice_perf.txt`):

```
[<name>] cores=64/64 device_kernel_ns=<N> achieved=<X>TFLOPs util=<Y>% pcc=<P>
```

- `device_kernel_ns` is the on-device kernel span (first kernel-start → last
  kernel-end across all cores/RISCs), read from the device profiler in-process
  (`ttnn.ReadDeviceProfiler` + `get_latest_programs_perf_data()` →
  `"DEVICE KERNEL DURATION [ns]"`). This is your optimization signal — drive it down.
- `util` is against **LoFi matmul peak ≈ 258 TFLOPs** (8·16·16 = 2048 MAC/cyc → 4096
  FLOP/cyc/core × 64 cores × ~0.9855 GHz). FLOPs counted = `4·B·H·S²·d` (the two
  matmuls QKᵀ and P·V; `mask=none` ⇒ full S×S).

`--run-all` is important: `run_safe_pytest.sh` appends `-x` by default, so without
it a single failing case hides the rest.

**Follow the method in `OPTIMIZATION_PROTOCOL.md`** — the disciplined one-change-at-a-time
loop, how to profile (and the gotcha that the auto CB-wait columns are empty for this
generic op → instrument with `DeviceZoneScopedN`), and the journal. Two tools support it:

- `scripts/run_safe_pytest.sh --tracy <test>` — runs under the Tracy profiler and emits
  the per-op CSV (`generated/profiler/.../ops_perf_results_*.csv`) for deeper analysis.
- `log_perf_attempt.py` — append a journal entry per attempt to
  `agent_logs/perf_journal.jsonl` (auto-attaches the latest measured numbers). Keep one
  entry per change with a keep/revert decision, so progress is auditable and reversible.

## The four cases (all fill the grid)

| name | B | H | S | head_dim | notes |
|---|---|---|---|---|---|
| favorable_h8_s4096_d64 | 1 | 8 | 4096 | 64 | classic shape |
| favorable_h8_s8192_d128 | 1 | 8 | 8192 | 128 | bigger matmuls (d=128 amortizes better) |
| favorable_h16_s8192_d128 | 1 | 16 | 8192 | 128 | more heads → more work units |
| very_favorable_h8_s16384_d512 | 1 | 8 | 16384 | 512 | the friendly one: huge K-dim + long seq |

## How the op parallelizes today (read this first)

Work **unit = one `(batch, head, q_chunk)`**: `c_q` Q tile-rows attended against
*all* of K/V (flash-style streaming over KV chunks of `c_kv`). Units are flattened
over `b·H·Nq` and dealt out contiguously to the grid:

```python
c   = max(1, min(4, 16 // Dt))     # Dt = head_dim / 32
c_q = min(c, Sq_t)                 # Q-chunk height in tiles
c_kv= min(c, Skv_t)                # KV-chunk width in tiles
Nq  = ceil(Sq_t / c_q)
total_units = B * H * Nq
num_cores   = min(64, total_units)
```

So for `head_dim ∈ {64,128}` → `c=4` (chunk = 4 tile-rows = 128 rows); for
`head_dim=512` → `Dt=16`, `c=1` (chunk = **1** tile-row, and **`c_kv=1` too** —
KV streamed one tile at a time). All four cases reach 64 cores.

## Baseline (measured, current kernel)

| case | device_kernel_ns | util (vs LoFi peak) | pcc | cores |
|---|---|---|---|---|
| favorable_h8_s4096_d64 | 12,232,917 (12.2 ms) | **1.09%** | 0.99999 | 64/64 |
| favorable_h8_s8192_d128 | 54,953,359 (55.0 ms) | **1.94%** | 1.00000 | 64/64 |
| favorable_h16_s8192_d128 | _(run to fill)_ | | | 64/64 |
| very_favorable_h8_s16384_d512 | _(run to fill — likely slow: c_kv=1)_ | | | 64/64 |

All cases fill the grid. The `util` column above is against **LoFi peak (258 TFLOPs)**.
Note the op actually runs at **`MATH FIDELITY = HiFi3`** (3 passes through the matrix
unit ⇒ peak ≈ LoFi/3 ≈ **86 TFLOPs**). So against the fidelity it's *really* using, FPU
util is ~3× the LoFi numbers (≈ **3.3%** and **5.8%**). Two readings, both true:
- **vs LoFi (~1–2%)** = headroom against best case — and "switch to LoFi" is *part* of that
  headroom: a ~3× matmul-throughput lever if PCC ≥ 0.99 survives it.
- **vs HiFi3 (~3–6%)** = how well the FPU is used at the current fidelity.

Either way it's far from compute-bound.

**Key finding from profiling:** the grid is *not* the problem on these shapes — it
fills all 64 cores. The bottleneck is **per-core efficiency**: each core runs a
modest matmul, then SFPU softmax with **online rescaling** (running max/sum, α
correction), then the second matmul, then reloads the next KV chunk — lots of
serialized non-FPU work, so the matrix engine sits idle most cycles. Per-active-core
utilization is a near-constant ~1% (d=64) / ~1.9% (d=128). **Fanning out more will
not help; you must keep the FPU busier.**

## Suggested levers (roughly highest-leverage first)

1. **KV chunk size for large head_dim.** `head_dim=512 ⇒ c=1 ⇒ c_kv=1`: the KV loop
   runs one tile-row at a time → the online-softmax bookkeeping fires per KV tile.
   Decouple `c_kv` from the `16/Dt` clamp and make it larger so each matmul does
   more work per softmax step. Likely the single biggest win for the d=512 case.
2. **Overlap softmax with matmul / reduce online-softmax overhead.** The running
   max/sum + α-rescale between the two matmuls serialize against the FPU. Fewer
   passes, fused reductions, or recomputing max less often (you have PCC 0.99 of
   slack to spend) all help.
3. **Bigger `c_q`.** More Q tile-rows per chunk amortizes the per-chunk softmax and
   KV reload across more matmul rows (raises matmul `M`). Watch L1 budget.
4. **Matmul sub-block / fidelity tuning.** LoFi vs HiFi2, sub-block dims for QKᵀ and
   P·V, DEST packing — make the two matmuls themselves efficient.
5. **Cut redundant CB copies / data movement.** Reader reloads, intermediate CB
   round-trips between scores/probs/PV.

Re-run the test after each change and watch `util` / `device_kernel_ns`. Target:
materially above the ~1–2% baseline (the owner is not expecting >15%, but there's
~50–100× of headroom against LoFi peak, so meaningful gains should be easy early).

## Notes

- The harness has device perf capture wired in generally now (`device_kernel_ns`
  column, on by default; `EVAL_CAPTURE_PERF=0` disables) — but this standalone test
  measures perf itself, so it works regardless.
- Keep the grid-fill assertion: if a refactor changes the work-unit math and a case
  drops below 64 cores, that's a regression for this task.
