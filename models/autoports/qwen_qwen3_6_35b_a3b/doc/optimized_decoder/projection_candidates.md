# Projection candidate harness (unmeasured)

`tests/optimized_projection_candidates.py` supplies `ProjectionCandidateDecoder`,
a diagnostic subclass of the current `OptimizedDecoder`. It changes projection
construction/dispatch only; attention, routing, recurrent state, KV cache,
prefill chunking, expert selection, and output assembly remain inherited. The
CLI explicitly installs the diagnostic subclass in `run_decoder.run`; there is
no functional fallback. This document proposes experiments, not results.

Weight packing uses the real supplied state dict only at setup. Forward methods
use only TTNN. Shared/expert precision and sparse geometry remain constructor
options inherited from `OptimizedDecoder`. The sensitive router and ordered
linear-attention projection arithmetic are untouched.

## Topology and policies

| Role | Baseline representation | Candidate alternatives |
|---|---|---|
| Full attention input | Q/K/V/gate packed to 9216 columns | Same packed projection, or four separate DRAM-sharded Q/K/V/gate matmuls plus concat |
| Full attention output | 4096 × 2048 BF16 | BFP8 or BF16 independently of expert/shared groups; HiFi4/HiFi2/LoFi |
| Shared input | Gate 512 + up 512 + padded scalar 32 | `packed_all`; `packed_gu_scalar` (GU1024 + scalar32); `split` (gate512 + up512 + scalar32) |
| Shared output | 512 × 2048 | DRAM-sharded or automatic interleaved matmul |
| Shared scalar | Included in packed dtype/fidelity | Independent BF16/HiFi4 scalar for GU+scalar or split modes |

HF Q projection stores Q/gate interleaved per head. Both candidates reorder to
Q4096/K512/V512/gate4096 before invoking inherited attention consumers. Shared
scalar is zero-padded from one output to 32, so every topology returns 1056
logical columns and its useful scalar stays at column 1024. Split topology
retains concat costs in end-to-end measurement. A future direct-consumer
candidate could remove concat; this harness does not claim to measure it.

## Proposed matrix

1. Full attention layer 3: `attention_dtype=bf16`, `attention_fidelity=HiFi4`
   control; BFP8/HiFi2; BFP8/LoFi; BF16/HiFi2. Keep the expert/shared settings
   fixed across these rows. Constructor defaults are a candidate, not a selection.
2. Lock the best correct attention precision, then compare readers 1/2/3 with
   `block_w` 2/4/8/16 and input cores 8/16/32. For output K4096, also try block
   width 32. Smaller-core inputs expose shard widths above two tiles. Compare
   bank tile multiples 1 and 3; the fused implementation padded each bank to
   three tiles. Padding is an independent traffic/worker-distribution parameter.
3. Compare packed and split attention under the same dtype/fidelity and tuned
   per-role legal geometry. Keep each best correct end-to-end result.
4. Shared: compare all three topologies, first with BFP8/HiFi2 and scalar BF16/
   HiFi4 on the separated modes; then BFP8/LoFi, and BFP4/LoFi only as stage-8
   evidence. `shared_dram=false` is the automatic-interleaved control.
5. Shared per-role geometry: gate/up/GU K2048 block widths 2/4/8/16; down K512
   widths 2/4/8/16 and input cores 4/8/16. Reader 1/2/3 comparisons must use the
   same precision. Scalar has only one useful output; independent interleaved
   dispatch may be a later candidate if DRAM sharding overhead dominates.
6. `prefill_block_w` exposes an explicit 8×8 1D matmul control, with widths
   2/4/8/16 where K permits; compare against unset (automatic) at 32, 256 and
   chunk-boundary lengths. This does not alter sequence acceptance or chunking.

Geometry JSON accepts `default` plus per-role overrides. Role names are
`attention_packed`, `attention_q`, `attention_k`, `attention_v`, `attention_gate`,
`attention_out`, `shared_packed`, `shared_gu`, `shared_gate`, `shared_up`,
`shared_scalar`, `shared_down`. Fields are `readers`, `input_cores`, `block_w`,
`per_core_n`, `bank_tile_multiple`. The result JSON includes the requested policy
and effective K/N/shard/program geometry for each installed DRAM projection.

Example (owner must serialize hardware runs and set the normal environment):

```bash
timeout 600 python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.optimized_projection_candidates \
  --layer 3 --length 256 --context 512 --output <artifact>.json \
  --build-kwargs '{"attention_dtype":"bfp8","attention_fidelity":"HiFi2","attention_topology":"split","shared_topology":"packed_gu_scalar","projection_geometry":{"default":{"readers":3,"block_w":8,"input_cores":8},"shared_down":{"input_cores":4}}}'
```

Apply the owner's current best expert/sparse policy in the same build kwargs
before ranking any candidate; the inherited defaults might change during the
ongoing expert sweep. Use tracking enabled for correctness/replay, disabled for
latency, and separate watcher and profiler runs. Real whole-output PCC, stable
traced replay, timing, and profiler dtype/fidelity rows remain mandatory before
integration. No hardware was opened by this implementation subtask.

## Source constraints and pending evidence

Inspected `test_matmul_dram_sharded_multicore.py`,
`matmul_multicore_reuse_mcast_dram_sharded_program_factory.cpp`,
`matmul_device_operation.cpp`, and `matmul_nanobind.cpp`.

- Readers 2/3 explicitly set `use_multicore_pipeline=True`; false with multiple
  readers uses a different legacy path and imposes a Blackhole-only check.
- Single-reader block widths may exceed the input shard width if they are exact
  multiples; the source gathers across storage shards. A small input shard is
  therefore not sufficient evidence to reject a larger block width.
- Multicore pipeline handles widths larger than storage shards and partitions
  narrow banks into K groups. Weight shards cover all K and all DRAM banks;
  activation shards must cover tiled K. The wrapper validates integral tiled
  shards and legal positive values before dispatch.
- The input/output L1 grid, actual L1 allocation, single-reader behavior with
  narrow scalar outputs, prefill program legality, numerical effects, and
  performance still need device validation. An initial API failure is a debug
  result, not an earned rejection. Adapt geometry/layout and retry.
- KV dtype/layout is unchanged. No context capacity reduction is introduced.
- Shared scalar defaults to BF16 only when split from the packed projection;
  `packed_all` necessarily uses the shared group's dtype/fidelity and records
  that actual policy rather than claiming an independently precise scalar.

CPU validation: Python AST/bytecode compilation passed and Black formatting was
applied with `--target-version py312`. No measured claim or final selection.
