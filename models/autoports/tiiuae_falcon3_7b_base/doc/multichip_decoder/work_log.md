# Multichip decoder work log

All commands are run from the repository root.  Hardware tests are serialized.

## 2026-07-18 — target selection before final implementation

- Baseline commit: `b35b3c1cbf7` on
  `mvasiljevic/model/tiiuae-falcon3-7b-base`.
- Preserved unrelated pre-existing modification:
  `.agents/skills/forge-functional-decoder-from-ir/SKILL.md`.
- `timeout 60 tt-smi -ls --local` reported four Blackhole p300c devices,
  numbered 0 through 3.
- A serialized TTNN mesh smoke opened and closed `MeshShape(1,4)` and reported
  `NUM_DEVICES 4`, `MESH_SHAPE [1,4]`, `MESH_SMOKE_OK`.
- The authoritative compiler provenance selects mesh `[1,4]`, TP axis 1,
  local heads Q=3/KV=1, column-parallel QKV and MLP gate/up, row-parallel O and
  down, local KV cache, and two ring all-reduces per decoder layer.
- Selected the complete compiler-proven replicated-residual TP4 path described
  in `README.md`.  Residual-sharded/fused variants remain measured probes, not
  an incomplete production path.
- Recovered an uncommitted older TP2 prototype only as source material.  Its
  1x2 results and ignored profiler files are not accepted as TP4 evidence.

## 2026-07-18 — implementation and correctness

- Implemented the fixed TP4 path in `tt/multichip_decoder.py`, using
  `OptimizedDecoder` as the declared numerical and local-kernel baseline.
- Runtime reported eight DRAM banks per chip, not the 12 assumed by the old
  prototype.  QKV remained `[3072,1280]`; decode MLP shards were padded from
  5,760 to 6,144 to satisfy both eight banks and 24 compute cores.  A first
  prefill exposed `Kt=180` not divisible by `in0_block_w=8`; the final path
  selects the largest legal divisor, six.
- The final safe correctness suite was:

```text
FALCON3_RUN_MULTICHIP_TOPOLOGY=1 \
FALCON3_RUN_MULTICHIP_FRACTURED_BOUNDARY=1 \
FALCON3_RUN_MULTICHIP_HETEROGENEOUS_POSITIONS=1 \
FALCON3_RUN_MULTICHIP_LONG_PREFILL=1 \
FALCON3_RUN_MULTICHIP_MAX_CONTEXT=1 \
FALCON3_RUN_MULTICHIP_BASELINE=1 \
python -m pytest -q -s \
  models/autoports/tiiuae_falcon3_7b_base/tests/test_multichip_decoder.py
```

Result on the hook-formatted final source: `8 passed, 2 skipped in 54.68s`.  The skips were
the separately completed performance and profiler gates.  The safe boundary
test carries a fractured partial through reduce-scatter, local residual add,
distributed RMSNorm, all-gather, and the next real QKV.  It produced PCC
0.999898 but was 1.1265x slower than the replicated boundary.  The repository's
fused matmul+reduce-scatter primitive was not invoked because its Blackhole
gate documents a nondeterministic synchronization race (issue #46181).  PCC
and shape results are in the README and `results/`.

## 2026-07-18 — tuning and warmed performance

The common performance command was:

```text
FALCON3_RUN_MULTICHIP_PERF=1 \
FALCON3_MULTICHIP_PERF_BATCH=<1-or-32> \
python -m pytest -q -s \
  models/autoports/tiiuae_falcon3_7b_base/tests/test_multichip_decoder.py::test_warmed_multichip_trace_performance
```

Candidate environment overrides swept `QKV_CORES`, `O_CORES`,
`GATE_UP_CORES`, `DOWN_CORES`, `CCL_DTYPE`, and `NUM_LINKS`, while retaining
the selected BFP4/LoFi precision policy.  The review-completion sweep measured
QKV16/O8, O12, gate8, down8, and gate8/down8 independently and records each
resolved physical grid.  The selected 8/8/24/8, BF16, two-link result is
0.356696 ms at batch 1 and 0.578799 ms at batch 32.  Corresponding optimized
single-chip results are 0.644047 and 0.768483 ms.  Batch-32 prefill is 2.740996
ms versus 3.262737 ms single-chip.  Final and rejected-candidate JSON files are
under `results/`.

The final completion audit found that the source also exposes distinct
12-core gate/up and down grids.  Three additional serialized runs on the exact
final implementation/test hashes measured gate12/down8 at 0.363515 ms,
gate24/down12 at 0.357040 ms, and gate12/down12 at 0.363764 ms.  The selected
gate24/down8 artifact is 0.356696 ms; its five samples do not overlap the
12-core-down sample range.  These runs close the midpoint geometry rather than
inferring it from the previously measured 8/24 endpoints.

For stale-artifact closure, the audit then reran every earlier QKV/O/MLP,
BF16-versus-BFP8 CCL, and one-versus-two-link control.  All thirteen
`sweep_*.json` files now bind to the final implementation SHA
`b249847705594bbf49e795eecdc1669a0016929929952d0767c9939cef1dd573`
and test SHA
`b6b10e32580ff5e4f9fdb465099272727a147049d5f191e91bd0dcae91ded557`
and include resolved physical grids and padding.  The refreshed ordering is
unchanged; the selected 8/8/24/8, BF16, two-link path remains fastest.

One QKV16 sweep completed its timing but initially failed while serializing a
report-only field (`local_intermediate` versus `local_intermediate_size`); it
was fixed and rerun successfully.  The first final profiler command omitted
Tracy's `-m pytest` switch and failed before device initialization or data
capture; the exact documented command below was then run successfully.

## 2026-07-18 — Tracy and tt-perf-report

Profiler and Watcher were run separately.  The profiler command was:

```text
FALCON3_RUN_MULTICHIP_PROFILE=1 python -m tracy -r -p -v \
  -o models/autoports/tiiuae_falcon3_7b_base/doc/multichip_decoder/tracy/tp4_selected \
  -m pytest -q -s \
  models/autoports/tiiuae_falcon3_7b_base/tests/test_multichip_decoder.py::test_multichip_profile_signposts
```

Human and CSV views were generated from the raw ops CSV with:

```text
tt-perf-report <ops_csv> \
  --start-signpost MULTICHIP_PREFILL --end-signpost MULTICHIP_PREFILL_END
tt-perf-report <ops_csv> \
  --start-signpost MULTICHIP_DECODE --end-signpost MULTICHIP_DECODE_END
```

The final source ops CSV SHA256 is
`181fb8acb8614c403404146b2cbafafa4106974da781bf667cf972798b0dfa25`.
See `tracy/tp4_selected/profile_provenance.json` for exact output hashes.

## 2026-07-18 — Watcher

Default Watcher, `DUMP_ALL`, and stack-usage-disabled Watcher each failed
before device initialization: Watcher-expanded active-Ethernet fabric code was
27,920 bytes versus a 25,600-byte kernel-config buffer.  After each startup
failure the four devices closed cleanly and `tt-smi` reported four healthy
DRAM devices.  The narrow accepted instrumentation was:

```text
TT_METAL_WATCHER=10 TT_METAL_WATCHER_DISABLE_ETH=1 \
python -m pytest -q -s \
  models/autoports/tiiuae_falcon3_7b_base/tests/test_multichip_decoder.py::test_real_layer_paged_non_aligned_prefill_decode_cache_and_trace
```

Result on the final source: `1 passed in 7.95s`, with the ring fabric and two all-reduces active.
All worker/dispatch checks remained enabled, four devices attached/detached,
and the exact log has zero error-pattern matches.  Both `watcher.log` and the
captured `pytest_stdout.log` are preserved under `watcher/`.

## 2026-07-18 — review remediation

The first independent `$stage-review` verdict was `more-work-needed` with
three findings:

1. The residual-sharded claim was based on a boundary-only microprobe.
2. Material decode geometries (QKV16/O8, O12, and 8-core gate/down) were not
   all measured under the selected precision.
3. BFP8 capacity used one byte per element rather than the physical 1,088-byte
   tile and omitted trace/transient reserves.

All three were corrected: the complete distributed-norm/next-QKV boundary is
now measured, the missing geometry sweep selected 8-core down, and
`context_contract.json` now records physical cache tiles plus calculated trace
and activation reserves.  The reviewer also requested pytest stdout from the
exact Watcher run; it is preserved and hashed.

The final completion audit found and corrected one additional conservative
capacity-ledger detail.  A host-only TTNN probe of a BF16 TILE-layout norm
weight reported logical shape `[3072]`, padded shape `[32,3072]`, and volume
98,304 elements.  The all-layer projection-and-norm term therefore uses
1,937,080,320 physical bytes rather than 1,926,414,336 logical-norm bytes.  The
same audit then accounted for full-sequence buffers retained around the
1,024-row linears: the explicit O-concatenation peak is 452,984,832 bytes
beyond the resident residual, so the transient reserve is 512 MiB and the
complete batch-1 resident-plus-reserve total is 3,307,954,432 bytes per device.
Neither correction changes the executed 32,768-token contract.

An earlier `$stage-review` returned `clean-pass` before this conservative
completion audit.  The final review is rerun below on the remediated tree so
that `stage_review.md` does not overstate which artifacts were inspected.

## 2026-07-18 — fused all-gather/matmul closure and recovery

The completion reviewer found that the residual-sharded topology audit had
measured explicit all-gather but had not exercised the repository's fused
all-gather/matmul family.  A focused probe now carries a row-parallel partial
through reduce-scatter, local residual addition, distributed RMSNorm, and the
real next layer-14 BFP4 QKV projection.  The serialized command was:

```text
FALCON3_RUN_MULTICHIP_FUSED_AGMM=1 timeout 900 python -m pytest -q -s \
  models/autoports/tiiuae_falcon3_7b_base/tests/test_multichip_decoder_agmm_probe.py::test_fractured_residual_fused_all_gather_qkv
```

The hook-formatted safe one-link candidate passed in 7.82 seconds.  Median
5x100 warmed trace times were 0.097549800 ms replicated, 0.109961499 ms
fractured with explicit all-gather, and 0.110680684 ms fractured with fused
all-gather/QKV.  PCC versus
replicated was 0.999898026 and 0.999858606; every trace was bitwise
deterministic.  The durable probe SHA is
`3cad18bf78b30efb54e853077a6cf8feef09c7925c6704d3840e71438611b998`.
The final bounded `tt-smi -ls --local` after that run again reported devices
0 through 3 as available and resettable.

AutoFix's source hypothesis that a hard-coded transfer count would block this
shape was refuted by hardware.  In the non-assert build the real reader loops
over four executed slices; the stale assertion matters only in Watcher or
lightweight-assert builds.  Because the candidate loses performance and is
not selected, that debug-build limitation does not enter production.

A temporary probe then added the target's two-link fused candidate (SHA
`8f501b918edbf4b6d3cfb7e2b33b4204f453b3d13ca06be4434960df42795665`).
It compiled and passed the eager PCC assertions, then made no progress during
trace replay for over three minutes.  Before terminating only PIDs 1509224 and
1509225, the required `tt-triage` report and summary were captured under
`triage/`.  Detailed NoC reads were limited by an installed tt-exalens binding
overload mismatch, while the summary's ARC, Ethernet, binary, Watcher-ring,
and broken-component checks passed.  Recovery used bounded
`tt-smi -ls --local`, `tt-smi -r`, and a second list; all four p300c devices
were visible, followed by `MESH_SMOKE_OK` from a `MeshShape(1,4)` open/close.
The exact hashes and recovery record are in
`results/fused_agmm_links2_trace_hang.json`.  The durable probe was restored
to its safe, hash-coherent one-link form.  The selected two-link decoder does
not call the experimental fused operation.

The final independent `$stage-review` then reread the complete remediated tree
and returned `clean-pass` with no required work.  Its anomaly ledger,
controlled concerns, and explicit later-stage gaps are recorded in
`stage_review.md`.

## Handoff

- Stage implementation/evidence commit: `b2666fe1505` (`Add Falcon3 TP4
  multichip decoder`).  The commit used `SKIP=check-large-files` only for the
  accepted 1.7-MiB raw Tracy ops CSV referenced by `profile_provenance.json`;
  every other pre-commit hook passed.
- Initial review-log commit: `069f3fc3fc2` (`Record Falcon3 TP4 stage review`).
- Completion-audit/evidence commit: `cdbf998b3d6` (`Complete Falcon3 TP4
  multichip audit`).
- The completion-audit checkpoint uses `SKIP=check-large-files` only for the
  4.9-MiB LLM-readable `tt-triage` dump from the rejected two-link fused trace
  hang.  Its SHA is bound in the compact JSON/summary; all other hooks pass.
- Implementation SHA256:
  `b249847705594bbf49e795eecdc1669a0016929929952d0767c9939cef1dd573`.
- Test SHA256:
  `b6b10e32580ff5e4f9fdb465099272727a147049d5f191e91bd0dcae91ded557`.
- `doc/context_contract.json` now advertises the fully executed 32,768-token
  batch-1 contract rather than the obsolete 128-token single-chip staging
  limit.
- No full-model, generation, or vLLM files were created or modified.
