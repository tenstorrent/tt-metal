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

The fresh final `$stage-review` returned `clean-pass` with no required work.
Its controlled anomaly ledger and residual risks are recorded in
`stage_review.md`.

## Handoff

- Stage implementation/evidence commit: `b2666fe1505` (`Add Falcon3 TP4
  multichip decoder`).  The commit used `SKIP=check-large-files` only for the
  accepted 1.7-MiB raw Tracy ops CSV referenced by `profile_provenance.json`;
  every other pre-commit hook passed.
- Implementation SHA256:
  `b249847705594bbf49e795eecdc1669a0016929929952d0767c9939cef1dd573`.
- Test SHA256:
  `b6b10e32580ff5e4f9fdb465099272727a147049d5f191e91bd0dcae91ded557`.
- `doc/context_contract.json` now advertises the fully executed 32,768-token
  batch-1 contract rather than the obsolete 128-token single-chip staging
  limit.
- No full-model, generation, or vLLM files were created or modified.
