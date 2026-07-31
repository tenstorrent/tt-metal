# Optimized decoder work log

Date: 2026-07-31 UTC. Base checkout: `e5623e55e69`.

## Hardware and safety

- `timeout 60 tt-smi -ls --local`: four Blackhole P300 devices healthy.
- Single-device `MeshShape(1, 1)` open/close: `MESH_SMOKE_OK`.
- Firmware 19.8 is newer than the repo-tested 19.5; all selected-path tests
  nevertheless passed.
- Profiler and watcher were run separately.
- Auditable final watcher run: `watcher_bfp8_lofi_final.log`; nine
  selected-path real-PCC, traced b1/b32, shared-cache, and mutable-input A/B/A
  cases passed with no watcher features disabled.

## Candidate chronology

1. Captured same-harness functional medians for layers 0 and 5, batches 1
   and 32.
2. Audited repeated-input projections, reshards, composite attention, router
   scales, sparse expert topology, and cache movement.
3. Added group-local dtype materialization and a guard that verifies device
   tensor dtypes. This caught an initial `to_memory_config(dtype=...)` trial
   that did not actually convert weights; all results from that trial were
   discarded.
4. Swept real-weight BF8/BF4 attention, dense, and expert groups. BF4 failed
   PCC. BF8 attention failed sliding decode. BF8 dense+expert passed PCC and
   improved prefill, but full decode regressed; it was rejected.
5. The original BF8/LoFi claim was not executable and has been superseded by
   the exact-current-source AutoFix fidelity sweep recorded below.
6. Folded router scale constants. This removed one runtime operation and
   slightly improved the repeated medians without changing PCC.
7. Packed dense gate/up. The first slice call exposed a missing `steps`
   argument; after using the supported API, both real layer kinds passed.
   Five-run medians then established it as the selected batch-1 winner.
8. Reviewer finding (AutoFix): the earlier precision runs did **not** exercise
   DRAM-sharded decode matmul program configs, sparse expert program-config
   alternatives, large-prefill configs, or a BF8 KV cache. They therefore
   reject only their stated dtype/fidelity hypotheses and are not evidence
   against those topology families. `optimization_candidate_matrix()` now
   makes the missing b1/b32 shapes, legal `in0_block_w` divisors, BF4/LoFi
   pairs, cache dtypes, prefill families, and movement families executable
   and reviewable. Static contract command:
   `pytest -q models/autoports/google_gemma_4_26b_a4b_it/tests/test_optimized_decoder.py -k 'candidate_matrix or owns_public or hot_path'`
   (`3 passed`). Hardware results must be attached before any of those
   candidates can be marked rejected.
9. Ran the exact-shape BF8 KV-cache candidate with real layer weights:
   `GEMMA4_RANGE_DOWNLOAD=1 GEMMA4_REAL_LAYER_CACHE=/tmp/gemma4_real_layer_cache GEMMA4_OPTIMIZED_CACHE_SWEEP=1 pytest -q models/autoports/google_gemma_4_26b_a4b_it/tests/test_optimized_decoder.py -k bfp8_kv_cache_candidate -vv`.
   Both representative kinds passed (sliding prefill/decode
   `0.9986195/0.9996786`; full `0.9984834/0.9998561`). Artifacts are under
   `candidates/kv_cache_bfp8/`.
10. Completed the BF8 cache decision with same-source BF16/BF8 controls.
    Real-weight trace replay passed at batches 1 and 32 for both layer kinds,
    including eager/replay and repeat PCC 1.0. The final exact-current-source
    five-run decode medians (decoder SHA `5604d3ec748b`, test SHA
    `a0ad898e5f66`) were BF16→BF8: sliding batch 1
    `2.050574→2.050694 ms` (+0.0059%), sliding batch 32
    `38.979322→38.969811 ms` (-0.0244%), full batch 1
    `2.241326→2.242059 ms` (+0.0327%), and full batch 32
    `38.763732→38.749591 ms` (-0.0365%).
    BF8 tiled storage is 1088 versus 2048 bytes/tile, a 46.875% reduction;
    at context 262144 this would save 25.78125 GiB across all 30 layer cache
    pairs. The changes are noise-scale, but the candidate was rejected because
    it does not beat BF16 for either primary batch-1 target.
    `candidates/kv_cache_bfp8/decision.json` records exact PCC, latency,
    source hashes, commands, layer-count, and byte evidence; the public
    context contract remains unchanged.
11. Corrected a first DRAM-sharded API attempt that passed an interleaved
    activation (runtime `bad optional access`). The adapted candidate used
    the coherent contract from `MLP1D`: packed dense shape
    `[32,2816] x [2816,4224]`, eight-bank DRAM width-sharded BF4 weight,
    eight-core L1 width-sharded activation/output, Blackhole LoFi,
    `per_core_M=1`, `per_core_N=17`, and the largest shard-local divisor
    `in0_block_w=11`. It completed:
    `DRAM_SHARDED_BFP4_LOFI_OK 8 Shape([1,1,32,4224])`.
    This proves the mandatory batch-1 DRAM-sharded family is legal on P300
    (logical batch 1 uses the tested one-tile physical height); it does not
    override the existing real-layer BF4 PCC rejection or constitute a
    traced whole-layer timing.
12. Swept sparse expert `in0_block_w={2,11}` through the full real-weight
    layer harness for both representative kinds. All four cases passed; this
    covers gate, up, and `is_input_a_sparse=True` down while preserving exact
    `nnz=8`. Sliding decode PCC was `0.9996699/0.9996786`; full was
    `0.9998610/0.9998561` for widths `2/11`. Artifacts:
    `candidates/sparse_in0_block_w_{2,11}/`.
13. The real packed prefill shape accepted a large explicit multicore config:
    `[1,1,1024,2816] x [2816,4224]`, grid `6x8`,
    `in0_block_w=11`, output subblock `1x2`, output block `4x22`,
    `per_core_M=4`, `per_core_N=22`; result
    `LARGE_PREFILL_PACKED_OK Shape([1,1,1024,4224])`. This is API/shape
    legality, not whole-layer PCC or latency selection.
14. Executed the full whole-layer frontier for both representative layer
    kinds and decode batches 1 and 32: `8 passed` PCC cases and `16 passed`
    warmed/traced performance cases. Corrected DRAM-sharded integration from
    BF16 four-core L1 overflow through a BFP8 four-core allocation clash to
    an eight-core packed gate/up geometry (logical N 4224 padded to 4352) and
    six-core down geometry (logical N 2816 padded to 3072). That coherent
    candidate passed PCC but regressed whole-layer latency
    (`3.023/3.218 ms` b1 and `69.031/68.777 ms` b32). Large prefill was
    neutral/slower. Sparse width 11 was the clear winner over width 2 and the
    selected incumbent: `2.047/2.239 ms` b1 and `38.950/38.750 ms` b32.
    It is now wired directly into the optimized decoder.
14a. Repaired the candidate-coherence review finding. The old non-sparse
    candidate artifacts predated the selected sparse-width-11 override, so
    their decode timings were not comparable to the later sparse frontier.
    Removed an ineffective monkeypatch of the functional module's sparse
    builder; candidates now select the optimized class attribute directly,
    and every machine-readable non-sparse contract records inherited sparse
    width 11. A same-source rerun (`optimized_decoder` SHA
    `5604d3ec748b`, test SHA `a0ad898e5f`) passed 16/16 timing cases.
    Sparse-11 versus large-prefill now measures 2.051 versus 2.069 ms
    sliding b1, 2.240 versus 2.244 ms full b1, 38.983 versus 38.955 ms
    sliding b32, and 38.738 versus 38.734 ms full b32. Therefore the former
    ~0.94 ms decode anomaly was stale wiring/provenance, not a decode effect
    from the prefill-only branch.
14b. Expanded the DRAM-sharded geometry/precision frontier on that same
    source. BF16 and BFP8/HiFi4 8-core gate-up / 6-core down with width 11
    both pass real-weight PCC. BF16 measures 2.100/2.309 ms b1 and
    39.098/38.846 ms b32; BFP8 measures 2.093/2.296 ms b1 and
    39.034/38.817 ms b32, all slower than the sparse-11 interleaved control.
    BFP8 4/3-core width 22 is exactly blocked by 2,864,128 B/core static
    circular buffers; 2/2-core width 44/33 is blocked by 10,475,264 B/core,
    versus 1,572,864 B/core available. PCC command: `GEMMA4_RANGE_DOWNLOAD=1
    GEMMA4_OPTIMIZED_WHOLE_LAYER_SWEEP=1
    TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback": true}' pytest -q
    models/autoports/google_gemma_4_26b_a4b_it/tests/test_optimized_decoder.py::
    test_optimized_whole_layer_candidate_pcc -k 'dram_sharded_dense_'`.
    Machine-readable blockers are in
    `candidates/whole_layer/candidate_failures.json`.
15. Reran the selected default cumulatively: both real layer PCC cases
    passed, five-sample traced medians passed at b1/b32, and a separate
    watcher run passed nine cases with no disabled features. Final
    `tt-smi -ls --local` showed all four P300 devices healthy.
16. Retained explicit functional SDPA, sparse top-8 expert composite, sharded
   norms, and paged-cache operations because they already represent the
   lower-movement TTNN forms.

## Gates and artifacts

- Whole-layer candidate gates: `8 passed` PCC and `16 passed` perf.
- Final selected watcher gate: `9 passed, 115 deselected`.
- Final suite with advertised-context and long-attention gates enabled:
  `22 passed, 102 opt-in evidence tests skipped`; retained in
  `final_suite.log`.
- Final advertised-context gate: `2 passed` at position 262143.
- Final long-attention gate: `2 passed` above the 32768-token dispatch
  boundary.
- Real PCC JSON: `pcc_layer*.json`.
- Trace/determinism JSON: `trace_*batch*.json`.
- Boundary and batch-2 JSON: `prefill_boundaries_*` and
  `prefill_batch2_*`.
- Repeated optimized medians: `*repeated_timings.json`.
- Same-harness functional medians: `baseline_repeated/`.
- Candidate precision and whole-layer evidence: `candidates/`; the latter
  includes exact commands/hashes/hardware/configs/PCC paths/timings.
- Invalid legacy Tracy timing is documented only as anomaly evidence; its
  large raw scratch directory was removed.
- Valid optimized trace aggregate plus current isolated per-op reports:
  `tracy/device_trace_latency.csv`, `tracy/*_perf_report.csv`, summary CSVs,
  and `tracy/provenance.md`.
- Mutable trace stress: `mutable_trace_aba_*.json`; A1/A2 PCC 1.0 and
  deliberately different A/B outputs for both layer kinds.

## Optimize checklist

- [x] operation-topology audit before selection
- [x] real-weight functional baselines
- [x] batch-1 and serving-batch layout/precision evidence
- [x] precision and compute-fidelity trials with dtype propagation guard
- [x] same-input packed projection selected with PCC and timing evidence
- [x] DRAM-sharded dense decode matmul sweep at b1 and b32; corrected
      eight-/six-core BFP8 candidate passed PCC but lost whole-layer timing
- [x] sparse expert gate/up/down width 2/11 whole-layer sweep; width 11 selected
- [x] large-prefill packed config whole-layer PCC/timing; rejected as neutral/slower
- [x] BF8 KV-cache real-weight correctness, b1/b32 trace, repeated perf, and
      capacity decision; rejected because every decode median regressed
- [x] coherent DRAM-sharded-weight/L1-width-sharded activation-output
      whole-layer movement timing; rejected as slower
- [x] warmed prefill and traced decode before/after at both required batches
- [x] non-aligned logical sequence lengths preserved
- [x] paged-cache semantics, shared physical cache, trace determinism,
      repeated execution, and watcher-clean run
- [x] no host fallback/conversion in optimized hot path
- [x] context contract reviewed; unchanged because cache allocation and dtype
      are unchanged
- [ ] independent stage review and local stage commit (filled at handoff)

## Limitations

The first combined modern Tracy join overflowed its marker buffer. Autofix
split the capture by exact test node and raised `--op-support-count` to
10000; both representative layer captures then joined successfully and
produced authoritative v2.1 Blackhole per-op reports. Invalid legacy
durations remain excluded. The firmware compatibility warning remains an
environment limitation.

## AutoFix reviewer-finding closure

- Hypothesis: optimized timing artifacts actually identify the functional
  implementation. Verified: inherited provenance named the functional test.
  Fix: optimized wrapper now rewrites implementation, command, source/test
  hashes, hardware, commit, and UTC timestamp; repeated JSON includes wrapper
  identity and command.
- Hypothesis: unchanged-input trace replay could hide stale input. Verified
  as an evidence gap, not an observed stale-input bug. Fix: a same-trace
  mutable A/B/A test now passes for sliding and full layers under watcher.
- Hypothesis: invalid legacy profiler data was the only device evidence.
  Verified. Fix: dedicated optimized trace profiling provides valid replay
  times, and isolated enlarged-buffer captures provide valid current per-op
  rows for both layer kinds.
- Hypothesis: the fallback audit ignored inherited methods. Verified. Fix:
  the AST audit now covers all reachable optimized and inherited decoder hot
  methods and rejects `from_torch`/`to_torch` within them.

Stage implementation/evidence commit SHA:
`6222e693ec8d8e83381535bd17f385508a93074d`.

## AutoFix reviewer-finding ledger

Starting evidence: stage-review findings 1 and 5 observed that chronology item
8 and the checked optimize boxes claimed program-config/topology coverage that
the retained artifacts did not prove.

| Hypothesis | Focused experiment | Verdict | Evidence / remaining uncertainty |
|---|---|---|---|
| BF8 cache is numerically viable at the exact model cache boundary | Rank-5-only cache dtype substitution in the normal real-weight layer harness | Verified for PCC and trace, rejected for the primary target | `candidates/kv_cache_bfp8/{pcc,trace,*repeated*,decision}*.json`; b32 improves by noise-scale amounts, but both b1 medians regress by 0.0059–0.0327% despite 46.875% storage reduction |
| DRAM-sharded packed dense can be made whole-layer legal and correct | BF16 and BFP8 eight-/six-core width-11 controls; BFP8 four-/three-core width 22 and two-/two-core width 44/33 | Verified at 8/6, rejected for speed; wider geometries exactly L1-blocked | `candidates/whole_layer/{candidate_failures.json,dram_sharded_dense_*_g8d6_w11/}`; PCC passes at 8/6, b1/b32 whole-layer latency loses |
| Sparse expert config cannot use larger `in0_block_w` | Full real-weight layers and traced timing with widths 2, 11, 22, and 44, covering gate/up/down and exact `nnz=8` | Refuted; width 11 selected | width 11 beats width 2; width 22 is slightly slower at b1 and hits a 1,280-byte L1 clash at b32; width 44 needs 2,319,616 B versus 1,572,864 B L1. See `sparse_extended_geometry.json` |
| A large packed prefill config is shape/API blocked | Real full-layer PCC and warmed timing with explicit 6x8 config | Refuted, rejected for speed | `candidates/whole_layer/large_prefill_multicore/`; PCC passes, prefill is neutral/slower |
| Width-sharded movement cannot satisfy the packed dense contract | Eight-core gate/up and six-core down DRAM weights plus coherent L1 activation/output boundaries | Refuted at whole-layer level, rejected for speed | Correct at b1/b32 for both kinds but slower than selected sparse-11 path |

Only the proven sparse width-11 candidate was selected into the default
runtime. The DRAM-sharded and large-prefill candidates remain reproducible
opt-in paths with machine-readable rejection evidence.

## AutoFix compute-fidelity closure

- Hypothesis: the declared fidelity sweep reached the dominant matmuls.
  Verified false: `_candidate_decoder()` changed only weight dtype, sparse
  expert up/down used framework defaults, and no candidate timing artifact
  carried fidelity.
- Fix: added role-specific executable dense/expert compute configs and wired
  the expert config through all sparse gate/up/down matmuls.
- BF8 HiFi2 and BF8 LoFi both pass real-weight PCC at 0.995 for sliding and
  full-attention layers. Five-run warmed/traced HiFi2→LoFi medians are
  `1.842→1.835 ms` sliding b1, `32.408→32.068 ms` sliding b32,
  `2.035→2.026 ms` full b1, and `32.302→31.925 ms` full b32. Prefill is
  effectively unchanged (`669.749→669.828 ms`, `670.982→671.016 ms`).
- BF4/LoFi is rejected by an exact adapted correctness blocker, not a first
  API error: sliding prefill/decode PCC `0.993716/0.994384`; full
  `0.992966/0.993189`, all below `0.995`.
- Isolated Tracy runtime evidence
  `candidates/expert_bfp8_lofi/tracy_sliding_b1_ops_perf_results.csv` shows
  gate/up/down `SparseMatmulDeviceOperation` rows with BF16 activations,
  BF8_B weights, and `MATH FIDELITY=LoFi`.
- Machine decision and exact commands:
  `candidates/fidelity_decision.json`. Frozen implementation/test SHA256:
  `3589c270b4c8f8b57348479eaf62479ace88702d0775e21d61ff2052133886b3` /
  `3e09e6704bc5977c1235baea61ad4275e4a72c15c28cd6d003509ccfcf001a27`.
- Follow-up selection: promoted expert BF8/LoFi into production because it is
  the best correct candidate at both required batches. Final selected-source
  medians are sliding prefill/b1/b32 `669.805/1.848/32.076 ms` and full
  `671.090/2.039/31.938 ms`. Production PCC is 3/3, trace contract 4/4,
  watcher selected path 9/9 with no disabled watcher features, and static
  contracts 4/4. Evidence:
  `watcher_bfp8_lofi_final.log`, final root PCC/trace/repeated JSON, and
  `candidates/fidelity_decision.json`.
- Context-capacity decision: unchanged. Only immutable expert weight dtype
  changed; KV-cache dtype, shape, paging, and allocation are identical.
