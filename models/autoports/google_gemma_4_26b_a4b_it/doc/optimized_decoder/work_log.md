# Optimized decoder work log

Date: 2026-07-29 UTC

Base checkout: `ed9dff651d7`  
Functional checkpoint: `b46b2396bd2`

Scope is restricted to `tt/optimized_decoder.py`, its optimized tests, and
this directory. No multichip, full-model, or vLLM work was started.

## Hardware and safety

- Host `qb2-120-p04t04`; four visible Blackhole P300C boards; stage mesh 1x1.
- `timeout 60 /home/mvasiljevic/.ttsmi-venv/bin/tt-smi -ls --local` passed
  before work, after the capacity probes, and after the watcher run.
- Mesh open/close smoke passed; compute-with-storage grid 11x10, DRAM grid 8x1.
- Profiler and watcher were separate processes. No reset or recovery was
  required.

## Operation-topology audit

This audit was written before candidate implementation.

| Region | Functional topology / issue | Candidates | Final action and evidence |
| --- | --- | --- | --- |
| attention projections | already-packed QKV, head-layout conversions, separate output projection | per-group dtype/fidelity; interleaved vs DRAM-sharded weights | retain packed QKV; BF16 with HiFi4 sliding / HiFi2 full required because BFP8 decode PCC was 0.994564; DRAM-sharded family measured but rejected by batch-32 tail correctness |
| SDPA/cache | paged update + paged SDPA decode; composite SDPA prefill; BF16 caller-owned cache | explicit configs, lower cache dtype | retain composite ops and correctness-derived configs; cache dtype/layout unchanged to preserve capacity/caller contract; profiler SDPA 712 us |
| dense MLP | repeated same-input gate/up, GELU, multiply, down; interleaved round trips | packed gate/up; BFP8/BFP4; DRAM-sharded decode; coherent L1 chain | BFP8/HiFi2 separate projections kept; packed+DRAM PCC failed; coherent sharded elementwise chain caused one batch-32 row PCC 0.7785; interleaved final |
| router | FP32 router matmul then on-device top-k/softmax/scatter | lower precision or fused scaling | retain FP32 sensitivity boundary; profiler marks 79 us `SLOW`, but it is small versus expert work and all routing remains device-side |
| routed experts | three sparse matmuls per user; conservative block width 1; exact top-8 | BFP8/BFP4, LoFi/HiFi2, block widths 1..22, L1 intermediates | BFP8/LoFi, block 11 kept; outputs/GeGLU/reduction remain L1; exact `nnz=8` preserved |
| prefill experts | canonical all-expert sparse path; optimized wrapper controls physical chunking | BF16/BFP8/BFP4; grouped chunk 32/64/128; legal sparse grids | BFP8 direct upload plus chunk 32 / `per_core_n=2` / gate block 44 / down block 11 kept; BFP4 failed non-aligned PCC (length 33: 0.994481); grouped chunks above 32 failed complete default boundary PCC and remain experimental; advertised capacity passed |
| prefill dense projections | large DRAM-interleaved matmuls | BFP8/BFP4, explicit large-grid configs | BFP8 dense MLP kept; profiler already selects 88 cores, block 8, subblock 4x2 and about 90% FLOPs, so no manual large-grid override beat the selected family |
| crossings/layout | no functional host fallback; conversions at op contracts | remove/chain conversions | static hot-path audit passes; attempted coherent dense chain was rejected by exact batch-32 correctness; remaining conversions serve explicit TTNN contracts |

## Candidate ledger

Times are whole-layer warmed host milliseconds at sequence/current position
1024. Each row was exercised at batch 1 and 32 unless identified as a
correctness-only precision localization. Final headline values come from a
fresh default run.

| Candidate | b1 decode | b32 decode | Correctness / decision |
| --- | ---: | ---: | --- |
| functional BF16 | 3.038 sliding | 68.969 sliding | baseline, PCC >=0.995 |
| BFP8 experts, block 1 | 3.021 | 69.099 | correct, geometry baseline |
| BFP8, gate/down block 2/2 | 2.446 | 50.325 | correct |
| BFP8, block 4/2 | 2.189 | 42.503 | correct |
| BFP8, block 8/2 | 2.078 | 38.931 | correct |
| BFP8, block 11/2 | 2.055 | 38.218 | correct |
| BFP8, block 11/11 | 2.038 | 37.432 | best robust sparse geometry |
| BFP8, block 22/22 | 2.041 | 37.816 | slower; rejected |
| BFP4/LoFi, block 11/11 | 1.861 | 32.056 | aggregate cases passed, but serving-batch per-user tail was weaker; rejected |
| BFP4/LoFi, block 22/22 | 1.889 | 32.546 | slower; rejected |
| BFP4/LoFi, block 8/11 | 2.320 | 32.450 | slower; rejected |
| BFP4 + dense DRAM-sharded | 1.775 | 31.949 | initially fastest, but exact sliding batch-32 HF PCC 0.989764 and user 15 cosine 0.778515; rejected |
| packed gate/up only | 1.817 | 32.005 | faster, but failed the real-weight PCC gate; rejected |
| packed + DRAM-sharded | 1.769 | 31.924 | decode PCC 0.99403 sliding / 0.99334 full; rejected |
| coherent L1 dense chain | 1.788 | 31.979 | same batch-32 user cliff; rejected |
| **final BFP8 block 11, interleaved dense** | **1.883** | **32.202** | aggregate PCC 0.999538, min-user tail 0.998238, identity row mapping; kept |

Sparse cross-product details: gate/up K is 88 tiles and down K is 22 tiles;
block candidates were legal divisors. The helper selected two sparse compute
cores for the 704-wide expert output and 88 cores for the 2816-wide down
output; `per_core_M=1` for both public batches because each user is serialized
as a one-row, one-tile activation. Output subblock is 1x1, an exact limitation
of the current sparse helper. L1 outputs and elementwise/reduction
intermediates were retained.

Precision/fidelity localization:

- global BFP8 initially failed decode at about 0.9935;
- BF16 attention + BFP8 dense + BFP8 experts passed;
- BFP8 attention failed decode at 0.994564, so attention stays BF16;
- dense BFP4 HiFi2/LoFi prefill PCC was 0.978582/0.975828;
- BFP4 attention LoFi PCC was 0.938627;
- BFP4 experts improved latency, but direct batch-32 independent inputs exposed
  the tail noted above;
- expert HiFi2 did not repair that tail; disabling dense DRAM sharding did;
- prefill experts remain BFP8 because BFP4 failed non-aligned length 33.

## AutoFix report

Starting evidence was the exact final-policy failure:

```text
...test_optimized_traced_decode_batch_contract -k batch32
sliding aggregate PCC 0.9897638802
eager/replay 1.0; repeat/replay 1.0; best-user identity
user 15 cosine 0.7785149217
```

The mandated fresh AutoDebug Codex runner could not create its bubblewrap
namespace; its Claude backend produced no report and was bounded/terminated.
An xhigh fresh fork completed read-only source/artifact diagnosis.

| Hypothesis | Focused A/B | Result / verdict |
| --- | --- | --- |
| BFP4 expert weight quantization | `GEMMA4_OPT_EXPERT_WEIGHT_DTYPE=bfp8` | aggregate improved 0.989764 -> 0.991640, row 15 remained; contributing, not root cause |
| LoFi expert accumulation | gate/general fidelity HiFi2 | unchanged 0.989764; refuted |
| dense DRAM-sharded decode family | disable family with other policy fixed | passed at 0.997314 with BFP4; verified root boundary |
| robust cumulative policy | BFP8 experts + interleaved dense | batch-32 sliding aggregate 0.999214, full 0.999836, repeat/eager exact; fixed |

The optimized wrapper now records per-user PCC and best-user mapping and
asserts a 0.99 tail guard in addition to the unchanged aggregate 0.995
functional contract. Full attention repeats one valid decode row in the
underlying functional oracle, so identity matching is asserted only for the
independent sliding rows.

### Grouped-prefill boundary repair

The clean shipped default later exposed a second correctness failure: chunk
128 passed physical length 32 but failed as soon as one sparse invocation
contained two or more tile groups. Sliding/full minima were
`0.888242`/`0.900904`; changing legacy versus fast block geometry did not
repair it. The same failing source with only chunk size changed to 32 passed
all 20 cases at `0.995340`/`0.998048` minima. The fresh source-only AutoDebug
review independently identified the same contract and ruled out tail geometry.

AutoFix changed both public construction defaults from 128 to `TILE_SIZE`
(32) and updated the defaults assertion. It retained the faster proven
`per_core_n=2`, gate block 44, and down block 11 geometry. Clean-default
boundary, full-suite, performance, context/capacity, profiler, and watcher
evidence were then regenerated against decoder SHA256
`803f0e19451926ce7f5529a05498aeadee5cc186c4e0cb408d53e0de8cef9e7e`.
See `AUTODEBUG.md` and `AUTOFIX_GROUPED_PREFILL.md`.

## Final correctness and capability evidence

| Gate | Result | Artifact |
| --- | --- | --- |
| material override identity / hot-path audit | pass | optimized test |
| real-weight sliding shared cache | prefill 0.998631, decode 0.999636 | `pcc_layer0_sliding_attention_shared1.json` |
| real-weight full natural/shared | prefill 0.997686, decode 0.999836 | `pcc_layer5_full_attention_shared{0,1}.json` |
| logical boundary lengths | all 20 cases >=0.995 | `prefill_boundaries_*.json` |
| batch-2 prefill | both layer kinds pass | `prefill_batch2_*.json` |
| traced batch 1/32 | both kinds pass; eager/repeat deterministic | `trace_*_batch*.json` |
| mutable stable buffers | both kinds pass at batch 32 | `trace_mutable_buffers_*.json` |
| advertised-position decode | current position 262143, both kinds pass | `advertised_context_decode_*.json` |
| non-aligned physical prefill | 262143 tokens, finite last token, both kinds pass | `prefill_capacity_*_262143.json` |
| watcher correctness | 7/7 cases, watcher log has no error/assert | command below |

The optimized policy changes only transient/weight precision, not the BF16 KV
cache, page table, cache allocation, or activation live-set topology.
`context_contract.json` therefore remains at 262144 with no reduction.

## Final performance

| Phase / kind | Batch | Functional ms | Final ms | Delta |
| --- | ---: | ---: | ---: | ---: |
| prefill sliding | 1 | 680.955 | 120.697 | -82.27% |
| prefill full | 1 | 681.880 | 121.500 | -82.18% |
| traced decode sliding | 1 | 3.019 | 1.883 | -37.63% |
| traced decode full | 1 | 3.201 | 2.051 | -35.93% |
| prefill sliding | 32 | 21780.254 | 3856.548 | -82.29% |
| prefill full | 32 | 21818.995 | 3884.307 | -82.20% |
| traced decode sliding | 32 | 68.879 | 32.202 | -53.25% |
| traced decode full | 32 | 68.646 | 31.994 | -53.39% |

## Profiler evidence

Final profiler command (separate process from watcher, reduced to one sliding
layer at batch 1, with mid-run device dumps to avoid buffer loss):

```bash
TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback": true}' \
GEMMA4_FUNCTIONAL_DECODER_PERF=1 GEMMA4_RANGE_DOWNLOAD=1 \
python -m tracy -r -p -v --check-exit-code --dump-device-data-mid-run \
--op-support-count=5000 \
-o models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_decoder/final_profile_tracy_midrun \
-m pytest -q \
models/autoports/google_gemma_4_26b_a4b_it/tests/test_optimized_decoder.py::test_optimized_decoder_perf_profile \
-k 'batch1 and sliding_attention'
```

The test and modern Blackhole-aware enrichment passed. The earlier
`final_profile_tracy` attempt passed its test but dropped device markers during
enrichment because the default profiler buffers filled. That attempt is
documented here but not used for conclusions; its duplicated raw logs were
moved to the desktop trash after the compact successful reports were verified.

`final_profile_tracy_midrun/ops_perf_results.csv` feeds the advice-enabled
`prefill_perf_report.*` and `decode_perf_report.*`:

- prefill: 527 ops, 119.993 ms device work + 0.482 ms gaps; sparse expert
  matmuls 99.627 ms (83.0%); QKV 485 us at 80.1% modeled FLOPs; SDPA 717 us;
  dense BFP8 gate/up 75 us each;
- decode: 74 ops, 1.822 ms device work + 0.072 ms gaps; sparse expert
  matmuls 742 us (40.7%); QKV 114 us at 80.4% modeled DRAM; SDPA decode
  43 us; dense BFP8 gate/up/down 52/51/46 us.

For a batch-1 sliding token at position 1024, the selected weights plus
sliding KV reads are approximately 145 MB. At the report's inferred 512 GB/s
peak, the memory-only roof is about 0.28 ms. The 1.822 ms device work is about
15% of that modeled bandwidth, consistent with the report's 15.2% whole-window
DRAM figure. Host trace replay (1.883 ms) and the device window (1.893 ms
including gaps) agree within measurement noise; there is no material
host/dispatch gap left.

## Commands

Representative final gates:

```bash
TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback": true}' \
GEMMA4_RANGE_DOWNLOAD=1 pytest -q \
models/autoports/google_gemma_4_26b_a4b_it/tests/test_optimized_decoder.py

GEMMA4_FUNCTIONAL_DECODER_PERF=1 GEMMA4_RANGE_DOWNLOAD=1 \
GEMMA4_OPTIMIZED_PERF_BASELINE=1 \
TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback": true}' pytest -q \
models/autoports/google_gemma_4_26b_a4b_it/tests/test_optimized_decoder.py::test_optimized_decoder_perf_profile

GEMMA4_FUNCTIONAL_DECODER_PERF=1 GEMMA4_RANGE_DOWNLOAD=1 \
GEMMA4_OPT_CANDIDATE_ID=final_fixed_default_perf \
TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback": true}' pytest -q \
models/autoports/google_gemma_4_26b_a4b_it/tests/test_optimized_decoder.py::test_optimized_decoder_perf_profile

GEMMA4_OPTIMIZED_PREFILL_BATCH32_PERF=1 GEMMA4_RANGE_DOWNLOAD=1 \
GEMMA4_OPTIMIZED_PREFILL_BASELINE=1 \
TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback": true}' pytest -q \
models/autoports/google_gemma_4_26b_a4b_it/tests/test_optimized_decoder.py::test_optimized_prefill_batch32_perf

GEMMA4_OPTIMIZED_PREFILL_BATCH32_PERF=1 GEMMA4_RANGE_DOWNLOAD=1 \
GEMMA4_OPT_CANDIDATE_ID=final_fixed_default_prefill_batch32 \
TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback": true}' pytest -q \
models/autoports/google_gemma_4_26b_a4b_it/tests/test_optimized_decoder.py::test_optimized_prefill_batch32_perf

GEMMA4_PREFILL_CAPACITY_LENGTH=262143 GEMMA4_RANGE_DOWNLOAD=1 \
TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback": true}' pytest -q \
models/autoports/google_gemma_4_26b_a4b_it/tests/test_optimized_decoder.py::test_optimized_prefill_capacity_probe

TT_METAL_WATCHER=10 GEMMA4_RANGE_DOWNLOAD=1 \
TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback": true}' pytest -q \
models/autoports/google_gemma_4_26b_a4b_it/tests/test_optimized_decoder.py \
-k 'real_weights_prefill_decode or (traced_decode_batch_contract and batch32) or trace_mutable_stable_buffers'
```

Watcher result: `7 passed, 19 deselected` in 111.35 s. The log contains no
watcher error, kernel assert, NoC timeout, or hang, and the post-run `tt-smi`
list passed.

## Optimize checklist

- [x] Started from a passing real-weight functional decoder and checkpoint.
- [x] Wrote the operation-topology audit before implementation.
- [x] Preserved prefill/decode, paged cache, trace, determinism, layer kinds,
  batch 1/2/32, non-aligned lengths, and the 262144 context contract.
- [x] Used named per-group precision/fidelity rather than a global policy.
- [x] Swept BF16/BFP8/BFP4 and LoFi/HiFi2 at real weight/shape boundaries.
- [x] Crossed dominant sparse precision with legal block geometries.
- [x] Measured legal DRAM-sharded decode matmuls at batch 1 and 32; rejected
  the fastest candidate on exact serving-batch correctness, not an API error.
- [x] Measured packed same-input gate/up and coherent lower-movement variants.
- [x] Retained paged SDPA/composite ops and explicit memory/program/compute
  configs where material and safe.
- [x] Profiled warmed prefill and traced decode with Tracy/`tt-perf-report`; attacked every
  material row within the stage file contract and recorded blockers.
- [x] Reported final default warmed prefill and traced decode before/after at
  batch 1 and 32; batch-1 primary wins and batch 32 does not regress.
- [x] Audited the measured hot methods for Torch/from/to/fallback crossings.
- [x] Ran stress/repeat replay, mutable buffers, physical capacity, watcher,
  and post-run health checks separately from profiler collection.
- [ ] Independent `$stage-review` clean pass.
- [ ] Local stage checkpoint commit (never push).

## Limitations

- Prefill's all-expert formulation is intrinsically expensive and remains the
  dominant row. Grouped sparse invocations above 32 tokens are numerically
  invalid in the current exact Gemma configuration; the first divergent
  gate/up-versus-down primitive remains a lower-level TTNN follow-up.
- Dense DRAM-sharded decode weights are legal at batch 1 and fast, but the
  current program family is not robust for all independent batch-32 inputs.
  It remains an opt-in diagnostic candidate, never the final default.
- The first Tracy attempt filled profiler buffers and dropped a device marker;
  the final mid-run-dump retry enriched successfully and is the sole source of
  profiler conclusions.
- MPI repeatedly warned about `/dev/shm` headroom even though reported
  available bytes exceeded the 16 MiB request. Repeated correctness/perf runs,
  watcher, and post-run health stayed clean; no reset was needed.
- No multichip, full-model, text generation, or serving claims are made.
