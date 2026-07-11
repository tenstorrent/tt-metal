# Gemma 4 31B optimized decoder (Stage 03)

This stage delivers the single-P150 `OptimizedDecoder` path. It preserves the
fused decoder's paged-cache, layer-kind, determinism, and logical-sequence
contracts while replacing decode projection matmuls with explicit
DRAM-width-sharded weights and L1-width-sharded activations. It does not begin
multichip, full-model, generator, or vLLM work.

## Result

The selected decode policy is BFP8/LoFi attention weights, BFP4/LoFi MLP
weights, BF16 residual/norm activations, and BFP8 paged KV cache. Prefill keeps
the fused stage's faster large-M 2D/HiFi2 programs. Decode uses eight
logical width shards over the eight Blackhole DRAM banks. The cache change
halves persistent KV bytes without reducing the advertised 262,144-token
logical context; `../context_contract.json` records the dtype and evidence.

| Layer kind | Mode | Fused baseline (ms) | Optimized warmed (ms) | Speedup | Final device window |
|---|---:|---:|---:|---:|---:|
| sliding, layer 0 | prefill-128 | 3.519 | 2.672 | 1.32x | 2.587 ms |
| full, layer 5 | prefill-128 | 4.326 | 3.395 | 1.27x | 3.300 ms |
| sliding, layer 0 | traced decode | 2.606 | 1.186 | 2.20x | 1.149 ms |
| full, layer 5 | traced decode | 2.949 | 1.332 | 2.21x | 1.297 ms |

Warmed values are medians from eight prefill calls or twelve traced decode
replays. Fused and optimized numbers use the same benchmark function; only
`GEMMA4_OPT_BENCH_IMPLEMENTATION` changes. Device values sum the matching
signposted `tt-perf-report` window.
The four final raw CSVs, advice-backed tables, filtered CSVs, and Tracy logs are
under `tracy/{sliding,full}/{prefill,decode}/final_bound/`.

## Correctness

All values below use real Gemma weights and the HF decoder reference; the
functional acceptance threshold is PCC 0.995.

| Logical length / mode | Sliding PCC | Full PCC |
|---|---:|---:|
| prefill 32 | 0.999219 | 0.999072 |
| prefill 33 (non-aligned) | 0.998574 | 0.998885 |
| prefill 128 | 0.998632 | 0.998786 |
| traced decode after prefill-32 | 0.998668 | 0.998983 |

The small delta from the fused BF16-cache result is explained by BFP8
attention weights and BFP8 cache quantization. Determinism is bit-exact across
eight trace replays. The suite also covers batch-2/32 prefill, batch-32 decode,
mutable trace buffers, sliding-window wrap, 1025/1057 non-aligned lengths,
long-context wrappers, and optimized-path type/source assertions.

The optimized BFP8 cache was also exercised at the inherited context limits.
The capacity harness deliberately uses bounded references: sliding prefill is
checked against the corresponding HF absolute-position window, full prefill
uses a bounded 2,049-token HF prefix, and its decode field is TT prefill/decode
self-consistency. These rows prove allocation, paging, modulo ownership, and
non-aligned execution; they are not labeled as full-history HF decode PCC.

| Logical length / mode | Sliding PCC | Full PCC |
|---|---:|---:|
| capacity prefill 262144 (bounded oracle) | 0.998922 | 0.998695 |
| capacity TT prefill/decode consistency | 0.998991 | 0.995178 |
| capacity prefill 262113 non-aligned (bounded oracle) | 0.998796 | 0.998695 |
| capacity TT prefill/decode consistency, non-aligned | 0.999042 | 0.998321 |

A separate periodic-history oracle compares a distinct late-position token to
HF at absolute position 262,143 after 262,143 populated history tokens. It
passes at PCC **0.997758 sliding / 0.998387 full**; deliberately using the
wrong position increases RMSE in both cases. Evidence is
`evidence/context_262144_distinct_hf_oracle.{xml,log}`.

The first combined exact-context command used pytest's default 300-second
per-test timeout. Sliding passed, while full attention timed out normally at
chunk 144384 without a device fault. The board remained healthy; rerunning the
unchanged node with `--timeout=900` completed in 269 seconds. The initial
harness artifact is retained under `evidence/rejected_harness/`; the clean
per-kind XML/logs are the acceptance evidence.

## Final measured topology

The decode report contains one packed QKV matmul, one O projection, separate
gate/up projections with GELU fused into gate, and one down projection. There
is no host fallback and no `torch`, `from_torch`, or `to_torch` in the measured
runtime path. Residual/norm layout conversions are 1–2 us each; the one MLP
up-result DRAM spill is required because both 21,504-wide BF16 outputs plus
the next matmul's static CBs cannot coexist in 1,572,864-byte P150 L1.

| Role | Shape MxKxN | Weight/fidelity | Program and logical shards | Input/output shard | Block / per-core N | Final latency |
|---|---|---|---|---|---|---:|
| packed QKV | 32x5376x16384 | BFP8 LoFi | DRAM-sharded, 8 | 1x21 / 1x64 tiles | 3 / 64 | 192 us |
| O | 32x8192x5376 | BFP8 LoFi | DRAM-sharded, 8 | 1x32 / 1x21 tiles | 8 / 21 | 97 us |
| gate | 32x5376x21504 | BFP4 LoFi | DRAM-sharded, 8 | 1x21 / 1x84 tiles | 7 / 84 | 201 us |
| up | 32x5376x21504 | BFP4 LoFi | DRAM-sharded, 8 | 1x21 / 1x84 tiles | 7 / 84 | 208 us |
| down | 32x21504x5376 | BFP4 LoFi | DRAM-sharded, 8 | 1x84 / 1x21 tiles | 21 / 21 | 196 us |

The reporter counts 12 active worker cores for these matmuls; the program
contract and activation shards use eight logical width shards. Output
subblock fields are not exposed for this program class. QKV/O sustain 88–90%
of modeled DRAM bandwidth; BFP4 MLP rows sustain 54–58% and are the remaining
decode limit.

## Candidate disposition

The selected 1.186 ms sliding traced decode is the best correct candidate.

| Candidate | Evidence | Decision |
|---|---|---|
| all BFP4/LoFi attention+MLP | real-weight traced-decode PCC 0.992205 | reject |
| BFP8 attention + BFP4 MLP, selected geometry | PCC 0.998668/0.998983; 1.186/1.332 ms | keep |
| BFP8 KV cache | final PCC table; cache rows show `BFP8, BF16 => BFP8` | keep |
| split Q/K/V, same dtype/config | PCC 0.998668; 1.211 ms vs packed 1.186 ms | reject; packed wins |
| packed gate/up BF16 output | block 7 OOM; block 3 static/live collision; block 1 PCC 0.998721 but 1.518 ms | reject; separate is faster |
| packed gate/up BFP8 output | block 7 OOM; block 3 static/live collision; block 1 PCC 0.998637 but 1.440 ms | reject; separate is faster |
| prefill DRAM-sharded QKV, M=128 | op only accepts M=1 | adapt, not reject |
| prefill as four legal M=32 DRAM-sharded chunks | PCC 0.998493; 3.356 ms vs 2.672 ms | reject |
| attention HiFi2 | 1.351 ms | reject |
| MLP gate/up HiFi2 | 1.539 ms | reject |
| MLP down HiFi2 | 1.363 ms | reject |
| QKV block 7 / gate-up block 21 / down block 28 | exact L1 overrun or static/live collision | reject |
| four cores, adapted QKV3/gate3/O8/down12 | down static end 638,976 vs live allocation 454,656 | reject |

The complete geometry matrix and exact L1 errors are in `work_log.md`.
`tt-perf-report`'s prefill tracing advice is not applicable to the requested
warmed untraced prefill metric; decode is traced. Its prefill DRAM-sharding
advice was adapted and measured above. Suggested decode HiFi2 and larger block
families were measured and lost or hit exact L1/divisibility limits. After the
packed block-7 OOMs, legal block-3 and block-1 variants were also tried; block
1 ran correctly for both output dtypes but remained 21.6% (BFP8) to 28.1%
(BF16) slower than separate.

MoE active-expert and CCL opportunities are not model-applicable: this decoder
has no MoE block and this stage is single-device. SDPA remains the dedicated
composite op with q/k chunks 32/64 for decode and the fused stage's proven 2D
prefill program. No decoder optimization is deferred to a later stage.

## Performance accounting

At decode position 32, nominal stored projection payloads plus BFP8 K/V reads
(excluding tile/container metadata) are
305,799,168 bytes for sliding attention and 371,724,288 bytes for full
attention. Using the reporter's 512 GB/s P150 bandwidth model gives rooflines
of 0.597 ms and 0.726 ms per layer/token.

| Kind | Roofline | Device window | Warmed end-to-end | Host/dispatch gap |
|---|---:|---:|---:|---:|
| sliding | 0.597 ms | 1.149 ms | 1.186 ms | 0.037 ms |
| full | 0.726 ms | 1.297 ms | 1.332 ms | 0.035 ms |

The remaining device/roofline gap is accounted for by the BFP4 MLP rows at
54-58% modeled DRAM bandwidth, QKV head transforms/SDPA, required norm and GQA
layout crossings, and the evidenced gate/up L1-capacity spill. The small
end-to-end gap is trace replay plus synchronization; the measured path has no
host conversion or fallback.

Repository and history searches found the same-family `models/demos/gemma4`
implementation but no same-model, same-stage, single-P150 optimized-decoder
artifact with comparable shapes and profiler rows. The strongest comparator is
therefore the completed fused stage plus this stage's candidate matrix.

## Reproduction

```bash
export PYTHONPATH=$PWD/build_Release:$PYTHONPATH
export LD_LIBRARY_PATH=$PWD/build_Release/lib:$LD_LIBRARY_PATH
export MPLCONFIGDIR=/tmp/mpl
pytest -q -s models/autoports/google_gemma_4_31b/tests/test_optimized_decoder.py
GEMMA4_OPT_BENCH=1 GEMMA4_OPT_BENCH_IMPLEMENTATION=fused pytest -q -s models/autoports/google_gemma_4_31b/tests/test_optimized_decoder.py -k warmed_latency
GEMMA4_OPT_BENCH=1 pytest -q -s models/autoports/google_gemma_4_31b/tests/test_optimized_decoder.py -k warmed_latency
GEMMA4_LONG_PREFILL=262144 pytest -q -s models/autoports/google_gemma_4_31b/tests/test_optimized_decoder.py -k optimized_long_nonaligned_prefill_capacity --timeout=900
GEMMA4_LONG_PREFILL=262113 pytest -q -s models/autoports/google_gemma_4_31b/tests/test_optimized_decoder.py -k optimized_long_nonaligned_prefill_capacity --timeout=900
GEMMA4_LONG_DECODE=262144 pytest -q -s models/autoports/google_gemma_4_31b/tests/test_optimized_decoder.py -k optimized_exact_context_distinct_traced_decode --timeout=900
TT_METAL_WATCHER=10 TT_METAL_LOGS_PATH=$PWD/models/autoports/google_gemma_4_31b/doc/optimized_decoder/watcher_final pytest -q -s models/autoports/google_gemma_4_31b/tests/test_optimized_decoder.py -k changed_trace_buffers
```

Profiler collection uses `GEMMA4_OPT_PROFILE=1 python -m tracy -r -p -v
--output-folder <artifact> -m pytest <single profile node> -s`, followed by
`tt-perf-report` filtered between `OPT_PERF_{PREFILL,DECODE}` signposts.
