# Llama 3.1 8B optimized decoder

This stage implements and validates the single-device optimized decoder for
`meta-llama/Llama-3.1-8B-Instruct`. It is deliberately limited to
`tt/optimized_decoder.py`, its tests, and this documentation. It does not begin
multichip decoder, full-model, generator, or vLLM work.

## Result

The optimized runtime is independent of the functional forward path. It keeps
the decode residual stream width-sharded in L1, uses packed QKV and dedicated
head/RoPE/SDPA/cache operations, fuses SiLU into the gated multiply, and uses
the shard-advisor-seeded 1-D matmul chain. All projection weights are BFP4,
projection matmuls use LoFi, activations and norms remain BF16, and the final
gate/up output subblock is `1x5`.

Layer 16 is representative: all 32 Hugging Face decoder layers use the same
dense Llama decoder kind and tensor geometry.

Real-weight PCC against Hugging Face, batch 32, prefill sequence 18, decode at
position 18:

| Check | Functional | Optimized |
| --- | ---: | ---: |
| prefill output | 0.999986 | 0.999805 |
| decode output | 0.999988 | 0.999862 |
| decode key append | at functional bar | 0.992893 |
| decode value append | at functional bar | 0.993318 |
| BFP8-cache decode output | not selected | 0.999862 |

The functional acceptance bar is PCC 0.99. The material PCC delta is the
expected result of selecting real-weight-validated BFP4/LoFi projection math;
it remains well above that bar. A batch-13, sequence-7 non-aligned run produced
PCC 0.999740 prefill and 0.999828 decode identically across three runs.
Five real-weight trace replays each produced decode PCC 0.999861 and exact
replay-to-replay equality. Refreshing the persistent traced input changed the
output and matched a second Hugging Face decode at PCC 0.999864; both original
and refreshed trace cache appends remained above PCC 0.993.

Warmed wall latency on one P300 Blackhole device, batch 32, sequence 18, ten
prefill iterations and 100 trace replays:

| Path | KV cache | Prefill (ms) | Traced decode (ms) |
| --- | --- | ---: | ---: |
| functional baseline | BF16 | 37.656 | 36.948 |
| final optimized | BF16 | 3.318 | 0.750 |
| final optimized | BFP8 | 3.346 | 0.713 |

The same-cache comparison is an 11.4x prefill and 49.2x traced-decode speedup.
BFP8 cache is an additional supported caller-owned policy and improves final
decode by 5.0% without changing cache shape or advertised context capacity.

At batch 1 with BF16 cache, the same 10/100 harness measured 1.461 ms
prefill and 1.369 ms traced decode for the functional path versus 1.263 ms
and 0.583 ms for the optimized path. The batch-32 result remains the primary
compiler-captured contract measurement; both ends of the supported batch
range are covered.

The advisor's complete residual/norm chain was also implemented and exercised
as the `advisor_exact_chain` candidate. It preserved real-weight PCC
(0.999805 prefill, 0.999855 decode), but measured 3.335 ms prefill and
0.749 ms BFP8-cache traced decode. The selected mixed chain is therefore kept:
it applies the advisor's projection grids, program configs, and projection
output shards while retaining the faster 32-way norm/residual layout.

The literal final file reproduced 3.345963 ms prefill and 0.712697 ms traced
decode under the 10/100 harness. The final advice-backed profile reports
0.697 ms of device time and 0.034 ms
of inter-op gap inside the traced decode window, consistent with the 0.713 ms
unprofiled end-to-end result. The BFP4 projection weights contain 109.052 MB;
including the position-18 BFP8 KV reads gives an approximately 0.215 ms
single-device 512 GB/s lower bound. Thus final device time is about 3.2x the
weight-plus-cache roofline, and the report's measured aggregate DRAM rate is
156 GB/s (30.6% of peak). The remaining cost is the five dense projection
matmuls plus required attention, cache, reshape, norm, and residual operations;
there is no unexplained host/dispatch gap in traced decode.

## Contract

- Input and output remain `[1, batch, seq_len, 4096]`, with batch in `[1, 32]`.
- Prefill accepts non-aligned logical sequence lengths; it does not expose a
  `seq_len % chunk == 0` restriction.
- Decode remains single-token and updates the same
  `[batch, 8, max_cache_len, 128]` paged-cache contract.
- Both BF16 and BFP8 cache tensors are accepted. Decode updates remain BF16 and
  are converted by the cache operation as required.
- Setup performs the only Torch/host tensor conversions. The measured forward
  paths contain no `torch`, `from_torch`, `to_torch`, or functional fallback.
- `doc/context_contract.json` remains at the previously advertised sequence-18
  validated capability. This stage did not reduce it or force a cache dtype.

## Evidence and artifacts

- The full operation-topology audit, candidate tables, advisor decisions,
  profiler conclusions, commands, failures, and checklist are in
  `work_log.md`.
- Mandatory shard-advisor outputs are
  `shard_advise/report.json` and `shard_advise/final_ir.mlir`. The capture
  adapter is preserved as `shard_advise/advise_llama.py`.
- Final Tracy and advice-backed reports are under `tracy/layer16/`:
  `decode_ops.csv.gz`, `decode_perf_report.{csv,txt}`, and
  `prefill_perf_report.{csv,txt}`.

## Reproduction

Set the real-weight shard once:

```bash
export LLAMA_31_8B_REAL_WEIGHT_FILE=/path/to/model-00002-of-00004.safetensors
```

Run optimized correctness and deterministic trace coverage:

```bash
pytest -q -s \
  models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_optimized_decoder.py
```

Run the final warmed benchmark:

```bash
RUN_OPTIMIZED_DECODER_PERF=1 \
OPTIMIZED_DECODER_PERF_VARIANT=optimized \
OPTIMIZED_DECODER_CACHE_DTYPE=bfp8 \
OPTIMIZED_DECODER_PREFILL_REPEATS=10 \
OPTIMIZED_DECODER_TRACE_REPLAYS=100 \
pytest -q -s \
  models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_optimized_decoder.py \
  -k optimized_decoder_perf
```

Run the watcher-clean real-weight and stress checks separately from profiling:

```bash
TT_METAL_WATCHER=10 pytest -q -s \
  models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_optimized_decoder.py \
  -k 'real_weight_prefill_decode_and_cache_contract or non_aligned_sequence_batch_and_repeated_run_determinism'
```

## Known limitations

- The functional stage's device-capacity evidence still limits validated
  prefill to sequence 18 and cache coverage to 128; long-context enablement is
  downstream work.
- Prefill cache fill is a per-user `ttnn.fill_cache` API loop. The profile
  attributes about 0.298 ms to dispatch gaps across those batch-32 updates;
  TTNN exposes a scalar `batch_idx` for the contiguous-cache operation. This
  does not affect traced decode.
- Two tile-layout reshape-view operations are required at the public decode
  input/output boundary and account for 0.089 ms of device time. Removing them
  would change the established public layout contract or the decode-head
  operator's internal batch placement.
- This is intentionally single-device code. No collectives or multichip
  layouts are introduced here.
