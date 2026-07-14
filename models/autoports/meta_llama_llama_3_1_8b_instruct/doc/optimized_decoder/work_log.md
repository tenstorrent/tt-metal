# Optimized Decoder Work Log

## Inputs

- Model: `meta-llama/Llama-3.1-8B-Instruct`
- Autoport: `models/autoports/meta_llama_llama_3_1_8b_instruct`
- Start commit: `edb87f7d8dcb7953eaae659ce98aa86b674d21d8`
- Scope: optimized decoder only; no multichip, full-model, or vLLM work.

## Implementation

- Added `tt/optimized_decoder.py`.
- Added `tests/test_optimized_decoder.py`.
- Added optimized decoder docs and artifacts under `doc/optimized_decoder`.

Key code decisions:

- Default policy uses BFP4_B projection/MLP weights with LoFi math.
- BF16 is preserved for activations, norm weights, and KV cache.
- Decode QKV/O/gate/up/down matmuls use `MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig`.
- Decode hidden-size residual adds and post-attention RMSNorm use L1 width-sharded layout from the advisor.
- Prefill uses `MatmulMultiCoreReuseMultiCast1DProgramConfig` where logical M is one tile or smaller.
- Decode SDPA keeps TTNN composite op and an explicit `SDPAMultiCoreProgramConfig`.

## Commands

Correctness:

```bash
pytest -q models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_optimized_decoder.py --tb=short -s \
  | tee models/autoports/meta_llama_llama_3_1_8b_instruct/doc/optimized_decoder/test_logs/full_optimized_decoder_after_l1.log
```

Clean timing:

```bash
python - <<'PY' 2>&1 | tee models/autoports/meta_llama_llama_3_1_8b_instruct/doc/optimized_decoder/test_logs/warmed_timing_after_l1.log
# inline timing script; see log for measured values
PY
```

Shard advisor:

```bash
export TTMLIR_ADVISOR_HOME=/localdev/mvasiljevic/tt-mlir
source .agents/skills/shard-advise/scripts/bootstrap.sh
export PYTHONPATH=/localdev/mvasiljevic/tt-metal/ttnn:/localdev/mvasiljevic/tt-metal:${PYTHONPATH:-}
ttnn-advise capture \
  models/autoports/meta_llama_llama_3_1_8b_instruct/doc/optimized_decoder/shard_advise/advise_llama31_8b.py:decode \
  --out /tmp/llama31_8b_shard_advise_dense_bfp4_1783987533
```

Advisor L1 candidate probe:

```bash
python models/autoports/meta_llama_llama_3_1_8b_instruct/doc/optimized_decoder/probes/l1_chain_candidate.py \
  2>&1 | tee models/autoports/meta_llama_llama_3_1_8b_instruct/doc/optimized_decoder/test_logs/l1_chain_candidate_final.log
```

Profiler:

```bash
python -m tracy -r -p -v -o models/autoports/meta_llama_llama_3_1_8b_instruct/doc/optimized_decoder/tracy/prefill/tracy \
  -m pytest -q models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_optimized_decoder.py::test_optimized_signposted_prefill_perf_smoke --tb=short -s

python -m tracy -r -p -v -o models/autoports/meta_llama_llama_3_1_8b_instruct/doc/optimized_decoder/tracy/decode/tracy \
  -m pytest -q models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_optimized_decoder.py::test_optimized_real_weight_decode_trace_replay_if_weights_available --tb=short -s

tt-perf-report models/autoports/meta_llama_llama_3_1_8b_instruct/doc/optimized_decoder/tracy/prefill/prefill_ops.csv \
  --start-signpost PERF_PREFILL --end-signpost PERF_PREFILL \
  --csv models/autoports/meta_llama_llama_3_1_8b_instruct/doc/optimized_decoder/tracy/prefill/prefill_perf_report.csv

tt-perf-report models/autoports/meta_llama_llama_3_1_8b_instruct/doc/optimized_decoder/tracy/decode/decode_ops.csv \
  --start-signpost PERF_DECODE --end-signpost PERF_DECODE \
  --csv models/autoports/meta_llama_llama_3_1_8b_instruct/doc/optimized_decoder/tracy/decode/decode_perf_report.csv
```

Watcher:

```bash
TT_METAL_WATCHER=10 pytest -q \
  models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_optimized_decoder.py::test_optimized_real_weight_single_layer_prefill_matches_hf_if_weights_available \
  models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_optimized_decoder.py::test_optimized_real_weight_decode_trace_replay_if_weights_available \
  --tb=short -s \
  | tee models/autoports/meta_llama_llama_3_1_8b_instruct/doc/optimized_decoder/test_logs/watcher_prefill_decode_after_l1.log
```

## Results

Correctness:

- Full optimized decoder suite: 13 passed in 148.05s.
- Watcher-clean real prefill/decode: 2 passed in 21.67s.
- Real prefill PCC: 0.9999944127056503.
- Real traced decode PCC: 0.9999949180661959.
- Real paged cache update PCC: key 0.9917272537490261, value 0.9932514722249101.

Performance:

- Functional prefill best: 108.031553 ms.
- Optimized prefill best: 26.645185 ms.
- Functional traced decode best: 106.335832 ms.
- Optimized traced decode best: 5.252113 ms.

Profiler conclusions:

- Prefill report shows BFP4/LoFi dense matmuls as the dominant device work; total device time is 26,259 us.
- Decode report shows DRAM-sharded BFP4/LoFi matmuls at about 62 us QKV, 49-50 us output projection, 139-140 us
  gate/up, and 134-135 us down projection.
- Decode report shows the applied L1 width-sharded residual/norm path as 64-core interleaved-to-sharded,
  sharded layernorm around 59-61 us, and sharded-to-interleaved rows.
- Decode report shows SDPA decode at 119 us and paged cache update ops at 26/27 us.
- Both reports lost test signposts in the CSV, so `tt-perf-report` used the whole captured test region.

## Shard Advisor Decisions

Artifacts:

- `shard_advise/report.json`
- `shard_advise/final_ir.mlir`
- `shard_advise/report.txt`

The raw advisor `pipeline.log` was 5.6 MB and was intentionally not committed because it exceeds the repository
large-file hook. The hard-gate artifacts, `report.json` and `final_ir.mlir`, are retained.

Result summary:

- `total_ops`: 13.
- `final_choices`: 12.
- `spill.total_spills`: 0.

Applied:

- Used the no-spill result as a capacity sanity check.
- Kept dense matmul focus as the optimization priority.
- Kept prefill dense matmuls in DRAM/interleaved with explicit program config.
- Applied the advisor L1 width-sharded recommendation to the hidden-size decode residual adds and post-attention RMSNorm.

Rejected with evidence:

- Full advisor L1 width-sharded intermediate MLP activation chain: rejected after two adapted attempts. The logical-height
  attempt failed with `Shard height 32 must match physical height 1024`; the physical-height adaptation reached sharded
  `silu` and failed with L1 OOM: 29,360,128 B allocation, 458,752 B per bank, only 59,616 B free. The reduced residual
  candidate preserved PCC and ran at 5.298410 ms, so it was applied to production.
- Advisor DRAM/interleaved decode matmul layout: rejected for decode because DRAM-sharded decode matmuls are present in
  `tracy/decode/decode_perf_report.txt` and are the measured fast path.
- Further gate/up packing: not applied in this stage because the TTNN surface here has no local packed MLP projection
  helper and the separate DRAM-sharded gate/up matmuls are 139/140 us in the decode profiler.

## Checklist

- [x] `tt/optimized_decoder.py` exists.
- [x] Tests exercise `OptimizedDecoder`, not a functional fallback.
- [x] Non-aligned logical prefill seq_len is accepted.
- [x] Paged KV-cache update behavior is covered.
- [x] Layer-kind coverage includes layers 0 and 31.
- [x] Decode trace replay and determinism are covered.
- [x] Real-weight prefill and decode correctness are covered.
- [x] Warmed prefill and traced warmed decode before/after are reported.
- [x] Final traced decode beats the functional traced-decode baseline.
- [x] Shard advisor run produced required `report.json` and `final_ir.mlir`.
- [x] Watcher-clean optimized correctness run exists.
- [x] Tracy and `tt-perf-report` artifacts exist for prefill and decode.
- [x] Context contract preserved without reduction.
- [x] No host fallback in optimized runtime methods by source guard.

## Open Limitations

- `tt-smi -ls --local` failed with `ModuleNotFoundError: No module named 'tt_smi'`; device use was validated through TTNN
  open/close, watcher, profiler, and tests.
- Shard advisor capture excludes SDPA/cache because the advisor tracer failed on `scaled_dot_product_attention_decode`
  with `TracedTensor` arguments.
- The profiler CSVs did not retain Tracy signposts.
