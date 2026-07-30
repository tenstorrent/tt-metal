# Optimized Multichip Decoder Work Log

Date: 2026-06-15

Scope: `$optimize` for the existing
`meta-llama/Llama-3.1-8B-Instruct` multichip decoder only. No full-model or
vLLM work was started. Target path is `MultiChipDecoder` on
`TARGET_MESH_SHAPE = (1, 8)` with `ttnn.FabricConfig.FABRIC_1D_RING`.

## Code Changes

- `tt/multichip_decoder.py`
  - Added env-selectable CCL tuning constants while keeping common defaults.
  - Added decode persistent all-gather output buffers keyed by `(mode, stage)`.
  - Added persistent MLP reduce-scatter intermediate buffers.
  - Switched `MultiChipDecoderPolicy` default to BFP8 activations and BFP4
    attention weights.
- `tests/test_multichip_decoder.py`
  - Added env-driven dtype/fidelity overrides for policy sweeps.
  - Added policy metrics to test logging.
  - Updated the contract test for the optimized multichip policy.
  - Added gated `MULTICHIP_DECODER_PROFILE_ONLY=1` profile-only multichip test
    for Tracy collection without opening the single-chip baseline mesh.

## Starting Baseline

Command:

```bash
MULTICHIP_DECODER_DECODE_REPLAYS=4 python_env/bin/pytest \
  --confcutdir=models/autoports/meta_llama_llama_3_1_8b_instruct/tests \
  models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_multichip_decoder.py::test_multichip_decoder_synthetic_paged_prefill_decode_trace_against_optimized \
  -q -s
```

Starting policy:
`llama31_8b_t3k_1x8_tp8_bfp8_attn_bfp4_mlp_decode_v1`.

- prefill PCC: `0.9999957546448677`
- decode PCC: `0.9999870318423426`
- single-chip warmed prefill: `3.5132989287376404 ms`
- single-chip traced decode: `0.798 / 0.802 ms` min/avg class
- multichip warmed prefill: `2.8265262953937054 ms`
- multichip traced decode: `0.4422650672495365 / 0.4453344736248255 ms`
- trace determinism/eager-vs-trace PCC: `1.0`
- fallback audit: `multichip_prefill_decode_clean`

## Optimization Trials

Buffer reuse:

- Prefill+decode all-gather persistent output buffers passed correctness but
  regressed prefill to about `5.49 ms`; prefill persistence rejected.
- Decode-only all-gather persistence accepted:
  prefill `2.821057103574276 ms`, decode
  `0.42658718302845955 / 0.4314172547310591 ms`, PCC unchanged.
- MLP reduce-scatter persistent intermediate buffer accepted:
  prefill `2.8937840834259987 ms`, decode
  `0.41330093517899513 / 0.41467417031526566 ms`, PCC unchanged.

CCL tuning used the multichip path and passed correctness. Defaults retained:
`chunks_per_sync=10`, `num_workers_per_link=2`,
`num_buffers_per_channel=2`.

| Trial | Prefill ms | Decode min ms | Decode avg ms | Decision |
| --- | ---: | ---: | ---: | --- |
| default | 2.893784083 | 0.413300935 | 0.414674170 | kept |
| workers/link=1 | 2.984880935 | 0.414488837 | 0.417536125 | rejected |
| workers/link=4 | 4.225681070 | 0.432888046 | 0.436189352 | rejected |
| chunks/sync=5 | 2.905189991 | 0.414498150 | 0.416087219 | rejected |
| chunks/sync=20 | 2.999338787 | 0.412575901 | 0.415038550 | rejected |
| buffers/channel=1 | 2.856312785 | 0.415639021 | 0.417203410 | rejected |
| buffers/channel=3 | 2.805924974 | 0.416041352 | 0.419544871 | rejected |

Fused matmul-CCL:

- Attention fused all-gather matmul is used.
- MLP fused matmul-reduce-scatter was rejected:
  `Batch fusion is required when input A is sharded`; then
  `Unsupported memory layout TensorMemoryLayout::WIDTH_SHARDED`; then a DRAM
  workaround failed on the second decode with `Tensor is not allocated` in
  `update_output_tensor_topologies<MatmulReduceScatterAsyncResult>`.

Precision/fidelity:

| Trial | Prefill PCC | Decode PCC | Prefill ms | Decode min ms | Decode avg ms | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| MLP HiFi2 | 0.9999957546 | 0.9999870318 | 2.981651109 | 0.434308778 | 0.436845236 | rejected |
| BFP8 activation | 0.9999642794 | 0.9999606333 | 2.882111818 | 0.401320867 | 0.409660744 | accepted candidate |
| BFP4 attention | 0.9997025853 | 0.9999302106 | 2.891752869 | 0.411103014 | 0.416283263 | accepted candidate |
| BF16 KV cache | 0.9999878017 | 0.9999861436 | 2.911374439 | 0.414206181 | 0.417795847 | rejected |
| BFP8 activation + BFP4 attention | 0.9996764336 | 0.9999044394 | 2.907341346 | 0.396286603 | 0.401859987 | accepted |
| Attention LoFi with BFP4 attention | 0.9996764336 | 0.9999044394 | 2.908406314 | 0.398709904 | 0.399877201 | rejected |

The final default path is
`llama31_8b_t3k_1x8_tp8_bfp4_attn_bfp4_mlp_bfp8_act_decode_v2`.

## Final Validation

Synthetic stress:

```bash
python_env/bin/python -m py_compile \
  models/autoports/meta_llama_llama_3_1_8b_instruct/tt/multichip_decoder.py \
  models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_multichip_decoder.py
MULTICHIP_DECODER_DECODE_REPLAYS=16 python_env/bin/pytest --timeout=1200 \
  --confcutdir=models/autoports/meta_llama_llama_3_1_8b_instruct/tests \
  models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_multichip_decoder.py \
  -q -s \
  2>&1 | tee models/autoports/meta_llama_llama_3_1_8b_instruct/doc/optimized_multichip_decoder/runs/final_full_test_16_replays_after_attention_trial_reverted.log
```

Result: `3 passed, 1 skipped in 18.39s`.

- prefill PCC: `0.9996764336199454`
- decode PCC: `0.9999044394414383`
- single-chip prefill: `3.317510709166527 ms`
- single-chip decode min/avg: `0.7905862294137478 / 0.7937005429994315 ms`
- multichip prefill: `2.8145541436970234 ms`
- multichip decode min/avg: `0.3974558785557747 / 0.40162313962355256 ms`
- trace determinism/eager-vs-trace PCC: `1.0`
- fallback audit: `multichip_prefill_decode_clean`

Real HF layer-0 check:

```bash
python_env/bin/python - <<'PY' 2>&1 | tee \
  models/autoports/meta_llama_llama_3_1_8b_instruct/doc/optimized_multichip_decoder/runs/final_real_weight_layer0_multichip_check.log
# Imports _real_state_dict(), _run_optimized_baseline(), and _run_multichip_case().
PY
```

- prefill PCC: `0.999867004626159`
- decode PCC: `0.999931400571509`
- single-chip prefill: `3.4264307469129562 ms`
- single-chip decode min/avg: `0.7903729565441608 / 0.7942511001601815 ms`
- multichip prefill: `3.080476075410843 ms`
- multichip decode min/avg: `0.39528682827949524 / 0.39838056545704603 ms`
- trace determinism/eager-vs-trace PCC: `1.0`
- fallback audit: `multichip_prefill_decode_clean`

## Profiler And Advice

Profiler-only multichip command:

```bash
MULTICHIP_DECODER_PROFILE_ONLY=1 MULTICHIP_DECODER_DECODE_REPLAYS=4 \
python_env/bin/python -m tracy -r -p -v \
  -o models/autoports/meta_llama_llama_3_1_8b_instruct/doc/optimized_multichip_decoder/tracy/synthetic_profile_only/.logs \
  -m pytest \
  --confcutdir=models/autoports/meta_llama_llama_3_1_8b_instruct/tests \
  models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_multichip_decoder.py::test_multichip_decoder_synthetic_paged_prefill_decode_trace_profile_only \
  -q -s \
  2>&1 | tee models/autoports/meta_llama_llama_3_1_8b_instruct/doc/optimized_multichip_decoder/tracy/synthetic_profile_only/tracy_run.log
```

Result: `1 passed in 167.67s`.

Stable tt-perf-report outputs:

- `tracy/synthetic_profile_only/multichip_prefill_perf_report.txt`
- `tracy/synthetic_profile_only/multichip_prefill_perf_report.csv`
- `tracy/synthetic_profile_only/multichip_decode_perf_report.txt`
- `tracy/synthetic_profile_only/multichip_decode_perf_report.csv`
- `tracy/synthetic_profile_only/multichip_decode_perf_report_per_device.txt`
- `tracy/synthetic_profile_only/multichip_decode_perf_report_per_device.csv`

Summary:

- Prefill report: 29 device ops, `1017.620 us` device op time.
- Decode report: 26 merged device ops, `370.881 us` device op time.
- Same profiler run traced decode e2e: `0.445081852 ms` min,
  `0.448176987 ms` avg.
- Minimal roofline estimate: about `0.047 ms/token` for final layer
  weight+KV reads on the 8-chip Wormhole mesh.

Advice outcomes:

- High op-to-op gap: final decode already uses traced replay; gap is larger
  than measured replay wall time and is recorded as a tt-perf-report
  signpost/multidevice trace artifact.
- Attention LoFi for BFP4 weights: tried and rejected with
  `runs/attention_lofi_trial.log`.
- Prefill input in L1: rejected for current `Attention1D`/MLP prefill
  contract; prefill is intentionally DRAM interleaved and changing it would
  require additional sharded prefill contracts and resharding.
- Output subblock advice: applicable prefill configs already use subblock
  width >= 2; no remaining exposed knob for DRAM-sharded decode matmuls.
- HiFi4/BF16 full-accuracy advice: rejected for performance goal; PCC already
  exceeds threshold.

## Watcher And Hardware

Watcher command:

```bash
TT_METAL_WATCHER=10 TT_METAL_WATCHER_NOINLINE=1 \
TT_METAL_LOGS_PATH=models/autoports/meta_llama_llama_3_1_8b_instruct/doc/optimized_multichip_decoder/watcher/synthetic_ring_final_after_recovery \
MULTICHIP_DECODER_PROFILE_ONLY=1 MULTICHIP_DECODER_DECODE_REPLAYS=1 \
python_env/bin/pytest --timeout=1200 \
  --confcutdir=models/autoports/meta_llama_llama_3_1_8b_instruct/tests \
  models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_multichip_decoder.py::test_multichip_decoder_synthetic_paged_prefill_decode_trace_profile_only \
  -q -s \
  2>&1 | tee models/autoports/meta_llama_llama_3_1_8b_instruct/doc/optimized_multichip_decoder/watcher/synthetic_ring_final_after_recovery/watcher_run.log
```

Result: `1 passed in 327.44s`. A grep scan for watcher failure signatures
under `watcher/synthetic_ring_final_after_recovery` returned no matches.

Final hardware status after the real-weight check:
`hardware_status_final_after_real_weight_check.log` lists chips `0-7`.

## Layout Contract

Inter-layer decoder boundary:
`[1, 1, batch_or_seq, 4096]` replicated across the `1x8` mesh.

Full-model bringup must preserve this boundary directly between adjacent
layers. No gather, reshard, or all-reduce should be inserted between decoder
layers. The remaining collectives are intra-layer attention AG+WO, attention
hidden all-gather, MLP reduce-scatter, and MLP hidden all-gather.

## Artifacts

- `runs/final_full_test_16_replays_after_attention_trial_reverted.log`
- `runs/final_real_weight_layer0_multichip_check.log`
- `runs/attention_lofi_trial.log`
- `tracy/synthetic_profile_only/tracy_run.log`
- `tracy/synthetic_profile_only/multichip_profile_only_ops_perf_results.csv`
- `tracy/synthetic_profile_only/multichip_prefill_perf_report.txt`
- `tracy/synthetic_profile_only/multichip_decode_perf_report.txt`
- `tracy/synthetic_profile_only/multichip_decode_perf_report_per_device.txt`
- `watcher/synthetic_ring_final_after_recovery/watcher_run.log`
- `watcher/synthetic_ring_final_after_recovery/generated/watcher/watcher.log`
- `hardware_status_recovered.log`
- `hardware_status_after_profiler_profile_only.log`
- `hardware_status_after_watcher.log`
- `hardware_status_final_after_real_weight_check.log`
- `perf_summary.json`

## Checklist

- Functional checks: passed.
- Synthetic and real-weight PCC: passed.
- Paged KV cache and trace replay: passed.
- Runtime fallback audit: clean.
- Stress/repeated-run coverage: 16 traced decode replays.
- Warmed prefill/decode before/after: reported.
- tt-perf-report tables/CSVs/provenance: present.
- Watcher: clean.
- Fully traced decode path: yes.
- Decode activation sharding and DRAM-sharded decode matmuls: present.
- Prefill DRAM interleaved with 2D matmul configs: present.
- Composite SDPA/Attention1D path: used.
- Explicit memory/program/compute configs: present.
- Fused matmul-CCL: attention kept, MLP rejected with evidence.
- Precision/fidelity reductions: tried and final checked with real weights.
- MoE and LM head/sampling: not applicable to this decoder-only dense model.
