# Llama 3.1 8B Instruct Optimized Multichip Decoder

Status: complete as of 2026-06-15.

This directory records the `$optimize` pass for the
`meta-llama/Llama-3.1-8B-Instruct` repo-local multichip decoder. The measured
path is `tt/multichip_decoder.py` on the target `1x8` T3K Ring mesh with
`ttnn.FabricConfig.FABRIC_1D_RING` and `ttnn.Topology.Ring`. Single-chip and
replicated fallback paths were not used as completion evidence.

## Final Path

Final policy:
`llama31_8b_t3k_1x8_tp8_bfp4_attn_bfp4_mlp_bfp8_act_decode_v2`.

- BFP8 activations, BFP4 attention weights, BFP4 MLP gate/up/down weights,
  BFP8 KV cache, and LoFi MLP math.
- Async CCLs are used for hidden all-gathers and MLP reduce-scatter.
- Attention decode uses fused all-gather plus WO matmul.
- Decode hidden/residual tensors are width-sharded in L1 within a decoder
  layer. Prefill tensors remain DRAM interleaved with explicit 2D matmul
  program configs.
- Decode hidden all-gathers reuse persistent output buffers. MLP
  reduce-scatter reuses a persistent intermediate buffer. Prefill all-gather
  persistence was rejected due measured prefill regression.

## Correctness And Latency

Main final synthetic stress command:

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

| Metric | Before optimize pass | Final default path |
| --- | ---: | ---: |
| Prefill PCC vs optimized decoder | 0.9999957546448677 | 0.9996764336199454 |
| Decode PCC vs optimized decoder trace | 0.9999870318423426 | 0.9999044394414383 |
| Decode trace determinism PCC | 1.0 | 1.0 |
| Decode eager vs trace PCC | 1.0 | 1.0 |
| Single-chip warmed prefill | 3.513298929 ms | 3.317510709 ms |
| Single-chip traced decode min | 0.798 ms class | 0.790586229 ms |
| Single-chip traced decode avg | 0.802 ms class | 0.793700543 ms |
| Multichip warmed prefill | 2.826526295 ms | 2.814554144 ms |
| Multichip traced decode min | 0.442265067 ms | 0.397455879 ms |
| Multichip traced decode avg | 0.445334474 ms | 0.401623140 ms |
| Runtime fallback audit | clean | `multichip_prefill_decode_clean` |

Real HF layer-0 check:

```bash
python_env/bin/python - <<'PY' 2>&1 | tee \
  models/autoports/meta_llama_llama_3_1_8b_instruct/doc/optimized_multichip_decoder/runs/final_real_weight_layer0_multichip_check.log
# Loads local meta-llama/Llama-3.1-8B-Instruct layer-0 weights and runs
# optimized single-chip baseline versus the optimized multichip path.
PY
```

Real-weight result:

| Metric | Value |
| --- | ---: |
| Prefill PCC vs optimized decoder | 0.999867004626159 |
| Decode PCC vs optimized decoder trace | 0.999931400571509 |
| Single-chip warmed prefill | 3.426430747 ms |
| Single-chip traced decode min / avg | 0.790372957 / 0.794251100 ms |
| Multichip warmed prefill | 3.080476075 ms |
| Multichip traced decode min / avg | 0.395286828 / 0.398380565 ms |
| Trace determinism / eager-vs-trace PCC | 1.0 / 1.0 |
| Runtime fallback audit | `multichip_prefill_decode_clean` |

The accepted PCC threshold is `0.995`; the final synthetic and real-weight
checks are above that bar for prefill and decode.

## tt-perf-report

Profiler-only multichip collection command:

```bash
mkdir -p models/autoports/meta_llama_llama_3_1_8b_instruct/doc/optimized_multichip_decoder/tracy/synthetic_profile_only/.logs
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

Profiler-run metrics:

| Metric | Value |
| --- | ---: |
| Profile-only prefill e2e | 4.707457963 ms |
| Profile-only traced decode min | 0.445081852 ms |
| Profile-only traced decode avg | 0.448176987 ms |
| Trace determinism / eager-vs-trace PCC | 1.0 / 1.0 |
| Runtime fallback audit | `multichip_prefill_decode_clean` |

Stable profiler artifacts:

- `tracy/synthetic_profile_only/multichip_profile_only_ops_perf_results.csv`
- `tracy/synthetic_profile_only/multichip_profile_only_profile_log_device.csv`
- `tracy/synthetic_profile_only/tracy_ops_data.csv`
- `tracy/synthetic_profile_only/tracy_ops_times.csv`
- `tracy/synthetic_profile_only/multichip_prefill_perf_report.txt`
- `tracy/synthetic_profile_only/multichip_prefill_perf_report.csv`
- `tracy/synthetic_profile_only/multichip_decode_perf_report.txt`
- `tracy/synthetic_profile_only/multichip_decode_perf_report.csv`
- `tracy/synthetic_profile_only/multichip_decode_perf_report_per_device.txt`
- `tracy/synthetic_profile_only/multichip_decode_perf_report_per_device.csv`

tt-perf-report commands used the raw
`tracy/synthetic_profile_only/.logs/reports/2026_06_15_15_48_45/ops_perf_results_2026_06_15_15_48_45.csv`
with `PERF_MULTICHIP_PREFILL` / `PERF_MULTICHIP_PREFILL_END` and
`PERF_MULTICHIP_DECODE` / `PERF_MULTICHIP_DECODE_END`.

| Window | Device ops | Device op time | Op-to-op gap reported | Host e2e from same profiler run |
| --- | ---: | ---: | ---: | ---: |
| Prefill | 29 | 1017.620 us | 2413.522 us | 4.707458 ms |
| Decode | 26 merged | 370.881 us | 2882.933 us | 0.445082 ms min / 0.448177 ms avg |

The decode high-gap advice is not an untraced loop in the final path: the
signposted decode window wraps a `ttnn.execute_trace(..., blocking=True)`
replay, determinism/eager-vs-trace PCC are both 1.0, and the runtime fallback
audit is clean. The gap reported by tt-perf-report is larger than the measured
signposted replay wall time and is treated as a Tracy/signpost association
artifact for this multidevice trace-replay window.

Performance accounting for one decoder layer, same profiler run:

- Minimal weight+KV roofline estimate: about `0.047 ms/token` on 8 Wormhole
  chips, using the final stored dtypes and the Wormhole DRAM peak implied by
  tt-perf-report's DRAM percentage normalization.
- Device op time: `0.371 ms/token` for the merged decode report.
- End-to-end traced replay: `0.445 ms/token` min, `0.448 ms/token` avg.
- Remaining gap over device op time is about `0.074 ms` on the measured min
  replay and is from trace replay enqueue/synchronization plus many small ops
  and intra-layer CCLs. No host fallback or single-chip fallback is present.

## Advice Review

| tt-perf-report advice or optimization item | Evidence | Decision |
| --- | --- | --- |
| Run decode with tracing to remove high op-to-op gap | Final decode is already captured and replayed with `ttnn.begin_trace_capture`, `ttnn.execute_trace`, and clean determinism/eager PCC. Reported decode gap exceeds the measured replay wall time. | No code change; classified as profiler/signpost artifact for this traced multidevice window. |
| Use LoFi for BFP4 attention matmuls | Temporary attention-LoFi trial passed PCC but measured prefill `2.908406314 ms` and decode min/avg `0.398709904 / 0.399877201 ms`; final default after reverting measured prefill `2.814554144 ms` and decode min/avg `0.397455879 / 0.401623140 ms`, and earlier recovered final measured decode min `0.396366231 ms`. | Rejected; no overall target win and prefill regressed. Artifact: `runs/attention_lofi_trial.log`. |
| Place prefill matmul input 0 in L1 if possible | `Attention1D.prefill_forward` and `_TensorParallelMLP.prefill_forward` are DRAM-interleaved prefill contracts; the surrounding prefill QKV/head/SDPA/cache-fill/concat and MLP residual path would require extra sharded prefill contracts and resharding. The optimize checklist also recommends DRAM-interleaved prefill activations for this large-M path. | Rejected for this decoder-stage contract; changing it would create extra movement and a new residual contract rather than remove movement. |
| Output subblock too small | Final long-prefill MLP configs use `out_subblock_w=2`, attention WO prefill uses `out_subblock_w=4`, and fused decode AG+WO reports `1x2` as good. DRAM-sharded decode matmul configs do not expose an output-subblock knob in the report. | No remaining applicable knob. |
| Use HiFi4/BF16 for full accuracy | PCC is already above threshold; MLP HiFi2 and BF16 KV/activation variants were measured and were slower or did not improve the target. | Rejected for performance goal. |

## CCL And Layout Findings

Async CCL default retained:
`chunks_per_sync=10`, `num_workers_per_link=2`,
`num_buffers_per_channel=2`.

| Trial | Prefill ms | Decode min ms | Decode avg ms | Decision |
| --- | ---: | ---: | ---: | --- |
| default | 2.893784083 | 0.413300935 | 0.414674170 | kept for CCL knobs |
| workers/link=1 | 2.984880935 | 0.414488837 | 0.417536125 | rejected, slower |
| workers/link=4 | 4.225681070 | 0.432888046 | 0.436189352 | rejected, slower |
| chunks/sync=5 | 2.905189991 | 0.414498150 | 0.416087219 | rejected, slower |
| chunks/sync=20 | 2.999338787 | 0.412575901 | 0.415038550 | rejected, prefill worse and avg not better |
| buffers/channel=1 | 2.856312785 | 0.415639021 | 0.417203410 | rejected, decode slower |
| buffers/channel=3 | 2.805924974 | 0.416041352 | 0.419544871 | rejected, decode slower |

Buffer reuse:

- Decode all-gather persistent output buffers: accepted. Decode improved from
  `0.442265067 / 0.445334474 ms` min/avg to
  `0.426587183 / 0.431417255 ms`.
- Prefill all-gather persistent output buffers: rejected. Prefill regressed
  from about `2.83 ms` to about `5.49 ms`.
- MLP reduce-scatter persistent intermediate buffer: accepted. Decode improved
  to `0.413300935 / 0.414674170 ms` min/avg.

Fused matmul-CCL:

- Attention decode uses `ttnn.experimental.all_gather_matmul_async`.
- MLP `ttnn.experimental.matmul_reduce_scatter_async` was tried and rejected:
  without `fuse_batch=True` it fails `Batch fusion is required when input A is
  sharded`; with `fuse_batch=True` it rejects
  `TensorMemoryLayout::WIDTH_SHARDED`; a DRAM workaround reached the op but
  failed on the second decode with `Tensor is not allocated` from
  `update_output_tensor_topologies<MatmulReduceScatterAsyncResult>`.

Precision/fidelity:

| Trial | Prefill PCC | Decode PCC | Prefill ms | Decode min ms | Decode avg ms | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| MLP HiFi2 | 0.9999957546 | 0.9999870318 | 2.981651109 | 0.434308778 | 0.436845236 | rejected, slower |
| BFP8 activation | 0.9999642794 | 0.9999606333 | 2.882111818 | 0.401320867 | 0.409660744 | accepted candidate |
| BFP4 attention | 0.9997025853 | 0.9999302106 | 2.891752869 | 0.411103014 | 0.416283263 | accepted candidate |
| BF16 KV cache | 0.9999878017 | 0.9999861436 | 2.911374439 | 0.414206181 | 0.417795847 | rejected, slower |
| BFP8 activation + BFP4 attention | 0.9996764336 | 0.9999044394 | 2.907341346 | 0.396286603 | 0.401859987 | accepted final |
| Attention LoFi with BFP4 attention | 0.9996764336 | 0.9999044394 | 2.908406314 | 0.398709904 | 0.399877201 | rejected, no overall win |

## Inter-Layer Layout Contract

The decoder layer boundary is replicated full-hidden residual:
`[1, 1, batch_or_seq, 4096]` replicated across the `1x8` mesh.

Full-model bringup should preserve this boundary directly between adjacent
decoder layers. Do not insert an extra gather, reshard, or all-reduce between
layers. Remaining collectives are intra-layer only:

- attention fused all-gather plus WO matmul,
- attention hidden all-gather before the first residual add,
- MLP reduce-scatter after the down projection,
- MLP hidden all-gather before the second residual add.

A hidden-sharded inter-layer boundary was not implemented in this pass because
it would require distributed RMSNorm plus a different residual-add contract.

## Watcher And Hardware Health

Watcher command:

```bash
mkdir -p models/autoports/meta_llama_llama_3_1_8b_instruct/doc/optimized_multichip_decoder/watcher/synthetic_ring_final_after_recovery
TT_METAL_WATCHER=10 TT_METAL_WATCHER_NOINLINE=1 \
TT_METAL_LOGS_PATH=models/autoports/meta_llama_llama_3_1_8b_instruct/doc/optimized_multichip_decoder/watcher/synthetic_ring_final_after_recovery \
MULTICHIP_DECODER_PROFILE_ONLY=1 MULTICHIP_DECODER_DECODE_REPLAYS=1 \
python_env/bin/pytest --timeout=1200 \
  --confcutdir=models/autoports/meta_llama_llama_3_1_8b_instruct/tests \
  models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_multichip_decoder.py::test_multichip_decoder_synthetic_paged_prefill_decode_trace_profile_only \
  -q -s \
  2>&1 | tee models/autoports/meta_llama_llama_3_1_8b_instruct/doc/optimized_multichip_decoder/watcher/synthetic_ring_final_after_recovery/watcher_run.log
```

Result: `1 passed in 327.44s`. A grep scan for watcher failure/error/timeout
signatures under `watcher/synthetic_ring_final_after_recovery` returned no
matches. `hardware_status_final_after_real_weight_check.log` shows chips `0-7`
discoverable after the final real-weight multichip check.

## Checklist

- Functional checks: passed on synthetic stress and real HF layer-0 weights.
- PCC: accepted for prefill and decode on synthetic and real-weight checks.
- Paged KV cache and warmed trace replay: passed.
- Runtime fallback audit: `multichip_prefill_decode_clean`.
- Stress/repeated-run coverage: full test file with 16 traced decode replays.
- Warmed prefill/decode before and after: reported above.
- tt-perf-report: text tables, CSVs, stacked CSVs/PNGs, and provenance logs
  are present under `tracy/synthetic_profile_only`.
- Watcher: clean on the optimized multichip profile-only path.
- Decoder path fully traced: yes for decode replay.
- Decode activations width-sharded in L1: yes inside each layer.
- Prefill activations DRAM interleaved with 2D matmul configs: yes.
- Optimized composite attention ops: `Attention1D`, SDPA, paged KV cache, and
  fused decode AG+WO are used.
- Explicit memory/program/compute configs: yes for important attention, MLP,
  and CCL ops.
- Shard specs/core grids: tile-aligned for Llama 3.1 8B dimensions.
- DRAM-sharded decode matmuls: yes for attention and MLP weight matmuls.
- Fused matmul-CCL: attention fused path kept; MLP fused path rejected with
  exact TTNN/runtime evidence.
- Reduced precision/fidelity: tried with synthetic and verified with real
  HF layer-0 final path.
- LM head/sampling: not applicable to this decoder-only goal.
- MoE: not applicable; Llama 3.1 8B is dense.
