# Optimized Decoder Work Log

Model: `Qwen/Qwen3-4B`

Stage base SHA: `07038677adc2222232fbb7db1f1c4c075e66cc65`

Skills used: `$optimize`, `$tt-device-usage`, `$stage-review`. `$autofix` is reserved for review findings or failed gates.

## Starting Point

The functional stage delivered `FunctionalDecoder.prefill_forward` only. `FunctionalDecoder.decode_forward` raises `NotImplementedError("decode path pending emitted-decode forge version")`. Functional evidence from `doc/context_contract.json`:

| Functional path | PCC |
| --- | ---: |
| Synthetic prefill seq 16 | `0.9996239896537468` |
| Synthetic prefill seq 64 | `0.9996795472223063` |
| Real-weight prefill seq 16 | `0.9999917295183249` |

Because functional decode did not exist, traced decode comparison is against legal optimized candidates and the final tuned candidate, not against a functional traced-decode implementation.

## Operation Topology Audit

Initial measured optimized topology was collected with Tracy and `tt-perf-report` around warmed prefill and traced decode windows. The final BFP4/LoFi plus L1-MLP-input plus DRAM-sharded-down capture is in `tracy/optimized_ops_final.csv`.

| Opportunity | Measured topology | Action | Evidence |
| --- | --- | --- | --- |
| Repeated same-input Q/K/V matmuls | QKV share the same hidden-state input. | Packed QKV weights. Prefill keeps `[V, Q, K]` split order for functional parity; decode uses `[Q, K, V]` for decode slicing. | Final traced decode has one `MatmulDeviceOperation 32 x 2560 x 6144` for QKV. |
| Same-input gate/up MLP matmuls | Gate and up share post-attention LN input. | Tried packed gate/up and split gate/up with post-norm L1. Kept separate gate/up because split+L1 beat packed variants. | `layout_geometry_trials.csv` and `layout_geometry_trials_round2.csv`; packed DRAM `0.8952450007200241 ms`, packed+L1 `0.9504840709269047 ms`, split+L1 `0.8917949162423611 ms` in the same round. |
| Avoidable host fallback | Functional setup uses host conversion; measured forward paths must not. | Kept conversion in setup helpers only; forward functions operate on TTNN tensors. Added static source test. | `test_optimized_runtime_has_no_host_fallback` passes; reports show 0 host ops. |
| Paged KV behavior | Functional stage had no decode cache. | Added paged BF16 K/V caches, page table helper, paged fill during prefill, and `paged_update_cache` during decode. | Prefix 16/17 decode tests and batch-2 disjoint-page test pass. |
| Decode SDPA | Manual attention would add extra data movement. | Used `ttnn.transformer.paged_scaled_dot_product_attention_decode`. | Final report has `SdpaDecodeDeviceOperation` at 11 us. |
| KV-cache precision | Final decode still uses paged BF16 cache, and reduced-cache dtype can affect attention/cache performance and capacity. | Tried BFLOAT8_B paged K/V cache with real weights, prefix 16/17, paged prefill, eager decode, trace replay, and explicit trace release. Kept BF16 because BFLOAT8_B did not beat the primary prefix-16 traced-decode path. | `kv_cache_dtype_trials.csv`: BF16 prefix16 min `0.501827 ms`, BFLOAT8_B prefix16 min `0.503337 ms`; BFLOAT8_B prefix17 min `0.501557 ms` but primary-prefix loss and final default min `0.501138 ms` keep BF16. All rows passed real-weight PCC and trace PCC. |
| Prefill SDPA | Functional prefill used TTNN SDPA. | Kept TTNN SDPA path with packed projections and no public alignment restriction. | Seq 16/17/64 prefill tests pass. |
| Decode sharding | Static one-core decode head layout broke batch > 1 cache updates. | Switched decode head memory config to batch-sized height sharding; added a width-sharded decode down-projection path. | Batch-2 disjoint page rows PCC `0.9834099823858623`; final down row is `DRAM Sharded=True`. |
| DRAM-sharded/L1 decode matmuls | Reports advise placing activation input 0 in L1 for QKV, gate/up, and down, and decode matmuls are DRAM-bound. | Accepted for gate/up by moving post-norm to L1. Then swept DRAM-sharded projection families and accepted down-only DRAM-sharded matmul. Rejected QKV, output, gate/up, gate/up/down, and all-projection DRAM-sharded families because they were slower than down-only. Removed a redundant `synchronize_device` after blocking trace replay. | `dram_sharded_trials.csv`: down-only `0.7435758598148823 ms`, all-projection `0.7583661936223507 ms`, gate/up/down `0.8032652549445629 ms`, gate/up `0.8467547595500946 ms`, QKV `0.8881948888301849 ms`, output `0.9088451042771339 ms`. Final default repeated min `0.501138 ms`; final report down row is `MatmulDeviceOperation 32 x 9728 x 2560`, `DRAM Sharded=True`, 12 report cores, BFP4/LoFi, `46.540 us`. |
| Large prefill program configs | Matmuls are device-time dominant in prefill. | Used TTNN default multi-core matmul configs with BFP4 attention/MLP and final post-norm L1 move; explicit down 2D larger-subblock config was invalid for the output block/grid shape. Decode-only DRAM-sharded down is not used in prefill, preserving broad prefill compatibility. Focused prefill L1 advice trials were tried, but repeated production attempts did not beat the final default path. | Final prefill report: 29 device ops, `696.664 us` summed device time, 0 host ops; invalid config error recorded in `layout_geometry_trials.csv`; focused advice evidence in `prefill_advice_trials.csv`; production repeated evidence in `production_prefill_l1_trials.csv`. |
| Runtime data movement | Slices, reshapes, transposes, interleaved-to-sharded ops are present around attention; final MLP adds one copy to L1 and one sharded/interleaved boundary around the DRAM-sharded down matmul. | Kept required layout conversions for TTNN decode composite ops and measured-profitable MLP movement; moved no measured conversion to host. Removed the redundant host sync after blocking trace replay and measured only the blocking trace replay body. | Traced decode report: 38 device ops, 0 host ops, `472.281 us` summed device time; final repeated traced decode min `0.501138 ms`. |
| MoE active expert execution | Qwen3-4B decoder layer is dense, not MoE. | Not applicable. | HF config/layer kind has standard attention and MLP only. |

## Precision and Fidelity Sweep

Command used:

```bash
python - <<'PY'
# Inline sweep over attention dtype, MLP dtype, and MLP math fidelity.
# Writes models/autoports/qwen_qwen3_4b/doc/optimized_decoder/precision_trials.csv.
PY
```

Rows:

| Candidate | Attention dtype | MLP dtype | Fidelity | PCC | Traced decode |
| --- | --- | --- | --- | ---: | ---: |
| `final_bfp8_hifi2` | BFP8 | BFP8 | HiFi2 | `0.9994021823670951` | `0.9246352128684521 ms` |
| `mlp_bfp8_lofi` | BFP8 | BFP8 | LoFi | `0.9993930603604769` | `0.9375647641718388 ms` |
| `mlp_bfp4_lofi` | BFP8 | BFP4 | LoFi | `0.999324241979847` | `0.8973558433353901 ms` |
| `attention_bfp4_mlp_bfp8_hifi2` | BFP4 | BFP8 | HiFi2 | `0.9993794545292246` | `0.9084348566830158 ms` |

Decision after the first precision sweep: keep BFP8 attention and BFP4/LoFi MLP. Stage-review required a final-topology attention precision signoff, and that later changed the selected attention policy to BFP4/LoFi.

## Final-Topology Attention Sweep

Command used:

```bash
python - <<'PY'
# Inline real-weight sweep on the final DRAM-sharded-down topology.
# Writes attention_final_topology_trials.csv.
PY
```

Rows:

| Candidate | PCC | Trace PCC | Traced decode |
| --- | ---: | ---: | ---: |
| `final_bfp8_attention_control` | `0.999104342904898` | `1.0` | `0.7515158504247665 ms` |
| `final_bfp4_attention_hifi2` | `0.9990924749345804` | `1.0` | `0.7871557027101517 ms` |
| `final_bfp4_attention_lofi_projection_probe` | `0.9990712074052157` | `1.0` | `0.7159649394452572 ms` |

Decision: keep BFP4 attention weights with LoFi compute. The final default also uses BFP4/LoFi MLP and removes a redundant post-`execute_trace(blocking=True)` sync; repeated final default timing min is `0.501138 ms`, beating the best correct candidate row.

## Layout, Sharding, and Packed Gate-Up Sweep

Commands used:

```bash
python - <<'PY'
# Inline real-weight sweep over packed gate/up, L1 activation placement,
# QKV input L1 placement, and an explicit down out_subblock_w=2 config.
# Writes layout_geometry_trials.csv and layout_geometry_trials_round2.csv.
PY
```

Rows:

| Candidate | PCC | Traced decode | Status |
| --- | ---: | ---: | --- |
| `baseline_split_dram_auto` | `0.999324241979847` | `0.908376183360815 ms`, `0.941774807870388 ms` | Replaced. |
| `packed_gate_up_dram_auto` | `0.999324241979847` | `0.8966038003563881 ms`, `0.8952450007200241 ms` | Legal, slower than selected split+L1. |
| `mlp_post_norm_l1_outputs_dram` / `split_post_norm_l1` | `0.999324241979847` | `0.8782949298620224 ms`, `0.8917949162423611 ms` | Selected. |
| `mlp_all_l1_inputs_outputs` | `0.999324241979847` | `0.9324047714471817 ms` | Rejected, slower. |
| `packed_gate_up_post_norm_l1` | `0.999324241979847` | `0.9504840709269047 ms` | Rejected, slower. |
| `qkv_input_l1_split_dram_mlp` | `0.999324241979847` | `0.9217248298227787 ms` | Rejected, slower. |
| `qkv_input_l1_split_post_norm_l1` | `0.999324241979847` | `0.9344853460788727 ms` | Rejected, slower. |
| `qkv_input_l1_packed_post_norm_l1` | `0.999324241979847` | `0.9236251935362816 ms` | Rejected, slower. |
| `down_out_subblock_2_explicit_2d` | n/a | n/a | TTNN validation error: `Num output blocks along x (40) must be smaller than or equal to the number of columns in compute grid (8)`. |

Decision: implement split gate/up with post-norm moved to L1. Packed/fused gate-up was legal and measured, but it was not kept because a well-tuned separate gate/up candidate was faster.

## DRAM-Sharded Decode Projection Sweep

Command used:

```bash
python - <<'PY'
# Inline real-weight sweep over DRAM-sharded decode QKV, output, gate/up,
# down, gate/up/down, and all-projection families.
# Writes dram_sharded_trials.csv.
PY
```

Rows:

| Candidate | PCC | Traced decode | Status |
| --- | ---: | ---: | --- |
| `baseline_final_l1_split` | `0.999324241979847` | `0.8990047499537468 ms` | Replaced. |
| `dram_qkv_only` | `0.9993251468406109` | `0.8881948888301849 ms` | Legal, slower than selected down-only. |
| `dram_o_only` | `0.9993220400294286` | `0.9088451042771339 ms` | Legal, slower. |
| `dram_gate_up_only` | `0.9993092802492918` | `0.8467547595500946 ms` | Legal, slower. |
| `dram_down_only` | `0.9993205880857916` | `0.7435758598148823 ms` | Selected. |
| `dram_gate_up_down` | `0.9993205880857916` | `0.8032652549445629 ms` | Legal, slower. |
| `dram_all_decode_projections` | `0.999312186669155` | `0.7583661936223507 ms` | Legal, slower than down-only. |

Decision: keep DRAM-sharded down-only for decode. The final default code reproduces the selected strategy and improves the sweep row with repeated host timing min `0.501138 ms`.

## MLP BFP4/LoFi Geometry Signoff

The first `$stage-review` returned `more-work-needed` because the final BFP4/LoFi MLP geometry signoff mixed evidence from older precision policies. I reran material gate/up/down geometry candidates under the selected BFP4/LoFi policy and wrote `mlp_geometry_trials.csv`.

| Candidate | PCC | Traced decode | Decision |
| --- | ---: | ---: | --- |
| `control_default_programs` | `0.999104342904898` | `0.7306160405278206 ms` | Control. |
| `gate_up_99c_i4_sub4` | `0.9764238582674671` | `0.6418656557798386 ms` | Rejected: faster but below real-weight 0.99 PCC. |
| `gate_up_99c_i8_sub4` | `0.9757735251608588` | `0.6322269327938557 ms` | Rejected: faster but below real-weight 0.99 PCC. |
| `gate_up_80c_i4_sub4` | `0.9764238582674671` | `0.6493963301181793 ms` | Rejected: faster but below real-weight 0.99 PCC. |
| `gate_up_64c_i5_sub5` | `0.9782249973361682` | `0.6240569055080414 ms` | Rejected: faster but below real-weight 0.99 PCC. |
| `gate_fused_silu_99c_i4_sub4` | `0.9764238582674671` | `0.6467062048614025 ms` | Rejected: faster but below real-weight 0.99 PCC. |
| `down_4core_i76` | n/a | n/a | Rejected: L1 CB allocation `1756416 B` exceeds max L1 `1572864 B`. |
| `down_8core_i19` | `0.999103663920951` | `0.7259668782353401 ms` | Legal, no final win. |
| `down_8core_i38_control_explicit` | `0.999104342904898` | `0.7252362556755543 ms` | Kept through final default. |

Decision: keep the current split gate/up and DRAM-sharded down `in0=38` family. Larger gate/up geometries were genuinely faster but fail real-weight layer PCC; synthetic/random PCC was not used as the veto.

## KV-Cache Dtype Trial

The second `$stage-review` found that the final BF16 cache policy needed a reduced-cache trial. I ran BFLOAT8_B K/V cache against the same single-layer real-weight paged decode contract: prefix 16 and non-aligned prefix 17, page table, paged prefill fill, eager decode, trace replay, PCC, and latency. The script released each trace with `ttnn.release_trace` before constructing the next trial. Results were written to `kv_cache_dtype_trials.csv`.

| Candidate | Prefix | PCC | Trace PCC | Traced decode rows | Decision |
| --- | ---: | ---: | ---: | --- | --- |
| `bf16_cache` | 16 | `0.9994682144570992` | `1.0` | `0.501827`, `0.505367`, `0.504637 ms` | Control; kept for primary prefix-16 path. |
| `bf16_cache` | 17 | `0.9992807441048474` | `1.0` | `0.505837`, `0.502887`, `0.505847 ms` | Control for non-aligned prefix. |
| `bfloat8_b_cache` | 16 | `0.9994673595855361` | `1.0` | `0.503647`, `0.503337`, `0.505298 ms` | Rejected: correct but slower than BF16 on the primary prefix-16 traced-decode path. |
| `bfloat8_b_cache` | 17 | `0.9992873744338733` | `1.0` | `0.504998`, `0.501557`, `0.503578 ms` | Correct, but the tiny non-aligned-path timing win does not beat the selected primary path or final default min `0.501138 ms`. |

Decision: keep BF16 KV cache. BFLOAT8_B does not change the advertised `current_supported_context=64` because the configured page count and block size are unchanged, and it was not selected.

## Commands and Results

Functional baseline confirmation:

```bash
pytest -q models/autoports/qwen_qwen3_4b/tests/test_functional_decoder.py
```

Result: 5 passed. PCC: synthetic seq 16 `0.9996239896537468`, synthetic seq 64 `0.9996795472223063`, real-weight seq 16 `0.9999917295183249`.

Optimized correctness:

```bash
pytest -q models/autoports/qwen_qwen3_4b/tests/test_optimized_decoder.py --tb=short
```

Result on the final code: `12 passed, 1 skipped in 65.33s`. The suite covers optimized prefill seq 16/17/64, real-weight prefill, paged decode prefix 16/17, real-weight non-aligned decode, batch-2 disjoint pages, traced replay determinism, no host fallback in measured paths, and the default context/cache contract tests.

Perf signposts without Tracy:

```bash
QWEN3_4B_OPT_RUN_PERF=1 \
QWEN3_4B_OPT_PERF_OUT=models/autoports/qwen_qwen3_4b/doc/optimized_decoder/perf_host_timings.csv \
pytest -q -s models/autoports/qwen_qwen3_4b/tests/test_optimized_decoder.py::test_optimized_perf_signposts --tb=short
```

Result: five final repeated rows were recorded. `perf_host_timings.csv`: warmed prefill seq 16 rows `1.626882`, `1.712621`, `1.658071`, `1.691041`, `1.837620 ms` with min `1.626882 ms`; warmed traced decode prefix 16 rows `0.502557`, `0.504977`, `0.502467`, `0.501138`, `0.501527 ms` with min `0.501138 ms`.

The optimized perf writer stores the warmed no-cache `prefill_forward` timing immediately after that call. An earlier draft accidentally overwrote the prefill row with a later cache-fill prefill, which is why stale docs reported `4.394558 ms`. The decoder body timer now excludes signpost call overhead while preserving signpost windows for Tracy.

Functional warmed prefill baseline:

```bash
QWEN3_4B_FUNCTIONAL_PREFILL_BASELINE=1 pytest ...
```

Result: `functional_prefill_baseline.csv` records functional prefill host timings `1.952949`, `1.675441`, `1.661491 ms`; min `1.661491 ms`. The final optimized prefill min `1.626882 ms` beats this baseline. Functional decode has no baseline because `FunctionalDecoder.decode_forward` is not implemented.

Final Tracy capture:

```bash
QWEN3_4B_OPT_RUN_PERF=1 \
QWEN3_4B_OPT_PERF_OUT=models/autoports/qwen_qwen3_4b/doc/optimized_decoder/perf_host_timings_tracy.csv \
python -m tracy -r -p -v --sync-host-device --dump-device-data-mid-run --check-exit-code \
  -o models/autoports/qwen_qwen3_4b/doc/optimized_decoder/tracy/perf_capture_final_bfp4_attention \
  -n qwen3_4b_opt_decoder_final_bfp4_attention \
  -m pytest -q -s models/autoports/qwen_qwen3_4b/tests/test_optimized_decoder.py::test_optimized_perf_signposts --tb=short
```

Result: 1 passed. `perf_host_timings_tracy.csv` records warmed prefill `1.876490 ms` and traced decode `0.531217 ms`. Ops CSV copied to `tracy/optimized_ops_final.csv`.

`tt-perf-report` commands:

```bash
~/.local/bin/tt-perf-report models/autoports/qwen_qwen3_4b/doc/optimized_decoder/tracy/optimized_ops_final.csv \
  --start-signpost PERF_PREFILL_WARMED --end-signpost PERF_PREFILL_WARMED_END \
  --csv models/autoports/qwen_qwen3_4b/doc/optimized_decoder/tt_perf_report_prefill.csv

~/.local/bin/tt-perf-report models/autoports/qwen_qwen3_4b/doc/optimized_decoder/tracy/optimized_ops_final.csv \
  --start-signpost PERF_TRACE_DECODE --end-signpost PERF_TRACE_DECODE_END \
  --csv models/autoports/qwen_qwen3_4b/doc/optimized_decoder/tt_perf_report_traced_decode.csv
```

Results:

| Report | Device time | Device ops | Host ops | Notes |
| --- | ---: | ---: | ---: | --- |
| Prefill | `696.664 us` | `29` | `0` | Packed QKV/output/gate/up/down rows show BFP4/LoFi; `.txt` artifact is the rendered table and `.csv` keeps machine-readable rows. |
| Traced decode | `472.281 us` | `38` | `0` | QKV/output/gate/up/down rows show BFP4/LoFi; down uses DRAM-sharded BFP4/LoFi matmul at `46.540 us`, `in0=38`; `.txt` artifact is the rendered table and `.csv` keeps machine-readable rows. |

Watcher:

```bash
TT_METAL_WATCHER=10 pytest -q -s models/autoports/qwen_qwen3_4b/tests/test_optimized_decoder.py --tb=short
```

Result on the final code: `12 passed, 1 skipped in 65.83s`; watcher attached/detached cleanly across all four local Blackhole devices and reported no errors.

Context contract:

```bash
.agents/scripts/check_context_contract.py --model-dir models/autoports/qwen_qwen3_4b
```

Result: `Context contract OK for models/autoports/qwen_qwen3_4b: target=40960, supported=64 (DRAM-limited).`

## Stage-Review Findings Closed

The previous `$stage-review` returned `more-work-needed` with two P2 findings.

| Finding | Fix | Evidence |
| --- | --- | --- |
| Prefill optimization closure was not earned because docs reported final optimized prefill `4.394558 ms`, slower than the functional `1.661491 ms`, and prefill advice rows were not addressed. | Fixed the perf writer to record warmed no-cache prefill before cache-fill prefill, cleaned the body timer, reran repeated perf, reran Tracy/`tt-perf-report`, and tried actionable prefill L1 advice. | Final optimized repeated prefill min `1.626882 ms`; functional baseline min `1.661491 ms`; focused advice rows `o_l1_post_dram=1.388303 ms`, `o_l1_post_l1=1.435913 ms`; production attempts preserved in `production_prefill_l1_trials.csv` were rejected after repeated paths failed to reproduce (`1.689821 ms` and `1.899130 ms`). |
| Context/cache defaults were inconsistent: `PagedKVConfig` default capacity was 64 tokens while `from_state_dict(max_seq_len=128)` default allowed larger position tables. | Introduced one default `PagedKVConfig`, made `from_state_dict` default to that capacity, rejected `max_seq_len` beyond paged cache capacity, and added tests. | `test_optimized_default_context_matches_default_paged_cache` and `test_optimized_rejects_context_beyond_paged_cache` pass in the final non-watcher and watcher suites. |
| BF16 KV cache had no reduced-cache trial. | Ran a BFLOAT8_B cache trial with real weights, prefix 16/17, paged prefill/decode, trace replay, PCC, latency, and trace release. | `kv_cache_dtype_trials.csv`; BFLOAT8_B passed PCC but was slower on the primary prefix-16 traced-decode path (`0.503337 ms` vs BF16 `0.501827 ms`) and did not beat final default min `0.501138 ms`. |
| `tt_perf_report_*.txt` files were command logs instead of rendered tables. | Regenerated prefill and traced-decode `.txt` files with non-CSV `tt-perf-report --no-color --no-summary`; kept machine-readable rows in `.csv`. | `tt_perf_report_prefill.txt` and `tt_perf_report_traced_decode.txt` now contain rendered performance and advice tables. |

## Rejected Advice and Evidence

| Advice or candidate | Decision | Evidence |
| --- | --- | --- |
| Use HiFi2/HiFi4 for MLP accuracy | Rejected for final MLP path. | Real-weight BFP4/LoFi decode PCC `0.999324241979847` exceeds 0.99 and is faster than BFP8/HiFi2. |
| BFP8 attention | Rejected for final path. | Final-topology control passed PCC but was slower (`0.7515158504247665 ms`) than BFP4/LoFi attention (`0.7159649394452572 ms` candidate, `0.501138 ms` final default). |
| Place matmul input 0 in L1 | Partially accepted. | Gate/up post-norm L1 won and was implemented. QKV L1 variants were slower (`0.9217248298227787 ms` or worse). All-MLP-L1 was slower (`0.9324047714471817 ms`). Decode down now uses a width-sharded L1 input with DRAM-sharded weights and measured faster than non-DRAM-sharded down. |
| Output subblock >= 2 for MLP down | Rejected for final path. | Explicit 2D down config with `out_subblock_w=2` failed TTNN validation: `Num output blocks along x (40) must be smaller than or equal to the number of columns in compute grid (8)`. |
| Packed/fused gate/up | Rejected for final path. | Packed gate/up was legal and measured (`0.8952450007200241 ms` DRAM, `0.9504840709269047 ms` post-norm L1) but slower than separate gate/up with post-norm L1 (`0.8917949162423611 ms` in the same round), and both were superseded by final production min `0.501138 ms`. |
| Larger explicit gate/up BFP4/LoFi geometries | Rejected for final path. | Faster rows `0.6240569055080414` to `0.6493963301181793 ms` failed real-weight 0.99 PCC (`0.9757735251608588` to `0.9782249973361682`). |
| Wider down 4-core DRAM-sharded geometry | Rejected for final path. | Exact L1 CB blocker: `1756416 B` requested beyond max L1 `1572864 B`. |
| DRAM-sharded decode projections | Accepted down-only; rejected other families. | `dram_down_only` was fastest in the sweep (`0.7435758598148823 ms`) and final default reproduced and improved it (`0.501138 ms`). QKV-only, output-only, gate/up-only, gate/up/down, and all-projection rows all passed PCC but were slower. |
| Reduced KV-cache dtype | Rejected for final path. | BFLOAT8_B cache passed real-weight prefix 16/17 and trace replay, but was slower than BF16 for the primary prefix-16 traced-decode path (`0.503337 ms` vs `0.501827 ms`) and did not beat final default min `0.501138 ms`. |
| Prefill input-0 L1 advice | Rejected for final production path after trials. | Focused rows showed `o_l1_post_dram=1.388303 ms` and `o_l1_post_l1=1.435913 ms`, but repeated production attempts preserved in `production_prefill_l1_trials.csv` did not reproduce the focused win (`1.689821 ms` for output-L1/post-DRAM and `1.899130 ms` for QKV/output/down-L1), while the final default path measured `1.626882 ms` and preserved traced decode. |
| Host fallback for RoPE/page tables | Rejected in measured path. | Setup helpers may create constants on host, but `prefill_forward` and `decode_forward` contain no host fallback calls and reports show 0 host ops. |
| Redundant post-blocking-trace sync | Removed. | `trace_decode_once` already calls `execute_trace(..., blocking=True)`. Removing the extra `synchronize_device` preserved trace PCC `1.0` and improved final repeated traced decode min to `0.501138 ms`. |

## Optimize Checklist

| Requirement | Status | Evidence |
| --- | --- | --- |
| Topology audit before optimization | Done | Table above and `tt_perf_report_*` artifacts. |
| Preserve functional semantics | Done | Prefill PCC at seq 16/17/64 and real-weight seq 16; residual, norms, attention, MLP covered. |
| Add paged KV decode | Done | Prefix 16/17 and batch-2 disjoint-page tests. |
| Support non-aligned logical sequence lengths | Done | Seq 17 prefill and prefix 17 decode tests. |
| Measure before and after | Done | Functional prefill baseline min `1.661491 ms`; optimized prefill final min `1.626882 ms`; functional decode absent, so optimized traced decode compared against all legal candidates. |
| Beat best correct traced decode candidate | Done | Final production min `0.501138 ms`; slower alternatives include BFP4/LoFi attention candidate `0.7159649394452572 ms`, `dram_down_only` `0.7435758598148823 ms`, all-projection DRAM-sharded `0.7583661936223507 ms`, packed gate/up `0.8952450007200241 ms`, QKV L1 `0.9217248298227787 ms`, and precision-only BFP4/LoFi `0.8973558433353901 ms`. |
| Try BFP4/LoFi when MLP dominates | Done | BFP4/LoFi selected from real-weight sweep. |
| Generate tt-perf-report CSV/tables | Done | `tt_perf_report_prefill.*`, `tt_perf_report_traced_decode.*`. |
| Eliminate unnecessary host fallback | Done | Static test and 0 host ops in reports. |
| Stress/repeated or watcher-clean run | Done | Watcher full suite passed. |
| Update docs | Done | This README and work log. |
| Stage review | Done | Final `$stage-review` returned `clean-pass`; earlier findings on context/cache default, prefill evidence, reduced KV-cache trial, and rendered report-table artifacts were fixed and rereviewed. |
| Commit stage-owned changes | Done | Stage-owned Qwen optimized-decoder files are committed locally after clean-pass; never push. |

## Scope Boundary

No multichip decoder, full model, generator, or vLLM integration work was started. The remaining stages must consume this optimized single-layer decoder as an artifact rather than expanding this goal.
