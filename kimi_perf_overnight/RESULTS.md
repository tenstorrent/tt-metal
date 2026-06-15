# Kimi prefill runner-vs-test perf — overnight results

Baseline (durably known): runner ~3.3 s/chunk @ MAX_SEQ_LEN=61440; no-PCC test ~1.94 s/chunk.
Gap ~1.4 s constant/additive per chunk. See RECOVERY.md for the question and method.

Results appended below as each experiment completes.

---
## 00_baseline_plain — Runner baseline MAX_SEQ_LEN=61440, no instrumentation. Confirms ~3.3 s/chunk AND that the added perf_probe code (toggles off) introduces no regression.

---
## 00_baseline_plain — Runner baseline MAX_SEQ_LEN=61440, no instrumentation. Confirms ~3.3 s/chunk AND that the added perf_probe code (toggles off) introduces no regression.

---
## 00_baseline_plain — Runner baseline MAX_SEQ_LEN=61440, no instrumentation. Confirms ~3.3 s/chunk AND that the added perf_probe code (toggles off) introduces no regression.

---
## STATUS 2026-06-13 20:12 — PAUSED (BLOCKED), 0 experiments completed

Device wedged (Device 15 ethernet core 26-25 timeout; ETH_LIVE_STATUS=0x0) — most likely caused by my
own `kill -KILL` of a runner mid-init at 19:51. ALSO: another user `asaigal` is actively running
migration-endpoint drivers on this shared Galaxy box. A `tt-smi -glx_reset` would clobber their work,
so I did NOT reset. Orchestrator stopped; monitoring loop polling ~every 25 min to auto-resume when the
box is free + healthy. Full detail in BLOCKER.md. No data lost; all 11 experiments still queued.

---
## 00_baseline_plain — Runner baseline MAX_SEQ_LEN=61440, no instrumentation. Confirms ~3.3 s/chunk AND that the added perf_probe code (toggles off) introduces no regression.

### 00_baseline_plain (runner) — 2026-06-14 01:56:02
per-iter pipeline.prefill() ms:
```
2951.19 ms
3001.93 ms
3049.68 ms
3062.79 ms
3101.57 ms
3100.01 ms
3091.99 ms
3196.22 ms
3145.75 ms
3178.00 ms
3151.28 ms
```

---
## 01_maxseq56320 — DECISIVE / cheapest: runner MAX_SEQ_LEN=56320 (=11*5120) to match the test's SEQ_CACHE=55*1024. If per-iter ms drops toward ~1.9 s, mla_seq_len (ring-buffer-proportional work) is the cause.

### 01_maxseq56320 (runner) — 2026-06-14 02:12:41
per-iter pipeline.prefill() ms:
```
```

---
## 01a_standalone_chunked — DECISIVE CONTROL (all 4 hypothesis agents converged here): runner in PREFILL_STANDALONE_CHUNKED mode @ MAX_SEQ_LEN=61440. Same pipeline/mla_seq_len as the request loop, but NO H2D socket, NO per-chunk metadata readback, NO ack channel, and NO request-mode clear_loaded_sub_device_manager (prefill_runner.py:578). PREFILL_PREFILL_SYNC=1 so each chunk syncs (per-chunk-comparable to the test). If total/11 ~1.9 s -> the 1.4 s is request-mode machinery (line-578 clear / sub-device baseline / socket). If still ~3.3 s -> genuine forward_chunk compute.

---
## 01_maxseq56320 — DECISIVE / cheapest: runner MAX_SEQ_LEN=56320 (=11*5120) to match the test's SEQ_CACHE=55*1024. If per-iter ms drops toward ~1.9 s, mla_seq_len (ring-buffer-proportional work) is the cause.

### 01_maxseq56320 (runner) — 2026-06-14 02:20:06
per-iter pipeline.prefill() ms:
```
2924.63 ms
2951.01 ms
3046.45 ms
3068.75 ms
3082.86 ms
3088.85 ms
3111.67 ms
3117.53 ms
3125.24 ms
3208.28 ms
3255.97 ms
```

---
## 01a_standalone_chunked — DECISIVE CONTROL (all 4 hypothesis agents converged here): runner in PREFILL_STANDALONE_CHUNKED mode @ MAX_SEQ_LEN=61440. Same pipeline/mla_seq_len as the request loop, but NO H2D socket, NO per-chunk metadata readback, NO ack channel, and NO request-mode clear_loaded_sub_device_manager (prefill_runner.py:578). PREFILL_PREFILL_SYNC=1 so each chunk syncs (per-chunk-comparable to the test). If total/11 ~1.9 s -> the 1.4 s is request-mode machinery (line-578 clear / sub-device baseline / socket). If still ~3.3 s -> genuine forward_chunk compute.

### 01a_standalone_chunked (standalone-chunked) — 2026-06-14 02:25:09
total + per-chunk timing:
```
2026-06-14 02:23:34.762 | INFO     | __main__:run_standalone_chunked_prefill_loop:358 - [standalone-chunked] prefilled chunk 1/11 (kv_actual=0)
2026-06-14 02:23:36.336 | INFO     | __main__:run_standalone_chunked_prefill_loop:358 - [standalone-chunked] prefilled chunk 2/11 (kv_actual=5120)
2026-06-14 02:23:37.970 | INFO     | __main__:run_standalone_chunked_prefill_loop:358 - [standalone-chunked] prefilled chunk 3/11 (kv_actual=10240)
2026-06-14 02:23:39.682 | INFO     | __main__:run_standalone_chunked_prefill_loop:358 - [standalone-chunked] prefilled chunk 4/11 (kv_actual=15360)
2026-06-14 02:23:41.472 | INFO     | __main__:run_standalone_chunked_prefill_loop:358 - [standalone-chunked] prefilled chunk 5/11 (kv_actual=20480)
2026-06-14 02:23:43.342 | INFO     | __main__:run_standalone_chunked_prefill_loop:358 - [standalone-chunked] prefilled chunk 6/11 (kv_actual=25600)
2026-06-14 02:23:45.282 | INFO     | __main__:run_standalone_chunked_prefill_loop:358 - [standalone-chunked] prefilled chunk 7/11 (kv_actual=30720)
2026-06-14 02:23:47.296 | INFO     | __main__:run_standalone_chunked_prefill_loop:358 - [standalone-chunked] prefilled chunk 8/11 (kv_actual=35840)
2026-06-14 02:23:49.393 | INFO     | __main__:run_standalone_chunked_prefill_loop:358 - [standalone-chunked] prefilled chunk 9/11 (kv_actual=40960)
2026-06-14 02:23:51.570 | INFO     | __main__:run_standalone_chunked_prefill_loop:358 - [standalone-chunked] prefilled chunk 10/11 (kv_actual=46080)
2026-06-14 02:23:53.815 | INFO     | __main__:run_standalone_chunked_prefill_loop:358 - [standalone-chunked] prefilled chunk 11/11 (kv_actual=51200)
2026-06-14 02:23:53.816 | INFO     | __main__:run_standalone_chunked_prefill_loop:361 - [standalone-chunked] 11 chunks prefilled in 20656.77 ms
```
PCC (record-only):
```
  PREFILL_STANDALONE_CHUNKED_PCC      = 0.88
  PREFILL_REQUEST_LOOP_PCC            = 0
2026-06-14 02:23:33.149 | INFO     | __main__:main:562 - Setup complete, running standalone chunked-prefill loop (golden KV-cache PCC check)
```

---
## 01b_standalone_sections — Standalone-chunked @ 61440 + section timing + construction dump. If 01a is still ~3.3 s (genuine compute), this localizes which section (embed/mla/moe) carries it WITHOUT any request-mode machinery, and dumps construction for diff vs the test (03). If 01a dropped to ~1.9 s, compare these sections against 02 (request-mode sections) to see which section the request machinery inflated.

---
## ANALYSIS (2026-06-14 02:29) — BREAKTHROUGH: the gap is request-loop machinery, not the model

### Numbers (per-chunk, 5120 tok, 61 layers, DEVICE_FP32 gate)
| exp | config | mean ms/chunk | notes |
|-----|--------|---------------|-------|
| 00 | request-loop, MAX_SEQ_LEN=61440 | **~3094** | 2951→3151, ramps w/ prefix (baseline reproduced) |
| 01 | request-loop, MAX_SEQ_LEN=56320 | **~3089** | 2925→3256 — IDENTICAL to 00 |
| 01a | **standalone-chunked**, 61440, per-chunk sync | **~1878** (20656ms/11) | NO socket / NO metadata readback / NO ack channel / NO line-578 clear |
| (ref) | no-PCC transformer TEST | ~1940 | the fast path |

### Verdicts
- **H3 (mla_seq_len 61440 vs 56320) — REFUTED.** Matching the test's buffer size (56320) left per-chunk
  unchanged (~3090 both). The KV-buffer/rope sizing is NOT the cost. (06/07 sweep at 81920/102400 will
  confirm flatness, but the decisive equal-to-test point already shows no effect.)
- **H1/H2 (request-mode machinery) — CONFIRMED.** The SAME pipeline object, SAME mla_seq_len=61440, run
  via `PREFILL_STANDALONE_CHUNKED` (which skips: the H2D socket, the per-chunk `ttnn.to_torch` metadata
  readback, the per-layer LayerAck channel, AND the request-mode-only `clear_loaded_sub_device_manager`
  at prefill_runner.py:578) runs at **~1878 ms/chunk — i.e. the TEST's speed**. So the ~1.2 s/chunk gap
  lives ENTIRELY in the request-loop driver path, NOT in forward_chunk compute, NOT in attention, NOT
  in buffer size.

### What this means / next
The remaining experiments now serve to localize WHICH piece of request-mode machinery costs ~1.2 s:
- **Strongest suspect:** the per-layer LayerAck callback fires `ttnn.synchronize_device` 61x/chunk in
  request mode (mla.py:638); standalone fires 0. (Caveat: an earlier PREFILL_DISABLE_LAYER_ACK run was
  reported "no change" — 04_skip_ack_sync and 08_disable_ack_sections will re-test cleanly and may
  overturn that.)
- **02 (request sections) vs 01b (standalone sections):** same instrumentation both sides → the section
  whose time differs (expect the per-layer/mla region carrying the 61 syncs) localizes it.
- Also in the request-only path: line-578 sub-device clear (H1), socket/metadata readback (H2).
Bottom line so far: **stop looking at the model; the fix is in the request-loop runner.**

### 01b_standalone_sections (standalone-chunked) — 2026-06-14 02:30:25
total + per-chunk timing:
```
2026-06-14 02:28:36.861 | INFO     | __main__:run_standalone_chunked_prefill_loop:358 - [standalone-chunked] prefilled chunk 1/11 (kv_actual=0)
2026-06-14 02:28:39.387 | INFO     | __main__:run_standalone_chunked_prefill_loop:358 - [standalone-chunked] prefilled chunk 2/11 (kv_actual=5120)
2026-06-14 02:28:42.008 | INFO     | __main__:run_standalone_chunked_prefill_loop:358 - [standalone-chunked] prefilled chunk 3/11 (kv_actual=10240)
2026-06-14 02:28:44.721 | INFO     | __main__:run_standalone_chunked_prefill_loop:358 - [standalone-chunked] prefilled chunk 4/11 (kv_actual=15360)
2026-06-14 02:28:47.517 | INFO     | __main__:run_standalone_chunked_prefill_loop:358 - [standalone-chunked] prefilled chunk 5/11 (kv_actual=20480)
2026-06-14 02:28:50.514 | INFO     | __main__:run_standalone_chunked_prefill_loop:358 - [standalone-chunked] prefilled chunk 6/11 (kv_actual=25600)
2026-06-14 02:28:53.471 | INFO     | __main__:run_standalone_chunked_prefill_loop:358 - [standalone-chunked] prefilled chunk 7/11 (kv_actual=30720)
2026-06-14 02:28:56.520 | INFO     | __main__:run_standalone_chunked_prefill_loop:358 - [standalone-chunked] prefilled chunk 8/11 (kv_actual=35840)
2026-06-14 02:28:59.634 | INFO     | __main__:run_standalone_chunked_prefill_loop:358 - [standalone-chunked] prefilled chunk 9/11 (kv_actual=40960)
2026-06-14 02:29:02.844 | INFO     | __main__:run_standalone_chunked_prefill_loop:358 - [standalone-chunked] prefilled chunk 10/11 (kv_actual=46080)
2026-06-14 02:29:06.138 | INFO     | __main__:run_standalone_chunked_prefill_loop:358 - [standalone-chunked] prefilled chunk 11/11 (kv_actual=51200)
2026-06-14 02:29:06.138 | INFO     | __main__:run_standalone_chunked_prefill_loop:361 - [standalone-chunked] 11 chunks prefilled in 31832.26 ms
```
section-timing (last 12 chunks):
```
2026-06-14 02:28:34.299 | INFO     | models.demos.deepseek_v3_d_p.tt.perf_probe:flush_sections:76 - [section-timing] total=10170.2ms dense=147.6(n=1) embed=9.8(n=1) mla=4794.5(n=61) moe=5218.4(n=60)
2026-06-14 02:28:36.860 | INFO     | models.demos.deepseek_v3_d_p.tt.perf_probe:flush_sections:76 - [section-timing] total=2360.8ms dense=36.1(n=1) embed=0.8(n=1) mla=656.0(n=61) moe=1667.9(n=60)
2026-06-14 02:28:39.387 | INFO     | models.demos.deepseek_v3_d_p.tt.perf_probe:flush_sections:76 - [section-timing] total=2327.1ms dense=2.4(n=1) embed=0.5(n=1) mla=656.5(n=61) moe=1667.7(n=60)
2026-06-14 02:28:42.008 | INFO     | models.demos.deepseek_v3_d_p.tt.perf_probe:flush_sections:76 - [section-timing] total=2409.0ms dense=2.6(n=1) embed=0.5(n=1) mla=744.3(n=61) moe=1661.7(n=60)
2026-06-14 02:28:44.721 | INFO     | models.demos.deepseek_v3_d_p.tt.perf_probe:flush_sections:76 - [section-timing] total=2502.5ms dense=2.6(n=1) embed=0.5(n=1) mla=820.8(n=61) moe=1678.6(n=60)
2026-06-14 02:28:47.517 | INFO     | models.demos.deepseek_v3_d_p.tt.perf_probe:flush_sections:76 - [section-timing] total=2579.4ms dense=2.6(n=1) embed=0.5(n=1) mla=904.7(n=61) moe=1671.7(n=60)
2026-06-14 02:28:50.514 | INFO     | models.demos.deepseek_v3_d_p.tt.perf_probe:flush_sections:76 - [section-timing] total=2784.1ms dense=2.5(n=1) embed=0.5(n=1) mla=988.6(n=61) moe=1792.5(n=60)
2026-06-14 02:28:53.470 | INFO     | models.demos.deepseek_v3_d_p.tt.perf_probe:flush_sections:76 - [section-timing] total=2739.7ms dense=2.6(n=1) embed=0.5(n=1) mla=1062.7(n=61) moe=1674.0(n=60)
2026-06-14 02:28:56.519 | INFO     | models.demos.deepseek_v3_d_p.tt.perf_probe:flush_sections:76 - [section-timing] total=2821.3ms dense=2.6(n=1) embed=0.5(n=1) mla=1146.5(n=61) moe=1671.7(n=60)
2026-06-14 02:28:59.633 | INFO     | models.demos.deepseek_v3_d_p.tt.perf_probe:flush_sections:76 - [section-timing] total=2898.6ms dense=2.6(n=1) embed=0.4(n=1) mla=1221.0(n=61) moe=1674.6(n=60)
2026-06-14 02:29:02.844 | INFO     | models.demos.deepseek_v3_d_p.tt.perf_probe:flush_sections:76 - [section-timing] total=2992.8ms dense=2.6(n=1) embed=0.5(n=1) mla=1303.8(n=61) moe=1686.0(n=60)
2026-06-14 02:29:06.138 | INFO     | models.demos.deepseek_v3_d_p.tt.perf_probe:flush_sections:76 - [section-timing] total=3064.3ms dense=2.6(n=1) embed=0.5(n=1) mla=1393.6(n=61) moe=1667.6(n=60)
```
construction-dump:
```
2026-06-14 02:28:20.721 | INFO     | models.demos.deepseek_v3_d_p.tt.perf_probe:dump_construction:119 - [construction-dump] TtPrefillTransformer:
  num_layers=61 seq_len=5120 is_chunked=True is_balanced=False padding_side=right
  block0: is_moe=False slot_num=? layer_num=?
  mla: mla_seq_len=? seq_len=? sp_factor=8 sp_axis=0 kv_lora_rank=512 scale=0.14467962580268923
  shared _chunked_kv_buf: shape=[1, 1, 61440, 576] dtype=DataType.BFLOAT8_B layout=Layout.TILE mem=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0)
2026-06-14 02:28:23.829 | INFO     | models.demos.deepseek_v3_d_p.tt.tt_deepseek_prefill_pipeline:compile:158 - TtDeepSeekPrefillPipeline.compile() — warming up one 5120-token chunk
2026-06-14 02:28:23.830 | INFO     | models.demos.deepseek_v3_d_p.tt.perf_probe:dump_first_chunk_inputs:128 - [input-dump] forward_chunk first call:
  kv_actual_isl=0 cache_user_id=0
```
PCC (record-only):
```
  PREFILL_STANDALONE_CHUNKED_PCC      = 0.88
  PREFILL_REQUEST_LOOP_PCC            = 0
2026-06-14 02:28:34.300 | INFO     | __main__:main:562 - Setup complete, running standalone chunked-prefill loop (golden KV-cache PCC check)
```

---
## 02_runner_sections — Runner 61440 + per-section timing + construction/input dump. Localizes the 1.4 s across embed / mla / moe (sync-bracketed) and records construction config + first-chunk input tensor specs for diff vs the test.

### 02_runner_sections (runner) — 2026-06-14 02:35:00
per-iter pipeline.prefill() ms:
```
3266.88 ms
3245.19 ms
3381.62 ms
3504.08 ms
3577.49 ms
3691.78 ms
3775.17 ms
3970.65 ms
3979.93 ms
4193.39 ms
4158.00 ms
```
section-timing (last 12 chunks):
```
2026-06-14 02:33:52.383 | INFO     | models.demos.deepseek_v3_d_p.tt.perf_probe:flush_sections:76 - [section-timing] total=10164.0ms dense=155.9(n=1) embed=5.1(n=1) mla=4790.8(n=61) moe=5212.2(n=60)
2026-06-14 02:34:08.707 | INFO     | models.demos.deepseek_v3_d_p.tt.perf_probe:flush_sections:76 - [section-timing] total=3013.4ms dense=37.7(n=1) embed=1.3(n=1) mla=1076.3(n=61) moe=1898.1(n=60)
2026-06-14 02:34:11.955 | INFO     | models.demos.deepseek_v3_d_p.tt.perf_probe:flush_sections:76 - [section-timing] total=2986.9ms dense=4.4(n=1) embed=0.6(n=1) mla=1086.9(n=61) moe=1894.9(n=60)
2026-06-14 02:34:15.338 | INFO     | models.demos.deepseek_v3_d_p.tt.perf_probe:flush_sections:76 - [section-timing] total=3117.3ms dense=4.4(n=1) embed=0.5(n=1) mla=1182.5(n=61) moe=1929.8(n=60)
2026-06-14 02:34:18.845 | INFO     | models.demos.deepseek_v3_d_p.tt.perf_probe:flush_sections:76 - [section-timing] total=3230.8ms dense=4.6(n=1) embed=0.5(n=1) mla=1272.9(n=61) moe=1952.8(n=60)
2026-06-14 02:34:22.425 | INFO     | models.demos.deepseek_v3_d_p.tt.perf_probe:flush_sections:76 - [section-timing] total=3303.7ms dense=4.6(n=1) embed=0.5(n=1) mla=1337.6(n=61) moe=1961.0(n=60)
2026-06-14 02:34:26.118 | INFO     | models.demos.deepseek_v3_d_p.tt.perf_probe:flush_sections:76 - [section-timing] total=3411.0ms dense=4.6(n=1) embed=0.6(n=1) mla=1430.5(n=61) moe=1975.4(n=60)
2026-06-14 02:34:29.895 | INFO     | models.demos.deepseek_v3_d_p.tt.perf_probe:flush_sections:76 - [section-timing] total=3501.9ms dense=4.7(n=1) embed=0.5(n=1) mla=1522.2(n=61) moe=1974.5(n=60)
2026-06-14 02:34:33.868 | INFO     | models.demos.deepseek_v3_d_p.tt.perf_probe:flush_sections:76 - [section-timing] total=3685.5ms dense=4.7(n=1) embed=0.5(n=1) mla=1596.9(n=61) moe=2083.3(n=60)
2026-06-14 02:34:37.849 | INFO     | models.demos.deepseek_v3_d_p.tt.perf_probe:flush_sections:76 - [section-timing] total=3700.9ms dense=4.6(n=1) embed=0.5(n=1) mla=1683.7(n=61) moe=2012.0(n=60)
2026-06-14 02:34:42.045 | INFO     | models.demos.deepseek_v3_d_p.tt.perf_probe:flush_sections:76 - [section-timing] total=3902.6ms dense=4.7(n=1) embed=0.6(n=1) mla=1788.2(n=61) moe=2109.1(n=60)
2026-06-14 02:34:46.205 | INFO     | models.demos.deepseek_v3_d_p.tt.perf_probe:flush_sections:76 - [section-timing] total=3870.5ms dense=4.8(n=1) embed=0.5(n=1) mla=1850.0(n=61) moe=2015.3(n=60)
```
construction-dump:
```
2026-06-14 02:33:38.947 | INFO     | models.demos.deepseek_v3_d_p.tt.perf_probe:dump_construction:119 - [construction-dump] TtPrefillTransformer:
  num_layers=61 seq_len=5120 is_chunked=True is_balanced=False padding_side=right
  block0: is_moe=False slot_num=? layer_num=?
  mla: mla_seq_len=? seq_len=? sp_factor=8 sp_axis=0 kv_lora_rank=512 scale=0.14467962580268923
  shared _chunked_kv_buf: shape=[1, 1, 61440, 576] dtype=DataType.BFLOAT8_B layout=Layout.TILE mem=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0)
2026-06-14 02:33:41.908 | INFO     | models.demos.deepseek_v3_d_p.tt.tt_deepseek_prefill_pipeline:compile:158 - TtDeepSeekPrefillPipeline.compile() — warming up one 5120-token chunk
2026-06-14 02:33:41.909 | INFO     | models.demos.deepseek_v3_d_p.tt.perf_probe:dump_first_chunk_inputs:128 - [input-dump] forward_chunk first call:
  kv_actual_isl=0 cache_user_id=0
```
input-dump:
```
2026-06-14 02:33:41.909 | INFO     | models.demos.deepseek_v3_d_p.tt.perf_probe:dump_first_chunk_inputs:128 - [input-dump] forward_chunk first call:
  kv_actual_isl=0 cache_user_id=0
  token_ids: shape=[1, 1, 640] dtype=DataType.UINT32 layout=Layout.ROW_MAJOR mem=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0)
  kvpe_cache: shape=[61, 1, 7680, 576] dtype=DataType.BFLOAT8_B layout=Layout.TILE mem=MemoryConfig(memory_layout=TensorMemoryLayout::ND_SHARDED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt,nd_shard_spec={"shard_shape":[1, 1, 32, 576],"grid":[{"start":{"x":0,"y":0},"end":{"x":0,"y":0}}, {"start":{"x":1,"y":0},"end":{"x":1,"y":0}}, {"start":{"x":2,"y":0},"end":{"x":2,"y":0}}, {"start":{"x":3,"y":0},"end":{"x":3,"y":0}}, {"start":{"x":4,"y":0},"end":{"x":4,"y":0}}, {"start":{"x":5,"y":0},"end":{"x":5,"y":0}}, {"start":{"x":6,"y":0},"end":{"x":6,"y":0}}, {"start":{"x":7,"y":0},"end":{"x":7,"y":0}}],"orientation":"ShardOrientation::ROW_MAJOR","shard_distribution_strategy":"ShardDistributionStrategy::ROUND_ROBIN_1D"},created_with_nd_shard_spec=1)
2026-06-14 02:33:41.914 | DEBUG    | models.demos.deepseek_v3_d_p.tt.tt_parallel_embedding:forward:237 - Forward: token_ids shape=Shape([1, 1, 640])
```

---
## 03_test_sections — No-PCC transformer test (fast path) + per-section timing + construction/input dump. SAME instrumentation as 02 -> direct apples-to-apples: which section is smaller in the test, and does its construction config differ (mla_seq_len 56320 vs 61440).

## ANALYSIS (2026-06-14 02:43) — section-timing localization (01b standalone vs 02 request)

Construction-dump IDENTICAL both paths: _chunked_kv_buf [1,1,61440,576] BFLOAT8_B, num_layers=61,
seq_len=5120, sp_factor=8, kv_lora_rank=512. => the gap is NOT a construction difference.

Section timing (sync-bracketed; absolute ms inflated ~2x by the instrumentation, but request−standalone
deltas localize the gap), last chunk (kv_actual~51200):
| section      | 01b standalone | 02 request | delta |
| mla (n=61)   | 1393.6 ms      | 1850.0 ms  | +456 ms |
| moe (n=60)   | 1667.6 ms      | 2015.3 ms  | +347 ms |
| embed/dense  | ~3 ms          | ~5 ms      | ~0 |
| total        | 3064 ms        | 3870 ms    | +806 ms |

Interpretation:
- Request mode is slower in BOTH mla AND moe — not a single localized op.
- mla +456ms is consistent with the 61 per-layer ack `synchronize_device` (mla.py:638), which fires in
  request mode (on_layer_complete set) but NOT in standalone.
- moe +347ms has NO ack in it → points to an ADDITIONAL systemic request-mode cost: the H2D stream
  service running in the background and/or the request-mode-only clear_loaded_sub_device_manager
  (prefill_runner.py:578) changing the sub-device baseline that per-layer MoE load/clear assumes.
- CAVEAT: section-timing's own syncs compress the gap (with timing ~800ms; the TRUE no-timing gap is
  ~1212ms). The async-dispatch component of the gap is masked here → 04_skip_ack_sync (no timing) is the
  decisive test for the ack-sync share; if it doesn't fully close the gap, the H2D-service / line-578
  sub-device cost (the moe-side excess) is the remainder.
Awaiting: 03 (test sections, expect ≈ 01b), 04 (skip_ack_sync, no timing — decisive), 08.

### 03_test_sections (test) — 2026-06-14 02:49:41
pytest outcome:
```
PASSED
  /home/ppopovic/tt-metal/python_env/lib/python3.10/site-packages/pydantic/_internal/_config.py:291: PydanticDeprecatedSince20: Support for class-based `config` is deprecated, use ConfigDict instead. Deprecated in Pydantic V2.0 to be removed in V3.0. See Pydantic V2 Migration Guide at https://errors.pydantic.dev/2.9/migration/
PASSED models/demos/deepseek_v3_d_p/tests/test_prefill_transformer_chunked.py::test_kimi_prefill_transformer_chunked_no_pcc[blackhole-kimi-mesh-8x4-L61-chunks11-iters20]
============ 1 passed, 8 deselected, 1 warning in 836.68s (0:13:56) ============
```
section-timing (last 12 chunks):
```
2026-06-14 02:48:44.042 | INFO     | models.demos.deepseek_v3_d_p.tt.perf_probe:flush_sections:76 - [section-timing] total=3065.8ms dense=2.7(n=1) embed=0.5(n=1) mla=1395.8(n=61) moe=1666.7(n=60)
2026-06-14 02:48:46.601 | INFO     | models.demos.deepseek_v3_d_p.tt.perf_probe:flush_sections:76 - [section-timing] total=2351.7ms dense=2.6(n=1) embed=0.5(n=1) mla=664.4(n=61) moe=1684.2(n=60)
2026-06-14 02:48:49.175 | INFO     | models.demos.deepseek_v3_d_p.tt.perf_probe:flush_sections:76 - [section-timing] total=2357.5ms dense=2.6(n=1) embed=0.6(n=1) mla=685.3(n=61) moe=1669.0(n=60)
2026-06-14 02:48:51.811 | INFO     | models.demos.deepseek_v3_d_p.tt.perf_probe:flush_sections:76 - [section-timing] total=2427.9ms dense=2.6(n=1) embed=0.5(n=1) mla=750.2(n=61) moe=1674.6(n=60)
2026-06-14 02:48:54.537 | INFO     | models.demos.deepseek_v3_d_p.tt.perf_probe:flush_sections:76 - [section-timing] total=2510.9ms dense=2.7(n=1) embed=0.5(n=1) mla=833.8(n=61) moe=1674.0(n=60)
2026-06-14 02:48:57.364 | INFO     | models.demos.deepseek_v3_d_p.tt.perf_probe:flush_sections:76 - [section-timing] total=2606.0ms dense=2.6(n=1) embed=0.6(n=1) mla=918.1(n=61) moe=1684.7(n=60)
2026-06-14 02:49:00.258 | INFO     | models.demos.deepseek_v3_d_p.tt.perf_probe:flush_sections:76 - [section-timing] total=2674.6ms dense=2.7(n=1) embed=0.5(n=1) mla=992.9(n=61) moe=1678.5(n=60)
2026-06-14 02:49:03.236 | INFO     | models.demos.deepseek_v3_d_p.tt.perf_probe:flush_sections:76 - [section-timing] total=2757.8ms dense=2.6(n=1) embed=0.5(n=1) mla=1072.1(n=61) moe=1682.5(n=60)
2026-06-14 02:49:06.281 | INFO     | models.demos.deepseek_v3_d_p.tt.perf_probe:flush_sections:76 - [section-timing] total=2822.0ms dense=2.6(n=1) embed=0.5(n=1) mla=1147.6(n=61) moe=1671.3(n=60)
2026-06-14 02:49:09.411 | INFO     | models.demos.deepseek_v3_d_p.tt.perf_probe:flush_sections:76 - [section-timing] total=2904.6ms dense=2.6(n=1) embed=0.5(n=1) mla=1231.4(n=61) moe=1670.0(n=60)
2026-06-14 02:49:12.634 | INFO     | models.demos.deepseek_v3_d_p.tt.perf_probe:flush_sections:76 - [section-timing] total=3002.1ms dense=2.6(n=1) embed=0.5(n=1) mla=1319.1(n=61) moe=1679.8(n=60)
2026-06-14 02:49:15.972 | INFO     | models.demos.deepseek_v3_d_p.tt.perf_probe:flush_sections:76 - [section-timing] total=3101.9ms dense=2.7(n=1) embed=0.5(n=1) mla=1420.4(n=61) moe=1678.3(n=60)
```
construction-dump:
```
2026-06-14 02:38:26.150 | INFO     | models.demos.deepseek_v3_d_p.tt.perf_probe:dump_construction:119 - [construction-dump] TtPrefillTransformer:
  num_layers=61 seq_len=5120 is_chunked=True is_balanced=False padding_side=right
  block0: is_moe=False slot_num=? layer_num=?
  mla: mla_seq_len=? seq_len=? sp_factor=8 sp_axis=0 kv_lora_rank=512 scale=0.14467962580268923
  shared _chunked_kv_buf: shape=[1, 1, 56320, 576] dtype=DataType.BFLOAT8_B layout=Layout.TILE mem=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0)
2026-06-14 02:38:29.007 | info     |           Metal | Enabling program cache on MeshDevice 0 (mesh_device.cpp:1033)
2026-06-14 02:38:29.010 | INFO     | models.demos.deepseek_v3_d_p.tt.perf_probe:dump_first_chunk_inputs:128 - [input-dump] forward_chunk first call:
  kv_actual_isl=0 cache_user_id=0
```
input-dump:
```
2026-06-14 02:38:29.010 | INFO     | models.demos.deepseek_v3_d_p.tt.perf_probe:dump_first_chunk_inputs:128 - [input-dump] forward_chunk first call:
  kv_actual_isl=0 cache_user_id=0
  token_ids: shape=[1, 1, 640] dtype=DataType.UINT32 layout=Layout.ROW_MAJOR mem=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0)
  kvpe_cache: shape=[61, 1, 7040, 576] dtype=DataType.BFLOAT8_B layout=Layout.TILE mem=MemoryConfig(memory_layout=TensorMemoryLayout::ND_SHARDED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt,nd_shard_spec={"shard_shape":[1, 1, 32, 576],"grid":[{"start":{"x":0,"y":0},"end":{"x":0,"y":0}}, {"start":{"x":1,"y":0},"end":{"x":1,"y":0}}, {"start":{"x":2,"y":0},"end":{"x":2,"y":0}}, {"start":{"x":3,"y":0},"end":{"x":3,"y":0}}, {"start":{"x":4,"y":0},"end":{"x":4,"y":0}}, {"start":{"x":5,"y":0},"end":{"x":5,"y":0}}, {"start":{"x":6,"y":0},"end":{"x":6,"y":0}}, {"start":{"x":7,"y":0},"end":{"x":7,"y":0}}],"orientation":"ShardOrientation::ROW_MAJOR","shard_distribution_strategy":"ShardDistributionStrategy::ROUND_ROBIN_1D"},created_with_nd_shard_spec=1)
2026-06-14 02:38:29.010 | DEBUG    | models.demos.deepseek_v3_d_p.tt.tt_parallel_embedding:forward:237 - Forward: token_ids shape=Shape([1, 1, 640])
```

---
## 04_runner_skip_ack_sync — Runner 61440 + PREFILL_SKIP_ACK_SYNC=1. The runner fires on_layer_complete per layer, doing ttnn.synchronize_device 61x/chunk; the test passes on_layer_complete=None (0 syncs). This keeps the cheap ack inject but drops those 61 syncs, isolating sync cost from inject cost (complements the earlier DISABLE_LAYER_ACK which removed both).

### 04_runner_skip_ack_sync (runner) — 2026-06-14 02:53:52
per-iter pipeline.prefill() ms:
```
3064.89 ms
3094.99 ms
3132.85 ms
3206.62 ms
3299.42 ms
3233.43 ms
3235.29 ms
3237.60 ms
3222.14 ms
3236.12 ms
3340.11 ms
```

---
## 05_maxseq56320_sections — Runner MAX_SEQ_LEN=56320 + per-section timing. If 01 improved per-iter ms, this shows WHICH section dropped (hypothesis: mla), confirming the ring_mla gather over _chunked_kv_buf scales with the allocated buffer (mla_seq_len) rather than logical_n.

### 05_maxseq56320_sections (runner) — 2026-06-14 02:58:15
per-iter pipeline.prefill() ms:
```
3248.54 ms
3234.43 ms
3382.10 ms
3476.51 ms
3596.60 ms
3780.55 ms
3781.10 ms
3870.30 ms
3998.87 ms
4180.35 ms
4135.97 ms
```
section-timing (last 12 chunks):
```
2026-06-14 02:57:06.934 | INFO     | models.demos.deepseek_v3_d_p.tt.perf_probe:flush_sections:76 - [section-timing] total=10279.5ms dense=159.8(n=1) embed=15.0(n=1) mla=4781.6(n=61) moe=5323.1(n=60)
2026-06-14 02:57:23.226 | INFO     | models.demos.deepseek_v3_d_p.tt.perf_probe:flush_sections:76 - [section-timing] total=2991.5ms dense=39.1(n=1) embed=2.6(n=1) mla=1069.1(n=61) moe=1880.7(n=60)
2026-06-14 02:57:26.462 | INFO     | models.demos.deepseek_v3_d_p.tt.perf_probe:flush_sections:76 - [section-timing] total=2979.8ms dense=4.3(n=1) embed=0.5(n=1) mla=1077.3(n=61) moe=1897.7(n=60)
2026-06-14 02:57:29.846 | INFO     | models.demos.deepseek_v3_d_p.tt.perf_probe:flush_sections:76 - [section-timing] total=3119.5ms dense=4.5(n=1) embed=0.5(n=1) mla=1187.3(n=61) moe=1927.2(n=60)
2026-06-14 02:57:33.324 | INFO     | models.demos.deepseek_v3_d_p.tt.perf_probe:flush_sections:76 - [section-timing] total=3210.8ms dense=6.2(n=1) embed=0.5(n=1) mla=1265.4(n=61) moe=1938.7(n=60)
2026-06-14 02:57:36.923 | INFO     | models.demos.deepseek_v3_d_p.tt.perf_probe:flush_sections:76 - [section-timing] total=3324.3ms dense=4.6(n=1) embed=0.6(n=1) mla=1348.5(n=61) moe=1970.6(n=60)
2026-06-14 02:57:40.705 | INFO     | models.demos.deepseek_v3_d_p.tt.perf_probe:flush_sections:76 - [section-timing] total=3508.1ms dense=4.6(n=1) embed=0.5(n=1) mla=1434.1(n=61) moe=2068.8(n=60)
2026-06-14 02:57:44.488 | INFO     | models.demos.deepseek_v3_d_p.tt.perf_probe:flush_sections:76 - [section-timing] total=3509.5ms dense=5.8(n=1) embed=0.5(n=1) mla=1522.6(n=61) moe=1980.6(n=60)
2026-06-14 02:57:48.360 | INFO     | models.demos.deepseek_v3_d_p.tt.perf_probe:flush_sections:76 - [section-timing] total=3594.4ms dense=4.7(n=1) embed=0.5(n=1) mla=1597.8(n=61) moe=1991.3(n=60)
2026-06-14 02:57:52.361 | INFO     | models.demos.deepseek_v3_d_p.tt.perf_probe:flush_sections:76 - [section-timing] total=3704.8ms dense=4.8(n=1) embed=0.5(n=1) mla=1687.6(n=61) moe=2011.9(n=60)
2026-06-14 02:57:56.543 | INFO     | models.demos.deepseek_v3_d_p.tt.perf_probe:flush_sections:76 - [section-timing] total=3881.8ms dense=4.8(n=1) embed=0.5(n=1) mla=1806.2(n=61) moe=2070.3(n=60)
2026-06-14 02:58:00.680 | INFO     | models.demos.deepseek_v3_d_p.tt.perf_probe:flush_sections:76 - [section-timing] total=3845.7ms dense=4.7(n=1) embed=0.5(n=1) mla=1833.2(n=61) moe=2007.3(n=60)
```

---
## 06_maxseq81920 — mla_seq_len scaling sweep point: MAX_SEQ_LEN=81920 (16*5120), LARGER than baseline. If per-iter ms scales UP with the allocated KV buffer, the ring_mla gather over _chunked_kv_buf is buffer-bound (proves the hypothesis). Producer still pushes the same 11 chunks.

### 06_maxseq81920 (runner) — 2026-06-14 03:03:00
per-iter pipeline.prefill() ms:
```
3058.01 ms
3103.27 ms
3160.53 ms
3162.89 ms
3215.67 ms
3302.10 ms
3262.43 ms
3275.20 ms
3294.63 ms
3257.37 ms
3277.32 ms
```

---
## 07_maxseq102400 — mla_seq_len scaling sweep point: MAX_SEQ_LEN=102400 (20*5120), largest. Fourth point (with 01=56320, 00=61440, 06=81920) to test LINEARITY of per-iter ms vs allocated KV buffer. Linear -> buffer-bound gather; flat -> mla_seq_len conclusively NOT the cause.

## ANALYSIS (2026-06-14 03:05) — ack-sync REFUTED; narrowing to H2D-service / line-578 sub-device clear

- 04_runner_skip_ack_sync (no instrumentation): per-iter 3065→3340, mean ~3210 ms ≈ baseline ~3094
  (even marginally higher). => Dropping the 61 per-layer ack synchronize_device does NOT close the gap.
  **The per-layer ack is NOT the cause** (corroborates the earlier DISABLE_LAYER_ACK "no change"; the
  section-timing "mla +456ms" in 02 was an artifact of instrumentation syncs interacting with the ack).
- 03_test_sections [section-timing]: mla ~1231-1420 / moe ~1670-1679 / total ~2905-3102 — matches 01b
  STANDALONE (mla 1394/moe 1668/total 3064). => test path == standalone path (both ~1.9s regime). Request
  is the sole outlier. (test PASSED, 836s.)
- 06_maxseq81920: mean ~3215 ms vs 56320=3089 / 61440=3094. Marginal (~120ms over +20k buffer), NOT the
  1.2s. H3 (mla_seq_len) stays refuted as the gap cause. (07=102400 pending to confirm.)

REMAINING SUSPECTS (request-mode-only setup, present in request loop but NOT in standalone/test):
  (a) the H2D stream service running in the background (socket polling threads / a command queue),
  (b) the request-mode-only clear_loaded_sub_device_manager at prefill_runner.py:578.
NEXT: env-gate skipping the line-578 clear (queue/09) — if per-iter drops to ~1.9s, that's the cause;
if not, the H2D service background activity is implicated (harder to disable; would profile its threads).

### 07_maxseq102400 (runner) — 2026-06-14 03:07:20
per-iter pipeline.prefill() ms:
```
3123.12 ms
3163.86 ms
3286.38 ms
3208.14 ms
3239.04 ms
3367.13 ms
3279.81 ms
3320.00 ms
3341.06 ms
3317.51 ms
3359.99 ms
```

---
## 08_disable_ack_sections — Runner 61440 + DISABLE_LAYER_ACK=1 + SECTION_TIMING. Removes BOTH the per-layer ack inject and its synchronize_device (on_layer_complete=None), matching the test which fires no callback. Compared against 02 (ack on) this attributes how much of the 'mla' section is the 61 per-layer acks/syncs vs genuine attention compute.

### 08_disable_ack_sections (runner) — 2026-06-14 03:12:01
per-iter pipeline.prefill() ms:
```
3235.39 ms
3256.23 ms
3364.09 ms
3570.89 ms
3577.01 ms
3675.41 ms
3764.25 ms
3844.57 ms
4006.24 ms
4082.96 ms
4235.82 ms
```
section-timing (last 12 chunks):
```
2026-06-14 03:10:49.404 | INFO     | models.demos.deepseek_v3_d_p.tt.perf_probe:flush_sections:76 - [section-timing] total=10347.4ms dense=151.5(n=1) embed=19.0(n=1) mla=4840.5(n=61) moe=5336.4(n=60)
2026-06-14 03:11:07.414 | INFO     | models.demos.deepseek_v3_d_p.tt.perf_probe:flush_sections:76 - [section-timing] total=2976.4ms dense=38.1(n=1) embed=1.2(n=1) mla=1060.0(n=61) moe=1877.0(n=60)
2026-06-14 03:11:10.672 | INFO     | models.demos.deepseek_v3_d_p.tt.perf_probe:flush_sections:76 - [section-timing] total=2990.7ms dense=4.6(n=1) embed=0.5(n=1) mla=1077.9(n=61) moe=1907.7(n=60)
2026-06-14 03:11:14.038 | INFO     | models.demos.deepseek_v3_d_p.tt.perf_probe:flush_sections:76 - [section-timing] total=3100.7ms dense=4.5(n=1) embed=0.5(n=1) mla=1165.9(n=61) moe=1929.7(n=60)
2026-06-14 03:11:17.611 | INFO     | models.demos.deepseek_v3_d_p.tt.perf_probe:flush_sections:76 - [section-timing] total=3203.2ms dense=4.5(n=1) embed=0.5(n=1) mla=1245.0(n=61) moe=1953.1(n=60)
2026-06-14 03:11:21.190 | INFO     | models.demos.deepseek_v3_d_p.tt.perf_probe:flush_sections:76 - [section-timing] total=3303.0ms dense=4.5(n=1) embed=0.6(n=1) mla=1341.6(n=61) moe=1956.2(n=60)
2026-06-14 03:11:24.867 | INFO     | models.demos.deepseek_v3_d_p.tt.perf_probe:flush_sections:76 - [section-timing] total=3400.4ms dense=4.5(n=1) embed=0.5(n=1) mla=1425.8(n=61) moe=1969.7(n=60)
2026-06-14 03:11:28.633 | INFO     | models.demos.deepseek_v3_d_p.tt.perf_probe:flush_sections:76 - [section-timing] total=3485.2ms dense=4.6(n=1) embed=0.5(n=1) mla=1503.6(n=61) moe=1976.5(n=60)
2026-06-14 03:11:32.480 | INFO     | models.demos.deepseek_v3_d_p.tt.perf_probe:flush_sections:76 - [section-timing] total=3567.7ms dense=4.7(n=1) embed=0.5(n=1) mla=1579.7(n=61) moe=1982.9(n=60)
2026-06-14 03:11:36.488 | INFO     | models.demos.deepseek_v3_d_p.tt.perf_probe:flush_sections:76 - [section-timing] total=3729.9ms dense=4.7(n=1) embed=0.5(n=1) mla=1667.2(n=61) moe=2057.5(n=60)
2026-06-14 03:11:40.573 | INFO     | models.demos.deepseek_v3_d_p.tt.perf_probe:flush_sections:76 - [section-timing] total=3791.4ms dense=4.7(n=1) embed=0.5(n=1) mla=1765.6(n=61) moe=2020.5(n=60)
2026-06-14 03:11:44.811 | INFO     | models.demos.deepseek_v3_d_p.tt.perf_probe:flush_sections:76 - [section-timing] total=3852.7ms dense=5.3(n=1) embed=0.5(n=1) mla=1840.8(n=61) moe=2006.1(n=60)
```

---
## 09_standalone_force_preclear — DECISIVE disambiguation: standalone-chunked @61440 + PREFILL_FORCE_PRECLEAR=1 — adds the request-mode-only clear_loaded_sub_device_manager (line-578 equivalent) to the otherwise-fast standalone path. If total/11 slows from ~1878ms to ~3090ms, the sub-device-manager clear (reverting compile()'s custom CCL/MoE manager to whole-chip default) IS the root cause of the request-loop gap (H1). If it stays ~1878ms, the clear is innocent and the H2D stream service background activity is the cause.

### 09_standalone_force_preclear (standalone-chunked) — 2026-06-14 03:17:01
total + per-chunk timing:
```
2026-06-14 03:15:25.455 | INFO     | __main__:run_standalone_chunked_prefill_loop:358 - [standalone-chunked] prefilled chunk 1/11 (kv_actual=0)
2026-06-14 03:15:27.038 | INFO     | __main__:run_standalone_chunked_prefill_loop:358 - [standalone-chunked] prefilled chunk 2/11 (kv_actual=5120)
2026-06-14 03:15:28.669 | INFO     | __main__:run_standalone_chunked_prefill_loop:358 - [standalone-chunked] prefilled chunk 3/11 (kv_actual=10240)
2026-06-14 03:15:30.381 | INFO     | __main__:run_standalone_chunked_prefill_loop:358 - [standalone-chunked] prefilled chunk 4/11 (kv_actual=15360)
2026-06-14 03:15:32.172 | INFO     | __main__:run_standalone_chunked_prefill_loop:358 - [standalone-chunked] prefilled chunk 5/11 (kv_actual=20480)
2026-06-14 03:15:34.039 | INFO     | __main__:run_standalone_chunked_prefill_loop:358 - [standalone-chunked] prefilled chunk 6/11 (kv_actual=25600)
2026-06-14 03:15:35.981 | INFO     | __main__:run_standalone_chunked_prefill_loop:358 - [standalone-chunked] prefilled chunk 7/11 (kv_actual=30720)
2026-06-14 03:15:37.994 | INFO     | __main__:run_standalone_chunked_prefill_loop:358 - [standalone-chunked] prefilled chunk 8/11 (kv_actual=35840)
2026-06-14 03:15:40.091 | INFO     | __main__:run_standalone_chunked_prefill_loop:358 - [standalone-chunked] prefilled chunk 9/11 (kv_actual=40960)
2026-06-14 03:15:42.268 | INFO     | __main__:run_standalone_chunked_prefill_loop:358 - [standalone-chunked] prefilled chunk 10/11 (kv_actual=46080)
2026-06-14 03:15:44.514 | INFO     | __main__:run_standalone_chunked_prefill_loop:358 - [standalone-chunked] prefilled chunk 11/11 (kv_actual=51200)
2026-06-14 03:15:44.515 | INFO     | __main__:run_standalone_chunked_prefill_loop:361 - [standalone-chunked] 11 chunks prefilled in 20674.50 ms
```
PCC (record-only):
```
  PREFILL_STANDALONE_CHUNKED_PCC      = 0.88
  PREFILL_REQUEST_LOOP_PCC            = 0
2026-06-14 03:15:23.834 | INFO     | __main__:main:570 - Setup complete, running standalone chunked-prefill loop (golden KV-cache PCC check)
```

## CONCLUSION (2026-06-14 03:29) — ROOT CAUSE: the H2D stream service, not the model

### The decisive experiment
09_standalone_force_preclear (standalone-chunked + PREFILL_FORCE_PRECLEAR=1, i.e. the otherwise-fast
path WITH the request-mode line-578 clear_loaded_sub_device_manager added): **20674 ms / 11 = ~1879
ms/chunk — IDENTICAL to standalone without the clear (~1878).** => the sub-device-manager clear is
INNOCENT.

### Full elimination (request ~3090 ms/chunk vs standalone/test ~1878 ms/chunk; gap ~1.2 s/chunk)
| hypothesis | experiment | verdict |
| mla_seq_len / KV buffer (H3) | 01/00/06/07 sweep 56320→102400 = 3089/3094/3215/3273 | REFUTED (weak ~6% secondary effect only) |
| per-layer ack synchronize_device | 04 skip_ack_sync ~3210; 08 disable_ack still mla~1840/moe~2010 | REFUTED |
| construction / config diff | 01b vs 02 construction-dump identical | REFUTED |
| request-mode sub-device clear (line 578) | 09 FORCE_PRECLEAR standalone ~1879 (unchanged) | REFUTED |
| **H2D stream service (background)** | the ONLY request-mode-only difference left; 08 (request, ack off) STILL elevated in BOTH mla & moe | **ROOT CAUSE (by elimination + corroboration)** |

### Root cause
The request-loop runner builds and runs the **H2D stream service** (`build_h2d_service`,
prefill_runner.py ~line 580) in-process for the entire prefill. Its mere presence slows EVERY
forward_chunk device op — mla and moe roughly uniformly (section timing 02/08) — adding ~1.2 s/chunk
(~40%). Standalone-chunked and the no-PCC test, which call the SAME forward_chunk but have NO service,
both run at ~1.88 s/chunk. Nothing in the model, attention length, KV buffer, sub-device manager, or
per-layer ack is responsible.

### Likely mechanism (to pin with tracy / a positive-confirmation run)
The H2D service reserves worker cores (H2D_SYNC_WORKER_CORES) and/or keeps a resident init program +
runs background socket-sync activity on the shared command queue / dispatch path. That contends with
the model's ops (fewer cores available, or host-dispatch / CQ contention), throttling every chunk.

### Fix direction (for the user)
1. Confirm directly (queued exp 10): build the service in the standalone path (unused) and check it
   slows to ~3090.
2. Reduce the service's footprint/contention: run it on a SEPARATE command queue from the model;
   shrink/relocate H2D_SYNC_WORKER_CORES off the model's compute grid; or make its background sync
   passive (event-driven, not polling) so it doesn't steal dispatch cycles while forward_chunk runs.
3. This is a RUNNER-side fix; the model (forward_chunk) is already at test speed.

---
## 10_standalone_force_service — POSITIVE CONFIRMATION of root cause: standalone-chunked @61440 + PREFILL_FORCE_BUILD_SERVICE=1 — builds the H2D stream service (UNUSED; tokens still from the local trace) in the otherwise-fast path. If total/11 slows from ~1878ms to ~3090ms, the H2D service's mere presence IS the ~1.2s/chunk request-loop cost (converts the by-elimination conclusion into a direct demonstration). If it stays ~1878ms, the service must be actively driven to cost — revisit.

### 10_standalone_force_service (standalone-chunked) — 2026-06-14 03:40:41
total + per-chunk timing:
```
2026-06-14 03:38:51.183 | INFO     | __main__:run_standalone_chunked_prefill_loop:358 - [standalone-chunked] prefilled chunk 1/11 (kv_actual=0)
2026-06-14 03:38:54.275 | INFO     | __main__:run_standalone_chunked_prefill_loop:358 - [standalone-chunked] prefilled chunk 2/11 (kv_actual=5120)
2026-06-14 03:38:57.412 | INFO     | __main__:run_standalone_chunked_prefill_loop:358 - [standalone-chunked] prefilled chunk 3/11 (kv_actual=10240)
2026-06-14 03:39:00.571 | INFO     | __main__:run_standalone_chunked_prefill_loop:358 - [standalone-chunked] prefilled chunk 4/11 (kv_actual=15360)
2026-06-14 03:39:03.838 | INFO     | __main__:run_standalone_chunked_prefill_loop:358 - [standalone-chunked] prefilled chunk 5/11 (kv_actual=20480)
2026-06-14 03:39:07.025 | INFO     | __main__:run_standalone_chunked_prefill_loop:358 - [standalone-chunked] prefilled chunk 6/11 (kv_actual=25600)
2026-06-14 03:39:10.230 | INFO     | __main__:run_standalone_chunked_prefill_loop:358 - [standalone-chunked] prefilled chunk 7/11 (kv_actual=30720)
2026-06-14 03:39:13.443 | INFO     | __main__:run_standalone_chunked_prefill_loop:358 - [standalone-chunked] prefilled chunk 8/11 (kv_actual=35840)
2026-06-14 03:39:16.697 | INFO     | __main__:run_standalone_chunked_prefill_loop:358 - [standalone-chunked] prefilled chunk 9/11 (kv_actual=40960)
2026-06-14 03:39:19.889 | INFO     | __main__:run_standalone_chunked_prefill_loop:358 - [standalone-chunked] prefilled chunk 10/11 (kv_actual=46080)
2026-06-14 03:39:23.083 | INFO     | __main__:run_standalone_chunked_prefill_loop:358 - [standalone-chunked] prefilled chunk 11/11 (kv_actual=51200)
2026-06-14 03:39:23.083 | INFO     | __main__:run_standalone_chunked_prefill_loop:361 - [standalone-chunked] 11 chunks prefilled in 34947.25 ms
```
PCC (record-only):
```
  PREFILL_STANDALONE_CHUNKED_PCC      = 0.88
  PREFILL_REQUEST_LOOP_PCC            = 0
2026-06-14 03:38:48.124 | INFO     | __main__:main:587 - Setup complete, running standalone chunked-prefill loop (golden KV-cache PCC check)
```

## ANALYSIS (2026-06-14 03:46) — H2D service DIRECTLY CONFIRMED
10_standalone_force_service (standalone-chunked + PREFILL_FORCE_BUILD_SERVICE=1): building the H2D
stream service — UNUSED, tokens still from the local trace, no producer/socket push — slowed the
standalone path from ~1878 to **34947ms/11 = ~3177 ms/chunk**, matching (slightly exceeding) the
request loop's ~3090. The service's MERE PRESENCE reproduces the full ~1.2s/chunk gap. Root cause
confirmed both by elimination AND by direct positive demonstration. Clean run, 0 crash markers.

## FINAL SUMMARY (2026-06-14)

### Question
Kimi K2.6 chunked prefill: the request-loop runner prefilled a 5120-tok chunk in ~3.3s while the
no-PCC transformer test did the SAME 61-layer chunk in ~1.94s — a ~1.2-1.4s constant per-chunk gap.

### Answer
**The H2D stream service running in the request-loop process is the cause.** Both paths call the same
`TtPrefillTransformer.forward_chunk`; the only thing that matters is whether the H2D service is present.
Its presence slows every forward_chunk device op ~40% (uniformly across MLA and MoE) — reserved worker
cores / resident init program / background socket-sync contending on the shared command queue / host
dispatch path. The model is NOT the problem; the fix is runner-side.

### All experiments (per-chunk mean, 5120 tok, 61 layers, DEVICE_FP32, mesh 8x4)
| exp | config | ms/chunk | takeaway |
|-----|--------|----------|----------|
| no-PCC test | transformer test (no service) | ~1940 | fast-path reference |
| 01a | standalone-chunked, 61440, NO service | ~1878 | == test → model is fast |
| 09 | standalone + FORCE_PRECLEAR (sub-dev clear) | ~1879 | sub-device clear INNOCENT |
| 10 | standalone + FORCE_BUILD_SERVICE (unused) | **~3177** | **service presence = the gap (CONFIRMED)** |
| 00 | request-loop, 61440 | ~3094 | baseline (service present) |
| 01 | request-loop, 56320 (=test buffer) | ~3089 | H3 mla_seq_len REFUTED |
| 06 | request-loop, 81920 | ~3215 | weak ~6% buffer effect only |
| 07 | request-loop, 102400 | ~3273 | weak ~6% buffer effect only |
| 04 | request-loop, SKIP_ACK_SYNC | ~3210 | ack-sync REFUTED |
| 02/08 | request section-timing (ack on/off) | mla~1840 moe~2010 | both elevated w/ or w/o ack |
| 01b/03 | standalone/test section-timing | mla~1394 moe~1668 | match each other |

### Fix direction (runner-side)
1. Run the H2D service on a SEPARATE command queue from the model (num_command_queues=2).
2. Shrink/relocate H2D_SYNC_WORKER_CORES so the service's cores don't overlap the model compute grid.
3. Make the service's background sync passive/event-driven (not polling) so it doesn't steal dispatch
   cycles while forward_chunk runs.
Profile with tracy (signposts already present) to pin which of cores/CQ/host-dispatch dominates.

### Artifacts
- Instrumentation + probes committed on branch `ppopovic/investigation`:
  d0abf44ed8f (perf_probe + section/construction dumps), 3b63b010766 (FORCE_PRECLEAR),
  0900424df25 (FORCE_BUILD_SERVICE), c708cb07a4d (RUNNER_PERF_INVESTIGATION.md conclusion).
- Harness: /home/ppopovic/kimi_perf_overnight (orchestrator.sh, lib.sh, queue/→done/, logs/).
- Raw per-iter logs: logs/<exp>.{runner,producer,test}.log.

### Operational notes from the night
- Device wedged ~19:58 after my mid-init `kill -KILL` (chip-lock/ethernet) while co-user `asaigal`
  was active; paused ~5.5h (did NOT reset while co-user present), reset with tt-smi -glx_reset_auto at
  01:45 once asaigal's device work ended, resumed. ETH_LIVE_STATUS 0x0 is normal (not a wedge signal).
- Two orchestrator bugs found+fixed: (a) cleanup must wait for FULL process reap (zombie holds sysmem);
  (b) stale /dev/shm/tt_prefill_layer_acks shm after SIGKILL broke the next request-loop run (now
  rm'd between experiments); (c) nullglob empty-queue made `ls queue/*.exp` list cwd (now uses find).

---
## 11_fix_offgrid_col11 — THE FIX: request-loop with H2D service worker core moved to col 11 (off the model's 0-10 compute grid; default now). If per-iter drops from ~3090 to ~1900ms, the persistent service kernel colliding with the model grid was the cause and this fixes it.

### 11_fix_offgrid_col11 (runner) — 2026-06-14 08:16:01
per-iter pipeline.prefill() ms:
```
3025.26 ms
3080.77 ms
3142.60 ms
3167.42 ms
3202.98 ms
3288.90 ms
3184.45 ms
3221.54 ms
3230.85 ms
3241.06 ms
3202.85 ms
```

---
## 12_repro_col0 — Control: force the OLD in-grid worker core (0,0) via PREFILL_H2D_WORKER_COL=0. Should reproduce ~3090ms, confirming the knob works and the col-11 fix (exp 11) is what changed it.

### 12_repro_col0 (runner) — 2026-06-14 08:20:17
per-iter pipeline.prefill() ms:
```
3047.67 ms
3062.24 ms
3137.50 ms
3189.17 ms
3194.49 ms
3319.49 ms
3248.87 ms
3277.51 ms
3256.91 ms
3246.72 ms
3252.28 ms
```

## FIX ATTEMPT 1 (2026-06-14 08:24) — core relocation: NEGATIVE

Hypothesis: the H2D service's persistent receiver kernel pinned on core (0,0) — inside the model's MoE
compute grid (cols 0-10, rows 0-9) — contends for that core. BH compute grid is 12x10, so col 11 is free.
- 11_fix_offgrid_col11 (service core → (11,0), off-grid): mean ~3190 ms/chunk (ran clean, 11 prefill lines)
- 12_repro_col0 (service core → (0,0), in-grid): mean ~3201 ms/chunk
=> NO difference, both ≈ broken baseline (~3090). **Core placement is NOT the mechanism.** The persistent
service kernel occupying a compute core is not what slows the model.

### Remaining mechanism (needs profiling, not more blind attempts)
The overhead is roughly proportional to op count in BOTH mla and moe sections (02 vs 01b) → a per-dispatch
/ sub-device-coexistence cost: when the H2D service's persistent program is resident, every model program
launch pays extra (dispatcher must coordinate the service's resident program / sub-device / CQ state).
This is NOT addressable by core placement, and the socket-buffer-in-L1 angle can't explain a uniform ~40%
slowdown (it is one core's L1). Ruled-out-cheaply candidates are exhausted.

### Recommended next step (for the user — needs tracy, not autonomous guessing on a shared box)
1. tracy profile exp 10 (standalone + FORCE_BUILD_SERVICE) vs 01a (standalone, no service): diff per-op
   DEVICE time vs inter-op GAPS. Gaps growing => host/dispatch coordination cost (the resident service
   program tax); per-op device time growing => on-device (NOC/L1/fabric) contention.
2. Likely fixes depending on profile: run the H2D service + model on SEPARATE command queues; or place the
   service's persistent program on a sub-device that the model's per-op dispatch does not have to
   re-validate/coordinate each launch; or tear down/suspend the service program during forward_chunk.
   Some of these are H2DStreamService (C++) changes — out of scope for autonomous Python-only edits.
3. The diagnostic knob PREFILL_H2D_WORKER_{COL,ROW} (default 0,0 = upstream) is committed for future use.

## TRACY PROFILE (2026-06-14 08:54) — VERDICT: dispatch tax, NOT on-device contention
1-layer (NCHUNKS=3) device perf reports, no-service vs service (PREFILL_FORCE_BUILD_SERVICE=1), sums:
| metric (5248 ops) | noservice | service | delta |
| DEVICE FW DURATION   | 961.6 ms  | 935.5 ms  | ~0 (equal) |
| DEVICE KERNEL DURATION | 741.9 ms | 745.9 ms | ~0 (equal) |
| OP-TO-OP LATENCY (gaps) | 38470 ms | 51336 ms | **+12866 ms (+33%)** |
On-device compute is IDENTICAL with/without the service; the slowdown is ENTIRELY inter-op latency,
spread UNIFORMLY across all ops (~+2.45us/op; OP NAME blank in this report mode, so it's not one op
type). => The resident H2D service program imposes a per-op-launch DISPATCH/host-coordination tax. At
61 layers this compounds to the ~1.2s/chunk gap. NOT a NOC/L1/fabric (on-device) contention.

### Fix that follows
Decouple the service's program from the model's per-op dispatch. Candidates, in order:
1. Separate command queue: open the mesh with num_command_queues=2 so the H2D service claims its own CQ
   (the runner's dtor comment "frees a command queue" implies it grabs one) and the model's CQ0 dispatch
   is uncontended. Python-feasible if open_mesh_device exposes num_command_queues — TESTING THIS NEXT.
2. If that doesn't help: the fix is at the H2DStreamService (C++) level — make its persistent program
   not participate in the fast-dispatch coordination of other programs (own sub-device / dispatch domain).

---
## 13_fix_2cq — FIX ATTEMPT 2 (dispatch-tax): full 61-layer request-loop with PREFILL_NUM_CQ=2 — open the mesh with 2 command queues so the H2D service claims its own CQ and the model's CQ0 dispatch is uncontended. If per-iter drops from ~3090 to ~1900ms, the separate-CQ decoupling is the fix. If unchanged or errors, the fix is H2DStreamService C++-level (document + stop).

### 13_fix_2cq (runner) — 2026-06-14 09:02:46
per-iter pipeline.prefill() ms:
```
3107.15 ms
3134.99 ms
3197.96 ms
3222.49 ms
3282.91 ms
3308.30 ms
3264.40 ms
3262.28 ms
3293.02 ms
3297.33 ms
3278.84 ms
```

## TRACY WARM (2026-06-14 09:17) — dispatch tax CONFIRMED on warm (post-compile) ops
Re-ran with ITERS=3 (pass1 compiles, passes 2-3 warm), binned the device perf rows by GLOBAL CALL COUNT
(warm = high count). Per-op medians:
| bin                | FW (device compute) median | OP-TO-OP LATENCY median |
| WARM no-service    | 55.4 us                    | 135.7 us |
| WARM service       | 55.3 us                    | 429.3 us (~3.2x) |
| (cold/compile bin) | 48 us                      | op2op mean ~10 ms (compile) |
=> On fully WARM ops: on-device compute IDENTICAL; the H2D service ~3x the op-to-op latency. The earlier
contamination concern (each chunk compiles a unique logical_n) is resolved — the cold bin held the
compile gaps; the warm bin shows the tax PERSISTS. (Absolute us are tracy-inflated/sampled — the ratio
and the FW-equality are the robust findings.) STEADY-STATE per-op-launch DISPATCH tax, not compile, not
on-device.

## FINAL STATUS (2026-06-14) — root cause diagnosed; fix is tt-metal/H2DStreamService (C++)
ROOT CAUSE: the H2D stream service's resident program imposes a per-op-launch DISPATCH/host-coordination
tax on every model op (op-to-op latency ~3x, on-device compute unchanged), compounding to ~1.2s/chunk
(~40%) over 61 layers. Standalone/test (no service) run at ~1.88s/chunk; request-loop ~3.09s/chunk.
PYTHON FIXES TRIED — BOTH NEGATIVE:
  - core relocation (service worker core (0,0)->(11,0), off the model grid): ~3190 vs ~3201ms (no change).
  - separate command queue (PREFILL_NUM_CQ=2): ~3241ms (no change).
=> The fix is NOT reachable from the runner (Python). It requires H2DStreamService (C++): give its
resident receiver program its OWN dispatch domain / sub-device so it is excluded from the fast-dispatch
launch coordination of other (model) programs — OR suspend/idle the service program while forward_chunk
runs. Hand to a tt-metal dispatch engineer with evidence: kimi_perf_overnight/devperf_{noservice,service}.csv
(per-op DEVICE FW DURATION vs OP TO OP LATENCY, warm bin). Diagnostic knobs committed (default-safe):
PREFILL_FORCE_BUILD_SERVICE, PREFILL_H2D_WORKER_COL/ROW, PREFILL_NUM_CQ, PREFILL_STANDALONE_CHUNKED_ITERS.
OPERATIONAL: the box disk hit 100% (tracy raw logs ~4.7GB/run on an already-near-full /); cleaned my raw
logs. Avoid repeated tracy captures here without disk headroom.

## CHUNK PROFILE 1st vs last (2026-06-14 09:41) — 1 layer, WARM (no compile)
Profiled one layer's forward_chunk WARM (ITERS=2, measured pass 2) at kv_actual=0 (logical_n=5120, "1st
chunk") vs kv_actual=51200 (logical_n=56320, "last chunk"). No H2D service. 1968 device ops each.
| metric (warm)        | 1st (logical_n 5120) | last (logical_n 56320) | ratio |
| DEVICE FW sum        | 388 ms               | 1940 ms                | 5.0x  |
| DEVICE KERNEL sum    | 310 ms               | 1060 ms                | 3.4x  |
| DEVICE KERNEL median | 45.8 us              | 45.7 us                | ~1.0  |
| heaviest single op   | ~2.8 ms              | ~14.1 ms               | ~5x   |
| OP-TO-OP LATENCY sum | 553 ms               | 371 ms                 | flat/lower |
(NB: per-op DEVICE-time SUMS overlap across the 32-chip mesh + async pipeline, so they are NOT wall
clock; the single-op kernel durations and the ratios are the robust signal.)

FINDING: the within-chunk compute ramp is the MLA ATTENTION scaling with the KV prefix.
- The MEDIAN kernel duration is IDENTICAL (46 us) for 1st vs last → MOST ops scale with chunk_size
  (5120), not prefix: matmuls (q/kv proj, wkv_b2), all_gathers, RotaryEmbeddingIndexed, nlp_create_qkv_heads.
- ONE dominant op grows ~5x (2.8 -> 14.1 ms): the `sdpa` op (ring_mla scaled-dot-product attention), which
  reads the [0, logical_n) KV window. logical_n grew 11x (5120->56320); the op grew ~5x (sublinear: SDPA
  has a fixed setup + linear-in-KV term). Named zones confirm: sdpa, MatmulDeviceOperation, AllGather*,
  RotaryEmbeddingIndexed, nlp_create_qkv_heads, update_padded_kv_cache.
- OP-TO-OP LATENCY (dispatch gaps) does NOT grow with prefix → dispatch is per-op-launch, prefix-independent.

INTERPRETATION: this is the EXPECTED, correct within-run ramp (request loop ~2950->3250ms across chunks)
— attention must read the growing KV cache. It is ORTHOGONAL to the H2D-service dispatch tax (the constant
~1.2s/chunk runner-vs-test offset, which is prefix-independent). Two distinct effects:
  (a) per-chunk ramp = MLA sdpa attention vs logical_n (device compute; correct/unavoidable, ~linear in KV);
  (b) runner-vs-test constant offset = H2D service per-op-launch dispatch tax (needs the C++ fix).

## 3-LAYER standalone (2026-06-14 11:06) — per-layer, slowest device, WARM (no service)
Slowest device, warm pass (op2op = NON-compile gaps). kernel_sum / op2op_sum per layer:
| layer | FIRST (logical_n 5120) kernel / op2op | LAST (logical_n 56320) kernel / op2op |
| layer0 DENSE | 6.13ms / 8.73ms  | 17.44ms / 9.64ms |
| layer1 MoE   | 25.27ms / 13.88ms | 35.56ms / 3.45ms |
| layer2 MoE   | 24.70ms / 5.77ms  | 38.12ms / -0.06ms |
Findings:
- RingJointSDPA (attention) scales with prefix: ~2.9ms (first) -> ~14.0ms (last), same in all 3 layers.
- MoE FFN is PREFIX-INDEPENDENT: UnifiedRoutedExpertFfn x12 ~8-9ms, CombineDeviceOperation ~3-4ms,
  TilizeDeviceOperation ~2.1ms, UnaryDeviceOperation ~1.5ms, DispatchDeviceOperation ~1.8-2ms,
  ReduceScatter ~3ms, MoeGroupedTopk(gate) ~0.07ms, MaskedBincount ~0.09ms — all ~same first vs last.
- MoE layers (~25-38ms kernel) are ~2x the dense layer; the extra ~20ms is the expert FFN + combine/dispatch.
- WARM op2op gaps are concentrated in the MLA-front CCL ops (AllGather ~1.9ms gap, gap-before-SDPA
  ~1.5-2ms = cross-chip sync wait) — the MoE expert ops stream with ~0.5-7us gaps. So warm dispatch gaps
  are dominated by cross-chip CCL synchronization, NOT the per-op dispatch. Full per-op tables:
  ops_slowest_device_3L_{first,last}.log. NEXT: with-service variant to see if these warm gaps grow.

## 3-LAYER WITH SERVICE vs standalone (2026-06-14 11:20) — warm op2op dispatch tax confirmed per-layer
Slowest device, WARM pass. kernel sums ~EQUAL (device compute unchanged); warm OP2OP gaps GROW with the service.
| chunk/layer | kernel std->svc (ms) | op2op std->svc (ms) | op2op delta |
| FIRST L0 dense | 6.13->6.20  | 8.73->20.56  | +11.8 |
| FIRST L1 MoE   | 25.27->23.68| 13.88->31.91 | +18.0 |
| FIRST L2 MoE   | 24.70->24.80| 5.77->30.41  | +24.6 |
| LAST  L0 dense | 17.44->16.99| 9.64->14.77  | +5.1  |
| LAST  L1 MoE   | 35.56->34.77| 3.45->11.12  | +7.7  |
| LAST  L2 MoE   | 38.12->38.62| -0.06->16.19 | +16.3 |
VERDICT: kernel/device time identical with vs without the service (within noise); the H2D service inflates
WARM (non-compile) op-to-op gaps by +5 to +25 ms PER LAYER — the per-op-launch DISPATCH TAX, now confirmed
at 3-layer granularity on warm ops. Which ops carry it: the MoE expert ops UnifiedRoutedExpertFfn go from
~7us op2op (standalone, streaming) to ~2500us (service); Matmul ~1200us->~3480us; gap-before-SDPA
~1450us->~2400us. So the service adds dispatch latency BROADLY across ops (worst on the many MoE experts),
exactly the mechanism behind the ~1.2s/chunk runner-vs-test gap over 61 layers. Logs:
ops_slowest_device_3Lsvc_{first,last}.log vs ops_slowest_device_3L_{first,last}.log.

## ROOT CAUSE of op2op gaps (2026-06-14) — dispatch-pipeline-bound, masked by heavy ops
Profiled per-op host vs device timing (ops_perf_results: HOST START/END TS, HOST DURATION, DEVICE KERNEL
DURATION, OP TO OP LATENCY), warm, slowest device, layer 0 (dense), last chunk:
- BEFORE the heavy RingJointSDPA: 24 small ops (kernel 3-160us) each have op2op GAP ~100-1900us (device idle
  >> compute). AFTER SDPA: identical op types run with gap ~0.5us.
- Host CPU is NOT the bottleneck: sum HOST DURATION over layer0 = 1.9ms, but host SPAN (first->last enqueue)
  = 16.5ms ~= device layer time (17ms). The host enqueues each op in ~1-2us then sits IDLE ~370us. Host
  idle ~88%.
=> The gaps are DISPATCH-PIPELINE-bound, not host-CPU-bound and not device-compute-bound. Each small op
waits ~370us for its dispatch commands + go-signal to be processed and multicast across the 8x4 mesh (up to
32 chips x ~110 worker cores), and the device sits idle in that window. A long op (SDPA ~14ms) lets the
prefetcher/dispatcher run far ahead and queue the following ops, so their go-signals overlap execution and
the gap vanishes — exactly why gaps appear ONLY before the first heavy op in each layer.

Mechanism (from tt-metal code): op2op = device worker-core idle between kernel ZONE_END and next ZONE_START
(device_post_proc_config.py). Host races ahead filling the prefetch_q (system_memory_manager.cpp
fetch_queue_reserve_back / wait_for_fetch_q_space; depth dispatch_settings.cpp prefetch_q_entries=1534, or
256 if DRAM-backed). Per-op dispatch assembles RTAs + launch msgs + go-signal with wait_count and writes
per-LOCAL-device (fd_mesh_command_queue.cpp write_program_cmds_to_subgrid -> up to 32 serial writes) +
write_go_signal_to_unused_sub_grids over all devices. The resident H2D service adds worker/sub-device
bookkeeping per dispatch -> measured ~3x the warm gaps (op2op_summary.md).

### FIX (primary): Metal Trace (capture/replay)
EnqueueTrace replays a pre-recorded command buffer via a single add_prefetch_exec_buf per device
(impl/trace/dispatch.cpp), eliminating per-op RTA/command assembly + per-chip CQ writes + serialized
go-signal issue. The device is fed from DRAM at full rate -> op2op collapses toward ~0, which removes the
bulk of the pre-SDPA idle AND the H2D-service dispatch tax. ttnn API: begin_trace_capture/end_trace_capture
/execute_trace (ttnn/cpp/ttnn/operations/trace.cpp). Caveat: requires static shapes/RTAs (each chunk's
logical_n differs -> one trace per distinct chunk shape, or pad to a fixed logical_n).
Secondary levers: 2nd command queue; verify prefetch_q isn't in the 256-entry DRAM-backed regime;
reduce sub-device-manager churn around the H2D service.
TO PIN the exact stall component: re-profile with `--profile-dispatch-cores` -> DISPATCH GO SEND WAIT TIME
attributes the ~370us between go-signal send and worker completion.

## OBSERVER EFFECT TEST (2026-06-14 12:36) — gaps are REAL, not profiler artifact
1-layer warm chunk (PROFILE_KV=51200), plain python, device profiler OFF vs ON, warm passes 2-6 mean wall:
  OFF (no TT_METAL_DEVICE_PROFILER) = ~24.8 ms   |   ON (=1) = ~24.5 ms   => OFF ~= ON.
=> The device profiler adds ~0 wall time; the op2op gaps are NOT observer-effect. A 1-layer warm chunk is
~24.7ms: SDPA kernel ~14ms + small-op kernels ~1.5ms + ~9ms (37%) of REAL op-to-op DISPATCH IDLE. The
~370us/op gaps before the heavy SDPA are genuine (standalone, no service, no profiler). Confirmed worth fixing.
Next: the host is idle ~88% (enqueues each op ~1-2us then blocks ~370us) => per-op dispatch backpressure;
checking async-dispatch / CQ / dispatch-core config and mesh go-signal cost.

## DISPATCH GAP — CONCLUSION (2026-06-14 12:46)
THE PROBLEMATIC THING: the prefill runs MANY small ops (~40 device ops/dense layer, more per MoE layer;
~2400+ ops per 61-layer chunk), and on the 8x4 (32-chip) mesh EACH op pays ~370us of per-op dispatch /
go-signal + worker-completion latency that the device cannot hide because the small ops (~30us kernel)
don't let the dispatcher run ahead. Result: ~9ms (37%) of a 1-layer warm chunk is device IDLE between ops.
Evidence (all standalone, no service):
- profiler OFF wall ~24.8ms == ON ~24.5ms  => NOT a profiler artifact (real).
- host CPU idle ~88% (enqueue 1-2us/op then block) => NOT host-CPU; no enable_async toggle (already async).
- gaps ~370us/op before the heavy RingJointSDPA, collapse to ~0.5us after it (dispatcher ran ahead during
  the 14ms op) => classic run-ahead-starved small-op dispatch.
- ~370us/op is ~10x typical single/small-mesh dispatch (~10-50us/op) => the cost is the 32-chip mesh
  go-signal + cross-chip completion round-trip (1D fabric, 8-long SP axis).
FIXES (ranked):
  (a) 2nd command queue (PREFILL_NUM_CQ=2): NO effect (~25.0ms vs ~24.7ms). Config knob does NOT fix it.
  (b) Metal Trace (capture/replay): the PURPOSE-BUILT fix for exactly this (small ops can't saturate
      dispatch). Replay issues one add_prefetch_exec_buf/device -> per-op dispatch eliminated -> the ~9ms/
      layer gap AND the H2D-service dispatch tax both collapse. User wants it "at a distance", but it is
      the architecturally-correct answer and likely ~halves standalone time. Caveat: static shapes (1 trace
      per logical_n or pad).
  (c) Op fusion (fewer, bigger ops in the model) — reduces op count so dispatch is hidden; DEEP model change.
  (d) tt-metal mesh-dispatch optimization (batch/parallelize the go-signal + completion across chips) —
      C++ change in tt_metal/distributed/fd_mesh_command_queue.cpp (go-signal path) + the worker-completion
      gate in tt_metal/impl/dispatch/kernels/cq_dispatch.cpp. Not attempted autonomously.
VERDICT: no config-level fix exists; the per-op multi-chip dispatch gap is fundamental for this many small
ops. Practical path = Metal Trace for the prefill chunk (recommended), or op-fusion, or a tt-metal go-signal
optimization. The ~370us is genuine and is the dominant non-compute cost of the chunk.

## GO-SIGNAL FIX ATTEMPT (2026-06-14 13:05) — host fan-out parallelization: NEGATIVE (reverted)
Parallelized write_program_cmds_to_subgrid (per-chip command writes across dispatch_thread_pool_) +
skip-unused-subgrid. Result: 1-layer warm chunk WORSE — ~58ms (passes 2-6: 72/51/56/54/56) vs ~24.7ms
baseline (~2.3x slower), no crash. => The serial 32-chip host fan-out was NOT the bottleneck (it was cheap
~47us/op); the thread-pool enqueue/wait added ~825us/op overhead. CONFIRMS the ~370us/op gap is DEVICE-SIDE
(go-signal mcast + worker-completion, per-chip-local on this all-MMIO mesh), NOT host fan-out. A host-side
dispatch optimization cannot fix it. Reverted fd_mesh_command_queue.cpp. Remaining non-Trace levers: the
device dispatch KERNEL (high risk) or the user's 4x8 mesh-axis-swap (may help the CCL-op gaps, which are the
largest). Next: evaluate 4x8 swap.

## 4x8 FALLBACK — feasibility (2026-06-14 13:10)
Blocked/longshot: (1) weight cache is keyed to mesh shape (resolve_weight_cache_path -> .../32dev/{sp}x{tp}/
= 8x4); a 4x8 layout assigns shards to different physical chips -> needs a full cache REBUILD from HF
(slow, disk-prohibitive at ~10G free). (2) dispatch is per-chip-LOCAL on this all-MMIO mesh (go-signal+
completion local, no fabric) -> axis swap won't change the bulk per-op gap; only the cross-chip CCL ops
(minority) could benefit. Cheap feasibility test possible: symlink .../4x8 -> .../8x4 + run 4x8 PCC (if PCC
passes, cache is layout-compatible and we can measure; if it fails, 4x8 needs a cache rebuild). Awaiting user
decision: (a) run the cheap symlink 4x8 PCC test, (b) pursue device-dispatch-kernel optimization (deep C++),
(c) accept Metal Trace (the purpose-built fix).

## DISPATCH SYNC PROBE (2026-06-14 13:34) — per-op stall source pinned
TT_DISPATCH_SYNC_DEBUG=1 on 1-layer warm chunk (compile+2 passes, ~120 ops): 245 forced kernel-config
syncs (~2/op). Breakdown:
  111 BYTE-WRAP idx0 (buf_bytes=70656) -> TENSIX kernel-config ring (~69KB free L1) fills with a layer's
      ~40 ops of CB/RTA/binary configs.
   72 TABLE-FULL idx0 -> kernel-config 8-entry table (kernel_config_entry_count) exhausts (7 in-flight).
   62 BYTE-WRAP idx3 (buf_bytes=7) -> launch-message ring (launch_msg_buffer_num_entries-1=7).
=> The ~370us/op gap is the dispatcher forced to STALL on worker-completion ~twice per op because the
kernel-config ring (idx0) both byte-wraps (69KB too small) AND table-exhausts (8 entries), plus the
launch-msg ring (idx3, 7) wraps. Levers: (a) kernel_config_entry_count 8->32 [host, safe] removes the 72
table-full; (b) launch_msg_buffer_num_entries 8->N [device, risky] reduces the 62 idx3; (c) the 111 idx0
byte-wraps are L1-size-limited (69KB ring vs working set) -> need more free L1 or smaller configs (model
L1 budget, risky). Starting with (a), measuring, then (b).

## DISPATCH FIX RESULT (2026-06-14 13:50) — bottleneck pinned; easy lever perf-neutral
kernel_config_entry_count 8->32: removed all 72 TABLE-FULL syncs (245->173, -29%), but 1-layer warm chunk
wall UNCHANGED (~25.0ms vs ~24.7ms baseline). => PERF-NEUTRAL. Reason: an op stalls if ANY buffer needs a
sync, and the 111 idx0 kernel-config-ring BYTE-WRAPs hit ~every op (111 over ~120 ops) regardless of the
table; removing the overlapping table-full syncs unstalls no ops. BINDING CONSTRAINT = the 69KB TENSIX
kernel-config L1 ring byte-wrapping because a layer's ~40 ops of CB/RTA/binary configs exceed 69KB free L1.
The launch-msg ring (idx3, 62) is secondary and also non-binding while idx0 dominates.
REAL FIX requires enlarging the kernel-config L1 ring (ring = worker_l1_unreserved_start - KERNEL_CONFIG_base
~ 69KB free) — i.e. reduce the model's per-op L1 footprint or move HAL L1 layout to give the config ring
more room. That is a deep model-L1-budget / HAL-firmware change, model-specific and risky. OR Metal Trace,
which eliminates per-op dispatch entirely (collapses ALL these syncs). The host table bump and the launch-
ring device bump do NOT move wall time here. Validating table=32 correctness (PCC) before deciding to keep.

## OPTION B CONCLUSION (2026-06-14 14:04)
PINNED: the per-op ~370us device-idle dispatch gap = the dispatcher forced to sync on worker-completion
~every op because the ~69KB TENSIX kernel-config L1 ring byte-wraps (a layer's ~40 ops of CB/RTA/binary
configs exceed the free L1 left after the model's tensors). Probe: 245 syncs/120ops (111 ring byte-wrap +
72 table-full + 62 launch-ring byte-wrap).
TRIED: kernel_config_entry_count 8->32 (committed a8239c53ea2) — removes the 72 table-full syncs (245->173),
PCC 0.965>=0.88 (correct), but PERF-NEUTRAL (~1863 vs ~1878 ms/chunk at 61 layers; 1-layer warm ~25 vs
24.7ms) because the 111 ring byte-wraps stall ~every op regardless. The launch-msg ring (device constant,
62 syncs) is non-binding while the ring byte-wrap dominates -> not worth a risky firmware change.
REAL FIX (needs human decision, not attempted autonomously - high blast radius):
  (a) Enlarge the kernel-config L1 ring so a layer's configs don't byte-wrap: reduce the model's per-op L1
      footprint, or adjust the HAL KERNEL_CONFIG base / worker_l1_unreserved layout to give the ring more
      room. Deep, model-specific, risks L1 collisions.
  (b) Metal Trace (capture/replay): eliminates per-op dispatch entirely -> collapses ALL these syncs AND
      the H2D-service dispatch tax. Architecturally correct; the biggest, safest win. (User wanted it at a
      distance; it is the recommended path.)
SUMMARY of the whole investigation: runner ~3.3s vs test ~1.9s/chunk had TWO independent causes —
(1) the H2D stream service's per-op dispatch tax (the ~1.2s constant offset), and (2) the within-chunk
per-op dispatch stalls (the kernel-config ring byte-wrap, ~370us/op, present even standalone). Both are
per-op dispatch overhead that Metal Trace eliminates. Diagnostic knobs committed: PREFILL_FORCE_BUILD_SERVICE,
PREFILL_H2D_WORKER_COL/ROW, PREFILL_NUM_CQ, PREFILL_STANDALONE_CHUNKED_ITERS, PREFILL_PROFILE_KV,
TT_DISPATCH_SYNC_DEBUG.
