# AutoDebug: exact-final fused CCL boundary experiment

## Headline finding

The existing fused result is not sufficient to accept or reject the fused residual family. `test_fused_matmul_ccl_fractured_residual_chain` measures only down projection -> reduction -> distributed norm -> QKV (`tests/test_multichip_decoder.py:1439-1681`). Both its baseline and candidate omit the residual add that exists after the down reduction (`tt/multichip_decoder.py:1375-1380`), and it does not measure the independent WO -> reduction -> residual add -> post-attention norm -> gate/up boundary (`tt/multichip_decoder.py:1362-1375`). Its successful `fused_interleaved` control also uses ordinary `all_reduce`, DRAM-interleaved 2D matmuls, and BF16 payloads, whereas the final default uses DRAM-sharded BFP4/LoFi matmuls plus persistent `all_reduce_async(... dtype=bfloat8_b)` (`tt/multichip_decoder.py:772-787`). The observed 8% isolated win therefore does not compare the material family against the final default.

The exact final DRAM-sharded matmul configs cannot be passed directly to fused MRS: `MatmulReduceScatterAsync` accepts only `MatmulMultiCoreReuseMultiCastProgramConfig` (2D multicast), while `AllGatherMatmulAsync` accepts only 1D/2D multicast. The tuned-blocker tests correctly expose this API restriction (`matmul_reduce_scatter_async_device_operation.cpp:37-47`, `all_gather_matmul_async_device_operation.cpp:46-66`). The focused test must retain exact-final controls and use a coherent 2D-multicast/interleaved-weight candidate, not restore the replicated contract immediately after MRS.

## Smallest correct experiment

Add one gated test, `test_fused_matmul_ccl_exact_final_boundaries`, alongside the current harness. Load `_real_state()` once and instantiate the decoder with its actual defaults: BFP4 weights, LoFi kernels, persistent BFP8 CCL, DRAM-sharded decode matmuls. Generate deterministic **real activations** by running `_hf_layer(state, config)` through `_reference_layer` once and capturing forward-pre-hook inputs to `self_attn.o_proj`, `post_attention_layernorm`, and `mlp.down_proj`. This supplies the actual WO input, attention residual, post-attention residual, and down input from the real layer rather than independent random tensors.

Measure four separately captured/replayed traces with identical warmup and iteration counts:

1. **A control, exact final default**
   - actual local WO: `decode_o_input_mem_config`, `output_weight`, `decode_o_program_config`, BFP4/LoFi;
   - `to_memory_config(... decode_mlp_output_mem_config)` then the decoder's persistent `all_reduce_async`, explicitly assert `collective_dtype == bfloat8_b`;
   - BF16 residual add in `decode_norm_mem_config`, `post_attention_norm` RMSNorm;
   - actual separate local `gate_weight` and `up_weight` with the same configs used by `_mlp_forward`;
   - return both gate and up tensors for per-rank PCC.

2. **A candidate, fractured residual**
   - WO + `matmul_reduce_scatter_async(dtype=bfloat8_b)` using the same real BFP4 WO values in an API-compatible DRAM-interleaved tensor and a 2D multicast config;
   - add the matching rank-local BF16 residual shard, keeping `[1,1,32,1280]` fractured;
   - distributed post-attention RMSNorm: `rms_norm_pre_all_gather`, gather only the small BF16 stats, then `rms_norm_post_all_gather` with rank-local gamma;
   - typecast the normalized shard to BFP8, then fuse its gather into the actual gate projection with `all_gather_matmul_async(dtype=bfloat16)`; use the returned gathered BFP8 activation for the actual separate up matmul. This performs one activation gather, not two, and returns actual gate/up results.

3. **B control, exact final default**
   - actual local down projection with `down_weight`, final DRAM-sharded down config, BFP4/LoFi;
   - persistent BFP8 all-reduce, BF16 add with the captured post-attention residual;
   - actual input RMSNorm and actual QKV projection with `qkv_weight` and final DRAM-sharded config;
   - return QKV for per-rank PCC.

4. **B candidate, fractured residual**
   - down + BFP8 MRS using the real BFP4 down values in API-compatible interleaved storage and a 2D multicast config;
   - add the matching rank-local post-attention residual shard;
   - distributed input RMSNorm, BFP8 typecast, then `all_gather_matmul_async` into the actual real BFP4/LoFi QKV projection;
   - return QKV for per-rank PCC.

Open the fixture with Ring-capable fabric, but do **not** change the control policy: the exact-final control remains `Topology.Linear`, two links, persistent BFP8 all-reduce, while MRS/AGMM candidates use their required `Topology.Ring`, one link. Give the candidate the same full worker subdevice, offsets, semaphore counts, barrier, `chunks_per_sync=10`, `num_workers_per_link=2`, and `num_buffers_per_channel=2`. Preallocate BFP8 MRS intermediate/output and AG buffers before compile/capture. Assert the control's all-reduce overload, workspace identity, dtype, Linear/two-link topology, weight dtype, compute fidelity, and program-config class so an accidental Ring/general/BF16 control cannot masquerade as the final default. If this checkout cannot issue Linear CCL on a Ring-capable fabric session, split control and candidate into two identically shaped tests using `DEVICE_PARAMS` and `RING_DEVICE_PARAMS`; never relabel a Ring/one-link control as exact-final.

The decision is based on direct sums:

```text
control_sum_ms = A_control_ms + B_control_ms
fused_sum_ms   = A_fused_ms   + B_fused_ms
```

Require gate, up, and QKV candidate/control PCC at the accepted fused threshold (at least 0.999 per rank), report each boundary ratio and the summed ratio, and print all logical/local shapes, dtypes, memory configs, program-config classes, and buffer bytes. If `fused_sum_ms` is slower, the family is rejected without a combined trace. If it wins or is within repeated-run noise (use <=1% as the trigger, then repeat 3x), add one combined trace chaining gate/up -> SwiGLU -> down MRS -> residual add -> distributed input norm -> QKV; compare it with the exact-control combined chain before changing production code.

## Concrete configs and byte ledger

Reuse the already-running interleaved configs as the first API-compatible candidates: 2D `(8,6)`, `per_core_M=1`; WO/down use `per_core_N=20`, QKV uses `per_core_N=6`. Gate/up `[5120,8192]` needs `per_core_N=32` across eight N columns; start with `in0_block_w=4`, `out_subblock_h=1`, `out_subblock_w=2`, `transpose_mcast=False`. If validation/L1 rejects it, adapt `in0_block_w`/subblock width while keeping the same semantic chain; a first TTNN error is not a rejection.

Tile payload sizes are BF16=2048 B, BFP8=1088 B, BFP4=576 B. Per device/rank at batch 32:

| Object | Local logical shape | Layout/dtype | Exact payload bytes |
|---|---:|---|---:|
| full residual / partial hidden | `[1,1,32,5120]` | tile BF16 | 327,680 |
| fractured residual | `[1,1,32,1280]` | tile BF16 | 81,920 |
| BFP8 MRS full intermediate | `[1,1,32,5120]` | tile BFP8 | 174,080 |
| BFP8 MRS output | `[1,1,32,1280]` | tile BFP8 | 43,520 |
| gathered norm activation | `[1,1,32,5120]` | tile BFP8 | 174,080 |
| local/gathered RMS stats | `[1,1,32,32]` / `[1,1,32,128]` | tile BF16 | 2,048 / 8,192 |
| WO input / QKV output | `[1,1,32,1024]` / `[1,1,32,1536]` | tile BF16 | 65,536 / 98,304 |
| gate or up/down activation | `[1,1,32,8192]` | tile BF16 | 524,288 |
| WO weight | `[1024,5120]` | tile BFP4 | 2,949,120 |
| QKV weight | `[5120,1536]` | tile BFP4 | 4,423,680 |
| gate/up/down weight (each) | `[5120,8192]` or `[8192,5120]` | tile BFP4 | 23,592,960 |
| final persistent AR workspace | `[1,1,32,20480]` | L1 width-sharded BF16 | 1,310,720 |

These are packed tile payload bytes, excluding allocator alignment/metadata. Record `buffer_address`, `memory_config`, and `padded_shape` at runtime to prove preallocation and expose any additional padding.

## Likely pitfalls

- Do not compare the fused candidate to its own interleaved baseline; controls must call the exact persistent BFP8 path and final DRAM-sharded programs.
- MRS `dtype` controls the matmul output feeding reduce-scatter. Its persistent intermediate and output buffers must therefore also be BFP8 for the requested final-precision family.
- AGMM has one weight input. For separate gate/up, fuse the gather into gate and feed AGMM's returned gathered activation to ordinary up matmul; two AGMM calls would duplicate the material gather.
- Slice residuals by mesh mapping (`ShardTensorToMesh(dim=3)`), not a per-device host loop whose rank ordering can drift. Shard RMS gamma on dim 0.
- Preserve the fractured residual across add and distributed norm. An immediate all-gather followed by the old replicated RMSNorm invalidates the family comparison.
- Use `try/finally` to release traces and restore the subdevice manager after API/config failures.

## Exact verification command

```bash
set -x
SNAPSHOT=/home/mvasiljevic/hf-cache/hub/models--mistralai--Mistral-Small-24B-Instruct-2501/snapshots/9527884be6e5616bdd54de542f9ae13384489724
EVIDENCE=models/autoports/mistralai_mistral_small_24b_instruct_2501/doc/optimized_multichip_decoder/evidence
timeout 1200 env TT_LOGGER_LEVEL=fatal \
  MISTRAL_SMALL_24B_REAL_WEIGHT_DIR="$SNAPSHOT" \
  MISTRAL_SMALL_24B_MULTICHIP_PERF_ITERS=300 \
  MISTRAL_SMALL_24B_MULTICHIP_COLLECTIVE_CANDIDATE=fused_exact_boundaries \
  pytest -q -s --junitxml="$EVIDENCE/fused_exact_boundaries_real_bfp8.xml" \
  models/autoports/mistralai_mistral_small_24b_instruct_2501/tests/test_multichip_decoder.py::test_fused_matmul_ccl_exact_final_boundaries \
  2>&1 | tee "$EVIDENCE/fused_exact_boundaries_real_bfp8.log"
```

No production change should be made until this test reports both boundary PCCs and the direct summed final-control comparison.
