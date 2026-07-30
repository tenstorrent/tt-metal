# Llama 3.1 8B Instruct Optimized Decoder

Model: `meta-llama/Llama-3.1-8B-Instruct`

Autoport path: `models/autoports/meta_llama_llama_3_1_8b_instruct`

Repo commit: `86f8bc022e6d526d9766539c6ea50137cabec799`

Hardware used: N300 Wormhole, single 1x1 `ttnn.MeshDevice`.

## Implementation

`tt/optimized_decoder.py` implements the optimized single-chip decoder stage.
It preserves the functional decoder public contract and paged KV-cache behavior,
but it does not subclass or call `FunctionalDecoder`.

The target has one meaningful decoder layer kind: dense Llama self-attention
plus SwiGLU MLP. Tests exercise layer 0 with full Llama 3.1 8B tensor shapes;
all 32 layers share this kind.

Final policy: `llama31_8b_single_chip_bfp8_attn_bfp4_mlp_decode_v1`.

| Tensor group | Policy |
| --- | --- |
| Activations and RMSNorm | BF16 |
| Attention weights | BFP8 |
| Paged KV cache | BFP8 |
| MLP gate/up/down weights | BFP4 |
| MLP mul intermediate | BFP8 |
| MLP math fidelity | LoFi |
| Decode residual stream | width-sharded L1 |
| Decode matmuls | DRAM-sharded weights and DRAM-sharded program configs |
| Prefill activations | DRAM interleaved with explicit 2D matmul program configs |

The runtime prefill/decode paths are TTNN-only. Tests patch `FunctionalDecoder`,
`ttnn.from_torch`, `ttnn.as_tensor`, `ttnn.to_torch`, common torch tensor
constructors, `torch.matmul`, and `torch.nn.functional.linear` to raise inside
the audited optimized hot paths.

The final signposted prefill window has no tilize, untilize, copy, host, or
reshard operations. The final signposted decode window has no tilize, untilize,
copy, or host operations; it retains two small TTNN device layout transitions
(`ShardedToInterleavedDeviceOperation` at 3 us and
`InterleavedToShardedDeviceOperation` at 1 us) required by the current TTNN
decode attention head APIs.

## Correctness

Acceptance threshold: PCC >= 0.995 for prefill and decode.

All decode PCC values are measured from replayed TTNN trace output. Repeated
trace and eager-vs-trace checks verify determinism.

| Case | Weights | Seq len | Decode context | Prefill PCC | Decode trace PCC | Repeated PCC | Eager vs trace PCC |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Functional baseline | synthetic | 128 | 129 | 0.9999777881890652 | 0.9999841394751932 | 1.0 | 1.0 |
| Optimized final | synthetic | 128 | 129 | 0.9995298149814705 | 0.9995144101749035 | 1.0 | 1.0 |
| Optimized final stress | synthetic | 128 | 129 | 0.9995298149814705 | 0.9995144101749035 | 1.0 | 1.0 |
| Optimized final long context | synthetic | 2048 | 2049 | 0.999534819041018 | 0.9995242130611273 | 1.0 | 1.0 |
| Functional baseline | real HF layer 0 | 128 | 129 | 0.9999812906688174 | 0.9999836008747124 | 1.0 | 1.0 |
| Optimized final | real HF layer 0 | 128 | 129 | 0.9994865842273941 | 0.9995098207830138 | 1.0 | 1.0 |

The optimized PCC delta is explained by the final BFP4 MLP policy. It remains
well above the functional acceptance bar while reducing decode latency. The
final full optimized run, including real-weight coverage, is preserved in
`final_full_optimized_run.log`.

Paged-cache coverage:

- non-identity page tables are used;
- prefill fills the paged K/V cache;
- decode updates and reads the same paged cache;
- a full-cache contract constructs a 128K-token paged cache with
  `page_block_size=64` and `max_num_blocks=2048`;
- decode uses a TTNN `current_pos` tensor and traced replay.

## Performance

Representative performance profile: synthetic weights, `seq_len=128`, decode
context 129, page block size 64, final BFP8 attention/KV plus BFP4 MLP policy.

| Window | Functional device time | Optimized device time | Speedup |
| --- | ---: | ---: | ---: |
| Warmed prefill | 3494.848 us | 2387 us | 1.46x |
| Traced warmed decode replay | 2482.910 us | 750 us | 3.31x |

Optimized host-side timing from the same final Tracy run:

- warmed prefill: `2.5199800729751587 ms`;
- traced warmed decode samples: `0.828934833407402 ms`,
  `0.8214409463107586 ms`;
- traced warmed decode average: `0.8243415504693985 ms`.

Performance accounting for the final decode run:

- roofline estimate: approximately `0.452 ms/token`;
- signposted device decode: `0.750 ms/token`;
- warmed traced end-to-end decode: `0.824 ms/token`.

Roofline estimate uses about 130.3 MB/token of BFP8/BFP4 weights plus BFP8
K/V cache reads at context 129, divided by 288 GB/s single-chip DRAM bandwidth.
The remaining device gap is the non-matmul attention/norm/cache/update work and
matmuls running at about 208-232 GB/s in the report. The host/device gap is
about 74 us in the final traced replay.

## Perf Report Conclusions

Artifacts:

- `final_full_optimized_run.log`
- `final_real_weights_run.log`
- `tracy/dense/optimized_ops_perf_results.csv`
- `tracy/dense/optimized_profile_log_device.csv`
- `tracy/dense/prefill_perf_report.txt`
- `tracy/dense/prefill_perf_report.csv`
- `tracy/dense/decode_perf_report.txt`
- `tracy/dense/decode_perf_report.csv`
- `tracy/dense/prefill_perf_report_stacked.csv`
- `tracy/dense/prefill_perf_report_stacked.png`
- `tracy/dense/decode_perf_report_stacked.csv`
- `tracy/dense/decode_perf_report_stacked.png`
- `tracy/dense/tracy_run.log`

Final `tt-perf-report` summary:

| Window | Device ops | Host ops | Device time | Advice summary |
| --- | ---: | ---: | ---: | --- |
| Prefill | 20 | 0 | 2387 us | Matmul tile configs are good; generic L1-input advice rejected for final path. |
| Decode | 19 | 0 | 750 us | All attention and MLP decode matmuls marked optimized. |

Advice decisions:

- Kept BFP8 attention weights and BFP8 KV cache. PCC remained above threshold
  and decode matmuls are DRAM-sharded.
- Switched MLP gate/up/down weights to BFP4 and MLP kernels to LoFi. Real
  weights stayed above PCC threshold and decode improved materially.
- Kept decode activations width-sharded in L1 across norm, attention residual,
  MLP, and output boundaries. The measured traced decode has no host fallback.
- Kept prefill activations DRAM interleaved. `Attention1D` exposes a static
  `prefill_input_memcfg`; forcing L1 input would not preserve the long-context
  contract because large prefill tensors exceed practical L1 residency.
- Tried short-prefill MLP L1 input after the report suggested L1 input 0. It
  added a `CopyDeviceOperation` and did not improve the device window
  (`2437 us` with the copy trial versus `2387 us` final), so it was rejected to
  avoid unnecessary movement.
- Rejected HiFi2/HiFi4 for final BFP4 MLP because LoFi preserved PCC and was
  faster. HiFi2 BFP4 trial is preserved in `precision_trials.log`.
- MoE active-expert optimization is not applicable; Llama 3.1 8B is dense.
- Fused matmul-CCL is not applicable in this single-chip decoder stage.
- LM head and sampling are not present in the decoder-only stage and are not
  part of this goal.

## Watcher

Clean watcher command:

```bash
TT_METAL_WATCHER=10 \
TT_METAL_WATCHER_DISABLE_ETH=1 \
TT_METAL_WATCHER_NOINLINE=1 \
TT_METAL_LOGS_PATH=models/autoports/meta_llama_llama_3_1_8b_instruct/doc/optimized_decoder/watcher/synthetic_disable_eth \
python_env/bin/pytest \
  --confcutdir=models/autoports/meta_llama_llama_3_1_8b_instruct/tests \
  models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_optimized_decoder.py \
  -k synthetic_paged_prefill_decode_trace -vv -s
```

Result: passed. Log scan found no fatal, assert, exception, error, timeout,
hang, NOC, ERISC, ARC, heartbeat, overflow, out-of-bounds, or fault signatures.

Artifacts:

- `watcher/synthetic_disable_eth/watcher_run.log`
- `watcher/synthetic_disable_eth/generated/watcher/watcher.log`
- `watcher/synthetic_disable_eth/generated/watcher/kernel_names.txt`
- `watcher/synthetic_disable_eth/generated/watcher/kernel_elf_paths.txt`
- `watcher/synthetic_disable_eth/generated/inspector/*.yaml`

## Final Checklist

| Requirement | Evidence |
| --- | --- |
| Optimized decoder file exists | `tt/optimized_decoder.py` |
| Tests exercise optimized path | `tests/test_optimized_decoder.py` patches functional fallback to raise |
| Paged KV-cache semantics preserved | synthetic, real, long-context, and full 128K cache tests pass |
| PCC at functional acceptance bar | all final PCC values >= 0.995 |
| Warmed prefill and traced decode latency before/after | performance table above |
| `tt-perf-report` tables and CSV logs | `tracy/dense/*perf_report*`, `optimized_ops_perf_results.csv` |
| Actionable advice tried or rejected with evidence | advice decisions above and `work_log.md` |
| No unnecessary host fallback in measured path | runtime fallback audit passes |
| Stress/repeated-run coverage | `test_optimized_decoder_repeated_trace_stress`, 8 trace replays |
| Watcher-clean correctness run | watcher command and artifacts above |
| MoE / CCL / LM-head applicability addressed | not applicable for dense single-chip decoder-only stage |

## Limitations

- This is the optimized decoder stage only. No multichip decoder, full-model, or
  vLLM work was started.
- Single 1x1 mesh only. Multi-chip parallelization belongs to the later
  multichip stage.
- Full 128K paged cache geometry is verified; HF-vs-TTNN long-context optimized
  correctness is covered at 2048 tokens in this stage because the functional
  stage already established larger HF-reference limits and full 128K cache
  allocation independently.
- Root pytest collection in this checkout still requires the autoport-local
  `--confcutdir` because root `conftest.py` imports unavailable
  `models.tt_transformers.demo.trace_region_config`.
