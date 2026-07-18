# GPT-OSS-20B EP4/QKV10 profiler report

## Provenance

This report covers one real GPT-OSS-20B decoder layer on the selected fixed
1x4 P300c ring. The final profile used the production EP4 policy: QKV
`(10,9,2,2)`, 9x10/subblock-1 gate/up and down prefill programs, post-sparse
BF16 materialization, and the existing ring all-reduce. It contains one S=128
prefill and three warmed decode trace replays. Watcher was not enabled in the
profile process.

```bash
env -u TT_METAL_WATCHER \
RUN_MULTICHIP_DECODER_PERF=1 MULTICHIP_DECODER_PREFILL_REPEATS=1 \
MULTICHIP_DECODER_TRACE_REPLAYS=3 \
MULTICHIP_PERF_RESULT_PATH=models/autoports/openai_gpt_oss_20b/doc/multichip_decoder/logs/profile_perf_final_autofix.json \
python -m tracy -r -p \
  -o models/autoports/openai_gpt_oss_20b/doc/multichip_decoder/perf/tracy_autofix \
  -n gpt_oss_20b_ep4_qkv10_autofix -m pytest -q \
  models/autoports/openai_gpt_oss_20b/tests/test_multichip_decoder.py::test_multichip_decoder_perf
```

Retained final-path provenance:

- `../logs/profile_perf_final_autofix.json`: profile-run wall timing/policy;
- `cpp_device_perf_report_ep4_qkv10_autofix.csv`: device-kernel report;
- `decode_report_autofix.csv`: merged TTNN decode op-level report;
- `decode_summary_autofix.csv.csv`;
- `prefill_report_autofix.csv` and `prefill_summary_autofix.csv.csv`;
- matching summary PNGs and `../logs/tt_perf_report_*_autofix.log` command logs.

The unmerged `ops_perf_results_ep4_qkv10_autofix.csv` remains locally but is
not checkpointed because its 4.48 MiB size exceeds the repository's 500 KiB
evidence-file limit. The raw Tracy directory was 1.4 GiB and was removed after
verifying these
compact artifacts. It is reproducible from the command above.

## Accepted warmed wall timing

Acceptance uses isolated unprofiled runs with 20 S=128 prefill iterations and
500 warmed trace replays. The one-chip optimized baseline and 1x4 process were
measured separately.

| Path | Prefill ms/layer | Decode ms/layer | Decode speedup | Four-chip efficiency | Prefill speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| optimized 1x1 | 13.408263 | 0.975834 | 1.000x | 100.00% | 1.000x |
| production 1x4 TP4-attention/EP4-experts | 22.618636 | 0.598719 | 1.62987x | 40.7468% | 0.59280x |

The profile-instrumented path is 23.782881 ms prefill and 0.658529 ms decode;
it is used for attribution, not acceptance. The original reviewed multichip
path was 26.676904 ms prefill, so the production prefill rewrite improves it
by 15.2%. Prefill remains 1.687x slower than one chip at S=128.

## Decode table

Three traced decodes contain 327 merged device operations and 1,660.50 us of
summed device-op time. The modeled aggregate DRAM roofline is 40.5% (207
GB/s).

| Operation family | Share | Device time / 3 traces | Count | Interpretation |
| --- | ---: | ---: | ---: | --- |
| AllBroadcast | 13.04% | 216.51 us | 6 | O and EP expert-result ring reductions |
| sparse matmul, DRAM input | 11.22% | 186.28 us | 6 | EP gate/up rows |
| sparse matmul, L1 input | 5.48% | 90.93 us | 3 | EP down rows |
| typecast, L1 | 5.31% | 88.21 us | 36 | routing/expert dtype boundaries |
| fill-pad, L1 | 4.93% | 81.81 us | 15 | tile-safe decode shapes |
| reshape views, L1 | 4.69% | 77.83 us | 12 | routed tensor topology |
| router FP32 matmul | 4.36% | 72.44 us | 3 | precision boundary protecting top-k |
| QKV width-sharded matmul | 3.73% | 61.98 us | 3 | selected QKV10 geometry |
| SDPA decode | 3.07% | 50.90 us | 3 | two local KV heads/rank |
| TopK | 2.72% | 45.14 us | 3 | exact global top-4 selection |
| block-sharded RMSNorm | 2.70% | 44.87 us | 6 | two norms per layer |

The `active=4` report input models four global routes. Hardware
instrumentation separately proves that the global total is exactly four while
each EP rank legitimately executes zero through four, which is why production
uses `nnz=None` for its rank-local sparse calls.

## Prefill table

The promoted prefill contains 67 merged device operations and 23,971.90 us of
summed device-op time. The modeled 107.4% aggregate DRAM roofline reflects the
sparse model's global-active-expert assumption and merged streams; it is not a
claim of physically sustained 550 GB/s by each device.

| Operation family | Share | Device time | Count | Interpretation |
| --- | ---: | ---: | ---: | --- |
| sparse matmul, DRAM input | 53.56% | 12,840.49 us | 3 | selected gate/up/down sparse expert rows |
| reshape views, DRAM | 13.60% | 3,260.31 us | 7 | token/expert topology |
| typecast, DRAM | 10.77% | 2,580.63 us | 6 | post-sparse BF16 and final collective boundary |
| unary, DRAM | 7.86% | 1,883.77 us | 7 | expert activation work |
| reduce-scatter | 5.84% | 1,399.05 us | 2 | two ring all-reduces' RS phase |
| binary, DRAM | 5.13% | 1,229.38 us | 10 | score/SwiGLU/residual math |
| fill-pad, DRAM | 1.05% | 251.43 us | 2 | tile-safe routed chunks |
| dense matmul | 0.52% | 124.19 us | 3 | QKV, O, router |
| RMSNorm | 0.39% | 92.51 us | 2 | replicated prefill norms |
| all-gather | 0.21% | 51.44 us | 2 | two ring all-reduces' AG phase |
| SDPA | 0.10% | 24.01 us | 1 | local attention |

Compared with the reviewed profile, typecast fell from 3,893.86 to 2,580.63
us, fill-pad from 1,594.60 to 251.43 us, and sparse matmul from 14,896.55 to
12,840.49 us. These reductions explain the full-layer improvement.

## EP4 prefill candidate disposition

All candidates preserve whole-expert EP4, exact gate-selected top-4 execution,
non-aligned public lengths, BFP8 weights, and the optimized attention policy.
Unless noted, timings use 10 prefill iterations and 100 trace replays.

| Candidate | Prefill ms | Decode ms | Disposition |
| --- | ---: | ---: | --- |
| reviewed 3x5 gate / 5x6 down | 26.6769 | 0.598641 | superseded |
| gate 5x6 | 24.9871 | 0.598996 | improved |
| down 5x9 only | 26.7416 | 0.598784 | rejected |
| down 9x10 only | 26.4462 | 0.598719 | improved |
| gate 5x6 + down 9x10 | 24.6387 | 0.598669 | improved |
| chunk 64 / chunk 32 | 27.0177 / 27.2026 | 0.5987 / noisy | rejected |
| post-sparse BF16 only | 24.8080 | 0.598731 | improved |
| 5x6 + 9x10 + BF16 | 22.8116 | 0.598888 | improved |
| 5x9 + 9x10 + BF16 | 22.9150 | 0.598728 | rejected |
| 9x10 + 9x10 + BF16 | 22.6662 | 0.598868 | selected; 22.6186 at 20/500 |

DRAM placement remains necessary because routed S=128 expert intermediates do
not fit a coherent all-L1 contract across all three sparse products. BFP8
reshape/fill cannot be elided before sparse output because block exponent
metadata must be reconstructed; the selected rewrite instead crosses to BF16
immediately after each sparse result and converts the local weighted sum back
to BFP8 immediately before the collective. Real sliding/full PCC and exact
route-count tests pass after promotion.

## Decode CCL candidate disposition

The remediation implemented and measured a persistent-semaphore alternative
at both existing decode reductions:

1. pad replicated width 2,880 to 2,944 (four tile-aligned 736-wide pieces);
2. `ReduceScatterMinimalAsyncDeviceOperation` on ring axis 1;
3. `AllGatherAsyncDeviceOperation` on the same persistent semaphores;
4. slice back to the public 2,880-wide replicated stream.

Both layer kinds pass correctness (minimum final PCC 0.99909233), mutable
trace replay, cache equivalence, and determinism. At 500 replays it measures
0.638599 ms versus 0.598641 ms for current all-reduce, a 6.7% regression. The
short candidate profile confirms six minimal-async RS rows (152.28 us, 8.98%),
six async AG rows (88.28 us, 5.21%), plus 39.27 us explicit pad and 10.04 us
slice. It is rejected.

Retained candidate artifacts include
`cpp_device_perf_report_ccl_rs_ag_pad64.csv`,
`decode_report_ccl_rs_ag_pad64.csv`, its summary CSV/PNG, and
`../logs/tt_perf_report_decode_ccl_rs_ag_pad64.log`. Its reproducible 1.4 GiB
raw Tracy directory was removed only after those files were verified. The
unmerged 4.63 MiB op dump remains locally but is omitted from the checkpoint
under the same repository file-size policy.

Fused matmul + minimal reduce-scatter is not legal on this target: the
GPT-OSS implementation disables that Blackhole family for race #46181.
Gathered-input/local-output O requires an alternate `[4096,2944]` output
sharding and another all-gather to restore the required replicated stack
boundary. The coherent sharded residual/norm/router/gather family was already
measured 2.089x slower, so neither alternative displaces current all-reduce.

## Communication, DRAM, compute, and data-movement conclusion

| Finding | Decision |
| --- | --- |
| Decode AllBroadcast is 13.04%; the persistent RS+AG substitute is correct but 6.7% slower. | Keep the current two mathematically required ring all-reduces. |
| Prefill sparse rows are 53.56%; data movement was materially reduced. | Keep EP4 9x10/BF16; it is 15.2% faster than the reviewed path and far faster than TP4's 39.82 ms. |
| Rank-local active count is data dependent. | Keep `nnz=None`; static four would read nonexistent local routes. |
| QKV is 3.73% of decode. | Keep `(10,9,2,2)`, already the fixed-policy winner. |
| FP32 router is 4.36%. | Keep it: near-tie stress proves top-k is discontinuous. |
| Remaining reshape/typecast/fill implements sparse routing and non-aligned lengths. | Retain only the now-measured minimal boundaries; no faster correct candidate remains. |
| Profile reports BFP8 LoFi experts and BF16 HiFi4 attention. | Keep inherited precision; earlier BFP4/LoFi correctness rejections remain controlling. |

The production path is accepted as the decode-focused layer-stack baseline.
Its remaining S=128 prefill slowdown is explicit and is not hidden by the
decode win.
