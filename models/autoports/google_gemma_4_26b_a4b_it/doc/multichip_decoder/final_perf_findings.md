# Final native performance findings

Both captures measure4096 logical prefill tokens and128 advancing decode positions at batch1/concurrency1 on all four ASICs. Capture integrity and normal exits pass. All native rows retain full metadata; every replay has the same op set per device. Tables contain advice and CSVs preserve dtype/config fields.

## sliding

Whole-layer device prefill 325830.533us; decode 705.420us. Useful prefill FLOPs/peak 0.102043%; estimated decode DRAM/peak 9.4246% (136158016 estimated bytes across four devices).

The merged tool decode op subtotal is 588.257us and prefill subtotal 91644.437us. These are diagnostic tool sums, not whole-layer denominators. RS/AG reported decode work totals 84.984us. Gaps and all nonmatmul operations remain in the device roofline denominator.

Largest reported decode rows:
- BinaryNgDeviceOperation: 110.331us
- SparseMatmulDeviceOperation active=8/128 x 32 x 2816 x 384: 44.672us
- TypecastDeviceOperation: 43.987us
- ReduceScatterMinimalAsyncDeviceOperation: 42.850us
- AllGatherAsyncDeviceOperation: 42.133us
- LayerNormDeviceOperation: 37.660us
- TilizeWithValPaddingDeviceOperation: 28.524us
- SdpaDecodeDeviceOperation: 27.208us
- FillPadDeviceOperation: 24.190us
- MatmulDeviceOperation 32 x 2816 x 2048: 18.905us

Native integrity: 2593 prefill programs/rank and128 replays of129 programs/rank; identical counts on all four devices.

## full

Whole-layer device prefill 309683.707us; decode 766.584us. Useful prefill FLOPs/peak 0.147866%; estimated decode DRAM/peak 10.7086% (168120384 estimated bytes across four devices).

The merged tool decode op subtotal is 667.124us and prefill subtotal 78031.425us. These are diagnostic tool sums, not whole-layer denominators. RS/AG reported decode work totals 91.588us. Gaps and all nonmatmul operations remain in the device roofline denominator.

Largest reported decode rows:
- TilizeWithValPaddingDeviceOperation: 85.347us
- RotaryEmbeddingHfDeviceOperation: 57.812us
- BinaryNgDeviceOperation: 56.248us
- LayerNormDeviceOperation: 52.793us
- SdpaDecodeDeviceOperation: 50.231us
- AllGatherAsyncDeviceOperation: 47.892us
- SparseMatmulDeviceOperation active=8/128 x 32 x 2816 x 384: 45.098us
- ReduceScatterMinimalAsyncDeviceOperation: 43.696us
- TypecastDeviceOperation: 28.910us
- MatmulDeviceOperation 32 x 2816 x 3072: 26.242us

Native integrity: 2469 prefill programs/rank and128 replays of106 programs/rank; identical counts on all four devices.

## Interpretation and accepted tradeoffs

Native sparse rows confirm indexed active8/128 execution, LoFi GU BFP8 input × BFP4 weights and BF16 down input × BFP4 weights; the activation after gate/GELU stays BF16. Shared GU is BFP4 both kinds, down BFP8 sliding/BFP4 full. QKV and WO are BFP8 with LoFi decode and FP32 output/accumulation. Shapes, core counts and K blocks match selected_policy_audit.md; full native rows and perf-report CSVs are the authority.

Communication, normalization and layout operations materially limit decode scaling. Sliding sharded BF16 RoPE cuts the native rotary subtotal to5.573us, but casts and layout conversions still cost time. Full interleaved rotary costs57.812us; the adapted BF16 split/sharded D512 implementation passed PCC/cache but was slower end-to-end (full_sharded_rope_adapted.json), so its isolated lower-movement intent did not justify selection.

Final-policy wider sparse geometries, DRAM shared MLP, and coherent hidden-sharded/Ring/fused-AGMM alternatives all pass but lose whole-layer latency (final_policy_alternatives.json). The sharded interface is consumed directly through following operations and is gathered only at the comparison boundary. Thus communication remains visible in the selected path, but the tested alternative does not remove it cheaply.

Prefill uses dynamic EP expert unions across32-token groups; useful FLOPs count only top8 per logical token, excluding additional union/padding work. The low useful-FLOP percentages also include profiler-inflated device gaps. Profiled prefill windows differ materially from ordinary warmed host intervals; neither host intervals nor kernel/op subtotals replace the required complete device window. This report makes no unprofiled-device prefill latency claim.

Remaining generic tool advice is assessed in optimization_advice_audit.md. Router width and K-block alternatives were measured and rejected, shared down already uses the full legal K17 decode block, and final sparse N2/K44/K88 alternatives were remeasured under BFP4/BFP8/LoFi. Small non-dominant prefill shared-down rows remain inherited; the stage does not claim every op reaches a roofline peak.

The final supporting table filters use distinct PERF_PREFILL_END/PERF_DECODE_END boundaries. A historical same-name filter warning was investigated and corrected offline; table row counts now exactly match native phase counts. Whole-layer JSON/windows/integrity files are byte-identical before and after this report-only correction. See final_signpost_filter_check.json.
