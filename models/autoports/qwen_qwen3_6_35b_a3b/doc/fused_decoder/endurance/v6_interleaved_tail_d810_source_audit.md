# d810 weighted-tail interleaved reader source audit

Host-only audit PASS for `v6_transport_conv_norm_decay_transpose_interleaved_tail_integration.py.snapshot`, SHA256 `d8106611f65edccdf069baab44acc9c0088baa709090b017a6a6df75f0a7bb87`. This is source review, not stage review or a hardware qualification claim. Production, the audit entrypoint, prior auditors, compatibility manifest, and git index were not edited.

`tests/audit_fused_interleaved_tail_v6.py` and its frozen `v6_interleaved_tail_d810_auditor.py.snapshot` have SHA256 `bea429df053982d9e4520243df7b2632c4d7284dc5e9f2743e87cad12b1a9ad7`. The upstream transpose auditor remains pinned to `e6e1de653bba87d2554011a82ca53b613c58aa5df92a8473f9d29067a7b0119d`. Full results: [source audit](v6_interleaved_tail_d810_source_audit.json).

## Exact delta and necessity

The candidate's weighted-tail call at line2409 accepts tiled FP32 `(1,32,128)` core and `(1,32,1)` sums, with explicit interleaved L1/DRAM guards. The unchanged caller at lines1308–1318 produces both in L1 after the same BF16 core rounding, FP32 widening, native square and sum. Replacing two input reshards with reader-owned local buffers removes transport without changing these arithmetic boundaries. The reader at line3479 fetches exactly4core+1sum 4096-byte pages, waits on the NoC read barrier, then publishes those buffers. Compile-time accessor offsets and runtime buffer order are pinned. The weighted output, gamma, debug bindings and downstream compact-SiLU helper remain unchanged.

Compute at line2377 is byte-identical to85ad after removing exactly the two old input reserve/push lines; arithmetic, rsqrt, explicit stores, BF16 rounding and output packing are untouched. The reader replaces these publications. Input local-CB capacity is20,480bytes; this replaces the old20,480bytes of transient input shards. This source accounting does not prove identical total program metadata, allocator peaks, or hardware capacity.

The only inverse changes permitted are the exact weighted runtime method, the compute constant, and removal of the appended reader constant. Their inverse reproduces the entire85ad module AST, hash `9a48213447b374e95cebb38ccb716d47fd11993bc082c4abcc01fb24314b42d0`. Entire `FusedDecoder` and `LocalSiluMultiply` ASTs remain identical. The constructor is unchanged; its existing individually reviewed setup allowance is retained, with **no new setup exemptions** and no whole-class/runtime-method exemption.

| Pin | SHA256 |
|---|---|
| WeightedNormTail AST | `ea3a9bd0d5f1c673822f6454a0535f025af17659d1ac64871a6c9a4ac893d2f4` |
| Runtime call AST | `0bb1cae71b4aed64e5d8311ce6b080070c0e29e2bca3c3e5faddb875ed03bfdf` |
| Unchanged constructor AST | `582b69dfbed36c930b713be87cf143df1f0f4f57bf2c0fe50cbb127278ea6045` |
| Reader source bytes | `fdfe6321306b121e20bcde3b4cc868ac81121e83add1f6786cc11e0956651d53` |
| Compute source bytes | `348f1d9a71ca559c08875853bfe07ab4c889ccd05eeb9753720eb812bd43f7a7` |

## Executed host checks and limitations

All47 non-setup runtime functions/methods pass the inherited host-fallback/dynamic-import audit. Fourteen new negative controls reject runtime readback/upload/exec, constructor upload/allocation, decoder edits, reader page count/stride/barrier edits, compute rounding/publication edits, reversed runtime buffers, incorrect CB capacity, and removed memory guards. Five also fail the independent syntax gate. All63 inherited negatives pass, for77 total. The source-identical planner/dispatch proof transfers:80,784valid planner cases (80,496 nonaligned starts or lengths),14invalid cases rejected and168mock dispatch cases, preserving262144context bound and originalV5 full-kind guards. No Torch or TTNN modules were imported.

The audit pins source and executable graph structure; it does not execute embedded kernels, prove device math, measure latency, or physically allocate advertised context. Existing component evidence remains separately recorded in [integration memo](v6_weighted_norm_interleaved_integration.md) and its linked result/build artifacts. Whole-decoder correctness, performance, capacity and final selection remain parent-owned gates. No new source-level contract risk or unintended delta was found.

Reproduce:

```bash
PYTHONPATH=. python_env/bin/python models/autoports/qwen_qwen3_6_35b_a3b/tests/audit_fused_interleaved_tail_v6.py \
  --snapshot models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v6_transport_conv_norm_decay_transpose_interleaved_tail_integration.py.snapshot \
  --output /tmp/v6_interleaved_tail_d810_source_audit.json
```
