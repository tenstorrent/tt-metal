# Expanded QSA34/66 accuracy investigation

Status: matched controls complete; independent AutoDebug is running. No runtime, fixture or threshold change has been made to suppress either failure.

New expanded HF fixtures fail prefill row30 at34 (whole0.99368006/contribution0.99350828) and row44 at66 (whole0.99313277/contribution0.99276608). All decode, state reset and replay checks pass. Both failing rows precede the compact tail.

Five device controls use the exact original fixture and preserve failing exits. Functional baseline fails the same rows (whole0.99369824 and0.99311984). The prior fused policy, writer-only, tail-only and combined produce byte-identical complete prefill and all three decode outputs for both lengths. See `qsa_tail_control_collection_v1.json` and `qsa_tail_control_comparison_v1.json`; the export wrapper copies already-read host metric operands after device readback and does not alter model calls or gates.

`qsa_tail_backend_cpu_v1.json` independently reproduces the literal HF oracle bit-for-bit, then changes only selected GR accumulation arithmetic. Replacing attention GR-down with FP64 F.linear, rounded at the unchanged BF16 boundary, reproduces the literal-HF whole/contribution failure at both lengths. Fused versus that control passes every prefill row:34 minimum0.99860567whole/0.99856806contribution;66 minimum0.99706912whole/0.99696749contribution. MLP-down-only does not reproduce the failure. Replacing all four GR projections has a different34 result, so this is a specifically scoped backend sensitivity control, not a universal higher-precision oracle. All weights, activation dtypes, model topology and fixture values remain unchanged in those CPU controls.

Literal34/66 acceptance remains false. The control identifies a shared attention-GR accumulation sensitivity and rules out a changed output from the new writer/tail paths; it does not waive the original bar. Required final-stage acceptance classification remains pending. Unrun original-contract batch/capacity/Watcher and candidate timing jobs continue separately in `integrated_writer_remaining_plan_v2.json`.

Device exports: `/home/ttuser/qwen38-flash/qsa_tail_control_<policy>_v1.pt`. CPU controls: `/home/ttuser/qwen38-flash/qsa_tail_backend_cpu_v1.pt`. Exact argv, source identity and logs accompany each named provenance/plan.
