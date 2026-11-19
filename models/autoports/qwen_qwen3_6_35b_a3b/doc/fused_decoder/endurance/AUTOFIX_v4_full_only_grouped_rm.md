# Full-attention-only grouped RM candidate

Prepared `v4_full_only_grouped_rm.py.snapshot`, SHA256 `09ebe6cc2f1ae06e07af713c54e096f3dca56aa0d09638ea9cdb7152d201b921`, against `v4_combined_tilize_widen.py.snapshot`, SHA256 `72d4f05d5e089c7f479e7d4bb35dcd2570aa4917dbd65c08961cc1ad0bcd9336`. The adjacent patch and provenance bind both sources. Production remains unchanged; this combined conditional source has no device qualification yet.

Only full-attention instances pad each grouped down expert's output width2048→2112, request native RM output and crop to2048 before combine. Linear instances retain exactly the original setup weight packing, TILE down output and untilize adapter. Each instance keeps one grouped-down representation. Added residency is zero for linear and16,777,216 bytes (16 MiB) for full attention. Decode arithmetic and sparse weight layouts are unchanged for either kind.

CPU AST specialization replaces comparisons against the fixed `self.kind` with their selected branches. The complete candidate module specialized to linear attention is AST-identical to the complete72d4 base module under the same specialization. This establishes unchanged linear tensor operations and setup allocations; it is not an end-to-end timing result. Separately, the candidate's full-attention prefill `_moe` specializes to the same AST as the unconditional padded-RM `_moe` whose all-four-bank actual device probe passed. No new operator consumer, data format or arithmetic expression is introduced by the conditional selection. The combined source must still be benchmarked and qualified on actual layer3 weights.

The unconditional control's five-case evidence and mixed timing remain in [AUTOFIX_v4_grouped_rm_padding.md](AUTOFIX_v4_grouped_rm_padding.md). Its direct private `_moe` S4097 diagnostic is valid and remains a real large-input regression; it is not discarded. The full-attention public `prepare_prefill` independently caps physical chunks at256, whereas linear attention can use4096. Therefore the conditional version avoids applying the measured large-input MoE change to the ordinary linear large-chunk path. This is a contract-based specialization, not a benchmark-size exception or a new public sequence restriction. Arbitrary valid logical lengths continue through the existing planner.

The small complete-MoE improvements do not establish a whole-decoder win and may be timing noise. Parent-owned whole-decoder controls should compare this snapshot against72d4 at full-attention S256/context512 and S4097/context8192. Both normal prefills include all adapters and continuation semantics. Required capacity validation must include the16 MiB addition if selected. No new standalone operator probe is needed because the source-specialized prefill consumer is unchanged from the existing exact five-case probe; actual integrated layer3 correctness/performance remains required.

Example parent-only whole-path command:

```bash
python_env/bin/python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.benchmark_fused_snapshot \
  --snapshot models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v4_full_only_grouped_rm.py.snapshot \
  --layer 3 --length 256 --context 512 --prefill-repeats 10 --decode-repeats 60 \
  --output models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v4_full_only_grouped_rm_l3_s256.json
```

No hardware execution, live source edit, shared-native change, production-selection decision or Stop Rule completion occurred in this preparation.
