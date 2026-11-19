# Supplemental source audit: transport, convolution, norm, direct decay

Host/source work only. The production audit entrypoint, original auditors, frozen snapshots and source-format compatibility manifest are unchanged. This is a source gate, not stage review or a new device qualification.

## Reviewed chain and exact changes

The frozen chain is ab28 → transport 13f78 → convolution b253 → norm 803685 → direct decay 44f87. `tests/audit_fused_composed_v6.py` reverses only the two transport placement statements, selector reshape expression, convolution guarded region, norm scan assignments, exact linear-only setup constructor statements and appended helpers. Whole-module AST equality is required at each intermediate source. `tests/audit_fused_decay_v6.py` separately restores the exact decay guarded region, setup statement and three appended definitions to 803. No whole method or class is replaced during these inverse proofs.

Source references in `v6_transport_conv_norm_decay_lookup_integration.py.snapshot`:

- Lines 601–612: linear-only construction of three helper objects. DecodeConvolution only assigns references. NormTerminalScan additionally queries mesh grid metadata. DecodeDecayLookup validates table shape, dtype, layout and memory placement; its only constructor calls are `tuple` and `table.memory_config`. None allocates or uploads tensors. Their constructors are included in runtime syntax inspection; **no setup exemption is added**.
- Lines 1191 onward: the convolution path requires `decode and cache_boundary`; its `else` is exactly the original convolution/QKV split/carry block. The helper at lines 3071–3171 reads existing four FP32 tap tensors and BF16 recurrent convolution state, writes BF16 QKV output and state on device. The exact Q/K/V slices are preserved. Public singleton prefill continues through the original block.
- Line 1049: the exact norm terminal scan replaces two scans/crops. The helper at lines 3179–3275 keeps the input placement and FP32 output boundary. All surrounding sum, reciprocal square root, multiplication, precision conversions and full-kind norm branches are unchanged.
- Lines 1222–1255: direct decay lookup is restricted to `decode and cache_boundary`. Its original `else` region remains exact, including public singleton prefill. The helper at lines 3282–3355 reads BF16 raw codes and existing per-head FP32 table values and writes FP32 TILE output. It does not perform an approximate exp or reconstruct floating-point values. Existing table generation and exhaustive compaction assertion are unchanged; compact and full table modes are supported. The kernel preserves raw payload bits, including subnormals and NaNs. `g=None` agrees with the original decode branch, which did not construct `g`.

The new helper class ASTs and every new C++ source string are pinned explicitly in the supplemental auditors and machine reports. Unreviewed helper, kernel, constructor, arithmetic, guard, setup or public-planner mutations reject. The original seven V5 full-attention guarded branches, `_full_attention`, public signatures, paged KV/cache/batch methods, context limit and planner remain covered by the original checker.

## Checks and evidence

803 report: [v6_composed_803_source_audit.json](v6_composed_803_source_audit.json). Auditor SHA256 `6ba4ffe724503feb0475625bc9293d0aa73d6cddbfaa4d73efd84bf4de7e4729`; preserved bytes [v6_composed_803_auditor.py.snapshot](v6_composed_803_auditor.py.snapshot). It checks 43 runtime functions, 80,784 valid aligned/non-aligned planner cases, 168 dispatch cases, 21 original negative controls, 9 transport controls and 14 composed controls. New constructor/runtime host-transfer controls reject independently in the original syntax checker as well as through exact-source checks.

44f87 report: [v6_decay_44f87_source_audit.json](v6_decay_44f87_source_audit.json). The separate decay audit reruns the frozen 803 audit, audits actual 44f87 runtime syntax/public contracts, proves exact planner/dispatch/public method AST transfer, and adds eight decay controls. It explicitly identifies the source on which mock planner/dispatch execution occurred; source-identical methods are not presented as new device runs. The actual candidate has 45 runtime functions checked. A constructor allocation mutation rejects through the exact helper proof even though device allocation is permitted in ordinary runtime methods.

Commands (from repository root):

```bash
python_env/bin/python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.audit_fused_composed_v6 --snapshot models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v6_transport_conv_norm_terminal_scan_integration.py.snapshot --output models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v6_composed_803_source_audit.json
python_env/bin/python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.audit_fused_decay_v6 --snapshot models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v6_transport_conv_norm_decay_lookup_integration.py.snapshot --output models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v6_decay_44f87_source_audit.json
```

The output paths are write-once; use fresh filenames to repeat. Pinned Black 23.10.1, autoflake 2.3.1 and isort 5.13.2 checks pass for both new auditors. Original audit source/entrypoint bytes are preserved under `v6_composed_preserved_*.snapshot`; compatibility manifest remains `fe5a3db52da5678a8d8041cd5078ff89f55838256bd9ec3c93fae220e87a085e`.

Component/device evidence is separate from these host checks:

- [Convolution component](v6_decode_conv_complete_all_device.json) and [build inverse](v6_transport_decode_conv_complete_integration_build.json).
- [Norm component](v6_norm_terminal_scan_device.json) and [build inverse](v6_transport_conv_norm_terminal_scan_integration_build.json).
- [Decay component](v6_decay_direct_lookup_all_device.json), [all raw-code proof](v6_decay_direct_lookup_rawcode_all_device.json), [table inventory](v6_decay_direct_lookup_table_inventory.json), and [build inverse](v6_transport_conv_norm_decay_lookup_integration_build.json).
- Parent-owned [44f87 layer34 stream](v6_transport_conv_norm_decay_layer34_seed154.json) and [matched 10/60 benchmark](v6_transport_conv_norm_decay_l0_s256_10pf.json). The parent reports unchanged stream PCC and 484.135795 ms warmed prefill / 1.450284 ms traced decode. These are existing device results, not measurements by this audit.

## Limits

Exact source restoration and syntax checks do not execute native kernels or prove external TTNN behavior. They do not substitute for unchanged PCC, repeated-run, watcher, context-capacity or performance gates. New constructors add no persistent tensors, but temporary CB usage and allocator feasibility still require capacity validation. The public advertised context remains 262144 without a new alignment restriction; this source audit does not claim that a new maximum-context device run has completed. The production entrypoint remains unchanged pending explicit parent promotion.
