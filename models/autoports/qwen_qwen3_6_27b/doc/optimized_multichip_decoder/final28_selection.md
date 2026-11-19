# Final28 selection recommendation at the user-directed search cap

CPU-only evidence review. This recommends the best established low-risk **candidate** for final-default validation; it does not report new timings, final acceptance, or satisfaction of the original roofline/device-time stop rule. The parent reports a subsequent authoritative user cap ending optimization search.

Choose the existing `ag_adapted9`/integrated family with **distributed decode norm off** for both kinds. For GDN retain local beta/decay and column grain1; for full attention retain split compact QKV plus fused QK normalization/RoPE. Do not enable new unmeasured whole-layer integrations merely because their isolated kernel passed.

## Measured selection

All numbers below are historical candidate warmed prefill / traced warmed decode milliseconds, with unchanged numerical gates passing. JSON receipts/commands referenced by the CSVs are authoritative. These are not final-default measurements.

| Kind and selected row | B1/S512 | B32/S33 | B32/S513 | Evidence |
|---|---:|---:|---:|---|
| GDN column1 | 2.078413 / 0.466628 | 5.661292 / 0.575519 | 43.662147 / 0.575418 | `gdn22_timing24.csv`, regression column1 rows |
| Full attention split+fused QK/RoPE | 1.727639 / 0.411243 | 3.635312 / 0.419471 | 35.254064 / 0.446391 | `attention22_timing24.csv`, regression control rows |

GDN column1 beats its same-family control in decode at B1 (0.484161→0.466628ms) and B32/S33 (0.578870→0.575519ms). Column2 is slower. Hidden-norm dispatch adds about4us B1 improvement, but lacks consistent B32 improvement; leave it off rather than introducing another final ownership/protocol combination at the cap. The paired real-text column1 rows corroborate the decode result.

Full attention direct-packed QK/RoPE is quality-passing and modestly faster at B32/S33 (0.419471→0.417983ms), but slower at B1 (0.411243→0.412949ms). Recommend the simpler measured split+fused default for all batches, leaving `packed_compact_qk_rope` off. A B-dependent selector could retain both results but that exact final dispatcher has not been validated; do not describe it as an existing measured default. The original fused QK/RoPE family itself gave 5.88–10.80% complete-layer improvements (`fusions18_summary.csv`).

Minimal norm is faster in all27 `norm_whole26` comparisons, but `norm_resources26_minimal_b1.json` fails while constructing persistent norm outputs: each of8 L1 banks needs10240B, with only5440B available (bank capacity1460288B, allocated1454848B). This is hard current-owner allocation evidence; it does not establish an unavoidable context limit. AutoDebug is ongoing. DiT has only two-owner proof; its64-owner jobs did not run after this failure. Therefore select explicit `distributed_decode_norm="off"`, retain the context contract, and do not silently substitute DiT. This excludes the unsafe resource family, without claiming its arithmetic failed.

## Exact candidate policy overlays

These overlays are copied from existing passing manifest rows above, with only an explicit norm-off value added (already the default for those rows). Apply through the parent's final default selection, not through runtime fallback.

GDN:

```json
{
  "flat_prefill": true,
  "flat_finish": true,
  "flat_prefill_correction": true,
  "persistent_ccl": true,
  "recurrent_variant": "cumulative_exp_scalar",
  "ccl_decode_memory": "DRAM",
  "mmrs_chunk_rows": 16384,
  "mmrs_max_rows": 32768,
  "mmrs_min_rows": 256,
  "local_fused_conv": true,
  "batched_conv_b1": true,
  "packed_correction_options": {
    "prefill": true,
    "release_separate": true,
    "prefill_config": {
      "row_cap": 4096,
      "grid": [
        8,
        8
      ],
      "out_block_h": 4
    }
  },
  "fused_packed_swiglu": true,
  "local_affine_gate": true,
  "tiled_prefill_conv": true,
  "flat_packed_gdn": true,
  "batched_prefill_cache": true,
  "local_beta_decay": true,
  "recurrent_column_grain": 1,
  "distributed_decode_norm": "off"
}
```

Full attention:

```json
{
  "flat_prefill": true,
  "flat_finish": true,
  "flat_prefill_correction": true,
  "persistent_ccl": true,
  "packed_qkvg": true,
  "direct_compact_attention": true,
  "gather_norm": "decode_or_single",
  "shard_hidden_norm": true,
  "ccl_decode_memory": "DRAM",
  "mmrs_chunk_rows": 16384,
  "mmrs_max_rows": 32768,
  "batched_prefill_rope": true,
  "mmrs_min_rows": 256,
  "fused_packed_swiglu": true,
  "local_affine_gate": true,
  "tiled_prefill_conv": true,
  "flat_packed_gdn": true,
  "batched_prefill_cache": true,
  "l1_mlp": true,
  "fused_compact_qk_norm": true,
  "decode_sdpa_grid": [
    8,
    4
  ],
  "fused_compact_qk_rope": true,
  "short_capacity_decode_k_chunk": 128,
  "distributed_decode_norm": "off"
}
```

Leave the unselected options off: `fused_hidden_norm`, `recurrent_prediction_registers`, `packed_compact_qk_rope`, `direct_prefill_qkv`, `prefill_swiglu_options`, `prefill_swiglu_tail_packed`, `flat_gather_prefill`, the combine-output-window flag, and `attention_gate_mode`. Exact names of unselected experimental flags should be checked at final editing; absence from the measured overlays is authoritative. No public input alignment restriction is introduced.

## B1 prefill remains an explicit unresolved final gate

Initial before measurements were GDN2.139ms/full attention1.618ms. They cannot establish same-current-harness nonregression against today's candidate, particularly full attention1.727639ms. `b1_diagnostic27.json` has4 prepared jobs and **zero result receipts** at this snapshot; its candidate rows still request minimal norm and must not be run unchanged for the recommended norm-off choice. Even those four jobs would be diagnostic only, not the required repeated independent-pair gate.

Source supports several conservative B1 decisions without claiming they solve the measured regression:

- `flat_prefill.py:15–31` admits compaction only for B>1 and S>1, preserving B1 legacy dispatch. Disabling flat prefill globally cannot be credited as a B1 speed fix from this branch.
- `multichip_decoder.py:_norm` gives full attention `gather_norm="decode_or_single"`; B1 prefill therefore takes the existing gather-first norm branch. Switching it to distributed stats would be an unmeasured new B1 family.
- `multichip_decoder.py:_prefill_cache_fill` sends B1 prefill through native single-user cache fill even when batched fill is enabled. Cache-flag ABBA observations reported no systematic penalty.
- `multichip_decoder.py:_qkv` adds direct-prefill QKV only when explicitly enabled; keep it disabled. Its standalone non-aligned movement regression is documented in `qkv_movement28_audit.md`, and whole-layer jobs are unrun.
- Compact fused QK/RoPE is implemented in `_local_compact_attention_decode` , separate from the ordinary multi-token `_qkv` path. It cannot be credited with repairing B1/S512 prefill without measurement.

The executable unchanged gate is `check_optimized_multichip_evidence.py:441–504`: at least5 independent original-TP4/final process pairs per kind, alternating arm order, exact paired median one-sided95% upper bound, zero-margin threshold. Original control SHA082c82faa6b9500f654dcc3f0e26f1c4fab656e987d3fd079bc1c79ec19a3c44 uses `--decoder-source`; final arm must use defaults and no source override. No existing evidence permits claiming this gate passed or inventing a relaxed tolerance. If the cap also precludes final validation, record this as unmet rather than converting historical candidate timing into final acceptance.

Final selection cross-check: `final28_selected_overrides.json` exactly equals the two measured overlays above after adding explicit norm-off; no proven policy mismatch found. Main source snapshot SHA256: `8ceccf19b6b77228210e5fd4e8b03777054efe0a97d92eec4c7d152543f2aa46`. Parent is applying final defaults; this document is a recommendation/evidence map, not an immutable final-source receipt.
