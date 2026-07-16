# AutoFix: DRAM-sharded BFP4 PCC collapse

## Trigger and outcome

The second independent stage review correctly rejected the original
`dram_mlp_bfp4_24c` and `dram_mlp_bfp4_48c` evidence: their near-zero PCC was
a corruption signature, not a valid reason to reject DRAM-sharded decode.
Fresh-context AutoDebug independently reached the same root cause as the
focused hardware experiments.  The repaired controls are correct and remain
slower than the selected shard-advisor path, so the final rejection is now
earned.

## Isolated hypotheses

| Hypothesis | Isolated evidence | Result |
|---|---|---|
| BFP4 upload/materialization corrupts DRAM-sharded weights | `results/autofix/weight_materialization.json`: advisor and DRAM-sharded gate/up/down round trips have the same approximately 0.993 PCC | Refuted |
| DRAM-sharded gate/up/down matmuls or 24/48-core geometry corrupt decode | `matmul_localization.json`: both DRAM geometries closely match advisor at gate, up, gated product, and down | Refuted |
| DRAM-sharded decode is independently bad | `stage_localization.json`: decode from an HF-filled clean cache reaches approximately 0.999999 full-output PCC for both geometries | Refuted |
| TT prefill corrupts state consumed by decode | Preserved failing `results/autofix/full_candidate/candidate_sweep.json`, followed by the first prefill localization console: attention and both KV caches were already near zero | Confirmed boundary |
| Splitting only MLP prefill/decode weights is sufficient | Preserved failing artifact still collapses after the MLP-only split | Refuted as incomplete |
| Decode-formatted DRAM width-sharded weights were incorrectly reused by generic large-M prefill | `AUTODEBUG.md` static analysis plus the focused prefill/cache A/B | Confirmed |

## Fix

`OptimizedDecoder.from_state_dict` now always materializes the prefill QKV, O,
gate, up, packed gate/up, and down weights as interleaved DRAM.  When and only
when `decode_matmul_mode="dram_sharded"`, it also materializes dedicated DRAM
width-sharded `*_decode_weight` copies.  Prefill call sites use the interleaved
copies; decode call sites use the sharded copies.  The selected
`decode_matmul_mode="shard_advisor"` default is unchanged and has no duplicate
weight overhead.

The test suite includes a regression guard that checks this layout contract.
The likely lower-level mismatch is documented, but not relied on for the fix:
Falcon QKV with prefill `grid_x=11` has `per_core_N=15`, while the 8-bank DRAM
shard storage width is 20 tiles.  The generic prefill validation does not reject
that DRAM-backed operand-B mismatch.

## Post-fix verification

`results/autofix/prefill_stage_localization.json`:

| Candidate | Attention PCC | Output PCC | K-cache PCC | V-cache PCC |
|---|---:|---:|---:|---:|
| Advisor MLP-BFP4 | 0.99999959 | 0.99998603 | 0.99988097 | 0.99985982 |
| DRAM BFP4, 24 cores | 0.99999959 | 0.99998603 | 0.99988097 | 0.99985982 |
| DRAM BFP4, 48 cores | 0.99999959 | 0.99998603 | 0.99988097 | 0.99985982 |

`results/autofix/candidate_sweep.json`:

| Candidate | Prefill PCC | Decode PCC | Warm prefill (ms) | Traced decode (ms) |
|---|---:|---:|---:|---:|
| Advisor MLP-BFP4 | 0.99998603 | 0.99999903 | 3.248070 | 0.782636 |
| DRAM BFP4, 24 cores | 0.99998603 | 0.99999904 | 3.264622 | 0.813781 |
| DRAM BFP4, 48 cores | 0.99998603 | 0.99999909 | 3.250064 | 0.813254 |

The comprehensive post-fix sweep independently reproduces the result.  The
DRAM controls are correct but 3.9% slower in traced decode than the selected
advisor configuration.

## Provenance

- Fresh-context report: `AUTODEBUG.md`, SHA256
  `cacfe6c6f12cfc0e62eed9b1e6f2a7ad7b0b5003e851427cc6153209257a72f1`
- Preserved failing artifact: `results/autofix/full_candidate/candidate_sweep.json`,
  SHA256 `9b46f5162d9f406d1c63754bad8e50035b48f9907046bcea923410ccd3c086a6`
- Fixed focused sweep: `results/autofix/candidate_sweep.json`, SHA256
  `b4be85cbeb6684adf027f9038097cf3aaab064d4656c23765ed1917727ba516e`
- Fixed prefill localization: `results/autofix/prefill_stage_localization.json`,
  SHA256 `099ab88b079cd4bc7219bfed6f66597ea9a8c0156c4346a458769a2bfc316732`
