# One-replay operator audits

These are **operator-audit-only** 4096/1 profiles. They cannot supply headline
latency, full 128-position accuracy, stage telemetry, or same-workload speedup
claims. The selected policies still require complete 4096/128 profiles and
their separate 512-step actual-input checks.

`tests/slice_optimized_operator_fixture.py` produced
`operator_audit_layer{0,5}_4096_1.pt` and matching JSON sidecars from the existing
`actual_text_layer{0,5}_4096_128.pt` files. It copies the complete transported/raw
prefill, first transported/raw decode input, and first 4097 token IDs exactly.
Metadata retains original HF/text provenance plus original fixture/manifest
and helper hashes, and sets `evaluation_scope=one_replay_operator_audit_only`,
`headline_eligible=false`, `telemetry_eligible=false`. The combined
`operator_audit_fixture_manifest.json` records the exact CPU command.

The existing harness allocates extent
`ceil((4096 + max(128, steps)) / 1024) * 1024`, so steps 1 and 128 both retain
5120 cache tokens with 32-token pages. The audited decode is absolute position
4096, the same first actual input as the headline run. Later positions are not
represented. No HF boundary is regenerated and no dtype is changed.

## Bounded candidate set

The completed `actual_direct_mlp_geometry_commands.json` entries below all pass
128 actual checks. Recorded host medians establish which alternatives deserve
operator-level inspection; they do not prove that the requested BFP4/LoFi or
program geometry was actually dispatched.

| Candidate result file | Host median, us | Minimum PCC |
| --- | ---: | ---: |
| `actual_direct_mlp_expert44k22_layer0.json` | 1091.684 | .995336754 |
| `actual_direct_mlp_expert22k22_layer0.json` | 1104.955 | .995336754 |
| `actual_direct_mlp_expert_separatek22_layer0.json` | 1113.009 | .995336754 |
| `actual_direct_mlp_expert44k22_layer5.json` | 1179.688 | .995221962 |
| `actual_direct_mlp_expert22k22_layer5.json` | 1197.879 | .995221962 |
| `actual_direct_mlp_expert_separatek22_layer5.json` | 1208.932 | .995221962 |

Use the selected fused-44-worker/K22 first-replay rows from the final complete
4096/128 profiles as the reference, if that policy is selected. The minimal four
additional operator audits are:

1. Sliding 22-worker/K22 fused expert gate/up, against the 44-worker/K22 reference.
2. Sliding separate expert gate/up at K22, the fastest measured separate variant.
3. Full 22-worker/K22 fused expert gate/up, against the 44-worker/K22 reference.
4. Full separate expert gate/up at K22, the fastest measured separate variant.

Preserve each original command's exact overrides. Change only its fixture to
`operator_audit_layer{layer}_4096_1.pt`, `--steps` to 1, and output names into an
explicit operator-audit directory; add `--profile` and remove `--timing` to avoid
the five extra 30-replay host batches. Keep `--real`, `--length 4096`, and
`--verify-program-cache`. Capture through the normal parent-owned Tracy path
with op-info caching disabled. Do not substitute later factory defaults for the
recorded candidate policy. If final selection changes, use its complete profile
as reference and audit only the missing comparison members.

Do not add separate-K11 runs: they are slower than separate-K22 for both kinds
(1163.729/1247.120 us). If two more audits are needed for shared-MLP attribution,
use full `actual_direct_mlp_shared_separate_layer5.json` (1198.931 us) and
`actual_direct_mlp_shared_r2k11_layer5.json` (1185.447 us), compared at the shared
operator rows against the selected full profile. Those original commands also
have different expert settings from the candidate final K22 policy; their
whole-layer differences are not isolated shared-MLP effects. A full Cartesian
matrix, separate-K11 repeats, down88, and the slower shared reader-3 candidate
are not required for this bounded audit.

## Required evidence from each actual CSV

Keep the complete prefill/decode `tt-perf-report` tables with advice enabled,
then link the exact native rows for the dominant expert/shared operations.
For each compared projection, record:

- Native operation code, logical/padded M/K/N and active expert count.
- Actual input/weight/output dtypes and observed math fidelity; the full expert
  and shared candidates must show BFP4/LoFi where claimed. Sliding expert gate/up
  remains BFP8, while expert down is BFP4; do not label all weights BFP4.
- Actual core count, memory layout and program attributes, including K block,
  subblock and multicast/reader geometry. Constructor fields alone are not proof.
- Device duration for the complete corresponding projection group: separate
  gate plus up plus activation must be compared with the fused projection and
  its associated activation, with conversion/movement rows visible.

One decode signpost window is expected in each candidate audit. The final
profile still needs all 128 complete windows; `summarize_perf.py` intentionally
rejects a one-replay file as headline evidence. No candidate operator timings
or executed candidate profiles are claimed by this preparation document.

CPU verification checked exact saved tensor slices, original and derived
hashes, truncated decode storage, scope metadata, and rejection of a mismatched
source manifest or layer. Black and Python compilation passed. No TTNN import
or hardware execution occurred.
