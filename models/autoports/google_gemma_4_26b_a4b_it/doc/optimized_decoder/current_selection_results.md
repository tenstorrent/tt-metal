# Completed configuration selection controls

This ledger covers all **40 completed controls** in the three linked journals:
16 output-projection, 15 integrated static-tuning and nine 512-step precision
runs. All use real weights and recorded text-derived layer inputs at PCC .995.
The four direct-attention BFP4 failures are retained. Final selected-default
validation and device-profile signoff are separate.

Each table cell is **warmed traced host median us / minimum decode PCC / status**.
Host samples are five batches of 30 fixed-position replays. They are not device
windows or refreshed-loop timings; 4096/128 and 1025/512 timings are not directly
compared. [current_selection_results.json](current_selection_results.json) stores
exact overrides, workload, raw timing samples, failure positions and all journal,
result, input and runtime hashes.

## Output projection, 4096/128

[actual_direct_output_commands.json](actual_direct_output_commands.json), result
pattern `actual_direct_output_{candidate}_layer{0,5}.json`. These are legal
test-wrapper controls on source `4a9050497d5d4734eb40715d8942adfc33413b958358a209dbbd31c9132a146e`;
they do not themselves prove integrated production dispatch.

| Candidate | Sliding | Full |
| --- | --- | --- |
| `i64k16` | 1076.194 / 0.995336754 / pass | 1162.457 / 0.995221962 / pass |
| `i110k16` | 1079.632 / 0.995336754 / pass | 1161.889 / 0.995221962 / pass |
| `d1k16` | 1128.509 / 0.995364756 / pass | 1256.527 / 0.995317677 / pass |
| `d2k16` | 1103.956 / 0.995364756 / pass | 1199.547 / 0.995317677 / pass |
| `d1k16_shard` | 1122.378 / 0.995364756 / pass | 1249.824 / 0.995317677 / pass |
| `i110k16_lofi` | 1079.124 / 0.995286691 / pass | 1161.558 / 0.995206910 / pass |
| `i110k16_hifi2` | 1078.960 / 0.995359222 / pass | 1161.057 / 0.995281013 / pass |
| `qkv_hifi2` | 1090.485 / 0.995341957 / pass | 1180.640 / 0.995319207 / pass |

`i` is interleaved output weight storage with 64/110 cores; `d1/d2` uses
DRAM-sharded weights with one/two workers per bank. `d1k16_shard` retains
the sharded output through the next normalization. All are legal; the
DRAM variants, including retained-shard, lose to explicit interleaved output.
The last row changes QKV fidelity, not output fidelity.

## Integrated static tuning, 4096/128

[actual_static_tune_commands.json](actual_static_tune_commands.json), result
pattern `actual_static_tune_{candidate}_layer{0,5}.json`, source
`51d24cd52a136a4095dfd74977dc4dda626437752f97a814cd79c6359ec4b868`.
Base uses QKV 110/K11, expert 44/K22, output 64 sliding/110 full, K16/DRAM,
output HiFi4; shared reader 1. Each row is its journaled isolated override.

| Candidate | Sliding | Full |
| --- | --- | --- |
| `base` | 1076.492 / 0.995336754 / pass | 1162.268 / 0.995221962 / pass |
| `out_k32` | 1079.616 / 0.995336754 / pass | 1165.998 / 0.995221962 / pass |
| `out_l1` | 1074.791 / 0.995336754 / pass | 1160.953 / 0.995221962 / pass |
| `out_lofi` | 1075.542 / 0.995286691 / pass | 1161.652 / 0.995206910 / pass |
| `expert_hifi2` | 1081.574 / 0.995274652 / pass | 1167.032 / 0.995221962 / pass |
| `shared_hifi2` | 1100.767 / 0.995483321 / pass | 1190.487 / 0.995221962 / pass |
| `prefill_2d` | 1076.267 / 0.995336754 / pass | 1162.275 / 0.995221962 / pass |
| `shared_r2` | Not run | 1158.437 / 0.995221962 / pass |

K32 and expert/shared HiFi2 lose; L1 output improves both kinds in this series.
Full shared reader2 improves over reader 1. Prefill medians base → 2D are
224043.243 → 224777.072 us sliding and 194209.884 → 194071.815 us full.
The full difference lies within the recorded sample spread; sliding is slower.
The existing prefill program remains selected. Small median differences are
selection observations, not robust speedup claims without final profiles.

## Direct-attention precision, 1025/512

[actual_final_precision_commands.json](actual_final_precision_commands.json),
result pattern `actual_final_precision_{candidate}_layer{0,5}.json`, same
`51d24...` source. Base now includes L1 output and full shared reader 2.

| Candidate | Sliding | Full |
| --- | --- | --- |
| `base` | 1065.665 / 0.995546781 / pass | 1105.775 / 0.996226953 / pass |
| `out_lofi` | 1066.546 / 0.995490377 / pass | 1104.566 / 0.996332265 / pass |
| `qkv_hifi2` | 1064.981 / 0.995523605 / pass | Not run |
| `qkv4` | 1050.295 / 0.993831699 / FAIL | 1088.080 / 0.989203513 / FAIL |
| `out4` | 1063.046 / 0.993118161 / FAIL | 1088.726 / 0.992120668 / FAIL |

BFP4 QKV fails six sliding / 16 full positions; BFP4 output fails six sliding /
four full positions. Matching BFP8 controls pass all 512. These final direct
projection controls close the historical attention-precision attribution gap;
Gaussian results do not determine the selected dtype. Sliding QKV HiFi2 passes
and improves its compatible base. Sliding output LoFi passes but loses against
HiFi4 in the compatible L1 path; full output LoFi passes and improves its base.

## Applied defaults and subsequent router repair

[optimized_decoder.py](../../tt/optimized_decoder.py), selected snapshot SHA256
`d6f4d858d7358f6d52332d9d1747a8bdb25cfa359f5c6f5aedde2f2e4f5d9f93`, uses
packed QKV 110/K11, expert 44/K22, BFP8 cache, BFP4 prefill experts, and explicit
output 64 sliding/110 full, K16/L1. Sliding uses QKV HiFi2, SDPA/output HiFi4,
expert gate 8/down 4 and shared8/reader 1. Full uses QKV/SDPA/output LoFi, expert
gate 4/down 4 with activation 8, and shared4/reader 2. Both retain generalized
routing with FP32 centering before BF16 conversion.

The full router changed after the controls above: a distinct recorded reuse
window at 2049 failed .994824794 on a fresh cache too. Isolated output HiFi4,
shared-reader1 and expert-activation-BF16 controls still failed. Centered
composite routing passed .998181955; FP32 routing passed .998172713.
[reuse_control_commands.json](reuse_control_commands.json),
[AUTOFIX_reuse.md](AUTOFIX_reuse.md) and [AUTODEBUG_reuse.md](AUTODEBUG_reuse.md)
localize this boundary without claiming measured expert rank swaps.

The new [validated_contract_commands.json](validated_contract_commands.json)
currently records passing full-layer heterogeneous-position B32, prefix65,
nine-request reuse, and BF16-cache prefix compatibility on the selected source.
Headline,512-step stress, both maximum contexts, Watcher and final profiles
remain subject to the acceptance index [optimize_checklist.md](optimize_checklist.md).
This ledger is not final signoff. It was generated from saved artifacts on CPU.
