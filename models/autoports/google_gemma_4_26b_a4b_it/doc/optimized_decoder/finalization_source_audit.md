# Finalization source audit

Source/CPU-only snapshot, 2026-09-26. Runtime SHA256:
`4a9050497d5d4734eb40715d8942adfc33413b958358a209dbbd31c9132a146e`.
No hardware execution or runtime changes. This checks the current actual-input
candidate and public methods; it is not final stage signoff. No concrete runtime
correctness defect was established in the inspected valid-input paths.

## Bounded findings

### SHOULD-FIX: close changed-policy contract evidence before promoting defaults

The source preserves the contracts below, but saved `default_batched_*`,
`default_prefix_continuation_*`, `default_request_reuse_*` and `default_long_*`
results name runtime `9fb650ad97e124dcee102e697542e3dba342d469af1b58ceb20eeda0c6e05088`.
They predate native SDPA, BFP8 caches and direct QKV. The current actual-input
1025/512 and 4096/128 runs do not replace batch/request/max-context coverage.
This is the existing final-validation task, not a request for another geometry
or precision matrix.

| Boundary | Source contract and concrete risk being covered |
| --- | --- |
| B32 logical users | `tt/optimized_decoder.py:582-599` dispatches each logical row with its own position and page-table slice. Padded rows are not promoted to active users. `NativePagedAttention` sees logical B1. `GeneralizedRouter` static output buffers are consumed into a new scatter result before the next sequential slot (`:708-726`). Existing/new batch harness has disjoint random pages and actual fixture windows; `tests/batched.py:108-110` still uses position33 for every slot, so it does not by itself prove heterogeneous per-slot positions. No source misindexing was found. |
| Prefix continuation | `:512-531` slices the requested page-table row and uses per-token device-position decode updates. It does not refill an occupied prefix page. `tests/prefix_continuation.py:87-107` checks boundaries0/31/33/65 and exact preservation of earlier rows and the other slot. |
| BFP8 prefill/decode asymmetry | `:951-957` explicitly casts fill values to destination cache dtype. `:936-938` keeps decode updates BF16, including when destination storage is BFP8; `tt/fused_decoder.py:631-642` uses those updates and forwards the device position/table to native attention. Matching BF16/BFP8 pairs and32-token pages are checked at `:484-491`. The contract harnesses now allocate the selected cache dtype. |
| Nonaligned prefill | `:538-570` pads only the physical work, rounds fills to32 tokens, and slices all returned chunks back to the logical valid length. A short sliding tail is padded to1024 queries; `:962-984` prepends the prior tail and drops the corresponding dummy query outputs. Fresh requests clear the tail at `:536`; prefix continuation bypasses it. The padded final cache rows remain within the documented request allocation. |
| Full-context native SDPA | `:510` preserves the HF262144 limit for prefill. `:871-880` retains FP32 destination and full-DST synchronization. At262144 tokens,32-token pages give8192 logical pages; the endpoint is also128-token aligned, so the documented read padding does not need an extra page. At262143, one padded token remains within that same allocation. No lowered maximum is introduced. |
| Direct-QKV cleanup | `:1036-1040` releases only broadcast rows/lane masks. Public fresh prefill pads M to at least32 (`:540-546`); nonzero-prefix continuation and B32 dispatch use M1 direct decode. Thus the delegated prefill branch `:1114-1116` does not need those released decode-only buffers. |

The bounded final contract run should consume the selected actual-input policy,
not an old default hash. Both262144/262143 actual fixtures and hash-bound sampled
HF references are ready in `actual_text_long/`; `LONG_CONTEXT_ACTUAL_INPUT.md`
and `verification.json` record their CPU checks. The long-context test includes
traced cache-consuming decode at the last two positions (`tests/long_context.py:129-169`).
A sampled reference remains `scope=subset`, not all-output parity.

The parent has already begun adding actual fixtures to the batch/prefix/reuse
harnesses. Their source changed during this audit; the statements above describe
the inspected versions and do not assert that those new device runs have passed.

### SHOULD-FIX: finish four required actual-input fidelity controls

`.agents/skills/optimize/SKILL.md:193,381` requires legal LoFi/HiFi2 comparisons
for each dominant decode projection group at fixed dtype. An inventory of the
179 available actual-fixture reports carrying `precision_policy` found only
LoFi for both `decode_expert_fidelity` and `shared_decode_fidelity`. Historical
expert sweeps and `tuning_shared_HiFi2_layer{0,5}.json` cover legal configurations,
but do not supply the matched current actual-input comparison. The parent has
accepted one expert-HiFi2 and one shared-HiFi2 control per layer kind; those four
controls close this specific gap without repeating the geometry sweeps.

| Group | Source wiring | Existing evidence / bounded closure |
| --- | --- | --- |
| Routed expert gate/up/down | `:160-182` builds and passes one decode compute config to both sparse projections (`:191,198`). | Current sliding gate8/down4 and full gate4/down4 LoFi have actual-input passes. Run the planned HiFi2 control at the same selected dtypes/geometry. It covers both projections; separate per-projection fidelity switches are not required. |
| Shared MLP gate/up/down | `:1442-1448,1497-1504` builds/uses the configured compute policy; `:1522-1535` supplies it to both projections. | Current sliding BFP8 and full BFP4 LoFi families have actual geometry/packed-separate evidence. Run the planned same-policy HiFi2 control for each kind. |
| Direct QKV | `:1042,1124-1129` inherits the configured compute policy and passes it directly to native linear. | Sliding direct LoFi fails actual512 (`actual_direct_isolate_lofi_no_gate_stress_layer0.json`, min.994262776); HiFi4 with centered routing passes (`actual_direct_isolate_hifi4_center_stress_layer0.json`, min.995546781). Direct HiFi2 headline exists (`actual_direct_output_qkv_hifi2_layer0.json`, min.995341957). Parent is completing full direct HiFi2; no extra lane-topology rerun is needed. Any newly selected cumulative policy still needs its normal stress gate. |
| Attention output | `tt/fused_decoder.py:538-564` uses the attention compute config in the unchanged production path. | Parent's current output probe already runs explicit geometry, LoFi/HiFi2 and adapted DRAM candidates. Do not treat this as another missing sweep. Integration/final rows must reflect the selected result. |
| Native SDPA | `:871-895` has an independent fidelity knob with full-DST sync. | `actual_native_sdpa_config_commands.json` and associated probe metadata already cover LoFi/HiFi2, grid32/64/110, and both attention kinds. No additional fidelity request. |

No blanket HiFi4 sweep is required. The router's selected projection is an FP32
multiply/reduce path (`:691-697`), not a matmul with a meaningful LoFi/HiFi2
switch. Norm/rotary SFPU work likewise does not create a new projection-fidelity
requirement. There is no LM head or CCL in this single-device decoder stage.

### SHOULD-FIX: refresh final policy and allocation reporting at selection

`doc/context_contract.json:47-64` still records BF16 caches and retained lane/
FP32 broadcast buffers. It explicitly labels that evidence historical, so this
is not a concealed capability claim, but it must be updated for the selected
BFP8/direct-cleanup policy. At unchanged page geometry the K+V tile payloads are
1,140,850,688 bytes sliding and570,425,344 bytes full per262144-token request
(BFP8 tile1088 bytes, versus BF16 tile2048). Preserve the262144 capability and
record actual retained buffers; `qkv_decode.setup_cleanup` reports which direct
cleanup actions occurred. Parent final-context execution remains necessary to
validate total allocation.

Keep phase precision explicit. `qkv_fidelity` changes decode compute via
`:1178-1187`; delegated prefill uses `self.source.compute` at`:1343-1356`, even
when direct decode uses LoFi. `prefill_attention_fidelity` changes the SDPA
compute paths (`:920-933,969-1005`), not QKV, output projection or shared MLP.
The final per-op report must establish actual input/weight dtypes and fidelity
as required by optimize skill line188; constructor/policy labels alone cannot.

Also read both `passed` and `decode.passed` in ordinary decoder JSONs. For
example `actual_direct_combined_1025_512_layer0.json` has `passed=true` for
prefill while `decode.passed=false` and minimum.994311051. This is the existing
report schema, not a runtime defect; the final table must not count that row
as a whole-layer pass.

## Selection snapshot

`actual_direct_geometry_base_layer{0,5}.json` names the audited runtime hash and
uses direct FP32 QKV with BFP8 attention weights/cache, generalized routing,
BFP4 prefill experts and LoFi prefill SDPA. Sliding uses centered routing,
HiFi4 QKV/native SDPA, BFP8 decode gate/shared and BFP4 expert down; full uses
uncentered routing, LoFi QKV/native SDPA, BFP4 decode gate/down/shared and BFP8
expert input. Their128-step minima are.995336754/.995221962. Later geometry
and output trials deliberately change pieces of this snapshot; it is neither
a selected final default nor proof that all later changes compose.

The public factory still defaults to native_sdpa=False and BF16 caches at
`:267-271`. Final promotion, same-policy contract validation, actual stress,
Watcher checks and final default profiler rows remain parent-owned completion
work. This audit adds no unrelated experiments.

Follow-up: `tests/batched.py --heterogeneous-positions` now adds the bounded
B32 positions32..63 check with independent per-slot HF references. CPU input,
oracle-wiring and position-buffer checks pass; hardware validation remains
pending. `actual_contract_inputs.md` records the commands and exact selection.
