# Read-only semantic inspection proposal

This inspection is not a formal stage review and grants no acceptance. Exact source,
document, config and closed-evidence hashes are in `inspection.json`. Proposed gate
bindings are in `gate_binding_proposals.json`; they have not been applied. All output
is outside the model tree.

## Public decoder contract

The documented methods match the implemented class: setup through `from_state_dict`
and `create_state`, all-row `prefill_forward`, and one-row `decode_forward`. Hidden
input/output is BF16 TILE `[1,1,S,10240]`; PLE additionally consumes UINT32 row-major
`[1,S,16]` words containing raw token ID and validity. The external width is four
2560-wide residual streams, not an altered target hidden width. HF's pinned config
loader maps serialized `full_attention` entries to `indexed_attention`; representative
real layers0/1/3 cover GDN, GDN+PLE and indexed QSA respectively.

`functional_decoder.py:144–177` permits arbitrary positive logical lengths up to the
unchanged HF maximum, computes chunks<=32 and returns all logical rows. It does not
promise that every allowed length fits every physical allocation profile. Tuple API
validation permits1..32 independent requests and returns a tuple; physical PLE batch5
is a separate measured profile. It does not check that callers supplied distinct state
objects. No Cartesian product of the maximum batch and maximum context is implied.

State is caller-owned. Linear state includes convolution and FP32 recurrent state;
PLE also requires convolution and exact token-hash history. There is no explicit
linear position counter: continuation is encoded in those states. QSA has paged K/V,
three raw indexer keys, compressed keys, and a device INT32 *next absolute position*.
`qsa_chunked.py:75–80` advances that position once by the logical count; per-query
positions, complete four-token groups and the incomplete causal tail remain device
operations. Prefix restoration must restore all corresponding state, not position alone.

The public QSA factory uses32-token pages and capacity128..262144 in multiples of128.
Host setup tables must be full permutations. Device table input and in-place remapping
are supported as caller preconditions: mappings must name allocated pages and cache
contents must be coherently remapped. The selected-cache reader validates layout/dtype,
but host-value permutation/position bounds are not redundantly read back inside forward.
The API does not itself check `position+S<=capacity` on the host, nor implement arbitrary
mask holes or multimodal positions. These limits are documented; they must stay explicit.

Torch conversions/weight construction/hash allocation occur in setup. Tracing requires
all used physical math/hash/I/O signatures warmed before capture, stable input/state
addresses, and external input refresh/restoration boundaries. `create_state` alone is
not a promise that every decoder math program is warm. The current README says so.
Current public continuation evidence verifies independent positions, unchanged other
state, changed/repeated inputs and live unseen logical lengths65/97/129. QSA additionally
proves same-trace page remap with stable addresses. Final Watcher coverage closes all
three representative layer kinds.

The API contract is therefore sufficiently documented for its causal-text functional
scope. The preliminary README/context acceptance banners and null final accepted-context
fields still need reconciliation after explicit review; this inspection does not set them.

## Largest feasible prefill interpretation

The detailed prepared-profile analysis is copied to `capacity_assessment.md`.
The reduced full resident PLE public profile supports a proposal of1..4768 logical rows:
all32 physical/hash shapes prepared, fixed resident packed table/BFP8 experts, batch1,
and the documented trace/output-release lifecycle. Complete literal-HF4768 and4767 pass;
a fresh complete4769 call fails at the first32-row GR temporary allocation. The failure
is fragmentation, not aggregate exhaustion. It does not prove that an optimized alternate
implementation could not exceed4768.

Some reduction of the advertised262144-row resident call is independently necessary:
BF16 hidden input alone is5,368,709,120B versus514,902,528B free after the recorded resident
model/state setup. No runtime-duration argument is used.320 residue-search observations,
all32 terminal proxies, fresh complete boundary calls, stable post-probe allocation
baselines and fixed-descriptor chunk sources provide finite evidence for the proposed
profile range. The proxies are not32 full public prefills, and no exhaustive integer
monotonicity theorem is claimed. No additional capacity experiment is presently required
before review of this explicitly bounded inference.

Maximum4768 repeated prefill releases the decode trace/output first. Live-trace
interleaving at shorter lengths is separately tested. Do not advertise4768 while a decode
trace remains resident, or batch5 at4768, from these records.

Long GDN and QSA evidence covers real attention/state prefixes through262144 and full
public final-token decoder bridges. Reduced prefix rows do not execute token-local MoE.
That scope is permitted by the reduced layer-level long-harness contract and must remain
visible. The PLE advertised-state-context run was still **running** when this inspection
was written: its mutable partial result was not hashed or promoted. `largest_feasible_prefill`
therefore cannot close yet under its existing dependencies.

## Controlled numerical findings and remaining review

The current declared leaf checks satisfy their own predicates except the running PLE
long dependency. This is not equivalent to all literal contribution rows passing:
layer0 fresh batch32, PLE fresh batch5, PLE unseen fresh prefills and the retained3707
finding have explicit independent changed-settings HF controls. Literal whole-output and
resumed/decode contribution bars remain. The controls retain>=.995 numerical thresholds,
but use a changed representation/accumulation comparator and therefore require explicit
model-specific review. They are not failures reproduced by the unchanged literal backend
and do not qualify for an automatic “original model same” exemption.

The proposed documented-contract binding checks actual API/state facts and requires an
explicit reviewer disposition of the supported scope. The proposed largest-prefill
binding retains all long/HF/capacity dependencies, binds4768/4767/4769 and the physical
input calculation, rejects universal optimality/maximum-live-trace claims, and requires
an explicit disposition of the prepared-profile range/reduction reason. No generic
`passed:true` substitutes for those judgments. Formal stage review, final documentation
hash reconciliation and the user-required local commit remain separate obligations.


## Owner reconciliation after final runs

This original read-only inspection is retained as a dated preliminary snapshot. Its
then-pending PLE dependency closed successfully on2026-11-13 at07:03:43UTC:262143
reduced PLE/hash/GDN prefix rows and a complete public final decode at context262144,
with literal gates/repeats/hash/runtime audit passing. It is not262144 separate reduced
component readbacks. Exact four-file formatting and post-format CPU source closures
follow the recorded transition; actual final performance uses formatted sources.
The final `doc/context_contract.json` distinguishes state context from the PLE resident
single-call limit. The owner adopts the prepared range1..4768 inference described here;
no maximum-live-trace, batch/context Cartesian-product or universal-optimality claim is
made. Changed-settings numerical controls are not unchanged-HF exemptions. Formal
independent stage review remains a separate requirement; this document grants none.
