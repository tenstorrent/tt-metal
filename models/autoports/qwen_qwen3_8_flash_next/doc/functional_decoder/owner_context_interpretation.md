# Proposed final context schema and non-circular owner closure

These files are proposals only; nothing is applied to the model tree. The new
`owner_inspection_gate_plan.json` supersedes the earlier proposal that required a
reviewer-disposition field inside each semantic gate.

The dependency order should be:

**Closed numerical/source leaves → owner semantic/capacity inspection → independent
stage_review → commit closure.**

Owner inspection records describe what was inspected, bind exact final source/docs and
evidence, and state the owner's explicit supported-profile inference with
`independent_review_required:true`. They do not require a stage-review verdict as an
input. The later stage-review gate independently accepts or rejects those inferences
and controlled findings. There is no circular semantic-gate/review dependency and no
owner-set blanket stage-pass flag.

`context_contract_patch_proposal.json` supplies a conditional RFC6902 patch, preserving
historical evidence and per-kind identities. It must **not** be applied until the active
PLE long run closes and all existing long-context dependencies pass. No active artifact
was hashed. After those conditions:

- `supported_context:262144` means autoregressive state/context, not one resident
  prompt call. The scope is explicitly supported by reduced state-prefix tests and
  complete public final-token bridges.
- Global/per-kind `largest_tested_prefill_context:262144` is labeled **reduced
  state/attention-prefix** coverage; prefix rows do not claim full token-local MoE.
- Separate `largest_tested_full_public_prefill_context` values are129 for GDN/QSA
  and4768 for PLE. The global4768 field is explicitly a maximum across kinds;
  common full-public tested length is129. These are test extents, not an assertion
  that every fresh contribution comparison is literal-strict.
- PLE's owner-declared single-call prepared profile is1..4768, with full real-HF4767
  neighbor and fresh4769 fragmentation failure. The exact physical reason some
  reduction is required is5,368,709,120B input alone versus514,902,528B resident free.
  It does **not** reduce total decode/state context to4768 or4771.
- No claim covers universal allocator optimality, maximum prefills with a live decode
  trace, or batch5 at the maximum prefill length. The profile prepares all32 physical
  signatures before large I/O and releases trace/output before repeated maximum prefill.

The owner capacity gate keeps every existing long-context/HF/capacity dependency.
Controlled fresh-contribution anomalies stay explicit and numerically unchanged; they
are not accepted by the physical argument or relabeled as literal-HF passes. Stage
review remains responsible for the required independent disposition.

Final adoption should also reconcile obsolete pending prose in existing per-kind
status/history fields, without deleting prior failures. Formal accepted-bound fields
may remain null during pre-review while the separate owner-declared inference is clear.
