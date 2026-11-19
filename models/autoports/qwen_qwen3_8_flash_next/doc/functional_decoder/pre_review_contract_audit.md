# Pre-review functional contract audit

This is an AutoFix source audit, **not** the independent stage review and not
an acceptance verdict. No hardware was opened for this audit. The user goal,
functional-decoder skill, public implementation, README, context contract,
and the current correctness harnesses were compared.

Additional coverage identified beyond the already pending long runs,
performance, fresh-process PLE capacity confirmation and PLE batch limit:

1. Current QSA continuation evidence tests the attention component. The full
   public decoder's older short test checks fresh prefill and decode, but not
   resumed contribution windows, dropped-history sensitivity, or another
   prefill while its decode trace stays live.
2. The public tuple batch test starts every prefill request at zero. It does
   not establish tuple prefill with different nonzero history lengths, even
   though decode subsequently exercises different current positions.
3. Changing page-table contents during a live trace is proved by the selected
   cache component, while the complete decoder uses fixed permutations. The
   public module explicitly documents in-place table refresh with remapped
   cache contents, so the complete path needs a matching check.

The new `tests/probe_public_continuation.py` addresses these gaps without any
runtime changes. It uses complete real HF layers and the public decoder for
layers 0, 1 and 3. Independent prefixes have lengths 31 and 33; one tuple
prefill resumes 8 and 7 rows respectively. Each resumed row's whole output and
contribution must meet PCC 0.995 against the full HF prefill. Removing all
history must fail that same contribution gate. There is no matched-backend
exception in this harness. A third fresh 5-token request runs while a tuple
decode trace remains live; repeated outputs, state isolation and two changed
decode inputs per request are checked. QSA then changes both page tables and
physical K/V placement in place and reuses the same trace, checking output,
logical cache, indexer and position equality. CPU-only independent HF
continuation controls have passed for all three kinds; device evidence is
still required.

Documentation points to resolve before stage review:

- Keep separate fields for full public prefill, reduced attention/state-path
  prefill, and complete public final-token decode after a reduced prefix.
  `probe_gdn_full_context`, `probe_ple_full_context`, and
  `probe_qsa_full_context` already state this distinction. Their successful
  final decode must not become a claim that all prefix rows traversed the
  complete 10240-wide decoder or that a single public prefill of that length
  was allocated. The skill permits reduced long-layer evidence, with scope
  and remaining risk stated explicitly.
- The context contract remains correctly `in_progress` with no accepted
  reduction. Replace historical/candidate values with final-source evidence
  when available. In particular, the README's 5792/5793 PLE candidate and the
  context JSON's 5664/5665 candidate are not interchangeable physical limits;
  failed-probe resource retention and subsequent lifetime changes are still
  being investigated. Record the final process lifecycle and byte evidence.
- Keep largest tested batch and length **per layer kind and storage policy**.
  Tuple API support for up to 32 requests is not proof that the resident PLE
  model can fit every such batch/length combination. The API returns tuples
  and executes requests serially; it does not expose a dense packed batch.
- Default PLE setup prepares all 1..32 physical hash lengths, allowing any
  legal logical tail. An explicitly reduced `hash_lengths` set only prepares
  those lengths; callers must prepare every required tail before capture.
  Public sequence lengths themselves have no chunk/tile divisibility rule.
- QSA capacity/position bounds are caller preconditions: the public contract
  requires the existing prefix plus new tokens to fit the allocated cache.
  The device-valued position is not read back to enforce a host-side bound.
- Preserve the layer-0 fresh-prefill GR contribution anomaly as carried
  evidence. `linear_final_batch32.json` is a matched-control-policy pass,
  while literal whole-output and decode/resume gates remain strict. Do not
  summarize it as every literal contribution passing.
- Link final runtime hashes, watcher scans, separate warmed prefill/traced
  decode tables and CSV provenance. Historical passing tables must remain
  visibly historical after precision or memory-lifetime changes. Formal
  stage review and the stage-owned local commit remain separate pending gates.

No additional public arbitrary-length restriction or hidden host-value path
was found in the inspected wrapper. Its valid input contract remains BF16
`[1,1,S,10240]`, explicit state, and raw PLE token words where applicable;
padding is excluded from logical input rows and all logical outputs return.
