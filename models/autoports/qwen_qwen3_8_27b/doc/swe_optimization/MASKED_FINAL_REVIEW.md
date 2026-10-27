# Independent masked-state stage review

Verdict: **clean-pass**, bounded local stage,2026-10-27. Reviewer:
`/root/qwen_masked_review`, fresh xhigh session, read-only source/artifact
inspection and CPU recomputation; no hardware execution or implementation edits.
The full review stays local at `/home/mvasiljev/build/qwen38-swe/MASKED_FINAL_REVIEW.md`.

The reviewer verified default-policy compatibility, canonical-mask routing,
disjoint FP32 state ownership, complete CB draining, unchanged HiFi4 compute,
all real-layer/watcher checks and inactive nonzero-state preservation.
Full64 strict tests match tokens, five complete384-entry recurrent-state
maps and all continuation logits; all shared qualitative outputs complete.
A separate clean three-repeat pair gives1.132216× decode gain, stable4 traces
and4 captures, no inner-loop host refresh/readback and unchanged TTFT.

The same-clock reduced profile gives3723.083→3540.077µs,896→888 device ops.
The reviewer recomputed envelopes, every raw/derived window invariant, exact
signature changes, fidelity interpretation and unchanged prefill. Only the
state-selection and copy disappear per ASIC. Raw parser assertions are retained.

Final inspection caught a formally incorrect generic-op declared output. The
read-only mask now precedes out/final; final FP32 state is last. Exact source
comparison proves descriptor/runtime-address/kernel/math construction and
helper returns unchanged; the final B16 tracking replay passes72 exact checks.
Old full/profile artifacts keep their original source hashes; see the compact
API-order proof. No unresolved local implementation issue remains.

This passes only the named local fixtures. Native-context serving behavior,
cache pressure, larger-cohort accuracy and SWE100/500 remain required for the
overall goal. The flag remains default off until serving gates are met. Only
code and compact docs are published; no raw CSV/Tracy/token arrays are included.
