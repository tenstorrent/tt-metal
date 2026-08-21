# REVIEW_RECORD — pin 18 (cc1plus 664bbf81b2ca)

Reviewer: orchestrator session (Claude, operated by nkapre@tenstorrent.com) — independent of lane ES (author); gates executed by lane ES, independently spot-verified by the orchestrator (frozen FAIL-set diff rc=0 vs the pin-17 reference; corpus OFF base-vs-mine tsv diff = byte-identical; ON-25 delta list = 24 variants in laneES-evidence-20260821/corpus/on25-flipped-24-variants-firstguard.txt).

Candidate: sfpi-gcc nkapre/sfpi @ b19915b18e2 = pin-17 + the lane ES P0 hang guard.
Installed: cc1plus 664bbf81b2ca0ba6fcde5eb9f796690647960be3af928ac0974abde2b7629127 (pin-install-fast, manifest appended).

## Reviewed
- ES-F1: no-exec replay record ingested inside the mod-6 store deferred-RWC retirement window while ADDR_MOD_*_SEC6 registers are reprogrammed = Tensix math-pipeline wedge (BH; two deterministic device reproductions, reset-bracketed). BH REPLAY expander semantics documentation gap recorded as an audited fact in rvtt-cost.md.
- Guard: named refusal mod-write-noexec-record-composition-unaudited in rtl-rvtt-dst-autoincr.cc — the mod-write terminator refuses when a no-exec record can land in its unaudited retirement window; lifts only when the BH expander semantics get audited.

## Gates
- Full rvtt.exp: FAIL set byte-identical to the pin-17 frozen reference (independent diff rc=0).
- Corpus: OFF base-vs-mine byte-identical; ON-25 delta vs installed pin-17 = EMPTY (.text identical 3213/3213) — the final window-rule guard preserves every witnessed-good composition byte-exactly (the 24-variant list was a narrowed-away DRAFT guard, superseded — the final identity proof is IN-REPO: attachments/pin18-on25-identity-certificate.txt (manifest sha256s equal, byte-identity re-verified at commit), and the reset-ledger timeline grounding the collateral exoneration is attachments/pin18-reset-ledger-20260821.txt; the off-box laneES-evidence dir is corroborating, no longer load-bearing).
- Post-fix device (flush-bracketed, solo): divint32floor corr node PASS; lcm record-hoist knob leg PASS (previously wedged the device).
- Collateral adjudication: the pin-17 sweep divint32floor corr FAIL and log-fresh corr FAIL were device-poisoning collateral of the lcm knob-leg hang (reset-ledger timeline in ES evidence); log-fresh re-measured solo = WIN -9.0% all-corr-PASS.
