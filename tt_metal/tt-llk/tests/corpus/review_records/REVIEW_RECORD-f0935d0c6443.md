# REVIEW_RECORD — pin 19 (cc1plus f0935d0c6443)

Reviewer: orchestrator session (Claude, operated by nkapre@tenstorrent.com) — independent of lane EV (author); gates executed by lane EV, verified by the orchestrator against laneEV-evidence-20260821 (EV-F1.md, SHA256SUMS 100 files).

Candidate: sfpi-gcc nkapre/sfpi @ 2b26eb61700 = pin-18 + the lane EV macro-planner inter-row drain fix.
Installed: cc1plus f0935d0c6443aad2c28abc70802ee42631cef650bbbdfeccbe338960c850fcae (pin-install-fast, manifest appended).

## Reviewed
- EV-F1 P0 wrong-code: emit_planner_run placed rows>1 back-to-back on a pinned VD with drain only at run end — 8 same-VD launches one cycle apart with 3-slot pending events = raced writebacks (corrupt values proven = float(int(input_bits))). Class PREDATES the unroll flag (bare 8-row shape mis-emits at -mtt-tensix-macro-planner alone, BH+WH). Replay former and unroll pass exonerated by control (no-replay leg fails identically).
- Fix: form_region derives an inter-row drain obligation (drain>0, rows>1, non-CC, fixed-VD value-carrier; new is_store_only spec) and emission places the full derived drain between rows — soundness = stream identity with the silicon-proven rolled calendar. Alternating/store-only/CC envelopes keep today's bytes.

## Gates
- dg macro-planner + replay-loop-unroll families 1141/1141 (4 new twins incl. unroll-composition; mul24-commuted twin re-pinned off an unproven race stream).
- Full rvtt.exp 5393 PASS, FAIL set line-identical to the frozen reference.
- Corpus OFF 3216/3216 .text identical; ON-25 delta = exactly 1 TU (mulint32-fresh latent-hazard hardening; fixed bytes CRAQ + device corr PASS). Next-sweep watch: mulint32-fresh ON-25 bytes change (enumerated, green).
- Device: signbit knob leg PASSES on the exact fixed bytes (flocked solo, bce8181e); 8-row knob-CRAQ matrix pin-vs-fix all green.
- Blast radius: every math.elf in both weeklies + corpus identity legs scanned — exactly 2 class instances, both closed.
