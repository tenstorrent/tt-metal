# OWNER RATIFICATION — castfp32tofp16a golden re-spec

Owner: nkapre@tenstorrent.com
Date: 2026-08-21 (this session; original in-session approval 2026-08-19)

Proposed ratification text (drafted by the orchestrator session):
  "I ratify the cast golden re-spec: golden = proven hardware cast semantics."
Owner's verbatim response: "amazeballs... approved"

Effect: the castfp32tofp16a correctness golden is DEFINED as the proven
hardware cast semantics (round-half-away, NaN->Inf, denorm flush, -0.0
handling — the 2^32-proof record of lanes CT/CX). laneCX's shipped
re-spec (1707546cdc, CRAQ 4/4 bit-exact) is retroactively covered by
this standing owner order.
