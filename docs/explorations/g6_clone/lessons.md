# Lessons from earlier ops

Handed in by the lead of the model bring-up; measured on earlier ops on this hardware. Advice, not a
requirement: each entry says in which situation it was measured and how often it held. An entry whose
situation is not true of this op does not apply: delete it, or add a line `- Matches here:` that names
the line of this bundle that makes it true. In the limits, dN is a decode call with N sequences and pN a
prefill call of N tokens.

### RB-036: Place each kernel group's cores as a few rectangles (few core ranges per RISC)

- Situation: A multi-core op runs inside a trace and the cores of a kernel group are scattered or
  choose their NoC one by one.
- Finding (do): Place every group as a few rectangles by choosing which core gets which role
  (descriptor only, kernels untouched); count core ranges per RISC for every shape; accept a
  slightly longer kernel for a much shorter start gap, judged as kernel + gap in the model. The
  dispatcher writes configuration, binaries and launch messages once per core range, so many ranges
  delay the program on every call.
- Evidence: held in 9 measurement(s) on 2 earlier op(s); largest measured gain 87 % of the measured
  time. A basis for looking, not an expectation.
- Limits: no gain measured at: same traces; R60 in the model, start gap per call.
- Matches here: risks.md, R8 (per-call overhead on the tiny cases) and R4; sources/local/contract.md:62 (the call runs inside a captured trace) and the multi-core split of tensix.md, Candidate regimes.

### RB-053: Do not compare programs whose cores end at different times on profiled kernel time alone

- Situation: Two versions of a multi-core program differ in which cores run a kernel and when those
  cores end (a role moved to a runtime argument, idle cores that return at once), and the difference
  to judge is a few percent of kernel time.
- Finding (do not): Do not accept or reject the change on the device profiler's kernel duration
  alone; measure the unprofiled host time of blocking trace replays of the program (steady
  repetition and with the program before it), and the model step, and judge on those. With the
  profiler on, every core writes its buffers to DRAM when its kernels end, so a version in which
  many cores end early reads slower although nothing in the plain build changed; a limit set on the
  profiled reading rejects a real gain.
- Evidence: held in 2 measurement(s) on 1 earlier op(s). A basis for looking, not an expectation.
- Limits: none measured.
- Matches here: tensix.md, Candidate regimes: the pieces of a split own unequal page counts (48 pages of [16,3,2560] or 1940 tiles of p4 over the chosen cores), so versions of the program differ in which cores end early.
