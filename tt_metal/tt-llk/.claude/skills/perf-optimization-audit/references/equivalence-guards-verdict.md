# perf-optimization-audit — equivalence gate, guards & verdict

## Semantic-equivalence gate (every finding must pass this)
A perf change is only valid if the result is bit-for-bit (or, in `APPROXIMATION_MODE`, within the op's stated tolerance) identical. For each finding, **state the equivalence argument** and check the edge cases the rewrite could perturb: **NaN / ±Inf / denormals / signed zero**, rounding mode, and the exact boundary values (e.g. `x==0` for a step/shrink op). A `max(x,0)`-for-`v_if` rewrite, for instance, must match the original's NaN and −0.0 behavior. If equivalence can't be established, downgrade to a **suggestion requiring test confirmation**, never a confident win.

## False-positive guards (scalar-CPU intuitions that do NOT apply)
- **Branch reordering by likelihood buys nothing** on SIMD predication — all lanes evaluate every branch; there is no branch prediction and no short-circuit. Do not file "put the common case first".
- **The `v_else` in an if/elseif/else costs no extra compare** (it reuses the complemented predicate) — a 3-region op genuinely needs 2 compares; that is the floor, not waste. (Canonical: `ckernel_sfpu_heaviside.h` / `ckernel_sfpu_sign.h` are already at that floor.)
- **Do not hand-insert NOPs / manual scheduling into sfpi-compiled code** — the compiler owns it (provenance lens).
- **Do not replace `0.0f`/`1.0f` (or other common constants) with `vConst0`/`vConst1`** — current `sfpi-gcc` already lowers those literals to the const registers, so the rewrite is a no-op (check A#5).
- **Do not remove a drain/stall for perf** unless it is provably redundant *for ordering too* — that is a correctness call owned by `reconfig-stall-audit` / `instruction-latency-audit`.

## Verdict
- **PERF-WIN** — a numerically-equivalent change that provably removes instructions/stores/bubbles on an arch where it matters; give the before→after and the magnitude class.
- **SUGGESTION (needs test confirmation)** — plausible win whose equivalence or magnitude isn't proven statically; recommend the perf-counter + `run-test` check.
- **ALREADY-OPTIMAL** — at the instruction floor for its semantics (say why; do not invent a win). e.g. a 3-region op at 2 compares.
- **NON-ISSUE (guarded)** — matched a pattern but killed by a false-positive guard or the provenance lens; record it so it isn't re-raised.
- **UNCERTAIN** — can't resolve pinned compiler / throughput facts → abstain, mark coverage bounded (never guess a latency/throughput number).
- **Arch-divergent** — report per-arch (WH vs BH vs QSR differ in scheduling, latency/throughput, and available instructions); never collapse to one verdict.
