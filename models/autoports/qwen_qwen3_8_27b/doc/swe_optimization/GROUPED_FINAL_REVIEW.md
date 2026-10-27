# Independent local optimization review

Verdict: **clean-pass**,27 October2026. This passes the scoped local grouped
recurrence optimization. Serving, clean SWE accuracy, and SWE100/500 remain
pending; the runtime flag stays default off.

A fresh read-only reviewer checked final production routing and ownership,
raw outputs/state receipts, trace lifetimes, watcher logs, clock repair,
compiled kernel descriptors, and both repaired-clock reduced profiles.
No implementation defect or unresolved local output anomaly was found.

- B19 automatic three-head and B32 varying nonzero-state/mask checks preserve
  every inactive row exactly. Primitive two-/three-/four-head families pass.
- Full64 B32 strict tracking includes program-cache allocations and no exemption.
  Five384-entry convolution/recurrent state maps per arm and three four-device
  continuation-logit sets match. These maps are not an exhaustive KV/cache-state
  proof; continuation tokens/logits and lifetime checks provide complementary
  evidence. One strict timed repeat is separate from three clean timed repeats.
- Every shared B32 qualitative prompt completes within512 tokens and matches its
  split control, including English, correct JSON counts and complete code.
- Trace IDs/capture counts remain stable through ragged/reset requests and
  continuation/eager work; autonomous inner loops have no host refresh/readback.
- No-inline watcher10 passes with all watcher features enabled. The earlier
  inline fabric startup overflow is instrumentation footprint, not timed evidence.
- Raw paired full64 medians rederive1.268112× B16 and1.442454× B32 decode gains.
- Matched fixed-clock reduced profiling rederives3725.584→3398.417µs,
  saving327.167µs with948→828 device ops. Prefill topology is unchanged;
  its0.372% single-window variation supplies no prefill gain claim.
- Fused-op fidelity labels are checked against nested attributes and actual
  descriptors: projection LoFi, head HiFi2, recurrence/reduction HiFi4.
  Derived reports exclude only four external predecessor edges per phase;
  raw rows/timestamps/durations remain intact and hashes match.

The reviewer independently checked runtime file hashes against B32 receipts.
The older strict harness hash differs only by later timing-only qualitative-skip
options, not acceptance/state/lifetime logic. All local runtime evidence uses
compiled backend3fbf3dbf1e88112aae1ca39cff9d1bc8a1da66e8 in Ubuntu24,
which differs from the pinned CI image. Deployment evidence must be bound
separately before serving claims.

Full review and raw artifacts remain local at
`/home/mvasiljev/build/qwen38-swe/GROUPED_FINAL_REVIEW.md`.
Compact comparisons are linked from GROUPED_RECURRENCE.md; no CSV/Tracy/raw token
arrays are published. The historical prototype review is superseded by this one.
