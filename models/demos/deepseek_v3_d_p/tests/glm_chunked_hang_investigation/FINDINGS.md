# GLM chunked-prefill layer-77 hang — investigation notes (2026-07-31/08-01)

## The bug

CI job `Blaze - Chunked GLM (code_debug 55k) [bh_sc1_high_power]` in the
`Blaze Models Prefill tests` workflow intermittently hangs running:

```
models/demos/deepseek_v3_d_p/tests/test_prefill_transformer_chunked.py::test_glm_prefill_transformer_chunked_no_pcc[blackhole-glm52-mesh-8x4-L78-preload0-chunks_eleven-ten_iters]
```

Reference hang: run `30517835185`, job `90794060416`, commit
`b788cf2ffdc5389026bd4878376d11a9a0facac7` (2026-07-30T05:51Z). Steady
per-layer progress (tracy signposts `forward_layer_N_start/end`,
`MLA_START/END`, `MoE_START/END`) all the way to `forward_layer_77_end`
(layer 77 of 78, i.e. the *last* layer), then **total silence** for ~28
minutes until the 45-minute job timeout killed it. No error, no exception,
no crash — just silence.

## CI history survey

Sampled 30 recent failed runs of the workflow (2026-07-29 to 2026-07-31),
pulled 8 representative job logs. Four distinct failure categories:

| Category | Seen (of 8) | Signature |
|---|---|---|
| **Target silent hang** | 1 | Steady progress to `forward_layer_77_end`, then silence until timeout. |
| **Same spot, clean crash** | 2 | `TypeError: cannot unpack non-iterable NoneType object` at `tt_prefill_block.py:551` (`mla_out, mla_indices = mla_out`), always at layer 77, right after `MLA_END`/before `MoE_START`. |
| **Known FABRIC_2D + all_gather JIT-compile hang** (unrelated, already fixed) | 4 | Stalls at `forward_layer_0_start` (never reaches `_end`) — layer **0**, not 77. Fixed on main by `38b47ca2` ("Fix Fabric_2D hang in ttnn.all_gather unicast kernels", #51422, merged 2026-07-29) — confirmed an ancestor of `b788cf2f`, so already picked up; do not confuse with the target bug. |
| **Pure infra flake** | 1 | K8s/Helm allocation or docker-download failures, never reaches the test. |

## Root cause (found via static analysis, not yet confirmed live)

The silent hang and the `TypeError` crash land at the *exact same call
site*: `models/demos/deepseek_v3_d_p/tt/tt_prefill_block.py`, right after
`self.mla.forward(...)`, in the block that unpacks
`return_kv_intermediates` / `return_indexer_indices` results.

`MLA.forward()` (`models/demos/deepseek_v3_d_p/tt/mla/mla.py:1029`) has an
early-return fast path:

```python
if self.kv_only:
    return self._forward_kv_only(...)   # always returns bare None
```

`_forward_kv_only` (mla.py:1256) is documented as the "last-layer fast
path" and its return type is literally `-> None`. Crucially, this
dispatch happens **before** `return_kv_intermediates` /
`return_indexer_indices` are even inspected — so if a `kv_only` layer is
ever called with either flag set, `mla.forward()` silently returns `None`
instead of the expected tuple, and the caller's unpack:

```python
elif return_indexer_indices:
    mla_out, mla_indices = mla_out    # tt_prefill_block.py:551
```

raises `TypeError: cannot unpack non-iterable NoneType object` when
`mla_out` is `None`. This exactly matches the crash signature.

**Working theory for the hang:** the crash and the hang are the same
underlying condition (a `kv_only` layer incorrectly asked for indexer
indices) with two different outcomes downstream — most of the time it
raises cleanly, occasionally something else ends up waiting on the
missing indices/output forever instead of raising. **Not yet confirmed**
— we never caught the hang live to verify this, see below.

This is presumably not a purely deterministic bug for a given test ID
(our exact parametrization passed clean 30/30 times locally — see below),
which suggests `kv_only` classification for layer 77 and/or whether
`return_indexer_indices` is requested for it depends on some runtime
state (chunked-prefill migration handoff timing — see the `on_layer_complete`
/ migration-worker comments nearby in `tt_prefill_block.py` — is the most
likely suspect), not just static per-layer construction config.

## Instrumentation added on this branch

Two `logger` calls (no behavior change) to make the above visible instead
of a bare `TypeError`, or silent in the hang case:

1. `mla.py`, right before the `kv_only` early return: `logger.warning(...)`
   whenever `return_kv_intermediates` or `return_indexer_indices` is set
   on a `kv_only` layer — fires on *every* occurrence of the hazardous
   combination, whether or not it later crashes/hangs.
2. `tt_prefill_block.py`, right before the unpack: `logger.critical(...)`
   dumping `layer_idx`, both `kv_only` flags, and both `return_*` flags
   whenever `mla_out is None` unexpectedly — gives full state right before
   the `TypeError` would fire.

Grep for `[hang-investigation]` in a run's log to find these.

If the hang reproduces on a high-power box: check whether the warning
(#1) appears before the stall. If it does, the hang is confirmed to be
the same root cause manifesting differently downstream, and the next
question is why the caller doesn't crash from the `None` unpack in that
case (dead device wait? a different code path?). If it does *not* appear,
the hang is a different mechanism and this theory is wrong.

## Local repro attempts: 30 iterations, zero repro

Test requires `is_high_power()` (>=130W TDP) to run at all — it's
`skipif`'d otherwise (`test_prefill_transformer_chunked.py:1715-1718`,
guards `exabox.tenstorrent.com/power=14kw`). The box used for this
investigation reports `TDP_LIMIT_MAX = 0x4b = 75W` via `tt-smi`, so the
skip guard was locally commented out (not committed — see `git log` on
this branch, that edit was reverted) purely to get real execution instead
of an instant skip.

Ran the full test **30 times** on that (real, but low-power) box:
- 15x via `run_loop.sh` (only resets on detected fabric wedge)
- 15x via `run_loop_reset_each.sh` (`tt-smi -glx_reset` before every
  single iteration, mirroring CI's fresh-allocation-per-job behavior)

Result: **29 genuine `1 passed` runs** (each ~23-28 min wall clock, full
78 layers x 11 chunks x 10 inner iters) **+ 1 pre-existing device wedge**
(leftover ethernet-core-timeout state from before the loop started,
auto-detected and recovered via `tt-smi -glx_reset`, unrelated to the
target bug). Zero repro of either the hang or the `TypeError`.

**Leading hypothesis for the null result:** the bug may be genuinely
power/clock-gated. The test is CI-gated to `>=130W` boxes specifically;
if the race window (in whatever decides a layer's `kv_only`/indices-reuse
status, or in the MLA/indexer op scheduling itself) is timing-sensitive,
a box running at roughly half that TDP ceiling may just not hit it in 30
tries where a high-power box hits it in CI at a rate of roughly 1-in-a-few
runs (per the CI survey above, both crash and hang variants combined were
3-of-8 sampled failures, i.e. not rare on real CI hardware).

## What to do on a high-power box

1. Fetch this branch (`ipotkonjak/glm-chunked-prefill-layer77-hang-investigation`)
   — the `is_high_power()` skip guard does **not** need bypassing there,
   since a genuinely >=130W box passes it naturally.
2. Run `run_loop.sh` (and/or `run_loop_reset_each.sh`) from this directory
   — see `README.md` here for exact usage.
3. Run `poller_loop.sh` alongside in a second terminal/tmux pane. It stays
   silent until it sees 5 minutes of no log growth on the current
   iteration while the process is still alive, at which point it runs
   `tools/tt-triage.py -vv`, waits for that log to actually land on disk,
   *then* (only then) kills the hung process and resets the device via
   `tt-smi -glx_reset`. It never touches a live process before triage is
   captured.
4. Check `generated/glm_chunked_hang_repro/` (default `LOGDIR`, override
   via `GLM_HANG_LOGDIR`) for `triage_pid*.log` once a hang is caught —
   that's the artifact this whole exercise exists to produce.
5. Grep whichever `run_N.log` stalled for `[hang-investigation]` to see
   whether the instrumentation warning fired before the stall.
