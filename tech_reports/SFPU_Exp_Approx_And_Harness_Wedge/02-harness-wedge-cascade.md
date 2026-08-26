# 02 — The harness turned one wedged core into a wall of kernel failures

Files:
* `tt_metal/tt-llk/tests/python_tests/helpers/device.py`
* `tt_metal/tt-llk/tests/python_tests/helpers/test_config.py`
* `tt_metal/tt-llk/tests/python_tests/helpers/llk_pytest_plugin.py`

## Why this is in a kernel report

It cost a wrong conclusion. A sibling branch (`sfpu-wh-lut-accuracy-and-tti-scheduling`, the
SFPU LUT accuracy work) shipped a report claiming accurate `Sqrt` hangs the TRISC on Blackhole,
"reproducibly, in isolation, on pristine `HEAD`". It does not. What had actually happened is that
an unrelated experiment — forcing the clamped approx-exp path on at `ITERATIONS = 32`, i.e. the
very path 01 documents, which genuinely hangs — wedged the core, and *every* test run afterwards,
in that process and in every later pytest invocation, failed with the same timeout naming whatever
kernel it happened to be. Two of those were Sqrt, so Sqrt looked guilty. That claim has been
retracted on the branch that made it; the harness half of the story is here, because the harness
is what made a single dead core look like a pile of kernel bugs.

The harness had all the information needed to say "this core is dead" and instead reported 57-73
independent-looking failures whose count drifted run to run. That is what is fixed here.

## The protocol, and the two ways it desynchronises

The host drives BRISC through a counter and a double-buffered command slot
(`tests/helpers/src/brisc.cpp`). Both sides start at 0. The host writes the command into slot
`counter & 1`, increments its counter, and waits for BRISC to publish the same value:

```python
if common_counter & 1: write(BriscCommand1, cmd)
else:                  write(BriscCommand0, cmd)
common_counter += 1
while time.time() < deadline:
    if read(BriscCounter) == common_counter: return
raise TimeoutError(...)                     # <-- counter left one ahead, forever
```

`common_counter` is a module-level global, so it is per worker process and never reset.

**Desync 1 — an unacknowledged command poisons the process.** The increment happens before the
wait, so a timeout leaves the host counter permanently ahead of anything BRISC can publish.
Every later command compares against an unreachable value *and* writes to the wrong slot. Once
one command times out, every subsequent one in that worker times out too — regardless of whether
the core recovered.

**Desync 2 — a successful mid-run bring-up also poisons the process.** BRISC's counter is a local
in its `main()`, so it restarts at 0 whenever the core leaves reset. The bring-up path zeroes
`BriscCounter` in L1 to match — but nothing zeroed the host's `common_counter`. So any
re-bring-up after the first test left BRISC polling slot 0 while the host wrote slot 1. This one
is latent: the existing bring-up comment says "a failed bring-up is retried on the next test
instead of poisoning the rest of this worker's run", which is true of a *failed* one and was not
true of a successful one.

**And nothing ever attempted recovery.** `TestConfig.BRISC_ELF_LOADED` is latched `True` after
the first successful boot and was never cleared on a command timeout, so every later test skipped
bring-up entirely and went straight to another command that could not be acknowledged.

## The fix

1. **`reset_brisc_command_counter()`**, called from the bring-up right where `BriscCounter` is
   zeroed. Fixes desync 2.
2. **`BriscCommandTimeout`**, a distinct exception, and the host counter is resynced to what BRISC
   last published before it is raised. Fixes desync 1, and gives callers something to catch that
   means "this core stopped answering" rather than "some timeout happened".
3. **`TestConfig.brisc_command()`** wraps the three steady-state call sites and, on that
   exception, drops `BRISC_ELF_LOADED` and the loaded-ELF cache so the next test re-runs the
   soft-reset kick that can recover a merely slow core.
4. **A wedge latch.** If a second timeout arrives after a whole test — including that forced
   bring-up — has run in between, the core is declared unrecoverable and the plugin `skip`s the
   remaining tests with a reason naming the core and telling the reader that the first failure is
   the real one. The count is per test, not per command, and is reset by
   `pytest_runtest_makereport` whenever a test passes.

Point 4 is a deliberate refusal to keep trying. A RISC soft reset does not clear Tensix-level
state — a hung kernel can leave the math pipeline in a condition BRISC's own `START_TRISCS`
sequence then blocks on — which is measured, not assumed: after a full BRISC bring-up,
`RESET_TRISCS` is still acknowledged and `START_TRISCS` still times out. Only a board reset
recovers it. So the honest thing the harness can do is stop and say so.

## Measured

Reproducer: force the clamped approx-exp path on (it hangs at `ITERATIONS = 32`, tt-llk#1486),
then queue three known-good perf tests behind it.

| | first test | rest of the run |
|---|---|---|
| before | 1 real failure | **3 failures**, all `TENSIX TIMED OUT`, each naming an innocent kernel |
| after | 1 real failure | 2 failures that *are* the diagnosis, then **skipped** with the reason |

Scaled up, that is the difference between 57-73 fabricated failures and a handful. The two
remaining failures are the latch's evidence — the first shows the core stopped answering, the
second shows a bring-up did not fix it — and reporting them is correct.

**Non-disruptive on a healthy board**, which is the property that actually matters:

```
before the harness change:  1885 passed, 473 skipped, 4 xfailed   (58.9s)
after  the harness change:  1885 passed, 473 skipped, 4 xfailed   (58.9s)
perf matrix, 32 node ids in a single consumer invocation:  32 passed
```

## What this does not fix

* **The wedge itself.** Forcing the clamped exp path at `ITERATIONS = 32` still hangs, because
  that body is hand-unrolled for 8 datums and does not respect `ITERATIONS` (tt-llk#1486). Fixing
  that is what would let 01's `SKIP_NEGATIVE_SANITIZE` finally be measured in cycles on
  Blackhole rather than in issue slots.
* **In-process recovery.** Nothing here can un-wedge a core; `tt-smi -r` is still required. The
  harness now says so instead of producing results that look like data.
* **The 1-second command timeout** is unchanged. It was never the problem — a real command is
  acknowledged in microseconds — but it is worth knowing that the value is not tuned for a
  loaded host.
