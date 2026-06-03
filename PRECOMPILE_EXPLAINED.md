# Up-Front Parallel Precompile — A Noob's Guide

A study guide for the JIT-precompile system that makes cold golden-test runs
~2× faster. Read top to bottom once; after that, use the **Reading List** as a
map.

All paths are relative to the repo root
(`.../groupnorm_sc_N_1_HW_C_full_run1/tt-metal/`). The eval system lives in the
`tt_ops_code_gen` submodule, symlinked as `.claude/`, so
`.claude/eval/precompile.py` and `tt_metal/third_party/tt_ops_code_gen/eval/precompile.py`
are the same file.

---

## 1. The problem, in one paragraph

When a golden test runs for the first time (a "cold cache"), the Tenstorrent
kernels it needs don't exist as compiled binaries yet. The framework compiles
them **just in time** (JIT) the first moment a test dispatches the op. Profiling
showed that on a cold cache **~99% of the entire test suite's wall-clock time is
this kernel compilation** — not the math, not PCC checks, not data movement
(see `.claude/eval/profiling/FINDINGS.md`). Worse, because pytest runs tests
**one at a time**, only one program (~5 kernels) compiles at any given moment,
so only ~3.8 of the machine's 8 CPU cores are ever busy. The compiler is
starved.

**The fix:** before the tests run, look at *all* the cases that are about to
run, figure out the **distinct set of programs** they need, and compile that
whole set **up front and in parallel** — filling every core. This populates the
on-disk compile cache. Then the actual tests run "warm": every kernel they need
is already a cached binary, so the whole suite executes in ~3.5 minutes instead
of ~88.

Result from the overnight A/B run: **88 min (cold, inline) → 46 min (precompile
+ warm)**, identical test outcomes.

---

## 2. The 30-second mental model

```
        WITHOUT precompile (cold)                WITH precompile (cold)
        ─────────────────────────                ──────────────────────
  test1 ─ compile(5 kernels) ─ run            ┌─ look at ALL tests
  test2 ─ compile(5 kernels) ─ run            │  find DISTINCT programs (2948)
  test3 ─ compile(5 kernels) ─ run            │  compile them ALL, in parallel  ← fills cores
   ...     (one core busy, 87 min)            └─ THEN:
                                                 test1 ─ run (warm, ~8ms)
                                                 test2 ─ run (warm, ~8ms)
                                                  ...    (warm, ~3.5 min total)
```

The trick that makes "look at all tests and find the programs they need"
possible *without actually running them* is the **generic-op intercept**
(Section 5). The trick that makes the parallel compile actually overlap is a
**GIL-released C++ entry point** (Section 6).

---

## 3. How to invoke it

It is **opt-in** via an environment variable and a **no-op when unset**, so it
never affects normal runs.

```bash
source python_env/bin/activate

# Normal golden run (no precompile — today's default behavior):
scripts/run_safe_pytest.sh --run-all .claude/eval/golden_tests/groupnorm_sc_N_1_HW_C/test_golden.py

# Same run, but warm the cache up front in parallel first:
EVAL_PRECOMPILE=1 \
  scripts/run_safe_pytest.sh --run-all .claude/eval/golden_tests/groupnorm_sc_N_1_HW_C/test_golden.py
```

Knobs (all optional, read in `precompile_plugin.py`):

| Env var | Default | Meaning |
|---|---|---|
| `EVAL_PRECOMPILE` | unset (off) | Set to `1` to turn the whole thing on. |
| `EVAL_PRECOMPILE_WORKERS` | `min(cpus, 4)` | Size of the Python thread pool driving the compiles. **Overnight proved 4 ≈ 8** — the build executor is the real limiter, so going higher buys nothing and risks memory blowup. |
| `EVAL_PRECOMPILE_DEVICE_ID` | `0` | Which device to open for the warm-up. |

Because it's wired into pytest's collection hook (Section 7), it works
identically under `run_safe_pytest.sh`, `eval_test_runner.sh`, or a bare
`pytest` — you just need `EVAL_PRECOMPILE=1` in the environment.

---

## 4. Reading list (in this order)

Read these in sequence. Each builds on the last. The ⭐ lines are the ones to
really stare at; they're reproduced and annotated in Sections 5–7.

1. **`.claude/eval/profiling/FINDINGS.md`** — *Why this exists.* The profiling
   evidence: "~99% of cold time is JIT compile," "warm cache makes the body 91×
   faster," "cold compile uses only ~3.8/8 cores." Read the **TL;DR** (lines
   15–30). This is the entire justification for the system.

2. **`.claude/eval/golden_tests/conftest.py`** (35 lines — read all) — *The
   entry point.* How the precompile gets hooked into pytest. Two jobs: put the
   `eval` package on `sys.path`, and register the precompile as a pytest hook.
   ⭐ Lines **30–35** (`pytest_collection_finish`).

3. **`.claude/eval/precompile_plugin.py`** (121 lines — read all) — *The pytest
   glue.* Decides which collected tests are eligible, turns each into a zero-arg
   "dispatcher" callable, opens a device, calls the core engine, prints the
   summary. ⭐ Lines **37–47** (`_eligible`), **50–54** (`_make_dispatch`),
   **80–81** (the worker cap), **92–97** (open device → run → the result).

4. **`.claude/eval/precompile.py`** (192 lines — the heart) — *The op-agnostic
   engine.* This is where the magic lives. Capture descriptors via the
   intercept, dedup by structural hash, parallel-compile. ⭐ Lines **55–69**
   (`intercept_generic_op`), **72–82** (`capture_descriptor`), **141–168** (the
   capture+dedup loop), **178–190** (the parallel compile).

5. **`ttnn/cpp/ttnn/operations/generic/generic_op_nanobind.cpp`** — *The C++
   foundation.* The two Python-callable functions the engine relies on. ⭐ Lines
   **65–75** (`compute_program_descriptor_hash` — the dedup key) and **77–104**
   (`precompile_program_descriptor` — compile-without-enqueue, GIL released).

6. **`.claude/eval/profiling/precompile/precompile_harness.py`** — *Optional /
   experimental.* A groupnorm-**specific** measurement version with cold A/B
   tests, used to gather the overnight numbers. Same idea as `precompile.py` but
   it knows about groupnorm's tensors so it can build them directly. Read this
   only if you want to see the experiment scaffolding; the **production** path is
   `precompile.py` + `precompile_plugin.py`.

7. **`precompile_overnight/SUMMARY.txt`** + the `*.log` files — *The results.*
   The A/B/C benchmark output. (Note: its "FAILED / likely OOM" verdict is a
   scripting artifact — `exit=1` is just the suite's 140 pre-existing precision
   failures; all three runs succeeded with identical results.)

---

## 5. The clever part: how the generic-op "intercept" works

### The goal

To compile a program up front, you need its **`ProgramDescriptor`** — a value
object that fully describes a program: which kernel source files, compile-time
args, core ranges, circular-buffer layout, semaphores. The op builds this
descriptor internally and hands it to `ttnn.generic_op`, which then compiles +
enqueues + runs it.

We want **only the descriptor**. We do *not* want to compile, enqueue, or run
anything during the "figure out what we need" phase. So we need to let the op
run far enough to *build* its descriptor, then snatch it the instant it tries to
dispatch — and stop it cold.

### The mechanism: monk-patch + exception

`ttnn.generic_op` is just a Python attribute. The engine temporarily **replaces
it** with a fake that grabs the descriptor and **raises an exception** to unwind
the op immediately.

From `.claude/eval/precompile.py:46-69`:

```python
class _Captured(Exception):
    """Raised by the generic_op interceptor to carry the descriptor out and
    short-circuit the op before it compiles / enqueues / verifies."""
    def __init__(self, descriptor):
        self.descriptor = descriptor          # ← smuggle the descriptor out via the exception object
        super().__init__()

@contextmanager
def intercept_generic_op():
    orig = ttnn.generic_op                     # ← remember the real function

    def _capture(tensors, program_descriptor):
        raise _Captured(program_descriptor)    # ← THE intercept: grab descriptor, abort immediately

    ttnn.generic_op = _capture                 # ← monkey-patch: swap the real one for our fake
    try:
        yield                                  # ← caller runs the op here; it will hit _capture and raise
    finally:
        ttnn.generic_op = orig                 # ← ALWAYS restore the real one, even if something throws
```

**Why an exception instead of just returning a fake tensor?** Because the op has
code *after* the `generic_op` call (output reshaping, verification, etc.) that
would run on garbage and waste time or crash. Raising `_Captured` unwinds the
entire op call stack instantly — we get the descriptor and nothing else runs.

The caller catches exactly that one exception (`.claude/eval/precompile.py:72-82`):

```python
def capture_descriptor(dispatch: Callable[[], object]):
    with intercept_generic_op():
        try:
            dispatch()                         # run one test case's op-dispatch path
        except _Captured as c:
            return c.descriptor                # got it — return the descriptor
    return None                                # dispatch() finished WITHOUT calling generic_op
                                               # → this op isn't generic_op-based → signal "skip me"
```

That `return None` is how the system stays **op-agnostic**: if an op dispatches
some other way (a C++ program factory instead of `generic_op`), the intercept
never fires, `capture_descriptor` returns `None`, and the precompile cleanly
skips that op instead of running every test case for real.

### Two subtleties worth internalizing

- **`@contextmanager` + `try/finally`**: the patch is *guaranteed* to be undone.
  If you monkey-patch a global and forget to restore it, you've corrupted the
  whole process. The `finally` block is non-negotiable.
- **"Single-threaded use only"** (docstring, line 59): because it mutates the
  *process-global* `ttnn.generic_op`, you can only capture one descriptor at a
  time. That's why **capture is serial** and only the **compile** (Section 6) is
  parallel — see the loop in Section 7.

> Side note: the experimental harness (`precompile_harness.py:118-135`) uses a
> slightly different intercept — it stores the descriptor in a dict and *returns*
> a fake tensor instead of raising, letting the op finish. Same idea, gentler
> unwind. The production `precompile.py` uses the raise-to-abort version because
> it's faster and doesn't depend on op internals after the dispatch.

---

## 6. The parallel compile + the C++ foundation

Once we have the list of descriptors, two C++ functions (Python-callable via
nanobind) do the heavy lifting.

### Dedup key: `compute_program_descriptor_hash`

From `ttnn/cpp/ttnn/operations/generic/generic_op_nanobind.cpp:65-75`:

```
Hashes kernel sources, compile-time args, core ranges, CB structure, and
semaphores. Excludes runtime arg values and buffer addresses, making it
suitable as a cache key for structural equivalence.
```

This is *why dedup works*. Two test cases that differ only in **where their
tensors happen to be allocated** (different buffer addresses) produce the **same
hash** — because the hash ignores addresses. So 2948 cases collapse to 2948
*distinct programs* to compile, and anything structurally identical is compiled
only once. It's the same key the device's program cache uses, so what we warm is
exactly what the tests will look up.

### Compile-without-enqueue: `precompile_program_descriptor`

From `ttnn/cpp/ttnn/operations/generic/generic_op_nanobind.cpp:77-104` — the
key body and binding line:

```cpp
mod.def(
    "precompile_program_descriptor",
    [](MeshDevice* device, const ProgramDescriptor& program_descriptor) {
        // Build the program and JIT-compile its kernels WITHOUT enqueueing.
        // CompileProgram populates the on-disk JIT cache (hash-keyed per kernel).
        // No command queue / program-cache state is touched, so this is safe
        // to call concurrently for distinct descriptors.
        Program program{program_descriptor};
        detail::CompileProgram(devices.front(), program);
    },
    nb::arg("device"),
    nb::arg("program_descriptor"),
    nb::call_guard<nb::gil_scoped_release>(),   // ⭐ THE line that makes parallelism real
    ...
```

Two things to understand here:

1. **It compiles, then stops.** No enqueue, no run, no program-cache insert. Its
   only side effect is writing compiled kernel binaries into the **on-disk JIT
   cache** (`TT_METAL_CACHE`). A later real `ttnn.generic_op` with a
   structurally-equal descriptor reads those binaries back = warm hit.

2. **`nb::call_guard<nb::gil_scoped_release>()`** releases Python's Global
   Interpreter Lock for the duration of the C++ compile. The GIL normally lets
   only one Python thread run at a time. By releasing it, *N* Python threads can
   each be inside a C++ compile **simultaneously** — that's what lets the thread
   pool actually fill the cores instead of taking turns.

### Driving it from a thread pool

From `.claude/eval/precompile.py:178-190`:

```python
def _compile(d):
    try:
        ttnn.precompile_program_descriptor(device, d)   # GIL released inside → real overlap
        return None
    except Exception as e:                              # collect errors, don't crash the pool
        return repr(e)

with ThreadPoolExecutor(max_workers=max_workers) as ex:
    for err in ex.map(_compile, descriptors):           # fan all distinct programs across N threads
        if err is not None:
            res.compile_errors.append(err)
```

> **Why threads, not processes?** Compiling is C++ work behind a GIL-released
> call, so threads genuinely run in parallel here *and* share the one open device
> handle + in-memory build dedup. Processes would each need their own device and
> couldn't share the build executor. (And the overnight data showed the build
> executor saturates around 4 workers anyway — `precompile_plugin.py:76-81`
> explains the cap.)

---

## 7. Pytest plugin concepts you need

You only need four ideas to understand the wiring.

### a) `conftest.py` is auto-discovered

Pytest automatically imports any file named `conftest.py` in or above the test
directory — you never reference it explicitly. It's the standard place to put
fixtures and **hooks** shared by a directory of tests. Here:
`.claude/eval/golden_tests/conftest.py` covers every golden suite under it.

### b) Hooks: named functions pytest calls at lifecycle moments

If you define a function with a specific magic name in a `conftest.py` (or
plugin), pytest calls it at the matching point in its lifecycle. The one used
here, from `conftest.py:30-35`:

```python
def pytest_collection_finish(session):       # pytest calls this automatically
    from eval.precompile_plugin import run_precompile
    run_precompile(session)                   # ...right after it finishes COLLECTING tests
```

`pytest_collection_finish` fires **after pytest has discovered every test it's
about to run, but before it runs any of them.** That is the perfect moment: we
have the full list of cases (`session.items`), and nothing has executed yet — so
we can warm the cache first. (The import is *inside* the function so the
conftest stays cheap to import when precompile is off.)

### c) Collection vs. execution — the two phases

This is the single most important pytest concept for this system:

```
COLLECTION phase                          EXECUTION phase
────────────────                          ───────────────
pytest discovers every test &      ──►    pytest runs each test one by one
its parametrizations (session.items)      (the actual test bodies)
        │                                          ▲
        └─ pytest_collection_finish fires here ────┘
             ↑ WE INSERT THE PRECOMPILE HERE
```

Each `session.items` entry is one **parametrized invocation** (e.g.
`test_op[1x1x4096x640-affine=gamma_beta-...]`), not just one function. The
groupnorm suite has one `test_op` function but thousands of items.

### d) Reading params off a collected item

Before execution, each item carries its parametrize values in `item.callspec.params`.
The plugin uses this to (1) filter and (2) rebuild a callable that dispatches
that exact case.

From `precompile_plugin.py:37-54`:

```python
def _eligible(item) -> bool:
    if item.get_closest_marker("skip") or item.get_closest_marker("xfail"):
        return False                          # don't precompile cases the suite would skip
    callspec = getattr(item, "callspec", None)
    if callspec is None:
        return False
    params = callspec.params
    return "inputs" in params and "axes" in params   # a standard golden parametrization

def _make_dispatch(func, params, device):
    kwargs = dict(params)
    return lambda: func(device=device, **kwargs)     # zero-arg callable: "run this one case"
```

Those `lambda`s are exactly the `dispatchers` that `collect_and_compile` feeds
through the intercept. Filtering out `skip`/`xfail` items matters: we only warm
the cache for cases that will actually dispatch.

### e) Best-effort: it can never break the run it's warming

Every layer swallows its own errors. The plugin wraps the whole pass in
`try/except` (`precompile_plugin.py:92-100`), and the engine counts per-case and
per-program errors instead of raising (`precompile.py:142-145`, `182-183`). If
precompile fails for any reason, you get a printed note and the suite just runs
cold — exactly as if `EVAL_PRECOMPILE` were unset.

---

## 8. End-to-end data flow (putting it together)

```
EVAL_PRECOMPILE=1  scripts/run_safe_pytest.sh ... test_golden.py
        │
        ▼
pytest COLLECTS all test_op[...] invocations  →  session.items
        │
        ▼  pytest_collection_finish  (conftest.py:30)
run_precompile(session)                          (precompile_plugin.py:57)
        │
        ├─ keep eligible items (not skip/xfail)  (_eligible)
        ├─ wrap each as a zero-arg dispatcher     (_make_dispatch → lambda)
        ├─ ttnn.CreateDevice(0)                   (our own warm-up device)
        │
        ▼  collect_and_compile(device, dispatchers)   (precompile.py:112)
        │
        │   ── CAPTURE (serial) ───────────────────────────────
        │   for each dispatcher:
        │       intercept ttnn.generic_op           (intercept_generic_op)
        │       run dispatcher → op builds descriptor → _Captured raised
        │       hash = compute_program_descriptor_hash(desc)   [C++]
        │       if hash unseen: keep descriptor      → 2948 unique
        │
        │   ── COMPILE (parallel) ─────────────────────────────
        │   ThreadPoolExecutor(max_workers):
        │       ttnn.precompile_program_descriptor(device, d)  [C++, GIL released]
        │           → CompileProgram → writes binaries to TT_METAL_CACHE
        │
        ├─ ttnn.close_device(device)
        │
        ▼  print "EVAL_PRECOMPILE: 2948 unique programs compiled in 2557.3s"
        │
        ▼
pytest EXECUTES test bodies one by one
        each test → real ttnn.generic_op → looks up TT_METAL_CACHE → WARM HIT (~8 ms)
        whole suite of 2948 cases runs in ~3.5 min instead of ~88 min
```

---

## 9. Mini-glossary

- **JIT compile** — "just in time": kernels are compiled the first moment a test
  needs them, not ahead of time. The thing we're trying to move earlier and
  parallelize.
- **Cold cache / warm cache** — cold = no compiled binaries on disk yet (must
  compile); warm = binaries already in `TT_METAL_CACHE` (just read them back).
- **`ProgramDescriptor`** — a self-contained value describing a program (kernel
  files, args, core ranges, CB layout). What `precompile_program_descriptor`
  compiles. Holds no live tensors, so it's safe to keep and compile later.
- **`ttnn.generic_op`** — the generic dispatch entry point an op calls to
  compile + enqueue + run a `ProgramDescriptor`. The thing we intercept.
- **Structural hash** — `compute_program_descriptor_hash`; ignores buffer
  addresses, so structurally-identical cases dedup to one compile.
- **GIL** — Python's Global Interpreter Lock; only one Python thread runs at
  once. Releasing it (`gil_scoped_release`) during C++ compile is what makes the
  thread pool actually parallel.
- **Monkey-patching** — replacing a function/attribute at runtime
  (`ttnn.generic_op = _capture`). The intercept technique.
- **pytest hook** — a magically-named function (`pytest_collection_finish`)
  pytest calls at a lifecycle point. How we inject precompile.
- **Collection vs execution** — pytest first discovers all tests, then runs
  them. We slip in between.

---

## TL;DR for the impatient

1. Turn it on with `EVAL_PRECOMPILE=1`.
2. A pytest hook (`pytest_collection_finish`) fires after pytest knows every test
   but before it runs any.
3. For each test we **fake `ttnn.generic_op`** so the op builds its
   `ProgramDescriptor` and we **steal it via an exception** — without compiling
   or running.
4. We **dedup** descriptors by a structural hash and **compile the distinct set
   in parallel** through a **GIL-released** C++ call that writes the on-disk
   cache.
5. The real tests then run **warm** — 88 min → ~46 min, identical results.
