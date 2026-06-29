# AGENT LESSONS — Spec-as-key perf work (read before measuring!)

Cumulative mistakes + gotchas so other agents don't repeat them. Append as you learn.

## 1. The editable-install worktree trap (cost real time this session)
`ttnn` is pip-installed **editable, pinned to the MAIN repo `/home/diego/tt-metal`** — via:
- a setuptools **MetaPathFinder** (`__editable___ttnn_..._finder.py` in the venv site-packages)
  with a hardcoded `MAPPING = {'ttnn': '/home/diego/tt-metal/ttnn/ttnn', ...}`, AND
- a `ttnn-custom.pth` that injects `/home/diego/tt-metal`, `/home/diego/tt-metal/ttnn`,
  `/home/diego/tt-metal/tools` onto `sys.path`.
**MetaPathFinders + .pth both run before PYTHONPATH**, so `PYTHONPATH=...wt-spec` does NOT redirect.
You will silently measure the MAIN repo build, not your worktree. ALWAYS
`print(ttnn.__file__)` and assert it's under the worktree.
**Fix (per-process, non-destructive):** strip the 3 ttnn-custom.pth entries from `sys.path`
(keep the `python_env` site-packages! it's also under /home/diego/tt-metal — don't strip it or you
lose loguru/torch), prepend the worktree's `ttnn`/`tools` paths, and patch the finder's `MAPPING`.
Baked into `/home/diego/i2s_perf.py` and `/home/diego/tp_perf_wt.py`.
Worktree `_ttnn.so` has an ABSOLUTE RPATH to its own `build_Release/lib`, so once the right
`_ttnn.so` loads, it pulls the right `_ttnncpp.so` (verified with `ldd`).

## 2. Don't measure while the build is still running
`./build_metal.sh --build-tests` relinks the libs EARLY (the python measurement is then valid) but
keeps compiling test binaries for many more minutes. Measuring during that = CPU contention =
inflated/noisy timings. First i2s number (38.3 µs) was taken under contention — flagged preliminary,
must re-measure clean. Check `pgrep -f 'ninja|build_metal.sh'` is empty before trusting a number.

## 3. Stale-binary discipline (from prior sessions, still applies)
- `build_metal.sh` on a tests-configured tree can exit while ninja stopped on a BROKEN test BEFORE
  relinking libs → STALE `.so`. Don't trust the exit code. Verify `_ttnncpp.so` mtime is AFTER your
  edits AND the build log got past the relink/install phase.
- Always full `./build_metal.sh`, never a partial `cmake --build --target X` (incoherent binary).

## 4. cwd resets between Bash tool calls
The shell cwd does NOT persist across Bash calls — it resets to the primary dir. Use absolute paths
or `cd <abs> && ...` in the SAME call. (A background `nohup ... &` did inherit the right cwd here,
but don't rely on it — verify with `readlink /proc/<pid>/cwd`.)

## 5. pre-commit clang-format aborts commits
Committing a non-clang-formatted .cpp: the pre-commit hook reformats it and ABORTS the commit (HEAD
unchanged, easy to miss). Run `clang-format -i <files>` first, then `git add` + commit.

## 6. Ratios aren't comparable across shapes
The doc's i2s "~48 µs / 2.0×" used a different shape than this session's 1024×1024/32-core run
(legacy 16.4 µs here vs ~24 µs there). Compare absolute µs at a FIXED shape; only compare ratios
within the same shape/condition. Always record shape + grid + INSPECTOR with every number.

## 8. PROFILE before attributing a cost to your fix (the memo was mis-targeted)
The #48252 memo was built to cut i2s's "~3 µs CoreRangeSet::merge." But a py-spy --native profile
showed `merge` (7.4% of samples) comes from the `CoreRangeSet(Span<CoreCoord>)` CONSTRUCTOR at i2s
factory line 87 (`all_cores`), inside create_program_spec — NOT from `from_shard_spec` (1.6%), which
is the only thing the memo touches. Lesson: the doc said "CoreRangeSet::merge ~3 µs" but didn't say
WHERE; an agent (me) assumed from_shard_spec and built a memo there. Always profile to find the
actual call site before optimizing, and re-profile to confirm the fix moved the needle.
py-spy --native at 250 Hz falls behind (slow unwinding) but still collects usable samples; aggregate
the speedscope JSON by leaf (self) + inclusive substring presence.

## 9. Dominant spec-as-key cost is create_program_spec (23.7%), not run-args
The per-dispatch spec REBUILD dominates the quasar-only overhead. Run-args (builder) and the memo are
single-digit-% levers. Don't over-invest in micro-opts; the big wins are reducing/caching the spec
build (CoreRangeSet ctor, KernelSpec, filesystem::path) or memoizing the whole spec (architecture).

## 10. WIN: CoreRangeSet solid-rectangle fast path (−14% i2s, behavior-preserving)
`CoreRangeSet(Span<const CoreCoord>)` ran `merge_ranges` (2D grid alloc + std::set churn) every
dispatch. Added an O(N) fast path: bbox + flat-bitset dup-check; if the coords fill a solid rectangle
(common: a full worker grid) emit one CoreRange. i2s 38.4→33.1 µs (−5.3 µs/−14%), PCC 1.0, transpose
unchanged. Lesson: the highest-leverage non-spec win was a SHARED metal primitive, not op-specific
code — and "optimize everything except the spec" (Diego's constraint) pointed straight at it. Always
check whether a hot primitive has a cheap common-case fast path before assuming the cost is intrinsic.

## 11. CoreRangeSet(Span<CoreCoord>) THROWS on duplicate coords — fast path must preserve it
Found while unit-testing the fast path: the original ctor builds one CoreRange per coord then
validate_no_overlap REJECTS duplicates (overlapping identical ranges throw). So duplicate coords were
never accepted. A naive solid-rect fast path (count == bbox area → one range) would SILENTLY ACCEPT a
duplicate-with-hole input (e.g. {(0,0),(0,0),(1,0),(1,1)}: area 4 == count 4) and not throw — a
behavior change. The flat-bitset dup-check catches the duplicate and forces the merge_ranges
fall-back, which throws exactly as before. Lesson: when adding a common-case fast path, enumerate the
inputs the SLOW path rejects/handles specially (throws, dedups, reorders) and prove the fast path
matches — a unit test asserting EXPECT_ANY_THROW on the dup case is what caught my wrong assumption
(I first expected dedup; the contract is throw).

## 7. The memo is a footgun (design caveat, not a bug yet)
`BufferDistributionSpec::from_shard_spec` memo is a process-global static unbounded cache. Fine for
benchmarking; the shippable form keys the layout sidecar on the SPEC HASH (the cache key), per
SPEC_AS_KEY_RUNARGS_PERF.md. Don't present the global-static version as production-ready.
