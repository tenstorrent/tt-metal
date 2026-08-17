# PR #46283 — findings and suggested fixes

We drove this PR's pipeline (`bring-up → emit-e2e → optimize`) end to end, twice, on a Blackhole
p150b, against a model it had never seen: **Voxtral-TTS**, 4.0 B parameters, three stacks (autoregressive
backbone + flow matching + audio codec). The tool was given only weights and a `config.json` in HF
format; the checkout it ran from contained no code for this model.

**What worked.** Bring-up is the strong half and we want to say so first. With no model-specific
code it rewrote **7 of 7 components** for the device, put **114 of 114 operations** on the chip with
zero CPU fallback, in **one round, in 42 minutes**. The pipeline it then wired up is genuinely
correct: **e2e PCC 0.9999834**, emitted audio codes exactly equal to the reference, zero code flips
— verified by running the tests directly rather than reading status files. The optimize stage ran
unattended for 17 hours, banked 11 measured wins, correctly rejected 12 regressions, and reached the
hand-written-kernel rung on its own. Several designs deserve credit explicitly: `beat_baseline` is
pinned false so the ledger, not the agent, decides a win; `gates_allow_banking()` treats an unrun
gate as a failed gate; and the emitted e2e test asserts exact audio-code equality on top of waveform
PCC, which is the right correctness criterion for a codec model and stricter than we expected.

**What this document is.** Every issue we hit that looks actionable for you, with a suggested fix
where we have one. Issues that were our own setup problems are excluded. Where we are unsure of the
fix, we say so rather than guess. Full evidence, reproduction commands and measurements are in
`TOOL_FINDINGS.md` (entry IDs below match it).

**The one thing to fix first is #1.** Most of the performance findings are downstream of it.

---

## 1. Correctness gates that can pass when they should not

- **[F27 — ROOT CAUSE] The captured KV cache is discarded instead of copied, and this single line
  produces the port's entire performance architecture.**
  `_captured/attention/args.pt` holds a real mid-generation call with a **208-deep KV cache**. The
  harness cannot hand the same cache object to the reference and then the stub, because
  `VoxtralAttention.forward` mutates it (`cache[cache_key] = (k, v)`) — so the stub would attend over
  a cache one position longer than the golden did. The harness drops the cache, and records the
  consequence itself: *"Dropping the cache instead makes the test vacuous: at S=1 with no cache the
  softmax is over a single key, so it returns 1.0 whatever q and k are."* Downstream: the generated
  stub never sees a cache, so a cacheless implementation scores 0.9999 and graduates → the demo,
  obliged to route through that stub, can only recompute the whole prefix every frame → the plan then
  codifies that (`gate_2_invoked: "tts_backbone == 1 + n_frames"`) → the trace gate becomes
  unreachable through the stubs → a second, unreachable decode implementation is written and receives
  all 17 hours of optimisation.
  **Fix:** `deepcopy` the captured args before handing them to each side. Then add `cache` to the
  generated test's recognised inputs (see next item) so the component is actually specified against
  it, let the stub carry the parameter its signature already declares, and re-derive the plan's
  invocation counts from the cached structure (`tts_backbone == 1`, not `1 + n_frames`).

- **[F42] The correctness gate returned `pcc: 33.612, pcc_verified: true` against a threshold of
  0.99 — four times in twelve hours.**
  Observed values: `33.612`, `47.779`, `1.525`, `11.021`, all reported `status: ok`. A Pearson
  correlation cannot exceed 1.0. Three mechanisms compose: `_PCC_RE = (?i)pcc[^\n]*?[:=]\s*(-?\d+\.\d+)`
  allows arbitrary text between "pcc" and the delimiter, so unrelated numbers on a line mentioning
  pcc match; `parse_pcc` takes `min()`, which only guards against values that are too *low*; and the
  pytest exit code is deliberately not consulted (defensibly — benign non-zero exits are real), which
  leaves the parsed number as the only signal. This is not merely a reporting bug:
  `gates_allow_banking()` reads `pcc.status` as its sole correctness input, so an unverified edit can
  be committed as a win. In our run the *perf* gate happened to reject all four; the correctness gate
  contributed nothing.
  **Fix:** reject out-of-range values as parse failures, **with a tolerance** — a real correlation can
  exceed unity by rounding (this port legitimately prints `pcc=1.000017`), so test
  `not (-1.0 - 1e-3 <= pcc <= 1.0 + 1e-3)` → `status: crash`, `pcc_verified: False`. Tighten the regex
  to require the delimiter close to `pcc`. When the parse is implausible, fall back to the exit code.

- **[F36] "per-component PCC tests will use real inputs" is false — the graduation gate runs on
  `torch.randn`.**
  Preflight captures real IO for 7/7 components (43 MB each for `attention` and `decoder_layer`,
  including the reference `output.pt`) and prints that the tests will use them. The generated tests
  contain **no `torch.load` at all** (0 hits across 7 files); every input is synthesised from the
  forward signature by argument name, with a `torch.randn(1, 64, 64)` fallback. The captures are read
  only to resolve which submodule to test.
  **Fix:** load `args.pt`/`kwargs.pt` when a capture exists and fall back to synthesis only when it
  does not; compare against the captured `output.pt` so the gate becomes a golden test; and make the
  log line state what the gate actually ran on.

- **[F37] The generated PCC test calls a helper it never defines, and two of its input defaults
  corrupt the golden rather than the port.**
  `_captured_submodule_path()` is called by 7 of 7 emitted tests and defined by 0 — a guaranteed
  `NameError` on the first component of every model. Separately, `bias` (the additive causal mask)
  defaults to `None` and is not in `_WELL_KNOWN_INPUTS`, so the template's own rule drops it —
  **silently making the golden non-causal**; and `_ttnn_from_torch_mesh_safe` stages every primary
  input as `bfloat16` unconditionally (`bringup_loop.py:619`, called at `:791` with no override), which
  is exact only to 256, so an index tensor is corrupted before the port sees it (codebook id
  8191 → 8192) and activations pick up ~4e-3 of input error the port is then blamed for.
  **Fix:** emit `_captured_submodule_path` into the template (or import it). Never drop a defaulted
  argument that changes semantics — fail loudly instead of producing a non-causal reference. Choose the
  staging dtype from the tensor: integer → `uint32`/`ROW_MAJOR`, float → `float32`/`TILE`.

- **[F29] The CLI's `--pcc-target` default of 0.95 overrides the engine's documented 0.99, and the
  threshold sets quality rather than merely gating it.**
  `e2e_mcp.py` documents 0.99 as *required*; `cli.py`'s `--pcc-target` defaults to 0.95 and wins.
  Measured on the same model, same code, same machine: at **0.95** the fix-loop reported
  `rounds=1 can_stop=True` and stopped at **e2e PCC 0.9586**; at **0.99** it did 45+ tool calls of real
  repair and reached **0.9986**. The 0.9586 was not a precision ceiling — the test's own bound reads
  1.0000 at that N. Worse, bring-up's closing "NEXT STEP" line prints the `emit-e2e` command *without*
  the flag, so copying what the tool just told you to run gives you the loose gate.
  **Fix:** have the CLI inherit the engine's default instead of overriding it, and print the
  threshold's provenance in the report (`pcc>=0.95 (CLI default; engine default is 0.99)`).

- **[F13] Generated stubs swallow fast-path exceptions, so a perf regression passes the PCC gate.**
  A stub that fails its fast path falls back silently, and correctness still passes — the gate cannot
  distinguish "fast path worked" from "fast path threw and we used the slow one".
  **Fix:** let the exception surface, or record the fallback and fail the gate on it.

- **[F26] The report states what was collected, not what the gate measured.**
  `captured 7/7` is printed where the meaningful claim is whether the tests *used* those captures.
  **Fix:** report the measured quantity, and where a gate ran on a proxy, say which proxy.

- **[F28] The entire end-to-end verdict rests on one prompt.**
  `E2E_ALL_TASKS=0` and a single default prompt decide PASS/FAIL for the whole port, and no flag
  widens it. A single sample cannot bound behaviour for a generative audio model.
  **Fix:** take the sample count from the batch dimension the test already has, or expose a flag; even
  n=4 distinguishes a systematic error from a lucky prompt.

- **[F39] The e2e report prints `Verdict: PASS` beside `e2e PCC n/a`.**
  The verdict is a literal passed in when `can_stop` is true; the number is read from
  `demo_dir/grader_report.json`, which the cc-engine path never writes. The value is not unavailable —
  `perf-mcp` stores the same figure to the digit in its gate-verdicts file.
  **Fix:** return the PCC the gate already measured and render that; never print a verdict beside
  `n/a` without saying why the number is missing.

---

## 2. The performance half measures something the product never runs

- **[F46] The profiled decode and the shipped decode are different algorithms.**
  `pipeline.py` contains an incremental, KV-resident `decode_step` (O(1) work per frame) and a
  `run_tts` loop that re-runs all 26 layers over the whole padded 224-token prefix every frame and
  keeps one row (O(n) per frame). `decode_step`/`decode_prefill` have exactly **two call sites in the
  whole demo directory**, both inside the trace harness — `run_tts` calls neither, and `demo_tts.py`
  calls `run_tts`. So every perf measurement, all 26 optimizer attempts and all 11 commits were
  ranked against code the product never executes. Measured consequence: the optimizer reported
  **−17.2%** while the shipped path improved **−13.2%** (297.69 → 258.26 ms/frame). The gains do reach
  the product, because most edits landed in shared helpers — but by side effect, not by measurement.
  **Fix:** assert that the perf test's hot function is reachable from the demo's entry point, before
  the first measurement. A static reachability check would have caught this. Note that simply pointing
  `run_tts` at `decode_step` breaks Gate 1 (the stubs would no longer be the routed bodies) and Gate 2
  (`tts_backbone` drops to 1) — the durable fix is F27's.

- **[F45] `E2E_REQUIRE_TRACE` is satisfied by proving a stage *can* be captured; the shipped pipeline
  never replays one.**
  `execute_trace` appears exactly once in `tt/pipeline.py`, inside `trace_capture_selftest`. `run_tts`
  — documented as *"THE pipeline. Real prompt → real 24 kHz waveform"* — contains no trace call, so
  every op is host-dispatched, every frame. Measured: **258.26 ms/frame** eager.
  **Fix:** gate on use, not capability — require `execute_trace` to be reachable from the generation
  entry point, or count trace replays during a real run and fail at zero.

- **[F44] The optimize objective times a capture-and-verify harness, not inference, at 1/6 the real
  workload.**
  `full_pipeline_ms` comes from `trace_capture_selftest`, which per stage runs an **eager execution**, a
  **trace capture**, one replay, **two device→host readbacks**, a **release** and a **host-side PCC**.
  Only the replay resembles serving. Shape: `ISL=32`, `OSL=4`, one decode step — against a product
  workload of a 200-token prompt and 24 frames. Measured proportions in a real utterance: decode
  **97.3%**, prefill 2.2%, codec 0.5% — so the cost that dominates entered the objective once, while
  per-stage capture overhead entered four times per iteration. There is a real reason the workload is
  small (`perf_test_gen.py:1184`: *"BOUNDED + profiler-safe so tracy's 12000-marker buffer never
  overflows"*) but the accept/reject timing runs **with Tracy off**, so no marker budget constrains it.
  **Fix:** separate the two jobs. Profile a **signposted window inside the real run** — the mechanism
  already exists (`probes.resolve_signposts`, `start_signpost`/`end_signpost`) and reported itself
  missing at Step 9/10 — so a steady-state window stays inside the marker budget while profiling real
  shapes. Run the accept/reject timing on the real 24-frame workload, and take capture, the eager pass,
  the readbacks and the host PCC out of the timed region.

- **[F43] `TRACE_PER_TOKEN_MS` is the per-call time, not per token.**
  `_ms = elapsed / iters` is milliseconds per call, where a call is `ISL=32` prompt rows plus `OSL=4`
  frames, and it is printed as both `FORWARD_WALL_MS` and `TRACE_PER_TOKEN_MS`. The ledger inherits it:
  `forward_wall_ms: 598.94` and `per_token_ms: 598.94`, identical — which is the signature. It scales
  with a knob, so raising `TT_PERF_OSL_TOKENS` to amortise prefill makes the reported per-token cost
  *rise*.
  **Fix:** divide by `OSL`; keep the undivided call time under its own name; assert
  `per_token_ms <= forward_wall_ms` whenever `OSL > 1`.

- **[F41] The depth-knob sanity check compares op sequences truncated at the same limit, and reads the
  shared limit as proof the knob is inert.**
  `_op_sig_probe.py:616` emits `json.dumps(_SEQ[:50000])`, and the check reported
  `op-count 50000->50000` for capped versus uncapped — the truncation constant on both sides. A cap
  from 26 layers to 2 producing byte-identical op counts is not a plausible measurement. The tool then
  declared the knob inert and profiled the full uncapped model, the opposite of a bounded window.
  Separately the knob is only half-wired: `test_main_perf.py:123` passes `layers=` but not
  `flow_layers`/`vocode_layers`, while its own comment claims it "caps every repeated stack".
  **Fix:** compare lengths before truncation, or hash the full sequence, and say when the signal
  saturated rather than reporting a conclusion. Pass all three depth arguments, or correct the comment.

- **[F40] A completed measurement is discarded because of a crash that happened afterwards.**
  The baseline ran cleanly — 8 iterations converging to `FORWARD_WALL_MS=756.4513`, all four stages
  traced PCC-clean — then segfaulted in `ttnn close_mesh_device` during **teardown**, called from the
  generated perf test's `_close_device`. The crash killed the process before Tracy wrote its CSV, so
  the run was reported `discovery failed … cpp_device_perf_report.csv not found` and the number thrown
  away; the stage then continued with no baseline. Recurred later on the `dtype` rung
  (`fullpipe_ms: None`, *"wedged/crashed when tried"*), losing 3 of 26 attempts to crashes rather than
  merit. Note `test_tts_e2e.py` opens and closes the same device cleanly seven times in the same
  session, so something the perf test leaves open makes teardown fatal.
  **Fix:** parse the numbers the run already printed — `FORWARD_WALL_MS` is on stdout in the tool's own
  format. Treat a post-measurement teardown failure as a warning, not a failed measurement. And either
  refuse to optimize with no baseline or say loudly that improvements will be unverifiable.

- **[F31] A profiled child that died of a signal is reported as a missing CSV.**
  `termination_check()` returned *"tracy run exit 1 … AssertionError: cpp_device_perf_report.csv not
  found"* when the actual event was `Fatal Python error: Bus error` / `Aborted (core dumped)` mid-forward.
  The assertion is the last link in a fallback chain; the abort is the cause. An agent handed the
  assertion goes looking for profiler plumbing.
  **Fix:** check the child's exit status first and report that (`tracy child aborted (SIGBUS) at
  <frame>`); include the log tail in the error, since the caller is already given the log path.

- **[F38] `optimize --devices` defaults to `"0,1"`, so a single-chip box is planned as 2-chip TP=2 —
  overriding an explicit `--mesh`.**
  On a P150 (`HARDWARE['P150'].chips == 1`, one device node) with `--box P150 --mesh 1,1` passed
  explicitly, the run printed `engine : … mesh 1,1` and on the next line
  `topology : 2-chip -> mesh 1x2 (TP=2 DP=1)`. Bring-up had written the correct topology minutes
  earlier into `parallelism_manifest.json` (`{"chips": 1, "tp": 1, "dp": 1}`), which the perf stage does
  not read back. Adding `--devices 0` fixes it. Relatedly the scorecard it then writes reads
  `mesh=unknown TP=unknown DP=unknown` beside a latency, which makes the record uncomparable.
  **Fix:** default `--devices` to what is present (`single` is already an accepted value); read
  `parallelism_manifest.json`; refuse to let a device-list default silently override an explicit
  `--mesh`; and stamp the resolved topology into `PERF_SCORECARD`.

- **[F47] "Host-free" is certified with the demo's per-frame host readback switched off.**
  `run_tts` defaults `early_stop=True` and `demo_tts.py` takes that default, so the demo performs a
  device→host readback of one value every frame (`_is_stop`). `host_op_selftest` calls
  `run_tts(..., early_stop=False)` and observes "a pure device chain" — i.e. with the loop's only host
  op removed. The reasoning (a stop condition is generation control, not arithmetic) is openly stated
  and defensible; the claim simply ships without the qualifier, and the cost becomes material once the
  pipeline is fast.
  **Fix:** state the configuration next to the claim, or compare the stop code on-device and read the
  flag every N frames.

- **[F12] The fusion rung reaches for a grid where it should reach for a program config.**
  On an op whose output width is only 32 tiles, no core grid can help — a 130-core grid cannot be
  filled by 32 tiles — but the ladder keeps applying the grid lever.
  **Fix:** when `N_tiles < cores`, treat the grid rung as exhausted and move to the program config or a
  structural change instead.

- **[F14] "Producer emits the consumer's shard" does not check the consumer's program-config grid.**
  A producer is asked to emit a layout the consumer will reshard anyway, because the check compares
  memory configs without consulting the consumer's program config grid.
  **Fix:** compare against the consumer's program config, not only its memory config.

---

## 3. Reproducibility — two runs from one commit are not the same run

- **[F34] The overlay store silently restores a model that was deleted from HEAD, so a from-scratch run
  is unreachable.**
  We deleted `models/demos/voxtral_tts_full/` and committed the deletion; `git status` was clean and
  the isolation worktree was correctly built from that commit. One line went past —
  `[isolation] applied 0 _shared + 1 model overlay(s)` — after which the worktree contained the entire
  previous port: 63 files including graduated stubs, their `.last_good_native` snapshots,
  `.bringup_cc_state.json` and the previous `RUN_REPORT.md`. Two runs from the same commit therefore
  start from different states, with nothing in the tree, the log or the report to say so. Compounding:
  `overlay-drop <model_id>`, documented as wiping the scope, left `locked_modules.json` behind pinning
  a structural decision from the previous run.
  **Fix:** say what was restored, not how many overlays applied
  (`restored models/demos/… (63 files, incl. 5 graduation markers)`); provide `--no-overlays` /
  `--from-scratch` and document it as the way to reproduce a clean port; make `overlay-drop` empty the
  scope; and never carry graduation markers in an overlay — ported source is legitimate to reuse, a
  "this component already passed its gate" marker earned under another threshold is not.

- **[F35] Backend selection is not reproducible.**
  Two `auto-up` runs on the same model and the same commit, four minutes apart, produced identical
  deterministic rankings (`hf_eager universal (TTS)` score 40, `XTTS-v2` score 30) and different
  winners — run 1 took rank 1, run 2 took rank 2. The LLM ranker overrode its own top score with no
  stated reason. Both candidates' `demo_path` values do not exist in the checkout.
  **Fix:** make the deterministic score authoritative unless the ranker gives a logged reason to
  depart from it; cache the resolved backend per (model, commit) so a re-run reproduces it; and exclude
  candidates whose `demo_path` is missing before ranking.

- **[F30] The registry drift gate detects the stale template and is wired never to block.**
  `sync-registry --check` exits non-zero on a mapped path that no longer exists, and its docstring says
  it exists so *"a pre-plan gate fails loudly instead of the planner silently mis-pointing at a stale
  sibling"*. `up`/`auto-up` reach the same check through `_warn_on_registry_drift()`, whose docstring
  states the opposite as a commitment: *"Never raises: neither a fetch nor a drift check may block
  bring-up."* It printed all 26 stale paths and proceeded to select a backend whose directory had been
  deleted six hours earlier. **The shipped registry itself has 26 missing paths** (`XTTS-v2`,
  `hf_eager universal` ×9, `tt_dit/minimax_h3`, …), entering through the tool's own commits.
  **Fix:** global drift can stay advisory, but a missing `demo_path` on the *selected* entry should be
  fatal. Also narrow the `except Exception: pass` around the check, and fix the verbosity guard —
  `os.environ.setdefault("TT_HW_PLANNER_VERBOSE", "0")` plus a bare truthiness test means the flag can
  never be off and `TT_HW_PLANNER_VERBOSE=0` does not turn it off.

- **[F19] Template dispatch can silently run a different model, and the template can be the tool's own
  earlier output.**
  Handed our three-block model, `auto-up` printed `Runs canonical HF id out-of-the-box:
  …/voxtral-tts-backbone` — a **different checkpoint**, the Block-1 export from a previous experiment —
  under a header reading `Compat verdict: READY`. The warning existed, as a prose bullet below the
  header. Left overnight this yields a finished run, with metrics, for a model nobody asked about.
  **Fix:** compare the template's `canonical_hf_id` against the model being ported and refuse to
  dispatch when they differ; do not print another model's id as a feature.

- **[F18] The architecture gate tests the model's name, not its structure.**
  `detect_family(cfg)` matches `cfg["model_type"]` against hardcoded name lists; every input to the
  decision (`model_type`, `architectures`, `is_encoder_decoder`) is a name, and an unknown name causes
  an early return with an empty block table. The structural answer is available in config fields the
  tool has already loaded.
  **Fix:** decide from structure — do the config's fields describe a decoder-only stack? A patch along
  these lines worked for us, but note it interacts with routing: making `compat` return READY for an
  inferred family flipped the generic-demo gate. Require the family to come from a *known* `model_type`
  before routing to a generic demo, and require the config to declare at most one stack.

- **[F22] The isolation worktree silently ignores uncommitted changes to the tool's own source.**
  The worktree is created from `HEAD`, so an edit in the working tree does not exist inside the run —
  and nothing says so. We lost a run to this: a patch was written, compile-checked, and the run failed
  with the identical error the patch fixes. The people most likely to edit this tool are the people
  extending it, and their natural loop is edit → re-run → observe.
  **Fix:** at worktree creation, if `git status --porcelain` is non-empty for tracked tool source, print
  one line naming the files that will *not* be included; print the worktree's commit sha in the banner.

- **[F15/F16/F17] `plan` and `compat` disagree about what the model is; the block table degrades to
  empty, which reads identically to "nothing needed"; and machine-readable structure is declared and
  never read.**
  **Fix:** have the two stages share one structural determination; distinguish "no blocks required"
  from "could not determine blocks" in the output; and consume the structure that is already emitted.

---

## 4. Operability — long unattended runs

- **[F6] `mcp` is declared nowhere, so all ten agent tools are silently absent.**
  The agent's tool server never starts, every tool call silently does nothing, and the run stalls
  without an error. Cost us 11 hours. This is the difference between "the tool does not work" and "the
  tool ported a 3.4 B model correctly in ten minutes".
  **Fix:** declare `mcp` as a dependency, and fail loudly at startup if the tool server does not come
  up.

- **[F2] The READY verdict can never fire.**
  `lambda _: []` where `lambda _: True` was meant, so an all-green compat report is refused: the best
  possible report is the surest failure.
  **Fix:** one line.

- **[F7] All progress flows through one channel, with no cross-check against disk and unbounded
  retries.**
  When that channel is dead nothing notices; a finished component reports `graduated 0` and the loop
  redoes completed work. Highest-value robustness fix in our view.
  **Fix:** cross-check reported progress against the artefacts on disk, and bound the retries.

- **[F32] `termination_check()` blocks for ~30 minutes with no progress channel, and the retry never
  returns.**
  Timings from the driving session: called 15:57:53, returned 16:27:45 (**29 min 52 s**) with an error,
  retried at 16:28:02, never returned. Nothing is emitted in between, so the normal case is
  indistinguishable from a hang — the run above was abandoned as hung when it had in fact returned once
  and was inside a second identical call.
  **Fix:** emit the sub-step and elapsed time from inside the call; bound it and return partial status;
  make a repeated identical call against an unchanged tree cheap or refused.

- **[F33] `worktree-list` can never print ORPHAN, so dead worktrees accumulate looking healthy.**
  `commands/worktree_list.py:20` is `status = "ORPHAN" if id(s) in orphans else "active"` — `id(s)` is
  the builtin object-address `id()`, while `list_orphans()` returns `List[WorktreeSession]`, so an `int`
  is never `in` a list of objects and the ORPHAN branch is unreachable. Six worktrees whose creator PIDs
  were all dead printed `active`; `worktree-cleanup`, which asks `list_orphans()` properly, then
  removed all six and printed `creator-pid=… dead` for the same PIDs. 2.7 GB was reclaimable while the
  listing said otherwise. Separately `_pid_alive` treats `PermissionError` as not-alive, but
  `os.kill(pid, 0)` raises it precisely when the process exists and belongs to another user — that PID
  is alive, and classifying it orphan makes it a `git worktree remove --force` against a live run.
  **Fix:** compare identity (`s in orphans`, or match on `s.path`); treat only `ProcessLookupError` as
  gone; have cleanup print the status the listing computes so the two cannot disagree.

- **[F11] The documented `--max-rounds` default is 20; the real one is 3, and it is the only exit that
  fires.**
  We also found `--target-band` cannot end a run on this model: the band is `[0.0, 0.0]` because
  `active_bytes` is 0, and the check is guarded by `if band and band[0] and band[1]`, where `0.0` is
  falsy — so the band is skipped and `--max-rounds` is the only real terminator.
  **Fix:** align the documented and real defaults; when the band cannot be derived, say so rather than
  silently skipping the stop condition.

- **[F1/F9/F10] A local model directory is handled inconsistently across stages.**
  A local directory gets a reduced probe and the run dies three stages later with a diagnosis pointing
  elsewhere (F1); `optimize <local dir>` is mistaken for a demo directory (F9); and the workaround —
  `optimize <demo-dir>` — loses the model id, so `optimize` cannot build the correctness gate it had
  already written (F10).
  **Fix:** treat a local directory as a first-class model id throughout, and carry it alongside the
  demo directory rather than substituting for it.

- **[F3] A Python exception is reported as "the PCC gate rejected the output".**
  This triggered hours of the wrong work — tuning numerics for what was a crash.
  **Fix:** distinguish "the gate ran and the number was low" from "the gate did not run".

- **[F8] A clean, deliberate refusal returns `rc=1`, so the supervisor resets the card three times.**
  The supervisor's refusal/crash distinction is otherwise good — this is one call site returning the
  wrong code, plus a second refusal path with the same problem.
  **Fix:** return the dedicated refusal exit code from both paths.

- **[F21] `trust_remote_code` is an allowlist containing exactly one model.**
  `if self.base_model_name in ["Phi-3-mini-128k-instruct"]`. So a custom-architecture checkpoint clears
  Step 0 (*"transformers can load … [ok]"*), clears static analysis, and then fails at execution with a
  message blaming the repository rather than the loader.
  **Fix:** decide from the checkpoint — `auto_map` in `config.json` *is* the declaration that a model
  ships custom modelling code, and HF refuses to load such a model without `trust_remote_code`, so its
  presence is decisive and needs no allowlist. Keep an env override for a checkpoint that should not be
  trusted. (We ran with this patched and it worked.)

- **[F25] Decomposition children lose their parent's path prefix, and the plan is copied from another
  model.**
  **Fix:** carry the parent path when emitting children, and key the plan to the model it was generated
  for.

- **[F5] The systemic-pattern detector counts error class names, so the broadest bugs are the ones it
  misses.**
  One root cause wearing three different exception types is counted as three unrelated singletons.
  **Fix:** group by failure family rather than by exception class name.

- **[F23] Capture drivers guess where the config already states the answer.**
  Note our own correction here: three of the four capture misses we originally attributed to this were
  ours, not the tool's. The residual point stands — the driver guesses at inputs the config declares.
  **Fix:** read the declaration before guessing.

- **[F20] The meta-plan is wired to stdout rather than to control flow.**
  The advisory meta-plan correctly described the three-stack structure and its consequences, unprompted,
  and nothing consumed it. We are **not** proposing a fix: on the run we watched, the pipeline ignored
  the meta-plan and was *right* to — F18's corrected routing gate reached a better decision from
  structure than the meta-plan's prose would have. Recorded so you have the observation, without a
  recommendation we cannot support.

- **[O10] `git_commit` sweeps untracked scratch files in the model directory into perf commits.**
  A stray `hifi3_verified_uncommitted.patch` was committed alongside a two-file perf change. Inert here
  — a `.patch` file does not execute, so that measurement stands — but a scratch `.py` in the demo
  directory would be committed *and* change behaviour while the message described something else. The
  agent noticed and removed it in a follow-up commit.
  **Fix:** stage only the files the attempt touched, or warn when the staged set exceeds them.

---

## Measurements, for reference

| | value |
|---|---|
| bring-up | 7/7 components, 114/114 operations on device, 1 round, 42 min |
| e2e correctness | PCC **0.9999834**, exact audio-code match, 0 code flips |
| optimize | 26 attempts, 11 wins, 12 rejected, 3 lost to profiler crashes, 17 h |
| optimizer's reported gain | **−17.2%** (1735.27 → 1437.41 ms, its own metric) |
| gain the product received | **−13.2%** (297.69 → 258.26 ms per audio frame) |
| correctness after optimisation | PCC **0.9999804**, exact code match, 0 flips |
| shipped pipeline speed | **258.26 ms/frame**, RTF 3.319 (real time is 80 ms/frame) |
| hand-written port, same model and board | **26.9 ms/frame**, RTF 0.357 |

The 9.6× difference is mostly structural — no KV cache on the shipped path, and no trace replay — not
a tuning deficit. Both trace back to item **#1**.

---

*Prepared from `TOOL_FINDINGS.md` (entry IDs match). That file carries the full evidence: exact log
excerpts, reproduction commands, the measurements above, a record of what we checked and found sound,
and a Corrections section listing the four claims of our own that did not survive scrutiny.*
