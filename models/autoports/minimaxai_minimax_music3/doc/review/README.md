# Stage 10 — review, clean-up, pull requests, final report

Work log for the review of the whole worktree diff (`git diff e946955cc15..HEAD -- models/autoports`, 217 files,
all added by stages 02-09), the fixes it produced, the repository hygiene pass, the top-level model README, the draft
pull requests and the final report (`~/mm3-bringup/REPORT.md`). Attempts 1 and 2 (attempt 1 ran the review, the fixes and the hygiene pass and was cut off while the re-verification
chain ran; attempt 2 read the chain results, committed, pushed and opened the pull request); headless, so every decision below was
taken without asking and is recorded here.

Hardware for the re-runs: one chip of the P300x2 host `qbge-devex-02` (`TT_METAL_VISIBLE_DEVICES=0`, board `p300c`,
`tt-smi -s` board id `000004613193411b`), 1x1 mesh on device 0, under the `with_hw_lock` flock.

## How the review was run

Skills: `tt-review-router` first, then `tt-review-core` plus two domain skills, `tt-model-bringup-review` (all of the
device code is model code under `models/`) and `tt-trace-review` (four trace capture sites: backbone decode, depth seed
and step, DiT Euler step). Not selected: `tt-precision-review` (the dtype policy was swept and reviewed in stage 07,
`doc/optimize/dtype_sweep.md`), `tt-test-coverage-review` (every stage has a gate test file; coverage questions were
folded into the core review). The `tt-comment-hygiene-review` checks (iteration-journey comments, magic numbers) were
part of the hygiene pass.

Three independent read-only reviewers, each holding the skill texts, covered:

1. AR side: `tt/llm.py`, `tt/ar_generator.py`, `tt/depth_decoder.py`, `tt/prompt.py`, `tt/constants.py` and their tests,
   against diffusers `encoders.py` / `minimax_music3_rvq_depth_decoder.py`.
2. DiT / pipeline side: `tt/flow_transformer.py`, `tt/denoiser.py`, `tt/scheduler.py`, `tt/condition_encoder.py`,
   `tt/vocoder.py`, `tt/vocoder_worker.py`, `tt/pipeline.py`, `tt/audio_metrics.py` and their tests, against
   `transformer_minimax_music3.py`, `before_denoise.py`, `denoise.py`, `classifier_free_guidance.py`.
3. Server, scripts, manifest, docs and repository hygiene (`server/`, `scripts/`, `tt-model.yaml`, `reference/` headers,
   tracked-file inventory, secrets, cross-checks of the numbers quoted in the work logs against the evidence JSONs).

Every finding was re-read against the code before acting. Severity labels are the `tt-review-core` ones
(MUST-FIX / SHOULD-FIX / CONSIDER). No MUST-FIX finding was raised; the reviewers' explicit "not flagged" lists cover
the reference arithmetic (constants, prompt assembly, CFG + top-k, end token, depth loop, chunk / overlap / crop
arithmetic, sigma schedule and Euler step, RoPE pairing and position offset, LayerNorm bias, gate / SiLU order, folded
convs), trace safety at all four capture sites (nothing host-side inside capture, warm compile with identical
signatures, PCC asserted on replay output, persistent buffers allocated before capture, state refreshed outside),
logical batch 2 preserved everywhere, and the asserted PCC bars matching the work logs.

## Findings and dispositions

### Fixed (code)

| # | severity | finding | fix | re-verified by |
|---|---|---|---|---|
| 1 | SHOULD-FIX | `ARGenerator` silently fell back to a full-vocabulary (12.8 MB / frame) host read-back when the backbone trace already existed without the logits window (`tt/ar_generator.py`); the docstring called this a precondition, the code continued past it (hidden host fallback, invariant enforced late) | `MusicLLM.release_trace()` added; the generator releases such a trace and installs the window, the trace is re-captured on the first frame; the fallback branch of `_decode` is deleted | gate 04 in the 07 chain |
| 2 | SHOULD-FIX | `_default_weight_cache_root()` read `TT_CACHE_PATH` back as the root, which `MusicLLM.__init__` had just pointed at `<root>/<policy>`; a second `MusicLLM` in one process (`scripts/policy_probe_device.py`) nested the converted-weight caches | the root is resolved once per process (module-level) | gate 02 |
| 3 | SHOULD-FIX | `load(vocoder_threads=...)` set the *parent's* torch threads; the vocoder runs in the spawned worker whose thread count was a fixed formula | `vocoder_threads` is passed to `VocoderProcess`; the parent setting is gone; docstring corrected | gate 06 |
| 4 | SHOULD-FIX | host-vs-device vocoder path chosen with `hasattr(vocoder, "mesh_device")` / `hasattr(vocoder, "release")` although `load` had the validated mode string | `MiniMaxMusic3Pipeline(vocoder_mode=..., preset=...)` constructor arguments; `overlap_vocoder = vocoder_mode == "host"`; `release()` calls the device vocoder only in device mode; `policy_report` no longer uses `getattr` | gate 06, 07 |
| 5 | SHOULD-FIX | a crashed vocoder worker (OOM kill, signal) left `ProcessPoolExecutor` broken for the life of the server, so one worker death failed every later request | `VocoderProcess._submit` catches `BrokenProcessPool`, discards the pool and re-spawns it once (logged); new host-only test `test_vocoder_worker_respawns_after_death` kills the worker with SIGKILL and checks the next submit returns the same waveform | test passed (1 passed, 2.0 s, host) |
| 6 | CONSIDER | `steps = steps or self.num_inference_steps` mapped `steps=0` to 30 in `ChunkDenoiser.denoise_chunk` | `None` check; the scheduler's `>= 1` assertion is the error path | gate 05 |
| 7 | CONSIDER | no size cap on `input` / `instructions` before the slow host tokenizer ran on the whole body | `max_length=200_000` on both fields (a 5000-token prompt is far below); documented in `server/README.md` | gate 08 |
| 8 | CONSIDER | `HF_MODEL` fell back to `MM3_WEIGHTS`, a variable that exists only on the bring-up host, in the shipped server | fallback removed (the launcher only ever exports `HF_MODEL`; the test exports it too) | gate 08 |
| 9 | CONSIDER | `tests/test_pipeline.py` relied on the repo-wide 300 s timeout; the session pipeline load (cold caches ~150 s) plus the golden replay left little margin | explicit `@pytest.mark.timeout` per hardware test, as in the sibling files | gate 06 |
| 10 | CONSIDER | `tests/test_llm_perf.py` defaulted the recorded board id to this host's, so a perf JSON written elsewhere would claim it | `MM3_BOARD_ID`, else `tt-smi -s` JSON (`board_id`, `board_type`) for the visible chip, else `unknown` | host check (`000004613193411b (p300c, tt-smi -s)`) |
| 11 | CONSIDER | `scripts/dtype_sweep.sh` sourced `~/mm3-bringup/common.sh` unconditionally and continued with unset variables when it was missing | requires `MM3_WT` / `MM3_PY` / `MM3_MODEL_DIR` from the environment with a clear error, defines a pass-through `with_hw_lock` when the flock helper is absent | `bash -n` |

### Fixed (documentation)

| # | severity | finding | fix |
|---|---|---|---|
| 12 | SHOULD-FIX | `server/README.md` and `doc/server/README.md` said a non-`wav` `response_format` gives 400; the check is a pydantic validator, so it is 422 (as the results JSON and the same log's table record) | both say 422 (body-validation error) |
| 13 | SHOULD-FIX | `doc/server/README.md` said "the tests assert exit code 0" while the test accepts 0 or -15 and -15 was observed | corrected |
| 14 | SHOULD-FIX | `doc/server/README.md` carried an editing note ("this line was added in the follow-up commit") and named a commit the results JSON does not record | states both commits and why they are equivalent |
| 15 | SHOULD-FIX | `doc/ar_generator/README.md` described the full-length host multinomial as the shipped sampling path; since stage 07 the hot path is the candidate multinomial (`_guided_window` + `sample_top_k_candidates`) | "superseded in stage 07" notes on the implementation line and the two transcription rows; `_full_logits` docstring says diagnostics-only |
| 16 | SHOULD-FIX | `doc/optimize/README.md` called the policy "LoFi decode matmuls" while `LI_FF1_FF3` / `LI_FF2` also cover the prefill MLP (tt_transformers has no prefill MLP group) | stated explicitly, with the note that the golden replay includes the prefill |
| 17 | CONSIDER | `doc/container/README.md` quoted 83.47 s for the stage-08 host wall of the model-card request; `doc/server/results.json` holds 83.58 s (83.47 was an earlier, overwritten run) | 83.58 s in the three places |
| 18 | CONSIDER | `server/README.md` shipped in the image with a launch recipe that only works on the bring-up host (`source ~/mm3-bringup/common.sh`) | plain `HF_MODEL=... python -m uvicorn ...` recipe and a plain pytest command; the bring-up shorthand kept as a note |
| 19 | CONSIDER | `.gitignore` ignores Tracy `ops.csv.gz` / stacked PNGs for stages 05 and 07 but stage 03 tracks them | kept (the stage-03 log derives and cites its device-time figures from them, 1 MB total); the reason is now in `.gitignore` |

### Not fixed (recorded decisions)

| # | severity | finding | decision |
|---|---|---|---|
| 20 | CONSIDER | `num_inference_steps` / `max_new_tokens` violate the lower bound with 422 (pydantic `ge`) and the upper bound with 400 (`frames()` / `steps()`) | Left: the codes are recorded in `doc/server/results.json` and the README table and are what the container gate validated; unifying them changes the served contract for cosmetics. |
| 21 | CONSIDER | `tt-model.yaml` `weights.ignore_patterns` lacks `scripts/*`, which the app's own `snapshot_download` excludes | Left: the manifest is the one the published package `jashansinghTT/MiniMax-Music3-tt` was built from; the difference is a few KB of upstream shell scripts. Noted for the next package revision. |
| 22 | CONSIDER | `/health` `status: "loading"` / 503 branch is unreachable while uvicorn runs with `--lifespan on` | Left as the shutdown-window response; harmless. |
| 23 | CONSIDER | `MM3_DIT_MATMUL_CONFIGS` / `MM3_DIT_IN0_BLOCK_W` / `MM3_DIT_SILU_MODE` change DiT numerics but are absent from `policy_report` | Left: these are sweep knobs (`scripts/dit_grid_sweep.py`) that default to the evidenced configuration; the server does not set them. Adding them to `policy_report` is a small follow-up. |
| 24 | CONSIDER | `getattr(pc, "fused_activation", None)` and `getattr(self, name, None)` in `DiTStepTrace.release` / `DepthStepTrace.release` for attributes that always exist | Left: cosmetic; touching the traced modules would require re-running the 05 / 03 gates for no behavioural change beyond what the chain already covers. |
| 25 | CONSIDER | teacher-forced generation runs the `_semantic_rank` diagnostic (a `[2, 200000]` fp32 tensor per frame) unconditionally, so the "teacher-forced frames/s" figure is slightly pessimistic | Left: before / after were measured like for like; the free-running figure (the headline) does not include it. |
| 26 | CONSIDER | `tests/test_ar_generator.py` records `bar: 0.99` while asserting `PCC_FRAME_MIN` (0.97 under the optimized policy) and does not record the LLM policy | Left: the committed JSON is a functional-policy run, consistent today; noted for the next edit of that test. |
| 27 | CONSIDER | four near-duplicate CFG implementations in `tt/ar_generator.py` (hot path + three diagnostics) | Left: `test_optimized` asserts the candidate sampler against the reference sampler; consolidating the diagnostics is a refactor without behavioural change. |
| 28 | CONSIDER | `embed_frame` converts a non-tile input instead of asserting; `prefill()` round-trips a device tensor through the host (stage-02 tests only); | Left: neither is on the generation path. |
| 29 | CONSIDER | `tests/test_flow_transformer_perf.py` writes traced and eager timings into the same `dit_forward_eager.json` when `MM3_DIT_TRACED=1` | Left: the committed evidence was produced with the default (eager) mode; the traced numbers the work log cites come from `scripts/dit_step_timing.py`. |
| 30 | CONSIDER | vocoder worker failures surface only after the chip warm-up; an orphaned worker is plausible if the parent is SIGKILLed (not reproduced) | Left: inside the container the PID namespace tears the worker down with the server; on the host a `timeout`-killed pytest can leave it (not reproduced). Recorded as a hardening follow-up. |

## Hygiene inventory (from the third reviewer, re-checked before committing)

- Tracked files over 1 MB: none (largest `doc/flow_dit/pcc/scheduler_triples.pt`, 506 KB, a test input). The gate's
  5 MB check passes.
- Secrets: none. `HF_TOKEN` appears only as a pass-through name (`--env HF_TOKEN`) in the container log.
- `reference/`: every file carries the Apache-2.0 header and the diffusers source path it was vendored from.
- Scripts: all 30 under `scripts/` are cited by a work log; none references a repository file that no longer exists;
  every hardware invocation in the `.sh` collectors has a `timeout`.
- Absolute `/home/...` paths: none in `tt/`, `server/`, `tests/`, `scripts/`, `reference/` code. They remain in
  evidence JSONs (weights directory, log paths) and in `tt-model.yaml` `source.tt_metal` (the build-time source
  pointer the package was built from; it is a build input, not read at serve time).
- No `TODO` / `FIXME` / debug prints / commented-out code in `tt/`, `server/`, `tests/`.
- `.gitignore`: `generated/` and the Tracy raw dumps are ignored; the small evidence files the work logs cite are
  re-included explicitly.
- The stage-09 gate re-runs had rewritten seven `doc/*/results.json` files (same PCC values, new timestamps / commit
  fields); they were restored with `git checkout` so the committed evidence matches the work logs that cite it.

## Re-verification on the device

Chain: `checks/07.sh` (= `tests/test_optimized.py` then the stage 02-06 gate tests: `test_llm.py`,
`test_depth_decoder.py`, `test_ar_generator.py`, `test_flow_transformer.py`, `test_pipeline.py`, plus the perf JSON
check) then `checks/08.sh` (`tests/test_server.py`), one process at a time under the hardware lock.

Run from `generated/gate10_chain.sh` (log `generated/gate10_chain.log`, 02:26-02:35 on 2026-09-10) on the working tree
that this stage commits, after all fixes above. Attempt 1 of this stage started the chain and was cut off while it
ran; attempt 2 let it finish and read the results (the chain kept running under the hardware lock).

| gate | test file | result | wall |
|---|---|---|---|
| 07 | `tests/test_optimized.py` | 6 passed | 47 s |
| 07 | `tests/test_llm.py` | 9 passed, 1 deselected (slow) | 26 s |
| 07 | `tests/test_depth_decoder.py` | 11 passed | 10 s |
| 07 | `tests/test_ar_generator.py` | 6 passed | 67 s |
| 07 | `tests/test_flow_transformer.py` | 12 passed, 1 deselected (slow) | 42 s |
| 07 | `tests/test_pipeline.py` (incl. the new `test_vocoder_worker_respawns_after_death`) | 8 passed, 1 deselected (slow) | 148 s |
| 07 | perf JSON check | `GATE07_OK ar 13.32->21.65 frames/s, dit chunk 3.5->2.6 s` | - |
| 08 | `tests/test_server.py` | 7 passed | 150 s |

Both gates exited 0. The runs rewrote the seven `doc/*/pcc/results.json` / `doc/server/results.json` evidence files
(same PCC values; timings within noise, new timestamps); as in the hygiene pass they were restored with `git checkout`
so the committed evidence stays the one the stage logs cite. The AR-generator log line of this run
(`14.36 frames/s warm`, LLM 37.7 ms + depth 32.5 ms) comes from that file's default fixture (functional backbone, bf16 depth
decoder; `MM3_LLM_POLICY` is not set by the gate), not from the optimized headline path, which `test_optimized.py` covers.

## Repository deliverables of this stage

- `README.md` (model root): what it is, hardware, how to run the tests, how to serve (host and container), the
  measured performance table, accuracy summary, known limitations (single-chip profiles and the dual-chip overlap
  follow-up, below-realtime AR loop, host vocoder, no streaming / batching, precision cost of the optimized policy).
- `doc/review/README.md` (this file).
- The fixes listed above.

## Pull requests

- tt-metal: branch `jashan/minimax-music3` pushed to `origin` (`tenstorrent/tt-metal`), draft pull request against
  `main`: `https://github.com/tenstorrent/tt-metal/pull/56043` (draft; base `main`, head `jashan/minimax-music3`). The branch is stacked on `agentic-research/hous/multigoal-claude` plus the local Muse-Glimmer
  commits the worktree was started from (the clone is shallow at `ecd7c64d0ff`), so the PR's commit list includes those
  parents; the MiniMax-Music3 change is exactly `models/autoports/minimaxai_minimax_music3/`. Decision: not rebased,
  because the stage work logs cite commit SHAs; stated in the PR body.
- tt-model-manager: `https://github.com/tenstorrent/tt-model-manager/pull/89` (stage 09, `fix(container): match container
  names exactly in running()`), confirmed still a draft (`gh pr view 89 --json isDraft` -> true).
- Both URLs are recorded in `~/mm3-bringup/state/10.prs.txt`.
- Commits of this stage: `e4630c9c021` (review fixes, README, review log) and the follow-up commit that records the PR URLs in this log.

## Open risks

- The review was performed by three model-driven reviewers holding the skill texts plus a human-style triage of their
  findings; it is not a substitute for a maintainer review of the tt_transformers integration (`tt/llm.py` drives
  `models/tt_transformers` through its public `Generator`-level API but re-implements the decode trace).
- The `CONSIDER` items left open above are all small; #23 (report the DiT env knobs) and #30 (worker liveness) are the
  two worth doing before the container is used outside this host.
