# AutoDebug report: DiffusionGemma GPQA doc 0 degenerates to `1`

Date: 2026-07-23
Scope: inspection plus a serialized on-device prompt-contract A/B; no source or external state changed
Issue: tt-metal #48291, comment 5062743522

## Verdict

The reported failure is real and deterministic, but a follow-up control resolves
its origin as an **invalid thinking-template invocation in the local GPQA
launcher**, not as evidence of a host-Gumbel or #48291 numerical defect.

The failing launcher put literal `<|think|>` in a system message but did not set
the server's `enable_thinking=true` chat-template argument. With the checkpoint
template, that combination renders both the trigger token and the disabled-mode
generation suffix:

`<|turn>model\n<|channel>thought\n<channel|>`

The suffix says that an empty thought channel has already ended. The supported
thinking invocation instead uses server-side
`--default-chat-template-kwargs '{"enable_thinking":true}'`, does not inject a
manual system token, and ends the generation prompt at `<|turn>model\n`.

A serialized same-device doc-0 rerun with that corrected contract, the same
host-Gumbel/seed-0/up-front-trace/early-halt stack, and the same 3,072-token
budget used by the earlier failing artifacts produced coherent physics
reasoning and the correct `\boxed{C}` answer (`exact_match=1`). Every block
converged, with K values `16,11,12,11,8,7`. The bad 3,072-token runs emitted
3,072 garbage tokens; the requested 6,144-token run had the same bad prefix.
This controlled contrast falsifies the issue comment's sampler/precision
attribution for this case.

The first point at which this artifact proves bad model output is the **final
clean-argmax commit of block 0 after all 48 denoise steps**, not the first
denoise step. Block 0 never satisfies the early-halt condition. The serving path
nevertheless commits its last clean argmax because both TT and upstream
DiffusionGemma define the max-step cap as a normal terminal condition. That bad
256-token commit is then written into KV and exposed as context for block 1.
This is a proven propagation/amplification mechanism and a missing product-level
failure-containment policy; it is not the originating cause in the corrected
prompt control, which never reaches the cap.

The production artifact alone did not localize the origin because it lacks
per-step tensors. The follow-up prompt-contract A/B localizes this reported case
one stage earlier, before prefill. Exact HF/TT injected-noise replay remains
useful for the broader #48291 question, but it is no longer required to explain
this doc-0 collapse.

## Proven headline findings

1. **Observed output is exactly as reported.** Re-tokenizing doc 0 with the
   checkpoint tokenizer gives 6,144 tokens (24 blocks of 256). Block 0 contains
   nine token IDs, dominated by:
   - token `621`, text ` \` (106 positions);
   - token `236770`, text `1` (104 positions);
   - token `6465`, text `ní` (20 positions).
   Blocks 1 through 23 are each exactly 256 copies of token `236770`.

2. **The repeated `1` is ordinary text, not EOS/pad or a control token.**
   Tokenizer identities are:
   - `0 = <pad>`;
   - `1 = <eos>`;
   - `98 = <|think|>`;
   - `236770 = "1"`.
   Therefore the repeated token cannot trigger EOS stopping.

3. **The log does not show degeneration at denoise step 0.** It has only
   block-level final metrics, not per-step token decisions. Doc 0 block 0 reports
   `denoise_steps=48`, `halted=false`. The strongest supportable statement is:
   degeneration is visible in block 0's final commit. Any claim that it began on
   the first denoise step requires a trajectory capture that does not exist here.

4. **The first nine blocks exhaust K=48 and are committed anyway.** Doc 0's
   block step counts are:

   `48,48,48,48,48,48,48,48,48,47,35,28,25,27,21,19,22,17,12,7,6,6,6,6`

   Blocks 0-8 have `halted=false`; block 9 and later report `halted=true`.
   Since blocks 1-23 are already all `1`, the later short halts mean the poisoned
   trajectory eventually becomes stable and very low-entropy around the repeated
   token. Early halt is faithfully recognizing that attractor; it is not recovery.

5. **Commit-after-cap is explicit, not an accidental early-halt branch.**
   `EarlyHaltTracedDenoiseController.denoise_block()` executes windows until
   either `eval_halt()` succeeds or `max_denoise_steps` is exhausted, then
   returns the last device clean argmax in both cases
   (`tt/traced_denoise.py:1140-1170`). The eager loop has the same terminal
   behavior (`tt/denoise_loop.py:688-751`). `denoise_and_commit_block()` always
   passes that result to `commit_fn` (`tt/generate.py:911-971`).

6. **The bad commit becomes subsequent model context immediately.**
   Serving writes all 256 committed tokens to KV, advances `next_pos`, and
   updates the denoise prefix (`tt/serving.py:380-418`). For doc 0:
   - natural prompt length: 161;
   - aligned cached prefix: 192;
   - block 0 commit writes positions `[192,448)`;
   - block 1 denoises with that committed block revealed in its prefix.
   The reveal-mask logs show the prefix advancing block by block. The reveal mask
   therefore propagates the bad commit by design; it does not independently prove
   how block 0 became bad.

7. **Upstream also commits a max-step clean argmax.** Official Transformers
   iterates at most K steps and appends `argmax_canvas` after the loop regardless
   of whether adaptive stopping fired
   (`generation_diffusion_gemma.py:755-803`). A blanket "do not commit unless
   halted" policy would be a product safeguard, not upstream algorithm parity.

8. **Host Gumbel refresh and seed derivation are internally coherent.**
   Every vLLM serving session is constructed with seed 0. The three independent
   streams are derived as:
   - initial canvas: seed 0;
   - renoise tokens: seed 1;
   - host Gumbel: base seed 2.
   Host Gumbel is deterministic and stateless per block/step:
   `seed = 2 + block_idx * 1,000,003 + step`
   (`tt/generate.py:517-549`). In the traced early-halt controller, the persistent
   Gumbel buffer is refreshed before each one-step replay window
   (`tt/traced_denoise.py:1140-1165` with window size 1). No artifact evidence
   shows a stale Gumbel buffer.

9. **All sessions resetting to seed 0 is a reproducibility policy and a risk,
   not proof of the bug.** It makes the same prompt deterministically revisit the
   same stochastic trajectory. Doc 1 also starts at seed 0 and is coherent, so
   seed reset does not imply global corruption. A seed sweep is required to test
   whether doc 0 is a narrow pathological trajectory.

10. **Doc 1 sharply lowers suspicion of global stale trace state.** Immediately
    after doc 0, the same server, trace family, reveal-mask mechanism, host-Gumbel
    mode, and seed policy produce coherent text. Its five blocks halt in
    `20,16,11,10,2` steps. This does not rule out prompt-length- or
    request-specific trace behavior, but it argues against a globally poisoned
    trace/Gumbel buffer or universally broken host sampling.

11. **EOS handling explains neither the onset nor continuation.** The TT session
    deliberately has internal stop IDs disabled so vLLM owns request-level
    stopping. Doc 0 has `stop=false` in all block metrics and emits ordinary
    token `236770`, not EOS ID 1. Doc 1 stopping successfully after coherent
    output is positive evidence that the outer serving stop path can work.

12. **The launcher produced a contradictory thinking prompt.** The rendered
    prompt contains one `<|think|>` token (ID 98), but token count alone is not
    sufficient. Because the server's `enable_thinking` template argument remained
    false, the generation suffix also contains
    `<|channel>thought\n<channel|>`, the checkpoint's disabled-thinking marker.
    A tokenizer control shows:
    - failing manual injection: system `<|think|>` plus an already-closed empty
      thought channel;
    - supported invocation: server-side `enable_thinking=true`, one injected
      `<|think|>`, and no empty thought suffix.
    The absent `--reasoning-parser` remains post-generation presentation-only;
    adding it is appropriate for evaluation but is not the generation fix.

13. **Existing 3,072-token counterparts are bit-for-bit identical prefixes.**
    Three prior doc 0 sample artifacts re-tokenize to the same 3,072-token ID
    sequence, SHA-256
    `e212c5a1412fefc00b3c82ef56f6dbcf613691285aadc0bfdc76552aaad92ef8`:
    - `/tmp/dg-upfront-gpqa-20260723T144509Z/smoke/.../samples_r1_gpqa_diamond_2026-07-23T14-56-46.299368.jsonl`
    - `/tmp/dg-upfront-gpqa-20260723T153006Z/smoke/.../samples_r1_gpqa_diamond_2026-07-23T15-41-23.225050.jsonl`
    - `/tmp/dg-upfront-gpqa-20260723T153006Z/full/.../samples_r1_gpqa_diamond_2026-07-23T17-59-48.210035.jsonl`

    The requested 6,144-token artifact has exactly that SHA for its first 3,072
    tokens. This proves deterministic reproduction under the same malformed
    thinking-template configuration. It does not identify host sampling,
    precision, or trace as the cause.

## Execution trace: why non-convergence is committed and propagated

### Session construction

`generator_vllm._make_session()` creates `BlockDiffusionServingSession` with the
configured denoise function and `gumbel_mode=host`
(`tt/generator_vllm.py:592-621`). The session constructor seeds all request-local
random sources from the same request seed (`tt/serving.py:195-203`). Metrics in
the requested log confirm `seed=0`, `gumbel_mode="host"`, and early halt enabled.

### Prompt geometry

`prefill_prompt_tokens()` retains the natural length but pads token IDs with zero
to a multiple of 32 (`tt/generate.py:246-282`). Zero is this checkpoint's real
`<pad>` ID. For doc 0 this transforms 161 natural tokens into a 192-token cached
prefix. Denoise uses the aligned cache length as both the visible prefix and the
canvas RoPE offset. Consequently, product TT starts the first canvas at absolute
position 192.

This is a concrete semantic difference from a normal official HF generation on
the unpadded 161-token prompt. The repository's HF-vs-TT replay utility also pads
HF to 32 and starts its decoder at the aligned position
(`demo/replay_hf_tt.py:43-48,290-295`), so existing replay parity does not test
the natural product prompt geometry. This difference is not proven causal—the
same product path handles doc 1—but it must be isolated before attributing the
failure purely to bf16 or TT kernels.

### Denoise and early halt

For each step, the controller:

1. refreshes host Gumbel for this `(block_idx, step)`;
2. executes the captured one-step trace;
3. reads scalar mean entropy and argmax mismatch;
4. halts only when mismatch is zero and mean entropy is below 0.005.

At block 0 none of steps 0-47 meets the joint condition. The controller reaches
the cap and returns the clean argmax produced by step 47. The log does not retain
the values of the 48 intermediate canvases, entropy vectors, acceptance masks,
or argmaxes, so it cannot show when the trajectory first departed from a good
reference.

### Commit and context poisoning

The serving commit callback writes the 256-token clean argmax into persistent KV,
advances `next_pos` by 256, and rebinds the denoise adapter to the expanded
prefix. Reveal-mask reuse then exposes exactly that expanded committed prefix
while hiding the uncommitted fixed-capacity tail. Block 1 is therefore
conditioned on block 0's mixed garbage. Once block 1 commits 256 ordinary `1`
tokens, that repetition itself is appended, and the feedback repeats. By block 9
the repeated trajectory is stable/confident enough to halt.

Thus:

- **proven origin point available in artifacts:** bad final block-0 clean argmax;
- **proven amplifier:** unconditional max-cap commit plus KV/reveal advancement;
- **unresolved:** the first internal denoise step/operation at which block 0
  diverged.

## Ranked root-cause assessment

### 1. Malformed thinking-template contract — proven cause of this reported collapse

The local launcher enabled thinking by supplying literal `<|think|>` as a system
instruction while leaving the server template at `enable_thinking=false`. That
rendered the disabled-mode empty thought suffix after the user turn. A corrected
server-side `enable_thinking=true` run, without the manual system instruction,
changed the natural prompt from 161 to 157 tokens (both paths used the same
aligned product prefill mechanism), converged in every block, and scored doc 0
correctly. The same host-Gumbel implementation, seed 0, trace controller, reveal
mask, and sparse-MoE stack were retained.

### 2. Exact prompt geometry and TT/HF precision — broader #48291 follow-up, not needed for doc 0

Natural-versus-aligned prompt positions and exact HF-fp32/HF-bf16/TT trajectories
remain valid questions for #48291. They are demoted for this case because the
corrected production-geometry TT run is coherent and correct. The malformed
prompt, not a TT/HF numerical comparison, explains the failing-versus-passing
boundary.

### 3. Request-specific traced/reveal behavior — possible but lower probability

This remains testable because the failing run uses upfront trace capture and
cross-prefix reveal-mask reuse. It is weakened by:

- coherent doc 1 immediately afterward;
- repeated reveal-mask reuse metrics with advancing prefix;
- existing eager-vs-trace exactness tests for host randomness and upfront reuse.

Those tests are not the exact 161/192 GPQA prompt at K=48. Only an exact
eager-vs-traced replay can rule this class out.

### 4. Host-Gumbel refresh/seed defect — low support

The code derives unique deterministic Gumbel noise per block/step and refreshes
the persistent device buffer before each one-step replay. Existing host
production runs on other prompts are coherent, including
`/tmp/dg48291_official_host_prod48_seed1_20260723.log` (TT halted in 10 steps).
An exact host-vs-argmax and seed sweep are still needed because the failing
trajectory is prompt-specific, but there is no direct evidence of stale or
reused Gumbel data.

### 5. Max-step commit without failure containment — proven amplifier, not origin

This policy guarantees that a non-converged block enters context. It explains
how one bad block poisons every later block and why the server continues for the
full token budget. It cannot explain why block 0's final clean argmax is bad, and
it matches upstream generation semantics. Treat it as a robustness gap.

### Effectively ruled out as originating explanations

- token `1` being EOS or special: false (`236770` is ordinary);
- missing reasoning parser: post-generation presentation only;
- premature early halt in block 0: false; all 48 steps ran;
- universally stale trace or globally broken host sampling: contradicted by
  coherent doc 1, though a request-specific defect remains possible;
- later server `trisc1` build/leak messages: they occur outside the observed
  smoke generation and cannot retrospectively produce its deterministic tokens.

## Decisive next controls

Use one serialized exact doc 0 prompt token tensor and save every random input and
per-step decision tensor. Do not regenerate "equivalent" randomness separately
inside each backend.

1. **Correct thinking-template control — completed.**
   - Failing: manual system `<|think|>`, server `enable_thinking=false`.
   - Passing: no manual system token, server `enable_thinking=true`, reasoning
     parser enabled for response extraction.
   - Result: exact match 1; six coherent blocks; K `16,11,12,11,8,7`.

2. **Prompt geometry control (broader #48291 follow-up).**
   - Official HF on the natural 161-token prompt, canvas at position 161.
   - Official HF on the identical 31-`<pad>`-extended 192-token prompt, canvas at
     position 192.
   - Same initial canvas, Gumbel tensors, renoise tensors, K=48, seed 0.
   This isolates the concrete product/reference semantic difference.

3. **HF-fp32 versus HF-bf16 with injected identical noise.**
   Record for every step: clean argmax, sampled IDs, entropy, accept mask,
   renoised canvas, mean entropy, mismatch, and halt decision.
   - both degenerate identically: prompt/seed/model trajectory, not TT;
   - fp32 coherent and bf16 degenerate: intrinsic precision bifurcation;
   - both HF modes coherent: proceed to TT localization.

4. **TT eager versus TT traced on exact replay tensors.**
   Use the same padded prompt/cache geometry and all saved step inputs.
   - eager good, traced bad: trace/reveal/buffer-refresh defect;
   - eager and traced bad at the same first step: shared TT forward/terminal path;
   - exact agreement: trace is exonerated for this failure.
   Also compare reveal-mask reuse against a fresh per-prefix capture.

5. **Host Gumbel versus argmax.**
   Keep prompt geometry, initial canvas, renoise tensors, and all other flags
   fixed. Argmax removes Gumbel sampling from denoiser token selection. A coherent
   argmax result only localizes sensitivity to the stochastic trajectory; it
   does not by itself prove the host refresh implementation is wrong.

6. **Thinking on/off.**
   Render both prompts through the checkpoint template and archive token IDs.
   This changes model input and therefore cannot isolate precision, but it tests
   whether this valid `<|think|>` path is the prompt-specific trigger. Keep
   geometry and seed handling explicit in both cases.

7. **Seed sweep.**
   Run at least seeds 0-9 on the exact host-Gumbel+thinking prompt. Report
   block-0 halt/K, final mean entropy/mismatch, unique-token count, EOS presence,
   and output quality. A seed-0-only failure supports a pathological trajectory;
   broad failure supports prompt geometry or a systematic implementation issue.

8. **Failure-containment experiment (after localization).**
   Flag or abort a block when K is exhausted without halt, rather than silently
   committing it. This would prevent context poisoning and expose the original
   bad block clearly. It is a serving robustness experiment, not a root-cause
   fix, and changes upstream behavior.

## Evidence reviewed

- `/tmp/dg-upfront-gpqa-20260723T190621Z/server.log`
- `/tmp/dg-upfront-gpqa-20260723T190621Z/smoke/google__diffusiongemma-26B-A4B-it/samples_r1_gpqa_diamond_2026-07-23T19-21-25.854591.jsonl`
- `/tmp/dg48291_doc0_correct_thinking_20260723/server.log`
- `/tmp/dg48291_doc0_correct_thinking_20260723/eval/google__diffusiongemma-26B-A4B-it/samples_r1_gpqa_diamond_2026-07-23T20-11-20.040786.jsonl`
- `/tmp/dg48291_doc0_correct_thinking_20260723/eval/google__diffusiongemma-26B-A4B-it/results_2026-07-23T20-11-20.040786.json`
- the three 3,072-token counterpart sample files listed above
- `/home/zni/dg_models/diffusiongemma-26B-A4B-it/chat_template.jinja`
- `/home/zni/dg_models/diffusiongemma-26B-A4B-it/generation_config.json`
- `/home/zni/dg_models/diffusiongemma-26B-A4B-it/README.md`
- `models/experimental/diffusion_gemma/doc/optimize_perf/run_upfront_gpqa.sh`
- `models/experimental/diffusion_gemma/tt/{generator_vllm,serving,generate,traced_denoise,denoise_loop,denoise_forward,sampling}.py`
- official Transformers `generation_diffusion_gemma.py`
- relevant generation, host-randomness, trace, reveal-mask, and upstream-parity tests

## Report review

- I tested the tempting "first denoise step broke" claim against the artifact:
  it is not observable; only the final 48-step block result is logged.
- I separated originating cause from propagation: cap-commit/KV advancement is
  proven propagation but cannot generate the first bad argmax.
- I checked the repeated token against checkpoint special IDs: it is ordinary
  text `1`, while EOS is token ID 1.
- I checked host seed derivation and trace refresh code: they are deterministic
  and per-step; no stale-buffer evidence appears in this run.
- I challenged deterministic-prefix reasoning: identical 3,072/6,144 prefixes
  prove reproducibility only.
- I ran the corrected server-side thinking contract on the same TT stack; its
  coherent exact-match result localizes this report's collapse to prompt
  construction and removes the need to infer a TT numerical defect.
- I identified the 161-to-192 prompt padding/position shift as a source-proven,
  semantic difference for broader fidelity work, but the passing corrected TT
  run demotes it as an explanation of this incident.
