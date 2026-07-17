# AutoDebug: non-greedy split-sampling RNG state

## Scope and verdict

Inspection only; no TT device was opened and no implementation file was edited.

**Headline (high confidence): the captured Gemma non-greedy sampler resets each
user core to the same seed on every replay.** This is a real stochastic-sampling
bug, not only a missing test. It does not imply the same token for changing
model logits, but it makes every decode step reuse the same random quantile
instead of advancing a PRNG stream. Constant candidate distributions will
therefore select the same token on every replay.

The separate token-feedback and changed-page-table concerns are evidence gaps,
not source-visible implementation defects.

## Starting evidence

- `Gemma4Generator` routes semantic greedy to `Gemma4GreedyTP4Sampler`, but
  routes every other `top_k`/`top_p`/temperature configuration through
  `Sampling1D` in the sampler trace
  (`tt/generator.py:561-585`).
- Gemma constructs `Sampling1DConfig` without supplying seeds
  (`tt/generator.py:319-332`). `Sampling1D` consequently materializes the
  persistent default seed tensor as `torch.arange(B)`
  (`models/common/modules/sampling/sampling_1d.py:728-740`). For batch one the
  unchanged seed is zero.
- No Gemma generator path updates `sampler._seeds`; a repository-wide search
  under the Gemma autoport finds no seed manager or `UINT32_MAX` transition.
- The Stage 06 model tests use only semantic greedy parameters (`k=1`, `p=0`,
  `temp=1`) and the custom greedy trace. There is no Gemma non-greedy
  multi-replay test. The generic `Sampling1D` trace test also uses `k=1`, runs
  only one replay, and explicitly expects the same token as eager execution
  (`models/common/tests/modules/sampling/test_sampling_1d.py:1470-1538`).

## Finding NG-1: fixed seed replay reinitializes the RNG every token

### Causal chain

1. `Gemma4Generator.prepare_token_out_decode()` allocates only `k`, `p`, and
   `temp` for a non-greedy trace (`tt/generator.py:789-797`). It neither accepts
   a request seed nor advances the common sampler's seed tensor.
2. `_capture_sampling_trace()` captures
   `self.sampler.decode_forward(..., tt_out_tok=state.token_input)`
   (`tt/generator.py:573-590`). Each later `_execute_sampling_trace()` replays
   that full graph unchanged (`tt/generator.py:643-647`).
3. `Sampling1D._sample_topk()` puts `ttnn.manual_seed()` immediately before
   `ttnn.sampling()` on every invocation, using the persistent config seed when
   no override is passed
   (`models/common/modules/sampling/sampling_1d.py:356-404`). Both operations are
   therefore part of every Gemma sampler replay.
4. The TTNN manual-seed contract says every non-`UINT32_MAX` seed calls
   `rand_tile_init(seed)`; only `UINT32_MAX` leaves PRNG state unchanged
   (`ttnn/cpp/ttnn/operations/reduction/manual_seed/docs/Manual_seed.md:5-10,
   187-216`). The tensor-seed compute kernel implements exactly that branch
   (`.../manual_seed/device/kernels/compute/manual_seed_receive_all_data.cpp:30-38`).
5. `ttnn.sampling()` receives no scalar seed from `Sampling1D`, so its compiled
   seed is zero and it calls `rand_tile()` using the core state established by
   `manual_seed` (`.../sampling/device/sampling_program_factory.cpp:30-35,
   103-105`; `.../sampling/device/kernels/compute/sampling.cpp:27-49,
   432-437`). Thus the replay sequence is: reset to seed S, consume the first
   random tile; reset to S, consume the first random tile; and so on.

### Controlled comparison

The older common sampler owns the missing state transition explicitly.
`SeedManager.get_new_values()` first pushes a real seed, then pushes
`MAX_UINT32`, and in steady state performs no host copy because `rand_tile()`
advances on device (`models/common/sampling/generator.py:856-906`). Its trace
wrapper also refuses internal traced sampling when an explicit request seed
requires per-token seed updates (`models/common/sampling/generator.py:371-418`).
This is strong contract evidence that repeatedly replaying a fixed real seed is
not intended safe behavior.

The generic `Sampling1D` test named
`test_sampling1d_deterministic_with_same_seed` confirms the callee behavior:
two calls with the same explicit seed must return the same tokens
(`models/common/tests/modules/sampling/test_sampling_1d.py:805-843`). That is a
useful reseeding test, but the exact opposite of the stateful autoregressive
contract Gemma needs.

### Symptom coverage and limits

- Explains deterministic reuse of one random CDF threshold at every non-greedy
  token and identical outputs when candidate probabilities are unchanged.
- Can bias or mechanically correlate a free-running completion even when tokens
  do not visibly repeat, because logits normally change between steps.
- Does not affect Gemma's selected semantic-greedy path; that path never calls
  `Sampling1D._sample_topk()`.
- Runtime manifestation was not reproduced in this inspection-only pass, but
  the reset behavior follows directly from the documented op contract and the
  captured call graph.

### Smallest focused verify/refute experiment

Add an isolated TP4 `Sampling1D` test; do not load Gemma weights or the full
model.

1. Use the Gemma physical contract: `(1,4)` mesh, vocab `262144`, batch one,
   `max_top_k=32`, FP32 gathered candidate values, and non-greedy parameters
   such as `k=32`, `p=1.0`, `temp=1.0`.
2. Construct constant logits with several equal-probability candidates and one
   persistent output tensor. Warm, capture `decode_forward`, then execute the
   same trace 16 times, synchronizing and recording the output after each
   replay.
3. Current-code prediction: all 16 outputs equal the first seeded draw. Also
   record the persistent seed buffer before and after to show it remains the
   same non-sentinel value.
4. A/B against the intended stateful contract: after warmup, call
   `ttnn.manual_seed` once with a fixed real seed, copy `UINT32_MAX` into the
   trace-bound seed tensor, capture/replay, and compare the exact 16-token
   sequence with eager `ttnn.sampling` calls that seed once and then preserve
   state. The sequence should advance and should reproduce after a fresh reset
   to the same request seed.

This experiment directly verifies RNG state; changing model logits is a weaker
test because the same quantile can still produce different tokens.

### Intervention boundary

Fix request/trace seed ownership at the Gemma generator/common-sampler boundary,
not in the LM head, model trace, or sampling kernel.

The smallest compatible design is:

- initialize each active request's sampler core once at the request boundary;
- after all non-greedy prewarm calls, reinitialize to the intended request seed;
- transition the trace-bound persistent seed tensor to `UINT32_MAX` before the
  first sampler replay, so captured `manual_seed` becomes a no-op and
  `rand_tile()` advances entirely on device;
- expose a request seed (with a documented entropy-derived default), fixed-slot
  reset, and reproducibility semantics in the generator;
- keep inactive slots at the skip sentinel.

The ordering after prewarm matters: `_prewarm_split_sampling_workloads()` itself
samples once and would otherwise consume or overwrite the intended stream.
Merely copying `UINT32_MAX` without a preceding request-boundary seed leaves the
stream dependent on stale core state. Incrementing a seed tensor inside the
trace is a larger alternative and does not match the existing common
seed-once/advance-state contract as cleanly.

Do not treat a host per-token seed copy as the final optimized fix; that would
repair randomness while violating the no-per-token-host-work split-trace goal.

## Evidence-gap assessment

### Token feedback: implementation appears correct; focused evidence is missing

The sampler trace writes to `state.token_input`
(`tt/generator.py:806-816`), the model trace was captured reading that exact
tensor (`tt/model.py:1070-1080`), and each step submits the model trace before
the sampler trace (`tt/generator.py:839-841`). The reduced test asserts the
shared tensor identity and zero host token refreshes
(`tests/test_full_model.py:368-380,392-398`). No conflicting token buffer is
visible in the source.

However, tensor identity is not the explicit two-step value/consumer proof
required by the trace contract. Add a focused replay test that records sampled
token N from the persistent buffer, executes replay N+1 without a host write,
and demonstrates that the next model result/cache update corresponds to token
N rather than a stale token. This is a test/evidence gap, not presently a
source-supported defect.

### Page tables: copy policy appears correct; changed-content evidence is missing

The model captures private stable clones, gates refresh by source identity plus
explicit generation, and submits one distributed `ttnn.copy(source, target)`
per unique changed target (`tt/model.py:949-972,1003-1039`). The existing test
proves one changed-generation copy and no repeat copy, but its alternate tables
are equal-content clones (`tests/test_full_model.py:366,400-418`). It therefore
tests lifecycle/copy suppression, not functional remapping.

Add distinct page-table contents and assert that the stable trace table matches
the new mapping after the changed replay, then prove a second unchanged replay
does not copy. Ideally use distinguishable KV blocks and check the model result.
This is also an evidence gap; static inspection found no incorrect copy target,
ordering, or unchanged-table branch.

## Evidence ranking

1. **High:** fixed-seed reset on every non-greedy sampler replay (documented
   manual-seed semantics plus exact captured call graph).
2. **High:** no Gemma seed transition or request-seed owner exists.
3. **Medium:** visible generation quality may be biased/repetitive; severity is
   distribution-dependent and needs the focused hardware experiment.
4. **Evidence gap only:** sampled token N is not value-checked as model input
   N+1, although source wiring is coherent.
5. **Evidence gap only:** changed-page-table contents are not tested, although
   source copy and suppression logic is coherent.

## Final status

`NG-1` is source-verified and requires a fix plus the focused non-greedy
multi-replay test before Stage 06 can claim the shared top-k/top-p path is
complete. Token feedback and page-table findings should be closed with focused
tests; they do not currently justify implementation changes.
