# Full model — work log

What actually happened building stage 05, in order, including what broke.

## 1. The harness was not in the checkout

`models/common/readiness_check/` does not exist on `main`, and stage 05's
contract is written entirely in terms of it (`run_prefill_check`,
`run_teacher_forcing`, `run_autoregressive`, `contract.Generator`, the
`build_generator` discovery convention), so it had to be vendored from a branch.

*Corrected at stage 05 review.* An earlier revision of this section also said it
does not exist on `origin/agentic-research/hous/multigoal-claude`. **It does** —
all seventeen files — and on eight other remote branches besides. That sentence
was simply false and the choice of branch never rested on it. The real reason to
take `origin/mvasiljevic/fmf/tiiuae-falcon3-7b-base` is concrete: it is the
newest branch carrying the harness, and its `mesh_device.py` is the one whose
`MESH_SHAPES` already contains `"P300X2": (1, 4)` — the label this 4-die host
runs under. `multigoal-claude`'s copy has only `N150`/`N300`/`T3K`/`TG`, so
vendoring from there would have required inventing the mesh label locally.

Vendored with `git checkout <branch> -- models/common/readiness_check/`.
Seventeen files, none of them previously present, so nothing was overwritten.

**Two** edits were then needed to make it run against *this* host and *this*
tokenizer, and both are compatibility fixes rather than model-specific ones:

* `generate.py` called `tokenizer.apply_chat_template(..., tokenize=True)` and
  iterated the result as a list of ints. On the transformers/tokenizers version
  in `python_env` that returns a `BatchEncoding`, so the loop hit
  `int('input_ids')`. Rendering to text and tokenizing explicitly with
  `add_special_tokens=False` works on both.
* `mesh_device.py` had no way to set `trace_region_size`, and TTNN's default is
  **0** (`ttnn._ttnn.device.DEFAULT_TRACE_REGION_SIZE`). Since the
  teacher-forcing runner *requires* traced decode, no traced model could ever
  have run through it on this checkout. Added `--trace-region-size` and threaded
  it through all three runners.

There was no third edit. An earlier revision listed "the same `--mesh-device
P300X2` label already mapped to `(1, 4)`, so no topology change was needed" as a
bullet in this list, which reads as a change and is not one: `MESH_SHAPES` in the
working tree is byte-identical to the falcon branch's. It is a *reason for
choosing that branch* and has been moved above, where it belongs.

## 2. The rotary op had to change, for a correctness reason

The stage-04 decode layer calls
`ttnn.experimental.rotary_embedding(q, cos, sin, token_index)` where
`token_index` is a **Python int**. A captured decode trace bakes the position
in, so every replayed token would be rotated at the position the trace was
captured at. That is not a performance question; it is a wrong model.

Stage 04's own investigation of `rotary_embedding_llama` is the obvious place to
look and it is a dead end — but for the *channel-order* reason, not the one
stage 04 stated. It needs Meta channel order, which the KV cache and prefill do
not have (that README's limitation 4: PCC 0.1933 against a prefill-primed cache
against 0.99997 fresh, plus a bfloat8_b requantisation `max|diff|` of
3.125e-01).

**Correction to stage 04, made here.** Stage 04 wrote "neither spelling can
advance the rotary position inside a replayed trace" and used it to justify
hoisting its cos/sin gather onto the first eager call. That is false for
`rotary_embedding_llama`: the nanobind signature
(`rotary_embedding_llama_nanobind.cpp:38-44`) takes `input_tensor`, `cos_cache`,
`sin_cache`, `trans_mat` and `is_decode_mode` — tensors only, no position — and
`models/tt_transformers/tt/rope.py:571,739` builds exactly that form. The op is
trace-replayable; stage 04's *wiring* of it was not. It had propagated into four
places and all four are corrected: `doc/optimized_multichip_decoder/README.md`,
that stage's `work_log.md`, `tt/multichip_decoder.py`'s docstring and
`probes/rope_hf_probe.py`. It changes no decision — stage 04 was right to reject
the lever and stage 05 is right to use `rotary_embedding_hf` — only the reason
given.

The op actually needed is **`ttnn.experimental.rotary_embedding_hf`** with
`is_decode_mode=True`. It is the same HF `rotate_half` convention — so no weight
permutation, no KV-cache convention change, prefill untouched — but it takes
cos/sin as **tensors** rather than a position as a compile-time int, so the
position can be a device tensor gathered with `ttnn.embedding` and advanced
inside the trace with `ttnn.plus_one`.

`probes/rope_hf_probe.py`, at the shipped per-die decode shapes
`[1, 1, 8, 128]` (Q) and `[1, 1, 1, 128]` (K), 13 positions from 0 to **262143**
— the last position of the advertised context — and, separately, batch 1/4/8/32
with **a distinct position per user**:

| | result |
|---|---|
| `max\|diff\|` vs the shipped op, all 34 cases | **0.000e+00** |
| PCC, all 34 cases | **1.0** |
| `rotary_embedding` (int position), trace slope | 5.43 µs |
| `rotary_embedding_hf` + the reshard it needs, trace slope | **3.75 µs** |
| cos/sin device gather at `rope_cache_len` 8192 | 52.43 µs |
| cos/sin device gather at `rope_cache_len` 262144 | 806.68 µs |

Bit-identical, and slightly *cheaper* per call. The gather is paid **once per
decode step**, not once per layer, so at the shipped 8192-row tables it is
~1 µs/layer.

**Extended at stage-05 review.** The first version of this probe ran `BATCH = 1`
only and stopped at position 4095. Both were gaps: per-user distinct decode
positions are the entire reason `rotary_embedding_hf` is needed — the shipped
op's `token_index` is one Python int for the whole tensor, so a mixed-position
batch is not expressible in it — and 4095 is 1.6% of a 262144-token contract.
The reference for the per-user rows is therefore the shipped op run **once per
user** at that user's own position, stitched back together.

Extending it also surfaced something the 8192-only run could not: the gather
cost scales with table length, because `ttnn.embedding` indexes the whole table.
At the full context the once-per-token gather is 806.68 µs against 52.43 —
0.81 ms, 3.6% of token-out decode. That is the reason `rope_cache_len` defaults
to 8192 and grows only on demand, and it is now a named limitation.

`probes/rope_hf_probe.log` is that run.

Three API details cost a cycle each and are recorded so they are not
rediscovered:

* `rotary_embedding_hf`'s `is_decode_mode` is **keyword-only**; passing it
  positionally raises a nanobind signature error.
* the first spelling of `rope_decode_tables` did
  `trimmed = transposed[:, :batch, :, :]` and then deallocated `transposed`.
  At `batch == 1` the slice is a *new Python object aliasing the same buffer*,
  so the `trimmed is not transposed` guard passed and the deallocate freed the
  tensor the next op read — `TT_FATAL: Input Tensor is not allocated`. Fixed by
  comparing shapes instead of identities.

## 3. Three failures between the first end-to-end call and a working one

All three were caught by `probes/smoke_probe.py --layers 2`, which loads in
about ten seconds. None of them would have been cheaper to find at 48 layers.

1. **`ttnn.sampling` compares logical shapes.** `input_indices_tensor
   .logical_shape() == input_values_tensor.logical_shape()` failed because the
   decode logits were logically `batch` rows padded to a tile, while the sampler
   works in 32 fixed user slots. The rows are already physically present — decode
   *is* one 32-row tile — so `decode_terminal` now pads the logical shape up to
   32 before the LM head. No extra data movement, only a shape.
2. **The warm-up must contain every op the trace will.** `_warm_decode_graphs`
   originally ran with `advance_position=False`, so the two `ttnn.plus_one`
   calls were not in the program cache and capture died on
   `Cannot load new binaries during trace capture`. The warm-up now advances the
   positions too and the restore afterwards puts them back.
3. A plain typo: `reset()` called `_copy_host(..., ttnn.uint32)` positionally
   against a keyword-only `dtype`.

## 4. What the reduced probe proved before the 48-layer run

At two layers (`probes/smoke_probe.py --layers 2`):

* device-traced split sampling and the host-sampling compatibility mode produce
  the **same four tokens**, which is the statement that split greedy is
  semantically greedy and not merely close;
* one steady-state replay moves exactly one counter — `replays` — and nothing
  else. No token, position, rotary, page-table or sampling-parameter host copy,
  no synchronisation;
* the runtime fallback audit is clean at the layer level (`dram_sharded_taken`,
  `in0_block_w` 16/12, expert intermediates in L1, `norm_shard_feeds_qkv_directly`)
  and at the wrapper level.

## 5. The 48-layer model, first run

`probes/smoke_probe.py --layers 48 --tokens 32 --context 8192`:

* weight load 203.0 s (48 layers streamed one at a time out of the sharded
  safetensors; the checkpoint is never fully materialised on the host);
* prompt: the chat-templated *"Write a Python function that reverses a string."*,
  18 tokens;
* output, verbatim:

  > `Here are a few different ways to write a Python function that reverses a string:\n\n## Method 1: Using String Slicing (Most Pythonic)\n`` ``` ``

* host-compat and device-traced paths agree on the first four tokens;
* steady-state replay again moves only `replays`.

That is a coherent, on-task, correctly-formatted answer from the first
end-to-end 48-layer run.

## 6. The sampler was 34.6% of decode, and both fixes were forced

The first end-to-end performance measurement (`probes/perf_full_model.py`, 48
layers, prompt 128 / generate 128) read:

| | ms | t/s/u |
|---|---|---|
| model trace (logits only) | 20.214 | 49.47 |
| token-out | 31.826 | 31.42 |
| token-out + readback | 32.492 | 30.78 |

The model trace was already **at the layer-stack lower bound** — 48 × the
stage-04 layer's 0.4286 ms is 20.57 — so the 11.6 ms gap to token-out was
entirely terminal work, and the same run priced it: the split sampler alone was
**11.005 ms** against force-argmax's **1.862 ms** on the same logits, both
returning token 16.

That is exactly the condition the full-model contract says must be fixed at the
LM-head/sampling contract before anything else is tuned, so it was, on a probe
that needs no model at all: `probes/sampler_probe.py` builds a
`[1, 1, 32, 37984]`-per-die logits tensor — the column-parallel LM head's real
output shape — and sweeps `Sampling1D` over it. Seconds per leg instead of four
minutes.

| leg | ms |
|---|---|
| `split_k32_padded` (the naive configuration) | 11.006 |
| `split_k32_unpadded` | **6.151** |
| `split_k16_padded` | 11.104 |
| `split_k16_unpadded` | 6.268 |

Two conclusions, one adopted lever each:

1. **`pad_to_power_of_2=False`.** The module's own comment calls the pad "a big
   device-perf win for non-power-of-2 vocab on the multi-device path". At a
   37984-wide shard the pad is to 65536, a 1.73× blow-up of the tensor
   `ttnn.topk` then scans, and it costs **1.79×**.
2. **Greedy routes to the force-argmax strategy**, 3.3× faster than the top-k
   path at the same token — and 5.5× after §10's watcher workaround unpinned
   the gather's worker count. Still `Sampling1D`, still traced, still `tt_out_tok`.
   Any `top_k > 1` or `top_p > 0` releases the traces and recaptures on the
   split path.

**What did not work, with the number, and what it cost.** Shrinking `max_top_k`
is not a lever. The split path all-gathers a `[1, 1, 32, max_top_k]` block; 32 is
one tile wide and below it `ttnn.all_gather` announces *"Using slower composite
all_gather: gather dim 3 is padded from 16 to 32"*. At 16 that is merely worse —
6.268 against 6.151 — despite gathering half the candidates. At **8 the
composite gather never returned**: the leg held the mesh for over twenty minutes
before being killed, and the mesh then needed `tt-smi -r` plus a ring-fabric
all-gather smoke test before anything else would run. The `k8` legs are removed
from the probe with that recorded in the source, rather than left in as a trap.

Re-measured with both fixes:

| | before | after | after the watcher workaround (§10) |
|---|---|---|---|
| model trace | 20.214 | 20.225 | 20.211 |
| **token-out** | 31.826 | **22.678** | **22.079** |
| token-out + readback | 32.492 | 23.347 | 22.748 |
| split sampler, eager | 11.005 | 6.155 | 6.155 |
| force-argmax sampler, eager | 1.862 | 1.859 | **1.125** |

**1.40× on token-out**, and the model trace is unmoved, which is the check that
nothing else was disturbed. The third column is the later watcher workaround
(§10), which turned out to make the greedy sampler 1.65× faster as well;
token-out against the same pre-fix figure is now 1.44×.

**Recorded at review: the "before" column above is unarchived.** Only the
"after" column is on disk, as `probes/perf_full_model.csv`. The before figures
(31.826 / 32.492 / 20.214 / 11.005) were read off a `perf_full_model.json`
produced by a code state that no longer exists, and `perf_full_model.py`
rewrites that file in place, so the post-fix run overwrote it. The 1.40×
headline therefore rests on an unarchived measurement. The sampler sweep behind
the *decision* is archived (`probes/sampler_probe.log`, 11.006 → 6.151), and so
is the post-fix in-model pair (6.155 against 1.859), but the end-to-end before
number is not, and the documents now say so instead of citing a file state that
is gone.

## 7. A one-token prompt segfaulted, and the cause does not raise

`test_non_aligned_prompt_lengths[1]` took SIGSEGV with the Python traceback
pointing at `prefill_norm`. Every component was probed individually at S=1 and
every one was fine: the embedding, the RoPE slice, the input norm,
`attention_prefill` against a real paged cache, `router_forward_multichip`,
`moe_prefill_optimized`, `all_reduce`, and `ttnn.rms_norm` on a one-row sliced
tensor with the shipped compute config. Six probes, no reproduction.

The defect is in the composition. `prefill_forward` retains the last prompt row
with `ttnn.slice` and then deallocates the chunk it came from. At every prompt
length but one that is a genuine copy. At `prompt_len == 1` the requested slice
covers the entire tensor, `ttnn.slice` hands back a **view**, and the deallocate
frees the buffer the retained row points at.

The first fix guarded with `piece is hidden` and **did not work**, which is the
interesting part: `probes/prompt_len_1_repro.py` prints

```
seq_len=1 row=0: piece is hidden -> False, same buffer address -> True
```

— a different Python object over the same buffer. The identity guard was
replaced with a shape test and a `ttnn.clone`. The repro runs both ways and is
archived at `probes/prompt_len_1_repro.log`.

## 8. The test module opens its own mesh

`tests/test_full_model.py` first used the repository `mesh_device` fixture and
every test errored with `ScopeMismatch`: that fixture is **function-scoped**, so
a module-scoped generator cannot depend on it. Function scope would mean
reopening the mesh and reloading the model for every test — ten seconds at two
layers, over three minutes at forty-eight — which makes the all-layer tier
unrunnable. `conftest.py`'s `use_module_device` marker does not help; it is
single-device and its own docstring says to avoid it for `mesh_device`. The
module therefore opens its own ring-fabric mesh in a module-scoped fixture,
which is what every probe in `probes/` already does.

## 9. What the gates said

| gate | result |
|---|---|
| `run_prefill_check` | top-1 0.980, **top-5 1.000, top-100 1.000** |
| `run_teacher_forcing` | top-1 0.990, **top-5 1.000, top-100 1.000** |
| `run_autoregressive` + `check_degenerate_output` | 128 tokens each side, **"No degenerate output detected"** |
| `tests/test_full_model.py`, 2 layers | 33 passed |
| `tests/test_full_model.py`, 48 layers | 33 passed |
| the shared qualitative prompt suite, 6 prompts x greedy+sampled | read and scored, `qualitative_check.log` |
| the watcher A/B | localized to an upstream `all_gather_async` parameter pair, `watcher_ab.log` |
| whole tree **under the watcher** | **145 passed, 0 tripped asserts**, `pytest_watcher_clean.log.gz` |
| stage-04 suite on this tree | **112 passed, 0 failed** — unchanged |
| `footprint_probe.py --context 262144` | 11.759 GB/die, 22.119 free, **no capability reduction** |

The teacher-forcing run was repeated after the sampler change, because greedy
now takes a different strategy inside `Sampling1D`; the top-k figures are
identical to the pre-change run and the decode rate moved 21.25 → 38.50 t/s/u.
**21.25 is unarchived**, for the same reason 31.826 is: the runner rewrites its
log in place, so the pre-change run's log no longer exists. 38.50 is in
`run_teacher_forcing.log`. Of the six pre-fix figures this stage quotes, only
the two sampler-sweep legs (11.006 and 11.104, both rows of
`probes/sampler_probe.log`) are backed by a file; **31.826, 32.492, 20.214 and
21.25 are not**, and are quoted with that said rather than implied.

## 10. A watcher assert: localized to an upstream op, then worked around locally

Running the full-model module under `TT_METAL_WATCHER=10
TT_METAL_WATCHER_DISABLE_ETH=1` **used to** abort on

```
Device 0 worker core(x= 0,y= 0) virtual(x= 1,y= 2): BRISC tripped an assert on line 119.
Current kernel: .../all_gather_async/device/kernels/minimal_default_writer.cpp
```

### What was written here first, and why it was wrong

The first version of this section said the assert "could not be localized below
the full-model path", that two isolation runs were inconclusive, and that the
second "never finished the two-layer build inside ten minutes". The review
caught that the log contradicts it: `pytest_full_model_watcher_FAILS.log` is a
**two-layer** run whose session starts at 05:06:25, reaches
`test_split_sampling_feeds_its_own_token_back_on_device` at 05:06:47 — 22
seconds, not ten minutes — and aborts inside it at 05:06:57. The assert was
already localized to the sampling test by the artifact that was in the tree.

It also framed the watcher-only nature as reassurance. It is the opposite: a
device `ASSERT` compiles out without the watcher, so a passing non-watcher run
means the invariant is **unchecked**, not satisfied. This one fires on the
shipped, traced, every-token greedy path.

### The A/B, properly run

Two new probes, both device-cheap because neither needs the model:
`probes/sampler_watcher_ab.py` drives `Sampling1D` over synthetic
`[1, 1, 32, 37984]`-per-die logits; `probes/ccl_watcher_ab.py` drops
`Sampling1D` and calls `ttnn.experimental.all_gather_async` directly.
`probes/run_watcher_ab.sh` runs every leg in its own process — the watcher
aborts on the first trip, so legs cannot share one — and writes
`watcher_ab.log` plus a per-leg log under `watcher_ab/`. Legs reproduce or stay
clean in **seconds**, which is what the ten-minute budget story had ruled out.

Findings, in the order they came:

1. The shipped force-argmax path trips on its **first eager call**, with zero
   layers. Reproduced.
2. **The barrier semaphore is not it.** The identical gather with
   `barrier_semaphore=` supplied trips exactly the same way, and returns the
   same token when it is allowed to finish without the watcher. That kills the
   README's leading hypothesis outright.
3. The split top-k path is clean, so it is not "the sampler".
4. Stopping the argmax path after the gather, after the vocab slice and after
   the untilize all trip identically — it is the gather.
5. But the *raw* gather with byte-identical arguments was **clean**, even with
   the sampler's buffers resident. The arguments therefore were not
   byte-identical, and they were not: `default_topology(mesh)` returns
   **`Topology.Linear`** on this 1x4 Blackhole mesh, so
   `_argmax_all_gather`'s Ring/no-barrier branch — the one whose comment about
   "trace-capture issues seen with some barrier-based configurations" this
   project had been reading as the relevant upstream workaround — **never runs
   here at all**. The fallback does: `Linear`, `cluster_axis=1`, a barrier
   semaphore, `chunks_per_sync=10`, `num_workers_per_link=1`.
6. Bisecting that fallback one knob at a time gives a two-parameter minimal
   trigger: **`Topology::Linear` + `num_workers_per_link=1`**. Ring with
   `num_workers_per_link=1` is clean; Linear with the default worker count is
   clean; Linear with `chunks_per_sync=10` is clean; Linear with a barrier is
   clean. Both together trip, at the sampler's 37984-wide shape and equally at
   the decoder layer's 512-wide one — so it is the op's parameters, not this
   model's tensors, and that is also why the layer is clean: it never passes
   `num_workers_per_link` and takes the default.

`Sampling1D._get_argmax_all_gather_config` forces `Linear` for any mesh under 8
devices and the fallback call hardcodes `num_workers_per_link=1`, so **every**
sub-T3K force-argmax sampling path in this repository lands on the combination.

Handing it upstream as two reports: the op bug against
`all_gather_async/.../minimal_default_writer.cpp`, with
`probes/ccl_watcher_ab.py --leg linear_workers1` as a model-free reproducer; and
the caller bug against `models/common/modules/sampling/sampling_1d.py:294-346`,
whose ring branch is unreachable on the meshes it was written for.

### Working around it, rather than shipping around it

Localizing it to two parameters made the fix obvious, because the same matrix
that names the tripping pair also names four clean spellings — including the one
the decoder layer has been using for four stages. The layer never passes
`num_workers_per_link`; it lets the op default. That is the whole reason it has
been clean.

`tt/model.py`'s `_WatcherCleanSampling1D` subclasses `Sampling1D` and overrides
`_argmax_all_gather` with the layer's spelling: same op, same `dim`, same
semaphores, `Topology.Ring`, a barrier semaphore, no tuning knobs pinned. The
seam works without touching shared code because `from_config` builds through
`object.__new__(cls)` and `_bind_strategy` binds `_pre_argmax_gather` by
attribute lookup on the instance, so the subclass's override is what gets bound.
`models/common/modules/sampling/sampling_1d.py` is **unmodified**.

Verified, not assumed: the `argmax_shipped` leg of `probes/sampler_watcher_ab.py`
runs the class the model actually instantiates and is clean, at the same sampled
token as before; and the whole tree under the watcher —
`pytest tests/ -m "not models_performance_bare_metal" -q` with
`TT_METAL_WATCHER=10 TT_METAL_WATCHER_DISABLE_ETH=1` — is **145 passed, zero
tripped asserts** (`pytest_watcher_clean.log.gz`). "Watcher-clean" is a standard
this project has met in every previous stage, and stage 05 now meets it.

The two upstream reports keep their value and should still be filed; the
subclass is a local workaround with a deletion condition (fix the op, delete the
class), not a substitute for fixing the op.
