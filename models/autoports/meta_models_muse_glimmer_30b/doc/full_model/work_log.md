# Full model — work log

Chronological. What was tried, what the evidence said, and what changed because of
it. Numbers and verdicts live in [README.md](README.md); this file is the trail.

Target: `meta-models/Muse-Glimmer-30B`, 4 x Blackhole (`ClusterType::P300_X2`),
`ttnn.MeshShape(1, 4)`, `FABRIC_1D_RING`, `ttnn.Topology.Ring`, 2 links.
Starting point: the optimized multichip decoder at `093c65bd2c2`
(`tt/multichip_decoder.py`), unchanged as the layer stack.

---

## 1. What the stage had to add, and the two parallelisation decisions

The decoder's contract is a **replicated** residual stream, so both terminal
weights are column-parallel in the only direction that keeps it replicated:

| tensor | fracture | consequence |
| --- | --- | --- |
| `embed_tokens` | hidden dim, `ShardTensorToMesh(-1)` | one all-gather of the *embedded rows*; the table is 672 MB/device instead of 2.7 GB replicated |
| `lm_head` | vocab dim, `ShardTensorToMesh(-1)` | **no logits gather at all** on the token-out path: the sampler consumes vocab-sharded logits |
| `norm.weight` | replicated | free (one tile row) |

Both were confirmed against a torch reference before any of the 52 layers were
loaded (`bench/terminal_probe.py`, `logs/terminal_probe.log`): the fractured
embedding plus one all-gather is PCC 1.000000000 against the replicated lookup at
batch 1, batch 32 and a 100-row prefill, and the terminal RMSNorm on the decoder's
width-sharded L1 boundary layout is 0.999987 against torch.

Three model-level facts had to be read out of the HF source rather than assumed
(`transformers/models/muse_glimmer/modeling_muse_glimmer.py`):

* `MuseGlimmerTextNormedEmbedding` applies a **weight-less** RMSNorm on top of the
  embedding lookup (`with_scale=False`), eps `rms_norm_eps`;
* `model.language_model.norm` is `MuseGlimmerRMSNorm(with_scale=True)`, which
  multiplies by `w` — **not** by `1 + w` the way the decoder's four centered norms
  do. The checkpoint's `norm.weight` is O(3), so folding a `+1` in would have been
  a ~30 % error on every channel;
* the head is `T * tanh(lm_head(h) * m / T)` with `m = output_multiplier = 0.19612`
  and `T = final_logit_softcapping = 20.0`. `m / T` is folded into the weight at
  setup, so the runtime path is one matmul, one `tanh` and one scalar `mul`.

`lm_head.weight` and `embed_tokens.weight` are **separate tensors** in this
checkpoint (`text_config.tie_word_embeddings` is False and the two differ
elementwise, checked at load), so `_tied_weights_keys` is inactive and the real
head is used. A silent fall back to the embedding would still produce plausible
text, which is why it is checked rather than assumed
(`test_lm_head_is_column_parallel_and_softcapped`).

---

## 2. Shared RoPE tables — the one change to the decoder file

`MultichipDecoder.from_state_dict` gained one optional keyword, `rope_cache`.
Default `None` keeps the single-layer behaviour exactly.

Why: every sliding layer in this checkpoint has the same `layer_rope_theta`
(500000.0; the 13 full-attention layers are NoPE and use no table), and the four
tables are 134 MB per layer at full context. Built per layer that is
**39 x 134 MB = 5.2 GB** of device DRAM holding 39 copies of one tensor — more
than the entire 52-layer weight footprint. `build_rope_cache` checks the
uniform-theta assumption instead of relying on it, and raises if a checkpoint ever
breaks it.

Measured: `per_device_rope_table_bytes = 134,217,728` for the whole 52-layer stack
(`logs/smoke_all_layers.log`), i.e. one shared set.

---

## 3. LM-head contract and geometry — measured, not inherited

`bench/terminal_probe.py` swept both legal matmul contracts at the real 32-row
decode payload (`logs/lm_head_sweep.log`). Two constraints shape the space:

* the DRAM-sharded matmul needs one weight shard per DRAM bank, so
  `dram_sharded_weight_memcfg` pads the per-device width to 256 and the vocab has
  to be padded to `4 x 50688 = 202752`;
* it also needs `K_tiles % cores == 0`, and K is `6656/32 = 208` tiles, so its
  legal core counts are the divisors of 208 that fit an 11x10 grid: 8, 13, 16, 26,
  52, 104. The 1D-mcast contract needs only tile alignment, so its minimum legal
  padding is `4 x 50528 = 202112`.

| contract | dtype | geometry | ms/step | weight/device |
| --- | --- | --- | --- | --- |
| **dram_sharded** | **BFP4** | **cores=52, in0_block_w=2** | **0.6029** | **190 MB** |
| mcast1d | BFP4 | in0_block_w=8 | 0.6765 | 190 MB |
| mcast1d | BFP8 | in0_block_w=8 | 0.9779 | 359 MB |
| dram_sharded | BFP8 | cores=16, in0_block_w=1 | 1.0396 | 359 MB |

Two findings worth naming:

* **core count is worth nothing** on the DRAM-sharded contract (1.0107–1.0147 ms
  across all six legal values at BFP4/`in0_block_w=1`; the BFP8 family at the same
  `in0_block_w` is a separate, disjoint band at 1.0396–1.0426 ms) and **`in0_block_w` is worth
  everything** (1.013 -> 0.603 ms going from 1 to 2). Above those values the op
  fails with an exact L1 blocker, *"Statically allocated circular buffers ... grow
  to 1821824 B which is beyond max L1 size of 1572864 B"* — which is also why BFP8
  cannot take `in0_block_w=2` here and loses to `mcast1d`;
* the two contracts were therefore swept as **coherent families**: each dtype was
  measured on both contracts across each contract's legal geometry, so the BFP4
  win is not an artefact of comparing one dtype's best geometry against another's
  worst.

BFP4 was **not** selected on the synthetic PCC (0.9937 against BFP8's 0.99976 on
i.i.d. Gaussian weights). It was selected on the real-weight accuracy gate; the
BFP8 control on the same gate is in [README.md](README.md).

---

## 4. Two shared-infrastructure gaps, both blocking

Neither is model code, and neither could be worked around from inside the port.

**(a) `models/common/readiness_check/mesh_device.py` had no 4-chip label and no
trace region.** `MESH_SHAPES` covered N150/N300/T3K/TG only, and
`open_readiness_mesh_device` called `ttnn.open_mesh_device(mesh_shape=...)` with
nothing else — so `trace_region_size` and `l1_small_size` both defaulted to 0.
The teacher-forcing runner *requires* `generate(..., enable_trace=True)`, so no
model whose decode must be traced could use these runners at all.

Change: added `"P300_X2": (1, 4)`, and made `open_readiness_mesh_device` accept
`trace_region_size` / `l1_small_size` / `fabric_packet_payload_bytes` as keywords
falling back to `TT_READINESS_*` environment variables and then to metal's own
defaults. Every existing caller passes nothing and behaves exactly as before.

**(b) `models/common/readiness_check/generate.py` mis-handled a chat template that
returns a dict.** `apply_chat_template(tokenize=True)` returns a flat id list for
some tokenizers and a `BatchEncoding` for others; this one returns the latter, and
iterating it yields its *keys*, failing as `int('input_ids')`. Change: unwrap
`input_ids` and a leading batch dimension. Both forms now work.

Two shims stayed inside the port, in `bench/readiness_cli.py`, because they are
genuinely model-specific:

* `AutoModelForCausalLM` does not know `MuseGlimmerConfig` — the checkpoint
  declares `MuseGlimmerForConditionalGeneration`, the multimodal wrapper, which
  *is* a `GenerationMixin` whose `forward` returns `.logits`. Note that
  `AutoModelForCausalLM.register` cannot fix this: `_LazyAutoMapping.register`
  silently returns for any config whose module starts with `transformers.`
  (`auto_factory.py:680`, a guard against remote code hijacking a native config),
  so the pair has to go into `_extra_content` directly;
* `refs/main` for this repo points at a **metadata-only** revision — config,
  tokenizer and the weight index, but no shards — so
  `from_pretrained("meta-models/Muse-Glimmer-30B")` resolves to a snapshot with no
  weights. Both the port and the shim now resolve by requiring every shard the
  index names to be present, which is strictly stronger than the
  "first snapshot with an index" rule `tests/reference.py` uses.

### An environment mistake, recorded

The first attempt at (b) passed `local_files_only=False`, which tried the network,
failed, and **created** a third metadata-only snapshot in the HF cache
(`a4e59da5…`) plus a *different* `chat_template.jinja` blob. That snapshot sorted
before the real one, so the "first snapshot with an index" rule would have picked
it — breaking `tests/reference.py` for every earlier stage. It was removed and
`refs/main` was pointed at the complete revision (`f84ecc3a…`); the resolution rule
in `tt/model.py` was then hardened so the same mistake cannot break it again.

---

## 5. The AIME24 chat reference

Generated fresh; no earlier artifact matched, and none existed.

```
python doc/full_model/bench/readiness_cli.py \
  models.common.readiness_check.generate \
  --hf-model meta-models/Muse-Glimmer-30B \
  --prompt-source aime24 --chat-template --gen-len 100 --top-k 100 \
  --output models/autoports/meta_models_muse_glimmer_30b/readiness_aime24_chat.refpt
```

204 chat-template prompt tokens, 100 continuation tokens, top-100 per position;
HF on CPU in bfloat16 (the checkpoint's own storage dtype — FP32 would be 112 GB
of host RAM and twice the traffic per token). `readiness_aime24_chat.metadata.json`
records the model id, revision, dtype, tokenizer, prompt source and index,
chat-template flag, gen length, top-k, the exact command and the artifact's
SHA-256, so a later stage can prove a match instead of regenerating.

One property worth flagging: the rendered system message embeds the **current
date** ("Current date: 2026-08-13"), so re-rendering the same AIME prompt on
another day gives different prompt text. The reference stores its own
`prompt_tokens`, so the TT run is unaffected — but a stage that re-renders rather
than reusing the reference will not reproduce it.

---

## 6. Bug found and fixed: non-aligned prompts were not reproducible

Found by the reduced probe, isolated by `bench/prefill_repeat_probe.py`.

```
Observed anomaly: two identical generate() calls on the same 37-token prompt
                  returned different tokens.
Evidence:         logs/determinism_probe.log, logs/prefill_repeat_probe.log
Affected path:    prefill, every non-tile-aligned prompt length
Control:          the same probe at prompt lengths 64 and 128 -- bit-identical
                  (top1/top2 gap 0.375 exactly, three runs); at 37 the gaps were
                  0.25 / 0.375 / 0.75 across three identical prefills
Likely subsystem: the embedded prompt's tile padding
Investigation:    four arms. (A) prefill alone, repeated -> unstable at 37,
                  stable at 64/128. (C) read the paged cache back: reset() does
                  zero it (0 non-zero entries). (B) prefill after a decode plus
                  reset -> stable. (D) prefill after a decode with *no* reset ->
                  also stable, so the cache is not the input that varies.
Resolution:       fixed.
```

Root cause: `ttnn.embedding` writes only the rows its input asks for, so embedding
a 37-token prompt leaves rows 37..63 of the tile-padded output holding whatever
was in that DRAM page. Those are real query rows to the prefill attention, their
K/V lands in the cache, and because the paged SDPA reads a **rounded** window
rather than exactly `seq_len` keys, they perturb the logits of the real rows. The
earlier decoder stages never saw this: their tests fed `hidden_states` built by
`ttnn.from_torch`, which zeroes tile padding.

Fix: one extra embedding row of zeros at index `vocab_size`
(`EMBED_PAD_ROWS`), and the generator pads the prompt's **token ids** up to a tile
boundary with that id. RMSNorm of a zero row is zero (`0 * (0 + eps) ** -0.5`), so
the padded rows are exactly the zeros every earlier stage validated its
non-aligned prefill PCC against. Cost: one 3.3 KB table row per device.

The generator now owns the whole padding contract, as `$full-model` requires: it
pads the ids, the layer stack sees a tile-aligned prompt (its own internal
`ttnn.pad` becomes a no-op), the junk-free K/V past the logical length is never
read because decode starts at `cur_pos = prompt_len`, and the logits are sliced
back to the logical last position. Pinned by `test_prefill_is_reproducible`.

That test was originally written for this bug alone, at 37 tokens, and that narrowness
is how a *second* reproducibility defect in the same path hid behind it for the rest of
the stage — see Section 15.

---

## 7. One decode trace, not two — a sampling-trace identity constraint

The first implementation captured two decode traces, one that advanced
position/RoPE on device (free-running) and one that did not (teacher forcing and
the caller-driven low-level path). The second trace then raised

```
ValueError: The provided logits tensor does not match the tensor used during
trace capture. Call `reset_trace()` before tracing with new tensors.
```

`SamplingGenerator` validates its trace by **tensor identity** and keys its slot
on `(penalties, log_probs, force_argmax)` — not on which logits tensor it was
captured over. Two decode traces therefore need two samplers, and `TTPenalties`
allocates ~45 MB/device of state whether or not penalties are used.

Resolution: one decode trace, always advancing on device. The in-trace
`ttnn.plus_one` runs *after* every read of the position tensors, so a caller that
restages them from the host simply overwrites the increment — correct in both
modes, and it costs two ops on ≤32-element tensors in the host-stepped path. Half
the trace memory and one fewer class of key-mismatch bug.

---

## 8. Sampling implementation: chosen and rejected

Both common implementations were read against this model's requirements before
either was wired in.

**Chosen: `models/common/sampling/` — `SamplingGenerator` + `TTSampling`.**

**Rejected: `models/common/modules/sampling/sampling_1d.py` — `Sampling1D`.**

The reasons, in the order they mattered:

1. **`Sampling1D` has no trace code at all.** `grep -n trace sampling_1d.py`
   returns one comment. Its only in-tree usage pattern is *inlined* into the
   caller's decode trace (`models/common/models/executor.py:2310`), which is not
   the split-sampling contract this stage owes: a model trace to sampler-ready
   logits plus a separately captured sampling trace. `SamplingGenerator` ships
   exactly that (`capture_trace` / `_execute_trace` / `reset_trace`, with
   identity-validated inputs).
2. **Stochastic sampling is broken in `Sampling1D`.** It re-seeds
   `ttnn.manual_seed` with a constant `arange(32)` on every call
   (`sampling_1d.py:404-410, 837-842`) and nothing ever advances it, so the device
   PRNG resets to the same state every step. `SeedManager`'s init -> SKIP -> steady
   state machine is what makes top-k/top-p produce a real stream — and its steady
   state does **no host copy**, which is why greedy decode stages nothing per
   token.
3. **Greedy determinism.** `TTSampling` asks `ttnn.topk` for the stable network
   where it exists (Blackhole qualifies) and then runs `_adjust_values_for_tiebreak`,
   which pins the greedy pick to the lowest global id among tied maxima.
   `Sampling1D` has neither, and at 202048 bf16 logits exact ties are common.
4. One API serves greedy and sampled: `format_sampling_params` rewrites
   `temperature=0` to `(temp=1, k=1, p=0)`, which is argmax expressed through the
   top-k op.

`tt_ccl=None` is passed deliberately. On the top-k path neither implementation
touches a semaphore — the two fixed-shape candidate all-gathers go through
`ttnn.all_gather`, which owns its own. A `TT_CCL` would put 36 global semaphores in
the **main** L1 pool, and the decoder's decode step has 7,296 B of headroom there;
twelve of them already broke its sharded norm once
(`doc/optimized_multichip_decoder/README.md`). Cost accepted: `TTPenalties`
allocates ~45 MB/device of unused penalty state.

Force-argmax is off, and that is a contract decision rather than a default: its
full-vocab all-gather (12.9 MB/step across this mesh) goes through
`self.tt_ccl.get_and_cycle_ag_semaphore_handles(...)`, and this port passes
`tt_ccl=None` deliberately — a `TT_CCL` puts 36 more global semaphores in the main L1
pool against 7,296 B of decode headroom (Section 9). With `tt_ccl=None` the arm does
not error, it **hangs**. The measured comparison is in [README.md](README.md).

> This paragraph originally gave a different reason — that `ttnn.argmax` returns rank-3
> `[1,1,32]` which "cannot be fed back into the rank-4 `[1,1,1,32]` token buffer in
> place". That was **wrong** and is withdrawn: upstream passes `output_tensor=tt_out_tok`
> straight into `ttnn.argmax`, so the op is built to write into a caller's persistent
> buffer, and the rank mismatch is a property of the buffer shape this port chose. The
> retraction was made in Section 14, in `README.md` and in `tt/generator.py` first and
> missed here for a round; a later stage reading only this section would have concluded
> force-argmax is structurally impossible and never tried `output_tensor=`.

---

## 9. L1_SMALL budget, recomputed for a stack

The decoder stage bounds this region hard (6144 B; both larger and smaller fail).
The full model's steady-state occupancy:

| user | semaphores | bytes |
| --- | --- | --- |
| decoder async prefill collectives (`_CCL_SEMAPHORES`) | 7 | 1,792 |
| model-level async all-gather (`_MODEL_CCL_SEMAPHORES`) | 3 | 768 |
| decode wrapper `reduce_scatter` + `all_gather` @ `[1,1,32,6656]` | 2 programs | 512 |
| sampler's two candidate all-gathers | 2 programs | 512 |
| **total** | | **3,584 of 6,144** |

The model's own gather uses the *async* primitive with three semaphores created
once per mesh, rather than `ttnn.all_gather`, precisely because the wrapper leaves
one semaphore per **program** in this region for the life of the program cache —
and the embedding gather has a distinct program per prompt length. A test session
that prefilled a dozen lengths would exhaust the region.

---

## 10. Device recovery

One hang, and it was self-inflicted. An earlier probe run was killed with
`pkill` while it held the mesh, and **no reset followed**; the next run hung inside
`TilizeWithValPaddingDeviceOperation` on the embedding output.
`tools/tt-triage.py` captured it (`triage/tt-triage.txt`,
`triage/triage-summary.txt`): the hung op and its predecessor
(`EmbeddingsDeviceOperation`) are named in `dump_running_operations`, `brisc` is
parked in `cb_wait_front`, and `check_noc_status` shows six ethernet cores with
undrained NOC transactions. Everything else passed, including `check_arc` and
`check_eth_status`.

`tt-smi` is not installed here and there is no network to install it, so recovery
went through the driver's own ioctl — `TENSTORRENT_IOCTL_RESET_DEVICE` with the
Blackhole `ASIC_RESET` then `POST_RESET` flag pair, which is what `tt-smi -r`
does. `bench/tt_reset.py` is that, kept as a tool. All four devices returned
`result=0`, and the mesh smoke (open 1x4 `FABRIC_1D_RING`, one `all_gather`, close)
passed immediately afterwards.

Recorded as infrastructure recovery, not a model result. The lesson is in the
first sentence: never `pkill` a device-holding process without resetting after it.

---

## 11. Two things the readiness discovery mechanism forces

Both found by running the stock runners rather than by reading them.

**`tt/generator.py` cannot use relative imports.** The runners load it with
`importlib.util.spec_from_file_location` under a synthetic module name and **no
package**, so `from .model import ...` raises *"attempted relative import with no
known parent package"* before the generator is ever constructed. It uses absolute
`models.autoports.meta_models_muse_glimmer_30b.tt.…` imports instead.
`tt/model.py` keeps relative imports: it is only ever imported as a package
module.

**The generator cache has to live in `tt/model.py`, and its key has to be
resolved.** Because the runners load `tt/generator.py` by path, a module-level dict
*there* is a second, unrelated copy — so the cache lives in `tt.model`, which is
imported normally and therefore exists once.

The key then has to be *resolved* before it is compared. The runners call
`build_generator(model_dir, mesh_device)` with no knobs, so their config carries
`max_seq_len=None` ("the HF-advertised context") while a driver that spells the
same number out carries `131072`. Keying on the unresolved value made those look
like different models: the first run of the driver rebuilt the whole 52-layer stack
inside `run_prefill_check`, which is not merely slow — it is another 7.18 GB/device
of DRAM per redundant copy, and four runners would have exhausted the part.
`_resolve_max_seq_len` fixes it, and both spellings now produce the same key.

## 12. Evidence runs

`bench/evidence.py` drives every stage over **one** build. The runners themselves
are the stock ones, called through their programmatic entry points, unmodified.

Commands, logs and results: [README.md](README.md).

---

## 13. The top-100 bar, and four controls — **superseded, kept as a record of a wrong diagnosis**

> **Read this section as history, not as findings.** Its conclusion — that prefill
> top-100 was stuck at 0.980 because of accumulated numerics in the 52-layer
> BFP8/BFP4 stack, and that doing better was physically impossible because BF16 weights
> would need 50 GB/device — is **wrong**. Every measurement below was taken through the
> nondeterministic prefill of Section 15. Once the embedding all-gather was fixed,
> prefill and decode both measure top-1 0.990, top-5 **1.000**, top-100 **1.000**,
> against the bf16 reference *and* the fp32 control, in two independent builds
> (`evidence_accuracy.json`, `evidence_fp32_gate.json`). No dtype, fidelity, cache or
> reference change was needed.
>
> Note the filenames in the table below are reused by later runs: `evidence_accuracy.json`
> and `evidence_perf.json` now hold 0.990 / 1.000 / 1.000, not the numbers tabulated here.
> The table is the historical reading; the files are always the latest run.
>
> The section is kept in full because the *shape* of the error is worth more than the
> numbers were. Five controls all reproduced the same two misses at the same two
> positions predicting the same two tokens, and that agreement was read as strong
> evidence of a systematic numerical cause. It was not evidence of a cause at all: the
> controls shared one defect, and each of them re-ran the same broken gather. Five
> controls that agree are not five pieces of evidence unless they can fail
> independently, and none of these could. What was never run was the cheapest control
> of all — the same configuration twice, asking whether the number was even
> reproducible.

The gate: `top-5 >= 98 %` and `top-100 = 100 %`.

| gate | reference | top-1 | top-5 | top-100 | run |
| --- | --- | --- | --- | --- | --- |
| prefill | bf16 | 0.910 | 0.980 | **0.980** | `evidence_accuracy.json` |
| prefill | bf16 | 0.910 | 0.980 | **0.980** | `evidence_perf.json` |
| prefill | fp32 | 0.910 | 0.970 | **0.980** | `evidence_perf.json` |
| decode | bf16 | 0.960 | 0.990 | 0.990 | `evidence_accuracy.json` |
| decode | bf16 | 0.920 | **1.000** | **1.000** | `evidence_perf.json` |
| decode | fp32 | 0.930 | 0.990 | 0.990 | `evidence_perf.json` |

top-5 clears the bar everywhere. **decode top-100 reaches 1.000** in the warmed
shipped run and 0.990 in the other two, so decode is 0.99-1.00 -- one position of
spread. **prefill top-100 is 0.980 in all four measurements**: 2 positions of 100,
stable. That is visible wrongness, so it gets the anomaly treatment rather than a
sentence, and the investigation below is about the prefill pair.

```text
Observed anomaly: 2/100 prefill and 1/100 decode positions where the TT argmax is
                  outside the HF reference's top 100, against a 100% bar.
Evidence:         doc/full_model/evidence_accuracy.json (prefill_check,
                  teacher_forcing), evidence_misses_bfp4.json (per-position),
                  logs/run_prefill_check.txt, logs/run_teacher_forcing.txt
Affected path:    the terminal logits, at flat positions
Likely subsystem: candidates were (a) the BFP4 LM head this stage chose,
                  (b) the bf16 reference's own ordering, (c) the carried-forward
                  decoder precision policy
```

**Control 1 — where are the misses, and how flat?** `bench/evidence.py --stages misses`
reports the HF rank of the TT token at every non-top-1 position. Of 10 non-top-1
positions, **8 are #1<->#2 swaps**: TT picked HF's second choice and HF's first
choice was TT's second. The 2 outside-top-100 positions are both flat -- TT's top-5
spans 2.0 and 1.6 logits, and one has a top-1/top-2 margin of **0.125** against a
bf16 quantum of 0.0625 at that magnitude. This is a precision-margin signature, not
a structural one: a wrapper bug (embedding, norms, positions, masks, cache
indexing) would move every position, not two.

**Control 2 — is the *reference* the problem?** `bench/fp32_reference_control.py`
keeps the bf16 reference's prompt *and its generated continuation* -- so every
position is the same position -- and recomputes only the top-100 with HF in
**float32**:

* top-1 disagreements between the fp32 and bf16 references: **0/100**;
* fp32 top-1 outside the bf16 top-100: **0/100**;
* mean top-100 *set* overlap: **0.9635**, i.e. ~4 of every 100 ranked ids differ,
  so the tail of the reference's top-100 *is* precision-sensitive in general;
* but at TT's two miss positions specifically, the two references agree on top-1
  and their top-100 sets overlap **1.0000** and **0.9802**.

So the reference is solid exactly where TT misses. The misses are the model's.

**Control 3 — is it the LM head?** The head is this stage's own choice, so it is the
first thing to convict or exonerate. Rebuilt with BFP8 weights on the `mcast1d`
contract, everything else identical:

| LM head | non-top-1 | outside top-100 |
| --- | --- | --- |
| BFP4, `dram_sharded`, LoFi, bf16 out (shipped) | 10/100 | **2** |
| BFP8, `mcast1d`, LoFi, bf16 out | 8/100 | **2** |

Raising the head improves top-1 by 2 positions -- which the three-build spread below
shows is inside the gate's own resolution -- and **does not move the top-100 misses at
all**. The maximum-precision head control settles it: BF16 weights, HiFi4 math, fp32
accumulation and fp32 logits out, every precision axis the head has, maxed, at 3.6x
the shipped weight footprint, still gives **2**. The head is exonerated.

**Control 4 — is it the carried-forward precision policy?** The remaining candidate
is the policy this stage is contractually required to preserve: BFP8 attention
weights, BFP4 MLP weights, **BFP8 KV cache**, LoFi math. The skill's prescribed
control is to change *only* the cache dtype, which is what
`--decoder-kv-cache-dtype bfloat16` does. (A full BF16-weight decoder is not a
runnable control here: 967 MB/layer x 52 = 50 GB/device against 31.5 GiB.)

| configuration | non-top-1 | outside top-100 | the two miss positions |
| --- | --- | --- | --- |
| shipped: BFP4 head, BFP8 KV cache | 10/100 | **2** | 40, 90 |
| BFP8 head, BFP8 KV cache | 8/100 | **2** | 40, 90 |
| BFP4 head, **BF16** KV cache | 10/100 | **2** | 40, 90 |
| **BF16 head, HiFi4, fp32 acc, fp32 logits** | 8/100 | **2** | 40, 90 |

Neither knob moves them, and in all three the misses are the **same two positions
predicting the same two tokens**. So they are not precision coin flips — they are a
stable disagreement, which is what makes the next step decoding them rather than
sweeping more dtypes.

### What the two positions actually are

```
gen_index 40  context: "... constant speed s km/h, walk takes her 4 hours"
              HF:  ','      TT: ' and'
gen_index 90  context: "... Aya walks at s+1/2 km/h."
              HF:  ' Find'  TT: '.'
```

Both are **punctuation-or-conjunction choices at a clause boundary**, in a passage
where the model is restating the problem statement. And index 90 is demonstrably a
coin flip between two *punctuation* tokens: across builds TT's top-1 there alternates
between `.` (15.75 vs 15.625) and `?` (15.375 vs 15.25) with a top-1/top-2 margin of
**0.125** either way, and HF ranks **both** outside its top 100. Whichever way the
flip lands, the position misses.

Scoring the shipped configuration against the **fp32** reference as well settles the
reference question at the gate level rather than position by position:

| reference | top-1 | top-5 | top-100 |
| --- | --- | --- | --- |
| bf16 (primary) | 0.910 | 0.980 | **0.980** |
| fp32 (control) | 0.910 | 0.970 | **0.980** |

A better-ordered reference does not recover the two positions. Neither is a semantic error, a
degenerate repeat, a language drift or a control-token leak. At index 90 TT ranks
three punctuation tokens (`.`, `?`, `...`) above HF's ` Find` by 1.6 logits, and HF
returns the favour by putting `.` outside its top 100 -- two peaked distributions
over *different* tokens at the same clause boundary.

**Resolution: controlled, not fixed.** The residual is the accumulated numerical
difference of the 52-layer BFP8-attention / BFP4-MLP / BFP8-KV stack against an
FP32-accumulating HF reference. It is not attributable to any single dtype this
stage chose, the two candidate dtypes were each measured and each ruled out, and the
one remaining lever -- raising the decoder's precision policy -- is both forbidden by
this stage's contract (carry the optimized policy forward unchanged) and, for BF16
weights, physically impossible on this part. The datatype-sweep stage owns that
lever. What this stage owes instead is that the *behaviour* is sound, which is what
the qualitative suite and the free-running degeneracy gate measure; see
[README.md](README.md).

### Run-to-run spread of the gate itself

The default configuration's prefill check ran three times, in three separate builds
of the same weights (once as the primary measurement, twice as the control runs'
incidental default rebuild -- see below):

| build | top-1 | top-5 | top-100 |
| --- | --- | --- | --- |
| 1 | 0.910 | 0.980 | **0.980** |
| 2 | 0.910 | 0.970 | **0.980** |
| 3 | 0.900 | 0.980 | **0.980** |

**top-100 is stable at 0.980**; top-1 and top-5 move by one position.

> **Superseded.** This paragraph originally argued that the spread was one flat
> position on a rank boundary — that the path was bit-reproducible within a process
> (citing the 37-token reproducibility test and `prefill_repeat_probe.py` arm A) and
> so the movement merely set the resolution of a 100-position gate. That was wrong.
> The prefill path was **not** reproducible: both of those controls ran at or below
> 64 rows, which is exactly where the embedding all-gather defect of Section 15 is
> invisible. The ±1 movement was that defect. The reasoning that a 1-point top-1 or
> top-5 difference between two configurations is not a result still holds, but for a
> different reason than the one given here, and the numbers in this section were
> re-measured after the fix.

(Two rebuild traps, both self-inflicted and both worth recording so the next stage
does not repeat them. First, the readiness runners **re-execute
`tt/generator.py` from disk** on every call, so editing that file while a run is in
flight either rebuilds the model against a changed cache key or, if the edit adds an
import that the already-loaded `tt.model` does not have, fails with *"cannot import
name ... from ..."* mid-run. Do not edit `tt/` while a device job is running.
Second: the readiness runners call
`build_generator(model_dir, mesh_device)` with **no** knobs, so a control run that
overrides the LM head or the cache dtype gets its override honoured by the driver's
own stages but *not* by `run_prefill_check` / `run_teacher_forcing`, which build the
default config in the same process. Non-default configurations therefore have to be
measured through `--stages misses`, which uses the in-process generator. That is why
the control tables above quote miss counts rather than runner rows.)

## 14. The sampler was 36 % of the token-out step

The first warmed perf run split the token-out step cleanly:

| | ms/token | share |
| --- | --- | --- |
| model decode trace (52 layers + terminal path) | 23.166 | 64 % |
| sampling trace | **12.970** | **36 %** |
| total | 36.164 | -- |
| layer-stack lower bound | 23.239 | -- |

Two readings. The model trace is **under the floor** -- 23.166 against 23.239 ms of
summed decoder-layer latency. That is a bound, not a subtraction: the per-layer figures
come from the decoder stage's traced decode at context 2048, a slower regime than the
128-256 positions measured here, so "the terminal path costs less than nothing" is not a
conclusion it supports.
The reduced two-layer probe prices that tail directly: 1.553 ms for two layers, of which
~0.90 ms is the layers, leaving **~0.65 ms** of full-model-only work inside the trace.

And 12.970 ms to pick one token out of a `32 x 50688` shard is not credible next to
that, so it was measured rather than accepted (`bench/sampler_ab.py`). The A/B runs on
the reduced model because sampling cost is layer-count independent -- checked, not
assumed: **12.9699 ms on two layers against 12.9702 on fifty-two**.

| arm | sampling trace | |
| --- | --- | --- |
| `max_top_k=32`, pad shard to 65536 (module default) | 12.9699 ms | |
| **`max_top_k=32`, no pad** | **9.6888 ms** | **-3.28 ms, -25 %** |
| `max_top_k=8` | did not complete | composite all_gather, killed after 2 min |
| force-argmax | not measured | ruled out on contract |

`pad_logits_to_power_of_2` is an inherited `TTSampling` default that upstream describes
as a "big device-perf win for non-power-of-2 vocab on the multi-device path". On this mesh
it is a net loss of 3.28 ms per token, so it went off, and the shipped numbers were
re-measured with it off.

**That was right about the number and wrong about the reason, and the wrong reason cost a
33x optimisation.** The explanation recorded here was that "the 32x65536 -inf write costs
more than `ttnn.topk`'s bitonic fast path saves". The stage review checked it against the
profiler this stage had already collected and it does not survive:
a Tracy capture of the **pre-split** sampling trace showed `TopKDeviceOperation` at
**98.41 %** of it, at `Cores = 1`, with the candidate all-gathers, the untilize and the
tie-break pass together under 1 %. That capture no longer exists — a post-split re-capture
of the same filename overwrote it, which is this stage's own recurring artifact defect — so
the percentage is history rather than evidence, and `tracy/sampling_perf_report_stacked.csv`
now describes the *shipped* sampler (TopK 50.7 %, 142 us/call, multi-core). What survives
and carries the argument is `topk_geometry_probe.json`: **9.486 ms** for one 50688-wide
call against **0.144 ms** for a 32768-wide one, at the same payload and `k` — the
single-core/multi-core boundary measured directly on the op. `topk_device_operation.cpp:select_program_factory`
takes the multi-core factory only when the reduced width is a power of two, **below
65535** (UInt16 indices), at least 8192, with `k <= 64`. 50688 is not a power of two, so
single core; 65536 is *over* the uint16 bound, so **also** single core, just 29 % wider.
Both arms ran the same single-core kernel and the ratio is pure width:
12.9699 / 9.6888 = 1.338 against 65536 / 50688 = 1.293. No fast path was ever engaged, so
there was nothing for the pad's write cost to lose to.

The lever that follows is to make each call's width a power of two *below* the bound --
pad to 65536 and **split into 2 x 32768**. Measured on the op directly, at the real
`[1, 1, 32, W]` payload and k=32 (`bench/topk_geometry_probe.py`):

| width | ms/call | factory |
| --- | --- | --- |
| 50688 | **9.486** | single core (not a power of two) |
| 65536 | 12.741 | single core (over the uint16 bound) |
| 32768 | 0.144 | multi-core |
| 8192 | 0.108 | multi-core |
| 4096 | 0.771 | single core (below `multi_core_min_width`) -- smaller *and* 5x slower, which is the control proving the boundary is the rule and not the size |
| 2 x 32768 | **0.286** | multi-core; what the split costs |

Shipped as `topk_split_to_power_of_2` in `models/common/sampling/tt_sampling.py`
(`_topk_multicore_split`), opt-in and off for every other model, on by default for this
port. Sampling trace **9.689 -> 0.632 ms** (the same-process ratio, from the two arms
of `sampler_ab.json`, is **15.39x**); token-out **32.888 ->
23.811 ms/token** (30.41 -> **42.00 t/s/u**) and TTFT 71.66 -> **65.48 ms**. Every A/B arm
returns the same four tokens from the same seeded prompt.

One implementation trap, recorded because the first version shipped it: the split returns
`pieces * max_top_k` candidates per device rather than reducing back to `max_top_k` on
device. Reducing back is **incorrect**. A 64-wide reduction is below
`multi_core_min_width`, and the single-core factory *ignores* `indices_tensor` and returns
positions into its own input; the unsplit path only survives that because its indices
tensor is the identity map, where position and index coincide. A second stage over
already-permuted indices does not have that property, and it returned candidate positions
0..63 as vocab ids -- visible as a constant sampled token **101376 = 2 x 50688**, device
2's offset plus local index 0. `bench/topk_split_correctness.py` pins it exactly: the
*values* match a single call and torch to the last decimal, the indices come back
`[0, 32, 2, 1, ...]` where they should be `[55461, 13147, 9722, ...]`. So the candidate
set is widened instead, and `candidates_per_device` is what the device-offset tensor is
built from.

`max_top_k=8` was previously recorded as a dead end that "did not complete": an 8-column
candidate tensor pads to a 32-column tile and drops `ttnn.all_gather` onto its composite
path -- *"Using slower composite all_gather: gather dim 3 is padded from 8 to 32"* -- after
which it made no progress for two minutes and was killed. With the split it contributes 16
per device, 64 gathered, the composite path is not taken, and it measures **0.794 ms** --
slightly *slower* than the shipped 32. So 32 now ships on a measurement rather than by
elimination.

Force-argmax is off, and the reason recorded here was also wrong. It said `ttnn.argmax`
returns rank-3 `[1,1,32]` which "cannot be written into the rank-4 `[1,1,1,32]` token
buffer in place, so it breaks the device-side token feedback this stage owes". Upstream
passes `output_tensor=tt_out_tok` straight into `ttnn.argmax`, so the op *is* built to
write into a caller's persistent buffer; the rank mismatch is a property of the buffer
shape this port chose, not of the op. The real blocker is in the same file and is exact:
force-argmax's full-vocab gather calls
`self.tt_ccl.get_and_cycle_ag_semaphore_handles(...)` and this port passes `tt_ccl=None`
deliberately (Section 9: a `TT_CCL` puts 36 more semaphores in the main L1 pool against
7,296 B of decode headroom). With `tt_ccl=None` the arm does not error, it **hangs** --
which is what it did when it was left in the arm list, and why it is now removed from it
rather than expected to fail. It is also no longer worth chasing: it would trade a
0.632 ms sampling trace for a 12.9 MB full-vocab collective.

What is left: at 0.632 ms the sampler is **2.7 %** of the token-out step and the model
decode trace is 96 % of it. The token-out path no longer has a sampler problem.

Harness flaw, now fixed rather than only recorded: `sampler_ab.py` seeded
`torch.manual_seed` once, so each arm drew a different random prompt and its
`first_tokens` column could not be used to check that a faster arm gives the same answer.
Each arm now draws from its own seeded generator, which is what makes the "same four
tokens across every arm" check above mean anything. The shipped change is additionally
validated the stronger way, by re-running the real accuracy gate with it.

One correction to that last sentence, in light of Section 15. When it was first written
the accuracy gate was not reproducible, so "re-running the real accuracy gate" was weaker
evidence than it sounded -- a gate that moves by a position between identical runs cannot
certify a sampler change to a position. It is sound now: the gate was re-run after the
gather fix, on a reproducible prefill. The sampler timings themselves were never in doubt,
since they are decode-path measurements and the decode path was always reproducible.

---

## 15. Bug found and fixed: the embedding all-gather made prefill nondeterministic

Found by `test_logits_are_reproducible_across_batch_positions`, a test added late in
the stage precisely because nothing else exercised more than one cache slot.

```
Observed anomaly: the same 200-token prompt prefilled into cache slot 0 and slot 1
                  returned different logits -- max abs 1.6875 in one pytest pass,
                  1.5955 in the next.
Evidence:         logs/full_test_run.log, logs/full_test_run_slow.log
Affected path:    prefill, every prompt above one tile row
Resolution:       fixed (mitigated with a measured envelope; the residual lead is
                  upstream in TTNN).
```

The differing value across runs killed the first hypothesis immediately: a page-table
row or position-tensor row mixed up by slot would give the *same* wrong number every
time. What followed is the part worth recording, because three plausible
explanations were wrong and each was killed by measurement rather than by argument.

**It is not the cache slot.** `bench/batch_slot_probe.py --mode repeat` prefills the
same prompt repeatedly into slot 0 alone: also nondeterministic. So "slot 1" was never
the variable — the second call was.

**It is not the batch dimension, and not the prefill CCL implementation.** The same
probe at `max_batch_size=1` fails identically, and re-running with
`prefill_ccl_impl="wrapper"` (which also disables the fractured prefill norm) fails
too. What the sweep *did* establish is a threshold: 32 rows bit-identical, 128 and up
not (`batch_slot_probe_len_b1.json`, `_b4.json`).

**It is not fixed by synchronising, and "three runs agreed" proves nothing.**
`bench/prefill_sync_bisect.py` put a `synchronize_device` at each stage boundary in
turn. Each single-sync arm passed and the arm carrying *every* sync failed — incoherent
for a synchronisation story, and the real lesson: the failure rate is about one run in
three, so a 3-sample arm comes back clean by luck. Every probe after this one repeats
until it catches a divergence instead of sampling three times. An earlier clean
3-run result from `prefill_divergence_probe.py` had to be discarded on the same
grounds; re-run to 20 repeats it caught the divergence on the first comparison.

`bench/prefill_divergence_probe.py` then walked the graph stage by stage and named the
first stage that moves: **the embedding**, before any layer. The layers only carry it
forward. `bench/embedding_gather_probe.py` split that stage in two and
`bench/ccl_reproducibility_probe.py` dropped the model entirely:

| context | gather input | implementation | rows | reproducible |
| --- | --- | --- | --- | --- |
| standalone | host-staged constant | async / composite | 32–8192 | yes, and exactly correct |
| in-model | host-staged constant | async / composite | 128 / 1024 / 4096 | yes |
| in-model | `ttnn.embedding` output | async | 32 | yes |
| in-model | `ttnn.embedding` output | async | 64 / 128 / 1024 / 4096 | **no** |
| in-model | `ttnn.embedding` output | async, semaphores per call | 64 / 128 | **no** |
| in-model | `ttnn.embedding` output | composite `ttnn.all_gather` | 64 / 128 | yes |
| in-model | `ttnn.embedding` output | composite `ttnn.all_gather` | 1024 / 4096 | **no** |
| in-model | `ttnn.embedding` output | clone, then async or composite | 128 / 1024 | yes |
| in-model | `ttnn.embedding` output | clone, then async or composite | 4096 | **no** |

25 repeats per row. Three things fall out. The local lookup is stable across runs in
every arm, so the *values* are right and the corruption is in the gather's output —
at small payloads confined to a single remote device's 1664-column shard, which is
what a transport fault looks like rather than a compute one. Sharing the semaphores
across shapes and callers is *not* the cause, which retires the L1_SMALL-driven
suspicion the docstring on `_MODEL_CCL_SEMAPHORES` invited. And the collective itself
is not broken: gathering a host-staged tensor of identical shape and contents is
exactly correct at every size, standalone *and* inside the built model with all the
weights resident. The distinguishing property is that the input came out of
`ttnn.embedding`.

Root cause, as far as this stage took it: the embedding's output is a bad gather
input, and the one structural difference left against a staged tensor is that it
reaches the collective through the `ttnn.unsqueeze_to_4D` view that turns
`ttnn.embedding`'s rank-3 result into the rank-4 tensor the CCL ops require. That
could not be tested by avoiding the view — `ttnn.embedding` returns rank 3 whatever
its input rank (`all_gather_async_device_operation.cpp:288` rejects `dim=3` on the
rank-3 result), so the view is not optional. That lead belongs upstream in TTNN.

Fix (`EMBED_GATHER_CHUNK_ROWS` in `tt/model.py`): the prefill embedding gather is
issued in 1024-row chunks into **freshly allocated** buffers rather than once over
the embedding's own output — the one configuration the table above shows reproducible
at every size tried. A decode step is a single tile row, where the gather is
reproducible as issued and the step is traced and at its measured latency floor, so
that path is gated out and left byte for byte alone.

Acceptance: four prompt lengths (128 / 200 / 1024 / 4096) × nine repeats, all
bit-identical, including the 4096 rows that failed through every gather variant
tried, plus the original cross-slot matrix now clean in all five comparisons
(`batch_slot_probe_after_fix.json`, `batch_slot_matrix_after_fix.json`).

Those lengths do not on their own clear the **layers'** collectives, which run the same
`all_gather_async` on activations at `prefill_chunk_size` granularity — up to 8192 rows,
larger than any gather the acceptance run above exercised. So two longer lengths were
run separately: 8192 (one full chunk) and 12345 (8192 + 4153, i.e. a continuation call
with the per-layer sliding-tail hand-off), six repeats each, all bit-identical
(`batch_slot_probe_long_chunks.json`). The layers were never the problem, and now that
is measured at the chunk size they actually use rather than inferred.

Every other collective call site was audited for the same exposure, since all of them
gather or scatter an *op output* rather than a staged tensor. There are two in
`tt/model.py` — the embedding gather (fixed) and `gather_and_untilize_logits`, which
already clones and only ever gathers one 32-row tile of logits, inside the proven-clean
regime on both counts — and the rest are the carried-forward decoder's own
`reduce_scatter_minimal_async` / `all_gather_async` pairs on layer activations, at 32
rows in decode and `prefill_chunk_size` rows in prefill. Those are covered by
measurement rather than by argument: decode replays bit-identically, and prefill is
bit-identical at 8192 and 12345 tokens, which is where their payload is largest.

Pinned by `test_prefill_is_reproducible`, parametrised over 37 / 128 / 200 / 1024
tokens with six repeats each. Both numbers are deliberate. The old test ran 37 tokens
only — one tile row past the padding bug of Section 6, and *below* the 64-row payload
where the gather starts to move — which is how this defect sat under a passing test
for the whole stage. And six repeats rather than two, because at a one-in-three
failure rate a two-sample test passes most of the time it is broken.

**What this invalidated.** Section 13 recorded that three builds of the default
configuration moved top-1 and top-5 by ±1 position between runs and attributed it to
gate resolution. That was this defect, not resolution; the attribution was wrong and
the accuracy figures in this document and in `README.md` were re-measured on a
reproducible prefill afterwards. The two systematic top-100 misses of Section 13 are a
separate matter — they were stable across five independent controls, which is not
behaviour this defect produces — but they were re-measured too rather than assumed.

---

## 16. Bug found and fixed: the padded vocab ids were drawable

Found by the round-3 stage review, in code this stage wrote.

```
Observed anomaly: the sampler could return a token id in [202048, 202752) -- ids that
                  are vocabulary padding, not tokens.
Evidence:         models/common/sampling/vocab_padding.py returns None -- no mask at
                  all -- when vocab_size == padded_vocab_size, and tt/generator.py's
                  _SamplingArgs set both to 202752.
Affected path:    device split sampling, for any non-greedy SamplingParams.
Resolution:       fixed.
```

`_SamplingArgs` carried one width where two were needed, and its docstring argued for the
wrong one convincingly enough that it survived two review rounds. The padded width
(202752) is what the **index arithmetic** needs: the sampler turns a per-device top-k
index into a global id by adding `device * padded_vocab_size / tp`, and an unpadded value
there shifts every id on devices 1-3 by `d * 176`. But `TTSampling` reads
`args.vocab_size` for something else entirely -- `_create_invalid_vocab_mask` -- and
`build_invalid_vocab_mask(202752, 202752)` returns `None`, because there is nothing
between the two widths to mask. So the mask was never built and
`_mask_invalid_vocab_logits` was a pass-through.

Why that is reachable rather than theoretical: the LM-head weight **zero-fills** the 704
padded columns, so each padded id carries logit exactly `20 * tanh(0) = 0.0`. That beats
every real token whose logit is negative, and on this model that is nearly all of them at
some positions -- `evidence_misses_bfp4.json` has positions whose 5th-best real logit is
already below zero (`gen_index 28`: `[16.25, 8.625, 4.0, 3.1875, -0.21]`). At such a
position the local top-32 of a device fills with padding ids ranked 4th onward. Greedy
survives whenever the argmax is positive; a top-k/top-p request does not have to. An
emitted id >= 202048 is outside the tokenizer, and fed back it indexes an embedding table
with 202049 rows -- 202048 reads the zero pad row, anything above reads out of bounds.

Fix: pass both widths. `vocab_size = 202048` for the mask, `padded_vocab_size = 202752`
for the offsets. The tail mask is then built over 704 columns.

Cost: **not attributed to a number**, deliberately. The pre-mask endpoints (a 0.592 ms
sampling trace, a 23.76 ms token-out step) came from runs whose consoles later reruns of
the same filenames have overwritten, so any "+0.04 ms" claim would subtract a figure that
exists from one that does not -- the exact defect this stage kept re-finding elsewhere.
What is measured and current: the shipped sampling trace is **0.632 ms**
(`sampler_ab.json` 0.6321, `evidence_perf.json` 0.63205) and the shipped token-out step is
**23.811 ms/token**. The mask is small enough to sit inside this measurement's own
run-to-run spread. Accuracy is unchanged at top-1 0.990 / top-5 1.000 / top-100 1.000 on
both references, and every sampler A/B arm still returns the same four tokens.

Three things this stage got wrong here are worth naming, because none of them was caught
by a passing gate:

* the record described a mechanism that did not exist. `valid_vocab_size` -- named in
  `tt/model.py`, in `context_contract.json` and in `_SamplingArgs`' own docstring as the
  thing that "makes the sampler mask the padded tail" -- is not an attribute of anything
  in `models/common/sampling/`. Naming a mechanism is not evidence that it runs;
* a test **pinned the defect**: `test_config_reports_the_carried_forward_contract`
  asserted `tt_sampling.vocab_size == padded_vocab_size`, under a comment describing the
  *offset* requirement, which the separate `padded_vocab_size` attribute already
  satisfied. The assertion was written from the docstring rather than from the consumer;
* nothing anywhere asserted the obvious end-to-end property -- that a sampled id is a
  valid token id. `test_sampling_never_returns_a_padded_vocab_id` now does, under greedy
  and under three sampled configurations including `temperature=2.0`, which is where the
  padded ids are actually reachable.

---

## 17. Bug found and fixed: a test that only passed because of its neighbours

Round 11's review noted that this stage had no watcher run, unlike the functional-decoder
stage, so I ran one over a six-case device subset (multi-chunk prefill reproducibility,
split-sampling token feedback, steady-state decode, the multi-core topk admission check,
per-row device sampling, and batch-32 mixed lengths). The watcher log came back clean.
The **pytest** run did not: `test_split_sampling_feeds_the_sampled_token_back_on_device`
failed with a bare `StopIteration`.

That was not a watcher effect. Re-running the same two node ids with watcher off
reproduced it exactly, and so did running the test on its own. The test did

```python
clean.generate(prompt_token_ids=prompt, max_new_tokens=1, enable_trace=True)
slot = next(iter(clean.sampling._trace_states.values()))     # StopIteration
```

and the traces are captured **inside the decode loop** (`generator.py:801-809`), which
`max_new_tokens=1` never enters -- that call is prefill, one sampled token, return. So
the test asserted on trace state that its own call had never created. It passed in the
full-file run only because the `generator` fixture is module-scoped and an earlier test
had captured the traces on the shared object; traces survive `reset()`, which
`test_reset_zeroes_the_cache_without_dropping_traces` pins deliberately. Every `-k`
selection, every subset, and every isolated run of that one test was failing.

Fix: ask for `max_new_tokens=2`, so the call itself enters the decode loop and owns the
state it asserts on. It now passes standalone (19.4 s).

Then the obvious follow-up question: how many *other* tests were passing on a
neighbour's side effects? Rather than argue about it, I collected all 46 node ids and
ran the file **in reverse order in one process** -- same fixtures, opposite ordering.
**46 passed in 230.19 s** (`logs/reverse_order_run.log`, the committed console; the
figure gate resolves the count and the time against it, because the first version of
this claim quoted a 226 s number that a *forward*-order run had produced). That is not a
proof of independence for every permutation, but it is the cheap experiment that would
have caught this one, and it now catches the class.

Two things worth keeping from this:

* the watcher run was requested as a *runtime-integrity* check and did not find a
  runtime-integrity problem -- it found a test defect, because running the suite a new
  way is itself the experiment. The watcher log's own verdict is clean.
* a module-scoped fixture that carries device state (traces, caches, counters) turns
  "this test passes" into "this test passes *here*". The reverse-order run is now a
  documented command in [README.md](README.md#how-to-reproduce) for exactly that reason.

---

## 18. Repo hygiene, and why it cost a re-measurement

Before committing I ran the repo's own pre-commit hooks over the stage's files rather
than assuming they would pass. Three of them had something to say, and the third is a
lesson about ordering:

* **`check-large-files` (>500 KB).** Ten console logs were 850 KB-1.4 MB, the raw Tracy
  ops CSVs were 535-656 KB *already gzipped*, and the watcher run had left 6.6 MB of raw
  Metal inspector dumps. Fixed by gzipping the consoles (`gzip` preserves mtime, so the
  ordering table survives), recompressing the ops CSVs with `xz -9` (656 KB -> 208 KB;
  `bench/run_tracy.sh` now does this itself), and deleting `watcher/generated/` while
  keeping the watcher log itself as `watcher/watcher_final13.log.gz`. Doc references and the provenance regex in
  `bench/check_reported_figures.py` were updated to the `.log.gz` names.
* **`prefer-expect-error`.** This repo requires tests to use the `expect_error` fixture
  from the root `conftest.py` rather than `pytest.raises`, so CI log triage can tell an
  expected error from a real one. Six call sites converted.
* **`black` / `isort`.** Nine files rewritten in all: `tt/model.py`, `tt/generator.py`
  and `tests/test_full_model.py` (two reformatted, two import blocks reordered), plus six
  bench scripts touched only by the whitespace/formatting hooks — `run_tracy.sh`,
  `sampler_ab.py`, `qualitative.py`, `batch_slot_probe.py`, `perf_window.py`, `smoke.py`.
  The bench rewrites looked harmless and were not written down, and one of them cost a
  review round: `bench/qualitative.py` moved to 22:52:04, which silently falsified the
  ordering table's claim that the HF control "postdates its own driver" — the control was
  produced at 11:01 and had not moved. A hook pass is an edit to every file it touches,
  including the ones whose *content* it barely changes.

The formatting is provably behaviour-preserving -- `ast.dump` of every touched file is
identical before and after for five of them, and for `tt/model.py` and `tt/generator.py`
the *non-import* statements are identical and the imported-name sets are equal, the only
difference being import order. But it still **edited implementation files after the
shipped console was captured**, which is exactly the condition this stage's own
provenance gate exists to catch, and the `expect_error` conversion is a genuine test
change that has to be run rather than argued about. So the whole device pass was run
again on the formatted tree, and the ordering table was rebuilt from the new mtimes.

What the re-measurement showed (`logs/final13.log.gz`, every step rc=0):

* **accuracy is bit-for-bit the same claim** -- 0.990 / 1.000 / 1.000 on all four
  gate x reference combinations, as in the two previous passes;
* the qualitative table reproduced **cell for cell** -- same adjacent-duplication, same
  trigram-loop figures, same first-divergence tokens on all six prompts;
* decode moved by 0.08 %, from 23.820 ms/token in `final12` to 23.800 in `final13`, and
  TTFT by 1.8 %, from 66.04 ms to 64.83 — both inside the process-to-process envelope
  this stage already documents;
* the sampler A/B, the Tracy composition and the fallback counters all reproduced, and
  the context contract recomputed to the same 131072.

So the formatting was as behaviour-neutral as the AST argument said. The point of
re-running was never that it might not be -- it was that "the shipped console describes
the shipped code" is a property of the *record*, and an argument cannot restore it.

The generalisable point: run the repo's hooks *before* the final measurement, not after.
Hooks are implementation edits, and they land at the worst possible moment.

---

## 19. Round 15: three record defects, and the perimeter that kept letting them through

Round 15 found the LM-head core-count band **wrong in `tt/model.py`** — the same
1.0107-1.0426 that round 11 corrected "in both records" four rounds earlier. There were
three copies, not two, and the third lived in a module docstring that no value check
read: every figure gate walked `README.md` (and later `work_log.md`), while only the
retracted-*string* sweep walked `tt/`. The same file also still mixed MiB into a
decimal-GB sentence (641 MB for what is 641 MiB / 672 MB), which round 11 had fixed in
the README alone.

That is the whole lesson of this round, and it is a perimeter lesson rather than a
number one: **a wrong measurement in the file a later stage reads first is worse than a
wrong one in the prose, not better.** The band derivation now runs over the markdown and
the source together, and it only demands a match from a file that quotes a band at all,
so adding a docstring table cannot silently escape it.

Fixing source meant the shipped console no longer described the shipped code, so
everything was measured again (`logs/final14.log.gz`, every step rc=0). Fourth pass in a
row with identical accuracy: 0.990 / 1.000 / 1.000 on all six gate x reference rows, and
`prefill_misses` again lands on `gen_index 64` with the reference's own rank-1 token.
TTFT read 65.48 ms (61.09 / 66.04 / 64.83 / 65.48 across four passes now) and decode
23.811 ms/token (23.810 / 23.820 / 23.800 / 23.811) — a 0.05 % decode spread against
7 % on TTFT, which is the same split this document already predicts between a traced
steady state and a once-per-process prefill path. Against the immediately preceding
pass: decode moved by 0.05 % (23.800 -> 23.811 ms/token) and TTFT by 1.0 %
(64.83 -> 65.48 ms).

Two more findings from the same round were structural rather than numeric:

* the artifact-ordering table's **exception groups carry reasons, and nothing checked
  them**. One said each HF/CPU control "postdates its own driver"; the HF qualitative
  control was produced at 11:01 and its driver had been rewritten at 22:52 by the
  formatting hooks (see the corrected §18), so the claim had been false for six rounds.
  The HF arm was regenerated in this pass — CPU only, no device — and the reason is now
  a gate: each named control must postdate the driver it names.
* the **15.3x** sampler ratio was wrong: it divided a pre-mask endpoint by a post-mask
  one. The two
  arms of a single `sampler_ab.json` give **15.39x** in one process with one seeded
  prompt, which is the number this document should have been quoting all along. The
  stage refused to *subtract* endpoints across eras and then divided them anyway.

One thing the re-run changed that is not a regression: the qualitative table's p0 and p3
cells moved, because the chat template renders **the current date** into the system
message and the previous pass ran on 2026-08-13. Both arms were regenerated together on
2026-08-14, so the comparison is internally consistent; it is simply a different prompt.

---

## 20. Commits

Recorded after the stage review returned `clean-pass`; see the final section of
[README.md](README.md).
