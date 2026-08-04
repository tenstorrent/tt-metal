# Gemma4 context-parallel prefill on Blackhole Galaxy

**Date:** 2026-08-03
**Status:** Implemented and passing — the whole prefill file runs under CP with no
skips (32 passed: single layer, the 60-layer body, and the full graph through
lm_head; all chunk sizes, eager and traced). See Implementation status.
**Scope:** Prefill only. Decode is disaggregated and runs outside this path.

## Implementation status

Landed on `svuckovic/gemma4-prefill`:

- **Single-layer CP works.** `test_prefill_layer` passes 16/16 on a BH Galaxy at
  `4x8` (TP=8 x CP=4): sliding and global, chunks 512/1024/2048/4096, eager and
  traced. PCC 0.9993-0.9999 against HuggingFace, with eager and traced identical
  in every case. High PCC also confirms the position plumbing lines up — a
  mismatch between input shard order, mask row sharding, K/V gather order, or the
  output gather would scramble positions and collapse PCC rather than nudge it.
- **All-gather path only.** As sequenced below, K/V are gathered across the whole
  CP axis for every chunk size and both layer types. The halo optimization is
  *not* implemented; it is a pure performance follow-up, and the fallback rule in
  "Halo generalization" means correctness never depended on it.
- **A1 came for free at this milestone.** `test_prefill_layer` runs with
  `kv_cache=None`, so there is no paged history and every rank's `base_offset` is
  0 — the scalar-offset problem does not bite until the whole-model path.
- **The 60-layer body works too** (`test_prefill_layers`, 8/8: four chunk sizes x
  eager/traced). Depth was never the obstacle — the layer is CP-shape-preserving,
  so chaining was already free. Three orthogonal things had to be fixed:
  1. *Token staging.* The body's input is token ids `[1, chunk]` and the model
     embeds on device. Sharding them needs `seq_dim=-1`; on a 2D tensor `-2` is the
     size-1 batch dim. `embed_tokens` composes cleanly because its all-gather runs
     on the TP axis (hidden), orthogonal to CP.
  2. *RoPE.* The model builds caches internally and slices
     `[start : start+seq]` with a mesh-wide scalar, which cannot differ per device.
     The 4D prefill caches are now sharded along positions, so the model's existing
     `[0:local_seq]` slice lands on the rows each rank owns — the same "offset as
     data" move as the mask. Requires `max_seq_len == chunk`, enforced by a guard.
  3. *Readback.* CP gather instead of reading device 0.

  Traced timings at 60 layers: 80.8 ms (512), 114.1 ms (1024), 158.9 ms (2048),
  273.1 ms (4096) — peak ~15k tok/s.
- **The full graph through lm_head works** (`test_prefill_full`, 8/8). It selects
  one absolute position's logits, and both paths pick it with a mesh-wide scalar
  index (`get_last_token=tile_start` eagerly, `process_logits_after_prefill_trace`
  after a trace) — which under CP addressed a different span on every rank. The
  sequence is now gathered before either slice: ~44 MB at 4k x 5376 bf16, and the
  head still only sees 32 rows. Gathering the *logits* instead would be ~4.3 GB at
  a 262k vocab, which is why the head is fed a slice at all. The gather sits only
  on the `get_last_token != -1` branch, since a caller wanting every row may keep
  logits CP-sharded; traced runs always pass -1, so it never lands in a capture.

  Eager and traced agree exactly on the top-5 next tokens at every chunk size, and
  the predictions move sensibly with context (512 tokens gives
  `As/Since/Because/While`; 1k-4k converge on a confident single token). Note this
  test is a smoke test, not a PCC test — there is no host-side 60-layer reference
  that fits in RAM, so the numerical evidence for CP comes from the single-layer
  PCC against HuggingFace.
- **CP branch guarded against the 32768 SDPA cliff.** It uses the non-chunked op,
  and under CP the relevant length is K (`cp * local`), so a check on the local Q
  length under-reports by `cp`. Concretely, chunk 32768 with CP=4 gives local Q
  8192 (looks safe) against K 32768 (on the cliff) — a silent-corruption window,
  now a loud failure. There is no correct path to fall through to: the chunked
  fallbacks take a scalar offset and reject `attn_mask`.
- **Mask cache is shared across layers.** It was per-`Gemma4Attention`, so a
  60-layer stack would have held 60 copies of an 8 MiB tensor (~480 MiB/device)
  where two suffice — the mask depends only on `(local_seq_len, sliding_window)`.
  Invisible at one layer; it scales with depth x chunk (at chunk 16384 the mask is
  128 MiB, so 60 layers would have been ~7.7 GiB/device). Now on `CCLManager`,
  which is already one-per-mesh.
- **Trace bug fixed along the way.** `_run_graph` ran `ttnn.clone` inside
  `begin_trace_capture` but never during warmup, so capture aborted with "Cannot
  load new binaries during trace capture" and left the device in capture state,
  hanging the next test. Pre-existing, unrelated to CP.

Not verified, and worth knowing:

- **No device-level CP=1 regression test was possible on this machine.** `1x8` and
  `1x4` fail fabric init (partial mesh of a 32-device system: "Ethernet handshake
  likely failed"), and `1x32` hangs in `all_gather_unicast` inside the TP=32
  all-reduce — a pre-existing 32-device collective problem this work does not
  touch. Only the 31B checkpoint is present locally, so a small-model `1x1` control
  was not available either. CP=1 neutrality therefore rests on a static argument:
  every new path is gated on `cp_degree(...) > 1`, which is false for any `(1,N)`
  mesh, and the mappers and readback fall back to the previous behaviour exactly.
- Physical-axis mapping of the logical `(4,8)` after rotation, and whether
  `FABRIC_2D` beats a torus variant, are still unmeasured. `FABRIC_2D` does
  initialize cleanly on all 32 devices.

## Goal

Speed up Gemma4-31B prefill on a 32-chip Blackhole Galaxy by adding a second
parallelism axis. Today the model uses tensor parallelism only. This design adds
context parallelism (CP) across a second mesh axis so the token dimension is
split as well, targeting a 256k maximum context prefilled in 4k chunks.

## Background: why the obvious options don't apply

Three findings constrain the design. Each was verified against the code or the
hardware rather than assumed.

**TP=32 hangs, for an unrelated reason.** `ttnn.all_gather` silently falls back
to `composite_all_gather` when a row-major input page is not 64 B aligned
(`ttnn/cpp/ttnn/operations/ccl/all_gather/all_gather.cpp`), and that composite
path — `ttnn::all_broadcast` + `concat` — deadlocks at 32 devices. With
`hidden_size = 5376`, the per-device shard at TP=32 is 168, giving a 336 B
row-major page; 336 = 5x64 + 16, so it is unaligned. TP=4 (1344 -> 2688 B) and
TP=8 (672 -> 1344 B) are both aligned, which is why 1x4 works. This is a
separate bug with a separate ~2-line fix (convert to `TILE_LAYOUT` before the
gather in `Gemma4Model.embed_tokens`); it is **out of scope here**, but it is why
TP=8 needs no workaround.

**gpt_oss's sequence parallelism is not portable.** All of it —
`_reshard_for_sequence_parallel`, `apply_sequence_parallel_allgather` — lives
inside `models/demos/gpt_oss/tt/experts/prefill.py`. Gemma4-31B is dense
(`enable_moe_block = False`, `num_experts = None`), and
`models/demos/gemma4/tt/layer.py` gates `MoEBlock` on that flag, so 31B never
enters the experts path. There is no dense-model SP in gpt_oss to copy. The
*pattern* (reduce-scatter -> pointwise compute -> all-gather) still transfers;
the code does not.

**Memory sets the CP degree.** CP replicates weights and KV cache across the CP
axis while TP shards them, so CP degree trades directly against per-device
memory. At 256k the KV cache makes this binding. The KV cache is head-sharded by
TP (`num_local_kv_heads = num_key_value_heads // tp`, 16 KV heads total), and the
full-context KV cache is ~108 GiB in bf8. Blackhole has 8 DRAM banks of
4,278,190,080 B, about 34 GB per device.

| config | weights/dev | KV cache/dev @256k | total | fits ~34 GB |
| --- | --- | --- | --- | --- |
| TP=32, CP=1 | ~1.9 GB | 108 GiB x 1/16 = 6.8 GiB | ~8.7 GiB | yes, easily |
| **TP=8, CP=4** | ~7.8 GB | 108 GiB x 2/16 = 13.5 GiB | **~21 GiB** | yes, ~13 GB spare |
| TP=4, CP=8 | ~15.5 GB | 108 GiB x 4/16 = 27 GiB | ~42 GiB | **no, OOM** |

TP=4 x CP=8 is therefore infeasible at 256k. **TP=8 x CP=4 is the target.**

## Target configuration

Mesh **`(4, 8)`**: TP=8 on axis 1, CP=4 on axis 0.

`MeshConfig` needs no functional change. Its prefill default is already
`ModeConfig(tp=decode.tp, sp=mesh_shape[0], ep=1)`, which on a `(4,8)` mesh
yields `tp=8, sp=4`, and `sp_axis = 0` follows from `tp_axis = 1`. The existing
field name `sp` is kept — it matches gpt_oss and costs no churn — and documented
as the context-parallel degree.

Use `(4,8)` rather than `(8,4)` with `tp_axis=0`. Keeping TP on axis 1 preserves
the existing "column axis is TP" assumptions, which are hardcoded in several
places: `tt/attention/kv_cache.py` computes `tp = mesh_device.shape[1]`,
`tt/rms_norm.py` sizes a buffer with `mesh_shape[1]`, and `tt/model.py` sets
`sampling_all_gather_axis = 1`. Choosing `tp_axis=0` would require auditing all
of them; `(4,8)` keeps them correct for free.

Two consequences:

- **Weight cache becomes `_tp8_`.** Needs one cold-cache run with
  `GEMMA4_PREFILL_LOAD_FULL_WEIGHTS=1` (~39 GB). The harness's `_require_cache`
  already skips cleanly with an actionable message when it is missing.
- **Fabric needs 2D routing**, since collectives now run on both axes. Start
  with `FABRIC_2D`.

## Data flow

The sequence is sharded on entry and stays sharded. Each rank owns
`T = chunk / cp` tokens — 1024 of a 4096-token chunk at CP=4.

Sharding happens at input staging, not via a collective: stage token IDs and
`position_idx` with a mesh mapper that shards the sequence dimension across
axis 0 and replicates across axis 1. gpt_oss uses a `reduce_scatter`-and-rescale
trick only because its inputs arrive already replicated; here staging is under
our control, so the scatter is free.

**Local, no communication** (all pointwise along sequence): token embed lookup,
all four layernorms, residual adds, QKV projection, O projection, `SharedMLP`
gate/up/down, final norm, lm_head.

**Unchanged**: the existing TP all-reduces on axis 1 (attention O-proj, MLP
down-proj). `ccl_allreduce` already targets `tp_axis`, so these need no edit.

**New**, one collective per layer on axis 0, only in attention. Sliding and full
layers differ; see below.

The KV cache needs no shape change. It is built with `ReplicateTensorToMesh` and
a per-device shape of `num_local_kv_heads`, so replication across the CP axis is
already the desired behaviour — and it is why full-attention layers can read
history locally instead of over fabric.

For CP rank 0, the sliding tail comes from the *previous 4k chunk*, which is
exactly today's in-memory `sliding_tail_in`. Ranks 1-3 get theirs over fabric.
Same code path, different source.

## Sliding-window layers (50 of 60)

These are fully CP-parallel with no complications.

A sliding query attends the preceding W tokens regardless of absolute position,
so the mask is position-*relative*. `chunked_prefill_sdpa_sliding` takes no
position offset for exactly that reason: it builds a square `[tail | chunk]`
slice and masks relative to the slice. Every CP rank therefore runs an
identically-shaped, identically-masked problem.

RoPE is the only consumer of absolute positions, and it applies them through
`ttnn.embedding(position_idx, cos_2d)` — a tensor lookup — so `position_idx`
shards per-rank for free.

### Halo generalization

The halo must cover W = 1024 tokens. One neighbour is correct only at chunk 4096:

| chunk (CP=4) | tokens/rank | halo ranks needed | path taken |
| --- | --- | --- | --- |
| 4096 | 1024 | 1 | halo |
| 2048 | 512 | 2 | halo |
| 1024 | 256 | 4 | all-gather |
| 512 | 128 | 8 | all-gather |

A one-neighbour design would be wrong for three of the four chunk sizes the
harness parametrizes. The rule:

```
T = chunk // cp
halo_ranks = ceil(W / T)
if halo_ranks >= cp - 1:
    all-gather K/V across the CP axis     # degenerate; cheaper than a multi-hop halo
else:
    fetch trailing K/V from halo_ranks preceding neighbours
```

Communication volume is small. The halo is W tokens x `num_local_kv_heads` (2 at
TP=8) x `head_dim` x 2 (K and V), about 1 MB per device per layer in bf8, so
~50 MB per 4k chunk across all sliding layers. The design is compute-bound.

## Full-attention layers (10 of 60): approach A1

These cannot be CP-parallelized without solving a scalar-argument problem.
`chunked_prefill_sdpa` takes `base_offset` as a **Python scalar** — the absolute
position of `tt_q`'s first row, used to build the causal mask. Under CP, rank
*r*'s Q begins at `chunk_offset + r*T`, which differs per rank, and a ttnn
program dispatches mesh-wide with one set of scalars.

**A1, chosen:** all-gather Q/K/V across the CP axis for these layers, run SDPA
with a uniform `base_offset` on every rank, then slice this rank's rows locally.
One collective, no scatter, and the MLP stays sharded because the attention
output is sliced back to T rows locally.

This forfeits parallelism on the 10 full-attention layers, which is the dominant
cost at long context. The trade was made deliberately for implementation safety;
see Expected performance.

The alternative (**A2**, deferred) is to dispatch these layers once per CP row on
a `1x8` submesh, each with its own `base_offset`. `create_submesh` exists and is
used in `models/tt_dit/`. It is deferred because trace-capture behaviour across
submeshes is unverified and host dispatch cost multiplies by CP.

## Expected performance

Rough FLOP shares per 4k chunk at the 256k tail. Order-of-magnitude estimates,
to be replaced by profiling.

| work | ~FLOPs | CP-parallel under A1 |
| --- | --- | --- |
| full-attention scores (10 layers, 256k history) | ~7e14 | no |
| MLP (60 layers) | ~1.7e14 | yes |
| QKV/O projections (60 layers) | ~6.5e13 | yes |
| sliding scores (50 layers) | ~2e13 | yes |

Amdahl on those shares gives **~1.24x at the 256k tail** and **~1.4x averaged**
over a full 256k prefill, where history averages 128k. For comparison,
parallelizing full attention as well (A2) would approach 4x.

This modest ceiling is a known and accepted property of A1. If measured gains
fall short of ~1.4x, the right response is to profile before adding complexity,
and A2 is the identified next step rather than incremental tuning of A1.

## Components

| file | change |
| --- | --- |
| `models/demos/gemma4/config.py` | no functional change; document `sp` as CP degree; validate `chunk % (cp*32) == 0` |
| `models/demos/gemma4/tt/ccl.py` | add `ccl_cp_allgather(tensor, dim)` and `ccl_cp_halo(tensor, halo_ranks)` on `sp_axis` |
| `models/demos/gemma4/tt/attention/prefill.py` | bulk of the work: K/V all-gather + halo for sliding; Q/K/V gather + local row slice for full |
| `models/demos/gemma4/tt/attention/kv_cache.py` | no shape change; assert CP-axis replication is intended |
| `models/demos/gemma4/tt/model.py` | stage tokens and `position_idx` sharded on `sp_axis`; gather logits only as tests require |
| `models/demos/gemma4/demo/text_demo_prefill.py` | `GALAXY_MESH = (4, 8)`, `FABRIC_2D`, cache key `_tp8` |
| `models/demos/gemma4/tests/test_factory.py` | add `(4,8)` with 2D fabric config |

## Testing and acceptance

`models/demos/gemma4/demo/text_demo_prefill.py` already compares against
HuggingFace by PCC, so acceptance is mostly reuse.

- **`test_prefill_layer` is the gate.** It already parametrizes `sliding` and
  `global`, covering both CP paths independently. Must meet PCC at CP=4 for both.
- **All four chunk sizes must pass**, not only 4096. 4096 is the 1:1 halo case;
  2048 exercises the 2-rank halo and 1024/512 the all-gather fallback. A
  one-neighbour bug would hide here.
- **CP=1 regression.** On a `(1,8)` mesh every new path must short-circuit and
  results must be identical to current behaviour. Cheapest guard against
  breaking the working configuration.
- `test_prefill_layers` (60 layers + final norm) and `test_prefill_full`
  (embed -> layers -> norm -> lm_head -> softcap).

Sequencing: implement the all-gather path first — it is correct for every chunk
size and both layer types — reach PCC green everywhere, then add the halo as an
optimization and re-verify. This also means A1's full-attention path and the
sliding fallback share one primitive.

## Error handling

- `cp == 1`: every new path short-circuits, preserving current behaviour exactly.
- `chunk % (cp * 32) != 0`: fail at config time, reporting the computed values,
  rather than failing mid-graph.
- `tokens_per_rank < 32`: explicit failure (sub-tile shard is not representable).
- Sliding layer with `sliding_window is None`: fail rather than silently attend
  the whole sequence.

## To verify during bring-up

These are assumptions the design rests on that have not been tested:

1. Which physical axis each logical axis of `(4,8)` lands on. The physical system
   mesh is 8x4, so `(4,8)` relies on the rotation path in
   `tt_metal/distributed/system_mesh.cpp`. This determines per-axis link
   bandwidth.
2. Whether `FABRIC_2D` is correct versus a torus variant
   (`FABRIC_2D_TORUS_X/Y/XY`), which the other 8x4 Galaxy models in this repo use.
3. That a `(4,8)` mesh opens at all on this system, and that CP-axis collectives
   on `sp_axis = 0` behave under trace capture.

## Out of scope

- **Decode.** Disaggregated, runs outside this path.
- **The TP=32 alignment fix.** Separate bug, separate change; TP=8 does not need
  it.
- **A2 submesh dispatch and ring attention.** Identified follow-ups, deferred.
- **Expert parallelism.** 31B is dense; `ep` stays 1.

## Operational note

A device hang wedges ethernet cores. Reset with
`python_env/bin/tt-smi -r` before the next run, or subsequent failures present as
misleading `Timed out while waiting for active ethernet core` errors rather than
the real behaviour.
