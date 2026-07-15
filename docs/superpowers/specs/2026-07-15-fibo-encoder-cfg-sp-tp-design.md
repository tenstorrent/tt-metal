# FIBO encoder: CFG=2 × SP=4 × TP=4 with fixed-bucket padding

**Date:** 2026-07-15
**Branch:** `fibo-pipeline`
**Status:** Approved (design)

## Goal

Re-parallelize the SmolLM3 text encoder in the Bria FIBO pipeline on the 4×8 Galaxy so all 32
devices are used for a single encode, instead of the current TP=8 that encodes the two CFG branches
sequentially.

New layout (per encode of both prompts):

| Factor | Value | Mesh axis | Purpose |
|--------|-------|-----------|---------|
| CFG    | 2     | axis 1 (split) | positive & negative prompts encoded **concurrently** on two submeshes |
| TP     | 4     | axis 1 (within submesh) | tensor-parallel Q/K/V/O, as today but factor 4 not 8 |
| SP     | 4     | axis 0 | **sequence-parallel across tokens** (new) |

`CFG(2) × TP(4) × SP(4) = 32` = the whole 4×8 mesh. The 4×8 is carved into **two (4,4)
submeshes**; within each, SP=4 on axis 0 and TP=4 on axis 1.

Also: pad the encoder input to a **fixed bucket length** (start with `[1024]`, extensible to
`[1024, 2048, ...]`) instead of the current "round up to a multiple of 32". A fixed bucket gives a
stable shape for SP sharding (bucket divisible by `sp_factor × 32`) and for program-cache reuse.

The DiT and VAE parallelization is **unchanged** (they stay cfg=1, whole-mesh, TP=8 / SP=4). Only
the encoder path changes.

## Scope

- `models/tt_dit/parallel/config.py` — extend `EncoderParallelConfig`.
- `models/tt_dit/encoders/smollm3/model_smollm3.py` — sequence-parallel attention (all-gather K/V).
- `models/tt_dit/pipelines/bria_fibo/text_encoder.py` — fixed-bucket padding, SP-aware host I/O.
- `models/tt_dit/pipelines/bria_fibo/pipeline_bria_fibo.py` — encoder submeshes, two encoder
  instances, concurrent `_encode`.
- `models/tt_dit/tests/models/bria_fibo/test_performance_bria_fibo.py` — `test_fibo_encode_perf`
  exercises the new layout (mostly unchanged; it calls `pipe._encode`).
- `models/tt_dit/tests/encoders/smollm3/test_smollm3.py` — PCC validation of SP attention.

Out of scope: making the whole pipeline (DiT/VAE denoise) CFG-parallel.

## Current state (baseline)

- Mesh `(4, 8)`: `sp_axis=0` (size 4), `tp_axis=1` (size 8).
- `EncoderParallelConfig.from_tuple((tp_factor=8, tp_axis=1))` — TP=8 only; `sequence_parallel`
  field exists but is unused.
- One `SmolLM3TextEncoderWrapper` built on `self._submesh` (the whole 4×8; DiT cfg=1).
- `_encode` calls `encode_prompt(pos)` then `encode_prompt(neg)` **sequentially**.
- Wrapper pads `seq_len` up to the next multiple of 32 (`padded_len = -(-seq_len // 32) * 32`),
  runs `is_causal` SDPA, and slices `[:, :seq_len, :]` back on host.
- Encoder attention (`model_smollm3.py:298-340`) runs full-sequence causal SDPA on each TP shard;
  **no sequence parallelism**. `all_gather` over `tp_axis` for the O-projection is the only CCL.

## Design

### 1. `EncoderParallelConfig`

Extend to carry cfg and sp alongside tp:

```python
class EncoderParallelConfig(NamedTuple):
    tensor_parallel: ParallelFactor
    sequence_parallel: ParallelFactor | None = None
    cfg_parallel: ParallelFactor | None = None

    @classmethod
    def from_tuples(cls, *, tp, sp=None, cfg=None):
        return cls(
            tensor_parallel=ParallelFactor(*tp),
            sequence_parallel=ParallelFactor(*sp) if sp else None,
            cfg_parallel=ParallelFactor(*cfg) if cfg else None,
        )
```

`from_tuple((tp, axis))` is kept for back-compat (existing device-profile test uses it).

Pipeline builds it as (on 4×8): `tp=(4, 1), sp=(4, 0), cfg=(2, 1)`.

### 2. Encoder submeshes (pipeline)

The encoder gets its **own** submesh split, independent of the DiT's single submesh. Reuse
`cfg.create_submeshes`, which slices the device into cfg-parallel submeshes sized by sp×tp:

```python
enc_cfg = DiTParallelConfig.from_tuples(cfg=(2, 1), sp=(4, 0), tp=(4, 1))  # shape-only, for the split
enc_submeshes = create_submeshes(device, enc_cfg)   # -> two (4,4) submeshes
```

For each submesh build:
- its own `CCLManager` (num_links for a (4,4) submesh — add `(4, 4): 2` to `_num_links` in the
  test and mirror the pipeline's link pick),
- a `SmolLM3TextEncoderWrapper` with `EncoderParallelConfig(tp=(4,1), sp=(4,0))`
  (cfg is expressed by *having two* encoders, not inside one).

Weights load twice (once per submesh); acceptable — each submesh is half the mesh.

Degradation on 2×2: `cfg=(2,1), sp=(2,0), tp=(1,1)` → two (2,1) submeshes, SP=2, TP=1.

### 3. Sequence-parallel attention (all-gather K/V)

The sequence dimension is sharded across `sp_axis`. At bucket 1024 and SP=4, each device holds 256
tokens. Token-wise ops need no change; only attention crosses shards.

Per `SmolLM3Attention.forward` when `sp_factor > 1`:

1. `qkv_proj` and `nlp_create_qkv_heads` produce local Q/K/V for this shard's 256 tokens.
2. Apply RoPE to local Q **and** local K using this shard's **global** positions
   `[rank*256, rank*256+256)`. (RoPE cos/sin are precomputed full-length and sharded on `sp_axis`
   along the seq dim, so each shard already holds its own slice — see §4.)
3. **All-gather K and V over `sp_axis`** (seq dim) → every device holds full 1024-token K/V,
   already RoPE'd. Use `ccl_manager.all_gather_persistent_buffer(..., mesh_axis=sp_axis)` (or the
   async all-gather) on the seq dim of the head-major K/V.
4. Local causal SDPA: local Q (256) vs full K/V (1024) with an **explicit rectangular causal
   bias** of shape `(1, 1, 256, 1024)` where query row `i` (global `rank*256+i`) attends key `j`
   iff `j ≤ rank*256+i`. `is_causal=True` cannot be used because Q is offset from K. Build the bias
   host-side per shard (0 / -inf) via the existing `prepare_attention_bias` path, or a small helper.
5. `concatenate_heads`, then the existing TP all-gather over `tp_axis` for O-proj — unchanged.

`sp_axis` and `tp_axis` are orthogonal mesh axes, so the two all-gathers don't interfere.

Output stays sequence-sharded on `sp_axis` through MLP/norm and all layers.

### 4. Fixed-bucket padding + SP host I/O (wrapper)

`SmolLM3TextEncoderWrapper.__init__` takes `pad_buckets: list[int] = [1024]` (sorted). Per encode:

- Tokenize at true length → `seq_len`.
- `bucket = smallest b in pad_buckets with b >= seq_len`; raise a clear error if none fits (the
  caller then adds a larger bucket, e.g. 2048). Assert `bucket % (sp_factor * 32) == 0`.
- Pad `input_ids` to `bucket`. Build full-length RoPE cos/sin for `bucket`.
- Shard `input_ids` and RoPE tensors along the seq dim across `sp_axis` when `sp_factor > 1`
  (`mesh_axes=[sp_axis, None]`-style placement); otherwise replicate (current behavior).
- Run the encoder; gather outputs back along the seq dim over `sp_axis` on `to_torch`
  (mesh-concat composer), then slice `[:, :seq_len, :]` on host as today.

Because attention is causal, padded tail positions never affect the real leading tokens, so
`attention_mask=None` (SP causal bias) stays numerically valid — same argument as the current
tile-pad path, now extended to the fixed bucket.

### 5. Concurrent CFG in `_encode` (pipeline)

```python
def _encode(self, prompt, negative_prompt, *, do_cfg=True):
    # dispatch both branches, then sync — they run on disjoint submeshes concurrently
    cond = self._text_encoders[0].encode_prompt(prompt)          # submesh 0
    uncond = self._text_encoders[1].encode_prompt(negative_prompt) if do_cfg else None  # submesh 1
    # encode_prompt already returns host tensors; the two device forwards overlap because the
    # to_torch reads are issued after both forwards are enqueued (verify enqueue ordering; if
    # encode_prompt blocks on to_torch internally, split enqueue vs. readback so they overlap).
```

The returned 4-tuple contract `(cond_embeds, cond_hidden_states, uncond_embeds,
uncond_hidden_states)` is unchanged. No CFG combine here (that's a denoise concept); each branch's
embeds are returned separately.

**Concurrency note:** to actually overlap the two branches, `encode_prompt` must enqueue the device
forward without immediately blocking on the host readback. If the current `to_torch` forces a sync,
refactor `encode_prompt` into `encode_prompt_enqueue` + `encode_prompt_readback`, or accept
sequential-on-host-but-parallel-nothing for a first cut and note it. Overlap is the point of CFG=2,
so the enqueue/readback split is part of this work, not a follow-up.

### 6. Test

`test_fibo_encode_perf` builds the full pipeline and times `pipe._encode(...)`; it needs no logic
change beyond confirming it runs on the new layout and updating the docstring/comment. Add `(4,4)`
to `_num_links`. The 2×2 param continues to work via the degradation in §2.

## Correctness / validation

- Extend `tests/encoders/smollm3/test_smollm3.py` with an SP case (e.g. `test_smollm3_encoder_sp`)
  that runs the encoder with `sp=(4,0), tp=(4,1)` on a 4×8 (and/or `sp=(2,0)` on 2×2) and asserts
  PCC vs. the torch `SmolLM3ForCausalLM` hidden states — this is where the RoPE-offset and
  rectangular-causal-bias math is proven. This gate must pass before wiring the pipeline.
- `test_fibo_encode_perf` (`-s`) reports the encode wall-clock; expected to drop vs. the TP=8
  sequential baseline (CFG branches overlap + SP splits each branch).

## Risks

- **SP causal bias / RoPE offset** is the only real algorithmic risk; the PCC test de-risks it.
- **K/V all-gather cost** per layer (36 layers) could offset SP gains at 1024 tokens; the perf test
  measures whether SP is net-positive. If not, the config can fall back to sp=1 without touching
  the attention code (guarded by `sp_factor > 1`).
- **Double weight residency** (two encoders): fine at 16 devices/submesh; note memory in review.
- **CFG overlap** depends on non-blocking enqueue (§5); if `to_torch` serializes, the CFG win is
  lost and needs the enqueue/readback split.

## Open items resolved

- Padding: fixed bucket list, start `[1024]`, extensible; error (not silent truncate) when a prompt
  exceeds all buckets. (User: "i want to be able to have multiple of them, just start with 1024".)
- SP method: all-gather K/V. (User choice.)
- Landing: shared encoder + FIBO wrapper + main pipeline `_encode`. (User: "Also update main
  pipeline".)
