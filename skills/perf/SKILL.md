---
name: perf
description: Pipeline-level perf for a use case (paged_update_cache + reusable metal trace + targeted tracy on the traced path). Use when optimizing end-to-end generation latency/throughput.
---

# SKILL: Pipeline Perf (paged_update_cache + reusable trace + targeted tracy)

## Purpose
Pipeline-level perf optimization for one use case. Distinct from
`skills/optimization/` (which is per-block); this skill covers cross-block
refactors and integrated-pipeline tuning. Two sub-passes: a STRUCTURAL pass that
unlocks a single reusable metal trace by switching the AR loop onto
`paged_update_cache` + persistent buffers, then a TARGETED pass that captures
tracy data and applies ONE optimization driven by what the profile actually
shows.

## When to use
- After a use case's generation phase is done (e2e test passes, HF parity
  holds on the known-good sample set).
- Before declaring the model "ready" for deployment.
- Re-run for each use case separately. The infrastructure (`kv_cache.py`,
  `<use_case>_generator.py`) is shared, but the trace and the tracy CSV are
  per-use-case — encoder shapes, decode budgets, and hot ops differ.

## Why this skill exists (and not just skills/optimization/)
Per-block optimization can tune one matmul's kernel config or shard one
`ttnn.linear`. The wins documented here REQUIRE touching multiple files at
once — the KV cache, the AR generator, and the cached-attention path inside
the attention block — and the correctness invariants span the whole AR loop,
not one block. Per-block work and pipeline work are NOT interchangeable:
- A block-level optimization shows up as a faster single op in isolation.
- A pipeline optimization shows up as fewer host dispatches per step OR as
  a smaller total kernel time across the whole AR loop.

If you find yourself editing one attention block to optimize it in isolation,
that's the optimization skill. If you find yourself editing `kv_cache.py`
together with `<use_case>_generator.py` together with the cached-attention
path inside the attention block, all to enable trace replay, that's THIS
skill.

## Prerequisites
- `use_cases.<name>.generation.status=done` — e2e test passes, HF parity
  holds. Without an HF-parity baseline you can't verify that perf work has
  not broken correctness.
- The block-level optimization skill has been run for the per-block
  primitives the use case touches (SDPA layouts, matmul `program_config`,
  L1 sharding choices). Do not re-tune blocks here.

## Process

### Sub-pass 1: structural — paged_update_cache + reusable metal trace

Skip this entire sub-pass if `needs_ar=false`. Encoder-only pipelines have
no AR loop to trace; the relevant perf knob is batching, not tracing.

The single biggest pipeline win is a single metal trace captured ONCE and
replayed for every decode position and every `generate()` call. That trace
captures the entire decode step (token embed → 24 layers of self-attn +
cross-attn + FFN → final LN → LM head) and amortises the per-op host
dispatch cost across all replays.

To get there you need three things to be true:

1. The KV-cache update reads the target position from device memory at
   replay time, not from a Python int baked into the kernel args at capture
   time.
2. Every per-step input (token id, position id, self-attn mask) flows
   through a PERSISTENT device buffer whose address is captured once and
   reused.
3. Every per-call input (cross-attn KV cache, encoder mask) sits in a
   persistent buffer that is OVERWRITTEN in place across `generate()`
   calls, never freed and reallocated.

Each of these is a small refactor on its own. Together they unlock
single-trace replay.

#### The trace pitfall

```python
# WRONG — bakes pos into the captured trace.
ttnn.update_cache(cache, k_new, update_idx=int(pos))
```

`ttnn.update_cache(update_idx=int_pos)` captures `int_pos` as a kernel
constant when the trace is recorded. Replaying the trace always writes to
that same position — every decode step after the captured one corrupts
slot 0 instead of advancing the cache. Single-trace replay across positions
is impossible.

#### The fix

```python
# RIGHT — reads pos from a device tensor at replay time.
ttnn.experimental.paged_update_cache(
    cache, k_new_sharded, update_idxs_tensor=cur_pos_tt,
)
```

`paged_update_cache` takes the position as a 1-element int32 device
tensor. Use a SINGLE logical page (no actual paging — the cache is just a
contiguous `[batch, num_heads, max_seq, head_dim]` buffer); the "paged"
part is overkill for non-batched decode but the op variant is the one
that accepts a tensor-valued position.

The cache wrapper that does this looks like:

```python
class SelfAttentionKVCache:
    def __init__(self, device, num_layers, batch, num_heads,
                 max_seq_len, head_dim, dtype=ttnn.bfloat16):
        # Per-layer [B, num_heads, max_seq, head_dim] in DRAM TILE_LAYOUT.
        zeros = torch.zeros(batch, num_heads, max_seq_len, head_dim,
                            dtype=torch.bfloat16)
        self.k_caches = [
            ttnn.from_torch(zeros, device=device, dtype=dtype,
                            layout=ttnn.TILE_LAYOUT,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG)
            for _ in range(num_layers)
        ]
        self.v_caches = [...]  # same

        # Persistent int32 [batch] position buffer. STABLE address — the
        # captured trace reads from this at replay time.
        self._persistent_pos_tt = ttnn.from_torch(
            torch.zeros(batch, dtype=torch.int32),
            device=device, dtype=ttnn.int32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Pre-built sharded memory config for paged_update_cache's
        # K/V input: [1, B, num_heads, head_dim] HEIGHT_SHARDED on L1.
        grid = device.compute_with_storage_grid_size()
        shard_grid = ttnn.num_cores_to_corerangeset(batch, grid, row_wise=True)
        shard_shape = (nearest_y(num_heads, ttnn.TILE_SIZE), head_dim)
        self._update_input_mem_cfg = ttnn.create_sharded_memory_config(
            shape=shard_shape, core_grid=shard_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            use_height_and_width_as_shard_shape=True,
        )

    def update(self, layer_idx, k_new, v_new, pos):
        # Resolve int -> persistent device tensor.
        if isinstance(pos, ttnn.Tensor):
            pos_tt = pos
        else:
            host = ttnn.from_torch(
                torch.tensor([int(pos)] * self.batch, dtype=torch.int32),
                dtype=ttnn.int32,
            )
            ttnn.copy_host_to_device_tensor(host, self._persistent_pos_tt)
            pos_tt = self._persistent_pos_tt

        # K/V leave the projection as [B, num_heads, 1, head_dim] TILE
        # in DRAM. paged_update_cache wants [1, B, num_heads, head_dim]
        # HEIGHT_SHARDED on L1. Reshape + interleaved_to_sharded here.
        k_r = ttnn.reshape(k_new, (1, self.batch, self.num_heads, self.head_dim))
        v_r = ttnn.reshape(v_new, (1, self.batch, self.num_heads, self.head_dim))
        k_s = ttnn.interleaved_to_sharded(k_r, self._update_input_mem_cfg)
        v_s = ttnn.interleaved_to_sharded(v_r, self._update_input_mem_cfg)
        ttnn.deallocate(k_r); ttnn.deallocate(v_r)

        ttnn.experimental.paged_update_cache(
            self.k_caches[layer_idx], k_s, update_idxs_tensor=pos_tt,
        )
        ttnn.experimental.paged_update_cache(
            self.v_caches[layer_idx], v_s, update_idxs_tensor=pos_tt,
        )
        ttnn.deallocate(k_s); ttnn.deallocate(v_s)
```

#### Persistent buffers

Allocate ONCE at generator init; overwrite per step via
`ttnn.copy_host_to_device_tensor`. Required set:

- `_persistent_input_ids_tt` — uint32 ROW_MAJOR DRAM `[1, 1]`. New
  token id per step.
- `_persistent_position_ids_tt` — uint32 ROW_MAJOR DRAM `[1, 1]`.
  Position used to gather into the sinusoidal / rotary table.
- `_persistent_self_mask_tt` — bf16 TILE DRAM `[1, 1, 1, max_seq]`.
  Pre-built per-position host tiles uploaded per step. Hides slots
  beyond the current cache fill with additive `-inf`.
- `_persistent_encoder_mask_tt` — bf16 TILE DRAM `[1, 1, 1,
  enc_seq_total]`. Invariant across the whole `generate()` call;
  rebuild per generate() and `copy_host_to_device_tensor` it into
  the persistent buffer.
- `self_attn._persistent_pos_tt` — int32 DRAM `[batch]`. Already
  defined in the cache class above. Drives `update_idxs_tensor`.

Cache the per-position host tensors at `_ensure_persistent_buffers`
time so the per-step path is JUST a `copy_host_to_device_tensor` (no
`from_torch` + tilize allocations on the hot loop):

```python
self._self_mask_hosts = {}
for p in range(max_seq_len):
    m = self.text_decoder._build_decode_self_attention_mask(
        batch=1, position=p, max_seq_len=max_seq_len, dtype=torch.float32,
    )
    self._self_mask_hosts[p] = ttnn.from_torch(
        m, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
    )
```

#### The hot loop

After the structural refactor, the steady-state per-step path is just
four H2D writes plus one trace execute:

```python
# Per-step inputs.
self._write_input_ids(next_token)       # H2D into persistent input-id buf
self._write_position_id(pos)            # H2D into persistent pos-id buf
self._write_cache_pos(pos)              # H2D into self_attn._persistent_pos_tt
self._write_self_mask(pos)              # H2D into persistent self-mask buf

# Replay the captured trace. All inputs read from the persistent buffers;
# all outputs land in self._decode_trace_output_tt.
ttnn.execute_trace(self.device, self._decode_trace_id,
                   cq_id=0, blocking=True)

# Read logits and argmax on host.
logits = ttnn.to_torch(self._decode_trace_output_tt).to(torch.float32)
next_token = int(torch.argmax(logits).item())
```

`ttnn.copy_host_to_device_tensor(host_tile, dst_persistent_tt)` is the
key primitive — it writes into the existing device buffer without
allocating a new one, so the captured trace continues to read from the
right address.

#### KV cache reset between generate() calls

Captured traces hold pointers to the cache buffers. Resetting a cache
must NOT free and reallocate the device tensors — that would invalidate
the trace. Instead, stream a pre-built host-side zero tensor into each
buffer:

```python
class SelfAttentionKVCache:
    def reset(self):
        for i in range(self.num_layers):
            ttnn.copy_host_to_device_tensor(self._zero_host, self.k_caches[i])
            ttnn.copy_host_to_device_tensor(self._zero_host, self.v_caches[i])

class CrossAttentionKVCache:
    def populate(self, layer_idx, k, v):
        # Overwrite the persistent slot via fill_cache. Do NOT replace
        # the tensor handles — the captured trace points at the old ones.
        ttnn.fill_cache(self.k_caches[layer_idx], k, 0)
        ttnn.fill_cache(self.v_caches[layer_idx], v, 0)
        ttnn.deallocate(k); ttnn.deallocate(v)
        self._populated[layer_idx] = True

    def reset(self):
        # No-op on the buffers themselves; just flip the flags.
        for i in range(self.num_layers):
            self._populated[i] = False
```

#### Trace lifecycle

Capture ONCE at the end of the first `generate()` warmup. Reuse for every
subsequent step AND every subsequent `generate()` call:

```python
def _generate_traced(self, tokens, max_total, eos_token_id, ...):
    self._ensure_persistent_buffers()

    # Position 0: untraced warmup. Logits discarded. Compiles all
    # kernels into the program cache and the captured trace.
    self._write_input_ids(tokens[0]); self._write_position_id(0)
    self._write_cache_pos(0); self._write_self_mask(0)
    if not self._decode_kernels_compiled:
        wu = self._run_decode_body(position=0, ..., use_pos_tensor=True)
        ttnn.synchronize_device(self.device); ttnn.deallocate(wu)
        self._decode_kernels_compiled = True
    else:
        ttnn.execute_trace(self.device, self._decode_trace_id,
                           cq_id=0, blocking=True)  # second-call warmup

    # Capture exactly once.
    if self._decode_trace_id is None:
        ttnn.synchronize_device(self.device)
        trace_id = ttnn.begin_trace_capture(self.device, cq_id=0)
        try:
            logits_tt = self._run_decode_body(position=0, ...,
                                              use_pos_tensor=True)
        finally:
            ttnn.end_trace_capture(self.device, trace_id, cq_id=0)
        ttnn.synchronize_device(self.device)
        self._decode_trace_id = trace_id
        self._decode_trace_output_tt = logits_tt

    # Steady state: write inputs, replay, argmax.
    for pos in range(1, max_total):
        self._write_input_ids(tokens[-1]); self._write_position_id(pos)
        self._write_cache_pos(pos); self._write_self_mask(pos)
        ttnn.execute_trace(self.device, self._decode_trace_id,
                           cq_id=0, blocking=True)
        logits = ttnn.to_torch(self._decode_trace_output_tt).to(torch.float32)
        next_token = int(torch.argmax(logits).item())
        tokens.append(next_token)
        if next_token == eos_token_id: break
```

Open the device with `trace_region_size=256_000_000` (or larger if the
trace dump indicates overflow). The default trace region is too small
for a 24-layer decoder.

### Sub-pass 2: targeted — tracy + ONE optimization

After sub-pass 1 lands and the e2e test still passes, run tracy on the
integrated pipeline and apply EXACTLY ONE additional optimization based
on what the profile shows. Resist the temptation to apply three at
once — the second one's win is unverifiable when the first one moved
the bottleneck.

#### Tracy harness skeleton

One file per use case: `tt/profile_<use_case>.py`. Pattern:

- 1 warmup `<verb>()` call followed by N timed `<verb>()` calls.
- Wrap the AR step with a `step_callback(position, ms, kind)` hook on
  the generator so per-step latency is bucketed by `kind` in
  `{"warmup", "capture", "replay"}`. Only the `"replay"` entries are
  steady-state.
- Open the device with the same `trace_region_size` you used in
  production.

```python
def main():
    args = ... # argparse: --src, --tgt-lang, --max-new-tokens,
              # --traced, --num-timed, --device-id

    device = ttnn.open_device(
        device_id=args.device_id, l1_small_size=32768,
        trace_region_size=256_000_000 if args.traced else 0,
    )
    try:
        model = <UseCaseModel>(device=device, hf_state_dict=hf_sd,
                                processor=processor)

        # Warmup: compiles kernels, captures trace if --traced.
        warmup_text, _ = _run_one(model, ..., use_trace=args.traced)

        # Timed: median across N calls is the steady-state report.
        totals, step_logs, prefill_logs = [], [], []
        for _ in range(args.num_timed):
            t0 = time.perf_counter()
            text, _ = _run_one(model, ..., use_trace=args.traced,
                               per_step_log=step_logs.append,
                               prefill_log=prefill_logs.append)
            totals.append((time.perf_counter() - t0) * 1000.0)

        # Split steady-state from warmup/capture; report medians.
        steady = [ms for ms, kind, _ in step_logs if kind == "replay"]
        print(f"SUMMARY prefill_ms={median(prefill_logs):.2f} "
              f"steady_decode_step_ms={median(steady):.2f} "
              f"total_ms={median(totals):.2f}")
    finally:
        ttnn.close_device(device)
```

Run it under tracy with the standard flags:

```bash
python -m tracy -p -v -r --op-support-count 3000 \
    --dump-device-data-mid-run -n <use_case> \
    models/demos/<model>/tt/profile_<use_case>.py --traced
```

Then read `generated/profiler/.logs/tracy_ops_data.csv` and bucket by
op-code and by `memory_config.memory_layout` / `buffer_type`. A handful
of dispatched ops will dominate.

#### Host-vs-device bound triage

Tracy gives you both per-op device kernel time and host-side zone time.
Use the relationship between them to choose where to attack:

```
total_ms ≈ steady_decode_step_ms * num_decode_steps + prefill_ms
kernel_time_sum = sum(per_op_device_kernel_time)   # from tracy CSV
```

Decision tree:

- **`total_ms < kernel_time_sum * num_steps + small_host_margin`** →
  This is the "still host-bound under tracing" case. Tracing has not
  fully amortised dispatch yet. Verify the trace is actually replaying
  on every step (the `step_callback` should show `kind="replay"` for
  positions 2..N). If it is, check that no per-step host-side work is
  sneaking in — `ttnn.from_torch`, mask rebuilds, fresh tensor
  allocations.

- **`total_ms ≈ kernel_time_sum * num_steps`** → Device-compute limited.
  Tracing has done its job; the floor is now kernel time. Pick ONE of
  the following based on which op dominates the tracy bucket:
  - Hottest matmul → revisit its `program_config` or shard it. See
    `skills/optimization/SKILL.md` for the kernel-config search.
  - Hot LN + Linear chain → consider fusing them, IF the block supports
    fused LayerNorm-Matmul on this arch.
  - Non-PCC-sensitive matmul → try `bfloat8_b` weights or
    `bfloat8_b` intermediate output. Re-run the block's PCC test
    AND the e2e test. If either fails, revert.

- **`steady_decode_step_ms` is a small fraction of `total_ms`** →
  Per-call overhead (encoder, cross-attn cache populate, EOS detection
  loop in Python) dominates. Trace the prefill phase too, or batch
  multiple `generate()` calls.

The decision tree is cheap to evaluate; the optimization itself can be
hours of work. Spend the up-front time on the triage.

#### Apply exactly one optimization

After triage, pick the single highest-leverage change for the profile
you have, NOT for the profile you expect. Common patterns:

- **Largest matmul L1-sharded** — convert it from DRAM-input
  interleaved to L1 height-sharded if the input fits.
- **Hot LayerNorm-Linear fusion** — replace the `ttnn.layer_norm` +
  `ttnn.linear` pair with the fused variant (where available).
- **Lower precision on a non-PCC-sensitive matmul** — `bfloat8_b`
  weights for the LM head can win N ms with no parity drop.

Implement, then VALIDATE: re-run the e2e test (HF parity preserved) and
re-run the per-block PCC test for whatever block you touched (PCC > 0.99
still holds). One change at a time means one number to compare against
the post-sub-pass-1 baseline.

## Reality check

Trace + reusable trace delivered **1.21× on SeamlessM4T-v2** end-to-end
(translation throughput). The hypothesis going in was "tracing buys
~50% because dispatch is the bottleneck." It was wrong for that model:

- Pre-trace `steady_decode_step_ms ≈ 20.88 ms`.
- Post-trace `steady_decode_step_ms ≈ 17.62 ms`.
- That's ~16% per step. Net wall-clock on a 9-token translation was
  closer to 1.21× because per-call overhead (encoder forward, cache
  populate, host argmax loop) didn't move.

The floor turned out to be device kernel time, not host dispatch.
~365 dispatches per step × ~50 µs/op overhead = ~18 ms of dispatch,
but kernel time was already in the same range; tracing only takes
away the dispatch slice. Further wins on that model need compute-side
attacks — sharded matmuls, LN fusion, possibly fp8 on the LM head.

Honest characterization matters more than aspirational targets.
"Trace gives 1.21× on this model; the next 1.5× needs compute work" is
a useful report. "Trace gives 50% (theoretical)" is not.

Write down both numbers — baseline and after — in `PERF_NOTES.md`
regardless of the multiplier. The pipeline-perf phase exists to make
the perf surface honest, not to chase a target.

## Output artifacts

- Modified `tt/kv_cache.py` — `SelfAttentionKVCache.update` uses
  `paged_update_cache(update_idxs_tensor=pos_tt)`; `reset()` streams
  a host zero tensor in place; cross-attn cache `populate()` uses
  `fill_cache` against pre-allocated buffers.
- Modified `tt/<use_case>_generator.py` — `_ensure_persistent_buffers`
  allocates the per-step buffer set; `_run_decode_body` reads from
  those buffers; `_capture_decode_trace` captures ONCE; `_generate_traced`
  replays. The `step_callback` hook is exposed for the profiler.
- Modified cached-attention path inside the attention block — accepts a
  `cur_pos_tensor` and the sharded K/V input layout that
  `paged_update_cache` requires.
- `tt/profile_<use_case>.py` — tracy harness, 1 warmup + N timed,
  splits replay/capture/warmup costs in its report.
- `PERF_NOTES.md` — Markdown with: setup (device, arch, branch SHA,
  prompt), baseline table (`prefill_ms`, `steady_decode_step_ms`,
  `total_ms`, `tokens_generated`), tracy findings (top ops by call
  count, top host zones), after-trace table, the one targeted
  optimization applied with before/after numbers, recommendations
  for the next pass.
- One row in `BRINGUP_LOG.md` under the `use_cases.<name>.perf`
  field, with the measured speedup and a pointer to `PERF_NOTES.md`.

## Failure modes

- **`paged_update_cache` shape mismatch.** The op expects K/V input as
  `[1, B, num_heads, head_dim]` HEIGHT_SHARDED on L1 with shard width
  == `head_dim` and shard height == `ceil(num_heads / TILE) * TILE`.
  The output of your `k_proj` / `v_proj` is typically
  `[B, num_heads, 1, head_dim]` TILE DRAM. The cache wrapper does the
  reshape + `interleaved_to_sharded`; don't try to skip it. If the
  reshape errors, check whether your batch dim leads or trails — TT
  conventions vary across attention blocks.

- **Trace capture fails on a specific op.** Some ops historically did
  not work inside trace (anything that allocates, anything that does
  D2H sync). Back out cleanly: keep the structural changes (persistent
  buffers, `paged_update_cache`), revert just the trace capture, and
  document which op was incompatible in `PERF_NOTES.md`. The structural
  changes alone usually still buy ~5-10% by amortising host-side
  allocation.

- **Trace replays but logits diverge from untraced.** A captured op is
  reading from a buffer whose address moved. Audit every input to
  `_run_decode_body`: anything passed by Python int OR by a
  freshly-allocated tensor breaks trace reuse. Common offenders: the
  encoder mask (must go through the persistent buffer, not be returned
  by a builder), the position id for sinusoidal lookup, the KV-cache
  update position. Cross-reference the persistent-buffer set in
  `_ensure_persistent_buffers`.

- **First `generate()` works, second one diverges.** The cross-attn
  KV cache is the most common culprit. `populate()` must OVERWRITE the
  persistent buffers via `fill_cache`, not allocate fresh ones. Same
  for `reset()` — it must not deallocate. If you see "pos 2 matches,
  pos 3 diverges" across two calls, you almost certainly reallocated
  somewhere in the prefill path.

- **Tracy shows no clear hot op.** Two flat ops at 8% each instead of
  one op at 40%. "Characterized, no single optimization yielded > 5%
  win" is a legitimate outcome — write it down in `PERF_NOTES.md` and
  stop. Forcing a "win" by tuning two ops at once is how you regress
  PCC without realizing it.

- **Device kernel time is the floor and matmul is already sharded.**
  You're at the per-block optimization frontier. Hand the work back to
  `skills/optimization/` — pipeline-level moves are exhausted. Document
  the floor in `PERF_NOTES.md` and the deferred matmuls.

## Reference implementation

- `models/demos/facebook_seamless_m4t_v2_large/tt/kv_cache.py` —
  `SelfAttentionKVCache.update` uses
  `paged_update_cache(update_idxs_tensor=pos_tt)` against a single
  contiguous cache; persistent `_persistent_pos_tt` buffer holds the
  position. `CrossAttentionKVCache.populate` writes encoder K/V into
  pre-allocated persistent buffers via `fill_cache`. ~400 lines.
- `models/demos/facebook_seamless_m4t_v2_large/tt/text_generator.py` —
  `_ensure_persistent_buffers` pre-allocates the input-id / position-id
  / self-mask / encoder-mask buffers AND caches per-position host
  tensors so the per-step path is just `copy_host_to_device_tensor`.
  `_capture_decode_trace` captures one trace replayed across all
  positions and all `generate()` calls. `_generate_traced` is the hot
  loop. ~870 lines.
- `models/demos/facebook_seamless_m4t_v2_large/tt/profile_t2tt.py` —
  tracy harness; 1 warmup + N timed; `step_callback` splits
  `replay` / `capture` / `warmup` per-step costs; reports
  `prefill_ms`, `steady_decode_step_ms`, `total_ms`. ~390 lines.
- `models/demos/facebook_seamless_m4t_v2_large/PERF_NOTES.md` —
  baseline table (`steady_decode_step_ms = 20.88 ms`), tracy
  findings (top ops + top host zones), the one-optimization pass
  (precomputed cross-attn mask, ~wash on the short prompt), the
  phase-9b metal-trace pass (`steady_decode_step_ms = 17.62 ms`,
  ~1.21× e2e), and the deferred work list. ~310 lines. Read this
  end to end before starting on a new model — the "why metal trace
  was deferred" section is the canonical write-up of the
  `update_cache(int_pos)` pitfall.

## Cross-references

- `skills/optimization/SKILL.md` — per-block performance. Pipeline-perf
  consumes its output (sharded matmuls, fused LN, kernel configs) and
  composes them; sub-pass 2 in this skill MAY discover a hot op that
  belongs in the optimization skill. Hand it back rather than retuning
  here.
- `skills/generation/SKILL.md` — produces the AR loop and KV cache
  this skill optimizes. The "do NOT rebuild the AR loop without
  `paged_update_cache` + persistent buffers" guidance in that skill
  is enforced here.
- `skills/debug/SKILL.md` — for the "trace replays but logits diverge"
  case, the Mode B chained PCC procedure is the right diagnostic.
- `models/common/sampling/tt_sampling.py` — on-device sampling
  primitives. When the host argmax loop becomes the bottleneck (rare
  but happens on very-short use cases) the next step is to fold
  sampling into the captured trace.
