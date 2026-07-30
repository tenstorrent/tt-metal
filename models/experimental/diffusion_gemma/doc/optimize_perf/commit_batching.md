# Batched commit — one causal prefill-append instead of 256 decode-appends (#47557)

Status: current. Both the batched commit and the one-op `fill_cache` KV write are the shipped
defaults; the batched-commit env override and its `batched_commit_enabled()` helper were deleted
2026-07-28, so that name now does nothing (see [flag triage](flag_triage_20260728.md)).
Owns: the sequential-vs-batched commit MoE defect and the ruling that `verify_commit_batching.py`
is an invalid gate; `ttnn.fill_cache` as a tile-aligned seq-offset write (safety argument,
head-boundary spill hazard, and the device proof that a traced fill + read observes its own write
on every replay).
See also: [refuted list](../REFUTED.md), [optimize_perf hub](README.md).

Over the 100-line cap: the MoE defect, four device-evidenced dead ends, two ttnn hazards and three
reproduction commands have no other home.

## What the commit is, and the current defaults

After each denoise block the generation loop commits the block's 256 clean-argmax canvas tokens into
the frozen Gemma4 KV cache (three-phase KV: prefill → denoise → **commit**). The **256 forwards, not
the KV writes, were the original cost**: the baseline sequential commit
(`tt/generate.py::commit_canvas_tokens`, one full 30-layer `commit_decode_forward` per token) runs
**~31.5 s per 256-token block** on QB2.

`tt/commit_batched.py::commit_canvas_tokens_batched` is the commit path, chosen by
`select_commit_fn()`; the sequential path is selected only for paged/vLLM caches (batched paged
SDPA-read is #47488). The KV write is one tile-aligned `ttnn.fill_cache` per K/V per layer;
`DG_COMMIT_KV_WRITE=position` selects the per-position reference write, and both are selectable per
call via `write_mode=`, so one process can A/B them against the identical pre-commit cache — which
is what makes an exact (`max_abs_diff == 0`) gate possible.

**Equivalence argument, in one line.** The batched commit writes algebraically identical K/V — same
absolute position `start_pos+i`, same per-head norm and rms eps, same RoPE offset, same K/V
projections, same causal visibility, same contiguous `[1, num_local_kv_heads, max_seq, head_dim]`
layout — differing only in prefill-vs-decode kernel numerics. So the correct assertion is high PCC,
never bit-identity. Sliding-window edge caveat, still unproven: the mask uses the HF causal-sliding
convention `0 ≤ q−k < window` and agrees with the decode SDPA's `sliding_window_size` only where the
window does not bite (`start_pos + C ≤ sliding_window`, i.e. ≤ 768 committed tokens at window 1024).

Measured: **~6.3x** on the commit itself vs the 256 sequential decode-appends (1031 ms vs 6503 ms at
L=6), and end-to-end (`serving_smoke`, full 30L, 24-step) sequential 49.94 s/block = 5.13
tok/block/s vs batched **19.64 s/block = 13.04** — a **2.54x** block speedup that is also a
correctness fix.

## The defect: the sequential commit MoE is wrong

Against a torch MoE oracle (hand-rolled to match HF `DiffusionGemmaTextRouter` +
`DiffusionGemmaTextExperts`, fp32, real layer-0 checkpoint weights, on the **identical bit-exact**
layer-0 commit MoE input the device computed):

| MoE output vs torch oracle | PCC |
|---|---|
| batched commit (`moe.experts` / `sparse_experts_forward`) | **0.9936** |
| sequential commit (`_commit_experts_decode_forward`, decode `sparse_matmul` nnz=8) | **0.1542** |

So the sequential decode-commit MoE is genuinely **defective** and the RUN commit path may be
writing slightly-wrong prefix KV. Layer-0 localisation (`probe_commit_l0attn.py`) puts the
divergence in the expert **kernel**, not routing: attention output 0.99992, shared_mlp input
0.99994, shared_mlp output 0.99975, router output 0.98097 with 99.6% expert-mask agreement and both
nnz=8, MoE expert output **0.16551**.

> **MEASUREMENT TRAP.** `verify_commit_batching.py` (batched-vs-sequential KV PCC, worst **0.43** at
> L=6 against a 0.997 threshold) is a **NON-GATE**: it compares against the defective sequential
> reference, so its FAIL is expected and meaningless. `probe_moe_vs_torch.py` (batched-vs-torch) is
> the correct gate. Three other candidates for that 0.43 were ruled out with device evidence — see
> [refuted list](../REFUTED.md).

Two bugs had to be fixed to make the batched path run at all, both worth knowing:
`paged_update_cache` requires a HEIGHT_SHARDED one-core-per-user update tensor with shard width ==
`head_dim`, so the DRAM-interleaved per-position canvas K/V slice tripped
`input_tensor.is_sharded()` (fixed by resharding each `[1,1,nkv,hd]` slice onto one core with the
tile-padded shard); and an unconditional `ttnn.to_memory_config(tt_q, DRAM)` on an already-DRAM
tensor returned a fresh **unallocated** alias that killed the SDPA input (fixed with a guarded
`if buffer_type != DRAM` convert).

## The KV write: one `fill_cache` instead of ~1536 ops

Where commit time went before the change: at L=8, full 266.1 ms / backbone 123.2 / KV write 142.9
(**53.7%**); at L=16, 495.1 / 241.9 / 253.2 (**51.1%**) — per layer write ~13.8 ms vs backbone
~14.9 ms. The per-position write was almost pure host dispatch: 256 iterations × (2 slice + 2
reshard + 2 `paged_update_cache`) ≈ **1536 tiny `[1,1,nkv,hd]` ops per layer** with essentially no
arithmetic.

| op-level (QB2 Blackhole, 11x10 grid, nkv=2, hd=256, C=256, 20 warm reps) | per layer (K+V) |
|---|---|
| per-position (256 × 6 ops) | 9.591 ms |
| one `fill_cache` per K/V | **0.012 ms** |

The ~52% write share collapses to ~0.1% of the commit. **Real-backbone gate PASS** (2026-07-24, QB2
1x4, full 30L): `commit_ms` 905.1 → 464.0 = **1.95x** with whole-cache `max_abs_diff = 0.0` across
30 layers × K/V × 4 device shards. Across three runs the ratio is **1.85x–1.95x** (905.1→464.0 and
882.4→478.2 warmed; 970.7→520.8 when the first-timed mode absorbs the cold program cache).

**Why `fill_cache` is safe here:** FILL is pure data movement (reader → CB →
`writer_unary_interleaved`, **no compute kernel**, unlike `paged_update_cache`'s
untilize/patch/tilize), so bf16 values cannot be perturbed, and it never touches the frozen prefix
`[0, start_pos)` or the tail. The write span is tile-aligned by construction, so no partial-tile
read-modify-write is needed.

> **TTNN HAZARD, no validator enforces it.** `fill_cache`'s interleaved path splits `nkv·(S/32)`
> tile-rows over the core grid and each core writes its rows contiguously from one `cache_start_id`,
> **assuming no core's range crosses a kv-head boundary**. Device-confirmed on QB2 that `nkv=8,
> C=1024, max_seq=2048` (256 rows > 110 cores) silently writes **49106 wrong elements**, while the
> same geometry with `C == max_seq` is fine. DiffusionGemma is inside the safe region
> (`nkv_local ≤ 2`, `C = 256` ⇒ ≤ 16 rows vs 110 cores), and `_fill_write_unsupported_reason`
> checks the spill plus dtype equality (FILL refuses to convert, unlike `paged_update_cache`),
> head/head_dim match, tile alignment, span bounds and cache batch, falling back once-with-a-log to
> the per-position write.

Upstream follow-ups worth filing: (a) `fill_cache` should `TT_FATAL` on the head-boundary spill
instead of silently corrupting; (b) gemma4's long non-paged prompt prefill (`nkv=2, S ≥ 8192` at
`S < max_seq`) can cross the same threshold.

**Device proof that a traced fill is replay-safe** (`tests/test_device_fill_cache_in_trace.py`, 3
passed on P150x4): eager `fill_cache(update_idx=p_max)` writes only the tail with the prefix
byte-identical; a traced fill + read replayed twice with different canvas contents has each replay
observe ITS OWN write, prefix intact, replays differing, cache `buffer_address()` stable; and a
traced fill + SDPA over the same tensor gives pcc > 0.99 vs a torch reference on both replays with
outputs differing across replays. So a Metal trace MAY write a scratch tail and read it back within
the same capture, and refreshing only the CONTENTS of a pre-capture input buffer is enough to drive
it. That run also warned `Allocating device buffers is unsafe due to the existence of an active
trace` because the test allocates its output inside the capture — it passed, but real wiring must
pre-allocate every output buffer BEFORE `begin_trace_capture`; do not copy the test's allocation
pattern into the model.

Two guard-design rules. **Cache batch > 1 must NOT fall back to the per-position write** —
`fill_cache` with `batch_idx=0` is legal on a multi-slot cache, but non-paged `paged_update_cache`
asserts `num_users == cache batch`, so a blanket fallback would turn a working write into a
`TT_FATAL`. **A ragged span is a contract violation, not a fallback** — `start_pos % 32` and
`canvas_len % 32` both raise in `commit_canvas_tokens_batched` before any device work, because a
ragged `canvas_len` would push tile-pad K/V past `start_pos+canvas_len` with the cache already dirty.

## TTNN aliasing trap fixed in passing

A full-span `ttnn.slice` (no_step, starts 0, ends max) short-circuits to an **ALIAS** of the input
(`slice.cpp` `finalize_into_preallocated(ret_adjustment(input_tensor))`), so deallocating the read
result frees the **KV cache itself** when a committed block ends exactly at `max_seq`.
`_read_cache_kv` now `ttnn.clone`s at that boundary, pinned by
`tests/test_device_commit_kv_write.py::test_cache_read_at_max_seq_does_not_alias_the_cache`. The
sibling `to_memory_config` aliasing trap is documented in
[per-layer prefix spans](per_layer_prefix_spans.md).

## Reproduction

env: see [plan](../../plan.md).

```bash
# Op-level, checkpoint-free, ~4 s — whole-cache bit-identity of the two write mechanisms,
# plus a torch oracle and the head-boundary spill guard:
DG_RUN_DEVICE=1 pytest models/experimental/diffusion_gemma/tests/test_device_commit_kv_write.py

# Real backbone — batched commit with fill vs with per-position, whole-cache exact:
DG_CKPT=/home/zni/dg_models/diffusiongemma-26B-A4B-it \
  python models/experimental/diffusion_gemma/doc/optimize_perf/verify_commit_kv_write.py \
  --mesh 1x4 --num-layers 30 --max-seq-len 1024 --prompt "The capital of France is"

# The correct MoE gate (batched vs torch):
DG_CKPT=/home/zni/dg_models/diffusiongemma-26B-A4B-it \
  python models/experimental/diffusion_gemma/doc/optimize_perf/probe_moe_vs_torch.py
```

Artifacts: `artifacts/leverB_verified_{batched,seq}_30L_s24.json` are present;
`artifacts/leverB_moe_vs_torch_L0.log` and the `.log` siblings are cited by the source record but
are not in the tree. `verify_commit_batching.py` and
`probe_attn_only.py` gained a `--kv-write` flag but still default to `position`, so their existing
batched-vs-sequential comparisons keep using the proven write.

Bit-exactness is not an achievable gate for either arm — see the bf16 chaos-amplification class in
[decision fidelity](../decision_fidelity/README.md).
