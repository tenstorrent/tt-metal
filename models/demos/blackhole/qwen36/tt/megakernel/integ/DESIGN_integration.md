# Megakernel ↔ vLLM integration (design v0, 2026-11-10)

Branch `anatarajan/qwen38-r16-mk-integ2`, off `anatarajan/qwen38-r16-megakernel` @0a59bea441f (spec v0.8.4).
Scope: how the one-generic_op-per-verify-pass megakernel goes into `models/demos/blackhole/qwen36` decode, as the
vLLM TT plugin captures and replays it. Everything is behind `QWEN36_R16_MEGAKERNEL`; flag off means today's
op-by-op path, byte for byte.

## 1. Where it plugs in

The traced verify body is `Qwen36Model._forward_verify_bucket_tp(n_users, bk, kv_split)` in `tt/model.py`, captured
per row bucket (n_users ∈ 1..B) by `spec_engine` and replayed by `verify()` → `model.verify_traced`. Today it runs:

```
x = embd(token_buf) → reshape → [VERIFY_DAR enter: AG → replicated RES40 residual]
for layer in layers:  attention (spec SDPA, KV write) | GDN (recurrent verify, state ring, conv sel, ctrl)
_spec_tail(x)  → final norm → LM head (sharded) → argmax rows / ids → drafter feed rows (+ normed, shard_logits)
```

With the megakernel, **only the layer loop is replaced**:

```
x = embd(...) → [DAR enter]                         (unchanged, op-by-op)
x = mk.forward(x, ctx)                              ONE ttnn.generic_op (MeshProgramDescriptor, all 8 TP devices)
_spec_tail(x)                                       (unchanged: final norm, LM head, argmax, drafter feed)
drafter (MTP) step(s)                               (unchanged, op-by-op traces)
```

Why this cut:
- The embedding, LM head and argmax are vocab-bound and already tuned (sharded argmax, device top-k).
- The drafter has its own traces (`_dr_trace`, MTP warm) and stays op-by-op; it reads the verify's drafter-feed rows
  and `normed`.
- The megakernel's contract is residual in → residual out, in the same replicated RES40 layout `_spec_tail` expects
  (`_spec_tail_replicated`).

`mk.forward` must be trace-capturable: fixed-address inputs and outputs, no host round-trips, and the phase table is
a persistent device tensor per bucket (§3).

## 2. Inputs, outputs, state (per replay)

| Item | Today (source) | Megakernel |
|---|---|---|
| x residual in | `embd` → DAR enter, RES40 replicated | CB_X seed. The engine's first phase is the layer-0 in-proj; the first AR's x-delivery contract also applies to the seed (owners in-row, x10-11 rows 0-7) |
| x residual out | last layer output | written to a persistent DRAM tensor (RES40 replicated) that `_spec_tail` reads |
| kvpos / cos / sin / ctrl | `_vfy_inputs(rows)` (packed aux) | passed as tensor addresses in runtime args; the SDPA phase (RoPE, KV write) and the GDN phases (ctrl: accepted-draft index `mi`, parity, rope delta) read them in-kernel |
| page tables | `_vfy_kvpt_buf` (rows), `_vfy_kvpt_users`, `_vfy_kvpt1_buf` (groups), `_vfy_kvpt_own` (kv-split) | the SDPA phase reads the per-group table (word 3 = table base) and the KV-write table. Same tensors, same host staging (`_vfy_page_table_host`) |
| KV cache | per full-attention layer k/v paged BFP8 tensors | same tensors, addressed per layer from a pointer list (one 16-u32 descriptor per attention layer: k base, v base, page size, block size) |
| GDN state | ring `state_blk_idx`, conv selectors `conv_sel`, stash (K4 LAZY) | same buffers. GDN_PREFIX keeps the compact conv window (spec §6 v0.5 rule a); GDN_STEP / GDN_STEP_U1 do the replay/WY on the ring. **Rollback is unchanged**: the step writes the per-candidate stash, and the next replay selects the committed position via `mi` in ctrl |
| logits / ids | `_spec_tail` | unchanged |

Nothing on the host changes between replays except what `spec_engine` already stages (tokens, aux, page tables).
The phase table is constant per bucket.

## 3. Buckets, users, rows

- Verify rows R = n_users × T (T = K+1 = 4). Buckets (`QWEN36_SPEC_BUCKETS`) capture one trace per n_users. The
  megakernel gets **one phase table + one program per bucket**, captured into that bucket's trace.
- Row-compact rule (spec v0.8/v0.8.2): AR and on-chip activation transfers move only R valid rows (R = 4·n_users);
  tiles stay 32 rows for compute. The bucket's R is a compile-time arg of the AR/x-delivery kernels, so each bucket
  is its own program.
- GDN step kernel per bucket: n_users == 1 uses `GDN_STEP_U1` (type 11); n_users > 1 uses GDN_PREFIX (10) +
  GDN_STEP (4).
- kv-split variants (`QWEN36_ATTN_KV_SPLIT=user`): v0 does not support them under the megakernel (falls back
  op-by-op for those traces). Revisit after the SDPA phase is final.

## 4. Load-time weight repack (`integ/weights_mk.py`, from the AR track's `weights.py`)

Done once at model load when the flag is on. It builds new device tensors and keeps the op-by-op weights unless
`QWEN36_R16_MK_FREE_OPBYOP=1` (DRAM budget: both copies fit at 27B BFP8/BFP4, ~2× weights, checked by
`spec_memory_costs`).

| Projection | Layout |
|---|---|
| MLP gate\|up, down | v12 split-K: 8 K-rows × 10 N-cols on the 80 STREAM cores; bank-local pages (each core's slice in its nearest DRAM bank, balanced by **bytes**); down output lands in the next layer's x-chunk layout |
| attention qkv (wqkvg_pair), wo | MKA2: 64 STREAM cores as K-halves + 40 ext cores at full K (v0.5 EXT); bank-balanced by bytes; ext slices prefetched in the preceding GDN layers (v0.8.1) |
| GDN in (qkvz/ba) / out | STREAM projection slices, bank-local; GDN conv weights + selectors resident in the GDN working region |
| RMSNorm γ (input / post-attn) | **folded into the consuming projection's weight rows behind `QWEN36_R16_MK_GAMMA_FOLD`** (spec v0.4); flag off applies x⊙γ in-kernel (~7 µs per projection phase). The folded weights must pass the GPQA output-length gate before they default on |
| q/k norm, gated norm | small vectors resident in L1 per block |

Dtypes are unchanged from prod (BFP8/BFP4 per tensor as `model_config`), KV is BFP8 (BFP4 failed the length gate).

## 5. Fabric / CCL

- The megakernel's ARs use the fabric directly (one-shot / row-compact port, `oneshot_ar.py`). The decode path
  already runs with `FABRIC_1D_RING` on the full 4×8 mesh, with TP8 submeshes per DP rank. The megakernel uses the
  same fabric config, plus global semaphores created once per model (spec §6 v0.1).
- Persistent connections: `setup_fabric_connection` is done in-kernel at pass start by the AR sender cores and torn
  down at pass end (open once per pass, spec §15).
- The embedding's DAR enter AG and `_spec_tail`'s CCLs stay as they are (op-by-op, outside the program).

## 6. Things that stay op-by-op (and why that's fine)

| Path | Why unchanged |
|---|---|
| Prefill (text, chunked) and vision / multi-image (c8c90fe) | different shapes; the megakernel is decode-verify only. Vision encoder + image-token prefill feed the same KV/GDN state the megakernel reads |
| Drafter (MTP) | separate traces; small; reads verify outputs |
| LM head / argmax / device top-k / sampled-path logits | vocab-bound, already tuned |
| Structured-output device mask (`_forward_verify_mask_body`), penalties, logprobs | applied on the logits after `_spec_tail`; the megakernel doesn't touch them |
| Host-sampling fallbacks (seed, logit_bias, …) | they run on the logits/ids the tail produces; unchanged |
| Plain decode (`device_decode_step`, non-spec) | not used in serving (spec always on); left op-by-op |

## 7. Flag and fallback

`QWEN36_R16_MEGAKERNEL`:
- `0` / unset: off, today's path.
- `stage0`: plumbing probe. A one-core-per-device probe program reads the bucket's phase table inside the verify
  trace and writes a checksum (proves the per-bucket table tensors, the mesh generic_op inside the captured trace,
  and replay). The layer loop still runs op-by-op, so outputs are bit-identical to off.
- `1`: the real megakernel replaces the layer loop (once the blocks are integrated).

Per-bucket fallback: if a bucket's program can't be built (unsupported n_users, kv-split variant, L1 overflow at
build), that bucket runs op-by-op, with a logged reason (same pattern as `device_decode_loop_unsupported_reason`).

## 8. Integration plan (increments, each gated by the step harness + md5/acceptance)

1. **stage0** (this branch): flag, per-bucket phase tables, probe generic_op captured inside the verify trace.
2. **MLP-only hybrid**: replace each layer's MLP (post-attn norm → gate|up → down → residual add) with the v12 split-K
   MLP as its own generic_op per layer. That proves the weight repack, row-compact x, and numerics vs op-by-op (PCC,
   then md5 with exact numerics, γ-fold off). Expected slower than the full megakernel (launch per layer); it's for
   correctness, not speed.
3. **Full pass, attention + MLP real, GDN stand-in**: one generic_op for the whole pass, with mixed3 attention, v12
   MLP and the row-compact AR. GDN layers still call the op-by-op GDN between megakernel segments (≤17 launches
   per pass), until GDN_PREFIX/STEP land.
4. **Full pass, all real**: GDN_PREFIX + GDN_STEP (u8) / GDN_STEP_U1 (u1).
5. Release gates: md5 (exact-numerics mode), acceptance, GPQA output-length (γ-fold), long-context needle, OR-style
   replay speed, structured-output conformance, multi-image conformance.

## 9. Open questions (for the megakernel agent / spec owner)

- Seed x delivery: does the engine's first phase take x from DRAM (after the DAR enter), or should the DAR enter AG
  itself be replaced by the megakernel's row-compact AR from the embedding shards? v0: from DRAM, with a one-off
  ~4 µs delivery cost.
- Final norm: `_spec_tail` applies the final RMSNorm. Under the deferred-norm contract the megakernel's output is
  the un-normed residual, which matches today's input to `_spec_tail`. OK.
- Per-bucket programs × 8 buckets × kernel compile: check the JIT cache size and warm-up time at capture.

## 10. Engine entry-point contract (agreed with the megakernel agent, spec v0.9.3)

`engine/build.py` (integration track) — `build_engine(mesh, layer_types, n_users, weights_mk, kv_desc, gdn_desc, fabric_ctx, T=4)
-> EngineProgram`:

| member | meaning |
|---|---|
| `R` | rows per activation transfer: 4·n_users rounded up to {4, 8, 16, 32}; phase type 11 (GDN_STEP_U1) at n_users=1, else 4 |
| `replaces_layers` | True: the program is the 64-layer stack. False: a stand-in; the op-by-op loop still runs on x_out |
| `alloc_x_out()` | persistent replicated DRAM residual [1,1,n_users·T,5120] bf16 TILE; called in `prepare`, never inside a capture |
| `launch(x_in, x_out, per_replay)` | one generic_op per pass, inside the trace; per_replay = the body's in-trace tensors (kvpos/ctrl/cos/sin/page tables) + GDN selectors, addresses only |

- Residual: x_in = DRAM copy of the replicated RES40 x (un-normed). The engine prologue loads each device's 2 owned chunks,
  computes layer-0 Σx² (no AR at entry). After the last layer's AG the owners write their chunk into x_out. The host converts
  x_out back to RES40 for `_spec_tail` (a fresh tensor; x_out itself is never freed).
- Fabric: the host guarantees a 1x8 FABRIC_1D_RING on axis 1 with 8 KB packets at device open and no other op on those links
  during the generic_op. The engine opens/closes its 4 fabric connections inside the op and resets its monotonic global
  semaphores at program end; the host never resets them.
- Weights: the HF state dict through the model's transforms, packed ONCE into engine layouts (γ·NORM_FIX fold re-quantizes);
  fallback = tile-granular repack of the model's TT tensors (lossless with γ unfolded). Layout functions:
  `engine/layout_mlp.py` (v12), `engine/layout_attn.py` (mixed3 track, first cut at mk-mixed3 bbc60fc8345),
  `engine/ar_phase.py` (AR).
- KV: per attention layer paged BFP8 (MKA_KVDT=bfp8; BFP4 failed the output-length gate) + block size; page tables in
  per_replay. GDN stays a stand-in until the GDN phases land.
- Stage1 is active only on the DAR residual path (`_verify_dar` not None, rows ≤ 32) and never on kv-split variants.

Kinds (`QWEN36_R16_MK_ENGINE_KIND`): `identity` (default until M3) copies x_in → x_out on 8 cores per device and keeps the
layer loop, so the ident md5 equals flag-off only if the copy ran on every replay; `mixed3` raises until M3 (mixed3 attention
+ v12 MLP + MLP-AR), and Stage1Verify falls back op-by-op with the reason.
