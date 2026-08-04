# W3a — Decode QKV head-split fusion (`nlp_create_qkv_heads_decode`)

Off-device design. No mesh opened, no code edited. Target: cut op-count on the dispatch-bound
decode hot path by replacing the hand-rolled `_split_qkv` (+ the separate `_shard_kv`) with the
single fused `ttnn.experimental.nlp_create_qkv_heads_decode`. Hard gate: decode-PCC ≥ 0.995
(accuracy-neutral).

All file:line citations are against the current tree.

---

## 0. Executive answer (for the report)

- **Does the fused op subsume `_shard_kv`?**
  - **V: YES, unconditionally.** The fused op already emits V height-sharded `[1,B,32,head_dim]`
    (one user/core), byte-identical in layout to `_shard_kv`'s output, and V never passes through
    norm or rope. `paged_update_cache` consumes it directly. `_shard_kv(v)` is removed.
  - **K: NOT in the conservative diff** (K must round-trip to interleaved for `_per_head_norm`, so
    it needs a re-shard afterward → `_shard_kv(k)` stays). K's shard is only subsumable in the
    **aggressive, full-rotary-only** follow-on (Stage 3 below), which is layer-gated and higher risk.
  - **`_split_qkv`: YES, fully removed** (6 ops → 1 fused op + 1 metadata reshape).
- **Estimated ops removed on the hot path (conservative Stage 1+2):** `_split_qkv` 3×slice+3×reshape
  (6) and `_shard_kv(v)` (1) go away; the fused op (1) + `sharded_to_interleaved` on q and k (2) come
  in. **Net ≈ −4 ops/decode/layer** (metadata reshape is a wash). Aggressive Stage 3 removes
  `_shard_kv(k)` for the 30 sliding layers → a further −1 there.
- **Top-2 breakage risks:** (1) **packed-QKV column order / head interleave** mismatch between what
  `_split_qkv` sliced and what the fused op assumes — would corrupt q/k/v silently (PCC collapse);
  (2) **K/V output shard-spec vs `paged_update_cache` expectation** (grid placement, `nkv_padded=32`
  vs `nkv32`, dtype) — would corrupt the cache write and thus SDPA reads.
- **Is the norm "L1 dance" required?** In the **conservative** design, **no** — Laguna's
  `_per_head_norm` runs on interleaved, so we just `sharded_to_interleaved` q and k after the fused op
  (one op each) and the entire norm→rope→shard→SDPA tail is byte-for-byte unchanged. The reference's
  in-place L1 round-trip (attention_1d.py:636-646) is only needed for the **aggressive** Stage 3,
  where q/k stay height-sharded through norm.

---

## 1. Op signature and layout (cited)

### 1.1 Python binding
`ttnn/cpp/.../nlp_create_qkv_heads_decode/nlp_create_qkv_heads_decode_nanobind.cpp:21-37`

```python
q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
    input_tensor,                 # positional, .noconvert()
    num_heads,                    # kw-only, required (LOCAL q heads)
    num_kv_heads=None,            # kw-only (LOCAL kv heads)
    output_tensors=None,          # kw-only optional pre-alloc [q,k,v]
    overlap_qk_coregrid=True,     # kw-only; MUST be True for interleaved input (see 1.3)
    batch_offset=None,            # kw-only (with slice_size, for batch slicing)
    slice_size=None,              # kw-only
    memory_config=None,           # kw-only: OUTPUT memcfg (HEIGHT_SHARDED template)
)
```
C++ decl: `nlp_create_qkv_heads_decode.hpp:13-21`.

### 1.2 Input layout (docstring + device-op validate)
Docstring (nanobind:24-26): input is
`[1, S=1, B, head_dim*(num_heads + 2*num_kv_heads)]`, packed as **[all Q heads | all K heads | all V
heads]**, `num_heads`/`num_kv_heads` padded to nearest 32 in the *output*. "Input must be sharded,
B=32 and S=1. If ttnn pads B from some number < 32 to 32, this op respects the unpadded B."

Validate (`device/nlp_create_qkv_heads_decode_device_operation.cpp:36-84`):
- device tensor, **TILE layout** (:42), dtype **FLOAT32 or BFLOAT16** (:38-41).
- `input_shape[0]==1`, `input_shape[1]==1` (:52-53); `num_users = input_shape[2] <= 32` (:46-51);
  `input_shape[3] % 32 == 0` (:47-50, head_dim multiple of TILE_WIDTH).
- If **sharded** → must be **WIDTH_SHARDED, ROW_MAJOR**, shard `shape[0]==physical_volume/padded_last`
  (:55-66). If **interleaved** → allowed, but `overlap_qk_coregrid` **must be True** (:82-84).

### 1.3 Output layout (`compute_output_specs`, same file :114-183)
- Q shape `[1, B, num_heads, head_dim]`; K,V shape `[1, B, num_kv_heads, head_dim]` (logical, actual
  head counts — :127-129).
- **HEIGHT_SHARDED** required (:87-90). Per-core shard shape is padded to 32 rows:
  `q: (ceil(num_heads/32)*32, head_dim)`, `k/v: (ceil(num_kv_heads/32)*32, head_dim)` (:131-156).
- One user per core: `num_cores >= num_users` (overlap) / `>= 2*num_users` (non-overlap) (:99-111).
  With `overlap_qk_coregrid=True`, Q,K,V all land on the first `batch` cores of the output grid
  (:138-152). Output dtype/layout = input's (:170-182).
- The `memory_config` arg is a **template**: the op reads its `memory_layout` (HEIGHT_SHARDED),
  `buffer_type`, and `grid`, and derives the three per-tensor shard specs itself (:157-168). So we pass
  **one** height-sharded memcfg whose grid covers ≥ B cores.

### 1.4 Reference wiring (studied)
- `models/common/modules/attention/attention_1d.py:621-632` — reshape all-reduced qkv to
  `(1,1,max_batch,qkv_w)` with **padded shape `(1,1,32,qkv_w)`** (:623), then the op with
  `overlap_qk_coregrid` + `memory_config=cfg.decode_create_qkv_head_memcfg`.
- Reference output memcfg default (`attention_1d.py:1620-1631`, Blackhole branch): HEIGHT_SHARDED,
  `shape=(32, head_dim)`, `core_grid=CoreGrid(y=4, x=8)` (=32 cores), ROW_MAJOR,
  `use_height_and_width_as_shard_shape=True`. This is the exact template Laguna should mirror.
- Norm workaround (:636-646): RMSNorm doesn't accept HEIGHT_SHARDED, so ref converts each of q/k to
  `L1_MEMORY_CONFIG`, runs `q_norm.decode_forward`, converts back to the sharded memcfg — the "L1
  dance". `tt_transformers/tt/attention.py:636-660` and `demos/gpt_oss/tt/attention/decode.py:72`
  follow the same shape.

---

## 2. How this maps onto Laguna's per-device decode

Laguna decode runs **per-device** on the TP=4 mesh. `MultichipDecoder.from_state_dict` mutates the
shared `cfg` to **LOCAL** head counts before constructing the layer
(`tt/multichip_decoder.py:268-271`): `cfg.num_heads = lqh`, `cfg.num_kv_heads = lkv`, and
`meta["q_w"/"kv_w"/"qkv_w"]` become the **per-device** widths (:275-277). Concretely (context_contract):
- **full-attention layers (10):** 48 global Q → **lqh=12** local; rotary partial **rd=64** (rd<hd).
- **sliding layers (30):** 64 global Q → **lqh=16** local; rotary **rd=128** (full, rd==hd).
- both: 8 global KV → **lkv=2** local; **head_dim=128**.

So the fused-op call uses `num_heads = cfg.num_heads` (12 or 16), `num_kv_heads = cfg.num_kv_heads`
(2). Padded output: Q `(32,128)`/core, K,V `(32,128)`/core — and `nkv32` in `_shard_kv`
(`optimized_decoder.py:1153`) is exactly `ceil(2/32)*32 = 32`, so **the fused op's V shard shape
equals `_shard_kv`'s V shard shape (32,128)**. That equality is what makes V-subsumption safe.

**Packed-order check (critical, and it matches).** The per-device packed weight is built as contiguous
`[Q_d | K_d | V_d]` blocks (`multichip_decoder.py:158-164`), so each device's qkv row is
`[local_q_w | local_kv_w | local_kv_w]` = `[Q|K|V]` — precisely the
`head_dim*(num_heads+2*num_kv_heads)` packing the op expects (nanobind:25), and precisely what
`_split_qkv` slices today (`optimized_decoder.py:887-889`). Within Q, heads are `head0..head_{nh-1}`
each `head_dim` contiguous, identical to `_split_qkv`'s `reshape(1,rows,nh,hd)` (:890). **No reorder
needed.**

---

## 3. Proposed diff

### 3.1 One-time setup (add an output-memcfg template next to the other decode configs)

Add in `OptimizedDecoder.__init__` (near the SDPA configs, ~`optimized_decoder.py:511-538`), so it is
built once, not per-decode:

```python
# Fused decode QKV head-split output template (HEIGHT_SHARDED, one user/core).
# The op derives per-tensor q/k/v shard specs from this; grid must cover >= B cores.
# Mirror attention_1d.py:1620-1631 (Blackhole): 32 cores, shard (32, head_dim).
self._qkv_heads_decode_memcfg = ttnn.create_sharded_memory_config(
    shape=(TILE, self.cfg.head_dim),           # (32, 128)
    core_grid=ttnn.CoreGrid(y=4, x=8),         # 32 cores >= max decode batch
    strategy=ttnn.ShardStrategy.HEIGHT,
    orientation=ttnn.ShardOrientation.ROW_MAJOR,
    use_height_and_width_as_shard_shape=True,
)
```
(`TILE` is imported in this module; `head_dim` is a multiple of 32 → satisfies validate:47-50.)

### 3.2 `decode_forward` — replace the split + V-shard (conservative, primary)

Current (`optimized_decoder.py:1092-1116`):
```python
        qkv = self._dram_mm(ln, self.w["wqkv"], self.w["wqkv_ds"], cfg.hidden, self.meta["qkv_w"], self._ck_qkv)
        if self.use_dram_sharded:
            qkv = ttnn.sharded_to_interleaved(qkv, ttnn.DRAM_MEMORY_CONFIG)
        q, k, v = self._split_qkv(qkv, B)
        q = self._per_head_norm(q, self.w["q_norm"])
        k = self._per_head_norm(k, self.w["k_norm"])
        ...rope...
        k_sh = self._shard_kv(k, B)
        v_sh = self._shard_kv(v, B)
```

Proposed:
```python
        qkv = self._dram_mm(ln, self.w["wqkv"], self.w["wqkv_ds"], cfg.hidden, self.meta["qkv_w"], self._ck_qkv)
        if self.use_dram_sharded:
            qkv = ttnn.sharded_to_interleaved(qkv, ttnn.DRAM_MEMORY_CONFIG)
        # Fused head-split: replaces _split_qkv (3 slice + 3 reshape) AND emits the decode
        # height-sharded per-batch layout directly. Interleaved input requires overlap_qk_coregrid=True
        # (device validate:82-84). Pad B to 32 in the padded-shape arg; op respects the unpadded B.
        qkv = ttnn.reshape(qkv, (1, 1, B, self.meta["qkv_w"]), (1, 1, TILE, self.meta["qkv_w"]))
        q_sh, k_sh_raw, v_sh = ttnn.experimental.nlp_create_qkv_heads_decode(
            qkv,
            num_heads=cfg.num_heads,
            num_kv_heads=cfg.num_kv_heads,
            overlap_qk_coregrid=True,
            memory_config=self._qkv_heads_decode_memcfg,
        )
        # Q,K -> interleaved so the UNCHANGED _per_head_norm / rope / (_shard_kv for K) tail runs
        # byte-for-byte as today. V is already in the exact _shard_kv layout -> feed the cache directly.
        q = ttnn.sharded_to_interleaved(q_sh, ttnn.DRAM_MEMORY_CONFIG)
        k = ttnn.sharded_to_interleaved(k_sh_raw, ttnn.DRAM_MEMORY_CONFIG)
        q = self._per_head_norm(q, self.w["q_norm"])
        k = self._per_head_norm(k, self.w["k_norm"])
        ...rope (UNCHANGED: _fused_rope_decode / _apply_rope, partial-rotary, rope_mats)...
        k_sh = self._shard_kv(k, B)     # K still needs re-shard after interleaved norm+rope
        # v_sh comes straight from the fused op — _shard_kv(v) REMOVED.
```

Everything from `if sequential_kv_write ...` onward (`:1117-1149`) is **unchanged**: `k_sh`/`v_sh`
have the same logical shape `[1,B,32,128]` and the same shard spec `(32,128)` as before, so
`paged_update_cache`, `_seq_kv_write` (`:1167-1205`, does its own `to_memory_config(...DRAM)` first),
`paged_scaled_dot_product_attention_decode`, the `_sdpa_pc`/`_sdpa_pc_decode` program-config gates,
and the `_gate`/`o_proj` tail all see identical inputs. Q reaches SDPA in DRAM interleaved (the layout
the current code already requires per the `:1094` note). **This is why Stage 1+2 is accuracy-neutral by
construction: only the *producer* of q/k and the *source* of v_sh change; the consumed layouts don't.**

`_split_qkv` (`:883-893`) becomes dead code for decode (still used by prefill? No — prefill uses
`nlp_create_qkv_heads`, `:904`). It can be deleted or left; leave it until Stage 1 validates.

### 3.3 Stage 3 (optional, sliding/full-rotary layers only) — also subsume `_shard_kv(k)`

Only when `cfg.rotary_dim == cfg.head_dim` (the 30 sliding layers), keep K height-sharded through norm
+ rope so its post-rope output is already `(32,128)` height-sharded and feeds the cache with no
`_shard_kv(k)`:
- norm K via the reference **in-place L1 dance** (attention_1d.py:636-646): `to_memory_config(k_sh,
  L1_MEMORY_CONFIG)` → `_per_head_norm` → `to_memory_config(back to the fused-op sharded memcfg)`.
- rope K via `_fused_rope_decode` but (a) skip its internal `_shard_batch` (input already sharded),
  and (b) skip the trailing `sharded_to_interleaved` (`optimized_decoder.py:1252-1254`) so `out_sh`
  (shape `(32, rd==hd==128)`) is returned sharded — identical to `_shard_kv(k)`.
- This is **not** valid for partial-rotary full-attention layers (rd=64): the pass-through concat
  (`:1255-1257`) reinterleaves, so those keep `_shard_kv(k)`. Gate Stage 3 on `rd==hd` and on
  `self._use_fused_rope` (the `_apply_rope` fallback is not sharded-aware — keep it on the conservative
  path). Do Stage 3 **after** Stage 1+2 is green so a PCC break localizes to the K path.

### 3.4 Optional: drop the `sharded_to_interleaved(qkv)` (feed width-sharded input)

The `_dram_mm` output is `L1_WIDTH_SHARDED` with shard `shape[0] = tile-padded batch rows`, which
*may* already satisfy the sharded-input contract (validate:55-66: WIDTH_SHARDED, ROW_MAJOR,
`shape[0]==physical_volume/padded_last`). If so, feed it directly and delete the `:1096` `s2i`,
saving one more op. Treat as a **separate, later** micro-opt — it changes the input program factory
(sharded vs interleaved path) so it must be PCC-gated on its own.

---

## 4. Risk list (ordered) + fast-loop staging

Add pieces in this order so a single-layer `test_decode_pcc` isolates any break:

1. **Stage 1 — split only, prove head values.** Fused op replaces `_split_qkv`, but immediately
   `sharded_to_interleaved` **all three** (q, k, **and v**) and keep `_shard_kv(v)` + `_shard_kv(k)`
   exactly as today. This isolates *"does the op produce the same q/k/v as `_split_qkv`?"*
   - Risk A (**highest**): **packed column order / head interleave.** If the op's Q|K|V slicing or
     within-Q head order differs from `_split_qkv`, PCC collapses (≪0.9). Analysis in §2 says it
     matches, but this stage is the guard. Also check the **B-padding**: reshape padded-shape must be
     `(1,1,32,qkv_w)` (nanobind:25 wants B padded to 32; op honors unpadded B).
   - Risk B: **dtype.** Op requires fp32/bf16 (validate:38-41). Laguna qkv is bf16 — OK. If a policy
     ever makes qkv fp32-accumulated, still fine; anything else FATALs (not silent).

2. **Stage 2 — remove `_shard_kv(v)`, feed `v_sh` to the cache.** Isolates *"is the fused V shard
   spec what `paged_update_cache` expects?"*
   - Risk C (**second-highest**): **K/V output shard placement / spec.** The op places V on the first
     `B` cores of `_qkv_heads_decode_memcfg`'s grid; `_shard_kv` used a `row=8` core layout
     (`:1154-1157`). Shard *shape* is identical `(32,128)`; if `paged_update_cache` is sensitive to
     core *placement* (it generally is not — it reads per-user shards), this passes. If PCC breaks
     only at Stage 2 (Stage 1 green), it is placement/spec, not head order. Mitigation: set the memcfg
     grid to the same `row=8` core ordering `_shard_kv` used, or pre-alloc via `output_tensors`.
   - Risk D: **B<32 / B=1.** The perf/PCC runs are batch-1. Op needs B≤32 and one user/core; at B=1 a
     single core is used. Padded head rows (12/16→32) are physical-only; the logical `[1,B,nh,hd]`
     shape drops them on `s2i` and on the cache write. Confirm the grid has ≥1 core (it does).

3. **Stage 3 — sharded K path (sliding layers only).** Isolates the norm L1 dance + sharded rope.
   - Risk E: **partial-rotary leakage.** Must be gated on `rd==hd`; do not enable for full-attention
     layers. Test both a sliding layer and a full-attention layer to prove the gate.
   - Risk F: **`_apply_rope` fallback + `sequential_kv_write`.** Keep both on the conservative
     (interleaved) K path; Stage 3 must be gated on `self._use_fused_rope and not sequential_kv_write`.

**dtype/tile/pad summary for B<32:** input TILE layout, bf16, logical `[1,1,B,qkv_w]` with padded
shape `[1,1,32,qkv_w]`; head_dim(128)%32==0; outputs bf16 HEIGHT_SHARDED `(32,128)`/core on B cores.

---

## 5. Fast-loop validation commands

Run per-stage (single P150x4 mesh; `LAGUNA_MC_CLASS=optimized` selects `OptimizedMultichipDecoder`).

**Accuracy (hard gate, PCC ≥ 0.995).** Cover a sliding *and* a full-attention layer (rope differs):
```bash
# sliding layer (full rotary) and full-attention layer (partial rotary rd=64)
LAGUNA_MC_CLASS=optimized pytest -q tests/test_multichip_decoder.py::test_decode_pcc \
  -k "layer4 or layer0" 2>&1 | tail -30
# multistep decode (paged_update_cache write path over several positions)
LAGUNA_MC_CLASS=optimized pytest -q tests/test_multichip_decoder.py::test_multistep_decode -k layer4
```
`test_decode_pcc` (`tests/test_multichip_decoder.py:150-167`) prefills then does one decode step and
asserts `pcc >= PCC_BAR`. Localize per §4: Stage-1 break ⇒ head order/pad; Stage-2 break ⇒ V shard
spec; Stage-3 break ⇒ K sharded rope/norm.

**Op-count / wall-clock delta (baseline to beat: OMC decode 0.7775 ms/token/layer @ layer 4):**
```bash
python tests/perf_trace_omc.py 4 4096 30   # <layer> <prefill_len> <decode_iters>
```
Reports `after_optimized.decode_ms_per_token` (`tests/perf_trace_omc.py:24-26,114-122`). Expect the
−4-op cut to move this below 0.7775; confirm with a single-layer tracy on layer 4 that
`nlp_create_qkv_heads_decode` replaced the 3×slice/3×reshape of `_split_qkv` and that only one
`paged_update_cache`-feeding shard remains for V.

---

## 6. Net verdict

- `_split_qkv` **fully replaced** (−6 ops → +2: fused op + reshape).
- `_shard_kv(v)` **removed** (V comes out cache-ready).
- `_shard_kv(k)` **kept** in the safe path (interleaved norm forces a re-shard); removable only in the
  gated Stage-3 sliding-layer path.
- Norm **L1 dance not required** for the conservative diff (Laguna's `_per_head_norm` is interleaved;
  a plain `sharded_to_interleaved` on q/k suffices). It *is* required for Stage 3.
- Accuracy-neutral by construction: the consumed layouts of rope, `paged_update_cache`, and the paged
  decode SDPA are unchanged; only the q/k producer and the v_sh source move to the fused op.
