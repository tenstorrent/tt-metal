# Plan: Fuse `fused_partial_rope` into the `matmul_decode` epilogue

**Status:** Draft
**Owner:** smanoj
**Date:** 2026-10-30
**Profile source:** `generated/profiler/reports/dsv41_decode_32k_32L/2026_10_30_08_39_55/ops_perf_results_dsv41_decode_32k_32L_2026_10_30_08_39_55.csv`

---

## 1. Motivation (from the decode profile)

The `dsv41_decode_32k_32L` profile shows `FusedPartialRopeDeviceOperation` is
**dispatch-bound**, not compute-bound:

| metric | value |
|---|---|
| call count | 3840 |
| device kernel total | 36.39 ms |
| host duration total | 36.40 ms |
| mean per call | 9.48 µs device + 9.48 µs host |
| cores used | `{'1': 2304, '64': 1368, '4': 84, '16': 84}` |

`host ≈ device` and **2304 of 3840 calls run on a single core** — i.e. the op
is paying full dispatch overhead for a tiny per-tile column rotation. Two of
those calls per attention block (the `q` rope after `q_b_proj` and the `kv`
rope after `kv_proj`) sit directly on a `matmul_decode` output and are the
fusible subset.

`LinearDecode` already fuses RMSNorm into the `matmul_decode` epilogue
(`enable_fused_rms_norm` / `_epilogue_kwargs`), so the epilogue-fusion
plumbing exists. This plan reuses it to add a `partial_rope` epilogue.

## 2. Current state

### 2.1 Call sites — `models/experimental/deepseek_v4_flash/tt/decode/attention.py`

The decode QKV path (`_qkv`, lines 1422–1465):

```python
1455:        q = self.q_b_proj(q_b_input)                       # matmul_decode -> [1,1,B,H*Dh]
1457:        q = _apply_rope(q, cos, sin, self.rot, self.rope_dim, head_dim=self.head_dim)
...
1463:        kv_raw = self.kv_proj(_decode_activation(self.kv_proj, tokens))   # matmul_decode -> [1,1,B,Dh]
1465:        kv = _apply_rope(kv, cos, sin, self.rot, self.rope_dim)
```

A third rope, on the SDPA **output**, is **not** fusible (it is not downstream
of a matmul) and stays:

```python
1275:        attn = _apply_rope(attn, cos, neg_sin, self.rot, self.rope_dim)
```

### 2.2 `_apply_rope` — `attention.py:557-590`

Delegates to `ttnn.experimental.fused_partial_rope(x, cos, sin, trans_mat,
rope_dim, head_dim=hd)`. Requires `x` to arrive **width-sharded L1** (ROW_MAJOR
`x` is computed as 1x32 faces). `cos`/`sin` are `[1,1,L,Rd]` DRAM-interleaved;
`trans_mat` is a single `[32,32]` rotate_half matrix.

### 2.3 Epilogue mechanism — `models/experimental/deepseek_v4_flash/tt/layers.py`

`LinearDecode._epilogue_kwargs` (lines 779–796) currently threads RMSNorm:

```python
779:    def _epilogue_kwargs(self, output_memory_config: ttnn.MemoryConfig) -> dict:
        ...
791:        if self.fused_rms_norm_eps is not None:
792:            kwargs["rms_norm"] = True
793:            kwargs["rms_norm_gamma"] = self.fused_rms_norm_gamma
794:            kwargs["rms_norm_epsilon"] = self.fused_rms_norm_eps
795:            kwargs["rms_norm_group_size"] = self.fused_rms_norm_group_size
796:        return kwargs
```

`enable_fused_rms_norm` (line 723) and `can_fuse_rms_norm` (line 712) gate it.
The same pattern is the template for the RoPE epilogue.

### 2.4 Standalone op — `ttnn/cpp/ttnn/operations/experimental/transformer/fused_partial_rope/`

Self-contained. The compute kernel rotates the trailing `rope_dim` of each
`head_dim` block:

```
out[..., b, :Hd-Rd] = x[..., b, :Hd-Rd]               # "nope" pass-through
out[..., b, Hd-Rd:] = x_rope * cos + (x_rope @ trans_mat) * sin
```

The rotation is **block-diagonal per tile** (pairs channels `2p`/`2p+1`), so
each output tile rotates independently of how columns are spread over cores
— this is what makes it composable onto a matmul output tile.

## 3. Design

Extend `matmul_decode`'s epilogue to optionally apply partial RoPE to the
output tile before writeback. The epilogue already runs on the B-core grid
that holds the `[M, N]` output; RoPE is a per-tile column transform that fits
after the reduce and before the writer.

### 3.1 New epilogue kwargs (mirror the `rms_norm_*` pattern)

```python
kwargs["partial_rope"] = True
kwargs["partial_rope_cos"] = cos            # [1,1,L,Rd] DRAM-interleaved
kwargs["partial_rope_sin"] = sin
kwargs["partial_rope_trans_mat"] = trans_mat  # [32,32] rotate_half
kwargs["partial_rope_rope_dim"] = rope_dim
kwargs["partial_rope_head_dim"] = head_dim
```

### 3.2 Composition with RMSNorm

`q_b_proj` already fuses RMSNorm (`group_size=head_dim`). The combined
epilogue order is **RMSNorm → RoPE**. RoPE is a rotation and is
norm-preserving, so the order is numerically safe; verify with a bit-exact
diff in `bf16`.

### 3.3 Constraints (enforce in factory selection / `create_program`)

- Output tile is `Tile(1,32)` (ROW_MAJOR) or native `Tile(32,32)`.
- `N % head_dim == 0`.
- `head_dim % 32 == 0` (tile-aligned).
- B-core grid owns full `head_dim` blocks on each core (true for `q_b_proj`
  and `kv_proj` today — both are full-width hub-mode).

## 4. Steps

### Step 1 — C++ device op: `matmul_decode` epilogue extension

- Locate the `matmul_decode` compute kernel's epilogue block (where
  `rms_norm` is applied today) and add a `partial_rope` branch.
- Port the inner tile transform from `fused_partial_rope`'s compute kernel
  (the `x_rope * cos + (x_rope @ trans_mat) * sin` per `head_dim` block,
  leading "nope" channels pass-through).
- Reader: extend the matmul reader to also stream `cos`/`sin` tile-rows
  (one per output row, or a single broadcast row — same logic as
  `fused_partial_rope`'s reader). `trans_mat` is a single `[32,32]` tile,
  replicated.
- Enforce the §3.3 constraints in factory selection.

### Step 2 — Python wrapper: `LinearDecode` (`layers.py`)

- Add `can_fuse_partial_rope()` (mirror `can_fuse_rms_norm`, line 712):
  requires `use_rm_hs` + full-width hub mode + `N % head_dim == 0`.
- Add `enable_fused_partial_rope(cos, sin, trans_mat, rope_dim, head_dim)`
  (mirror `enable_fused_rms_norm`, line 723). Store the tables as
  WIDTH_SHARDED L1 in the layout the reader expects; reuse the grid-alignment
  logic from `fused_rms_norm_gamma_memory_config` (line 81).
- Extend `_epilogue_kwargs` (line 779) to emit the `partial_rope_*` kwargs
  when enabled.

### Step 3 — Call sites: `decode/attention.py`

- In `DeepSeekV4Attention.__init__` (around `attention.py:829+` /
  `layers.py:941-960`), after building `q_b_proj` and `kv_proj`, call
  `enable_fused_partial_rope(...)`.
- `cos`/`sin` are **per-step** (position-dependent), so they cannot be baked
  at `__init__`. Two options:
  - **(a) Preferred:** pass the tables as runtime kwargs through `__call__`
    (matches how `cur_pos` is passed to SDPA).
  - **(b)** keep the tables in fixed DRAM buffers that
    `prepare_static_decode` rewrites per step (matches the existing
    trace-captured pattern for `cos_win`/`sin_win`).
- Remove `_apply_rope(q, ...)` (`attention.py:1456`) and
  `_apply_rope(kv, ...)` (`attention.py:1465`) from `_qkv`.
- **Leave** `_apply_rope(attn, ...)` (`attention.py:1275`) — output RoPE after
  SDPA, not downstream of a matmul.

### Step 4 — Trace capture

- The fused path must be capture-safe (the whole decode is a trace).
- `cos`/`sin` are already persistent DRAM buffers rewritten in place per
  step, so fusing them into the matmul reader is trace-safe as long as the
  reader addresses are baked at capture (same as the standalone op).
- Verify with a single-step trace replay vs eager.

### Step 5 — Validation

- **Unit test:** compare fused-epilogue `q_b_proj`/`kv_proj` output vs
  `matmul_decode → fused_partial_rope` eager, over a sweep of
  `head_dim`/`rope_dim`/batch, at `bf16` and the matmul's actual `HIFI4`
  fidelity. Tolerance matches `tests/ttnn/unit_tests/operations/experimental/transformer/test_fused_partial_rope.py`.
- **Trace replay:** run one decode step eager and one via trace; assert
  bit-equality (RoPE is exact in `bf16`).
- **End-to-end:** re-run the `dsv41_decode_32k_32L` profiler and diff the
  new `ops_perf_results`. Expect:
  - `FusedPartialRope` count drops by ~576 (2/block × 288).
  - `MatmulDecode` device time rises by the fused compute.
  - `ATTN_MIX`/`ATTENTION` signpost wall time drops by the removed dispatch
    latency.

## 5. Benefit

### Direct (from the profile)

- Eliminates **576 `FusedPartialRope` dispatches** (the `q` and `kv` ropes;
  2 per attention block × 288 blocks). At ~9.5 µs host dispatch + ~9.5 µs
  device each:
  - **~11 ms off the host dispatch critical path**
  - **~11 ms off device dispatch**
- Saves **one L1→L1 round-trip** of the `q` (`[1,1,B,H*Dh]`) and `kv`
  (`[1,1,B,Dh]`) tensors per block: the matmul output tiles are rotated
  in-place in the epilogue's L1 CB instead of being written, re-read by the
  rope reader, and re-written. The `q` round-trip (full `H*Dh` width) is the
  larger save.
- Removes the **layout-conformance constraint**: today `_apply_rope` requires
  width-sharded L1 input (`attention.py:577-578`), forcing the matmul output
  into a specific layout. Fusing lets the matmul pick its cheapest output
  layout.

### Indirect (more valuable than the direct savings)

- **Flatter critical path.** Each standalone op on the decode path is a
  dispatch point that can stall the trace. The `SendDirectAsync` tail
  (median 74 µs, p90 2.0 ms, 138 calls >1 ms = 97.6% of send time) shows the
  pipeline is sync-sensitive. Removing 576 dispatch points reduces the
  chance of a dispatch bubble aligning with a cross-submesh send stall.
- **Fewer programs in the trace.** 576 fewer programs → smaller capture,
  faster replay setup, less CQ command-buffer pressure.
- **Unlocks further fusion.** Once the epilogue accepts a column-transform,
  the same mechanism can fuse:
  - the output RoPE (`attention.py:1275`) into a future fused `sdpa_decode`
    epilogue, and
  - the compressor/window ropes into their matmuls.
  The same plumbing serves all of them.

### Honest sizing

The rope fusion alone is **~36 ms device + ~36 ms host = ~72 ms out of a
9.67 s trace (~0.7%)**. It is **not** the big win — the `SendDirectAsync`
tail (92.3% of device time) is. But it is:

- the **cheapest** of the fusions to land (the epilogue mechanism already
  exists for RMSNorm),
- **numerically risk-free** (RoPE is exact in `bf16`), and
- removes **dispatch points from a sync-sensitive path**, which is where
  the real `ATTN_MIX` stall lives.

## 6. Risks

- **Epilogue register pressure.** Adding RoPE to the matmul compute kernel
  raises register/L1 usage in the epilogue. `q_b_proj` already fuses RMSNorm,
  so the combined epilogue (RMSNorm + RoPE) needs an L1 budget check. If it
  spills, fall back to fusing RoPE only on `kv_proj` (which has no norm) and
  leave `q_b_proj`'s rope standalone.
- **`cos`/`sin` per-step mutation.** Must confirm the tables are rewritten
  in place at the same DRAM addresses every step (the existing trace pattern)
  so the fused reader's baked addresses stay valid.
- **BFLOAT4_B compatibility.** `ReshardDeviceOperation` output shows
  `BFLOAT4_B` is in use; the RoPE math must be validated at the matmul's
  actual output dtype, not just `bf16`.

## 7. Open questions

1. **`cos`/`sin` runtime path** — option (a) runtime kwargs through
   `__call__`, or (b) fixed DRAM buffers rewritten per step? (a) is cleaner
   but requires threading the tables through every `__call__`; (b) matches
   the existing trace pattern. Decide before Step 2.
2. **Combined RMSNorm + RoPE epilogue on `q_b_proj`** — does it fit in L1,
   or do we fuse RoPE only on `kv_proj` (no norm) for v1?
3. **`head_dim` tile-alignment** — confirm all layer types have
   `head_dim % 32 == 0`; if not, the op needs a fallback for the
   non-tile-aligned case.

## 8. Suggested order of work

1. Step 1 (C++ epilogue) — sketch the RoPE branch in the `matmul_decode`
   compute kernel, gated off by default.
2. Step 2 (Python wrapper) — `LinearDecode.enable_fused_partial_rope` +
   `_epilogue_kwargs`.
3. Step 3 (call sites) — wire up `kv_proj` first (no norm, simplest), then
   `q_b_proj` (combined epilogue).
4. Step 5 (validation) — unit test + trace replay + end-to-end profile diff.
5. Step 4 (trace capture) — confirm and fix capture safety, if needed.
