# LOCAL vs EXP17 — the L1 activation chain (op-by-op shapes, configs, and why LOCAL falls back to DRAM)

Companion to `LOCAL_vs_EXP17_op_shapes_configs.md` (that one is about DRAM-sharding the matmul **weights**). **This doc is about the activation stream: for every decode op, its shape + full memory config + program config, so you can see op-by-op why LOCAL doesn't keep activations L1-width-sharded and chain op→op the way EXP17 does.**

Reasons in the "Why not L1" column come **only from the code and the trial logs** (`work_log.md`, `sharded_residual_norm_trials.md`, `*_trials.csv`). Where nothing in the logs tested it, the reason is simply **"not attempted."** Where a trial did test it, the result is cited.

Sources: LOCAL `tt/optimized_decoder.py` (`decode_forward`/`_decode_qkv`/`_mlp`); EXP17 `tt/optimized_decoder.py` + `models/common/modules/attention/attention_1d.py`. µs from `optimized_decoder/tt_perf_report_decode_gate_up_geometry.txt`. Shapes are decode with batch padded to `M=32`, one token/user.

Constants: `hidden=4096`, `intermediate=14336`, `heads=32`, `head_dim=128`, `kv_heads=8`, packed QKV = `6144`, tile = 32.

Memory-config shorthand: `DRAM` = DRAM interleaved; `L1 WS [r,c]@Nc(x·y)` = L1 WIDTH_SHARDED, shard `[rows,cols]`, N cores on grid x·y; `L1 HS` = L1 HEIGHT_SHARDED. Chain: 🟢 L1 sharded, 🔴 DRAM, 🔁 pure layout-move (reshard/copy).

The **EMIT** column in §2 is the same op's config in the codegen emit (`ttnn-models/meta-llama/Llama-3.1-8B-Instruct/model/graph_0/model_ttnn.py`, decode branch). Core counts there are the emit's own, taken on an **11-wide** grid (e.g. QKV `@96c`, gate/up `@90c`), so `@Nc` values don't map 1:1 onto the 8×8 target grid; the intent (WIDTH_SHARDED L1, shard width) is what carries over.

---

## 0. TL;DR

An L1 chain means each op reads a sharded L1 input and writes a sharded L1 output, so the next op runs with no DRAM round-trip and no reshard. EXP17 does this for the **whole decode layer**. LOCAL does it **only for rows 14–23** (post-attn residual → post-attn norm → MLP-down); the input norm, the entire attention front, and gate/up/SiLU/multiply all live in DRAM.

Documented status of each break (see §3):
- **Input RMSNorm DRAM** — not attempted (explicitly out of scope of the one norm trial).
- **Attention front DRAM (QKV/O)** — moving the attention **input** to L1 was **tried and rejected as slower**; a DRAM-sharded attention matmul + fused heads was **not attempted**.
- **gate/up/SiLU/multiply DRAM** — the 1D gate/up geometry was tuned, but DRAM-sharding gate/up and fusing SiLU were **not attempted**.

For reference, the **tt-forge emit that LOCAL was rewritten from already kept the whole decode layer L1-sharded** (WIDTH_SHARDED norms/matmuls/residuals, L1-interleaved head glue; only SDPA runs in DRAM) — see the EMIT column in §2. So LOCAL's DRAM breaks are a regression from the emit's own layout, not an inherent constraint. The emit reaches L1 differently from EXP17 though: its matmuls are **1D-multicast over interleaved weights**, not DRAM-width-sharded, and it fuses SiLU into the **gate matmul** rather than into the mul.

---

## 1. LOCAL decode — op by op with shape + full config

"Shape vs EXP17" column: `=` = same op exists in EXP17 with the **same tensor shape**, so only the memory layout differs — i.e. *the L1 version is provably doable, it just isn't done here*; `x` = same op but a **different shape** (does not occur here); a **LOCAL-only op** with no EXP17 counterpart (reshard, hand-rolled head glue, or standalone SiLU that EXP17 fuses away) gets **no sign, just a note**. There is no `x` row — the divergence is layout/op-decomposition, never shape.

| # | Op | Shape in → out | Shape vs EXP17 | Output mem config | Program config | µs | Chain |
| ---: | --- | --- | :---: | --- | --- | ---: | --- |
| 1 | input `rms_norm` | `[1,1,32,4096]` → `[1,1,32,4096]` | `=` | `DRAM` | none (1-core) | **94** | 🔴 |
| 2 | QKV `matmul` | `[1,1,32,4096]`·`[4096,6144]` → `[1,1,32,6144]` | `=` | `DRAM` | **none** (default) | 102 | 🔴 |
| 3 | slice q/k/v | → `[..,4096]`,`[..,1024]`,`[..,1024]` | EXP17 fuses in `create_qkv_heads` | `DRAM` | — | 10 | 🔴 |
| 4 | reshape/permute q,k | → `[1,32,32,128]` / `[1,8,32,128]` | fused in EXP17 | `DRAM` | — | 25+ | 🔴 |
| 5 | RoPE q, RoPE k | same | `=` | `DRAM` | — | 32 | 🔴 |
| 6 | permute/slice q,k | same | fused in EXP17 | `DRAM` | — | 40+ | 🔴 |
| 7 | `to_memory_config` q,k,v | → head layout | LOCAL-only reshard | `L1 HS [32,128]@32c` | — | ≈4 | 🔁 |
| 8 | paged_update_cache ×2 | into cache | `=` | cache (DRAM) | — | 27 | — |
| 9 | SDPA decode | q·cache → `[1,32,32,128]` | `=` | `DRAM` | `SDPAProgramConfig 8×8` | 13 | 🔴 |
| 10 | `to_memory_config` sdpa | → head layout | LOCAL-only reshard | `L1 HS [32,128]@32c` | — | 1 | 🔁 |
| 11 | `nlp_concat_heads_decode` | → `[1,1,32,4096]` | `=` | `L1` | — | 2 | 🟢* |
| 12 | slice attn | → `[1,1,32,4096]` | LOCAL-only | `DRAM` | — | 3 | 🔴 |
| 13 | O `matmul` | `[1,1,32,4096]`·`[4096,4096]` → `[1,1,32,4096]` | `=` | `DRAM` | **none** (default) | 58 | 🔴 |
| 14 | `to_memory_config` hidden, attn_out | → residual | LOCAL-only reshard | `L1 WS [32,128]@32c(8×4)` | — | 6 | 🔁 |
| 15 | add attn_residual | → `[1,1,32,4096]` | `=` | `L1 WS [32,128]@32c` | — | 2 | 🟢 |
| 16 | post-attn `rms_norm` | → `[1,1,32,4096]` | `=` | `L1 WS [32,128]@32c` | `LNShardedMC 8×4, block_h=1, block_w=4, subblock_w=4` | 10 | 🟢 |
| 17 | gate `matmul` | `[1,1,32,4096]`·`[4096,14336]` → `[1,1,32,14336]` | `=` | **`DRAM`** | `Matmul1D 8×8, in0_block_w=4, per_core_N=7, subblock 1×7, mcast_in0` | 202 | 🔴 |
| 18 | SiLU | same | EXP17 fuses into mul | **`DRAM`** | — | 14 | 🔴 |
| 19 | up `matmul` | same as gate | `=` | **`DRAM`** | same 1D config | 201 | 🔴 |
| 20 | multiply (gate·up) | → `[1,1,32,14336]` | `=` | **`DRAM`** | — | 15 | 🔴 |
| 21 | `to_memory_config` gated | → dram-input | LOCAL-only reshard | `L1 WS [32,1792]@8c(8×1)` | — | 7 | 🔁 |
| 22 | down `matmul` | `[1,1,32,14336]`·`[14336,4096]` → `[1,1,32,4096]` | `=` | `L1 WS [32,512]@8c` | `DRAMSharded in0_block_w=14, per_core_N=16` | 134 | 🟢 |
| 23 | add output residual | → `[1,1,32,4096]` | `=` | `L1 WS [32,128]@32c` | — | — | 🟢 |
| 24 | `to_memory_config` output | → DRAM | LOCAL-only; EXP17 ends in L1 | `DRAM` | — | 3 | 🔁 |

*Row 11 output is L1 but is immediately sliced (12) and consumed by the interleaved O matmul (13), so the chain does not carry forward.

**Takeaway:** every `=` row is the *same shape* in both implementations, yet LOCAL runs several of them (1, 2, 9, 13, 17, 19) in DRAM while EXP17 runs them in L1 — identical shapes prove the L1 version is doable, so those DRAM placements are pure config choices, not shape-driven. Every unsigned (LOCAL-only) row is an op LOCAL adds that EXP17 doesn't have at all (reshards + hand-rolled heads + standalone SiLU), which is where the 38-vs-19 op-count gap comes from.

## 2. Same-shape ops — LOCAL config vs EXP17 config, side by side

These are the `=` rows from §1 (identical tensor shape in both). Read each row as: *for this shape, LOCAL does X, EXP17 does Y, and the codegen EMIT did Z.* The last column shows who runs it in L1. EXP17's extra fused ops that replace LOCAL's glue (`nlp_create_qkv_heads_decode`, SiLU-in-mul) are noted inline; LOCAL-only reshard/glue ops stay in §1.

| Op (shape) | LOCAL: mem + program | EXP17: mem + program | EMIT (`model_ttnn.py`): mem + program | L1 |
| --- | --- | --- | --- | :---: |
| input `rms_norm` `[.,4096]` | **`DRAM`**, 1-core, no program config | `L1 WS`, sharded RMSNorm (`decode_in/out_sharded=True`) | `L1 WS [32,192]@22c`, `program_config=None` (sharded via out memcfg), HiFi4 fp32-acc | EXP17 + EMIT |
| QKV `matmul` `[.,4096]·[4096,6144]` | **`DRAM`**, **no program config** (default matmul) | `L1_WIDTH_SHARDED`, `DRAMSharded` (`_dram_matmul_config`) | `L1 WS [32,64]@96c`, **`Matmul1D` mcast** `in0_block_w=2, per_core_N=2` (weights **interleaved**, not DRAM-sharded) | EXP17 + EMIT |
| RoPE q,k (head tensors) | **`DRAM`** | `L1` (in-attention) | `L1 interleaved` (cos/sin also moved to L1 interleaved) | EXP17 + EMIT |
| paged_update_cache | cache (DRAM) | cache (DRAM) | cache (DRAM); K/V head input reshard to `L1 HS [32,128]` | = all |
| SDPA decode | **`DRAM`** out, `SDPAProgramConfig 8×8` | `L1` out, sharded SDPA | **`DRAM`** in + out, default program config | EXP17 only (LOCAL≈EMIT DRAM) |
| concat heads `→[.,4096]` | `L1` (`nlp_concat_heads_decode`) | `L1` (`nlp_concat_heads_decode`) | `L1 WS [32,128]` (`nlp_concat_heads_decode`), then L1-interleaved reshape | = all |
| O `matmul` `[.,4096]·[4096,4096]` | **`DRAM`**, **no program config** | `L1_WIDTH_SHARDED`, `DRAMSharded` (`decode_attn_output_prg_config`) | `L1 WS [32,64]@64c`, **`Matmul1D` mcast** `in0_block_w=8, per_core_N=2` (weights **interleaved**) | EXP17 + EMIT |
| add attn residual `[.,4096]` | `L1 WS [32,128]@32c` | `L1 WS` | `L1 WS [32,64]@64c` | = all |
| post-attn `rms_norm` `[.,4096]` | `L1 WS`, `LNShardedMC 8×4, block_w=4, subblock_w=4` | `L1 WS`, sharded RMSNorm | `L1 WS [32,192]@22c`, `program_config=None`, HiFi4 fp32-acc | = all |
| gate `matmul` `[.,4096]·[4096,14336]` | **`DRAM`**, `Matmul1D` 64c, `in0_block_w=4, per_core_N=7` (weights interleaved) | `L1_WIDTH_SHARDED`, `DRAMSharded` 32c, `in0_block_w=4, per_core_N=14` (weights DRAM-width-sharded) | `L1 WS [32,160]@90c`, **`Matmul1D` mcast** `in0_block_w=2, per_core_N=5` (weights interleaved), **SiLU fused into this matmul** | EXP17 + EMIT |
| up `matmul` (same as gate) | **`DRAM`**, same 1D config | `L1_WIDTH_SHARDED`, `DRAMSharded` | `L1 WS [32,160]@90c`, `Matmul1D` `in0_block_w=2, per_core_N=5`, no fused act | EXP17 + EMIT |
| multiply gate·up `→[.,14336]` | **`DRAM`**, standalone (SiLU is a separate DRAM op) | `L1 WS [32,448]@32c`, **SiLU fused into the mul** | `L1 WS [32,160]@90c`, plain mul — **SiLU was fused into the gate matmul instead** | EXP17 + EMIT |
| down `matmul` `[.,14336]·[14336,4096]` | `L1 WS [32,512]@8c`, `DRAMSharded in0_block_w=14, per_core_N=16`, **8 cores** | `L1_WIDTH_SHARDED`, `DRAMSharded in0_block_w=14, per_core_N=4`, **32 cores** | `L1 WS [32,64]@64c`, **`Matmul1D` mcast** `in0_block_w=8, per_core_N=2` (weights **interleaved**, not DRAM-sharded); input reshard `L1 WS [32,256]@56c` | = all (EMIT 1D-mcast, LOCAL/EXP17 DRAM-sharded) |
| add output residual `[.,4096]` | `L1 WS [32,128]@32c` | `L1 WS` | `L1 WS [32,64]@64c` | = all |

Reading the "L1" column: on identical shapes, LOCAL keeps only **6 of 14** in L1 (post-attn residual/norm, concat, down, both residual adds); EXP17 keeps **all 14**; and the **EMIT keeps 13 of 14** in L1 (everything except SDPA, which it runs in DRAM like LOCAL). The six shapes where EXP17 is L1 and LOCAL is DRAM (input norm, QKV, RoPE, O, gate/up/mul) are exactly the shapes the emit already had in L1 — so the L1 version isn't just "provably doable", it's what the source graph did. The one difference between EXP17 and the emit is *how* they reach L1: EXP17 uses **DRAM-width-sharded matmuls**, the emit uses **1D-multicast matmuls over interleaved weights** (and fuses SiLU into the gate matmul, not the mul).

---

## 3. Why each op is / isn't L1 — from the logs only

Status key: ❌ **not attempted** (no candidate in any log) · 🔻 **tried and rejected** (measured slower) · ✅ **attempted and kept** · ⚙️ **required layout move** (reshard the path needs).

### 🔴 3a. NOT in L1 — why (the breaks)

| Op(s) | Status | Reason (source) |
| --- | :---: | --- |
| Input RMSNorm (row 1) — DRAM, 1-core, 94 µs | ❌ | The only norm trial targeted the post-attention path; it states "the remaining 94 µs LayerNorm row is the input RMSNorm before packed QKV… **was not the stage-review target**" (`sharded_residual_norm_trials.md`; `work_log.md` §Sharded Residual/Norm). |
| QKV / O matmul + inputs (rows 2, 13) — DRAM, no program config | 🔻 / ❌ | Input→L1 was **tried and rejected slower**: `l1_movement_trials.csv` "L1 attention input and prefill down input" = **1.307 ms vs 1.283 ms kept** (`work_log.md` step 13). That trial kept the interleaved matmul; a **DRAM-sharded attention matmul was not attempted** (no such candidate). |
| Head build reshape/permute/slice (rows 3–6, 12) — DRAM, hand-rolled | ❌ | No log tests `nlp_create_qkv_heads_decode` or an L1-sharded head path; QKV sweeps only covered packed-vs-separate topology (`qkv_projection_trials.csv`). |
| gate / up (rows 17, 19) — DRAM output, 1D multicast | ❌ | `gate_up_geometry_trials.csv` swept grid/`in0_block_w` for the **1D** config only and kept 64-core `in0_block_w=4` (1.059 ms). No candidate stored gate/up weights DRAM-width-sharded or used a DRAM-sharded matmul (unlike `down`), and the `DRAM` output was never A/B'd against an L1 output. |
| SiLU (row 18) — separate DRAM op | ❌ | No log tests fusing SiLU into the multiply. |

### 🟢 3b. IN L1 — why (kept / required)

| Op(s) | Status | Reason (source) |
| --- | :---: | --- |
| Post-attn residual + norm (rows 14–16) — L1 WS sharded | ✅ | A/B in `sharded_residual_norm_trials.md`: DRAM baseline 1.285 ms → sharded residual/norm+MLP-input+final-residual 1.129 ms; kept. |
| down matmul (row 22) — L1 WS, DRAM-sharded | ✅ | `down_geometry_trials.csv`: `in0_block_w=14` fastest valid (`56` failed L1 CB alloc); `work_log.md` step 6. |
| q/k/v & sdpa reshards (rows 7, 10) — L1 head layout | ⚙️ | Required by SDPA decode (needs head-sharded L1). |
| gated reshard (row 21) — 8-core L1 | ⚙️ | Required to feed the DRAM-sharded `down`. |
| final output → DRAM (row 24) | ⚙️ | Decoder output contract returns DRAM. |

**Net:** of the DRAM breaks in 3a, **one was tried and rejected** (attention input → L1, slower) and **the rest were never attempted** as L1/DRAM-sharded candidates. The only L1/sharding effort that landed (3b) went to the post-attn residual/norm and the `down` matmul.

---

## 4. What the breaks cost (from the perf report)

Pure chain-tax — layout moves and hand-rolled head glue a fully L1 + fused path would not run:

| Category | Ops | Approx µs |
| --- | --- | ---: |
| Reshard (Interleaved↔Sharded) | I2S ×7 (rows 7,10,14,21), S2I ×1 (row 24) | ≈20 |
| Head glue in DRAM (reshape/transpose/slice) | reshape ×3, transpose ×4, slice ×7 | ≈95 |
| Standalone DRAM SiLU (row 18) | 1 | 14 |
| **Total chain-tax** | | **≈130 µs of ~1020 µs** |

LOCAL runs **38 device ops**; EXP17's decode is **19** — the difference is almost entirely these reshard + glue ops.

---

## 5. Fix — extend the L1 chain across the whole layer

1. **Shard the input RMSNorm** like the post-attn one (reuse `_decode_residual_norm_program_config` + `residual_memcfg`). Not attempted yet.
2. **Keep the attention front L1** with a DRAM-sharded QKV/O matmul (see companion doc) so its output is L1, then use `nlp_create_qkv_heads_decode`. Note the logs show only the weaker "L1 input, interleaved matmul" was tried (and was slower) — the DRAM-sharded variant is untested.
3. **Keep gate/up in L1** (`memory_config=L1_WIDTH_SHARDED`) with DRAM-sharded weights like `down`, and **fuse SiLU into the multiply** (`ttnn.mul(gate, up, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])`), removing the DRAM SiLU op and the row-21 reshard. Not attempted yet.

Or structurally adopt EXP17's `RMSNorm1D` + `Attention1D` + DRAM-sharded `_OptimizedMLP`, which keep the L1 chain by construction.

---

*Evidence — LOCAL: `tt/optimized_decoder.py`, `optimized_decoder/tt_perf_report_decode_gate_up_geometry.txt`, `optimized_decoder/work_log.md`, `optimized_decoder/sharded_residual_norm_trials.md`, `optimized_decoder/*_trials.csv`. EXP17: `experiment-17/.../tt/optimized_decoder.py`, `models/common/modules/attention/attention_1d.py`, `models/common/modules/rmsnorm/rmsnorm_1d.py`.*
