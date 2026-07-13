# Bug: `paged_scaled_dot_product_attention_decode` diverges when valid KV spans >1 k_chunk

**Op:** `ttnn.transformer.paged_scaled_dot_product_attention_decode`
(`ttnn/cpp/ttnn/operations/transformer/sdpa_decode/`, compute kernel `sdpa_flash_decode.cpp`).
**Build:** tt-metal @ ASR branch (main-based), Blackhole P150x4 (P300×2), mesh (1,4), TP=4.
**Found via:** Qwen3.6-27B serving — decode output collapses into multilingual garbage once the
generated sequence passes a fixed cached length, regardless of block_size.

## Symptom

Teacher-forced, identical inputs, per-position next-token logits, **paged decode vs the non-paged
`scaled_dot_product_attention_decode` on internal concat caches (oracle)**, full 64-layer model:

| cached length (cur_pos) | paged-vs-oracle logits PCC |
|---|---|
| 64 … 127 | 0.999 (matches) |
| **128** | 0.988 (starts to drop) |
| 129 → 140 | 0.95 → 0.91 → … → **0.51** (catastrophic) |

- The drop is at **cur_pos = 128 regardless of block_size** (32 and 64 give byte-identical PCC drops)
  → it is NOT block-index related; it tracks an absolute valid-length / k_chunk boundary.
- RoPE and cur_pos are correct (host builds `freqs = outer(pos, inv_freq)` with the true `pos`, no
  128 wrap; matches the coherent oracle).
- The **non-paged** op with the **same** `SDPAProgramConfig` and `compute_kernel_config` does NOT
  diverge across the same multi-chunk boundary — only the **paged** path does.

## Isolation — it is the paged multi-chunk online-softmax

Varying only the paged op's `k_chunk_size`:

| k_chunk_size | collapse onset |
|---|---|
| 0 (auto → dynamic, ≈128) | cur_pos = 128 |
| 32 | cur_pos ≈ 96, worse (min PCC 0.19) |
| **512 (≥ cached length ⇒ single valid chunk)** | **none — all positions PCC ≥ 0.99** |

The paged op processes `S = page_table.padded_shape[-1] * block_size` (the whole allocated cache) in
`k_chunk_size` chunks, masked by `cur_pos`. It is correct while the **valid** KV (positions
`[0, cur_pos]`) fits in a **single** chunk, and diverges — worsening with position — as soon as the
valid KV spans **two or more** chunks (i.e. `cur_pos ≥ k_chunk_size`). The single-chunk case
(`k_chunk_size ≥ cur_pos`) is exact. ⇒ the defect is in the **paged cross-chunk online-softmax
reduction / boundary-chunk masking**, not in the QKᵀ/AV math (which the single-chunk and non-paged
paths share and get right).

## Minimal repro

`models/demos/blackhole/qwen36/tests/test_paged_vs_internal.py` (env `T`, `N_DEC`, `BLOCK_SIZE`,
`N_LAYERS`): prefill T tokens, then teacher-force N_DEC decode steps through BOTH the paged and the
internal-cache paths with identical tokens, print per-step logits PCC. Reproduces the pos-128 drop at
default k_chunk and its disappearance at k_chunk=512.

## Workaround (in use)

`attention/tp.py` paged decode branch, env `QWEN_PAGED_KCHUNK`: set the paged `SDPAProgramConfig`
`q/k_chunk_size` to a single-chunk value (≥ served context). Validated: serving pure decode with
`QWEN_PAGED_KCHUNK=512` produces coherent output for the whole generation (no collapse), vs a ~30-token
collapse at the default.

**Limit:** `k_chunk_size = 2048` overflows L1 — `Statically allocated circular buffers grow to
4671424 B > max L1 1572864 B` (`program.cpp:1554`). So single-chunk is only viable up to
`k_chunk ≈ 512` (context ≤ ~512). Longer contexts need the real fix.

## Ask

Fix the paged cross-chunk online-softmax in `sdpa_flash_decode.cpp` so paged decode matches the
non-paged op when valid KV spans multiple k_chunks. Candidate areas: the boundary/partial chunk mask
and the running-max/running-sum rescale across chunks on the paged path (`k_num_chunks`,
`k_chunk_start/end`, `window_start_*`, and the `cb_cur_max`/`cb_prev_max` rescale loop).

## ROOT CAUSE CONFIRMED + fix landed (2026-07-12)

The defect is the **cross-CORE tree reduction**, not the reader/gather. `max_cores_per_head_batch=1`
(all chunks of a head on one core → the correct within-core lazy-softmax; no cross-core softmax
correction) makes paged decode match the internal-cache oracle **for any context length**:

| paged config | paged-vs-internal PCC (cur_pos 64→144) |
|---|---|
| default (auto k_chunk, multi-core) | collapses at 128 → 0.51 |
| k_chunk=512 (single chunk, 1 core) | clean, but L1-caps context ≤512 |
| **max_cores_per_head_batch=1 + k_chunk=128 (multi-chunk, 1 core)** | **clean at all positions (≥0.99), any context** |

Validated end-to-end: serving pure decode with the single-core config stays coherent past cur_pos=128
(vs ~30-token multilingual collapse at default). **Landed as the default in the Qwen3.6 paged decode
branch (`attention/tp.py`)**: `max_cores_per_head_batch=1`, `k_chunk=128`; env `QWEN_PAGED_MAXCORES=0`
restores the stock config. Trade-off: one core per head is slower — acceptable for B=1; the proper
upstream fix is to correct the cross-core tree reduction so multi-core paged decode is usable.

**Upstream bug (still worth fixing):** `sdpa_flash_decode.cpp` tree reduction (`correction_block` /
`cb_cur_max`/`cb_prev_max` cross-core combine, ~line 506+) gives wrong results for PAGED multi-chunk
decode while the identical within-core path and the non-paged multi-chunk path are correct.
