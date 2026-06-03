# Ring Joint SDPA — Chunked-Prefill Latent-V Perf Sweep Findings

Branch: `skrstic/ring_joint_sdpa_optional_latent_v_fix`
Date: 2026-06-02
Test: `tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py::test_ring_joint_attention_create_chunked_perf_table`

## Fixed configuration for this sweep

| Param | Value |
|---|---|
| Model | `kimi50k` (MLA: nhq=16/ring, nhk=1, d_q=d_k=576, d_v=128, Q bf16, KV bf8) |
| seq len (per-device Q rows / chunk) | 1280 |
| q_chunk | 64 |
| Ring | size 4 (sp=4), tp=2 → two rings of 4, all 8 devices |
| chunk_size | 5120 (= 1280 × sp 4), total_seq 56320, 11 chunks |
| SDPA grid | 12×10 → 11 wide × 10 = **110 cores** (col 11 = CCL) |
| Hardware | P150_X8 (8× Blackhole P150) |

`Math Util` = measured device-kernel FLOPs / theoretical, per chunk. Per-chunk work is
rectangle (Q_chunk vs prefix K/V, non-causal) + triangle (Q_chunk vs current chunk, causal),
so util climbs with prefix size across chunks 0→10.

---

## Hardware / topology findings (P150_X8)

1. **Device-count bug.** `MeshConfig.detect()` globbed `/dev/tenstorrent/*`, which includes the
   `by-id` symlink dir → counted **9 devices**, forcing an invalid `[1,9]` mesh. Fixed by
   counting only digit-named entries (`tests/nightly/sdpa_perf_utils.py`).

2. **Wrong grid assumed.** `NON_GALAXY_GRID` defaults to `(11,10)`; this box is **12×10**.
   With 11 the SDPA grid was 10 wide (100 cores, CCL on col 10) instead of 11 wide
   (110 cores, CCL on col 11). The 110-core grid is ~12–15% faster — all tables below use it.

3. **No native ring of 4.** The interconnect is a **cube graph (Q3)**, not a 2×4 grid/torus:
   ```
   0:{1,2,4}  1:{0,3,5}  2:{0,3,6}  3:{1,2,7}
   4:{0,5,6}  5:{1,4,7}  6:{2,4,7}  7:{3,5,6}
   ```
   The default row-major mapping `[0,1,2,3]` is **not** a 4-cycle (`1-2` is not an edge), so
   `tp=2/sp=4` failed with `Could not find any forwarding direction from src (M0,D0) to dst (M0,D3)`.
   The cube's faces *are* 4-cycles. Pinning `physical_device_ids=[0,1,3,2, 4,5,7,6]` makes each
   sp-row a real face — Ring A `0-1-3-2-0`, Ring B `4-5-7-6-4` — with tp links `0-4,1-5,3-7,2-6`
   all present. This forms two valid rings of 4. The ring-of-8 (`MeshShape(1,8)`) works because
   the cube has a Hamiltonian cycle through all 8.

---

## The latent-V toggle

Single module flag `CHUNKED_LATENT_V` in the test file:

- **`True`** — V in latent space: V is rematerialized on-device from the first `d_v` columns
  of the (single-head, shared) latent K. No separate V tensor, no V all-gather.
  Wired via `head_dim_v=d_v` and passing `input_v / joint_v / persistent_output_buffer_v = None`.
- **`False`** — separate decompressed V tensor (nhv heads), with its own all-gather (original path).

Reference V in latent mode = `K_full[..., :d_v]`; `take_heads()` already broadcasts the single
KV head across the 16 Q heads (GQA). Latent-V chunked accuracy passes (PCC ≥ ~0.999).

---

## Results — LATENT V (`CHUNKED_LATENT_V=True`)

### k_chunk = 256
| Chunk | logical_n | prefix_K | Duration (ms) | Cores | FLOPs (G) | Math Util |
|------:|----------:|---------:|--------------:|------:|----------:|----------:|
| 0 | 5120 | 0 | 0.870 | 110 | 73.82 | 27.9% |
| 1 | 10240 | 5120 | 1.710 | 110 | 221.46 | 42.6% |
| 2 | 15360 | 10240 | 2.438 | 110 | 369.10 | 49.8% |
| 3 | 20480 | 15360 | 3.250 | 110 | 516.74 | 52.3% |
| 4 | 25600 | 20480 | 3.978 | 110 | 664.38 | 54.9% |
| 5 | 30720 | 25600 | 4.785 | 110 | 812.02 | 55.8% |
| 6 | 35840 | 30720 | 5.523 | 110 | 959.66 | 57.1% |
| 7 | 40960 | 35840 | 6.331 | 110 | 1107.30 | 57.5% |
| 8 | 46080 | 40960 | 7.055 | 110 | 1254.94 | 58.5% |
| 9 | 51200 | 46080 | 7.857 | 110 | 1402.58 | 58.7% |
| 10 | 56320 | 51200 | 8.594 | 110 | 1550.21 | 59.3% |

### k_chunk = 512
| Chunk | logical_n | prefix_K | Duration (ms) | Cores | FLOPs (G) | Math Util |
|------:|----------:|---------:|--------------:|------:|----------:|----------:|
| 0 | 5120 | 0 | 0.891 | 110 | 73.82 | 27.3% |
| 1 | 10240 | 5120 | 1.552 | 110 | 221.46 | 46.9% |
| 2 | 15360 | 10240 | 2.424 | 110 | 369.10 | 50.1% |
| 3 | 20480 | 15360 | 3.065 | 110 | 516.74 | 55.4% |
| 4 | 25600 | 20480 | 3.827 | 110 | 664.38 | 57.1% |
| 5 | 30720 | 25600 | 4.436 | 110 | 812.02 | 60.2% |
| 6 | 35840 | 30720 | 5.323 | 110 | 959.66 | 59.3% |
| 7 | 40960 | 35840 | 5.952 | 110 | 1107.30 | 61.2% |
| 8 | 46080 | 40960 | 6.717 | 110 | 1254.94 | 61.4% |
| 9 | 51200 | 46080 | 7.325 | 110 | 1402.58 | 63.0% |
| 10 | 56320 | 51200 | 8.221 | 110 | 1550.21 | 62.0% |

### k_chunk = 640
| Chunk | logical_n | prefix_K | Duration (ms) | Cores | FLOPs (G) | Math Util |
|------:|----------:|---------:|--------------:|------:|----------:|----------:|
| 0 | 5120 | 0 | 0.864 | 110 | 73.82 | 28.1% |
| 1 | 10240 | 5120 | 1.610 | 110 | 221.46 | 45.2% |
| 2 | 15360 | 10240 | 2.319 | 110 | 369.10 | 52.3% |
| 3 | 20480 | 15360 | 3.032 | 110 | 516.74 | 56.0% |
| 4 | 25600 | 20480 | 3.743 | 110 | 664.38 | 58.4% |
| 5 | 30720 | 25600 | 4.448 | 110 | 812.02 | 60.0% |
| 6 | 35840 | 30720 | 5.165 | 110 | 959.66 | 61.1% |
| 7 | 40960 | 35840 | 5.880 | 110 | 1107.30 | 61.9% |
| 8 | 46080 | 40960 | 6.584 | 110 | 1254.94 | 62.7% |
| 9 | 51200 | 46080 | 7.294 | 110 | 1402.58 | 63.2% |
| 10 | 56320 | 51200 | 8.004 | 110 | 1550.21 | 63.7% |

### k_chunk = 768 → **OOM**
`Statically allocated circular buffers ... grow to 1620496 B which is beyond max L1 size of 1572864 B` (over by ~47.6 KB).

---

## Results — NON-LATENT / separate V (`CHUNKED_LATENT_V=False`)

### k_chunk = 256
| Chunk | logical_n | prefix_K | Duration (ms) | Cores | FLOPs (G) | Math Util |
|------:|----------:|---------:|--------------:|------:|----------:|----------:|
| 0 | 5120 | 0 | 0.995 | 110 | 73.82 | 24.4% |
| 1 | 10240 | 5120 | 1.803 | 110 | 221.46 | 40.4% |
| 2 | 15360 | 10240 | 2.565 | 110 | 369.10 | 47.3% |
| 3 | 20480 | 15360 | 3.351 | 110 | 516.74 | 50.7% |
| 4 | 25600 | 20480 | 4.118 | 110 | 664.38 | 53.1% |
| 5 | 30720 | 25600 | 4.902 | 110 | 812.02 | 54.5% |
| 6 | 35840 | 30720 | 5.655 | 110 | 959.66 | 55.8% |
| 7 | 40960 | 35840 | 6.447 | 110 | 1107.30 | 56.5% |
| 8 | 46080 | 40960 | 7.205 | 110 | 1254.94 | 57.3% |
| 9 | 51200 | 46080 | 7.999 | 110 | 1402.58 | 57.7% |
| 10 | 56320 | 51200 | 8.754 | 110 | 1550.21 | 58.2% |

### k_chunk = 384
| Chunk | logical_n | prefix_K | Duration (ms) | Cores | FLOPs (G) | Math Util |
|------:|----------:|---------:|--------------:|------:|----------:|----------:|
| 0 | 5120 | 0 | 1.108 | 110 | 73.82 | 21.9% |
| 1 | 10240 | 5120 | 1.787 | 110 | 221.46 | 40.7% |
| 2 | 15360 | 10240 | 2.512 | 110 | 369.10 | 48.3% |
| 3 | 20480 | 15360 | 3.363 | 110 | 516.74 | 50.5% |
| 4 | 25600 | 20480 | 4.041 | 110 | 664.38 | 54.1% |
| 5 | 30720 | 25600 | 4.781 | 110 | 812.02 | 55.8% |
| 6 | 35840 | 30720 | 5.613 | 110 | 959.66 | 56.2% |
| 7 | 40960 | 35840 | 6.310 | 110 | 1107.30 | 57.7% |
| 8 | 46080 | 40960 | 7.015 | 110 | 1254.94 | 58.8% |
| 9 | 51200 | 46080 | 7.857 | 110 | 1402.58 | 58.7% |
| 10 | 56320 | 51200 | 8.553 | 110 | 1550.21 | 59.6% |

### k_chunk = 512
| Chunk | logical_n | prefix_K | Duration (ms) | Cores | FLOPs (G) | Math Util |
|------:|----------:|---------:|--------------:|------:|----------:|----------:|
| 0 | 5120 | 0 | 1.098 | 110 | 73.82 | 22.1% |
| 1 | 10240 | 5120 | 1.724 | 110 | 221.46 | 42.2% |
| 2 | 15360 | 10240 | 2.572 | 110 | 369.10 | 47.2% |
| 3 | 20480 | 15360 | 3.218 | 110 | 516.74 | 52.8% |
| 4 | 25600 | 20480 | 4.047 | 110 | 664.38 | 54.0% |
| 5 | 30720 | 25600 | 4.689 | 110 | 812.02 | 56.9% |
| 6 | 35840 | 30720 | 5.526 | 110 | 959.66 | 57.1% |
| 7 | 40960 | 35840 | 6.179 | 110 | 1107.30 | 58.9% |
| 8 | 46080 | 40960 | 6.997 | 110 | 1254.94 | 59.0% |
| 9 | 51200 | 46080 | 7.633 | 110 | 1402.58 | 60.4% |
| 10 | 56320 | 51200 | 8.486 | 110 | 1550.21 | 60.1% |

### k_chunk = 640
| Chunk | logical_n | prefix_K | Duration (ms) | Cores | FLOPs (G) | Math Util |
|------:|----------:|---------:|--------------:|------:|----------:|----------:|
| 0 | 5120 | 0 | 0.969 | 110 | 73.82 | 25.0% |
| 1 | 10240 | 5120 | 1.731 | 110 | 221.46 | 42.1% |
| 2 | 15360 | 10240 | 2.468 | 110 | 369.10 | 49.2% |
| 3 | 20480 | 15360 | 3.201 | 110 | 516.74 | 53.1% |
| 4 | 25600 | 20480 | 3.931 | 110 | 664.38 | 55.6% |
| 5 | 30720 | 25600 | 4.661 | 110 | 812.02 | 57.3% |
| 6 | 35840 | 30720 | 5.400 | 110 | 959.66 | 58.4% |
| 7 | 40960 | 35840 | 6.137 | 110 | 1107.30 | 59.3% |
| 8 | 46080 | 40960 | 6.868 | 110 | 1254.94 | 60.1% |
| 9 | 51200 | 46080 | 7.589 | 110 | 1402.58 | 60.8% |
| 10 | 56320 | 51200 | 8.323 | 110 | 1550.21 | 61.2% |

### k_chunk = 768 → **OOM**
Same ceiling: `grow to 1620496 B which is beyond max L1 size of 1572864 B`.

---

## Comparison & conclusions

**Last-chunk (chunk 10, full prefix) summary:**

| k_chunk | Latent ms | Latent util | Separate-V ms | Separate-V util | Latent speedup |
|--------:|----------:|------------:|--------------:|----------------:|---------------:|
| 256 | 8.594 | 59.3% | 8.754 | 58.2% | 1.9% |
| 384 | — (not run) | — | 8.553 | 59.6% | — |
| 512 | 8.221 | 62.0% | 8.486 | 60.1% | 3.1% |
| 640 | 8.004 | 63.7% | 8.323 | 61.2% | 3.8% |
| 768 | OOM | — | OOM | — | — |

- **Latent V is consistently faster** (no V all-gather; V rematerialized locally from K), and the
  gap widens with larger k_chunk (1.9% → 3.8% at the full-prefix chunk). Early chunks benefit most
  (e.g. chunk 0 at k=256: 0.870 vs 0.995 ms, ~13%).
- **Larger k_chunk → faster + higher util** in both modes (fewer/larger K tiles, better MM
  subblock utilization), up to the L1 ceiling.
- **L1 ceiling = k_chunk 768** for both modes at seq=1280/q=64 — the static CB total (1,620,496 B)
  exceeds the 1.5 MB L1 by ~48 KB. Best stable config here is **k_chunk = 640, latent V**
  (63.7% peak util, 8.004 ms full-prefix chunk).
- Note: latent 768 and non-latent 768 OOM at the *same* CB total — at this config the V CBs are
  not the dominant term pushing over the edge; the K-chunk-driven CBs are.

## Reproduce

Set `CHUNKED_LATENT_V` (True=latent / False=separate) at the top of the test, then:
```
scripts/run_safe_pytest.sh \
  "tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py::test_ring_joint_attention_create_chunked_perf_table[kimi50k-q64-k<K>-chunk5120]"
```
Sweep scaffolding (sp=4/tp=2 cube-face ring, grid 12×10, per-device chunk 1280, k_chunk_sizes list)
lives in the same file under "TEMP sweep" comments — revert before merge.
