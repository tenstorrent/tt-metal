# Hypotheses for the ~1.4 s/chunk runner-vs-test gap (ranked)

Compiled 2026-06-13 from a 4-agent repo sweep (construction lens, device/dispatch lens, MoE/gate lens,
input/rope lens) plus the prior investigation. Gap = **~1.4 s CONSTANT additive per chunk,
prefix-independent**. Both paths call `TtPrefillTransformer.forward_chunk`.

## Already ruled out (prior work + this sweep)
- Timer/sync placement (`PREFILL_PREFILL_SYNC`, `PREFILL_PRESYNC`) — no change.
- Per-layer ack inject (`PREFILL_DISABLE_LAYER_ACK`) — no change (inject is a cheap shm atomic).
- num_users (1 vs 2) — no change. Compile/warmup (2-pass cold/warm) — no change.
- Attention sequence length — `logical_n` is prefix-bounded in both (`mla.py:649`), and the gap is FLAT
  in prefix, so attention ramp is not it.
- **Program cache** — DISPROVEN as a difference: default-ON in C++ since PR #24073
  (`program_cache.hpp:131 is_enabled_=true`); the test's `enable_program_cache()` is a redundant no-op,
  the runner inherits it enabled. Both cache identically.
- **Gate mode** — CONFIRMED identical: both use `GateComputeMode.DEVICE_FP32` (pure on-device, no host
  topk, no per-chunk host round-trip). Runner default for kimi = DEVICE_FP32 (`runner_utils.py:76`).
- MoE forward host round-trips — NONE on this path (`return_intermediates`/`debug_token_count` both off).
- MoE construction config — identical (capacity_factor=8, experts_per_chip, dispatch buffer,
  `overlap_shared_expert_with_dispatch=True` hardcoded). Token input prep + rope identical and OUTSIDE
  the timed region; `indexed_rope` built ONCE in both paths (not per chunk).

## Live hypotheses (highest first)

### H1 — Sub-device-manager thrash + request-mode-only top-level clear  [MED-HIGH]  ← NEW this sweep
MoE does `load_sub_device_manager(sd_manager_id)` + `clear_loaded_sub_device_manager()` **every MoE
layer** (`tt_moe.py:493-494, 518-519`) → ~60×/chunk. In **request mode only**, the runner ALSO does a
top-level `clear_loaded_sub_device_manager()` after compile (`prefill_runner.py:578`, to let the H2D
service program validate), changing the baseline manager from the shared-CCL manager
(`tt_ccl.py:89-91`, loads its manager + stall group at init; MLA CCL ops keyed to its
`worker_sub_device_id`, `mla.py:666,912`) to the whole-chip default. The test never does this. A
dedicated benchmark exists (`tests/op_unit_tests/test_sub_device_load_clear_timing.py`) — implying
load/clear was already suspected expensive. 60 × ~10-20 ms ≈ 0.6-1.2 s ≈ the gap, and it's per-layer →
prefix-independent (constant). 
- **Tests:** `01a_standalone_chunked` (skips the line-578 clear → if it drops, strong signal); section
  timing `02` vs `08` localizes to the `moe` section; future: gate a skip of line-578 (risky, H2D needs it).

### H2 — Request-mode machinery as a whole (socket + metadata readback + line-578 clear)  [decisive split]
The request loop does `h2d_socket_sync` + a per-chunk `ttnn.to_torch(...).view(int32)` metadata readback
(`prefill_runner.py:408,414`) BEFORE the timer; PRESYNC excluded a one-time drain at timer start but not
host work interleaved during dispatch. Standalone-chunked bypasses ALL of it.
- **Test:** `01a_standalone_chunked`. standalone≈1.9 s ⇒ gap is request machinery; standalone≈3.3 s ⇒
  genuine `forward_chunk` compute (then H1 via section timing, or H3 via the sweep).

### H3 — mla_seq_len (61440 runner vs 56320 test) sizes KV buffer + rope cos/sin  [LOW-MED]
The only construction delta. `_chunked_kv_buf` (`mla.py:231`), the KV cache, AND the rope cos/sin
tensors (`rope.py:163`) are all sized to `mla_seq_len`. Static analysis says `ring_mla` reads are
`logical_n`-bounded so this should NOT cost compute — but the fused all-gather into `_chunked_kv_buf`
(`mla.py:644-656`) may touch the full buffer. Gap being flat-in-prefix is consistent with a
buffer-proportional (not logical_n-proportional) fixed cost.
- **Tests:** scaling sweep `01`(56320) / `00`(61440) / `06`(81920) / `07`(102400). Linear in MAX_SEQ_LEN
  ⇒ buffer-bound; flat ⇒ conclusively excluded. `05` adds section timing at 56320.

### H4 — fabric reliability_mode RELAXED_INIT (runner) vs STRICT_INIT (test)  [LOW]  ← NEW this sweep
Only genuine device-init divergence (`runner_utils.py:133` vs conftest default `STRICT_INIT`,
`conftest.py:484`). Init-time property, weak per-chunk suspect.
- **Test (future, needs 1-word code edit):** set runner to STRICT_INIT and re-time. Queue if H1/H2/H3 all
  miss.

## Experiment → hypothesis map
| exp | targets |
|-----|---------|
| 00 baseline | regression check + H3 anchor (61440) |
| 01 maxseq56320 | H3 (decisive point) |
| 01a standalone_chunked | **H1 + H2 (decisive split)** |
| 01b standalone_sections | H1 localize (no request machinery) |
| 02 runner_sections | localize mla vs moe (request mode) → H1 |
| 03 test_sections | fast-path localize + construction diff → H3 |
| 04 skip_ack_sync | isolate the 61 per-layer MLA syncs |
| 05 maxseq56320_sections | H3 localize |
| 06/07 maxseq 81920/102400 | H3 scaling (linearity) |
| 08 disable_ack_sections | attribute ack/sync share of moe/mla section |

## Not yet queued (need code edits — add after wave 1 if unresolved)
- Skip the line-578 `clear_loaded_sub_device_manager` in request mode (gate it) — direct H1 test.
- STRICT_INIT in the runner (H4).
- Finer MLA sub-section timing: split `update_padded_kv_cache` / `ring_mla` / `wkv_b2 linear`.
