# Plan: Path B — Fused matmul+CCL decode MLP (qwen3.6 galaxy, ring-40)

Branch: `ssinghal/qwen36_vlm` (in place — device tests need the in-place `build_Release`;
do NOT create a worktree). Fusion stays behind `QWEN36_FUSE_RS_MATMUL` (default OFF) until
the 1L PCC + perf gates pass, so the default demo/server path stays green throughout.

## Goal
Fuse the FF2 all-gather INTO the w2 matmul in the real decode MLP
(`tt/llama_decoder.py::_mlp_decode_qwen36`), on a ring-40 padded-consistent config, to
absorb the ~32%-of-token AllGather. All CCL via `tt_ccl` wrappers — never hand-built raw CCL
(raw `ttnn.experimental.*` + manual semaphore/sub-device/topology HUNG the board last session).

## M0 findings (already measured on-device — do not re-derive)
- The fused w1/w3 `matmul_line_reduce_scatter` path WORKS at M=32, padded dims
  (`w1_red_s`/`w3_red_s` = (1,1,32,960) at ring-24/3840). The reduce-scatter fusion is fine.
- The ONLY break is the FF2 all-gather: the decode `line_all_gather`
  (`use_optimal_ccl_for_llama=True`, `buffer_key="BINARY_MUL"`) collapses M 32→1 and emits
  the native width 2176, so `w2` (padded K=3840) rejects it
  (`TT_FATAL a_shape[-1]==b_shape[-2]: width=2176 height=3840`, `llama_decoder.py:289`).
- A clean ring matmul FORCES padding (native 2176 = 68 tiles shares no ring divisor with
  K=1280 = 40 tiles). Ring-24 pads N→3840 (+76%); ring-40 pads N→2560 (+18%) and divides
  K=1280 cleanly (40 tiles → 1/core). w2 at ring-40: K=2560, N=1280 (40 tiles, no pad).
- `num_to_coregrid(40)` → `CoreGrid(y=5,x=8)` = 40 cores. FF1_3 ring-40
  (M=32,K=1280,N=2560): in0_block_w=1, out_block_w=2, num_blocks=40 ✓.
  FF2 ring-40 (M=32,K=2560,N=1280): in0_block_w=2, out_block_w=1, num_blocks=40 ✓.

## Keystone facts (verified)
- Raw fused op (proven on BH GLX, num_links=2, global_cb=None, PCC 0.99999):
  `ttnn.experimental.llama_all_gather_matmul_async(input, weight, intermediate_buffer,
   dim=3, cluster_axis=1, mesh_device, multi_device_global_semaphore, ag_memory_config,
   mm_memory_config, topology, num_links, subdevice_id, program_config,
   compute_kernel_config, dtype, global_cb)`.
- Proven-on-BH reference test: `tests/ttnn/unit_tests/operations/ccl/test_llama_all_gather_matmul.py`
  (`-k ff2_qwen`). It builds sub-device/semaphores MANUALLY — the wrapper's job is to
  encapsulate that via tt_ccl's existing pools.
- Wrapper templates to MIMIC in `tt/llama_ccl.py`:
  - `matmul_line_reduce_scatter` (line 1284) — how a fused matmul+CCL wrapper pulls
    `reduce_scatter_buffers` / `gather_semaphore_handles` / `worker_sub_device_id` /
    `CCL_TOPOLOGY` and cycles `gather_idx`.
  - `line_all_gather` (line 1535) — how the decode all-gather pulls `all_gather_buffers` +
    semaphores + `worker_sub_device_id`.

---

## Task 1 — `tt_ccl.all_gather_matmul` wrapper (TDD, keystone)
Add a wrapper `all_gather_matmul(self, input_tensor_mesh, weight, dim=3, cluster_axis=1,
num_links, ag_memory_config, mm_memory_config, program_config, compute_kernel_config,
dtype, global_cb=None, buffer_key=None)` to `tt/llama_ccl.py`, wrapping
`ttnn.experimental.llama_all_gather_matmul_async`. It must source the intermediate
(all-gather) buffer + the cross-device semaphore from tt_ccl's existing pools exactly like
`line_all_gather`/`matmul_line_reduce_scatter`, use `self.worker_sub_device_id`,
`self.model_config["CCL_TOPOLOGY"]`, and cycle `gather_idx` after the call. global_cb=None.

**TDD:**
- RED: write `tests/test_all_gather_matmul_wrapper_pcc.py` that builds the bh_glx mesh +
  a real tt_ccl (mimic how `profile_decode_eager.py` / the model build constructs TT_CCL —
  read it; reuse the `bh_glx_mesh` fixture), constructs a sharded ring input + ring weight at
  the **ff2_qwen-proven dims** (use the exact dims/configs from `test_llama_all_gather_matmul.py`
  `ff2_qwen` so the only new variable is the wrapper), calls `tt_ccl.all_gather_matmul(...)`,
  and asserts PCC > 0.99 vs a torch reference (all_gather then matmul). Run it; it MUST fail
  with AttributeError (wrapper doesn't exist). Capture the failure.
- GREEN: implement the wrapper. Re-run; PCC > 0.99, output M preserved, no board hang.
- Use num_links from `model_config["GALAXY_NUM_LINKS"]` (=2 on BH).

**Acceptance:** wrapper unit test PASSES (PCC > 0.99), no hang, output shape correct.
**Files:** `tt/llama_ccl.py` (+ wrapper), `tests/test_all_gather_matmul_wrapper_pcc.py` (new).
**If you see ethernet/fabric timeout or `exit 134`:** STOP, report BLOCKED "possible board
hang" — do NOT retry (retrying wedges the board further; only the human can `tt-smi -r`).

## Task 2 — ring-40 decode-MLP config set + extend wrapper test (TDD)
Add a ring-40 config set LOCAL to the decode MLP (do NOT change global `RING_SIZE=24`; the
768/24-core padding is pervasive in attention QKVG). In `tt/qwen36_model_config.py`:
- Decode-only ring-40 weight memcfgs: `W1W3_RING40_MEMCFG` (k=1280, n=2560),
  `W2_RING40_MEMCFG` (k=2560, n=1280).
- `FF1_3_RING40_PROGCFG = matmul_1d_ring_config(1,32,1280,2560,40)`,
  `FF2_RING40_PROGCFG = matmul_1d_ring_config(1,32,2560,1280,40)`.
- Ring-40 sharded memcfgs: `SHARDED_FF12_RING40` (in K shard), `SHARDED_FF12_OUT_RING40`
  (N 2560), `FF2_IN_RING40` (2560), `REDUCE_SCATTER_OUT_RING40`, derived from 40 cores
  (`num_to_coregrid(40)`=5x8).
In `tt/llama_mlp.py`: add decode-only ring-40 weights `self.w1_ring40 / w3_ring40 / w2_ring40`
sharded/padded to the ring-40 dims (parameterize the existing dram-sharded-memcfg path; leave
the shared 3840 prefill ring weights `self.w1/w3/w2` untouched).

**TDD:**
- RED: a shape unit test `tests/test_ring40_mlp_config.py` (CPU-only OK) asserting the ring-40
  progcfg `num_blocks==40` and the per-device weight/memcfg shard widths (w1/w3 N=2560,
  w2 K=2560). Also extend the Task-1 wrapper test with a ring-40-dims case (PCC > 0.99 at
  K=2560,N=1280). Run; fails (keys/weights don't exist).
- GREEN: add the configs/weights. Re-run; shapes correct + ring-40 wrapper PCC > 0.99.

**Acceptance:** config shape test passes; wrapper PCC > 0.99 at ring-40 dims.
**Files:** `tt/qwen36_model_config.py`, `tt/llama_mlp.py`, tests above.

## Task 3 — wire fused ring-40 FF2 into `_mlp_decode_qwen36` (TDD, integration)
In `tt/llama_decoder.py::_mlp_decode_qwen36`, in the `_fuse_rs` branch: switch w1/w3 to the
ring-40 weights/progcfgs (so RS output is 2560/4=640 per col, M=32), then REPLACE the broken
`line_all_gather` + separate w2 matmul (lines ~279-298) with ONE
`mlp.tt_ccl.all_gather_matmul(ff_ring, mlp.w2_ring40, dim=3, cluster_axis=1,
num_links=GALAXY_NUM_LINKS, ag_memory_config=FF2_IN_RING40, mm_memory_config=FF2_OUT_RING40,
program_config=FF2_RING40_PROGCFG, compute_kernel_config=hifi2, dtype=bf8b, global_cb=None)`,
then the existing `line_all_reduce` on the w2 output. Keep gated behind `QWEN36_FUSE_RS_MATMUL`.

**TDD (the test is the 1L profiler — it builds a real tt_ccl + real decode):**
- RED: run the 1L profiler with fusion ON; confirm it currently fails (it does — M0).
  `QWEN36_FUSE_RS_MATMUL=1 QWEN36_N_LAYERS=1 QWEN36_PROFILE_LAYER_TYPE=linear
   QWEN36_PROFILE_DECODE_STEPS=4 python -m pytest --noconftest
   models/demos/qwen3_6_galaxy_v2/tests/profile_decode_eager.py -s -x`
- GREEN: wire the fused FF2. Re-run; it must complete 4 decode steps AND the first decode
  token must equal the unfused baseline `220` (prefill is identical; first token is
  deterministic). Then run the SAME command with fusion OFF and confirm next_tok sequence
  matches fusion ON for all 4 steps (fused == unfused == correctness gate).

**Acceptance:** 1L profiler fusion ON completes; next_tok sequence == fusion OFF (PCC-equiv
correctness). Demo with fusion default-OFF unchanged.
**Files:** `tt/llama_decoder.py`.

## Task 4 — measure the device-kernel win (1L Tracy)
Profile fused-ON vs fused-OFF at 1L (linear layer) and report MLP device-kernel µs.
`QWEN36_FUSE_RS_MATMUL={0,1} QWEN36_N_LAYERS=1 QWEN36_PROFILE_LAYER_TYPE=linear
 python -m tracy -p -v -r --op-support-count 20000 -m pytest --noconftest
 models/demos/qwen3_6_galaxy_v2/tests/profile_decode_eager.py -s`
Aggregate: drop rows > 1e9 ns (wrap cutoff); critical-path = max-over-devices per logical op.
Baseline: unfused MLP w2 RS+AG ≈ 58 µs/layer. Report fused MLP µs/layer + the delta, and the
padding-trim contribution (3840→2560).

**Acceptance:** documented fused vs unfused MLP device-kernel µs + delta. If win real and
PCC holds → recommend default-on (separate decision); else keep gated + document.

---

## Guardrails (every task)
- `tt_ccl` wrappers only — never raw CCL + manual semaphore/sub-device (board-hang risk).
- Fusion default OFF until Task 3 + Task 4 gates pass.
- Board hang (ethernet/fabric timeout, exit 134) → BLOCKED, escalate to human (no auto-reset).
- "We only care about device-kernel duration" for perf.
- Commit per task only when its test passes. Reviewed by Codex downstream.
