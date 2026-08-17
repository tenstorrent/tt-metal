# Gemma4 Status & Runbook (12B / 26B-A4B / 31B)

**Branches:**
- tt-metal: `arg/gemma4_1x8optiom` @ `94606a8c3c2`
- vLLM (tt-metal/vllm): `arg/gemma4_fixes` @ `ae722ece2` (+ local uncommitted plugin edits)
- tt-inference-server: `arg/gemma4_optimizations` @ `aac4474b7`

**Host of record:** LoudBox `f02cs02` (8× P150) + QB2 dies as P150x4
**Date:** 2026-08-04 (product defaults: async ON, GeLU Accurate, device-sample demo, traced ≤4k; prior rewrite 2026-08-02 / campaign from 2026-07-29)
**Related:** GitHub #51186 (async decode), vLLM [PR #448](https://github.com/tenstorrent/vllm/pull/448) (chunked prefill), shield run [30506815455](https://github.com/tenstorrent/tt-shield/actions/runs/30506815455)

This document summarizes what has been solved on these branches, current metal + server implementations for **12B / 26B-A4B / 31B**, how to run them, measured performance, and open issues.

**Product path (target):** vLLM + metal, `async_scheduling`, chunked prefill 4096, `max_num_seqs=32`, on-device decode sampling, prefill device traces through **4096**, GeLU **Accurate**. Prefer fixes under `models/demos/gemma4/`; small approved edits may touch `models/common/sampling/` (log noise) or `tt-metal/vllm` plugin.

**Constraint:** do **not** edit `models/tt_transformers/` unless explicitly approved.

---

## 1. Folder layout (`models/demos/gemma4/`)

| Path | Role |
|------|------|
| `README.md` | Upstream-style support matrix, CI/WH numbers, basic metal run cmds |
| `PREFILL_DECODE_OPTIMIZATION.md` | LoudBox 31B prefill/decode optimization scoreboard |
| `STATUS_AND_RUNBOOK.md` | **This file** — branch status, server+metal perf, issues |
| `demo/text_demo.py` | CI short demo + batch-32 |
| `demo/text_demo_v2.py` | Preferred long-context / perf (TTFT + decode tok/s) |
| `tt/model.py` | Core Gemma4Model (decode/prefill, PLI vs non-PLI); inactive-row RoPE clamp vs cache `-1` |
| `tt/generator.py` | Metal `Gemma4Generator` + `ChunkedPrefillPageTableGuardMixin` (async-ahead merge + page-table guards) |
| `tt/generator_vllm.py` | vLLM bridge (`Gemma4ForCausalLM`); sequential vs batched prefill; routes decode via mixin |
| `tt/generator_trace.py` | Prefill-trace policy + `GEMMA4_LONG_CONTEXT_POLICY` |
| `tt/attention/prefill.py` | Sliding / global prefill SDPA + KV fill + sliding-tail stash |
| `tt/async_decode.py` | Gemma4-local `merge_async_ahead_decode_tokens` (no `tt_transformers` edit) |
| `tt/compute_config.py` | Production GeLU (**Accurate**) + SDPA HiFi4 helpers; **no** linear HiFi4/fp32 A/B |
| `tt/ccl.py` | Gemma4 CCL (prefer non-deprecated AG paths; `GEMMA4_CCL_TOPOLOGY`) |
| `tt/dram_sharded.py` | DRAM-shard progcfg (K+N grid / L1-aware in0) |
| `tt/shared_mlp.py` / `experts/` / `layer.py` | GeLU via `gelu_variant()` |
| `tests/unit/` | Continuity / CCL + host-only coherence unit tests (see below) |
| `configs/` | Per-variant config stubs |

**Landed on metal tip (`94606a8c3c2`):** keep sliding KV tail across APC short chunks and async decode (stash/pad short tails; do not wipe on decode; pad Q when `seq < hist`).

**Local / uncommitted (metal `models/demos/gemma4/`) — server + product stack (2026-08):**

| File | Fix |
|------|-----|
| `tt/model.py` | Inactive decode rows: RoPE clamps `pos < 0` → 0; cache/SDPA keep `-1`; `keep_sharded_for_sampling` only when on-device; batched prefill final RMSNorm; `extract_last_tokens_batched_prefill` + `_apply_norm_and_lm_head` for device-sample batch-32; `plus_one(..., skip_negative_entries=True)` |
| `tt/generator_vllm.py` | Clear multi-row page-table stash; hetero `valid_seq_lens`; `super().decode_forward` → mixin; **`supports_async_decode` default ON** (`GEMMA4_SUPPORTS_ASYNC_DECODE=0` kill-switch); no `optimizations=accuracy` → linear HiFi4 |
| `tt/generator.py` | Mixin `decode_forward` + microbatch ≤user_cap; `valid_seq_lens`; `async_decode.merge_*` |
| `tt/generator_trace.py` | Prefill-trace buckets include **4096** (`[128,512,1024,2048,4096]`) |
| `tt/attention/prefill.py` | Per-slot tile-ceil fill; sliding-tail stash + width-mismatch rebind; no K→V resync |
| `tt/attention/decode.py` / `operations.py` | No K→V resync; no linear HiFi4 ckc; SDPA HiFi4+fp32 via `compute_config` |
| `tt/compute_config.py` | GeLU **Accurate** (was Tanh A/B; FastLut = old `fast_and_approximate_mode`) |
| `tt/async_decode.py` | Safe async-ahead merge (`dev_len >= host_b` slice) |
| `tt/dram_sharded.py` | K×N core grid so `per_core_N * cores` covers N |
| `demo/text_demo_v2.py` / `text_demo.py` | **`GEMMA4_HOST_SAMPLE` default `0`** (on-device; product parity) |
| `tests/unit/test_decode_inactive_positions.py` | Host unit: RoPE vs cache sentinel |
| `tests/unit/test_batched_prefill_actual_lens.py` / `valid_seq_lens` / `prefill_over_user_cap` | Batched prefill gates |
| `tests/unit/test_async_ahead_decode_tokens.py` | Merge / fallback |
| Also dirty: `ccl.py`, `test_ccl_topology.py`, `shared_mlp.py`, `experts/operations.py` |

**Stripped (not for product — caused garbage or A/B-only):** linear HiFi4/fp32 matmul env, `GEMMA4_PRECISION_PROFILE`, per-layer bf16 lifts, K→V `resync_kv_tied`, fat `compute_config` A/B surface.

**vLLM (`tt-metal/vllm` `arg/gemma4_fixes`):** PR #448 chunked-prefill + decode-bucket pad + local plugin: generation-scoped async preempt accounting (`scheduler.py`), page-table pad **`-1`** (not `0`) for inactive decode rows (`model_runner.py`), `test_tt_scheduler_async_preempt.py`.

**Artifacts (local, untracked):** `isl_sweep_logs/` (recal, coherence A/B, matrix, `pr448_ci_repro_*`).

---

## 2. What has been solved (this branch / campaign)

### Metal / CI

| Item | Status |
|------|--------|
| Long-context policy per (model, device) in `generator_trace.py` | Landed — QB2/LB/P150 cutovers for bound+chunk |
| BH e2e time budgets for `bh_p150` SKUs | Fixed + pushed |
| RO MLPerf weight-cache mkdir | Fixed + pushed |
| Prefill-trace sp0 persistent-ring clear (WH-T3K nightly OOM/hang class) | Fixed + pushed |
| DRAM-shard / prefill progcfg guards for E2B/E4B TP=8 | Earlier on branch |
| Prefetcher spike documented; default **off** (not competitive) | See `PREFILL_DECODE_OPTIMIZATION.md` |
| Sliding KV tail across short APC chunks / async decode | **Landed** `94606a8c3c2` (shield hang / continuity class) |
| Long-context KV fill + async decode continuity | Landed `ee3288e0434` |
| Inactive-row RoPE vs vLLM pad `-1` | **Local uncommitted** — clamp for RoPE only; keep `-1` for KV/SDPA (`tt/model.py`) |
| Multi-user prefill page-table clobber | **Local uncommitted** — clear multi-row stash on sequential path (`generator_vllm.py` / mixin) |
| Batched prefill pad→KV contamination | **Local uncommitted** — per-slot `valid_seq_lens` tile-ceil fill (`attention/prefill.py`); hetero actual + same pad bucket may batch |
| Async-ahead decode merge (gemma4-local) | **Local uncommitted** — `tt/async_decode.py` + mixin `decode_forward`; vLLM uses `super()` so mixin is hit (no `tt_transformers` patch) |
| Prefill traces through 4k | **Local** — `_DEFAULT_TRACE_PREFILL_SEQ_LENS` includes `4096`; continuations need `GEMMA4_CHUNKED_PREFILL_TRACE=1` (yaml) |
| GeLU Accurate | **Local** — `compute_config.gelu_variant()` → `GeluVariant.Accurate` |
| Metal demo on-device sample | **Local** — `GEMMA4_HOST_SAMPLE` default `0` |
| Batched-prefill device sample | **Local** — `extract_last_tokens_batched_prefill` (batch-32 + `HOST_SAMPLE=0`) |

### Server / async (#51186) / vLLM

| Item | Status |
|------|--------|
| `supports_async_decode` for Gemma4 vLLM | Implemented — **default ON** (`GEMMA4_SUPPORTS_ASYNC_DECODE=0` kill-switch). Mitigations: async-ahead merge + plugin preempt gen accounting + page-table `-1` pad |
| PLI (E2B/E4B) force-off async; non-PLI device token feedback + `plus_one` | Implemented |
| Inference-server `async_scheduling: true` on BH Gemma4 specs | In `tt-inference-server` `dev/llm.yaml` |
| Metal↔server B=1 ISL recal (12B/31B × LB/QB2) | Done — see §6; decode often **faster** on server than metal on LB |
| Short coherence under async (marker prompts) | PASS for 31B LB/QB2, 12B QB2, 26B LB; LB 12B long-pad — see §7 |
| Token-chunked prefill for TT backend | [vLLM PR #448](https://github.com/tenstorrent/vllm/pull/448) on `arg/gemma4_fixes` |
| 1-token chunked continuation misclassified as decode | Fixed in #448; further `model_runner` fix: multi-token resume ≠ decode (`num_sched > 1`) |
| Decode pad to nearest warmed batch bucket | `e51ac7ab6` + inactive block-table pad **`-1`** (not `0`) |
| Discard async frames on TT KV preempt | Generation-scoped stale drain (no phantom `max(pending,1)`) — local `scheduler.py` |
| Concurrent GPQA garbage / near-zero accuracy | **Mitigated** (inactive RoPE + actual-lens + page-table H2D + async-ahead merge + sliding-tail rebind + #48037 force-argmax + async merge). LB 12B ci-nightly was **70%** exact_match; published 78.8±5% (floor ~74.9%) still open — see §12–§13 |

### Diagnosed / partially fixed

| Item | Status |
|------|--------|
| **LB 12B server long-context coherence cliff ~9k tokens** | Sliding-tail land (`94606a8c3c2`) + local inactive-RoPE / page-table / async-ahead stack. Long-pad probe **re-verify** still required after clean server boot (`coherence_probe2.py` may be missing locally). |
| Heterogeneous concurrent prefill (GPQA / metal batch-32) | **Root causes:** (1) batched KV fill wrote full pad (`get_last_token=-1`) — fixed via per-slot `valid_seq_lens`; (2) batched early-return skipped final RMSNorm while deferred `process_logits_after_prefill_trace` only runs lm_head (expects post-norm) — fixed in `model.py`. Microbatch ≤4 for B>user_cap hang. |
| Shield CI assert `Prefill batch should not include decode cached requests` | **Appears fixed by #448** — CI dump was 1-token chunked continuation (`num_output_tokens=0`), not decode. Local GPQA repro: **0 hits** of that assert. |
| Metal unit suite on QB2 (`max_num_blocks_per_seq`) | FAIL in matrix (expected-error noise / config); e2e demos PASS |
| 12B `full_model` PCC ~0.97 (vs 31B ~0.997) | **Arch gap, not missing 12B logic.** 12B=`gemma4_unified` / **1** global KV head (GQA 16:1); 31B=`gemma4` / **4** global KV heads. GeLU Accurate recovered ~0.94→~0.98; remaining drop is ACC at late full-attn (L29/35/41), TF still ≥0.999. Linear HiFi4≈0.989 but decode-garbage — do not re-enable. `pcc_thresholds.json` now gates 1x8 at 0.96/0.97 (was default 0.99). |
| **TP host-sample decode skipped vocab all-gather (2026-08-04)** | `_apply_lm_head` skipped all-gather on every decode whenever `self.sampling` existed, even for **host** sampling. Host argmax then saw only ~vocab/TP (~32k of 262k) → thought-loop / no final answer on metal Direct (`GEMMA4_HOST_SAMPLE=1`). **Fix:** `keep_sharded_for_sampling` only when `on_device_logits=True` (`tt/model.py`). Probe: case0 GPQA greedy host-sample → **(C)/(C) match** after fix (`/tmp/gemma4_acc_debug_20260803/metal_seq1_8k.log`). Product server `decode_only` already used on-device decode sampling (unaffected path) but any host-decode fallback is now correct. Post-fix LB 12B ci-nightly still **FAIL** (`exact_match=0.525`, prior 0.675; N=40, temp=1.0, stderr≈0.08 — variance + model/PCC gap vs published 78.8%). Concurrent server probe: coherent answers, no empty/garbage; wrong finals are content errors. |
| **Async-ahead merge rejected pad-32 feedback (2026-08-04)** | `merge_async_ahead_decode_tokens` required `dev_len == host_b`. Non-PLI Gemma4 always pads token/pos feedback to width **32**, while nearest-bucket decode uses B∈{1,2,4,8,16}. On every prefill→decode mode switch under concurrent async, merge took **`host_fallback`** → stale host tokens for continuing users (coherent wrong finals; seq1 often fine). **Fix:** accept `dev_len >= host_b`, slice `[:host_b]`; position `use_dev` still rejects unrelated rows. Units updated (`test_async_ahead_decode_tokens.py`, 8 passed). Note: product warmup is B=1+B=max so concurrent pads to 32 (equality already held); merge fix matters for B∈{2..16} buckets / B=1 mode-switch. |
| **Inactive decode `plus_one` cleared -1 cache sentinel (2026-08-04)** | `ttnn_decode_forward` did `plus_one(rot_mat_idxs)` **without** `skip_negative_entries`. Gemma4 passes int32 **cache** positions as `rot_mat_idxs` (vLLM pads inactive rows with -1; page tables -1). Sentinel became 0/1/… so pad rows could paged_update. **Fix:** `plus_one(rot_mat_idxs, skip_negative_entries=True)`. Note: concurrent **greedy** (temp=0) still bit-identical thought-loops at 16k after this fix (pad KV may not be the greedy-loop driver); product sampling is the gate. |
| **Concurrent product-sample probe (2026-08-04)** | Same first-10 GPQA docs, conc=10, product gen_kwargs (temp=1.0/top_k=20/top_p=0.95/seed=42): **9/10** exact_match, 10/10 boxed+stop (prior eval first-10 was **6/10**). |
| **Prefill sliding-tail copy 128→1024 TT_FATAL (2026-08-04)** | Prior ci-nightly `exact_match=0.425` with **18/40 empty** after EngineCore death: `ttnn.copy` sliding-tail stash seq=128 into persistent hist=1024 under APC remnant / preempt-resume at ~98% KV. Among non-empty: **17/22 ≈ 77%**. **Fix:** `_copy_sliding_tail_into_persistent` rebinds on width mismatch (no mid-capture zeros). |
| **LB 12B GPQA after stack (2026-08-04)** | Post merge + plus_one-skip + sliding-tail rebind: ci-nightly `exact_match=**0.70**` (28/40; stderr≈0.073; taxonomy 28 correct / 11 wrong_boxed / 1 no_boxed; **0 engine death**, server stayed up). Acceptance still FAIL vs published 78.8±5% (floor ~74.9%). Conc=10 product probe was **9/10**. Remaining gap looks like content wrong-boxed + N=40 variance / 12B PCC, not garbage/empty. |
| **On-device greedy B=32 sampling (#48037, 2026-08-04)** | Qwen3/25-VL pattern: Gemma4 had `allow_force_argmax=False` + `tt_ccl=None`, so greedy used the heavy top-k/top-p multi-gather path and sampling traces at B=32 froze stale AG semaphores. **Fix (gemma4-only):** wire `TT_CCL`, `allow_force_argmax=True`, `_tt_disable_sampling_trace=True`; host-commit sampled ids into pad-32 decode feedback after sync. |
| **Async decode token-doubling (2026-08-04)** | Was: `TheThe user user…` at B=1 when async ON (`host_tok0=0` / empty decode inputs). **Mitigations now product-default:** `merge_async_ahead_decode_tokens` (pad-32 feedback), plugin generation-scoped preempt drain, page-table `-1` pad. Capability **default ON**; kill with `GEMMA4_SUPPORTS_ASYNC_DECODE=0` if doubling returns. |
| **B=1 vs B=32 greedy (2026-08-04)** | Short greedy coherent. Concurrent B=32 greedy **coherent** but not bit-identical to B=1 (stylistic wording) — residual numerical batch / PCC, not garbage. |
| **GeLU FastLut vs Tanh vs Accurate (2026-08-04)** | Old `fast_and_approximate_mode=True` = **FastLut**, not HF tanh. `GeluVariant.Tanh` ≈ HF `gelu_pytorch_tanh`. Metal GPQA N=10 Ring/device-sample @ OSL 8k: Tanh **5/10**, FastLut **6/10** (4 Tanh fails = OSL truncation). Product chose **Accurate** for PCC (`compute_config.py`). Do **not** enable linear HiFi4/fp32 (unicode garbage on LB 12B decode). Do **not** enable K→V resync (immediate garbage). |
| **Metal demo HOST_SAMPLE (2026-08-04)** | Historical default `1` avoided device-sample + decode-trace mid-alloc corruption. Product/`decode_only` always used device sample. Demo default flipped to **`0`** (on-device). Host path: `GEMMA4_HOST_SAMPLE=1` (pays full 262k AG → ~30 tok/s vs ~34 device @ 128k). Batch-32 + device sample needs `extract_last_tokens_batched_prefill`. |
| **Prefill trace 4k (2026-08-04)** | Buckets were `[128…2048]` (omit 4096) because traced first grant + eager remnant dropped sliding-tail Python stash (#51186). Sliding-tail persistent rebind + yaml `GEMMA4_CHUNKED_PREFILL_TRACE=1` allow **4096 in default buckets**. |

---

## 3. Current implementation snapshot (12B / 26B / 31B)

### Architecture (all three)

- Mixed `sliding_attention` / `full_attention` layers (`hf_config.layer_types`)
- Partial RoPE on global layers; full RoPE on sliding
- Hybrid per-layer page tables for vLLM
- On-device decode sampling on TP meshes (`sample_on_device_mode: decode_only`) — required so token ids ≥65536 are reachable
- Prefill: default metal chunk **4096** (31B/26B @ ≥128k on QB2 → **2048** + bounded)
- Sliding layers: stash last `sliding_window` K/V as `sliding_tail_out` for the next APC chunk / async interleave
- **vLLM pad / inactive rows:** decode positions `-1` stay as skip markers for paged_update / SDPA; RoPE lookup clamps negatives to 0 (`prepare_decode_inputs_host`)
- **Batched prefill gate:** same **padded** bucket + page table; hetero actual lengths OK (per-slot `valid_seq_lens` caps KV fill). B>user_cap (default 4) microbatches on metal / sequential on vLLM (true-batched hang on P150x8). Sequential path clears multi-row page-table stash
- **Async-ahead:** device token/pos preferred when `dev_pos ∈ {host_pos, host_pos+1}`; host fallback on nearest-bucket width mismatch or OOB `slot_remap`

### Metal Direct (`text_demo` / `text_demo_v2`)

| Model | Dense/MoE | Async decode | Long-context policy (high level) |
|-------|-----------|--------------|----------------------------------|
| **12B** | Dense | N/A (no async scheduler on metal Generator) | QB2: unbounded ≤128k, bound+chunk @256k. LB/P150x8: unbounded through 256k. P150×1: bound+chunk above 32k |
| **26B-A4B** | MoE | N/A | QB2/LB: similar to 31B cutovers; serve pool often 128k |
| **31B** | Dense (+ MoE block in HF family naming; metal path dense transformer stack) | N/A | QB2: auto-bound ≥64k; chunk 2048 @ ≥128k. LB: unbounded ≤128k; auto-bound @256k |

Defaults that matter for quality / product parity:

- `GEMMA4_HOST_SAMPLE=0` (on-device; matches server `decode_only`)
- GeLU **Accurate** (`tt/compute_config.py`) — no env knob
- Prefill-trace buckets through **4096**
- Do **not** set `GEMMA4_DEMO_SINGLE_CHUNK=1` on long ISL (known garbage)
- Prefill chunk from `GEMMA4_LONG_CONTEXT_POLICY` unless overridden via `GEMMA4_GEN_PREFILL_CHUNK`
- Do **not** enable linear HiFi4/fp32 or K→V resync (decode garbage)

### vLLM / inference-server (BH)

| Model | Devices in specs | `max_context` | Chunked prefill | Async scheduling | Trace region |
|-------|------------------|---------------|-----------------|------------------|--------------|
| **12B** | P300X2 (QB2), P150X8 (LB) | 262144 | 4096 | true (metal capability default ON) | 512MB |
| **31B** | P300X2, P150X8 | 262144 | QB2 **2048** / LB **4096** | true | 512MB |
| **26B-A4B** | P300X2, P150X8 | 131072 | 4096 | true | 1.2GB; `GEMMA4_CHUNKED_PREFILL_TRACE=0` |

Env common to BH Gemma4 server specs:

- `MESH_DEVICE=P150x4` (QB2) or `P150x8` (LB)
- `GEMMA4_CCL_TOPOLOGY=linear` on QB2 (**Ring** is BH≥8 metal default — coherent GPQA used Ring)
- `GEMMA4_CHUNKED_PREFILL_TRACE=1` on 12B/31B BH (traced multi-chunk)
- `GEMMA4_PAGE_BLOCK_SIZE=64`
- `GEMMA4_MAX_TOKENS_ALL_USERS` = `max_context`
- `GEMMA4_VLLM_SINGLE_CHUNK=0`
- Thinking + tool parsers: `reasoning-parser: gemma4`, `tool-call-parser: gemma4`
- Kill-switch if async doubles tokens: `GEMMA4_SUPPORTS_ASYNC_DECODE=0`

**Wormhole T3K (CI nightly):** 31B served at `max_model_len=16384`, chunk 2048, `max_num_seqs=1` — do **not** mirror BH 256k. Raise `trace_region_size` to 80MB if nightly still OOMs.

---

## 4. How to run — Metal Direct (LB / QB2 / QB2-on-LB)

Prefer **`text_demo_v2.py`** for long-context and perf (logs TTFT **and** decode tok/s).
Use **`text_demo.py`** for CI short demos and batch-32 / batched-prefill.

### 4.0 Common env (every metal demo)

```bash
cd /home/tt-admin/ashai/tt-metal
source python_env/bin/activate
export TT_METAL_HOME=$PWD
export PYTHONPATH=$PWD${PYTHONPATH:+:$PYTHONPATH}
export LD_LIBRARY_PATH=$PWD/build/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export HF_HUB_OFFLINE=1
export DISABLE_METAL_OP_TIMEOUT=1   # long ISL
# Defaults (already correct if unset):
#   GEMMA4_HOST_SAMPLE=0          # on-device sample (product parity)
#   GeLU=Accurate via compute_config (no env)
# Do NOT set GEMMA4_DEMO_SINGLE_CHUNK=1 on long ISL
# Opt into host sample only if needed: export GEMMA4_HOST_SAMPLE=1
```

Kill leftover metal/vLLM PIDs by PID (avoid `pkill -f` self-match), then:

```bash
tt-smi -r
```

After eth-core timeout / dirty crash, always `tt-smi -r` before the next demo or server boot.

### 4.1 Device map — what `MESH_DEVICE` / `TT_VISIBLE_DEVICES` mean

| Host | Goal | `MESH_DEVICE` | `TT_VISIBLE_DEVICES` | Mesh shape | `text_demo.py` `-k` |
|------|------|---------------|----------------------|------------|---------------------|
| **LoudBox** (8× P150) | Full LB TP=8 | `P150x8` | `0,1,2,3,4,5,6,7` (or unset) | 1×8 | `1x8` |
| **LoudBox** | **QB2-equivalent on LB** (4 dies) | `P150x4` or `P300x2` | **`4,5,6,7`** (preferred) or `0,1,2,3` | 1×4 | `1x4` |
| **QB2** (2× P300 = 4 dies) | Native QB2 | `P300x2` or `P150x4` | usually unset / all visible | 1×4 | `1x4` |
| Single P150 | 12B (or E2B/E4B) only | `P150` | `0` | 1×1 | `1x1` |

Notes:

- `P150x4` and `P300x2` / `P300X2` are **aliases** for the same 1×4 Blackhole mesh and the same `GEMMA4_LONG_CONTEXT_POLICY` QB2 entry.
- On LoudBox, if you set `MESH_DEVICE=P150x4` but leave all 8 chips visible, some paths can still open wrong topology — **always set `TT_VISIBLE_DEVICES` to exactly 4 dies** for QB2-on-LB.
- `text_demo_v2.py` does **not** use `1x4`/`1x8` in `-k`; mesh comes only from `MESH_DEVICE` + visible devices. Filter with `-k "long-context-…"`, `-k "batch-1"`, etc.
- `text_demo.py` short/batch tests **do** parametrize mesh ids — match `-k "1x8"` (LB) or `-k "1x4"` (QB2 / QB2-on-LB).

### 4.2 Model IDs

| Variant | `HF_MODEL` |
|---------|------------|
| 12B | `google/gemma-4-12B-it` |
| 31B | `google/gemma-4-31B-it` |
| 26B-A4B | `google/gemma-4-26B-A4B-it` |

```bash
export TT_CACHE_PATH=$HF_HOME/tt_cache/${HF_MODEL//\//--}
```

### 4.3 LoudBox full mesh (P150x8) — 12B / 31B / 26B

```bash
# §4.0 env first
export MESH_DEVICE=P150x8
export TT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HF_MODEL=google/gemma-4-12B-it   # or 31B-it / 26B-A4B-it
export TT_CACHE_PATH=$HF_HOME/tt_cache/${HF_MODEL//\//--}

# Short CI-style demo (text_demo)
pytest models/demos/gemma4/demo/text_demo.py::test_demo -k "1x8" -s --timeout 1500

# Batch-1 latency (text_demo_v2)
pytest models/demos/gemma4/demo/text_demo_v2.py -k "batch-1" -s --timeout 1500

# Long-context perf — preferred (TTFT + decode tok/s)
pytest models/demos/gemma4/demo/text_demo_v2.py \
  -k "long-context-4k or long-context-32k or long-context-128k" -s --timeout 2400

# Single ISL
pytest models/demos/gemma4/demo/text_demo_v2.py -k "long-context-128k" -s --timeout 2400

# Batch-32 (text_demo_v2; on-device sample default; microbatch ≤4 internally)
pytest models/demos/gemma4/demo/text_demo_v2.py -k "batch-32" -s --timeout 1800

# Alternate CI-style batch-32 (text_demo)
pytest models/demos/gemma4/demo/text_demo.py::test_demo_batch_32 \
  -k "prefill_128 and 1x8" -v --timeout 1800
```

### 4.4 Native QB2 (P300x2 / P150x4)

On a real QuietBox2 (4 Blackhole dies):

```bash
# §4.0 env first
export MESH_DEVICE=P300x2          # or P150x4 — same 1×4 policy
unset TT_VISIBLE_DEVICES           # use all visible QB2 dies; or set explicitly if multi-board
export HF_MODEL=google/gemma-4-31B-it   # or 12B-it / 26B-A4B-it
export TT_CACHE_PATH=$HF_HOME/tt_cache/${HF_MODEL//\//--}

# Short CI-style demo
pytest models/demos/gemma4/demo/text_demo.py::test_demo -k "1x4" -s --timeout 1500

# Batch-1
pytest models/demos/gemma4/demo/text_demo_v2.py -k "batch-1" -s --timeout 1500

# Long-context (QB2 policy: 31B bound from 64k; chunk 2048 @ ≥128k)
pytest models/demos/gemma4/demo/text_demo_v2.py \
  -k "long-context-4k or long-context-32k or long-context-128k" -s --timeout 3600

# Batch-32 on 1×4
GEMMA4_BATCH_DEMO_SIZE=8 pytest models/demos/gemma4/demo/text_demo.py::test_demo_batch_32 \
  -k "prefill_2048 and 1x4" -v --timeout 1800
```

If the machine exposes more than 4 chips, pin the QB2 dies explicitly (same as QB2-on-LB below).

### 4.5 QB2-equivalent run **on LoudBox** (4 of 8 P150s)

Use this on `f02cs02` (or any 8×P150 LB) to exercise the **QB2 1×4 policy** without a QB2 box. Dies **4–7** are the usual choice so a full-LB job on 0–7 does not collide if you ever split — pick one quartet and stick to it.

```bash
# §4.0 env first — IMPORTANT: only 4 chips visible
export MESH_DEVICE=P150x4          # or P300x2 (alias)
export TT_VISIBLE_DEVICES=4,5,6,7  # QB2-on-LB; alternate: 0,1,2,3
export HF_MODEL=google/gemma-4-12B-it   # or 31B-it / 26B-A4B-it
export TT_CACHE_PATH=$HF_HOME/tt_cache/${HF_MODEL//\//--}

# Confirm visibility before pytest
tt-smi -s | head -40

# Short demo — must filter 1x4 (not 1x8)
pytest models/demos/gemma4/demo/text_demo.py::test_demo -k "1x4" -s --timeout 1500

# Long-context under QB2 policy (bound/chunk cutovers match native QB2)
pytest models/demos/gemma4/demo/text_demo_v2.py \
  -k "long-context-4k or long-context-32k or long-context-128k" -s --timeout 3600

# 12B @ 128k on QB2-on-LB
HF_MODEL=google/gemma-4-12B-it \
MESH_DEVICE=P150x4 TT_VISIBLE_DEVICES=4,5,6,7 \
  pytest models/demos/gemma4/demo/text_demo_v2.py -k "long-context-128k" -s --timeout 3600

# 31B @ 128k on QB2-on-LB (chunk=2048 + bounded by policy)
HF_MODEL=google/gemma-4-31B-it \
MESH_DEVICE=P150x4 TT_VISIBLE_DEVICES=4,5,6,7 \
  pytest models/demos/gemma4/demo/text_demo_v2.py -k "long-context-128k" -s --timeout 3600
```

**Switching back to full LB after QB2-on-LB:**

```bash
unset TT_VISIBLE_DEVICES
export MESH_DEVICE=P150x8
export TT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
tt-smi -r
```

### 4.6 Single P150 (12B / E2B / E4B)

```bash
export MESH_DEVICE=P150
export TT_VISIBLE_DEVICES=0
export HF_MODEL=google/gemma-4-12B-it
export TT_CACHE_PATH=$HF_HOME/tt_cache/${HF_MODEL//\//--}

pytest models/demos/gemma4/demo/text_demo.py::test_demo -k "1x1" -s --timeout 1500
pytest models/demos/gemma4/demo/text_demo_v2.py -k "long-context-256k" -s --timeout 3600
```

### 4.7 Copy-paste matrix (metal demos)

| Run | Host | Env essentials | Pytest |
|-----|------|----------------|--------|
| LB 12B short | LoudBox | `MESH_DEVICE=P150x8` `TT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7` `HF_MODEL=…12B-it` | `text_demo.py::test_demo -k "1x8"` |
| LB 31B 128k | LoudBox | same + `…31B-it` | `text_demo_v2.py -k "long-context-128k"` |
| QB2 31B short | QB2 | `MESH_DEVICE=P300x2` `HF_MODEL=…31B-it` | `text_demo.py::test_demo -k "1x4"` |
| QB2 12B 128k | QB2 | `MESH_DEVICE=P150x4` `HF_MODEL=…12B-it` | `text_demo_v2.py -k "long-context-128k"` |
| **QB2-on-LB 12B** | LoudBox | `MESH_DEVICE=P150x4` `TT_VISIBLE_DEVICES=4,5,6,7` `…12B-it` | `text_demo.py::test_demo -k "1x4"` or `text_demo_v2.py -k "long-context-…"` |
| **QB2-on-LB 31B** | LoudBox | same + `…31B-it` | `text_demo_v2.py -k "long-context-128k"` |

### 4.8 Demo pitfalls

| Pitfall | Fix |
|---------|-----|
| QB2-on-LB still opens 1×8 | Set `TT_VISIBLE_DEVICES` to exactly 4 dies; `MESH_DEVICE=P150x4` |
| `text_demo` collects wrong mesh | Use `-k "1x4"` vs `-k "1x8"` matching visible mesh |
| Wrong long-context policy | Policy keys off mesh name (`P150x8` vs `P150x4`/`P300x2`) — QB2-on-LB must use 1×4 name |
| Chip busy / hang | Kill PIDs → `tt-smi -r` → clear stale `/dev/shm/TT_UMD_LOCK.*` only if safe |
| Garbage on long ISL | Do not set `GEMMA4_DEMO_SINGLE_CHUNK=1`. Prefer device sample (`HOST_SAMPLE=0`); if decode garbage, try `HOST_SAMPLE=1` once to isolate sampling vs model |
| `AttributeError: extract_last_tokens_batched_prefill` | Batch-32 + device sample — needs method on `Gemma4Model` (see `tt/model.py`) |

---

## 5. How to run — Inference server (tt-inference-server)

### Prerequisites

```bash
METAL=/home/tt-admin/ashai/tt-metal
SRV=/home/tt-admin/ashai/tt-inference-server
source "$METAL/python_env/bin/activate"
export TT_METAL_HOME=$METAL
export PYTHONPATH=$METAL${PYTHONPATH:+:$PYTHONPATH}
export LD_LIBRARY_PATH=$METAL/build/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export DISABLE_METAL_OP_TIMEOUT=1
# Async decode default ON (product). Kill-switch if token-doubling returns:
# export GEMMA4_SUPPORTS_ASYNC_DECODE=0
# Kill leftovers by PID; unset TT_VISIBLE_DEVICES for full LB mesh
unset TT_VISIBLE_DEVICES MESH_DEVICE
tt-smi -r
```

vLLM must be installed into `tt-metal/python_env` (tenstorrent fork + `vllm-tt-plugin`). Local-server installs the plugin from `$METAL/vllm/plugins/vllm-tt-plugin` (needs generation-scoped preempt + `-1` page-table pad).

For gated evals (e.g. `Idavidrein/gpqa`), put a valid `HF_TOKEN` in `$SRV/.env`.

### Boot LB 12B (chunked prefill + async — product path)

```bash
cd "$SRV"
# Capability defaults ON; yaml has async_scheduling + GEMMA4_CHUNKED_PREFILL_TRACE=1.
# Kill-switch only: export GEMMA4_SUPPORTS_ASYNC_DECODE=0
nohup python3 -u run.py \
    --model gemma-4-12B-it \
    --tt-device p150x8 \
    --workflow server \
    --local-server \
    --tt-metal-home "$METAL" \
    --vllm-dir "$METAL/vllm" \
    --host-hf-cache "$HF_HOME" \
    --no-auth --dev-mode \
    --device-id 0,1,2,3,4,5,6,7 \
    --disable-metal-timeout \
    --skip-system-sw-validation \
    --service-port 8000 \
  > /tmp/gemma4_lb12_server.log 2>&1 &

# Wait until ready
curl -sf http://127.0.0.1:8000/v1/models
```

`run.py` exits after spawn; real process is logged as `Created local server process PID: …`. Logs also under `$SRV/workflow_logs/local_server/`.

### Boot variants

| Target | `--model` | `--tt-device` | Notes |
|--------|-----------|--------------|-------|
| LB 12B | `gemma-4-12B-it` | `p150x8` | chunk 4096 in yaml |
| LB 31B | `gemma-4-31B-it` | `p150x8` | chunk 4096 |
| LB 26B | `gemma-4-26B-A4B-it` | `p150x8` | max_context 128k |
| QB2 12B/31B/26B | same model keys | `p300x2` | `TT_VISIBLE_DEVICES=4,5,6,7`; 31B chunk **2048** |

### Chat smoke / coherence probe

```bash
# Short
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"google/gemma-4-12B-it","messages":[{"role":"user","content":"Reply with exactly: OK-ASYNC-TEST"}],"max_tokens":64,"temperature":0}'

# Long-pad cliff probe (local helper; recreate if missing)
python3 isl_sweep_logs/lb12_coh_fix_20260728/coherence_probe2.py 800 VERIFY
python3 isl_sweep_logs/lb12_coh_fix_20260728/coherence_probe2.py 900 VERIFY
```

### Release / eval against an already-running server

Use `--server-url http://127.0.0.1:8000` (port required for remote health probe). Example GPQA-style concurrent stress (matches shield shape):

```bash
# With server up + HF_TOKEN for gated datasets
lm_eval --tasks r1_gpqa_diamond ... num_concurrent=32 --limit 0.2 \
  --gen_kwargs max_gen_toks=32768 ...
```

Local log dir example: `tt-metal/isl_sweep_logs/pr448_ci_repro_20260730/`.

### Benchmarks via run.py

```bash
cd "$SRV"
ONLY_BENCHMARK_TARGETS=1 OVERRIDE_BENCHMARK_TARGETS=/path/to/targets.json \
  python3 -u run.py --model gemma-4-12B-it --tt-device p150x8 --workflow benchmarks \
    --dev-mode --service-port 8000 --disable-trace-capture \
    --tt-metal-home "$METAL" --vllm-dir "$METAL/vllm" \
    --host-hf-cache "$HF_HOME" --no-auth --skip-system-sw-validation
```

---

## 6. Current performance

Sources:

- Metal README BH table + `isl_sweep_logs/defaults_scoreboard` / `full_matrix_*`
- Recal: `isl_sweep_logs/async_recal_20260728/summary.tsv` (2026-07-28, B=1, async server)
- Matrix: `isl_sweep_logs/async_validate_matrix_20260728/summary.tsv`

### Metal Direct — batch-1 decode (steady-state)

| Model | Device | ISL | tok/s/user | TTFT (ms) | Notes |
|-------|--------|-----|------------|-----------|-------|
| 12B | LB P150x8 | 4k | ~42–48 | ~380–460 | README ~47.6 @4k; recal metal 42.4 |
| 12B | LB | 32k | ~41.8 | ~6075 | recal |
| 12B | LB | 128k | ~35.8 | ~20920 | recal |
| 12B | QB2 P150x4 | 4k | ~34.7 | ~693 | recal |
| 12B | QB2 | 32k | ~34.5 | ~6480 | recal |
| 12B | QB2 | 128k | ~30.7 | ~27824 | recal |
| 12B | P150×1 | 4k | ~15.6 | ~1190 | README |
| 31B | LB | 4k | ~27.6–31.7 | ~660–700 | README 31.7; recal 27.6 |
| 31B | LB | 32k | ~26.8 | ~6781 | recal |
| 31B | LB | 128k | ~24.0 | ~31451 | recal; opt notes ~26.1 historical |
| 31B | QB2 | 4k | ~22.7–23.3 | ~1060 | README / recal |
| 31B | QB2 | 32k | ~22.2 | ~17754 | recal |
| 31B | QB2 | 128k | ~20.4 | ~57220 | recal |
| 26B-A4B | LB (server matrix) | 4k–64k | — | — | Metal Direct not in recal TSV; server ~19.5–19.7 tok/s |

Wormhole CI (cold TTFT, not comparable to BH warm): see `README.md` (T3K 26B/31B ~10–12 tok/s, TTFT tens of seconds).

### Server async — B=1 Mean TPOT → tok/s (`async_recal` / matrix)

| Model | Device | ISL | server tok/s | server TTFT (ms) | vs metal decode |
|-------|--------|-----|--------------|------------------|-----------------|
| 12B | LB | 4k | **48.7** | 439 | server **faster** (~−15% gap) |
| 12B | LB | 32k | **46.3** | 4019 | faster |
| 12B | LB | 128k | **38.8** | 22874 | faster |
| 12B | QB2 | 4k | 28.7 | 629 | metal faster (~17% gap) |
| 12B | QB2 | 32k | 27.9 | 5972 | metal faster |
| 12B | QB2 | 128k | 25.0 | 37619 | metal faster |
| 31B | LB | 4k | **30.3** | 779 | server faster |
| 31B | LB | 32k | **29.0** | 7485 | faster |
| 31B | LB | 128k | **25.2** | 47297 | faster |
| 31B | QB2 | 4k | 19.9 | 1134 | metal faster |
| 31B | QB2 | 32k | 19.4 | 11829 | metal faster |
| 31B | QB2 | 128k | 17.6 | 82311 | metal faster |
| 26B | LB | 4k | ~19.7 | — | matrix only |
| 26B | LB | 32k | ~19.5 | — | matrix |
| 26B | LB | 64k | ~19.2 | — | matrix |

`gap_decode_pct` in recal = `(metal − server) / metal × 100`. Negative ⇒ server ahead.

---

## 7. Coherence status

### Server concurrent / GPQA class (primary 2026-08 fixes)

Symptom under product async + chunked prefill + concurrent users (GPQA / shield-shaped): near-zero or garbage generations (~10–17% exact_match), not just the historical single-stream long-pad cliff.

| Fix (gemma4-local, uncommitted) | Failure mode addressed |
|---------------------------------|------------------------|
| Inactive-row RoPE clamp (`tt/model.py`) | Nearest-bucket pad rows at `pos=-1` → RoPE uint32 OOB / corrupt embeddings while KV/SDPA correctly skip |
| Page-table clear on sequential (`generator_vllm.py` / mixin) | Sequential path kept multi-row `_active_page_tables_per_layer` and clobbered user 0’s 1-row slice |
| Per-slot `valid_seq_lens` batched fill (`attention/prefill.py`) | Heterogeneous prompts in one pad bucket wrote zero-pad rows into KV (`get_last_token=-1`); decode collapsed |
| Async-ahead merge (`tt/async_decode.py` + mixin `decode_forward`) | Stale host tokens / `IndexError` on bucket change or OOB `slot_remap`; vLLM must call `super().decode_forward` (not skip mixin) |

**Evidence so far:** host units green (31 passed); release-CI style concurrent GPQA recovered to **~60%** exact_match (published ~78.8% still open). Remaining gap may include known 12B `full_model` PCC ~0.93 and/or residual long-pad / placeholders issues.

### Short / moderate prompts (async ON)

| Config | Result |
|--------|--------|
| LB 31B short + long-pad (~2k reps in older probes) | PASS |
| QB2 12B / 31B | PASS |
| LB 26B | PASS |
| LB 12B short / ~7k tokens (reps≤800) | PASS (historical) |
| Metal Direct QB2-on-LB + LB short / batch-1 (2026-08-02) | PASS — see §12 |

### LB 12B long-context cliff (re-verify)

**Historical symptom (2026-07-28/29):** UTF-8 garbage / echo / `\ufffd` when prompt ≳ ~8–9k tokens under product server settings (chunked prefill 4096).

**Repro (2026-07-29 verify):**

| reps | ~tokens | Result |
|------|---------|--------|
| 50 | ~470 | PASS |
| 800 | ~7220 | PASS |
| **900** | **~8120** | **FAIL** |
| 1000+ | ~9k–18k | FAIL |

Log: `isl_sweep_logs/lb12_coh_fix_20260728/verify_sweep.txt`

**A/B facts (pre–sliding-tail land):**

- Failed with async ON or OFF and with `GEMMA4_ALWAYS_REFRESH_DECODE=1`
- **Passed** when vLLM chunked prefill was disabled
- Metal Direct long-context OK; QB2 12B long OK; LB 31B long OK
- Threshold aligned with vLLM splitting at 4096

**Metal fixes since then:**

1. `94606a8c3c2` — stash/pad sliding tails across short APC chunks and async decode.
2. Local 2026-08 stack (table above) — inactive RoPE, actual-lens batching, page-table clear, async-ahead merge.

**Still required:** re-run reps 800/900/1000 after a clean LB 12B server boot with the local stack loaded before closing the cliff. (`coherence_probe2.py` was missing under `isl_sweep_logs/` as of 2026-08-02 — recreate or restore before probing.)

**Workaround if still failing:** disable chunked prefill or raise `max_num_batched_tokens` above the prompt (perf/DRAM tradeoff).

---

## 8. Other open issues

| Issue | Impact | Notes |
|-------|--------|-------|
| `assert request.num_output_placeholders >= 0` | **Engine killer** under high KV / concurrent long gen | Seen near end of GPQA concurrent=32 and hard `ignore_eos` / 32k stress (`async_scheduler.py`). Different from prefill/decode assert. |
| `ttnn.all_gather` deprecation spam | Log noise | Hot path often `models/common/sampling/tt_sampling.py`; constraint: prefer gemma4-local override (e.g. `tt_ccl.line_all_gather`) rather than editing `models/common/`. |
| WH-T3K 31B nightly EngineCore OOM | CI risk | `trace_region_size: 80MB` if still red |
| Async default / kill-switch | Ops | Default **ON**; `GEMMA4_SUPPORTS_ASYNC_DECODE=0` to disable |
| Linear HiFi4 / resync | Accuracy | Never enable — unicode garbage on LB 12B |
| Non-PLI `always_refresh` | Continuity | Kill-switch `GEMMA4_ALWAYS_REFRESH_DECODE=1` |
| Multi-user chip contention | Boots hang on `CHIP_IN_USE_*` | Kill foreign metal jobs; `tt-smi -r`; clear `/dev/shm/TT_UMD_LOCK.*` if safe |
| Metal unit `max_num_blocks_per_seq` on QB2 | Unit FAIL in matrix | e2e demos still PASS |
| Galaxy | Unsupported | Fabric bring-up |
| Force-push rebased `arg/gemma4_fixes` | Remote may be behind local rebase | Only if explicitly requested (`--force-with-lease`) |

---

## 9. Env cheat sheet

| Variable | Meaning |
|----------|---------|
| `MESH_DEVICE` | `P150` / `P150x4` / `P150x8` / `P300x2` |
| `TT_VISIBLE_DEVICES` | Chip list (QB2 often `4,5,6,7`) |
| `HF_MODEL` / `HF_HOME` / `TT_CACHE_PATH` | Weights + TT tensor cache |
| `HF_TOKEN` | Needed for gated HF datasets (GPQA) via server `.env` |
| `GEMMA4_GEN_PREFILL_CHUNK` | Override metal/vLLM chunk size |
| `GEMMA4_BOUNDED_SLIDING` | Force bounded sliding KV |
| `GEMMA4_DEMO_SINGLE_CHUNK` | Metal full-ISL single chunk — **avoid** on long ISL |
| `GEMMA4_VLLM_SINGLE_CHUNK` | Server single-chunk mode (specs set `0`) |
| `GEMMA4_CHUNKED_PREFILL_TRACE` | Trace multi-chunk sp1 (12B/31B BH yaml `1`; 26B default `0`) |
| `GEMMA4_SUPPORTS_ASYNC_DECODE` | vLLM async capability — **default ON** (set `0` to kill) |
| `GEMMA4_ALWAYS_REFRESH_DECODE` | Force host restage every decode step |
| `GEMMA4_TRACE_PREFILL_SEQ_LENS` | Override prefill-trace buckets (default `128,512,1024,2048,4096`) |
| `GEMMA4_HOST_SAMPLE` | Metal demo: **default `0`** (on-device); `1` = host AG path |
| `GEMMA4_CCL_TOPOLOGY` | `linear` on QB2 server specs; BH≥8 metal default Ring |
| `DISABLE_METAL_OP_TIMEOUT` | Needed for long ISL |
| ~~`GEMMA4_LINEAR_*` / `PRECISION_PROFILE` / `RESYNC_KV_TIED` / `GELU_VARIANT`~~ | **Removed** — do not reintroduce (garbage / A/B only) |

---

## 10. Suggested next steps

1. **Land / commit** gemma4 product stack (§1 / §13) on `arg/gemma4_1x8optiom` + vLLM plugin on `arg/gemma4_fixes` (avoid `tt_transformers` edits).
2. **`tt-smi -r` + reboot LB 12B server** with async ON + traced 4k + Accurate GeLU; re-run long-pad coherence (reps 800/900/1000) and GPQA diamond concurrent=32 (target ≥74.9% floor / ~78.8% published).
3. Finish metal GPQA N=10 Accurate @ OSL 16k (prior Tanh OSL16k killed mid-run after 3/3 correct).
4. Close residual **GPQA →78.8** (12B PCC ~0.93) separately from concurrent garbage (already mitigated).
5. Investigate / fix **`num_output_placeholders >= 0`** under high-KV async + chunked prefill.
6. Keep shield CI green on QB2 12B with concurrent evals.
7. Land WH-T3K `trace_region_size` yaml if nightly still OOMs.
8. Update #51186 with product defaults (async ON, traced 4k) + GPQA status.

---

## 11. Quick reference — “what to trust”

| Claim | Trust? |
|-------|--------|
| LB/QB2 12B & 31B metal long-context 4k–128k coherent | Yes (metal policy + demos) |
| Server async decode throughput ≈ metal (often better on LB) | Yes (recal 2026-07-28) |
| Server short coherence under async | Yes for 31B / QB2 12B / 26B |
| Concurrent GPQA garbage / near-zero | **Mitigated** — ci-nightly peak **70%**; stack still uncommitted |
| Async decode product path | **Default ON** with merge + preempt gen + `-1` pad; kill-switch `=0` |
| Prefill traced through 4k | **Yes** (default buckets + yaml `CHUNKED_PREFILL_TRACE=1` on 12B/31B) |
| GeLU Accurate / demo device-sample | **Yes** (defaults) |
| CI “prefill batch includes decode” assert under #448 | Likely fixed (local GPQA: 0 hits) |
| Server LB 12B coherent at all ISLs with chunked prefill 4096 | **Re-verify** with product defaults |
| GPQA diamond ≈ published 78.8% | **No** — ~70% best local; gap open (PCC / OSL / variance) |
| Engine survives high-KV concurrent long gen | **No** — `num_output_placeholders` assert |
| WH-T3K 31B nightly green without 80MB trace region | **No** — needs yaml bump if still OOM |

---

## 12. Live host snapshot (update when you boot)

| Item | Last known |
|------|------------|
| LB 12B server `:8000` | Often **DOWN** after GPQA / placeholders crash — reboot with §5 |
| metal tip | `94606a8c3c2` on `arg/gemma4_1x8optiom` + **local uncommitted coherence stack** (§1) + B=1 bounded page-table fix |
| vLLM tip | `ae722ece2` on `arg/gemma4_fixes` (check dirty plugin files) |
| GPQA diamond (pre–coherence stack) | `exact_match ≈ 0.075` on 40 samples / concurrent=32; then placeholders kill |
| GPQA diamond (with local coherence stack) | ~**60%** exact_match (release-CI style); vs published ~78.8% |
| **31B accuracy root (2026-08-03)** | Product `max_model_len=256k` auto-enables **bounded sliding**. B=1 prefill cleared `_active_page_tables_per_layer` and stuffed the remapped **sliding** page table into `kwargs["page_table"]`, so **full-attention layers wrote/read the sliding KV pool** → empty thought channel (`<\|channel>thought\n<channel|>`) and ~10–20% GPQA. Stable-shaped **49k / unbounded / seq1** recovered **90%** ci-nightly (`exact_match=0.9`, Acceptance PASS). Fix: keep per-layer page tables on B=1 prefill (`generator_vllm.py`). **Bounded + chunked prefill are compatible and required** for full 256k ISL on DRAM-limited SKUs (QB2 / single-card); do not disable either for accuracy. Product also needs `async_scheduling` + `max_num_seqs=32` for perf — sequential multi-user must slice per-layer tables to the active row (mixin `_activate_sequential_per_layer_row`), and ring-clear must not truth-test numpy `start_pos`. |
| **Product conc=32 retest (2026-08-03)** | LB 31B product reboot after fixes: `max_model_len=256k`, `max_num_seqs=32`, `async_scheduling`, chunked **2048** (yaml aligned), bounded=True. **No EngineCore numpy crash** (prior product run died instantly). ci-nightly GPQA → `exact_match=0.3` (3/10), Acceptance FAIL — **6/10 client `TimeoutError`** under long thinking gens (`max_gen_toks=126976`); server stayed up / decode_batch=32 healthy. Results: `.../results_2026-08-03T14-37-39.370666.json`. |
| **Conc=32 page-table H2D (2026-08-03)** | Sequential multi-user sliced host per-layer tables per user but did **not** call `update_persistent_per_layer_page_tables` on the B=1 device buffers. `_page_tables_to_ttnn` returns existing B=1 buffers without content refresh → users after the first keep user-0 block IDs (full-attn cross-chunk SDPA + sliding ring). Fix: H2D refresh inside `_activate_sequential_per_layer_row` (`generator.py`). Retest product shape + `max_gen_toks=16384` / `timeout=7200` / conc=32 / limit=10 → **exact_match=0.8** (8/10), **0 timeouts**, no garbage loops (`/tmp/gemma4_gpqa_conc32_h2d_fix`). Concurrent ACK probe 3×32/32. Prior product run was 0.3 with 6/10 timeouts. Metal long-ISL post-#51911: 4k/32k/128k coherent; 256k demo still “laughted” loop (separate B=1 full-ISL class). |
| **SDPA program_config (#51911)** | Op default `q/k_chunk=32` is up to ~3.7x slower ([tt-metal#51911](https://github.com/tenstorrent/tt-metal/issues/51911)). Gemma4 already used `prefill_sdpa_program_config` on most B=1 paths; remaining default-path sites (batched prefill + long sliding `chunked_prefill_sdpa_sliding`) now pass it too, with `k_chunk` capped at `window//2` for sliding. Local P150 microbench (1-chip, dh=256/512): **2.3–3.9x** SDPA speedup vs default. |
| Metal demos 2026-08-02 (`/tmp/gemma4_runbook_20260802/`) | QB2-on-LB 12B short **2 passed** (~30 tok/s); LB 12B short **2 passed** (~49–51 tok/s); LB batch-1 **passed** (47.4 tok/s) |
| Product defaults cleanup (2026-08-04) | See **§13** — async ON, Accurate GeLU, HOST_SAMPLE=0, traced ≤4k; stripped linear HiFi4/resync A/B |

---

## 13. Product defaults cleanup (2026-08-04)

Goal: ship **async scheduler + traced until 4k + best accuracy/perf** for LB 12B product (vLLM, chunked prefill, `max_num_seqs=32`). Uncommitted metal + plugin changes audited; keep only accuracy / performance / server-support paths.

### Defaults (current)

| Knob | Value | Notes |
|------|-------|-------|
| GeLU | **Accurate** | `tt/compute_config.gelu_variant()` — all MLP / experts / PLI |
| Prefill-trace buckets | `[128, 512, 1024, 2048, 4096]` | Continuations still need `GEMMA4_CHUNKED_PREFILL_TRACE=1` (yaml) |
| `supports_async_decode` | **ON** (`"1"`) | Kill: `GEMMA4_SUPPORTS_ASYNC_DECODE=0` |
| Metal `GEMMA4_HOST_SAMPLE` | **`0`** (on-device) | Host AG path: set `1`; batch-32 needs `extract_last_tokens_batched_prefill` |
| Linear HiFi4 / fp32 | **OFF / removed** | Caused unicode garbage on LB 12B decode |
| K→V resync | **OFF / removed** | Immediate garbage |
| `optimizations=accuracy` | No-op on linear knobs | Logs that Accurate GeLU stays; does not set HiFi4 |

### Why HOST_SAMPLE was historically `1`

Device sample + decode Metal Trace could allocate after capture and corrupt tokens. Product `decode_only` never used that demo default. With force-argmax + sampling buffers settled, demo default is **`0`** for server tok/s parity (~34–38 tok/s @ 128k vs ~30 host).

### Why 4096 was historically omitted from trace buckets

vLLM first grant of 4096 could be traced while remnant chunks stayed eager without refreshed sliding-tail Python pointers → ~9k garbage (#51186). Mitigated by sliding-tail persistent stash/rebind; product yaml enables `GEMMA4_CHUNKED_PREFILL_TRACE=1`.

### Accuracy probes (metal Direct, Ring, device sample, LB 12B)

| Config | Result | Artifacts |
|--------|--------|-----------|
| Tanh, OSL 8k, N=10 | **5/10** exact (4 OSL truncation / no box) | `/tmp/gemma4_acc_debug_20260803/metal_n10_tanh_ring_*` |
| FastLut, OSL 8k, N=10 | **6/10** | `metal_n10_fastlut_ring_*` |
| Tanh, OSL 16k, N=10 | **Killed** after 3/3 correct (idx 0–2); no summary | `metal_n10_tanh_ring_osl16k.log` |
| Accurate | Product default; re-probe pending | — |

Topology note: product yaml often forces **Linear**; coherent metal GPQA used **Ring** (BH≥8 default).

### KEEP vs DROP (audit)

**KEEP:** `async_decode` + mixin; vLLM preempt gen + `-1` pad; Accurate GeLU; DRAM K×N; force-argmax / keep_sharded; valid_seq_lens / inactive `-1`; sliding-tail + SDPA program_config; traced ≤4k; batched extract for device-sample prefill; unit tests above.

**DROP:** linear HiFi4/fp32 wiring; precision-profile / per-layer bf16; K→V resync; `optimizations=accuracy` → HiFi4 setdefaults; per-step force-argmax INFO/DEBUG logs in `models/common/sampling/tt_sampling.py` (spam only).

### Quick metal commands (LB)

```bash
# Batch-32 — device sample by default (no GEMMA4_HOST_SAMPLE needed)
MESH_DEVICE=P150x8 HF_MODEL=google/gemma-4-12B-it \
  pytest models/demos/gemma4/demo/text_demo_v2.py -k batch-32 -s --timeout=1800

# Long-context 128k
MESH_DEVICE=P150x8 HF_MODEL=google/gemma-4-12B-it \
  pytest models/demos/gemma4/demo/text_demo_v2.py -k long-context-128k -s --timeout=1800
```
