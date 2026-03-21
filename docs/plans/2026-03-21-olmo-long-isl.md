# OLMo Long-ISL Support (16k / 32k / 64k, batch=1) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extend OLMo-3.1-32B-Think Galaxy to serve 16k, 32k, and 64k input sequences at batch=1, both in the standalone demo and via tt-inference-server vLLM.

**Architecture:**
- **16k**: Add 16384 to `support_seqlens` → pre-allocated CCL buffers → traced prefill (same path as 8k). One-line CCL change; MLP dtype automatically handled by the existing `seq_len in support_seqlens` check.
- **32k / 64k**: Already designed as eager-mode (no trace). The CCL barrier-sync fallback path is used. The `intermediate_memory_config=DRAM` fix committed in the last session should already unblock these. The task is to verify they work end-to-end and wire them into vLLM.
- **vLLM**: Increase `max_context` in `model_spec.py` to 65664 so the server allocates a KV cache large enough for 64k sequences.

**Tech Stack:** TTNN, `llama_ccl.py` (TT_CCL), `demo_olmo_decode.py`, `tt-inference-server` model spec, `patch_olmo_image.sh`.

**Environment setup for every command:**
```bash
cd /home/tt-admin/ssinghal/tt-metal
export ARCH_NAME=wormhole_b0
export TT_METAL_HOME=$(pwd)
export PYTHONPATH=$(pwd)
source python_env/bin/activate
```

---

## Background: How `support_seqlens` Controls Traced vs. Eager Prefill

`llama_ccl.py` line 119:
```python
self.support_seqlens = [8192, 4096, 2048, 1024, 128]
```

A seqlen in this list gets:
1. **Pre-allocated persistent DRAM buffers** for each CCL op (ring_reduce_scatter, line_all_gather).
2. **Traced prefill** — `generator.py` line 227: `if prefill_seq_len not in support_seqlens: enable_trace = False`.
3. **bfloat8_b dtype** in MLP `minimal_matmul` — required to match the typed persistent buffer.

A seqlen NOT in this list (e.g. 32768, 65536):
- Uses the **barrier-sync fallback** path (no persistent buffer).
- Runs **eager mode** only.
- MLP `minimal_matmul` uses `dtype=None` (bfloat16).
- Works as long as `intermediate_memory_config=DRAM` is set on `ring_reduce_scatter` (already done in last commit).

`demo_olmo_decode.py` line 309:
```python
MAX_TRACE_SEQLEN = max(tt_model.tt_ccl.support_seqlens)  # = 8192 currently
use_trace = padded_prefill_len <= MAX_TRACE_SEQLEN
```

Adding 16384 to `support_seqlens` raises `MAX_TRACE_SEQLEN` to 16384 and auto-enables traced 16k prefill.

---

## Task 1: Enable 16k Traced Prefill (demo)

**Files:**
- Modify: `models/demos/llama3_70b_galaxy/tt/llama_ccl.py:119`

**Step 1: Make the one-line change**

```python
# llama_ccl.py line 119 — change from:
self.support_seqlens = [8192, 4096, 2048, 1024, 128]
# to:
self.support_seqlens = [16384, 8192, 4096, 2048, 1024, 128]
```

**Step 2: Run the 16k demo test**

```bash
pytest models/demos/llama3_70b_galaxy/demo/demo_olmo_decode.py::test_OLMo_demo[isl-16k-b1] \
  -v -s --timeout=600 2>&1 | tail -30
```
Expected: PASS. TTFT ~2.5–3s (2× the 8k TTFT of ~1.2s). Output coherent.

If it hangs: check that `intermediate_memory_config=DRAM` is on `ring_reduce_scatter` (already committed), and that the MLP dtype is bfloat8_b for seq_len=16384 (automatically handled).

**Step 3: Verify warmup includes 16k**

Check the log output contains:
```
Warming up prefill traces for all supported sequence lengths
Prefilling User 1, use_batched_prefill: False  # for seqlen=16384
```

**Step 4: Commit**

```bash
git add models/demos/llama3_70b_galaxy/tt/llama_ccl.py
git commit -m "feat(olmo): add 16384 to support_seqlens for traced 16k ISL prefill"
```

---

## Task 2: Verify 32k and 64k Eager Prefill (demo)

**Files:**
- Read: `models/demos/llama3_70b_galaxy/demo/demo_olmo_decode.py` (isl-32k-b1, isl-64k-b1 test cases)
- No code changes expected — the `intermediate_memory_config=DRAM` fix from the last session should already make these work.

**Step 1: Run the 32k demo test**

```bash
pytest models/demos/llama3_70b_galaxy/demo/demo_olmo_decode.py::test_OLMo_demo[isl-32k-b1] \
  -v -s --timeout=900 2>&1 | tail -30
```
Expected: PASS. TTFT ~14–28s (eager, proportional to seqlen). Output coherent.

**Step 2: Run the 64k demo test**

```bash
pytest models/demos/llama3_70b_galaxy/demo/demo_olmo_decode.py::test_OLMo_demo[isl-64k-b1] \
  -v -s --timeout=1800 2>&1 | tail -30
```
Expected: PASS. TTFT ~28–56s. Output coherent.

**Step 3: If 32k/64k hang — diagnose**

Check for:
- `TT_FATAL: ... barrier semaphore deadlock` → `intermediate_memory_config=DRAM` not set on all `ring_reduce_scatter` calls. Search for any remaining calls missing it: `grep -n "ring_reduce_scatter\|reduce_scatter_minimal" models/demos/llama3_70b_galaxy/tt/llama_ccl.py`.
- OOM / `TT_THROW: ... out of memory` → KV cache too large. Reduce `page_max_num_blocks` in the test case.
- `TT_FATAL: ... M_block_size` assertion in `olmo_model_config.py` → FF2 config for very long seqlen not handled. Fix: extend the M_block_size selection in `olmo_model_config.py` for seq_len > 16384.

**Step 4: Commit if any fixes were needed**

```bash
git add models/demos/llama3_70b_galaxy/tt/
git commit -m "fix(olmo): ensure 32k/64k eager prefill works end-to-end"
```

---

## Task 3: Wire Long ISLs into vLLM (tt-inference-server)

**Files:**
- Modify: `/home/tt-admin/ssinghal/tt-inference-server/workflows/model_spec.py` (max_context)
- Modify: `/home/tt-admin/ssinghal/tt-inference-server/benchmarking/benchmark_config.py` (add long-ISL benchmarks)
- Modify: `/home/tt-admin/ssinghal/tt-inference-server/benchmarking/benchmark_targets/model_performance_reference.json` (add perf targets)

**Step 1: Update max_context in model_spec.py**

Find the OLMo entry (currently `max_context=8320`) and raise it to 65664 (capacity for 64k ISL + 128 OSL):

```python
# workflows/model_spec.py ~line 1514
max_context=65664,  # 64k ISL + 128 OSL: 8208 blocks × 64 tok/block / 8 batch_per_group
```

> Note: The vLLM KV cache allocates blocks to fill available DRAM. With `max_context=65664`, the paged attention manager will allocate enough blocks for the longest supported sequence. Verify the device has sufficient DRAM (each additional 4096 blocks × 64 × 128 × 2 × 64L ≈ 4.3 GB).

**Step 2: Start the server and verify max_model_len**

After rebuilding the patched image (Task 4), start the server and check the log:
```
max_seq_len=65664
```

**Step 3: Add long-ISL benchmark cases to benchmark_config.py**

Find the OLMo benchmark config and add ISL=16384, 32768 con=1 entries:
```python
BenchmarkTask(isl=16384, osl=128, max_concurrency=1, num_prompts=4),
BenchmarkTask(isl=32768, osl=128, max_concurrency=1, num_prompts=2),
```
(Small num_prompts because long-ISL prefill is slow; 2–4 samples is enough for throughput measurement.)

**Step 4: Add perf reference targets**

In `model_performance_reference.json`, add entries for the new ISLs with conservative targets (e.g., TTFT ≤ 6s for 16k, ≤ 14s for 32k; output tok/s ≥ 12).

**Step 5: Run the long-ISL benchmarks**

```bash
cd /home/tt-admin/ssinghal/tt-inference-server
python run.py \
  --model OLMo-3.1-32B-Think \
  --workflow benchmarks \
  --no-auth --skip-system-sw-validation --tt-device galaxy
```

Verify TTFT and output tok/s for 16k and 32k con=1.

**Step 6: Commit**

```bash
cd /home/tt-admin/ssinghal/tt-inference-server
git add workflows/model_spec.py benchmarking/
git commit -m "feat(olmo): extend max_context to 64k + long-ISL benchmark tasks"
```

---

## Task 4: Rebuild Patched Docker Image

**Files:**
- Script: `/home/tt-admin/ssinghal/tt-inference-server/scripts/patch_olmo_image.sh`

**Step 1: Rebuild with long-ISL support baked in**

```bash
cd /home/tt-admin/ssinghal/tt-inference-server
TT_METAL_HOME=/home/tt-admin/ssinghal/tt-metal bash scripts/patch_olmo_image.sh
```

This re-runs `docker create` → `docker cp` → `docker commit`, picking up the updated `llama_ccl.py` (with 16384 in `support_seqlens`) from TT_METAL_HOME.

**Step 2: Retag for auto-resolution**

```bash
docker tag \
  ghcr.io/.../vllm-tt-metal-src-dev-ubuntu-22.04-amd64:0.11.0-b0c2f63-8f36910-olmo \
  ghcr.io/.../vllm-tt-metal-src-dev-ubuntu-22.04-amd64:0.11.0-b0c2f63-8f36910
docker tag \
  ghcr.io/.../vllm-tt-metal-src-dev-ubuntu-22.04-amd64:0.11.0-b0c2f63-8f36910-olmo \
  ghcr.io/.../vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.11.0-b0c2f63-8f36910
```

**Step 3: Smoke test the server**

```bash
python run.py \
  --model OLMo-3.1-32B-Think \
  --workflow server \
  --docker-server \
  --no-auth --skip-system-sw-validation --tt-device galaxy
```

Send a test request at 16k ISL:
```bash
python -c "
import requests, json
prompt = 'A ' * 16000
r = requests.post('http://localhost:8000/v1/chat/completions', json={
    'model': 'allenai/OLMo-3.1-32B-Think',
    'messages': [{'role': 'user', 'content': prompt}],
    'max_tokens': 10
})
print(r.json()['choices'][0]['message']['content'])
"
```
Expected: coherent 10-token response within 10s.

---

## Task 5: Update BRINGUP_LOG.md

**File:** `models/demos/llama3_70b_galaxy/BRINGUP_LOG.md`

Add a new dated session entry recording:
- ISLs now supported: 16k (traced), 32k (eager), 64k (eager)
- TTFT numbers from demo and vLLM
- Block hash of the commit

```bash
cd /home/tt-admin/ssinghal/tt-metal
git add models/demos/llama3_70b_galaxy/BRINGUP_LOG.md
git commit -m "docs(olmo): record long-ISL support results"
```

---

## Summary of Changes

| File | Change |
|------|--------|
| `tt/llama_ccl.py` | Add 16384 to `support_seqlens` |
| `demo/demo_olmo_decode.py` | No change needed (isl-16k/32k/64k tests already exist) |
| `tt-inference-server/workflows/model_spec.py` | `max_context=65664` |
| `tt-inference-server/benchmarking/benchmark_config.py` | Add ISL=16k, 32k con=1 tasks |
| `tt-inference-server/benchmarking/benchmark_targets/...json` | Add perf targets for new ISLs |
| `scripts/patch_olmo_image.sh` | Re-run to rebuild patched image |

## Known Risks

| Risk | Mitigation |
|------|-----------|
| 16k CCL buffer allocation OOM | Buffer is 2× 8k size (~14 MB/device for RS, ~7 MB for AG). Total overhead for adding 16384 is modest; unlikely to cause OOM. |
| 32k/64k barrier-sync hang | `intermediate_memory_config=DRAM` already committed. If still hangs, add to all `ring_reduce_scatter` call sites in eager path. |
| vLLM KV cache OOM at 64k | KV cache is paged — vLLM allocates blocks to fill DRAM. Galaxy has ample DRAM. Monitor with `kv_cache_usage` in scheduler stats. |
| 16k warmup trace capture slow | First-request trace capture for seqlen=16384 takes ~3× the traced execution time. Done once at startup in `warmup_prefill_traces`. |
