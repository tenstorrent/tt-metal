# OLMo-3.1-32B Open Issues

Two open issues for OLMo-3.1-32B-Think on TG (Galaxy, 8×4 mesh, 32 chips).

---

## Issue 1: 64K ISL Prefill Hang

### Summary

Prefilling a 64K-token sequence hangs (NOC deadlock on multiple devices). 32K ISL works reliably.

### Root Cause (Hypothesis)

OLMo prefill uses fully **synchronous CCL** (`ttnn.reduce_scatter` and `ttnn.all_gather`, `num_links=1`)
for both `ring_reduce_scatter` and `ring_all_gather` in `llama_ccl.py`. Each sync CCL op leaves
Ethernet ring fabric completion signals outstanding. At 64K ISL, 64 layers × ~6 CCL ops per layer
= ~384 sequential sync CCL calls, saturating the ring fabric before the forward pass completes.

A single 32K warmup run (done during server load in `generator.py`) primes the ring fabric enough
for a subsequent 32K inference prefill to succeed. The same approach does not appear to be sufficient
for 64K — even with a 64K warmup run first.

Llama 70B does not have this issue because it uses **async CCL** for prefill
(`reduce_scatter_minimal_async` + `all_gather_async`). OLMo was switched to sync CCL because
earlier async CCL attempts caused NOC hangs at 8K+ ISL — but the root cause of those async
hangs was not fully isolated.

### Current Code State

**`models/demos/llama3_70b_galaxy/tt/llama_ccl.py`**

```python
# ring_reduce_scatter — OLMo prefill path (line ~1259)
if self.is_olmo and self.mode == "prefill":
    ttnn_tensor_out = ttnn.reduce_scatter(
        input_tensor_mesh, dim,
        cluster_axis=cluster_axis,
        memory_config=memory_config,
        topology=ttnn.Topology.Ring,
        num_links=1,
        subdevice_id=None,
    )

# ring_all_gather — OLMo prefill path (line ~1409)
if self.is_olmo and self.mode == "prefill":
    ttnn_tensor_out = ttnn.all_gather(
        input_tensor_mesh, dim,
        cluster_axis=cluster_axis,
        topology=ttnn.Topology.Ring,
        num_links=1,
        memory_config=memory_config,
    )

# line_all_gather — OLMo prefill path (line ~1344)
if self.is_olmo and self.mode == "prefill":
    ttnn_tensor_out = ttnn.all_gather(
        input_tensor_mesh, dim,
        cluster_axis=cluster_axis,
        topology=topology,
        num_links=1,
        memory_config=memory_config,
    )
```

**`models/demos/llama3_70b_galaxy/tt/generator.py`** — warmup at server load:

```python
# Single warmup at largest supported ISL (currently 64K if max_seq_len >= 64K)
warmup_seqlen = max(sl for sl in [8192, 16384, 32768, 65536] if sl <= max_sl)
self.long_isl_warmup_seqlens = [warmup_seqlen]
```

### How to Reproduce

```bash
cd /path/to/tt-metal
export ARCH_NAME=wormhole_b0
export TT_METAL_HOME=$(pwd)
export PYTHONPATH=$(pwd)
export HF_MODEL=~/models/OLMo-3.1-32B-Think
source python_env/bin/activate

pytest models/demos/llama3_70b_galaxy/demo/text_olmo_demo.py \
    -k "long-64k-b1" -v 2>&1 | tee olmo_64k_hang.txt
```

The test will hang after printing `"Prefilling User 1"`. `tools/tt-triage.py` will show NOC hangs
on multiple ERISC cores across devices.

### Proposed Fix Directions

1. **Enable async CCL for OLMo prefill** (aligns with Llama): Restore `reduce_scatter_minimal_async`
   + `all_gather_async` for OLMo prefill in `llama_ccl.py`. Previous attempts deadlocked at 8K+
   ISL because `barrier_semaphore` was used without `persistent_buffers` (which are only allocated
   for ISLs ≤ 4096). Need to investigate why Llama's async CCL works at 128K with the same
   `barrier_semaphore` path and OLMo's did not.

2. **Persistent buffers for large ISLs**: Allocate `persistent_buffers` for 8K/16K/32K/64K in
   `get_ring_prefill_reduce_scatter_buffers()`. The dev team identified this as the "correct fix"
   but it costs ~300 MB per seqlen at 32K, which may OOM.

3. **`ttnn.synchronize_device` per-layer**: Adding a device sync between every prefill layer
   drains ring fabric signals (the `DEBUG_PREFILL_LAYERS=1` approach that worked in March). This
   is slow (~12s overhead per 32K pass) but reliable. Used as last resort.

---

## Issue 2: Long-Generation Accuracy (Garbage / Repetition After ~530 Decode Steps)

### Summary

With `bfloat8_b` KV cache and `sliding_window_size=4096` active in decode, the model produces
coherent output for the first ~530 decode steps, then output degrades to garbage or repetitive
tokens. This is a **hardware precision accumulation bug**, not a model behavior issue.

### Root Cause (Confirmed)

The KV cache stores K and V tensors in `bfloat8_b` (8-bit). Each decode step writes one new
token to the cache with quantization error. Over 530+ steps, accumulated quantization error
in the K cache corrupts attention scores enough to collapse the logit distribution.

The sliding window (4096) makes it worse: corrupted K tokens at the edge of the window receive
disproportionate attention weight, amplifying the signal corruption.

### Experimental Evidence

All three experiments run to completion on TG:

| KV cache dtype | Decode sliding_window | Steps before failure | Failure type |
|---|---|---|---|
| `bfloat8_b` | 4096 (SWA active) | ~530 | **Garbage** (hardware corruption) |
| `bfloat16` | None (full attn) | ~3680 | Dashes — model mode collapse (expected) |
| `bfloat16` | 4096 (SWA active) | ~4630 | Dashes — model mode collapse (expected) |

The `bfloat16` KV cache runs both lasted much longer (3680 and 4630 steps) and the
"failure" was model-level repetition (OLMo's sliding window forgets early context),
NOT hardware corruption. This confirms `bfloat8_b` is the root cause.

### Current Workaround

`sliding_window_size=None` is applied to decode SDPA, effectively disabling the sliding window
in decode. This gives ~3680 coherent steps (model mode collapse, not hardware) at the cost of
slightly degraded attention quality (full attention over all tokens instead of last 4096).

**`models/demos/llama3_70b_galaxy/tt/llama_attention.py`** (~line 108):
```python
# _use_sliding_window_decode = False means decode always uses sliding_window_size=None
self._use_sliding_window_decode = False
```

### Proper Fix (Blocked on Kernel)

`paged_fill_cache` does not support `bfloat16` input for long ISLs — it hangs at 32K ISL with
`bfloat16` K/V tensors. This kernel would need to be updated to accept `bfloat16` input and
write to a `bfloat16` KV cache. Once that works:

1. Initialize KV cache as `bfloat16` instead of `bfloat8_b`
2. Remove the `_use_sliding_window_decode = False` workaround
3. Re-enable `sliding_window_size=4096` in decode

### How to Reproduce

```bash
cd /path/to/tt-metal
export ARCH_NAME=wormhole_b0
export TT_METAL_HOME=$(pwd)
export PYTHONPATH=$(pwd)
export HF_MODEL=~/models/OLMo-3.1-32B-Think
source python_env/bin/activate

# Run AIME math problem — generates long chain-of-thought, shows failure mode
pytest models/demos/llama3_70b_galaxy/demo/text_olmo_demo.py \
    -k "aime-b1" -v -s 2>&1 | tee olmo_aime_accuracy.txt
```

With `bfloat8_b` KV cache and `sliding_window_size=4096` re-enabled (revert the
`_use_sliding_window_decode = False` fix), garbage output appears around decode step 530.

### Test Prompt (AIME 2024 Problem I/5)

File: `models/demos/llama3_70b_galaxy/demo/sample_prompts/input_data_aime_test.json`

```json
[
  {
    "prompt": "Every morning Aya goes for a 9-kilometer-long walk and stops at a coffee shop afterwards. When she walks at a constant speed of s kilometers per hour, the walk takes her 4 hours, including t minutes spent in the coffee shop. When she walks s+2 kilometers per hour, the walk takes her 2 hours and 24 minutes, including t minutes spent in the coffee shop. Suppose Aya walks at s+1/2 kilometers per hour. Find the number of minutes the walk takes her, including the t minutes spent in the coffee shop.\nPlease reason step by step, and put your final answer within \\boxed{}."
  }
]
```

Correct answer: **\boxed{204}**

The model should produce a `<think>---...` preamble (OLMo-Think reasoning delimiter), then
chain-of-thought reasoning, then `</think>` and the answer. With the hardware bug active,
the reasoning degrades to repeated tokens or garbage before reaching the answer.

### Sampling Parameters (HF-recommended)

```python
{"temperature": 0.6, "top_p": 0.95, "seed": 42}
# Note: HF does NOT recommend top_k for the -Think variant
```

---

## Environment

| Parameter | Value |
|---|---|
| Hardware | TG Galaxy (8×4 mesh, 32 Wormhole B0 chips) |
| Model | `allenai/OLMo-3.1-32B-Think` |
| Model path | `~/models/OLMo-3.1-32B-Think` |
| Demo script | `models/demos/llama3_70b_galaxy/demo/text_olmo_demo.py` |
| Branch | `ssinghal/olmo-3-32b` |
| `ARCH_NAME` | `wormhole_b0` |

## Key Files

| File | Role |
|---|---|
| `models/demos/llama3_70b_galaxy/tt/llama_ccl.py` | CCL ops — sync/async guards for OLMo prefill |
| `models/demos/llama3_70b_galaxy/tt/llama_attention.py` | KV cache dtype, sliding_window decode flag |
| `models/demos/llama3_70b_galaxy/tt/generator.py` | Long-ISL warmup during server load |
| `models/demos/llama3_70b_galaxy/tt/olmo_model_config.py` | `sliding_window=4096`, `ccl_dtype=bfloat8_b` |
| `models/demos/llama3_70b_galaxy/demo/text_olmo_demo.py` | Demo test cases including `aime-b1`, `long-64k-b1` |
| `models/demos/llama3_70b_galaxy/demo/sample_prompts/input_data_aime_test.json` | AIME eval prompt |
