# Why decode tok/s differed between tt-metal direct and the vLLM server —

**Device:** `MESH_DEVICE=P300x2` (Blackhole QB2, TP=4)  ·  **Harness:** `models/demos/gemma4/demo/text_demo_v2.py` (B=1, device sampling, `GEMMA4_HOST_SAMPLE=0`, greedy `temperature=0`)  ·

**Models measured:** `google/gemma-4-31B-it` and `google/gemma-4-12B-it`

Two separate things are explained here:

1. **Part 1** — the original decode tok/s gap between tt-metal direct and the vLLM server was caused by *host vs device sampling*, not by different kernels.
2. **Part 2** — device sampling was slow because of one operation (`ttnn.topk`), and switching greedy decode to `argmax` removes it, giving **+17–20% decode tok/s on 31B** and **+29–34% on 12B**.

---

## Part 1 — The metal-vs-server gap was a sampling-mode difference

### What "sampling" means here

After the model computes scores (logits) for all 262,144 possible next tokens, something has to *pick* one. That picking step can happen in two places:

| | Where the picking happens | What has to move |
|---|---|---|
| **Host sampling** | on the CPU | read logits back from the chip to the CPU |
| **Device sampling** | on the chip itself | only the chosen token id comes back |

### The two harnesses had different defaults

This is the whole origin of the confusion:

- The **tt-metal demo** historically defaulted to **host** sampling (`GEMMA4_HOST_SAMPLE=1`); on `ign/gemma4_tt_inf` the default is now **device** sampling (`GEMMA4_HOST_SAMPLE=0` — see env table in Part 2)
- The **vLLM server** defaulted to **device** sampling (`sample_on_device_mode=decode_only`)

So the two were never doing the same work. Metal looked faster, and it was easy to assume the server had extra overhead (HTTP, scheduling, Python).

### Proof that sampling mode was the cause

Switching *the server* to host sampling closed most of the gap, with no other change:

| ISL | Server, device sampling | Server, host sampling | tt-metal direct, host sampling |
|----:|------------------------:|----------------------:|-------------------------------:|
| 4096 | 17.8 tok/s | **20.5** | ~21.3 |
| 32768 | 17.3 tok/s | **19.8** | ~20.4 |
| 131072 | 15.8 tok/s | **17.8** | ~18.9 |

Host sampling gained the server **+2.0 to +2.7 tok/s**, leaving only ~3–6% versus metal. That residual is genuine server overhead (scheduling, readback), but it is small.

The reverse check agrees: when **both** were set to device sampling, metal and server landed within **~1–2%** of each other:

| ISL | tt-metal direct (device) | Server (device) | difference |
|----:|-------------------------:|----------------:|-----------:|
| 4096 | 18.02 | 17.79 | −1.3% |
| 32768 | 17.48 | 17.28 | −1.1% |
| 131072 | 16.15 | 15.79 | −2.2% |

**Conclusion:** the tt-metal kernels and the server were performing almost identically. The apparent gap was one side sampling on the CPU and the other on the chip.

---

## Part 2 — Why device sampling was slow, and what argmax fixes

Device sampling should be the cheaper option — it only sends back a single token id instead of 65,536 scores. Measured on the actual hardware, it was not. Timing the sampling chain in isolation (P300x2, TP=4, logits `[1,1,32,65536]` per chip):

| Step | Time |
|---|---:|
| **`ttnn.topk`** | **10.89 ms  ← 94%** |
| 2 × `all_gather` (values + indices) | 0.23 ms |
| everything else (typecasts, offsets, untilize, `ttnn.sampling`) | ~0.42 ms |
| **whole device chain** | **11.06 ms** |
| *host path for comparison (read back + `torch.argmax`)* | *~1.0 ms* |

So device sampling cost ~11 ms per token while the host path cost ~1 ms. That ~10 ms difference per token is exactly the gap seen in Part 1.

**`ttnn.topk` alone is 94% of the cost**, and it is stubborn:

- Same ~10.9 ms whether `k=1` or `k=32`
- Same ~10.9 ms whether the batch is 1 row or 32 rows
- 4 MiB scanned in 10.9 ms ≈ 0.4 GB/s — roughly three orders of magnitude below Blackhole's memory bandwidth

Nothing the model controls makes it faster. It looks like an unoptimized kernel path.

### Greedy argmax bypasses top-k entirely

For `temperature=0`, the sampler does not need score values or random draws — only the global argmax. That path all-gathers the full logits and calls `ttnn.argmax`, skipping `ttnn.topk` altogether. Measured in isolation:

| Path | Time |
|---|---:|
| top-k chain | **11.06 ms** |
| gather + untilize + argmax (batch padded to 32) | **2.05 ms** |
| gather + untilize + argmax (batch 1, unpadded) | **0.56 ms** |

The win is not fewer all-gathers (the argmax gather moves *more* data). **The entire saving is skipping `ttnn.topk`.**

### `text_demo_v2` sampling env vars (`ign/gemma4_tt_inf`)

**Defaults (no env set):**

- Device sampling ON (TP>1, vocab shard ≤ 64K)
- Greedy decode (`temperature=0` in demo configs → `top_k=1` on device)
- Force-argmax fast path OFF (`GEMMA4_FORCE_ARGMAX_SAMPLING` unset / `0`)

| Env var | Default | Set to | Effect |
|---------|---------|--------|--------|
| `GEMMA4_HOST_SAMPLE` | `0` (device) | `1` | Sample on CPU from host logits instead of on-device. |
| `GEMMA4_FORCE_ARGMAX_SAMPLING` | `0` (top-k kernel, even for greedy `k=1`) | `1` | Greedy-only fast path (all-gather + argmax). Needs `temperature=0`. Also set on **vLLM server** process to enable there. |

Quick examples:

```bash
# Default — device greedy (top-k path with k=1)
pytest models/demos/gemma4/demo/text_demo_v2.py -k long-context-4k -v

# Host sampling (~23 tok/s style recal; incorrect on TP for tokens ≥ 65536)
GEMMA4_HOST_SAMPLE=1 pytest models/demos/gemma4/demo/text_demo_v2.py -k long-context-4k -v

# Device greedy + force-argmax fast path
GEMMA4_FORCE_ARGMAX_SAMPLING=1 pytest models/demos/gemma4/demo/text_demo_v2.py -k long-context-32k -v
```

Full argmax A/B reproduce (both models, 4k/32k/128k):

```bash
export HF_HUB_OFFLINE=1 HF_HOME=~/.cache/huggingface MESH_DEVICE=P300x2 GEMMA4_HOST_SAMPLE=0
# 31B:
export HF_MODEL=google/gemma-4-31B-it
export TT_CACHE_PATH=~/gemma4-vllm/vllm/tt-inference-server/persistent_volume/volume_id_tt_transformers-gemma-4-31B-it-v0.18.0/tt_metal_cache/cache_gemma-4-31B-it/P300x2
# 12B:
export HF_MODEL=google/gemma-4-12B-it
export TT_CACHE_PATH=~/gemma4-vllm/vllm/tt-inference-server/persistent_volume/volume_id_tt_transformers-gemma-4-12B-it-v0.18.0/tt_metal_cache/cache_gemma-4-12B-it/P300x2
# argmax ON/OFF:
export GEMMA4_FORCE_ARGMAX_SAMPLING=1   # or 0
pytest models/demos/gemma4/demo/text_demo_v2.py -k long-context-{4k,32k,128k} -s --timeout 7200
```

### Metal direct argmax A/B — `text_demo_v2`

#### gemma-4-31B-it

| ISL | argmax OFF | argmax ON | gain |
|----:|-----------:|----------:|-----:|
| 4096 | 18.04 tok/s | **21.62** | **+19.8%** |
| 32768 | 17.48 | **20.84** | **+19.2%** |
| 131072 | 16.14 | **18.96** | **+17.5%** |



#### gemma-4-12B-it

| ISL | argmax OFF | argmax ON | gain |
|----:|-----------:|----------:|-----:|
| 4096 | 27.61 tok/s | **37.09** | **+34.3%** |
| 32768 | 26.61 | **35.28** | **+32.6%** |
| 131072 | 24.25 | **31.22** | **+28.8%** |



12B starts from a higher baseline (smaller model → faster decode overall) but the **absolute** sampling saving is similar (~9 ms/token on 31B; ~9–9.5 ms on 12B). The **relative** gain is larger on 12B because sampling was a bigger fraction of the shorter decode step.

Generated tokens are **byte-identical** to the top-k path at 4k / 32k / 64k / 128k on metal.

### Where device sampling + argmax lands (31B @ 4k)

| Configuration | 4k decode | Correct on TP? |
|---|---:|:--:|
| device sampling, top-k (original) | 18.02 tok/s | yes |
| host sampling | 20.5 tok/s | **no** — only sees 1/4 of the vocab |
| **device sampling + argmax** | **21.62 tok/s** | **yes** |

With argmax ON, **12B @ 4k** reaches **37.09 tok/s** vs **27.61 tok/s** top-k (same harness).

---
