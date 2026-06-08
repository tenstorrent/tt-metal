# Qwen3-Coder-Next — TT-Symbiote Integration

End-to-end generation test for [Qwen/Qwen3-Coder-Next](https://huggingface.co/Qwen/Qwen3-Coder-Next) accelerated with TT-Symbiote on Tenstorrent Wormhole hardware.

**Test file:** [`test_qwen3_coder_next.py`](test_qwen3_coder_next.py)

## Target Device

| Setting | Value |
|---------|-------|
| Hardware | **Wormhole Loudbox** (T3K) |
| Mesh shape | `(1, 8)` — 8 devices in a 1×8 ring |
| `MESH_DEVICE` | `T3K` (required; test skips on other devices) |
| Fabric | `FABRIC_1D_RING` |
| Trace region | 50 MB |

## Model Overview

Qwen3-Coder-Next is an 80B-parameter MoE coding model with ~3B activated parameters per token. It uses a hybrid attention layout that alternates between linear (Gated DeltaNet) and full (gated softmax) attention blocks.

**HuggingFace model:** [Qwen/Qwen3-Coder-Next](https://huggingface.co/Qwen/Qwen3-Coder-Next)

### Layer Structure

The 48 decoder layers follow a repeating hybrid pattern:

```
12 × ( 3 × (Gated DeltaNet → MoE)  →  1 × (Gated Attention → MoE) )
```

Each `Qwen3NextDecoderLayer` contains:

1. **Input RMSNorm** (`Qwen3NextRMSNorm`) — stays on PyTorch
2. **Attention sub-layer** — one of:
   - `Qwen3NextGatedDeltaNet` → `TTNNGatedDeltaNet` (linear attention, 3 of every 4 layers)
   - `Qwen3NextAttention` → `TTNNQwen3NextGatedAttention` (full gated attention, 1 of every 4 layers)
3. **Post-attention RMSNorm** (`Qwen3NextRMSNorm`) — stays on PyTorch
4. **MoE FFN** (`Qwen3NextSparseMoeBlock`) → `TTNNQwen3MoE`

Top-level model:

```
Qwen3NextForCausalLM
└── Qwen3NextModel
    ├── embed_tokens          (PyTorch — not replaced)
    ├── Qwen3NextDecoderLayer × 48
    │   ├── input_layernorm   (PyTorch)
    │   ├── self_attn         (TTNN — GatedDeltaNet or GatedAttention)
    │   ├── post_attn_layernorm (PyTorch)
    │   └── mlp               (TTNN — MoE)
    ├── norm                  (PyTorch)
    └── lm_head               (PyTorch — excluded for correct full-vocab logits)
```

## TT-Symbiote Module Mapping

| PyTorch Module | TT-Symbiote Replacement | Source |
|----------------|------------------------|--------|
| `Qwen3NextSparseMoeBlock` | `TTNNQwen3MoE` | `modules/moe.py` |
| `Qwen3NextAttention` | `TTNNQwen3NextGatedAttention` | `modules/attention.py` |
| `Qwen3NextGatedDeltaNet` | `TTNNGatedDeltaNet` | `modules/gated_deltanet.py` |
| `nn.Linear` | `TTNNLinearIColShardedWRowSharded` | `modules/linear.py` |

### MoE Sub-modules (inside `TTNNQwen3MoE`)

| Component | TT-Symbiote Class |
|-----------|-------------------|
| Router / gate | `TTNNGlm4MoeTopkRouter` |
| Token routing (decode) | `TTNNQwen3CoderNextMoERouterDecode` |
| Routed experts | `TTNNQwen3CoderNextExperts` |
| Shared expert MLP | `TTNNQwen3SharedExpertMLP` |
| Shared expert gate | `TTNNLinear` |

### Modules Left on PyTorch

- `embed_tokens` — token embedding lookup
- `Qwen3NextRMSNorm` / `Qwen3NextRMSNormGated` — layer normalization
- `Qwen3NextRotaryEmbedding` — rotary position embeddings
- `lm_head` — output projection (excluded to preserve full-vocabulary logits accuracy)

## Prerequisites

- Model weights cached locally under `~/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-Next/`
- `triton` installed (required by the HuggingFace model loader)
- `transformers` with Qwen3-Next support
- Active T3K Wormhole device (8-chip mesh)

## Running the Test

The test prompt is a single-turn chat message: *"Write a quick sort algorithm."*

Decode length is controlled by the `main_max_new_tokens` pytest parameter. The test **defaults to 128** new tokens for a shorter smoke run, but the integration supports decode caps **up to 262,144** (the model's native context length). Set `main_max_new_tokens <= 0` to use the full remaining context budget. Generation stops early on EOS regardless of the cap.

### Normal Mode

Standard TTNN execution without trace capture:

```bash
MESH_DEVICE=T3K TT_SYMBIOTE_RUN_MODE=NORMAL pytest --timeout=0 \
  models/experimental/tt_symbiote/tests/test_qwen3_coder_next.py
```

### Traced Mode

Uses `TracedRun` to capture and replay device traces for `@trace_enabled` modules (e.g. `TTNNLinearIColShardedWRowSharded`):

```bash
MESH_DEVICE=T3K TT_SYMBIOTE_RUN_MODE=TRACED pytest --timeout=0 \
  models/experimental/tt_symbiote/tests/test_qwen3_coder_next.py
```

### Changing Decode Length

The default parametrize value in `test_qwen3_coder_next.py` is **128**:

```python
@pytest.mark.parametrize(
    "main_max_new_tokens",
    [128],  # default — change to any value up to 262144
    ids=["decode_max_128"],
)
```

Supported values range from `128` up to **`262144`** (256K). Common caps used in perf benchmarking:

| `main_max_new_tokens` | Use case |
|-----------------------|----------|
| `128` | Default smoke test (fastest) |
| `4096` | Medium-length generation |
| `32768` | Long-form generation |
| `262144` | Full native context decode cap |

To sweep multiple caps, add values with matching `ids`:

```python
@pytest.mark.parametrize(
    "main_max_new_tokens",
    [128, 4096, 32768, 262144],
    ids=["decode_max_128", "decode_max_4096", "decode_max_32768", "decode_max_262144"],
)
```

### Timing Output

After each run the test writes:

- `qwen3_coder_next_timing_stats.csv` — per-op timing breakdown
- `qwen3_coder_next_timing_stats_pivot.csv` — pivoted summary

## Performance Results

Measured on Wormhole Loudbox (T3K) with prompt length 14 tokens. Actual generated token counts depend on EOS; the `main_max_new_tokens` value is the decode cap.

### `main_max_new_tokens = 128` (NORMAL only)

| Metric | Value |
|--------|-------|
| Generated tokens | 128 |
| Prefill compile (warmup) | 96.22 s |
| TTFT (measured) | 2.63 ms |
| Prefill throughput | 5,330 tok/s |
| Decode speed | 6,498 ms/tok (0.15 tok/s) |
| Decode wall time | 831.77 s |
| Full `generate()` wall time | 831.77 s |
| Output quality | Correct quicksort answer |
| Top module (total) | `TTNNQwen3MoE` — 572.1 s |

### `main_max_new_tokens = 4096` (NORMAL only)

| Metric | Value |
|--------|-------|
| Generated tokens | 497 (stopped at EOS) |
| Prefill compile (warmup) | 15.95 s |
| TTFT (measured) | 1.76 ms |
| Prefill throughput | 7,939 tok/s |
| Decode speed | 5,904 ms/tok (0.17 tok/s) |
| Decode wall time | 2,934.49 s (~49 min) |
| Full `generate()` wall time | 2,934.49 s |
| Output quality | Correct full quicksort implementation |
| Top module (total) | `TTNNQwen3MoE` — 2,016.8 s |

### `main_max_new_tokens = 32768` (NORMAL only)

| Metric | Value |
|--------|-------|
| Generated tokens | 503 (stopped at EOS) |
| Prefill compile (warmup) | 15.87 s |
| TTFT (measured) | 1.91 ms |
| Prefill throughput | 7,333 tok/s |
| Decode speed | 4,860 ms/tok (0.21 tok/s) |
| Decode wall time | 2,444.70 s (~41 min) |
| Full `generate()` wall time | 2,444.70 s |
| Output quality | Correct full quicksort implementation |
| Top module (total) | `TTNNQwen3MoE` — 1,926.4 s |

### `main_max_new_tokens = 262144` / 256K (NORMAL only)

| Metric | Value |
|--------|-------|
| Generated tokens | 438 (stopped at EOS) |
| Prefill compile (warmup) | 16.50 s |
| TTFT (measured) | 1.91 ms |
| Prefill throughput | 7,328 tok/s |
| Decode speed | 4,832 ms/tok (0.21 tok/s) |
| Decode wall time | 2,116.52 s (~35 min) |
| Full `generate()` wall time | 2,116.52 s |
| Output quality | Correct full quicksort implementation |
| Top module (total) | `TTNNQwen3MoE` — 1,675.6 s |

### Performance Observations

- **MoE dominates runtime.** `TTNNQwen3MoE` / `TTNNQwen3CoderNextExperts` account for ~65–75% of total module time across all runs.
- **Prefill is fast.** TTFT is consistently under 3 ms once warmup compile completes.
- **Decode is the bottleneck.** Per-token decode latency is ~4.8–6.5 s/tok, yielding ~0.15–0.21 tok/s throughput.
- **First-run compile cost.** The initial 128-token NORMAL run incurred a 96 s prefill compile; subsequent runs with larger caps compiled in ~16 s.

## Known Limitations / WIP

| Issue | Status | Details |
|-------|--------|---------|
| TRACED mode correctness at 128 tokens | **WIP** | TRACED mode produces degenerate repetitive output (e.g. `"the the the in in in..."`) instead of a valid quicksort answer. NORMAL mode output is correct at the same token cap. |
| TRACED mode at larger decode caps (≥ 4096) | **WIP** | TRACED runs at higher `main_max_new_tokens` values take excessively long and need optimization before they are usable. |
| `lm_head` on PyTorch | By design | Output projection stays on host PyTorch for full-vocab logit accuracy. |

## Related Tests

| Test | Description |
|------|-------------|
| [`test_qwen3_moe.py`](test_qwen3_moe.py) | Unit PCC test for `Qwen3NextSparseMoeBlock` → `TTNNQwen3MoE` |
| [`test_gated_deltanet.py`](test_gated_deltanet.py) | Unit test for `TTNNGatedDeltaNet` |
| [`test_attention.py`](test_attention.py) | Gated attention PCC test including `TTNNQwen3NextGatedAttention` |
