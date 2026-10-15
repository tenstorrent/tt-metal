# Muse-Glimmer-30B

Text-only decoder inference for [meta-models/Muse-Glimmer-30B](https://huggingface.co/meta-models/Muse-Glimmer-30B) on Tenstorrent Blackhole, using TT-NN with tensor parallelism (and data parallelism on Galaxy). The vision tower is not implemented.

The implementation builds on the attention helpers of `models/demos/gemma4` (paged KV cache, per-user decode RoPE, SDPA program configs, CCL patterns).

## Architecture

| Item | Value |
|---|---|
| Layers | 52 |
| Hidden / intermediate | 6656 / 19968 |
| Vocab | 202048 (padded to 202752 for the LM head) |
| Attention pattern | 3 sliding (window 2048, RoPE theta 5e5) : 1 full (NoPE); layer i is full/NoPE iff (51-i)%4==0 |
| Heads | 32 Q / 2 KV, head_dim 128 |
| Attention extras | Gated attention (`o * sigmoid(gate_proj(x))`); weightless per-head QK RMSNorm, with the 3.87 scale folded into the SDPA scale |
| Norms | Sandwich (pre/post attention and FFN), centered `(1 + w)` weights (precomputed at load time); final norm is a plain weighted RMSNorm |
| Embedding | Normalized: weightless RMSNorm of the token embedding (no sqrt(H) scaling) |
| LM head | Untied; `logits = 20 * tanh(lm_head(h) * 0.196 / 20)` (output multiplier 0.196, softcap 20) |

## Supported systems

| System | Devices | Mesh | Parallelism | Batch | Tokens/s/user | Tokens/s | TTFT |
|---|---|---|---|---|---|---|---|
| QB2 (P300x2 / 4x Blackhole) | 4 | 1x4 | TP4, DP1 | 1 | 23.8 | 23.8 | 0.52 s |
| QB2 (P300x2 / 4x Blackhole) | 4 | 1x4 | TP4, DP1 | 32 | 23.4 | 728 | - |
| Blackhole Galaxy | 8 | 1x8 submesh | TP8, DP1 | 1 | 30.3 | 30.3 | 1.00 s |
| Blackhole Galaxy | 8 | 1x8 submesh | TP8, DP1 | 32 | 29.5 | 918 | - |

Measured on a Blackhole Galaxy submesh with DP1, 128 new tokens, `MUSE_ATTN_FIDELITY=hifi2`, `MUSE_MLP_FIDELITY=lofi`. The QB2 rows are measured on a 1x4 submesh of a Blackhole Galaxy, which runs the same TP4 code path. Replica submeshes are created with `models.tt_transformers.tt.generator.create_submeshes`.

Accuracy (teacher forcing against the reference, TP4, 52 layers): top-1 0.977, top-5 1.0.

## Parallelism

One model replica runs on a `1 x TP` mesh (TP in {1, 2, 4, 8}). The residual stream is replicated on every rank (bf16).

- **Attention**: fused QKVG projection, column-parallel; per-rank columns are `[Q_i | K_i | V_i | G_i]`.
  - TP4: 8 Q heads + 1 KV head per rank (the KV head is replicated: ranks 0,1 own KV head 0, ranks 2,3 own KV head 1).
  - TP8: 4 Q heads + 1 KV head per rank (replicated the same way).
  - TP<=2: KV heads are split contiguously across ranks.
- **O projection**: row-parallel, followed by all-reduce.
- **MLP**: gate/up column-parallel (4992 per rank at TP4, 2496 at TP8), down row-parallel with all-reduce.
- **Embedding**: sharded on the hidden dim, followed by all-gather and the weightless RMSNorm.
- **LM head**: vocab-sharded, weight padded with zero columns to 202752; padded logits are ignored.
- **CCL**: all-reduce / all-gather on `cluster_axis=1`. Topology is Ring if TP>=8 else Linear; 2 links on Blackhole.
- **KV cache**: paged, block size 64, bf16, local KV heads per rank.
- **Weights**: attention, MLP and LM head in bfloat8_b; embedding and norms in bfloat16.

## Prerequisites

1. Download the checkpoint:
   ```bash
   huggingface-cli download meta-models/Muse-Glimmer-30B --local-dir <dir>
   export HF_MODEL=<dir>      # default: /mnt/tt-data/Muse-Glimmer-30B
   ```
2. Weight cache: converted TT weights are cached in `<ckpt>/tt_cache/tp{TP}` (override the base dir with `MUSE_TT_CACHE`; if not writable, falls back to `$TT_METAL_HOME/generated/muse_glimmer_tt_cache/tp{TP}`). The first load is slow; later loads and other DP replicas reuse the cache.
3. `transformers` in the Python env does **not** need `muse_glimmer` support. The torch reference (`reference/model.py`) is pure torch, and the tokenizer is loaded from `tokenizer.json` + `chat_template.jinja`.

## Running the demo

```bash
# QB2 (1x4, TP4, DP1)
MESH_DEVICE=P300x2 pytest models/demos/muse_glimmer_30b/demo/text_demo.py -sv

# Blackhole Galaxy, DP8 x TP4
MESH_DEVICE=BHGLX MUSE_DP=8 pytest models/demos/muse_glimmer_30b/demo/text_demo.py -sv

# Blackhole Galaxy, DP4 x TP8
MESH_DEVICE=BHGLX MUSE_DP=4 MUSE_TP=8 pytest models/demos/muse_glimmer_30b/demo/text_demo.py -sv
```

The demo runs chat-templated prompts from `demo/sample_prompts.json` (cycled to fill the global batch), with paged KV and traced decode, and logs TTFT and tok/s/user.

| Env var | Meaning | Default |
|---|---|---|
| `MESH_DEVICE` | Optional. `P300x2`/`P150x4`/`QB2` (1x4), `P150x8` (1x8), `BHGLX`/`GLX` (8x4), or `RxC`. When unset, the root fixture opens the full system mesh (also correct on a Galaxy) | unset: full system mesh |
| `MUSE_TP` | Tensor-parallel width per replica | 4 (8 if `MUSE_DP=4` on 32 devices) |
| `MUSE_DP` | Data-parallel replicas (4 or 8 on 32 devices; must be 1 otherwise) | 8 (TP4) / 4 (TP8) on 32 devices, else 1 |
| `MUSE_ATTN_FIDELITY` | Decode (M<=32) matmul fidelity for attention QKVG/O: `hifi2` (HiFi2 + fp32 acc) or `lofi` (no config, LoFi) | hifi2 |
| `MUSE_MLP_FIDELITY` | Same for MLP gate/up/down (separate DRAM-sharded gate and up weights) | lofi |
| `MUSE_MM_FIDELITY` | Legacy: if set (`lofi`/`default`/`hifi2`), overrides both of the above (logs a warning) | unset |
| `MUSE_BATCH` | Batch per replica (<= 32) | 32 |
| `MUSE_MAX_NEW_TOKENS` | Tokens to generate | 128 |
| `MUSE_SAMPLING` | `greedy` or `topp` (temperature 1.0, top-p 0.95, top-k 64) | greedy |
| `MUSE_NUM_LAYERS` | Truncate the model to N layers (debug) | all 52 |
| `MUSE_FABRIC` | `ring` selects FABRIC_1D_RING instead of FABRIC_1D | FABRIC_1D |
| `MUSE_TRACE_REGION_SIZE` | Trace region size in bytes (`device_params()` default 256000000 when unset) | 200000000 (demo) |
| `MUSE_DEVICE_SAMPLING` | `0` disables on-device sampling (full-logits host readback + host sampling); default on for TP>1 (greedy = force-argmax, top-k capped at 32); only a `[32]` token tensor is read back from device 0 | 1 |
| `MUSE_TT_CACHE` | Weight cache base directory | `<ckpt>/tt_cache` |
| `HF_MODEL` | Checkpoint directory | `/mnt/tt-data/Muse-Glimmer-30B` |
| `MUSE_CCL_TOPOLOGY`, `MUSE_CCL_NUM_LINKS` | Read in `tt/ccl.py` for topology/link-count selection; the CCL calls themselves are the default synchronous `ttnn.all_reduce` / `ttnn.all_gather`, so treat these as informational | Ring if TP>=8 else Linear; 2 links |

The demo uses `max_seq_len=4096`.

## Tests

Run from the repo root with `PYTHONPATH=$PWD`.

**Host-only** (no device):

| File | Checks |
|---|---|
| `tests/test_model_config.py` | Per-rank head/KV layout, derived sizes for TP 1/2/4/8, layer-type and NoPE tables, weight loading, weight-cache dir logic |
| `tests/test_weight_packing.py` | Simulates TP sharding on host: fused QKVG packing, MLP and LM-head padding, `1+w` norm offset, real checkpoint shapes |

```bash
pytest models/demos/muse_glimmer_30b/tests/test_model_config.py models/demos/muse_glimmer_30b/tests/test_weight_packing.py -v
```

**Device** (compare against `reference/model.py` with real weights; parametrized on `tp` (ids `tp4` / `tp8`, select with `-k`); the system mesh is opened once (full system mesh, or `MESH_DEVICE`) and the test carves a `1 x tp` submesh from it):

| File | Checks | Threshold |
|---|---|---|
| `tests/test_attention.py` | Attention prefill / decode for layer 0 (sliding) and layer 3 (full, NoPE) | PCC 0.99 |
| `tests/test_decoder.py` | MLP, decoder-layer prefill and decode (layers 0 and 3) | PCC 0.99 |
| `tests/test_model.py` | Multi-layer prefill+decode vs reference (4 and 8 layers); full-depth smoke generation | logits PCC 0.98 |
| `tests/test_accuracy.py` | Teacher-forcing top-1 / top-5 vs golden tokens, eager and trace | see file |

```bash
pytest models/demos/muse_glimmer_30b/tests/test_attention.py -k tp4 -sv     # QB2 (1x4 mesh) or a 1x4 submesh of the Galaxy
pytest models/demos/muse_glimmer_30b/tests/test_decoder.py -k tp8 -sv     # needs >= 8 devices (skipped on QB2)
```

All device tests use the ids `tp4` / `tp8`. On a Galaxy, directly opening a 1x4/1x8 mesh with fabric enabled fails (fabric router handshake timeout), so the full 8x4 mesh is opened (with or without `MESH_DEVICE=BHGLX`; unset works too) and TP submeshes are created inside it (`tt/mesh_utils.py`). `tp8` tests are skipped when fewer than 8 devices exist. Device tests read the checkpoint from `HF_MODEL` (`test_decoder.py` and `test_attention.py` use the default path `/mnt/tt-data/Muse-Glimmer-30B`).

**Reference validation**: `reference/model.py` is bit-exact against HF transformers 5.19 (bf16 and fp32) on the real weights. It is a standalone torch implementation (`MuseGlimmerTextConfig`, `TextModel`, `CausalLM.from_checkpoint`, ...).

## Known limitations

- Prefill is single-pass per user, up to `max_prefill_len` (default 8192, padded to a multiple of 128). Chunked prefill is not supported; SDPA for S > 32k is unsupported (silently wrong if non-chunked), so the length is capped.
- Sliding layers allocate a full-length KV cache (masking via `sliding_window_size`); there is no ring buffer.
- Vision tower is not implemented.
- QB2 ring topology is untested; QB2 results come from a Galaxy 1x4 submesh.
