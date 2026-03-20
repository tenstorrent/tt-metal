# MiniMax-M2.5 TTNN Bringup Log

## N300 General Agentic Mode (Session 2026-03-18)

| Status | Component |
|--------|-----------|
| ✅ Done | `models/demos/minimax_m2/agentic/` — full directory structure |
| ✅ Done | `tool_wrappers/llm_tool.py` — Llama 3.1 8B **fully TTNN** (internal KV cache, fabric-enabled full mesh) |
| ✅ Done | `tool_wrappers/whisper_tool.py` — WhisperGenerator wrapping (TTNN) |
| ✅ Done | `tool_wrappers/speecht5_tool.py` — SpeechT5 TTS (TTNN hybrid) |
| ✅ Done | `tool_wrappers/owlvit_tool.py` — OWL-ViT zero-shot detection (TTNN) |
| ✅ Done | `tool_wrappers/bert_tool.py` — BERT Large QA (TTNN) |
| ✅ Removed | `tool_wrappers/vit_tool.py` — deleted (ViT incompatible with shared device setup) |
| ✅ Removed | `tool_wrappers/sbert_tool.py` — deleted (Turkish model, CI-coupled) |
| ✅ Done | `tools.py` — 5 TOOL_SCHEMAS + dispatch_tool() |
| ✅ Done | `loader.py` — load_all_models(), open_n300_device() |
| ✅ Done | `orchestrator.py` — run_agentic_loop(), run_one_turn(), process_single_query() |
| ✅ Done | `demo.py` — CLI entry point (interactive + single-query modes) |
| ✅ Done | `tests/test_agentic.py` — 34 offline pytest tests, all passing |
| ✅ Done | `tests/conftest.py` — session-scoped device + synthetic data fixtures |
| ✅ Done | `tests/test_device_individual.py` — 43 per-tool warmup + trace-reuse tests (require N300) |
| ✅ Done | `tests/test_device_integration.py` — 34+ end-to-end agentic pipeline tests (require N300) |

### LLM TTNN Implementation (key detail)
- `kv_cache=None` passed to all Generator calls → uses model's internal pre-allocated KV cache
- No paged attention required; no external KV cache tensors to manage
- `warmup_prefill=True` on first `prefill_forward_text` → compiles kernels for all supported seq lengths
- Decode trace captured lazily on first `decode_forward(enable_trace=True)` call
- `prefill_forward_text` returns torch `[1, 1, vocab_size]`; `decode_forward` returns `(logits, log_probs)` pair

### Test Hierarchy (run in this order)
1. **Offline** (no device): `pytest tests/test_agentic.py -m offline` — 34 tests
2. **Individual** (per model): `pytest tests/test_device_individual.py -m device` — warmup + trace reuse per tool
3. **Integration** (full pipeline): `pytest tests/test_device_integration.py -m "device and integration"` — end-to-end

**Test Results:** 34/34 offline pass. Device tests require N300 + model weights.

**Block hash:** `git log --oneline -1` on branch `ssinghal/minimax`

### Session 2026-03-19 update

- **Status:** In progress — staged Whisper-first load + trace release before co-resident models.
- **PCC:** N/A (workflow orchestration and runtime loading/debug session, not a PCC block validation run).
- **Block Hash:** `7e0537a26f`
- **Notes:**
  - `run_all_tools.py`: **PHASE 0a/1a** load Whisper only → warmup (trace capture); **release Whisper decoder trace**; **PHASE 0b** load OWL/BERT/SpeechT5; **PHASE 1b** warmups; **PHASE 2** infer. Whisper-only teardown was insufficient — OWL/BERT still resident caused Whisper stall; Whisper-first staging fixes that.
  - After Whisper warmup, **`WhisperTool.release_decoder_trace()`** (`WhisperGenerator.cleanup()`) — Metal warns allocations are unsafe while a persistent trace is active; this avoids hangs loading BERT after Whisper (re-capture on next `transcribe`).
  - `create_functional_whisper...` attaches **`whisper_generator`** on the pipeline callable for the above.
  - Script supports `--skip-llm` when HF gated model auth is unavailable.
  - SpeechT5: `warmup_on_init` + init checkpoints in `speecht5_tool.py`.
  - LLM: HF gated `meta-llama/Llama-3.2-3B-Instruct` needs token/cache unless `--skip-llm`.

### Session 2026-03-20: Systematic Pairwise Testing → **RESOLVED**

- **Status:** ✅ **RESOLVED** — All 4 models run successfully on chip0 submesh.
- **PCC:** N/A (multi-model workflow testing, not PCC block validation).
- **Block Hash:** `7e0537a26f`

**Test Framework Created:**
- `tests/level0/` — Standalone tests for each model
- `tests/level1/test_pairwise.py` — Parameterized pairwise tests
- `tests/level2/test_triple.py` — Three-model combination tests
- `tests/TESTS_DONE.md` — Passing test log
- `tests/NOT_POSSIBLE.md` — Architectural blockers

**Final Results:**

| Level | Test | Status |
|-------|------|--------|
| L0 | Whisper standalone | PASS |
| L0 | BERT standalone | PASS |
| L0 | OWL-ViT standalone | PASS |
| L0 | SpeechT5 standalone | PASS |
| L1 | Whisper + BERT | PASS (trace release required) |
| L1 | BERT + OWL-ViT | PASS |
| L1 | OWL-ViT + SpeechT5 | PASS |
| L1 | BERT + SpeechT5 | **PASS** (both on chip0) |
| L2 | BERT + OWL + SpeechT5 | **PASS** (all on chip0) |
| L3 | **All 4 models** (--skip-llm) | **PASS** |

**Root Cause Confirmed:** Mixing full-mesh models with chip0-submesh models causes deadlocks. When any model uses the full mesh while others use chip0 submesh, warmup hangs.

**Solution:** Run ALL models on chip0 submesh. This sacrifices multi-chip parallelism (chip1 unused) but enables stable co-residency for all models.

**Architecture:**
```
N300 (1×2 mesh)
├── chip0 (all models)
│   ├── Whisper STT
│   ├── BERT QA
│   ├── OWL-ViT detection
│   ├── SpeechT5 TTS
│   └── LLM (if loaded)
└── chip1 (unused — available for future optimization)
```

**DRAM Budget (chip0 only):** ~9.9 GB / 12 GB (~2 GB headroom)

### Session 2026-03-20: Full (1,2) Mesh Support + Llama 8B — COMPLETE ✅

**All 5 models now work on full (1,2) mesh with fabric enabled!**

| Model | Full Mesh (1,2) | Notes |
|-------|-----------------|-------|
| OWL-ViT | ✅ PASS | Added `mesh_composer` to `ttnn_owl_vit.py` |
| BERT | ✅ PASS | Native support |
| Whisper | ✅ PASS | Native support |
| SpeechT5 | ✅ PASS | Added scalar handling via `get_device_tensors` |
| LLM (Llama 3.1 8B) | ✅ PASS | Upgraded from 3B to 8B, requires HF auth |

**Key Fixes:**

1. **Fabric Config for LLM on Full Mesh:**
```python
# Must call set_fabric_config BEFORE open_mesh_device
ttnn.set_fabric_config(
    ttnn.FabricConfig.FABRIC_1D,
    ttnn.FabricReliabilityMode.STRICT_INIT,
    None,
    ttnn.FabricTensixConfig.DISABLED,
)
mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), **device_params)
```

2. **Multi-Device Scalar Handling:**
```python
# Scalar tensors on multi-device can't use ConcatMeshToTensor (no dim to concat)
# Solution: use get_device_tensors to extract from device 0
if is_scalar:
    device_tensors = ttnn.get_device_tensors(tensor)
    return ttnn.to_torch(device_tensors[0])
```

**Files Modified:**
- `models/demos/wormhole/owl_vit/tt/ttnn_owl_vit.py` — Added `_to_torch` helper
- `models/demos/wormhole/owl_vit/tests/test_end_to_end.py` — Updated `ttnn.to_torch` calls
- `models/experimental/speecht5_tts/demo_ttnn.py` — Added `_to_torch` with scalar handling
- `models/experimental/speecht5_tts/tt/ttnn_speecht5_generator.py` — Added `_to_torch` helper
- `models/demos/minimax_m2/agentic/tool_wrappers/owlvit_tool.py` — Updated for mesh_composer
- `models/demos/minimax_m2/agentic/loader.py` — Added fabric config before mesh open
- `models/demos/minimax_m2/agentic/tool_wrappers/llm_tool.py` — Upgraded to Llama 3.1 8B

**Benchmark Results (full (1,2) mesh, 3 iterations):**

| Model | Mean (ms) | Std (ms) | Min (ms) | Max (ms) |
|-------|-----------|----------|----------|----------|
| Whisper | 186.2 | 12.8 | 175.2 | 200.2 |
| BERT | 150.6 | 8.6 | 142.3 | 159.4 |
| OWL-ViT | 154.0 | 72.4 | 105.4 | 237.2 |
| SpeechT5 | 2731.7 | 316.0 | 2544.6 | 3096.6 |
| **TOTAL** | **~3.2s** | — | — | — |

**Architecture (full mesh with fabric):**
```
N300 (1×2 mesh) — BOTH CHIPS ACTIVE via FABRIC_1D
├── chip0 + chip1 (sharded/replicated as appropriate)
│   ├── Whisper STT
│   ├── BERT QA
│   ├── OWL-ViT detection
│   ├── SpeechT5 TTS
│   └── LLM (Llama 3.1 8B — sharded across both chips)
```

**Notes:**
- SpeechT5 longer time (~2.7s) due to autoregressive TTS generation (~80 steps)
- OWL-ViT variance from JIT compilation on first iterations
- All models run sequentially without hangs
- Whisper decoder trace released after warmup for model coexistence

### Session 2026-03-20: Agentic Workflow Testing & Segfault Fix

**Branch:** `ssinghal/agentic-workflow`

**Commits:**
1. `900b6c1f74` — feat(agentic): N300 multi-model agentic workflow with Llama 8B on full mesh
2. `707a7e803d` — fix(agentic): improve system prompt and prevent tool call loops
3. `3d4bdcb442` — fix(agentic): prevent segfault by releasing traces before device close

**Key Fixes:**

1. **Segfault Prevention:**
   - Root cause: Python GC runs `__del__` after `ttnn.close_mesh_device()`, causing trace release to fail
   - Fix: Added `cleanup_models()` function to explicitly release traces BEFORE device close
   - Added `close()` method to `LLMTool` that deletes the generator to trigger trace release

2. **Tool Call Loop Prevention:**
   - LLM was repeatedly calling the same tool
   - Added `called_tools` set to track used tools
   - Reduced `_MAX_TOOL_TURNS` from 10 to 3
   - On duplicate tool call, force final answer

3. **System Prompt Improvements:**
   - Clear rules: answer simple questions directly, only use tools for attachments
   - Explicit "AFTER receiving tool results, respond with FINAL ANSWER"

**Test Results:**
```
# LLM-only (simple math)
Query: "What is 5 times 7?"
→ Answer: 35 (correct, though LLM still tries to use tools)

# Cleanup verified
Cleaning up models (releasing traces)...
LLMTool closed (traces released).
Model cleanup complete.
Device closed.  # No segfault!
```

**Files Modified:**
- `loader.py` — Added `cleanup_models()` function
- `llm_tool.py` — Added `close()` method, upgraded to Llama 3.1 8B
- `orchestrator.py` — Added duplicate tool prevention, improved system prompt
- `demo.py` — Call `cleanup_models()` before device close

### Shared N300 multi-model run — issues & blockers (detail)

See **`models/demos/minimax_m2/agentic/SHARED_DEVICE_BLOCKERS.md`** for:

- Observed symptoms (Whisper trace stall with co-residents, Metal trace+alloc warning, BERT warmup hang, device locks, post-kill init warnings, HF 401).
- Working theory (staged Whisper-first load, decoder trace release, BERT vs chip0 submesh interaction — **not fully resolved**).
- What still blocks proving **load → warmup → infer** for all tools on one mesh end-to-end.

## Target Platform
Galaxy (TG) — mesh device `(8, 4)` = 32 × Wormhole B0 chips

## Parallelism Strategy

| Component | Strategy | Sharding details |
|---|---|---|
| Attention QKV | TP=4, column-parallel | `[H, (NQ+NK+NK)*D]` → `[H, (NQ+NK+NK)*D/TP]` per col device |
| Attention O-proj | TP=4, row-parallel | `[NQ*D, H]` → `[NQ*D/TP, H]` per col device |
| Attention all-reduce | `mesh_config.allreduce` (reduce-scatter + all-gather) | axis=cols (axis=1) |
| QK-norm | Replicated weight, local norm per TP shard | Approximation: norm is over `NQ*D/TP` instead of `NQ*D` |
| Partial RoPE | Local per device | cos/sin replicated; no CCL needed |
| MoE router gate | Replicated `[H, E]` | Selection on CPU, weights on device |
| MoE expert gate/up | EP=8 + TP=4, `dims=(1, -1)` | `[1, E, H, FF]` → `[1, E/EP, H, FF/TP]` per device |
| MoE expert down | EP=8 + TP=4, `dims=(1, -2)` | `[1, E, FF, H]` → `[1, E/EP, FF/TP, H]` per device |
| MoE EP all-reduce | `ttnn.all_reduce` | axis=rows (axis=0) |
| MoE TP all-reduce | `ttnn.all_reduce` | axis=cols (axis=1) |
| Embeddings / norms / lm_head | Replicated | `ReplicateTensorToMesh` |

## Files Changed

| File | Change summary |
|---|---|
| `tt/model_config.py` | Added `make_mesh_config()` using `gpt_oss.MeshConfig`; mesh (8,4), TP=4, EP=8 |
| `tt/rms_norm.py` | Added `mesh_mapper` parameter; defaults to `ReplicateTensorToMesh` for `MeshDevice` |
| `tt/rope.py` | cos/sin replicated via `ReplicateTensorToMesh`; `apply_partial_rope` is local per device |
| `tt/attention.py` | Full rewrite: TP=4 col-parallel QKV + QK-norm + row-parallel O-proj + `apply_allreduce` |
| `tt/moe.py` | Full rewrite: EP=8+TP=4 on-device expert weights; dense batched matmul; EP+TP all-reduce |
| `tt/model.py` | Opens MeshDevice; `CCLManager`; replicated embeddings/norms/lm_head |
| `tests/test_minimax_m2_tt.py` | `device` fixture: `open_mesh_device(8,4)` + `set_fabric_config(FABRIC_1D_RING)`; `tt_to_torch` reads from `device[0]`; 8 test cases |

## Test Results

### Passing ✅

| Test | PCC | Notes |
|---|---|---|
| `test_rmsnorm` | 0.999983 | Replicated norm weight across mesh |
| `test_partial_rope` | Q: ~0.9999, K: ~0.9999 | Replicated cos/sin; local RoPE per device |
| `test_attention` | 0.994625 | TP=4; local QK-norm approximation causes small PCC loss |

### Failing ❌

| Test | PCC | Root cause |
|---|---|---|
| `test_moe` | 0.940 | Dense batched matmul (E_local=32 experts) PCC below 0.99 threshold — under investigation |

## Known Issues & Root Causes

### 1. Fabric must be initialized before opening MeshDevice

**Error:** `TT_FATAL: Trying to get un-initialized fabric context`

**Cause:** `mesh_config.allreduce` (used for attention TP all-reduce) uses
`reduce_scatter_minimal_async` + `all_gather_async`, which require the Ethernet
fabric. Without calling `ttnn.set_fabric_config(FABRIC_1D_RING)` before
`ttnn.open_mesh_device`, all CCL ops that use the fabric fail.

**Fix applied:** Updated the `device` fixture to call `ttnn.set_fabric_config`
with `FABRIC_1D_RING` before `open_mesh_device`, and reset to `DISABLED` on teardown:

```python
ttnn.set_fabric_config(
    ttnn.FabricConfig.FABRIC_1D_RING,
    ttnn.FabricReliabilityMode.STRICT_INIT,
    None,
    ttnn.FabricTensixConfig.DISABLED,
    ttnn.FabricUDMMode.DISABLED,
    ttnn.FabricManagerMode.DEFAULT,
)
mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(8, 4))
```

### 2. CCL all-reduce ops require 4D tensors

**Error:** `TT_THROW: ShapeBase[] index out of range. 3 not in [-4, 3)`

**Cause:** `mesh_config.allreduce` internally calls `reduce_scatter_minimal_async`
with `dim=3`, requiring a 4D tensor. Our attention O-proj output was 3D `[B, S, H]`.

**Fix applied:** Unsqueeze to 4D before `apply_allreduce`, reshape back to 3D after:

```python
out_4d = ttnn.unsqueeze_to_4D(out)
out_4d = apply_allreduce(out_4d, self.mesh_config, self.ccl_manager, H)
out = ttnn.reshape(out_4d, (B, S, H))
```

**Important:** `ttnn.unsqueeze_to_4D` returns a view sharing the same buffer.
Do NOT call `out.deallocate(True)` after unsqueeze — it frees the shared buffer,
causing `TT_FATAL: Buffer must be allocated on device!` on the next access.

### 3. `ttnn.matmul` does NOT support broadcast in batch dims

**Error:** `TT_FATAL: bmm expects input tensors of shapes BCMK*BCKN=BCMN`

**Cause:** Attempting `[1, 1, T, H] × [1, E_local, H, FF]` — TTNN requires exact
batch dim match. Broadcasting (e.g., 1 → E_local) is not supported by `ttnn.matmul`.

**Fix applied:** Use `ttnn.repeat(x_flat, ttnn.Shape([1, E_local, 1, 1]))` to
explicitly expand `x_flat` to `[1, E_local, T, H]` before the matmul.

### 4. MoE PCC = 0.94 (UNDER INVESTIGATION)

**Symptom:** `test_moe` consistently gives PCC ≈ 0.940, below the 0.99 threshold.

**Confirmed not caused by:**
- Routing tensor shape mismatch (T vs T_pad) — fixed by using `T_pad` in `_route`
- Wrong reduction op (`ttnn.sum` vs `fast_reduce_nc`) — both give same PCC
- Wrong weight convention (w1/w2/w3 transpose) — verified against reference

**Leading hypothesis:** Dense batched matmul over E_local=32 experts in bfloat16
introduces accumulated rounding error that the sparse reference avoids. The reference
computes only the 8 selected experts per token, while our dense implementation computes
all 32 local experts and multiplies non-selected ones by 0 routing weight.
Even though `0.0 × finite_value = 0.0` exactly in IEEE 754, possible sources of error:
- `ttnn.repeat` may not produce exact copies on MeshDevice (view semantics unknown)
- `fast_reduce_nc` over 32 values in bfloat16 may accumulate ~6% error

**Next steps to investigate:**
1. Test with CPU-reference routing injected into TTNN (bypass TTNN sigmoid) to isolate routing error vs expert error
2. Test with single expert loop (`ttnn.linear` per expert in Python) to bypass batched matmul
3. Try `ttnn.sparse_matmul` following `gpt_oss` patterns (requires `ttnn.moe_routing_remap`)

## Architecture Notes

### MiniMax-M2.5 Specific

- **QK-norm**: Applied per TP shard (local approximation). Norm is over `NQ*D/TP`
  instead of the full `NQ*D`, so results differ slightly from reference. This
  causes ~0.5% PCC loss in attention (0.9946 vs 1.0).

- **Partial RoPE**: Only first `rotary_dim=64` of `head_dim=128` get rotary embedding;
  remaining 64 are NoPE (no positional encoding). Each TP device applies RoPE
  locally to its head shard.

- **Sigmoid routing with bias**: Router uses sigmoid (not softmax) + additive bias
  `e_score_correction_bias` only for TOP-K selection, not for actual routing weights.
  Routing weights are normalized sigmoid values.

- **SwiGLU**: Standard `silu(gate) * up`, no gpt_oss-style SwiGLU variant (no clamp, no alpha).

### Memory Per Device (estimated)

| Component | Per device | Basis |
|---|---|---|
| Attention weights | ~22 MB × 62 = 1.4 GB | TP=4: QKV [3072, 2048], O [1536, 3072] |
| Expert weights | ~225 MB × 62 = 14 GB | EP+TP: [1,32,3072,384]×3 per layer |
| Embeddings/norms/lm_head | ~1.5 GB | Replicated |
| **Total** | **~17 GB/device** | Fits in 12 GB DRAM? See note |

> **Note:** The 14 GB estimate for expert weights exceeds the 12 GB per-chip DRAM.
> May need FP8 quantization for expert weights, or further EP/TP factoring.
> Actual measurement on hardware pending successful test run.

## Dependencies on gpt_oss

The following are directly imported from `models/demos/gpt_oss`:

| Import | Used for |
|---|---|
| `gpt_oss.config.MeshConfig` | mesh shape, TP/EP axis, `column_parallel`, `row_parallel`, `allreduce` |
| `gpt_oss.config.ModeConfig` | decode/prefill mode config |
| `gpt_oss.tt.ccl.CCLManager` | semaphore management for CCL ops |
| `gpt_oss.tt.attention.operations.apply_allreduce` | TP all-reduce for attention O-proj |
| `gpt_oss.tt.experts.operations.apply_expert_parallel_allreduce` | EP all-reduce for MoE |
| `gpt_oss.tt.experts.operations.apply_tensor_parallel_allreduce` | TP all-reduce for MoE |

## Run Commands

```bash
export ARCH_NAME=wormhole_b0
export TT_METAL_HOME=$(pwd)
export PYTHONPATH=$(pwd)
source python_env/bin/activate

# Individual block tests
pytest models/demos/minimax_m2/tests/test_minimax_m2_tt.py::test_rmsnorm -xvs
pytest models/demos/minimax_m2/tests/test_minimax_m2_tt.py::test_partial_rope -xvs
pytest models/demos/minimax_m2/tests/test_minimax_m2_tt.py::test_attention -xvs
pytest models/demos/minimax_m2/tests/test_minimax_m2_tt.py::test_moe -xvs

# All tests
pytest models/demos/minimax_m2/tests/test_minimax_m2_tt.py -v
```
