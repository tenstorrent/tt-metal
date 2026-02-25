---
name: OLMo-3-32B Galaxy
overview: "Implement OLMo-3-1125-32B on Galaxy TT hardware in `/home/ttuser/ssinghal/PR-fix/main/debug/tt-metal`, branch `ssinghal/olmo-3-32b`, reusing the `llama3_70b_galaxy` stack. Three targeted changes versus Qwen3-32B (already working): new model config, YaRN RoPE, and per-layer sliding-window attention."
todos:
  - id: branch-setup
    content: Create branch ssinghal/olmo-3-32b in /home/ttuser/ssinghal/PR-fix/main/debug/tt-metal
    status: pending
  - id: phase0-ring-sdpa-cpp
    content: "Phase 0: Extend ring_distributed_sdpa C++ kernel to accept sliding_window_size (7 files, ~25 lines)"
    status: pending
  - id: phase0-ring-sdpa-build
    content: "Phase 0: Build tt-metal and run test_ring_sdpa_sliding_window.py with OLMo tensor shapes [1,5,seq_len,128] for seq_len in [4096, 8192, 32768], sliding_window_size=4096"
    status: pending
  - id: phase1-config
    content: Create tt/olmo_model_config.py (TtOlmoModelArgs) with correct dims, vocab, layer_types, sliding_window
    status: pending
  - id: phase1-weights
    content: Verify load_checkpoints.py key mappings work for OLMo HF keys (no qk_norm); add test_olmo_weight_loading
    status: pending
  - id: phase1-validate
    content: Run pytest test_olmo_model.py::test_olmo_weight_loading - validate Phase 1
    status: pending
  - id: phase2-yarn
    content: Integrate YaRN RoPE into llama_rope.py using tt_transformers YarnRotaryEmbedding; map attention_factor to mscale
    status: pending
  - id: phase2-validate
    content: Run pytest test_olmo_model.py::test_olmo_rope_pcc - validate Phase 2
    status: pending
  - id: phase3-sliding
    content: Add per-layer sliding_window_size to llama_decoder.py and llama_attention.py (all 4 SDPA variants)
    status: pending
  - id: phase3-validate
    content: "Run pytest test_olmo_model.py::test_olmo_sliding_window_decoder AND test_paged_sdpa_decode_sliding_window_olmo_shapes with cur_pos in [4095, 8191, 65535] - validate Phase 3"
    status: pending
  - id: phase4-demo
    content: Create text_olmo_demo.py and test_olmo_accuracy.py
    status: pending
  - id: phase4-validate
    content: Run full accuracy test batch_size=32; assert top-1 >= 80%, top-5 >= 98%
    status: pending
isProject: false
---

# OLMo-3-32B on Galaxy: Implementation Plan

## Architecture Delta (OLMo-3-32B vs. Qwen3-32B in Galaxy)


| Parameter            | Qwen3-32B (working)      | OLMo-3-32B (target)                |
| -------------------- | ------------------------ | ---------------------------------- |
| n_q_heads / head_dim | 64 / 128                 | 40 / 128                           |
| intermediate_dim     | 25600                    | 27648                              |
| vocab_size           | 151936                   | 100278                             |
| RoPE type            | linear (Llama-3 scaling) | YaRN (factor=8, orig=8192)         |
| Attention            | Full only                | 48 sliding (window=4096) + 16 full |
| Q/K norm             | Yes                      | No                                 |


Layer pattern (64 total): `[sliding, sliding, sliding, full] × 16`

## Working Area & Branch

```bash
cd /home/ttuser/ssinghal/PR-fix/main/debug/tt-metal
git checkout -b ssinghal/olmo-3-32b
export ARCH_NAME=wormhole_b0
export TT_METAL_HOME=$(pwd)
export PYTHONPATH=$(pwd)
export HF_MODEL="allenai/OLMo-2-1124-32B"
source python_env/bin/activate
```

---

## Phase 1: Branch + TtOlmoModelArgs + Weight Loading

**Files touched:**

- NEW `[models/demos/llama3_70b_galaxy/tt/olmo_model_config.py](models/demos/llama3_70b_galaxy/tt/olmo_model_config.py)` — copy of `qwen_model_config.py` with:
  - `n_q_heads = 40`, `n_kv_heads = 8` (40 heads × 128 = 5120 = hidden_size; no projection mismatch)
  - `intermediate_dim = 27648`, `intermediate_dim_per_tp = 27648 // 8 = 3456`
  - `intermediate_dim_per_tp_padded_24_cores` recalculated to next multiple of 32: `3488`
  - `padded_vocab_size = 100352` (ceil(100278 / 1024) × 1024, so 98 tiles × 32 × 32 devices)
  - `qk_norm = False`, `is_qwen = False`
  - Add `self.layer_types = (["sliding_attention"] * 3 + ["full_attention"]) * 16`
  - Add `self.sliding_window = 4096`
  - Rename class to `TtOlmoModelArgs`; update `LOCAL_HF_PARAMS` dict
- VERIFY `[models/demos/llama3_70b_galaxy/tt/load_checkpoints.py](models/demos/llama3_70b_galaxy/tt/load_checkpoints.py)` — OLMo uses standard Llama-style HF keys (`model.layers.*.self_attn.*`, `model.layers.*.mlp.*`), no `q_norm`/`k_norm`. The existing `map_hf_to_meta_keys` path (non-qk_norm branch) should work without changes; confirm by dry-running key mapping.
- NEW `[models/demos/llama3_70b_galaxy/tests/test_olmo_model.py](models/demos/llama3_70b_galaxy/tests/test_olmo_model.py)` — minimal pytest: instantiate `TtOlmoModelArgs`, call `load_hf_state_dict`, run `map_hf_to_meta_keys`, assert no missing/unexpected keys.

**Validation gate:** `pytest tests/test_olmo_model.py::test_olmo_weight_loading -v` passes (weights load, no key errors).

---

## Phase 2: YaRN RoPE Integration

OLMo rope_scaling from HF config:

```json
{"rope_type": "yarn", "factor": 8.0, "original_max_position_embeddings": 8192,
 "attention_factor": 1.2079, "beta_fast": 32, "beta_slow": 1.0}
```

`tt_transformers/tt/rope.py` already has `YarnRotaryEmbedding`; `tt_transformers/tt/common.py` has `RopeScalingYarn(beta_fast, beta_slow, mscale, mscale_all_dim)`.

**Files touched:**

- MODIFY `[models/demos/llama3_70b_galaxy/tt/llama_rope.py](models/demos/llama3_70b_galaxy/tt/llama_rope.py)`:
  - In `TtLlamaRotarySetup.__init__`, detect `args.rope_type == "yarn"` branch
  - When `yarn`: construct `RopeScalingYarn(beta_fast=32, beta_slow=1.0, mscale=attention_factor, scaling_factor=factor)` and call `rotary_embedding_factory(dim, max_pos, theta, rope_scaling=yarn_config)` from `tt_transformers` instead of `precompute_freqs` + `apply_scaling`
  - When not `yarn`: keep existing Llama-3 `apply_scaling` path
- MODIFY `[models/demos/llama3_70b_galaxy/tt/olmo_model_config.py](models/demos/llama3_70b_galaxy/tt/olmo_model_config.py)`:
  - Set `self.rope_type = "yarn"` (read from HF config `rope_scaling.rope_type`)
  - Set `self.attention_factor = 1.2079` (from HF config `rope_scaling.attention_factor`)
- ADD to `[models/demos/llama3_70b_galaxy/tests/test_olmo_model.py](models/demos/llama3_70b_galaxy/tests/test_olmo_model.py)`: `test_olmo_rope_pcc` — compare TT cos/sin output vs. `transformers` `OlmoRotaryEmbedding` reference, assert PCC > 0.999.

**Validation gate:** `pytest tests/test_olmo_model.py::test_olmo_rope_pcc -v` passes.

---

## Phase 3: Per-layer Sliding Window Attention

Reference for sliding_window SDPA calls: `models/demos/gpt_oss/tt/attention/decode.py` (lines 121-148) and `prefill.py` (lines 132-141).

**Files touched:**

- MODIFY `[models/demos/llama3_70b_galaxy/tt/llama_decoder.py](models/demos/llama3_70b_galaxy/tt/llama_decoder.py)`:
  - In `TtTransformerBlock.__init__`, compute: `sliding_window_size = args.sliding_window if args.layer_types[layer_num] == "sliding_attention" else None`
  - Pass `sliding_window_size=sliding_window_size` to `TtLlamaAttention(...)` constructor
- MODIFY `[models/demos/llama3_70b_galaxy/tt/llama_attention.py](models/demos/llama3_70b_galaxy/tt/llama_attention.py)`:
  - Accept `sliding_window_size=None` in `__init__`, store as `self.sliding_window_size`
  - In prefill SDPA call (~line 749): add `sliding_window_size=self.sliding_window_size`
  - In ring-SDPA prefill call (~line 740): add `sliding_window_size=self.sliding_window_size`
  - In decode SDPA call (~line 519): add `sliding_window_size=self.sliding_window_size`
  - In paged-decode SDPA call (~line 508): add `sliding_window_size=self.sliding_window_size`
- ADD to `tests/test_olmo_model.py`: `test_olmo_sliding_window_decoder` — run a sliding-window layer (layer 0) and a full-attention layer (layer 3) single-forward pass; compare outputs vs. CPU reference; assert PCC > 0.99.

**Validation gate:** `pytest tests/test_olmo_model.py::test_olmo_sliding_window_decoder -v` passes.

---

## Phase 4: End-to-End Demo and Accuracy

**Files touched:**

- NEW `[models/demos/llama3_70b_galaxy/demo/text_olmo_demo.py](models/demos/llama3_70b_galaxy/demo/text_olmo_demo.py)` — copy of `text_qwen_demo.py`; replace `TtQwenModelArgs` → `TtOlmoModelArgs`; adjust default `max_seq_len=65536`, `batch_size=32`.
- NEW `[models/demos/llama3_70b_galaxy/tests/test_olmo_accuracy.py](models/demos/llama3_70b_galaxy/tests/test_olmo_accuracy.py)` — copy of `test_qwen_accuracy.py`; use `TtOlmoModelArgs`; target `min_top1_acc=80, min_top5_acc=98`.

**Validation gate:** `pytest tests/test_olmo_accuracy.py -v` passes with top-1 ≥ 80%, top-5 ≥ 98%.

---

---

## Phase 0 (prerequisite): Extend Ring SDPA Kernel for sliding_window_size

**Why first:** Ring SDPA is triggered for `seq_len > 1024 and batch_size == 1` — exactly the OLMo long-sequence single-user prefill case. 48 of 64 OLMo layers are sliding-window. Without this, ring SDPA silently uses full attention for all sliding-window layers.

**Root cause confirmed:** `sliding_window_size` is hardcoded to `0` at two compile-time arg positions in `ring_distributed_sdpa_program_factory.cpp` (lines 268 and 302). The kernel tile logic (`kernels/compute/sdpa.cpp`) is shared with regular SDPA which already implements sliding window — it just needs to receive the correct value.

**Exact changes — 7 C++ files, ~25 lines:**

1. `ttnn/cpp/ttnn/operations/transformer/sdpa/device/ring_distributed_sdpa_device_operation_types.hpp`
  - Add `std::optional<uint32_t> sliding_window_size;` to `RingDistributedSDPAParams`
2. `ttnn/cpp/ttnn/operations/transformer/sdpa/device/ring_distributed_sdpa_device_operation.hpp`
  - Add `std::optional<uint32_t> sliding_window_size = std::nullopt` to `ring_distributed_sdpa()` signature (line 48)
3. `ttnn/cpp/ttnn/operations/transformer/sdpa/device/ring_distributed_sdpa_device_operation.cpp`
  - Add `sliding_window_size` to function signature
  - Add `.sliding_window_size = sliding_window_size,` when constructing `operation_attributes` (lines 303-311)
4. `ttnn/cpp/ttnn/operations/transformer/sdpa/device/ring_distributed_sdpa_program_factory.cpp`
  - Extract: `const auto& sliding_window_size = operation_attributes.sliding_window_size;`
  - Line 268: replace `0, //(uint32_t)sliding_window_size,` → `sliding_window_size.value_or(0),`
  - Line 302: replace `0, //(uint32_t)sliding_window_size,` → `sliding_window_size.value_or(0),`
5. `ttnn/cpp/ttnn/operations/transformer/sdpa/sdpa.hpp`
  - Add `std::optional<uint32_t> sliding_window_size = std::nullopt` to `ExecuteRingDistributedScaledDotProductAttention::invoke` (line ~135)
6. `ttnn/cpp/ttnn/operations/transformer/sdpa/sdpa.cpp`
  - Add `sliding_window_size` to `invoke` signature and forward it in `ttnn::prim::ring_distributed_sdpa(...)` call (~line 289)
7. `ttnn/cpp/ttnn/operations/transformer/sdpa/sdpa_nanobind.cpp`
  - Add `std::optional<uint32_t> sliding_window_size` to the ring pybind lambda (~line 613) and bind it as `nb::arg("sliding_window_size") = nb::none()`

**Validation gate — ring SDPA tensor-size test:** Build tt-metal, then run `pytest tests/test_ring_sdpa_sliding_window.py -v`.

OLMo-3-32B per-device shapes at the ring SDPA call site (Galaxy 8×4 mesh, `num_devices_per_group = n_kv_heads = 8`):

- `n_local_heads = 40 // 8 = 5`
- `n_local_kv_heads = 8 // 8 = 1`
- `head_dim = 128`

Test spec — `tests/test_ring_sdpa_sliding_window.py`:

```python
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
@pytest.mark.parametrize("seq_len", [4096, 8192, 32768])  # all > 1024 to trigger ring path
@pytest.mark.parametrize("sliding_window_size", [4096])   # OLMo sliding window
def test_ring_sdpa_sliding_window_olmo_shapes(seq_len, sliding_window_size, mesh_device):
    # Q: [1, n_local_heads=5, seq_len, head_dim=128] per device
    # K/V: [1, n_local_kv_heads=1, seq_len, head_dim=128] per device
    q = torch.randn(1, 5, seq_len, 128, dtype=torch.bfloat16)
    k = torch.randn(1, 1, seq_len, 128, dtype=torch.bfloat16)
    v = torch.randn(1, 1, seq_len, 128, dtype=torch.bfloat16)
    # Put on device, call ring_distributed_scaled_dot_product_attention
    # with sliding_window_size=sliding_window_size, ring_size=4
    # Reference: ttnn.transformer.scaled_dot_product_attention with same sliding_window_size
    # Assert: output PCC > 0.99 vs reference; no crash for all three seq_len values
```

The three `seq_len` values cover: equal-to-window (4096), 2× window (8192), and 8× window near OLMo max (32768 = 65536/2).

---

## Risk Resolution Summary

**Risk 1 (ring SDPA + sliding_window):** Resolved by Phase 0. The underlying kernel already supports it; only the parameter wiring was missing. Tensor-size test above confirms OLMo shapes work before any model integration.

**Risk 2 (paged decode + sliding_window):** No blocker. Investigation confirmed:

- `paged_scaled_dot_product_attention_decode` C++ kernel fully accepts `sliding_window_size` (it's in the struct, pybind, and forwarded to the primitive)
- The `attention_1d.py` rejection only applies to `chunked_scaled_dot_product_attention` in the gpt_oss module — the Galaxy stack (`llama_attention.py`) never uses that module
- Galaxy uses non-chunked `scaled_dot_product_attention` for prefill, which also accepts `sliding_window_size`
- **Fix in Phase 3**: Just pass `sliding_window_size=self.sliding_window_size` at the 4 existing SDPA call sites in `llama_attention.py`

**Paged decode tensor-size test** (added to Phase 3 validation):

OLMo-3-32B per-device shapes at paged decode call site:

- `batch_per_device_group = max_batch_size // n_groups = 32 // 4 = 8`
- Q: `[1, n_local_heads=5, batch_per_device_group=8, head_dim=128]`
- KV cache (paged): `[max_batch_size=32, n_local_kv_heads=1, max_seq_len=65536, head_dim=128]`

```python
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
@pytest.mark.parametrize("cur_pos", [4095, 8191, 65535])  # at/beyond sliding window boundary
@pytest.mark.parametrize("sliding_window_size", [4096])
def test_paged_sdpa_decode_sliding_window_olmo_shapes(cur_pos, sliding_window_size, mesh_device):
    # Q: [1, 5, 8, 128] - 5 heads, 8 users per device, head_dim=128
    # KV cache: [32, 1, max_seq_len, 128] - 32 users, 1 kv_head, max_seq_len=65536
    # page_table: [32, num_pages] with page_block_size=64
    # cur_pos_tensor: [32] with value=cur_pos for all users
    # Assert: paged_scaled_dot_product_attention_decode with sliding_window_size=4096 runs
    #         without error; when cur_pos >= sliding_window_size, output differs from
    #         full-attention output (confirms window masking is active)
```

The three `cur_pos` values test: just before the window fills (4095), 2× window (8191), and at max context (65535). The last two confirm that tokens outside the 4096-token window are correctly masked.

**Risk 3 (YaRN attention_factor mapping):** OLMo's `attention_factor=1.2079` maps to `mscale` in `RopeScalingYarn`. Validate numerically in Phase 2 gate test.
