# Mistral-Small-4-119B: which `deepseek_v3_d_p` tests are relevant?

Source: `pytest --collect-only` over `models/demos/deepseek_v3_d_p/tests/`
(+ `tests/sparse_mla` passed as its own top-level arg — see note 1).

**Totals: 192 test functions, 97,109 collected parametrizations, 71 test-bearing files.**
Concentration is extreme: `test_ds_prefill_transformer` alone is 73,920 params (76%);
the top 6 functions are 91,524 of 97,109.

## Classification scheme

Three-way, by how a model appears in the test:

| tag | meaning | count | cost to add mistral4 |
|---|---|---|---|
| **NH** | name-hardcoded — model is in the *function name* | 58 | a new test function |
| **VP** | variant-parametrized — model appears only in parametrize IDs | 36 | one param entry |
| **MF** | model-free — no model token in name or IDs | 98 | nothing |

**Caveat on MF:** `MF` is derived from ID vocabulary, *not* from the body. Some MF tests still
hardcode DeepSeek dims internally (e.g. `cache/test_embedding_cache.py` hardcodes
`vocab_size=129280, emb_dim=7168`). "98 MF" does not mean "98 tests already cover mistral4".

## Relevance verdicts

### A. RELEVANT — needs a new mistral4 test function (NH)

| file | new function | why relevant |
|---|---|---|
| `test_prefill_block.py` | `test_mistral4_prefill_block` | one MLA+MoE layer; the core per-layer check |
| `test_prefill_transformer.py` | `test_mistral4_prefill_transformer` | full-stack multi-layer prefill |
| `pcc/test_ttnn_moe.py` | `test_mistral4_moe` | MoE-every-layer model; MoE is half the compute |
| `test_kv_cache_table.py` | `test_mistral4_kv_cache_table` | kvpe latent is 320 wide, not 576 — geometry worth asserting |
| `perf/test_mla_perf.py` | `test_mistral4_mla_perf_galaxy` | per-op perf baseline |
| `perf/test_prefill_block_perf.py` | mistral4 case | block-level perf baseline |
| `test_mla.py` | **already exists** — `test_mistral4_mla:553`, 8 params | dense MLA PCC + KVPE PCC |

### B. RELEVANT — one-line variant param addition (VP)

| file | function | existing variant set |
|---|---|---|
| `cache/test_mla_weights_cold_warm_cache.py` | `test_mla_weights_cold_warm_cache` | dsv3, k3 |
| `pcc/test_moe_gate_prefill2d.py` | `test_forward_pass`, `test_hash_gate_forward_pass` | glm, kimi, minimax |
| `pcc/test_parallel_embedding.py` | `test_parallel_embedding` | — |
| `op_unit_tests/test_prefill_dispatch.py` | `test_ttnn_dispatch` | dsv3, dsv4_pro, dsv4_flash, gptoss_120b, kimi_k26, minimax_m27 |
| `op_unit_tests/test_prefill_combine.py` | `test_ttnn_combine` | same 6 |
| `op_unit_tests/test_ttnn_dispatch_combine.py` | `test_ttnn_dispatch_combine` | same 6 |
| `op_unit_tests/test_reduce.py` | `test_ttnn_reduce_models` | same 6 |
| `perf/test_dispatch_combine_perf.py` | `test_device_perf_dispatch_combine` | dsv3, glm52, kimi |
| `perf/test_prefill_dispatch_combine.py` | `test_ttnn_dispatch_combine` | dsv3, glm52, kimi |

### C. RELEVANT ONLY IF THE CHUNKED PATH IS AVAILABLE

mistral4 currently binds the single-shot dense attention path, not `_chunked_attn`
(`tt/mla/mla.py:598-616`). Verdict pending a separate investigation.

| file | functions |
|---|---|
| `test_prefill_block_chunked.py` | 6 NH functions |
| `test_prefill_transformer_chunked.py` | 8 NH functions — **the only place ttnn op-trace (`traced`/`notrace`) lives** |
| `test_mla.py::test_mla_chunked_prefill` | VP: deepseek_v3_d_p, kimi_k2_6, kimi_k3 |
| `test_chunked_trace_helpers.py` | VP helpers |

### D. NOT RELEVANT — architecture-specific to another model

This is the bucket the "HCA/DSA ones for DeepSeek" rule points at.

| path | why not |
|---|---|
| `sparse_mla/` (5 files, 137 params) | DSA sparse attention + indexer. mistral4 is **dense** MLA with no indexer; `mla.py` never binds the sparse path for it. Nothing to wire. |
| `dflash_prefill/` (2 files, 6 params) | DeepSeek-V4-flash attention variant. |
| `torch/test_kimi_k3_mla_reference.py` (9) | asserts Kimi-K3's own vendored reference. |
| `didt/test_deepseek_v3_128k_matmul.py` (32) | di/dt stress on hardcoded DeepSeek 7168-dim matmul shapes. |
| `pcc/test_deepseek_v3_matmul_pcc.py` (14) | same, PCC form. |
| `didt/sweep_deepseek_v3_matmul_tune.py` (1) | tuning sweep; not auto-collected (filename doesn't match `test_*.py`). |

### E. NO WIRING NEEDED — genuinely model-free infrastructure

Reported for completeness; adding a mistral4 entry to these would be hollow.

`test_dequant_utils.py` (7) · `test_runner_utils.py` (3) · `test_prefill_summary_utils.py` (9) ·
`test_sparse_kv_cache_contract.py` (15) · `test_zero_op_mem_validate.py` (1) ·
`test_disaggregation.py` (12) · `test_d2d_socket_sync.py` (4) · `test_h2d_socket_sync.py` (1) ·
`test_embedding_socket.py` (1) · `test_tokenizer.py` (18) · `test_prefill_block_loop.py` (3744, MF) ·
most of `op_unit_tests/` (combine_subdevices, rotary_embedding_indexed, update_padded_kv_cache,
fp8_kv_cache_gather, masked_bincount, mla_matmuls, moe_padding_config, offset_cumsum,
ring_joint_mla, rope_prefill, sub_device_load_clear_timing, zero_padded_kv_cache) ·
`pcc/{test_ffn,test_lm_head,test_moe_routing_setup,test_rmsnorm,test_shared_expert}.py` ·
`torch/{test_moe,test_moe_reference_comparison,test_torch_dispatch_combine}.py` ·
`cache/` (all except `test_mla_weights_cold_warm_cache`)

## Collection gotchas found

1. **`sparse_mla/` hard-errors as a subdirectory** — `Failed: Defining 'pytest_plugins' in a
   non-top-level conftest is no longer supported` (`tests/sparse_mla/conftest.py`). Collecting
   `tests/` yields 96,971 tests, **zero** from `sparse_mla`, and `Interrupted: 1 error during
   collection`. Pass `tests/sparse_mla` as its own top-level argument to collect it (137 tests).
2. `pytest.ini`'s `addopts` must be overridden with `-o addopts=--import-mode=importlib` or
   `--collect-only` prints `<Function>` blocks instead of usable node IDs.
3. Unregistered marks warn at collection: `extended_model`, `fp8_disp_compression`.
