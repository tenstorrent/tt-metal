# Llama 3.1 8B Instruct Functional Decoder Work Log

## Stage

- Skill: `forge-functional-decoder`
- Hardware practice: `tt-device-usage` for serialized TTNN/device-facing commands.
- Target: `meta-llama/Llama-3.1-8B-Instruct`
- Autoport: `models/autoports/meta_llama_llama_3_1_8b_instruct`
- Forge source: `/localdev/mvasiljevic/ttnn-models/meta-llama/Llama-3.1-8B-Instruct/model/graph_0`

## Translation Notes

- Read `model_ttnn.py`, `consteval.py`, `params.py`, and `model_pt.py`.
- Implemented a single-layer `FunctionalDecoder(LightweightModule)`.
- Preserved Llama forge/HF math: RMSNorm eps `1e-5`, RoPE through the HF Llama 3.1 scaled RoPE setup, SDPA scale `1/sqrt(128)`, raw HF projection weights with `transpose_b=True` where projections are not setup-transposed, and SwiGLU MLP.
- Preserved the emitted workload batch size `32` as the default and tested shape.
- Did not copy static shard specs, paged-cache updates, seq/cache-position code, or static matmul program configs.
- `decode_forward` is an intentional `NotImplementedError` stub pending an emitted-decode graph.
- Note: the user goal text mentioned Qwen-style per-head `q_norm`/`k_norm` and eps `1e-6`; the supplied Llama forge emit and HF config do not have Q/K norm weights and use Llama RMSNorm eps `1e-5`, so the implementation translates the actual supplied emit/HF layer.

## Commands And Evidence

Hardware-facing commands were run serially per `tt-device-usage`.

```bash
timeout 60 tt-smi -ls --local
```

Result: failed because `tt-smi` is not on `PATH` in this environment (`timeout: failed to run command 'tt-smi': No such file or directory`).

```bash
python - <<'PY'
import ttnn
mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1), trace_region_size=0)
ttnn.close_mesh_device(mesh)
print('MESH_SMOKE_OK 1x1')
PY
```

Result: passed; opened local chip id `{0}` and printed `MESH_SMOKE_OK 1x1`.

```bash
.agents/scripts/check_context_contract.py --model-dir models/autoports/meta_llama_llama_3_1_8b_instruct
```

Result: passed with output `Context contract OK for models/autoports/meta_llama_llama_3_1_8b_instruct: target=131072, supported=64 (DRAM-limited).` The `DRAM-limited` text is the checker's label for any reduced context accepted through its DRAM-evidence branch; this stage did not measure or claim a physical DRAM limit.

```bash
pytest -q models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_functional_decoder.py -k runtime_has_no_host_fallback --tb=short
```

Result: passed in 1.51s.

```bash
pytest -q -s models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_functional_decoder.py -k 'synthetic_weight_prefill or decode_forward_documents_pending_path' --tb=short
```

Result: passed in 38.30s. Synthetic seq 16 PCC `0.9985012578882686`; synthetic seq 64 PCC `0.9981899661680298`; decode stub test passed.

```bash
pytest -q -s models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_functional_decoder.py -k real_weight_single_layer_prefill_matches_hf --tb=short
```

Result: passed in 4.26s. Real-weight batch 32 seq 16 PCC `0.9999980524146453`.

```bash
pytest -q -s models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_functional_decoder.py --tb=short
```

Result: passed in 30.42s. All 5 tests passed. PCCs printed in this full run: synthetic seq 16 `0.9985012578882686`, synthetic seq 64 `0.9981899661680298`, real weights seq 16 `0.9999980386971122`.

## PCC

- Synthetic batch 32 seq 16: `0.9985012578882686`.
- Synthetic batch 32 seq 64: `0.9981899661680298`.
- Real weights batch 32 seq 16: `0.9999980524146453` in the targeted run; `0.9999980386971122` in the full-file run.

## Runtime Contract

- `from_state_dict(state_dict, *, hf_config, layer_idx, mesh_device, max_seq_len=128, batch=32, **kwargs)`
- `prefill_forward(hidden_states, *, position_cos=None, position_sin=None, attention_mask=None) -> ttnn.Tensor`
- Input shape: `[1, batch, seq_len, 4096]`
- Output shape: `[1, batch, seq_len, 4096]`
- Runtime `prefill_forward` contains no torch conversion, `ttnn.from_torch`, `ttnn.to_torch`, or host fallback.
- `decode_forward(...)` raises `NotImplementedError("decode path pending emitted-decode forge version")`.

## Limitations

- Prefill-only.
- Static forge origin; this pass generalizes setup-time RoPE and mask construction but validates only selected lengths.
- The emitted graph in this checkout is decode/cache-oriented after a CPU prefill, not a complete generalized prefill graph.
- Decode pending a future emitted-decode forge version.
- No paged KV path.
- Current supported context is below the HF-advertised 131072 context.

## Stage Review

- Independent `$stage-review` subagent `019f3dac-3f17-7d03-ba65-737f050adbba` returned `clean-pass`.
- Required work: none.
- Other concerns: `prefill_forward` stores setup-time causal mask but defaults to `is_causal=True`; accepted because the skill allows causal SDPA without mask and PCC tests pass. Ignored `__pycache__` files exist in the autoport tree and are excluded from the checkpoint commit.
- Hard-check gaps noted by reviewer: review did not rerun hardware tests; tests validate seq 16 and 64 with `max_seq_len=seq_len` rather than a separate `seq_len < max_seq_len` case.
- Anomalies classified by reviewer: Qwen-like prompt details conflict with actual Llama forge/HF config and are controlled by translating actual source semantics; context-checker `DRAM-limited` label is a controlled schema compatibility issue; supplied forge graph is decode/cache-oriented after CPU prefill and this prefill-only limitation is documented.

## Local Commit

- Repo: `/localdev/mvasiljevic/tt-metal`
- Branch: `mvasiljevic/llama-bs32-rerun`
- Implementation checkpoint commit: `ffa42eee731`
- SHA-record follow-up commit: see final stage handoff.
- Scope: stage-owned files under `models/autoports/meta_llama_llama_3_1_8b_instruct`
- Unrelated untracked files excluded: `multigoal-logs/`
