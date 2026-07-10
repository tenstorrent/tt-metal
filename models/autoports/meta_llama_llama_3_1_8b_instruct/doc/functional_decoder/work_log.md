# Functional Decoder Work Log

## Provenance

Translated from:

- `/localdev/mvasiljevic/ttnn-models/meta-llama/Llama-3.1-8B-Instruct/model/graph_0/model_ttnn.py`
- `/localdev/mvasiljevic/ttnn-models/meta-llama/Llama-3.1-8B-Instruct/model/graph_0/consteval.py`
- `/localdev/mvasiljevic/ttnn-models/meta-llama/Llama-3.1-8B-Instruct/model/graph_0/params.py`
- `/localdev/mvasiljevic/ttnn-models/meta-llama/Llama-3.1-8B-Instruct/model/graph_0/model_pt.py`

The emit is batch 32, active decode seq 1, cached length 128. The functional decoder exposes prefill only.

## Implementation Notes

- Added `tt/functional_decoder.py` with `FunctionalDecoder(LightweightModule)`.
- `from_state_dict` accepts canonical HF keys and layer-local keys.
- Runtime uses bf16/TILE/DRAM defaults and supports arbitrary `seq_len` at the TTNN op level.
- Runtime validates that the input batch matches the configured emitted batch.
- Decode raises `NotImplementedError` with pending emitted-decode text.
- Llama-3.1 has no q/k norm weights; RMSNorm epsilon is `1e-5`.
- The actual emit QKV order is `[Q,V,K]` for layers 0-30.
- `ttnn.experimental.rotary_embedding` was tested and found to pad prefill sequence to 32; runtime uses an equivalent TTNN rotate-half formula.

## Commands

```bash
timeout 60 tt-smi -ls --local
```

Result: failed, `tt-smi` not on PATH.

```bash
timeout 120 python - <<'PY'
import ttnn
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
ttnn.close_mesh_device(mesh)
print("MESH_SMOKE_OK")
PY
```

Result: `MESH_SMOKE_OK`.

```bash
timeout 600 pytest -q models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_functional_decoder.py --tb=short -s
```

Result: 5 passed.

Measured PCC:

- `SYNTHETIC_PREFILL_SEQ_1_PCC=0.998992`
- `SYNTHETIC_PREFILL_SEQ_8_PCC=0.998662`
- `REAL_WEIGHT_PREFILL_SEQ_4_PCC=0.999998`

## Stage Review

First independent review returned `more-work-needed`:

- Context contract overstated supported context as 131072 despite validation only to seq 8.
- Sharding recommendations were aggregated and some emitted program configs were abbreviated.

Remediation:

- `context_contract.json` now records `current_supported_context=8` and keeps HF advertised context separate.
- `forge_sharding_recommendations.json` now declares a per-layer/per-role template schema for repeated decoder ops and expands the previously abbreviated MLP program configs to complete emitted field sets.
- `prefill_forward` now rejects runtime batches that do not match the configured emitted batch.

Note: `.agents/scripts/check_context_contract.py --model-dir models/autoports/meta_llama_llama_3_1_8b_instruct` fails with the honest context contract because the checker currently requires DRAM evidence for every below-HF context. This functional stage is below advertised due to validation scope, not a proven DRAM limit.

Rereview after remediation returned `clean-pass` with no required work. The reviewer classified the context-checker conflict and decode-shaped source graph as controlled anomalies.

## AutoFix Follow-Up: Context Checker

An AutoFix-style independent diagnostic pass inspected the remaining failed gate:

```bash
python .agents/scripts/check_context_contract.py --model-dir models/autoports/meta_llama_llama_3_1_8b_instruct
```

Result:

```text
models/autoports/meta_llama_llama_3_1_8b_instruct/doc/context_contract.json supports context 8, below HF-advertised 131072, without device-DRAM capacity evidence.
```

Verdict: no honest model-dir-only JSON change can both preserve:

- `hf_advertised_context = 131072`
- `current_supported_context = 8`, the largest validated prefill seq
- non-DRAM, validation-scope reduction notes
- checker success

The checker condition is:

```python
if supported < target and not has_dram_limit_evidence(contract):
    return 2
```

The only mechanical JSON edits that pass are to overstate supported context as 131072 or falsely claim a DRAM capacity limit. Both would violate the stage evidence, so this artifact intentionally keeps the honest context contract.

## Checkpoint Commit

- Repo: `/localdev/mvasiljevic/tt-metal`
- Branch: `mvasiljevic/llama-forge-seed-rerun`
- Implementation checkpoint commit: `77600fb40a5`
- Context-checker diagnostic commit: pending
- Push: not pushed
