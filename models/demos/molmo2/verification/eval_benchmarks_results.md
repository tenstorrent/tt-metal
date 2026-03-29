# Molmo2-8B Eval Benchmarks Results

Date: 2026-03-29

## Summary

| Benchmark | Samples Run | TT Score | Published Score | Status |
|-----------|-------------|----------|-----------------|--------|
| chartqa | 1250 (50%) | 9.36% | 85.7% | Completed |
| docvqa_val | 1026 (~38%) | N/A | 88.7% | Server crashed |
| mmmu_val | 0 | N/A | 51.0% | Not started |

## Configuration

- **Model:** allenai/Molmo2-8B
- **Device:** T3K (8 Wormhole devices)
- **Server:** vLLM with TT backend
- **Eval Framework:** lmms-eval v0.4.1
- **Eval Mode:** ci-nightly (50% of dataset)
- **Eval Class:** openai_compatible

## Eval Config Added

Added `EvalConfig` for Molmo2-8B in `tt-inference-server/evals/eval_config.py`:

```python
EvalConfig(
    hf_model_repo="allenai/Molmo2-8B",
    tasks=[
        EvalTask(
            eval_class="openai_compatible",
            task_name="chartqa",
            workflow_venv_type=WorkflowVenvType.EVALS_VISION,
            score=EvalTaskScore(
                published_score=85.7,
                published_score_ref="https://huggingface.co/allenai/Molmo2-8B",
                ...
            ),
        ),
        EvalTask(task_name="docvqa_val", ...),
        EvalTask(task_name="mmmu_val", ...),
    ],
)
```

## ChartQA Results (Completed)

### Metrics
- **Samples:** 1250 / 2500 (50%)
- **Accuracy:** 9.36%
- **Published:** 85.7%
- **Gap:** -76.34 percentage points

### Sample Outputs Analysis

| Target | Model Response | Match | Issue |
|--------|----------------|-------|-------|
| 14 | Thirteen | ❌ | Spelled number instead of digit |
| 0.57 | 10.0 | ❌ | Wrong value |
| 3 | Four | ❌ | Spelled number, wrong count |
| No | Yes | ❌ | Opposite answer |
| 23 | 23 | ✅ | Correct |
| 6 | 7 | ❌ | Off by one |
| 62 | 222222222222 | ❌ | Garbage output |
| Yes | Yes | ✅ | Correct |
| Inspired | Inspired | ✅ | Correct |
| 0.03 | 4444444444444444 | ❌ | Garbage output |

### Identified Issues

1. **Number Format Mismatch:** Model outputs spelled numbers ("Thirteen") instead of digits ("14")
2. **Garbage Outputs:** Some decimal value questions produce repeated digit strings
3. **Counting Errors:** Model miscounts items in charts
4. **Yes/No Inversions:** Some boolean questions answered incorrectly

## DocVQA Results (Incomplete)

- **Samples Completed:** ~1026 / 2675 (~38%)
- **Status:** Server crashed during evaluation
- **Estimated Time/Sample:** Started at ~2s, degraded to ~47s before crash

## Commands Used

### Start Server
```bash
cd /home/ttuser/ssinghal/PR-fix/tt-metal
export TT_METAL_HOME=$(pwd) && export PYTHONPATH=$(pwd) && source python_env/bin/activate
python -m vllm.entrypoints.openai.api_server \
  --model allenai/Molmo2-8B \
  --trust-remote-code \
  --max-model-len 4096 \
  --max-num-seqs 1 \
  --block-size 64
```

### Run Evals
```bash
cd /home/ttuser/ssinghal/PR-fix/tt-metal/tt-inference-server
source python_env/bin/activate
OPENAI_API_KEY="dummy" python run.py --model Molmo2-8B --workflow evals \
  --tt-device t3k --limit-samples-mode ci-nightly
```

### Smoke Test (3 samples)
```bash
OPENAI_API_KEY="dummy" python run.py --model Molmo2-8B --workflow evals \
  --tt-device t3k --limit-samples-mode smoke-test
```

## Output Files

- **Results JSON:** `workflow_logs/evals_output/eval_id_tt-transformers_Molmo2-8B_t3k/allenai__Molmo2-8B/20260329_041134_results.json`
- **Samples JSONL:** `workflow_logs/evals_output/eval_id_tt-transformers_Molmo2-8B_t3k/allenai__Molmo2-8B/20260329_041134_samples_chartqa.jsonl`

## Next Steps

1. **Investigate Low Accuracy:**
   - Check if lmms-eval prompt format is optimal for Molmo2
   - Verify generation settings (temperature, max_tokens)
   - Compare with direct API calls that work correctly

2. **Fix Server Stability:**
   - Server crashed during docvqa eval
   - May need memory optimization for large evals

3. **Complete Remaining Evals:**
   - Restart docvqa_val from beginning
   - Run mmmu_val

4. **Tune Eval Config:**
   - Consider adding model-specific prompts in lmms-eval
   - Adjust generation parameters if needed

## Notes

- Video verification tests (105/105) pass correctly with coherent responses
- Direct image/video API calls work well
- Discrepancy between API quality and eval scores suggests prompt/format issue
