# Final64 tracked gate collector

`run_final64_gates.py` is prepared, **not executed**. Run only after the parent validates native64, canonical sampling, and the reduced adapter, and starts an exclusive healthy tracked server with the final source/configuration. The server should have trace/audit events enabled and the usual startup/warmup baseline needed by the lifecycle summarizer. The parent owns its separate 14400-second server lifetime and shutdown.

From the tt-metal checkout, with `python_env` and `../lane-env.sh` already activated:

```bash
export PYTHONPATH="$TT_MODEL_BRINGUP_ROOT/runtime:$PYTHONPATH"
export PYTEST_ADDOPTS='-p vllm'
python models/autoports/qwen_qwen3_6_35b_a3b/doc/vllm_integration/run_final64_gates.py \
  --server-log models/autoports/qwen_qwen3_6_35b_a3b/readiness_vllm/final64_tracked_server/server.log \
  --server-url http://localhost:8000 \
  --overall-timeout 12000 \
  --execute
```

Replace `--server-log` with the actual parent-owned tracked server log. Omit `--execute` to print the complete plan without HTTP, subprocess execution, or creating output directories. The default new output root is `readiness_vllm/final64_gate/`; it must not already exist. `--output-dir` can choose a fresh root. No retry/resume behavior is built in.

The serialized commands are:

1. Lifecycle: nonaligned 4101-token long prompt, 96 output tokens and sampled top-k 40 to exercise both native sampling widths, followed by the existing CPU summarizer using the complete server log (including startup capture/baseline events).
2. Shared plugin top-k selectors: all three `test_topk` batch sizes, `test_top1_is_greedy`, `test_batch1_seed_reproducible[1]`, `test_different_seeds_produce_different_outputs`, and `test_same_seeds_reproduce_across_batches`. The last three exercise native k=50, above the old boundary of 32. These canonical assertions remain intact; isolated 64-wide fixtures and canonical/reduced-adapter checks remain prerequisites.
3. P2: all four shared profiles, users 1 and 32, 128 output tokens, critical load 32, seeded policy `auto`. Before any overwrite, copy the existing normalized `doc/eval_readiness/harness_requests.json` and complete `harness_requests/` tree to `P2_previous/`, with an inventory of preserved paths. P2 writes its required normalized location; fresh results and raw runs are then copied to `final64_gate/P2/`. No capability override or threshold relaxation is supplied.
4. Shared `run_vllm_server --stages sampling --sampling-profile full`, with max sequences 32, model length 262144, block 32, mesh P300x2, the existing TT config, and its own new `sampling/` output directory.
5. Existing model collector for the shared six prompts, greedy and sampled, thinking off and 256-token budget, into `qualitative/`.

`execution.json` records exact effective commands, UTC times, raw subprocess exits, per-job time limits, server byte offsets and copied slices, actual host/device decode marker counts, adapter prefill/decode event counts, errors, idle observations, and pytest XML results. The per-job caps are 1800/120/1200/2400/4800/1800 seconds; the independent 12000-second overall budget shortens remaining job timeouts. GNU timeout affects only the gate client and its subprocesses, never the separately started server. The driver waits for two zero-running/zero-waiting metric scrapes before and after jobs.

A crash, request exception, incomplete artifact/pytest result, unexpected non-P2 gate failure, missing route evidence, or failure to become idle stops subsequent jobs. An ordinary completed P2 gate failure remains a failure but permits remaining collection. The sole full-suite `test_mixed_params_batch` assertion may also permit remaining collection **only** when it is the only failure, no error exists, and all announced tests have results. It is recorded as `ISOLATED_MIXED_REPRODUCIBILITY_ASSERTION_REQUIRES_REVIEW`, retaining the raw nonzero exit and assertion text; historical routing attribution is not asserted for a fresh failure. The overall driver exits nonzero if any gate fails. It cannot turn an exception disposition into runner success.

The collector does not judge or waive qualitative output. Read all 12 outputs, compare the planned same-code final64 standalone shared-prompt control, and record the required qualitative verdict separately. Existing all-ten constrained host-sampling controls remain separate unchanged evidence. P3, P5 accuracy, served max context, and benchmarks are not rerun here. Their carry-forward conditions and the final tracking-disabled performance server are parent responsibilities. On success the server is left idle; on failure the driver attempts a bounded idle check and records any failure, without issuing any server signal or shutdown.

Source-only validation: Python compile, dry-plan path (nonexistent server log; no output directory created), and CPU checks of route counting and strict singleton mixed-assertion classification passed. No HTTP, server, or TT operation was performed while preparing this driver.
