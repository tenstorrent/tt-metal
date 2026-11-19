# Current-production reduced sampler geometry probe

Prepared source only; no model or device execution by the author. This is fresh diagnostic evidence for the actual common-sampler prediction after forced token44, including the B2-history singleton survivor. It does not waive the frozen failed cross-geometry comparison or promote full-model acceptance.

`tests/probe_current_geometry_hf.py` constructs one production generator with real layers0/1/3, configured batch2, capacity128 and a16-block shared BF16 paged pool. Single-active controls execute the configured-B2 generator's n1 route; they are explicitly **not true configured-B1 runs**. Prompts, physical page mappings and forced42/43/44 histories match `probe_shared_pool_generator.py`; HF references are the hash-verified original `shared_pool_hf_control_v2.pt` histories.

There are five histories, each repeated once: solo request0, solo request1, uninterrupted B2, migration after42 followed by B2 steps43/44, and the same migration followed by release of request1 before singleton-survivor44. This gives10 prefills and30 decode calls,62 saved prediction rows. Repeated rows establish reproducibility and are excluded from accuracy denominators. Output reports top1/top5/top100 separately by history and prefill/decode, including the raw device token, true torch argmax, all maximum-tie IDs and strict HF rank. Top1 uses the actual device token versus HF argmax, not `topk()[0]`. Every sampled token must equal its TT row's maximum value; a valid tied greedy token need not equal torch's lowest-index argmax.

Every prefill and decode uses the production common device sampler, with explicit per-request temperature/top-k/top-p/**seed lists**. Migration supplies authoritative two-row sampling parameters and resets sampling state. Full-vocabulary diagnostic readbacks occur after the sampled call and never drive the next input;42/43/44 are externally forced. Thus this probe is neither token-out timing evidence nor an autoregressive feedback-loop acceptance test. It verifies all-rank sampled feedback, logical/QSA positions, exact repeated tokens/logits, same-history continuous-versus-migrated outputs, immutable pool addresses and second-suite trace/program-cache reuse. It does not add cache-state equivalence claims beyond the recorded positions/feedback.

The output preserves the historical v5 FAILED result and its hashes/43/44 metrics. It also recomputes current cross-geometry comparisons using the unchanged threshold function, reporting their verdicts separately. Exit success requires execution, exact scoped checks, semantic greedy correctness, unchanged source hashes and the direct-HF top5>=98%/top100=100% bar. It makes no combined acceptance claim and intentionally does not convert a separately reported frozen cross-geometry failure into a pass.

Device-owner command, after the usual accepted checkpoint/packed-PLE lane setup, restoring devices0/1/2/3 and the P300x2 descriptor:

```bash
export TT_VISIBLE_DEVICES=0,1,2,3
export TT_MESH_GRAPH_DESC_PATH=tt_metal/fabric/mesh_graph_descriptors/p300_x2_mesh_graph_descriptor.textproto
export TT_METAL_TRACE_ALLOC_TRACKING=1
export TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0
timeout --signal=INT --kill-after=60 900 python_env/bin/python \
  -m models.autoports.qwen_qwen3_8_flash_next.tests.probe_current_geometry_hf \
  --output models/autoports/qwen_qwen3_8_flash_next/doc/full_model/current_geometry_sampler_v1.json \
  --tensors /home/ttuser/qwen38-flash/reference/current_geometry_sampler_v1.pt
```

Both output paths must be absent; the probe refuses overwrite. Use an external owner-managed log/wait wrapper. Expected duration is roughly1–4 minutes based on earlier reduced model loads/probes, not a measurement of this new harness;900 seconds is an outer diagnostic bound. Device ownership and timeout triage remain with the parent. Only a few tens of MiB of logits are saved; the real reduced-model weight/cache footprint remains the usual existing probe footprint.

CPU validation completed:

```bash
python_env/bin/python -m models.autoports.qwen_qwen3_8_flash_next.tests.probe_current_geometry_hf --self-test
python_env/bin/python -m py_compile models/autoports/qwen_qwen3_8_flash_next/tests/probe_current_geometry_hf.py
/home/ttuser/.cache/pre-commit/repo5lt0iwfz/py_env-python3/bin/isort --check-only models/autoports/qwen_qwen3_8_flash_next/tests/probe_current_geometry_hf.py
/home/ttuser/.cache/pre-commit/repow99qs9xl/py_env-python3/bin/black --check --include '\.py$' models/autoports/qwen_qwen3_8_flash_next/tests/probe_current_geometry_hf.py
/home/ttuser/.cache/pre-commit/repoqd0pcgsr/py_env-python3/bin/autoflake --check --remove-all-unused-imports --remove-unused-variables models/autoports/qwen_qwen3_8_flash_next/tests/probe_current_geometry_hf.py
```

All passed. Self-test imports neither TTNN nor model weights, covers valid tied greedy selection, a tied HF maximum distinct from HF argmax, non-greedy selection, genuine HF top5/top100 misses, invalid token IDs, nonfinite logits and positive/negative repeat equality. No production source, prior probe, scorer threshold or historical artifact was edited.
