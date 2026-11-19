# Complete native TT attribution control

Prepared CPU-only; no device or HTTP workload was launched. Prefer one complete ten-seed run in a fresh artifact over merging old seeds 0/3 with a new eight-seed run. The existing collector already accepts `--seeds`; no collector or model edit is needed.

From the tt-metal root, after the parent has serialized hardware and verified it is free:

```bash
source python_env/bin/activate
source ../lane-env.sh
TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 VLLM_PLUGINS='' \
timeout 1800 python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.run_vllm_native_host_control \
  --reference models/autoports/qwen_qwen3_6_35b_a3b/readiness_vllm/sampling_controls/http_bad_words_card_1024.json \
  --checkpoint /home/ttuser/qwen36-35b-a3b/hf/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/345d467b117d1fd5eda9b879a5988de9ae5f47a7 \
  --seeds 0 1 2 3 4 5 6 7 8 9 \
  --output models/autoports/qwen_qwen3_6_35b_a3b/readiness_vllm/sampling_controls/tt_native_host_card_all10.json \
  > models/autoports/qwen_qwen3_6_35b_a3b/doc/vllm_integration/logs/tt_native_host_card_all10.log 2>&1
```

This is the collector's existing 1800-second outer bound, matching its recorded `required_outer_timeout_s`. The saved HTTP corpus contains 4620 tokens; the prior two native cases took 271.52 seconds for 1794 tokens, excluding setup. That supports using the existing bound prospectively but does not guarantee completion. Do not extend a running watchdog. If a larger bound is chosen before launch, the collector currently hardcodes 1800 in its docstring/artifact, so correct that diagnostic metadata first rather than leaving contradictory provenance.

The command uses all 40 layers; selected precision from `doc/datatype_sweep/selected_precision_config.json`; mesh 1×4; capacity 32; retained decode buckets 2/32; the canonical generator traced model path; and the existing explicit diagnostic CPU host sampler. It is **not** the on-device sampling performance path. It reads raw logits and feeds host-selected tokens through the documented diagnostic compatibility route, which must remain outside performance evidence. Standalone context is 2048, sufficient for the exact 14-token prompt plus 1024-token cap; its allocator differs from serving, as the collector records.

Prompt IDs, singleton bans, thinking-on setting, temperature 1, top-p 0.95, top-k 20, presence 1.5, seeds, and token cap come from the saved HTTP artifact. Each seed resets the canonical generator and constructs a fresh request-seeded sampler; model and sampling traces remain captured across cases. The sampler helper fixes top-k 20 and the collector asserts that the reference requests it. EOS IDs come from the pinned checkpoint. These are the exact intended card-control settings, not new test recipes.

The collector finishes all requested seeds before returning comparison failure, so a changed early seed does not skip later attribution. `collection_status=PASS` alone is insufficient: inspect `comparison_status`, all ten rows, shutdown capture/release counters, and the process exit. The parent must separately confirm clean device release.

## Comparing actual arrays and existing blind coverage

The adjacent CPU-only `compare_native_card_controls.py` reads the actual HTTP response `token_ids`, not merely the native artifact's copied expected arrays. It verifies request identity, all-layer mesh identity, response IDs, and completed rows; multiple native inputs require identical recorded source/config identities and disjoint seeds. It joins the frozen blind review through its key and transfers a served judgment only for an exactly reproduced token sequence. It never changes the failed HF comparison or invents native scores for differing text.

```bash
python_env/bin/python models/autoports/qwen_qwen3_6_35b_a3b/doc/vllm_integration/sampling_debug/compare_native_card_controls.py \
  --native models/autoports/qwen_qwen3_6_35b_a3b/readiness_vllm/sampling_controls/tt_native_host_card_all10.json \
  --controls models/autoports/qwen_qwen3_6_35b_a3b/readiness_vllm/sampling_controls \
  --output models/autoports/qwen_qwen3_6_35b_a3b/readiness_vllm/sampling_controls/native_card_all10_attribution.json
```

Expected success is `ALL_TEN_EXACT`, ten exact seeds, and six of six existing served secondary failures reproduced. This is a prospective outcome, not an observed result. Any differing native trajectory needs its own prompt-correct assessment and potential first-divergence investigation; no saved served score automatically transfers.

CPU validation against the existing two-seed artifact correctly returned exit 1 with `INCOMPLETE_OR_DIFFERENT exact=2/10 attributed_failures=2/6`; output is `native_card_existing_coverage.json` beside this report. This deliberately demonstrates the remaining eight-seed coverage gap without running a model.

## Reuse decision

At preparation time, the old artifact's recorded source hashes matched current files except `tt/generator_vllm.py`. The standalone collector does not execute that adapter, so this difference alone does not invalidate the old native trajectories. Nevertheless, one complete run avoids a mixed-provenance result and rechecks seeds 0/3 after intervening work. It is the preferred command above.

An eight-seed-only run would use `--seeds 1 2 4 5 6 7 8 9` and a fresh output. Reusing old 0/3 would require explicit proof that all executed model, collector, CPU sampler, selected precision, and native runtime configuration remained unchanged. The supplied strict comparison script intentionally rejects an indiscriminate merge when any recorded source identity differs. Do not rewrite old hashes or overwrite the original artifact to bypass that check.

Even all ten exact native matches would only complete attribution of this served behavior to the TT full model under the tested recipe. The frozen HF 2/10 versus served 6/10 comparison stays failed; no HF equivalence, unavoidable-quantization claim, or acceptance waiver follows.
