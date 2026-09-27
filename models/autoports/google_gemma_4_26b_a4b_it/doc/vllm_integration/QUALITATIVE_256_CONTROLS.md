# Matching the long serving continuations

The saved serving suite uses256-token limits, while the selected-policy standalone controls stop at128. The flagged greedy text (`shared_2`: `tiny-brass-heart`; `shared_3`: `In any own-contained system`) therefore requires longer matched controls. The sampled `shared_2` text also contains `brass-bound-and-etched` and `pair-o-glasses`; these remain preserved observations requiring review.

`tests/check_vllm_qualitative_256.py` loads the full selected-policy generator and produces256-token greedy controls for `shared_2`/`shared_3`. It verifies both rendered prompts and token IDs against the saved serving prompt metadata and prior selected control using tokenizer revision `4d7ae4984b7db7de8f8457170b3f1a419ee76d52`. It records source hashes, complete precision policy/runtime verification, raw generated token IDs, visible/special-token text, prior128-token prefix equality, current serving text equality, first differing text context, and presence/context of every flagged phrase. It emits no automatic quality verdict.

After the main agent has exclusive mesh ownership, run from `/workspace/tt-metal`:

```bash
TT_METAL_LOGS_PATH=/workspace/tt-metal/models/autoports/google_gemma_4_26b_a4b_it/readiness_vllm/runtime_logs \
python -m models.autoports.google_gemma_4_26b_a4b_it.tests.check_vllm_qualitative_256 \
  --output models/autoports/google_gemma_4_26b_a4b_it/readiness_vllm/qualitative_256_controls.json \
  > models/autoports/google_gemma_4_26b_a4b_it/readiness_vllm/qualitative_256_controls.log 2>&1
```

The default performs exactly the two required greedy generations. `--prepare-only` validates pinned cached tokenizer prompts and writes the comparison inputs without opening any device; this CPU preparation passed for both target prompts. Python compilation and formatting checks also passed. No hardware generation was run by the preparing subagent.

The original shared harness sends temperature0.7/top_p0.9 for sampled requests, but sends no request seed or top_k and saves only completion text. The current `vllm_qualitative_outputs.json` therefore does not contain original sampled token IDs or the effective random draw seed. The engine's logged seed0 is not an individual sampled request seed. Exact same-seed reconstruction is unavailable from these artifacts.

An optional **new** sampled control can be added with both `--sampled-seed <seed>` and `--sampled-top-k <1..32>`. Its parameters and explicit non-equivalence to the original random draw are recorded; it must not be represented as reproducing the original seeded request. Matching that sampled text would require new serving and standalone requests with a recorded common seed/effective parameters, or original request-level evidence that is currently absent.

Since serving saved text only, the report labels serving token comparisons as retokenized visible text. Those reconstructed IDs do not establish identity with the original generated token IDs; exact text equality and original prior-control token prefix equality are separate fields. Matching a questionable phrase in standalone would locate it outside the serving adapter for this policy; it would not by itself establish acceptable quality or HF equivalence. This check uses no profiler and preserves all anomalies for the reviewer.
