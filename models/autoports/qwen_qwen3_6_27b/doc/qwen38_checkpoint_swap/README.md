# Qwen3.8-27B on the Qwen3.6-27B autoport — evidence

Question: can the pipeline-produced Qwen3.6-27B autoport run Qwen/Qwen3.8-27B weights well
enough to be the deliverable for 3.8, instead of running the 11-stage bringup pipeline?

Answer so far: **yes at layer and logit level**, with the release eval still in flight.

- tt-metal branch `mvasiljevic/fmf/qwen-qwen3-8-27b-via-3-6` (from `mvasiljevic/fmf/qwen-qwen3-6-27b`)
- worktree `/home/mvasiljevic/tt-metal-q36`, built Release with profiler, clang-20 in container
- implementation `models/autoports/qwen_qwen3_6_27b` — contains **no** `models/demos` or
  `models/experimental` imports, i.e. a genuine from-scratch port
- `TARGET_MESH_SHAPE = (1, 4)`, which is exactly this machine's 4 Blackhole chips

## Why the substitution is legitimate

`Qwen/Qwen3.8-27B`'s `config.json` is identical to `Qwen/Qwen3.6-27B`'s except
`transformers_version` (`5.8.0.dev0` vs `4.57.1`): same 64 layers, hidden 5120, 24 heads / 4 KV,
head_dim 256, vocab 248320, 262144 context, 48 `linear_attention` + 16 `full_attention`, one MTP
layer, same vision tower. Same graph; only weights and tokenizer files differ.

## Checkpoint identity — proven, not assumed

Verified because a silent 3.6-vs-3.8 mixup would invalidate every number:

| check | result |
|---|---|
| loaded config `transformers_version` | `5.8.0.dev0` (3.6 reads `4.57.1`) |
| shard count in index | 18 (3.6 ships 15) |
| snapshot revision | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |
| `_name_or_path` | `Qwen/Qwen3.8-27B` |
| tensor fed to BOTH HF and TT vs the 3.8 shard | `torch.equal` → **True** |
| 3.6 checkpoint present on host | **no** — only `models--Qwen--Qwen3.8-27B` exists |

Structurally unmixable: the PCC test builds one `config` and one `state` dict and passes those same
objects to `_hf_layer(config, state)` and `from_state_dict(state, hf_config=config)`. The full-model
test loads HF from `default_snapshot()` with `local_files_only=True`.

## Results

| measurement | 3.6 (port's own record) | 3.8 (measured) |
|---|---|---|
| DeltaNet layer-0 decode PCC (48 of 64 layers) | 0.999922 | **0.999868** |
| Full-attention layer-3 decode PCC (16 of 64) | 0.997297 | **0.997952** |
| AIME24 teacher-forced top-1 | 97–98/100 | **95/100** |
| AIME24 teacher-forced top-5 / top-100 | 100/100 | **100/100** |
| teacher-forcing decode | 6.262–6.98 t/s/u | **6.43 t/s/u** |
| teacher-forcing TTFT | 5.13–16.60 s | **12.65 s** |
| full-model greedy HF-vs-TT, token-exact | 3/6 | **0/6** |
| full-model greedy HF-vs-TT, first-80-char | 4/6 | **1/6** |
| r1_gpqa_diamond (CI subset) | 0.60 (author flags as inflated) | **0.70 ± 0.153 — also inflated, see below** |

**Reading the full-model divergence.** 0/6 token-exact looks alarming next to 3.6's 3/6, but
top-5 = 100/100 means HF's chosen token is inside TT's top-5 at *every one* of the 100 teacher-forced
positions — so there is no systemic error in embeddings, LM head, layer accumulation or MTP. Only 5
positions flip the argmax, i.e. near-ties. Under greedy decoding one early near-tie sends the
trajectory somewhere different but equally valid, which matches what was observed: fluent, coherent
text on both sides, not degradation. This is *not* the degraded-text defect the 3.6 port documented
for its own GPQA run.

## ★ The blocking finding: the text-quality defect reproduces on 3.8

`r1_gpqa_diamond` scored **0.70 ± 0.153** (n=10, CI subset) — nominally better than the 3.6 port's
0.60, and far below the 84.74% bar. But the score is not the result. Sample inspection:

| signal | value |
|---|---|
| contains `\boxed` | 9/10 |
| **closed `</think>`** | **4/10** (the 3.6 run got 10/10) |
| response length | min **11 chars**, median 1051, max 21391 |

- **docs 3, 4, 5 scored 1 on 11–15 characters**: the entire response is `\boxed{B}` / `\boxed{C}` /
  `\(\boxed{A}\)` with **no reasoning at all** and no `</think>`. With four choices that is 25% luck
  per item, so three of the seven "correct" answers carry no evidence of having been reasoned.
- **doc1** (3859 chars) is raw byte garbage:
  `'.e: \u, butD:\x86\x46\x40\x41\x42\x43\x96\x80…\x02\x01\x02\x01…'`
- **doc7** (12492 chars) is a repetition loop: `'the"g "the"the"the"the"the"the … Control controlthe'`
- **doc9** scored 1 after a corrupted preamble: `' </think> </think> <password>> . </think> To determine…'`
- only doc0 and doc2 read as clean, coherent answers

**This is the same defect the port documented for itself.** `doc/SAMPLING_TEXT_QUALITY.md` records
the 3.6 run as "0.60, and a text-quality defect the score hides", with the same signature —
`'Let's Let's Let's'`, `'** **'`, degraded tails, only one well-formed document out of ten.

So the defect is **in the implementation or serving path, not in the checkpoint**: it reproduces
across two different checkpoints of the same architecture. Our 3.8 run is arguably worse on two
counts — `</think>` closure fell from 10/10 to 4/10, and raw byte sequences appear, which the 3.6
run did not show.

Note where it does *not* appear: layer PCC, teacher-forced top-1/top-5, and the 50-token greedy
full-model test are all clean. The defect belongs to **long free-running sampled generation** —
which is exactly what serving does, and exactly what a release eval measures.

**Consequence for the decision.** "Use the 3.6 implementation instead of the pipeline" is sound on
correctness (PCC, logits) but inherits a known, unfixed serving-quality defect. That defect must be
root-caused before this can ship as the Qwen3.8 deliverable; it is pre-existing work on the 3.6 port,
not something introduced by the checkpoint swap.

## Fixes required (each from an observed failure)

In-tree, on the branch:

1. `functional_decoder.py` — `MODEL_ID`/`MODEL_REVISION` read `QWEN_AUTOPORT_MODEL_ID` /
   `QWEN_AUTOPORT_MODEL_REVISION`; new `default_snapshot()` derives the snapshot from `$HF_HOME/hub`
   rather than the hardcoded `/huggingface/hub` the port was written against. Defaults unchanged.
2. `linear_attention_real_pcc.py` — resolve the layer's shards from `model.safetensors.index.json`.
   The hardcoded `model-000NN-of-00015.safetensors` names could never match 3.8's 18 shards. The
   full-model `SnapshotReader` was already index-driven.
3. `full_model_qualitative.py` — report `MODEL_ID` instead of a hardcoded `"Qwen/Qwen3.6-27B"`, which
   otherwise stamps a 3.6 label onto 3.8 outputs.
4. `models/common/sampling/generator.py` — `reset_trace()` interpolated a ttnn trace handle inside a
   `logger.debug` f-string. Eagerly evaluated, so it raised `TypeError` at any log level and aborted
   `reset()` **and** `teardown()`, leaking traces and masking the real error. **Still live on
   `origin/main`.** Worth upstreaming; will hit anything using traced sampling.

Out of tree (vLLM checkout is not part of this repo):

5. `TTQwen3_5ForConditionalGeneration` was hardwired to
   `models.demos.blackhole.qwen36.tt.qwen36_vllm`. Added a selector mirroring the Mistral entry,
   `TT_QWEN3_5_TEXT_VER=qwen3_6_27b_autoport|demo`, **defaulting to the autoport** — this
   architecture is meant to be served by `models/autoports/qwen_qwen3_6_27b`, not the demo (user's
   call, 2026-08-18). The default matters rather than being cosmetic: the tt-inference-server release
   workflow spawns its own server and knows nothing about this variable, so a demo default would
   silently evaluate the demo. Patch: `vllm-register-qwen36-autoport.patch`.

   **Consequently the release-flow branch is wrong about this model's code path.** In
   `reference_config/benchmarking/benchmark_targets/model_performance_reference.json`, the
   Qwen3.8-27B entry's comment states it "rides the same tt-metal code path
   (models/demos/blackhole/qwen36)". That should say `models/autoports/qwen_qwen3_6_27b`. Its
   performance targets (`ttft_ms 62.0`, `tput_user 41.0`) are themselves flagged "ASSUMED, NOT
   VALIDATED", extrapolated from Qwen3-32B on a **t3k (8 devices)** while this model runs on 4, so
   they need real measurement before they mean anything.

## Serving environment recipe (undocumented gap)

`create_venv.sh` alone cannot serve. Three successive startup failures established the full recipe:

```bash
./create_venv.sh                                   # tt-metal env (has uv, NOT pip)
uv pip install -r vllm/requirements/common.txt \
  -c vllm-deps-constraints.txt                     # pin torch: ttnn is compiled against it
uv pip install -e vllm/plugins/vllm-tt-plugin      # editable: registers the entry points
```

- plain `pip` is absent from the venv, so `pip install` escapes to system Python and fails PEP 668
- `VLLM_PLUGINS=tt,tt_model_registry` only *selects* plugins; without installing `vllm-tt-plugin`
  the entry points do not exist and vLLM dies with `Failed to infer device type`
- editable matters: it makes the patched `platform.py` the live code

Serving verified: vLLM resolved `qwen_qwen3_6_27b.tt.generator_vllm`, model `Qwen/Qwen3.8-27B`,
`Fabric initialized on 4 devices`.

## Eval configuration

Mirrors the `Qwen/Qwen3.8-27B` entry on tt-inference-server
`vvukoman/add-8-models-to-release-flow` (`60f80c4b`, `reference_config/evals/eval_config.py` ~L1374):
`r1_gpqa_diamond` on EVALS_COMMON, chat endpoint, `max_length=262144`, `stream=false`,
temperature 1.0 / top_k 20 / top_p 0.95. Published GPQA Diamond 89.2 with **no**
`gpu_reference_score`, so the check is `accuracy >= 89.2 * (1 - 0.05)` = **84.74%** — a bar that
entry's own comment expects to fail on early runs (status EXPERIMENTAL, evals informational).

Two deviations, printed into every run log so the number is never mistaken for a full-spec result:
`max_gen_toks=32768` (spec 81920; at ~56 ms/token 80K is up to ~76 min per document) and
`--limit 0.05` (the ~10-document CI subset, not the full 198).

## Open decisions (not measurements)

- Should the vLLM registration change be upstreamed? It decides which implementation owns
  `TTQwen3_5ForConditionalGeneration` for this architecture.
- Should the deliverable live under `models/autoports/qwen_qwen3_8_27b`, or remain the 3.6 port
  driven by `QWEN_AUTOPORT_MODEL_ID`? The latter is what is validated today.
- The `reset_trace()` fix should go upstream regardless of this model's outcome.
