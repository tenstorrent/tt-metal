# Stage 11 prerequisites on BH QuietBox 2 (p300x2)

Date: 2026-08-14 UTC
Purpose: what must exist before the Stage 11 release workflow can produce a
*valid* report on this host, and what is already prepared. Written during the
host move; not a stage verdict.

## The integration-side work from Stages 09-11 was never pushed

tt-metal kept everything. Both integration repos lost their patches, verified
against the public APIs:

| Repo | Autoport work | Present upstream? |
| --- | --- | --- |
| tt-metal | model, generator, `tt/generator_vllm.py`, all 11 stages' evidence | **yes**, branch `mvasiljevic/fast-models-fast/gemma4-31b` |
| tenstorrent/vllm | `TT_GEMMA4_TEXT_VER` selector + autoport registration, recorded as nested plugin commit `91c467d6fc18c4386eda14360baf0bee0e0f684c`, checkout `44b7853d448f3f8c5db7ed068a4f82ebfcd1065d` | **no**. Both SHAs return `422 No commit found`; `TT_GEMMA4_TEXT_VER` and `gemma4_31b_autoport` appear nowhere in `dev` (head `bf98d556bb46`) |
| tenstorrent/tt-inference-server | nine harness fixes, `f1a89cb4b` .. `b803374e0` | **no**. `b803374e04c2460ea3bfabec4bfed832f2af532a` returns `422 No commit found`, while tag `v0.18.0` resolves normally to `d5913e816ac5`, so the API check is sound |

## Critical: without the plugin patch, the release report is silently invalid

The checkpoint declares `architectures: ["Gemma4ForConditionalGeneration"]`.
Upstream vLLM `dev` maps that arch to
`models.demos.gemma4.tt.generator_vllm:Gemma4ForCausalLM`, and
`models/demos/gemma4/tt/generator_vllm.py` exists in this checkout. A server
started without the patch therefore comes up cleanly, serves requests, and
produces a release report against **`models/demos/gemma4`, not the autoport**.

Stage 11's contract forbids exactly this: "Do not run stock `tt-transformers`,
`models/demos`, or another packaged implementation. Such a report is invalid
even if `run.py` exits 0." Nothing errors, so this must be checked positively
rather than assumed.

Stage 09's review recorded the intended target: "Shared TT vLLM registration
selects `models.autoports.google_gemma_4_31b.tt.generator_vllm:Gemma4ForCausalLM`",
selected by `TT_GEMMA4_TEXT_VER=gemma4_31b_autoport`.

**Verify before trusting any Stage 11 report**: confirm the served class resolves
to `models.autoports.google_gemma_4_31b.tt.generator_vllm`, not
`models.demos.gemma4`.

## Prepared on this host

- `tenstorrent/vllm` `dev` cloned at `/home/mvasiljevic/vllm`, head `bf98d55`.
- The `TT_GEMMA4_TEXT_VER` selector has been re-applied to
  `plugins/vllm-tt-plugin/src/vllm_tt_plugin/platform.py`, following the existing
  `TT_LLAMA_TEXT_VER` / `TT_QWEN3_TEXT_VER` convention in the same function. It
  **defaults to `demos`**, so upstream behavior is unchanged unless
  `TT_GEMMA4_TEXT_VER=gemma4_31b_autoport` is set. Uncommitted working-tree
  change; it is not a tt-metal change and cannot ride this branch.
- `tenstorrent/tt-inference-server` cloned at
  `/home/mvasiljevic/tt-inference-server`. `v0.18.0` confirmed as
  `d5913e816ac5`, matching the recorded release base. `v0.19.0` and `v0.20.0`
  also available.
- Verified checkpoint at `/home/mvasiljevic/models/google/gemma-4-31B` and a
  warmed 30 GB TTNN tensor cache. See `../full_model/revalidation_p300x4/`.

## Not prepared

- **vLLM is not installed.** Neither `python_env` nor `/opt/venv` has it, so
  `tests/test_vllm_adapter_contract.py` fails at import. Both venvs are
  otherwise identical (torch 2.11.0+cpu, Python 3.10.19) and `ttnn` imports from
  either. The documented procedure is
  `source plugins/vllm-tt-plugin/docs/install-vllm-tt.sh` from the vLLM repo root
  with the tt-metal env active, which runs
  `VLLM_TARGET_DEVICE=empty uv pip install -e .` then installs the plugin. The
  plugin's `pyproject.toml` states it "runs only inside a tt-metal python_env"
  and that vLLM must be the locally built `empty` target, never the PyPI CUDA
  wheel.
- **The nine TTI harness fixes.** Rebuildable from the recorded diagnoses in
  `autofix/*/FIX_RESULT.md` and `autofix/*/AUTODEBUG.md`. Oldest first:
  `f1a89cb4b` external autoport release specs, `6ad299582` service port for
  external workflows, `c5eb37b7a` release eval and target wiring, `e4d2307cd`
  eval context and prompts, `8a69f76d4` IFEval scorer schema, `569f62b01` vLLM
  conformance pytest discovery, `507c74673` long vLLM conformance requests,
  `fdc353375` bounded determinism conformance samples, `b803374e0` Meta GPQA
  answer filtering.
- **Serving ceiling re-probe.** The 113,280-token limit in
  `../context_contract.json` came from per-bank DRAM allocator probes on P150b,
  not arithmetic. Per-chip DRAM geometry is identical here, so it is expected to
  carry, but it is a probe result and must be re-probed.

## Host differences that change the Stage 11 commands

This host is `qb2-120-p02t03`, a BH QuietBox 2: two `p300c` boards, four
Blackhole chips, 11x10 worker grid, 249 GB RAM, 16 physical cores.

- **TTI device selector is `p300x2`, not `p150x4`.** Upstream v0.20.0 documents
  this platform at `docs/model_support/llm/gemma-4-31B-it_p300x2.md`
  (`python3 run.py --model gemma-4-31B-it --device p300x2 ...`). The recorded
  Stage 11 run used `--tt-device p150x4`, which does not describe this host.
- **No Docker.** There is no `docker` CLI, so Stage 11's Docker fallback is
  unavailable and the external-server path is mandatory. That path is the one
  the prompt prefers anyway.
- **Hugepages are not preallocated** (`HugePages_Total: 0`); UMD warns and falls
  back to regular pages, which costs host-device DMA bandwidth. Worth fixing
  before quoting serving throughput.
- For reference only, upstream lists the stock `tt-transformers` `-it` variant on
  this platform at max batch 1 and **max context 49,152**. That is a different
  implementation and a different checkpoint, so it is not a comparand for the
  autoport's 113,280, but it is a reminder that the ceiling is platform-specific.

## The readiness gate is unchanged by any of this

Both mandatory Meta rows still lack a canonical base-checkpoint reference:
`meta_ifeval` 25.181850822484343 and corrected `meta_gpqa_cot`
26.339285714285715. `stage_review_final.md` priced the exact local control at
67+ hours and refuted the batching, MTP, and prompt-lookup shortcuts. This host
has no CUDA or ROCm device and 16 physical cores, so it cannot produce that
reference.

Realistic ceiling for Stage 11 here is `release-workflow-pass /
readiness-fail`, the same verdict already recorded. Reaching `readiness-pass`
needs GPU access for the reference run, an evidence-backed row waiver, or a
product-owned replacement gate; it is not a stage-execution problem.
