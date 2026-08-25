# arch_donor — which supported model is the new one most like?

Given a new decoder LLM's HuggingFace config, rank the models already
implemented in tt-metal by **mechanism similarity, per block**, so each block of
a new bring-up can start from the closest existing implementation instead of
from a guess.

Host-only. No device, no build, no weights — it reads `config.json` files.

## The rule that must not be broken

**The comparison is HuggingFace config vs HuggingFace config.** Nothing about
Tenstorrent hardware may enter the similarity metric: no mesh shapes, no
TP/SP/EP, no bfp8/bfp4, no kernel names, no chunked prefill. Those are
*consequences*. Once you know "same attention mechanism, 1.5x wider", you go
read that donor's recipe and adapt it.

If you find yourself scoring donors by how their TT implementation looks, the
tool has been inverted. The whole point is that implementation reuse is derived
from architectural similarity, not mixed into it.

Corollary: categorical **mechanism** and numeric **shape** are never combined
into one score. "Identical mechanism, different shapes" is the most valuable
verdict this produces, and it only exists because the two stay separate.

## Usage

```bash
# a local config, or any HuggingFace repo id (fetched on demand)
python -m models.tooling.arch_donor.compare path/to/config.json
python -m models.tooling.arch_donor.compare mistralai/Mistral-Medium-3.5-128B

python -m models.tooling.arch_donor.compare <target> --all-sizes  # include sub-Galaxy donors
python -m models.tooling.arch_donor.compare <target> --json       # machine-readable
python -m models.tooling.arch_donor.corpus                        # list the donor corpus + tiers

pytest models/tooling/arch_donor/tests/test_arch_donor.py
```

Add `--noconftest` to pytest if your local `ttnn` build is stale; these tests
never import it, but the repo-root `conftest.py` does.

## Reading the output

Per-block verdicts, ordered by how much work they imply:

| Verdict | Meaning | Implication |
|---|---|---|
| `identical` | every mechanism field equal | reuse the block; retune shapes only |
| `compatible` | differs only in `host` / `ingest` fields | reuse the dataflow; regenerate a table or add a dequant step |
| `near` | exactly **one** dataflow difference | right skeleton, one swap — usually the most useful real donor |
| `unverified` | a field is `unknown` on one side | go read the modeling source; **not** a match |
| `different` | two or more dataflow differences | wrong donor for this block |

Severity is what separates them:

- `dataflow` — the computation graph changes (a rewrite): attention kind, QK-norm,
  RoPE coverage, sparsity, sinks, GLU flavour, MoE routing.
- `host` — only a host-side parameter changes: RoPE scaling type, norm epsilon.
- `ingest` — weight loading changes: fp8, mxfp4.

Donor **tier** matters as much as the verdict:

- `proven` — has an active `*galaxy*` SKU in `models/model_targets.yaml`.
- `in-flight` — a bespoke `models/demos/<model>/` bring-up, no galaxy target yet.
- `reference` — supported elsewhere; mechanism reference only.

A mechanism-identical `reference` 7B model is not a parallelism donor, which is
why `galaxy_class` (≥50B) gates the default ranking.

`shapeΔ` is the mean absolute log-ratio over the block's dimensions; `0.00`
means the shapes are *equal*, not merely close.

## The step you must not skip

The ranking comes from config fields plus a `MODEL_TYPE_TRAITS` table in
`signature.py`. **Before committing to a donor, open its modeling source for
that block and confirm the mechanism is really what the tool claims.**

Configs lie by omission. `qwen3` has no QK-norm flag but applies per-head
QK-norm unconditionally; `gpt_oss` has no attention-sinks flag at all. Both are
only handled because the traits table says so, and that table is a claim about
source that someone read once.

## Extending it

When a new `model_type` appears, the tool announces the gap
(`no traits entry: qk_norm/sinks/norm_style may be under-reported`) and reports
`unknown`, which surfaces as an `unverified` verdict. To close it:

1. Add field aliases to the tuples in `signature.py` (`N_EXPERTS`, `TOP_K`, …)
   if the family spells a dimension differently. Five spellings of "number of
   experts" already exist across deepseek / gpt-oss / minimax / kimi / gemma4.
2. Add a `MODEL_TYPE_TRAITS` entry for mechanisms no config field encodes,
   **with an inline comment naming the source file you read**. Never guess here.
3. Add the model to `PARAM_TRUTH` in the test with its published card numbers.
   The parameter estimate is the alias canary: a misread expert count fails by
   10–100x, not by a plausible margin.

## Known limits — repeat these when reporting results

- **Totals are asserted to 5%; "activated" params are a bracket.** Vendors
  disagree on whether embeddings count (MiniMax's 23B excludes them, OpenAI's
  5.1B counts one copy), so the tool reports
  `[active_no_embed_B, active_embed_once_B]` and a published figure should land
  inside it.
- **Vision towers and multi-token-prediction modules are excluded** from every
  estimate. Text backbone only.
- **The divisibility section covers head / hidden / FFN splits only.** It says
  nothing about sequence parallelism, which does not divide those dimensions.
- **Corpus freshness follows `models/model_targets.yaml`.** A model whose galaxy
  perf CI has not landed reads as `in-flight`, not `proven` — true of
  `minimax_m3` and `deepseek_v3` as of 2026-08.
- **`unverified` is not a soft `compatible`.** It means the traits table has a
  hole; fill the hole.

## Validation

`tests/test_arch_donor.py` checks parameter estimates against seven published
model cards, mechanism signatures against five hand-verified configs, and two
retrodictions:

- Target **MiniMax-M3** returns `deepseek_v3` as the MLP donor, verdict `near`,
  the single dataflow difference being the activation — which is what actually
  happened: `a4e461ee4cc` extended DeepSeek's `unified_routed_expert_ffn` with a
  SwiGLU-OAI activation.
- Target **Mistral-Medium-3.5-128B** returns `Llama-3.3-70B-Instruct`
  (`models/demos/llama3_70b_galaxy`) on 5/5 blocks — attention `compatible`
  (only YaRN vs llama3 RoPE scaling, host-side), MLP `identical` at shapeΔ 0.00.

**If you change the taxonomy, these must still pass.** A taxonomy that cannot
rediscover reuse that already happened is wrong, and that is the only real
check we have on it.

## Status

Not merged. Living on a branch for feedback — the taxonomy is the part most
worth arguing about, so please push back on the block/severity split before
this goes to `main`.
