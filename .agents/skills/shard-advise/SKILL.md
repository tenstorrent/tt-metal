---
name: shard-advise
description: "Get a good-enough L1 sharding strategy for a pre-written ttnn model (or a single block) from the tt-mlir compiler, instead of hand-deriving it. The greedy optimizer decides which tensors to L1-shard, how, on what grid, plus the matmul program config for its pick; use it to skip the mechanical sharding search during model bringup / porting, apply the result as a baseline, then tune the rest and re-query specific blocks as needed. Produces a structured report.json (per-op layout, program_config, reshards, L1 spill). Not for perf profiling and not for dtype-precision decisions."
---

# Shard Advise

## Mission

Get a good-enough L1 sharding strategy for a pre-written model **without spending
your own reasoning budget rediscovering it**. Deriving which tensors to shard,
how, and on what grid is exactly the mechanical search tt-mlir's greedy L1
optimizer already does — so ask the advisor instead of exploring it yourself.

Give it the model (or one block). It traces the ttnn function, runs the greedy
optimizer, and hands back, per op, the layout to use and the matmul program
config for it (`report.json`). Apply that as your sharding baseline, then spend
your effort on the parts the advisor doesn't cover — dtype precision, the
DRAM-sharded-weight strategy, kernel configs (see Scope) — rather than on the
sharding the compiler can hand you for free.

Intended loop:

1. **Get a baseline** — run the advisor on the model → apply the reported
   layouts + program configs. Cheap, fast, compiler-validated.
2. **Tune what's left** — profile, then adjust the axes the advisor doesn't own.
3. **Re-query a piece** — changed one block, or want a fresh strategy for just
   the MLP / attention? Point `advise_decoder.py` at that piece and ask again.

## When to use

- You have a pre-written (or freshly ported) model and need a sharding strategy
  to start from — don't hand-derive it, ask the advisor first.
- You want a per-op layout + program-config map for a block you didn't write.
- You changed a block and want the compiler's current best layout for it.

It is a fast baseline, not the last word: it reasons about L1 layout + the
program config for its pick (see Scope). Do **not** use it for perf numbers
(profile instead) or to decide tensor dtype precision (bf16 vs bfp8/bfp4).

## Setup (once per shell)

```bash
source .agents/skills/shard-advise/scripts/bootstrap.sh
```

`bootstrap.sh` activates the pre-built tt-mlir advisor env (from
`$TTMLIR_ADVISOR_HOME`) and ensures `SYSTEM_DESC_PATH` is set. If it reports the
advisor env is missing, that is one-time operator setup — see the integration
README; do not try to build tt-mlir from inside an experiment.

## Run it

Point the `advise_decoder.py` capture target at the experiment's decoder, then
run the advisor in a **fresh process** and read `report.json` — never scrape
stdout:

```bash
# edit scripts/advise_decoder.py: set MODEL_DIR / config / layer to the experiment
ttnn-advise capture .agents/skills/shard-advise/scripts/advise_decoder.py:decode \
    --out /tmp/shard-advice 2>/dev/null

python -c "import json; d=json.load(open('/tmp/shard-advice/report.json')); \
  print('\n'.join(f\"{o['index']:>3} {o['op']:<45} {o['layout']}\" for o in d['ops']))"
```

Or, if a TTIR `.mlir` dump already exists (no device needed):

```bash
ttnn-advise mlir path/to/model.ttir.mlir --out /tmp/shard-advice 2>/dev/null
```

## Read the result

`/tmp/shard-advice/report.json`:
- `ops[]`: `{index, op, layout}` — e.g. `l1/width_sharded/1x64 cores=(0,0)-(7,7)`
- `reshards[]`: `{kind, producer, consumer, from, to, output_revert}`
- `spill`: `{ran, total_spills}` — near-zero is healthy
- `total_ops`, `final_choices`, `artifacts{...}`

Also written: `report.txt` (human-readable), `final_ir.mlir` (authoritative TTNN
IR), `pipeline.log` (captured native output, for debugging only).

**Apply it as the baseline:** `ops[].layout` + `ops[].program_config` are the
strategy to write onto each op's `memory_config=` / `program_config=`. Typically
the advisor width-shards the L1-resident projections across the grid with a
1d-multicast matmul config and keeps SDPA-decode / KV cache in DRAM — take that
as given rather than re-deriving it. If the model already sets something and the
advisor disagrees, prefer the advisor's pick unless you have a measured reason
not to (then it becomes a tuning question, step 2).

## Scope — do not over-read

The advisor advises L1 layout / sharding **and** the matmul **program config**
the optimizer picks for that strategy (e.g. `matmul_multi_core_reuse_multi_cast_1d
@8x8`, in each op's `program_config`). It faithfully traces the dtypes the model
already chose (bfp4/bfp8 weights included), so layout reasoning uses the real
footprint — but it does not *recommend* a dtype change.

It does **not** pick the **DRAM-sharded-weight** matmul strategy (a distinct
optimizer feature landing soon; once chosen its program config surfaces the same
way) or tune **compute-kernel configs** (hifi2/hifi4). Comparing to a hand-tuned
model, expect agreement on the layout skeleton + chosen-strategy program config,
and gaps on precision and the DRAM-sharded-weight strategy.

## Gotchas

- **Fresh process per run** — the optimizer's device context is process-global.
- **ttnn version skew** — the advisor traces against tt-mlir's ttnn, not the
  experiment's tt-metal branch. If tracing fails on a ttnn op, that op's tracer
  handler needs aligning (bounded work in tt-mlir); report it rather than
  working around it.
- Read `report.json`; the CLI keeps stdout to a 5-line summary and routes all
  pipeline/device logging to `pipeline.log`.
- `nlp_concat_heads_decode` needs a sharded input; the advisor marks it unfixable
  instead of emitting the reshard. Insert the interleaved->sharded conversion (or a
  valid sharded seed) yourself around it — the one op that needs hand-repair here.
- Advisor geometry is batch-shaped: capture at the batch the downstream stage serves
  (and a small batch when they differ), not only batch-1.
- A first constraint failure is not a rejection — run the full legal layout family and
  the matmul-only isolate, and preserve any spill recommendation (dropping one can break PCC).
