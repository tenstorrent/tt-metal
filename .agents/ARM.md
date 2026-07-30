# skillexp arm: mvasiljevic/qb2/skillexp/nofuse-noadvise

|            | this arm |
|------------|----------|
| `$graph-fusing` (stage 01b + optimize step 1) | **no** |
| `$shard-advise` (OPT-015 + stage-02 hard gate) | **no** |

Shared parent: `mvasiljevic/qb2/skillexp/base`. Factor definitions, what is held constant, and why:
`.agents/EXPERIMENT.md`. Run plan: `docs/skillexp-run-plan.md`.

Stage sequence for this arm:

```

.agents/prompts/model_bringup_multigoal/02-optimized-decoder.txt
```

Stage 01 (`01-functional-decoder.txt`) is run once per model from `mvasiljevic/qb2/skillexp/base`, never from an
arm branch — all four arms optimize the *same* functional decoder for a given model.
