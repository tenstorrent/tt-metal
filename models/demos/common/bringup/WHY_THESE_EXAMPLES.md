# Bring-up kit — what to keep, and why

The selection rule: **keep the thing that would have saved a session.** Not the Llama answer —
the pattern, the template, or the measured fact that made the answer reachable.

```
models/demos/common/bringup/
├── README.md                     what the kit is, how to start a bring-up
├── BRINGUP_RECIPE.md             the recipe (facts folded into the body)
├── templates/
│   ├── 00_MODEL_CARD.md          fact table w/ mandatory Source column + "does NOT have"
│   ├── 05_DECISIONS.md           the DEC block, incl. Falsifier + Blast radius
│   ├── 06_GATES.md               ledger row + detail block (measured/floor/gap/dist/ref-dtype)
│   ├── 07_RISKS.md               risk entry w/ owner + how-to-close
│   └── 08_INTEGRATION.md         coverage table: what a gate proves vs what it does not
├── examples/
│   ├── module_test_vs_ref.py     the canonical gate test, incl. negative control
│   ├── noise_floor.py            quantize_like_device + err_ratio (the E.2 primitive)
│   └── verify_citations.py       machine-checks every path:line in code and docs
├── LANDMINES.md                  the failure playbook, all measured
└── scripts/new_bringup.sh        scaffold a package + empty logs + the nine files
```

## Per-part rationale

| Kept | Because, concretely |
|---|---|
| **Model card w/ `Source` column** | Every dimension traced to `config.json` or a `path:line`. The "what this model does NOT have" section is the anti-bloat control: it is what stopped MoE, attention sinks, sliding-window and QK-norm being copied in from the two nearest templates. |
| **DEC block** | The two fields that did the work are **Falsifier** and **Blast radius**. Without a falsifier a decision cannot be revisited; without blast radius nobody knows what a reversal costs. |
| **Gate ledger block** | Forces `measured / floor / gap / input distribution / reference dtype policy`. Each of those five caught a real error in this run. |
| **`noise_floor.py`** | The single highest-value artefact. Gating against another implementation's published PCC is invalid (its reference may share the device's rounding) and gating against a README number is a guess. |
| **`module_test_vs_ref.py`** | Identical random weights both sides, no checkpoint needed, and a **negative control**. A positive PCC without a control cannot distinguish a pass from a symmetric bug. |
| **`mesh_config.py` + `ccl_manager.py`** | The repo's converged answer: semaphores allocated once and cycled, collectives *inside* modules, TP the only knob. |
| **`dense_mlp.py`** | One module carried end to end — weight load, cache-only branch, collective tail, its gate. Every other module is a variation on it. |
| **`verify_citations.py`** | Caught wrong line numbers in the recipe *and* in agents' own first drafts. An unverified `path:line` is worse than none: it reads as authoritative. |
| **`LANDMINES.md`** | Each entry cost a session. |
| **Integration coverage table** | Separates "what this gate proves" from "what it does not", which is how `G-LOOPBACK` got correctly scoped out instead of being recorded as a blocker. |

## Deliberately NOT kept

- The Llama implementation (`tt/`), its tests, golden scripts, and filled-in logs. Those are the
  answer key. Shipping them makes the validation run measure transcription.
- Anything Llama-specific in the templates — dims, layer counts, key names.


## A note on what is NOT shipped as an example

`MeshConfig`, `CCLManager` and a worked dense MLP are **not** duplicated here. They already exist,
maintained, in `models/demos/gpt_oss_d_p/` and `models/demos/minimax_m3/`, which the recipe names as
the structural templates. Copying them into the kit would create a second version to drift.
