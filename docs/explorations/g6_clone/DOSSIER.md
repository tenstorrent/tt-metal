# g6_clone: exploration dossier

Status: ready for golden-tests
Date: 2026-11-25
Request: "Explorator ticket for the library op g6_clone (group 6, data movement): bundle and op spec under the lead's contract; support every case of scenarios.json and beat the incumbent stock op on device time"

## In one screen

- Computes: a bit-identical copy of one DRAM-interleaved device tensor into a new tensor of the
  same shape, dtype, layout and placement, `out[i] = x[i]` bit for bit, out new storage
- Comes from: the library's group 6 (data movement), called by FLUX.2-dev and Qwen3.8-27B
  text-only (29 cases); reference PyTorch `torch.clone`, pytorch/pytorch@70d99e998b49 (torch
  2.11.0); the case list "lead of the library: traced production runs"; no paper
- Contract: `from ttnn.operations.g6_clone import g6_clone`; `g6_clone(input_tensor: ttnn.Tensor) -> ttnn.Tensor`
- Phase 0: dtype ttnn.float32, layout ttnn.TILE_LAYOUT, DRAM interleaved, every rank and
  alignment. Target: bfloat16 TILE; bfloat16, int32 and uint32 ROW_MAJOR (the production pairs)
- Oracle evidence: E1 for the PyTorch function the contract names (`torch.clone` executed on this
  host, the pinned build), max relative difference 0.0 in float64; bit-identity probes on all
  four production dtypes and all 29 production shapes passed; no comparison at another level
- Oracle accepted without execution by: nobody
- Cold reads (planner, cheater, skeptic): by independent readers (three subagents, each given only the files its pass names)
- Largest risks: R1 an engine or conversion path changes bits; R2 output aliasing the input;
  R3 partial tiles of the three non-aligned TILE shapes; R5 4 B and 32 B ROW_MAJOR pages
- Waiting for a human: Q1 (lead, blocks nothing), Q2 and Q3 (golden-tests)

## Where to look

| You need | Read |
|---|---|
| The contract the op must meet | `spec/g6_clone.txt` |
| What the reference does, statement by statement | `reference.md#semantic-statements` |
| The formula, symbols and edge cases | `math.md#definition` |
| Which results span which axes, and what a cut along an axis would have to carry | `math.md#axes-and-dependence` |
| Equivalent forms that suit tiles and streaming | `math.md#reformulations` |
| Value ranges, accumulation depth, where precision goes | `math.md#numerics` |
| Tensor sizes, tile grids, bytes and operation counts | `tensix.md#geometry-and-counts` |
| Compute phases in tile terms | `tensix.md#compute-phases` |
| Which primitive could carry each phase, and what is missing | `tensix.md#candidate-primitives`, `tensix.md#gaps` |
| What is likely to go wrong, and how it would show | `risks.md` |
| A runnable PyTorch reference and its checks | `oracle/pytorch_g6_clone.py`, `oracle/check_oracle.py` |
| Answers for the golden-tests interview | `golden_brief.md` |
| The upstream code and the paper, offline | `sources/SOURCES.md` |
| What the lead decided for this op: name, boundary, tensors at the boundary, call matrix | `sources/local/contract.md` |
| Every case production calls the op with | `sources/local/scenarios.json` |
| The model-wide rules, as the lead handed them in | `sources/local/rules.md` |
| What earlier ops of the campaign found, as advice | `lessons.md` |
| The same facts for a program | `facts.json` |

## Provenance

| What | Value |
|---|---|
| tt-metal checkout | llk_helper_library@33bde179040e (2026-11-24), read only |
| tt_ops_code_gen | 240db7457123 (2026-11-24) |
| Reference implementation | pytorch/pytorch@70d99e998b4955e0049d13a98d77ae1b14db1f45 (torch 2.11.0), license BSD-3-Clause (given: GitHub names none) |
| Paper | none: a library function |
| Model configuration | the lead's contract (`sources/local/contract.md:24`) and case list (`sources/local/scenarios.json`), origin "lead of the library: traced production runs", license unknown |
| Oracle run | python_env of the tt-metal checkout, torch 2.11.0+cpu, 2026-11-25: 22 of 22 checks pass, 2 precision rows informational |
| Written by | explorator skill; bundled references verified at tt-metal 204116418f5a, tt_ops_code_gen b93830bfbd5b |

## Not read on purpose

An existing form of this op is in the checkout: 126 paths (kinds: C++ op, Python op, tt-train,
tests, model code, fused kernel header) matched a name this op goes by. None was opened and
nothing in this bundle comes from them; the names and the paths are in the handoff reply, not
here.

## Decisions and assumptions

| Id | Decision or assumption | Why | Alternative considered |
|---|---|---|---|
| D1 | Name g6_clone, kind op; one call computes a copy of the input into a new tensor. Source: lead | `sources/local/contract.md:8-10` | none: decided |
| D2 | Boundary: one call is one device program, never merged with a neighbour, never split. Source: lead | `sources/local/contract.md:17-20` | none: decided |
| D3 | Tensors: 29 cases, one input each; the output has the input's shape, dtype, layout and DRAM-interleaved placement; five (dtype, layout) pairs; ranks 1 to 4. Source: lead | `sources/local/contract.md:26-39`, `sources/local/scenarios.json:7-32` and every other case (oracle probe "call matrix") | none: decided |
| D4 | The reference's `memory_format` is not a parameter | a device tensor has no strides to preserve; no call passes it | a keyword with one accepted value: an untestable parameter |
| D5 | No `dtype` or `memory_config` keyword: the output takes the input's dtype and placement | no production call passes either (`sources/local/contract.md:39`); a dtype change is a conversion, not a copy; Q1 asks the lead | both keywords, accepting only the input's values: parameters the suite never exercises |
| D6 | The oracle keeps the input dtype (no float32 round trip) | int32 and uint32 above 2^24 are not exact in float32 (oracle probe) | the suite's usual float32 form: wrong for the integer cases |
| D7 | Phase 0: float32, TILE, DRAM interleaved, every rank and alignment | the model's largest TILE calls and the flagged cases p1, p2 are float32 TILE; an engine path shows first in float32 | bfloat16 TILE (14 cases): defers the two flagged float32 cases |
| D8 | Registry: `dtype` and `layout` projected from scenario dicts; no `alignment`, `rank` or `memory_layout` axis | BB-22 limits the suite to production cases; a copy has no kernel path that differs by alignment or rank | iterated `dtype`: builds non-production pairs such as int32 TILE |
| D9 | Gate: bitwise exact for every dtype; the suite compares bit patterns, NaN payloads and -0.0 included. Source: lead for exactness | `sources/local/contract.md:73` | value comparison: NaN would never equal itself |
| D10 | Oracle evidence E1 against `torch.clone` of the installed torch, the pinned commit | the contract names the PyTorch function as the reference (`sources/local/contract.md:72`) | none |
| D11 | Perf focus on the flagged cases p1 to p4. Source: lead | `sources/local/contract.md:50-57` | none: decided |
| D12 | The cores rule stands in the spec | no golden case fails on a single core (risks.md R4), and p4 is one tile row of 1940 tiles | silence: a single-core op would pass every cell |
| D13 | Advice: no bullet kept in the spec; of the lessons RB-036 and RB-053 kept | the filter of the lead's hand-in; reasons in the change log | RB-036 as a spec bullet: it repeats BB-17 |
| D14 | The PyTorch reference is named by its PyTorch name, which is also a name probed for existing implementations | the contract names it (`sources/local/contract.md:10`); no TT-Metal path, namespace or symbol is written | renaming the reference: hides what the contract says |
| D15 | The bundle lies in `bundle/g6_clone/` under the ticket's output directory | the bundle script requires the directory to end in the op name | none |

## Open questions

| Id | Question | Blocks | Owner |
|---|---|---|---|
| Q1 | The contract's sentence "with the dtype and memory placement the call asks for" (`sources/local/contract.md:10`) against "Other arguments of the call: none" (`sources/local/contract.md:39`): the spec has no `dtype` or `memory_config` keyword. Confirm, or name the keywords the model's call needs | nothing | lead |
| Q2 | Projecting `dtype` from a scenario dict (so that only the five production pairs are built) has no precedent in a shipped suite: accept it, or iterate `dtype` and build non-production cells | golden suite | golden-tests |
| Q3 | Can the suite's host-to-device path carry NaN payloads and subnormals unchanged, so that the random-bit-pattern generator may include them | nothing | golden-tests |

## How to re-verify

From the bundle directory, with EXPLORATOR set to the directory of the explorator skill and
TT_METAL_HOME to the tt-metal root:

    python3 -B "$EXPLORATOR"/scripts/fetch_source.py verify --out sources
    python3 -B oracle/check_oracle.py
    python3 -B "$EXPLORATOR"/scripts/lint_spec.py spec/g6_clone.txt --rules sources/local/rules.md --tt-metal "$TT_METAL_HOME"
    python3 -B "$EXPLORATOR"/scripts/check_dossier.py . --tt-metal "$TT_METAL_HOME"

## Change log

- 2026-11-25: created from the request above, with the lead's hand-in (contract revision 2,
  rules, advice, lessons) and the case list pinned.
- 2026-11-25: advice and lessons filtered. Kept in lessons.md: RB-036 (traced multi-core
  program, tiny cases), RB-053 (unequal page counts per core). Dropped: RB-038 and RB-059 (no
  large or fused kernel text in a copy, INFERENCE), RB-061 (no evidence that this program starts
  late), RB-001 and RB-002 (boundary and producer layout are the lead's contract), RB-010 (no
  input shared by several cores), RB-014, RB-005, RB-012 (no state), RB-027 (no arithmetic; an
  engine path would break bit identity). No advice bullet stays in the spec: RB-036 repeats
  BB-17, RB-053 is a measurement method, the other four do not match.
- 2026-11-25: cold reads by independent subagents. Skeptic: citations, oracle rerun (22 of 22),
  tile figures and case counts confirmed; fixed the d_0 range in math.md (1 to 32), a case range
  in reference.md, and the logical-row transfers of a partial tile (two 32 B face pieces per
  tile, risks.md and tensix.md). Planner: named the dtype of the partial-tile shapes, stated that
  no dimension grows without bound, stated that the suite's tolerances are exact; added a rule
  against gating production cases by axis, tagger or EXCLUSIONS. Cheater: added a rule that the
  entry point dispatches its own program (no delegation to an existing device op), that the input
  is unchanged by the call, that an L1-bound refusal of a golden shape is a defect; the brief now
  asserts output dtype, layout and placement on every cell, reads the input back in "new storage",
  keeps the NaN payload case as an expected failure instead of dropping it, and marks the
  random-bit-pattern draw [spec]. Kept as is: the cores rule in its sanctioned form (the cheater
  asked for a stronger one; the spec may not name a core count).
