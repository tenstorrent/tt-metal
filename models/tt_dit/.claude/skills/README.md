# TT-DiT model skills

Four skills covering a diffusion model from port to tuned device time on a
Tenstorrent mesh. Each writes state to a durable journal before advancing, so
any of them can be resumed cold or run unattended.

| Skill | Answers | Component state | Never |
|---|---|---|---|
| `tt-dit-add-model` | Does it produce the right numbers? | Not written, or not green yet | Optimizes |
| `tt-dit-benchmark-profile` | Where does the time actually go? | Green; needs numbers or a profile read | Changes code |
| `tt-dit-performance` | How do we make it faster, provably? | Green and profiled; too slow | Guesses without a profile |
| `tt-dit-kernel-research` | Does the op already exist, and can it be made to fit? | A lever needs an op that may not exist | Writes a kernel before checking |
| **`tt-dit-loop`** | How do we drive this to the goal over many sessions? | Any — it is the **outer loop** invoking the four above | Does the work itself |

`tt-dit-loop` is scoped by *duration*, not phase: multi-round, unattended,
cross-session. A single-phase question goes straight to one of the four.

**Where the vocabulary overlaps, these are separated by component state, not
topic.** PCC, shapes, conv3d, meshes and collectives appear in `add-model`,
`benchmark-profile` and `performance` alike, so topic cannot route between them —
"isn't green yet" vs "was green and got slow" vs "was green and broke under
tracing" can, and each of those three descriptions carries that boundary plus a
pointer to the sibling it defers to.

`kernel-research` is the exception and shows the rule: its vocabulary (nanobind,
config knob, fused kernel, does this op exist) is distinctive enough that topic
alone routes it correctly. Measured — it needed no description tuning at all.

If you add a fifth skill: distinctive vocabulary is enough on its own; if you
overlap an existing skill's terms, add a state boundary and name the sibling.

## Shared

| File | Read when |
|---|---|
| **`shared/device-hangs.md`** | **Before the first device run of any session.** Timeouts, watcher, tt-triage/exalens, recovery |
| `shared/known-issues.md` | Any unexplained hang, allocation failure or precision surprise |
| `shared/reference-models.md` | Before writing any new layer |
| `shared/parallelism.md` | Choosing how to spread work across the mesh |
| `shared/journal-protocol.md` | Recording that a measurement contradicted the plan |

Two lookup tables carry most of the discovery value:
`tt-dit-benchmark-profile/existing-fast-paths.md` (which fused ttnn op already
covers this pattern) and `shared/reference-models.md` (which model already
solved this problem). Check both before writing anything.

## Two rules that govern everything

**Every device run is timeout-gated, and every kill is followed by a reset.** A
hang leaves the device dirty and makes the *next* run fail somewhere unrelated,
which reads as a new bug. `shared/device-hangs.md`.

**Correctness before performance.** `tt-dit-add-model` gates on quality;
`tt-dit-performance` re-checks that gate every iteration and aborts on
regression.

## Optimization order

Fixed, because getting it wrong is how loops stall:

```
dtype matches the reference (a contract, not a lever)
  1. parallelism  →  2. kernel research  →  3. layout round-trips
  4. math fidelity  →  5. fusion/folding  →  6. blocking sweeps  →  7. trace
```

Trace is last precisely because it is a guaranteed win where it applies — it
survives every other change, and chasing it first ships a model that dispatches
efficiently and computes slowly.

## Measurement contract

Warm device time per step, at a stated mesh shape and input shape. Weight upload
is one-time construction cost and is never counted. A number without its mesh
shape, input shape and warm-window method is not a measurement.
