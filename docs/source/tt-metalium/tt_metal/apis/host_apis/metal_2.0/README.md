# Metal 2.0 Op-Porting Recipe

Documentation for porting TTNN ops to the Metal 2.0 host API. Find your role below and
follow the route.

## Start here — who reads what

| You are… | Start with | Then |
|---|---|---|
| **An AI porter** (Claude) porting an op | [`ai/port_op_to_metal2_audit.md`](ai/port_op_to_metal2_audit.md) — run the port-readiness audit **first** (the go/no-go gate) | [`ai/port_op_to_metal2_recipe.md`](ai/port_op_to_metal2_recipe.md) — do the port |
| **A human** getting oriented | [`human/user_orientation.md`](human/user_orientation.md) | [`human/CB-to-DFB-flowchart.svg`](human/CB-to-DFB-flowchart.svg) — the CB→DFB decision flowchart |
| **Anyone** who needs the concepts / API reference | [`metal2_migration_guide.md`](metal2_migration_guide.md) | — |

## Directory map

- **[`metal2_migration_guide.md`](metal2_migration_guide.md)** (root) — the shared conceptual +
  API reference (CB→DFB, first-class bindings, the spec / run-params split). Lives at the root
  because both humans and AI porters draw on it.
- **[`human/`](human/)** — human-facing material.
  - `user_orientation.md` — orientation for people.
  - `CB-to-DFB-flowchart.svg` — the CB→DFB classification flowchart.
- **[`ai/`](ai/)** — the AI porter / auditor recipes (the working procedures).
  - `port_op_to_metal2_audit.md` — port-readiness audit: the per-op go/no-go gate. **Run first.**
  - `port_op_to_metal2_recipe.md` — the port procedure itself.
  - `metal2_port_patterns.md` — patterns & anti-patterns catalog (referenced throughout the recipe and audit).
  - `port_op_to_metal2_ttnn_factory.md` — TTNN factory-concept specifics for the port.
  - `metal2_workspace_setup.md` — environment / workspace setup.
- **`analyses/`** — reference tables (data, not procedures): op port-readiness, the
  TensorAccessor 3rd-arg taxonomy, pre-port-issue sweep results. These are *decaying
  snapshots* — each is dated and stamped with its authoritative source; re-verify before
  relying. *(Populated as curations and sweeps land.)*
