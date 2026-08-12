# Metal 2.0 Op-Porting Recipe

Documentation for porting TTNN ops to the Metal 2.0 host API. Find your role below and
follow the route.

## Start here — who reads what

| You are… | Start with | Then |
|---|---|---|
| **A human** getting oriented | [`human/READ_ME_FIRST.md`](human/READ_ME_FIRST.md) | [`human/intro_to_metal_2.0.md`](human/intro_to_metal_2.0.md)|
| **An AI porter** (Claude) porting an op | [`ai/audit/metal2_audit.md`](ai/audit/metal2_audit.md) — host/spec port-readiness audit | [`ai/port/metal2_port.md`](ai/port/metal2_port.md) — do the port |
| **An AI** applying one post-port fix to an already-ported op | [`ai/post_port/pass_procedure.md`](ai/post_port/pass_procedure.md) — the shared pass procedure | the one fix recipe your invoker named, under [`ai/post_port/`](ai/post_port/) |
| **An AI porter** uplifting a Metal 2.0 op to **Quasar** | [`ai/audit/quasar_audit.md`](ai/audit/quasar_audit.md) — Quasar-uplift feasibility audit | uplift recipes forthcoming |
| **An AI** who needs the concepts / API reference | [`ai/shared/migration_guide.md`](ai/shared/migration_guide.md) | — |
| Auditing kernel CB/DFB usage for **Quasar-uplift** readiness <br> (separate from Metal 2.0 porting)| [`ai/audit/cb_dfb_quasar_audit_helper.md`](ai/audit/cb_dfb_quasar_audit_helper.md) — **standalone device-side CB/DFB audit** ([How to use](ai/audit/cb_dfb_quasar_audit_helper.md#how-to-use-this-doc)) | Optional: cross-ref the host audit if doing a full op port |

## Directory map

The `ai/` recipes are organized by **porting phase** — audit → port → post-port — over a
`shared/` pool of reference docs the phases draw on.

- **[`ai/`](ai/)** — the AI porter / auditor working procedures.
  - **[`ai/audit/`](ai/audit/)** — port-readiness audits (the per-op go/no-go gates).
    - `metal2_audit.md` — the Metal 2.0 **host/spec** feasibility audit (pre-port, WH/BH).
    - `quasar_audit.md` — the **Quasar-uplift** feasibility audit: run *after* the WH/BH Metal 2.0
      port, to gate uplifting the op to Quasar. A young scaffold; delegates the CB→DFB
      analysis to the kernel audit below.
    - `cb_dfb_quasar_audit_helper.md` — **standalone device-side** CB/DFB kernel audit
      (per-op buffer portability report for Quasar; classifies CBs **or** already-ported DFBs;
      no host-audit prerequisite). The analysis `quasar_audit.md` delegates to.
  - **[`ai/port/`](ai/port/)** — the port procedures.
    - `metal2_port.md` — the Metal 2.0 port procedure itself.
  - **[`ai/post_port/`](ai/post_port/)** — small targeted fixes applied *after* a working port, one
    per pass.
    - `pass_procedure.md` — the shared procedure every pass follows (the *how*); the individual fix
      recipes in the two subdirectories are the *what*. Also defines the style / semantic split and
      the batch contract.
    - **[`ai/post_port/style/`](ai/post_port/style/)** — expressive changes: what the program *means*
      is untouched, and the op's tests are a real check on the result. Batchable.
    - **[`ai/post_port/semantic/`](ai/post_port/semantic/)** — changes to how the program *runs*, where
      whether behaviour is preserved rests on reasoning the tests cannot fully police. Run one at a
      time, with the launching human reachable; not batched.
  - **[`ai/shared/`](ai/shared/)** — cross-cutting reference, drawn on by both the audits and the recipes.
    - `migration_guide.md` — the shared conceptual + API reference (CB→DFB, first-class
      bindings, the spec / run-params split). Currently AI-facing; a trimmed human version is planned.
    - `port_patterns.md` — patterns & anti-patterns catalog (referenced throughout the recipe and audit).
    - `ttnn_factory.md` — TTNN factory-concept specifics for the port.
    - `workspace_setup.md` — environment / workspace setup.
    - `cb_dfb_api_whitelist.md` — the CB→DFB API-swap whitelist.
- **[`human/`](human/)** — human-facing material.
  - `READ_ME_FIRST.md` — orientation for people.
  - `intro_to_metal_2.0.md` - Metal 2.0 overview
  - `CB-to-DFB-flowchart.svg` — the CB→DFB classification flowchart.
  - `Porting-Strategy.png` — the three-step porting/uplift strategy figure (used by `READ_ME_FIRST.md`).
- **`analyses/`** — reference tables (data, not procedures): op port-readiness, the
  TensorAccessor 3rd-arg taxonomy, pre-port-issue sweep results. These are *decaying
  snapshots* — each is dated and stamped with its authoritative source; re-verify before
  relying. *(Populated as curations and sweeps land.)*
