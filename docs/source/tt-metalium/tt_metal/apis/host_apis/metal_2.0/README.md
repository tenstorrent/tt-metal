# Metal 2.0 Op-Porting Recipe

Documentation for porting TTNN ops to the Metal 2.0 host API. Find your role below and
follow the route.

## Start here — who reads what

| You are… | Start with | Then |
|---|---|---|
| **An AI porter** (Claude) porting an op | [`ai/audit/metal2.md`](ai/audit/metal2.md) — host/spec port-readiness audit | [`ai/port/metal2.md`](ai/port/metal2.md) — do the port |
| **An AI porter / reviewer** auditing **kernel CB→DFB** readiness | [`ai/audit/cb_dfb_kernel_audit_helper.md`](ai/audit/cb_dfb_kernel_audit_helper.md) — **standalone device-side CB audit** ([How to use](ai/audit/cb_dfb_kernel_audit_helper.md#how-to-use-this-doc)) | Optional: cross-ref the host audit if doing a full op port |
| **A human** getting oriented | [`human/user_orientation.md`](human/user_orientation.md) | [`human/CB-to-DFB-flowchart.svg`](human/CB-to-DFB-flowchart.svg) — the CB→DFB decision flowchart |
| **Anyone** who needs the concepts / API reference | [`ai/shared/migration_guide.md`](ai/shared/migration_guide.md) | — |

## Directory map

The `ai/` recipes are organized by **porting phase** — audit → port → (post-port,
forthcoming) — over a `shared/` pool of reference docs the phases draw on.

- **[`ai/`](ai/)** — the AI porter / auditor working procedures.
  - **[`ai/audit/`](ai/audit/)** — port-readiness audits (the per-op go/no-go gates).
    - `metal2.md` — the Metal 2.0 **host/spec** audit.
    - `cb_dfb_kernel_audit_helper.md` — **standalone device-side** CB→DFB kernel audit
      (per-op CB portability report; no host-audit prerequisite). Destined to become an
      offshoot of the forthcoming Quasar audit.
    - `temp_cb_dfb_reference_info.md` — *temporary* frozen snapshot that keeps the
      device-side CB audit's links pointed at stable content during the CB→DFB pivot;
      to be deleted once the kernel audit is self-contained.
  - **[`ai/port/`](ai/port/)** — the port procedures.
    - `metal2.md` — the Metal 2.0 port procedure itself.
  - **[`ai/shared/`](ai/shared/)** — cross-cutting reference, drawn on by both the audits and the recipes.
    - `migration_guide.md` — the shared conceptual + API reference (CB→DFB, first-class
      bindings, the spec / run-params split). Currently AI-facing; a trimmed human version is planned.
    - `port_patterns.md` — patterns & anti-patterns catalog (referenced throughout the recipe and audit).
    - `ttnn_factory.md` — TTNN factory-concept specifics for the port.
    - `workspace_setup.md` — environment / workspace setup.
    - `cb_dfb_api_whitelist.md` — the CB→DFB API-swap whitelist.
- **[`human/`](human/)** — human-facing material.
  - `user_orientation.md` — orientation for people.
  - `CB-to-DFB-flowchart.svg` — the CB→DFB classification flowchart.
- **`analyses/`** — reference tables (data, not procedures): op port-readiness, the
  TensorAccessor 3rd-arg taxonomy, pre-port-issue sweep results. These are *decaying
  snapshots* — each is dated and stamped with its authoritative source; re-verify before
  relying. *(Populated as curations and sweeps land.)*
