# Ease-of-Use Refactoring Epic — AI Codegen Infra Fit Analysis

I read the milestones doc and the 4 refactoring epics (tt-metal#33823, tt-llk#911, tt-metal#35739, tt-metal#22907), then pulled all 28 open sub-issues via the GitHub API and read their bodies. #35739 has no sub-issues yet. Below is a ranking by AI‑Codegen‑Infra fit, based on whether the change is (a) mechanical/rule-based, (b) has a crisp spec in the issue, (c) verifiable via existing CI (APC/BPC/LLK tests, pipelines), and (d) doesn't need HW experimentation or a design call first.

## Tier 1 — strong candidates (ship these first)

| Issue | Why it's a fit |
|---|---|
| [tt-metal#34587](https://github.com/tenstorrent/tt-metal/issues/34587) — pack tilize/untilize bools → `PackMode` enum | **Already has a Claude Code plan attached** (`PACK_MODE_REFACTORING_PLAN.md`). Cross-repo mechanical bool→enum refactor — exactly the shape codegen nails. |
| [tt-llk#1148](https://github.com/tenstorrent/tt-llk/issues/1148) — `DataCopyType` enum → `enum class` | Purely mechanical C++ refactor + callsite updates. Low ambiguity, high file count. |
| [tt-metal#23995](https://github.com/tenstorrent/tt-metal/issues/23995) — remove default CB-id values in `llk_math_*unary_*` | Labeled `good-first-issue`. Clear DoD: remove defaults, chase failing callers until APC/BPC green. Loop-on-CI is ideal for an agent. |
| [tt-llk#880](https://github.com/tenstorrent/tt-llk/issues/880) — remove redundant `llk_math_eltwise_binary_init` w/o-operands overload | Drop one overload + rewrite callers. Tight scope. |
| [tt-llk#1036](https://github.com/tenstorrent/tt-llk/issues/1036) — make x‑start/x‑end transient (add `TTI_SETADCXX`) | Clear directive: add instruction to missing inits, remove elsewhere. Good for a grep-and-patch agent. |
| [tt-llk#1161](https://github.com/tenstorrent/tt-llk/issues/1161) — fix y‑/z‑dim in `_llk_unpack_tilize_uninit_` (WH) | Pinpoint one-function fix with exact expected values. |
| [tt-metal#18347](https://github.com/tenstorrent/tt-metal/issues/18347) — doc strings for compute kernel APIs | Doc generation is a canonical codegen task; issue enumerates target files and gaps. |

## Tier 2 — good candidates (need a short spec-up before handoff)

| Issue | Caveat |
|---|---|
| [tt-metal#34495](https://github.com/tenstorrent/tt-metal/issues/34495) — drop `tile_size` param from `llk_unpack_AB_matmul_init` | Some tests expected to break; agent must triage. |
| [tt-metal#34499](https://github.com/tenstorrent/tt-metal/issues/34499) — clean up reconfig calls (`is_tile_dim_reconfig_en`, `to_from_int8`, `llk_pack_mop_config`) | Unused-param/function removal + computed tile size. |
| [tt-metal#16974](https://github.com/tenstorrent/tt-metal/issues/16974) — CBs as compile-time/template params | Large sweep; pair with #13653. |
| [tt-metal#13653](https://github.com/tenstorrent/tt-metal/issues/13653) — `reconfig_data_format` → templates + tests | Well-scoped, needs a new test kernel as in the issue. |
| [tt-metal#40510](https://github.com/tenstorrent/tt-metal/issues/40510) — remove custom `face_r_dim`/`num_faces` from hw_configure/inits | **Reference branch exists** (`ndivnic/experiment_cb_write_face_geometry`) — agent can rebase/port. |
| [tt-llk#989](https://github.com/tenstorrent/tt-llk/issues/989) — fix `pack_reads_per_xy_plane` default/reduce handling | Precise spec but packer semantics require care. |
| [tt-llk#1089](https://github.com/tenstorrent/tt-llk/issues/1089) — separate transpose settings in `llk_unpack_AB` | Mirror the pattern already in `llk_unpack_A`. |
| [tt-metal#35819](https://github.com/tenstorrent/tt-metal/issues/35819) — Quasar uninits | Mostly empty-body templates mirroring BH/WH; needs Quasar build access. |
| [tt-llk#1033](https://github.com/tenstorrent/tt-llk/issues/1033) — tile-size standardization, **Step 1 only** | Step 1 (param removal / `TensorShape` plumbing) is mechanical; Steps 2–3 need human judgment. |

## Tier 3 — not a good fit right now (investigation, HW experiment, or design call)

- [tt-metal#36411](https://github.com/tenstorrent/tt-metal/issues/36411) DST bit 11 — CLAUDE.md says **offloaded to Anil**, just track.
- [tt-llk#951](https://github.com/tenstorrent/tt-llk/issues/951), [#960](https://github.com/tenstorrent/tt-llk/issues/960), [#966](https://github.com/tenstorrent/tt-llk/issues/966) — "check/explain" HW behavior; needs experiments, not codegen.
- [tt-metal#17641](https://github.com/tenstorrent/tt-metal/issues/17641) — cross‑arch refinement with known GS regression.
- [tt-metal#17419](https://github.com/tenstorrent/tt-metal/issues/17419) — open question ("check why…"), not a change spec.
- [tt-metal#37671](https://github.com/tenstorrent/tt-metal/issues/37671) — three alternatives listed; needs a design decision first.
- [tt-llk#1015](https://github.com/tenstorrent/tt-llk/issues/1015) — stride management inside `configure_unpack_AB`; HW reasoning heavy.
- [tt-metal#35020](https://github.com/tenstorrent/tt-metal/issues/35020) — "trigger the pipelines and verify my hunch" — exploratory.
- [tt-metal#26202](https://github.com/tenstorrent/tt-metal/issues/26202), [#28870](https://github.com/tenstorrent/tt-metal/issues/28870), [tt-llk#1287](https://github.com/tenstorrent/tt-llk/issues/1287) — need human design call before any code change.

## Suggested pilot

Start with **#34587** (has a prewritten plan and an explicit "consider using Claude Code" note — lowest-risk way to prove the loop end-to-end) and **#23995** (good-first-issue with a trivial DoD — great for validating the agent's CI-iteration loop on APC/BPC). Both give measurable signal in a few days before committing agents to the Tier‑2 sweep.
