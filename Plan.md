# Plan: FSD-driven mapping for `tt-run` / `generate_rank_bindings`

Tracking issue: [#52859](https://github.com/tenstorrent/tt-metal/issues/52859) — *tt-run / generate_rank_bindings: accept FSD argument, map from FSD when available, fall back to PSD discovery.*

Today the auto-mapper flow (`tt-run -m/--mesh-graph-descriptor` + `--hosts`, and Fabric Manager's `generate_rank_bindings`) always maps against a **Physical System Descriptor (PSD)** that is **discovered live** from the cluster at runtime. The goal is to optionally map against the **Factory System Descriptor (FSD)** — the "what the cluster *should* look like" descriptor — and fall back to live PSD discovery when no FSD is supplied or it does not cover the requested topology.

This work is split into two stages. **Stage 1 (PR #53451) is plumbing only** — the FSD path travels end-to-end but is not yet consumed for mapping. **Stage 2 is the consumption** — actually deriving the topology from the FSD. This PR is the shared-library prerequisite for Stage 2 (the runtime-free topology library).

---

## Stage 1 — Plumbing (DONE in PR #53451)

The FSD path is threaded from the `tt-run` CLI down to `generate_rank_bindings` (Phase 1) and the workload (Phase 2), carried by a new RTOption and its environment variable. Nothing consumes it for mapping yet — `generate_rank_bindings` still runs live PSD discovery and only logs that an FSD was supplied.

### RTOption (`tt_metal/llrt/rtoptions.{hpp,cpp}`)
- New member `std::string factory_system_descriptor_path` with accessors:
  `has_factory_system_descriptor_path()`, `get_factory_system_descriptor_path()`, `set_factory_system_descriptor_path()`.
- New `EnvVarID::TT_METAL_FACTORY_SYSTEM_DESCRIPTOR_PATH` + a `HandleEnvVar` case that stores the value.
  The env var name matches the `--factory-descriptor-path` convention used by the validation tooling.

### `tt-run` (`ttnn/ttnn/distributed/ttrun.py`)
- New `--factory-system-descriptor <path>` click option (both legacy and new mode).
- Managed the **same way as `--mock-cluster-rank-binding`**: set explicitly per-rank in `get_rank_environment()` as
  `TT_METAL_FACTORY_SYSTEM_DESCRIPTOR_PATH`, and added to `ENV_BLOCKLIST` so a stray parent-env value cannot override the CLI value.
- Carried on `TTRunConfig.factory_system_descriptor_path`, applied to every rank.
- Propagated to **Phase 1** (`build_generate_rank_bindings_mpi_cmd` / `run_phase1_generate_rank_bindings`) and **Phase 2** (`legacy_flow`).
- Folded into the **Phase 1 cache fingerprint** (`compute_phase1_cache_fingerprint_full`) so a different/added FSD invalidates the cache.

### `generate_rank_bindings` (`tools/scaleout/src/generate_rank_bindings.cpp`)
- Reads the FSD path from RTOptions (`get_factory_system_descriptor_path()`), populated from the propagated env var —
  **no CLI argument** on this binary (deliberate: env/RTOptions is the single source of truth).
- Currently only logs "FSD provided, falling back to live PSD discovery." No behavior change.

### Docs / tests
- `README_ttrun.md`: documented `--factory-system-descriptor` under Additional Options.
- `tests/ttnn/distributed/test_ttrun.py`: unit tests for env-var injection (real + mock cmd), `get_rank_environment`, and fingerprint invalidation.

---

## Stage 2 — Consumption (TODO, follow-up PR(s))

The pivotal substitution point is `run_psd_discovery()` in `generate_rank_bindings.cpp` (line ~55): it returns a
`PhysicalSystemDescriptor psd` that flows into `build_topology_mapping_inputs()` →
`build_physical_multi_mesh_adjacency_graph(psd, pgd, mgd, ...)` → `run_topology_mapping()`. Everything downstream is
keyed on the `PhysicalSystemDescriptor` type, so Stage 2 replaces that `psd` with one derived from the FSD.

### 1. Derive a topology from the FSD
- Reuse `generate_cluster_descriptor_from_fsd()` in `tools/scaleout/factory_system_descriptor/utils.cpp`
  (declared in `utils.hpp`): it text-parses the `FactorySystemDescriptor` proto and writes cluster-descriptor YAML(s)
  (one for single-host; per-host + a rank→descriptor mapping for multi-host), returning the generated path.
- Build a `PhysicalSystemDescriptor` from the FSD-derived cluster descriptor **without** live discovery. Note the existing
  seam: `run_physical_system_discovery(...)` in `tt_metal/fabric/physical_system_discovery.{hpp,cpp}` already takes a
  `bool run_live_discovery = true` — evaluate whether passing `false` (fed with the FSD-derived cluster description) yields
  the right non-live `PhysicalSystemDescriptor`, or whether a dedicated `build_psd_from_fsd()` helper is cleaner.

### 2. Wire it into `generate_rank_bindings::main()`
- When `rtoptions().get_factory_system_descriptor_path()` is non-empty, produce the FSD-derived
  `PhysicalSystemDescriptor` instead of calling `run_psd_discovery()`. Keep the rest of the pipeline unchanged
  (`build_topology_mapping_inputs`, `run_topology_mapping`, `extract_rank_bindings`).

### 3. Coverage check + fallback
- If the FSD does not cover the requested hosts/topology (missing hosts, board-type mismatch, shape mismatch), log a
  clear warning and **fall back to live PSD discovery** — matching the "PSD-based discovery stays the default/fallback"
  requirement. Decide the coverage predicate (hostname set ⊇ requested hosts; board types compatible with the MGD).
- Consider a strict mode later (fail instead of fall back) if pre-flight/CI wants hard guarantees.

### 4. Multi-host / mock interplay
- `generate_rank_bindings` gathers mock cluster descriptor paths across ranks (`gather_mock_cluster_desc_paths`).
  Define precedence when both a mock cluster mapping and an FSD are present (likely: mock wins for mock runs; document it).
- For multi-host FSDs, ensure the per-host cluster-descriptor mapping produced by `generate_cluster_descriptor_from_fsd()`
  lines up with MPI rank → host assignment.

### 5. Tests (Stage 2)
- C++: map from a supplied FSD (golden FSD → expected rank bindings) and verify fallback to live PSD when the FSD is
  absent/incomplete. Reuse fixtures from `tools/tests/scaleout/test_factory_system_descriptor.cpp`.
- Python (`test_ttrun.py`): end-to-end mock-cluster run with `--factory-system-descriptor` produces bindings without
  live discovery; and coverage-miss falls back.

### 6. Docs (Stage 2)
- `README_ttrun.md`, `tt_metal/fabric/MGD_README.md`, and the Automapper Guide: when to use the FSD, the fallback
  semantics, and the "skinny FSD"/offline/CI/pre-flight use cases from the issue's Motivation.

---

## Key references
- `tools/scaleout/src/generate_rank_bindings.cpp` — `run_psd_discovery()`, `build_topology_mapping_inputs()`, `main()`.
- `tools/scaleout/factory_system_descriptor/utils.{hpp,cpp}` — `generate_cluster_descriptor_from_fsd()`.
- `tt_metal/fabric/physical_system_discovery.{hpp,cpp}` — `run_physical_system_discovery(..., run_live_discovery)`.
- `tools/scaleout/validation/run_cluster_validation.cpp` — existing `--factory-descriptor-path` / FSD precedent.
- `tt_metal/llrt/rtoptions.{hpp,cpp}` — the new `TT_METAL_FACTORY_SYSTEM_DESCRIPTOR_PATH` RTOption.
- `ttnn/ttnn/distributed/ttrun.py` — `--factory-system-descriptor` plumbing.
