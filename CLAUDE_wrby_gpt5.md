## CLAUDE.md — Implementation Brief for Claude Code (TTTv2)

Read this whole file first; then execute the milestones in order. Favor small PRs that finish a milestone slice end-to-end (code + tests).

### Grounding
- Primary spec: `models/tt_transformers_v2_prototype_2/docs/TTTv2_design.md`.
- Treat core as a Python library: building blocks + patterns + testing utilities. Models live outside core and pin to a version.
- Non-goals: migrating all TTTv1 models; owning a model zoo inside core; inventing a new graph DSL.

### Operating Mode
- Be autonomous. Propose diffs; implement; write tests; run them.
- Keep the public API strict; expose only what is documented in `__init__.py` files.
- Prefer pure, stateless functions; where state is needed (caches), keep it explicit.
- Optimize for readability and maintainability. No premature micro-optimizations without tests.

### Repository Layout (target)
```
tt_transformers_v2/
  src/
    tt_transformers_v2/
      building_blocks/      # attention, ffn, normalization, embeddings
      patterns/             # decoder_layer, encoder_layer, CausalLM
      testing/              # TestSuite utilities
      __init__.py
    __init__.py
  tests/
    building_blocks/
    patterns/
    testing/
models/
  adapters/                 # reference adapters (optional, non-core)
  llama3/ qwen/ mistral/    # reference examples (minimal)
```

If dirs already exist under different names, adapt, but preserve the design intent above.

### API Conventions (enforce)
- Each building block defines:
  - `Spec` dataclass: shapes, dims, options
  - `ImplConfig` dataclass: TTNN op choices, dtypes, tiling, kernel flags
  - `get_default_impl_config(device, mode)` → sane defaults
  - `forward(...)` pure function; separate prefill/decode when applicable
- Patterns (e.g., `CausalLM`) compose blocks and surface prefill/decode.
- Public symbols are re-exported from top-level `tt_transformers_v2/__init__.py` only.

### Testing Strategy (gate merges)
- Unit tests for building blocks (correctness + perf budgets as markers)
- Composition tests for `patterns/` with real dims
- A small fluent `TestSuite` utility for capturing inputs from real forwards
- Fail fast on accuracy regressions (e.g., `atol`/`rtol` or `pcc`), and mark perf expectations as soft caps with pytest markers

### Acceptance Criteria (per milestone)
- Code compiles; tests pass locally
- Public API stable and minimal
- Docstrings explain non-obvious constraints and TTNN-specific caveats
- No dead code; no broad try/except; clear early returns; explicit types at boundaries

---

## Milestone 0 — Bootstrap (skeleton + CI hooks)
Deliverables:
- Create package skeleton under `tt_transformers_v2/src/tt_transformers_v2` with `building_blocks/`, `patterns/`, `testing/`, `__init__.py` files
- Add placeholder tests in `tests/` that import modules
- Add a minimal `pytest` config if missing

Commands to run:
```bash
pytest -q || true
```

---

## Milestone 1 — Building Blocks API stubs
Deliverables (minimal but typed):
- `building_blocks/attention.py`
- `building_blocks/ffn.py`
- `building_blocks/normalization.py`
- `building_blocks/embeddings.py`

Each file must include: `Spec`, `ImplConfig`, `get_default_impl_config`, `forward` (and `prefill_forward`/`decode_forward` if split), with NotImplemented internals plus shape checks and docstrings.

Tests:
- Import tests that instantiate specs/configs and validate basic shape contracts.

---

## Milestone 2 — Patterns (decoder/encoder + CausalLM)
Deliverables:
- `patterns/decoder_layer.py`, `patterns/encoder_layer.py`, `patterns/causal_lm.py`
- Compose blocks; wire `prefill_forward`/`decode_forward` and explicit caches

Tests:
- Composition tests with small dims; verify forward shape & a smoke numerical check against a trivial reference when feasible

---

## Milestone 3 — Testing utilities
Deliverables:
- `testing/suite.py` exposing a fluent `TestSuite`
- Helpers to compare against a reference function and assert `pcc`/`atol`/`rtol` thresholds

Tests:
- Unit tests for the suite itself; demonstrate both explicit-input and trace-capture modes

---

## Milestone 4 — Reference adapters (non-core, optional)
Deliverables:
- A minimal `models/adapters/model_factory.py` that shows how an external repo would construct a model using `patterns.CausalLM`
- One tiny reference (e.g., toy-llama-like spec) to prove the path end-to-end

Tests:
- Smoke test that builds the toy model and runs prefill/decode for a few steps

---

## Performance & Correctness Guards
- Where possible, include perf expectations as pytest markers, e.g., `@pytest.mark.perf(latency_ms=5.0, p=0.9)`
- Correctness thresholds: prefer `pcc ≥ 0.99` against a golden; fall back to `atol/rtol` where golden lacks stochastic parity

---

## Coding Standards (quick)
- Type annotate public functions and dataclasses
- No broad `except:`; no hidden global state
- Early returns > deep nesting; tiny helpers over giant functions
- Keep imports local if they are heavy or optional; otherwise keep them top-level and explicit

---

## Workflow Checklist (repeat per PR)
1) Read/update the design doc subsection if behavior changes: `models/tt_transformers_v2_prototype_2/docs/TTTv2_design.md`
2) Edit code + tests for a single milestone slice
3) Run tests locally: `pytest -q`
4) Ensure public API only exposes intended symbols
5) Add/adjust docstrings; include rationale for non-obvious TTNN choices

---

## Kickoff Tasks (execute now)
- [ ] Create the package skeleton and empty test files matching the layout above
- [ ] Add `__init__.py` files to define the strict public API
- [ ] Add stub modules for attention/ffn/norm/embeddings with typed signatures and `NotImplementedError`
- [ ] Add `patterns/causal_lm.py` with method stubs (`prefill_forward`, `decode_forward`)
- [ ] Add `testing/suite.py` skeleton and one simple test that proves import + run
- [ ] Wire a toy spec in `tests/` to exercise one block end-to-end

When skeleton lands and tests import, proceed to fill implementations incrementally starting with normalization (usually simplest), then embeddings, then FFN, then attention.

---

## Notes
- If `TTNN` or hardware-specific libs are unavailable locally, keep adapters thin and protect imports; write CPU fallback refs for tests.
- The architecture diagram referenced by the design doc may be external; do not block on it.
- Keep model-specific code out of core; put any reference usages under `models/` and mark as non-core.
