# Release-notes layer map (reference)

Quick reference for how `release_notes_by_layer.py` assigns a PR to a section,
based on the paths of its changed files:

- LLK — `tt_metal/tt-llk/`, `tt_metal/hw/ckernels/`
- Metalium — other `tt_metal/`
- TT-NN — `ttnn/`
- Infrastructure & CI — `.github/`, `infra/`, `scripts/`

A PR that touches multiple layers appears under each.
