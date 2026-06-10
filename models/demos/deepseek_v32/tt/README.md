# DeepSeek V3.2 — multichip TT model

V3.2 = V3 with one architectural change: DeepSeek Sparse Attention (DSA) inside the
MLA layer (a lightning indexer selects top-k tokens; SDPA attends only over those).
Everything else is identical, so this package reuses `models/demos/deepseek_v3_d_p`
as a library and only carries the MLA delta.

## What lives here vs. what is imported

| Path | Status |
|---|---|
| `mla/` | **v32 code** — `ttMLA` subclass of v3 `ttMLA`; all DSA work lands here |
| `tt_prefill_block.py` | copy of v3, only the `ttMLA` import changed |
| `tt_prefill_transformer.py` | copy of v3, only the `TtPrefillBlock` import changed |
| norms, FFN, MoE, CCL, embedding, LM head, rope, runners, test fixtures | imported from `deepseek_v3_d_p` |

Do not edit the two copied files beyond their import lines — sync them from v3
instead (`diff` against the originals should show only imports + header note).

## FUTURE WORK: remove the copied files

The copies exist only because v3 hardcodes its composition:

- `deepseek_v3_d_p/tt/tt_prefill_block.py` hardcodes `ttMLA`
  (ctor + `check_cache_complete` / `build_ttnn_cache` / `kv_cache_to_host`).
- `deepseek_v3_d_p/tt/tt_prefill_transformer.py` hardcodes `TtPrefillBlock`.

Propose to the v3 team: an `mla_class=ttMLA` parameter on `TtPrefillBlock` and a
`block_class=TtPrefillBlock` parameter on `TtPrefillTransformer` (defaulted, so v3
is untouched). Then both copies here are deleted and v32 passes its MLA class.

## Tests

`tests/test_mla.py` runs the v3 e2e MLA harness with the v32 class swapped in
(passthrough today, so it must match v3 PCC). V3.2 checkpoints with indexer
weights are not wired into the conftest yet — runs use the v3 variant.

Reference implementations: `../reference_cpu` (PyTorch truth), `../reference_tt_single_chip`
(single-chip ttnn port, indexer included), specs in `../reference_tt_single_chip/spec*.md`.
