# Mistral 4 MoE routing capture

## What it is

`test_dispatch_combine_perf` replays a **recorded** MoE routing pattern rather than random indices,
because Dispatch/Combine cost is a function of how tokens spread across experts — even routing
spreads the work, skewed routing makes one chip a hotspot. It is worth measuring: Dispatch+Combine is
**26% of a layer and 3.1x the cost of the expert math it feeds**, the largest non-math term.

The capture is that recording: for each MoE layer, which **4 of 128** experts each of 5,120 tokens
picked, measured on device on the real 56,320-token trace. The test replays one Galaxy column's worth
(8 chips x 640 tokens x 4 picks) on an 8-chip LoudBox standing in for that column.

DeepSeek V3 / Kimi K2.6 / GLM 5.2 got theirs in #51426. **Mistral 4 had none, and there was no
generator in the tree** — that patch was never committed, which is why this had to be rebuilt.

## The file

`expert_routing_mistral4.safetensors` — 2.9 MB, 35 keys `expert_ids_layer_0..34`, int32,
8 x 640 x 4 **raw global** expert ids (the per-column remap happens in `load_captured_routing`).

- Distributed as a PR attachment, as the other three models' captures were in #51426
- Verified: `load_captured_routing(layer=18, col=2, model="mistral4")` -> `indices=(8,640,4)`,
  in-col share 42.9%, all 128 experts present
- Hot `(layer, col)` picks measured from it — hottest **18/2 (42.9%)**, **16/2 (41.3%)**;
  ~uniform **19/2 (25.0%)**, **15/1 (25.0%)** — the 2-hottest + 2-average pattern
  `test_dispatch_combine_perf` selects, computed straight from the capture with no extra tooling.

## How it was generated

This branch adds `tt/moe/routing_capture.py` plus
3 lines in `tt_moe.py`. Off unless `TT_DS_DUMP_ROUTING` is set.

```bash
cd <tt-metal> && source python_env/bin/activate
export TT_METAL_HOME=$PWD PYTHONPATH=$PWD MESH_DEVICE=TG LOGURU_LEVEL=INFO
export MISTRAL4_HF_MODEL=/mnt/models/blaze/mistralai/Mistral-Small-4-119B-2603
export TT_MISTRAL4_PREFILL_TTNN_CACHE=/mnt/models/blaze/mistralai/Mistral-Small-4-Cache/CI
export TT_DS_DUMP_ROUTING=/abs/path/expert_routing_mistral4.safetensors
python3 -m pytest models/demos/deepseek_v3_d_p/tests/test_prefill_transformer_chunked.py::test_mistral4_prefill_transformer_chunked_padded \
  -k "mistral4 and torus-xy-8x4 and L36 and full55k and notrace" -vvv --tb=short
```

11m39s on `bh-glx-120-b03u02`, whole galaxy idle. Look for `locked chunk 4` then `wrote 35 layer keys`.

## Four traps, each of which yields a plausible, non-erroring, WRONG capture

1. **`..._chunked_no_pcc` uses synthetic in-vocab ids by design** — its routing is meaningless. Use
   the padded golden-trace leg above.
2. **Every chunk is padded to 5,120 on device**, so width cannot tell real from padded: a 1,024-token
   chunk still reads 640 tokens/chip. Only `actual_isl` can. Our capture is chunk 4 — fully real.
3. **Identify the SP shards by mesh coordinate, never by value-deduplicating the 32 device tensors** —
   two chips can route identically (an all-padding shard does). Equality only *verifies* the TP
   columns of a row are replicas.
4. **The padded leg is built `kv_only_last_layer=True`**, so layer 35's MoE never runs. 35 layers, not 36.

## Where the file belongs

`$DEEPSEEK_V3_TRACE_DIR/code_debug_5k_chunked/expert_routing_mistral4.safetensors`

| environment | root | state |
|---|---|---|
| exabox | `/mnt/models/deepseek-prefill-cache` | dir absent; parent is writable, so we can create it |
| metal CI | `/mnt/MLPerf/deepseek-prefill-cache` | not mounted on exabox; staged separately |

Same relative layout, different root per environment (see #45521 comment). The two roots are staged
**per artifact, not mirrored**: `test_mla_output` exists under both, `code_debug_5k_chunked` only on CI.

## Wiring it up

1. **Attach the safetensors**, as #51426 did. That distributes it to people; CI still cannot read a
   PR attachment, so note the canonical path in the body.
2. **Add `"mistral4"` to the allow-list at `init_helpers.py:779`** so it resolves the default path like
   the other three. Do *not* rely on `TT_DS_USE_CAPTURED_INDICES`: it is inherited process-wide, and
   one invocation covers every model, so a mistral4 path would fail the filename check
   (`init_helpers.py:767`) for dsv3/kimi26/glm52.
3. **The row cannot be gated yet.** `test_dispatch_combine_perf` is commented out in
   `blackhole_e2e_tests.yaml` pending **#47287** — ~5% cross-runner skew against its 3%/4.5% margins.
   Until that is resolved a mistral4 case is record-only, as it is for the other three models.
