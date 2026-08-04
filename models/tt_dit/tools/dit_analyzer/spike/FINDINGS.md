# Dry-run spike: findings

**Verdict: the design works.** The real `LTXTransformerBlock.forward` runs under a
metadata-only `ttnn`, on a laptop, with no device, no checkpoint and (here) not
even torch installed — and the analyzer finds the same 6 provable duplicate
gathers as the hand-written oracle, byte for byte.

Run it:

```bash
python3 models/tt_dit/tools/dit_analyzer/spike/run_ltx_block.py            # BH 4x8, Ring
python3 models/tt_dit/tools/dit_analyzer/spike/run_ltx_block.py --linear   # BH 2x4, Linear
```

## Results against the acceptance criteria

| | BH 4x8 (Ring) | BH 2x4 (Linear) |
|---|---|---|
| 1. real forward completes | yes — 212 nodes, 341 symbols | yes — 206 nodes, 343 symbols |
| 2. per-device shapes / branches | no analyzer diagnostics; shape-dependent branch at `attention_ltx.py:483` matches | same |
| 3. collectives vs `examples/ltx.py` | **31 vs 31, identical multiset** of (op, mesh axis, gathered extent, logical shape) | **25 vs 25, identical** |
| 4. findings vs oracle | **6 `duplicate_gather`, all provable**, 128.7 GiB/forward — byte-identical to the oracle, same per-call split (3 × 912 MiB video, 3 × 3 MiB audio) | **0 findings**, as the oracle |
| unregistered ops | none | none |

The block ran with `has_audio=True`, `apply_gated_attention=True`,
`cross_attention_adaln=True` — all 6 attention instances — and 84 injected
parameters. No graph was hand-written: `examples/ltx.py` was used only as the
oracle to diff against.

## The five risk questions the spike existed to answer

1. **Does `get_matmul_config` survive fake shapes?** Yes. Its arithmetic and
   assertions ran on shim `padded_shape`s without firing. The program-config
   objects it constructs come back as opaque stubs, which is all the model needs.
2. **How much `Module`/`Parameter` machinery needs support?** Almost none.
   Setting `Parameter._data` to a metadata tensor was enough (84 parameters);
   `load_torch_state_dict` and `_prepare_torch_state` never run, so the
   `_interleave_heads` / swiglu permutations were never needed for *shapes* —
   only their *layout meaning* matters, and that is a declaration (blocker 12).
3. **Does `CCLManager.__init__` stub cleanly?** Yes. `_init_subdevice`,
   `_init_semaphores` and `synchronize_device` no-op; semaphore and subdevice
   objects flow through as stubs into op kwargs and are ignored. Real semaphore
   identity is only needed for phase 11's buffer-liveness gate.
4. **How many ops does one block touch?** **18 distinct ttnn ops** (of 53 the
   shim implements): `from_torch`, `all_gather_minimal_matmul_async`,
   `dit_fused_distributed_rmsnorm` (+ `_create_stats_buffer`), `addcmul`,
   `multiply`, `rotary_embedding_llama`, `chunk`, `sigmoid`, `permute`,
   `nlp_create_qkv_heads`, `concatenate_heads`, `unsqueeze`,
   `scaled_dot_product_attention`, `all_gather_async`,
   `ring_joint_scaled_dot_product_attention`, `minimal_matmul_split`,
   `minimal_matmul_strided_reduce_scatter_async`. The ~100-op estimate for the
   whole DiT surface still looks right, but a *transformer block* is 18 — the
   long tail is VAE/encoder, not the DiT.
5. **Hidden value-dependence?** None on this path. `fake_torch.Tensor.item()`
   raises by design and was never called.

## Two bugs found, both the expected class

Both were in the shim, not the analyzer or the design — and both are exactly the
failure mode the roadmap calls the top risk (blockers 36, 42).

1. **`num_heads_per_device` defaults to `1`** (`normalization.py:198`), which is
   the *no split* case; the shim split heads anyway. Result: logical shapes like
   `[1, 4, 38912, 1024]` where the truth is `[1, 1, 38912, 4096]`.
2. **Chunked AGMM reused the fused weight's symbol.** `to_qkv(chunks=3)` /
   `to_kv(chunks=2)` consume a column block of a wider weight; passing the fused
   `[4096, 12288]` symbol made the analyzer read 3072 columns per device instead
   of 1024, so head ranges came out as `[0:24)` instead of `[0:8)`.

**What this cost, before the fix: 15 spurious findings** (4 `dead_collective`,
6 `overwide_gather`, 5 `participant_shrink`) alongside the 6 real ones. A shape
error does not perturb the graph, it invents redundancy. That is the argument for
phase 11's per-op conformance suite, made concrete.

**What caught them:** the analyzer's own diagnostics (`K_COVERAGE` ×416,
`GATHER_AXIS_MISMATCH` ×8) plus the oracle diff. Neither bug was subtle once the
report was read — which is evidence the honesty rules (diagnostics surfaced with
findings, `examples/` kept as oracles) do their job.

## Environment notes, not design findings

* This machine has no torch, numpy or loguru, so the spike fakes the slice of
  torch the import path touches (`fake_torch.py`). **A real dry run should use
  real torch with `device='meta'`** — same idea, nothing to maintain.
* The repo needs **Python ≥ 3.10** (PEP 604 unions in evaluated annotation
  positions, `types.NoneType`); this box has 3.9.6. Worked around by compiling
  tt_dit sources with the `annotations` future flag via an import hook. A real
  dry run just uses the repo's interpreter.
* `models.common.utility_functions` pulls in numpy *and pytest*; tt_dit only
  wants `is_blackhole` from it. Worth splitting upstream, or the shim stubs it.

## New blocker the spike surfaced

**44. Source attribution points at shared library code.** Findings landed on
`layers/linear.py:250` — the AGMM call site inside `ColParallelLinear` — rather
than `attention_ltx.py:428` (`to_qkv`) where the *hand model* pointed. Both are
true; only the second is actionable. Capture should record a short caller stack
(2–3 tt_dit frames) and the report should show the outermost model frame with the
library frame underneath. Cheap, and it materially changes how a finding reads.

## Calibration for the roadmap

* Phase 6 (shim core, 3–4 weeks): plausible. This spike is ~1,000 throwaway lines
  covering one block; the production version needs real torch-meta support, the
  `unregistered` node kind, and pipeline-level construction.
* Phase 7 (shape fidelity): the two bugs above are exactly its content. Tiling is
  *not* modelled here (`padded_shape` just rounds the last two axes to 32) and
  uneven shards assert rather than divide — both real work.
* Phase 8 (op coverage): 18 ops per transformer block is encouraging; conv/VAE is
  where the count grows.
* Phase 11 (conformance): non-negotiable, and the spike shows why in one page.

## What the spike deliberately did not do

Tiling; uneven shards; the VAE/encoder stages; multi-submesh and CFG; pipeline
entry points; block rollup (the block is emitted once with `calls=48`); any
device conformance. The weight-chunk layout is modelled as separate per-chunk
weights, which is right for the maths but wants a real `chunked_weight` spec so
the per-device interleave is explicit.
