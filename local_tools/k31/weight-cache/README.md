# K3.1 TTNN weight-cache producer (local only)

`test_k31_weight_cache.py` builds one layer of Kimi-K3.1's TTNN weight cache (8x4 mesh, TP axis 1) from the
real dequantized checkpoint and points the runner's `8x4` link at it. It is not part of PR #64830: it was
removed from `models/demos/deepseek_v3_d_p/tests/kimi_k3/` by `tt_metal_tracker-8v1.3` (review finding M3;
user decision: local only, no repo script). Copied from `mvasilijevic/k31/main` @ `e376caa411d`, last changed
there by `50218a57b20`.

## What it writes

- Tensorbins and one `.layer_<i>.complete` marker per built layer under
  `cache_root(checkpoint, (8, 4), 1, KimiK31Adapter)`, by default
  `/mnt/weka/model-cache/scratch/moonshotai/Kimi-K3.1-Cache/Kimi-K3.1-Cache-prefill/kimi_k3_1_bh_32dev/<checkpoint-id>/mesh8x4.tpaxis1`
  (`$TT_KIMI_K31_PREFILL_TTNN_CACHE` relocates the root).
- The link `.../kimi_k3_1_bh_32dev/8x4 -> <checkpoint-id>/mesh8x4.tpaxis1`, which is the path the runner
  (`KimiK31Adapter.weight_cache_path`) and `tests/kimi_k3/k31_stage.weight_cache_path()` read. Nothing in the
  repo creates this link or builds K3.1 layers from real weights; CI only consumes the populated cache
  (the runner raises on an incomplete cache, `k31_stage` skips).

On 2026-11-13 the shared cache was complete: link to `e280adedf7fbe8f1/mesh8x4.tpaxis1`, 93 markers.

## Restore and run

Rebuild when the checkpoint changes (new checkpoint id) or a layer reports stale (missing/old-format KDA or
indexer tensorbins).

`test_k31_weight_cache.py` here is byte-identical to the repo copy at `e376caa411d` and calls
`cache_root(checkpoint, mesh, TP_AXIS, KimiK31Adapter)`. Review round 1 of `tt_metal_tracker-8v1.3` restored
`tt/kimi_k3/weights.py::cache_root` to its K3-only base form (no `adapter` argument), so:

- in a checkout of `e376caa411d` (before that restore), the copy runs as is: skip the `patch` step below;
- in a checkout of the PR head or later, apply `cache_root_k31.patch` after copying. It replaces the call with
  the same path computed from `KimiK31Adapter` (`$TT_KIMI_K31_PREFILL_TTNN_CACHE` or `ttnn_cache_default` /
  `kimi_k3_1_bh_32dev` / `<checkpoint-id>` / `mesh8x4.tpaxis1`, empty env var disables) and imports
  `_checkpoint_identity` instead of `cache_root`. Verified 2026-11-13: for the real checkpoint it gives
  `.../kimi_k3_1_bh_32dev/e280adedf7fbe8f1/mesh8x4.tpaxis1`, the current `8x4` target, identical to the old call.

In a built tt-metal checkout:

```bash
cp /home/mvasilijevic/repos/k31-local-tools/weight-cache/test_k31_weight_cache.py \
   models/demos/deepseek_v3_d_p/tests/kimi_k3/
(cd models/demos/deepseek_v3_d_p/tests/kimi_k3 && \
  patch -p1 < /home/mvasilijevic/repos/k31-local-tools/weight-cache/cache_root_k31.patch)   # PR head or later only
source python_env/bin/activate && export PYTHONPATH=$PWD
umask 0002   # shared tree: files 664 / dirs 2775, readable by others
for L in $(seq 0 92); do
  K31_CACHE_LAYER=$L scripts/run_safe_pytest.sh \
    models/demos/deepseek_v3_d_p/tests/kimi_k3/test_k31_weight_cache.py || break
done
rm models/demos/deepseek_v3_d_p/tests/kimi_k3/test_k31_weight_cache.py   # never commit it
```

One layer per run keeps each device job short and releases the device lock between layers; a cold MoE layer
holds ~59 GB of routed experts in host memory. A layer already marked is a cache hit (load only). Needs the
checkpoint at `$KIMI_K31_HF_MODEL` (default `/mnt/weka/model-weights/llm/moonshotai/Kimi-K3.1-Pro-dequantized`)
and an 8x4 galaxy. Imports it relies on: `tt/kimi_k3/weights.py` (`cache_root` or, patched, `_checkpoint_identity`; `layer_is_cached`,
`load_layer_state_dict_cached`, `load_tensors`), `tests/attn_res/checkpoint_utils.load_attn_res_state_dict`,
`tests/kda/checkpoint_utils.resolve_model_root`, `TtKimiK3Transformer`; if a later refactor renames them,
adapt the copy.
