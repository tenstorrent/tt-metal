# tt_transformers Generator: gotchas when bootstrapping it outside the pytest fixture

Most of these are absorbed by `prepare_generator_args` (in `models/tt_transformers/demo/simple_text_demo.py`) and the pytest `mesh_device` fixture (in `conftest.py`). If you build on tt_transformers utilities directly from your own generator — instead of going through the pytest demo path — you have to replicate them, or paged attention will silently break in confusing ways.

Reference correct calling sequence: `models/tt_transformers/demo/simple_text_demo.py` (specifically `prepare_generator_args`, `create_tt_page_table`, and the prefill+decode loop). Diff your code against it when a gotcha hits.

## 1. kv_cache nesting needs an outer wrap

`Generator.prefill_forward_text` indexes the cache with one outer level on top of the per-layer list:

    kv_cache[0][layer_id][k_or_v]

But `create_tt_model` returns only `[layer_id][k_or_v]`. Skipping the outer wrap → the wrong index gets passed into `Attention.forward_prefill`, the K tensor appears as 3-D inside `paged_fill_cache`, and the C++ validator crashes with:

    TT_THROW: ShapeBase[] index out of range. 3 not in [-4, 3)

Fix:

```python
_, _, _tt_kv_cache_inner, _ = create_tt_model(mesh_device, ...)
self._tt_kv_cache = [_tt_kv_cache_inner]   # one-element outer list
```

## 2. `max_seq_len` must be a power of 2

`warmup_prefill` enumerates supported sequence lengths from `max_seq_len` and asserts each is a power of 2. Round up before passing — `1280` will fail; `2048` will not.

## 3. `decode_forward` returns a tuple even with `sampling_params=None`

`Generator.decode_forward(...)` always returns `(logits, log_probs)`. `Generator.prefill_forward_text` with `sampling_params=None` returns just `logits`. Unwrap conditionally:

```python
logits = decode_out[0] if isinstance(decode_out, tuple) else decode_out
```

## 4. `HF_MODEL` + `TT_CACHE_PATH` interaction

`HF_MODEL` accepts either a HuggingFace repo id or a local checkpoint directory. Local paths are convenient when there's no HF auth. But:

- `TT_CACHE_PATH` defaults to `os.path.join("model_cache", HF_MODEL, device_name)`.
- `os.path.join` discards the relative prefix when `HF_MODEL` is **absolute**.
- The default cache path then collapses to a subdir of the (often read-only) checkpoint dir.
- First write attempts hit `errno=13 Permission denied`.

**Always set `TT_CACHE_PATH` explicitly when `HF_MODEL` is an absolute path.** A reasonable default is `<repo>/model_cache/<model_name>/<mesh>`. Set it via `os.environ.setdefault("TT_CACHE_PATH", ...)` in your generator's bootstrap before `ModelArgs` constructs.

## Canonical reference run

When in doubt, the demo's batch-1 paged-attention test is the source of truth for single-prompt N150 prefill+decode:

```bash
MESH_DEVICE=N150 HF_MODEL=<path-or-repo> TT_CACHE_PATH=<writable-path> \
  /localdev/tcheda/tt-metal/python_env/bin/python3 \
  -m pytest models/tt_transformers/demo/simple_text_demo.py \
  -k "batch-1 and performance and not accuracy" -s
```

The `-s` flag preserves stderr output from any temporary instrumentation — see [dev-environment.md](dev-environment.md).
