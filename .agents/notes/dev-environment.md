# Dev-environment essentials for tt-metal

Concrete facts about the `/localdev/tcheda` Tenstorrent dev box. Most are not documented in code comments — they come from trial and error.

## Python env

Always use:

```
/localdev/tcheda/tt-metal/python_env/bin/python3
```

The system Python is missing torch and ttnn. CPU-only torch is installed here (`torch 2.11.0+cpu`); the device is accessed via ttnn. Running an 8B HF model on CPU is ~1.7 tok/s — slow but tractable for reference generation.

## Hardware

Four N300 boards (8 dies total). An N300 is essentially two N150s — opening an **N150 mesh** picks one of the two dies:

| Label | `ttnn.MeshShape(...)` | Notes |
|---|---|---|
| `N150` | `(1, 1)` | One die of an N300; default for small/medium models. |
| `N300` | `(1, 2)` | Both dies of one N300. |
| `T3K` | `(1, 8)` | Multi-board. |
| `TG` | `(8, 4)` | Galaxy. |

## Mesh device lifecycle

Open with `ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(*shape))`. Modern ttnn sizes `trace_region_size` dynamically — omit it unless a "no trace region" error appears.

**Cleanup:** `ttnn.close_mesh_device(md)`. **NOT** `md.close()` — that method doesn't exist; `getattr(md, "close", None)` silently no-ops.

## Fabric

Single-device meshes don't need fabric setup, even though the demo's `device_params` says `fabric_config=True`. The tt-transformers conftest converts that to `None` on single-device. Multi-chip meshes (N300+) need:

```python
ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D, ttnn.FabricReliabilityMode.STRICT_INIT)
```

called **before** `open_mesh_device`.

## Crash recovery

After any crash or hang, run `tt-smi -r` before retrying. Symptoms that indicate a device reset is needed:

- The next run hangs without progress.
- The next run produces an identical error to the previous one despite code changes.
- Logs mention "device not initialized" or similar.

## Reading tt-metal crash logs

A typical failure dumps the Python traceback followed by **100+ lines of C++ backtrace**. Useful grep:

```bash
grep -nE "Error|Traceback|TT_FATAL|TT_THROW|^info:" /tmp/run.log | head -20
```

The line immediately above `Traceback` is usually the real error. The C++ backtrace is mostly noise; the first `info:` line after a `TT_THROW` is the actionable message.

## pytest output capture

pytest captures stderr by default. To see runtime debug prints (e.g. a temporary `sys.stderr.write` instrumentation), pass `-s`:

```bash
pytest path/to/test.py -k "..." -s
```

Without `-s`, your debug prints disappear into pytest's buffer and never appear in logs.

## Local weight paths (read-only)

| Model | Path |
|---|---|
| Llama 3.1 8B Instruct | `/proj_sw/user_dev/llama31-8b-data/Llama-3.1-8B-Instruct/` |
| Llama 3.2 family | `/proj_sw/user_dev/llama32-data/` |
| Llama 3.3 70B Instruct | `/proj_sw/user_dev/llama33-data/Llama-3.3-70B-Instruct/` |

Use these for `HF_MODEL` to avoid HF auth — but read [tt-transformers-gotchas.md](tt-transformers-gotchas.md) §5 for the `TT_CACHE_PATH` interaction.

## Long-running commands

Prefer foreground for anything with tqdm/progress output. Backgrounding kills live progress visibility. `tee` mangles tqdm carriage returns; the terminal renders them fine but the log file becomes noisy.
