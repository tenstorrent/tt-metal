# SDXL in ComfyUI — in-process POC (Tenstorrent)

Run SDXL text-to-image from a ComfyUI graph, with the ComfyUI node driving a
Tenstorrent mesh **directly** through tt-metal. No tt-inference-server, no media
server, no FastAPI in the loop.

This is a proof of concept, not a merge candidate.

## Dependency boundary

The whole integration is one Python import. A single-file ComfyUI custom node
imports tt-metal's `SDXLGenerator` and calls it:

```
ComfyUI graph
  └─ TT SDXL node           (custom_nodes/tt_sdxl.py)
       └─ SDXLGenerator     models/demos/stable_diffusion_xl_base/tt/generator.py   ← tt-metal
            └─ TtSDXLPipeline ─→ TT mesh   (N300: 2 chips, CFG-parallel)
```

The only requirement that makes the import work is **tt-metal on `PYTHONPATH`**.
There is no Settings singleton, no `MODEL` / `DEVICE` / `MODEL_RUNNER` env, no
HTTP hop.

`SDXLGenerator` owns the device lifecycle (open mesh → warm trace → generate →
close); it lives on this branch at `tt/generator.py` — read it there.

## The ComfyUI custom node (`custom_nodes/tt_sdxl.py`)

ComfyUI gitignores `custom_nodes/`, so the node is **not** committed with
tt-metal. It is a single file — ComfyUI's loader picks up any top-level `.py` in
`custom_nodes/` and calls its module-level `comfy_entrypoint` (the V3 node API).
Recreate it verbatim:

```python
"""TT SDXL — in-process text-to-image ComfyUI node (Approach B).

One file, one dependency: it imports tt-metal's SDXLGenerator and drives the TT
mesh directly. No tt-inference-server, no media server, no FastAPI. tt-metal must
be on PYTHONPATH (the standard tt-metal env).

Drop this at ComfyUI/custom_nodes/tt_sdxl.py.
"""
import os
import threading

import numpy as np
import torch
from comfy_api.latest import ComfyExtension, io
from typing_extensions import override

_SEED_MAX = 0xFFFFFFFFFFFFFFFF
TENSOR_PARALLEL = int(os.environ.get("TT_SDXL_TENSOR_PARALLEL", "2"))

class SDXLGeneratorCache:
    """Holds one warm SDXLGenerator for the process. The mesh + captured trace are
    single-tenant, so access is serialized; steps and guidance are runtime args on
    generate(), so a single open serves every request."""

    def __init__(self):
        self._lock = threading.RLock()
        self._gen = None

    def generate(self, prompt, negative_prompt, num_inference_steps, seed,
                 guidance_scale, number_of_images):
        with self._lock:
            if self._gen is None:
                # Imported lazily so ComfyUI can register the node without pulling
                # in ttnn at startup; the device is only touched once a job runs.
                from models.demos.stable_diffusion_xl_base.tt.generator import SDXLGenerator

                g = SDXLGenerator(TENSOR_PARALLEL)
                g.open()
                self._gen = g
            return self._gen.generate(
                prompt,
                negative_prompt or None,
                seed=int(seed),
                num_images=int(number_of_images),
                num_inference_steps=int(num_inference_steps),
                guidance_scale=float(guidance_scale),
            )


def _pils_to_image_tensor(pils):
    # ComfyUI IMAGE = torch.float32 [B, H, W, C] in [0, 1], RGB.
    if not pils:
        raise ValueError("SDXL generator returned no images")
    arrs = [np.asarray(p.convert("RGB"), dtype=np.float32) / 255.0 for p in pils]
    return torch.from_numpy(np.stack(arrs, axis=0))


class TTSDXLTextToImage(io.ComfyNode):
    # ComfyUI calls execute() as a stateless classmethod, so the warm generator and
    # its open mesh live on the class to survive across jobs.
    _cache = SDXLGeneratorCache()

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TTSDXLTextToImage",
            display_name="TT SDXL — Text to Image (in-process)",
            category="Tenstorrent",
            inputs=[
                io.String.Input("prompt", multiline=True, default=""),
                # Not optional, so it renders directly under prompt; empty == no
                # negative prompt (coerced to None below).
                io.String.Input("negative_prompt", multiline=True, default=""),
                io.Int.Input("num_inference_steps", default=20, min=12, max=200),
                io.Int.Input("seed", default=0, min=0, max=_SEED_MAX),
                io.Float.Input("guidance_scale", default=5.0, min=1.0, max=20.0, step=0.1),
                io.Int.Input("number_of_images", default=1, min=1, max=4),
            ],
            outputs=[io.Image.Output()],
        )

    @classmethod
    def execute(cls, prompt="", negative_prompt=None, num_inference_steps=20, seed=0,
                guidance_scale=5.0, number_of_images=1):
        pils = cls._cache.generate(prompt, negative_prompt, num_inference_steps, seed,
                                   guidance_scale, number_of_images)
        return io.NodeOutput(_pils_to_image_tensor(pils))


class TTExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [TTSDXLTextToImage]


async def comfy_entrypoint() -> TTExtension:
    return TTExtension()
```

`TT_SDXL_TENSOR_PARALLEL` (default `2`) is the one config knob: `2` for N300
(CFG-parallel across both chips — **required** on N300, the 1-chip path mis-indexes
the UNet time embeddings), `1` for a single-chip N150.

## Boot-up

### 0. Build tt-metal (once)

Produces `python_env/` (a uv-managed venv) and an editable `ttnn`. Follow the
repo's `INSTALLING.md` / `build_metal.sh`.

### 1. Add ComfyUI into tt-metal's python_env (once)

ComfyUI runs in the *same* venv as tt-metal so the node can import
`models.demos…`. tt-metal pins its torch trio; install ComfyUI's requirements
**without** letting them move torch. Write a constraints file
`comfy-torch-constraints.txt`:

```
torch==2.10.0+cpu
torchvision==0.25.0+cpu
torchaudio==2.10.0+cpu
```

then install ComfyUI's deps under it, and drop the node in place:

```bash
cd $TT_METAL_HOME
uv pip install --python python_env -c comfy-torch-constraints.txt -r ../ComfyUI/requirements.txt
cp tt_sdxl.py ../ComfyUI/custom_nodes/tt_sdxl.py
```

### 2. Environment (every boot)

```bash
export TT_METAL_HOME=/path/to/tt-metal
export PYTHONPATH=$TT_METAL_HOME          # required: resolves `import models.demos…`
export HF_HOME=/path/to/hf_cache          # weights cache; without it, first run
                                          # downloads to ~/.cache/huggingface
```

`PYTHONPATH=$TT_METAL_HOME` is the load-bearing one — it's what lets the node
import the generator (the node carries no `sys.path` fallback). `TT_METAL_HOME` is
the standard tt-metal variable; keep it set (tooling and profiler paths read it).
Nothing else is needed: no device-selection, cache, or model-runner env — the JIT
kernel cache and the mesh come from defaults.

### 3. Start the backend

```bash
cd $TT_METAL_HOME/../ComfyUI
$TT_METAL_HOME/python_env/bin/python -u main.py --cpu --listen 0.0.0.0 --port 8188
```

`--cpu` keeps ComfyUI's own torch off any (nonexistent) CUDA device — the TT node
drives the mesh out-of-band. `--listen 0.0.0.0` lets the SSH tunnel reach it.
ComfyUI serves its own web UI; there is no separate frontend process.

### 4. Reach the UI from your laptop

How you reach the UI (backend port 8188) depends on your network path to the box.
Two cases:

- **Backend host is directly SSH-reachable:**

  ```bash
  ssh -L 8189:localhost:8188 $USER@<your-box>
  ```

- **Backend runs in a container not routable from your laptop** (common on shared
  boxes): land on the container's *host* and forward to the container's own IP. The
  two machine-specific bits are discoverable on the box — run this there and it
  prints the command to paste on your laptop:

  ```bash
  # on the box (inside the container):
  echo "ssh -L 8189:$(hostname -i):8188 $USER@<your-box-host>"
  ```

  `hostname -i` is the container IP, `$USER` your login. `<your-box-host>` is the
  machine you SSH into to reach this box (the one you reserved) — it is *not*
  discoverable from inside the container, so fill it in yourself.

Then open `http://localhost:8189` in your browser (8189 is an arbitrary free local
port). Add the **TT SDXL — Text to Image** node (category *Tenstorrent*), type a
prompt, and run. First run warms the trace (~115s); every subsequent run is ~5s,
including ones that change steps, guidance, prompt, or seed.
