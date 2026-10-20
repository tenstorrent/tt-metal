# TRELLIS.2-4B on Tenstorrent: image to 3D mesh demo

Turns one RGBA image of an object into a textured 3D mesh (vertex-coloured PLY) with
[TRELLIS.2-4B](https://github.com/microsoft/TRELLIS.2). The three flow transformers (structure, shape, texture: about
95% of the model's compute) run on **one Blackhole chip**. The rest stays on the host CPU in fp32: DINOv3
conditioning, the structure decoder, the cascade upsampler and the SC-VAE mesh decoders. The demo drives the reference
pipeline code and only swaps the flow models' forward, so stage order, sampler settings and RNG order are TRELLIS.2's own.

Status (branch `nkira/trellis2-port`): bring-up. Every TT flow model matches the CPU fp32 reference on single passes
(PCC >= 0.9999). Over a full 12-step sample it tracks a CPU bf16 run at 1.0 to 1.8x the bf16 error, depending on the
input. Untraced; host-driven sampler; 1024 and above not yet validated against a reference.

## Hardware and host

- One Blackhole chip. Tested on one chip of a Blackhole Galaxy. Any single Blackhole chip with 32 GB should work
  (p150, one chip of a p300 or QuietBox 2), but that is untested. On a multi-chip box, pick a chip with
  `TT_VISIBLE_DEVICES=<n>`.
- Host: about 40 GB free RAM for `512`; `1024` and above need more and take longer, since the mesh decoders run on
  the CPU (see Timings).
- Disk: about 11 GB of weights for `512`, plus 5 GB for the 1024 models.

## Setup

1. Build this branch of tt-metal and its Python env the usual way (`./build_metal.sh`, `./create_venv.sh`), then
   activate the env and work from the repo root:

   ```bash
   cd tt-metal && source python_env/bin/activate && export PYTHONPATH=$(pwd)
   uv pip install -r models/experimental/trellis2/demo/requirements.txt   # or: pip install easydict
   ```

2. Clone the reference code at the tested commit and apply the CPU overlay (removes the CUDA-only pieces: a torch
   sparse convolution, a torch dual-grid mesh extractor, sdpa attention for the sparse transformers):

   ```bash
   git clone https://github.com/microsoft/TRELLIS.2.git ~/TRELLIS.2
   cd ~/TRELLIS.2 && git checkout 75fbf0183001ed9876c8dbb35de6b68552ee08bd
   git apply <tt-metal>/models/experimental/trellis2/demo/trellis2_cpu_overlay.patch && cd -
   ```

3. DINOv3 weights are gated. Request access at
   https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m (Meta approves manually), then log in once with
   `hf auth login` (or set `HF_TOKEN`). The other weights download anonymously on the first run, into
   `~/.cache/trellis2_tt` (change with `--weights-dir` or `TRELLIS2_WEIGHTS_DIR`).

## Run

The input must be one object on a transparent background (RGBA PNG or WebP, about 1024 px). An object on plain white
also works with `--matte-white`. Use the full-resolution image: TRELLIS.2's structure stage is sensitive to the input,
and a downscaled or lossy copy can come out as a different shape.

```bash
python models/experimental/trellis2/demo/demo.py --trellis-dir ~/TRELLIS.2 --image my_object.png --out trellis2_out
#   --pipeline-type 512 (default) | 1024 | 1024_cascade | 1536_cascade      --seed 42      --threads <host threads>
```

Or as a test (same code, the chip opened by the tt-metal pytest fixture):

```bash
TRELLIS2_REPO=~/TRELLIS.2 TRELLIS2_IMAGE=my_object.png TRELLIS2_OUT=trellis2_out \
  pytest models/experimental/trellis2/tests/test_demo_e2e.py -s
```

Outputs in `--out`, named after the image, the pipeline type and the seed:

| File | What |
|---|---|
| `<name>_<type>_seed<seed>.ply` | raw dual-grid mesh with per-vertex base colour (opens in Blender, MeshLab or any PLY viewer) |
| `<name>_<type>_seed<seed>_views.png` | input next to the textured voxels from six axis views: a quick check without a 3D viewer |
| `<name>_<type>_seed<seed>.json` | voxel and token counts, mesh size, per-stage wall time and TT time, peak host memory |
| `<name>_preprocessed.png` | the cropped image the model saw |

The PLY is the decoder's raw mesh: no simplification, hole filling, UV unwrap or texture bake. The reference GLB export
needs CUDA-only libraries.

## Timings (512, one BH chip at 160 W, AMD EPYC 9354P host, untraced)

Measured on a 1024 px Tenstorrent-logo image, 6,320 voxels, mesh 3.1M vertices / 6.5M faces:

| Stage | Where | Time |
|---|---|---|
| load pipeline + TT models | host | 55 s, once |
| DINOv3 | CPU | 1.3 s |
| structure flow, 12 steps (22 DiT calls) | TT | 16.9 s |
| shape flow, 12 steps (21 calls) | TT | 19.6 s |
| texture flow, 12 steps (12 calls) | TT | 12.4 s |
| shape + texture decoders | CPU | 60 to 160 s, depending on host load |

The first run of each new token count also compiles kernels; later runs are faster. The CPU fp32 reference takes about
1,030 s for the three flow stages. At `1024` (~30K tokens) a shape step takes about 9 s on the chip. At
`1536_cascade` (~48K tokens) it takes about 18 s, and TRELLIS.2's 49,152-token cap may lower the output to 1280³.

## Files

- `demo.py`: the demo (`run_e2e()` is also what the test calls).
- `trellis2_cpu_overlay.patch`: CPU overlay for TRELLIS.2 75fbf018.
- `requirements.txt`: extra Python packages on top of the tt-metal env.
- `../tt/`: the TT flow models; `../reference/`: host-side tables and the torch mirror used for bring-up;
  `../tests/`: PCC tests against CPU goldens and the end-to-end test.

Licences: TRELLIS.2 is MIT. DINOv3 has its own licence (attribution and use limits). The demo never loads RMBG-2.0,
the reference pipeline's non-commercial background remover; inputs bring their own alpha.
