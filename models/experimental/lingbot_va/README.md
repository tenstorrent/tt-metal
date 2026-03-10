# Lingbot-VA (TT-Metal)

Lingbot-VA reference and TT implementations for video–action modeling. This directory includes a **demo script** that runs inference (action prediction) or multi-chunk video generation from checkpoint and camera images.

## Demo

The demo script builds the same observation format as the Lingbot-VA server and runs inference or generation locally (no WebSocket server).

**Script:** `tests/demo/lingbot_va_inference.py`

Run from the **tt-metal** repo root (or ensure the `lingbot_va` package root is on `PYTHONPATH`).

### Prerequisites

- Checkpoint directory with subdirs: `vae/`, `tokenizer/`, `text_encoder/`, `transformer/` (e.g. from `tests/download_pretrained_weights.py` or `reference/checkpoints/`).
- For image input: a directory containing three PNGs named:
  - `observation.images.cam_high.png`
  - `observation.images.cam_left_wrist.png`
  - `observation.images.cam_right_wrist.png`
  Example: `tests/demo/sample_images/robotwin/`.

---

### Inference (action prediction)

Runs reset + one infer chunk: encodes the observation, runs the diffusion loops, returns an action array. Does **not** run `--generate`.

```bash
# From tt-metal repo root
python3 models/experimental/lingbot_va/tests/demo/lingbot_va_inference.py \
  --checkpoint models/experimental/lingbot_va/reference/checkpoints/ \
  --images-dir models/experimental/lingbot_va/tests/demo/sample_images/robotwin/ \
  --prompt "Lift the cup from the table"
```

**Optional:**

- `--output <path>` – save the action array to a `.npy` file.
- `--save-dir <path>` – directory for internal outputs (e.g. `latents_*.pt`, `actions_*.pt`). Default: `tests/demo/out_inference`.
- `LINGBOT_VA_CHECKPOINT` – environment variable can set default `--checkpoint`; if set, `--checkpoint` can be omitted.

**Output:** Prints action shape (e.g. `(16, 2, 16)`) and, if `--output` is set, writes the action array to disk.

---

### Generate (multi-chunk video → demo.mp4)

Runs **only** generation: loads the same three camera images as initial observation, runs multiple inference chunks, decodes latents to video, and saves `demo.mp4`. Does **not** run the single-chunk infer path.

```bash
# From tt-metal repo root
python3 models/experimental/lingbot_va/tests/demo/lingbot_va_inference.py \
  --generate \
  --checkpoint models/experimental/lingbot_va/reference/checkpoints/ \
  --images-dir models/experimental/lingbot_va/tests/demo/sample_images/robotwin/ \
  --prompt "Lift the cup from the table" \
  --save-dir models/experimental/lingbot_va/tests/demo/out_generate \
  --num-chunks 10
```

**Options:**

- `--generate` – use generate path only (no infer).
- `--num-chunks` – number of chunks to generate (default: 10).
- `--save-dir` – directory where `demo.mp4` is written (e.g. `.../out_generate/demo.mp4`).

**Output:** Prints the path to the saved video (e.g. `.../out_generate/demo.mp4`).

---

### Summary

| Mode        | Command focus              | Output |
|------------|-----------------------------|--------|
| **Inference** | No `--generate`            | Action array (printed; optional `--output action.npy`). Internal saves in `--save-dir`. |
| **Generate**  | `--generate` + `--checkpoint` | `demo.mp4` in `--save-dir`. |

Both modes require `--checkpoint` (or `LINGBOT_VA_CHECKPOINT`) and, for the default image source, an `--images-dir` that contains the three camera PNGs listed above.
