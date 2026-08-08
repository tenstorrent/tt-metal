# DINO-5scale Swin-L — Reference (inference + config)

Backup reference for **DINO-5scale Swin-L** (object detection, 1333×800, COCO 80 classes). Use for PyTorch baseline and TTNN bring-up.

## Files here

- **`infer.py`** — MMDetection inference script (load config + checkpoint, run on image/dir, save JSON + vis).
- **`dino_5scale_swin_l.py`** — Full config (36e, test scale 1333×800).

## Download weights / checkpoint

See the **parent README** for the full section **“Download weights / checkpoint”**:

- Use tt-metal **python_env** (`source python_env/bin/activate`).
- Install: `pip install openmim` then `mim install mmdet`.
- Download: `mim download mmdet --config dino-5scale_swin-l_8xb2-36e_coco --dest checkpoints/dino_5scale_swin_l`.
- Optional: symlink the `.pth` to `dino_5scale_swin_l.pth` and copy this folder’s config into `checkpoints/dino_5scale_swin_l/` for default script paths.

## Run inference

From repo root (checkpoint under model folder):

```bash
export PYTHONPATH=/home/ubuntu/.local/lib/python3.10/site-packages  # if mmdet in user site-packages
python models/experimental/dino_5scale_swin_l/references/infer.py \
  --config models/experimental/dino_5scale_swin_l/reference/dino_5scale_swin_l.py \
  --checkpoint models/experimental/dino_5scale_swin_l/checkpoints/dino_5scale_swin_l/dino_5scale_swin_l.pth \
  --input /path/to/image.jpg --output-dir results/standalone_dino --save-vis
```

Dependencies: torch, cv2, tqdm, mmdet (mmengine, mmcv).
