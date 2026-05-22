# WAN 2.2 S2V Pipeline Notes

Brief notes on S2V-specific design decisions that aren't obvious from the
code or the reference repo.

## Reference image preprocessing

Today `WanPipelineS2V.__call__` matches `WanPipelineI2V`: it hands the input
PIL image directly to `VideoProcessor.preprocess(..., height=height,
width=width)`. That uses `resize_mode='default'` which **stretches** the image
to the target shape. If the input aspect ratio doesn't match the target,
features (faces, etc.) will be visibly squashed.

### Alternative: aspect-preserving letterbox (off by default)

To preserve aspect ratio, replace the preprocess call with:

```python
image_prompt_padded = ImageOps.pad(
    image_prompt, (width, height), method=Image.Resampling.LANCZOS
)
ref_tensor = self.video_processor.preprocess(
    image_prompt_padded, height=height, width=width
).to("cpu", dtype=torch.float32)
```

`ImageOps.pad` scales-to-contain (preserves all source content) then pads
with black to match exactly `(width, height)`. The mesh config and program
caches stay valid because `(height, width)` is unchanged.

Why not `ImageOps.fit`: cropping loses ~57% of a portrait image when fit into
a landscape target — typical headshots get their face cropped off.

Keeping behavior consistent with `pipeline_wan_i2v` for now; the letterbox
fix is a one-line swap when needed.
