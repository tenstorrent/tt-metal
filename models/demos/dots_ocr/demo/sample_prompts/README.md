# Dots OCR sample prompts

Each file holds a JSON array of `{ image, prompt }` entries consumed by `demo.py --prompts-json`.
`image` may be `null` (text-only) or a local path / URL to an image understood by `PIL.Image.open`.

This mirrors `models/demos/qwen25_vl/demo/sample_prompts/` in spirit; Dots uses the simpler
single-turn `(image, prompt)` shape since it's single-user OCR.

Quick usage:

```bash
# Text-only, HF reference
python models/demos/dots_ocr/demo/demo.py \
    --prompts-json models/demos/dots_ocr/demo/sample_prompts/text_only.json \
    --backend hf

# Full TTNN stack (requires MESH_DEVICE)
MESH_DEVICE=N150 python models/demos/dots_ocr/demo/demo.py \
    --prompts-json models/demos/dots_ocr/demo/sample_prompts/demo.json \
    --backend ttnn \
    --dummy-weights
```
