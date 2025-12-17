### Steps:

1. `pip install -r ../requirements.txt`
2. `wget https://github.com/mlcommons/inference/raw/4b1d1156c23965172ae56eacdd8372f8897eb771/text_to_image/coco2014/captions/captions_source.tsv`
    - rename file to `captions.tsv`
3. `wget https://github.com/mlcommons/inference/raw/4b1d1156c23965172ae56eacdd8372f8897eb771/text_to_image/tools/val2014.npz`
4. Login to huggingface: `huggingface-cli login`
    - enter your token
    - Accept the license agreement for SD3.5 at: https://huggingface.co/stabilityai/stable-diffusion-3.5-large
5. Load weights locally:
    - `pip install accelerate`
    - ```
      huggingface-cli download stabilityai/stable-diffusion-3.5-large \
    --repo-type model \
    --local-dir ./sd35_large \
    --include "*"
    ```

6. Run eval: `pytest model.py::test_accuracy_sd35 -v -s --start-from=0 --num-prompts=5`

### Notes:
- SD3.5 uses three text encoders: CLIP-L, CLIP-G, and T5-XXL
- The T5 encoder provides enhanced text understanding and longer prompt support (up to 256 tokens)
- Default settings: 1024x1024, guidance_scale=3.5, 40 inference steps
- Requires ~40GB VRAM on H100 for full resolution inference
- Results are saved to `test_reports/sd35_test_results.json`
