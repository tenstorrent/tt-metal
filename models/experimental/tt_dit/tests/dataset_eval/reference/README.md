# Reference Implementations for Dataset Evaluation

This directory contains reference implementations for evaluating models on H100 hardware. These implementations use standard diffusers pipelines and provide baseline metrics for comparison with TT-Metal implementations.

## Setup

1. `pip install -r requirements.txt`
2. Download COCO captions:
   - `wget https://github.com/mlcommons/inference/raw/4b1d1156c23965172ae56eacdd8372f8897eb771/text_to_image/coco2014/captions/captions_source.tsv`
   - Rename file to `captions.tsv`
3. Download COCO validation statistics:
   - `wget https://github.com/mlcommons/inference/raw/4b1d1156c23965172ae56eacdd8372f8897eb771/text_to_image/tools/val2014.npz`
4. Login to HuggingFace: `huggingface-cli login`
   - Enter your token
   - Accept license agreements as required for each model

## Flux Evaluation

**File**: `flux.py`

**Steps**:

1. Load weights locally:
   - `pip install accelerate`
   - ```
     huggingface-cli download black-forest-labs/FLUX.1-schnell \
       --repo-type model \
       --local-dir ./flux_schnell \
       --include "*"
     ```

2. Run eval:
   ```
   pytest flux.py::test_accuracy_model -v -s --start-from=0 --num-prompts=5
   ```

**Notes**:
- Default settings: 1024x1024, guidance_scale=1.0, 4 inference steps (schnell model)
- Results are saved to `test_reports/flux_test_results.json`

## Stable Diffusion 3.5 Evaluation

**File**: `sd35.py`

**Steps**:

1. Accept the license agreement for SD3.5 at: https://huggingface.co/stabilityai/stable-diffusion-3.5-large

2. Load weights locally:
   - `pip install accelerate`
   - ```
     huggingface-cli download stabilityai/stable-diffusion-3.5-large \
       --repo-type model \
       --local-dir ./sd35_large \
       --include "*"
     ```

3. Run eval:
   ```
   pytest sd35.py::test_accuracy_sd35 -v -s --start-from=0 --num-prompts=5
   ```

**Notes**:
- SD3.5 uses three text encoders: CLIP-L, CLIP-G, and T5-XXL
- The T5 encoder provides enhanced text understanding and longer prompt support (up to 256 tokens)
- Default settings: 1024x1024, guidance_scale=3.5, 40 inference steps
- Requires ~40GB VRAM on H100 for full resolution inference
- Results are saved to `test_reports/sd35_test_results.json`

## Motif Evaluation

**File**: `motif.py`

**Note**: There isn't an existing diffusers pipeline for Motif yet, so this uses the official Motif inference script.

**Steps**:

1. Clone the Motif repository:
   ```
   git lfs install
   GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/Motif-Technologies/Motif-Image-6B-Preview
   cd Motif-Image-6B-Preview
   huggingface-cli download Motif-Technologies/Motif-Image-6B-Preview --local-dir ./ --include "checkpoints/*"
   ```

2. Download captions for eval:
   - `wget https://github.com/mlcommons/inference/raw/4b1d1156c23965172ae56eacdd8372f8897eb771/text_to_image/coco2014/captions/captions_source.tsv`
   - `mv captions_source.tsv captions.tsv`

3. Accept the license agreement for Motif if required

4. Run eval:
   ```
   python motif.py \
     --images-dir outputs/seed_7777/scale_4.0 \
     --prompts-file captions.tsv \
     --start-from 0 \
     --num-prompts 500
   ```

**Notes**:
- This script uses the official Motif inference.py script to generate images
- Results are saved to `dataset_eval_results.json`
