Note: There isn't an existing diffusers pipeline for Motif yet.

### Steps:

1. `pip install -r ../requirements.txt`
2.
- `wget https://github.com/mlcommons/inference/raw/4b1d1156c23965172ae56eacdd8372f8897eb771/text_to_image/coco2014/captions/captions_source.tsv`
- `mv captions_source.tsv captions.tsv`
3. `wget https://github.com/mlcommons/inference/raw/4b1d1156c23965172ae56eacdd8372f8897eb771/text_to_image/tools/val2014.npz`
4. Login to huggingface: `huggingface-cli login`
    - enter your token
    - Accept the license agreement for Motif if required
5. Load weights locally:
-
```
git lfs install
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/Motif-Technologies/Motif-Image-6B-Preview
cd Motif-Image-6B-Preview
hf download Motif-Technologies/Motif-Image-6B-Preview --local-dir ./ --include "checkpoints/*"
```

6. Run eval: `pytest model.py::test_accuracy_motif -v -s --start-from=0 --num-prompts=5`
