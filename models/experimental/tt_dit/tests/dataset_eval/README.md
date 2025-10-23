## Dataset Eval

### Steps:

1. `wget https://github.com/mlcommons/inference/raw/4b1d1156c23965172ae56eacdd8372f8897eb771/text_to_image/coco2014/captions/captions_source.tsv`
2. `wget https://github.com/mlcommons/inference/raw/4b1d1156c23965172ae56eacdd8372f8897eb771/text_to_image/tools/val2014.npz`
3. Login to hugging face: `huggingface-cli login`
4. Tmux for lasting terminal:
    - if don't have tmux: `sudo apt install tmux`
    - `tmux new -s new_session`
4. Run desired test, ex. `pytest models/experimental/tt_dit/tests/dataset_eval/flux/test_flux_accuracy.py --num-prompts 500`

Results will generate to the corresponding results .json file, ex. `test_reports/flux_test_results.json`
