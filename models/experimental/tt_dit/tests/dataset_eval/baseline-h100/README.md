### Steps:

1. `pip install -r requirements.txt`
2. `wget https://github.com/mlcommons/inference/raw/4b1d1156c23965172ae56eacdd8372f8897eb771/text_to_image/coco2014/captions/captions_source.tsv`
    - rename file to `captions.tsv`
3. `wget https://github.com/mlcommons/inference/raw/4b1d1156c23965172ae56eacdd8372f8897eb771/text_to_image/tools/val2014.npz`
4. Login to huggingface: `huggingface-cli login`
    - enter your token
5. Load weights locally, ex:
    - `pip install accelerate`
    - ```
      hf download black-forest-labs/FLUX.1-schnell \
    --repo-type model \
    --local-dir ./flux_schnell \
    --include "*"
    ```

4. Run eval: `pytest model.py::test_accuracy_model -v -s --start-from=0 --num-prompts=5`
