# ATSS Swin-L DyHead COCO Evaluation Guide

This guide runs the same four-slice COCO evaluator in three configurations:

1. **TTNN stage-2 precision:** the current ATSS configuration, with Swin-L
   stage-2 MLP linears promoted to HiFi2/BF16.
2. **TTNN vanilla:** the original Swin-L precision path, with no
   high-precision stages.
3. **PyTorch reference:** the standalone reference model on the same four
   tiles, processed sequentially on CPU.

Both TTNN commands use a 1×4 device mesh. The vanilla wrapper changes the
backbone factory only inside its Python process; it does not edit the source
defaults.

## Create and verify the Python environment

From the tt-metal repository root, create the repository-managed environment
once:

```bash
cd /path/to/tt-metal
./create_venv.sh --python-version 3.10
```

`create_venv.sh` creates `python_env/` and installs tt-metal together with its
required Python and development dependencies. Use this script rather than a
plain `python -m venv` environment.

Activate the environment and build tt-metal:

```bash
source python_env/bin/activate
./build_metal.sh --release
```

Verify that the active interpreter comes from `python_env` and that TTNN can
be imported:

```bash
which python
python -c "import ttnn; print('TTNN import OK:', ttnn.__file__)"
```

`which python` should print:

```text
/path/to/tt-metal/python_env/bin/python
```

For later shells, the environment only needs to be reactivated:

```bash
cd /path/to/tt-metal
source python_env/bin/activate
```

If `python_env/` already exists but the TTNN import fails, refresh the editable
installation while the environment is active:

```bash
uv pip install -e .
./build_metal.sh --release
```

## Evaluation setup

Run from the repository root:

```bash
cd /path/to/tt-metal
source python_env/bin/activate

export TT_METAL_HOME=$PWD
export PYTHONPATH=$PWD
export ARCH_NAME=wormhole_b0
export WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml
export ATSS_CHECKPOINT=$PWD/models/experimental/atss_swin_l_dyhead/weights/best_coco_bbox_mAP_epoch_28.pth
```

The evaluator requires `pycocotools`:

```bash
python -c "import pycocotools"
```

If that import fails, install it in the active environment:

```bash
pip install pycocotools
```

Define these optional shell variables to shorten the commands:

```bash
MODEL_DIR=models/experimental/atss_swin_l_dyhead
DATASET_DIR=$MODEL_DIR/boat-detection-marina.v2i.coco-segmentation
EVAL=$MODEL_DIR/demo/evaluate_coco_slice_4dev.py
VANILLA_EVAL=$MODEL_DIR/demo/evaluate_coco_slice_4dev_vanilla.py
RESULTS=$MODEL_DIR/results/coco_eval_slice_4dev
```

## PCC validation

PCC uses the same environment and `ATSS_CHECKPOINT` configured above.

### Strict E2E PCC with stage-2 precision

Run the checked-in ATSS configuration:

```bash
pytest models/experimental/atss_swin_l_dyhead/tests/pcc/test_ttnn_e2e.py -v -s
```

Expected result with the custom boat checkpoint: **pass**. The stage-2
precision configuration raises head level-0 centerness PCC above the strict
`0.96` threshold (measured at approximately `0.962101`).

### Strict E2E PCC with vanilla precision

Use the process-local vanilla wrapper:

```bash
python models/experimental/atss_swin_l_dyhead/tests/pcc/run_ttnn_e2e_vanilla.py
```

This invokes the same strict test after disabling all high-precision backbone
stages. With the custom boat checkpoint, the expected baseline result is a
failure only at head level-0 centerness:

```text
Head cent level 0 PCC 0.959824 < 0.96
```

This expected failure demonstrates the difference that the stage-2 precision
change resolves; the wrapper does not lower the test threshold.

Additional pytest arguments can be appended to the wrapper command, for
example:

```bash
python models/experimental/atss_swin_l_dyhead/tests/pcc/run_ttnn_e2e_vanilla.py --tb=short
```

### Full ATSS PCC suite

To run all ATSS PCC tests with the checked-in stage-2 precision configuration:

```bash
pytest models/experimental/atss_swin_l_dyhead/tests/pcc -v -s
```

## Score threshold

- Use `--score-threshold 0.05` for the primary COCO AP/AR evaluation. Keeping
  low-confidence predictions allows COCOeval to construct the full
  precision–recall curve.
- Use `--score-threshold 0.3` to match the visible-box operating point used by
  the demos and visual comparisons. AP/AR will generally be lower because
  low-confidence true positives are discarded.

## Test split

### TTNN with stage-2 precision

The normal evaluator uses the checked-in ATSS default
`ATSS_HIGH_PRECISION_MLP_STAGES = (2,)`:

```bash
python "$EVAL" \
    --annotations "$DATASET_DIR/test/_annotations.coco.json" \
    --images-dir "$DATASET_DIR/test" \
    --score-threshold 0.05 \
    --output-dir "$RESULTS/test_stage2_hp"
```

### TTNN vanilla

The vanilla wrapper passes empty high-precision stage tuples to the same model
and evaluator:

```bash
python "$VANILLA_EVAL" \
    --annotations "$DATASET_DIR/test/_annotations.coco.json" \
    --images-dir "$DATASET_DIR/test" \
    --score-threshold 0.05 \
    --output-dir "$RESULTS/test_vanilla"
```

### PyTorch reference

```bash
python "$EVAL" \
    --pytorch-only \
    --annotations "$DATASET_DIR/test/_annotations.coco.json" \
    --images-dir "$DATASET_DIR/test" \
    --score-threshold 0.05 \
    --output-dir "$RESULTS/test_pytorch"
```

## Validation split

### TTNN with stage-2 precision

```bash
python "$EVAL" \
    --annotations "$DATASET_DIR/valid/_annotations.coco.json" \
    --images-dir "$DATASET_DIR/valid" \
    --score-threshold 0.05 \
    --output-dir "$RESULTS/valid_stage2_hp"
```

### TTNN vanilla

```bash
python "$VANILLA_EVAL" \
    --annotations "$DATASET_DIR/valid/_annotations.coco.json" \
    --images-dir "$DATASET_DIR/valid" \
    --score-threshold 0.05 \
    --output-dir "$RESULTS/valid_vanilla"
```

### PyTorch reference

```bash
python "$EVAL" \
    --pytorch-only \
    --annotations "$DATASET_DIR/valid/_annotations.coco.json" \
    --images-dir "$DATASET_DIR/valid" \
    --score-threshold 0.05 \
    --output-dir "$RESULTS/valid_pytorch"
```

## Run at score threshold 0.3

Use the same commands with:

```bash
--score-threshold 0.3
```

Always use a distinct output directory, for example:

```bash
--output-dir "$RESULTS/test_stage2_hp_score_0p3"
--output-dir "$RESULTS/test_vanilla_score_0p3"
--output-dir "$RESULTS/test_pytorch_score_0p3"
```

## Other useful options

```text
--num-samples N       Evaluate only the first N image IDs.
--no-trace            Disable the traced TTNN pipeline.
--overlap 128         Tile overlap in pixels.
--merge-iou 0.55      IoU threshold used to merge tile detections.
--category-id 1       COCO category mapped from model label 0.
```

Do not pass `--pytorch-only` to the vanilla wrapper; that script is
specifically for vanilla TTNN.

## Outputs

Every output directory contains:

- `metrics.json`: all 12 COCO AP/AR metrics, timing, configuration, and total
  prediction count.
- `predictions.json`: COCO-format bbox predictions used by COCOeval.

Use separate directories for each model, split, and score threshold so results
are never overwritten.

## Visual comparison

After generating all three prediction files at threshold 0.3, render a 2×2
comparison containing COCO ground truth, PyTorch, vanilla TTNN, and stage-2
TTNN:

```bash
python "$MODEL_DIR/demo/visualize_coco_comparison.py" \
    --annotations "$DATASET_DIR/test/_annotations.coco.json" \
    --images-dir "$DATASET_DIR/test" \
    --pytorch-predictions "$RESULTS/test_pytorch_score_0p3/predictions.json" \
    --ttnn-vanilla-predictions "$RESULTS/test_vanilla_score_0p3/predictions.json" \
    --ttnn-stage2-hp-predictions "$RESULTS/test_stage2_hp_score_0p3/predictions.json" \
    --score-threshold 0.3 \
    --output-dir "$MODEL_DIR/results/coco_visual_comparison/test_score_0p3"
```

Replace `test` with `valid` and use the corresponding validation prediction
paths for the validation split.

## Latest verified test rerun

The following results were produced on 2026-08-05 with score threshold 0.05:

| Metric | TTNN vanilla | TTNN stage-2 precision |
|---|---:|---:|
| AP | 0.616219 | 0.616243 |
| AP50 | 0.951621 | 0.951534 |
| AP75 | 0.740675 | 0.747381 |
| AR @100 | 0.684871 | 0.683395 |
| Total predictions | 414 | 419 |
| Mean sliced inference | 188.629 ms | 193.373 ms |
