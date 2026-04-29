# X-VLA evaluation harness

Two kinds of evaluation live here. The fixed `benchmark/run_benchmark.py`
(the metric oracle for optimization iterations) stays frozen; these
scripts are separate and can evolve freely.

## 1. Relative PCC — implementation fidelity vs torch fp32 reference

What it answers: *how much numerical error did the TT-NN port introduce,
holding the algorithm fixed?* Differs from `run_benchmark.py`'s PCC line
in that both the reference and the test run use the same
`num_denoising_steps` (default 10), so only precision/implementation
differences show up.

```
python3 eval/eval_relative_pcc.py \
  --backends torch_cpu,ttnn \
  --steps 10 \
  --seeds 5
```

Prints, per backend: mean / min PCC vs the fp32 torch reference, plus
max abs error on the action chunk. A healthy port is `>= 99.9%` mean PCC.

## 2. GT-dataset evaluation — action prediction error on real data

What it answers: *how well does the model match real robot actions from
a LeRobot dataset?* Input observations are taken from the dataset (so
we exercise real images and state vectors), and predicted action chunks
are compared to the dataset's ground-truth future actions.

```
# LIBERO image benchmark, 100 samples, torch fp32 reference + ttnn port
python3 eval/eval_gt_dataset.py \
  --dataset lerobot/libero_10_image \
  --num-samples 100 \
  --backends torch_cpu,ttnn
```

Metrics reported per backend:
- **MSE** and **MAE** of the predicted action chunk vs GT.
- **Per-action-step degradation** (bucketed: 1st action vs chunk-end).
- **Backend delta**: MAE(ttnn) - MAE(torch_cpu_fp32) — the portion of
  the error attributable to the port vs the model itself.

Pass `--rename-images` to map your dataset's image keys to the X-VLA
schema (`observation.images.image{,2,3}`). Two forms are accepted:

- **Short form** — rewrites the image-key SUFFIX:
  `--rename-images top=image,wrist=image2,left_wrist=image3`
  (`observation.images.top` -> `observation.images.image`, etc.)
- **Full form** — dots in either side map full keys, including
  duplicating a single camera into multiple X-VLA slots:
  `--rename-images observation.image=observation.images.image,observation.image=observation.images.image2,observation.image=observation.images.image3`

Example (pusht — single-view dataset, populate all 3 X-VLA views from
the same camera):

```
python3 eval/eval_gt_dataset.py \
  --dataset lerobot/pusht_image \
  --num-samples 100 --steps 10 --backends torch_cpu,ttnn \
  --rename-images 'observation.image=observation.images.image,observation.image=observation.images.image2,observation.image=observation.images.image3'
```

Note that base X-VLA was NOT fine-tuned on pusht, so absolute MAE will
be large. The useful signal is the delta between backends.

## 3. [FUTURE] Task-success in simulation

Strongest signal but highest setup cost: run the policy closed-loop in a
simulator (LIBERO / ALOHA sim / robosuite) and report success rate
over 50–100 episodes per task. Catches "PCC looks fine but the robot
is confused" failures that synthetic and open-loop dataset evals both miss.

Blocked on:
- Simulator install (`robosuite`, `robocasa`, or LIBERO gym wrapper).
- A language-instruction-conditioned eval loop using `policy.select_action`
  (which already handles the 30-step action chunk queue).
- Seeding / variance budget for the success-rate estimator.

Tracked as a follow-up; not implemented here.
