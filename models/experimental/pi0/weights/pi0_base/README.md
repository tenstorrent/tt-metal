---
language:
- en
library_name: lerobot
pipeline_tag: robotics
tags:
- vision-language-action
- imitation-learning
- lerobot
inference: false
license: gemma
---

# π₀ (Pi0) (LeRobot)

π₀ is a Vision-Language-Action (VLA) foundation model from Physical Intelligence that jointly reasons over vision, language, and actions to control robots, serving as the base architecture that later enabled π₀.₅’s open-world generalization.


**Original paper:** π0: A Vision-Language-Action Flow Model for General Robot Controlion
**Reference implementation:** https://github.com/Physical-Intelligence/openpi
**LeRobot implementation:** Follows the original reference code for compatibility.


## Model description

- **Inputs:** images (multi-view), proprio/state, optional language instruction
- **Outputs:** continuous actions
- **Training objective:** flow matching
- **Action representation:** continuous
- **Intended use:** Base model to fine tune on your specific use case


## Quick start (inference on a real batch)

### Installation

```bash
pip install "lerobot[pi]@git+https://github.com/huggingface/lerobot.git"
```
For full installation details (including optional video dependencies such as ffmpeg for torchcodec), see the official documentation: https://huggingface.co/docs/lerobot/installation

### Load model + dataset, run `select_action`

```python
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors

# Swap this import per-policy
from lerobot.policies.pi0 import PI0Policy

# load a policy
model_id = "lerobot/pi0_base"  # <- swap checkpoint
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

policy = PI0Policy.from_pretrained(model_id).to(device).eval()

preprocess, postprocess = make_pre_post_processors(
    policy.config,
    model_id,
    preprocessor_overrides={"device_processor": {"device": str(device)}},
)
# load a lerobotdataset
dataset = LeRobotDataset("lerobot/libero")

# pick an episode
episode_index = 0

# each episode corresponds to a contiguous range of frame indices
from_idx = dataset.meta.episodes["dataset_from_index"][episode_index]
to_idx   = dataset.meta.episodes["dataset_to_index"][episode_index]

# get a single frame from that episode (e.g. the first frame)
frame_index = from_idx
frame = dict(dataset[frame_index])

batch = preprocess(sample)
with torch.inference_mode():
    pred_action = policy.select_action(frame)
    # use your policy postprocess, this post process the action
    # for instance unnormalize the actions, detokenize it etc..
    pred_action = postprocess(pred_action)
```


## Training step (loss + backward)

If you’re training / fine-tuning, you typically call `forward(...)` to get a loss and then:

```python
policy.train()
batch = dict(dataset[0])
batch = preprocess(batch)

loss, outputs = policy.forward(batch)
loss.backward()

```

> Notes:
>
> - Some policies expose `policy(**batch)` or return a dict; keep this snippet aligned with the policy API.
> - Use your trainer script (`lerobot-train`) for full training loops.


## How to train / fine-tune

```bash
lerobot-train \
  --dataset.repo_id=${HF_USER}/<dataset> \
  --output_dir=./outputs/[RUN_NAME] \
  --job_name=[RUN_NAME] \
  --policy.repo_id=${HF_USER}/<desired_policy_repo_id> \
  --policy.path=lerobot/[BASE_CHECKPOINT] \
  --policy.dtype=bfloat16 \
  --policy.device=cuda \
  --steps=100000 \
  --batch_size=4
```

Add policy-specific flags below:

- `-policy.chunk_size=...`
- `-policy.n_action_steps=...`
- `-policy.max_action_tokens=...`
- `-policy.gradient_checkpointing=true`


## Real-World Inference & Evaluation

You can use the `record` script from [**`lerobot-record`**](https://github.com/huggingface/lerobot/blob/main/src/lerobot/scripts/lerobot_record.py) with a policy checkpoint as input, to run inference and evaluate your policy.

For instance, run this command or API example to run inference and record 10 evaluation episodes:

```
lerobot-record  \
  --robot.type=so100_follower \
  --robot.port=/dev/ttyACM1 \
  --robot.cameras="{ up: {type: opencv, index_or_path: /dev/video10, width: 640, height: 480, fps: 30}, side: {type: intelrealsense, serial_number_or_name: 233522074606, width: 640, height: 480, fps: 30}}" \
  --robot.id=my_awesome_follower_arm \
  --display_data=false \
  --dataset.repo_id=${HF_USER}/eval_so100 \
  --dataset.single_task="Put lego brick into the transparent box" \
  # <- Teleop optional if you want to teleoperate in between episodes \
  # --teleop.type=so100_leader \
  # --teleop.port=/dev/ttyACM0 \
  # --teleop.id=my_awesome_leader_arm \
  --policy.path=${HF_USER}/my_policy
```
