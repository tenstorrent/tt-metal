# Real-robot demo (`demo/`)

Drive a real robot with the pi0.5 policy running on **TT kernels** — the same
policy validated in the LIBERO simulator, with the simulator swapped for live
cameras + a real arm.

> **Note on "using the LIBERO sim with real cameras":** LIBERO *is* the simulator
> — it *generates* the camera images. You don't feed real cameras into LIBERO.
> Instead, the policy is already decoupled from the sim behind a clean adapter,
> so real cameras/robot replace the sim's obs/action I/O while the TT policy runs
> unchanged. That's what this folder provides.

## Contents
- `policy.py` — headless policy surface: re-exports `Pi0_5LiberoAdapter` and adds
  `build_policy()` (opens the device/mesh + builds the adapter). No LIBERO/MuJoCo needed.
- `robot_runtime.py` — `RobotInterface` (the hardware boundary), `run_realrobot`
  (the control loop, mirrors the sim's), and `MockRobot` (no-hardware stub).
- `demo_realrobot.py` — in-process CLI demo (log-only by default; `--enable-motion` to move).
- `policy_server.py` — remote policy server (TT box serves; robot POSTs obs → gets actions).

## How it works
The policy entry point is `Pi0_5LiberoAdapter.predict_chunk(agent_img, wrist_img,
state, task_desc, num_denoising_steps)`:
- **Inputs:** two `(H,W,3)` uint8 images (any resolution) + an 8-D state
  `[eef_pos(3), eef_axis_angle(3), gripper_qpos(2)]` + a task string.
- The adapter does ALL preprocessing internally (resize/pad→224, [-1,1], tokenize,
  normalize) and returns `(chunk, 7)` actions **already denormalized** into raw
  robot space: `[Δpos(3), Δaxis_angle(3), gripper]`.
- Backends: `ttnn_1x8` (8-chip mesh, trace+2CQ, ~31 ms/chunk) · `ttnn` (single chip) ·
  `pytorch` (CPU reference).

**Embodiment:** the upstream `pi05_libero` checkpoint is trained for the LIBERO
7-DoF delta-EE + gripper action space and 8-D state. A LIBERO-compatible arm works
as-is; a different robot needs its own `norm_stats` + a checkpoint trained for it.

## Quick start (no hardware — MockRobot, 1×8 mesh)
```bash
export PI05_CHECKPOINT_DIR=/home/tt-admin/pi05_cache/pi05_libero_upstream
export PI0_TOKENIZER_PATH=/home/tt-admin/pi05_cache/tokenizer/paligemma_tokenizer.model
export TT_METAL_HOME=$PWD PYTHONPATH=$PWD TT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15

python_env/bin/python models/experimental/pi0_5/demo/demo_realrobot.py \
  --checkpoint $PI05_CHECKPOINT_DIR --backend ttnn_1x8 \
  --task "pick up the black bowl" --steps 40
# Runs the full policy loop against MockRobot (log-only): the first chunk captures
# the trace (slower), then replays at ~31 ms/chunk.
```

## Driving a real robot
1. Implement `RobotInterface` (in `robot_runtime.py`) for your arm + cameras:
   - `capture()` → `(agentview_img, wrist_img)` uint8, **upright** (training orientation);
     handle any camera-mount rotation here.
   - `get_state()` → 8-D `[eef_pos(3), eef_axis_angle(3), gripper_qpos(2)]`
     (compute the EEF axis-angle from your arm's pose quaternion).
   - `send_action(a)` → command one 7-D delta-EE + gripper action.
   - `reset()` / `is_done()` as needed.
2. In `demo_realrobot.py`, replace `MockRobot()` with your implementation.
3. Dry-run **log-only** first (default), confirm the printed actions look sane, then
   add `--enable-motion` to command the arm.

Camera/robot bindings (RealSense/USB via OpenCV, ROS topics, or a vendor SDK) live
entirely inside your `RobotInterface` — the policy + loop don't change.

## Remote deployment (robot on a different machine)
Start the server on the TT box:
```bash
python_env/bin/python models/experimental/pi0_5/demo/policy_server.py \
  --checkpoint $PI05_CHECKPOINT_DIR --backend ttnn_1x8 --port 8000
```
The robot POSTs `/predict` with `{agent_image, wrist_image, state, task}` (images as
base64 uint8 + shape) and receives `{"actions": [[7 floats], ...]}`. `GET /health`
reports readiness. For an openpi-native robot stack, swap the transport for openpi's
websocket policy-server protocol — reuse the same `predict_chunk`.

## Safety
`--enable-motion` is required to command the arm; the default is log-only. Always
dry-run first and keep an e-stop within reach.
