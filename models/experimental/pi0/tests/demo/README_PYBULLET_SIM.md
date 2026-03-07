# PI0 PyBullet Simulation

Real-time robotics simulation demonstrating PI0 inference with PyBullet physics engine. This script runs closed-loop control of a Franka Panda robot (7-DOF) using the PI0 vision-language-action model on Tenstorrent hardware.

## Important Note: Tokenizer Selection

**By default, this script uses `SimpleRoboticsTokenizer` (word-based, no setup required).**

To use the official Gemma tokenizer (better task understanding), you must:
1. Set up HuggingFace authentication (see Tokenizer Setup section)
2. **Add the `--use-gemma-tokenizer` flag** to your command

```bash
# Default: Uses SimpleRoboticsTokenizer
python run_pybullet_sim.py --steps 400

# With Gemma: Add the flag
python run_pybullet_sim.py --steps 400 --use-gemma-tokenizer
```

---

## Overview

**What it does:**
- Simulates a Franka Panda robot arm manipulating a cube in PyBullet environment
- Captures multi-view RGB observations (2 cameras at 224x224 or custom resolution)
- Runs PI0 model inference on TT device for action prediction
- **Action buffering** for smooth, efficient control (executes multiple predicted actions before re-planning)
- **Delta action control** (default) for stable, oscillation-free motion
- Applies actions to robot in real-time control loop
- **Spatial tracking** monitors distance to target cube
- Records high-quality video (720p @ 20fps)
- Tracks detailed performance metrics (inference time, control frequency, timing breakdowns)

**Robot:** Franka Emika Panda (7-DOF collaborative arm) - same robot used in PI0's training data!

**Key Features:**
- ✅ **Action buffering** reduces oscillation and improves task completion
- ✅ **Configurable re-planning** interval (5-30 actions) for smooth motion
- ✅ **10-25 Hz control frequency** (up from 2-3 Hz) with optimization
- ✅ **Delta action mode** (default) for smoother, more stable motion
- ✅ **Spatial tracking** monitors end-effector distance to cube in real-time
- ✅ **Performance optimizations** (image resolution, warm-up, timing breakdowns)
- ✅ **Reproducible behavior** with random seed control
- ✅ **Diagnostic tools** for debugging trajectory and tokenization issues
- ✅ **Two tokenizer options:**
  - **SimpleRoboticsTokenizer** (default, no setup required)
  - **Gemma tokenizer** (opt-in with `--use-gemma-tokenizer` flag, best quality)

---

## Requirements

### System Dependencies
```bash
# Virtual display (for headless video recording)
sudo apt-get install xvfb

# Python packages (should already be installed in your environment)
pip install pybullet numpy torch imageio[ffmpeg]
```

### Environment Variables
```bash
# Required: TT-Metal installation path
export TT_METAL_HOME=/path/to/tt-metal

# Optional: HuggingFace token (for Gemma tokenizer)
export HUGGING_FACE_HUB_TOKEN="your_token_here"
```

### Model Weights
```bash
# Download PI0 pretrained weights (if not already done)
python tests/download_pretrained_weights.py
```

---

## Command-Line Arguments

### Basic Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--checkpoint` | string | `$TT_METAL_HOME/models/experimental/pi0/weights/pi0_base` | Path to PI0 model weights |
| `--steps` | int | `500` | Number of simulation steps to run |
| `--task` | string | `"pick and place"` | Natural language task instruction for the robot |

### Display & Recording

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--headless` | flag | `False` | Run without GUI window (no visual display) |
| `--record-video` | flag | `False` | Enable video recording of the simulation |
| `--video-path` | string | `pi0_simulation_{timestamp}.mp4` | Custom path for output video file |

### Hardware & Performance

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--device` | int | `0` | Tenstorrent device ID to use for inference |
| `--image-size` | int | `224` | Image resolution for observations (NxN pixels). Lower values = faster inference. Try 160 or 112 for speed. |
| `--replan-interval` | int | `5` | Number of actions to execute before re-planning (1=original behavior, 5-10=smoother motion, 20-30=maximum speed) |

### Motion Control

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--use-delta-actions` | flag | `True` | Use incremental position changes (default, smoother motion with less oscillation) |
| `--use-absolute-actions` | flag | `False` | Use absolute position control instead of delta (overrides --use-delta-actions) |
| `--max-velocity` | float | `0.5` | Maximum joint velocity in rad/s (lower = smoother but slower, higher = faster but may overshoot) |

### Behavior & Debugging

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--demo-mode` | flag | `False` | Use scripted sinusoidal motion instead of PI0 predictions (good for testing robot/sim works) |
| `--verbose-actions` | flag | `False` | Print predicted and scaled actions every 50 steps |
| `--seed` | int | `42` | Random seed for reproducibility (controls diffusion sampling) |

### Advanced

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--use-gemma-tokenizer` | flag | `False` (off) | **REQUIRED flag** to use official Gemma tokenizer. Default is SimpleRoboticsTokenizer. Requires HuggingFace authentication for google/gemma-2b. |

---

## Usage Examples

### 1. **Basic Run (With GUI)**
```bash
python run_pybullet_sim.py
```
- Shows PyBullet GUI window
- Runs for 500 steps with default task "pick and place"
- No video recording

### 2. **Headless with Video Recording** (Recommended)
```bash
xvfb-run -a python run_pybullet_sim.py --headless --record-video --steps 400
```
- No GUI (faster)
- Records 720p video at 20 FPS
- Saves as `pi0_simulation_YYYYMMDD_HHMMSS.mp4`

### 3. **Custom Task**
```bash
xvfb-run -a python run_pybullet_sim.py \
  --headless --record-video --steps 400 --task "pick up cube"
```
- Gives robot specific task instruction
- Other task examples: "grasp object", "move left", "reach forward"

### 4. **Debug Mode (See Actions)**
```bash
xvfb-run -a python run_pybullet_sim.py \
  --headless --verbose-actions --steps 200
```
- Prints predicted actions every 50 steps
- Shows action range, mean, std
- Shows scaled actions (normalized → radians)

### 5. **Demo Mode (Guaranteed Motion)**
```bash
xvfb-run -a python run_pybullet_sim.py \
  --demo-mode --record-video --steps 800
```
- Uses scripted sinusoidal motion (not PI0)
- Good for verifying robot/simulation works correctly
- Produces smooth, predictable movements

### 6. **Reproducible Behavior**
```bash
xvfb-run -a python run_pybullet_sim.py \
  --headless --record-video --steps 400 --seed 42
```
- Same seed = same behavior every run
- Try different seeds (1-100) to find good trajectories
- Useful for demos and debugging

### 7. **With Official Gemma Tokenizer** (Requires Flag!)
```bash
# First: Set up HuggingFace authentication (see Tokenizer Setup below)
# Then: Add --use-gemma-tokenizer flag

xvfb-run -a python run_pybullet_sim.py \
  --headless --record-video --steps 400 \
  --task "pick up cube" --use-gemma-tokenizer
```
- ⚠️ **Must include `--use-gemma-tokenizer` flag** to use Gemma tokenizer
- Uses authentic Gemma SentencePiece tokenizer
- Best task understanding (matches PI0's training)
- Requires HuggingFace access to google/gemma-2b
- Without flag: Uses SimpleRoboticsTokenizer (default)

### 8. **Custom Video Output**
```bash
xvfb-run -a python run_pybullet_sim.py \
  --headless --record-video --video-path my_demo.mp4 --steps 1000
```
- Saves video with custom filename
- 1000 steps ≈ 50-67 seconds of video (depends on control frequency)

### 9. **Multiple Device Setup**
```bash
xvfb-run -a python run_pybullet_sim.py \
  --device 1 --headless --record-video
```
- Uses TT device ID 1 instead of default 0
- Useful if you have multiple Tenstorrent accelerators

### 10. **Reduced Re-planning for Smoother Motion**
```bash
xvfb-run -a python run_pybullet_sim.py \
  --headless --record-video --steps 1000 \
  --replan-interval 10 --seed 42
```
- Executes 10 actions before re-planning (instead of 1)
- **Reduces oscillation/wiggling** near targets
- **5-10x faster** control loop (fewer inference calls)
- Better temporal consistency in robot motion
- Recommended for tasks requiring precise movements

**Comparison:**
- `--replan-interval 1`: Original behavior (re-plan every step, may oscillate)
- `--replan-interval 5`: Balanced (default, good for most tasks)
- `--replan-interval 10`: Smooth motion (best for precise tasks, less reactive)
- `--replan-interval 20`: Very committed plans (may be slow to adapt)

### 11. **Performance Optimization** (New!)
```bash
# Fast mode: 20-25 Hz control frequency
xvfb-run -a python run_pybullet_sim.py \
  --headless --record-video --steps 400 \
  --replan-interval 30 --image-size 160

# Balanced mode: 10-15 Hz, good quality
xvfb-run -a python run_pybullet_sim.py \
  --headless --record-video --steps 400 \
  --replan-interval 20 --image-size 224
```
- **--replan-interval 20-30**: Run inference less frequently
- **--image-size 160 or 112**: Lower resolution = faster inference
- Dramatically improves loop frequency (from 2-3 Hz to 10-25 Hz)
- See `PERFORMANCE_GUIDE.md` for detailed optimization strategies

### 12. **Absolute vs Delta Action Control**
```bash
# Delta mode (default): Incremental position changes
xvfb-run -a python run_pybullet_sim.py \
  --headless --record-video --use-delta-actions

# Absolute mode: Direct position targets
xvfb-run -a python run_pybullet_sim.py \
  --headless --record-video --use-absolute-actions

# Adjust motion speed
xvfb-run -a python run_pybullet_sim.py \
  --headless --record-video --max-velocity 1.0
```
- **Delta actions (default)**: Smoother, reduces oscillation, recommended
- **Absolute actions**: Original behavior, may be less stable
- **max-velocity**: Higher = faster motion, lower = smoother/safer

---

## Tokenizer Setup

The script supports two tokenization methods:

### Option 1: SimpleRoboticsTokenizer (Default - Always Active Unless You Use Flag)
- **No setup required** - works immediately
- **Active by default** - used unless you add `--use-gemma-tokenizer` flag
- Word-based tokenizer with robotics vocabulary
- Semantic encoding for: pick, place, grasp, move, cube, etc.
- Better than character-based, but not identical to PI0's training
- **Recommended for:** Quick testing, no authentication hassles

### Option 2: Official Gemma Tokenizer (Best Quality - Opt-In)
**⚠️ Important: You MUST add `--use-gemma-tokenizer` flag to use this tokenizer!**

**Requires HuggingFace authentication:**

1. **Request access:**
   - Go to https://huggingface.co/google/gemma-2b
   - Click "Agree and access repository"
   - Wait for approval (usually instant)

2. **Create token:**
   - Go to https://huggingface.co/settings/tokens
   - Create a "Read" token
   - Copy the token

3. **Authenticate:**
   ```bash
   huggingface-cli login
   # Paste your token when prompted
   ```

4. **Run with `--use-gemma-tokenizer` flag (REQUIRED):**
   ```bash
   python run_pybullet_sim.py --use-gemma-tokenizer
   ```

**Without the flag, the script will use SimpleRoboticsTokenizer by default, even if you have HuggingFace authentication set up.**

**How to verify it's working:**
You should see this in the output:
```
📝 Initializing tokenizer...
   ✅ Loaded official Gemma tokenizer from google/gemma-2b

Tokenizer: ✅ Official Gemma (SentencePiece)
```

**Benefits of Gemma tokenizer:**
- ✅ Matches PI0's training exactly
- ✅ Better task understanding
- ✅ More consistent behavior
- ✅ Improved semantic parsing of complex instructions

---

## Output & Performance

### Console Output
```
======================================================================
   PI0 REAL-TIME SIMULATION - Franka Panda Robot (7-DOF)
======================================================================
   Random seed: 42 (for reproducibility)

📦 Initializing PyBullet...
   ✅ Franka Panda loaded: 7 arm joints (7-DOF)
   🤖 This robot was used in PI0's training data!

📦 Loading PI0 model...
   ✅ Model loaded successfully

📹 Video recording enabled: pi0_simulation_20260226_012345.mp4

======================================================================
🚀 Running episode: 'pick up cube'
   Robot: Franka Panda (7-DOF) - trained in PI0 dataset!
   Steps: 400
   Robot DOF: 7 (using first 7 of 32 predicted actions)
   State: 14-dim (7 pos + 7 vel), padded to 32
   Action horizon: 50
   Image size: 224x224 (lower = faster inference)
   Re-plan interval: 20 (execute 20 actions before re-planning)
   Action mode: ✅ Delta (incremental)
   Max velocity: 0.5 rad/s
   Random seed: 42
   Tokenizer: ✅ Official Gemma (SentencePiece)
======================================================================

⏳ Warming up model (first inference includes JIT compilation)...
   Running 3 warm-up iterations to ensure full compilation...
   Warm-up 1/3: 450.0ms
   Warm-up 2/3: 340.0ms
   Warm-up 3/3: 335.0ms
✅ Warm-up complete! Starting control loop...

   🔍 Task tokenization debug:
      Prompt: 'pick up cube'
      Tokens: [2, 100, 403, 200]... (first 10)
      Mask: [True, True, True, True]... (first 10)
      Vocab size: 256000

   🔄 Re-planning at step 5 (new action buffer)
Step    0 | Cap: 90.3ms | Prep: 4.4ms | Inf: 334.5ms | Loop: 430.7ms | Freq: 2.3 Hz | Inferences: 1/1
   📍 EE pos: [0.307, 0.000, 0.487]
   🎯 Cube: [0.500, 0.000, 0.025]
   📏 Distance to cube: 0.523m
   🔄 Re-planning at step 10 (new action buffer)
Step   50 | Cap: 88.9ms | Prep: 3.3ms | Inf: 332.0ms | Loop: 85.2ms | Freq: 11.7 Hz | Inferences: 10/50
Step  100 | Cap: 88.5ms | Prep: 3.1ms | Inf: 330.8ms | Loop: 82.3ms | Freq: 12.2 Hz | Inferences: 20/100
   📍 EE pos: [0.398, 0.012, 0.352]
   🎯 Cube: [0.500, 0.000, 0.025]
   📏 Distance to cube: 0.342m
...

======================================================================
✅ Episode complete!
======================================================================

📊 Performance Summary:
   Total steps:      400
   Inference steps:  80 (20.0%)
   Buffered steps:   320 (80.0%)
   Re-plan interval: 5

   Capture time:     88.90 ± 0.67 ms
   Preprocess time:  2.79 ± 4.74 ms (when inferencing)
   Inference time:   331.58 ± 7.09 ms (when inferencing)
   Total loop time:  85.43 ± 120.31 ms
   Control frequency: 11.7 Hz

   Min inference:    327.81 ms
   Max inference:    370.58 ms

📹 Saving video to pi0_simulation_20260226_012345.mp4...
   ✅ Video saved successfully!
   Duration: 20.0 seconds
```

### Performance Metrics Explained

| Metric | Typical Value | Description |
|--------|---------------|-------------|
| **Inference steps** | 80 (20%) | Number of steps that ran model inference |
| **Buffered steps** | 320 (80%) | Number of steps using buffered actions (with replan_interval=5) |
| **Capture time** | ~90ms | Time to capture RGB images from 2 cameras |
| **Preprocess time** | ~3ms | Convert images/state/tokens to TTNN tensors (only when inferencing) |
| **Inference time** | ~330ms | PI0 model inference on TT device (only when inferencing) |
| **Loop time** | ~85ms avg | Total time for one control step (much lower with buffering!) |
| **Control frequency** | ~11.7 Hz | Steps per second (5x faster with replan_interval=5) |
| **Inferences** | 10/50 | Number of inference calls vs total steps (shown every 50 steps) |
| **EE pos** | [x, y, z] | End-effector position in world coordinates (printed every 100 steps) |
| **Cube pos** | [0.5, 0.0, 0.025] | Cube position on table surface |
| **Distance to cube** | varies | Euclidean distance from end-effector to cube (should decrease!) |

### Spatial Tracking (New!)

Every 100 steps, the simulation prints spatial information:

```
📍 EE pos: [0.398, 0.012, 0.352]  ← End-effector position
🎯 Cube: [0.500, 0.000, 0.025]    ← Target cube position
📏 Distance to cube: 0.342m        ← Should DECREASE over time
```

**How to use this:**
- **Distance decreasing:** ✅ Robot moving toward cube (trajectory correct)
- **Distance constant/increasing:** ❌ Wrong trajectory (see troubleshooting)
- **Distance reaches ~0.05m:** Robot very close to cube
- **Compare over time:** Step 0 vs Step 100 vs Step 200, etc.

### Video Output
- **Format:** MP4 (H.264 codec)
- **Resolution:** 1280x720 (720p)
- **Frame rate:** 20 FPS
- **Duration:** `num_steps / control_frequency` seconds
- **Camera angle:** Optimized view of Franka Panda workspace

---

## Troubleshooting

### Issue: "PyBullet not installed"
**Solution:**
```bash
pip install pybullet
```

### Issue: "imageio not installed" or "Video save failed"
**Solution:**
```bash
pip install imageio[ffmpeg]
```

### Issue: "Cannot access gated repo google/gemma-2b"
**Solution:**
- Make sure you requested access at https://huggingface.co/google/gemma-2b
- Run `huggingface-cli login` with a valid token
- Or run without `--use-gemma-tokenizer` flag to use fallback

### Issue: "I have HF authentication but it's still using SimpleRoboticsTokenizer"
**Solution:**
You must **explicitly add** the `--use-gemma-tokenizer` flag to your command!

```bash
# Wrong (uses SimpleRoboticsTokenizer by default)
xvfb-run -a python run_pybullet_sim.py --headless --record-video

# Correct (uses Gemma tokenizer)
xvfb-run -a python run_pybullet_sim.py --headless --record-video --use-gemma-tokenizer
```

**Verify it's working:** Look for this in the output:
```
Tokenizer: ✅ Official Gemma (SentencePiece)
```

If you see `Tokenizer: SimpleRoboticsTokenizer (word-based)`, the flag is missing!

### Issue: "TT_METAL_HOME environment variable not set"
**Solution:**
```bash
export TT_METAL_HOME=/path/to/your/tt-metal
```

### Issue: "Checkpoint not found"
**Solution:**
```bash
# Download pretrained weights
cd $TT_METAL_HOME
python models/experimental/pi0/tests/download_pretrained_weights.py
```

### Issue: Robot barely moves in video
**Possible causes:**
1. **Action scaling issue** - Should be fixed in latest version
2. **Poor random seed** - Try different seeds (--seed 1, --seed 2, etc.)
3. **Short run** - Try 400-800 steps for better motion
4. **Verify with demo mode:**
   ```bash
   python run_pybullet_sim.py --demo-mode --record-video
   ```
   If robot moves well in demo mode, the simulation works correctly.

### Issue: Robot reaches midway to target but starts wiggling/oscillating
**This is a common issue with frequent re-planning!**

**Symptoms:**
- Robot moves toward goal initially
- Gets close to target but then oscillates/wiggles
- Never completes the final approach

**Solution:**
Increase the `--replan-interval` to reduce oscillation:
```bash
# Try interval of 10 (execute 10 actions before re-planning)
xvfb-run -a python run_pybullet_sim.py \
  --headless --record-video --steps 1000 \
  --replan-interval 10 --seed 42
```

**Why this happens:**
- Re-planning every step causes the robot to constantly change its approach
- Near the target, small observation changes cause action oscillations
- Action buffering provides temporal consistency and smoother motion

**Recommended values:**
- `--replan-interval 5`: Default, good balance
- `--replan-interval 10`: Better for precise tasks with oscillation
- `--replan-interval 1`: Original behavior (may oscillate)

### Issue: Robot takes wrong trajectory to cube / doesn't understand task
**This is likely a tokenization issue!**

**Symptoms:**
- Robot moves smoothly but goes in wrong direction
- Robot doesn't approach the cube
- Distance to cube stays constant or increases

**Solution 1: Use official Gemma tokenizer (recommended)**
```bash
# Install transformers and authenticate
pip install transformers
huggingface-cli login

# Run with official tokenizer
xvfb-run -a python run_pybullet_sim.py \
  --headless --record-video --use-gemma-tokenizer --task "pick cube"
```

**Solution 2: Try different task prompts**
```bash
# Simple prompts with basic vocabulary
xvfb-run -a python run_pybullet_sim.py --task "pick cube"
xvfb-run -a python run_pybullet_sim.py --task "grasp object"
xvfb-run -a python run_pybullet_sim.py --task "reach forward"
```

**Solution 3: Test tokenization**
```bash
cd demo/
python test_tokenization.py
```
This shows how your task is being tokenized and helps identify vocabulary issues.

**Solution 4: Visualize camera views**
```bash
cd demo/
python visualize_cameras.py --image-size 224
```
Check if the cube is visible in the saved images (./camera_debug/).

**Why this happens:**
- PI0 was trained with Gemma's SentencePiece tokenizer
- SimpleRoboticsTokenizer (fallback) uses different token IDs
- Model may not understand the task correctly with wrong tokenization

**See also:** `TROUBLESHOOTING_TRAJECTORY.md` for detailed debugging steps

### Issue: Simulation runs too slowly (< 5 Hz)
**This is expected behavior for the full model!**

**Current performance:**
- Inference takes ~330ms per call
- With `--replan-interval 5` (default): ~10-12 Hz average
- With `--replan-interval 1`: ~2-3 Hz (very slow)

**Solution: Optimize performance**
```bash
# Fast mode (20-25 Hz)
xvfb-run -a python run_pybullet_sim.py \
  --replan-interval 30 --image-size 160 --headless

# Balanced mode (10-15 Hz)
xvfb-run -a python run_pybullet_sim.py \
  --replan-interval 20 --image-size 224
```

**Performance trade-offs:**
- Higher `--replan-interval`: Faster but less reactive to changes
- Lower `--image-size`: Faster inference but less visual detail
- `--headless`: Removes GUI rendering overhead

**See also:** `PERFORMANCE_GUIDE.md` for comprehensive optimization strategies

### Issue: Video recording fails on headless server
**Solution:**
```bash
# Use xvfb-run to provide virtual display
xvfb-run -a python run_pybullet_sim.py --headless --record-video
```

### Issue: "No display" error when recording
**Solution:**
Install Xvfb:
```bash
sudo apt-get update
sudo apt-get install xvfb
```

---

## Tips & Best Practices

### For Demonstrations
- **400-800 steps** - Good balance of motion without drift
- **Use seed 42** or test seeds 1-10 to find best behavior
- **Record video** - Much easier to share than live GUI
- **Try demo mode first** - Verify simulation works before testing PI0
- **Use replan-interval 20** - Smooth motion, 10-15 Hz control frequency
- **Use image-size 160-224** - Good quality without excessive slowdown

### For Debugging
- **Use --verbose-actions** - See what PI0 is predicting
- **Monitor distance to cube** - Watch console output every 100 steps
- **Test tokenization** - Run `python test_tokenization.py` to check prompt encoding
- **Visualize cameras** - Run `python visualize_cameras.py` to see robot's view
- **Start with 200 steps** - Faster iteration
- **Test different tasks** - "pick cube", "grasp object", "reach forward"
- **Check spatial tracking** - Distance to cube should decrease over time

### For Development
- **Use SimpleRoboticsTokenizer** - No auth required, good enough for testing
- **Fix the seed** - Reproducibility is critical for debugging
- **Profile performance** - Check which operations are slow
- **Test on single device** - Use --device 0 for consistent results
- **Read diagnostic guides** - See TROUBLESHOOTING_TRAJECTORY.md and PERFORMANCE_GUIDE.md

### For Performance
- **Replan-interval 20-30** - Dramatically improves control frequency
- **Image-size 160** - Best balance of speed and quality
- **Headless mode** - Remove GUI overhead
- **Expected performance:**
  - Default (replan=5, img=224): ~10-12 Hz
  - Optimized (replan=20, img=160): ~15-20 Hz
  - Maximum (replan=30, img=112): ~20-25 Hz

### Recommended Starting Command (Fast & Simple)
```bash
xvfb-run -a python run_pybullet_sim.py \
  --headless --record-video --steps 400 \
  --task "pick cube" --seed 42 \
  --replan-interval 20 --image-size 160
```
**Performance:** 10-15 Hz control frequency
**Features:** Video output, smooth motion, reproducible, fast
**Tokenizer:** SimpleRoboticsTokenizer (no HF auth required)

### Recommended Starting Command (With Gemma Tokenizer)
```bash
xvfb-run -a python run_pybullet_sim.py \
  --headless --record-video --steps 400 \
  --task "pick cube" --seed 42 \
  --replan-interval 20 --image-size 224 \
  --use-gemma-tokenizer
```
**Performance:** 10-15 Hz control frequency
**Features:** Best task understanding, smooth motion, reproducible
**Tokenizer:** Official Gemma (requires HuggingFace authentication)

### For Tasks with Oscillation Issues
```bash
xvfb-run -a python run_pybullet_sim.py \
  --headless --record-video --steps 1000 \
  --task "pick cube" --seed 42 \
  --replan-interval 10
```
Higher re-plan interval (10) provides smoother, more committed motion toward targets.

### For Trajectory Debugging
```bash
# Monitor distance to cube (printed every 100 steps)
python run_pybullet_sim.py --task "pick cube" \
  --replan-interval 10 --steps 400

# Test tokenization
cd demo/ && python test_tokenization.py

# Visualize camera views
cd demo/ && python visualize_cameras.py --image-size 224

# Try different prompts
for prompt in "pick cube" "grasp object" "reach forward"; do
  python run_pybullet_sim.py --task "$prompt" --steps 200
done
```

---

## Technical Details

### Simulation Architecture

1. **Environment:** PyBullet physics engine with Franka Panda URDF
   - Robot: Franka Panda (7-DOF) at origin [0, 0, 0]
   - Cube: Small cube at [0.5, 0.0, 0.025] (on table surface, reachable)
   - Plane: Ground plane for physics

2. **Observations:**
   - 2x RGB images (configurable resolution, default 224x224)
   - Normalized to [-1, 1] for SigLIP vision encoder
   - Camera 1: Front view at [1.0, 0.0, 0.5]
   - Camera 2: Side view at [0.3, 1.0, 0.5]
   - Robot state: 14-dim (7 joint positions + 7 velocities), padded to 32-dim

3. **Actions:**
   - PI0 predicts 32-dim actions at 50-step horizon
   - First 7 dimensions used for 7-DOF arm
   - **Delta mode (default):** Actions are position changes scaled by 0.3
   - **Absolute mode:** Actions normalized from [-1, 1] to joint limits
   - Position control with configurable max velocity (default: 0.5 rad/s)

4. **Inference:**
   - TTNN-optimized PI0 model on TT device
   - Diffusion-based action generation (flow matching)
   - Multiple warm-up iterations ensure full compilation
   - ~330ms per inference call on TT hardware

### Action Buffering & Re-planning Strategy

**Traditional Receding Horizon (replan_interval=1):**
```
Step 0: Observe → Infer 50 actions → Execute action[0] → Discard [1-49]
Step 1: Observe → Infer 50 actions → Execute action[0] → Discard [1-49]
Step 2: Observe → Infer 50 actions → Execute action[0] → Discard [1-49]
...
Problem: Wasteful, can cause oscillation near targets
```

**Action Buffering (replan_interval=5, default):**
```
Step 0: Observe → Infer 50 actions → Buffer [0-49] → Execute action[0]
Step 1: (buffered) → Execute action[1]
Step 2: (buffered) → Execute action[2]
Step 3: (buffered) → Execute action[3]
Step 4: (buffered) → Execute action[4]
Step 5: Observe → Infer 50 actions → Buffer [0-49] → Execute action[0]
...
Benefits: 5x fewer inferences, smoother motion, better temporal consistency
```

**Why This Helps:**
- ✅ **Reduces oscillation** - Robot commits to short-term plans instead of constantly changing its mind
- ✅ **Faster execution** - 5-10x speedup (only 20% of steps need inference)
- ✅ **Better use of predictions** - PI0 predicts 50 steps ahead; we should use more than just the first
- ✅ **Temporal consistency** - Multi-step action sequences execute as planned
- ✅ **Reduced wiggling** - Especially helpful when robot is near target objects

### Action Space Details

**Two action modes:**

1. **Delta Actions (default, `--use-delta-actions`):**
   - Actions interpreted as position changes (deltas)
   - `new_position = current_position + (action * 0.3)`
   - Smoother motion, reduces oscillation
   - Better for precise manipulation tasks
   - Automatically clamped to joint limits

2. **Absolute Actions (`--use-absolute-actions`):**
   - Actions interpreted as target positions
   - Linear mapping from [-1, 1] to joint limits
   - Example: -0.16 → -0.46 rad ≈ -26 degrees
   - Original behavior, may be less stable

**Control parameters:**
- **Type:** Position control with velocity limits
- **Force limit:** 87N (Franka Panda max)
- **Max velocity:** Configurable via `--max-velocity` (default: 0.5 rad/s)
- **Safety:** All targets clamped to joint limits

### Random Seed Behavior
- Seeds control **diffusion sampling** during action generation
- Same seed = same trajectory (deterministic)
- Different seeds = different behaviors (stochastic exploration)
- **Why it matters:** PI0 uses flow matching for actions, which involves random noise

### Understanding "Steps"

**What does `--steps 1000` mean?**

`steps` refers to **control loop iterations**, not unique actions:

```
steps=1000 means:
├─ 1000 control loop iterations
├─ 1000 robot commands applied to simulation
├─ But NOT 1000 unique PI0 inference calls
└─ Actual inferences depend on --replan-interval
```

**Example with replan_interval=5:**
- Total steps: 1000
- Inference calls: 200 (1000 ÷ 5)
- Buffered actions used: 800

**Example with replan_interval=1 (original):**
- Total steps: 1000
- Inference calls: 1000 (re-plan every step)
- Buffered actions used: 0

**Control loop at each step:**
1. Check if re-planning needed (based on replan_interval)
2. If yes: Capture observations → Run PI0 → Buffer 50 actions
3. If no: Use next action from buffer
4. Apply action to robot
5. Step physics simulation
6. Repeat

**Duration calculation:**
- With replan_interval=5: ~1000 steps ÷ 11.7 Hz ≈ 85 seconds
- With replan_interval=1: ~1000 steps ÷ 2.4 Hz ≈ 417 seconds

---

## License & Commercial Use

### Gemma Tokenizer (Optional)
- **License:** Gemma Terms of Use
- **Commercial use:** ✅ Allowed with attribution
- **Requirements:** Include license, pass-through restrictions
- **More info:** https://ai.google.dev/gemma/terms

### Alternative (No License Concerns)
- Use default `SimpleRoboticsTokenizer` (custom implementation)
- No external dependencies or licensing restrictions
- Good semantic understanding for robotics tasks

---

## Contact & Support

For issues specific to this simulation:
- Check the troubleshooting section above
- Review the console output for error messages
- Test with `--demo-mode` to isolate PI0 vs simulation issues

For PI0 model or TT-Metal issues:
- Refer to main PI0 documentation
- Check TT-Metal installation and device status

---

## Quick Reference Card

```bash
# Minimal command (GUI, no recording, uses SimpleRoboticsTokenizer)
python run_pybullet_sim.py

# RECOMMENDED: Fast & high quality (10-15 Hz, SimpleRoboticsTokenizer)
xvfb-run -a python run_pybullet_sim.py --headless --record-video \
  --steps 400 --replan-interval 20 --image-size 160 \
  --task "pick cube" --seed 42

# Maximum performance (20-25 Hz, SimpleRoboticsTokenizer)
xvfb-run -a python run_pybullet_sim.py --headless --record-video \
  --steps 400 --replan-interval 30 --image-size 112 --seed 42

# Best quality - WITH Gemma tokenizer (REQUIRES FLAG!)
xvfb-run -a python run_pybullet_sim.py --headless --record-video \
  --steps 400 --use-gemma-tokenizer --replan-interval 20 \
  --image-size 224 --task "pick cube" --seed 42

# Debug command (verbose, short run, spatial tracking)
python run_pybullet_sim.py --verbose-actions --steps 200 \
  --replan-interval 5

# Demo command (scripted motion, verify sim works)
python run_pybullet_sim.py --demo-mode --record-video

# For oscillation issues (more committed motion)
xvfb-run -a python run_pybullet_sim.py --headless --record-video \
  --steps 1000 --replan-interval 10 --seed 42

# Absolute positioning mode (if delta actions cause issues)
xvfb-run -a python run_pybullet_sim.py --headless --record-video \
  --use-absolute-actions --steps 400

# Test different tokenization approaches
cd demo/ && python test_tokenization.py

# Visualize camera views
cd demo/ && python visualize_cameras.py --image-size 224

# Original behavior (slow, may oscillate)
xvfb-run -a python run_pybullet_sim.py --headless --record-video \
  --steps 400 --replan-interval 1 --image-size 224
```

---

## Recent Updates

### 2026-02-27 - Performance Optimizations & Diagnostic Tools

**Performance Impact Summary:**

| Configuration | Control Frequency | Notes |
|--------------|------------------|-------|
| **Before** (replan=1, img=224) | 2-3 Hz | Very slow, may oscillate |
| **Default** (replan=5, img=224) | 10-12 Hz | Good balance |
| **Optimized** (replan=20, img=160) | 15-20 Hz | Recommended |
| **Maximum** (replan=30, img=112) | 20-25 Hz | Fastest |

**Performance Improvements:**
- **Image resolution control:** Added `--image-size` parameter (default: 224)
  - Try `--image-size 160` or `--image-size 112` for faster inference
  - 20-40% speedup with acceptable quality trade-off
- **Improved warm-up:** Now runs 3 warm-up iterations (was 1)
  - Ensures TTNN ops are fully compiled before control loop
  - Shows timing for each warm-up iteration
- **Performance monitoring:** Added detailed timing breakdown in output
  - Shows inference time separately from buffered steps
  - Tracks capture, preprocessing, and inference times independently

**Bug Fixes:**
- **Fixed cube position:** Changed from `[0.5, 0.0, 0.5]` (floating!) to `[0.5, 0.0, 0.025]` (on table)
  - This was causing robot to reach for wrong location
  - Cube now properly placed on table surface

**Motion Control:**
- **Delta actions enabled by default:** Uses incremental position changes
  - Smoother motion, reduces oscillation
  - Can disable with `--use-absolute-actions` flag
- **Configurable max velocity:** Added `--max-velocity` parameter (default: 0.5 rad/s)
  - Lower = smoother/safer, higher = faster but may overshoot

**Diagnostic Features:**
- **Tokenization debugging:** Prints token IDs and mask on first inference
  - Helps identify if task prompt is being understood correctly
  - Shows vocabulary size and token ranges
- **Spatial tracking:** Every 100 steps, prints:
  - End-effector position
  - Cube position
  - Distance to cube (should decrease if trajectory is correct!)
- **New diagnostic tools:**
  - `test_tokenization.py` - Compare tokenizers, test different prompts
  - `visualize_cameras.py` - Save camera view images for debugging
  - `TROUBLESHOOTING_TRAJECTORY.md` - Step-by-step guide for trajectory issues
  - `PERFORMANCE_GUIDE.md` - Comprehensive optimization strategies
  - `README_FIXES.md` - Summary of all recent changes

**Recommended Command for Best Performance:**
```bash
xvfb-run -a python run_pybullet_sim.py \
  --headless --record-video --steps 400 \
  --replan-interval 20 --image-size 160 \
  --task "pick cube" --seed 42
```
This achieves 10-15 Hz control frequency (vs. 2-3 Hz before optimizations).

### 2026-02-26 - Action Buffering Feature & Tokenizer Clarification
- **Action Buffering:** Added `--replan-interval` parameter for smoother robot motion
  - Default: Execute 5 actions before re-planning (was 1)
  - **Fixes oscillation/wiggling** near target objects
  - **5-10x faster** control loop (fewer inference calls)
  - Improved temporal consistency in robot behavior
  - See "Action Buffering & Re-planning Strategy" in Technical Details

- **Tokenizer Clarification:**
  - Made it explicit that `--use-gemma-tokenizer` flag is **required** to use Gemma tokenizer
  - Default is `SimpleRoboticsTokenizer` (word-based, no auth required)
  - Gemma tokenizer is opt-in, not automatic even with HF authentication

---

## Diagnostic Tools

The demo folder now includes several diagnostic tools to help debug issues:

### test_tokenization.py
Compare how different tokenizers encode your task prompts.

```bash
cd demo/
python test_tokenization.py
```

**Use this when:**
- Robot seems to ignore task instructions
- You want to understand if tokenization is correct
- Comparing SimpleRoboticsTokenizer vs. Gemma tokenizer

### visualize_cameras.py
Capture and save what the robot "sees" from each camera.

```bash
cd demo/
python visualize_cameras.py --image-size 224
```

**Use this when:**
- Verifying cube is visible in camera views
- Checking if image resolution is sufficient
- Debugging visual input issues

**Output:** Saves RGB and depth images to `./camera_debug/`

### TROUBLESHOOTING_TRAJECTORY.md
Step-by-step guide for debugging wrong trajectories.

```bash
cat demo/TROUBLESHOOTING_TRAJECTORY.md
```

**Covers:**
- Tokenization issues
- Visual input problems
- Action scaling issues
- How to monitor distance to cube
- Testing with different prompts

### PERFORMANCE_GUIDE.md
Comprehensive guide to optimization strategies.

```bash
cat demo/PERFORMANCE_GUIDE.md
```

**Covers:**
- Why inference is slow (335ms per call)
- Quick fixes (replan-interval, image-size)
- Advanced optimizations
- Expected performance improvements
- Recommended settings by use case

---

**Last updated:** 2026-02-27
