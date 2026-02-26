# PI0 PyBullet Simulation

Real-time robotics simulation demonstrating PI0 inference with PyBullet physics engine. This script runs closed-loop control of a Franka Panda robot (7-DOF) using the PI0 vision-language-action model on Tenstorrent hardware.

## Overview

**What it does:**
- Simulates a Franka Panda robot arm in PyBullet environment
- Captures multi-view RGB observations (2 cameras)
- Runs PI0 model inference on TT device for action prediction
- Applies actions to robot in real-time control loop
- Records video of simulation
- Tracks performance metrics (inference time, control frequency)

**Robot:** Franka Emika Panda (7-DOF collaborative arm) - same robot used in PI0's training data!

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

### Behavior & Debugging

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--demo-mode` | flag | `False` | Use scripted sinusoidal motion instead of PI0 predictions (good for testing robot/sim works) |
| `--verbose-actions` | flag | `False` | Print predicted and scaled actions every 50 steps |
| `--seed` | int | `42` | Random seed for reproducibility (controls diffusion sampling) |

### Advanced

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--use-gemma-tokenizer` | flag | `False` | Use official Gemma tokenizer instead of word-based fallback (requires HuggingFace authentication) |

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

### 7. **With Official Gemma Tokenizer**
```bash
# First: Set up HuggingFace authentication (see Tokenizer Setup below)

xvfb-run -a python run_pybullet_sim.py \
  --headless --record-video --steps 400 \
  --task "pick up cube" --use-gemma-tokenizer
```
- Uses authentic Gemma SentencePiece tokenizer
- Best task understanding (matches PI0's training)
- Requires HuggingFace access to google/gemma-2b

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

---

## Tokenizer Setup

The script supports two tokenization methods:

### Option 1: SimpleRoboticsTokenizer (Default)
- **No setup required** - works immediately
- Word-based tokenizer with robotics vocabulary
- Semantic encoding for: pick, place, grasp, move, cube, etc.
- Better than character-based, but not identical to PI0's training

### Option 2: Official Gemma Tokenizer (Best Quality)
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

4. **Run with flag:**
   ```bash
   python run_pybullet_sim.py --use-gemma-tokenizer
   ```

**Benefits of Gemma tokenizer:**
- ✅ Matches PI0's training exactly
- ✅ Better task understanding
- ✅ More consistent behavior

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
   Random seed: 42
   Tokenizer: ✅ Official Gemma (SentencePiece)
======================================================================

⏳ Warming up model (first inference includes JIT compilation)...
✅ Warm-up complete! Starting control loop...

Step    0 | Cap: 90.3ms | Prep: 4.4ms | Inf: 334.5ms | Loop: 430.7ms | Freq: 2.3 Hz
Step   50 | Cap: 88.9ms | Prep: 3.3ms | Inf: 332.0ms | Loop: 424.8ms | Freq: 2.4 Hz
...

======================================================================
✅ Episode complete!
======================================================================

📊 Performance Summary:
   Capture time:     88.90 ± 0.67 ms
   Preprocess time:  2.79 ± 4.74 ms
   Inference time:   331.58 ± 7.09 ms
   Total loop time:  423.91 ± 8.57 ms
   Control frequency: 2.4 Hz

📹 Saving video to pi0_simulation_20260226_012345.mp4...
   ✅ Video saved successfully!
   Duration: 20.0 seconds
```

### Performance Metrics Explained

| Metric | Typical Value | Description |
|--------|---------------|-------------|
| **Capture time** | ~90ms | Time to capture RGB images from 2 cameras |
| **Preprocess time** | ~3ms | Convert images/state/tokens to TTNN tensors |
| **Inference time** | ~330ms | PI0 model inference on TT device |
| **Loop time** | ~425ms | Total time for one control step |
| **Control frequency** | ~2.4 Hz | Steps per second |

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

### For Debugging
- **Use --verbose-actions** - See what PI0 is predicting
- **Start with 200 steps** - Faster iteration
- **Test different tasks** - "pick cube", "grasp object", "move left"
- **Check performance metrics** - Should see ~2-3 Hz control frequency

### For Development
- **Use SimpleRoboticsTokenizer** - No auth required, good enough for testing
- **Fix the seed** - Reproducibility is critical for debugging
- **Profile performance** - Check which operations are slow
- **Test on single device** - Use --device 0 for consistent results

### Recommended Starting Command
```bash
xvfb-run -a python run_pybullet_sim.py \
  --headless --record-video --steps 400 \
  --task "pick up cube" --verbose-actions --seed 42
```
This gives you: video output, clear motion, action debugging, reproducibility.

---

## Technical Details

### Simulation Architecture
1. **Environment:** PyBullet physics engine with Franka Panda URDF
2. **Observations:**
   - 2x RGB images (224x224, normalized to [-1, 1])
   - Robot state: 14-dim (7 joint positions + 7 velocities)
3. **Actions:**
   - PI0 predicts 32-dim actions at 50-step horizon
   - First 7 dimensions used for 7-DOF arm
   - Normalized actions scaled to joint limits (±2.87 rad)
   - Position control applied to each joint
4. **Inference:**
   - TTNN-optimized PI0 model on TT device
   - Diffusion-based action generation
   - KV caching for efficiency

### Action Space Details
- **Input:** Normalized actions from PI0 (roughly [-1, 1])
- **Scaling:** Linear mapping to joint limits
  - Example: -0.16 → -0.46 rad ≈ -26 degrees
- **Control:** Position control with force limit (87N)
- **Safety:** Clamped to Franka Panda joint limits

### Random Seed Behavior
- Seeds control **diffusion sampling** during action generation
- Same seed = same trajectory (deterministic)
- Different seeds = different behaviors (stochastic exploration)
- **Why it matters:** PI0 uses flow matching for actions, which involves random noise

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
# Minimal command (GUI, no recording)
python run_pybullet_sim.py

# Production command (headless, video, 400 steps)
xvfb-run -a python run_pybullet_sim.py --headless --record-video --steps 400

# Debug command (verbose, short run)
python run_pybullet_sim.py --verbose-actions --steps 200

# Demo command (scripted motion, verify sim works)
python run_pybullet_sim.py --demo-mode --record-video

# Best quality (Gemma tokenizer, reproducible)
xvfb-run -a python run_pybullet_sim.py --headless --record-video \
  --steps 400 --use-gemma-tokenizer --seed 42 --task "pick up cube"
```

---

**Last updated:** 2026-02-26
