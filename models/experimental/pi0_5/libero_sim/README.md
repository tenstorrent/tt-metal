# LIBERO simulator rollout (`libero_sim/`)

Runs the pi0.5 policy in the LIBERO robosuite/mujoco simulator and reports task
success. Backends: `pytorch` (CPU reference), `ttnn` (single Blackhole chip),
`ttnn_1x8` (1×8 Blackhole mesh).

Contents:
- `libero_rollout.py` — the rollout entry point (checkpoint → policy → success rate / videos).

## Setup (one-time)

```bash
export PI05_SIM=$HOME/pi05_sim        # any writable dir

# 1. Upstream pi05_libero checkpoint (torch/safetensors) — downloads, fills in
#    config.json / norm_stats, and verifies it loads. Gated repo: run
#    `huggingface-cli login` first. See ../weights/README.md.
python_env/bin/python models/experimental/pi0_5/weights/download_pi05_libero.py \
  --out $PI05_SIM/pi05_libero_upstream
export PI05_CHECKPOINT_DIR=$PI05_SIM/pi05_libero_upstream

# 2. PaliGemma tokenizer (~4 MB, public)
curl -L -o $PI05_SIM/paligemma_tokenizer.model \
  https://storage.googleapis.com/big_vision/paligemma_tokenizer.model

# 3. LIBERO from source (the PyPI package is broken)
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git $PI05_SIM/libero_repo

# 4. System packages for headless MuJoCo render
sudo apt install -y libosmesa6 libegl1-mesa xvfb ffmpeg

# 5. Python deps into python_env (uv-managed). robosuite 1.4.0 is REQUIRED
#    (1.5.x breaks libero 0.1.0's import path); do NOT pin numpy (<2 downgrades ttnn).
export VIRTUAL_ENV=$PWD/python_env
uv pip install "robosuite==1.4.0" mujoco bddl easydict cloudpickle gym imageio-ffmpeg
uv pip install --no-deps -e $PI05_SIM/libero_repo
```

## Run

```bash
source models/experimental/pi0_5/common/pi05_production.env    # perf flags + checkpoint path

PI0_TOKENIZER_PATH=$PI05_SIM/paligemma_tokenizer.model \
LIBERO_REPO_PATH=$PI05_SIM/libero_repo \
MUJOCO_GL=osmesa TT_METAL_HOME=$PWD PYTHONPATH=$PWD:$PI05_SIM/libero_repo \
python_env/bin/python -u models/experimental/pi0_5/libero_sim/libero_rollout.py \
  --checkpoint $PI05_CHECKPOINT_DIR \
  --suites libero_spatial libero_object libero_goal libero_10 \
  --task-range 0 9 --num-episodes 1 --steps-sweep 5 \
  --backend ttnn --replan-steps 5
# → 40 episodes (1 init/task × 4 suites). Add --video-dir <dir> for per-episode mp4s;
#   scale up with --num-episodes N (up to 50 canonical inits/task).
#   Use --backend ttnn_1x8 for the 1×8 mesh (pin devices with TT_VISIBLE_DEVICES if needed).
```

Machine-specific env vars (not in `pi05_production.env`): `PI0_TOKENIZER_PATH`,
`LIBERO_REPO_PATH`, `MUJOCO_GL=osmesa`, and `TT_METAL_CACHE` if `$HOME/.cache` is a
dangling symlink.

## Key flags

| flag | default | meaning |
|---|---|---|
| `--backend` | `pytorch` | `pytorch` (CPU ref) / `ttnn` (single chip) / `ttnn_1x8` (1×8 mesh) |
| `--steps-sweep` | `10 4` | denoise-step counts (one rollout per N). **We run `5`.** |
| `--action-horizon` | `10` | chunk size (10 = upstream; auto-read from `config.json`) |
| `--state-in-prompt` | `false` | upstream default (task-prompt-only) |
| `--replan-steps` | `10` | actions applied per chunk before re-planning (we use `5`) |
| `--num-episodes` | `3` | initial states per task (≤ 50 canonical inits/task) |
| `--suites` / `--task-range` | `libero_spatial` / `0` | suites + inclusive task range |
| `--max-steps` | per-suite | env step cap (spatial 220 / object 280 / goal 300 / libero_10 520) |
| `--video-dir` / `--video-fps` | none / 20 | write `<dir>/N{N}/<suite>/task{XX}_ep{NN}_..._<success\|failure>.mp4` |

## Verify the environment

```bash
python_env/bin/python -c "
import ttnn, torch, numpy, robosuite, mujoco
from libero.libero.envs import OffScreenRenderEnv
print('ok — numpy', numpy.__version__, '| robosuite', robosuite.__version__, '| mujoco', mujoco.__version__)"
```
