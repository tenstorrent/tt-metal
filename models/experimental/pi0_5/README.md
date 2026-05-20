# PI0.5 (`pi0_5`) — Tenstorrent

End-to-end TTNN implementation of the **π₀.₅** (PI0.5) vision-language-action policy on Blackhole, with a PyTorch reference, LIBERO simulator integration, and a real-weights perf path that runs at **~55.8 ms / chunk** (N=10) — **~37.8 ms / chunk** at N=5 — with the upstream openpi `pi05_libero` checkpoint and TTNN trace. End-to-end LIBERO simulator success: **788/800 (98.5%)** across all four suites.



---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                             PI0.5 Model                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────┐   ┌──────────────────────────┐│
│  │         PREFIX EMBEDDING            │   │    SUFFIX EMBEDDING      ││
│  │                                     │   │                          ││
│  │  ┌───────────┐   ┌───────────────┐  │   │           ┌────────┐     ││
│  │  │  Images   │   │ Lang + State  │  │   │           │ Noisy  │     ││
│  │  │  (224x224)│   │ (200 tokens,  │  │   │           │Actions │     ││
│  │  │  (3 views)│   │  state→bins)  │  │   │           │(50, 32)│     ││
│  │  └─────┬─────┘   └───────┬───────┘  │   │           └───┬────┘     ││
│  │        │                 │          │   │               │          ││
│  │        ▼                 │          │   │               ▼          ││
│  │  ┌───────────┐           │          │   │      ┌────────────────┐  ││
│  │  │  SigLIP   │           │          │   │      │ action_in_proj │  ││
│  │  │  Vision   │           │          │   │      │  (32 → 1024)   │  ││
│  │  │  Tower    │           │          │   │      └────────┬───────┘  ││
│  │  │(27 blocks)│           │          │   │               │          ││
│  │  └─────┬─────┘           │          │   │               │          ││
│  │        │                 │          │   │  ┌──────────┐ │          ││
│  │        ▼                 │          │   │  │ sincos(t)│ │          ││
│  │  ┌───────────┐           │          │   │  └─────┬────┘ │          ││
│  │  │Projector  │           │          │   │        ▼      │          ││
│  │  │(1152→2048)│           │          │   │  ┌──────────┐ │          ││
│  │  └─────┬─────┘           │          │   │  │ time_mlp │ │          ││
│  │        │                 │          │   │  │ in→silu→ │ │          ││
│  │        ▼                 ▼          │   │  │ out→silu │ │          ││
│  │  ┌───────────────────────────────┐  │   │  └─────┬────┘ │          ││
│  │  │  Image Embeds + Lang Embeds   │  │   │        │      │          ││
│  │  │  (Gemma 2B embedding)         │  │   │  adarms_cond  │          ││
│  │  └───────────────┬───────────────┘  │   └────────┼──────┼──────────┘│
│  │                  │                  │            │      │           │
│  └──────────────────┼──────────────────┘            │      │           │
│                     │                               │      │           │
│                     ▼                               │      ▼           │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │               DUAL-EXPERT TRANSFORMER (18 layers)                │  │
│  │  ┌────────────────────────┐    ┌────────────────────────┐        │  │
│  │  │     Gemma 2B VLM       │    │   Gemma 300M Expert    │        │  │
│  │  │   (processes prefix)   │    │ ★ AdaRMSNorm variant ★ │        │  │
│  │  │   Plain RMSNorm        │    │                        │        │  │
│  │  │                        │    │   adaRMS Dense per     │        │  │
│  │  │  Q_vlm ──┐             │    │   layer:               │        │  │
│  │  │  K_vlm ──┼─► SHARED ◄──┼────┼─ scale, shift, gate    │        │  │
│  │  │  V_vlm ──┘   ATTN      │    │   ← from adarms_cond   │        │  │
│  │  │                        │    │                        │        │  │
│  │  │  MLP_vlm               │    │   normed = RMS(x)      │        │  │
│  │  │                        │    │   out = normed*(1+s)+b │        │  │
│  │  │                        │    │   residual ← gate * .  │        │  │
│  │  └────────────────────────┘    └────────────────────────┘        │  │
│  └────────────────────────────────────┬─────────────────────────────┘  │
│                                       │                                │
│                                       ▼                                │
│                       ┌──────────────────────────────┐                 │
│                       │     FLOW MATCHING DENOISER   │                 │
│                       │     (10 denoising steps)     │                 │
│                       │                              │                 │
│                       │  for t in [1.0 → 0.0]:       │                 │
│                       │    v_t = action_out_proj(    │                 │
│                       │           expert_out)        │                 │
│                       │    x_t ← x_t + dt · v_t      │                 │
│                       └──────────────┬───────────────┘                 │
│                                      │                                 │
│                                      ▼                                 │
│                        ┌──────────────────────────┐                    │
│                        │      Action Output       │                    │
│                        │     [batch=1, 50, 32]    │                    │
│                        └──────────────────────────┘                    │
└─────────────────────────────────────────────────────────────────────────┘
```

**Key architectural details:**
- **Shared Attention**: VLM and Expert share K,V tensors (concatenated), but have separate Q and MLPs (same as PI0)
- **AdaRMSNorm in Expert**: each layer reads `(scale, shift, gate)` from a Dense projection of `adarms_cond` (time-derived). `out = RMSNorm(x)·(1+scale) + shift`; residual is gated.
- **State in language tokens**: 8-dim robot state → MEAN/STD normalize → discretize to 256 bins → append to prompt as `Task: …, State: <bins>; Action:`. No separate state token in the suffix.
- **Flow Matching**: same Euler integration as PI0; 10 denoising steps from N(0,I) → actions.
- **Dual Experts**: VLM (2B) processes images+language+state-as-tokens; Expert (300M, adaRMS) processes only action tokens.

### What differs from PI0

| Component        | PI0                                              | **PI0.5**                                                     |
| ---------------- | ------------------------------------------------ | ------------------------------------------------------------- |
| Suffix tokens    | `[state_token, action_0, …, action_{H-1}]`       | `[action_0, …, action_{H-1}]` (state encoded into lang tokens)|
| Time injection   | concat(action, sincos(t)) → 2-layer MLP, fused   | sincos(t) → MLP → `adarms_cond` (fed to adaRMSNorm)           |
| Expert RMSNorm   | Plain RMSNorm                                    | adaRMSNorm: `normed * (1+scale) + shift`, with gated residual |
| `max_token_len`  | 48                                               | 200                                                           |
| State input      | continuous (state_proj)                          | discretized into 256 bins, embedded as language tokens        |

Everything else (SigLIP-27, Gemma-2B VLM, Gemma-300M expert, flow-matching denoising with Euler integration, KV-cache prefill of the prefix) follows the openpi/lerobot reference.

---

## Directory layout

```
pi0_5/
├── common/
│   ├── configs.py            # GemmaConfig, SigLIPConfig, PaliGemmaConfig,
│   │                         #   SuffixConfig, PrefixConfig, DenoiseConfig,
│   │                         #   PI0ModelConfig, Pi0_5ModelConfig
│   └── weight_loader.py      # PI0Config, PI0WeightLoader, categorize_weights,
│                             #   Pi0_5WeightLoader (subclass; strips "model." prefix)
├── reference/                # PyTorch reference (used for PCC + libero pytorch backend)
│   ├── torch_denoise.py
│   ├── torch_gemma.py        # GemmaAttn/MLP/Block + AdaRMSGemmaBlock
│   ├── torch_paligemma.py    # PaliGemmaBackbone + Pi0_5PaliGemmaBackbone
│   ├── torch_prefix.py
│   ├── torch_siglip.py
│   ├── torch_suffix.py       # SuffixEmbedding + Pi0_5SuffixEmbedding (sincos→silu→MLP→silu)
│   └── torch_pi0_5_model.py  # Pi0_5Model
├── tt/                       # TTNN implementation
│   ├── ttnn_common.py
│   ├── ttnn_gemma.py         # GemmaAttn/MLP TTNN + AdaRMSGemmaBlockTTNN +
│   │                         #   build_matmul_pcfg (2D + 1D width-shard small-M)
│   ├── ttnn_paligemma.py     # PaliGemmaBackboneTTNN + Pi0_5PaliGemmaBackboneTTNN
│   ├── ttnn_prefix.py
│   ├── ttnn_siglip.py
│   ├── ttnn_suffix.py        # SuffixEmbeddingTTNN + Pi0_5SuffixEmbeddingTTNN
│   └── ttnn_pi0_5_model.py   # Pi0_5ModelTTNN
├── eval/
│   └── libero_rollout.py     # LIBERO simulator → policy → success rate / videos
├── scripts/
│   └── download_weights.sh   # Download pi05_base / pi05_libero from HuggingFace
├── tests/
│   ├── pcc/                  # Reference-vs-spec correctness
│   └── perf/                 # Latency / throughput on Blackhole
└── weights/
    ├── pi05_base/             # lerobot/pi05_base safetensors (see download script)
    ├── pi05_libero_finetuned/ # lerobot/pi05_libero_finetuned_v044 (LIBERO eval)
    └── pi05_libero_upstream/  # upstream openpi pi05_libero (JAX→PyTorch converted)
```

---

## Quickstart

### Inference (PyTorch reference)

```python
from models.experimental.pi0_5.common.configs import Pi0_5ModelConfig
from models.experimental.pi0_5.common.weight_loader import Pi0_5WeightLoader
from models.experimental.pi0_5.reference.torch_pi0_5_model import Pi0_5Model

loader = Pi0_5WeightLoader("/path/to/pi05_base")
model  = Pi0_5Model(Pi0_5ModelConfig(), loader)

actions = model.sample_actions(
    images=[img_tensor],            # list of (B, 3, 224, 224)
    img_masks=[mask],
    lang_tokens=tok_ids,            # (B, L) — includes discretized state
    lang_masks=tok_mask,
    state=None,                     # ignored on the pi0.5 path
)  # → (B, action_horizon, action_dim)
```

### Inference (TTNN, Blackhole)

```python
import ttnn
from models.experimental.pi0_5.common.configs import Pi0_5ModelConfig
from models.experimental.pi0_5.common.weight_loader import Pi0_5WeightLoader
from models.experimental.pi0_5.tt.ttnn_pi0_5_model import Pi0_5ModelTTNN

device = ttnn.open_device(device_id=0, l1_small_size=24576, trace_region_size=134_217_728)  # 128 MiB — trace is ~81 MB
loader = Pi0_5WeightLoader("/path/to/pi05_base")
model  = Pi0_5ModelTTNN(Pi0_5ModelConfig(), loader, device)

actions = model.sample_actions(
    images=[image_ttnn],            # list of ttnn.Tensor
    img_masks=[mask_ttnn],
    lang_tokens=tokens_ttnn,
    lang_masks=lang_masks_ttnn,
    state=None,
)
ttnn.synchronize_device(device)
actions_torch = ttnn.to_torch(actions)
```

---

## Tests

All tests below are skipped automatically if `models/experimental/pi0_5/weights/pi05_base/model.safetensors` is missing.

### PCC (correctness) tests

The suite is layered: per-component → sub-model → end-to-end → diagnostic drilldowns. All TTNN-vs-torch tests need a Blackhole device and the real `pi05_base` weights.

```bash
# Run the whole suite
PYTHONPATH=$PWD python_env/bin/pytest -xvs models/experimental/pi0_5/tests/pcc/

# Component-level (TTNN module vs torch reference)
python_env/bin/pytest -xvs models/experimental/pi0_5/tests/pcc/test_pcc_gemma_vs_torch.py     # GemmaMLPTTNN
python_env/bin/pytest -xvs models/experimental/pi0_5/tests/pcc/test_pcc_prefix_vs_torch.py    # PrefixEmbeddingTTNN
python_env/bin/pytest -xvs models/experimental/pi0_5/tests/pcc/test_pcc_siglip_vs_torch.py    # SigLIPVisionTowerTTNN
python_env/bin/pytest -xvs models/experimental/pi0_5/tests/pcc/test_pcc_suffix_vs_torch.py    # Pi0_5SuffixEmbeddingTTNN

# Sub-model
python_env/bin/pytest -xvs models/experimental/pi0_5/tests/pcc/test_pcc_paligemma_vs_torch.py # PaliGemma backbone (VLM + Expert)

# End-to-end (full 10-step denoise)
python_env/bin/pytest -xvs models/experimental/pi0_5/tests/pcc/test_pcc_pi05_model_vs_torch.py    # single seed
python models/experimental/pi0_5/tests/pcc/test_pcc_pi05_per_step_vs_torch.py                     # primary gate: 10-seed E2E sweep

# Diagnostic block-level drilldowns
python_env/bin/pytest -xvs models/experimental/pi0_5/tests/pcc/test_pcc_vlm_block_drilldown.py     # one VLM block, sub-step PCC
python_env/bin/pytest -xvs models/experimental/pi0_5/tests/pcc/test_pcc_expert_block_drilldown.py  # one AdaRMS expert block
```

**Primary gate:** `test_pcc_pi05_per_step_vs_torch.py` runs the real `sample_actions()` path across N seeds (default 10) and gates on the **mean** E2E PCC. E2E PCC is intrinsically seed-sensitive because flow-matching with random initial noise is a chaotic dynamical system — a 10-step Euler integration of a learned velocity field amplifies tiny per-step bf16 drift differently per input. The sweep is the implementation-quality signal; a single seed is not.

**Latest E2E PCC distribution (10-seed sweep, real `pi05_base` weights):**

| metric | value |
|---|---|
| mean | **0.991** |
| stdev | 0.007 |
| min | 0.978 |
| max | 0.998 |
| pass rate (PCC ≥ 0.95) | **10/10** |
| per-step velocity PCC (worst step, single seed) | 0.993 |

### Perf tests (Blackhole)

Full headline run — `sample_actions` captured as a TTNN trace and replayed 20 times — against the upstream openpi `pi05_libero` checkpoint at 5 denoising steps:

```bash
# Assumes TT_METAL_HOME already points at your tt-metal checkout.
# (Set it once in your shell rc, e.g. `export TT_METAL_HOME=/path/to/tt-metal`.)
TT_VISIBLE_DEVICES=0 \
PI0_UPSTREAM_MASKS=1 \
QWEN_NLP_CONCAT_HEADS_HEAD_SPLIT=1 \
QWEN_NLP_CREATE_HEADS_HEAD_SPLIT=1 \
PI05_CHECKPOINT_DIR=$TT_METAL_HOME/models/experimental/pi0_5/weights/pi05_libero_upstream \
PI05_NUM_DENOISE_STEPS=5 \
PYTHONPATH=$TT_METAL_HOME \
python_env/bin/python -m pytest -s \
  $TT_METAL_HOME/models/experimental/pi0_5/tests/perf/test_perf_ttnn_full_e2e_trace.py
```

For N=10 denoising steps, drop `PI05_NUM_DENOISE_STEPS=5` (the test defaults to 10). To target the lerobot `pi05_base` or `pi05_libero_finetuned` checkpoint, point `PI05_CHECKPOINT_DIR` at it — `action_horizon` is auto-read from `<checkpoint>/config.json` (openpi `action_horizon` or lerobot `chunk_size`).

Without trace (un-amortized host dispatch):

```bash
PYTHONPATH=$PWD python_env/bin/pytest -xvs \
  models/experimental/pi0_5/tests/perf/test_perf_ttnn_full_e2e.py
```

**Flag reference for the trace command above:**

| Flag | Meaning |
|---|---|
| `TT_METAL_HOME` | Root of the tt-metal checkout (required by the runtime to find kernel sources, JIT cache, pre-compiled firmware). |
| `TT_VISIBLE_DEVICES=0` | Restrict the process to one TT card. Picks physical device 0; the test then opens logical device 0 inside that filtered set. |
| `PI0_UPSTREAM_MASKS=1` | Switch the attention-mask + RoPE-position path to the openpi convention (cumsum-based RoPE indices + logical-pad attention mask). Required for upstream openpi `pi05_libero`; harmless for `pi05_base` but unnecessary. |
| `QWEN_NLP_CONCAT_HEADS_HEAD_SPLIT=1` | Enable the head-split kernel path for `nlp_concat_heads` (action-expert side). Halves dispatch overhead for the small-M matmul/concat pattern. |
| `QWEN_NLP_CREATE_HEADS_HEAD_SPLIT=1` | Same as above for `nlp_create_qkv_heads` (Q/K/V split on the expert side). |
| `PI05_CHECKPOINT_DIR` | Override the symlinked `weights/pi05_base` default. Points at any locally converted checkpoint (`pi05_base`, `pi05_libero_finetuned`, upstream `pi05_libero`, …). |
| `PI05_NUM_DENOISE_STEPS=5` | Number of flow-matching Euler steps. Default 10; LIBERO sweep showed N=5 is accuracy-neutral with ~1.48× throughput. |
| `PYTHONPATH` | So `import models.experimental.pi0_5.…` resolves from a fresh shell. |
| `python_env/bin/python -m pytest -s` | Run pytest in the project's venv with `-s` to forward stdout (per-call latencies print as they happen). |

**Latest measured trace-mode perf (Blackhole, upstream openpi `pi05_libero`, action_horizon=10):**

| Denoise steps | Per-call latency | Chunk throughput | Action throughput |
|---:|---:|---:|---:|
| N=10 | **55.75 ms** | 17.94 chunks/s | 179 actions/s |
| N=5  | **37.79 ms** | 26.46 chunks/s | **265 actions/s** |

Standard deviation across 20 traced replays: 0.05 ms (both N). Trace capture: ~230–330 ms (one-time per (task, N) key).

Each denoise step ≈ **3.6 ms** through the 18-layer Gemma 300M action expert; fixed cost (SigLIP encode + VLM prefill + projection + dispatch) is ≈ 19.8 ms.

On the lerobot `pi05_base` checkpoint (action_horizon=50, N=10, with SigLIP block-sharded encoder + adaRMS expert + sharded LayerNorm/RMSNorm), the same test path measures **~64.85 ms / chunk** (~770 actions/s) — the larger suffix dominates the per-step cost.


---

## LIBERO simulator rollout

End-to-end real-robot benchmark on the LIBERO suites (`libero_spatial`, `libero_object`, `libero_goal`, `libero_10`).

### Latest simulator results

Full 4-suite, 10-init-per-task sweep on the **upstream openpi `pi05_libero`** checkpoint, TTNN traced rollout (`PI0_LIBERO_TRACE=1`, default on), 8-way parallel across Blackhole devices.

| Suite          | N=10              | N=5              |
|----------------|-------------------|------------------|
| libero_spatial | 100/100 (100.0%)  |  99/100 ( 99.0%) |
| libero_object  |  98/100 ( 98.0%)  |  99/100 ( 99.0%) |
| libero_goal    |  98/100 ( 98.0%)  |  98/100 ( 98.0%) |
| libero_10      |  98/100 ( 98.0%)  |  98/100 ( 98.0%) |
| **TOTAL**      | **394/400 (98.5%)** | **394/400 (98.5%)** |

**Grand total: 788/800 (98.5%)** — N=5 matches N=10 accuracy with **~1.48× throughput** (37.8 ms vs 55.8 ms per chunk). At LIBERO's 20 Hz env, N=5 is ~13× faster than real-time.

### Upstream openpi `pi05_libero` support

The rollout adapter now supports both checkpoint variants. Defaults target upstream openpi:

| Variant | `--checkpoint` | `--action-horizon` | `--state-in-prompt` |
|---|---|---:|---|
| **upstream openpi** (default) | `pi05_libero_upstream` | `10` | `false` |
| lerobot finetune | `pi05_libero_finetuned` | `50` | `true` |

The three flags are coordinated: overriding one without the others produces garbage (untrained position embeddings beyond the trained range, or state tokens the model never saw during training).

Required env vars for the traced upstream path:
```bash
PI0_UPSTREAM_MASKS=1                   # cumsum-RoPE + logical-pad attention mask
QWEN_NLP_CONCAT_HEADS_HEAD_SPLIT=1     # NLP head-split kernel paths (action expert)
QWEN_NLP_CREATE_HEADS_HEAD_SPLIT=1
PI0_SIGLIP_HF=1                        # optional: HF SigLIP bridge for vision encoder
# PI0_LIBERO_TRACE defaults to on for backend=ttnn; opt out with =0
```

### One-time setup (not in git)

```bash
# 1. PaliGemma tokenizer (used to encode the prompt + discretized state)
mkdir -p /storage/sdawle/pi05_weights
curl -L -o /storage/sdawle/pi05_weights/paligemma_tokenizer.model \
  https://storage.googleapis.com/big_vision/paligemma_tokenizer.model

# 2. Download pi05_libero checkpoint (see "Weights" section above for details)
./models/experimental/pi0_5/scripts/download_weights.sh libero

# 3. LIBERO env from source (PyPI install is broken)
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git /storage/sdawle/libero_repo

# 4. System packages for MuJoCo headless render
sudo apt install -y libosmesa6 libegl1-mesa xvfb ffmpeg

# 5. Python deps in the active venv
python_env/bin/pip install mujoco imageio-ffmpeg lerobot gym-aloha bddl easydict robosuite sentencepiece 'numpy<2'
```

### Running a rollout

```bash
cd $TT_METAL_HOME

PYTHONPATH=$PWD:/storage/sdawle/libero_repo \
MUJOCO_GL=osmesa HF_HOME=/storage/sdawle/hf_cache \
python_env/bin/python -u models/experimental/pi0_5/eval/libero_rollout.py \
  --checkpoint /storage/sdawle/pi05_weights/pi05_libero_finetuned \
  --suites libero_spatial libero_object libero_goal libero_10 \
  --task-range 0 9 \
  --num-episodes 1 \
  --steps-sweep 4 10 \
  --backend ttnn \
  --replan-steps 5 \
  --video-dir /storage/sdawle/libero_videos --video-fps 20
```

### CLI flags

| flag | default | meaning |
|---|---|---|
| `--checkpoint` | `pi05_libero_finetuned` | path to model.safetensors + normalizer stats |
| `--suite` / `--suites` | `libero_spatial` | one (`--suite`) or many (`--suites` nargs+) LIBERO suites |
| `--task-idx` / `--task-range` | `0` | single task or inclusive `(start, end)` range |
| `--num-episodes` | `3` | initial states per task (max 50; LIBERO ships 50 canonical inits per task) |
| `--max-steps` | per-suite default | env step cap; defaults: spatial=220, object=280, goal=300, libero_10=520 |
| `--backend` | `pytorch` | `pytorch` (CPU ref) or `ttnn` (Blackhole) |
| `--replan-steps` | `10` | apply this many actions per chunk before requesting a new chunk (openpi convention=5) |
| `--steps-sweep` | `10 4` | denoise-step counts to evaluate (one rollout per N) |
| `--video-dir` | none | write one mp4 per episode under `<dir>/N{N}/<suite>/task{XX}_ep{NN}_<title>_<success\|failure>.mp4` |
| `--video-fps` | `20` | playback fps (sim runs at 20 Hz) |

### Example: 5-episode sanity check

```bash
PYTHONPATH=$PWD:/storage/sdawle/libero_repo MUJOCO_GL=osmesa HF_HOME=/storage/sdawle/hf_cache \
python_env/bin/python -u models/experimental/pi0_5/eval/libero_rollout.py \
  --num-episodes 5 --max-steps 220 --steps-sweep 4 \
  --backend ttnn --replan-steps 5
# → ~3 minutes wall, 5 episodes, prints per-episode success + final summary
```

### Example: full demo across all 4 suites with videos

```bash
PYTHONPATH=$PWD:/storage/sdawle/libero_repo MUJOCO_GL=osmesa HF_HOME=/storage/sdawle/hf_cache \
python_env/bin/python -u models/experimental/pi0_5/eval/libero_rollout.py \
  --suites libero_spatial libero_object libero_goal libero_10 \
  --task-range 0 9 --num-episodes 1 --steps-sweep 4 10 \
  --backend ttnn --replan-steps 5 \
  --video-dir /storage/sdawle/libero_videos
# → 80 episodes, ~50 min wall, 80 mp4s organized as N{4,10}/<suite>/...
```



---

## Weights

Three checkpoint variants are exercised in-tree. Two come pre-converted on HuggingFace; the upstream openpi `pi05_libero` is published as a JAX/Orbax checkpoint on Google Cloud Storage and must be converted to PyTorch `safetensors` first.

| Variant | Source | Size | Format | Use |
|---|---|---|---|---|
| `pi05_base` | [`lerobot/pi05_base`](https://huggingface.co/lerobot/pi05_base) (HF) | ~14.5 GB | safetensors | PCC tests, perf tests, general inference |
| `pi05_libero_finetuned` | [`lerobot/pi05_libero_finetuned_v044`](https://huggingface.co/lerobot/pi05_libero_finetuned_v044) (HF) | ~7.5 GB | safetensors | LIBERO eval (lerobot finetune, `action_horizon=50`, `state_in_prompt=true`) |
| `pi05_libero` (upstream openpi) | [`gs://openpi-assets/checkpoints/pi05_libero/`](https://storage.googleapis.com/openpi-assets/checkpoints/pi05_libero/) (GCS, public) | ~12.4 GB JAX → ~6.8 GB safetensors after convert | JAX/Orbax → convert | LIBERO eval (upstream, `action_horizon=10`, `state_in_prompt=false`) |

### HuggingFace download (lerobot variants)

```bash
# Install the HuggingFace CLI (if not already installed)
pip install huggingface_hub[cli]

# Log in (required — Gemma-family weights are gated)
huggingface-cli login

# Download both lerobot checkpoints into weights/
./models/experimental/pi0_5/scripts/download_weights.sh

# Or download individually:
./models/experimental/pi0_5/scripts/download_weights.sh base    # lerobot/pi05_base (~14.5 GB)
./models/experimental/pi0_5/scripts/download_weights.sh libero  # lerobot/pi05_libero_finetuned_v044 (~7.5 GB)
```

### Upstream openpi `pi05_libero` — download + JAX→PyTorch convert

The upstream openpi `pi05_libero` checkpoint is published as a JAX/Orbax tree. Convert it to PyTorch `safetensors` using openpi's conversion script — [`examples/convert_jax_model_to_pytorch.py`](https://github.com/Physical-Intelligence/openpi/blob/main/examples/convert_jax_model_to_pytorch.py):

```bash
# 1. Clone openpi and install its conversion deps
git clone https://github.com/Physical-Intelligence/openpi.git /path/to/openpi
cd /path/to/openpi
uv venv --python 3.11 .venv
uv pip install -e . --no-deps
uv pip install flax==0.10.2 orbax-checkpoint==0.11.13 jax==0.5.3 jaxlib==0.5.3 \
  ml-dtypes==0.4.1 tensorstore==0.1.74 numpy'<2' safetensors torch==2.7.1 \
  transformers==4.53.2 tyro 'fsspec[gcs]'
# Apply openpi's transformers patch (required by the converter):
cp -r src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/

# 2. Download the public JAX checkpoint (~12.4 GB) from GCS
mkdir -p /path/to/jax_pi05_libero
curl -s "https://storage.googleapis.com/storage/v1/b/openpi-assets/o?prefix=checkpoints/pi05_libero/&maxResults=500" \
  | python -c "import json,sys,os; \
      [os.makedirs(os.path.dirname(p), exist_ok=True) or os.system(f\"curl -sLo {p} {i['mediaLink']}\") \
       for i in json.load(sys.stdin)['items'] \
       for p in [os.path.join('/path/to/jax_pi05_libero', i['name'].replace('checkpoints/pi05_libero/', '', 1))]]"

# 3. Run the converter (~5 min)
.venv/bin/python examples/convert_jax_model_to_pytorch.py \
  --checkpoint_dir /path/to/jax_pi05_libero \
  --config_name pi05_libero \
  --output_path /path/to/pi05_libero_upstream \
  --precision bfloat16

# 4. Copy norm-stats assets next to the safetensors
cp -r /path/to/jax_pi05_libero/assets /path/to/pi05_libero_upstream/
```

After conversion, point the rollout / perf test at the output directory with `--checkpoint /path/to/pi05_libero_upstream` (rollout) or `PI05_CHECKPOINT_DIR=/path/to/pi05_libero_upstream` (perf test).

---

## License

SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
SPDX-License-Identifier: Apache-2.0
