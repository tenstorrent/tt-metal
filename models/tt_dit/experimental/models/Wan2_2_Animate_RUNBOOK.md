# Wan2.2-Animate-14B on a Galaxy: runbook

This runbook is written for a **local Claude Code agent driving a Cirrascale Galaxy over SSH**. The
design and its rationale are in [Wan2_2_Animate.md](Wan2_2_Animate.md). The code was written
without hardware: the CPU checks pass, and everything that touches a device is unverified. The goal
here is to find and fix what breaks, in a fixed order, and record what happens.

Ground rules:

- Work on branch `claude/wan22-animate-poc` only. Commit fixes there and push.
- Never push to `main`. Don't open or merge PRs.
- Keep weights, videos and caches out of git.
- Record every run in the [RESULTS](#results) section at the end of this file.
- Run the stages in order. Each stage localizes failures for the next one.

## 0. Conventions

```bash
GALAXY=user@galaxy-host            # fill in
REPO=~/tt-metal                    # tt-metal checkout on the Galaxy
ssh $GALAXY 'cd ~/tt-metal && git status -sb | head -3'
```

Run long commands inside `tmux` on the Galaxy, and capture logs with `2>&1 | tee logs/<stage>.log`.
The first conversion of each module from torch weights takes minutes and looks like a hang. The
weight loader prints a heartbeat every 5 s.

## 1. Environment and build

```bash
ssh $GALAXY
cd ~/tt-metal                                     # or: git clone https://github.com/tenstorrent/tt-metal ~/tt-metal
git fetch origin claude/wan22-animate-poc && git checkout claude/wan22-animate-poc
git submodule update --init --recursive
./build_metal.sh                                  # see INSTALLING.md; reuse an existing build if one is current
./create_venv.sh && source python_env/bin/activate
export TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd)
export TT_DIT_CACHE_DIR=/localdev/$USER/tt_dit_cache   # converted-weight cache, makes reruns fast (~40 GB)
export HF_HOME=/localdev/$USER/hf                      # ~110 GB of weights, see section 2
mkdir -p $TT_DIT_CACHE_DIR $HF_HOME logs out

python -c "import diffusers, transformers; print(diffusers.__version__, transformers.__version__)"  # expect diffusers 0.38.0
python -c "import ttnn; print('blackhole' if ttnn.device.is_blackhole() else 'wormhole')"
tt-smi -s | head -40                              # 32 chips visible, none in a bad state
```

The architecture decides which test IDs to use for the rest of the runbook:

| Galaxy | Module tests (`-k`) | End-to-end (`--mesh`) |
|---|---|---|
| Blackhole | `bh_4x8sp1tp0_linear`, then `bh_4x8sp1tp0_ring` | `bh_4x8_linear`, then `bh_4x8_ring` |
| Wormhole | `wh_4x8sp1tp0` | `wh_4x8_ring` |

The examples below use the Blackhole IDs; substitute if needed. Blackhole uses linear first because
`Wan2_2.md` runs I2V on BH Galaxy with the linear fabric.

## 2. Weights and upstream code

```bash
# Diffusers-format Animate checkpoint: transformer (69 GB, fp32), UMT5, CLIP ViT-H, VAE, scheduler.
huggingface-cli download Wan-AI/Wan2.2-Animate-14B-Diffusers

# From the original repo: preprocessing models (pose, detector, SAM2) and the relighting LoRA. Not the
# 14B original-format weights.
huggingface-cli download Wan-AI/Wan2.2-Animate-14B \
    --include "process_checkpoint/det/*" "process_checkpoint/pose2d/*" "process_checkpoint/sam2/sam2_hiera_large.pt" \
              "relighting_lora/*" \
    --local-dir /localdev/$USER/Wan2.2-Animate-14B

# Upstream code, for the preprocessing scripts only. Use a separate venv so tt-metal's pins stay intact.
git clone --depth 1 https://github.com/Wan-Video/Wan2.2 ~/Wan2.2
python3 -m venv ~/wan_pre && ~/wan_pre/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
~/wan_pre/bin/pip install onnxruntime decord opencv-python-headless loguru matplotlib pandas "diffusers>=0.35" transformers einops
# Replacement mode only:
~/wan_pre/bin/pip install "git+https://github.com/facebookresearch/sam2.git@0e78a118995e66bb27d78518c4bd9a3e95b4e266"
```

Set `PRE=/localdev/$USER/Wan2.2-Animate-14B/process_checkpoint` and
`LORA=/localdev/$USER/Wan2.2-Animate-14B/relighting_lora/adapter_model.safetensors`.

If the Galaxy host has no internet, download on a machine that does and `rsync` `$HF_HOME`. Then
`export HF_HUB_OFFLINE=1`.

## 3. Tests, in order

Each stage must pass before the next. Thresholds are the test asserts; record the PCC actually
printed (`assert_quality` logs it).

| # | Command | What it isolates | Expected | Time |
|---|---|---|---|---|
| 3.0 | `pytest models/tt_dit/experimental/tests/test_wan_animate_cpu.py -q` | The math, on CPU only | 21 passed (passed when written) | <1 min |
| 3.1 | `pytest models/tt_dit/tests/models/wan2_2/test_transformer_wan.py -k "test_wan_transformer_block and 4x8sp1tp0nl2_line_is_fsdp0 and 14b-480p"` | Regression: the `attention_wan.py` edits must not change Wan 2.2 | PCC ≥ 0.9995 (unchanged) | ~10 min first run (CPU reference) |
| 3.2 | `pytest models/tt_dit/experimental/tests/test_wan_animate_modules.py -k "test_animate_patch_embedding and bh_4x8sp1tp0_linear"` | Pose fusion in the patch embedding | PCC ≥ 0.9999 | <2 min |
| 3.3 | `... -k "test_animate_face_encoder and bh_4x8sp1tp0_linear and (tiny or full)"` | Face encoder: selection-matrix convs, LN and silu on device | PCC ≥ 0.999, shape [1, T', 5, D] | <3 min |
| 3.4 | `... -k "test_animate_face_adapter and bh_4x8sp1tp0_linear and tiny and small"` | Masked cross-attention with per-head QK norm, in sdpa and matmul modes | PCC ≥ 0.999 (output), ≥ 0.995 (adapter delta) | <3 min |
| 3.5 | `... -k "test_animate_face_adapter and bh_4x8sp1tp0_linear and full and 480p_segment"` | The same at real dims and real sequence length | same | ~10 min |
| 3.6 | `... -k "test_animate_block_with_image_kv and bh_4x8sp1tp0_linear and tiny"`, then `full and small`, then `full and 480p` | A Wan block with the CLIP image K/V branch | PCC ≥ 0.9995, relative RMSE ≤ 0.04 | full/480p ~15 min (CPU reference) |
| 3.7 | `... -k "test_animate_transformer_model and bh_4x8sp1tp0_linear and tiny"` | The whole Animate transformer, tiny: embeddings, 5 blocks, adapter, host modules | PCC ≥ 0.99, RMSE ≤ 0.15 | <5 min |
| 3.8 | `WAN_ANIMATE_REAL_WEIGHTS=1 pytest ... -k "real and bh_4x8sp1tp0_linear"` | Real checkpoint weights: face encoder, adapter 0, block 0, 1-block model | as 3.3–3.7 | 20–40 min |
| 3.9 | Section 4 end to end, 480p, about 2 s | Everything | see section 5 | see RESULTS |

`...` stands for `pytest models/tt_dit/experimental/tests/test_wan_animate_modules.py`. Add `-x -s`
for logs.

After 3.2–3.8 pass on `_linear`, run 3.6 and 3.7 once on `_ring` too, and note which fabric is
faster.

## 4. End to end on a short clip

Inputs: a 2–3 s clip of one person dancing, full body in frame (`dance.mp4`), and a character image
(`cat.png`). For animation mode, the character should be front-facing, roughly upright, with its
limbs visible.

```bash
# Animation mode, 480p-area, 2 s. The script runs the official preprocessing (CPU) first,
# then the device run inside pytest.
python models/tt_dit/experimental/wan_animate/run_animate.py \
    --image cat.png --video dance.mp4 --mode animate --seconds 2 \
    --wan-repo ~/Wan2.2 --preprocess-ckpt $PRE --pre-python ~/wan_pre/bin/python --retarget \
    --resolution-area 832 480 --mesh bh_4x8_linear --out out/cat_dance_animate_480p.mp4 2>&1 | tee logs/e2e_animate_480p.log
```

Equivalently, run the preprocessing by hand and pass `--src-root`:

```bash
(cd ~/Wan2.2/wan/modules/animate/preprocess && ~/wan_pre/bin/python preprocess_data.py \
    --ckpt_path $PRE --video_path $PWD/dance.mp4 --refer_path $PWD/cat.png \
    --save_path ~/work/pre_animate --resolution_area 832 480 --retarget_flag)
python models/tt_dit/experimental/wan_animate/run_animate.py --src-root ~/work/pre_animate \
    --mode animate --seconds 2 --mesh bh_4x8_linear --out out/cat_dance_animate_480p.mp4
```

Inspect `~/work/pre_animate/src_pose.mp4` and `src_face.mp4` before blaming the model. Retargeting
(`--retarget_flag`) helps when the character's proportions differ from the dancer's (a cat!).

For replacement mode, drop `--retarget_flag` and add `--replace_flag` to the preprocessing; this
needs SAM2. Then:

```bash
python models/tt_dit/experimental/wan_animate/run_animate.py --src-root ~/work/pre_replace \
    --mode replace --seconds 2 --relighting-lora $LORA --mesh bh_4x8_linear \
    --out out/cat_dance_replace_480p.mp4
```

**SAM2 on a CPU-only host.** Upstream `sam_utils.py` calls `.cuda()`. Patch your local copy only;
never commit it to tt-metal:

```bash
cd ~/Wan2.2/wan/modules/animate/preprocess
sed -i 's/device="cuda"/device="cpu"/; s/\.cuda()/.cpu()/g' sam_utils.py
grep -rn "cuda" sam_utils.py process_pipepline.py    # remaining hits must be in code paths you don't use
```

Alternatively, run the preprocessing on any GPU machine and copy the output directory over.

Outputs:

- `out/<name>.mp4`;
- `out/<name>_stats.json`, with timings by section, frame stats and the size;
- a `RESULTS {...}` line in the log.

Once 480p works, repeat at `--resolution-area 1280 720` and at `--seconds 3`. 2–3 s at 30 fps is
60–90 frames, so 1–2 segments of 77 with a 1-frame overlap.

## 5. What good output looks like

- **Animation**: the cat, keeping the reference image's identity, colours and background, does the
  dancer's moves in time. Arms follow the skeleton in `src_pose.mp4`, and the head turns and
  expression follow `src_face.mp4`. With 2 segments there is no visible pop at frame 77.
- **Replacement**: the dancer is gone and the cat dances in the original scene, with the background
  unchanged outside the mask. With the relighting LoRA, the cat's shading matches the scene.
- In `_stats.json`:
  - `pixel_std` is well above 5; a flat or grey video points to NaNs or a dead VAE;
  - `temporal_mad` is clearly above 0.2; a static video means the motion is not arriving.
- The test asserts only these sanity checks. Quality is judged by eye: open the mp4.

Bad output, by symptom:

| Symptom | Likely cause | Look at |
|---|---|---|
| A still image of the cat | Pose or face conditioning ignored: wrong pose channel order, or motion tokens zero | 3.2, 3.7; `cond` channels 36..52 |
| Correct motion but a different character | Image K/V branch or CLIP context broken | 3.6; `prepare_image_conditioning` |
| Correct body but a frozen face | Face adapter or face encoder | 3.3–3.5 |
| Noise or colour soup | Patch-embedding layout or `_model_input` interleave | 3.2, 3.7 |
| Flicker or pop at the segment boundary | Temporal guidance latent or mask (`prev_len`, `i2v_mask`) | the CPU test `test_segment_conditioning_*` |

## 6. Likely failure points and how to debug them

1. **Import or signature errors** (no hardware has run this code). Fix them in place; they will be
   in `experimental/wan_animate/*`, `attention_wan.py` or `transformer_wan.py`. Rerun 3.0 after
   every fix.
2. **SDPA rejects the mask, or the face-adapter PCC is low in `sdpa` but fine in `matmul`.** Set
   `WanAttention.cross_attn_impl = "matmul"` as a class default, or per instance in
   `WanAnimateFaceAdapter.__init__` and for the blocks' `attn2`. Note it in RESULTS. If both modes
   fail, check the mask shard: per device it must be `[1, 1, N_pad / sp, K]`, with
   `N_pad % (32 * sp) == 0`.
3. **Per-head RMSNorm** (`dit_fused_distributed_rmsnorm` with `per_head_norm=True` and no RoPE)
   errors, or gives a low PCC in 3.4 while 3.6 is fine. Work around it with whole-row norm on a
   reshaped tensor: create heads first (`nlp_create_qkv_heads`), then `ttnn.rms_norm` over head_dim
   with the [head_dim] weight. Patch `WanAttention` for `per_head_qk_norm`.
4. **The face encoder PCC is low only for `face_frames=77`.** This is tile padding: check that
   `causal_conv_selectors` gets `rows` and `cols` equal to the padded tile counts, and that the
   inputs to `ttnn.matmul` are `[1, 1, rows, cols] @ [1, 1, cols, C]`.
5. **Strict load fails** (`missing` or `unexpected Torch state keys`). Diff the key list against
   the checkpoint index. Every key must be consumed: `image_embedder` and `motion_encoder` are
   popped to the host modules, and `pose_patch_embedding` is fused. Do not switch to
   `strict=False`; that is how the earlier attempt silently lost the motion conditioning.
6. **Out of DRAM.**
   - At init on WH: confirm `is_fsdp=True`.
   - During the VAE encode at 720p: run 480p first, or try `WAN_ANIMATE_VAE_ENC_CHUNK=4`. The value
     must divide 76; 4 is the default and the smallest.
   - Free the text encoder after encoding as a last resort.
7. **VAE encode drops frames** (logs say `dropping ... trailing frame(s)`). The chunk does not
   divide `segment_frame_length - 1`; use 4.
8. **NaN latents** (`FloatingPointError: non-finite latents`). Find the first bad block by running
   `test_animate_transformer_model` with `tiny`, then `real`. Likely culprits are a fully masked
   softmax row (it should be impossible; check `face_attention_mask` rows ≥ N) or bf16 overflow in
   the adapter residual.
9. **Hangs in CCL.** Check the fabric mode matches the test ID (linear or ring), and that
   `num_links` is 2 on BH and 4 on WH. Reset with `tt-smi -r` and retry once; otherwise record it.
10. **Slow.**
    - The first run compiles kernels and converts weights; that takes tens of minutes. Rerun with
      `TT_DIT_CACHE_DIR` set before timing.
    - The loop is untraced, so expect host-bound steps.
    - The time per section is in `_stats.json`.

When a fix touches shared code (`attention_wan.py`, `transformer_wan.py`), rerun 3.1 before pushing.

## 7. Reporting back

- Commit each fix to `claude/wan22-animate-poc` with a message naming the stage it unblocks, for
  example `wan-animate: matmul fallback for face adapter (3.4)`. Push with
  `git push origin claude/wan22-animate-poc`.
- Append to RESULTS below:
  - one row per stage: the PCCs printed, pass or fail, and the fix commit;
  - one row per end-to-end run: the `_stats.json` timings, the size, the frames and a one-line
    verdict;
  - where the output mp4s are on the Galaxy (don't commit them; a GIF under 2 MB in a PR comment is
    fine later).
- If a stage cannot be made to pass, write down the exact error, what you tried, and the smallest
  reproducer (test ID), then move on to the next independent stage.

## RESULTS

<!-- Append below. Keep the newest at the bottom. -->

| Date | Machine (BH/WH, fabric) | Stage | Result (PCC / timing) | Commit | Notes |
|---|---|---|---|---|---|
| 2026-11-16 | dev container, CPU | 3.0 | 21 passed | (initial) | Written without hardware; nothing on device yet |
| 2026-11-20 | cirrascale-13, BH Galaxy, linear | 3.1 | PCC 99.9941 % (regression, unchanged) | - | Wan 2.2 block unaffected by the attention_wan.py edits |
| 2026-11-20 | cirrascale-13, BH Galaxy, linear | 3.2 | 99.9997 % (tiny and full) | - | |
| 2026-11-20 | cirrascale-13, BH Galaxy, linear | 3.3 | 99.9990 % (tiny); real weights pass | - | |
| 2026-11-20 | cirrascale-13, BH Galaxy, linear | 3.4 | sdpa 99.9990 / delta 99.974 %; matmul 99.9989 / 99.973 % | tiny config 4 -> 8 heads | per-head norm kernel needs more than 1 head per device; the 4-head tiny config had 1 at TP=4 |
| 2026-11-20 | cirrascale-13, BH Galaxy, linear | 3.5 | 99.9987 / delta 99.9736 % | - | both modes pass at 480p_segment |
| 2026-11-20 | cirrascale-13, BH Galaxy, linear | 3.6 | tiny 99.9975, full small 99.9973, full 480p 99.9978 % | - | |
| 2026-11-20 | cirrascale-13, BH Galaxy, linear | 3.7 | 99.9968 % | - | |
| 2026-11-20 | cirrascale-13, BH Galaxy, linear | 3.8 | all 9 real-weights tests pass; adapter delta 97.3 % | delta threshold 0.97 for real weights | the real adapter's delta is ~1 % of the residual; bf16 rounding of the sum alone caps its PCC at 98.5 % (measured on CPU); output PCC 99.9997 % |

End-to-end runs:

| Date | Mode | Size | Frames | init_s | generate_s | denoise_s / segment | Verdict | Output path |
|---|---|---|---|---|---|---|---|---|
| 2026-11-20 | replace | 832x464 | 60 | 28.4 | 102.5 | 25.6 | upstream replace example (boy -> robed man): scene, pose and framing preserved, character relit; vae_encode 36.4 s, vae_decode 28.3 s | ~/jgeilich/animate/out/example_replace_480p.mp4 on cirrascale-13 |
