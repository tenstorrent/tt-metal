# tencent/HunyuanVideo-1.5 — TTNN end-to-end pipeline

On-device TTNN pipeline for **HunyuanVideo-1.5** (an MMDiT / dual-stream video diffusion
transformer): **Qwen text-encode → DiT denoise → tiled VAE decode**, running full
text→video / image→video generation on a 32-chip Blackhole Galaxy. The DiT reproduces
`diffusers.HunyuanVideo15Transformer3DModel.forward(...)` — the golden the sampler calls
each denoise step. `demo/demo_t2v` and `demo/demo_i2v` share the one pipeline
(`tt/pipeline.py`); they differ only in whether `image_embeds` is populated (t2v masks the
image tokens out via a per-key additive attention bias, so it matches the reference).

## Status
- **Branch:** `sdawle/hunyuanvideo-bringup_bh_glx`
- **18/18 DiT modules on device** (native ttnn, per-component PCC-verified)
- **Full 121-frame generation validated across all four (task × resolution) checkpoints,**
  32-chip sp=8×tp=4 + tile-sharded VAE (DiT + VAE on device all 32 chips, text-encode on host):
  | task | tier | resolution | denoise | **e2e** |
  |---|---|---|---|---|
  | **t2v** | 480p | 480×848 | 1:51 (2.02 s/it) | **5:09** |
  | **t2v** | 720p | 720×1280 | 5:00 (5.76 s/it) | **9:12** |
  | **i2v** | 480p | 480×848 | 1:58 (2.37 s/it) | **5:19** |
  | **i2v** | 720p | 720×1280 | 5:07 (5.91 s/it) | **9:27** |

  **With the heads-major layout flags** (`HY_DIT_FUSED_HEADS=1
  HY_DIT_FUSED_QKV_HEADS=1`) and `HY_CFG_PADDING_POLICY=masked`, measured on a
  freshly reset Galaxy at the same 121 frames / 50 steps. Output is
  bit-identical to the rows above (frame PCC 1.00000000, max abs pixel diff 0.0):
  | task | tier | denoise | **e2e** | vs baseline |
  |---|---|---|---|---|
  | **i2v** | 480p | 1:20 (1.60 s/it) | **4:50** | −32.2% denoise |
  | **i2v** | 720p | 3:56 (4.72 s/it) | **8:11** | −23.1% denoise |

  t2v was not re-measured: only the two i2v checkpoints are in the local HF
  cache, so the t2v rows above remain at their original measurement.

  720p is opt-in (`HY_720P=1`); i2v via `HY_I2V=1` (+ `HY_IMAGE=<path>` for the conditioning
  frame). i2v conditioning (SigLIP `image_embeds` + a VAE-encoded first frame concatenated
  into the DiT's 65 input channels) is PCC-verified by the e2e gate (0.999979, both regimes)
  and produces coherent animated video (first frame reconstructs the input, then it moves).
- **Per-step DiT `device_ms`: 6.43 → 5.10 (1.26×)** via 16 fusion/sharding wins + an L1 guard
  (tt-hw-planner `optimize`, on the capped 2-layer profiling workload).
- **Device-resident FlowMatch Euler + latent path is available as an opt-in**
  (`HY_DEVICE_RESIDENT_DENOISE=1`, SP only). Equal-length CFG is batched. Mixed
  positive/negative or batch-row lengths default to exact-length condition slots.
  `HY_CFG_PADDING_POLICY=masked` packs valid image/byT5/Qwen tokens per row and
  supplies per-row key lengths to fused ring joint SDPA, keeping one padded DiT
  batch and trace shape. Invalid query states are zeroed after every block, and
  Qwen refinement still runs at each row's exact length. **It now runs end to end
  on hardware** (i2v 480p, 13 frames, 8 steps, 225.04s wall, frame PCC 1.000000);
  the earlier failure was the fabric 2×4 open bug, not the mask. It stays opt-in
  pending a 121-frame 50-step quality run, and must be paired with `HY_TRACE=0`.
- **`HY_TRACE=1` is currently a correctness blocker, not just unqualified.**
  Capture is exact — a 1-step traced generation is bit-identical to eager
  (PCC 1.000000) — but per-step replay is wrong: 8 steps give aggregate PCC
  0.237300 with per-frame PCC falling 0.9747 → 0.8648 → negative. Frame 0 only
  survives because i2v anchors it to the conditioning image. Same signature as
  the already-rejected heterogeneous trace (0.235647), so it is one defect.
- **The shared tt_dit T5 encoder now supports independent attention width.**
  Hunyuan byT5 (`d_model=1472`, q/k/v width `6×64=384`) runs on a genuinely
  disjoint TP1/TP2 mesh and unloads before DiT construction. **Its real-weight
  hardware PCC gate passes on all 5 cases** (TP1 0.999935, TP2 0.999938, full
  sequence without padding neutralization 0.999931). Host byT5 remains the
  default for a committed 32-chip run because no disjoint mesh is left, which is
  a placement result rather than a pending measurement.

## Download the weights
The pipeline pulls the **community diffusers conversions** from the HF hub (cached under
`$HF_HOME`, default `~/.cache/huggingface`). There are four checkpoints, one per
(task × resolution); each is **~54 GB total** (transformer ~33 GB + VAE ~5 GB + Qwen/byT5
text encoders):

| checkpoint | HF repo |
|---|---|
| 480p t2v *(default)* | `hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v` |
| 720p t2v | `hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v` |
| 480p i2v | `hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_i2v` |
| 720p i2v | `hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_i2v` |

```bash
# the 480p t2v pipeline auto-downloads on first run, or pre-fetch it explicitly:
hf download hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v

# 720p reuses the 480p VAE + text encoders (byte-identical across tiers), so it only
# needs the 720p transformer -- HY_720P=1 swaps just that over the cached 480p pipeline:
hf download hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v \
    --include "transformer/*" "model_index.json"

# i2v needs its own checkpoint (adds a SigLIP image_encoder); 720p_i2v is again transformer-only:
hf download hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_i2v
hf download hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_i2v \
    --include "transformer/*" "model_index.json"
```
Set `HF_HUB_OFFLINE=1` once cached to skip the network round-trip. *(Heads-up: all four
checkpoints together are ~216 GB; keep an eye on disk.)*

## Directory structure
```
hunyuanvideo_1_5/
├── tt/pipeline.py          # the ONE shared chained forward (demo + tests call it)
├── _stubs/                 # the 18 graduated TTNN stubs (native on-device modules)
├── demo/{demo_i2v,demo_t2v}.py   # single-denoise-step demos
├── tests/pcc/              # per-component PCC tests (18 modules + mesh variant)
├── tests/e2e/              # e2e gate, real-weight PCC, generation, perf tests
├── OPTIMIZATION_REPORT.md  # cross-pipeline optimization inventory and blockers
└── e2e_plan.json
```

## How to run
```bash
# setup (once per shell)
source python_env/bin/activate
export TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd)
PY=./python_env/bin/python
DIR=models/demos/hf_eager/hunyuanvideo_1_5

# --- fastest 121-frame t2v generation: 32-chip sp=8×tp=4 + tile-sharded VAE ---

# 480p (480×848), ~5:09 e2e
HY_MESH=8,4 HY_DIT_SP=1 HY_DIT_BF16=1 HY_TT_QWEN=1 HY_TT_VAE=1 HY_VAE_TILE=1 HY_VAE_TILE_PX=128 \
HY_FRAMES=121 HY_STEPS=50 HY_H=480 HY_W=848 \
$PY -m pytest $DIR/tests/e2e/test_stage2b_gen.py::test_stage2b_gen_qb2 -svv

# 720p (720×1280), ~9:12 e2e -- add HY_720P=1 (swaps in the 720p transformer + shift=9.0)
HY_720P=1 HY_MESH=8,4 HY_DIT_SP=1 HY_DIT_BF16=1 HY_TT_QWEN=1 HY_TT_VAE=1 HY_VAE_TILE=1 HY_VAE_TILE_PX=128 \
HY_FRAMES=121 HY_STEPS=50 HY_H=720 HY_W=1280 \
$PY -m pytest $DIR/tests/e2e/test_stage2b_gen.py::test_stage2b_gen_qb2 -svv

# --- image->video (i2v): HY_I2V=1 + HY_IMAGE=<first-frame>. Resolution is derived from the
#     image + the checkpoint's target_size (no HY_H/HY_W). 480p_i2v ~5:19, 720p_i2v ~9:27.
HY_I2V=1 HY_IMAGE=/path/to/first_frame.png \
HY_MESH=8,4 HY_DIT_SP=1 HY_DIT_BF16=1 HY_TT_QWEN=1 HY_TT_VAE=1 HY_VAE_TILE=1 HY_VAE_TILE_PX=128 \
HY_FRAMES=121 HY_STEPS=50 \
$PY -m pytest $DIR/tests/e2e/test_stage2b_gen.py::test_stage2b_gen_qb2 -svv
# 720p i2v: add HY_720P=1 (swaps the 720p_i2v transformer, target_size 960, shift=7.0)

# correctness
$PY -m pytest $DIR/tests/e2e/test_e2e_pipeline.py -s        # e2e gate (i2v+t2v, PCC ≥ 0.95)
$PY -m pytest $DIR/tests/e2e/test_real_weight_pcc.py -s     # real-weight DiT PCC (single + mesh)
$PY -m pytest $DIR/tests/pcc/ -s                            # all 18 per-module PCC tests

# per-step DiT perf (device_ms, trace+2CQ, capped 2-layer workload)
$PY -m pytest $DIR/tests/e2e/test_main_perf.py::test_main_perf -svv
```

### Generation flags (`test_stage2b_gen_qb2`)
All optional env vars; defaults below.

| flag | default | meaning |
|---|---|---|
| `HY_I2V` | `0` | `1` = **image→video** (`HunyuanVideo15ImageToVideoPipeline` + i2v checkpoint). Resolution derives from the image + `target_size`, so `HY_H`/`HY_W` are ignored |
| `HY_IMAGE` | (saved cat frame) | i2v conditioning first-frame image path; falls back to a saved frame, then a synthetic gradient |
| `HY_720P` | `0` | `1` = swap the 480p DiT for the **720p** transformer (+ scheduler shift 9.0 t2v / 7.0 i2v). For t2v pair with `HY_H=720 HY_W=1280` |
| `HY_SCHED_SHIFT` | `9.0` | Flow-match scheduler shift used by the 720p swap (720p t2v=9.0, i2v=7.0; 480p=5.0) |
| `HY_MESH` | (parametrized) | DiT mesh `rows,cols`. With `HY_DIT_SP=1`: **rows → sp, cols → tp** |
| `HY_DIT_SP` | `0` | Sequence parallelism (shard the latent seq across rows). **Required for multi-row meshes** |
| `HY_DIT_BF16` | `0` | bf16 DiT weights + block matmuls (used for all perf numbers) |
| `HY_DIT_QKV_SPLIT` | `0` | Opt in to tt_dit's fused `minimal_matmul_split` for both joint-attention QKV projections; retained legacy matmul + slices is the safe default pending 50-step quality validation |
| `HY_DIT_MMRS_OVERLAP` | `0` | Opt in to Hunyuan latent-stream fused matmul+TP reduce-scatter for attention-out and FFN-down projections. Requires Blackhole TP4 + `HY_DIT_BF16=1`; intended for SP8×TP4. Context projections and the final persistent all-gather retain the legacy path. Unqualified pending block PCC, 121f A/B, and generated-frame PCC |
| `HY_DIT_FUSED_HEADS` | `1` | Merge the joint-SDPA output with `nlp_concat_heads` instead of permute + reshape. Skips the `(B, S, H, D)` intermediate whose 4-wide local-head axis tile-pads to 32. Output is **bit-identical**; measured −12% denoise alone at 480p/121f/50 steps. **Default on**; set to `0` to restore the legacy permute + reshape path |
| `HY_DIT_FUSED_QKV_HEADS` | `1` | Build Q/K/V heads-major with `nlp_create_qkv_heads` straight from the fused `[q\|k\|v]` projection, removing three slices, three reshapes and six permutes, and running RoPE in `(B, H, S, D)`. Output is **bit-identical**; with `HY_DIT_FUSED_HEADS`, and against a `HY_CFG_PADDING_POLICY=masked` baseline that reproduces this README's published 1:58, measured **1:58 → 1:20 denoise (−32.2%)** and 345.5s → 289.5s e2e at 480p/121f/50 steps; −36% denoise against the shipped `separate` default. 720p I2V 121f/25 steps: 6.40 → 4.79 s/step (−25.2%), also bit-identical — smaller gain because quadratic attention dominates more at 111,600 tokens |
| `HY_FREE_DIT_BEFORE_VAE` | `0` | Release the 54 DiT blocks' device weights on the first VAE decode call. Decode is the last stage of a one-shot generation and never touches the DiT, which otherwise holds ~99% of device DRAM (measured 4.24 of 4.27 GB per bank). Frees 1350 tensors. Weights are **not** offloaded — the torch modules stay resident and `HY_DIT_WEIGHT_CACHE` reloads them in seconds. Required for `HY_VAE_HW_SHARD=1` at 121f; on the tiled path it is neutral-to-slightly-slower, so leave it off unless something downstream needs the DRAM |
| `HY_VAE_HW_SHARD` | `0` | Decode the VAE with the latent fractured across the mesh H/W and per-conv halo exchange instead of overlapping tiles. Runs at 121f with `HY_FREE_DIT_BEFORE_VAE=1 HY_VAE_ATTN_SDPA=1`, but is **~72s SLOWER** than the tiled path (222.8s vs 150.4s): it pays a per-convolution halo exchange across 32 chips, and on this fabric collectives never overlap compute, so that costs more than the 45-50% redundant tile decode it removes. **Keep the tiled path.** Output PCC 0.99864 vs tiled (tiling blends overlaps; sharding computes each pixel once) |
| `HY_FAST_WRITEOUT` | `0` | Write the output PNGs from a thread pool instead of serially. Measured **19.0s → 0.41s** for 121 frames (46x); PNG bytes are identical (frame PCC 1.00000000) |
| `HY_SAVE_GIF` | `1` | Write the animated GIF. It is the single most expensive writeout artifact — **13.2s of a 121-frame run**, and ~25 MB on disk — while the mp4 covers the same purpose at 1.4 MB. Set to `0` to skip it. With `HY_FAST_WRITEOUT=1` this takes total writeout from ~32s to ~0.4s (**−40.4s wall**) |
| `HY_DIT_SKIP_PARTS_STUBS` | `0` | Skip building the 4x54 `ada_layer_norm_zero` / `feed_forward` part stubs. They are read only by `_transformer_block_from_parts` (granularity `mid`); generation runs `composite` through the fused `s_blocks`, which builds its own AdaLN and FF weights — so on the production path these upload a second copy of the largest weights in the model and nothing reads them. Measured **48.3s → 17.3s DiT weight upload (−64%), −28s wall**, output bit-identical. Any access to a skipped stub raises, so `mid` fails loudly rather than diverging |
| `HY_VAE_WEIGHT_CACHE` | `0` | Cache the VAE's prepared causal-conv weights. `prepare_conv3d_weights` reformats every conv for the conv3d kernel on the host at adapter construction; caching its output makes VAE weight upload **12.6s → 0.81s (15.6×)**, output bit-identical. Costs **1.7 GB** per configuration (a tenth of the DiT cache). The cold run pays ~6.4s extra to populate. Directory from `HY_VAE_WEIGHT_CACHE_DIR`, else `HY_DIT_WEIGHT_CACHE_DIR`, else `TT_DIT_CACHE_DIR`, else `~/.cache/tt-dit`; keyed on mesh shape, core grid, dtype and H/W-sharding |
| `HY_DIT_WEIGHT_CACHE` | `0` | Cache prepared DiT weights to disk and reload them instead of re-running `ttnn.from_torch` every process. Uses `ttnn.DumpTensorMode.LOCAL`, which persists each device's own shard and restores placement, so the round trip is **bit-identical** (frame PCC 1.00000000, max abs pixel diff 0.0) and the cache is 1.00x the logical weight size. Measured 231.5s -> 181.9s wall (**-49.6s**, a setup cost so the saving is constant regardless of step count). **Costs ~16 GB of disk per configuration** — check free space first. Directory from `HY_DIT_WEIGHT_CACHE_DIR`, else `TT_DIT_CACHE_DIR`, else `~/.cache/tt-dit`; keyed on mesh shape, tp/sp, axes, dtype and `HY_DIT_RS_DOMAIN_BIAS` |
| `HY_DIT_SDPA_PRESET` | `hunyuan` | Ring-SDPA chunk preset. `wan_bh_sp8tp4` (q=288) is **measured and unusable at Hunyuan shapes**: at 121f it aborts with a statically-allocated circular-buffer clash against L1 buffers on core range [0-0 - 11-8]. The retained `hunyuan` preset (q=128, k=512) is the only working setting |
| `HY_DIT_SDPA_Q_CHUNK` / `HY_DIT_SDPA_K_CHUNK` | preset | Explicit positive, tile-aligned SDPA chunk overrides; invalid values fail at pipeline construction |
| `HY_TT_QWEN` | `0` | Request Qwen text-encode on device. A disjoint encoder submesh is preferred; otherwise it remains on host unless `HY_TT_QWEN_SHARED=1` |
| `HY_TT_QWEN_SHARED` | `0` | Sequentially load/encode/unload Qwen on the DiT's full mesh before constructing the DiT (TP on a KV-head-compatible axis + FSDP on the other axis; no overlapping mesh context) |
| `HY_TT_QWEN_KEEP_RESIDENT` | `0` | Keep TT Qwen weights after pre-encoding. Intended only for served repeated prompts with enough DRAM; sequential unload is safer for one-shot generation |
| `HY_TT_BYT5` | `0` | Request TT byT5 on a dedicated disjoint 1-device (TP1) or 1×2/2×1 (TP2) mesh. The checkpoint config is enforced fail-closed except `tie_word_embeddings`, which HF does not round-trip and which only ties an LM head an encoder does not have; full 8×4 placement and overlapping submeshes fail closed. `tests/pcc/test_byt5_encoder_pcc.py` passes (5/5); host remains the default because a committed 32-chip run has no disjoint mesh to spare |
| `HY_BYT5_VERIFY` / `HY_BYT5_PCC` | `1` / `0.99` | Fail-closed first-call check of the TT byT5 against the wrapped host encoder over the masked-in tokens (12 layers × ≤256 tokens, so a fraction of a second once per generation) |
| `HY_BYT5_ZERO_PAD` | `1` | Zero byT5 embeddings at masked positions, matching `HY_QWEN_ZERO_PAD`: the fused joint-attention kernel accepts no key mask, so padding surviving `_trim_to_valid` must be neutral rather than arbitrary. Hardware shows this is **defensive, not load bearing** — the full-sequence case passes at PCC 0.999931 with it off |
| `HY_PROMPT_CACHE` / `HY_PROMPT_CACHE_DIR` | `0` / `$TT_DIT_CACHE_DIR` | Persist the complete Qwen+byT5 positive/negative embedding tuple. A warm hit skips both encoders |
| `TT_DIT_CACHE_DIR` | unset | Enables prepared TT weight caches via `cache.load_model`; Qwen cold setup otherwise falls back to direct state-dict preparation |
| `HY_CFG_PADDING_POLICY` | `separate` | `separate` runs exact-length rows (mixed trace fails closed). `masked` is an opt-in SP-only fused-ring key mask: exact-length Qwen refinement + packed image/byT5/Qwen valid prefixes, zeroed invalid queries, one padded DiT batch/trace shape. Validated end to end on hardware at 13f/8 steps (225.04s, frame PCC 1.000000) but not yet at 121f/50 steps, and **only with `HY_TRACE=0`**. `error` fails; `legacy` retains unsafe longest-row batching |
| `HY_TT_SIGLIP` | `0` | I2V only: run the 27-layer SigLIP transformer on a reserved, disjoint 1x1 chip when spare hardware exists; stays on host for SP=8 and when VAE/Qwen claim the spare rows |
| `HY_TT_VAE` | `0` | VAE decode on device; auto **tile-shards** across the mesh when `ndev>1` |
| `HY_VAE_TILE` / `HY_VAE_TILE_PX` | `0` / `0` | Enable tiled VAE / per-tile px. **Use 128 at 121f** (192 fragments DRAM) |
| `HY_VAE_LEGACY_TILE_READBACK` / `HY_VAE_LEGACY_TILE_BLEND` | `0` / `0` | Restore per-round D2H or scalar host blending for VAE A/B and fallback |
| `HY_VAE_DEVICE_STITCH` | `0` | Opt-in TTNN tile blend/crop/stitch. Currently enabled only for a one-device VAE; multi-device tile batches remain fractured by tile index and safely fall back to the validated one-readback host stitch |
| `HY_VAE_HW_SHARD` | `0` | Experimental full-latent H/W-fractured decoder: per-convolution neighbor halo, global mid-block attention gather/repartition, logical edge repair, and one final D2H. Requires a 2D multi-device mesh. Keep off: real-weight 480p was slower than tile/single-readback and 720p OOMed in global attention |
| `HY_VAE_ATTN_CHUNK` | `0` | Opt-in mid-block attention query-block size in **tokens**. `0` means *disabled* (the monolithic `seq × seq` mask + score path), not frame-granular. Requests are clamped to `H*W` because a block may never cross a latent frame — that is exactly what lets a block drop the additive mask. Non-integer or negative values fail closed. **Hardware-validated** (7/7 device cases, PCC ≥ 0.9999984; 720p decode now fits) |
| `HY_VAE_ATTN_DIST` | `0` | Opt-in H/W-**distributed** mid-block attention: keep queries fractured, all-gather only K and V, and skip the post-attention `mesh_partition`. Only takes effect with `HY_VAE_HW_SHARD=1`. Host-proven; the device cases are unblocked but not yet run |
| `HY_VAE_ATTN_SDPA` | `0` | Opt-in `ttnn.transformer.scaled_dot_product_attention` for each mask-free query block instead of explicit matmul/softmax/matmul. Implies block-wise attention. The key chunk is derived from the head dim, not copied from Wan. **The `num_heads=1`/`head_dim=1024` geometry works on hardware** (PCC 0.9999913 at the derived `k_chunk=64`); not yet measured inside a real decode |
| `HY_FRAMES` / `HY_STEPS` | `13` / `50` | Frame count / denoise steps |
| `HY_H` / `HY_W` | `480` / `848` | Output height / width |
| `HY_TRACE` | `0` | **Do not enable.** Trace capture is exact but per-step replay is incorrect: 1 step is bit-identical to eager (PCC 1.000000), 8 steps give 0.237300. Same signature as the rejected heterogeneous trace (0.235647). Applies to every policy, including `masked` |
| `HY_DEVICE_RESIDENT_DENOISE` | `0` | `1` = keep the SP latent and FlowMatch Euler/CFG update on device in eager or trace mode; opt-in until real-weight generation and timing are validated |
| `HY_PROMPT` / `HY_NEG_PROMPT` | cat / — | Positive / negative prompt |
| `HY_OUT` / `HY_FPS` | `/tmp/hy15_stage2b_qb2` / `24` | Output dir / video fps |

## Performance

> **Every VAE timing below is PROVISIONAL and probably mixes cold and warm
> decodes.** 480p H/W-sharded was recorded at 92.97s and later measured at
> 44.23s; one chunked run measured 140.40s cold against 64.37s warm; and 720p
> tile decode measured 13.07s against 480p tile at 37.55s, which cannot be right
> at a higher resolution. Numbers are labelled `PROVISIONAL (cold?)` at the point
> of use and retained rather than deleted so a clean all-warm table can be diffed
> against them. Do not pick a VAE path on these figures. PCC and memory results
> are unaffected — they do not depend on cache state.

### Full 121-frame t2v generation (real weights, 50 steps)
| config | denoise | VAE | **e2e** |
|---|---|---|---|
| 480p, baseline (8-chip) | ~15:00 | — | 18:52 |
| 480p, 16-chip sp=2 | 6:21 | ~2:30 | 10:38 |
| 480p, 32-chip sp=4, replicated VAE | 3:05 | ~5:00 | 9:47 |
| 480p, 32-chip sp=4, tile-sharded VAE | 2:49 | ~2:00 | 5:59 |
| **480p t2v, 32-chip sp=8×tp=4 + persist-RS + SDPA-cfg** | 1:51 | ~2:00 | **5:09** ✅ |
| **720p t2v (720×1280), same sp=8×tp=4 config** | 5:00 | ~2:00 | **9:12** ✅ |
| **480p i2v (480×848), same config** | 1:58 | ~2:00 | **5:19** ✅ |
| **720p i2v (720×1280), same config** | 5:07 | ~2:00 | **9:27** ✅ |

The denoise is **attention/CCL-bound at 121f** (seq ~49k) — a different regime than the
dispatch-bound `device_ms` work below. 720p is **~2.85× the 480p denoise/step** (5.76 vs
2.02 s/it): the latent grid is ~2.25× the tokens (720×1280 → 45×80 vs 480×848 → 30×53) and
attention is O(seq²)/sp. The two 480p levers, in order:
1. **sp=8×tp=4** (`HY_MESH=8,4`): sequence-parallel divides the O(seq²) attention, so
   doubling SP cut denoise **3.24 → 2.15 s/it**, e2e 5:59 → 5:13 (zero code).
2. **Two block levers** (`_stubs/hunyuan_video15_transformer_block.py`): reduce-scatter uses
   its persistent buffer in the bf16 path, and SDPA gets its own compute config with
   `fp32_dest_acc_en=False` → **2.02 s/it**, e2e 5:13 → 5:09.

The **VAE decode is tile-sharded** (`_decode_batch_sharded`, `085be79`): the ~45 tiles are
batched + sharded so each device decodes one tile/round (2 passes, not 45 sequential),
reusing the full mesh after denoise. Tile-round outputs now stay on device, concatenate
there, and use one final readback instead of one readback/synchronization per round.
The remaining host stitch is vectorized and bit-exact against the prior scalar blend in
40 property cases covering zero/oversized overlaps and odd edge tiles; one boundary-weight
case and four device/round ordering cases also pass (45 static tests total).
No VAE or end-to-end latency delta is claimed yet because the shared Galaxy had active
generation jobs during validation. *(sp=2×tp=16 / any tp=16 are impossible on the 8×4
Galaxy; CCL Ring / num_links>2 need a torus fabric FABRIC_1D doesn't provide.)*

The experimental `HY_VAE_HW_SHARD=1` path now threads the tt_dit
`VaeHWParallelConfig`/`CCLManager` contract through the complete decoder. The uneven
30×53 480p and 45×80 720p latent grids use equal-storage H/W shards; every 3×3 causal
convolution performs a one-cell neighbor exchange, each upsample doubles logical H/W,
and mid-block attention gathers/crops full H/W before repartition. T is never sharded:
31 latent frames still expand as `1 + (31 - 1) * 4 = 121`. Because Hunyuan uses
replicate rather than zero boundary padding, uneven storage-only cells are restored
after convolution/channel-to-space before the next halo. The repair is rank-local:
each rank forms candidate H/W tails from its local logical edge, and cached masks
fractured over the same mesh select them only on the final uneven ranks. H is repaired
before W so the bottom-right corner exactly replicates the logical corner. This path
uses no collective; full H/W gather remains only around global mid-block attention.

Random-weight 1x2, 2x2, and 8x4 decoder gates passed at PCC 1.0; four-chip halo convolution,
uneven temporal/spatial upsample, and gathered-attention gates also passed at PCC 1.0.
Rank-local production-grid edge tests cover every 8x4 rank for 30x53 and 45x80 through
all four 2x upsample stages; exact 30x53 and 45x80 device tests pass on 8x4 at PCC >=
0.999 with zero edge-fill gathers. In the small decoder, graph gathers changed from 11
to 1 on 1x2 and 22 to 2 on both 2x2 and 8x4—the remaining count is one per active
attention mesh axis. The H/W path performs one full-latent decode (no overlapping tile
decode) and one final D2H. Edge masks add one cached H2D upload per uneven axis/scale:
10 masks (about 2.44 MB BF16 total) for 480p and 5 masks (about 2.62 MB) for 720p across
stages 0..4; later layers reuse them. Cached test call time changed 0.31s to 0.30s on
1x2 and 0.25s to 0.21s on 2x2; this is not a production performance result.

Real-weight offline 8x4 validation used the cached 480p I2V VAE (also shared by 720p).
The smallest supported `(T,H,W)=(2,8,4)` latent matched the host decoder at PCC
0.999980645 in both H/W-sharded and replicated modes. First decode was 35.41s H/W
versus 25.81s replicated *(PROVISIONAL (cold?))*; H/W issued two global gathers. A
one-latent-frame probe is unsupported by the current temporal upsample and fails
before PCC with a zero-length branch.

For 121-frame 480p `(31,30,53)`, H/W decode took 92.97s plus 77.6ms final D2H, versus
37.55s total for the current tile/single-readback path: H/W was 2.48x slower.
***PROVISIONAL (cold?): 480p H/W was later measured at 44.23s for the same work, and
the 37.55s tile figure is itself suspect because 720p tile measured 13.07s. Do not
quote the 2.48x verdict until the all-warm table lands.*** The
full-video PCC between those device paths was 0.998960435 and final-frame PCC was
0.999074800 — those, and the gather/D2H counts, do not depend on cache state and
stand. The tile path decodes 15 padded 16x16 tiles (2.42x latent-area work) in
one 32-device round. Synchronized post-decode DRAM allocation was 8.61 GiB for
H/W versus 1.80 GiB for tile; these checkpoints are not allocator peak measurements.
At 720p `(31,45,80)`, global attention OOMed requesting a 24,916,262,912-byte matmul
buffer with only 579,611,264 bytes free per bank; **chunked attention resolves this**
— the same gate now needs 0.74 GiB and 720p chunked decode completes at output
`[1,3,121,720,1280]`. Keep `HY_VAE_HW_SHARD=0` until the matched VAE-only gates are
repeated on warm numbers.

#### Where the mid-block attention memory goes

The mid-block is a **single-head** attention over the entire spatiotemporal extent
(`C = 1024`, one such block in the graph) with a block-causal mask: a query in latent
frame `f` attends to every token of frames `0..f`. The monolithic form allocates three
`S_pad × S_pad` bf16 tensors — the cached mask, the raw scores, and the scaled/softmaxed
scores — where `S = T*H*W` and `S_pad` rounds up to the 32-row tile grid. At 720p
`(31,45,80)`, `S = 111,600` and `S_pad = 111,616`, so each one is **23.21 GiB**; that is
precisely the 24,916,262,912-byte request in the OOM. At 480p the same term is 4.53 GiB,
which accounts for 4.53 of the 8.61 GiB observed still resident after decode, because
`_mask_cache` deliberately keeps the mask alive across tile rounds.

Because every query row in a block that stays inside one frame shares the same key
prefix, the block can slice K/V to `[0, (f+1)*H*W)` and drop the mask **entirely** —
this is a rearrangement, not an approximation. Peak then falls to one query block:

| resolution | monolithic (each of 3) | `HY_VAE_ATTN_CHUNK=1024` peak | reduction |
|---|---:|---:|---:|
| 720p `(31,45,80)` | 23.21 GiB | 218 MiB | 109× |
| 480p `(31,30,53)` | 4.53 GiB | 96 MiB | 48× |

182 host cases agree with the masked reference to 1e-12 in float64, including a
cross-check against the diffusers mask and `torch.nn.functional.scaled_dot_product_attention`.

**Chunking is validated on hardware.** All 7 device cases pass — PCC 0.9999985 at
query chunks 1/7/32, 0.9999988 at 512, 0.9999984 sharded — and the block's peak DRAM
falls 9,207,808 → 1,228,800 B. The pre-existing VAE suite passes unchanged with
chunking forced on (38 tests), the 720p gate that demanded 23.2 GiB now needs
0.74 GiB, and a matched 480p A/B held cross-path PCC 0.9999899 while reclaiming
4753 MiB of device memory and dropping host peak RSS 32.3 → 9.4 GiB. The flash-SDPA
variant also works at the awkward `num_heads=1`/`head_dim=1024` geometry: PCC
0.9999913 at the derived `k_chunk=64` and 0.9999936 at `C=64`/`k_chunk=1760`. That
was the only question a static reading of the kernel could not settle.

**Chunking does not unlock larger tiles.** At the production `HY_VAE_TILE_PX=128` the
decoder tail dominates at 484 MiB against only 22.5 MiB of attention, so attention is not
what caps tile size; it only overtakes the tail past 512px. Any claim that chunked
attention would allow bigger tiles is wrong at the sizes actually used.

#### Distributing the attention instead of replicating it

`norm` reduces only over channels and `to_q` is a 1×1×1 causal convolution — with
`kernel == 1` its `t_front` and `pad_hw` are both zero, so it performs no replicate
padding and issues no neighbor exchange. Both are therefore pointwise in H/W, which means
**Q never needs the all-gather**; only K and V do, because attention reduces over every
spatial token of the causal prefix. `HY_VAE_ATTN_DIST=1` exploits that: each rank
normalizes and projects only its own shard, all-gathers K and V, computes exactly the
query rows it already stores, and returns them still fractured. The post-attention
`mesh_partition` disappears, and on the 8×4 Galaxy the score-element count per rank drops
**28.4×** at 480p and **30.0×** at 720p (the shortfall from 32× is equal-storage padding
overhead: 1.127× and 1.067×). Even with no chunk request, one frame of rank-local queries
is a 6.0 MiB block at 480p and 27.2 MiB at 720p.

Replicate-padded storage rows need no explicit output repair. Given the shared keys and
values, every remaining stage is a per-position map, so a padded row holding a copy of the
last logical row necessarily produces a copy of that row's output — exactly the semantics
`replicate_pad_to_plan` and `canonicalize_replicated_shard_edges` maintain elsewhere. The
block canonicalizes its input first so it does not depend on upstream hygiene; padded K/V
rows are still cropped before the sequence is flattened, since duplicated keys would
otherwise reweight the softmax. Partitions that leave a rank with no logical row at all
are rejected, unchanged from the existing edge-fill contract; both production grids are
legal on 8×4 (30 rows → 4-row shards with a 2-row tail, 45 rows → 6-row shards with a
3-row tail).

85 host cases pass, including 24 rank-decomposition equivalence cases (agreeing to 1e-12
in float64 across even, H-only, W-only, and both-uneven partitions plus the real 30×53
grid on a simulated 8×4 mesh) and a test that isolates the query path by freezing K/V to
prove a remote rank's rows cannot reach this rank's query.

The 18 device cases were blocked by their own parameterization, not by hardware: they
asked for a 5-row latent on a 4-rank H axis, which is 2-row shards with a 3-row tail,
so the final rank held nothing but padding and had no logical row to replicate an edge
from. That partition is illegal by design. Each device case is now paired with a
geometry its mesh can partition — even, H-only-uneven, W-only-uneven and both-uneven,
plus the real 30×53 and 45×80 production grids on 8×4 — and
`test_every_hardware_case_partitions_legally` guards the parameter lists on host so the
same edit cannot reach hardware again. The rejection message now names the offending H,
the per-rank row count, the rank count, and the smallest legal H for that mesh.

### Per-step DiT `device_ms` (trace+2CQ, capped 2-layer / M=32 profiling workload)
| phase | key levers | device_ms |
|---|---|---|
| baseline | — | 6.43 |
| bias→matmul fusion | fold every `Linear` bias into the `ttnn.linear` epilogue | 6.02 |
| matmul fusion | AdaLN 6-way modulation (C→6C); column-parallel QKV | 5.37 |
| ternary fusion | `addcmul` for AdaLN modulation + gated residuals | 5.31 |
| LayerNorm width-sharding | `norm2` / `_adazero` / `norm_out` LN over a core row | 5.17 |
| launch dedup | row-linear bias fold, RoPE `mul+addcmul`, AdaLN slice, silu/RoPE hoist | **5.10** |

16 wins + an L1-fit guard (`d0d07d0`). A hand-authored **tt-lang fused-FFN kernel** was
on-device correct (PCC 0.99999) but **+3%** — the FF matmul is dispatch-bound at M=32, so a
compute kernel can't beat the tuned `ttnn` matmul. (These `device_ms` wins are at the
profiled scale; the 121f e2e win comes from the sequence-parallel + VAE levers above.)

The joint-attention QKV projections now reuse tt_dit's `minimal_matmul_split`, returning
Q/K/V directly instead of launching three slices after each fused projection (six slices
removed per block, 324 across 54 blocks). The focused hardware comparison passed all four
legacy/reference gates at PCC >= 0.99. In a cached 480p I2V 13f/4-step A/B with the default
mixed-length conditioning, wall time changed 239.29s → 234.71s (-1.9%), but generated-frame
PCC was only 0.950332. With equal-length conditioning, wall time changed 226.23s → 229.91s
(+1.6%) and frame PCC was 0.996543. Because the latency result was not repeatable and the
default-conditioning frame divergence needs a 50-step check, `HY_DIT_QKV_SPLIT=0` remains
the safe default.

### Trace: replay is incorrect, and the economics are better than believed
**Correctness first: do not enable `HY_TRACE=1`.** Capture is exact — a 1-step traced
generation is bit-identical to eager (PCC 1.000000) — but per-step replay is not:
8 steps give aggregate PCC 0.237300 with per-frame PCC degrading 0.9747 → 0.8648 →
negative (−0.0657 … −0.3122). Frame 0 survives only because i2v anchors it to the
conditioning image, not because the first replayed step is right. This is the same
signature as the already-rejected heterogeneous trace (0.235647 in the flag table
above), so the two are one defect in per-step replay rather than two. Ruled out at
source level: time-embed and patch-embed placement, the `traced=True` copy at
`tt/pipeline.py:1135`, and the Euler update (shared with the exact eager path).

Warm timing fits over `n` steps are eager ≈ `1.72 + 1.016·n` s and trace ≈
`7.27 + 0.416·n` s, so break-even is **~9.3 steps** — well inside the 50-step default,
and far better than the ~71-step figure below, which came from a cold trace. The
break-even is never reached today because `tests/e2e/test_stage2b_gen.py:288-289`
releases the trace in a `finally`, so every run repays the ~7.27 s capture. The
pipeline already guards capture on `_trace_id is None`, so reuse needs no new
mechanism and would drop break-even to ~3 steps. Worth doing only after replay is
correct.

The earlier (cold) matched user A/B measured eager steady state at ~2.33 s/step and
trace replay at ~2.21 s/step, a real ~5% trace benefit. Eager denoise/e2e were
1:58/5:19; trace was 1:59/5:24 because trace step 1 took 10.78 s versus 2.21 s for
steps 46–49. The old fixed startup excess was therefore ~8.5 s, giving a one-shot
crossover of about 71 steps.

Hardware showed that mesh capture cannot load uncached programs, so a compile forward is
required before capture. Capture itself does not execute; the safe sequence is therefore
compile warmup → capture → explicit first execution. Attempts to reuse the compile output
while deferring the first trace execution passed a small synthetic replay gate but failed
real 13f generated-frame checks (PCC 0.422903 and 0.004703 in two intermediate variants).
The implementation now uses the conservative sequence above. Its host regression passes;
the final mesh retest and all further VAE/121f work were stopped when another generation
job acquired the Galaxy.

Matched equal-length 13f/4-step eager wall/denoise results were: legacy 226.23s/5s,
QKV 229.91s/5s, device-resident 238.54s/8s (frame PCC 0.998612), and QKV+resident
236.03s/5s (PCC 0.998012).
All had 8 transformer calls and 4 physical device runs. Neither device residency nor QKV
improved this short one-shot wall-time measurement, so eager legacy QKV remains the best
safe configuration. Trace remains disabled.

### What runs where (sp=8×tp=4)
| stage | placement | note |
|---|---|---|
| Qwen 2.5-VL | **host by default; device opt-in** | full-mesh mode is sequential load → encode → unload → DiT load; valid-token PCC target >=0.999 |
| byT5 | **host by construction on a committed run** | TP must divide both 6 heads and `d_model=1472`, so only 1- and 2-device meshes are legal; it can never share the 8×4 DiT mesh |
| Latent init + scheduler step | **host by default; device opt-in** | `HY_DEVICE_RESIDENT_DENOISE=1` keeps CFG/Euler + latent resident and gathers once at the end |
| **DiT denoise** (50 steps) | **on device** | all 32 chips, sp=8 × tp=4 |
| **VAE decode** | **on device** | tile-sharded across all 32 chips |
| Frame post-proc → mp4/gif | **host (CPU)** | one-time save |

Only the two heavy compute stages (DiT, VAE) run on device; host work is one-time or a
cheap per-step update. On-device Qwen is available at any sp via `HY_TT_QWEN_SHARED=1`
(TP on a compatible mesh axis + FSDP on the other axis) but was ~1 min *slower* for a
cold one-shot video (5:59 host versus 6:58 shared TT). Prepared-weight and prompt-embedding
caches target warm served requests; no warm timing has been measured yet.

## PCC / correctness
- **Per-component (18/18):** every stub has `tests/pcc/test_<module>.py` (+ a `_mesh` variant
  for the transformer block); all pass native ttnn.
- **End-to-end gate** (`test_e2e_pipeline.py`, 2-layer seed-0 reference weights): runs **both
  the i2v and t2v regimes** across 3 granularities, asserts min PCC ≥ 0.95 and that all 18
  graduated stubs are invoked (**min PCC 0.999979**).
- **Real-weight DiT** (`test_real_weight_pcc.py`, single + 24-chip mesh): threshold 0.99.
- **sp-degree consistency:** validated by 1-step frame-PCC on the cached community weights
  (matched prompt/seed) — sp=8 vs sp=4 = 0.9971.
- **Harness determinism (so cross-path PCC means something):** the seed is pinned at
  `torch.Generator().manual_seed(0)` inside the shared `_run()` helper
  (`tests/e2e/test_stage2b_gen.py:280`), and an eager-vs-eager repeat is bit-identical
  (PCC 1.000000, maxabs 0.000000). Every cross-path generated-frame PCC quoted here,
  including the masked-CFG 1.000000, is therefore a real comparison.
- **Qwen conditioning:** the fp32 eager attention core measured valid-token PCC 0.99984
  versus host. The default `HY_CFG_PADDING_POLICY=separate` still trims and runs each
  mixed row independently. The opt-in `masked` path preserves exact Qwen refinement
  lengths, packs all valid conditioning regions into a row prefix, masks invalid fused
  ring-SDPA keys, and zeroes invalid query states. It now completes a real i2v 480p
  13-frame/8-step generation in 225.04 s at frame PCC 1.000000 with
  `HY_DEVICE_RESIDENT_DENOISE=1`; 121-frame/50-step quality is still pending.
- **byT5 placement is settled, not pending:** tensor parallelism has to divide both
  `num_heads` (6) and `d_model` (1472 = 2⁶ × 23), so the only legal factors are 1 and 2 and
  neither Galaxy mesh axis (8 or 4) can express either. byT5 therefore can never share the
  DiT mesh, and on a committed 32-chip run there is no disjoint mesh left to give it — it
  stays on host by construction. It costs a fraction of a second there (12 layers × ≤256
  tokens), so this is not a latency concern. The TP1/TP2 device path remains useful only
  for a deployment that reserves chips outside the DiT mesh.
- **byT5 conditioning:** the real-weight hardware gate passes on all 5 cases — TP(1,1)
  0.999935, TP(1,2) 0.999938, full sequence without zero-padding 0.999931, batched-row
  consistency ~1.000, adapter self-check passed, all outputs finite and non-zero.
  33 host-only tests pass. Two findings from that run: padding neutralization is **not**
  load bearing (the full-sequence case passes with it off), and the
  `(mask − 1)·inf → 0·inf = NaN` hazard flagged in the shared T5 additive-mask
  expression does **not** manifest on device.
- **byT5 checkpoint contract — one field is deliberately not enforced.** The gate first
  failed closed on `tie_word_embeddings=True (expected False)` and nothing else. The
  checkpoint's own `config.json` stores `false` while `T5Config.from_pretrained` returns
  `True`: HuggingFace does not round-trip the field, so the strict check was rejecting
  the exact checkpoint this port targets. The field only ties an LM head to the input
  embedding, and `T5EncoderModel` has no LM head, so it cannot move an encoder
  activation — it is the one field in the contract that provably cannot affect numerics.
  It was dropped from the strict set and is instead reported through
  `ByT5Support.reason` when the parsed value disagrees. Every other check stays
  fail-closed, and a checkpoint that genuinely carried an LM head is still rejected by
  the strict unexpected-key check in `load_torch_state_dict`.
- **Resident scheduler contract:** 18 host property/guard cases match diffusers exactly across
  shifts 5/9, two shapes, native/original CFG, and CFG enabled/disabled. Two focused
  Blackhole tests cover the actual TTNN CFG/Euler kernels. A real-weight 13f/4-step
  equal-conditioning run passed at frame PCC 0.998612 versus eager, but was 12.31s slower
  wall-clock; keep it opt-in.
- **VAE tile stitch:** 40 host property cases match every pixel bit-for-bit against the
  legacy blend, including odd edge tiles; five boundary/order cases also pass. The focused
  2-chip single-readback test passed at PCC 0.999998593. The requested representative
  legacy/single-readback timing A/B was not started after another generation job acquired
  the Galaxy.

## Notes
- **Opening a 2×4 mesh directly fails; open 8×4 and take a submesh.** On a completely
  quiet machine with no other job, a direct 2×4 open fails during fabric-router
  synchronization on device 1 (master `chan=3` at `0xa1b1c1d1`, `chan=4` stuck at
  `STARTED` `0xa0b0c0d0`), while an 8×4 open on the same machine succeeds. An earlier
  session blamed contention for this; that is disproven. Any test wanting a smaller
  mesh should open 8×4 and submesh out of it.
- **720p color fix:** the 720p checkpoint's flow-match scheduler needs `shift=9.0` (480p=5.0;
  i2v=7.0). `FlowMatchEulerDiscreteScheduler.set_timesteps` reads `self.shift` (property
  backed by `self._shift`), **not** `self.config.shift` — so `register_to_config(shift=9)` is
  a silent no-op that under-shifts the trajectory into oversaturated/blown-out frames. The
  `HY_720P` path calls `set_shift(9.0)`.
- **i2v generation:** uses a separate checkpoint (`*-i2v`, ~54 GB) and the
  `HunyuanVideo15ImageToVideoPipeline`, which (a) runs a **SigLIP `image_encoder`** on the
  input image → `image_embeds` (729 tokens, to the DiT's image projection), and (b)
  **VAE-encodes the first frame** and concatenates it into the DiT's conditioning channels
  (`in_channels=65` = 32 noise + 32 image-cond + 1 mask). It takes **no `height`/`width`** —
  the output resolution comes from the image's aspect ratio + the checkpoint's `target_size`
  (`calculate_default_height_width`, then a crop-resize). The DiT itself is unchanged (same
  stubs; `task="i2v"` keeps the image tokens active instead of masking them). The 720p_i2v
  transformer-swap must also refresh the pipeline's cached `target_size` (640→960) or it
  keeps bucketing to 480p.
- **Reference vs real weights:** the per-component / e2e-gate tests use deterministic-random
  (seed 0) weights shrunk to `num_layers=2` for CPU feasibility; `test_real_weight_pcc*` and
  generation use the real checkpoint (community diffusers conversion).
- **L1-fit guard:** width-shard LN needs ~16 MB/core at 121f, and its static circular
  buffers measured 2.35 MiB even at the 13f `block_h=25` shape (bank is 1.5 MiB). `_wln`
  now keeps the sharded optimization to `block_h<=8` and otherwise uses interleaved LN.
