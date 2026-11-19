# Host operations on the TT inference path

`Alpamayo2EdgeTT.infer` (`tt/model.py`) runs the vision tower, the LM prefill and decode, the expert, and
the action in/out projections on the device. This page lists everything that still runs on the host
or crosses the PCIe link for one frame of B samples, with its call site and the reason it stays on
the host. Paths are relative to `models/experimental/alpamayo2_edge/`.

## Per frame, once

| Host op | Call site | Transfer | Why on host |
|---|---|---|---|
| Input preparation: frame ordering, uint8 bicubic smart-resize, block-major patchify, rescale + normalize, chat template, tokenization, history-token fusion, `get_rope_index` | `reference/inputs.py` `prepare_inputs` (the caller runs it; the demo in `demo/demo.py`) | none | String and tokenizer work with data-dependent shapes, ported from the upstream processor; the TT and reference paths share it, so their inputs are bitwise equal |
| Patch upload, `[N, 768]` bf16 | `tt/model.py` `encode_images` | ~17 MB up (N = 11568) | The input itself |
| Vision `cu_seqlens` | `tt/model.py` `encode_images` -> `reference/vision.py` `vision_cu_seqlens` | op argument | A Python list of image boundaries that the SDPA op takes as an argument |
| Prompt ids `[1, S]`, M-RoPE positions `[3, 1, S]` fp32 and the image-token mask `[1, S]` | `tt/generate.py` `generate_cot` (prefill call) | ~60 KB up | Inputs of the prefill |
| Prefill last-token logits `[1, V]` readback | `tt/generate.py` `generate_cot` | 270 KB down | The first CoT token is sampled on the host, as every later one is (below) |
| `rope_deltas` per sample `[B]` int32 | `tt/generate.py` `generate_cot` | tiny up | Decode rope offset, constant over the frame |
| Anchor search and prefix length | `tt/generate.py` `generate_cot` -> `reference/model.py` `find_anchor_offsets` | none | Plain index bookkeeping over the generated ids |
| Expert positions `rope_deltas + offset + arange(64)` | `tt/expert.py` `make_expert_frame` -> `reference/action.py` `expert_position_ids` | 256 B x B up; cos/sin computed on device | Depends on each sample's anchor offset, known only after decode |
| Expert attention mask `[B, 1, 64, p_pad + 64]` | `tt/expert.py` `make_expert_frame` -> `reference/action.py` `expert_mask` | ~440 KB x B up (bf16, p_pad = 3392) | Data-dependent per sample (offset, prompt padding); built once per frame and shared by all 28 layers and all 10 flow steps |
| Flow noise `[B, 64, 2]` | `tt/model.py` `run` (`torch.randn` with the seeded generator) | none (enters through the flow features) | Seeded host draw, identical to the reference's |
| `action_to_traj`: t0 state estimation (regularized least squares over the 20-pose history) and unicycle integration to xyz and rotations | `tt/model.py` `run` -> `reference/action.py` `action_to_traj` | none | 64 x 2 numbers; cumulative sums and small linear solves that want fp32 and cost microseconds |

## Once per grid layout or row count, cached on device

| Host op | Call site | Why on host |
|---|---|---|
| Vision position embeddings: antialiased bilinear resize of the 16x16 learned grid to each camera grid, block-major packing | `tt/vision.py` `PosEmbedCache.get` -> `reference/vision.py` `vision_position_embeddings` | Antialiased interpolation has no device op; the camera resolutions are fixed, so each grid layout is computed and uploaded once and reused by every frame |
| Identity RoPE tables (cos = 1, sin = 0) for the vision blocks | `tt/vision.py` `TtVisionTower.identity_rope` | Constant per row count; uploaded once |

## Per CoT decode step (up to 256 steps, 128 in the teacher-forced tests)

| Host op | Call site | Transfer | Why on host |
|---|---|---|---|
| Next-token ids `[B]` uint32 upload | `tt/generate.py` `generate_cot` (`decode`) | tiny up | Input of the decode step |
| Logits `[B, V]` readback, bf16 -> fp32 | `tt/generate.py` `generate_cot` (`decode`) | 270 KB x B down | Sampling runs on the host |
| Logit processing and sampling: ban the trajectory codes and EOS, temperature 0.6, top-p 0.98 (a sort over 135168 entries), multinomial draw with the seeded generator | `tt/generate.py` `sample` -> `reference/model.py` `process_logits` | none | Bitwise the reference's sampler and generator stream, so a seed gives the same draw on both paths; tt_dit's device top-k does not implement this processor chain |
| Stop logic (every sample has emitted the anchor) | `tt/generate.py` `generate_cot` | none | Data-dependent loop exit |

The teacher-forced path (`forced_cot`) skips the sampling but still reads back the logits; with
`keep_intermediates` they are kept for the tests.

## Per flow step (10 Euler steps)

| Host op | Call site | Transfer | Why on host |
|---|---|---|---|
| Fourier features of the noisy actions and the timestep, fp32 sin/cos | `tt/action.py` `action_velocity_tt` -> `action_features` | `[B, 64, 60]` bf16 up (7.5 KB x B) | Arguments reach 2π·100·\|x\| rad, hundreds of radians a few sigma out; bf16 sin/cos on the device cannot resolve them |
| Velocity readback `[B, 64, 2]` | `tt/action.py` `action_velocity_tt` | 0.5 KB x B down | The Euler update runs on the host |
| Euler update `x += dt * v` with fp32 `linspace` timesteps | `tt/action.py` `sample_actions_tt` -> `reference/action.py` `euler_sample` | none | Keeps x in fp32 between steps, bitwise the reference schedule; moving it to the device needs a device sin/cos with fp32 range reduction first |

## Test-only readbacks

With `keep_intermediates=True` (`tt/model.py` `run`) the vision embeddings `[V, 2048]` and every
step's logits are read back for the PCC tests. `infer` does not request them.
