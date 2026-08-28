# FLUX.2 Klein 9B — transformer (TTNN, T3K / TP=8)

End-to-end TTNN pipeline for the **transformer** component of
[`black-forest-labs/FLUX.2-klein-9B`](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B),
i.e. `diffusers.Flux2Transformer2DModel`: 9.08 B parameters, 8 dual-stream DiT
blocks + 24 single-stream parallel blocks, `inner_dim = 4096` (32 heads x 128).

This is the denoiser, not a whole image pipeline. The VAE and the text encoder
are separate models and are **not** in this checkpoint — the checkpoint dir ships
`config.json` plus two safetensors shards and nothing else (no tokenizer, no
processor, no `generation_config`). So the component's real input is what the
enclosing latent-diffusion pipeline hands it, and its real output is the
velocity prediction that pipeline integrates.

## The two Calls

| | entrypoint | output | golden |
|---|---|---|---|
| **Call 1** `denoise_step` | `tt/pipeline.py::run_denoise_step` | velocity prediction `[1, S_img, 128]` | `Flux2Transformer2DModel.forward` — this checkpoint has no `generate()`; `forward` **is** the reference callable |
| **Call 2** `denoise_latents` | `tt/pipeline.py::run_denoise_latents` | final denoised latents `[1, S_img, 128]` + the unpacked grid `[1, 128, h, w]` | the same forward driven through the identical flow-match Euler schedule |

Call 2 is the real task. The latents stay **resident on device** for the whole
loop and the Euler update `x <- x + (σ' − σ)·v` is done in ttnn, so step *i+1*
consumes step *i*'s actual TT output — no reference tensor is injected at any
joint.

### How long the loop runs

A denoise loop has no stop token: there is no `eos_token_id` or model-specific
stop id in `config.json`, and the checkpoint ships no `generation_config.json`
and no scheduler config. Neither the stop-token rule nor the config-length rule
has anything to read, so **N is chosen**: 4 by default, because this is the
distilled Klein variant (`config._name_or_path = klein-9b-distilled-diffusers`),
which is built for few-step sampling. It is clamped to `[1, 50]`
(`tt/inputs.py::MIN_STEPS`/`MAX_STEPS`) so the loop cannot run away.

What is *not* chosen is the schedule. `tt/inputs.py::sigma_schedule` reproduces
Source A's own `Flux2Pipeline`: `np.linspace(1, 1/N, N)`, exponential time-shift
with the empirical `mu` computed from `image_seq_len`, trailing `0.0`. It is
computed **once** and the same list drives the TT loop and the HF golden, so
`--num-steps` moves both sides together.

## Running it

```bash
export TT_METAL_HOME=$PWD PYTHONPATH=$PWD ARCH_NAME=wormhole_b0
source python_env/bin/activate

# Call 1 -- one forward, with the PCC against the HF golden
python -m models.demos.flux_2_klein_9b_transformer.demo.demo_denoise_step \
    --height 256 --width 256 --txt-len 64 --check-pcc

# Call 2 -- the full 4-step Euler loop
python -m models.demos.flux_2_klein_9b_transformer.demo.demo_denoise_latents \
    --height 256 --width 256 --txt-len 64 --num-steps 4 --check-pcc
```

Flags: `--height --width --txt-len --num-steps --seed --layers --dual-layers
--single-layers --tp --trace-region-size --check-pcc --output-dir`.
Both demos write their real output tensors and a json summary under
`generated/flux_2_klein_9b_transformer/`.

`--check-pcc` also loads the float32 reference (~36 GB of host RAM) and prints
the same `e2e PCC=` line the tests print — both call the same
`tt/pipeline.py` function, so there is one implementation and one number.

### Tests

```bash
pytest models/demos/flux_2_klein_9b_transformer/tests/e2e -s
```

One `pytest` run, one mesh, one pipeline build (session fixtures in
`tests/e2e/conftest.py`), at the checkpoint's full depth and the gate shape
256x256 / `S_txt=64` (`S_img=256`, `S_joint=320`, 4 steps).
`TT_FLUX2_E2E_{LAYERS,DUAL_LAYERS,SINGLE_LAYERS,HEIGHT,WIDTH,TXT_LEN,STEPS}`
cap it for a fast wiring loop; every test prints the depth it actually ran at.

| file | what it gates |
|---|---|
| `test_e2e_denoise_step.py` | Gate 1 (native stubs, unmodified), Gate 2 (all 18 invoked, exact counts), Gate 3 (Call 1 PCC) |
| `test_e2e_denoise_latents.py` | one shared schedule, Gate 2 over the loop, Gate 3 (per-step + final-latents PCC) |
| `test_trace_contract.py` | `PIPELINE_STAGES` + the `denoise_*` hooks, the depth knobs, the host-op observer, a real device trace capture |

## Measured

At TP=8 on a T3K (mesh 1x8, `FABRIC_1D`), full depth (8 dual + 24 single),
256x256 / `S_txt=64` (`S_joint=320`), against the float32 HF golden on
byte-identical input. Required threshold: 0.95.

| | PCC |
|---|---|
| Call 1 — velocity prediction | **0.9947** |
| Call 2 — final latents, 4 steps | **0.9990** |
| Call 2 — per step | 0.99998, 0.99996, 0.99989, 0.99903 |
| traced vs untraced forward | **1.0000** |

Call 1 is the *lower* of the two on purpose: the velocity prediction is the raw
output of all 32 blocks, so it carries the whole accumulated bfloat16 rounding of
a full-width residual stream, whereas the latents it is integrated into are
dominated by the (identical) starting noise. The per-component PCCs the 18
graduated bodies earned during bring-up are 0.99991 .. 1.0 at TP=8, so what is
left here is that accumulation and nothing else.

Wall clock on the same box: 208 s to build the full pipeline (9.08 B parameters
staged onto the mesh), 0.55 s per full-depth forward at `S_joint=320` once
kernels are warm, 0.19 s for a 4+3-block build. The whole e2e gate — three test
files, one mesh, two builds, both goldens and a real trace capture — is 441 s.

## How it is built

```
tt/pipeline.py    THE single chained forward + PIPELINE_STAGES + the denoise_* trace
                  contract + the selftest hooks. Imported by BOTH demo/ and tests/e2e/.
tt/stubs.py       loads the 18 graduated stubs by name, records provenance,
                  proves live == frozen snapshot (Gate 1). Host-only.
tt/inputs.py      the seeded Source-A-recipe input builder, the _captured/ readers,
                  the sigma schedule, latent packing/unpacking. Host-only.
tt/reference.py   the HF golden. The ONLY place HF is called to compute. Host-only.
demo/             one runnable entrypoint per Call, plus their shared plumbing.
tests/e2e/        the gates.
```

Every layer of the model is owned by one of the **18 graduated stubs** in
`models/tt_dit/pipelines/flux_2_klein_9b_transformer/_stubs/`, composed **as-is**
— never edited, never re-implemented here. `tt/stubs.py` proves that by comparing
each live body byte-for-byte against its own frozen `.last_good_native` /
`.last_good_sharded` snapshot *and* against the sha256 `e2e_plan.json` certified,
and by recording which stub module built every routed object.

What `tt/pipeline.py` contributes is the chaining, in
`Flux2Transformer2DModel.forward`'s exact order:

```
timestep --> timesteps --> timestep_embedding --> 3x flux2_modulation
         \-> flux2_timestep_guidance_embeddings ------------------\
img_ids, txt_ids --> flux2_pos_embed (x2) --> concat text-first --> RoPE tables
latents      --> patch_embed ---\                                   |
prompt embeds--> patch_embed ---+--> dual blocks 0,1 (assembled by hand out of
                                     layer / flux2_attention / flux2_feed_forward /
                                     mlp / flux2_swi_g_l_u)
                                 --> dual block 2      (flux2_transformer_block)
                                 --> dual blocks 3..7  (encoder_stack, one call)
                                 --> concat [txt, img]
                                 --> single blocks 0,1 (layer + the two
                                     parallel-attention stubs)
                                 --> single blocks 2..23 (flux2_single_transformer_block)
                                 --> drop the text tokens
                                 --> ada_layer_norm_continuous(temb) --> decoder_head
                                                                        = the output
```

Four blocks are assembled explicitly from fine-grained stubs so that each of the
18 owns a real, load-bearing layer rather than being called to tick a counter:
dual block 1's feed-forward, for instance, is built around the standalone
`flux2_swi_g_l_u` gate (column-parallel `linear_in` → `all_gather` → the gate →
`mesh_partition` → row-parallel `linear_out` + `all_reduce`), which is that
stub's own documented replicated-in / replicated-out placement on a real layer.
Removing any routed stub changes the final output, because every one of them sits
on the residual path to `proj_out`. `e2e_plan.json::routing.table` is the full
map; the exact per-step call counts are asserted in the tests.

### TP=8 layout

The residual stream is **full-width and replicated** on every chip. Each
sub-layer is Megatron column-then-row internally and closes with its own
collective — `all_gather` after a widening projection whose consumer needs every
feature, `all_reduce` after a projection that reduces back to the model dim — so
the blocks compose with no extra collective in the wiring. Norms (all
`elementwise_affine=False` here), QK-norm gammas over `head_dim`, the RoPE tables
and the modulation shift/scale/gate vectors are elementwise or per-head and stay
replicated.

The model's *fused* projections are the wrinkle: `Flux2FeedForward.linear_in`
emits SwiGLU's two halves and `to_qkv_mlp_proj` emits q, k, v and the MLP's two
halves from one weight, so a contiguous shard would pair features that live on
different chips. `_flux2_ttnn.py::_regroup` reorders the columns at load time so
a plain `ShardTensorToMesh` hands each chip a matching slice of every group. Same
arithmetic, partitioned differently.

Two places deliberately run in **float32** rather than bfloat16, both measured
during bring-up: the sinusoidal timestep features (the model scales the timestep
by 1000, so the phase reaches ~1000 rad, where `ttnn.cos` of a bfloat16 argument
is off by up to 1.6 absolute) and the RoPE phase `ids * inv_freq`.

### Depth knobs

`build_pipeline(device, model=None, layers=None, denoise_layers=..., dual_layers=...,
single_layers=..., height=..., width=..., txt_len=...)` constructs and returns the
resident pipeline; it never runs the model.

Precedence: **per-stack > stage > `layers` > full depth**. The minimums are
`dual >= 4` and `single >= 3` — below those a graduated aggregate would hold
*zero* layers, i.e. be structurally absent rather than merely shallower, so the
build clamps up and says so. `layers=0` is not a zero-layer model and clamps the
same way.

Everything **outside** the two repeated stacks (both embedders, `pos_embed`, the
timestep stack, the three modulations, `norm_out`, `proj_out`) is always built, so
a capped build still runs every distinct op the full model runs — which is the
point, because profiling is per-op, not per-layer.

### Trace

`PIPELINE_STAGES == ["denoise"]`. There is one recurring graph — the joint
text-then-image forward — and only the timestep scalar and the latents change
between steps, so there is no prefill/decode split to model and no KV cache.

`denoise_trace_setup` pins the joint sequence to a fixed capacity
`C = S_txt + S_img` and pre-uploads, outside the capture, the latents, the prompt
embeddings, a persistent 1-element float32 timestep buffer, and the RoPE cos/sin
(taken from the HF reference on the capacity ids and concatenated text-first
exactly as `forward` does). `denoise_trace_step` is then a host-op-free forward
that reads only those buffers, and `denoise_trace_set_timestep` rewrites the
timestep **on device** (`ttnn.mul` on a persistent 1.0 buffer, then `ttnn.copy`
into the traced buffer) so one capture serves every Euler step.

**Padding is honest, not silent.** Flux2's joint attention takes no attention
mask — diffusers' own `Flux2Pipeline` passes no prompt mask, and the graduated
attention body calls `scaled_dot_product_attention(is_causal=False)` with no mask
argument — so a padded position *would* participate in the softmax. Masking it
would mean editing a graduated body, which Gate 1 forbids. The contract is
therefore "pin `C` to the deployment length": `denoise_trace_inputs()` returns
inputs at exactly `C`, so nothing is padded and the traced output is
bit-identical to the untraced one. A shorter input is padded and the fallback is
**printed**.

`trace_capture_selftest` / `host_op_selftest` exist twice on purpose: as methods
on the pipeline (what the tests call, with the device they already own) and as
module-level zero-argument functions (what
`scripts/tt_hw_planner/_trace_capture_probe.py` and `_host_op_probe.py` call).
The module-level pair runs the work in a child process whose `__main__` owns the
mesh, because `tt/pipeline.py` must never open a device on an import path — it
runs on the `device` passed into `build_pipeline`, and the test fixture is the
sole opener.
