# MOSS-Image implementation design

This is model bring-up, with accuracy taking precedence over performance. It
supports two checkpoint weight sets through one architecture implementation.
The 5.082B-parameter DiT has 42 main blocks, width 3072, 24 attention heads of
128 dimensions and FFN width 8192. Two text refinement blocks precede the main
image-first joint sequence. The prompt encoder is Qwen3.5-9B's language model,
with 24 Gated DeltaNet layers and eight full-attention layers.

## Device ownership and placement

Model classes accept an explicit MeshDevice. They do not open devices. The demo
CLI owns its mesh; pytest calls the same generation function using fixture-owned
meshes. The prompt encoder uses one chip, then releases its layer/model tensors
before DiT weights are resident. Two-card DiT shards QKV heads and FFN channels,
and reduces row-parallel projection results before replicated residual additions.
Biases are applied after reductions where needed, avoiding double-counting.
The VAE replicates its weights and output on a two-chip mesh; it is not a
channel-parallel decoder. Generation saves the first replica for the image.

On a two-chip caller-owned mesh, encoding uses a one-chip child view. Before
denoising, generation quiesces the parent hierarchy to hand shared physical
command queues and event counters back from the child to the parent. Deleting
the local child variable is insufficient because the Python binding retains it.
The caller's parent mesh remains open. This follows the upstream distributed
child/parent workload pattern and is exercised by the two-card integrated test.

Module boundary transfers serialize TT-produced conditioning, with no captured
CUDA input. This path is CUDA independent; it is not a claim of eliminating host
participation. Host code handles tokenization, initial noise, position/time
constants, schedule construction and output serialization.

## Numerical choices

DiT matrix operands use BF16 with HiFi4 and FP32 destination accumulation.
RMS normalization explicitly uses the official cast/scale semantics. The FFN
uses MOSS's bounded `situ` function rather than substituting ordinary SwiGLU.
The attention sequence is image first followed by the actual text tokens; padded
keys are masked and the split-half rotary layout matches the official model.
Classifier-free guidance and Euler latent updates use FP32.

BF16 pointwise arithmetic executes one operation in FP32 and explicitly casts
back to BF16 at each PyTorch eager boundary. This preserves ties-to-even rounding:
the qualified runtime's direct BF16 addition instead rounds halfway values away
from zero. It also avoids prematurely quantizing constants such as 1/25. Split-half
RoPE uses two separately rounded BF16 products followed by a separately rounded
sum, rather than the native fused rotary operation. Biased linear projections
retain FP32 matmul output through bias addition and round once afterwards.
Attention uses the model-local `AccurateAttention` helper in both the DiT and
prompt encoder. It retains score, scaling, additive mask and stable-softmax
arithmetic in FP32, rounds probabilities to BF16, and accumulates PV into FP32
before the reference BF16 output boundary. Groups of at most six query heads
bound temporary score memory. Encoder GQA groups stay within each KV head and
repeat that KV only within its query subgroup; causal masks contain position
metadata. DiT masks retain padded-key exclusion on each TP device. These helpers
do not install a process-wide TTNN monkeypatch or move activations to the host.

The preceding native-SDPA implementation selected 128-token query/key chunks in
matched-input tests. Its remaining drift motivated this explicit attention path.
The earlier manual-attention experiments were slower, so the replacement favors
accuracy and must retain separate runtime evidence. Neither isolated component
improvement nor precision settings establish final independent image parity.

The prompt encoder retains full normalized token output and the official thinking
chat template. Its GDN path uses FP32 L2 normalization, decay, explicit SFPU recurrence and
residual state arithmetic where required. Zero-centered RMS weights are offset
in FP32. Causal convolution and the gated normalization preserve reference
rounding order. These details were needed for all-token component accuracy and
are implemented in the model-local GDN recurrence.

The VAE reverses the packed latent batch normalization before unpacking 2×2
patches. It uses the checkpoint's decoder widths [96, 192, 384, 384], not the
encoder widths. Convolution operands are BF16 with HiFi4 and FP32 accumulation;
normalization, residuals and nonlinear operations retain FP32 activations.
Prepared convolution cache keys include shape, dtype, layout and memory config.
No fitted gain correction or output waveform/image rescaling is used to improve
reported error metrics; the final clamp and pixel conversion are model output
processing.

## Validation boundary

CUDA captures execute the pinned official pipeline, with tensor-only artifacts
and a manifest that hashes consumed weights, configurations and artifacts.
Runtime selects only canonical/index-declared safetensors and checks every
consumed component tensor against its qualified shape schema. The schema was
extracted from the pinned Base checkpoint and is shared by the identical SFT
architecture. Replay checks the selected files against the capture manifest.
All Torch artifact loads use `weights_only=True`.

Component tests report PCC, relative RMS error and maximum absolute error.
The detailed accuracy review also reports RMSE normalized by the reference's
standard deviation, explicitly distinguishing it from relative RMS normalized
by the reference RMS. The attention tests include an independent FP64 arithmetic
reference, the BF16 probability-rounding policy, causal/GQA grouping and padded
key invariance. Two-card attention replica checks do not replace full DiT TP
collective or independent-generation qualification.
Independent integration uses the same seed and CPU RNG algorithm on both
backends, ensuring equal initial noise without injecting a saved activation.
It compares conditioning, initial predictions, every updated latent and final
pixels after completion. Final image error must be reported even when component
checks pass; diffusion trajectories can amplify systematic numeric differences.
Results from an older build are not evidence that a newly rebased native runtime
passes. The results record distinguishes historical and current-source runs.

The contribution includes no HTTP monitor, SSH launcher, remote locking utility,
background experiment scheduler or nested environment/lock file. Tests use the
repository device fixtures directly; no scheduled model CI job is registered.
