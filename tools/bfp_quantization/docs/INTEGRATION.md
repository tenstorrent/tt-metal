# Integration details

## Layout contract

`search_linear` and `gptq_search` accept CPU/GPU PyTorch weights in `[N,K]` and
return a new CPU float32 tensor in `[N,K]`; they never mutate the supplied weights.
The quantization assumes TT's normal 32×32 tile with 16×16 faces and exponents
shared across 16 contiguous entries in the packed tensor's last dimension.
For Linear weights packed as `[K,N]`, each group spans 16 output channels.

`output_splits` is a list of physical output-shard widths, not a list of input
widths or shard indices. It must sum to N. Groups restart at every shard boundary;
partial groups are padded with zeros independently. For N=80 split over two
devices, use `[40,40]`; quantizing all 80 outputs as one row is a different grouping.
Input sharding alone does not change this grouping; a full-input Hessian still
allows GPTQ to compensate across the original input channels before distribution.

The API does not infer tensor-parallel topology. A concatenation, head permutation,
fused projection, custom tile, or later reshard can change groups. For weight-only
search, call `search_packed` on each 2D shard in its final logical layout immediately
before the existing native conversion. For GPTQ, provide the weight in `[N,K]`
for that final output ordering and a Hessian matching its exact input ordering.
Do not transpose the Hessian just because the weight is transposed for TT storage.

The library does not install model hooks at inference, alter the model's dtype
selection, build device kernels, or implement a model loader. The
[checkpoint exporter](CHECKPOINTS.md) produces files for an existing HF loader.
Keep existing
attention/GDN, normalization, convolution, bias and activation behavior unchanged
unless those parameters are explicitly included in your own experiment.

## Capture activations in any implementation

```python
from tt_bfp_quant import HessianAccumulator, gptq_search

acc = HessianAccumulator(in_features, device="cpu", chunk_rows=2048)
# Inside your temporary calibration adapter, before the target projection:
acc.add(x_cpu_or_gpu, mask=valid_tokens)  # x [...,K], boolean mask [...]
# Repeat on representative calibration prefixes, then remove the adapter.
H = acc.value()
Q, metadata = gptq_search(W, H)
```

For TTNN inputs, reconstruct the true logical X: concatenate input-channel shards,
remove duplicate replicas, and remove tile/sequence padding. Copying all replicas
and treating them as distinct channels produces the wrong Hessian. Instrumentation
should observe inputs without changing outputs; compare logits with and without
the capture adapter on a small control input. Disable traces temporarily if your
capture point is otherwise bypassed by a captured execution.

The convenience `capture_linear_inputs` hook observes every row supplied to a
named `torch.nn.Linear`, including padding if present. It supports positional and
keyword `input`, removes hooks on exceptions, and does not change training/eval
mode. Use unpadded batches or the explicit accumulator for padding and custom
modules. A `max_samples` cap retains the first observed rows, so shuffle/select
representative prefixes before using a cap. MoE layers need routed expert inputs.

Calibration is a forward-only pass on representative training data. The original
Qwen experiment used 65,536 calibration tokens and evaluated a separate 297,192
WikiText-2 test targets. Those sizes are provenance, not universal requirements.
Capture from the same activation arithmetic used in deployment when possible;
PyTorch calibration is a convenient alternative to evaluate, not numerically
identical to TT calibration.

## Efficiency and validation

The native backend parallelizes independent output groups using OpenMP when
available. The NumPy backend is a readable oracle and portable fallback. Dense
matrix multiplication and Cholesky use PyTorch's CPU libraries; set
`torch.set_num_threads(...)` separately from the API's native-loop `threads`.
No tensor-valued gradients are stored. Hessians and their factors are reusable.
Quantization is once per weight variant, not once per inference request.

`validate_repacking(..., native=True)` invokes TTNN's installed host packer.
No device is opened; some TTNN builds still need an architecture descriptor even
for host packing. Configure that normally for your build rather than installing
a different TTNN wheel. The check compares numerical values (signed zero is not
distinguished). Run it once per exported tensor or representative packing case;
it need not be repeated every model load if the tensor and packing recipe are
unchanged and verified. Clear or namespace TT's weight caches by variant.

Returned values normally fit exactly in BF16. `to_bf16_exact` verifies this before
storage. A BF16 carrier still occupies two bytes per weight; native BFP compression
happens at your existing TTNN conversion. Loading those values into a PyTorch
Linear is useful for offline reconstruction checks but does not provide native
BFP storage, TT activation arithmetic, or a TT performance measurement.
