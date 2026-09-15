# Captured model Q/K/V evaluation contract

[captured_inputs.py](captured_inputs.py) validates an externally supplied local artifact. [captured_fullchip.py](captured_fullchip.py) evaluates it in the existing private full-chip harness without modifying kernels, frozen variants, chunk sizes or input buffering. No model weights are downloaded and no model code is imported or executed from the artifact.

This is an operator comparison on particular captured activations, **not evidence of model-level quality, downstream evaluation scores or production dispatch performance**. A synthetic capture is useful for integration tests but is never described as a model capture. Capture provenance is declared by the producer, not independently authenticated by this loader.

## Format and supported semantics

A `torch.save` artifact must contain exactly:

```python
{
    "schema": "sdpa-captured-inputs-v1",
    "q": q, "k": k, "v": v,
    "metadata": {
        "causal": False,
        "mask": None,
        "scale": 1 / math.sqrt(128),
        "provenance": {
            "source_kind": "model",  # or "synthetic", explicitly
            "model_id": "producer-specified model/revision",
            "layer_id": "producer-specified layer",
            "capture_stage": "operator-boundary Q/K/V after applicable RoPE/transforms",
            # Optional JSON-primitive provenance: dataset/sample IDs, tokenizer,
            # capture script revision, batch/head selection, transformations, etc.
        },
    },
}
```

Q/K/V must be plain, detached, contiguous, dense BF16 tensors with identical `[1,H,N,128]` shapes, `H>0`, and positive `N` divisible by 512. Every input value must be finite. The loader rejects non-BF16, noncontiguous, tensor-subclass, unequal-length/head-count, or unsupported-layout input; it does not cast, transpose, pad, truncate, expand GQA, or modify values. Loading maps storage to CPU explicitly.

The operation is square, unmasked, noncausal `softmax(Q K^T / sqrt(128)) V`, with no attention bias, padding mask, window, dropout, custom scale or folded scaling. A capture from a causal/masked operator cannot truthfully declare those semantics absent to obtain a model-operator qualification. It could support a separately labeled synthetic unmasked ablation, but does not reproduce that original operator. Unequal Q/KV head counts, decode and cross-attention are outside this version. Capture BF16 inputs at the actual SDPA boundary, after transformations the model really applies. If upstream capture preparation changes the original dtype/layout or selects heads/examples, document that explicitly; this helper performs none of those transformations.

Metadata permits finite JSON primitives, lists and string-key dictionaries only, with bounded nesting and serialized length. Unknown top-level fields or semantic metadata fields are rejected; additional descriptive fields belong inside `provenance`. Do not store prompts, credentials or sensitive examples unless the storage and intended sharing are appropriate.

## Safe loading and provenance

The loader uses only `torch.load(weights_only=True, map_location="cpu")`, with **no unsafe fallback**. It temporarily clears PyTorch's user-extendable safe-global allowlist, restoring it in `finally`; some builds populate this list at import. Run this helper in an isolated evaluation process, not concurrently with other deserializers or safe-global mutations. Unsupported/custom objects remain rejected even if another component had allowlisted their class. This is not an OS sandbox or a guarantee against malicious resource-exhaustion artifacts: use captures from known sources and an updated supported PyTorch build. The default 4 GiB limit bounds serialized file size, not malicious decoded allocation size.

Recorded hashes include the unchanged serialized artifact, canonical metadata, and each tensor's logical C-order BF16 bit patterns encoded as little-endian uint16. Signed zeros therefore have different tensor hashes. Equal values in two separately saved artifacts can have identical tensor hashes and different file hashes. The driver verifies original CPU tensor hashes after each variant. The loader hashes the same open file before and after deserialization, checks its device/inode/size/mtime against both the opened descriptor and current path, and records the opened file's byte count. Atomic path replacement during loading is rejected. It neither edits the capture nor downloads external content referenced by metadata.

## Host validation and reference

```sh
python -B experiments/sdpa-l2/bfp4-lofi-v2/captured_inputs.py /path/to/capture.pt --sample-rows 128
python -B experiments/sdpa-l2/bfp4-lofi-v2/test_captured_inputs.py
```

The first command validates and prints JSON; it does not import TTNN. Query rows are deterministic integers: for `S=min(requested,N)>1`, row `i` is `floor(i*(N-1)/(S-1))`; one requested row selects zero. The exact row list is recorded, shared across variants and includes endpoints when possible. Use `--sample-rows N` for all-output reference coverage on sufficiently small captures. Deterministic sampling is not a probabilistic guarantee of catching a rare failing query.

The FP64 reference uses the **original captured BF16 values**, every head and every K/V row. Stable online softmax processes bounded key blocks and small query batches; there is no quantized-input replacement reference. Reported accuracy includes relative L2 in percent, PCC, gain (reported only, never fitted away), maximum absolute and exact elementwise-relative error, RMS error, and the BF16-output rounding floor. Exact zero reference elements with nonzero error make maximum relative error unbounded (`null` plus count/flag), not epsilon-clamped. A zero-norm output reference makes L2 undefined. Centered error subtracts the same FP64 mean of original V from both outputs, with no gain alignment; constant-V/zero residual references are explicitly undefined. PCC is undefined when its centered norm product is zero.

## Device evaluation

From the built Blackhole repository environment:

```sh
python experiments/sdpa-l2/bfp4-lofi-v2/captured_fullchip.py /path/to/capture.pt \
  --output /path/to/fresh-capture-eval.jsonl --cores 22 \
  --variants main fast balanced accurate lofi_fp32_b8 lofi_fp32_b4 \
  --sample-rows 128 --iters 5
```

Use a core count divisible by H and within the actual device grid. The driver caps it at the available query jobs, without changing Q256/K512 or the existing BF16 two-slot / FP32 one-slot K/V buffering. `--iters 0` skips timing but still checks finite outputs, sampled accuracy and exact trace replay. An output file is exclusively created: an existing file is never overwritten. Incomplete JSONL without a `complete` footer is not a completed qualification. There is no fallback to a different attention algorithm if construction or qualification fails.

The first four names select the established numerical choices in the **private common-reader harness**, not byte-for-byte replay of their frozen original drivers. `fast` explicitly enables the private Q256 correction reset. Balanced is FP32 QK4/PV2; accurate is the existing full-precision treatment. The two additional LoFi FP32 choices use the qualified native-exp path, Q7 preparation and B8 or B4 K/V preparation. No centering, Q rescaling, identity4, adaptive exponent search or new LUT option is silently enabled. Original inputs are uploaded, and applicable quantization is executed on device. Exact preprocessing checks default on; `--no-check-preprocess` is an explicit reported relaxation. CPU oracles are validation only, never substituted as device-preprocessed inputs.

Every output is checked finite and hashed; accuracy is only for the recorded query rows. Combined trace replay must be bitwise identical. Optional `--max-l2` and `--min-pcc` gates reject before timing; no universal threshold is chosen implicitly. Sources are pinned before device evaluation and compared after each variant and at completion. The source list captures the important local dependencies, not a full compiler/firmware closure.

Timing uses blocking trace-replay wall time with warmup and medians, separately for attention, real device preprocessing, and their captured combination. Initial input transfer, CPU validation/reference, compilation and artifact loading are excluded. Combined time is measured, not obtained by adding independent medians. Useful FLOPs are `4*H*N*N*128`; extra quantization, compensation, softmax or preprocessing operations are not counted as useful attention work. TFLOPs divide this count by measured time. This driver does **not** claim measured FPU utilization, infer an ISA-specific peak, or substitute resident/no-data-movement throughput for end-to-end chip throughput.

## Verification status

The 13 companion CPU tests use synthetic inputs only and cover load/hash preservation, rejection cases, restricted deserialization, allowlist restoration, atomic path replacement rejection, deterministic sampling, FP64 reference versus dense softmax, zero/constant-reference metrics, all six host configurations, and source-pin resolution.

The completed [synthetic interface smoke v2](captured-interface-smoke-v2.jsonl) used [the explicitly synthetic fixture script](capture_smoke_fixture.py): seed 1240, N1024/H2/D128, four cores, square noncausal attention, **every query row** against original-input FP64 attention. All six variants returned finite outputs with unchanged original input hashes and bitwise-identical combined trace replay. Applicable LoFi device preprocessing passed its exact oracle checks; the four BF16-input controls do not preprocess and correctly record that gate as not applicable. No timing was collected (`iters=0`). V2's six complete-output hashes and metric objects are identical to [historical v1](captured-interface-smoke-v1.jsonl). After source formatting, both are historical records; the fresh [final captured smoke](final-captured-v1.jsonl) now supplies current-source qualification and exactly matches all six v2 complete-output hashes. See the [strict final audit](final-smoke-audit-v1.json) and [qualification scope](FINAL_QUALIFICATION.md).

| Private-harness choice | L2 (%) | PCC |
|---|---:|---:|
| Main BF16 | 2.450737 | 0.99970123 |
| FAST BF16, private correction reset | 2.445195 | 0.99970238 |
| Balanced FP32 QK4/PV2 | 0.372492 | 0.99999309 |
| Accurate FP32 | 0.178175 | 0.99999841 |
| LoFi FP32 B8 K/V, native exp | 2.836418 | 0.99960075 |
| LoFi FP32 B4 K/V, native exp | 16.469779 | 0.98642441 |

This qualifies the tested **synthetic interface path**, not real model activations, an accuracy band over distributions, performance, or model quality. No real model activation data was supplied.

Historical v1 provenance has a disclosed gap: its manifest omitted the four frozen MAIN/FAST compute headers and their two separately selected SFPU headers. The captured driver now pins these and additional directly selected SDPA/dataflow helpers. V1 is preserved unchanged and is not retrospectively assigned new hashes. V2 repaired the manifest in a fresh pre-format run, but its source hashes are also historical after formatting. The checkpoint validator preserves those historical warnings; the separate strict final validator requires current runtime pins for `final-captured-v1.jsonl`, with no historical-source fallback. The original synthetic fixture's generator metadata remains historical and is distinguished from the current runtime sources.
