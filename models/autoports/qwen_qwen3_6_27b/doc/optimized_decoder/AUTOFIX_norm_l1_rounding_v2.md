# Norm L1 handoff V2: attribute and preserve conversion rounding

The first isolated test (`norm144_probe.json`) disproved bit equivalence of
native `typecast` and the existing sharded-to-interleaved conversion. For
input norm B1 at scale .001, 2,534/5,120 values differed, with maximum absolute
difference .015625. Zero-input rows matched. The next projection's buffer
alias check passed. This is a conversion-precision failure, not evidence that
the L1 handoff topology is illegal or slow. The first candidate remains
immutable, and the parent separately measures its full decoder PCC/latency
under the unchanged acceptance bar.

## Source distinction

The baseline `sharded_to_interleaved(..., output_dtype=BF16)` uses
`eltwise_copy_metal2.cpp`: copy tile into DST, then pack. Its factory explicitly
uses default ComputeHardwareConfig with `fp32_dest_acc_en=false` and default
source-register unpack. See
`ttnn/cpp/ttnn/operations/data_movement/sharded/sharded_to_interleaved/device/`
`sharded_to_interleaved_program_factory.cpp:193-223`.

By contrast, `ttnn.typecast` sets `preserve_fp32_precision=true` and
`fp32_dest_acc_en=true` for every FP32 input
(`ttnn/cpp/ttnn/operations/copy/typecast/typecast.cpp:33-51`), invoking a
different conversion path. Source inspection explains why equivalence cannot
be assumed, but does not alone establish whether the observed old values are
truncated or rounded. The diagnostic probe compares both with CPU arithmetic
from the actual device FP32 result.

## Native repair candidate

`ttnn.clone` exposes both output dtype and compute config. Its sharded factory
uses the owner grid, and its pure-copy kernel performs copy_tile/pack with no
SFPU typecast. V2 replaces only the proposed L1 `typecast` with:

```python
ttnn.clone(
    y, dtype=ttnn.bfloat16, memory_config=memory,
    compute_kernel_config=ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    ),
)
```

This configuration is intended to match baseline conversion while retaining
eight-core `[32,640]` BF16 L1 output. It is source-supported but **not yet device
validated**. The full FP32 RMSNorm, affine, epsilon, its compute configuration,
and residual residency are unchanged. A native same-memory-config `ttnn.assign`
copy is included as an independent diagnostic control; its work distribution
differs and it is not selected in the decoder candidate.

References:

- `ttnn/cpp/ttnn/operations/data_movement/clone/clone_nanobind.cpp:39-45`:
  supported dtype/memory/compute arguments.
- `.../clone/device/clone_program_factory.cpp:39-49,205-233`: sharded placement
  and supplied compute hardware configuration.
- `.../clone/device/kernels/compute_kernel.cpp`: pure library copy.
- `ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp:109-113`: pure copy is
  CopyTile followed by PackTile, distinct from the typecast convenience op.

## Artifacts and recommended checks

- Frozen decoder: `candidates/norm_l1_handoff_v2_on134.py`.
- Generator: `candidates/make_norm_l1_handoff_candidate_v2.py`.
- Probe: `tests/run_norm_l1_handoff_probe_v2.py`.
- Source evidence: `candidates/norm_l1_handoff_v2_source.json`.

Python syntax checks passed. No TTNN/device execution occurred during source
preparation. The new probe records its own SHA, candidate SHA and six relevant
C++ source SHAs. Its candidate argument is provenance only: the isolated probe
executes explicit baseline/typecast/clone/assign API paths.

For both real folded hidden-norm gamma tensors, B1/3/32 and four input scales,
the probe reads the **same device-produced FP32 norm result** and constructs
CPU BF16 round-to-nearest-even, BF16 truncation, and TF32-truncate-then-BF16-RNE
oracles. It records mismatch counts, maximum error and relative RMS for every
conversion path. Every mode also checks projection-buffer aliasing and memory
config. Recoverable API exceptions are recorded per mode and collection
continues; they are not used to reject the family. A custom decoder-local
conversion remains an option if supported native copy paths fail.

Run through the parent's serialized queue:

```bash
python_env/bin/python -m models.autoports.qwen_qwen3_6_27b.tests.run_norm_l1_handoff_probe_v2 \
  --layer 3 --batch 1 3 32 --require-clone-exact \
  --output models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/norm_conversion_v2.json
```

Require baseline bit equality for V2, then full-attention short/repeated/long
correctness and warmed traced whole-path latency. Keep only a measured win;
fewer transfers alone do not select the candidate. If clone matches baseline
truncation while typecast matches CPU RNE, report that factual precision
distinction rather than calling the more precise path numerically broken.
