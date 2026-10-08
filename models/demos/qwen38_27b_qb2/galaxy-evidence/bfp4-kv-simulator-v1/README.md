# BFP4 KV accuracy with CPU ttsim

The simulator completed **24 paged-attention cases**: three K/V formats,
two causal bounds, and four contexts (1,024-token smoke plus 32,768, 131,072,
and 262,016 tokens). All outputs were finite and all cases passed the existing
kernel gate against a quantized-input reference. The virtual device closed
cleanly. **BFP4 quantization produced substantially larger attention-output
error than BFP8 on these synthetic inputs. This is not model or GPQA qualification.**

The experiment did not change model precision, checkpoint weights, native
runtime sources or installation, or physical-device jobs. Simulator elapsed
time is not a hardware latency or bandwidth measurement.

## Results

These ranges cover the **18 requested long-context cases**, excluding smoke.
Relative RMS is `norm(actual - reference) / norm(reference)`. The columns use
different references and are not additive.

| K/V format | Quantization only | Execution on quantized K/V | Total versus original BF16 inputs |
|---|---:|---:|---:|
| K8/V8 | 1.005–1.073% | 0.812–1.484% | 1.281–1.861% |
| K4/V4 | 15.576–16.516% | 0.893–1.507% | 15.647–16.641% |
| K8/V4 | 11.073–11.159% | 0.830–1.532% | 11.116–11.288% |

Total RMS at each full-length causal bound:

| Active tokens | K8/V8 | K4/V4 | K8/V4 |
|---|---:|---:|---:|
| 32,768 | 1.313% | 15.661% | 11.116% |
| 131,072 | 1.595% | 16.455% | 11.198% |
| 262,016 | 1.861% | 16.637% | 11.285% |

The second causal bound is 37 tokens shorter in each context. Its results are
included in the ranges and [raw receipt](receipts/long-v2.json). The smallest
execution PCC over long cases was **0.9998881906**. The unchanged kernel gate
requires **at most 2% per-user relative RMS and at least 0.999 per-user PCC**
against the CPU FP32 reference on device-roundtripped K/V. No acceptance
threshold was added for quantization or total error; no format was promoted.

Worst individual live-query-head total relative RMS over long cases was
**2.115% for K8/V8, 17.957% for K4/V4, and 12.066% for K8/V4**. The existing
gate is per user, not per individual head. The raw receipt reports every
head's RMS, PCC and maximum absolute error, plus overall finite status,
PCC, maximum absolute error and mean signed error for each comparison.

Relative to the corresponding BFP8 control, K4/V4 had **8.94–12.22 times**
the total error, and K8/V4 had **6.06–8.69 times** the total error. Direct
output disagreement with BFP8 is a separate metric: **15.697–16.692%** for
K4/V4 and **11.091–11.293%** for K8/V4. See the
[summary](receipts/summary.json) for all comparisons.

## Inputs and numerical references

- Use Qwen's local TP4 geometry: batch 1, six live BF16 query heads, one K/V
  head and head dimension 256. Append 26 zero query heads to expose a full
  logical 32-head tile, preserving the accurate-exp path for the live heads.
- Generate synthetic BF16 Gaussian Q/K/V using seed `20261007 + context`.
  These are not real model activations or captured KV. All three formats
  share the original inputs, query and physical-page mapping for each bound.
- Use 32-token pages with a shuffled physical mapping. Allocate
  `ceil((context + 127) / 512) * 512` cache tokens. Test active lengths
  `context` and `context - 37` with explicit position tensors. Set every
  future V row to sentinel 32; the shorter bound exercises a partial page.
- Keep HiFi4, FP32 destination accumulation, `math_approx_mode=False`,
  `packer_l1_acc=True`, `exp_approx_mode=False`, chunk 256 and
  `max_cores_per_head_batch=16` fixed. Q and output remain BF16. Use one eager
  invocation and readback per case, without trace capture or replay.
- Upload K/V through native TTNN conversion, then round-trip them to obtain
  the actual quantized operands. Independent K and V formats are accepted
  by the pinned native factory; K8/V4 was executed, not merely inferred.

The probe separates three calculations:

1. **Quantization only:** CPU FP32 causal attention on roundtripped K/V
   versus CPU FP32 causal attention on original BF16 inputs.
2. **Execution on quantized operands:** actual simulator output versus CPU
   FP32 causal attention on roundtripped K/V.
3. **Total:** actual simulator output versus CPU FP32 causal attention on
   original BF16 inputs.

CPU reference attention uses six live query heads and a single shared KV
head. It gathers causal pages without materializing six copies of the cache.
Error reductions use float64. Query, page-table, original/roundtripped KV,
reference and output hashes are recorded for reproducibility.

## Failed attempt and simulator compatibility

The initial [smoke](receipts/smoke-v1.json) completed six cases without a
compiler compatibility flag. Long v1 then aborted before its first 32K BFP8
output with `UnsupportedFunctionality: tensix_sfploadmacro: explicitly out of scope`.
Its [log](receipts/long-v1.log.gz), [partial receipt](receipts/long-v1.json),
[source](receipts/probe-v1.py.gz) and [launch](receipts/long-launch-v1.json)
are preserved. The process exited inside the simulator, so its partial
receipt still says `executing` and `cleanup_completed=false`; the authoritative
service result is `failed`, exit status 1. Before retry, its MainPID and
ControlPID were zero and its ControlGroup was empty.

V2 used **`TT_METAL_DISABLE_SFPLOADMACRO=1`** and a fresh separate JIT cache.
This native compiler option emits `-DDISABLE_SFPLOADMACRO` and chooses explicit
LLK arithmetic instructions instead of unsupported macro instructions.
**It does not disable simulator checks or skip arithmetic.** Unsupported
instruction, unpredictable-behavior and undefined-behavior checks remained
intact. Every format, including BFP8, used the same compiler option.

Preserved native source evidence explains the fallback:

| Source snapshot | Relevant lines and behavior |
|---|---|
| [JIT build](receipts/native-source-evidence/tt_metal/jit_build/build.cpp.gz) | 457–458: emit `-DDISABLE_SFPLOADMACRO` |
| [Runtime options](receipts/native-source-evidence/tt_metal/llrt/rtoptions.cpp.gz) | 910–914: define the runtime flag |
| [Native ttsim regression launcher](receipts/native-source-evidence/tt_metal/tt-llk/tests/run_ttsim_regression.sh.gz) | 260–261: document simulator compatibility and default the flag to 1 |
| [Blackhole binary max/min](receipts/native-source-evidence/tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_binary_max_min.h.gz) | 23–31: explicit LOAD/LOAD/SWAP/STORE fallback; 52/54: macro path |
| [Paged-decode compute](receipts/native-source-evidence/ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/compute/sdpa_flash_decode.cpp.gz) | 487–488: subsequent local chunks merge the previous maximum |
| [Shared attention compute](receipts/native-source-evidence/ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/compute_common.hpp.gz) | 290–294: that merge calls `binary_max_tile` |

The failure log names an instruction but gives no PC; this source evidence
identifies a relevant path, not a uniquely traced failing instruction address.
All six smoke output hashes were **bit-identical with and without the flag**.
That check does not establish long-context parity with the production
macro-enabled kernel. **Physical-hardware and production-instruction parity
remain unqualified.**

## Pins and source provenance

| Component | Pin |
|---|---|
| Native Metal revision | `a08819ddbe23077f8037d3802303939064868ff6` |
| Simulator release | ttsim `v1.11.2`, `libttsim_bh.so` |
| Simulator SHA256 | `d01be2a094f9f0f6a00e771defeae2311cb82f051b8af89c2df509e78b8432b7` |
| Exact executed v2 source SHA256 | `241176996e994a39b9ef0134f084843d2f207e855f7e6c5c2fe16e96854fe53f` |
| Published source SHA256 | `160f612722935206e88c3c10855d57d1994ec85f5f983baddf730dfb878797dc` |

[Exact executed v2 source](receipts/probe-v2.py.gz) is immutable evidence.
The [published probe](../../demo/probe_bfp4_kv_simulator.py) differs only in
Black formatting, isort import ordering and removal of unused top-level
`math` and reference-local `torch` imports, including an empty placeholder.
The verifier checks AST equality after precisely those import cleanups.
The formatted source was compiled and checked locally; no numerical experiment
was rerun after formatting. [Publication metadata](publication.json) records
both hashes and the hook/config pins: Black 23.10.1, autoflake 2.3.1, isort
5.13.2, 120-column Black profile.

Native library and relevant source hashes are in the raw receipts. Original
sources, logs and native snapshots use deterministic gzip with `mtime=0` so
repository formatters cannot rewrite the evidence. Collection hashes refer
to the **decompressed original bytes**.

## Verification and reproduction

From the repository root, verification uses only the Python standard library
and does not import TTNN, open a device or launch an experiment:

```bash
python3 models/demos/qwen38_27b_qb2/galaxy-evidence/bfp4-kv-simulator-v1/verify_receipts.py
```

Expected result: **17 collected hashes verified, six identical smoke outputs,
24 complete finite cases passing the unchanged kernel gate, and published
source provenance verified**. [Collection evidence](receipts/collection.json)
includes terminal service states and journal lines. Successful transient
units were garbage-collected (`LoadState=not-found`); the completed numerical
receipt and full log additionally show clean virtual-device shutdown at
**2026-10-08 05:28:56 UTC**. No experiment service retained a PID or cgroup.

[Long-v2 launch](receipts/long-launch-v2.json) contains the exact argument
array and environment. It caps the service at one CPU (`CPUQuota=100%`),
8 GiB and two hours, enables slow dispatch, disables inspector RPC and uses
an isolated JIT cache. The probe verifies simulator SHA and native revision,
then requires exactly one visible virtual chip before opening mesh 1×1.

The recorded host was `ttuser@10.228.203.98`; its native runtime was
`/home/ttuser/kimi-prefill.Ubx2wY/runtime/qwen38-27b-20261006`. The simulator
library remains under
`/home/ttuser/qwen38-artifacts-20261007/bfp4-simulator-v1/libttsim_bh.so`.
No checkpoint is required. Copy this evidence directory to that host and
run the following **there**, from the evidence directory, only when another
bounded CPU run is intended. It replays the exact archived v2 source and
recorded command with fresh artifact, cache, output and service names:

```python
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import subprocess

evidence = Path.cwd()
receipt = json.loads((evidence / "receipts/long-launch-v2.json").read_text())
old = Path("/home/ttuser/qwen38-artifacts-20261007/bfp4-kv-simulator-v1")
stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
new = old.parent / f"bfp4-kv-replay-{stamp}"
new.mkdir(exist_ok=False)
source = gzip.decompress((evidence / "receipts/probe-v2.py.gz").read_bytes())
assert hashlib.sha256(source).hexdigest() == receipt["probe_sha256"]
(new / "probe-v2.py").write_bytes(source)
command = [part.replace(str(old), str(new)) for part in receipt["command"]]
unit = f"qwen38-bfp4-kv-replay-{stamp}"
command = [f"--unit={unit}" if part.startswith("--unit=") else part for part in command]
(new / "replay-launch.json").write_text(json.dumps(command, indent=2) + "\n")
subprocess.run(command, check=True)
print(f"systemctl --user status {unit}.service")
print(f"Receipts and logs: {new}")
```

Do not use `run_safe_pytest.sh` for this reproduction. Retain direct Python,
the pinned binary and explicit environment in the recorded command; do not
disable simulator checks. Preserve failed attempts and verify authoritative
terminal service state before retrying. The probe refuses an existing output
receipt. Progress is atomically saved after each variant, including the
quantization reference before execution in v2.

## Limits

This is a single seeded synthetic sample per context, with two causal bounds
sharing that sample. There is no captured KV, RoPE-derived distribution,
full-layer execution, end-to-end model evaluation, perplexity/logit study or
GPQA run. The measured percentages are tensor errors, not task-score losses
or percentages of incorrect tokens. The BFP4 results do not establish a
model-quality acceptance threshold, a hardware speedup or a deployment policy.
