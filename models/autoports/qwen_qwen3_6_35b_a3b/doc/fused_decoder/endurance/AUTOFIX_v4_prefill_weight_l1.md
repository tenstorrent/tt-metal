# Diagnostic: transient shared L1 weights, sequential PF projection blocks

Base `v4_combined_adapters.py.snapshot` is SHA256 `82c8c62950b85fdea309f2eed91fbc0f7b9691cb5da78fa97b0728154e8be1cb`. Candidate `v4_prefill_weight_l1.py.snapshot` is SHA256 `8ef2301f4bb475b4ea8dfee050091191ea1e22984c5ddf4b1db4bff961bc609b`. Only the multirow `linear(..., "attention_packed")` branch changes; setup and singleton K512/K1024 are AST-unchanged. No production/native/core-harness files changed. No hardware was executed by the preparer.

The serial PF baseline rereads the24.25MiB compact BF16 weight matrix from DRAM for every token. This candidate tests reuse of that exact matrix in transient L1 while reducing each of the two K1024 blocks separately. Each block still accumulates1024 products sequentially; the final FP32 sum remains partial0 + partial1. RM input placement, compact pair gather, RM output retention, final concat/tilize, public prefill boundaries, and recurrent cache rounding are preserved.

## Adapter and source legality

Inside each measured PF call, two `ttnn.slice` calls copy the tile-aligned K ranges `[0,1024)` and `[1024,2048)` from the persistent DRAM BF16 matrix directly into separate interleaved L1 tensors. Explicit `memory_config=L1` is accepted by the slice wrapper/factory and selects the destination buffer. All starts/ends are tile-aligned, width6208 is tile-aligned, and strides are one, so no row-major conversion is necessary. This avoids allocating a full extra L1 matrix plus sliced duplicates. `slice_nanobind.cpp:125–137` exposes the sequence wrapper; its omitted steps become ones. `slice.cpp:139–146` carries explicit output memory configuration, and `slice_device_operation.cpp:354–410` uses it in the output TensorSpec.

For each token the adapter computes the same reordered BF16 coefficients. It multiplies one staged BF16 weight half by its coefficient half into FP32 L1, runs the existing uncompensated FP32 height cumsum in L1, selects its final row via the aligned terminal32-row band, and releases the product/prefix before proceeding to the other block. Both final partials remain FP32.

`ttnn.add(partial0, partial1, memory_config=L1)` then replaces the two-input FastReduce. The native FP32 FastReduce kernel copies operand0 and performs exactly one SFPU add with operand1 (`fast_reduce_nc/device/kernels/reduce_nc.cpp:42–59`). BinaryNG ADD selects SFPU for two equal FP32 operands regardless of the Python default fast-mode flag (`binary_ng_device_operation.cpp:49–58`); its program factory unpacks both operands directly to FP32 destination (`binary_ng_program_factory.cpp:1370–1381`). No fused activation or BF16 boundary is introduced. Explicit `fast_and_approximate_mode=False` must **not** be passed for this FP32 ADD: its validation allows that explicit flag only for BF16 output. The candidate intentionally leaves it unset.

The sum is cast once to the requested final output dtype, matching the original boundary, and converted to compact DRAM RM before retention. Both staged weights are explicitly released after the row loop and before final concat/tilize; no L1 weight attribute survives on the decoder. All staging, two scans, terminal slices, addition, conversions, concat, and final tilize remain in the method and therefore in complete eager/traced timing.

## Capacity bound and cost tradeoff

Two staged BF16 weights total25,427,968 bytes. One block's FP32 product and prefix each occupy25,427,968 bytes, so their simultaneous pair occupies50,855,936 bytes. The three main live allocations total76,283,904 bytes (72.75MiB). One earlier completed partial adds794,624 padded bytes while computing block1. The terminal32-row band is also794,624 bytes, after the product has been released. Coefficient buffers are small and remain DRAM.

Using the previously observed110 interleaved banks, each weight half has6208 BF16 tiles, rounded to57 tiles per bank. Each FP32 product/prefix also has6208 tiles. The main per-bank reservation is684KiB, plus roughly8KiB for the retained first partial and32KiB for native cumsum buffers: about724KiB before binary buffers, other live tensors, alignment, reserved regions and fragmentation. This is a nominal capacity argument, not a device allocation guarantee or a claim about other meshes.

The rewrite exchanges two weight copies per PF call for repeated DRAM matrix reads per token, but doubles scan/multiply launches and halves independent block parallelism. Native cumsum still allocates/writes every prefix; there is no last-only or in-place claim. Small sequences may lose on staging/dispatch, and L1 bandwidth contention may erase the expected benefit. The earlier losing DRAM cumsum controls and passing serial L1 control remain intact; this is a weight-reuse experiment against the newer combined RM adapter, not a relabelled old control.

## Discriminator

`tests/v4_prefill_weight_l1_probe.py` invokes the actual baseline/candidate `linear` methods. It uses real checkpoint K/V/A/B weights and explicitly labelled synthetic normalized BF16 inputs, checks every6208 FP32/BF16 output for S2/32/33/128/129 plus singleton K512/K1024, and skips timing on mismatch. It compares complete synchronized eager execution and warmed traced execution, including transient weight movement on every call/replay. First/last replay outputs are checked. Physicalchip0, trace32MiB, L1-small32768, clean timing-environment checks, source/weight/input hashes, trace release and `finally` device close are explicit.

`v4_prefill_weight_l1_cpu.json` passes selected-column FP32 partial/final order checks on33 rows. This verifies the mapping and two-block arithmetic on those fixtures, not all possible signed-zero patterns or device operations. The existing cumsum positive-zero start convention remains subject to the strict final output checks. Source-scope validation finds only `linear` changed and its singleton branch identical. Snapshot/probe/CPU hashes are bound in the provenance sidecar.

```sh
python_env/bin/python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.v4_prefill_weight_l1_probe --output models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v4_prefill_weight_l1_probe.json
```

Device allocation, exactness and speed remain pending. A passing adapter requires the parent's complete decoder benchmark and acceptance queue before selection.

## Parent device result: exact, rejected for complete-adapter latency

The parent completed `v4_prefill_weight_l1_probe.json`, SHA256 `2e411bbf7c314efb5453bd4e7a0bed282a8e4c4fa2bc2b07110ffd793b9c9ced`, with `status=exact`, all seven cases exact at FP32/BF16 boundaries, and clean device close. Candidate/probe hashes match the frozen preparation files. The provenance sidecar binds this parent-owned report; no hardware was executed by the preparer.

| Rows | Baseline traced ms | Candidate traced ms | Baseline eager ms | Candidate eager ms |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 30.579791 | 33.885947 | 31.137396 | 34.089250 |
| 129 | 30.839801 | 34.166113 | 31.427881 | 34.379805 |

S2 also regressed from0.499750 to0.623556ms traced. At S128, traced latency increased approximately10.8%, despite exact outputs and successful L1 allocation. Every transient weight copy, both sequential scans, final add, RM conversion and concat is included; the result cannot be improved by accounting away setup or host dispatch. This **measured split-K-block transient L1 weight reuse candidate is rejected**.

The experiment establishes that the complete rewrite costs more than the baseline on the measured workload. Its extra launches and reduced independent block parallelism are plausible contributors, but this report does not claim a kernel-level causal attribution without a profile. No further layout-only variant is proposed from these results. The frozen snapshot, probe and CPU history remain intact; selection metadata alone records the earned rejection.
