# Measured tiny norm scan: independent and composed integration candidates

Component `v6_norm_terminal_scan_device.json` passed all19 complete norm fixtures, both synthetic batches (768 midpoint/adjacent FP32 values and five zero/subnormal/cancellation controls), exact intermediate FP32 lanes/total and complete BF16 output gates. All three complete-adapter timing pairs remain exact after100 trace replays each. Device closed11:37:17. The component source remains frozen at85d00325f34ab3ac803fbc9b4acb917b9bd6d3decacb4b3d15c0c52c8104408c; no helper/source pin was rewritten.

| Complete norm adapter | Baseline traced ms | Candidate traced ms | Baseline eager ms | Candidate eager ms |
|---|---:|---:|---:|---:|
| Actual227 input DRAM | 0.168472 | 0.155246 | 0.435242 | 0.394685 |
| Actual227 postnorm DRAM | 0.168383 | 0.155363 | 0.430334 | 0.486491 |
| Actual227 postnorm WIDTH_SHARDED | 0.173741 | 0.160764 | 0.480209 | 0.489947 |

Traced component gain is about13µs per norm. Eager postnorm regresses; no whole-decoder or prefill gain is claimed. Both candidates intentionally exercise the scan for all linear norm row counts, so parent must evaluate complete warmed prefill and traced decode before selection. Full-attention stays unchanged.

## Frozen candidates and exact source proof

Builder: `tests/v6_build_norm_terminal_scan_snapshot.py`, SHA256 `f16a01d61bba5bc3bc0d34af0bf055849729680714e4f392edd9104146732d20`.

- Independent: `v6_ab28_norm_terminal_scan_integration.py.snapshot`, SHA256 `d6f926660343d8b12466e83fd9d7d2bbd6653d5d6a1912f4422d83c39ca1cca2`; base ab28a91bb3cb5e8b88e5371da0defe3b0d731e86a131c8523483e4b6b65b0334.
- Composed: `v6_transport_conv_norm_terminal_scan_integration.py.snapshot`, SHA256 `803685e105d7ef1d46a6f9a8ceafb1758d9b1131e1c8c35aed1ab05cbf1475a5`; base b253be4403326047c86d94b5e460f6a6e32bfb43373ba85af6f491cd05df1257 (`v6_transport_decode_conv_complete_integration.py.snapshot`).

Each matching `_build.json` records the exact measured gate hash, full timing samples, unchanged method hashes, builder/kernel snapshots, alpha equivalence and precise whole-module AST restoration. Only the linear `norm` branch's two cumsum/crop assignments become `total = self.norm_terminal_scan(groups)`. Setup appends a linear-only `NormTerminalScan` constructor before return. Removing those exact additions and reinstating those two assignments restores each whole base AST exactly. The full-kind norm and setup early branches are independently compared. Composed transport/convolution methods and helpers remain untouched.

Four static definitions are appended: `NormTerminalScan`, `NORM_TERMINAL_READER`, `NORM_TERMINAL_COMPUTE`, `NORM_TERMINAL_WRITER`. They are alpha-equivalent to the measured helper; kernel literal bytes match its gate. There are no test imports or runtime AST operations in either candidate. The constructor stores TTNN/mesh references and queries grid metadata only; it creates/uploads no tensors. All emitted candidates and the builder pass the pinned Black/autoflake/isort toolchain with executable AST unchanged.

Source auditor hand-off: add explicit reviewed candidate hashes and pin the four definitions; no new SETUP exemption is needed. Helper AST SHA256 `dda73a29d720375872db2cc1187d4361f18bf3545233c520d99d14919589635f`; constructor AST `ca944acfa058e4b62a631313e1615961ae446e930beb311dc83e40c24127cef6`. Keep constructor and call body under runtime audit; add host-readback/constructor-upload mutation controls if extending review. Existing exact `FusedDecoder.from_state_dict` setup hashes should be admitted only through these reviewed snapshots.

## Memory and capacity

No persistent tensor, table, KV-cache, recurrent state or dtype is added or changed. The helper has seven4096-byte FP32 CB pages per active core (28KiB), with at most `min(rows, grid.x*grid.y)` cores active. The scalar output is FP32 TILE `(rows,1,1,1)`,4096bytes per logical norm row, identical padded terminal output geometry to the old scan/crop result. Existing group input geometry is unchanged. Intermediate native scan/crop tensors are replaced by local CBs. Debug lane output is disabled in integration, and no persistent decoder setup allocation occurs.

These facts do not establish a new advertised capacity. The context contract is unchanged; parent must retain capacity/long-stream checks, valid nonaligned sequence support, singleton prefill, all meaningful layer kinds and exact state/output quality gates. If a device allocation limit appears, investigate it before changing any advertised value. No context capability reduction was made here.

## Exact source-only build commands used

```bash
PYTHONPATH=. python_env/bin/python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.v6_build_norm_terminal_scan_snapshot \
  --output models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v6_ab28_norm_terminal_scan_integration.py.snapshot \
  --report models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v6_ab28_norm_terminal_scan_integration_build.json
PYTHONPATH=. python_env/bin/python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.v6_build_norm_terminal_scan_snapshot --compose \
  --output models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v6_transport_conv_norm_terminal_scan_integration.py.snapshot \
  --report models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v6_transport_conv_norm_terminal_scan_integration_build.json
```

The original paths already exist and are preserved. A rerun requires fresh output/report paths. Source integration used no hardware and changed no production file, frozen helper, compatibility manifest, or historical evidence. Final runtime choice remains pending independent/combined whole-decoder tests and benchmarks.
