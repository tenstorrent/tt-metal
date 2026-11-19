# AgentX-like prefill throughput, galaxy 8x4 (2026-11-19)

Text: `agentic_cc` (Claude Code harness sessions from nvidia/Open-SWE-Traces, rendered with the checkpoint's chat
template; reference/texts.py F287). One steady 5120-token chunk at each history position, DECODER_SKIP=decode
(layers 0-19 + layer 20's KV head), the resident runner (layers 0-20 loaded), harness walls (tests/test_skip_positions.py).
Serving adds the ids readback, ~1.5 ms a chunk (tests/perf/test_ids_readback.py): about -1.3k tok/s at each point.

## Current tree: 04c14a09d00 (4-bit history, host hiding, C++ engram host path, mHC fast, grouped wo_a)

| Context | Steady wall ms | tok/s | Resumed wall ms (first chunk of a request) | Device ms | Exposed host ms |
|---|---|---|---|---|---|
| 50k (51200) | 74.1 | 69.1k | 75.8 | 60.7 | 13.4 |
| 128k (131072) | 75.2 | 68.1k | 79.5 | 62.8 | 12.4 |
| 256k (262144) | 77.2 | 66.3k | 80.6 | 64.3 | 12.9 |
| 512k (524288) | 79.5 | 64.4k | 81.4 | 69.8 | 9.7 |
| 768k (786432) | 84.9 | 60.3k | 87.3 | 77.5 | 7.4 |
| 1M (1024000) | 87.4 | 58.6k | 89.0 | 80.3 | 7.1 |

## Earlier the same day: before mHC fast (P.83) and grouped wo_a (P.81)

| Context | Steady wall ms | tok/s | Resumed wall ms | Device ms | Exposed host ms |
|---|---|---|---|---|---|
| 50k | 75.9 | 67.5k | 80.1 | 62.8 | 13.1 |
| 128k | 77.5 | 66.1k | 79.6 | 64.5 | 13.0 |
| 256k | 78.8 | 65.0k | 81.8 | 67.1 | 11.7 |
| 512k | 82.3 | 62.2k | 83.7 | 71.9 | 10.4 |
| 768k | 87.0 | 58.9k | 88.7 | 79.9 | 7.1 |
| 1M | 90.4 | 56.6k | 90.6 | 83.1 | 7.3 |

Comparison bar (owner): another stack ~67k new tok/s a Galaxy on layers 0-20 with packed 8k-row passes.
