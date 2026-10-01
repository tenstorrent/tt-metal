# DeepSeek V4 Flash: traced prefill and long-context decode issues

## Summary

1. **Fixed: traced prefill produced garbage for chunks of 512 tokens or more.**
   The cause is a `ttnn.broadcast` bug: a single row-major page larger than one fabric packet (about 4.3 KB) arrives corrupted on the receiving devices. The stage-0 input packet crosses that size at chunk ≥ 512. Worked around in `tt/model.py` by broadcasting the packet as rows of ≤ 4 KB.
2. **Open: decode-only (prompt fed token by token) goes wrong between 8192 and 10752 prompt tokens.** Prefill is correct at that length; decode-only is not.
3. **Open: end-to-end answer quality.** On longbench question 112 the model answers "(A)" (expected C) and repeats the sentence until the token limit.
4. **Minor, open:** an "allocating under active trace" warning after the traced prefill run, from the `_export_states` slices.

## Setup

- Test: `tests/prefill/test_prefill_decode_demo.py::test_prefill_decode_demo`, 8 pipeline stages × TP4 on 32 chips, prefill and decode sharing the same chips.
- Prompt: longbench question 112 (`length == "short"`, 10752 tokens after tokenization), expected answer C.
- Relevant knobs:
  - `DEEPSEEK_V4_LONGBENCH_INDICES`: question index.
  - `DEEPSEEK_V4_E2E_CHUNK`: prefill chunk size (minimum 128).
  - `DEEPSEEK_V4_DECODE_LAYERS`: layer count (4 for fast checks).
  - `DEEPSEEK_V4_E2E_MAX_INPUT`: truncate the prompt to the first N tokens.
  - `DEEPSEEK_V4_E2E_TRACE_CHECK=1`: compare traced prefill against eager prefill (packet contents per rank, per-layer state PCC, logits PCC), then skip.
  - `DEEPSEEK_V4_E2E_COMPARE=1`: compare prefill logits against decode-only logits.
  - `DEEPSEEK_V4_PREFILL_INDEXER`: override whether the CSA lightning indexer is used.

## Issue 1: traced prefill garbage (fixed)

### Symptoms

- With chunk ≥ 512, traced prefill produced all-zero top-5 logits and PCC `nan` against decode-only.
- Eager prefill on the same prompt was coherent (it answered A with an explanation and reached EOS).
- With chunk 128 (and max input 128), traced prefill matched eager exactly.

### How it was narrowed down

1. `DEEPSEEK_V4_E2E_TRACE_CHECK` on 4 layers, chunk 512, max input 512:
   - The TP rank 0 state matched eager.
   - TP ranks 1–3 had corrupt CSA / HCA compressed entries and bad logits.
2. Disabling the indexer did not change the result, so the indexer was not the cause.
3. Checked the packet each stage-0 rank received after the broadcast:
   - Rank 0 (the H2D receiver) was correct.
   - On ranks 1–3, 24 slots of the packet were wrong. The rest was correct.
4. The packet holds token ids `[0, C)`, positions `[C, 2C)`, entry positions per compress rate, then the last-token slot, rounded up to 64 B. It is sent as a single row-major page `[1, 1, 1, W]`. At C = 128 the page is below about 4352 B; at C ≥ 512 it is above. Corruption only appears above that size, which is about one fabric packet.

### Fix

In `TracedPrefill._stage_forward`, stage 0:

```python
ttnn.experimental.recv_async_h2d(stage.pkt, self._pkt_socket)
if stage.device.get_num_devices() > 1:
    width = self._pkt_bcast_width
    rows = ttnn.reshape(stage.pkt, [1, 1, self._pkt_w // width, width])
    sent = ttnn.broadcast(rows, ttnn.MeshCoordinate(0, 0), cluster_axis=1, topology=ttnn.Topology.Ring)
    pkt = ttnn.reshape(sent, [1, 1, 1, self._pkt_w])
```

`_pkt_bcast_width` is the largest divisor of the packet width that is ≤ 1024 int32 (4 KB).

### Verification

- Chunk 512, 4 layers: packets on all ranks correct, all state PCCs 1.0.
- Full 10752-token prompt, chunk 1024, indexer on, 4 layers: logits PCC 0.99999, every per-layer tensor (`kv_tail`, `compressed_kv`, `csa_prev_kv`, `csa_prev_gate`, `idx_keys`, `idx_prev_kv`, `idx_prev_gate`) PCC 1.0000.

### Follow-up

The underlying bug is in `ttnn.broadcast` (row-major pages larger than one fabric packet). Any other caller that sends large row-major pages is affected. It should be reproduced in a standalone op test and filed.

## Issue 2: decode-only wrong at long context (open)

### Observations

Prefill logits against decode-only logits (full model, question 112, prompt truncated with `DEEPSEEK_V4_E2E_MAX_INPUT`):

| Prompt tokens | Logits PCC | Same argmax | Top-10 overlap |
|---|---|---|---|
| 1024 | 0.93 | yes | 7/10 |
| 4096 | 0.95 | yes | not recorded |
| 8192 | 0.98 | yes | 9/10 |
| 10752 | 0.60 | no | not recorded |

At 10752, decode-only predicts `' to'`. Prefill predicts `'</think>'` (24.88), which is the plausible continuation of the chat template. Because traced prefill matches eager prefill (Issue 1), the decode side is the one that is wrong.

### Hypotheses (not yet tested)

- **CSA lightning indexer capacity.** Decode uses `fused_lightning_select_kv` with `index_topk = 512`, `CSA_MAX_COMPRESSED_ENTRIES = 512`, `CSA_INDEX_BLOCK_SIZE = 64`. The compressed entry count grows with the prompt, and some capacity (such as `max_blocks_per_core` in the program factory, or a fixed entry buffer) may be exceeded between 8192 and 10752 tokens.
- **Pipelined feeding.** The prompt is now fed through `decode_prompt_traced` (writer thread writing packets ahead, reader thread consuming outputs). It matched sequential decode at shorter lengths, but has not been checked at 10752.

### Suggested next steps

1. Bisect the failure length: `DEEPSEEK_V4_E2E_COMPARE=1 DEEPSEEK_V4_LONGBENCH_INDICES=112 DEEPSEEK_V4_E2E_MAX_INPUT=<N>` for N in 8192…10752.
2. At the first failing length, compare sequential `decode_traced` against `decode_prompt_traced` to rule pipelining in or out.
3. Check whether the failing length corresponds to a compressed-entry count crossing a power of two or a capacity constant, and inspect the reader, writer and compute kernels of `fused_lightning_select_kv`.

## Issue 3: answer quality (open)

- Full model, traced prefill (chunk 1024, indexer on), question 112: prefill at 2908 tok/s, decode at 28.2 tok/s.
- Output: "The correct answer is (A)." repeated until the 128-token limit. Expected C.
- Eager prefill on the same prompt also chose A but produced an explanation and reached EOS. The repetition suggests decode drift after prefill, and may share a root cause with Issue 2, because generation starts beyond 10752 tokens of context.
