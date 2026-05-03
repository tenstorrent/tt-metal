# Gemma 4 Trace Signatures

## Decode Trace V0

| Field | Value |
| --- | --- |
| Entry point | `models/demos/gemma4/demo/text_demo.py::run_generation` |
| Mode | Decode after prefill |
| Batch | `1` |
| Mesh | `1x8` for 26B-A4B target |
| Page block size | `64` |
| Max sequence in demo | `4096` unless overridden |
| Hidden input | `[1, 1, 1, 2816]` BF16 |
| Position RoPE input | `[1, 32]` UINT32 |
| Cache position input | `[1]` INT32 |
| Optional PLI | Not used by 26B-A4B (`hidden_size_per_layer_input=0`) |
| Persistent outputs | `trace_output` from captured decode forward |

## Warmup Order

1. Load config and weights.
2. Create model, RoPE caches, page table, and KV caches.
3. Run prefill for the prompt.
4. Run one untraced decode iteration to compile.
5. Capture decode trace with fresh persistent input buffers.
6. Replay trace with copied host input contents and fixed device addresses.

## Acceptance Checks

The accepted run must log trace capture and replay, generate long decode output under `enable_decode_trace=True`, and report TTFT plus decode tokens/sec/user. Compile iterations must be reported separately from steady replay.

## Strict Decode Trace V1

| Field | Value |
| --- | --- |
| Entry point | `models/demos/gemma4/demo/strict_device_feedback_demo.py` |
| Mode | Decode after prefill, batch=1 |
| Mesh | `1x8` for 26B-A4B target |
| Page block size | `64` |
| Max sequence in accepted run | `512` |
| Token input | `[1]` UINT32, ROW_MAJOR, device-resident |
| Token output | `[32]` UINT32, ROW_MAJOR, device-resident padded sampling output |
| Hidden input | Produced inside trace by `model.embed_tokens(token_in)` |
| Position RoPE input | `[1]` UINT32, ROW_MAJOR, device-resident |
| Cache position input | `[1]` INT32, ROW_MAJOR, device-resident |
| Optional PLI | Not used by 26B-A4B (`hidden_size_per_layer_input=0`) |
| Persistent outputs | `token_in`, `token_out`, `pos_rope`, `pos_cache` mutated by trace replay |

## Strict Warmup Order

1. Load config and weights.
2. Create model, RoPE caches, page table, and KV caches.
3. Run prefill for the prompt and choose the first decode token on host.
4. Allocate persistent device buffers: `token_in`, padded `token_out`, `pos_rope`, `pos_cache`.
5. Run one untraced strict decode iteration to compile and advance positions once.
6. Capture strict decode trace using the persistent buffers.
7. Replay trace without host token or position copies.
8. Read final token/position buffers only after the timed replay window.

## Strict Acceptance Checks

The accepted strict run must log trace capture and replay, report TTFT plus decode tokens/sec/user, and show final RoPE/cache positions equal `prompt_len + 1 + replay_count` on every device. Compile iterations must be reported separately from steady replay. The strict harness currently validates token handoff by final buffer state; it does not reconstruct the full generated text stream.
