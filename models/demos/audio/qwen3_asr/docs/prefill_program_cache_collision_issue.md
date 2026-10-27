# tt-metal issue draft: prefill program-cache collision across the MLP reshape batch dim

**Component:** `models/tt_transformers` (prefill path) + ttnn matmul program cache
**Arch:** Blackhole (P150). Observed with a Qwen3-1.7B decoder built on `tt_transformers.tt.model.Transformer`.
**tt-metal:** branch `yito/qwen3_asr` @ `b8319d60dd4` (merged `main` @ `98169aec057`).

## Summary

Running two single-user prefills whose padded sequence lengths fall in **different 512-buckets**
(e.g. 512 then 1024) in one long-lived process **`TT_FATAL`s** in the attention output matmul. Each
length **in isolation is fine**. The cached program compiled for the first prefill shape is incorrectly
reused for the second, because the two shapes differ only in a tensor dimension that the matmul program
hash does not distinguish.

This is the root cause behind the "length-keyed prefill corruption" that forces the Qwen3-ASR demo to pin
every prefill to a single fixed length (op-level 512-pad + server-level fixed 14 s chunking), which costs
transcription accuracy (full-clip CER 0.045 → 0.065) and blocks long single-shot / variable-length audio.

## Root cause

`models/tt_transformers/tt/mlp.py:135-137` reshapes the prefill activation when `seq_len >= prefill_len_cutoff`
(`= 512` on Blackhole, `model_config.py:555`):

```python
if mode == Mode.PREFILL and seq_len >= self.args.prefill_len_cutoff:
    x = ttnn.reshape(x, [1, seq_len // self.args.prefill_len_cutoff, self.args.prefill_len_cutoff, -1])
```

So the prefill activation for different padded lengths differs **only in the batch dim `-3`**:

| padded seq_len | reshaped activation |
|---|---|
| 512  | `[1, 1, 512, d]` |
| 1024 | `[1, 2, 512, d]` |
| 1536 | `[1, 3, 512, d]` |

The ff1/ff3/ff2 matmul program configs on this path are length-invariant (fixed grids, `m` pinned to
`min(seq_len, prefill_len_cutoff) = 512`; `mlp2_grid`/`mlp1_3_grid` are `find_prefill_grid(...)`, independent
of seq_len — `model_config.py:756-765`). The attention output (`wo`) matmul config is likewise built from
`m = min(seq_len, 1024)` with a seq-len-independent grid (`get_attn_wo_program_config`, `model_config.py:1988`).

Because the program configs are identical and the input tensors differ only in dim `-3`, the **program-cache
key does not separate the two shapes** for the prefill matmuls (`ttnn.experimental.minimal_matmul` in ff2,
`mlp.py:275-281`, and/or the attention `wo` `ttnn.linear`, `attention.py:1156`). The program compiled for the
first prefill length is reused for the second and asserts on shape.

## Reproduction (observed)

Qwen3-1.7B decoder on one Blackhole (P150), non-paged single-user prefill:

1. Prefill with a padded length of **512** → OK.
2. Prefill with a padded length of **1024** in the same process → **`TT_FATAL`**:

```
TT_FATAL @ ttnn/cpp/ttnn/operations/matmul/device/matmul_device_operation.cpp:58:
a_shape[-1] == b_shape[-2]
The width of the first tensor must be equal to the height of the second tensor.
Mismatch: width=3072 height=2048
```
(backtrace: attention `forward_prefill` → `wo` `ttnn.linear`, `models/tt_transformers/tt/attention.py:1156`.)

3. A **1024** prefill run first, in isolation (no prior 512), **works** — as do 512-only and repeated
   same-length prefills. The crash requires a *prior different-bucket* prefill in the same process.

### Minimal repro sketch (any tt_transformers Transformer on Blackhole)

```python
import ttnn, torch
from models.tt_transformers.tt.model_config import ModelArgs
from models.tt_transformers.tt.model import Transformer

dev = ttnn.open_device(device_id=0, l1_small_size=32768, trace_region_size=200_000_000)
args = ModelArgs(dev, max_batch_size=1, max_seq_len=2048)
model = Transformer(args, ttnn.bfloat16, dev, args.load_state_dict(), args.weight_cache_path(ttnn.bfloat16),
                    use_paged_kv_cache=False)

def prefill(S):
    toks = (torch.arange(S) % args.vocab_size).reshape(1, S)
    inp = model.prepare_inputs_prefill(toks, page_table=None, batch_size=1, user_id=0, last_token_idx=S - 1)
    tl = model.ttnn_prefill_forward(inp[0], rot_mats_global=inp[1], rot_mats_local=inp[2], user_id=0,
                                    page_table=None, get_last_token=((S - 1) // 32) * 32, kv_cache=None, batch_size=1)
    ttnn.from_device(tl)

prefill(512)    # OK, compiles the [1,1,512,d] program
prefill(1024)   # TT_FATAL: reuses the 512 program for the [1,2,512,d] input
```

## Expected vs actual

- **Expected:** each distinct prefill shape compiles (or looks up) its own correct program; interleaving
  512- and 1024-pad prefills produces correct logits for both.
- **Actual:** the second bucket reuses the first bucket's cached program and `TT_FATAL`s (or, on other
  op/shape combinations, silently produces corrupt output — the "locks to first length" symptom).

## Suggested fix

Ensure the prefill matmul program-cache key incorporates the **full input rank/shape**, specifically the
batch dim `-3` that the MLP reshape varies — for `ttnn.experimental.minimal_matmul` (ff2) and the attention
`wo` `ttnn.linear`. Either:
- include dim `-3` (batch) in the program hash of these matmul ops, or
- have the MLP reshape keep the varying length in a hashed dimension.

## Impact / why it matters

With the fix, `tt/qwen3_asr_decoder.py` could drop the min-512 pad and `server/qwen3_asr_server.py` could
drop `FIXED_INFER_SEC = 14.0`, enabling variable-length / long single-shot prefill and recovering the
chunking accuracy loss (CER 0.065 → 0.045). Any tt_transformers model doing multi-length prefill in a
long-lived process (servers, batched eval) is exposed to the same collision.

## References

- `models/tt_transformers/tt/mlp.py:135-137` (prefill reshape), `:275-281` (ff2 `minimal_matmul`)
- `models/tt_transformers/tt/model_config.py:555` (`prefill_len_cutoff`), `:756-765` (fixed grids),
  `:1988` (`get_attn_wo_program_config`)
- `models/tt_transformers/tt/attention.py:1156` (`wo` linear, crash site)
- `models/demos/audio/qwen3_asr/tt/qwen3_asr_decoder.py:prefill_logits` (the 512-pad workaround)
- `models/demos/audio/qwen3_asr/README.md` → "Known limitations"
