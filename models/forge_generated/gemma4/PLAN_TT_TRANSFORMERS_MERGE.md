# Plan: tt_transformers-style merge for gemma4_31b

## Branch state at plan time
- Branch: `sdjordjevic/gemma4_31b`
- HEAD: `47b2a58f19 gemma4_31b: real multi-token generation in demo.py`
- 16 commits since `da2bbbf4b9` (Phase 0 baseline import)
- Total LOC: 6,307 (down from 10,599 baseline = -42%)
- PCC baseline: prefill=0.999464, decode=1.000000

## Goal

Adopt the canonical `tt_transformers` orchestration pattern:

1. **One `Gemma4ForCausalLM` instance** (drop the two `is_decode=True/False` instances)
2. **`mode: Mode` kwarg** on `__call__` instead of construction-time `is_decode`
3. **KV caches preallocated on `self.layer_past`** (one K, one V tensor per layer) instead of synthesized into runtime input slots
4. **`current_pos` runtime kwarg** (instead of position-id input slots)
5. **`Generator` class** that wraps the model and runs prefill → decode loop with per-step `current_pos` tracking

This unblocks fast multi-token generation: prefill writes the prompt's K/V into `self.layer_past`, decode reads them and writes one new K/V slot per step. Replaces the current `demo.py`'s O(seq_len²) prefill-loop with O(seq_len + N) prefill-then-decode.

## Reference: tt_transformers patterns

| Pattern | tt_transformers location |
|---|---|
| `mode` runtime kwarg through forward | `tt/attention.py:1219` (`Attention.forward()`) |
| KV cache on instance | `tt/attention.py:384-439` (`Attention.init_kv_cache()` → `self.layer_past = [K, V]`) |
| External cache override (vLLM) | `tt/attention.py:723-728` (the `kv_cache=` kwarg) |
| `current_pos` runtime kwarg | `tt/attention.py:630-800+` (`forward_decode(current_pos, ...)`) |
| OpGroup per-mode tuning | `tt/model_config.py:67-82` (LI_QKV_DECODE / LI_QKV_PREFILL etc.) |
| Generator API | `tt/generator.py:54-100` + `2211+` (manual prefill→decode loop) |

## Phased plan

PCC verified after every phase. If a phase drops PCC, revert and diagnose before proceeding.

### Phase 0: Recon and audit (no code changes)

Goal: confirm assumptions before touching code.

Tasks:
- Inventory all `ttnn.deallocate(k_cache, ...)` / `ttnn.deallocate(v_cache, ...)` / `ttnn.deallocate(pos_ids, ...)` calls in the four attention bodies (`_sliding_decode`, `_sliding_prefill`, `_full_decode`, `_full_prefill`).
- Map each layer's K/V cache slot indices via `layer_table.LAYER_TABLE_PREFILL` and `LAYER_TABLE_DECODE`. Confirm prefill slots == decode slots (they should — both produced by `_expand_common_recipes` in `runtime_inputs.py`).
- Confirm KV cache shapes from `runtime_inputs.py:_COMMON_ZEROS_TILE_BF16_4x256x256` (sliding K/V: `[1, 4, 256, 256]`) and `_COMMON_ZEROS_TILE_BF16_1x256x512` (full K/V: `[1, 1, 256, 512]`). These are the shapes `self.layer_past` must allocate.
- Map every input slot read in the orchestration (`_call_decode` / `_call_prefill`) and decoder bodies — categorize as: KV cache | position-id helper | scratch zeros | token IDs | scalar 1.0.
- Find every place position-id helpers are read from input. Identify which slots feed which ops.

Deliverable: a small inventory file (or notes in the planning doc) listing the K/V cache slots per layer, KV shapes, and position-helper consumers.

### Phase 1: Dealloc surgery in attention bodies

Goal: stop the four attention bodies from deallocating KV cache and position tensors. PCC must stay green (the caches just live longer; the test runs once so memory isn't an issue yet).

Tasks:
- In `attention.py`, comment out or guard with `if False:` the lines that dealloc K cache, V cache, and pos_ids (~6-12 lines per body, ~30 total).
- Run PCC for both prefill and decode.
- Commit: `gemma4_31b: don't dealloc KV cache / pos_ids in attention bodies`.

Risk: low. If something else implicitly relies on those tensors being freed, PCC will catch it.

### Phase 2: Allocate KV caches on model; layers reference them

Goal: KV caches live on a shared object that both prefill and decode model instances reference. The decoder layer's `__call__` reads from `self.k_cache` / `self.v_cache` instead of `input[k_slot]` / `input[v_slot]`.

Tasks:
- Add a `Gemma4Caches` class (or just per-layer dicts) that preallocates K and V tensors for all 60 layers based on layer type:
  - sliding (50 layers): K, V each `[1, 4, 256, 256]` BFLOAT16 TILE replicated
  - full (10 layers including L59): K `[1, 1, 256, 512]` BFLOAT16 TILE replicated; V same
- `Gemma4ForCausalLM.from_state_dict(... caches=None)` accepts optional shared caches; if None, allocate fresh.
- During construction, pass the per-layer K/V references into each `SlidingDecoderLayer` / `FullDecoderLayer`'s `__init__` (new kwargs `k_cache`, `v_cache`).
- In each layer's `__call__`, change `kv = (input[k_slot], input[v_slot], input[pos_slot])` to `kv = (self.k_cache, self.v_cache, input[pos_slot])`.
- Add a `model.reset_kv_caches()` method that zeros all caches. Call it at the start of `model.__call__()` for now (so prefill PCC reproduces).
- Update `runtime_inputs.synthesize_*_inputs` to NOT allocate K/V cache slots (or to allocate them and have them be ignored; cleaner to drop them).
- Update tests / demo to use the new API.

Run PCC for both modes from the same caches (build prefill model, run it; build decode model with the same caches argument, run it).

Commit: `gemma4_31b: KV caches on shared Gemma4Caches; layers reference them`.

Risk: medium. Cache shape mismatches will trip immediately. Cache contamination across calls will subtly drop PCC.

### Phase 3: Position kwarg

Goal: `current_pos` is a runtime kwarg through `model.__call__`. The model builds the position tensor(s) internally, replacing the position-id input slots.

Tasks:
- Add `current_pos: int = 0` to `Gemma4ForCausalLM.__call__`. For prefill, it's typically 0 (start of sequence); for decode, it's the new token's position.
- Build the per-call position tensors inside `_call_prefill` / `_call_decode` from `current_pos` (instead of reading them from input slots).
- Drop the position-id slots from `runtime_inputs.py`.
- Run PCC. Decode reference at `current_pos=0` (matches the canonical decode test setup); prefill reference at `current_pos=0`.

Commit: `gemma4_31b: current_pos as runtime kwarg`.

Risk: medium. There are subtle position-helper tensors (slot 0, slot 26) whose semantics need to be preserved exactly. Compare what `synthesize_decode_inputs` synthesizes against what the orchestration actually consumes.

### Phase 4: Mode kwarg merge

Goal: one `Gemma4ForCausalLM` class. `is_decode` is a runtime arg (`mode: Mode`), not a constructor flag.

Tasks:
- Add `Mode` enum (`Mode.PREFILL`, `Mode.DECODE`) — could just use a Python enum or two string constants.
- Drop `is_decode` from `Gemma4ForCausalLM.__init__`. The model holds both `_call_decode` and `_call_prefill` methods (already does); `__call__(input, mode)` dispatches.
- The model needs to construct BOTH sets of layer_table entries (sliding-decode vs sliding-prefill specifics). One set of layers can serve both modes — Attention already takes `is_decode` per call, RMSNorm and FeedForward are mode-agnostic.
- The two preludes need both decode and prefill versions on the same instance: `self.sliding_prelude_decode`, `self.sliding_prelude_prefill`, `self.full_prelude_decode`, `self.full_prelude_prefill`.
- L58 (the per-prefill-mode special) is already a `SlidingDecoderLayer` instance — it can become `self.layers[58]` for both modes (unification across modes).
- L59 (terminal) is already a `FullDecoderLayer(is_terminal=True)` for both modes — keep `self.l59` shared.
- Update `decoder_layer.py`: `SlidingDecoderLayer` / `FullDecoderLayer` drop `_is_decode` from `__init__`; `__call__` takes `is_decode` kwarg, dispatches.

Run PCC for both modes from the merged model instance.

Commit: `gemma4_31b: merge prefill/decode into one model with mode kwarg`.

Risk: high. This restructures the most code at once. Consider sub-phasing: 4a) merge decoder layers (mode kwarg per call), 4b) merge model class.

### Phase 5: Generator class

Goal: a `Generator` that owns the model and runs prefill → decode loops.

Tasks:
- New file `gemma4/generator.py` with `class Generator`:
  - `__init__(self, model, tokenizer)`
  - `prefill(self, prompt_tokens) -> int` — pads prompt to `seq_len`, calls `model(input, mode=PREFILL, current_pos=0)`, returns next token. Side effect: `self.current_pos = len(prompt_tokens)`.
  - `decode_step(self, prev_token) -> int` — builds decode input list with single token, calls `model(input, mode=DECODE, current_pos=self.current_pos)`. Increments `self.current_pos`.
  - `generate(self, prompt_tokens, max_new_tokens) -> list[int]` — calls prefill once, then decode_step in a loop.
- Update `demo.py` to use `Generator`.
- Verify generated text matches the prefill-loop demo's output: "As an AI, I don't have personal feelings or the ability to travel".

Commit: `gemma4_31b: add Generator class for fast prefill→decode generation`.

Risk: medium. KV cache continuity between prefill and decode is the key correctness property. If decode reads stale or wrong cache state, generation diverges.

### Phase 6: Cleanup

Goal: remove dead code from the runtime_inputs, simplify the test API.

Tasks:
- Drop K/V cache slots and position slots from `synthesize_*_inputs` (they're dead now).
- Drop construction-time `is_decode` machinery (layer_table per-mode lookups, etc.) where redundant.
- Update `tests/conftest.py` if needed (probably not — tests just call the new API).
- Final PCC run on both modes.

Commit: `gemma4_31b: drop dead runtime input slots after KV/pos lift`.

## Things to NOT lose

These are existing properties that must survive the refactor:

1. **PCC**: prefill=0.999464, decode=1.000000 at default seq_len=19.
2. **seq_len parameterization**: built up in commit `a68f8e2481`. The merged model must still accept `seq_len=N` at construction.
3. **L58/L59 unification**: commits `e3d5d88ee4` and `5657df49af`. Don't reintroduce `_full_layer_59_decode` / `_full_layer_59_prefill` / `_sliding_layer_58_prefill` special methods.
4. **Multi-token demo**: `demo.py` should still produce coherent generation from the canonical prompt. Ideally faster after Phase 5 (one prefill + N decodes vs N prefills).

## Hardware / env setup

```bash
source python_env/bin/activate
export TT_METAL_HOME=$(pwd) ARCH_NAME=blackhole
export PYTHONPATH=$(pwd):$(pwd)/models/forge_generated
```

PCC test commands:
```bash
pytest models/forge_generated/gemma4/test_prefill.py -v -s
pytest models/forge_generated/gemma4/test_decode.py -v -s
```

Demo:
```bash
python -m gemma4.demo "What is your favorite city?" --seq-len 64 --max-new-tokens 16
```

Hardware: 4 Blackhole devices, mesh (1,4). Tests take ~60s each.

## Estimated scope

~4-6 hours of careful work with hardware verification. Each phase commits independently; safe to land any subset.

The earliest "real win" is Phase 5 (the Generator), but it depends on Phases 2-4 to be in place. Phases 1-2 alone are useful (KV cache lifetime cleanup) and ship a meaningful refactor.
