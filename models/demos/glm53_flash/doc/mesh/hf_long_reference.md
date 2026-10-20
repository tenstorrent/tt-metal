# HF CPU reference for the long-context needle prompts

Host-side (CPU, transformers 5.20.2) greedy references for the needle-in-a-haystack prompts of
`tests/test_mesh_long_context.py`, so the mesh run's first token and greedy chain can be checked token by token.
Branch `arg/glm53_hfref`; no device code is involved.

## Pieces

| file | role |
|---|---|
| `reference/hf_long_reference.py` | builds the test's prompt (imports `build_needle_ids`), runs the HF text model on the host, writes `NEEDLE_<N>/` |
| `reference/compare_to_tt.py` | lines a mesh JSON (`generated/glm53_flash_long_context/mesh<R>x<C>_N<S>.json`) up with a `NEEDLE_<N>/` reference |
| `tests/test_mesh_long_context.py` | one-line fix: `apply_chat_template(..., return_dict=False)` (see below) |

## How the host model is run

The bf16 `Glm5NextTextModel` is ~620 GB (304 B of the 313 B parameters are the 42 x 288 routed experts), so it is
not instantiated. `hf_long_reference.py` keeps every non-expert weight resident (8.9 B params; bf16, strict-fp32
groups kept, exactly `cast_like_from_pretrained`) and replaces each MoE layer's `Glm5NextTextExperts` by
`LazyExperts`: the HF eager per-expert loop verbatim, with `gate_up_proj[e]` / `down_proj[e]` read from the fp8
shards and dequantized by the same `dequantize_weight_tensor` call `build_layer` uses, for the experts a forward
hits (8 per layer per decode token; all 288 per prefill chunk, ~8 s per MoE layer). Everything else is the stock
HF module: KDA (`chunk_kda` reference path), DSA attention through HF's `sdpa` interface (the CPU flash kernel
with the top-k bool mask, memory-blocked; `from_pretrained`'s default), indexer, mHC, router (fp32), shared
expert, norms, embedding, `lm_head` in bf16 -> argmax.

Two memory measures for long prompts, both without changing the result:
- the prompt goes through the HF `DynamicCache` in chunks (`--chunk`, default 16384): the KDA conv/recurrent
  states, the latent KV and the indexer keys carry over exactly as in a single pass;
- the indexer's `[B, S, heads, pools]` fp32 pool scores are evaluated `--index-qblock` (2048) queries at a time
  (`install_blocked_indexer`; `--selftest` checks the selected top-k set equals the stock forward's).

Validation against the existing host reference `FULLMODEL_G16` (13-token prompt, 16 greedy tokens with per-step
top-5 logits, produced layer-by-layer by `hf_layer_reference.py --generate 16` with eager attention): see the
results table.

Prompt: `build_needle_ids` from the test (same seeds, text and chat template). Under transformers 5
`apply_chat_template(tokenize=True)` returns a `BatchEncoding`, so the test's `len(ids)` was 2 and the filler
length exploded (the call hangs); the test now passes `return_dict=False` and the reference script shims the
same. The chat template prepends `<|system|>Reasoning Effort: Max` and ends the generation prompt with
`<think>`, so the greedy tokens are the start of the model's reasoning, not the bare passphrase: the `needle_found`
flag is only meaningful for runs long enough to leave the thinking block.

## Commands

```
export W=/home/ttuser/ashai/glm53/tt-metal WT=/home/ttuser/ashai/glm53/wt_hfref
export PYTHONPATH=$WT:$W/ttnn:$W TT_METAL_HOME=$W
cd $WT
$W/python_env/bin/python models/demos/glm53_flash/reference/hf_long_reference.py --selftest
$W/python_env/bin/python models/demos/glm53_flash/reference/hf_long_reference.py --attn eager --validate-g16 <FULLMODEL_G16/generate.json>
$W/python_env/bin/python models/demos/glm53_flash/reference/hf_long_reference.py --n 4096 16384 --chunk 16384 --out <refs>
$W/python_env/bin/python models/demos/glm53_flash/reference/hf_long_reference.py --n 65536 --chunk 16384 --out <refs>
$W/python_env/bin/python models/demos/glm53_flash/reference/compare_to_tt.py \
    --tt generated/glm53_flash_long_context/mesh8x4_N16380.json --ref <refs>/NEEDLE_16384 --snapshot <snapshot>
```

`<refs>` = `/home/ttuser/ashai/glm53/wt_mesh/bringup/references/glm53_flash` (the shared, git-excluded references
directory). Each `NEEDLE_<N>/` holds `prompt_ids.json` (ids + sha256), `reference.json` (first token, generated ids,
text, top-20 last-token logits, per-step top-5, wall times, needle flag, config) and `last_logits.pt` (fp32 [vocab]).

## Results

(filled in as the runs complete; see `/home/ttuser/ashai/glm53/_agents/hfref/STATUS.md` for the live state)
