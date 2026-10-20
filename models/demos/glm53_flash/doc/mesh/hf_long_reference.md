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

Prompt: `build_needle_ids` from the test (same seeds, text and chat template; wt_mesh 8c5a3cb5902). Under
transformers 5 `apply_chat_template(tokenize=True)` returns a `BatchEncoding` (the original test's `len(ids)` was 2
and the filler length exploded); the builder now takes `input_ids` and appends `</think>` after the generation
prompt (`<|assistant|><think></think>`), so the greedy continuation is the answer itself. Prompt lengths and the
sha256 of `json.dumps(ids)` (also in each `prompt_ids.json` / `reference.json`):

| N | ids | sha256 |
|---|---|---|
| 4096 | 4079 | 3e4b9cd2cf96e5df99b628a96a286d5c57b22a8d19fa7be6173297f8f52af932 |
| 16384 | 16359 | f7110fd30b4f493e02a15277cfa5860b5cf7d434102b4240b06a1ea029d45db4 |
| 65536 | 65493 | 8574d55f4456f1ef7991990638bf85d385e41415e4e137e0fc4f426d61dfe7dc |

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

| run | result |
|---|---|
| `--selftest` (blocked indexer vs stock, S=300 + 1-token step, S=257 + 5-token step) | same top-k sets |
| `--validate-g16 --attn eager` (13-token prompt, 16 greedy tokens) | all 16 tokens identical to FULLMODEL_G16, top-5 ids identical at every step, max abs logit difference 0.000 |
| `--validate-g16 --attn sdpa` | first 7 tokens and the step-0 top-5 identical (max abs logit difference 0.31 at step 0, <= 0.5 through step 7); at step 7 a near-tie flips (eager top-2 [73538, 11478] -> sdpa [11478, 73538]) and the greedy chains part. HF-internal numerics: eager multiplies P x V in bf16 after an fp32 softmax, the sdpa CPU flash kernel accumulates in fp32. The needle references use sdpa (the `from_pretrained` default; the only path that fits 65K), so a TT-vs-HF greedy divergence at a near-tie is expected too; `compare_to_tt.py` prints HF's top-5 at the divergence step |
| NEEDLE_4096 (4079 ids, sdpa, chunk 16384) | first token 15209 "amber" (top-5 15209 amber / 785 The / 20431 secret / 40 I / 1925 On); greedy [15209, 2220, 45935, 12, 22, 101294, 16, 154827] = 'amber-falcon-7391' + eos -> needle FOUND; prefill 819.7 s (5 tok/s under the shared load), decode 45.7 s for 7 steps (4-9 s/token), wall 865.9 s |
| NEEDLE_16384 / NEEDLE_65536 | pending (see `/home/ttuser/ashai/glm53/_agents/hfref/STATUS.md`) |

Host (AMD EPYC 9354P 32c/64t, avx512_bf16, shared with two other jobs at the time): 13-token prefill 61 s, decode
16-63 s per token (8 experts x 42 layers dequantized per step, ~8 GB of shard reads).
