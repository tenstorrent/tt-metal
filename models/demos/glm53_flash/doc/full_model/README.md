# GLM-5.3-Flash full model on one chip: weight streaming, teacher-forced check (Phase 3, step 1)

Design of record: `~/ashai/GLM_5.3_DESIGN.md` (Phase 3 "full model: 45 layers, embedding/head/sampling,
teacher-forcing accuracy"). Code: `tt/model.py` (`GLM53Model`), `tests/test_full_model.py`; host side
`reference/hf_layer_reference.py --generate`. Host `sjc3-glx-a04u07`, chip `TT_VISIBLE_DEVICES=0`,
`MeshShape(1,1)`, branch `arg/glm53_fullmodel` (from `arg/glm53_mesh` @ 21b109e5193).

## What it is

`GLM53Model` chains the stage-1 `FunctionalDecoder` through all 45 layers plus `TtEmbedding` and
`TtModelHead` on ONE chip. A chip holds one layer's weights (7.7 GB bfp8 for a MoE layer) next to the
embedding table (1.27 GB bf16), the LM head (0.65 GB bfp8) and the autoregressive state of every layer,
so each pass over the stack streams the weights:

    for each layer i:  build (checkpoint -> device) -> run (prefill chunk / decode step) -> release

| piece | role |
|---|---|
| `FunctionalDecoder.release_weights()` / `attach_layer()` | the decoder object is the state owner (KDA user slots, DSA paged latent cache + pooled-key index, `next_start`) and lives for the whole model; `release_weights` walks the attached `TtDecoderLayer` (`device_tensors_of`: every `ttnn.Tensor` reachable through the model's own objects; the per-mesh shared KDA history exchange is skipped) and deallocates all of them, `attach_layer` re-arms it with a freshly built layer of the same index |
| `GLMMesh.upload_cached` | `upload` through a tensorbin cache: `ttnn.load_tensor` on a hit (dtype / layout / shape checked, a stale file raises), `from_torch` + `ttnn.dump_tensor` (temp file + rename) on a miss; the host tensor is a callable so a hit never dequantizes the checkpoint |
| `TtRoutedExperts` / `TtMoE(weight_cache_path=)` | the routed W13 / W2 (95 % of a MoE layer's bytes) go through that cache (`experts_w13_<dtype>_mesh1x1.tensorbin`, `experts_w2_...`); with a complete cache the state dict may omit `mlp.experts.N.*` (`TtMoE.expert_cache_complete`) and `GLM53Model.build_layer` filters them out of `load_layer_raw` |
| `GLM53Model.prefill(ids, all_rows=)` / `decode(ids, positions)` / `generate(ids, n)` | embedding -> streamed layers -> head; decode runs the `max_batch` user slots; `generate` is greedy for one user (host argmax of the device logits, the device argmax recorded next to it) |
| `GLM53Model.timing` / `timing_table()` / `dram_view` | per-layer build / run / release seconds of every pass, expert-cache hits, DRAM allocated after each build |

Cache root: `/home/ttuser/ashai/glm53/_cache/fullmodel/layer<i>/` (expert tensorbins + the KDA prefill
weight tensorbins of `load_kda_weights`). The embedding / head / norms / attention projections / shared
expert / mHC tensors are converted on every build (~3-4 s per layer, the dense / attention bfp8 conversions),
the experts load in ~2.6 s against ~50 s of host bfp8 conversion (measured on layers 3 / 4).

## Running

```bash
source ~/ashai/glm53/env.sh; export TT_VISIBLE_DEVICES=0
pytest models/demos/glm53_flash/tests/test_full_model.py -s --timeout 7000 -k test_dump_chain     # (1)+(2)
python models/demos/glm53_flash/reference/hf_layer_reference.py --generate 16 \
    --prompt "The capital of France is Paris. The theory of general relativity" \
    --out bringup/references/glm53_flash/FULLMODEL_G16                                       # HF side of (3), CPU
pytest models/demos/glm53_flash/tests/test_full_model.py -s --timeout 7000 -k test_generate_vs_hf  # (3)
```

`GLM53_FULLMODEL_LAYERS=0-4` runs the chain on a layer prefix (mechanics; head gates skipped),
`GLM53_FULLMODEL_CACHE=none` disables the tensorbin cache, `GLM53_FULLMODEL_N` / `GLM53_FULLMODEL_PROMPT`
set the generation. The HF generation streams one layer at a time per token (prefill pass + 15 decode
passes, ~10 s per MoE layer build on a warm page cache, slower under load) and resumes from
`generate_state.pt`.

## Evidence (real weights; logs under `~/ashai/glm53/_logs/fullmodel_*.log`)

### 1. Dump chain: the 128-token random prompt through all 45 layers (`fullmodel_dump_chain.log`)

TT chain = every layer fed the TT output of the previous one; PCC vs the dump's HF `x_out` of that layer.
"standalone" = the same layer on its exact HF input (`test_all_layers.py`, `_logs/layers45_chip*.log`).
"HF+1 %" = the HF bf16 chain itself with a 1 % relative Gaussian error added to every layer's output
(`reference/hf_chain_sensitivity.py --noise 0,0.01,0.02,0.04`, `fullmodel_hf_chain_sensitivity.log`; the
`r=0` stream reproduces the dump at 1.000000 on every layer).

| layer | kind | standalone | TT chain prefill (worst row) | TT chain decode 0 / 1 (pos 128 / 129) | HF+1 % chain | HF+2 % | HF+4 % |
|---|---|---|---|---|---|---|---|
| 0 | KDA+dense | 0.999799 | 0.999799 (0.99960) | 0.999865 / 0.999886 | 1.00000 | 1.00000 | 1.00000 |
| 1 | KDA+dense | 0.999949 | 0.999766 (0.99896) | 0.999885 / 0.999912 | 0.99996 | 0.99988 | 0.99956 |
| 2 | KDA+dense | 0.999978 | 0.999926 (0.99870) | 0.999945 / 0.999959 | 0.99998 | 0.99993 | 0.99974 |
| 3 | DSA+MoE | 0.999846 | 0.999712 (0.99354) | 0.999851 / 0.999952 | 0.99966 | 0.99910 | 0.99748 |
| 4 | KDA+MoE | 0.999974 | 0.999625 (0.99310) | 0.999846 / 0.999942 | 0.99949 | 0.99868 | 0.99625 |
| 5 | KDA+MoE | 0.999965 | 0.999459 (0.99272) | 0.999839 / 0.999934 | 0.99920 | 0.99784 | 0.99383 |
| 6 | KDA+MoE | 0.999968 | 0.999336 (0.99218) | 0.999836 / 0.999923 | 0.99880 | 0.99680 | 0.99037 |
| 7 | DSA+MoE | 0.999965 | 0.999217 (0.99121) | 0.999703 / 0.999915 | 0.99853 | 0.99615 | 0.98824 |
| 8 | KDA+MoE | 0.999974 | 0.999098 (0.99113) | 0.999695 / 0.999897 | 0.99831 | 0.99556 | 0.98660 |
| 9 | KDA+MoE | 0.999983 | 0.998921 (0.99051) | 0.999676 / 0.999095 | 0.99799 | 0.99480 | 0.98447 |
| 10 | KDA+MoE | 0.999968 | 0.998370 (0.98782) | 0.999671 / 0.998914 | 0.99716 | 0.99236 | 0.97901 |
| 11 | DSA+MoE | 0.999957 | 0.996634 (0.98491) | 0.998203 / 0.997914 | 0.99612 | 0.99116 | 0.97535 |
| 12 | KDA+MoE | 0.999972 | 0.996412 (0.98443) | 0.998382 / 0.998137 | 0.99583 | 0.99086 | 0.97491 |
| 13 | KDA+MoE | 0.999982 | 0.995399 (0.97577) | 0.997540 / 0.997710 | 0.99487 | 0.98963 | 0.97194 |
| 14 | KDA+MoE | 0.999944 | 0.995043 (0.96583) | 0.997111 / 0.994040 | 0.99460 | 0.98929 | 0.97096 |
| 15 | DSA+MoE | 0.999924 | 0.992941 (0.94551) | 0.996078 / 0.989140 | 0.99272 | 0.98670 | 0.96074 |
| 16 | KDA+MoE | 0.999960 | 0.991646 (0.94569) | 0.996011 / 0.988196 | 0.99153 | 0.98514 | 0.95366 |
| 17 | KDA+MoE | 0.999963 | 0.986426 (0.94387) | 0.995457 / 0.979847 | 0.98721 | 0.98055 | 0.93841 |
| 18 | KDA+MoE | 0.999935 | 0.983102 (0.92282) | 0.993936 / 0.978192 | 0.98578 | 0.98007 | 0.93951 |
| 19 | DSA+MoE | 0.999760 | 0.966700 (0.84224) | 0.988705 / 0.962279 | 0.97365 | 0.96824 | 0.90531 |
| 20 | KDA+MoE | 0.999938 | 0.958340 (0.78668) | 0.986709 / 0.916892 | 0.96834 | 0.96444 | 0.90638 |
| 21 | KDA+MoE | 0.999364 | 0.938064 (0.76289) | 0.960606 / 0.808846 | 0.95533 | 0.95124 | 0.89054 |
| 22 | KDA+MoE | 0.999521 | 0.915535 (0.69016) | 0.803622 / 0.675390 | 0.94056 | 0.93399 | 0.87633 |
| 23 | DSA+MoE | 0.998858 | 0.856822 (0.55181) | 0.630910 / 0.564984 | 0.90682 | 0.90243 | 0.84630 |
| 24 | KDA+MoE | 0.999686 | 0.838809 (0.54787) | 0.626850 / 0.554957 | 0.88679 | 0.88775 | 0.83075 |
| 25 | KDA+MoE | 0.999896 | 0.832117 (0.55275) | 0.640174 / 0.554143 | 0.87752 | 0.88030 | 0.82563 |
| 26 | KDA+MoE | 0.999853 | 0.793667 (0.53310) | 0.624066 / 0.542373 | 0.84304 | 0.86582 | 0.81223 |
| 27 | DSA+MoE | 0.999761 | 0.750281 (0.52873) | 0.605923 / 0.525106 | 0.80480 | 0.83879 | 0.78511 |
| 28 | KDA+MoE | 0.999779 | 0.725192 (0.50726) | 0.573502 / 0.488403 | 0.78077 | 0.81471 | 0.76498 |
| 29 | KDA+MoE | 0.999758 | 0.690644 (0.47848) | 0.525927 / 0.467138 | 0.74966 | 0.78255 | 0.73597 |
| 30 | KDA+MoE | 0.999918 | 0.691752 (0.40267) | 0.547685 / 0.493671 | 0.75280 | 0.78935 | 0.74702 |
| 31 | DSA+MoE | 0.999667 | 0.706953 (0.49653) | 0.663447 / 0.605452 | 0.77122 | 0.80782 | 0.75275 |
| 32 | KDA+MoE | 0.999932 | 0.690303 (0.47571) | 0.628456 / 0.598886 | 0.75558 | 0.78858 | 0.73337 |
| 33 | KDA+MoE | 0.999928 | 0.683751 (0.44016) | 0.597613 / 0.588983 | 0.74806 | 0.77317 | 0.72684 |
| 34 | KDA+MoE | 0.999947 | 0.787148 (0.60851) | 0.718067 / 0.735707 | 0.83985 | 0.86279 | 0.83760 |
| 35 | DSA+MoE | 0.999695 | 0.787512 (0.62248) | 0.737277 / 0.693932 | 0.82943 | 0.85613 | 0.80904 |
| 36 | KDA+MoE | 0.999985 | 0.819342 (0.65460) | 0.756459 / 0.734931 | 0.85811 | 0.87863 | 0.83802 |
| 37 | KDA+MoE | 0.999995 | 0.813154 (0.64429) | 0.749034 / 0.728810 | 0.85267 | 0.87298 | 0.83078 |
| 38 | KDA+MoE | 1.000000 | 0.810930 (0.64278) | 0.736202 / 0.726874 | 0.85047 | 0.86976 | 0.82715 |
| 39 | DSA+MoE | 0.999908 | 0.803414 (0.65663) | 0.723772 / 0.730450 | 0.84422 | 0.86416 | 0.81092 |
| 40 | KDA+MoE | 0.999959 | 0.803990 (0.65258) | 0.730295 / 0.727282 | 0.84466 | 0.86426 | 0.81052 |
| 41 | KDA+MoE | 0.999939 | 0.809082 (0.68091) | 0.750723 / 0.736503 | 0.84911 | 0.86786 | 0.81382 |
| 42 | KDA+MoE | 0.999887 | 0.819191 (0.71312) | 0.778319 / 0.757278 | 0.85771 | 0.87601 | 0.82414 |
| 43 | DSA+MoE | 0.999742 | 0.812367 (0.71120) | 0.784757 / 0.758919 | 0.85205 | 0.87142 | 0.82374 |
| 44 | KDA+MoE | 0.999894 | 0.858327 (0.75103) | 0.856999 / 0.859909 | 0.88616 | 0.92297 | 0.90608 |

Logits of the 128 rows (TT chain + TT head vs the host head `lm_head(norm(mean))` in fp32 on the dump's
layer-44 `x_out`): PCC 0.783, top-1 agreement 0.234, top-5 overlap 0.425. The head alone (TT head on the
dump's own layer-44 streams vs the host head): PCC 0.999967, top-1 0.969, top-5 0.978. Decode steps on the
TT state: step 0 (token 17324 at 128) logits PCC 0.762, TT argmax 99484 vs host 44210 (99484 is the host's
3rd); step 1 (21039 at 129) PCC 0.731, TT argmax 102647 vs host 99484 (102647 the host's 4th). The device
argmax equals the host argmax of the device logits on both steps. The test's layer-44 > 0.99 / top-1 >= 0.9
gates FAIL on this prompt (0.858 / 0.234).

Reading: the per-layer TT error inside the chain is the standalone one (every layer >= 0.9989 on its exact
input, the chain's first layers reproduce the standalone numbers exactly), but the stack amplifies small
errors on this random-token prompt: the HF bf16 chain itself with a 1 % relative error per layer (per-layer
PCC 0.99995, well below the TT per-layer error of 1-5 %) decays to 0.907 at layer 23, 0.75 at layers 29-33
and 0.886 at layer 44 (2 % / 4 %: 0.923 / 0.906 at layer 44) -- the same profile as the TT chain (0.857 /
0.69 / 0.858), including the recovery after layer 33. The coordinator's independent probe (the HF chain
perturbed once by 5e-4 relative rms at the layer-0 input: 0.93 at layer 24, 0.83 at 33, 0.93 at 44;
`L45_S128_perturb1e-3`, `tests/test_chain_sensitivity.py`) says the same. The random prompt is therefore
not a usable end-to-end accuracy metric for this model; the real-prompt generation (2.) is.

### 2. Greedy generation, real prompt (`fullmodel_generate.log`, `fullmodel_hf_generate.log`)

Prompt "The capital of France is Paris. The theory of general relativity" (13 tokens), 16 greedy tokens.
HF side: `hf_layer_reference.py --generate 16` (bf16 head path, layers streamed per token, 300-870 s per
pass on the loaded host). See the status table below.

GENERATION_PLACEHOLDER

### 3. Timings, memory, cache (`fullmodel_dump_chain.log`, host load 35-70 on 64 cores from the other sessions)

| item | value |
|---|---|
| embedding + head build | 1.810 GiB resident (table 1.27 GB bf16 + head 0.65 GB bfp8), 30.02 GiB free of 31.83 |
| MoE layer build, expert cache miss | 27-50 s mean of 3 builds incl. the miss (the miss itself: load_layer_raw 6 s + bfp8 conversion ~50 s + tensorbin dump 12 s = ~85 s under load) |
| MoE layer build, expert cache hit | 5-10 s (layer 44: 6.9 / 8.5 s; layer 3 DSA: 5.0 / 7.4 s; quiet host 3.4-4.1 s) |
| dense KDA layer build (no cache) | 1.3-4.5 s |
| layer run, prefill S=128 (warm) | 0.04-0.11 s (KDA+MoE), 0.58 s (layer 3 DSA compile pass excluded: first-pass compiles 31 s layer 0, 45 s layer 3) |
| layer run, decode B=1 | 0.02-0.04 s (compile 15-17 s once per layer kind) |
| release | 0.2-0.6 s (37 buffers of a MoE layer, 30 of a DSA layer, 32 of a dense layer) |
| pass over 45 layers | prefill pass 3688 s with the cache being built (40 misses); decode passes 312 / 313 s (all hits); generation passes 450-460 s under the heavier host load |
| DRAM peak after a build | 9.65 GiB (KDA+MoE), 9.32 GiB (DSA+MoE), 2.59 GiB (dense); back to 1.84-1.97 GiB after every release (= embedding + head + the 45 layers' state) |
| tensorbin cache | 313 GiB (335.2 GB) under `/home/ttuser/ashai/glm53/_cache/fullmodel`: 42 x (5.13 GB w13 + 2.57 GB w2) + the KDA prefill tensorbins |
