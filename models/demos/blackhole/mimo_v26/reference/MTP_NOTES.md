# MiMo-V2.6-Flash MTP notes (track 0, no implementation)

Legend: [Q] quoted from a local source, [M] measured from checkpoint headers/index, [I] inferred (not stated by any source we have).
Sources: `/mnt/tt-data/MiMo-V2.6-Flash-RL/MiMo_V2_6_technical_report.pdf` (44 pages; "p." = PDF page; text extracted with a throw-away stdlib PDF parser because no
PDF tools are installed, so wording below is verbatim only where in quotes), `README.md`, `config.json`, `modeling_mimo_v2.py`, `model.safetensors.index.json`, `model_mtp.safetensors`,
`dflash/` (code in `/mnt/tt-data/ssinghal/bringup_study/mimo/dflash/`). `bringup_study/mimo/sgl.py` is a 404 stub: there is NO sglang source on disk.

## What the report says
* p.4 sec 2.1 [Q]: "MiMo-V2.6 also integrates an SWA and dense FFN based MTP ... to improve model performance during pre-training." Hybrid blocks: N consecutive SWA blocks then a GA block;
  "The only exception is the very first Transformer block, which uses global attention with a dense FFN"; window W = 128; SWA and GA blocks both use sparse MoE FFN without shared experts.
  (Consistent with config: layer 0 = global + dense, 0,5,11,... global.)
* p.5 sec 2.4 [Q]: "Speculative decoding in MiMo-V2.6 uses a multi-token prediction (MTP) module following the block diffusion design of DFlash (Chen et al., 2026). The drafter comprises 5
  Transformer layers with dense feed-forward networks. Conditioned on backbone hidden features and a clean anchor token, it predicts 7 subsequent tokens in a single forward pass ... All draft layers use
  sliding-window attention (SWA) with grouped queries ... Tokens attend bidirectionally within their draft block and to at most 1,024 backbone context positions preceding the anchor." p.6 Table 1 [Q]:
  Speculative Decoder 5 layers (5 SWA / 0 GA), hidden 4096 (Flash), SWA heads 64/8, window 1024, head dims 128/128.
  So the report's "MTP module" in sec 2.4 IS the 5-layer DFlash drafter (= `dflash/` in the repo), NOT the 3 in-checkpoint MTP layers.
* p.32-33 sec "Draft Model Acceleration" [Q]: "RL rollouts use block-6 DFlash for speculative decoding by default, replacing the multi-token prediction (MTP-3) configuration inherited from SFT. ...
  Under this default configuration, average accepted length is 31.3% higher than with the MTP configuration." Block-6 improves throughput ~6% over block-8; FP8 draft ~10.3% higher per-node throughput on long context.
* The report gives NO details of the MTP-3 layers: nothing on eh_proj concat order, chaining, window, shared embedding/head, acceptance numbers for MTP-3, nor a GSM8K score for MiMo-V2.6-Flash
  (its benchmark tables are agentic: SWE-bench, Terminal-style, MiMo Code Bench, etc.). README.md also has no GSM8K figure. => no published GSM8K number exists locally; the accuracy gate threshold
  cannot be derived from the paper.

## What the checkpoint has [M]
`config.json`: `num_nextn_predict_layers: 3`. Tensors `model.mtp.layers.{0,1,2}.*` in `model_mtp.safetensors`, per layer:
`eh_proj.weight` bf16 [4096, 8192]; `enorm.weight`, `hnorm.weight`, `input_layernorm.weight`, `pre_mlp_layernorm.weight`, `final_layernorm.weight` bf16 [4096];
`self_attn.qkv_proj.weight` fp8 [14848, 4096] + `weight_scale_inv` [116, 32] (SWA dims: 64 q heads x192, 8 kv x192, 8 v x128; same per-TP4-rank layout as the main model, see LOG qkv decision, [I] but same shapes);
`self_attn.o_proj.weight` bf16 [4096, 8192]; `self_attn.attention_sink_bias` bf16 [64] (so SWA with sinks);
`mlp.{gate,up,down}_proj.weight` fp8 dense 16384 (+ scale_inv). No embedding / lm_head tensors under `model.mtp.*` [M: only `model.mtp.layers.*` keys exist].
The HF modeling file ignores them (`_keys_to_ignore_on_load_unexpected` has `model\.mtp\..*`, modeling_mimo_v2.py:1694): no reference code.
README [Q]: sglang launch uses `--speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 --enable-multi-layer-eagle`.

## Inferred structure (DeepSeek-V3-style MTP; NOT verified, no code on disk)  [I]
* Layer k input = eh_proj( concat[ enorm(embed(token_{t+k+1})), hnorm(h_prev) ] ) with eh_proj [4096, 8192] mapping the 8192-wide concat to 4096. Concat ORDER (embedding first vs hidden first) is
  the DeepSeek convention (embedding first) but unconfirmed; it can be settled empirically by acceptance on real text (two candidates, pick the one with acceptance > chance).
* Then input_layernorm -> SWA attention (window 128, sinks, V*0.707, partial RoPE, rope theta 1e4 like main SWA layers [I: dims identical]) -> o_proj -> residual -> pre_mlp_layernorm ->
  dense SwiGLU FFN (16384) -> residual -> final_layernorm -> shared lm_head [I]. h_prev for layer 0 = main model's last-layer hidden (pre final norm or post? unconfirmed), for layer k>0 = layer k-1's output [I]
  ("--enable-multi-layer-eagle" with 3 steps and topk 1 suggests one distinct MTP layer per draft step, chained).
* Embedding and lm_head shared with the main model (no tensors of their own) [I].
* Verify count: 3 drafts + 1 root = 4 tokens per verify step (README num-draft-tokens 4) [Q/I].
* KV: each MTP layer would need its own SWA KV ring (window 128) [I].

## Recommendation
Treat MTP-3 as "needs an SGLang/vLLM source or an experiment": no device work should assume the above. The DFlash drafter has real reference code (`dflash/dflash.py`) and is the path the
report says is used in RL and is 31.3% better in accepted length than MTP-3 (p.32).
