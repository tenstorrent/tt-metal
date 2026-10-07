---
name: tt-model-reference
description: Build the CPU oracle for a new LLM (HF modeling code + config, layer-by-layer CPU reference) and dump per-layer activations, router decisions and decode state in a reusable format. Use in track 0 before any device module is written, and when a device test needs a dump that does not exist.
---

# Reference model and dumps (track 0)

Last validated against commit: 8fe1f078f57 (branch ssinghal/mimo-ref, MiMo-V2.6-Flash), ONLY for: sec 0 (local-weights check), sec 1 step 3 on a real fp8/fp4 checkpoint
(rank-blocked fp8 qkv layout proof, fp4 nibble/e8m0 check by routing selectivity), sec 2 (independent fp32 oracle == HF layer, PCC 0.9999999, same router sets; short chain),
sec 3 prefill dump format + `single_layer.py` CLI + noise floor (S=128 b16 and S=512 b4, L0-5). NOT validated: decode-step dumps against a device, S>=2048 noise floor, any model other than MiMo.

Policy (user decision): the CPU reference does NOT run the complete model. Prefill-only and decode-only checks are enough.
1. Per-layer teacher-forced checks: the oracle runs ONE layer on the real hidden state captured from the device (or from a short CPU prefix) and
   compares layer output, router decisions, KV writes and decode-step outputs per layer. Error does not compound in this mode.
2. Short CPU CHAIN over a layer prefix (first 4-8 layers, or one of each layer type: dense, sliding-window MoE, global MoE) to catch compounding drift.
   Teacher-forced per-layer checks alone do not catch it (DSV4: per-layer 0.996-0.999 but a long chain was much lower, see L-chain).
3. Prefill and decode are validated as separate phases on device (prefill: layer outputs + KV contents vs reference; decode: one step from a
   reference-prefilled KV state vs the reference step).
4. End-to-end acceptance without a CPU full-model run: see `tt-moe-correctness` section I (GSM8K, determinism, first-token, hosted-reference prompts).
Residual risk, stated plainly: compounding error beyond the CPU prefix is only caught by the GSM8K/behavioural gates, so those gates are REQUIRED.
Host RAM: 566 GB (527 GB available), 64 cores. Stream the reference layer by layer and dequantize on the fly (MiMo: ~13 GB bf16 of experts per layer;
the full model ~618 GB bf16 does not fit). Never load the full model into RAM.

Pass criterion (log it): (a) the single-layer oracle reproduces the HF layer output on a real hidden state (PCC > 0.999, same router set) for each
layer type, and the short chain reproduces HF logits/greedy tokens on a short real prompt; (b) per-layer dumps (inputs, outputs, routing, KV writes,
decode step) for at least one real-text prompt set exist, with command, commit, S, B and prompt sha256 in LOG.md; (c) bf16 and fp32 runs of the
same layers exist so the reference's own noise floor is known.

## 0. Local weights first
Before anything: find the checkpoint on disk (`ls /mnt/tt-data`, HF caches, `bringup_study/<model>/`, the index json's shard names). Study folders often hold only config/index/modeling code
(MiMo: `sgl.py` there was a 404 stub, there was NO checkpoint). If no complete copy exists, STOP and ask the coordinator/owner before downloading (170-600 GB); do not range-fetch
weights from the network on your own. Record the checkpoint path in LOG.md (env var `<MODEL>_CKPT`).

## 1. Get the model
1. Read, do not guess: the HF repo `config.json`, `modeling_*.py`, the quantisation config, the tokenizer chat template, the
   safetensors index. Save a local copy path in LOG.md (weights live on NFS; never copy 300+ GB to a worktree).
2. List per layer type: attention kind (sliding window size / global / GQA heads, head dims, sinks, rope params per type),
   router (score fn, bias use, normalisation, scale factor, top-k, group limits, hash-routed layers), expert FFN (activation,
   clamp), shared expert yes/no, norms, dense-vs-MoE layer pattern, MTP/drafter modules. Write this table into LOG.md as `decision`
   entries (INFERRED until checked by (a)). DSV4 lesson: details read from a summary page were wrong several times; read the code.
3. Checkpoint format: fp8/mxfp4 blocks need dequant; check scale layout. A loader bug (fp8 weight read without its scale) cost a day.
   Verify fake-quant vs unquantised layer PCC on the CPU (DSV4: > 0.998 for most layers, 0.97-0.99 for a deep one) before blaming the device.

## 2. CPU reference (slow, exact, no device)
* Pattern: `reference/ref_layer.py` (one layer: attention + MoE + norms), `ref_kernels.py` (the model's own special ops in torch),
  `ref_chain.py` (a layer prefix; the DSV4 script ran all layers, you need not). Write it from the HF code, not from the device code. Run in bf16 (what the device approximates) and in fp32.
* Router: run it in fp32 and store the decisions; the device router is the main source of divergence (see `tt-moe-correctness`).
* Check against HF: logits PCC > 0.999 and identical greedy tokens on 3 prompts. Then freeze it.
* Runtime reference: S=128, 16 users ~ 1 min/layer on CPU; plan CPU hours for 2k/4k/8k dumps and run them in the background (nice).

* Teacher-forced single layer for ANY layer (no CPU chain beyond the short prefix): `python -m <pkg>.reference.single_layer --layer N --h-in <h.pt or layer_*.pt> --out out.pt [--dtype bf16|fp32] [--dec-in dec.pt]`
  runs the oracle on a hidden state handed in (e.g. captured from the device) and writes the same prefill/state/decode fields as the chain dump (MiMo: `reference/single_layer.py`, tested on L20, L46, L47 with a random real-scale input).
* Per-layer device gate beyond the CPU prefix: run the oracle in fp32 AND bf16 on the DEVICE h_in of every layer and compare device attn_out/ffn_out/h_out/router sets to both, next to the bf16-vs-fp32 floor (MiMo: `reference/probes/device_vs_oracle.py`, ~40 s/layer). h_out PCC saturates (residual outliers): judge by rel L2 of attn_out/ffn_out versus the floor ratio, and isolate the device router by running the fp32 router on the DEVICE ffn_in.
* Write the oracle independently from the spec and prove it equal to the HF layer on real tokens (MiMo: PCC 0.9999999 fp32, identical expert sets) BEFORE trusting it; PCC saturates on residual-outlier models, also log relative L2 error.
* Check quantised-layout assumptions with the real HF forward, not statistics alone: wrong qkv layout gave layer outputs 10-90x the residual (MiMo), a functional test (routed tokens must excite their expert more) fixed the fp4 nibble order.
* The checkpoint's modeling file may target an older transformers than installed (MiMo: 5.3 vs 5.12: mask helper kwargs changed); build masks with the installed API and compare to the oracle mask.

## 3. Dumps (format the device tests load)
Real text prompts only (GSM8K chat-template prompts + one long book-text prompt); random tokens give meaningless routing
(near-ties everywhere, 88-100% router flips on a 9-token random prompt, wrong skew for dispatch tests).
Per layer `layer_{L}.pt` (torch.save dict), files under `/mnt/tt-data/ssinghal/<model>-prefill-s{S}[b{B}]/`:
* `prefill`: `h_in`, `h_out` (layer in/out [B,S,...] bf16), `attn_in`, `attn_out`, `ffn_in`, `ffn_out`, `routing_idx` [B*S, topk] (+ routing weights);
* `state`: decode state after prefill (KV window ring with position p at slot p % window, global KV, any compressor state);
* decode-chain dump (`ref_chain.py`): `dec_in`, `dec_out`, `state`; `final.pt`: `prefill_logits` (last prompt token), `prefill_argmax`,
  decode-step logits/argmax; `tokens.pt`: prompt and decode tokens.
* End-to-end generation (`ref_e2e_gen.py`): `meta.pt` (prompts, gold), `results.pt` (greedy stream, top-50 values/ids, lse), `logits_tf.pt`
  (teacher-forced logits), `state/`, `kv_final/` for GSM8K accuracy and per-layer KV error.
* Also store per layer the bf16-vs-fp32 difference summary (`reference/chain_analyze.py --ref ... --ref32 ...`): this is the noise floor.
Naming: keep S in the directory name; tests read it from `DSV41_PREFILL_DIR` / `DSV41_S` style env vars (see
`tests/test_e2e_prefill_decode.py`, `tests/test_chain_steps_device.py`, `tests/test_decode_steps.py`). Mirror that: one env var for the dump dir per test.

## 4. Dump shapes to produce (minimum)
S=128 b16 (every layer type, streamed layer by layer, fast), S=512 b4, one S=2048 b1 and one S=4096..8192 b1 (layers that use sparse/global
attention and long-context paths), plus the 128-user GSM8K set for accuracy. Tile a 1-user dump to the batch only where the test says so.
Include layer types that appear rarely (MiMo: the 9 global layers) in every dump; a dump that skips them hides bugs.

## 5. Record
LOG entry per dump: command, commit, host/CPU time, dir, S, B, prompt sha256 (`PROMPTS_SHA256.txt` in the DSV4 package), noise floor numbers.
Do not regenerate dumps silently: a changed dump invalidates every PCC logged against it (supersede, state the dir name change).

## Mistakes we made
* Chained 9-token random-prompt reference gave chain PCC 0.74-0.87 and looked like a device bug; it was reference noise + router flips on near-ties (real 128-token prompt: 0.98).
* Teacher-mode harness wrote the Engram/aux state twice at one layer: harness bug looked like a device fault. Diff harness against the model path first.
* Synthetic random KV caches for attention tests made top-k selection near-uniform: PCC 0.97 was the noise floor, not an error. Use real activations.
* Reference's own fp4/bf16 simulation makes two reference variants differ by 3-12%: measure the floor before setting a threshold.

Escalate: if (a) fails and the HF code and checkpoint disagree (ask the human which is the truth), or a dump needs > 6 CPU hours.
