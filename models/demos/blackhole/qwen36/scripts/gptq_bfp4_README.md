# GPTQ-BFP4 for Qwen3.8-27B (MLP + GDN in/out as bfloat4_b, attention + LM head stay BFP8)

Plan item #4 (GPTQ-BFP4). Today the dense MLP and GDN projections are BFP8. Plain round-to-nearest (RTN) BFP4 costs
Qwen3.8 several points of teacher-forced top-1, because 3.8 is about 7x more precision-sensitive than 3.6
(QWEN38_HANDOFF.md section 3). GPTQ picks the BFP4 values to minimise each layer's output error on calibration
activations, not the weight error, so most of that loss should come back. Expected gain: -2.0 to -2.9 ms per B=8
step, from about 45% fewer weight bytes on the MLP and GDN matmuls. The accuracy gate below decides if it ships.

## How it fits ttnn (and how it differs from tt-lab)

* ttnn `bfloat4_b` shares one exponent per 16 elements along the **last** dim of the `[K, N]` weight, that is,
  16 **output** rows of one input column ("N-grouped"). ttnn recomputes that exponent from the bf16 values at every
  conversion (`blockfloat_common.cpp`).
* tt-lab's `requant.cpp:691-960` groups 16 **input** columns per exponent and writes the packed bits itself. It fixes
  each section's exponent at the section's first GPTQ column (max exponent, or max-1 if the H-diagonal-weighted
  error is lower), and its act-order permutes whole 16-column sections.
* Our adaptation (`utils/gptq_bfp4.py`):
  * Each 16-row block lives inside one GPTQ column. The exponent is chosen when that column is quantized, from its
    error-updated values, so act-order can permute single columns.
  * The pipeline writes **bf16 grid values**, not packed bits. Each value is m*2^(E-129), and each block keeps one
    element with m>=4. ttnn therefore re-derives the same exponent, and its rounding is a no-op. The conversion is a
    fixed point, checked on the host (`verify`) and on device (`test_gptq_bfp4_tp.py`).
  * tt-lab's "max exponent - 1" choice is still a fixed point. Use it with `--exp-search` (tt-lab default ON; ours
    OFF until the emulator says it helps).
  * Block membership follows the **device layout**. The GDN fused qkvzab weight puts per-device `[a_d | b_d]` rows
    into shared blocks (TP8: `[a6 b6 pad4]`), so a checkpoint is calibrated for one TP degree. The marker records
    it, and the model refuses a mismatch under the preset.
* MTP drafter weights are not calibrated. Under the preset its MLP loads as RTN BFP4, because it shares
  `Qwen36MLP`. Spec acceptance is part of the gate.

## Flags (all default OFF; behaviour without them is unchanged)

| Env | Effect |
|---|---|
| `QWEN36_GPTQ_BFP4=1` | Preset for a GPTQ checkpoint. Sets the defaults of the four flags below to BFP4. Requires `gptq_bfp4.json` in `HF_MODEL` with the running TP. |
| `QWEN36_MLP_GATEUP_BFP8` | Existing flag. Default 1; 0 under the preset. |
| `QWEN36_BFP4_MLP_DOWN` | Existing flag. Default 0; 1 under the preset. |
| `QWEN36_BFP4_GDN_IN` | Existing flag (BFP4 + LoFi in decode). Default 0; 1 under the preset. |
| `QWEN36_BFP4_GDN_OUT` | **New.** GDN out_proj (decode row-parallel and prefill colpar copies) in BFP4, HiFi2 unchanged. Default 0; 1 under the preset. |

An explicitly set flag always wins over the preset. If `HF_MODEL` contains `gptq_bfp4.json`, the tensor cache moves to
`$TT_CACHE_PATH/tensor_cache_bfp8_mesh1x8_gptq<12 hex>`, a NEW directory. `as_tensor` cache names carry only the
tensor name and dtype, so with the same cache dir the model would silently load the RTN tensors.

**Name the output dir `.../Qwen3.8-27B`** (for example `/data/models/gptq-bfp4-v1/Qwen3.8-27B`). The demo's
accuracy `.refpt`, perf targets and `env38.sh` key on `basename(HF_MODEL)`.

## 1. Offline pipeline (CPU or CUDA box, no TT device)

Environment: torch >= 2.4, safetensors, and the transformers version pinned by the tt-metal python env (it must have
`transformers.models.qwen3_5`; check with `pip freeze | grep transformers` in the TT container). `datasets` is needed
only for `--hf-dataset`. Optional: `flash-linear-attention` and `causal-conv1d` make the GDN forward on CUDA much
faster. Run from the tt-metal root, so that `models.` imports resolve.

```bash
cd $TT_METAL_HOME; export PYTHONPATH=$PWD
SRC=/data/models/Qwen3.8-27B                 # bf16 HF checkpoint (the one being served)
OUT=/data/models/gptq-bfp4-v1/Qwen3.8-27B    # NEW dir, same basename
W=/data/gptq-work; mkdir -p $W

# (a) calibration set: 128 x 2048 tokens. Best source is the model's own chat-formatted reasoning traces on
#     prompts DISJOINT from AIME/GPQA/IFBench (jsonl {"messages": [...]}), plus generic text.
python models/demos/blackhole/qwen36/scripts/gptq_bfp4.py calib-data --model $SRC --out $W/calib.pt \
  --nsamples 128 --seqlen 2048 --jsonl $W/self_traces.jsonl --hf-dataset allenai/c4:en:train:text --seed 0
# (b) held-out set for the KL gate (different seed, skip half the docs)
python models/demos/blackhole/qwen36/scripts/gptq_bfp4.py calib-data --model $SRC --out $W/heldout.pt \
  --nsamples 16 --seqlen 2048 --jsonl $W/self_traces_heldout.jsonl --hf-dataset allenai/c4:en:validation:text --seed 1 --skip-frac 0.5

# (c) smoke test first: 2 layers (1 GDN + 1 attention), ~5 min on a GPU
python models/demos/blackhole/qwen36/scripts/gptq_bfp4.py quantize --model $SRC --calib $W/calib.pt \
  --out $W/smoke/Qwen3.8-27B --tp 8 --layers 0,3
# (d) full run (resumable: rerun the same command after a crash; finished layers load from $OUT/.gptq_parts)
python models/demos/blackhole/qwen36/scripts/gptq_bfp4.py quantize --model $SRC --calib $W/calib.pt --out $OUT --tp 8 \
  2>&1 | tee $W/quantize.log
# (e) bit-level check: GPTQ tensors are BFP4 fixed points in the TP8 layout, every other tensor is untouched
python models/demos/blackhole/qwen36/scripts/gptq_bfp4.py verify --model $SRC --out $OUT
# (f) emulator gate (exit 1 = do not proceed to device/evals)
python models/demos/blackhole/qwen36/scripts/gptq_bfp4.py kl --model $SRC --gptq $OUT --eval $W/heldout.pt \
  --nsamples 16 --json $W/kl.json
```

Per layer, `quantize.log` prints the relative output error, GPTQ vs RTN; expect GPTQ to be 2-5x lower. The marker
`$OUT/gptq_bfp4.json` stores every per-target stat. Variants: `--targets mlp_gateup,mlp_down` (MLP only),
`--exp-search`, `--damp 0.01..0.1`.

### Resource estimates (27B, 128 x 2048 calibration tokens)

| | GPU box (1x A100/H100 80 GB) | CPU-only box |
|---|---|---|
| RAM | 96 GB min, 128 GB recommended (54 GB bf16 model + activations + export shard) | same |
| GPU memory | ~10-15 GB peak (one layer, a 17408^2 fp32 Hessian, 2x2.7 GB activations) | n/a |
| Disk | 54 GB output + ~47 GB `.gptq_parts` during the run (deleted at the end unless `--keep-parts`) | same |
| quantize time | ~1.5-3 h (5 calibration passes per GDN layer, 3 per attention layer; ~10 s of GPTQ per layer) | ~15-30 h on 64+ cores. Use `--nsamples 64 --seqlen 1024` (~4x less) and `--act-device cpu` |
| kl time | ~30-60 min (4 streams x 16 x 2048 tokens) | several hours. Use `--nsamples 4` |
| verify | ~10-20 min (re-reads both checkpoints) | same |

Add `--act-device cpu` if GPU memory is short. The TT host also needs about 30-57 GB more disk for the new tensor
cache on the first device load, which takes about 150 s.

### Emulator gate (`kl`)

It compares, against BF16 logits on held-out text and with weights stored exactly as ttnn would:

* `deployed`: today's defaults, all BFP8.
* `rtn`: the preset dtypes with plain rounding.
* `gptq`: this checkpoint.

PASS needs both of these (the thresholds are flags):
* gptq top-1 agreement is at most 1.0 pt below deployed (`--max-top1-drop`);
* gptq mean KL is at most 0.6x rtn mean KL (`--max-kl-vs-rtn`).

For reference, the handoff emulator measured these teacher-forced top-1 scores: all-BFP8 99.2; RTN down BFP4 95.9;
RTN GDN-out BFP4 96.9; RTN gate/up BFP4 91.8. If GPTQ does not reach about 98.5 on the equivalent device test
(below), it will not pass the task evals.

## 2. On-device validation (Blackhole Galaxy, inside the tt-metal container, this branch built)

Use a test galaxy or a spare 1x8 submesh, never a live serving galaxy without sign-off.
`source qwen38_handoff/scripts/env38.sh`-style env, with `MESH_DEVICE=P150x8` and `TT_CACHE_PATH` set.

```bash
T=models/demos/blackhole/qwen36
# 0. host tests (no device; ~1 min)
pytest $T/tests/test_gptq_bfp4_host.py $T/tests/test_gptq_bfp4_flags_host.py -q
#    expect: all pass (the end-to-end tiny-model and layer-loop tests skip only if transformers lacks qwen3_5)

# 1. device-layout proof: GPTQ grid weights load bit-exactly as BFP4 through gdn/tp.py + mlp.py at TP8 (~3 min)
MESH_DEVICE=P150x8 HF_MODEL=$SRC pytest $T/tests/test_gptq_bfp4_tp.py -v
#    expect: PASS. A failure on a key names the weight whose device grouping disagrees with utils/gptq_bfp4.py

# 2. teacher-forced accuracy, three arms (same TT_CACHE_PATH is safe: the GPTQ arm gets its own cache dir)
HF_MODEL=$SRC pytest $T/demo/text_demo.py -k accuracy_512                                   # A: deployed (BFP8)
HF_MODEL=$SRC QWEN36_MLP_GATEUP_BFP8=0 QWEN36_BFP4_MLP_DOWN=1 QWEN36_BFP4_GDN_IN=1 QWEN36_BFP4_GDN_OUT=1 \
  pytest $T/demo/text_demo.py -k accuracy_512                                               # B: RTN BFP4 control
HF_MODEL=$OUT QWEN36_GPTQ_BFP4=1 pytest $T/demo/text_demo.py -k accuracy_512                # C: GPTQ BFP4
#    read "Top-1 token accuracy" in each log. Expect C >= A - 1.0 pt and C well above B (B ~93-95).
#    The log of C must show "GPTQ-BFP4 checkpoint: ... cache suffix _gptq..." and the four "BFP4" weight lines.
#    Repeat with -k accuracy_1536 for A and C.

# 3. decode perf, A vs C (the metric: B=8 spec step ms and t/s/user)
HF_MODEL=$SRC QWEN36_SPEC_TIMING=1 pytest $T/demo/text_demo.py -k "spec_128_b8"
HF_MODEL=$OUT QWEN36_GPTQ_BFP4=1 QWEN36_SPEC_TIMING=1 pytest $T/demo/text_demo.py -k "spec_128_b8"
HF_MODEL=$SRC pytest $T/demo/text_demo.py -k "batched_128_b8 or (traced_128 and not traced_128k)"
HF_MODEL=$OUT QWEN36_GPTQ_BFP4=1 pytest $T/demo/text_demo.py -k "batched_128_b8 or (traced_128 and not traced_128k)"
#    expect: spec step at B=8 2.0-2.9 ms lower; plain decode t/s/u +8-12%; spec accepted tokens/step within 0.05 of A
#    (MTP's own MLP is RTN BFP4 under the preset).

# 4. prefill path with the BFP4 colpar out-proj + packed gate_up AGMM, and long context
HF_MODEL=$OUT QWEN36_GPTQ_BFP4=1 pytest $T/demo/text_demo.py -k "traced_4k or spec_4k_b8"
#    expect: pass, TTFT within noise of A, no hang. Watch board current in prefill (tt-lab saw chip drops at 140-190 A with BFP4).

# 5. spec correctness under the preset
HF_MODEL=$OUT QWEN36_GPTQ_BFP4=1 pytest $T/tests/test_spec_batched.py $T/tests/test_spec_determinism.py -q
```

## 3. Task-eval gate (vLLM server with the GPTQ checkpoint on a test galaxy, DP4xTP8)

Bring up the server as for production, with `HF_MODEL=$OUT QWEN36_GPTQ_BFP4=1` (plus a separate `TT_CACHE_PATH`, or
the same one: the cache dir differs). Run the same commands against a baseline server with `$SRC` and no flags.
Score both with the same seeds. Gate: each task within 1 SE of baseline. score.py prints a question-clustered bootstrap
95% CI (SE ~ CI half-width / 1.96); compare.py prints a paired McNemar test.

```bash
E=models/demos/blackhole/qwen36/qwen38_handoff/eval
for arm in base gptq; do   # point --base-url at the matching server
  python $E/task_eval.py --task aime --out runs/$arm/aime.jsonl --repeats 8 --concurrency 32          # AIME 2026 x 8 repeats
  python $E/task_eval.py --task gpqa --out runs/$arm/gpqa.jsonl --repeats 2 --concurrency 32          # GPQA-Diamond, 198 q x 2
done
python $E/score.py runs/base/aime.r*.jsonl; python $E/score.py runs/gptq/aime.r*.jsonl
python $E/compare.py runs/base/gpqa.r0.jsonl runs/gptq/gpqa.r0.jsonl
# IFBench (allenai/IFBench, 300 prompts): generate with the served model, then score with its official checker
git clone https://github.com/allenai/IFBench && cd IFBench && pip install -r requirements.txt
python - <<'EOF'   # writes responses_<arm>.jsonl; set BASE/ARM per server
import json, os, openai
c = openai.OpenAI(base_url=os.environ["BASE"], api_key="x"); m = c.models.list().data[0].id
with open(f"responses_{os.environ['ARM']}.jsonl", "w") as f:
    for r in map(json.loads, open("data/IFBench_test.jsonl")):
        out = c.chat.completions.create(model=m, messages=[{"role": "user", "content": r["prompt"]}],
                                        temperature=1.0, top_p=0.95, max_tokens=32768, extra_body={"top_k": 20, "seed": 0})
        f.write(json.dumps({"prompt": r["prompt"], "response": out.choices[0].message.content}) + "\n")
EOF
python -m run_eval --input_data=data/IFBench_test.jsonl --input_response_data=responses_gptq.jsonl --output_dir=out_gptq
```

Ship only if the emulator gate, device accuracy (step 2) and all three tasks pass. Roll back by unsetting
`QWEN36_GPTQ_BFP4` and pointing `HF_MODEL` back at `$SRC`.
