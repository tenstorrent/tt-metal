# Functional-decoder work log

## Source discovery

- Loaded the cached Hugging Face config for `mistralai/Mistral-Small-24B-Instruct-2501`: 40 layers, hidden size 5120, 32 query heads, 8 KV heads, head dimension 128, intermediate size 32768, RMS epsilon `1e-5`, SiLU, RoPE theta `100000000.0`, and advertised context 32768.
- Ran `.agents/skills/forge-functional-decoder-from-ir/scripts/classify_graphs.sh` on the requested bundle. Compiler/runtime graph pairs g0/g2 classified as prefill (`fill_cache=2560`); g1/g3 classified as decode (`paged_update_cache=80`, decode SDPA=40). g2/g3 additionally contain logits, so no-logit compiler graphs g0/g1 were selected for the decoder layer.
- Ran `scripts/ir_to_emit.sh` on g0 and g1 with output under `/tmp/mistralai_mistral_small_24b_instruct_2501/`. The generated prefill and decode files contained 38,539 and 11,159 lines respectively.
- Read the raw MLIR alongside the flat emit. Segmented representative middle layer 20 by repeating RMS-norm boundaries. Confirmed source mesh `1x4`, TP degree 4, actual emitted batch 32, prefill sequence 18, decode sequence 1, and cache length 128.

## Translation decisions

- Collapsed TP4 local projections and post-projection all-reduces to dense single-device math over full, unsharded HF weights. No layout glue or collective remains in the runtime path.
- Preserved emitted Q/K/V constant-evaluation order: reverse local V/K/Q operands, transpose them, and concatenate Q then K then V. The dense loader performs the equivalent operation on full HF tensors.
- Preserved the two RMS norms, GQA head split, Q/K RoPE, cache mutation, SDPA scale, output residual, SwiGLU MLP, and final residual.
- Corrected an initial assumption that attention width equaled the 5120 residual width. The raw IR's local Q/O width of 1024 under TP4 proves a dense attention width of 4096; the final implementation and synthetic weights use Q `[4096,5120]` and O `[5120,4096]`.
- Normalized the source BF8/compiler-specific functional graph to the required correctness baseline: BF16, TILE layout, DRAM. Retained only a small L1 head layout required by decode head operations.

## Commands and gates

Static import/compile and runtime-fallback audit:

```text
python -m py_compile \
  models/autoports/mistralai_mistral_small_24b_instruct_2501/tt/functional_decoder.py \
  models/autoports/mistralai_mistral_small_24b_instruct_2501/tests/test_functional_decoder.py
pytest -q models/autoports/mistralai_mistral_small_24b_instruct_2501/tests/test_functional_decoder.py::test_runtime_forwards_have_no_host_fallback
```

Hardware selection followed the TT device-use contract. Four local Blackhole p300c UMD chips were visible. Exposing only chip 0 produced the expected incomplete P300 pair topology; opening pair 0/1 then reported a fixed-address sysmem mapping conflict consistent with another process. No external process was killed or reset. The complete free pair 2/3 passed a `1x1` mesh smoke, so all model tests were serialized with `TT_VISIBLE_DEVICES=2,3` and closed their device fixtures normally.

Synthetic and real-weight gates:

```text
TT_VISIBLE_DEVICES=2,3 pytest -q -s \
  models/autoports/mistralai_mistral_small_24b_instruct_2501/tests/test_functional_decoder.py

MISTRAL_SMALL_24B_REAL_WEIGHT_DIR=/home/mvasiljevic/hf-cache/hub/models--mistralai--Mistral-Small-24B-Instruct-2501/snapshots/9527884be6e5616bdd54de542f9ae13384489724 \
TT_VISIBLE_DEVICES=2,3 pytest -q -s \
  models/autoports/mistralai_mistral_small_24b_instruct_2501/tests/test_functional_decoder.py
```

The real layer-20 weights come from `model-00005-of-00010.safetensors` and `model-00006-of-00010.safetensors` as selected by the checkpoint index.

## PCC results

| Case | Output PCC | K-cache PCC | V-cache PCC |
| --- | ---: | ---: | ---: |
| synthetic prefill, batch 32, sequence 4 | 0.9994167 | 0.9998630 | 0.9998662 |
| synthetic prefill, batch 32, sequence 18 | 0.9995333 | 0.9998639 | 0.9998673 |
| synthetic prefill, batch 32, sequence 128 | 0.9996013 | 0.9998620 | 0.9998669 |
| synthetic prefill, batch 13, sequence 4 | 0.9994267 | used by decode | used by decode |
| synthetic decode, batch 13, position 4 | 0.9995202 | 0.9998693 | 0.9998690 |
| real prefill, batch 32, sequence 18 | 0.9999699 | used by decode | used by decode |
| real decode, batch 32, position 18 | 0.9999697 | 0.9998788 | 0.9998838 |

All required PCC thresholds passed, and every output path exceeded the 0.995 aim. No functional-stage performance number was collected because this stage is correctness-scoped.

## Context evidence and review remediation

- A fresh one-device mesh reported 8 DRAM banks of 4,272,341,376 bytes each, for 34,178,731,008 total bytes.
- The first independent stage review returned `MORE-WORK-NEEDED`: the arithmetic proved HF context 32768 impossible, but did not prove that the initially recorded 128 was the largest feasible value.
- Added an opt-in, isolated-process batch-32 capacity test in `tests/test_context_capacity.py`. Full-shape synthetic BF16 weights have the same device allocation sizes as real weights, and each passing run copies the entire output to host to force completion.
- Swept 256, 512, 1024, 2048, 3072, 3328, 3456, 3520, 3552, 3553, 3568, 3576, 3580, 3582, and 3583 successfully. Sequence 4096 failed first during the geometric sweep; bisection found that 3583 passes while the immediately adjacent 3584 fails.
- Reproduced 3584's TT DRAM OOM twice. The retained expected-failure run requested a 1,174,405,120-byte buffer; per bank, 3,958,020,096 bytes were allocated, 314,321,280 were free, and the largest free block was 114,688,000 bytes versus 146,800,640 required. The fixture closed normally and `tt-smi -ls --local` saw all four chips afterward.
- Updated `current_supported_context` to the exact emitted-batch boundary 3583. This is capacity evidence, not a new PCC claim; PCC remains validated through sequence 128.
- At batch 32 and HF context 32768, the BF16 hidden input is 10,737,418,240 bytes, the BF16 K/V cache is 4,294,967,296 bytes, and one `[32,32768,32768]` BF16 MLP activation is 68,719,476,736 bytes, independently confirming that the advertised context cannot fit this unchunked path.

Capacity boundary commands:

```text
TT_LOGGER_LEVEL=fatal MISTRAL_SMALL_24B_CONTEXT_PROBE_LEN=<length> \
TT_VISIBLE_DEVICES=2,3 pytest -q -s \
  models/autoports/mistralai_mistral_small_24b_instruct_2501/tests/test_context_capacity.py

TT_LOGGER_LEVEL=fatal MISTRAL_SMALL_24B_CONTEXT_PROBE_LEN=3584 \
MISTRAL_SMALL_24B_CONTEXT_EXPECT_OOM=1 TT_VISIBLE_DEVICES=2,3 pytest -q -s \
  models/autoports/mistralai_mistral_small_24b_instruct_2501/tests/test_context_capacity.py
```

## Review and checkpoint

- Fresh reviewer `/root/stage_review_mistral` returned `MORE-WORK-NEEDED` only for the unproven initial context cap; all graph, translation, API, PCC, runtime-fallback, provenance, and scope gates passed.
- After the adjacent capacity boundary was measured and documented, a different fresh reviewer, `/root/stage_review_mistral_rereview`, returned `CLEAN-PASS` with no required work. Its explicit graph/path, IR-fidelity, TP-collapse, runtime, correctness, context, documentation, and scope assessments all passed.
- The post-remediation final serialized correctness run passed all four tests in 26.84 seconds with the real-weight environment enabled. Python compile, JSON parse, `git diff --check`, and the context-contract checker also passed.
- Local functional-decoder checkpoint: `6154b41e1f85baf4d621eb128bb651eb7f245f8f` (`Add Mistral Small 24B functional decoder from TTNN IR`).

## Multichip provenance

The retro-generated [`multichip_provenance.json`](multichip_provenance.json) records the complete layer-20 sharding prior from the selected prefill and decode IR. The capture uses a `1x4` mesh with TP degree 4 on mesh/cluster axis 1; each path has sum `all_reduce` collectives on that axis after `o_proj` and `down_proj`, with no other collective type in the segmented layer.
