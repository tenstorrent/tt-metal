# Initial P1 plan audit

The proposed two-text run is valid as an initial P1 measurement **if the
reference first fits the full262144 context, including prepared live traces**.
`e8_three_hifi2` is a placeholder pending that device result and the HiFi4
comparison below, not a proven highest-precision fitting policy in this CPU
audit. It promotes expert gate/up/down in layers
0,1,47 to BFP8 and uses expert HiFi2; the other45 layers retain BFP4. Preserve
the physical-fit evidence and required-to-fit precision limitation. Do not
describe this as an all-BFP8 reference.

The current `p1_candidates_initial.json` still names `e8_four_hifi2`; replacing
the command's reference ID alone will fail argument validation. The following
template creates a separate candidate file from existing construction kwargs,
then initially scores only the fitting reference against original HF. Run only
after the parent has established fit and confirmed the device is free.

```bash
export TT_MODEL_BRINGUP_ROOT=/home/ttuser/qwen38-flash/codex_home/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.24
export TT_AUTODEBUG_ROOT=/home/ttuser/qwen38-flash/codex_home/plugins/cache/tenstorrent-skills/tt-autodebug/0.1.8
BRINGUP_EXPORTS=$(python "$TT_MODEL_BRINGUP_ROOT/scripts/environment.py") || exit
eval "$BRINGUP_EXPORTS"
export PYTHONPATH="$PWD:$PYTHONPATH"
export HF_HOME=/home/ttuser/qwen38-flash/hf
export TT_VISIBLE_DEVICES=0,1,2,3
export TT_MESH_GRAPH_DESC_PATH=tt_metal/fabric/mesh_graph_descriptors/p300_x2_mesh_graph_descriptor.textproto
export TT_METAL_TRACE_ALLOC_TRACKING=1
export TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0
python_env/bin/python - <<'PY'
import json
from pathlib import Path
root = Path('models/autoports/qwen_qwen3_8_flash_next/doc/datatype_sweep').resolve()
ids = ('e8_three_hifi2', 'inherited_e4_lofi')
candidates = {cid: json.loads((root / 'configs' / f'{cid}.kwargs.json').read_text()) for cid in ids}
assert all(row['max_seq_len'] == 262144 and row['max_batch_size'] == 1 for row in candidates.values())
(root / 'p1_candidates_three.json').write_text(json.dumps(candidates, indent=2) + '\n')
PY
timeout --signal=INT --kill-after=60 14400 python_env/bin/python -m \
  models.autoports.qwen_qwen3_8_flash_next.tests.run_datatype_p1 \
  --model-dir models/autoports/qwen_qwen3_8_flash_next \
  --hf-model Qwen/Qwen3.8-Flash-Next \
  --mesh-device P300x2 --fabric-config FABRIC_1D_RING \
  --candidates models/autoports/qwen_qwen3_8_flash_next/doc/datatype_sweep/p1_candidates_three.json \
  --reference e8_three_hifi2 --selected e8_three_hifi2 --only e8_three_hifi2 \
  --num-texts 2 --max-new-tokens 6144 --max-text-tokens 8192 \
  --min-answer-tokens 512 --min-positions 1000 \
  --sampling '{"temperature":1.0,"top_p":0.95,"top_k":20,"seed":1000}' \
  --chat-template-kwargs '{"enable_thinking":true,"thinking":true}' \
  --topk 64 --decode-tokens 256 --decode-texts 2 --hf-positions 2048 \
  --nll-floor 0.004 --kl-max-x1000 5.0 --hf-kl-max-x1000 4.0 \
  --no-score-cache \
  > models/autoports/qwen_qwen3_8_flash_next/doc/datatype_sweep/logs/p1_three_initial.log 2>&1
```

No threshold is relaxed. Two accepted answers of at least512 tokens provide
at least1024 scored positions, above1000. The runner may try more than two
prompts if answers are too short or repetitive. CPU rendering of bundled r01
and r02 using the pinned tokenizer produced111 and113 prompt tokens, both
ending in the assistant `<think>` opening. Even their full6144-token outputs
fit the8192 text bound. Model-card temperature1/top-p0.95/top-k20 fits the
device sampler's maximum top-k32. The adapter's preserved signature exposes
`sampling_params`, and its EOS trimming excludes post-answer output while
keeping teacher forcing greedy. Actual retained text/position counts, EOS
events and gate output must still be checked after execution.

## Expert HiFi4 reference question

Expert HiFi4 is source-legal and is not ignored or overridden. `MeshMoe`
constructs `expert_compute` directly from
`compute_fidelities.experts` (`tt/multichip_decoder.py:757`) and passes it to
`moe_fused_swiglu` for both prefill and decode (`:850`). The native validator
rejects FP32 destination accumulation, packer-L1 accumulation, and full-DST
sync, but imposes no HiFi4 prohibition
(`moe_fused_swiglu/device/moe_fused_swiglu_device_operation.cpp:300`). The
factory forwards the requested fidelity into `ComputeConfigDescriptor`
(`moe_fused_swiglu_program_factory.cpp:601`), and the matmul helpers use the
normal compile-time fidelity path. Blackhole LLK programs the number of math
phases from the fidelity enum (`tt_llk_blackhole/llk_lib/llk_math_matmul.h:433`),
so HiFi4 is not an alias for HiFi2 in this path.

HiFi4 does not restore bits already discarded from BFP4/BFP8 weights, and its
effect after BF16 intermediates/accumulation cannot be guaranteed from source.
However, neither the op nor the LLK proves HiFi2 equivalent for these BF16
activation / BFP8-or-BFP4 weight products. The three BFP8 layers especially
make dismissing the extra phases without target-model evidence unsupported.

Recommendation: evaluate an otherwise identical `e8_three_hifi4` policy before
calling HiFi2 the highest-precision fitting reference. It changes kernel
fidelity, not resident weight/KV byte counts; prepared-trace fit still needs a
device check. If it fits, use it as the conservative P1 reference or measure
its real-model difference from HiFi2 and document an evidence-backed choice.
If the already-planned initial HiFi2 run starts first, retain it as provisional
evidence; it does not eliminate this comparison. Keep the same LoFi/HiFi2
candidates for performance selection. No HiFi4 execution or accuracy claim is
made by this audit.

Selecting the reference does not pass through self-comparison. Without a
`--higher-precision` candidate, `reference_selected_check` requires the original
HF comparison to satisfy KL×1000 <=4. The adapter loads all48 original HF
layers from revision812d96e10a0dd82b69d749e68693698cf6c36daa using exact mmap
weights and exact PLE rows. It bypasses the runner's dense-resident RAM estimate;
this does not guarantee host working-set fit. `--hf-positions 2048` stops
**between whole texts**: a first6144-token answer causes all6144 answer
positions to be HF-scored. HF results are recomputed on every invocation.

`prefill_logits` returns `[1, logical_length,248320]` and accepts nonaligned
lengths. Device work stays in1/2/4/8/16/32-token chunks, but host vocabulary
logits are accumulated.6144 positions need5.68GiB in FP32 before prompt rows;
chunk concatenation and padded output can temporarily retain roughly three
copies. `score_logits` adds a512-row FP32 log-softmax workspace. Original HF
scores the whole text in one causal forward, adding attention/workspace memory
and mmap page pressure. Keep CPU HF work serial with performance measurements.

The runner requests `enable_trace=True` for generation and the final256-token
teacher-forced decode. The public generator rejects `enable_trace=False` and
uses `_replay_decode` after the prefill prediction; thus this source path is
traced. The adapter now records per-call counter deltas under the actual
`generator.model.precision` config ID and rejects zero model replays. The
CPU mock check in `p1_adapter_cpu_check.json` exercised the current nested
wrapper with the installed runner's256-token teacher-forcing function: it
recorded255 replays, retained EOS predictions, preserved the sampling signature,
and rejected the zero-replay negative control. These are host contract checks,
not measured device results. P1 records no teacher-forcing throughput; its
decode comparison is advisory and cannot replace the sweep's performance
evidence. Cache hits bypass construction and therefore produce no new wrapper
counter records; retain the original uncached sidecar alongside cached results.
No device execution occurred in this audit.

Cache behavior requires care:

- `--no-score-cache` refreshes scores, but does not regenerate existing
  `doc/eval_readiness/precision_reasoning/text.jsonl`. Existing text bypasses
  `--num-texts` and generation settings. Preserve old evidence before replacing
  an unsuitable text cache; do not silently label old text as newly generated.
- Scores are keyed by text, candidate kwargs, reference role/kwargs, model
  Python source, and precision-file contents. Adapter/kernel source, common
  build kwargs, and scoring settings are not all included. Use explicit
  candidate kwargs and refresh scores when any unkeyed dependency changes.
- Follow-up candidates can use `--only inherited_e4_lofi --selected
  inherited_e4_lofi` without `--no-score-cache`, reusing a matching reference.
  Contrary to the skill reference's aggregation wording, this installed
  runner loads only the current `order`: with `--only`, that is reference plus
  the named candidate. Other cached candidates are reported unscored in that
  invocation. Preserve per-run results; run without `--only` to aggregate all
  offered candidates from cache, or finish with the selected candidate and
  reference only. The final invocation must name the shipped selected ID.
- Two texts use128-position blocks for uncertainty estimates because fewer
  than eight texts are present. This is runner-supported initial evidence,
  not a broad prompt-suite or qualitative-quality claim.
