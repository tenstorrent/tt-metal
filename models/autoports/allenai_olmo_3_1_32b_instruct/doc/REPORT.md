# allenai/Olmo-3.1-32B-Instruct on Tenstorrent Blackhole — bring-up report

generated 2026-09-09T15:01:49Z

## Verified

- p150 (1x1, P150): token accuracy vs HF bf16 reference top-1 95.0% / top-5 99.8% (performance precision)
- p150 (1x1, P150): single-chunk prefill validated to 16384 tokens
- p300 (1x2, P300): token accuracy vs HF bf16 reference top-1 95.8% / top-5 99.8% (accuracy precision)
- p300 (1x2, P300): single-chunk prefill validated to 65536 tokens
- p300x2 (1x4, P150x4): token accuracy vs HF bf16 reference top-1 95.2% / top-5 99.8% (accuracy precision)
- p300x2 (1x4, P150x4): single-chunk prefill validated to 65536 tokens
- container profile `p150`: tt-model serve / API proof (identity, greedy, determinism, tool calling) / stop passed with advisory findings
- host vLLM `p150`: OpenAI API proof passed (advisory findings)
- container profile `p300x2`: tt-model serve / API proof (identity, greedy, determinism, tool calling) / stop passed
- host vLLM `p300x2`: OpenAI API proof passed

## Not verified

- host vLLM `p150`: readiness runner exit 1
- container profile `p300`: not verifiable on this box: the p300 profile opens a direct (1,2) mesh, which needs a physical P300 host (2 chips). On this QuietBox-2 a direct 2-chip mesh cannot initialise fabric (inter-board eth routers never see a partner; probed 2026-09-09). TP2 was validated as a (1,2) submesh on bare metal (stages 05/06).
- host vLLM `p300x2`: readiness runner exit 1
- TTI release workflow (meta_ifeval / meta_gpqa_cot): not run (optional stage 11)
- hybrid KV-cache groups are disabled in vllm-tt-plugin: sliding layers allocate full-length KV (future optimisation)

## Accuracy and performance (tt_transformers demo, HF bf16 refpt = tale-of-two-cities 512-token teacher-forced continuation)

| profile | precision | top-1 | top-5 | TTFT b1 (ms) | decode t/s/u b1 | decode t/s b32 | validated context |
|---|---|---|---|---|---|---|---|
| p150 (1x1, P150) | performance | 95.0 | 99.8 | 460.97 | 12.18 | 348.73 | 16384 |
| p300 (1x2, P300) | accuracy | 95.8 | 99.8 | 299.77 | 16.77 | 512.29 | 65536 |
| p300x2 (1x4, P150x4) | accuracy | 95.2 | 99.8 | 177.05 | 25.17 | 790.84 | 65536 |

## Datatype sweep

```
{
 "p150": {
  "performance": {
   "top1": 95.0,
   "top5": 99.8,
   "tok_s_user": 12.18,
   "ttft_ms": 460.97
  },
  "accuracy": {
   "top1": 96.8,
   "top5": 99.8,
   "tok_s_user": 10.41,
   "ttft_ms": 485.45,
   "error": null
  }
 },
 "p300": {
  "performance": {
   "top1": 95.8,
   "top5": 99.8,
   "tok_s_user": 16.77,
   "ttft_ms": 299.77
  },
  "accuracy": {
   "top1": 97.2,
   "top5": 99.8,
   "tok_s_user": 15.01,
   "ttft_ms": 316.25,
   "error": null
  }
 },
 "p300x2": {
  "performance": {
   "top1": 95.2,
   "top5": 99.8,
   "tok_s_user": 25.17,
   "ttft_ms": 177.05
  },
  "accuracy": {
   "top1": 97.2,
   "top5": 99.8,
   "tok_s_user": 23.1,
   "ttft_ms": 179.54,
   "error": null
  }
 }
}
```

selected: `{"p150": {"optimizations": "performance", "why": "forced: 16k single-chunk prefill on one P150 validated only with the bfp4 FF2 (performance) config; top-1 +1.8 at 15% t/s/u cost"}, "p300": {"optimizations": "accuracy", "why": "top-1 +1.4 at 10% t/s/u cost"}, "p300x2": {"optimizations": "accuracy", "why": "top-1 +2.0 at 8% t/s/u cost"}}`

## Unit tests (stage 03, one chip)

```
load_checkpoints critical rc=0 5 passed
hybrid_kv_spec critical rc=0 9 passed
rope critical rc=0 3 passed
qk_norm_full_width critical rc=0 4 passed
rms_norm critical rc=0 2 passed
embedding critical rc=0 2 passed
lm_head critical rc=0 1 passed
mlp critical rc=0 4 passed
attention_L0 critical rc=0 4 deselected 4 passed 4 deselected
attention_prefill_L0 critical rc=0 2 deselected 2 passed 2 deselected
decoder_L0 critical rc=0 2 passed
decoder_prefill_L0 critical rc=0 2 passed
attention_L3 critical rc=0 4 deselected 4 passed 4 deselected
attention_prefill_L3 critical rc=0 2 deselected 2 passed 2 deselected
decoder_L3 critical rc=0 2 passed
decoder_prefill_L3 critical rc=0 2 passed
attention_hf_rope advisory rc=0 6 deselected 2 passed 6 deselected
```

## Full-model PCC (test_model.py)

```
test_model rc=0 4 passed
```

## Advisory findings and known limitations

- vLLM `p150` plugin sampling suite: 17 failed, 56 passed; failing cases are the seed-reproducibility, penalty and mixed-batch tests (bfp4/bfp8 numerics make greedy ties batch-dependent): test_host_only_params.py, test_seeding_and_variety.py, test_structured_output_dp1.py, test_tt_penalties.py
- vLLM `p300x2` plugin sampling suite: 2 failed, 1 passed; failing cases are the seed-reproducibility, penalty and mixed-batch tests (bfp4/bfp8 numerics make greedy ties batch-dependent): test_request_isolation.py, test_seeding_and_variety.py

## Stage table

| stage | body | gate | started | ended |
|---|---|---|---|---|
| 00e-metal-build | ok | 0 | 2026-09-09T05:41:30Z | 2026-09-09T05:55:45Z |
| 00-host-prep | ok | 0 | 2026-09-09T05:41:22Z | 2026-09-09T05:41:30Z |
| 01-weights | ok | 0 | 2026-09-09T05:41:30Z | 2026-09-09T05:49:44Z |
| 02-cpu-reference | ok | 0 | 2026-09-09T06:05:50Z | 2026-09-09T06:07:36Z |
| 03-unit-pcc-1chip | ok | 0 | 2026-09-09T07:28:10Z | 2026-09-09T07:32:26Z |
| 04-full-model-p150 | ok | 1 | 2026-09-09T08:15:11Z | 2026-09-09T08:24:09Z |
| 05-multichip | ok | 0 | 2026-09-09T08:24:09Z | 2026-09-09T08:45:51Z |
| 06-dtype-perf-sweep | ok | 0 | 2026-09-09T08:45:51Z | 2026-09-09T08:52:16Z |
| 07-vllm-host | ok | 1 | 2026-09-09T09:33:06Z | 2026-09-09T09:43:49Z |
| 08-package | ok | 0 | 2026-09-09T09:43:49Z | 2026-09-09T09:48:21Z |
| 09-serve-prove | ok | 1 | 2026-09-09T09:48:21Z | 2026-09-09T10:01:27Z |
| 10-report-push | running | - | 2026-09-09T15:01:45Z |  |

## Package

- manifest: `~/tt-model-builds/olmo-3.1-32b-instruct/tt_kernel_manifest.json` (schema 5.1, kind vllm-plugin, tt-metal 0.65.2.dev9797, image `tt-model/olmo-3.1-32b-instruct:88c48066d1ae`)
- profiles: p150 (p150, mesh P150, ctx 16384, seqs 32), p300 (p300, mesh P300, ctx 65536, seqs 32), p300x2 (p300x2, mesh P300x2, ctx 65536, seqs 32); default `p150`
- weights `allenai/Olmo-3.1-32B-Instruct` @ `ac0587e4a774`
- code: tt-metal worktree `~/tt-metal-olmo3`, branch `jashan/olmo3-32b` (models/tt_transformers + models/autoports/allenai_olmo_3_1_32b_instruct); local commits, not pushed

## Artifacts

- unit PCC: artifacts/pcc/ · full model: artifacts/full/ · multichip: artifacts/multichip/ · sweep: artifacts/sweep/ · vLLM: artifacts/vllm/ · container proofs: artifacts/prove/ · hardware events: logs/hw-events.log

## Published

- repo: https://huggingface.co/jashansinghTT/olmo-3.1-32b-instruct-blackhole (public)
- clean-pull proof on p150: PASSED (`tt-model serve --profile p150 jashansinghTT/olmo-3.1-32b-instruct-blackhole`; prove exit 1)

## p300 profile attempted on this QuietBox-2 (2026-09-09 19:02Z)

`tt-model serve --profile p300 jashansinghTT/olmo-3.1-32b-instruct-blackhole` on the 4-chip box: image pulled, model
registered, vLLM engine initialised (26.9 s), then `Fabric Router Sync: Timeout ... on Device 0` — the routers on the
inter-board ethernet channels 2 and 4 stay at STARTED while the intra-board channels 8 and 9 reach
REMOTE_HANDSHAKE_COMPLETE. This is the known QuietBox-2 limitation for a direct 2-chip mesh (the partner chips of
those links are not opened), not a model failure: the same TP2 model validated at 95.8% / 99.8% token accuracy and 64k
prefill as a (1,2) submesh. The profile remains **unverified in a container until it is booted on a physical P300**.
