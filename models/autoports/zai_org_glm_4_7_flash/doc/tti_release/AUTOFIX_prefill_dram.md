# AutoFix Report - TR-001: prefill OOM at prompts above ~40k through the serving path

## Starting Evidence

No `AUTOTRIAGE.md` / `AUTODEBUG.md` existed for this failure. The failure was
already fully captured (a hard `TT_FATAL` with a Python traceback, not a hang),
so no fresh `$autotriage` pass was needed; the source-level diagnosis below was
done from the preserved logs plus the model source, and every hypothesis was
then verified or refuted on hardware.

Original failing command: the TTI release stage's benchmark sweep against the
live autoport vLLM server. Sweep point 21 (`isl=65536 osl=128
max_concurrency=1`) killed the vLLM EngineCore:

```
TT_FATAL: Out of Memory: Not enough space to allocate 402653184 B DRAM buffer across 8 banks,
where each bank needs to store 50331648 B, but bank size is 4228587904 B
(allocated: 4196763008 B, free: 31824896 B, largest free block: 20877312 B) (assert.hpp:104)
RuntimeError: TT_FATAL @ tt_metal/impl/allocator/bank_manager.cpp:462
```

Call site: `tt/fused_decoder.py:429 _moe_prefill -> ttnn.transpose(gu, 1, 3)`,
reached from `generator_vllm.prefill_forward -> generator.prefill_and_sample ->
model.prefill_forward_last_logits_device -> model.run_layer_stack_prefill ->
fused_decoder.prefill_forward -> functional_decoder._mlp`.

Preserved logs (outside the repo, not copied here):

- `/home/stisi/glm47_tti_release/crash_evidence/server_oom_crash.log`
- `/home/stisi/glm47_tti_release/crash_evidence/server_boot_head.log`
- `/home/stisi/glm47_tti_release/crash_evidence/server_full_run1.log`
- `/home/stisi/glm47_tti_release/logs/tti_release.log`

Everything up to and including `isl=32768` (including a 32768/128
`max_concurrency=6` point) had completed in the same server process.

## Hypothesis Experiments

### H1 - the failing 402,653,184 B buffer is the MoE gate_up transpose, and it needs *two* of them live

- **Experiment (source, exact arithmetic).** `FusedDecoder._moe_prefill` builds
  `gu` as `[1, G, 1, E, 32, 2*inter]`, then does `gu = ttnn.transpose(gu, 1, 3)`.
  With the serving `VLLM_PREFILL_CHUNK_SIZE = 1024`: `G = 1024/32 = 32`,
  `E = n_routed_experts = 64`, `2*inter = 2*1536 = 3072`, bf16.
  `32 * 64 * 32 * 3072 * 2 = 402,653,184 B`.
- **Result.** Byte-exact match with the failing allocation. `ttnn.transpose`
  allocates its output while the input is still live, so the peak is
  `2 * 402,653,184 = 805,306,368 B`, which is *exactly* the 0.75 GiB
  `safety_margin_gib` already in `get_max_tokens_all_users`. The buffer scales
  with the chunk, not with the prompt, so this term was correctly sized and is
  not the bug.
- **Verdict: verified (and exonerated as the cause).**
- **Fix: none.**

### H2 - the unreserved term is the whole-prompt prefill activation pair

- **Hypothesis.** `GLM47FlashModel.run_layer_stack_prefill` (tt/model.py) does
  `nxt = layer.prefill_forward(x, ...)`, `deallocate(x)`, `x = nxt`, and
  `FusedDecoder.prefill_forward` allocates `out_acc` for the *whole* physical
  prompt before its chunk loop. So at each of the 47 layer boundaries two
  `[1, 1, phys, 2048]` bfloat16 tensors are live: `phys * 8192` B, growing with
  the prompt and reserved nowhere. Prediction: the shortfall is
  `phys`-proportional, and the boundary sits where
  `free_before_gu(phys) < 805,306,368 B`.
- **Experiment A (source audit of everything else O(prompt)).** Read
  `FusedDecoder.prefill_forward`, `FusedDecoder._attn_prefill_chunk`,
  `OptimizedDecoder._attn_prefill_chunk`, `FusedDecoder._moe_prefill`,
  `_swiglu_linear`, `model.prefill_forward_last_logits_device`,
  `model._slice_rows` / `_pad_to_sampler_rows` / `lm_head_prefill`. Every other
  prefill tensor is chunk-sized (1024 rows) or tile-sized (32 rows): `x_c`, `h`,
  `attn`, `res`, `h2`, `mlp`, `out_c`, `qkv_a`, `q_abs`, `attn_lat`, `kvpe`,
  `gate`/`up`/`h`, the terminal slab/norm/logits. `out_acc` is written per
  chunk via `slice_write`, and `S_pad == phys` at every reachable serving
  length so the trailing `ttnn.slice` never doubles it. vLLM prefills one row at
  a time (`generator_vllm.prefill_forward` loops `slots`), so one pair is the
  peak even when the scheduler batches several prompts into one step.
- **Result A.** The activation pair is the only unreserved O(prompt) DRAM term.
- **Experiment B (hardware, pre-fix, fresh server).** Launched the exact release
  server command, probed exact token-id prompt lengths 32768 then 65536.
- **Result B.** 32768 -> HTTP 200, `usage.prompt_tokens=32768`. 65536 -> HTTP 500,
  EngineCore dead with the byte-identical `TT_FATAL` at the same
  `fused_decoder.py:429` call site (`allocated: 4196373888`, `free: 32214016`,
  `largest free block: 20877312` per bank). Boot line reproduced the crashed
  run exactly: `469104 tokens`, `GPU KV cache size: 471,168 tokens`,
  `num_blocks=7362`.
- **Experiment C (hardware, pre-fix, second fresh server, sharpened boundary).**
  From B's allocator numbers, device-wide free before the `gu` allocation was
  `257,712,128 + 402,653,184 = 660,365,312 B`, so pure-capacity arithmetic
  predicted the cliff at `phys <= 47104` passing and `phys >= 48128` failing.
  Probed 47104 then 48128.
- **Result C.** *Both failed.* 47104 died with device-wide free
  `411,951,104 B` - already **more** than the `402,653,184 B` requested - but a
  largest contiguous per-bank free block of only `23,799,872 B` against the
  `50,331,648 B/bank` needed. So the activation pair does not just consume the
  space the transpose needs, it **fragments** it; the pure-capacity prediction
  was too optimistic by roughly one 384 MiB buffer's worth of contiguity. This
  refuted the naive capacity-only model of the boundary while confirming the
  mechanism, and it is why the fix reserves the whole pair rather than only the
  arithmetic shortfall.
- **Verdict: verified.**
- **Fix.** `tt/generator_vllm.py`, `GLM47FlashForCausalLM.get_max_tokens_all_users`:
  subtract a third, prompt-length-scaled term from the KV budget, alongside the
  existing prompt-length-independent `safety_margin_gib = 0.75` (which stays
  untouched):

  ```
  max_prefill_phys      = ceil(max_model_len / block_size) * block_size   # 202752
  hidden_size           = load_hf_config(...).hidden_size                 # 2048, from the checkpoint
  reservation           = 2 * max_prefill_phys * hidden_size * 2          # 1,660,944,384 B = 1.547 GiB
  total_tokens          = (headroom - 0.75 GiB - reservation) / 29376
  ```

  `PREFILL_LIVE_WHOLE_PROMPT_ACTIVATIONS = 2` and
  `PREFILL_ACTIVATION_DTYPE_BYTES = 2` are new module constants that name the
  two model facts behind the `2 *` and the `* 2`; `hidden_size` is read from the
  checkpoint's own HF config through the existing
  `resolve_checkpoint_dir` / `load_hf_config` helpers, and `block_size` from
  `doc/context_contract.json`'s `kv_cache` block. Nothing is hard-coded.
  `prefill_physical_len` clamps `phys` to `max_seq_len_physical`, so the
  block-aligned served context is the exact worst case.

- **Verification (hardware, post-fix, one server process, in order).**

  | requested prompt tokens | HTTP | `usage.prompt_tokens` | exact echo | wall s |
  |---|---|---|---|---|
  | 10000 (non-aligned) | 200 | 10000 | yes | 22.1 |
  | 65536 | 200 | 65536 | yes | 295.1 |
  | 131072 | 200 | 131072 | yes | 1022.0 |
  | 131071 (non-aligned) | 200 | 131071 | yes | 978.6 |
  | 202751 (full context) | 200 | 202751 | yes | 2199.5 |

  Token-id prompts (no chat template) so the length is exact; `temperature 0.0`;
  `max_tokens = min(8, 202752 - N)`. Zero `Out of Memory` / `EngineCore
  encountered` lines in the whole post-fix server log. The exact release sweep
  point that had died (`isl=65536 osl=128 max_concurrency=1`) was re-run through
  `vllm bench serve` and completed 1/1 with 65536 input and 128 output tokens.

### H3 - shrinking the KV pool regresses decode

- **Hypothesis.** The pool drops 471,168 -> 414,656 tokens; if decode cost
  depended on pool size (page-table width, cache stride, trace shapes) the
  optimized-vLLM headline would move.
- **Experiment.** `--stages benchmark --server-url http://127.0.0.1:8000` at
  128 in / 128 out / `--benchmark-concurrency 1`, twice: on the freshly booted
  post-fix server, and again on the same process after all five long prefills.
- **Result.**

  | run | mean TPOT ms/token | t/s/u | mean TTFT ms |
  |---|---|---|---|
  | optimized-vLLM stage reference | 29.496 | 33.903 | ~274 |
  | post-fix, fresh server | 29.490 | 33.910 | 280.9 |
  | post-fix, after the 202751 prefill | 29.494 | 33.905 | 275.8 |

  TPOT is within 0.02% of the recorded headline in both runs. Decode does not
  depend on the pool size: `blocks_per_user` is computed from `max_seq_len`, not
  from `num_blocks` (VS-011), so the page-table width and every decode trace
  shape are unchanged.
- **Verdict: refuted (no regression).**
- **Fix: none.**

### Infrastructure note (not a model finding)

Between the second pre-fix run and the first post-fix run, EngineCore startup
failed with `RuntimeError: NOC0 is hung on PCIe device ID 1` - fallout from the
two deliberately OOM-killed EngineCore processes, not from the code change.
Recovered per `$tt-device-usage`: no stale processes, `tt-smi -ls --local`
showed all four p300c boards, `build_Release/tools/umd/warm_reset
--max-attempts 3` succeeded on attempt 1, a 1x1 `open_mesh_device` /
`close_mesh_device` smoke passed, and the server booted normally on the retry.
All four boards were healthy again after the full post-fix run.

## Final Status

**Fixed.**

- Before: prompts of 47104 and 65536 tokens killed EngineCore; only <= ~40k was
  servable, against an advertised and gate-enforced 202752-token context.
- After: 10000, 65536, 131072, 131071 and 202751 all return 200 with the exact
  prompt length echoed, in one server process, with no OOM.
- KV pool: 469,104 -> 412,563 budgeted tokens; vLLM `GPU KV cache size`
  471,168 -> 414,656 tokens (7362 -> 6479 blocks), which vLLM reports as
  `Maximum concurrency for 202,752 tokens per request: 2.05x`. Still 2.05x the
  served context, so a single full-context user fits with room to spare.
- Decode TPOT unchanged at 29.49 ms/token (33.9 t/s/u).
- No capability reduction: the served context, `max_num_seqs`,
  `VLLM_PREFILL_CHUNK_SIZE` and the datatype-sweep precision policy are all
  untouched. The advertised context became *reachable*, not smaller.

Commands that prove the final state:

```bash
# server (held open for all probes and benchmarks below)
./python_env/bin/python -m models.common.readiness_check.run_vllm_server \
  --stages serve --model-dir models/autoports/zai_org_glm_4_7_flash \
  --hf-model zai-org/GLM-4.7-Flash --mesh-device N150 \
  --max-num-seqs 32 --max-model-len 202752 \
  --tt-config '{"trace_region_size": 350000000}' \
  --additional-server-args "--reasoning-parser glm47"

# exact-length prefill probes (token-id prompts)
python probe.py 10000 65536 131072 131071 202751

# decode regression
./python_env/bin/python -m models.common.readiness_check.run_vllm_server \
  --stages benchmark --server-url http://127.0.0.1:8000 \
  --model-dir models/autoports/zai_org_glm_4_7_flash --hf-model zai-org/GLM-4.7-Flash \
  --benchmark-prompt-len 128 --benchmark-output-len 128 \
  --benchmark-num-requests 1 --benchmark-concurrency 1 --no-benchmark-ci-serving

# host-side unit tests over the changed function
./python_env/bin/python -m pytest \
  models/autoports/zai_org_glm_4_7_flash/tests/test_generator_vllm_adapter.py \
  -k "get_max_tokens_all_users or full_vllm_plugin_contract" -q     # 2 passed

# context-contract gate
./python_env/bin/python .agents/scripts/check_context_contract.py \
  --model-dir models/autoports/zai_org_glm_4_7_flash --hf-model zai-org/GLM-4.7-Flash \
  --stage tti-release --require-contract                            # exit 0
```

Machine-readable evidence: `doc/tti_release/autofix_prefill_dram.json`.

## Remaining risks / follow-up evidence needed

- **The reservation is measured against one config, not swept.** It is exact for
  the activation pair itself, but the amount of *residual* room it leaves the
  MoE transients is a derived figure, not a directly instrumented one. Working
  back from the two pre-fix crash dumps, DRAM free at the start of a prefill was
  about 1.20 GB (`257,712,128 + 536,870,912 + 402,653,184 = 1,197,236,224` at
  prompt 65536; `411,951,104 + 385,875,968 + 402,653,184 = 1,200,480,256` at
  prompt 47104). Subtracting the activation pair, that left about 930 MB free at
  the prompt 32768 that passed and a measured `814,604,288 B` at the prompt
  47104 that failed - so the fragmented requirement sits between those two.
  Post-fix the pool gives back `56,512 * 29,376 = 1,660,096,512 B`, so at the
  full 202752 context the same subtraction leaves about 1.20 GB, roughly 29%
  more than the largest known-good pre-fix figure. That is a comfortable margin
  and it is backed by a measured pass at 202751, but it is not a swept
  fragmentation study. A larger `VLLM_PREFILL_CHUNK_SIZE`, more experts, or a
  wider `moe_intermediate_size` would grow the 384 MiB transpose buffers and
  re-tighten it.
- **The 0.75 GiB fixed margin is still a committed constant**, even though it
  now provably equals `2 * VLLM_PREFILL_CHUNK_SIZE * n_routed_experts *
  2*moe_intermediate_size * 2` byte-for-byte. Deriving it the same way the new
  term is derived would remove the last magic number here, but that is a
  refactor of an already-validated value and was deliberately left out of this
  fix.
- **The real headroom cost is a capacity trade, and it is worth naming.** The
  pool pays 1.547 GiB permanently for a transient that only exists during
  prefill. Halving the peak at source - streaming the gate_up transpose, or
  slicing `out_acc` so the layer-to-layer activation is not held whole - would
  buy that back as servable KV tokens. Both are real optimizations, both change
  the prefill compute path, and neither belongs in an OOM fix.
- **Long-prompt prefill is slow enough to matter for the release sweep.** The
  202751-token probe took 2199.5 s wall, part of it first-use program
  compilation for chunk offsets past the warmed buckets (a known limitation
  recorded in `full_model.notes`). Release sweeps at these lengths need their
  timeouts sized accordingly.
- **Not re-measured:** the multi-concurrency sweep points above 32768 (the
  release stage's 32768/128 `max_concurrency=6` point passed pre-fix and the
  pool is now smaller, so admission at high concurrency *and* long context is
  bounded by 2.05x the served context rather than 2.32x). vLLM's own admission
  accounting enforces that; no code path in this adapter assumes the larger
  pool.
