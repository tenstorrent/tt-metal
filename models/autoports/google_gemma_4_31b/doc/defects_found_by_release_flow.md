# Defects found by running the release flow locally

Date: 2026-08-18 UTC. Four distinct defects, one per attempt. Two of them produce
wrong answers or crashes in normal serving and were invisible to every existing
test in this repo.

| # | Defect | Status |
| --- | --- | --- |
| 1 | Greedy-only adapter vs upstream `vllm bench serve` | Fixed (spec data) |
| 2 | Prefill SDPA circular buffers clash with decode-resident L1 | Mitigation under test |
| 3 | `gcd`-sized block shard overflows L1 for 54% of prompt lengths | Fixed (code) |
| 4 | Chunked prefill path degrades accuracy | **Open regression, caused by my own mitigation for #2** |

## 1. Greedy-only vs the upstream benchmark client

Symptom: benchmark reported 8 "successful" requests with **0 generated tokens** in
0.03 s, then EngineCore died.

```text
generator_vllm.py:297  prefill_forward -> _require_semantic_greedy(sampling_params)
ValueError: on-device Gemma 4 31B serving sampling is greedy-only
```

The client said why itself: *"vllm bench serve no longer sets temperature==0
(greedy) in requests by default. The default will be determined on the server
side."* The adapter advertises `sample_on_device_policy: "greedy_only"` expecting
the plugin to route non-greedy requests to host sampling, but the plugin never
implemented that hook (the same four missing Stage 10 hooks
`test_vllm_adapter_contract` xfails on), so the request reaches the model and the
guard raises.

`GEMMA4_31B_VLLM_HOST_SAMPLING_COMPAT` does **not** help: it gates only the
`sampling_params is None` path, and here params are present and non-greedy.

Fix: `override_generation_config: '{"temperature": 0.0, "top_k": 1}'` in the spec,
the same mechanism AFM-4.5B uses. No fidelity cost -- greedy is the only mode the
model supports on device and the sampler does not affect throughput.

Note the earlier Shield benchmark pass predates the vllm-fork ->
vllm-tt-plugin migration, whose bench client changed this default.

## 2. Prefill SDPA circular buffers clash with decode-resident L1

```text
TT_THROW: Statically allocated circular buffers in program N clash with L1 buffers
on core range [0-0 - 7-3]. L1 buffer allocated at 1374336
and static circular buffer region ends at 1374976
```

The CB region **always ends at 1374976**, in every occurrence. So the SDPA
program's L1 footprint is fixed, not sequence-dependent; the clash happens
whenever another L1 buffer sits below that mark. Observed twice: buffer at
1,374,912 (64 B overlap) and at 1,374,336 (640 B overlap).

What is resident is decode-owned. `MLP_DECODE_CORES = 14` places width-sharded
decode activations on cores 0..13, and the prefill SDPA grid `[0-0 - 7-3]` covers
them. `generator_vllm._release_decode_state()` releases decode **traces** but not
those buffers, so the collision depends on allocator state.

**Correction to an earlier diagnosis.** This was first read as a direct-vs-chunked
path problem, and mitigated by lowering `GEMMA4_PREFILL_SDPA_MAX_SEQ` to 1024 to
route traffic to the chunked path. That was wrong on both counts:
`_chunked_full_attention_concatenated` builds an **identical** config
(`CoreCoord(8,4)`, q=k=128), so the chunked path has the same footprint and was
never safer -- it just did not collide on that run's allocation history. And the
reroute caused defect #4.

Mitigation under test: `_prefill_sdpa_config()` in `multichip_decoder.py` halves
the K chunk (128 -> 64) for the `head_dim>=512` layers to buy real headroom, used
by both SDPA call sites; sliding layers keep the shared (8,8) q=256/k=128
geometry. The principled fix is to move decode's L1 residents off the prefill
grid, as the persistent CCL buffer already does by using tail cores.

## 3. `gcd`-sized block shard overflows L1 for most prompt lengths

`_chunked_attention_output_projection` sized its write shard with

```python
height_cores = math.gcd(tile_rows, 8)
```

which collapses to 1 for any tile-row count coprime with 8, putting a whole chunk
on one core row:

```text
TT_FATAL: Out of Memory: Not enough space to allocate 18923520 B L1 buffer across
7 banks, where each bank needs to store 2703360 B, but bank size is 1461504 B
```

Checked over sequence lengths 1..70,000:

| | Worst per-core | Lengths over budget |
| --- | ---: | ---: |
| Old (`gcd`) | 6,242,304 B | **38,080 of 70,000 (54%)** |
| New | 786,432 B | **0** |

`seq=1760` computes 2,703,360 B under the old code -- byte-identical to the crash
log. Power-of-two lengths give `tile_rows` 64/128 and `gcd` 8, so they fit, which
is exactly why a power-of-two benchmark sweep passes while real eval prompt
lengths crash.

Fix: align the chunk **boundaries** to 8 tile rows (256 rows) so every full chunk
divides by 8 by construction and only a sub-256-row tail can land on one core;
`gcd` replaced by the largest divisor <= 8. Padding each chunk up to a power of
two was considered and rejected: it needs a real allocation and copy of up to
twice the chunk and is pure waste just past a boundary.

## 4. OPEN: the chunked prefill path degrades accuracy

Scores are deterministic -- two runs produced bit-identical results -- which makes
this unambiguous:

| Run | `GEMMA4_PREFILL_SDPA_MAX_SEQ` | MMLU | GPQA |
| --- | --- | ---: | ---: |
| 1 | 32768 (default) | 0.7837 | **0.3750** |
| 2 | 32768 (default) | 0.7837 | **0.3750** |
| 4 | **1024** | 0.7847 | **0.1750** |

GPQA fell 20 points, to **below the 25% random floor** for 4-choice questions,
while MMLU moved by ~2 samples. For scale, Google publishes GPQA Diamond 84.3 for
the instruction-tuned model (thinking mode), TTI measures 83.33 for it on H100,
and the n-shot variant costs the instruct model ~30 points -- so ~53 is the
instruct expectation on this task and 37.5 was a credible base-model result.
17.5 is not. The mechanism fits: GPQA's 5-shot science prompts
are long and cross 1024 into the chunked path, whereas most MMLU prompts stay
under it.

So the #2 mitigation traded a crash for **silently wrong answers**, which is
worse. The chunked full-attention path was previously reachable only above 32,768
tokens -- effectively never -- and is not trustworthy.

### Why no test caught it

`test_multichip_sliding_nonaligned_window_wrap_matches_baseline` runs a prefill at
seq 1025/1057 and then PCC-checks the **decoded token**, asserting only
`output.shape[-2] == seq_len` for the prefill result. Since
`_chunked_attention_output_projection` produces the hidden states passed to later
layers -- after this layer's KV cache is written -- an error there cannot move the
decode check. The chunked prefill's **output activations are unverified**, and the
full-attention chunked path has no moderate-length PCC coverage at all.

### Next step

Revert the threshold to the default so traffic returns to the validated direct
path, keep fix #3 (a genuine bug, inert when the chunked path is unused), and rely
on the #2 headroom change instead. Then confirm GPQA returns to 0.3750. A
direct-vs-chunked prefill-output PCC comparison at seq 1500 for both layer kinds
is the diagnostic that localises the numerical error.
