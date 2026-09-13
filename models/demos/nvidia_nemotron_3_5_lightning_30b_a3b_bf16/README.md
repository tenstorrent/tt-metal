# NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 on Tenstorrent

A real end-to-end TTNN pipeline for
[`nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16),
a 31.6 B-parameter hybrid **Mamba2 / GQA-attention / MoE** causal LM
(`NemotronHForCausalLM`, 52 blocks, 128 routed experts top-6, `relu2` MLPs).

Tokenizer in → chained graduated TTNN stubs → generated text out, on a **4-chip
TP=2 × DP=2 mesh**.

---

## Calls (task heads)

`config.architectures == ["NemotronHForCausalLM"]` registers exactly **one**
task head, so there is one Call.

| Call | Task | Entrypoint | Reference |
|---|---|---|---|
| 1 | text → text (causal generation) | `demo/demo_text_generation.py` | `model.generate(..., do_sample=False)` |

```bash
# demo (32 prompts, 32 independent continuations)
python -m models.demos.nvidia_nemotron_3_5_lightning_30b_a3b_bf16.demo.demo_text_generation --compare-hf

# the gate
./python_env/bin/python -m pytest \
  models/demos/nvidia_nemotron_3_5_lightning_30b_a3b_bf16/tests/e2e/test_e2e_pipeline.py -s
```

The demo and the test call the **same** function — `tt/pipeline.py::NemotronHPipeline.run_text_generation`.
There is no second copy of the wiring, so a green test is a working demo.

---

## Layout

```
demo/demo_text_generation.py   Call 1 entrypoint (argparse + __main__)
tt/pipeline.py                 THE chained forward pass + trace/perf contract
tt/_hf_ref.py                  Source-A model loading + the depth cap
tt/_hf_compat.py               CPU shims so HF NemotronH loads without CUDA/mamba-ssm
tt/_invocation.py              the Gate-2 "which stub actually ran" registry
tests/e2e/test_e2e_pipeline.py Gates 1 / 2 / 3
tests/e2e/make_golden.py       the 32-prompt Source-A input
tests/pcc/                     per-component PCC tests (from bring-up)
_stubs/                        the 10 graduated TTNN stubs
```

---

## Where each graduated module runs (Gate 2)

All **10** graduated stubs are on load-bearing edges of the forward pass. Where a
whole-block stub and its constituents overlap, *different layers use different
variants* rather than calling one redundantly — so nothing is invoked merely to
tick a counter, and there is no coverage-sweep function anywhere in the package.

| Layer | Block type | Variant | Graduated stubs used |
|---|---|---|---|
| 0 | mamba | `MAMBA_A` | `nemotron_h_block` (owns norm + SSD mixer + residual) |
| 1 | moe | `MOE_A` | `nemotron_h_r_m_s_norm`, `nemotron_h_mo_e` |
| 2 | mamba | `MAMBA_B` | `nemotron_h_r_m_s_norm`, `nemotron_h_mamba2_mixer`, **`zamba2_r_m_s_norm_gated`** (injected as the mixer's gated grouped norm) |
| 3 | moe | `MOE_B` | `nemotron_h_r_m_s_norm`, `nemotron_h_topk_router`, `nemotron_h_experts`, `nemotron_h_m_l_p` (shared expert) |
| 4 | mamba | `MAMBA_C` | `nemotron_h_r_m_s_norm`, `nemotron_h_mamba2_mixer` |
| 5 | attention | `ATTN` | `nemotron_h_r_m_s_norm`, `nemotron_h_attention` |
| 6 | moe | `MOE_C` | `nemotron_h_r_m_s_norm`, `nemotron_h_topk_router`, `nemotron_h_experts`, **`re_l_u_squared_activation`** (shared expert = matmul → relu² → matmul) |

`nemotron_h_r_m_s_norm` additionally serves the model's final norm. Embedding,
lm_head and greedy sampling are pipeline-owned `ttnn` ops (`ttnn.embedding`,
`ttnn.matmul`, `ttnn.argmax`) — no host compute in the loop.

The graduated set is derived from disk (`_stubs/*.py.last_good_{native,sharded}`)
and asserted equal to `RUN_REPORT.md`'s `ON_DEVICE (10)` list by
`test_graduated_set_matches_bringup`, so it cannot silently drift.

---

## Parallelism — TP=2 × DP=2 on 4 chips

`ttnn.set_fabric_config(FABRIC_1D)` then `open_mesh_device(MeshShape(2, 2))`;
rows = DP, cols = TP. Every sharded stub computes `_tp_axis = len(mesh)-1 = 1`
and `all_reduce`s on that axis, so they all agree on which physical axis is TP.

| Component | Scheme | Requirement | This config | Verdict |
|---|---|---|---|---|
| `nemotron_h_attention` | column/row-parallel MHA + all_reduce | heads % TP, kv_heads % TP | 32 % 2, **2 % 2** | ok |
| `nemotron_h_mamba2_mixer` / `nemotron_h_block` | head-parallel SSD + all_reduce | mamba_heads % TP, n_groups % TP | 64 % 2, 8 % 2 | ok |
| `nemotron_h_mo_e` / `nemotron_h_experts` | **expert-parallel** (64 experts/chip) + all_reduce | n_routed_experts % TP | 128 % 2 | ok |
| `nemotron_h_m_l_p` | column/row-parallel MLP + all_reduce | shared inter % TP | 3712 % 2 | ok |
| `nemotron_h_topk_router` | replicated **by design** (tiny weight, full-width logits) | — | — | ok |
| `nemotron_h_r_m_s_norm` | replicated (normalizes the un-split hidden dim) | — | — | ok |
| `zamba2_r_m_s_norm_gated` | grouped RMS, TP-sharded when inside the sharded mixer | (inter/TP) % group_size | (4096/2) % 512 | ok |
| `re_l_u_squared_activation` | elementwise, runs on whatever shard it gets | — | — | ok |

**TP=2 is the largest viable degree for this model**: `num_key_value_heads == 2`,
and `kernel_findings.json` flags TP=4/8/32 as blockers for exactly that reason.
The pipeline genuinely shards — `test_gate1_sharding_is_live_on_device` fails if
no built stub took its `ShardTensor2dMesh` branch.

---

## Batch

**B = 32** independent samples per call: 32 distinct prompts, left-padded to a
common length, threaded as a leading batch dim through embedding → every block →
final norm → lm_head → argmax. One program per step feeds all 32 rows; there is
no python loop over samples. `nemotron_h_block` hard-coded a leading `1` in its
slice/reshape bounds (which would have silently dropped samples 2..32); those
bounds now come from `x.shape[0]`, and `test_batch_row0_matches_unbatched`
proves row 0 of the B=32 run still equals a B=1 run.

The PCC gate compares **each sample to its own golden row** and takes the
minimum, and separately asserts the 32 outputs are not all identical.

---

## The depth cap — a real hardware ceiling

**The gate builds 7 of the 52 decoder blocks.** This is a measured DRAM limit,
not a convenience:

* the 23 MoE blocks hold `128 × 2 × 2688 × 1856 × 23 = 29.4e9` parameters
  = **58.8 GB** in bf16;
* expert-parallel at TP=2 halves that to **~29 GB per chip**, against ~12 GB of
  DRAM per chip;
* TP > 2 (which would split them further) is blocked by `num_key_value_heads=2`.

No TP degree this model permits makes a resident 52-block build fit on 4 chips.

`layers_block_type[:7] == [mamba, moe, mamba, moe, mamba, attention, moe]` — the
shortest prefix that carries all three block types with enough of each to host
every graduated stub, so a capped build still exercises **every distinct op** the
full model runs, just fewer times. Embeddings, final norm and lm_head are intact.

**The golden is capped identically** (`tt/_hf_ref.py` truncates
`model.model.layers` to the same 7 blocks), so TT and HF compute the same
function and the PCC comparison is exact-in-scope. The generated *text* from a
7-block truncation is not meaningful English — that is a property of the
truncation, not of the port.

`build_pipeline(device, layers=None)` still means every layer; the gate and demo
pass `layers=7`. `TT_E2E_LAYERS` / `TT_PERF_LAYERS` override it.

---

## Decode horizon

Stop-token first: `generation_config.eos_token_id == [2, 11]`; decode halts when
every row has emitted a stop id. The safety cap is
`min(TT_E2E_MAX_NEW_TOKENS (default 16), max_position_embeddings - prompt_len)`.

16 is hardware-forced, not arbitrary: the graduated stubs are **stateless
full-sequence bodies** (no KV / SSM cache), so step *k* recomputes the whole
prefix and decode is O(N²) in tokens. The same `N` and the same stop set are
passed to `model.generate()`, so both sides always produce the same length.

---

## Trace / perf contract

`PIPELINE_STAGES = ["prefill", "decode"]` (derived: `ForCausalLM`,
`is_encoder_decoder == False`). Exposed on the pipeline object:

* `<stage>_trace_setup(inputs)` — pins the sequence axis to a fixed capacity
  `C` and pre-uploads the padded ids plus every shape-dependent constant
  outside the trace
* `<stage>_trace_step()` — one host-op-free forward at that fixed shape
* `<stage>_trace_inputs()` — zero-arg, returns exactly what `_trace_setup` takes
* `<stage>_trace_items()` — `B × C` for prefill, `B` for decode
* `trace_capture_selftest(device)` — captures / executes / PCCs / **releases**
  each stage in turn
* `host_op_selftest()` — runs encoded-ids → logits under
  `host_op_observer.observe_host_ops()`; tokenization and the one-time weight
  build are outside the observed region

`build_pipeline(device, model=None, layers=None, prefill_layers=None,
decode_layers=None, **kwargs)` **returns the resident object** and never runs it.
NemotronH has one repeated stack shared by both stages, so the per-stage
overrides resolve against it. `self.layers` is a plain list of same-typed
`TtNemotronHLayer`s and `self.hf` stays reachable, so the stack is discoverable
structurally.

---

## Changes made to the graduated stubs

Everything below is **additive** — no sharded body was rewritten to replication
and no torch compute op was introduced. `test_gate1_*` enforces both.

| Stub | Change | Why |
|---|---|---|
| `nemotron_h_experts` | optional `routing_dense` ttnn kwarg; host-side expert-axis split at build | the graduated body built its routing matrix with a **host** `torch.scatter_add_`, which would put host compute in the hot path and fail `host_op_selftest` |
| `nemotron_h_mo_e` | host-side expert-axis split at build | 128 per-expert `ttnn.slice`s out of a 1.3 GB tiled stack cost seconds *each* |
| `nemotron_h_mamba2_mixer` | optional `gated_norm` hook | lets the graduated `zamba2_r_m_s_norm_gated` stub supply step 12 instead of the inline copy |
| `zamba2_r_m_s_norm_gated` | optional `tp_shard` | so it can sit inside the head-sharded mixer (local width 2048, group_size 512) |
| `nemotron_h_block` | batch bounds from `x.shape[0]` | it hard-coded a leading `1`, silently dropping samples 2..32 |

`nemotron_h_attention`, `nemotron_h_m_l_p`, `nemotron_h_topk_router`,
`nemotron_h_r_m_s_norm` and `re_l_u_squared_activation` are byte-identical to
their `last_good_*` snapshots.

---

## Results

See `RESULTS.md` for the measured PCC numbers from the last gate run.
