# ZAYA1-8B (Zyphra MoE) on Blackhole P150a

Port of [Zyphra/ZAYA1-8B](https://huggingface.co/Zyphra/ZAYA1-8B) — 8.4 B total / 760 M
active params, 80 layers strictly alternating **CCA attention** (Compressed Convolutional
Attention) and **MoE** (16 experts, top-1, MoD skip, EDA router state) — to a single
Tenstorrent Blackhole **P150a**, token-exact against the HuggingFace reference.

> End-to-end check: `"The capital of France is"` → **` Paris`** (logits PCC 0.996,
> last-token argmax = golden 9079), reproduced by both the non-traced and ttnn-traced
> decode paths.

- **Branch:** `yito/zaya`
- **Jira:** [CSEJ-110](https://tenstorrent.atlassian.net/browse/CSEJ-110) (parent, under Epic [CSEJ-61 Model Support](https://tenstorrent.atlassian.net/browse/CSEJ-61))
- **Full design + ground-truth shapes:** [`PORTING_SPEC.md`](./PORTING_SPEC.md)

## Status & milestones

| Phase | Jira | Status | Milestone commit |
|---|---|---|---|
| Prefill bring-up (CCA + MoE, ref-validated op-by-op PCC) | [CSEJ-111](https://tenstorrent.atlassian.net/browse/CSEJ-111) | ✅ Done | `9ff2e7cfab5` |
| Incremental decode + ttnn trace | [CSEJ-112](https://tenstorrent.atlassian.net/browse/CSEJ-112) | ✅ Done | `9ff2e7cfab5` |
| Single-card decode perf (trace / L2-vec / bfp8) | [CSEJ-113](https://tenstorrent.atlassian.net/browse/CSEJ-113) | ✅ Done | `9ff2e7cfab5` |
| Multi-card tensor parallel (tt_ccl) | [CSEJ-114](https://tenstorrent.atlassian.net/browse/CSEJ-114) | 🔄 In Progress | `9ff2e7cfab5` (foundation) |

> Update this table (and add a Jira comment) at each milestone commit so progress,
> performance, and approach stay traceable from the ticket.

## Performance (single P150a, greedy decode, token-exact)

| Stage | ms/tok | notes |
|---|---:|---|
| Naive (rebuild model per token) | 3523 | 17 GB reload/token |
| Build-once + program cache + batched-dense MoE | 508 | device-resident |
| **+ ttnn trace** | **108.2** | fixed-MAX persistent KV, replayed graph |
| **+ vectorized CCA per-head L2** | **99.6** | 10 tok/s — current best |

Profiled traced step (MAX=64): `execute_trace` 86.9 %, `lm_head`+argmax 10.9 %, host
input-writes 2.2 %. **Decode is op/dispatch bound, not DRAM-bandwidth bound** — bfp8
expert weights (half the bytes) give only ~3 %, so the lever is op-count reduction, not
weight sharding. Larger context is cheap: MAX=256 is token-exact at ~111 ms/tok (+3 ms).

## Implementation approach

- **Residual stream in fp32**, learned per-channel residual scaling with delayed/parallel
  merge at the next layer's start (mirrors HF `ZayaModel.forward`).
- **CCA** ([`tt/cca.py`](./tt/cca.py)): the two causal depthwise/grouped `Conv1d(k=2)`
  compose into `qk@Cm^T + shift1@Bm^T + shift2@Am^T + bias` with host-precomputed
  block-diagonal matrices; `(q+k)/2` mean residual; per-head L2-norm (vectorized in head
  layout) × `sqrt(head_dim)`; per-kv-head temperature; 2-stream value (current + 1-step
  shift); partial RoPE (50 %, θ=5e6); GQA. **Manual fp32-softmax attention** matches HF
  eager and removes the routing drift that ttnn SDPA introduced. 3-part decode cache
  (KV + conv_states + prev_hs).
- **MoE** ([`tt/moe.py`](./tt/moe.py)): MLP router → softmax(17) → +balancing bias →
  top-1 one-hot gate; MoD skip-expert (idx 16) = identity; EDA 256-d router state carried
  across MoE layers. All 16 experts run as **one batched matmul** (gate-weighted sum) —
  no host sync, ~13× fewer dispatches than a per-expert loop. `ZAYA_EXPERT_DTYPE=bf16|bfp8`
  toggle (bfp8 halves expert memory, token-exact here).
- **ttnn trace** ([`tt/trace.py`](./tt/trace.py)): fixed-`MAX` persistent KV / conv / prev_hs
  buffers; per-position inputs (cos/sin/mask/onehot) updated host-side **outside**
  `execute_trace`; KV written by masked add into the fixed window. **Gotcha:** an op's
  *cold* first run issues host→device writes that are forbidden during capture, and prefill
  (S = prompt_len) does not warm the S=1 decode programs — so `capture()` runs the S=1
  backbone once eagerly to warm the program cache *before* `begin_trace_capture`.
- **Multi-card** ([`tt/`](./tt) + `run_zaya_multi.sh`): `ttnn.set_fabric_config(FABRIC_1D)`
  **before** opening the `(1,N)` mesh is required for tt_ccl (else "un-initialized fabric
  context"). `all_gather` validated; vocab-sharded TP `lm_head` is token-exact. Measured TP
  decode speedup ~1.08× → TP targets prefill throughput / model capacity, not the
  op-bound single-token decode latency.

## Run

```bash
# single card (default device 1)
TT_DEVICE=1 /home/yito/work/run_zaya.sh python models/demos/zaya1_8b/demo/demo.py

# traced decode validation + bench
TT_DEVICE=1 /home/yito/work/run_zaya.sh python models/demos/zaya1_8b/tests/run_trace.py

# multi-card mesh + CCL smoke / TP lm_head
TT_DEVICES=0,1 /home/yito/work/run_zaya_multi.sh python models/demos/zaya1_8b/tests/run_mesh_smoke.py 2
TT_DEVICES=0,1 /home/yito/work/run_zaya_multi.sh python models/demos/zaya1_8b/tests/run_tp_lmhead.py 2
```

## Layout

```
tt/        model.py · cca.py · moe.py · standard.py · cache.py · trace.py · model_args.py
tests/     run_phase{1,2,3,5}.py · run_trace*.py · run_decode_*.py · run_mesh_smoke.py · run_tp_lmhead.py · run_fallback_audit.py
reference/ zaya_hf/ (mirrored modeling code) · dump_golden.py · golden/ (gitignored .pt)
demo/      demo.py
PORTING_SPEC.md
```

## Next levers (single-card, op-count bound)

`to_heads`/`from_heads` via `ttnn.experimental.nlp_create_qkv_heads_decode` /
`nlp_concat_heads_decode` (replace ~36 slice/concat ops per layer); fused SDPA-decode;
sharded/streamed argmax for the `lm_head` tail.
