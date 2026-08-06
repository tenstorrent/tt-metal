# Resume point — GLM blaze integration

Written at a context checkpoint. Everything below is measured on the 32-chip BH Galaxy unless
marked otherwise. Full context: [`../BLAZE_EVALUATION.md`](../BLAZE_EVALUATION.md).

## State of the two GLM fused ops

Both are **correctness-gated and cluster-measured**. Neither is in the model yet.

| op | correctness | cluster timing | vs |
|---|---|---|---|
| `GLMQKVAProjection` (q_a + kv_a, shared act) | PCC **0.9999** both outputs | 45.1 → **9.5 µs**, **4.76x** | ttnn's *fused* `q_kv_a` (the real shipping path) |
| `GLMQANormQBProjection` (q_a RMSNorm → q_b, chained) | `pcc_vs_device` **0.99989**, `pcc_vs_model` **0.99988** | 25.9 → **21.1 µs**, **1.22x** | ttnn `rms_norm` + `q_b` matmul |

Sources live in the tt-blaze tree; copies archived here under `glm_qkv_a_projection/` and
`glm_qa_norm_qb_projection/`, with `GLM_FUSED_OP_HANDOFF.md` for the second.

**Upper bound if both landed and fully translated: ~1.9 ms/token of 33.2 ms (~5.7%).** Treat as a
ceiling, not a forecast — see the two null results below.

## IMPORTANT: device state

The last run **hung** (EXIT=124, and 0 JIT compiles in its log, so it was not merely slow).
Per F12 a hang degrades the device and `open/close` still succeeds on a degraded one. **Before
trusting any measurement, run the control:**

```bash
cd /home/ttuser/sdawle/tt-blaze && source env.sh && unset TT_MESH_GRAPH_DESC_PATH
timeout 300 python -m pytest glm47_all_shapes_check.py -q     # expect 6 passed, ~11 s
# if it hangs:  tt-metal/python_env/bin/tt-smi -r   then re-run the control
```

## The one open blocker: feeding the model's activation to `GLMQKVAProjection`

`DRAMStreamingMatmul` wants the activation replicated per DRAM-bank worker as **1×32 tiles**.
The model holds `[1,1,32,2048]` in 32×32 tiles. Building the former from the latter costs
**4.8 µs/layer** against ~36 µs of headroom, so the reshard is affordable — that is settled.

What is not settled is declaring the CB. Findings, in order:

1. Passing the raw row-major tensor fails: blaze calls `tensor.get_tile()` →
   `'NoneType' object has no attribute 'height'`.
2. Pass a **CBHandle** instead, built with `f.cb_from_tensor(t, tile=Tile([1,32]), page_size=64)`.
   `tile=` alone is ignored — `page_size=` is what selects the branch that honours it
   (`fused_program.py:1380-1387`). This clears the tile error.
3. Next wall: `K mismatch: act gives 32, weights gives 2048`.
   `K_from_act = num_pages * tile_w` (`dram_streaming_matmul/common.py:278`), and for row-major
   `n_pages = shard.shape[0]` (`fused_program.py:1408`) — the shard **row count**, which ignores
   `page_size`. With shard `(1, 2048)` that is 1 page, so K reads as 32.
4. `total_size=` would set page count directly, but it reaches `_resolve_tensor_geometry` and is
   then **rejected by `BlazeProgram.cb_from_tensor`** — not plumbed through.
5. **Attempted and UNRESOLVED:** declare the per-core shard as `(64, 32)` instead of `(1, 2048)`
   — identical bytes, but makes `n_pages = 64` and `page_size = 64` fall out naturally. This
   **hung the device**. Script preserved at
   `/tmp/.../scratchpad/gate5_script.py` (regenerate from this doc if gone).
   Unknown whether the hang is the reshape/reshard chain or the CB view; use `triage-hang`
   (`TT_METAL_INSPECTOR=1`, logs land in `<cwd>/generated/inspector`, **not** `/tmp/tt-metal`).

Two cleaner options than fighting this from the caller:

- **(a)** Plumb `total_size` through `BlazeProgram.cb_from_tensor` (one-line-ish, blaze-side).
- **(b)** Do the retilize **inside** the fused op as its first phase. blaze ships `Retilize` for
  the 1×32 → N×32 direction; the input direction needs the reverse. This is the architecturally
  right answer: it removes the 4 ttnn conversion ops entirely instead of paying them.

`GLMQANormQBProjection` has the **same** activation requirement, so whichever fix lands unblocks
both.

## Remaining steps to the end-to-end swap

1. Control the device (above).
2. Close the CB gap via (a) or (b); re-gate PCC ≥ 0.99 for both ops.
3. Wire into `tt/blaze_ops.py` (the seam already exists, import-guarded and inert on our tree),
   then swap the call sites behind `GLM4_MOE_LITE_BLAZE_QKV_A=1`:
   - `attention_decode.py:144` — the fused `w_q_kv_a` matmul + its two `_safe_slice`s
   - `attention_decode.py:327-328` — `q_a_layernorm` then `attn_linear(w_q_b)`
4. Per-layer PCC gate against the ttnn path, then traced end-to-end.

### The end-to-end command, and what to compare against

The model runs traced **in blaze's tree** (that work is done). Three optimizations must be off
there because blaze's older ttnn lacks the parameters they use:

```bash
cd /home/ttuser/sdawle/tt-blaze && source env.sh && unset TT_MESH_GRAPH_DESC_PATH
HF_HOME=/dev/shm/hf \
GLM4_MOE_LITE_FUSE_DOWN_ROUTING_SCALE=0 \
GLM4_MOE_LITE_FUSED_COLLECTIVE_EPILOGUE=0 \
GLM4_MOE_LITE_FUSED_ROUTER=0 \
timeout 560 python tt-metal/models/experimental/glm4_moe_lite/scripts/debug_run_full_tt_greedy.py \
  --prompt "What is the capital city of Australia? Answer with just the city name." \
  --simulate-context-len 128 --min-cache-tokens 256 --max-new-tokens 32 --batch-size 1 \
  --mesh-rows 4 --mesh-cols 8 --kv-cache-dtype bf16 --phase both \
  --enable-trace --trace-mode sampling --cache-dir /dev/shm/ttnn_cache_blaze
```

**Compare against 34.8 ms/token** — same tree, same flags, no blaze op. NOT against 33.2 ms,
which is our tree with all optimizations on. Keeping those straight is the whole point:

| config | our tt-metal | blaze's tt-metal |
|---|---:|---:|
| full default flags | 33.2 | cannot run |
| the three off | 34.1 | **34.8** ← the baseline to beat |

## Two null results that should temper expectations

Both measured in the shipping traced regime, and both argue the cluster wins may not translate:

- **`GLM4_MOE_LITE_DS_CORE_CAP=8`** — giving ttnn blaze's 8-core bank layout changed the step
  33.2 → **33.4 ms**. No gain.
- **The bandwidth ceiling** — doubling every dense weight's bytes costs only **+3.9 ms (11.7%)**,
  yet the six per-op wins sum to 31.5% of the step. Most of that cannot be on the critical path.

Also relevant: fusing two projections that *share* an input contributed only ~4% (the rest was
`DRAMStreamingMatmul` beating ttnn), and the chained fusion that genuinely removes a DRAM
round-trip is worth 1.22x. Mechanism 2 works; at these cluster sizes it is small.

## Environment

- `/dev/shm` holds the 62.5 GB checkpoint (`HF_HOME=/dev/shm/hf`) and two converted-weight caches
  (`ttnn_cache`, `ttnn_cache_blaze`), ~93 GB total. Needed for any end-to-end run; `rm -rf` to
  reclaim.
- tt-blaze working tree holds uncommitted F3 (`mla_q_grid`, `scattered_q_heads`) and F11
  (`num_total_experts`, `zero_tail`) work — **validated on hardware this session**: F3 33/33;
  F11's hang fixed (17.8 s vs >700 s), GLM-5 dims pass, GLM-4.7's 64 experts run but PCC 0.0235.
  Do not clobber: syncs must only replace `tt-metal/models/experimental/glm4_moe_lite`.
