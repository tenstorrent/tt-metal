# Task: find what actually causes the PCC decay in mistral_medium_d_p prefill attention

You are debugging an accuracy problem in `models/demos/mistral_medium_d_p` in the tt-metal repo at
`/data/dgolubovic/repos/tt-metal` (branch `dgolubovic/mistral-attention`). Work on a 32-chip
Blackhole Galaxy.

## The finding

The prefill attention block's PCC against a float32 torch reference **decays monotonically with the
length of the attended prefix**. Measured on a *single unchunked* call, so no chunking, no cache
rotation and no chunk seam are involved:

| attended prefix (tokens) | 1,024 | 2,048 | 4,096 | 51,200 |
|---|---|---|---|---|
| tail-block PCC | 0.99185 | 0.98379 | 0.93517 | **0.60253** |

Production target is a **128K** prefill, 2.5× deeper than the worst point measured. At 0.60 the block
is not usable, so the question is what to fix.

## What is already ruled out — do not re-litigate these

1. **Chunking is not the cause.** Running the same tokens as two chunks vs one gives PCC **0.99982**
   device-to-device, and chunked-vs-reference (0.99763) is indistinguishable from
   single-vs-reference (0.99765). The per-block PCC curve is *continuous across the chunk seam* — a
   rotation / RoPE-start-row / Q-offset bug would put a visible step there. There is none.
2. **Math fidelity is not the cause.** HiFi4 → HiFi2 changed PCC by <2e-6 (ring SDPA 0.9998858 →
   0.9998847). The block currently runs HiFi2 + `packer_l1_acc=True`.
3. **It is not a sharding or reassembly bug.** Per-TP-column PCC is uniform (0.98714 / 0.98754 /
   0.98750 / 0.98747 at 2048), i.e. all four column shards are equally wrong, which a mis-assembled
   gather would not produce.

## Leading hypothesis (unproven — your job is to prove or kill it)

The KV cache is **bf8** (`bfloat8_b`, block-float, ~7 mantissa bits + shared exponent per 16 values).
K is read from that cache straight into the attention scores, and YaRN's `attention_factor` (1.4159,
applied to **both** cos and sin) makes the scores run about **2× hot**, so softmax amplifies the
quantisation noise. The more positions compete in the softmax, the worse it gets.

Competing explanations you must also test, because nobody has:
- the **float32 reference itself** losing accuracy at 51,200 (it reduces over 51,200 terms too);
- **random activations** being a pathological worst case — unstructured scores give a near-uniform
  softmax that reshuffles under the slightest perturbation, whereas real activations attend peakily;
- error contributed by the **projections / o_proj**, not the attention core, notably the
  `ttnn.typecast(tensor, ttnn.bfloat8_b)` in `apply_output_projection` (`tt/attention/operations.py`);
- accumulation inside the ring op's **online softmax** across 8 hops.

## The decomposition to run (this is the core of the task)

Build a ladder and measure PCC at prefix lengths **1K / 4K / 16K / 51K** at each rung. The single
most informative comparison is **L1 vs L3**: if the ring SDPA alone already shows the decay, the
cache/op owns it; if L1 is clean and only L3 degrades, the projections/RoPE/o_proj own it.

- **L0 — is the reference trustworthy?** Recompute `_reference_tail` in float64 for a handful of rows
  and compare against the float32 version. If they diverge materially, every number below is suspect
  and this must be fixed first.
- **L1 — ring SDPA alone, bf8 cache.** Place Q/K/V directly, no projections and no RoPE. Extend
  `tests/unit/test_ring_joint_sp_vs_ref.py` (which already does exactly this at 1024) to longer
  sequences. Reference is a plain causal GQA.
- **L2 — ring SDPA alone, bf16 cache.** Same as L1 with `allocate_kv_cache(cache_dtype=ttnn.bfloat16)`.
  **Note:** `tt/attention/dense_sp.py` asserts a bf8 cache, and the persistent ring-gather buffers in
  `tt/ccl.py::get_ring_gather_buffer` are also allocated bf8 — both must change together, and the op
  may simply reject bf16. If it does, say so and fall back to quantising K/V to bf8 *on the host* and
  feeding that through the float reference — that isolates the dtype's effect arithmetically even if
  the device path cannot run bf16.
- **L3 — full `attention_forward`.** Already exists:
  `tests/unit/test_attention_accuracy_at_depth.py` (51,200 tokens, 10 chunks of 5,120; it computes Q
  for the final 5,120 rows against K/V for all 51,200, one head at a time, because a full-sequence
  float reference at that length is ~960 GB).
- **L4 — RoPE's contribution.** Rebuild the tables with `attention_factor` forced to 1.0 (see
  `tt/rope_tables.py::yarn_inv_freq`, which returns it) and re-run L1. If PCC recovers substantially,
  the 2×-hot scores are a real amplifier and that is a design finding, not just a numerics one.
- **L5 — activation realism.** Repeat the best rung with activations that produce *peaked* attention
  rather than uniform — e.g. strongly correlated `x`, or real checkpoint weights via `HF_MODEL`. This
  decides whether 0.60 is a real risk or an artifact of random test data. **This is the one that
  determines whether anyone needs to act.**

## Deliverable

1. A table: rung × prefix length → PCC.
2. A one-line verdict naming the dominant error source, with the evidence that isolates it.
3. Whether 128K is viable as-is, and if not, the cheapest fix (bf16 cache? fp32 accumulation in the
   softmax? a different cache layout?) with its cost — note that bf8 halves both cache bytes and
   ring-gather traffic, and the ring gather is **70–73% of the block's device time** at depth, so a
   bf16 cache is not free.
4. Commit tests you add; do not commit throwaway scripts.

## Environment — four traps, all previously hit

```bash
# 1. container (only /data and /home are shared; /opt/venv and the image are per-machine)
docker pull ghcr.io/tenstorrent/tt-metal/tt-metalium/ubuntu-22.04-dev-amd64:latest
docker run -d --name tt-metalium-dev --user $(id -u):$(id -g) \
  --device /dev/tenstorrent:/dev/tenstorrent:rwm \
  -v /dev/hugepages-1G:/dev/hugepages-1G -v /home/dgolubovic:/home/dgolubovic -v /data:/data \
  --cap-add CAP_SYS_PTRACE --shm-size 8g --restart unless-stopped \
  -w /data/dgolubovic/repos/tt-metal \
  -e USER=dgolubovic -e LOGNAME=dgolubovic -e HOME=/home/dgolubovic \
  ghcr.io/tenstorrent/tt-metal/tt-metalium/ubuntu-22.04-dev-amd64:latest sleep infinity

# 2. ALWAYS redo this — the venv is container-local, and without it `import ttnn` silently
#    resolves to an empty namespace package (repo-root ttnn/ shadows ttnn/ttnn/).
#    `export PYTHONPATH=$TT_METAL_HOME` does NOT fix this; it causes it.
docker exec tt-metalium-dev bash -lc 'uv pip install -e .'

# 3. Run tests with USER set, or pytest dies during COLLECTION with
#    `KeyError: getpwuid(): uid not found` from inside torch's cache-dir setup.
docker exec -e USER=dgolubovic -e LOGNAME=dgolubovic tt-metalium-dev bash -lc \
  'export TT_METAL_HOME=$PWD && python3 -m pytest <path> -q --no-header'

# 4. If FABRIC_1D_RING fails to map ("Graph specified in MGD could not fit ... STRICT"), the box is a
#    plain-mesh Galaxy with no wrap-around links. Add MISTRAL_LINEAR_FABRIC=1. A torus reports a
#    degree histogram {4:32}; a grid reports {2:4, 3:16, 4:12}. Correctness is topology-independent,
#    so this does not affect any PCC number — only performance.
```

The build on `/data` is current and profiler-enabled; a rebuild should not be needed
(`grep ENABLE_TRACY build_Release/CMakeCache.txt` → `ON`).

## Ground rules

- **Check `who` before anything disruptive.** These boxes are shared; `tt-smi -r` resets all 32 cards
  and will destroy other people's in-flight work.
- A previous 51,200-token run ended in a **bus error that wedged the driver** (`tt-smi` then failed
  with "Query mappings failed on device 0"). If that recurs, it is worth investigating in its own
  right — it may be a real resource bug at long sequence lengths, not just bad luck.
- Report PCC to 5 decimal places, and state which rung and prefix length each number came from.
- If a hypothesis dies, say so plainly and move to the next; do not narrow the threshold to make a
  test pass.
