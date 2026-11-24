# Zig-zag load balancing for causal context parallelism

Opt-in: `device_config.cp_sharding: zigzag`. The default is `contiguous`.

## 1. Problem

Context parallelism (CP) splits the sequence over the N devices of the CP axis. Ring attention keeps Q on each
device and passes K/V around the ring: at ring step s, device d attends to the K/V of device
src = (d + s) mod N.

With a contiguous split, device d holds sequence block d. Under a causal mask:

| K/V source | Work on device d |
|---|---|
| src < d | full block, no mask |
| src == d (step 0) | diagonal block, causal mask |
| src > d | nothing (fully masked), the device waits |

Device N-1 computes N blocks and device 0 computes one. Every ring step waits for the slowest device, so the attention
time is set by device N-1, which does about twice the average work.

Causal attention over the whole sequence, N = 4. Rows are the Q blocks each device holds, columns the K/V blocks;
`█` is a full block, `◣` a causal diagonal block, `·` a block the mask removes:

```
              K/V block:   0    1    2    3      work (blocks)
device 0   Q block 0       ◣    ·    ·    ·      0.5
device 1   Q block 1       █    ◣    ·    ·      1.5
device 2   Q block 2       █    █    ◣    ·      2.5
device 3   Q block 3       █    █    █    ◣      3.5   <- every step waits for this device
```

What each device does at each ring step:

| step | device 0 | device 1 | device 2 | device 3 |
|---|---|---|---|---|
| 0 | ◣ | ◣ | ◣ | ◣ |
| 1 | idle | idle | idle | █ |
| 2 | idle | idle | █ | █ |
| 3 | idle | █ | █ | █ |

## 2. Design

### 2.1 Layout

Cut the sequence into 2N blocks. Device d holds blocks d and 2N-1-d, in that order:

```
N = 4, blocks 0..7:   device 0: [0 | 7]   device 1: [1 | 6]   device 2: [2 | 5]   device 3: [3 | 4]
```

This is the same device layout as ttnn's `ring_joint_sdpa` with `is_balanced` and `ring_distributed_sdpa`.

The same causal attention in half blocks, N = 4. Each cell shows the device that computes it (the device that holds
that Q half); `◣` is a causal diagonal half block:

```
              K/V half:   0  1  2  3  4  5  6  7      device: full + diagonal half blocks
Q half 0  (device 0)      ◣  ·  ·  ·  ·  ·  ·  ·      device 0:  0 + 7 full, 2 diagonal
Q half 1  (device 1)      1  ◣  ·  ·  ·  ·  ·  ·      device 1:  1 + 6 full, 2 diagonal
Q half 2  (device 2)      2  2  ◣  ·  ·  ·  ·  ·      device 2:  2 + 5 full, 2 diagonal
Q half 3  (device 3)      3  3  3  ◣  ·  ·  ·  ·      device 3:  3 + 4 full, 2 diagonal
Q half 4  (device 3)      3  3  3  3  ◣  ·  ·  ·
Q half 5  (device 2)      2  2  2  2  2  ◣  ·  ·      every device: 7 full + 2 diagonal
Q half 6  (device 1)      1  1  1  1  1  1  ◣  ·
Q half 7  (device 0)      0  0  0  0  0  0  0  ◣
```

Call the local halves chunk 0 (block d) and chunk 1 (block 2N-1-d). At a ring step s >= 1 with source src:

| Q half × K/V half | Visible? |
|---|---|
| Q chunk 1 × K/V chunk 0 | always fully visible |
| Q chunk 0 × K/V chunk 0 | fully visible if src < d, else fully masked |
| Q chunk 1 × K/V chunk 1 | fully visible if src > d, else fully masked |
| Q chunk 0 × K/V chunk 1 | never visible |

So at every step >= 1 every device computes exactly two unmasked half × half blocks: the cross block plus one of the
two diagonal pairs. Step 0 is one causal call over the local sequence on every device. No device ever idles, and no
block needs a mask other than "none" or "causal".

What each device does at each ring step, N = 4 (`cross` = Q chunk 1 × K/V chunk 0; `c0` / `c1` = the diagonal pair
`get_ring_step_work` picks):

| step | device 0 | device 1 | device 2 | device 3 |
|---|---|---|---|---|
| 0 | causal, full local | causal, full local | causal, full local | causal, full local |
| 1 | cross + c1 | cross + c1 | cross + c1 | cross + c0 |
| 2 | cross + c1 | cross + c1 | cross + c0 | cross + c0 |
| 3 | cross + c1 | cross + c0 | cross + c0 | cross + c0 |

Per device, the attention work becomes N/2 full-block units instead of up to N - 1/2 on the busiest contiguous device,
so the compute-bound ceiling on the attention speedup is (2N - 1) / N: 1.5x at CP2, 1.88x at CP8.

### 2.2 Pieces

| Piece | Change |
|---|---|
| CP sharding setting | `core::distributed::CpSharding {Contiguous, ZigZag}` in `ContextParallelConfig::sharding` (next to `reuse_forward_kv`); `cp_global_position` maps a device-order slot to its global position; `validate_cp_sequence_length` checks `S % (2N) == 0` for ZigZag. |
| RoPE | `build_rope_params` builds each device's sin/cos rows at the global positions of the tokens it holds. Contiguous is the identity, so its tables do not change. |
| Ring SDPA chunk list | `ring_sdpa_fw` / `ring_sdpa_bw` take a list of chunks and a `RingLoadBalancing {None, ZigZag}` mode. `get_ring_step_work(device, step)` returns `{execute, mask, chunk}`; each device's program descriptor binds `chunks[work.chunk]`, and a cache hit rebinds by list position. Outputs are caller-owned. No SDPA kernel change. |
| Zig-zag ring attention | `ring_attention_sdpa_zigzag`, chosen by a switch on the sharding (the context's, or a per-call override). nano_gpt reorders every sample's tokens and targets once into zig-zag order before sharding. Pipeline-parallel Llama rejects ZigZag. |

### 2.3 Why two calls per step

ttml's SDPA kernels need Sq == Sk, so one call cannot cover "Q full × K/V half". Two half × half calls cover the two
visible blocks with the existing kernels. One call per step needs a kernel change (see §6).

## 3. Costs

| Cost | Contiguous | Zig-zag |
|---|---|---|
| SDPA calls per ring step >= 1 (fw; bw is the same) | 1 | 2 |
| ring_shift dispatches per step (K, V) | 2 full tensors | 4 half tensors (same bytes) |
| Online-softmax merges per step (fw) | 1 | 3 (one on half the rows has weight 0 on each device) |
| Extra device memory while a layer runs | none | half-size buffers for the cross and the two chunk calls: outputs and intermediates in the forward, dQ/dK/dV in the backward |

Each extra call is one program per mesh coordinate, so at small local sequences the extra host dispatch can exceed the
balance gain (see §4 for where that happens).

## 4. Results

Setup: BH Galaxy. Op level: `ring_attention_benchmark`, one attention layer
fw+bw, 8 q / 2 kv heads per device, head dim 128, 1 warm-up and 5 timed iterations, median, forward K/V kept for
the backward (`--reuse-forward-kv`). Model level: Llama-8B dimensions (32 layers, 32 q / 8 kv heads, vocab 32000), random init, batch 1,
`memory_efficient` runner, TP4, `cp_reuse_forward_kv: true`, 20 steps, mean step time of steps 2-20.

### 4.1 Op level (one attention layer fw+bw, ms)

BH Galaxy; CP2 and CP4 on 1x2 / 1x4 submeshes, CP8 on the 8x4 mesh (CP8 x TP4 replicas).

| CP | S | S_local | contiguous fw / bw | zig-zag fw / bw | fw+bw change | speedup |
|---|---|---|---|---|---|---|
| 2 | 8k | 4k | 10.54 / 27.88 | 7.85 / 21.72 | -23.0% | 1.30x |
| 2 | 16k | 8k | 38.33 / 97.07 | 26.98 / 69.57 | -28.7% | 1.40x |
| 2 | 32k | 16k | 146.07 / 366.10 | 100.17 / 252.91 | -31.1% | 1.45x |
| 4 | 16k | 4k | 23.72 / 63.09 | 15.76 / 44.71 | -30.3% | 1.44x |
| 4 | 32k | 8k | 86.55 / 222.53 | 54.14 / 142.02 | -36.5% | 1.58x |
| 4 | 64k | 16k | 331.45 / 843.44 | 200.86 / 511.05 | -39.4% | 1.65x |
| 8 | 16k | 2k | 15.18 / 43.11 | 15.86 / 29.25 | -22.6% | 1.29x |
| 8 | 32k | 4k | 51.02 / 136.28 | 32.07 / 91.51 | -34.0% | 1.52x |
| 8 | 64k | 8k | 188.20 / 488.02 | 109.30 / 290.10 | -40.9% | 1.69x |
| 8 | 128k | 16k | 726.23 / 1870.46 | 408.67 / 1063.72 | -43.3% | 1.76x |

Short local sequences are slower: with 1k tokens per device (CP8, 8k) zig-zag takes 36.3 ms against 20.5 ms
  contiguous, because each step is too short to hide the doubled SDPA dispatches (§3). From 2k tokens per device
  (CP8, 16k) zig-zag wins.

### 4.2 Model level (Llama-8B training step, ms)

| Mesh | S | contiguous | zig-zag | change | max \|Δ loss\| over 20 steps |
|---|---|---|---|---|---|
| CP2 x TP4 (2x4 submesh) | 8k | 2800.0 | 2438.7 | -12.9% | 0.031 |
| CP2 x TP4 (2x4 submesh) | 16k | 7716.0 | 6191.1 | -19.8% | 0.031 |
| CP8 x TP4 (8x4) | 16k | 3180.5 | 2522.6 | -20.7% | 0.016 |
| CP8 x TP4 (8x4) | 32k | 9152.6 | 6201.5 | -32.2% | 0.016 |
| CP8 x TP4 (8x4) | 64k | 29160.7 | 18238.6 | -37.5% | 0.016 |

- Losses match over 20 steps: the largest per-step difference is one bf16 step of the printed loss at about 6.5
  (mean difference <= 0.007), from the different rounding of the two attention paths.
- As at op level, short local sequences are slower: at CP8 x TP4, 8k (1k tokens per device) the step takes 1944.6 ms
  against 1310.2 ms contiguous.

## 5. Tests

| Test | What it checks | CP |
|---|---|---|
| `CpShardingTest` (host) | identity for Contiguous; hand-written ZigZag layouts at CP2/CP4; every position once; length guard; yaml names; pipeline-parallel rejection | — |
| `Mesh1x2ZigZagRopeTest` | each device's RoPE rows equal the full table at its tokens' global positions; length guard | 2 |
| `RingSDPAScheduleTest` (host) | zig-zag work tables, balance, chunk index in range, causal-mask guard | 2-8 |
| `RingSDPAChunkTest` | per-device chunk binding fw/bw on cache miss and hit; swapped chunk list on a hit; contiguous skipped devices keep caller outputs; validation | 2, 4, 8 |
| `Mesh1x2RingSDPATest` | ring attention fw+bw vs host SDPA, contiguous (default re-shift) and zig-zag, GQA zig-zag, reuse vs re-shift bit-exact, zig-zag guards | 2 |
| `SystemMeshRingSDPATest` | zig-zag fw+bw with re-shifted and kept K/V, contiguous fw+bw, zig-zag reuse vs re-shift bit-exact; skips below CP4, the smallest size where a device switches chunk and the backward re-shifts more than once | 4, 8 |
| `GalaxyRingSDPATest` | contiguous and zig-zag fw+bw on an 8x4 Galaxy (CP8 x TP4), reuse vs re-shift bit-exact | 8 |

## 6. Limits and follow-ups

- Causal attention only; a non-causal mask with ZigZag is rejected.
- The local sequence must split into two tile-aligned halves (S_local % 64 == 0), and S % (2N) == 0.
- The data loader must deliver tokens in zig-zag order; nano_gpt does. The Python trainers have no binding for the
  setting yet.
- Not with pipeline parallelism.
- Follow-ups:
  - one program per device range instead of per mesh coordinate, to cut host dispatch at small local sequences;
  - one merge per zig-zag step for both chunks (needs the ring op to write both chunks into one buffer);
  - no per-step resets of unassigned buffers (the ring op writes zeros / -inf for the chunk a device skips);
  - one SDPA call per step with Sq != Sk support.
