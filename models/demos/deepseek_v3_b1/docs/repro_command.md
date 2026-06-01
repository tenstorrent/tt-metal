# Reproducing the half-DEST flash_mla bug on craq-sim (no hardware)

> Companion to [`debug_log.md`](./debug_log.md). This is the **hardware-free** repro of
> Attempt 19 / Attempt 20: the bank-asymmetric `flash_mla` per-core output bug, run
> entirely on the `craq-sim` simulator. Reaches the hash signal in **~18 s** of wall
> time per run, with zero silicon allocation.

---

## TL;DR

```bash
cd /localdev/ncvetkovic/work/tt-metal

TT_METAL_SIMULATOR=/tmp/craq-sim-bh/libttsim.so \
ARCH_NAME=blackhole \
TT_METAL_HOME=$PWD PYTHONPATH=$PWD \
TT_METAL_DRAM_BACKED_CQ=1 \
TT_METAL_DRAM_BACKED_CQ_DIRTY_FLUSH=auto \
TT_METAL_SIMULATOR_CQ_WAIT_CLOCKS=10000 \
TT_METAL_SIMULATOR_DIRECT_TENSOR_LOAD=1 \
TT_METAL_SIMULATOR_PARALLEL_CLOCK_THREADS=16 \
TTSIM_HANG_WATCHDOG_CLOCKS=10000000 \
TT_METAL_DPRINT_CORES="(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(3,2)" \
TT_METAL_DPRINT_RISCVS="BR,NC,TR0" \
TT_METAL_DPRINT_FILE=/tmp/flashmla_dprint.log \
python_env/bin/pytest \
  "models/demos/deepseek_v3_b1/tests/unit_tests/test_flash_mla.py::test_flash_mla_decode[32768-128-127-1]" \
  -x --timeout=500 -s

# Then inspect the per-core hashes:
grep 'hash\[' /tmp/flashmla_dprint.log
```

Expected signal (the bug): on **every** SDPA worker core, the **iteration-0** hash
differs from the **iteration-1..9** hashes, and iterations 1..9 are bit-identical to
each other. That iter-0-vs-rest split is the bank-asymmetric `flash_mla` output bug
(half-DEST `dst_full_sync_en=False`, residual DEST state at kernel boot — see
`debug_log.md` Attempt 19/20).

---

## Why this test (and not `test_decoder_mlp`)

The original `test_decoder_mlp` runs the full DeepSeek decoder block on a **4×2 (8-chip)
mesh**. On craq-sim that path accumulates a large command-queue backlog during host-side
tensor setup, and the simulator drains it at only ~2.6 kHz — the run takes hours and may
never finish within a sane timeout. The actual bug, however, is **single-Tensix**:
bank-parity asymmetry in the `flash_mla` SFPU/FPU DEST round-trips.

`test_flash_mla_decode` exercises exactly that op:
- **single chip** (uses the `device` fixture, not a mesh),
- runs `FlashMLADecode.op` in a **10-iteration loop** on one program,
- `decode_position=127` → 1 K-chunk, the cheapest case,
- reaches the simulator at ~13 kHz (no multichip CQ overhead) → ~18 s total.

The 10-iteration loop is what surfaces the bug: iter-0 runs on freshly-booted DEST
(bank 0 / residual), iters 1..9 reach steady state. The hashes split iter-0 vs the rest.

---

## Prerequisites (one-time setup)

### 1. tt-metal checkout / branch

```bash
cd /localdev/ncvetkovic/work/tt-metal
git checkout ncvetkovic/multichip_half_dest   # off ridvan/nkapre-multichip-metal-v2
git submodule update --init tt_metal/third_party/umd tt_metal/third_party/tracy
```

This branch carries the half-DEST debug port:
- `flash_mla.hpp`: `hash_cb(sdpa_output_cb, out_chunk_tiles, 0x30)` after the final
  `cb_push_back`, gated on `#ifdef DEBUG_CB_HASH`.
- `micro_ops/flash_mla/op.py`: passes `defines=[("DEBUG_CB_HASH", "1")]` to the
  `UnifiedKernelDescriptor`.
- `tests/unit_tests/test_flash_mla.py`: the PCC-vs-torch assert is relaxed to non-fatal
  (sim numerics differ from torch golden), and the iter-vs-iter0 assert is converted to a
  log line (that drift IS the bug, so we don't want it to abort).
- The `cb_hash` debug LLK files under `tt_metal/hw/...debug/`.

### 2. Build tt-metal (Release)

```bash
./build_metal.sh --build-type Release --enable-ccache
```

### 3. Build the craq-sim simulator

`craq-sim` lives at `/localdev/ncvetkovic/work/craq-sim` on the `multichip` branch.

```bash
cd /localdev/ncvetkovic/work/craq-sim
git checkout multichip

# IMPORTANT: this box has only AVX2 (x86-64-v3), not AVX-512.
# The default -march=x86-64-v4 produces a libttsim.so that SIGILLs on init.
./make.py --env TTSIM_MARCH=-march=x86-64-v3 src/_out/release_bh/libttsim.so
```

Two small simulator patches are on this `multichip` checkout (in `src/tensix.cpp`),
relaxing over-conservative `TTSIM_VERIFY` guards whose function bodies already handle
the values the deepseek kernels use:
- `tensix_execute_unpacr_nop`: allow `bank_clr_ctrl=1` (both-bank clear is implemented).
- `tensix_execute_cfgshiftmask`: allow `scratch_sel` `0..3` (switch handles all cases).

### 4. Stage the simulator directory

`libttsim.so` must sit next to a `soc_descriptor.yaml`. Re-run this if `/tmp` gets
cleaned (it does, periodically):

```bash
mkdir -p /tmp/craq-sim-bh
cp /localdev/ncvetkovic/work/craq-sim/src/_out/release_bh/libttsim.so /tmp/craq-sim-bh/
cp /localdev/ncvetkovic/work/tt-metal/tt_metal/soc_descriptors/blackhole_140_arch.yaml \
   /tmp/craq-sim-bh/soc_descriptor.yaml
```

---

## Environment variables explained

| Var | Why |
|---|---|
| `TT_METAL_SIMULATOR=…/libttsim.so` | Routes tt-metal through the simulator instead of silicon. |
| `ARCH_NAME=blackhole` | The simulated chip arch. |
| `TT_METAL_DRAM_BACKED_CQ=1` + `…_DIRTY_FLUSH=auto` | Fast-dispatch path used by the craq-sim CCL baseline (see `craq-sim/MULTICHIP.md`). **Do not** use `TT_METAL_SLOW_DISPATCH_MODE` — slow dispatch is glacial here. |
| `TT_METAL_SIMULATOR_CQ_WAIT_CLOCKS=10000` | CQ polling cadence in the sim. |
| `TT_METAL_SIMULATOR_DIRECT_TENSOR_LOAD=1` | Host loads tensors straight into sim memory, skipping a slow DMA path. |
| `TT_METAL_SIMULATOR_PARALLEL_CLOCK_THREADS=16` | Parallel Tensix-tile clocking. |
| `TTSIM_HANG_WATCHDOG_CLOCKS=10000000` | Fires only on true no-progress; classifies real deadlocks. |
| `TT_METAL_DPRINT_CORES="(0,1),…,(3,2)"` | The 8 S1 flash_mla worker cores (see `FlashMLADecode.ProgramConfig.grid.BLOCKS[0]` in `flash_mla/op.py`). The op does **not** run on (0,0). |
| `TT_METAL_DPRINT_RISCVS="BR,NC,TR0"` | `hash_cb` emits from the UNPACK thread (TR0/TRISC0). |
| `TT_METAL_DPRINT_FILE=…` | Where DPRINT (and thus the hashes) lands. |

Note: `test_flash_mla_decode` uses the plain single-chip `device` fixture, so the mock
8-chip cluster descriptor + mesh-graph-descriptor are **not** needed for this repro.

---

## Reading the result

```bash
grep 'hash\[' /tmp/flashmla_dprint.log
```

Each line looks like:

```
0:0-1:TR0: hash[0x30] cb=9 tiles=16 = 0x140373bc
```

- `0` — device id
- `0-1` — core (x=0, y=1)
- `TR0` — TRISC0 (UNPACK thread)
- `0x30` — the label we passed to `hash_cb` (sdpa_output_cb)
- `cb=9 tiles=16` — circular buffer id and tile count hashed
- `0x140373bc` — FNV-1a hash of the CB's L1 bytes

There are 10 emissions per core (one per iteration). The bug = **iter-0 hash differs
from iter-1..9, which are all equal**. Example from a clean repro run:

| Core | Iter 0 | Iter 1..9 (all identical) |
|---|---|---|
| (0,1) | `0x140373bc` | `0x98ee870c` |
| (1,1) | `0xf012ccba` | `0x899f19e9` |
| (2,1) | `0x98fe0517` | `0xcaae7950` |
| (3,1) | `0xc9fbe666` | `0x9fe3a2a0` |
| (0,2) | `0x177efb9a` | `0x757075a1` |
| (1,2) | `0x2c761c59` | `0x784e3d82` |
| (2,2) | `0x4660eb74` | `0xb9a45853` |
| (3,2) | `0x39de5d6e` | `0x59e144d5` |

All 8 cores diverge iter-0-vs-rest → the bank-asymmetric `flash_mla` output bug is
present and reproducible without hardware.

To sort/group the hashes per core:

```bash
grep 'hash\[' /tmp/flashmla_dprint.log \
  | awk -F'[: ]' '{print $2, $NF}' | sort
```

---

## Sweeping / iterating

- Other decode positions (more K-chunks) are available in the test parametrize list:
  `127, 255, 383, 511, …, 32767`. Larger positions = more chunks = slower sim but the
  same hash diagnostic.
- To test a candidate fix (e.g. ZEROACC at kernel boot — Attempt 20), edit
  `flash_mla.hpp` / the LLK and re-run. The JIT cache is content-addressed, so editing
  source is enough; no host rebuild needed. A run with the bug fixed would show
  **iter-0 hash == iter-1..9** on all cores.
- The DPRINT deprecation warning in the log is benign (DPRINT still works).

---

## The alternating MLP PCC (#43563) — what's reproducible and what isn't

The headline #43563 symptom is the **alternating MLP PCC**: `test_decoder_mlp` gives
`0.99318986` for `num_internal_iterations=1` and `0.99344836` for `=2`. That number is
measured at the **decoder MLP output**, after the full downstream stack
(post-SDPA tree-reduce → matmul4/5 → all-reduce → MoE shared expert → reduce-to-one).

What craq-sim reproduces today (single chip, fast):

1. **Root cause — per-core `sdpa_output_cb` bank asymmetry.** `test_flash_mla_decode`
   shows iter-0 ≠ iter-1..9 hashes on every SDPA core (see above). This matches the
   hardware Attempt 19 observation.

2. **flash_mla final output does NOT alternate.** Reading the actual attention output
   tensor back (`torch.equal` across the 10-iteration stress loop) gives
   **`equal=True, max_diff=0`** at both `decode_position=127` (1 chunk) **and**
   `decode_position=511` (4 chunks). This is the flash-attention **scale-invariance
   cancellation** from `debug_log.md` Attempt 23: the tail `compute_sdpa_recip`
   (mm2 ÷ sum) cancels the bank-asymmetric scale factor per core, so the normalized
   output is bit-identical even though the intermediate accumulator differs.

Consequence: **the MLP-level alternating PCC cannot be reproduced at the `flash_mla` op
alone** — by construction the per-core recip cancels it. It only becomes observable once
the post-SDPA tree-reduce + matmuls + MoE break that per-core cancellation across the
multi-chunk accumulation. That requires the full `test_decoder_mlp` on the 8-chip mesh.

### Why the full `test_decoder_mlp` is currently impractical on craq-sim

`test_decoder_mlp` on the 8-chip sim stalls before producing a kernel iteration:

- **Host weight prep** (`prepare_dense_layer_weights`) is ~150 s of pure CPU torch —
  unavoidable, and unrelated to the simulator.
- **Tensor creation + `synchronize_device`** enqueues a large fast-dispatch CQ backlog
  (all decoder weights sharded across 8 chips). Draining it on the sim parks the host in
  `FDMeshCommandQueue::finish_nolock` for tens of minutes (confirmed via gdb:
  `Synchronize → finish → finish_nolock`, sim clocking at only a few kHz).

A `TT_METAL_SIMULATOR_DIRECT_TENSOR_WRITES=1` "fastwrite" path (≈5.55× speedup, pushed to
tt-metal on 2026-05-31 per `craq-sim/HANDOFF.md`) would mitigate the CQ-drain portion, but
it is **not present** in the `ridvan/nkapre-multichip-metal-v2` tt-metal base used here.

### Harness for the alternation (when the decoder run is feasible)

`test_decoder_mlp` on this branch is wired for a golden-free alternation check:
- golden torch reference is skipped (its KV-cache d2h also stalls the CQ),
- after the kernel runs, the final reduce-root MLP output is saved to
  `$CRAQSIM_ALT_PCC_DIR/decoder_mlp_out_pos{POS}_niter{N}.npy`,
- when both `num_internal_iterations` 1 and 2 outputs exist for a position, they are
  diffed directly (`niter=1` output vs `niter=2` output). They are the **same** golden,
  so any difference IS the alternation.
- parametrized over `position_id ∈ {127 (control), 511 (bug)}` × `num_internal_iterations
  ∈ {1, 2}`.

Run it (long timeout; may not finish without fastwrite):

```bash
/tmp/run_alt_pcc.sh        # see script; sets CRAQSIM_ALT_PCC_DIR + full multichip env
```

Expected once it completes: `pos=127` → niter-1 and niter-2 outputs **equal** (control);
`pos=511` → outputs **differ** (the alternation), logged as
`*** ALTERNATING OUTPUT REPRODUCED ***`.

---

## File / branch reference

- tt-metal branch: `ncvetkovic/multichip_half_dest` (base `ridvan/nkapre-multichip-metal-v2`)
- craq-sim branch: `multichip`, built with `TTSIM_MARCH=-march=x86-64-v3`
- Key commits on the tt-metal branch:
  - `900d180` — port Attempt 19 `hash_cb` to multichip `flash_mla.hpp`
  - `17cf304` — enable `DEBUG_CB_HASH` in `flash_mla/op.py` + relax PCC for hash-only run
