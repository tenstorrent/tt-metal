# README — DeepSeek-V4-Flash on ONE galaxy: ttnn prefill + ttnn traced decode (hand-off)

Branch **`sdawle/dsv4-flash-prefill-decode-1glx`** (tenstorrent/tt-metal). It merges the pure-ttnn DeepSeek-V4-Flash
**prefill** (`models/demos/deepseek_v3_d_p/tt/v4`, from branch `sdawle/dsv4-flash-prefill-ttnn`) with Sankar Manoj's
ttnn **traced decode** (`models/experimental/deepseek_v4_flash`, from branch `smanoj/ds_v4_flash`), and adds the
hand-off between them. It is all tt-metal: nothing imports tt-blaze (`blaze` is not even importable in the venv).

> **Status, 2026-09-30 00:00.** Merged, built and pushed. **The decode alone runs on device**: on host 41's firmware
> 19.11 (where its weight prefetcher cannot run) the 4-layer decode builds, replays the 128-token prompt and generates
> 32 tokens (`1 passed`, 2026-09-29 23:58, after three prefetcher-off fixes). The 43-layer decode run (first real text
> check) started at 23:58 (§5). **The prefill→decode e2e has not run yet.**

## 1. Where to run it

| | |
|---|---|
| machine | **host 41** (`10.82.97.41`), one Blackhole galaxy, 32 chips |
| checkout | `/home/ttuser/sdawle/tt-metal-dsv4-1glx` (a fresh tt-metal clone, own build and venv) |
| venv | `source python_env/bin/activate; export TT_METAL_HOME=$PWD PYTHONPATH=$PWD` |
| logs | `/home/ttuser/sdawle/e2e_runs/` (the launchers' default `E2E_LOG_DIR`) |
| weights / caches | `/mnt/tt-data/sdawle/...` (NFS `10.82.97.2:/mnt/TT-Data`, the same on hosts 29/41/42) |

Somewhere else: `git clone https://github.com/tenstorrent/tt-metal.git && git checkout sdawle/dsv4-flash-prefill-decode-1glx
&& git submodule update --init --recursive && ./build_metal.sh && ./create_venv.sh`, then
`pip install transformers==5.12.1` in `python_env`, and satisfy the per-host prerequisites below.

### Per-host prerequisites

| item | host 41 | why |
|---|---|---|
| KMD `options tenstorrent dma_address_bits=58` in `/etc/modprobe.d/tenstorrent.conf` (+ module reload) | **set** | with the default 64-bit mask every large host→device write hangs (a 1 GB upload never completed; with 58: 5.1 s). The quad29 hosts (29/48/30/31) have it; **hosts 42/43 do not** |
| Blackhole firmware bundle | **19.11.0.0** | decode's DRISC weight prefetcher needs ≥ 19.12.0.0; on older firmware the decode now turns it off by itself (§4) |
| `transformers==5.12.1` in `python_env` | installed | the decode imports `transformers.models.deepseek_v4` |
| HF checkpoint in the hub layout | `~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash-0731` → `/mnt/tt-data/sdawle/models/DeepSeek-V4-Flash-0731` | the decode tests hard-code the hub path |
| decode bf4 tile cache `DEEPSEEK_V4_CACHE_DIR=/mnt/tt-data/sdawle/dsv4_sankar_cache` | layers 0-3 only | the first 43-layer build converts every weight (~85-97 s per layer measured cold, so ~1 h+) |
| prefill tile cache `PREFILL_TTNN_CACHE=/mnt/tt-data/sdawle/dsv4_flash_ttnn_cache` | `…_bh_32dev/8x4` only (282 GB); **no `4x4`** | the first prefill build on the 4x4 submesh converts every weight (unmeasured). It can be built on another galaxy in parallel (shared NFS; the prefill does not use the prefetcher) |

Never force `TT_METAL_ENABLE_BLACKHOLE_DRAM_PROGRAMMABLE_CORES=1` on firmware 19.11: tt-metal documents a collision with
the syseng firmware on a DRAM core (`tt_metal/llrt/firmware_capability.cpp`, #45751).

## 2. How to run

All from the checkout root on host 41. Each launcher resets the galaxy (`tt-smi -glx_reset_auto`), opens the `(8,4)`
TORUS_XY mesh as a gate (resetting up to 4 times: after a reset host 41 sometimes cannot map the mesh, "Graph specified in
MGD could not fit"), then runs the pytest under a timeout and prints the key lines.

**a. The decode alone** (next step; §5):
```bash
TAG=demo STAGES="A B" models/experimental/deepseek_v4_flash/tests/decode/run_decode_demo_galaxy.sh
```
Stage A = 4 layers (SWA, SWA, CSA, HCA), 32 new tokens, ~2.5 min on a warm cache; stage B (only if A passes) = all 43
layers, 64 new tokens, >1 h the first time. Or the test directly:
`DEEPSEEK_V4_DSPARK=0 DEEPSEEK_V4_DECODE_LAYERS=4 DEEPSEEK_V4_MAX_NEW_TOKENS=32 DEEPSEEK_V4_CACHE_DIR=… pytest -s models/experimental/deepseek_v4_flash/tests/decode/test_full_model_decode_demo.py -k tp4_32chip`.

**b. The e2e** (after a; §5):
```bash
TAG=e2e NEW=64 PLEN=1000 COMPARE=decode models/experimental/deepseek_v4_flash/tests/prefill/run_ttnn_prefill_decode_e2e.sh
```
Knobs: `NEW` (new tokens), `PLEN` (prompt tokens, < 2048 in total with `NEW`), `COMPARE` = `decode` (replay the same
tokens through the decode, read its caches back, per-layer PCC vs our hand-off) | `sankar` (Sankar's own prefill on rows 2-3
as the reference) | unset (no check), `BUDGET` (timeout s).

**c. Useful env knobs**: `DEEPSEEK_V4_PREFETCHER` (unset = auto, `0`/`1` pin it), `DEEPSEEK_V4_DSPARK=0` (no MTP link;
required on 19.11), `DEEPSEEK_V4_DECODE_LAYERS=N`, `DEEPSEEK_V4_MAX_NEW_TOKENS`, `PREFILL_FORCE_EXPERT_DEQUANT=1`
(re-dequantize experts even on a warm prefill cache).

## 3. What the e2e does (`tests/prefill/test_ttnn_prefill_decode_e2e.py`)

One process, one `(8,4)` mesh (`FABRIC_2D_TORUS_XY`, 2 command queues, `l1_small_size=1152`):

| rows | what |
|---|---|
| 0-1 | Sankar's traced decode, TP4, two 1x4 stages (layers 0-21 / 22-42) |
| 2-3 | idle, or Sankar's prefill when `COMPARE=sankar` |
| 4-7 | our ttnn prefill on a `(4,4)` submesh (SP4, TP4, EP 16 experts/chip) |

Steps: tokenize → build the decode (+ `prepare_static_decode`, session, a throw-away step that captures the traces) →
build our prefill on the submesh (`kv_only_last_layer=True`, `use_trace=False`) → prefill `T0 = floor((len-1)/128)*128`
tokens → **hand-off** (`tt/prefill/handoff_ttnn_prefill.py`) → replay the ragged tail `T0..len-1` through
`decode_traced` → generate.

The hand-off reads the prefill's per-layer WORKING state (bf16/fp32) and writes the buffers `prepare_static_decode`
allocated: `sliding_carry` → the 128-row window ring; CSA / HCA `compressed_kv[:entry_count]` → entry rows `128 + w`
(HCA through the session's page table); CSA `prior_c` last 4 rows → `prev_kv` and `prev_gate` with the position bias
subtracted (decode adds it inside `csa_pool_window`). **Limit: under 2048 tokens of conversation** — the lightning indexer's
state is not handed over yet (the hand-off raises past it).

Why partitioned rows instead of both models on all 32 chips: the decode's prefetcher rings are permanent L1 (288 KB per
receiver core) on every decode chip, and the prefill is L1-tight. Analysis: tt-blaze postmortem DS4F-0282.

## 4. What is done (commits on the branch, oldest first)

| commit | what | verified |
|---|---|---|
| `6038f1e42a8` | merge of `smanoj/ds_v4_flash`: 30 conflicts + 7 breakages outside the markers (sparse_sdpa sink shape, indexer weights layout, `SiluClamped`→`ClampedSiluGlu` with a silent `Silu` fallback removed, runner trace assert, `prefill_chunk` kwarg) | builds |
| `4bf2909167d` | hand-off module, e2e test; the prefill skips the host expert dequant when its cache is complete (was ~27 min of a warm load) | host only |
| `50ba89a634f` | decode without the DRISC prefetcher where the firmware cannot run it (`is_tensor_prefetcher_supported`), `DEEPSEEK_V4_PREFETCHER`; the CSA indexer no longer hard-wired to it | **device: every layer builds** |
| `37ff4948dc9` | `COMPARE=decode` check, e2e launcher, this README (first version) | host only |
| `b3fd4b277db` | prefetcher off: re-replicate ONE copy of an activation replica from another core grid (`LinearDecode.to_replicated_rm_hs_activation`) | **device: layer 0 attention runs** |
| `172db491b2a` | prefetcher off: router gate + shared expert stay on `LinearDecode` hub mode (`hub_mode`) instead of falling back to `ttnn.linear` | **device: the 4-layer decode runs end to end** (prompt replay + 32 tokens) |
| (this commit) | this README; `tests/decode/run_decode_demo_galaxy.sh` | — |

## 5. Run history on host 41 (all measured)

| run | reached | stopped by | fix |
|---|---|---|---|
| e2e run1 | decode weight upload | hang on the 1 GB embedding write: KMD 64-bit DMA mask | `dma_address_bits=58` on host 41 (DS4F-0283) |
| e2e run2 | mesh open | "Graph specified in MGD could not fit" after a reset | launchers gate on a mesh open |
| e2e run3 | decode build, 17 s in | `DRAM-sender GlobalCircularBuffer requires programmable DRAM cores … firmware >= 19.12.0.0` | `50ba89a634f` (DS4F-0284) |
| decode A try1 | all 4 layers built (cold, ~85-97 s/layer); first step, layer 0 `kv_proj` | `Number of shards along height 512 must not exceed number of cores 16` | `b3fd4b277db` |
| decode A try2 | layer 0 attention + hyper-connection; layer 0 MoE router | `per_core_M must be greater than 0` (`ttnn.linear` fed the hub-mode replica) | `172db491b2a` |
| decode A try3 | **PASS**: 4 layers (warm cache, ~1.5 s/layer), 128-token prompt replay, 32 generated tokens, `1 passed` in 66 s | — | the text is meaningless with 4 of 43 layers, and 204 tok/s for 4 layers is not a model number |
| decode B | started 2026-09-29 23:58: all 43 layers, 64 new tokens; converts layers 4-42 cold first (~1 h) | running at hand-off | see §5a |

Logs: `/home/ttuser/sdawle/e2e_runs/{run1,run2,run3}.log`, `demoA_try1.log`, `demoA_try2.log`, `demoA.log` (try3),
`demoB.log` (the 43-layer run), `demo.out` (the launcher's summary lines).

### 5a. Running on host 41 at hand-off

The 43-layer decode run (stage B of `/home/ttuser/sdawle/e2e_runs/run_decode_demo.sh`, the pre-repo copy of
`tests/decode/run_decode_demo_galaxy.sh`), started 2026-09-29 23:58, launcher PID 684208 (its own session / process group).
Check it: `tail -f /home/ttuser/sdawle/e2e_runs/demo.out` and `grep -a -E "Layer [0-9]+ is|GENERATED|decode throughput|passed|failed|TT_FATAL" /home/ttuser/sdawle/e2e_runs/demoB.log`.
It builds the 43-layer bf4 cache, so let it finish if you can. To stop it: `kill -- -684208` (the whole process group),
then `sudo tt-smi -glx_reset_auto`. Do not `pkill -f` with a pattern (it matches your own shell).

## 6. What is next, in order

1. ~~Decode alone, 4 layers~~ — **done** (try3, above).
2. **Decode alone, 43 layers** (running at hand-off, §5a; else `STAGES="A B"`): coherent text on the default prompt, and
   tok/s (not comparable with Sankar's prefetched numbers). If it fails, expect a prefetcher-off blocker like the three
   fixed so far -- the prefetcher-off decode path was never run by its authors ("always on"). The rule that has held: keep
   every projection a `LinearDecode` with the prefetched path's layout, only without the GCB (`hub_mode`). Watch L1 (the
   DRAM→L1 weight copies are L1 the prefetched path does not use). Alternative: flash firmware ≥ 19.12 and run with the
   prefetcher (needs the pod owner's approval).
3. **Our prefill on the rows 4-7 4x4 submesh under TORUS_XY**: 4 layers first (golden PCC ≥ 0.9997), then 43 (builds the
   `4x4` cache; watch per-chip DRAM). Not scripted yet: the e2e's `_build_ttnn_prefill` builds exactly this; the existing
   device tests (`models/demos/deepseek_v3_d_p/tests/pcc/test_v4_transformer.py`, `test_v4_kv_table_device.py`) run
   on 2x4 / 8x4 with `FABRIC_2D`.
4. **e2e with `COMPARE=decode`** at `PLEN=1000`: the per-layer hand-off PCC is the first correctness gate; then coherent
   text; then ≥ 3 consecutive requests without a reload. Record TTFT, hand-off seconds, decode tok/s.
5. **Past 2048 tokens**: hand over the indexer (`idx_key_cache` from the prefill's un-rotated `index_k`, `comp_kv`, the
   indexer compressor windows from `prior_i` minus the bias); then 16k-128k.

Estimate for 1-4: ~4-5 h of wall clock if nothing else breaks, 1-2 working days realistically (every failure costs a
model-load cycle; the two cold cache builds are sequential on one galaxy).

## 7. Known limits and risks

- ≤ 2048 tokens of conversation until step 5.
- Prefetcher-off decode is slower than Sankar's measured numbers and uses more L1 per step; run on device for layers 0-3
  only so far (layers 4-42 are the same four kinds, but their L1 at 43 layers is unmeasured).
- Our prefill has never run on a 4x4 submesh, with SP4, or under TORUS_XY (it ran on the full 8x4 with `FABRIC_2D`).
- The hand-off is unvalidated on device (`COMPARE=decode` is the check).
- The prefill still allocates its old KV export caches (tt-blaze ring format, `tt/v4/kv_contract.py`); the hand-off does
  not use them. Dropping them frees DRAM on the prefill chips.
- `tt/v4/weights/{dequant,layer_weights}.py` are copies of tt-blaze's loaders (tt-metal must not import blaze).

## 8. Record

Every defect, fix and refuted hypothesis: tt-blaze `docs/deepseek_v4_flash_postmortem.md` (branch
`sdawle/dsv4-flash-prefill-ttnn`): DS4F-0282 (the merge and the co-existence analysis), DS4F-0283 (DMA mask),
DS4F-0284 (prefetcher firmware floor). The decode-bring-up fixes above are to be recorded as DS4F-0285+.
