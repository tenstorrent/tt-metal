# Qwen3-VL-4B e2e on Quasar

One pytest (`test_qwen3_vl_e2e.py`) runs Qwen3-VL-4B-Instruct cut down to V vision blocks, T text layers and K decode steps, and compares every stage against the same cut-down HF model on the host. Three wrapper scripts run it on WH/BH (ttsim or hardware), craq-sim and the emu-quasar-2x3 emulator with identical settings.

Ops that work on WH/BH but not on Quasar, simulator issues and model-copy bugs found along the way are logged in [QUASAR_GAPS.md](QUASAR_GAPS.md).

## Run

```bash
source python_env/bin/activate
export TT_METAL_HOME=$(pwd) TT_METAL_RUNTIME_ROOT=$(pwd) PYTHONPATH=$(pwd):$PYTHONPATH

E2E=models/experimental/ops/quasar/qwen3_vl/tests/e2e
export QWEN_CRAQ_SIM=/localdev/$USER/quasar-sim/libttsim.so  # the staged craq-sim build; run_craq.sh's default is an old copy
$E2E/run_wh_bh.sh --ttsim wh           # WH baseline on ttsim, emulator-sized grid
$E2E/run_wh_bh.sh --grid native        # WH/BH hardware, full grid
$E2E/run_craq.sh                       # craq-sim, emulator-sized grid
$E2E/run_craq.sh --grid 8x4            # craq-sim, full grid
$E2E/run_emu.sh                        # emulator (needs NNG_SOCKET_ADDR from your IRD reservation)
```

The emulator backs only 64 MiB of DRAM per channel although its descriptor says 1 GiB; larger allocations silently overwrite earlier ones, and the e2e (~1.3 GiB of weights) does not fit (QUASAR_GAPS S7, #55417).

Script defaults: `--size tiny --vision-layers 2 --text-layers 2 --decode-steps 1 --deepstack-at 0 --debug default` (`--debug fast` on craq-sim, where the watcher slows each op 150-300x).

| Flag | Meaning |
|---|---|
| `--size tiny\|demo` | tiny: 256x256 image, 64 image tokens, 78-token prompt padded to 128. demo: the graph-capture sizes (2752 image tokens, 4096 prefill). |
| `--vision-layers N`, `--text-layers N` | blocks of the 24-block vision tower / 36-layer text model to run |
| `--decode-steps K` | teacher-forced decode steps after prefill (0 = prefill only) |
| `--deepstack-at I\|real` | move the deepstack tap to vision block I (in TT and HF alike); `real` keeps 5/11/17 |
| `--kv-blocks N` | paged KV-cache blocks of 32 tokens (default: preset) |
| `--host-ops a,b\|all` | run these ops on the host (bisecting only; the run becomes DIAGNOSTIC) |
| `--disable-wa a,b` | turn off named workarounds from `op_overrides.py` |
| `--debug fast\|default\|deep` | fast: no watcher. default: watcher + LLK asserts, NoC sanitize off. deep: plus dump-all, noinline, per-RISC DPRINT, debug logging |
| `--noc-sanitize` | re-enable NoC sanitize (20-30x slower) |
| `--fp32-dest-acc` | keep fp32 dest accumulation in the Quasar config (hardware A/B; undefined on ttsim WH) |
| `--timeout S` | pytest timeout (WH/BH and emulator 14400, craq-sim 10800) |
| `--ttsim wh\|bh`, `--grid 2x3\|native`, `--config quasar\|native` | `run_wh_bh.sh` only; `--config native` runs the unmodified bf8 config as a hardware control |
| `--grid 2x3\|8x4` | `run_craq.sh` only |
| `-- <args>` | passed to pytest, e.g. `-- --qwen-dump-stages` to save golden and TT stage tensors, or `-- --qwen-allow-uncertified` to use host fallbacks not yet checked against the real op (`test_fallbacks.py`) |

Exit codes: 0 PASS, 3 DIAGNOSTIC (some ops ran on the host), anything else FAIL or error.

## Output

Each run writes `generated/qwen3_vl_quasar/<target>/<UTC time>/`:

| File | Content |
|---|---|
| `pcc.md` | verdict, first failing stage, per-stage PCC / threshold / max abs error / seconds, workaround and host-op hit counts |
| `verdict.txt` | `PASS`, `FAIL` or `DIAGNOSTIC` |
| `progress.log` | one `PRE`/`POST` line per ttnn op with shapes, memory configs and model stage. After a hang, the last `PRE` without a `POST` is the op that never finished. |
| `run.log`, `command.txt`, `env.txt`, `git.txt` | full output, exact command, environment, commit and branch commits |
| `ops.pt` | with `-- --qwen-capture-ops STAGE_PREFIX` (e.g. `vision.block0`, `text.decode`): every top-level op output of those stages. `repros/diff_ops.py RUN_A RUN_B` prints where two runs first diverge |
| `prefill_snapshot.pt` | KV cache, decode position, rope delta, page table and prefill stage tensors at the end of prefill. `-- --qwen-resume-prefill <run folder>` skips vision and prefill and decodes from it (same size, layers, grid and config only; the run notes it in `pcc.md`) |

Thresholds per stage and preset live in `thresholds.json`.

`progress.log` also records each op's program, compute and memory configs, so a failing call can be replayed in a repro. `-- --qwen-vision-only` stops after the vision tower (vision stages only), for short runs on slow targets.

## Grids

The emulator build exposes a 2x1 compute grid (`tt_metal/core_descriptors/quasar_simulation_2x3_arch.yaml`). `--grid 2x3` reproduces it elsewhere with `TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE`, which sets the grid's *end* coordinate, so the value depends on where each target's compute grid starts:

| Target | Native grid | Override for 2x1 |
|---|---|---|
| ttsim WH | 8x8 | `1,0` |
| ttsim BH | 11x10 | `1,0` |
| craq-sim | 8x4 | `3,2` |
| emu-quasar-2x3 | 2x1 | none |

## ttsim setup (WH/BH baselines)

```bash
for a in wh:wormhole_b0_80_arch bh:blackhole_140_arch; do
  n="${a%%:*}"; y="${a##*:}"; d="/localdev/$USER/ttsim/sim_${n}"
  mkdir -p "$d" && cp "/localdev/$USER/ttsim/src/_out/release_${n}/libttsim.so" "$d/" \
    && cp "tt_metal/soc_descriptors/${y}.yaml" "$d/soc_descriptor.yaml"
done
```

Override locations with `QWEN_TTSIM_DIR`, `QWEN_CRAQ_SIM`, `QWEN_EMU_DIR`. `run_craq.sh` defaults to `/localdev/$USER/sim/libttsim.so`, an old craq-sim copy (2026-10-01); every craq-sim run recorded here used `QWEN_CRAQ_SIM=/localdev/$USER/quasar-sim/libttsim.so`, except the 2026-10-28 2x1 runs, which used a build of craq-sim#538 before it was merged and staged.

On WH/BH the vision-token merge copies rows on the host: `QuasarModelArgs.device_scatter` is on only on Quasar. No simulator can check the device path there yet: ttsim WH aborts on scatter's int32 index (QUASAR_GAPS S3, a ttsim gap, not a WH bug) and ttsim BH aborts at the first op (tt-blaze#5898). The upstream demo (`models/demos/qwen3_vl`) scatters on device on WH.

## Hangs

A hang shows up as the pytest timeout. The script prints the last lines of `progress.log`; the watcher log has per-core waypoints (default/deep profiles). On simulators `TT_METAL_OPERATION_TIMEOUT_SECONDS` is ignored (`tt_metal/llrt/llrt.cpp`, `wait_until_cores_done`). For a hard-to-localize hang, temporarily patch it locally to honor that variable when set; do not commit that change.

## Graph-capture op tests (`qwen3_vl_ops`)

`models/experimental/ops/quasar/tests/qwen3_vl_ops/` (on main) replays every distinct call of a WH N150 demo capture: demo sizes, bfp8/bfp4 dtypes and WH program/shard configs, so some cases do not fit Quasar as captured. On craq-sim we run it with `TTNN_GRAPH_OPS_BF8_AS_BF16=1` (block-float cases run as bf16), one pytest invocation per case (a craq-sim abort or hang loses only that case), and `test_add.py` / `test_multiply.py` a second time through `ttnn.experimental.quasar.add` / `.multiply` (the base ops are not ported to Quasar; that is expected, not a finding).

| Date | Target | Cases | Pass | Skip (captured config too large for the device) | Fail | Timeout | Wall time |
|---|---|---|---|---|---|---|---|
| 2026-10-26 | craq-sim 8x4 (.so md5 74c8eb33), branch on main fb6d34ceaed | 134 (+12 experimental add/mul) | 94 (+10) | 23 (+2): 8x5 L1 shard grids, 12-bank DRAM sharding | 16 | 1 | 1 h 45 min (95 min in tests; 12-50 s per case) |
| 2026-10-26 | craq-sim 8x4, main 5285b13d773 with the DFB drain (no #57646 revert, no tensor_layout droppable) | 134 (+12) | 94 (+10) | 23 (+2) | 16 | 1 | 1 h 34 min (88 min in tests); every case's outcome identical to the run above |

The 16 failures: base add (7) and multiply (3), all passing through the experimental ops; `linear` with a captured 8x8 matmul grid (2); `plus_one` int32 and uint32 (2, Q10); `scatter` (1, Q1); `layer_norm` `[2752, 4096]` at PCC 0.998996 against 0.999 (1). The timeout is prefill SDPA `[16, 12288, 64]` (more than 20 min on craq-sim; the `[32, 4096, 128]` case passes in 13 min). Per-case results and times: `generated/qwen3_vl_ops_sweep_1026/summary.md` and `results.tsv` on the run host.

### e2e ops vs graph-capture cases

Compared on 2026-10-27: every distinct call in `progress.log` of the craq-sim 8x4 and 2x1 e2e runs (`craq_8x4/20261026T220230Z`, `craq_2x3/20261026T164502Z`) against every `CASES` entry, by op, input shapes, dtype (bf8/bf4 counted as bf16) and memory layout. `concat` was not compared (the log records its list argument only as `list`).

| | Graph-capture tests | e2e |
|---|---|---|
| Source | one WH N150 demo run: 24 vision blocks, 36 text layers, 200 decode tokens x 2 | 2 vision blocks, 2 text layers, 1 decode step |
| Inputs, check | random inputs vs a torch golden per op | real weights, image and prompt; each stage vs truncated HF |
| Tokens | vision 11008 (padded to 12288), prefill 4096, KV 1024 blocks | vision 256, prefill 128, KV 8 blocks |
| Prefill tensor shape | chunked: `[1,4,1024,…]`, `[1,6,2048,…]`, `[1,12,1024,…]` | flat: `[1,1,128,…]`, `[1,1,256,…]` |
| Dtypes | bf8/bf4 weights and KV cache (bf16 with `TTNN_GRAPH_OPS_BF8_AS_BF16=1`) | bf16 everywhere |
| Memory and program configs | WH: DRAM-sharded weights, L1 width-sharded decode activations, 8x8/8x5 grids | DRAM-interleaved weights; configs clipped to the device grid (Q3, G1-G5) |
| Compute config | dropped (op default) | HiFi4, bf16 dest acc |
| add, multiply, SDPAs | base ops | `ttnn.experimental.quasar.*` |

**Same** (shapes, dtype and memory layout), all on the decode path, whose 32-row batch padding makes the shapes size-independent: `rms_norm` width-sharded `[32,2560]` (00, 03; same sharded program config), Q/K norms `[32,128]`/`[8,128]` (01, 02), final norm (07); decode `rotary_embedding_llama` (00, 01); `nlp_create_qkv_heads_decode`, `nlp_concat_heads_decode`; decode residual `add` (00, 01) and MLP-gate `multiply` (00) in L1 width-sharded; small ops (`reshape` 00-02, `to_memory_config` 00-05/10/13/16, `sharded_to_interleaved` 00, `interleaved_to_sharded` 00, `transpose`, `expand`, `__getitem__` 00, `unsqueeze_to_4D` 00/02, `to_device` 01-03).

**Close** (same weights and kernels, only token or KV-capacity dims differ): vision `layer_norm` (00-02); vision QKV/WO/FC2 `linear` (08, 09, 07) and patch-merger FC2 (11); vision bias `add` (06, 07); prefill SDPA (00 `[16,12288,64]`, which times out on craq-sim, and 01); prefill rotary, `nlp_create_qkv_heads`, `nlp_concat_heads`, `typecast`, `rms_norm`, residual `add`, `multiply`; `paged_fill_cache`, `paged_update_cache` and decode SDPA (KV cache `[8,…]` vs `[1024,…]`; the decode update and query tensors are identical); rope `embedding` (00); `slice`, `pad`, `unsqueeze`, `mesh_partition`.

**Different, same weights:**
- Decode text linears (QKV, WO, FF1/3, FF2): same shapes as linear 00-03, but the captured cases use L1 width-sharded inputs, DRAM-sharded weights and `MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig` (not brought up on Quasar, Q3/#58912); the e2e runs them interleaved with ttnn's default program.
- Text prefill: QKV and FF2 are `ttnn.linear` at 128 tokens but `ttnn.experimental.minimal_matmul` at the captured 4096 (cases 00, 01); FF1/3 and WO are `linear` in both (06, 10) but chunked, with bf4/bf8 DRAM-sharded weights.
- LM head: the e2e splits the vocabulary into grid-sized widths (`[2560, 11136|12800|1280|896]`), interleaved, with a single-core untilize of `[32,151936]` (G5); the capture uses DRAM-sharded `[2560,26720]`/`[2560,18336]` (linear 04, 05), `split`, `untilize` `[32,37984]` and `to_memory_config` 11/12.

**Only in the graph suite:**
- No case exists for the vision MLP FC1 or the patch-merger FC1, both `linear(activation="gelu")`: the generator could not rebuild the `activation` argument (`test_linear.py` "NOT generated": 72 + 12 calls). The e2e runs both (it found the Quasar gelu gap, since fixed by #57753).
- Run on the host by the e2e: token `embedding` over `[151936,2560]` (01, Q8).
- On-device sampling and position increment: `argmax`, `plus_one` (Q10); the e2e reads logits on the host and teacher-forces one step.
- Demo-size prefill only: `minimal_matmul` 00/01, chunked `reshape` 07-09/12/15/16.
- Others: `multiply` 02 on the KV cache `[1024,8,32,128]`, `interleaved_to_sharded` 01, `unsqueeze_to_4D` 01, `to_device` 00 (page table `[1,1024]`).

**Only in the e2e:** ops added by the small-grid workarounds: `untilize`/`pad`/`tilize`/`reshape` on `[32,128]`/`[1,4096]` (G1), `sharded_to_interleaved` on `[32,2560|4096|9728]` and the `to_memory_config` around unsharded linears (G4), and the LM-head splits. They grow as the grid shrinks: 1212 device ops on 2x1 against 564 on 8x4.

To make the suite reflect what Quasar runs: shorten the prefill and vision token dims (never weight dims), replace the DRAM-sharded and 8x8/8x5 decode configs with ones that fit Quasar, and add cases for the two gelu linears.

## Baselines and run times

Wall time is the first-to-last-op span from `progress.log` (setup and weight loading add a few minutes). The repro suite (`tests/e2e/repros`, as run by the checks) takes about 12 min on craq-sim 2x1. Tiny size, 2 vision / 2 text layers, 1 decode step, unless noted. The staged craq-sim `.so` with md5 74c8eb33 is craq-sim's `release_qsr` build from a928410c; `src/` is identical up to 1bfae892 (2026-10-21), so either commit names the same simulator. Since 2026-10-27 the staged build is b53e4d5c (md5 484339f9); since 2026-10-28 it includes craq-sim#538 (md5 7f4a2163, MOP inner loops up to 0x3FF, QUASAR_GAPS S8). The 2026-10-28 runs had the uncommitted #62275 fix (#62817, now #62968) in the tree; the model does not use that path (Q11).

| Target | Grid | Verdict | Wall time | Date | Run folder (`generated/qwen3_vl_quasar/...`) |
|---|---|---|---|---|---|
| craq-sim with craq-sim#538 (md5 7f4a2163), main 9c941a0cc3a; vision merge scattered on device | 2x1 | PASS (prefill 0.99698, decode0 0.99861; lowest `text.layer1.mlp` 0.99427); logits identical to the host merge | 42 min | 2026-10-28 | `craq_2x3/20261028T151400Z` |
| craq-sim (b53e4d5c, md5 484339f9), main 9c941a0cc3a; scatter on device | 8x4 | PASS (prefill 0.99681, decode0 0.99841; lowest `text.layer1.mlp` 0.99433) | 15 min | 2026-10-28 | `craq_8x4/20261028T145829Z` |
| emu-quasar-2x3, DRAM clamp bypassed (`emu-quasar-2x3-uncapped`), main 1e85fb6a52e with #62744; scatter on device | 2x1 | runs every stage (1211 device ops, scatter included; no hang or crash); FAIL on PCC from `vision.block0.attn` on, the accepted DRAM-wrap effect (S7) | 217 min ops (3 h 38 min pytest) | 2026-10-28 | `emu_2x3/20261028T020831Z` |
| craq-sim with craq-sim#538 (md5 7f4a2163), main 1e85fb6a52e; host merge | 2x1 | PASS (prefill 0.99698, decode0 0.99861; lowest `text.layer1.mlp` 0.99427); first 2x1 pass since #61421 (S8) | 42 min | 2026-10-28 | `craq_2x3/20261028T020043Z` |
| craq-sim (b53e4d5c, md5 484339f9), main 1e85fb6a52e with #62744; scatter on device | 8x4 | PASS (prefill 0.99681, decode0 0.99841) | 14 min | 2026-10-27 | `craq_8x4/20261027T212609Z` |
| craq-sim (b53e4d5c, md5 484339f9), main 1e85fb6a52e (#62543 in, old Q9 dropped) | 8x4 | PASS (prefill 0.99681, decode0 0.99841) | 14 min | 2026-10-27 | `craq_8x4/20261027T183639Z` |
| craq-sim (b53e4d5c, md5 484339f9), main c17e19d50bc; no fp32 upload cast, rms_norm with fp32 dest acc (both workarounds removed) | 8x4 | PASS (prefill 0.99681, decode0 0.99841; lowest `text.layer1.mlp` 0.99433) | 15 min | 2026-10-27 | `craq_8x4/20261027T164520Z` |
| ttsim WH (md5 e9c03997), same branch; `ttsim_wh_host_cast_fp32_upload` | 2x1 | PASS (prefill 0.99943, decode0 0.99950) | 10 min | 2026-10-27 | `wh_bh_wh_2x3_quasar/20261027T170038Z` |
| ttsim WH, without any fp32 upload cast | 2x1 | abort at the first op (fp32 -> bf16 `as_tensor`, S2) | — | 2026-10-27 | `wh_bh_wh_2x3_quasar/20261027T162408Z` |
| craq-sim (b53e4d5c), main c17e19d50bc, `--disable-wa host_cast_fp32_upload` (rms_norm workaround still on) | 8x4 | PASS (prefill 0.99732, decode0 0.99917) | 14 min | 2026-10-27 | `craq_8x4/20261027T155243Z` |
| craq-sim (md5 74c8eb33), main f288db45673 | 8x4 | PASS (prefill 0.99710, decode0 0.99915) | 13 min | 2026-10-26 | `craq_8x4/20261026T220230Z` |
| craq-sim (staged .so md5 74c8eb33), main 5285b13d773 with the DFB destructor drain (no #57646 revert, no tensor_layout droppable) | 8x4 | PASS (prefill 0.99710, decode0 0.99915) | 14 min | 2026-10-26 | `craq_8x4/20261026T213739Z` |
| craq-sim (staged .so md5 74c8eb33), main fb6d34ceaed | 2x1 | PASS (prefill 0.99727, decode0 0.99918) | 42 min | 2026-10-26 | `craq_2x3/20261026T164502Z` |
| craq-sim (same) | 8x4 | PASS (prefill 0.99710, decode0 0.99915) | 14 min | 2026-10-26 | `craq_8x4/20261026T172737Z` |
| emu-quasar-2x3, DRAM clamp bypassed (`emu-quasar-2x3-uncapped`, see the plan's open decision) | 2x1 | runs every stage (1212 device ops, no hang or crash); FAIL on PCC from `vision.block0.attn` on, the accepted DRAM-wrap effect (S7) | 203 min ops (3 h 38 min pytest; 13 min Zebu queue before it) | 2026-10-26 | `emu_2x3/20261026T154632Z` |
| emu-quasar-2x3, with the #60444 DRAM clamp | 2x1 | FAIL at weight upload: OOM (2 x ~41 MiB usable) | 4 min | 2026-10-26 | `emu_2x3/20261026T151437Z` |
| craq-sim (a928410c / staged .so md5 74c8eb33) | 2x1 | PASS (prefill logits 0.99727, decode0 0.99918) | 41 min | 2026-10-22 | `craq_2x3/20261022T134319Z` |
| craq-sim (same) | 8x4 | PASS (prefill 0.99710, decode0 0.99915) | 14 min | 2026-10-22 | `craq_8x4/20261022T142425Z` |
| craq-sim, without `quasar_rms_norm_bf16_dest` | 2x1 / 8x4 | PASS (prefill 0.99685 / 0.99687) | 42 min / 15 min | 2026-10-21 | `craq_2x3/20261021T170032Z`, `craq_8x4/20261021T175443Z` |
| craq-sim (older build) | 2x1 / 8x4 | PASS | 36 min / 12 min | 2026-10-16 | `craq_2x3/20261016T165030Z`, `craq_8x4/20261016T183306Z` |
| craq-sim (older build, 695 device ops) | 2x1 | PASS | 8 min | 2026-10-08 | `craq_2x3/20261008T161235Z` |
| emu-quasar-2x3 | 2x1 | FAIL: DRAM wraps at 64 MiB per channel (S7), logits NaN | 165 min (2 h 47 min wall) | 2026-10-08 | `emu_2x3/20261008T212107Z` |
| emu-quasar-2x3, vision only, V=1 | 2x1 | FAIL (S7) | 13-43 min | 2026-10-09 | `emu_2x3/20261009T0*` |
| ttsim WH | 2x1 / 8x8 / 8x4 | PASS | 10 / 9 / 10 min | 2026-10-16 | `wh_bh_wh_2x3_quasar/20261016T180210Z`, `wh_bh_wh_native_quasar/20261016T18*` |
| WH N150 silicon | 8x9 | PASS (all stages >= 0.9993), with the #59510 fix | — | 2026-10-06 | aus-wh-08 `wh_bh_hw_native_quasar/20261006T182734Z` |

The 2026-10-08 craq-sim run logged 695 device ops against 1212 from 2026-10-16 on (more of the model runs on device now), so its 8 min is not comparable with today's 41 min. Blackhole (ttsim and hardware) is not part of the baseline.
