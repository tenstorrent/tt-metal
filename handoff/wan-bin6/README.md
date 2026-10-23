# Handoff: Wan2.2-T2V-A14B on a Bin6 (7-bank) BH Galaxy

Branch: `jameslee/test-pr60190-dram-endpoints` (`main` @ `7086ed4c7a0` + the two commits of PR #60190, cherry-picked).
Machine: `bh-glx-120-b10u08` (single BH Galaxy, 32 Blackhole chips).
Written 2026-10-14, updated 2026-10-15 (TDP cap results), by a Claude Code session for jameslee@tenstorrent.com.

This directory is test scaffolding for an experiment branch. It is not meant to merge.

## TL;DR

1. **Issue #59504 (OOM at Wan pipeline creation on Bin6) reproduces here, byte for byte.** Chunked VAE decode (`vae_t_chunk_size=7` for the BH 4x8 preset) makes pipeline creation succeed.
2. **On `main`, Wan output on this machine is garbage** (flat gray / static frames, CLIP ≈ 19.4). **PR #60190 fixes it** (CLIP 37.07 at 480p). Likely cause: CCL kernels compute remote DRAM addresses with the *local* chip's bank-to-core map, and this Galaxy mixes all 8 DRAM harvest masks.
3. **720p + chunked VAE on the PR branch hung every time without a power cap: 17 of 17 runs** (plus 1 of 3 at 480p). Every device-side hang that triage could read (14) has the straggler on **device 9 (T2L2)**, with the **same physical core, `4-10` (logical `(3,8)`)**, as the single outlier, across ring SDPA and matmul, `tt-smi -r` and `-glx_reset`, and with `TT_MM_THROTTLE_PERF=5`. In the last of those hangs, all 32 chips stayed on PCIe (no `rev ff`), so no chip or tray dropped out.
4. **New (2026-10-15): `TT_METAL_TDP_LIMIT_WATTS=130` stops the hang.** With the firmware TDP limit at 130 W on all 32 chips, **4 of 4 runs finished without hanging**. This is strong evidence that the hang is power-related (power delivery, or a marginal core under full power), not a pure logic bug.
5. **New: with the hang gone, the output is wrong in a different way (open).** Same prompt and seed (42), yet every run gives a **different video**: different art style, lighting and set, and in run 3 cats that vanish and teleport (CLIP 25.3). On top of that, **flat mid-grey rectangles** (decoded zeros) appear in almost every run, **always in the VAE shard at mesh coordinate (3, 6)** (pixels x 960–1120, y 540–720), sometimes also (0, 6) / (0, 5). CLIP passes in 2 of 4 runs. See finding 6.

## Your task

The hang now has a workaround (`TT_METAL_TDP_LIMIT_WATTS=130`). The open question is the **wrong, non-reproducible output** that shows up once the run gets through:

1. **Run with the cap:** `SKIP_VBENCH=1 TT_METAL_TDP_LIMIT_WATTS=130 hang_loop.sh 3 720p chunk`, then `find_grey_patches.py` on each `run_NN/*.mp4`. Is it still always shard (3, 6)?
2. **Is it power or software?** Try a middle cap (e.g. 200 W) and a lower one (e.g. 100 W). If the garbage gets worse as the cap rises and disappears at a lower cap, it is the same marginal-power problem in a milder form (silent corruption instead of a hang). If it does not change, it is a software bug that the hang was hiding.
3. **Find the physical chip behind VAE shard (3, 6)** (and (0, 6)). Map the mesh coordinate to a chip ID / tray / PCI address (e.g. `mesh_device.get_device_id(ttnn.MeshCoordinate(3, 6))` on an open mesh). If it is device 9 (T2L2), the hang and the patches are the same bad chip.
4. **Check reproducibility on `main`'s math:** run the 480p preset (no chunking) 2–3 times with the cap and compare videos. 480p does not use the patched chunk size, so it separates the chunked-VAE path from the rest.
5. Still worth doing: rebuild `main` and compare; stress device 9 alone; report the flaky ethernet links and device 9 / core `4-10` to whoever owns this machine.

## Machine facts (verified)

- **Bin6 DRAM:** every chip has 1 of 8 DRAM channels harvested. `device.dram_grid_size()` = `7x1`. Harvest masks are **mixed**: each of `1, 2, 4, …, 128` appears on 4 chips (`check_links.py --dram`). UMD warns "inconsistent DRAM harvesting information" at startup; that warning is expected here.
- **Firmware bundle 19.13.2.** PR #60190 notes that with this firmware no DRAM subchannel is free on every harvest case. This affects its second commit (DRISC core), not this test.
- **No X-axis wraparound cabling.** None of the 8 four-chip rows has a link between its end chips (e.g. T1L1↔T1L4). Of the single-Galaxy descriptors, only `single_bh_galaxy_torus_y` opens; `torus_x` and `torus_xy` fail placement. On a Galaxy, `FABRIC_1D_RING` always requests `TORUS_XY` (`tt_metal/fabric/fabric_host_utils.cpp:58-60`). So the issue's exact config (`4x8sp1tp0nl2_ring_is_fsdp0`) **cannot run here**. Use `4x8sp1tp0nl2_linear_is_fsdp0`. It has the same `is_fsdp=False` weight placement, and it hits the identical OOM.
- **Reset with `tt-smi -glx_reset`, not `tt-smi -r`.** On this Galaxy `-r` warns that it needs CPLD FW v1.16+. It may not fully reset the chips. `-glx_reset` does an IPMI power reset of all 32 chips (about 1–2 minutes). All 7 device-9 hangs happened after `-r`. The first run after a `-glx_reset` (480p) did not hang.
- **Ethernet links are flaky across `-r` resets.** Most of them leave a different set of links down. Device 9 (T2L2) failed to train its ports 0/1 to T1L2 on 2 consecutive resets. Always run `check_links.py` after a reset. A neighbour pair with **zero** links makes the 8x4 mesh unbuildable, so reset again.
- **The machine is shared.** Another user (apaola) ran fabric jobs here earlier. Two jobs on the chips at once caused a Bus error and `Sysmem mapped at unexpected NOC address`. Before every reset or run, check `ps -eo user,cmd | grep -E "[p]ytest|[t]trun|[t]t-smi"`.

## Setup

```bash
cd ~/tt-metal && source python_env/bin/activate
# VBench quality gate (pinned set, --no-deps as the requirements file says). Already installed in jameslee's python_env.
uv pip install --no-deps -r models/tt_dit/utils/vbench-requirements.txt
```

`run_wan.sh` sets the rest: `TT_MESH_GRAPH_DESC_PATH`, `NO_PROMPT=1`, `TT_DIT_CACHE_DIR`, `VBENCH_CACHE_DIR`, `TORCH_HOME`, and `--timeout 2100`. pytest.ini's default 300 s timeout is too short for model load.

VBench cache: CI uses `/mnt/models/vbench`, but on this machine it holds no model files and is read-only, so VBench fails with "Permission denied". The scripts default to `~/.cache/vbench`, where VBench downloads its models on first use. VBench also shells out to `unzip` (not installed here) for the RAFT model. If you hit `FileNotFoundError: 'unzip'`, extract it by hand: `cd ~/.cache/vbench/raft_model && python -I -c "import zipfile; zipfile.ZipFile('models.zip').extractall('.')"`. jameslee's `~/.cache/vbench` is already populated. VBench reads only the `.mp4`, so you can score a saved video on CPU without the board: `assert_vbench_quality(path, prompt=..., thresholds=...)` from `models/tt_dit/utils/vbench.py`.

### Switching builds

Python loads the *installed* copies (`build/lib/*.so`, `ttnn/ttnn/_ttnn.so`). `cmake --build build` alone does not update them. Use:

```bash
git switch <branch>
cmake --build build -j$(nproc) && cmake --build build --target install   # incremental ≈15 min
strings build/lib/libtt_metal.so | grep -c harvested_dram_views           # >0 means the PR #60190 build
tt-smi -glx_reset                                                          # required after switching (PR #60190 notes)
python -I handoff/wan-bin6/check_links.py                                  # no output means all links trained
```

Git on this machine has no identity configured. Pass `-c user.name=… -c user.email=…` per command; don't change the global config.

## Running

```bash
handoff/wan-bin6/run_wan.sh 720p chunk   run.log     # 720p, vae_t_chunk_size=7 via vae_chunk_patch.py
handoff/wan-bin6/run_wan.sh 720p nochunk run.log     # reproduces the #59504 OOM
handoff/wan-bin6/run_wan.sh 480p nochunk run.log     # fits without chunking
```

Healthy timeline (720p): mesh up in about 30 s; weight load from cache in about 30 s (first run without a cache: about 7 min); `Pipeline allocation run...` (warmup: 2 denoise steps + VAE decode) in about 20–70 s; 40 denoise steps at about 4.9 s each; then decode, CLIP, VBench. **If the log stops for more than 2–3 minutes after `Pipeline allocation run...`, it has hung.** pytest will not notice until the 2100 s timeout.

### When it hangs

```bash
python tools/tt-triage.py --llm-output --disable-progress --skip-version-check > triage.txt   # while the process is alive
```

- `dump_op_mesh.py`: an 8x4 table showing which op each device is in. The straggler is the device(s) still in an earlier op, marked `[!]`.
- `dump_callstacks.py`: per-core callstacks. Group the straggler's cores by callstack; the core whose stack differs from all others is the one to look at. Take two snapshots minutes apart (`--run=dump_callstacks`). If that core's program counters are identical in both, it is frozen, not polling.
- If every device shows `(idle)`, the hang is on the host. Attach `gdb -p <pid> -batch -ex "thread apply all bt 15"`.

Then kill the test (`kill <python pid>`; don't use a `pkill -f` pattern that also matches your own shell), `tt-smi -glx_reset`, and run `check_links.py`.

## Results so far

All runs use `4x8sp1tp0nl2_linear_is_fsdp0` with `TT_MESH_GRAPH_DESC_PATH=single_bh_galaxy_mesh_graph_descriptor.textproto`.

| Build | Res | VAE chunk | Result |
|---|---|---|---|
| main | 480p | none (preset) | ran 40 steps; output flat gray (pixels 124–129), CLIP 19.42 → FAIL |
| main | 720p | none (preset) | **OOM at pipeline creation**, identical to #59504 |
| main | 720p | 7 | ran 40 steps; output static (7/8 frames CLIP 19.42), CLIP 19.41 → FAIL |
| PR #60190 | 480p | none (preset) | ran 40 steps; pixels 0–255, **CLIP 37.07 → PASS**; then failed only because `vbench` was not installed (now installed) |
| PR #60190 | 720p | 7 | **hang** at denoise step 14/40: device 9 stuck in ring SDPA, core `4-10 (3,8)` compute frozen |
| PR #60190 | 720p | 7 | **hang** after warmup: all devices idle, host waiting for a completion-queue event that never arrived |
| PR #60190 | 720p | 7 | **hang** at denoise step 1/40: device 9 stuck in MinimalMatmul, odd core again `4-10 (3,8)` |
| PR #60190 | 480p | none (preset) | after `tt-smi -r`: **hang** at denoise step 15/40, device 9 matmul, odd core `4-10 (3,8)` |
| PR #60190 | 480p | none (preset) | after **`tt-smi -glx_reset`**: **no hang**. Video written; frames show the prompt (two cats boxing on a spotlit stage, moving). CLIP 36.81 PASS. VBench PASS (scored offline): subject 0.9535, background 0.9727, motion smoothness 0.9789, dynamic degree 1.0, imaging 0.5678. The in-test VBench failed only on the read-only cache |
| PR #60190 | 720p | 7 | `hang_loop.sh` ×5 (`tt-smi -r` before each run): **5/5 hang**, all on device 9, odd core `4-10 (3,8)` every time: SDPA after step 1, SDPA after step 2, matmul in warmup, SDPA after step 4, matmul after step 2 |

| PR #60190 | 720p | 7 | `hang_loop.sh` ×3 with **`tt-smi -glx_reset`**: **3/3 hang**: device 9 matmul in warmup (odd core `4-10`); 2× host-side (all devices idle, host waits on a completion event; in one, triage could not open chip 0: "No such device") |
| PR #60190 | 720p | 7 | `hang_loop.sh` ×3 with `-glx_reset` + **`TT_MM_THROTTLE_PERF=5`** (verified: 19 kernel builds with `MM_THROTTLE 5`; warmup steps 10.6 s vs 4.5 s): **3/3 hang**, all on device 9 in RingJointSDPA, odd core `4-10 (3,8)` every time |

| PR #60190 | 720p | 7 | `-glx_reset`, then `lspci -d 1e52: \| grep "rev ff"` sampled 4× during the hang: **hang** after warmup step 1, **all 32 chips still on PCIe**, no `rev ff` (no tray or chip dropped). About 3.5 min into the hang the test died with `Fatal Python error: Bus error` (host MMIO access failed), before triage could read the cores. Reports in `~jameslee/wan_pcie_check/` |
| PR #60190 | 720p | 7 | **`TT_METAL_TDP_LIMIT_WATTS=130`** (log confirms "firmware TDP limit on chip N is now 130 W" for all 32), `hang_loop.sh`: **no hang in any run**, 40 steps at about 3.6 s each, about 4 min per run with a warm cache. **Output wrong instead:** see the table below and finding 6. Reports in `~jameslee/wan_720p_tdp130/run_NN/` (`test.log`, the `.mp4`, `index.txt`) |

**Runs with `TT_METAL_TDP_LIMIT_WATTS=130`** (same prompt, seed 42, `-glx_reset` before each; frames 0 / 22 / 45 / 80 in `img/`):

| Run | Hang | CLIP mean (≥ 36 passes) | What the video shows | Grey patches (`find_grey_patches.py`) |
|---|---|---|---|---|
| 1 | no | 34.92 FAIL | realistic cats on a dark spotlit stage, coherent motion | frames 17–32, shard (3, 6) only |
| 2 | no | 37.54 pass | realistic cats on a grey starry stage, coherent motion | frames 19–24, shard (3, 6) |
| 3 | no | 25.30 FAIL | flat cartoon cats on a white stage; cats vanish and jump between frames, stray dots | frames 47–52 and 62–64, shards (3, 6), (0, 6), (0, 5) |
| 4 | no | 36.47 pass (test PASSED, VBench skipped) | realistic cats on a blue-lit stage, coherent motion | frame 54 in shard (3, 6) (+1 tile in (0, 1)); smaller blocks also visible by eye in the same spot in frames 0 and 22 (below the scanner's 3-tile cutoff) |

Run 2 was stopped during VBench (CPU, more than 25 minutes at 720p); runs 3+ use `SKIP_VBENCH=1`.

Totals before the power cap: 17 runs at 720p chunked on the PR build, 17 hangs. The breakdown of the first 15: 12 device-side, all on device 9 / core `4-10 (3,8)`; 3 host-side. Plus 1 hang at 480p (device 9 / core `4-10`). Overall 16 hangs, 13 device-side, every one on that core. The 15th 720p run (a final `-glx_reset` retry, no throttle) hung at denoise step 2 in MinimalMatmul, outlier `4-10`; reports in `~jameslee/wan_720p_retry/`.

**Power / di/dt mitigations.** `TT_MM_THROTTLE_PERF=N` (1–5; level 5 caps matmul math at 33%) inserts NOPs between MVMULs. Via `throttle_mm_perf()` it reaches Wan's `MinimalMatmul` but **not** `RingJointSDPA`, whose Q·Kᵀ/PV math (`sdpa_matmul_block_no_mop`) is unthrottled, and that is where most hangs are. `TT_MM_STAGGER_TYPE`/`TT_MM_STAGGER_VALUE` have no effect on Blackhole: `apply_mm_stagger()` exists only in the Wormhole `llk_io_unpack.h`. Neither Wan op calls `add_stagger_defines_if_needed()` either. Testing the "weak Galaxy power delivery" theory properly would need throttling or staggering added to the SDPA compute kernel (a code change).

Loop reports: `~jameslee/wan_720p_glx/`, `~jameslee/wan_720p_throttle5/`, `~jameslee/wan_hang_reports/run_01..05/` (`summary.txt`, `triage.txt`, `triage_callstacks_2.txt`, `host_stacks.txt`, `test.log`, `links.txt`), index in `~jameslee/wan_hang_reports/index.txt`. The state after reset (0–4 chips with down ports) made no difference.

Not yet run: `main` 720p chunked a second time; anything on an 8-bank Galaxy for comparison.

Inside core `4-10` the three compute threads are in phases that don't fit together. In SDPA hangs the packer always sits in pack setup (`configure_single_tile_pack` / `configure_pack_width`, `compute_streaming.hpp:318/325`) while the unpacker waits on a CB that the packer should fill. In matmul hangs the threads sit at `compute_metal2.cpp:403–434`, while 100+ healthy cores sit at `:412/:441`. The "frozen" flag in `summary.txt` (TRISC PCs identical across two snapshots) is unreliable for this core: in some runs its PCs shift slightly between snapshots. The outlier grouping is the reliable signal.

## Findings in detail

### 1. Fabric auto-discovery fails on this board (software bug, worked around)

With no `TT_MESH_GRAPH_DESC_PATH`, control-plane setup fails with `Graph specified in MGD could not fit in the discovered physical topology`. `get_num_connections_per_direction` (`tt_metal/fabric/topology_mapper.cpp:1594`) takes the **maximum** links per neighbour pair. Four corner pairs (T1L1↔T2L1, T1L4↔T2L4, T3L1↔T4L1, T3L4↔T4L4) have 4 links by design, and every other pair has 2. So the generated MGD demands 4 channels per edge, under STRICT policy (the default reliability mode), and no 8x4 placement satisfies that. The BH Galaxy CI job always sets `TT_MESH_GRAPH_DESC_PATH`, so CI never hits this path. Not yet reported.

### 2. Issue #59504: OOM at pipeline creation

`TT_FATAL: Out of Memory: Not enough space to allocate 457850880 B DRAM buffer across 7 banks … largest free block: 65405568 B`. Every number matches the issue: 1,728 B short per bank. It happens in the warmup pass in `WanPipeline.__init__` (`pipeline_wan.py:460`), at a `conv3d` in a VAE decoder up-block ResNet (`models/tt_dit/models/vae/vae_wan2_1.py:477`). The BH (4, 8) preset combines `is_fsdp=False` and `dynamic_load=False` (both transformers resident) with `vae_t_chunk_size=None` (all 21 latent frames in one pass). Setting `vae_t_chunk_size=7` (the BH 2x4 value) fixes it at 81 frames / 720p. A real fix should probably chunk only when `mesh_device.dram_grid_size().x == 7`; chunked decode is slower and fits fine on 8 banks. Longer videos than 81 frames are untested.

### 3. Garbage output on main; PR #60190 fixes it

On a DRAM-harvested Blackhole, `main` gives each chip DRAM endpoints that depend on *which* channel is harvested. CCL kernels build the neighbour's address from the local accessor and write it over fabric, e.g. `ttnn/cpp/ttnn/operations/ccl/all_gather/device/kernels/unicast_writer.cpp:177-179`:

```cpp
uint64_t neighbor_addr = tt::tt_fabric::addrgen_detail::get_noc_address(output_tensor_accessor, page_id, byte_off);
fabric.async_write(l1_read_addr, neighbor_addr);
```

With mixed harvest masks, bank *k* sits on different cores on sender and receiver, so these writes land in the wrong place. PR #60190 gives every harvested chip the same translated DRAM endpoints. Note that the PR is a draft marked "[DO NOT MERGE]", and its author calls it "not the intended fix". The PR author (jbaumanTT) should hear that it fixes Wan output on a mixed-mask Bin6 Galaxy.

### 4. Hangs at 720p + chunked VAE on the PR branch (open)

See "Results so far". Device 9 = T2L2 = tray 2, ASIC location 2, logical mesh row 6 / col 0. The repeated core is `4-10` (virtual) / `(3,8)` (logical).

- Run 1 (SDPA): core `(3,8)`'s unpacker waits on the row max (`compute_streaming.hpp:602` ← `:1573`), its packer sits at the pack init for the same max CB (`:325` ← `:1693`), and math is at `:610`. Its program counters were identical across two snapshots minutes apart. The other 10 cores in that row's K chain wait on it (`chain_link.hpp:190/207`, `ring_joint_reader.cpp:1225`).
- Run 3 (matmul `compute_metal2`): 109 cores sit at `compute_metal2.cpp:441`. Core `(3,8)` alone is at `:431`, with its unpacker in `ckernel.h:265`.
- Run 2 was a host-side hang instead: the completion reader thread spun in `SystemMemoryManager::completion_queue_wait_front` while every device was idle.

### 5. A firmware TDP cap of 130 W stops the hang

`TT_METAL_TDP_LIMIT_WATTS` (`tt_metal/llrt/rtoptions.cpp`, applied in `Cluster::apply_tdp_limit_override`, `tt_metal/llrt/tt_cluster.cpp:363`) sends ARC message `SET_TDP_LIMIT` to every PCIe chip when the cluster opens, then reads the limit back and logs it. The accepted range is 50–500 W, and it needs Blackhole firmware 19.11.0+ (this board has 19.13.2). **The limit outlives the process and stays until the chip is reset** (`tt-smi -glx_reset`). Anyone who uses the board afterwards runs capped too.

With 130 W the run is no slower in practice: 3.6 s per denoise step versus 4.5–4.9 s seen earlier without a cap (those runs also had other variables). The firmware throttler lowers clocks only when the chip reaches the limit. Unlike `TT_MM_THROTTLE_PERF`, it covers every op, including RingJointSDPA, where most of the hangs were.

Taken together (17/17 hangs without the cap, 0/4 with it; the same core every time; chips stay on PCIe during a hang), the most likely story is **a marginal core or power path on device 9 that fails under full load**. That matches the "weak Galaxy power delivery" theory and the advice to try `TT_MM_THROTTLE_PERF`. The cap only works around it; the machine owners should still hear about device 9.

### 6. With the cap: non-reproducible output and grey patches in VAE shard (3, 6) (open)

Two separate symptoms. Both appear only now that runs finish on the PR build at 720p:

1. **Different output every run, with the same prompt and seed.** `torch.manual_seed(seed)` + `torch.randn` on the host (`pipeline_wan.py:678`, `:586`) make the initial latents identical across runs, and the device math has no atomics, so the video should be the same each time. Instead, the four videos differ completely in style and content, and run 3 is barely coherent. So the latents (or the text embeddings) are being corrupted somewhere in the denoise loop, by a different amount each run. Not yet proven: whether the pipeline is bit-exact on a healthy machine; compare two runs on an 8-bank Galaxy to be sure.
2. **Flat mid-grey rectangles.** Grey ≈ 124 is what an all-zero latent region decodes to (the garbage output on `main` was the same value over the whole frame). On the BH 4x8 preset the VAE splits height over mesh axis 0 (4 × 180 px at 720p) and width over axis 1 (8 × 160 px). The patch in all 4 runs sits inside the shard at **mesh coordinate (3, 6)**, pixels x 960–1120, y 540–720, in very different scenes. It fades in over 2–3 frames (the VAE's temporal convolutions spread it), lasts 6–16 frames, and moves in time from run to run. Run 3 also had patches in (0, 6) and (0, 5).

![run 1 frame 26](img/tdp130_run01_frame26_grey_patch.jpg)

A fixed shard points to one device (or its links) delivering zeros for part of its region, not random noise. Open questions: which physical chip is (3, 6) (is it device 9?); whether the patches and the run-to-run divergence go away at a lower cap (power) or not (software, e.g. the CCL/DRAM-address issue PR #60190 only partly fixes); and whether they appear at 480p.

## Files

- `run_wan.sh`: runs the test with the right environment.
- `vae_chunk_patch.py`: pytest plugin that sets BH 4x8 `vae_t_chunk_size=7` in memory only (`-p vae_chunk_patch`). With `SKIP_VBENCH=1` it also skips the VBench check, which takes more than 25 minutes on CPU at 720p.
- `find_grey_patches.py video.mp4 [--save-dir DIR]`: frames with flat mid-grey (decoded zero) tiles, their pixel box and which VAE shard (mesh row, col) they fall in.
- `img/`: frame strips (frames 0 / 22 / 45 / 80) from the 130 W runs, and a full frame showing the grey patch.
- `check_links.py`: ethernet link state (`--map`: full neighbour map; `--dram`: harvest masks), via UMD without opening a device.
- `hang_loop.sh [runs] [res] [chunk|nochunk] [out_dir]`: N × (check no other users → `tt-smi -glx_reset` → links → run). On a stall over 3 minutes it captures triage twice, gdb host stacks and an `analyze_triage.py` summary, then kills the run. It writes one line per run to `<out_dir>/index.txt` and copies the run's `.mp4` into `run_NN/`. Environment variables pass through, e.g. `SKIP_VBENCH=1 TT_METAL_TDP_LIMIT_WATTS=130 hang_loop.sh 3 720p chunk ~/out`.
- `analyze_triage.py triage.txt [triage_callstacks_2.txt]`: running ops per device, straggler device(s), cores grouped by kernel callstack with the outliers listed, and frozen-PC cores.

Full triage dumps and logs from the session above were in a session scratch directory and were **not** committed (about 110k lines each).
