# Handoff: Wan2.2-T2V-A14B on a Bin6 (7-bank) BH Galaxy

Branch: `jameslee/test-pr60190-dram-endpoints` (`main` @ `7086ed4c7a0` + the two commits of PR #60190, cherry-picked).
Machine: `bh-glx-120-b10u08` (single BH Galaxy, 32 Blackhole chips).
Written 2026-10-14 by a Claude Code session for jameslee@tenstorrent.com.

This directory is test scaffolding for an experiment branch. It is not meant to merge.

## TL;DR

1. **Issue #59504 (OOM at Wan pipeline creation on Bin6) reproduces here, byte for byte.** Chunked VAE decode (`vae_t_chunk_size=7` for the BH 4x8 preset) makes pipeline creation succeed.
2. **On `main`, Wan output on this machine is garbage** (flat gray / static frames, CLIP ≈ 19.4). **PR #60190 fixes it** (CLIP 37.07 at 480p). Likely cause: CCL kernels compute remote DRAM addresses with the *local* chip's bank-to-core map, and this Galaxy mixes all 8 DRAM harvest masks.
3. **Open problem: 720p + chunked VAE on the PR branch hangs every time: 15 of 15 runs** (plus 1 of 3 at 480p). Every device-side hang (13 of 16) has the straggler on **device 9 (T2L2)**, with the **same physical core, `4-10` (logical `(3,8)`)**, as the single outlier, across ring SDPA and matmul, `tt-smi -r` and `-glx_reset`, and with `TT_MM_THROTTLE_PERF=5`. The other 3 were host-side (all devices idle). This points strongly to a defective or marginal Tensix core on device 9. `main` has completed 2 runs and hung 0, too few to rule the PR build out.

## Your task

Find out whether the 720p hangs are hardware (device 9) or software (PR #60190 / chunked VAE):

0. **First, run `hang_loop.sh 3 720p chunk` (it now uses `tt-smi -glx_reset`).** Every hang so far followed a `tt-smi -r`, which may not fully reset this Galaxy. If the hangs disappear with `-glx_reset`, they were stale chip state, not a bad core or a software bug.
1. Rebuild `main` (see "Switching builds"), reset, and run `run_wan.sh 720p chunk` 2–3 times. No such run on `main` has hung yet, but there has only been one.
2. Run `run_wan.sh 720p chunk` on this branch 2–3 more times. Each time it hangs, run triage (below) and record **which device and which core** is the straggler. If it is always device 9 / core `4-10`, that is strong evidence of a bad chip.
3. If possible, stress device 9 on its own (for example, a single-device matmul or SDPA unit test in a loop on that chip) to confirm a hardware fault without Wan.
4. Report the ethernet links that fail to train intermittently (below) to whoever owns this machine.

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

Totals: 15 runs at 720p chunked on the PR build, 15 hangs: 12 device-side, all on device 9 / core `4-10 (3,8)`; 3 host-side. Plus 1 hang at 480p (device 9 / core `4-10`). Overall 16 hangs, 13 device-side, every one on that core. The 15th 720p run (a final `-glx_reset` retry, no throttle) hung at denoise step 2 in MinimalMatmul, outlier `4-10`; reports in `~jameslee/wan_720p_retry/`.

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

## Files

- `run_wan.sh`: runs the test with the right environment.
- `vae_chunk_patch.py`: pytest plugin that sets BH 4x8 `vae_t_chunk_size=7` in memory only (`-p vae_chunk_patch`).
- `check_links.py`: ethernet link state (`--map`: full neighbour map; `--dram`: harvest masks), via UMD without opening a device.
- `hang_loop.sh [runs] [res] [chunk|nochunk] [out_dir]`: N × (check no other users → `tt-smi -glx_reset` → links → run). On a stall over 3 minutes it captures triage twice, gdb host stacks and an `analyze_triage.py` summary, then kills the run. It writes one line per run to `<out_dir>/index.txt`.
- `analyze_triage.py triage.txt [triage_callstacks_2.txt]`: running ops per device, straggler device(s), cores grouped by kernel callstack with the outliers listed, and frozen-PC cores.

Full triage dumps and logs from the session above were in a session scratch directory and were **not** committed (about 110k lines each).
