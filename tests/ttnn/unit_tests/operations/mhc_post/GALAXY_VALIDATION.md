# mhc_post on a Galaxy: validation and comparison with the bring-up op

You have a Blackhole Galaxy (8x4 system mesh). This file says what to run and what to report. It covers the
`mhc_post` op that an eval run generated (this branch), on the hardware DeepSeek-V4.1 runs on. Nothing here has
been run on a Galaxy yet. Everything so far ran on a QuietBox 2 (4 chips, physically 2x2, opened as a 1x4 ring).

## Background

- **What the op computes:** the mHC post-sublayer for DeepSeek-V4.1 (T 160 tokens per chip, C 5120, n = 4
  residual streams).
  - It computes X'_j = post_j·F + Σ_i comb[i][j]·X_i.
  - It computes stats (X'·W, ΣX'², Σy²), the Sinkhorn hc, the collapse y = Σ P_i·X'_i, and the RMS norm of y.
  - With `feed_output`, it also performs the all-to-all of y over the 4 chips of the TP group (cluster axis 1;
    in_dim -2, out_dim -1). That all-to-all runs in the same program, over the fabric, from 4 reserved port
    cores per chip.
- **Op location:** `ttnn/ttnn/operations/mhc_post/` (Python program descriptor plus kernels in `kernels/`).
- **Golden suite and eval prompt:** `tt_metal/third_party/tt_ops_code_gen` (submodule pinned at `65d9aa91`,
  branch `malimpic/mhc-post-fused`). It is symlinked at the repo root as `eval/`.
- **The two calls DeepSeek-V4.1 makes ("focus cases"):** both are `LOOSE_CASES[0]` and `[1]` in
  `eval/golden_tests/mhc_post/feature_spec.py`. Both use bf16, `coef_source=packed`, HiFi2 and fp32 DEST.
  - **feed:** collapse + norm + feed, `feed_num_links=2`. The model's attn_residual call.
  - **no feed:** collapse + norm. The model's ffn_residual call.
- **Results on the QuietBox 2.** Measurement: traced, mean of the fastest chip, `eval.ccl_perf` protocol.

  | Case | Generated op | Bring-up op |
  |---|---|---|
  | Feed | 113.1 µs | not measured: it refuses the QB2's 1x4 group ("routes leave by more than two directions") |
  | No feed | 96.4 µs | 131.3 µs, one chip, device kernel duration |

  Roofline 37.3 µs, target 46.6 µs.
- **Why a Galaxy:**
  - On a Galaxy the TP group is a real row of the 8x4 mesh, so the bring-up op can run the feed. That makes the
    feed comparison possible.
  - The generated op's feed has never run on a Galaxy, so its correctness and speed there are unknown. Its feed
    plan was tuned on the QB2's 2x2 physical layout: one direction slot carries twice the data.
- **Fabric packet payload:**
  - The golden suite opens the fabric with a 14 KiB + 64 B packet payload (`production_router_config()` in
    `eval/golden_tests/mhc_post/conftest.py`).
  - The DeepSeek-V4.1 runner uses **10880 B** (`DS41Config.FABRIC_PAYLOAD_SIZE` on `dnijemcevic/model_bringup`).
  - The op sizes its packets from `ttnn.get_tt_fabric_max_payload_size_bytes()`: 7 tiles at 14 KiB, 5 tiles at
    10880 B. The 5-tile path has never run, so Part B runs both payloads.

## Rules for this box

- Run every device test through `scripts/run_safe_pytest.sh` (`--run-all` for full counts). Never use plain
  `pytest` or `python3` with device code, and never run `tt-smi -r` yourself. If the Galaxy has its own
  device-access or reservation rules, those come first.
- Do not change the op, the kernels or the golden suite to make something pass. The one exception is the payload
  edit in Part B step 4, which is local and never committed. If something fails or hangs, record it with the
  triage report and move on to the next step.
- Push only branches whose name starts with `malimpic/`.

## Part A: set up the generated op

```bash
git clone git@github.com:tenstorrent/tt-metal.git tt-metal-mhc-gen   # or a worktree of an existing clone
cd tt-metal-mhc-gen
git checkout malimpic/2026_11_13_2257_run1_mhc_post
git submodule update --init --recursive
./build_metal.sh
./create_venv.sh && source python_env/bin/activate
python -c "import ttnn; print(ttnn.get_arch_name())"            # expect blackhole
```

Record the following. The QB2 had an 11x10 compute grid; if the Galaxy's grid differs, the op's core split will
differ from the one measured.

- the system mesh shape:
  `python -c "import ttnn; print(ttnn._ttnn.multi_device.SystemMeshDescriptor().shape())"`. Expect 8x4.
- one chip's compute grid.
- the commit hashes (`git rev-parse HEAD`, and the submodule's).

## Part B: the generated op on the Galaxy

On an 8x4 system mesh, `eval/golden_tests/mhc_post/topology.py` runs every feed cell on the full 8x4 mesh. That
means 8 TP groups of 4 at once, each chip with its own inputs, under `FABRIC_2D_TORUS_XY` (the model's fabric).

1. **Single-chip golden suite.** This is a sanity check that the build is good on this box; it opens one chip.
   ```bash
   scripts/run_safe_pytest.sh --run-all eval/golden_tests/mhc_post/test_golden.py eval/golden_tests/mhc_post/test_regression.py
   ```
   - Expected: everything passes except `test_projection_hifi4` (2 failures, about 3e-4 against a 1e-5 gate; the
     same as on QB2).
   - Report any other failure.
2. **Feed correctness on 8x4, suite payload (14 KiB):**
   ```bash
   scripts/run_safe_pytest.sh --run-all eval/golden_tests/mhc_post/test_feed.py
   ```
   - On QB2 this was 149 passed. Here the cell count is similar, but every cell spans 32 chips.
   - This file includes:
     - every TARGET cell with `feed=on`;
     - feed against no-feed bit-identity at links 1 and 2 (`test_feed_matches_no_feed`);
     - `FABRIC_2D` instead of the torus (`test_feed_fabric_2d`);
     - 5 repeated calls into alternating buffers (`test_feed_repeated_calls`);
     - the refusals.
   - Do not use the eval runner's L1-profiling mode. It returns a zero peak on mesh workloads and fails correct
     cells.
3. **Perf of the two focus cases, suite payload:**
   ```bash
   CCL_PERF_EXPERIMENTS=5   CCL_PERF_OUT=galaxy_results/gen_14k_check scripts/run_safe_pytest.sh --no-precompile tests/ttnn/unit_tests/operations/mhc_post/perf_feed_measure.py -s
   CCL_PERF_EXPERIMENTS=100 CCL_PERF_OUT=galaxy_results/gen_14k        scripts/run_safe_pytest.sh --no-precompile tests/ttnn/unit_tests/operations/mhc_post/perf_feed_measure.py -s
   python -m eval.ccl_perf show galaxy_results/gen_14k/feed.json
   python -m eval.ccl_perf show galaxy_results/gen_14k/nofeed.json
   ```
   - `perf_feed_measure.py` checks each case against the golden reference once, then traces it.
   - Each JSON reports the fastest-chip mean (the metric), the slowest chip, the per-chip means, the spread and the
     dispatch check.
   - Do not use `eval/golden_tests/mhc_post/perf_measure.py`. It uploads inputs inside the trace and TT_FATALs.
4. **The model's payload (10880 B).**
   - Edit `production_router_config()` in `tt_metal/third_party/tt_ops_code_gen/eval/golden_tests/mhc_post/conftest.py`
     to set `cfg.max_packet_payload_size_bytes = 10880`. This is a local edit; do not commit it.
   - Rerun step 2 and step 3, with `CCL_PERF_OUT=galaxy_results/gen_10880`.
   - Restore the file afterwards: `git -C tt_metal/third_party/tt_ops_code_gen checkout -- eval/golden_tests/mhc_post/conftest.py`.
   - If step 2 fails only at 10880 B, the 5-tile packet path is broken. Report the first failing test and its
     message.

## Part C: the bring-up op on the Galaxy, same harness

The bring-up op is `ttnn.bringup.mhc_post` on `dnijemcevic/model_bringup`. The QB2 numbers used
`54b8cfd0793`; use that commit, and also the current tip if it differs and time allows. The adapter in this
directory (`bringup_adapter.py`) exposes it with the eval's signature and registry, so the same suite and the
same perf harness run it unchanged.

```bash
git clone git@github.com:tenstorrent/tt-metal.git tt-metal-mhc-bringup   # or a second worktree
cd tt-metal-mhc-bringup
git checkout 54b8cfd0793
git submodule update --init --recursive
./build_metal.sh
./create_venv.sh && source python_env/bin/activate
GEN=/abs/path/to/tt-metal-mhc-gen
test -e ttnn/ttnn/operations/mhc_post && echo "STOP: the branch has its own mhc_post package; report it"
mkdir -p ttnn/ttnn/operations/mhc_post
cp $GEN/tests/ttnn/unit_tests/operations/mhc_post/bringup_adapter.py ttnn/ttnn/operations/mhc_post/__init__.py
ln -s $GEN/tt_metal/third_party/tt_ops_code_gen/eval eval                     # the same golden suite
mkdir -p tests/ttnn/unit_tests/operations/mhc_post_bringup
cp $GEN/tests/ttnn/unit_tests/operations/mhc_post/{conftest.py,perf_feed_measure.py} tests/ttnn/unit_tests/operations/mhc_post_bringup/
python -c "import ttnn; from ttnn.operations.mhc_post import mhc_post, SUPPORTED; print('ok')"
```

None of these files are committed on the bring-up branch; they are local scaffolding.

1. **Feed correctness, so the comparison compares correct outputs:**
   ```bash
   scripts/run_safe_pytest.sh --run-all eval/golden_tests/mhc_post/test_feed.py -k "matches_no_feed or repeated_calls or fabric_2d"
   ```
   - The raw-y feed tests skip, because the bring-up op only feeds the normalized y.
   - On QB2 (2x2, group of 2) the bring-up op passed these tests but failed the T=150 feed cells (`h_non_aligned`,
     D differs from the all-to-all). Report whether that still happens here. Optional:
     `-k "test_op and h_non_aligned"` on `test_feed.py`.
2. **Perf, the same two cases, both payloads:**
   ```bash
   CCL_PERF_EXPERIMENTS=100 CCL_PERF_OUT=galaxy_results/bringup_14k scripts/run_safe_pytest.sh --no-precompile tests/ttnn/unit_tests/operations/mhc_post_bringup/perf_feed_measure.py -s
   # then the conftest payload edit from Part B step 4 (in $GEN's submodule, which the symlink points at), and:
   CCL_PERF_EXPERIMENTS=100 CCL_PERF_OUT=galaxy_results/bringup_10880 scripts/run_safe_pytest.sh --no-precompile tests/ttnn/unit_tests/operations/mhc_post_bringup/perf_feed_measure.py -s
   ```
   - Restore the conftest afterwards.
   - If the bring-up op refuses or hangs on 8x4, record the error or the triage report. Do not modify it.
3. **Compare the two ops:**
   ```bash
   python -m eval.ccl_perf compare galaxy_results/bringup_14k/feed.json   $GEN/galaxy_results/gen_14k/feed.json
   python -m eval.ccl_perf compare galaxy_results/bringup_14k/nofeed.json $GEN/galaxy_results/gen_14k/nofeed.json
   ```
   Do the same for the 10880 B pair.

## Part D (optional, only if Parts B and C are done)

- **Larger T on the feed:** pass `CCL_PERF_CASES=98,99` to `perf_feed_measure.py`. These are the `LOOSE_CASES`
  indices of the T 320 and T 640 feed cases (collapse + norm + feed, C 5120).
  - T ≥ 640 at C 5120 does not fit the op's L1 landing and uses its slower DRAM-staged path.
  - QB2 numbers: T 320 252.1 µs, T 640 about 429.5 µs.
- **Zone profile of the generated op (feed case), to see where the time goes on this topology:** set
  `MHC_POST_PERF_ZONES=port` and run `perf_feed_measure.py` with `CCL_PERF_EXPERIMENTS=5`
  `CCL_PERF_OUT=galaxy_results/zones_port`. Then repeat with `reader`, `writer` and `compute`.
  - Profile one kernel group at a time: all four at once overflow the 70 656 B kernel-config buffer. The knobs
    are documented in `_diag_defines` in `ttnn/ttnn/operations/mhc_post/mhc_post_program_descriptor.py`.
  - The interesting zones are the port kernels: `p_arm` / `p_wait_wave` / `p_stage` and `s_ready` / `s_send` /
    `s_arrival` / `s_landing`.
  - Report median and max per zone, per chip, from the device profiler log
    (`generated/profiler/.logs/profile_log_device.csv`).

## What to report

Write `galaxy_results/RESULTS.md` in the generated-op checkout, next to the JSON directories, containing:

1. **The box:** system mesh shape, compute grid per chip, both commit hashes (and the submodule's), the fabric
   config actually opened.
2. **Correctness:** pass / fail / skip counts for each test run in Parts B and C, at both payloads, and every
   failure with its first error line. For hangs, the path of the triage report and the stuck kernel from
   `dump_callstacks.py`.
3. **Perf table** (µs; fastest-chip mean, slowest chip, spread across chips), with the QB2 numbers alongside:

   | Case | Payload | Generated | Bring-up | QB2 generated |
   |---|---|---|---|---|
   | collapse + norm + feed | 14 KiB | | | 113.1 |
   | collapse + norm + feed | 10880 B | | | — |
   | collapse + norm | 14 KiB | | | 96.4 |
   | collapse + norm | 10880 B | | | — |

4. **Anything unexpected:** a chip or TP group much slower than the others, dispatch-bound warnings from
   ccl_perf, the bring-up op refusing a case.

If you can push, commit `galaxy_results/` (JSON plus RESULTS.md, no profiler dumps) on a new branch named
`malimpic/mhc-post-galaxy-results` based on this branch, and push it. Otherwise leave everything in place and
report the paths.
