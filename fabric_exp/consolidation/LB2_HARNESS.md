# Second-LB harness: study device jobs on extra BH P150x8 LoudBoxes via Slurm

Started 2026-11-03 (user: "if the queue is backed up we can start dispatching to other BH LBs as well"). The main
host bh-lb-120-a09u32 keeps the tt-device broker queue. Selected jobs go to an extra LB through Slurm. That LB runs a
node-local copy of the study tree at the same path, `/home/aagarwal/tt-metal`, so every study script runs unchanged.

## Nodes

- Approved: **bh-lb-120-a07u28** (partition `bh_lb_C45`), **bh-lb-120-a08u28** (`bh_lb_C67`) **bh-lb-120-a04u24** (`bh_lb_120_A45`, from 2026-11-03) and **bh-lb-120-a04u32** (`bh_lb_120_A456`, from 2026-11-04). The user OKed renaming old trees aside on these nodes (`--move-aside`). At most one job runs
  per node at a time (Slurm singleton, job name `lb2-<node>`), so there are at most two LB allocations.
- **`-n any` (user, 2026-11-03):** ONE job that starts on whichever approved LB frees first.
  - Slurm gets the approved nodes' partitions (`bh_lb_C45,bh_lb_C67,bh_lb_120_A45`, `-N 1`) and `--exclude` for every
    other node in them. That list is built from `sinfo` at submit time, so a09u32 is always excluded.
  - An approved node that `sinfo` doesn't list is left out.
  - The job is named `lb2-any`, its log is `logs/lb2-any-<job>.log`, and it has no per-node singleton (`--exclusive`
    still keeps one job per node).
  - It runs on `$SLURMD_NODENAME`, and lb2.sbatch's guard, host-check baseline and reset all use that node. `SUMMARY`
    and lb2_collect.sh's manifest (`results/lb2/<job>_<node>.tsv`) record the node.
  - Use it for jobs whose A/B arms all run in the one job (same host).
  - Never submit the same work twice, once per node: both could start, and running jobs are never cancelled.
- **`--test-only`** shows what Slurm accepts without submitting anything that can run. `sbatch --test-only` answers
  "allocation failure" while every candidate is allocated, so the job is also submitted held. Its
  partitions / excluded nodes / node count are printed, and it is cancelled while still held.
- Never use bh-lb-120-a04u24 (released, do-not-use) or a08u24 (another tenant). Ask the user before using any other
  node. `bh_lb_single` nodes are single cards, so they are not usable.
- The nodes are shared Slurm nodes. They also run a tt-device broker, k8s agents and other users' shells. Another
  user can take a node between our jobs (a08u28 was taken at 00:37 on 2026-11-03 by arichins' `buildall`).
- **Node status (2026-11-04)**:
  | node | /home free | links | any-mode |
  | --- | --- | --- | --- |
  | a07u28 | 936 GB | all 48 mesh link ends UP / PASS / 400G in host check 172557 (06:00, read-only, no reset); the new baseline has 60 UP links | in |
  | a04u32 | 1247 GB | all 48 mesh link ends UP / PASS / 400G in host check 172564 (first job there; wiring identical to a09u32); baseline 55 UP links | in |
  | a04u24 | 223 GB (11-03) | | in, but held by another user's 4 h job at 06:00 |
  | a08u28 | 13 GB (job 171943) | | **out**: below the 25 GB floor; `-n a08u28` still reaches it |
  - a07u28's chip 6 eth (7,1) did not train after any reset in 172043, 172149, 172157 or 172175. 172175 used the new
    poll: 3 resets with 90 s each. It was UP again at host check 172557. Its next job's reset shows whether it holds.
  - Any-mode leaves out the nodes in lb2_run.sh's `ANY_SKIP_DEFAULT`. Set `LB2_ANY_SKIP` to change that for one
    submit; an explicit `-n` list overrides it.
  - The busy check ignores zombie processes (`<defunct>`). An unreaped zombie of another user's test stopped the first
    a04u32 host check (172556); a zombie holds no device file.
  - The host check's mock runs (step 4: torusns, torusns_scatter, VC1 split) failed on both a07u28 and a04u32 on
    2026-11-04 (exit 1 / 134). They are reported, not gating. Failing the same way on two nodes points at the tree's
    mock setup rather than the nodes; not looked into.
- Each node gets a host check before it is used (below). Its baseline is in `/data/aagarwal/lb2/nodes/<node>/`:
  `cluster_desc.yaml`, `eth_baseline.txt` and `mock_mapping.yaml`. Every job's link check compares against that
  baseline.

## Files

| file | what |
|---|---|
| `lb2_stage.sh [refresh\|full]` | runs on a09u32 and only reads the live tree. It writes the snapshot to `/data/aagarwal/lb2/stage` (see below). |
| `lb2_pack.py` | packs a scope as a zstd tarball and writes a sha256 reference list from the same bytes. `check` reads the archive back; `live` re-hashes the tree. |
| `lb2_run.sh -n NODE [opts] "<cmd>"` | submit wrapper, run on a09u32. It first checks that the live inputs match the stage, then submits `lb2.sbatch`. `-n any`: see below. |
| `lb2.sbatch` | the job (see the header). Steps: node guard, busy check, space, swap restore, sync + verify, preflight, `tt-smi -r`, link check, the command, hang handling, swap restore, copy-back. |
| `lb2_hostcheck.sh` | read-only host check: topology, wiring vs a09u32, links, and mock runs of the torus rows and the VC1 split. |
| `lb2_linkcheck.py` | the link gate after each reset (section "Link gate"): `live` on the node, `replay` on saved files. |
| `lb2_calib_runs.sh TAG` | calibration rows: main_fw and v4rxg3 on pcs3 ew4352 / hop3 / lin3, plus the 2D / 1D size sweep. |
| `lb2_collect.sh JOBID` | runs on a09u32. It copies a finished job's results into `fabric_exp/results` (it never overwrites) and writes the manifest `fabric_exp/results/lb2/<job>_<node>.tsv`. |

Logs go to `/data/aagarwal/lb2/logs/lb2-<node>-<job>.log` (streamed). Job outputs go to `/data/aagarwal/lb2/jobs/<job>/`:
`results/`, `builds/`, `caches/<cache>/` (the router ELFs and generated headers the job compiled: what PC-sample
mapping and offline analysis need), `MANIFEST.tsv`, `SUMMARY`, `COMMAND`, `command.log`, and the verify / eth / reset
outputs.

## Stage and refresh

NFS creates small files at ~10/s (measured 2026-11-03), so the stage is a set of tarballs, not a file tree:

- `full-<time>/`: build_Release, python_env (without `*.pyc`), runtime (sfpi) and the uv Python (cpython-3.10.19)
  that python_env links to. Each is a `.tar.zst` with a `REF_<x>.sha256` list, plus `FULL_STAMP` (checksums of the
  host binaries).
  - Rebuild it with `lb2_stage.sh full` only after a host-library rebuild on a09u32. A refresh warns when
    build_Release has drifted from the stamp.
- `src-<time>/`: `src.tar.zst` holds the top-level files plus tt_metal, ttnn, tt_stl, tests, tools, scripts,
  third_party and fabric_exp (no results, consolidation/builds or variants/*/elf). It also holds `REF_src.sha256`,
  `GIT` (the live tree's git status and diff checksums), `CHANGED` (what differs from the previous stage) and
  `STAGE_TIME`.
  - `current` points at the newest src stage. The two newest are kept. A job resolves `current` once at its start.
- `git/`: an rsync of `.git`, used only for the node's git status check. It is never copied to a node.
  - A full stage syncs it, and so does a refresh when the live HEAD moved (since 2026-11-03).
  - Before that, a commit after the last full stage failed every job's git check: the node computed git status with
    the old HEAD, so every committed file showed as changed. This stopped lb2 171279 at verify, before any device
    work.
- The snapshot is the **unswapped** tree. A broker job's active swap (`~/.tt-metal-p1-swap-backup/manifest.json`) is
  undone in the copy: saved files are taken from the backup and created files are left out. A pack is repeated when
  a swap was applied or restored during it.
- `lb2_stage.sh refresh` takes about a minute. It prints the files packed, the MB written to /data, and the files
  changed since the previous stage. On the node, the sync extracts to local disk and rsyncs into the tree. The job
  log prints files and MB moved per directory, and only changed files move. build_Release, python_env and runtime
  sync again only when the full stage changed.
- Verification, every job:
  - the node copy is checked by `sha256sum -c` against every reference list. Those lists come from the live a09u32
    tree (unswapped view) at stage time;
  - git status and git diff checksums over the source scope are compared with the live tree's. fabric_exp/results
    and the swapped files are excluded on both sides.

## Link gate (2026-11-04)

The job's link check, after every reset poll, gates only on the links the fabric uses
(`lb2_linkcheck.py live`, called by lb2.sbatch's `eth_check`).

**Why it changed.** The old check required every Ethernet core that was UP / PASS / 400G in the host-check
baseline to be so again. That baseline included ports the fabric never uses, mostly cables to other hosts
(`ethernet_connections_to_remote_devices`), whose state follows the other machine. Two sweep jobs stopped before
any device work on such ports:
- lb2 176008 on a04u24: chip 1 (11,1), chips 4 and 7 (6,1);
- lb2 176130 on a07u28: chip 6 (6,1).
The coordinator chose option (b), fail closed.

**The gate** (exit 0 only when all three pass):
- **A. Link set, mapping-free.** A fresh UMD cluster descriptor (TopologyDiscovery, as topo_dump.py makes it) must
  have exactly the baseline's `ethernet_connections`. They are compared as unordered (chip unique id, channel) pairs,
  so a chip renumbering does not matter. A missing pair is a fabric link down; an extra pair is unexpected wiring.
- **B. Channel to core, per chip.**
  - Each connection end's channel goes to a core two ways: UMD's soc descriptor of that device
    (`get_eth_cores_for_channels`), and the descriptor (the arch file's eth order with the chip's
    `eth_harvesting_mask` positions removed).
  - Both must give the same single core, or the gate fails.
  - Descriptor chips are matched to PCI devices by BDF, else by bus id; one-to-one, or it fails.
- **C. State.** Every connection end's core must read port UP, train PASS, speed 400.

Every other Ethernet core goes into the job log as information ("port down (a link to another host)" or "(not a
fabric link)"), with counts of UP and unused ports. It never gates. Each poll keeps `eth_<label>_eth.txt` (each
core's state and channel) and `eth_<label>_cluster_desc.yaml` in the job directory. Any error fails the gate.

**The channel mapping.** Two readings disagreed:
- the arch file's raw order puts (6,1) at index 10 and (11,1) at 11;
- this harness's a07u28 notes put them at channels 8 and 9.

With eth_harvesting_mask 288 (positions 5 and 8 harvested, on every chip of all four nodes), removing the harvested
positions gives channels 8 and 9. Only that mapping puts all 48 connection ends on cores present in the readouts: the
raw order would place two ends on harvested cores, which the readout does not list. The live gate cross-checks it
against UMD on every poll and fails closed on any disagreement.

**Tests** (offline, `lb2_linkcheck.py replay BASELINE_DESC NOW_DESC NOW_ETH`):
- Every saved poll of 176008 (30 readouts, a04u24) and 176130 (30, a07u28) passes. Each node's baseline descriptor
  stands in for the current one, since those runs saved none, so gate A is not exercised by them. All 48 fabric link
  ends were UP / PASS / 400; the failing ports were links to another host or unconnected.
- Every node's own baseline readout passes (a04u24, a04u28, a04u32, a07u28).
- Synthetic cases:
  - one fabric link end down: FAIL (gate C);
  - one pair missing from the current descriptor: FAIL (gate A);
  - an end's core missing from the readout: FAIL (gate C);
  - chips renumbered with the same wiring: PASS (gate A).
- **Live, first run** (lb2 176206 on a04u24, 2026-11-05 00:03): the gate passed at the first poll after the reset.
  - A: 24 of 24 links present.
  - B: 48 of 48 ends mapped, with UMD's `get_eth_cores_for_channels` and the descriptor mapping in agreement. This
    settles the mapping on that node: with this harvesting, (6,1) = ch8 and (11,1) = ch9.
  - C: 48 of 48 UP/PASS/400.
  - Other ports: 8 up, 32 unused, 8 down; logged, not gating.
  - The dry run 176205 passed before it.
- **Finding: descriptor chip ids are not PCI device numbers.** On a04u24 the gate mapped descriptor chips to PCI
  devices as {0: 2, 1: 3, 2: 1, 3: 0, 4: 6, 5: 7, 6: 5, 7: 4}.
  - read_eth_speed.py (the old check, and lb2_hostcheck.sh step 3) labels its readout "chip N" by PCI device N, and
    those labels were compared with descriptor chip N.
  - That is harmless only because the permutation stays inside each 2x2 group, and every chip of a group uses the
    same channels.
  - The replay tests above read the saved readouts with the same assumption, so their per-chip labels are PCI
    numbers. Their pass / fail results hold for the same reason.
  - The live gate maps chips by BDF.

**Staging.** The gate is in lb2.sbatch, which Slurm stores at submit time, plus lb2_linkcheck.py from the stage.
Jobs submitted before the change keep their old sbatch, and their old check needs only read_eth_speed.py, which is
unchanged. Stages are never modified in place.

## Rules: which jobs go to an extra LB

The main session applies these (user, 2026-11-03):

1. **Frozen inputs only**, all of them already on the node:
   - built variants that won't change, yamls and scripts. `lb2_run.sh` refuses a submit when any live script, yaml,
     variant, router or host source, or any file named in the command, differs from the stage. In that case run
     `lb2_stage.sh refresh` first;
   - nothing that depends on a09u32-only caches or data: trimming capture caches, coverage case caches (`cov_m`,
     `~/.tt-metal-v3r-builds`), offline-build bases (`~/.tt-metal-p1-base`), or earlier results read as inputs. The
     stage holds no `fabric_exp/results`; stage a needed results input explicitly.
2. **No host-library change.** The node runs the staged build_Release. A job that needs a host patch or a rebuild
   stays on a09u32, or waits for a full re-stage after a09u32's build changed.
3. **Batches worth the overhead:** about 20 min of device time or more per dispatch. Per-job overhead is the queue
   wait plus about 3-5 min: sync, sha256 verify, reset and link check, copy-back. Both passes of an A/B run on the
   same host, ideally in the same job.
4. **The sender pass's round (broker 499-501) stays on a09u32.**
5. **Hosts don't mix within a comparison.** A same-job A/B on an extra LB is valid on its own. Absolute numbers
   transfer across hosts only as far as the calibration shows (below). The manifest records each label's host.
6. Ring/torus yamls not yet run on a09u32 hardware go through the mock first (memory: mock-vetting rule). The host
   check runs the torus rows and the VC1 split on the node's own descriptor.
7. Never cancel a running job. The main session cancels queued broker copies of moved jobs; agents don't.

After a hang, the job kills the session, captures the state (`hang_capture.py`, unless the study script already
did), resets and checks the links. A hang rules nothing out on the new host until that reset and check are done.

## Settings and overrides

- The command string runs as written: `bash -c "<cmd>"` in `/home/aagarwal/tt-metal` on the node. Environment
  overrides go inside it, exactly as in a broker command. Example: a mesh graph descriptor for a 4x2 shape,
  `TT_MESH_GRAPH_DESC_PATH=$PWD/fabric_exp/torus/<file>.textproto bash $C/<script>.sh ...`, or the variable a script
  reads.
- The job clears every inherited `TT_*` variable before it starts, and removes `SLURM_*` / `PMI*` / `OMPI_*` before
  the command runs, so mpirun (ttrun.py) behaves as it does on a09u32's broker. `TT_FABRIC_NO_BW_TELEMETRY=1` is
  exported.
- A fabric_exp path named in the command (a yaml, a descriptor, a script) must exist in the node tree, or the
  preflight stops the job. `--inputs "path ..."` adds paths a script reads internally.
- New files anywhere under fabric_exp (e.g. `fabric_exp/torus/`) reach the node with the next
  `lb2_stage.sh refresh`.

## Typical use (on a09u32)

```
bash fabric_exp/consolidation/lb2_stage.sh refresh                       # inputs changed since the last stage
bash fabric_exp/consolidation/lb2_run.sh -n a07u28 -t 01:30:00 "C=fabric_exp/consolidation && ..."
bash fabric_exp/consolidation/lb2_collect.sh <jobid>                       # after it ends
```

## Calibration

### bh-lb-120-a07u28 (2026-11-03, Slurm jobs 169530 host check, 169531 calibration; stage src-20261103-005613)

Host check (read-only):
- 8 chips, cube wiring **identical to a09u32 at (chip, channel) level**: 24 links, 12 pairs x 2. Chips {0-3} and
  {4-7} are two 2x2 faces with 8 links between them. PCIe map in order.
- All 48 mesh link ends are UP / PASS / 400G. 56 cores are UP in total.
- Ports down, none of them mesh links:
  - chips 0, 1, 2 and 3: chan 11 eth (10,1);
  - chip 3: chan 9 (11,1);
  - chips 5 and 6: chan 8 (6,1);
  - chip 7: chan 10 (7,1).
- Firmware bundle 19.16.0 and KMD 2.11.0, the same as a09u32.
- 64 CPUs, not confined by the Slurm allocation, 591 GB memory free.
- CPU-only mocks on the node's descriptor all get past fabric setup:
  - torusns and torusns_scatter;
  - the VC1 split, vc1_alloc and vc1_inter: two MPI ranks, no `--oversubscribe`, with SLURM_* removed.

Calibration (cycles/packet, same scripts and settings as a09u32; `lb2_calib_runs.sh a07`):

| row | router | a07u28 | a09u32 | diff |
|---|---|---|---|---|
| pcs3 ew4352 | v4rxg3 | 127.0 / 127.0 | 127.0 (jobs 493/494) | 0 |
| pcs3 hop3 | v4rxg3 | 147.3 / 147.3 | 146.7 | +0.6 (+0.4%) |
| pcs3 lin3 | v4rxg3 | 124.4 / 124.4 | 124.4 | 0 |
| pcs3 ew4352 | main_fw | 222.6 / 222.6 | sweep Mesh1HopEW 222.7 / 222.8 | 0 |
| pcs3 hop3 | main_fw | 221.8 / 221.8 | sweep Mesh3HopEW 221.7 / 222.0 | 0 |
| pcs3 lin3 | main_fw | 187.5 / 187.5 | sweep Linear3HopEW 187.6 | 0 |

The 2D + 1D size sweep (`rszlbca07_*`, 260 rows) compared with a09u32's jobs 465-468:
- median |diff| is 0.01-0.03% per suite and router.
- **Observation (cause untested):** v4rxg3's Mesh3HopEW scatter rows run 1.1-3.5% *faster* on a07u28 at every size
  (3072 B: 142.9 vs 148.1; a09u32's two passes agree to 0.1%). main's same rows match within 0.03%. Different Tensix
  harvest masks across hosts are only a hypothesis.
- v4rxg3 Linear3HopEW write at 256-3072 B is 0.34-0.47% slower, and Mesh3HopEW write at 4352 B 0.4% slower.
- main HalfRingMulticast differs within its own pass spread.

Verdict:
- Same-job A/Bs on a07u28 are valid on their own.
- Absolute rates transfer within ~0.5% for most rows. **Cross-host comparisons of forwarded multi-hop rows are not
  valid** (v4rxg3 scatter at 3 hops: up to 3.5%). Compare those within one host.

### bh-lb-120-a08u28

Not used yet. Another user held it from 00:37 to about 01:15 on 2026-11-03. Its `/home/aagarwal/tt-metal` is an
older tree (2026-10-09) that must not be touched without the user's OK. The harness needs that path (`--move-aside`
renames the old tree; it never deletes it). /home is 99% full (59 GB free), so the job's cache floor
(LB2_MIN_FREE_GB 25) matters there.

## Harness jobs so far (2026-11-03)

- 169448 / 169449: read-only srun probes of a08u28 / a07u28 (disk, tools, device holders); 169497 was refused (a08u28
  taken).
- a07u28 dry runs:
  - 169509: stopped at the git check (submodule .git files): fixed with `--ignore-submodules=all`;
  - 169521: our own unmarked tree hit the move-aside guard: fixed with the `.lb2_owned` marker;
  - 169525: relative lb2_pack.py path: fixed;
  - 169529: clean.
- 169530 host check (rc 0) and 169531 calibration (rc 0; collected, `fabric_exp/results/lb2/169531_a07u28.tsv`).
- 169543: dry run of broker 510's command, cancelled: 510/511 stayed on a09u32.
- 169565: dry run of the XY-torus command. Queued behind another user's job on a07u28 (ahossain 169537 from ~01:07).

## Prepared: XY-torus rows (user, 2026-11-03)

Files: `fabric_exp/torus/` (torusxy_2x4_sizes.yaml, torusxy_4x2_sizes.yaml, val_torus_xy_p150.yaml,
p150_x8_4x2_torus_xy_mgd.textproto), `torus_xy_mock.sh` (CPU-only vetting; every group got past setup in both shapes and
on both routers; DA = 1 on all four directions), `torus_xy_runs.sh`, `report/scripts/build_torus_xy.py` ->
`report/data/bandwidth_torus_xy.json`. The host is chosen by the main session:

```
# extra LB (both passes in one dispatch; on a08u28 only after its host check and calibration)
bash fabric_exp/consolidation/lb2_run.sh -n a07u28 -t 00:50:00 --inputs "fabric_exp/torus/torusxy_2x4_sizes.yaml fabric_exp/torus/torusxy_4x2_sizes.yaml fabric_exp/torus/val_torus_xy_p150.yaml fabric_exp/torus/p150_x8_4x2_torus_xy_mgd.textproto" "C=fabric_exp/consolidation && bash \$C/torus_xy_runs.sh a 2x4 4x2 && ROUTERS='v4rxg3 main_fw' TXY_VAL=0 bash \$C/torus_xy_runs.sh b 2x4 4x2"
# a09u32 broker (two jobs, each well under 1500 s)
cd /home/aagarwal/tt-metal && C=fabric_exp/consolidation && bash $C/torus_xy_runs.sh a 2x4 4x2
cd /home/aagarwal/tt-metal && C=fabric_exp/consolidation && ROUTERS='v4rxg3 main_fw' TXY_VAL=0 bash $C/torus_xy_runs.sh b 2x4 4x2
```

**2026-11-03, XY torus on a09u32 (broker 512):** the job ran on a09u32 because its queue was idle. It claimed
/data/aagarwal/lb2/locks/torus_xy and the LB copies were cancelled. Bug: jobs_lib.sh defines `VAL="stream mesh val1d"`,
which overrode torus_xy_runs.sh's VAL flag, so 512 skipped the validation and DA checks; the flag is now `TXY_VAL`. The
benchmark logs confirm the axes (2x4: Y=LINE X=RING; 4x2: Y=RING X=LINE). Validation reran as pass c.
