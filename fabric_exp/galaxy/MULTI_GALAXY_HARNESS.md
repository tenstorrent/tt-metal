# Multi-Galaxy harness: one command on a cabled BH Galaxy quad (2026-11-03)

Prepared for the new router's multi-Galaxy regression runs: fabric tests and CCL / model tests on multi-Galaxy meshes.
Nothing has run on a quad yet. The first real job waits for the go (section 6).

## 1. What a quad is here

A **cabled quad** is four BH Galaxies cabled as a ring, 8 QSFP-DD cables between ring neighbours. That is the 32x4
torus of `tt_metal/fabric/mesh_graph_descriptors/32x4_quad_bh_galaxy_torus_xy_graph_descriptor.textproto`
(device_topology 32x4 RING x RING, host_topology 4x1, channels 2 RELAXED).

On the 110-a nodes (study rule: bh-glx-110-a* only) there are nine:

| quad | Slurm | ring order (physical) | tt-cluster-config leaf |
| --- | --- | --- | --- |
| pod3 | bh_sc36_6_110a_pod3 | a06u02, u08, u14, u20 | 110-SC4_A06 |
| pod6 | bh_sc36_6_110a_pod6 | a04u02, a04u08, a05u08, a05u02 | 110-SC4_A04-05_u02-08 |
| pod7 | bh_sc36_6_110a_pod7 | a01u02, u08, u14, u20 | 110-SC4_A01 |
| pod8 | bh_sc36_6_110a_pod8 | a02u02, a02u08, a03u08, a03u02 | 110-SC4_A02-03_u02-08 |
| pod9 | bh_sc36_6_110a_pod9 | a02u14, a03u14, a04u14, a05u14 | 110-SC4_A02-05_u14 |
| a07 ... a10 | bh_110a_sc16 + those four nodes | aNNu02, u08, u14, u20 | 110-SC16_4x4_A7-10 |

Machine-readable form: `glxq_quads.tsv`.

**Where this comes from:**
- Ring orders come from each leaf's `cabling_guide.csv` in the tt-cluster-config checkout `/data/pbara/tt-cluster-config`
  (`exabox/`). Each quad's 8-cable neighbour pairs close a 4-cycle; pod6 and pod8 are not in name order.
- **Pod partitions** (pod3, 6, 7, 8, 9) are exactly one quad each, OverSubscribe=EXCLUSIVE with no time limit: that
  is tt-sbatch's `--topo quad / sc4`. pod1, 2, 4 and 5 exist but have no nodes.
- **SC16** (bh_110a_sc16, 6 h limit) is four per-rack rings (a07-a10) plus single cables between racks. So a rack is a
  quad, requested with `-w` and that rack's nodes.
- **State on 2026-11-03 ~20:00:**
  - pod6 drained (MINFRA-2421), a03u02 drained;
  - pod3, pod8 and pod9 each hold a job with no time limit;
  - pod7 and the SC16 racks are busy.
  - A quad will queue until a whole pod frees.

**FSD:** the published, aggregated exabox Factory System Descriptor, production channel:
`/data/scaleout_configs/tt-cluster-config-sources/exabox-latest-main/factory_system_descriptor.textproto`.
- It covers 212 hosts and is identical to `exabox-latest-staging`; the `bh_glx_exabox` symlink points at staging.
- The control plane keeps the allocated hosts' part (`ControlPlane::ingest_factory_system_descriptor`,
  fsd_host_filter), so one file serves every quad.
- The per-quad leaf FSDs agree with it inside a quad: 312 Ethernet endpoints per host with both ends in the quad. The
  aggregated file lists 314-318 per host because of the single cables to other pods.
- In tt-run the FSD is optional: without it, Phase 1 and the control plane use live discovery. `generate_rank_bindings`
  does not consume it yet (issue 52859).

**FSD and health mode:** with an FSD passed to the run, a RELAXED mesh graph (the quad MGDs) needs
`RELAXED_SYSTEM_HEALTH_SETUP_MODE`. The control plane stops with "A RELAXED mesh graph cannot be used with STRICT system
health setup" otherwise.
- test_tt_fabric builds its first control plane, right after the master seed, in STRICT, before it reads the yaml's
  `fabric_reliability_mode`. `RELIABILITY_MODE=relaxed` does not reach it either (mock run, section 5.1).
- So the harness uses the FSD for its own link preflight (a file read) and passes it to tt-run only with `--fsd-run`.
  That is for apps that open the fabric RELAXED themselves; the job then also exports `RELIABILITY_MODE=relaxed`.
- `glxq_smoke.yaml` sets `fabric_setup.fabric_reliability_mode: RELAXED_SYSTEM_HEALTH_SETUP_MODE`. It is harmless
  without an FSD (the mock reached launch with it).

## 2. How the ranks start

**tt-run in rank-binding (legacy) mode**, as the exabox CCL runbook
(`tests/ttnn/unit_tests/operations/ccl/blackhole_CI/exabox/multi_host/RUNBOOK.md`) and `run_fabric_tests.sh --config 4x32`
do:
- 4 ranks on mesh 0 (`32x4_quad_bh_galaxy_rank_bindings.yaml` shape, MGD path written in);
- `--mpi-args "--host <ring order>"`, so rank k lands on the k-th ring host;
- `--tcp-interface ens5f0np0`, which becomes `--mca btl self,tcp --mca btl_tcp_if_include ens5f0np0`;
- `--factory-system-descriptor` only with `--fsd-run` (section 1).

The new MGD mode (`-m` + `--hosts`) needs `build/tools/scaleout/generate_rank_bindings`, which the study build does
not have.

**MPI:** `/usr/local/bin/mpirun-ulfm` (Open MPI 5.0.7, PRRTE) is on the Galaxies, with `plm: slurm`.
- Inside the batch job the Slurm environment stays set, so mpirun starts its daemons on the allocated nodes with srun.
  No ssh, no containers.
- This differs from the single-node harnesses (lb2.sbatch strips SLURM_* for single-host mpirun).
- tt-sbatch runs its ranks in docker containers with an sshd. Here they run on the hosts, from the /data tree, like
  every other study Galaxy job.

**Environment:** tt-run passes every `TT_*` variable (and `LD_LIBRARY_PATH`, `PATH`, `HOME`) to every rank with `-x`.
It sets per rank `TT_MESH_ID`, `TT_MESH_GRAPH_DESC_PATH`, `TT_RUN_RANK`, `TT_METAL_HOME` and the FSD path.
The job sets:
- the tree's library path;
- `TT_FABRIC_NO_BW_TELEMETRY=1`, `TT_METAL_FABRIC_OPT_LEVEL=O3`, `TT_METAL_CLEAR_L1=1`;
- `TT_METAL_FABRIC_ROUTER_SYNC_TIMEOUT_MS=1000000` (as run_fabric_tests.sh does, for slow Ethernet handshakes after a
  reset);
- the router: `new` = kernel v2 slot, flags 13129088, keepalive 1000; `main` = kernel v1, flags 0;
- the job's own paths as `TT_GLXQ_JOB` / `TT_GLXQ_TREE` (the TT_ prefix carries them to the ranks).

**Per rank (`glxq_rank.sh`):**
- `TT_METAL_HOME` = a symlink farm of the tree with its own `generated/`, so test outputs do not collide;
- `TT_METAL_CACHE` = `<job>/cache/r<rank>`, kept for PC mapping and hang decoding;
- output goes to `<job>/ranks/r<rank>_<host>.log`, with `r<rank>/info.txt` (host, env, exit code);
- mpirun's tagged output keeps one status line per rank.

**Tree sharing:** everything the nodes read is on /data.
- The trees are `/data/aagarwal/frs-glx/trees/<name>`, read-only by convention, like the single-Galaxy jobs.
- The submit side copies the helpers to `/data/aagarwal/frs-glx/glxq/stage/<stamp>-<quad>/`.
- This host's `/home/aagarwal` is not on the Galaxies: a01u14 has its own `/home/aagarwal` without `tt-metal`.
- tt-sbatch copies its checkout to node-local disk because /data is slow per file; our test binaries and kernels build
  from /data, as the single-Galaxy jobs do.

## 3. Files (fabric_exp/galaxy/)

| file | what |
| --- | --- |
| `glxq_run.sh` | submit side (run on a09u32): checks, staging, `sbatch`; `--dry` = no submission |
| `glxq.sbatch` | the job: guard, busy, tools, reset, links, smoke, run, hang handling |
| `glxq_node.sh` | one node's preflight / cleanup step (srun, one task per node): busy, tools, reset, eth, kill, capture |
| `glxq_rank.sh` | one MPI rank: own tree farm, cache, log |
| `glxq_links.py` | Ethernet preflight: per host, channels UP and trained >= the FSD's endpoints inside the allocation |
| `glxq_smoke.yaml` | data-checked all-to-all on the quad mesh, the first run of every job |
| `glxq_quads.tsv` | the quads (section 1) |
| `glxq_mock.sh` | the plumbing on one host, CPU only: 4 mock ranks of a 32x4 quad (section 5) |

## 4. Using it

```
G=fabric_exp/galaxy
bash $G/glxq_run.sh --quad pods --tree v4txa11_1103 --dry -- \
    ./build/test/tt_metal/tt_fabric/test_infra/test_tt_fabric --test_config <yaml, tree-relative or absolute>
bash $G/glxq_run.sh --quad pods --tree v4txa11_1103 --router new --time 02:00:00 --timeout 3600 -- <same command>
```

**`--quad`:** `pods` (whichever pod partition frees first), `pod3` ... `pod9`, or `a07` ... `a10`.
- Defaults: `--mgd` the 32x4 quad torus, `--router new`, and `--fsd` the published exabox FSD for the link preflight
  (`none`: counts printed only).
- The run uses live discovery unless `--fsd-run`.
- For main vs new in one comparison, submit one job per router on the same quad (`--quad podN`), one after the other.
  The single rule of the exabox runbook: one tt-run at a time per cluster.

**The job's steps** (any failure stops it; it never cancels a job):
1. **guard:** every node bh-glx-110-a*, and the allocation is one quad of the table; the ring order comes from there.
2. **busy:** other users' device processes on any node stop the job.
3. **tools:** mpirun-ulfm, tt-smi, the tree and its python_env, the MPI NIC, /data writable, HOME.
4. **reset:** `tt-smi -glx_reset` on every node in parallel, then 32 or more Blackhole chips listed on each. This is
   tt-sbatch's reset.
   - Cluster validation through `recover.sh` is **not** used: its header forbids agents from running it directly. The
     approved entry point is the operator's `recover-hosts` helper, which records the run.
5. **links:** `read_eth_speed.py 32` on every node.
   - Channels UP and trained must be at least the FSD's endpoints inside the quad (`glxq_links.py`; 312 per host).
   - The per-cable comparison is the control plane's own when the FSD is passed.
6. **smoke:** `glxq_smoke.yaml` through tt-run, 900 s.
7. **run:** the command through tt-run, `--timeout`.

**On a timeout:**
- this job's test processes are killed on every node (the job holds them `--exclusive`);
- the routers are captured read-only on every node (`hang_capture.py` into `results/<label>_hangcap_<host>`);
- every node is reset, and the job stops.

**Outputs:** `/data/aagarwal/frs-glx/jobs/<id>/`
- `results/`: quad.txt, eth dumps, reset logs, smoke.log and run.log (tagged mpirun output), the source snapshot, and
  copies of each rank's `generated/fabric/*.csv` as `r<k>_*.csv`;
- `ranks/`: per-rank logs and info;
- `cache/r<k>`;
- `inputs/`: the staged files.
- Slurm log: `/data/aagarwal/frs-glx/logs/glxq-<quad>-<id>.log`.

**Rules the wrapper enforces:**
- It refuses to submit while any of this user's Galaxy jobs (bh_sc36_6*, bh_110a_sc16) is running or pending: a quad
  takes all four study allocations.
- It never cancels anything.
- The other direction is not enforced: while a quad job is queued, submit no single-Galaxy jobs. The single-Galaxy
  sbatch files do not check for a quad.

## 5. Validation done without a quad

- **`glxq_run.sh --dry`** for pod3, pod8 and `pods` (all five pod quads):
  - staging and the rank binding are written;
  - `tt-run --dry-run` prints the launch. It is one `mpirun-ulfm --bind-to none --tag-output --mca btl self,tcp
    --mca btl_tcp_if_include ens5f0np0 --host <ring>` with four one-process segments. Each carries `TT_MESH_ID=0`,
    the MGD, the FSD, `TT_GLXQ_*` and its `TT_RUN_RANK`.
  - tt-run already adds `--tag-output`, so the harness does not pass it twice.
- **`sbatch --test-only`** cannot give a start time for a 4-node quad request now. It answers "Requested node
  configuration is not available" for any request that includes a node currently allocated, including a single
  allocated node; a single idle pod node (a06u20, a01u14) tests fine. Every quad has such nodes now, so it is an
  availability answer, not a refused request (partition, `-N 4 --ntasks-per-node=1 --exclusive` accepted).
- **Stub run of `glxq.sbatch`** (`GLXQ_STUB=1`, local; srun, tt-smi, the ETH reader and tt-run replaced by stubs; no
  Slurm, no device) for pod8 from a sorted node list:
  - quad and ring order found (also for pod3 from a shuffled node list);
  - the link check against the published FSD (312 per host);
  - smoke and run launches with ranks in ring order; the FSD not passed to tt-run by default;
  - per-rank farm, log and info; results collected.
  - The **timeout path** (tt-run stub sleeping past a 5 s timeout): exit 124, kill, capture, reset, job exit 1.
  - In stub mode the kill and capture steps are inert: they would act on the submit host, which holds the broker's
    P150 cards.
- **Read-only probe of one idle Galaxy** (a01u14, 1 node, ~1 min, no device access): mpirun-ulfm 5.0.7 with
  `plm: slurm`, tt-smi, srun, `ens5f0np0` UP (MTU 9000), the /data tree visible, HOME writable, 32 devices.
- **Mock plumbing (`glxq_mock.sh`, CPU only on a09u32):** tt-run rank-binding mode with 4 mock ranks of a revC 32x4
  subtorus quad (SC36 aisle D d03u02-u20 descriptors), the quad MGD, `glxq_rank.sh`, `glxq_smoke.yaml`, v4txa11's tree.
  - Without the FSD: all 4 ranks reached program launch (the mock's end) on the 128-chip mesh.
  - With the FSD: section 5.1.

### 5.1 Mock with the FSD

- Without `--factory-system-descriptor`: all 4 ranks reach program launch (twice, before and after the smoke yaml got
  the RELAXED mode field).
- With the published FSD: every rank stops at control-plane construction: "A RELAXED mesh graph cannot be used with
  STRICT system health setup".
  - This holds with `RELIABILITY_MODE=relaxed` exported and the yaml field set: test_tt_fabric's first control plane
    (after "Master seed sent", before "Opening devices") is built STRICT.
  - The FSD therefore stays out of the run by default (`--fsd-run` to opt in).
  - Not checked: whether a STRICT copy of the quad MGD would pass. A STRICT graph may use RELAXED health, but STRICT
    channels fail on any missing link.

## 6. Needs a go / not verified

**Needs a go:** any real quad job (4 nodes, all four study allocations). Proposed first job:
```
glxq_run.sh --quad pods --tree v4txa11_1103 --router new --timeout 900 -- \
    ./build/test/tt_metal/tt_fabric/test_infra/test_tt_fabric --test_config /data/aagarwal/frs-glx/glxq/stage/<stamp>/glxq_smoke.yaml
```
That is the smoke row only; its job time is ~15-25 min including reset and links.

**Not verified until a real quad runs:**
- mpirun's Slurm launch across four Galaxies from inside the batch job, including whether PRRTE's out-of-band
  channel needs the NIC pinned too (only the btl is pinned now);
- `tt-smi -glx_reset` in parallel on four cabled Galaxies (tt-sbatch does the same; recover.sh resets and retrains
  with retries);
- the ETH counts on real nodes against 312;
- the FSD ingest on real hosts (`--fsd-run`; see 5.1 for test_tt_fabric);
- `read_eth_speed.py` on a Galaxy (it has run on P150 only; it is generic Blackhole UMD);
- the hang capture on four nodes;
- the cost of building kernels from /data on four ranks at once.

**CCL / model tests:** pytest through tt-run, with the runbook's command and `MESH_DEVICE=QUAD_BH`, needs a tree with
the ttnn Python package and pytest in its python_env.
- The lean study trees have `ttnn/ttnn` and `_ttnn.so`; pytest is not checked.
- Not prepared yet; the harness takes any command.
