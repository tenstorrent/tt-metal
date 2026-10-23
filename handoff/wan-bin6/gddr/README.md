# Toggling the Bin6 GDDR soft harvest on a BH Galaxy

How to switch a Bin6 BH Galaxy between 7 and 8 DRAM banks with the scripts in this directory, run perf in both states, and put it back.

## Background

On `bh-glx-120-b10u20`, Bin6 is a **soft harvest**. Every chip trains all 8 GDDR channels (`bh-mod` prints `[8/8] DRAM`), and `bh-mod`'s `dram_table.soft_harvest_dram_mask` turns one off per chip. The masks are mixed: chip N gets 2^(N mod 8), so each mask is on 4 chips. The same `bh-mod` config also sets `chip_limits.tdp_limit` to **75 W, against a firmware default of 130 W**. So "75 W" on this machine is an override, not the stock limit.

That makes a like-for-like 7-bank vs 8-bank test possible on one machine: same chips, same code, same power limit.

Command shapes follow the Exabox page [GDDR Harvesting — Blackhole Galaxy Bin6 Command Reference](https://tenstorrent.atlassian.net/wiki/spaces/Exabox/pages/2946695384/GDDR+Harvesting+Blackhole+Galaxy+Bin6+Command+Reference).

## Before you start

- **Get the owner's OK.** The harvest was set up on purpose for Bin6 runtime testing (DCAMP-1704, MINFRA-1230). `bh-mod` changes persist across resets for every user of the machine.
- **Make sure nobody else is on the board.** Two jobs on the chips at once have caused bus errors here.
  ```bash
  who
  ps -eo user,cmd | grep -E "[p]ytest|[t]t-smi|[b]h-mod"     # should print nothing
  ```
- **Have `sudo`.** `bh-mod set`/`res` and `tt-smi -glx_reset` need it. The read-only scripts work without it (`SUDO= ./snapshot.sh`).
- **Have a built tt-metal checkout with `python_env`.** `verify.sh` opens the mesh to read the DRAM bank count.
- **Always finish with `restore.sh`.** Don't leave the machine unharvested.

## The scripts

| Script | Changes the board? | What it does |
|---|---|---|
| `snapshot.sh [out_dir]` | no | Records each chip's mask and TDP override, keyed by ASIC ID, plus the evidence the Exabox page asks for (`tt-smi -ls`, `lspci`, `tt-smi -s`). Default `out_dir`: `~/gddr-state/<host>-<timestamp>/` |
| `unharvest.sh <snapshot.tsv> [--apply]` | only with `--apply` | `bh-mod res dram_table.soft_harvest_dram_mask` on every chip, `tt-smi -glx_reset`, then `verify.sh unharvested`. Without `--apply` it only prints the commands |
| `restore.sh <snapshot.tsv> [--apply]` | only with `--apply` | Sets every chip's mask back to the snapshot's value (matched by ASIC ID), resets, then `verify.sh <snapshot>`. Without `--apply` it runs `bh-mod set -n` (dry run) on every chip |
| `verify.sh unharvested\|<snapshot.tsv>` | no | Checks the masks per chip, that all 32 chips are on PCIe, the DRAM bank count tt-metal sees (`dram_grid_size().x`: 8 or 7), and the ethernet links |

All of them run from the repo root (`cd ~/tt-metal`) or anywhere else; they find the repo from their own path. Environment overrides: `BHMOD=` (path to `bh-mod`, default `/data/local-syseng/dc-deploy-pkgs/bh-mod`), `SUDO=` (empty to skip `sudo`), `EXPECTED_CHIPS=` (default 32).

## Step by step

```bash
cd ~/tt-metal
G=handoff/wan-bin6/gddr
```

### 1. Snapshot the current state

```bash
$G/snapshot.sh ~/gddr-state/before
```

Keep this directory until you have restored. It is the only record of which chip had which mask. Expected output, one row per chip, ending with:

```
masks: 1 x4  2 x4  4 x4  8 x4  16 x4  32 x4  64 x4  128 x4
tdp:   75 x32
PCIe devices: 32
```

Then check the board is healthy before changing anything:

```bash
$G/verify.sh ~/gddr-state/before/snapshot.tsv
```

```
[..] 1/4 bh-mod soft_harvest_dram_mask per chip
  32 chips checked
[..] 2/4 PCIe enumeration
  32 Tenstorrent PCIe devices
[..] 3/4 DRAM banks seen by tt-metal
  dram_grid_size().x = 7 (expected 7)
[..] 4/4 ethernet links
[..] PASS
```

If links are down here, run `sudo tt-smi -glx_reset` and verify again before going on.

### 2. Unharvest (back to 8 banks)

Dry run first. It prints one `bh-mod ... res` command per chip and changes nothing:

```bash
$G/unharvest.sh ~/gddr-state/before/snapshot.tsv
```

Then apply. It resets every chip's mask, power-resets the Galaxy (1–2 min), and runs `verify.sh unharvested`:

```bash
$G/unharvest.sh ~/gddr-state/before/snapshot.tsv --apply
```

Success ends with `dram_grid_size().x = 8 (expected 8)` and `PASS`.

### 3. Run perf

Use the same environment as `../perf_ring_4x8.md` ("Reproduce"), then:

```bash
pytest --timeout 5000 models/tt_dit/tests/models/wan2_2/test_performance_wan.py -k "ring_bh_4x8_sp1tp0"                 # 75 W override
TT_METAL_TDP_LIMIT_WATTS=130 pytest --timeout 5000 models/tt_dit/tests/models/wan2_2/test_performance_wan.py -k "ring_bh_4x8_sp1tp0"
sudo tt-smi -glx_reset                                                                                                     # drops the 130 W setting
```

With 8 banks, the branch's `is_dram_harvested()` returns false, so the Wan preset decodes the VAE full-T (no chunking). Compare against the 7-bank numbers in `../perf_ring_4x8.md` at the same power limit.

### 4. Restore Bin6

Dry run first (`bh-mod set -n` on every chip; it checks the snapshot and that every ASIC is present):

```bash
$G/restore.sh ~/gddr-state/before/snapshot.tsv
```

Then apply. It sets each chip's mask, resets, and runs `verify.sh` against the snapshot:

```bash
$G/restore.sh ~/gddr-state/before/snapshot.tsv --apply
```

Success ends with `dram_grid_size().x = 7 (expected 7)` and `PASS`. Take a final snapshot for the ticket and diff it against the first. Only the timestamp and device paths may differ:

```bash
$G/snapshot.sh ~/gddr-state/after
diff <(cut -f1,2,4,5 ~/gddr-state/before/snapshot.tsv | grep -v '^#') <(cut -f1,2,4,5 ~/gddr-state/after/snapshot.tsv | grep -v '^#')
```

## If something goes wrong

| Symptom | What to do |
|---|---|
| `board looks busy, stop these first` | Someone (maybe you) is running pytest, tt-smi or bh-mod. Wait or stop your own job; don't kill other users' jobs |
| `snapshot ... was not taken on <host>` | You passed a snapshot from another machine. Use this machine's |
| `no device node for ASIC ...` | A chip is missing from `/dev/tenstorrent/by-id`. Run `sudo tt-smi -glx_reset`, then `lspci -d 1e52: \| wc -l` (expect 32). If it is still missing, stop and escalate |
| `verify.sh`: `MISMATCH ASIC ...` | That chip's mask didn't take. Re-run the same `--apply` command; the scripts are safe to repeat |
| `verify.sh`: `chips have links down` | Ethernet training is flaky across resets on these machines. `sudo tt-smi -glx_reset` and run `verify.sh` again |
| `verify.sh`: wrong `dram_grid_size().x` | The masks are right but the reset didn't apply them. `sudo tt-smi -glx_reset` and verify again |
| Still failing after one extra reset | Stop. Per the Exabox page: escalate if reset fails, ASICs disappear, or the topology doesn't return. Restore from the snapshot as soon as the board is back |

## Notes

- **Chips are keyed by ASIC ID** (`tt_asic_id`), and the device node comes from `/dev/tenstorrent/by-id/` each time. `/dev/tenstorrent/N` does not follow PCI order here (PCI `81:00.0` is `/dev/tenstorrent/24`), so a restore by device number could put a mask on the wrong chip.
- **The TDP override is left alone.** Pick the power per run with `TT_METAL_TDP_LIMIT_WATTS`. The setting lasts until the next reset, and `0` goes back to the 75 W override, not the firmware's 130 W. To remove the override itself (not needed here): `sudo bh-mod res chip_limits.tdp_limit` per chip, then reset.
- **Tested so far:** `snapshot.sh`, `verify.sh` and the `unharvest.sh` dry run, on b10u20, with no changes made. The `--apply` paths and `restore.sh` have not been run yet. Watch the first real run closely.
