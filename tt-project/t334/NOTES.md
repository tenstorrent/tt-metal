# t334: per-layer conv3d table, conv VAE decode on 4x8 (feeds t332/FINDINGS.md section 5, lever 1)

## State (2026-10-10)
- 2x4 baseline (#96 job 436, 1.15 GHz) analysed: `results/t96_2x4_table.txt`, `t96_2x4_conv3d.csv`, `t96_2x4_ops.csv`.
  conv3d 360.7 ms (sum of per-layer max), 26.4 TFLOP/chip, 51.7 % of HiFi4 peak weighted.
  Every conv3d runs at **HiFi2** with fp32 dest acc (FINDINGS section 5 assumed HiFi4), so that is ~26 % of HiFi2 peak.
  Halo per chip (42 ops): min/median/max 8.89/23.00/54.36 ms; the slowest-compute chip has the least halo time
  (halo durations include CCL wait).
- 4x8 job queued on the blx03 serial runner: `t334-prof4x8-r1` (spec `spec-t334-prof4x8-r1`, TIMEOUT 600, cold JIT).
  Marker: `g14blx03:/var/tmp/fasth3/runner/done/t334-prof4x8-r1.done`.
  Probe: `ssh -o BatchMode=yes -o ConnectTimeout=20 g14blx03 'bash ~/fasth3/runner/probe.sh t334-prof4x8-r1'`.

## Setup of the 4x8 job
- Code: overlay of origin/ttp/t48-ltx25-integrated @ f6547442b30 (models, conftest, pytest.ini) at
  /var/tmp/fasth3/t334/src plus `test_vae_ltx_prof_4x8.py`; built tree ~/fasth3/t315 (same commit, read-only) as TT_METAL_HOME.
- Full (4,8) mesh, FABRIC_1D, H on axis 0 (4), W on axis 1 (8), Linear CCL, 2 links, decoder's own blockings;
  t48 defaults HALO_ONLY, FOLD_TIME_PAD, EXACT_SHARD on; LTX_FUSE_YUV_OUTPUT=1 (parity with #96); LTX_PIN_CORES=0.
- Weights: random-init decoder (torch.manual_seed(42) on _TorchLTXVideoDecoder, prod blocks), NOT the real
  LTX-2.5 conv VAE. Per-op timing does not depend on the weights (user #235).
- Latent: saved 1080p latent /home/smarton/fasth3/out/t37/s2reuse0/lat.gen0.pt (128x19x34x60).
- Tracy: tracy-capture started by run334.sh in its own process group, test run with `--no-capture-tool`, report
  with `--process-logs-only` (tracy's own capture path setsid's the test and launches a WASM server into TT_METAL_HOME).
- Clock: whatever blx03 has (not changed); numbers are relative only.

## Next step (on wake)
1. `ssh g14blx03 cat /var/tmp/fasth3/runner/done/t334-prof4x8-r1.done`; on failure quote the last 20 lines and the first error line of the log.
2. Copy back `/var/tmp/fasth3/t334/{ops_perf_4x8.csv.gz,t334_4x8_ops.csv,t334_4x8_conv3d.csv,t334_4x8_table.txt}` to `results/`.
3. Compare with the 2x4 table (per-layer ms, %HiFi4, halo totals), write FINDINGS here.
4. Clean up blx03 /var/tmp/fasth3/t334 (jit, prof, src).

## Side observation
blx03 /var/tmp/fasth3 holds 457G (cache 295G, models 161G): not "small" per the charter. Not ours to delete in this task.
