# #43 incident: blx03 unreachable after job 989 (2026-10-01, UTC)
- 02:09:43 job 988 (smarton, run43.sh): pytest setup error (root conftest not loaded), no device opened.
- 02:10:38 job 989 (smarton, run43.sh, 1150 MHz cap): opened a bare (2,4) mesh_device with FABRIC_1D on the galaxy.
- 02:11:08 open_mesh_device threw: "Fabric Router Sync: Timeout after 10000 ms on Device 1: expected 0xa2b2c2d2.
  Master chan=3 got 0xa1b1c1d1 (REMOTE_HANDSHAKE_COMPLETE); chan 4,5 stuck at 0xa0b0c0d0 (STARTED)";
  chans 2,3,6,7 completed handshake. No decode ran. Job exited ~02:11:12 (34 s). /dev/tenstorrent had 33 entries right after.
- 02:11:12 broker auto health check "fabric traffic pass across all links (~45s)" (job 990, fabric-check.sh) started.
  Still running at ~50 s (last seen ~02:12:05).
- ~02:12-02:14: ssh to g14blx03 hangs, then connection timed out; ping 100% loss at 02:14:30.
Not seen: whether a chip left PCIe (no access to the box after 02:12). Likely trigger candidates: the broker's
all-link fabric check after a failed fabric init, or the tray instability seen 2026-09-30. Unknown which.
Our state on blx03: ~/fasth3/t43 (test, run43.sh, analyze43.py, run43.log), /var/tmp/fasth3/t43 (jit, prof): to clean up.
- ~02:16: ssh answers 'System is booting up' -> blx03 rebooted. Chip state after boot not checked (no device work allowed).
