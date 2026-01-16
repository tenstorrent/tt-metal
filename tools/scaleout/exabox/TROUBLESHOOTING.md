# Exabox Validation Troubleshooting

Real issues encountered and their solutions.

---

## SSH Agent Forwarding for MPI

MPI tasks require passwordless SSH between hosts.

**Setup**:
```bash
ssh-agent
ssh-add
ssh-add -L  # verify your key is listed
```

**Login with agent forwarding**:
```bash
ssh -A [username]@[hostname]
```

If you get a `known_hosts` warning, delete the offending line from `~/.ssh/known_hosts`.

**Important**: Do NOT copy SSH keys between machines for MPI. Always use agent forwarding (`ssh -A`).

---

## SLURM Nodes Showing Down

**Symptom**: `sinfo` shows nodes as "down" instead of "idle".

**Cause**: Node may have rebooted or had a transient issue.

**Solution**: Run `sudo tt_reidle_downed_node.sh` on the login node to bring them back to idle.

**Diagnosis**: Check `uptime` on the affected node. If uptime is low, the machine rebooted unexpectedly - may need further investigation.

---

## tt-smi Reset Requires Sudo Password in SLURM Session

**Symptom**: Running `tt-smi -glx_reset` via mpirun in a SLURM session fails with:
```
IPMI command failed: sudo: a terminal is required to read the password
```

Running `groups` in the SLURM session shows wrong groups (e.g., another user's group).

**Cause**: UID/group mismatch between slurm-login node and galaxy nodes. The SLURM session inherits incorrect group IDs.

**Workaround**: SSH directly to the galaxy node (not via srun) and run the reset:
```bash
ssh <your-username>@<galaxy-host>.exabox.tenstorrent.com
tt-smi -glx_reset
```

If this issue persists, report to infra team to fix group synchronization.

---

## Tests Fail with "Permission denied" on tt-metal-cache

**Symptom**: Running tests with your personal account fails with:
```
cannot map elf file into memory: Permission denied
```
Error points to `/tmp/tt-metal-cache/...`

**Cause**: The `/tmp/tt-metal-cache` directory was created by a different user (e.g., local-syseng) from a previous test run.

**Solution**: Remove the cache directory:
```bash
rm -rf /tmp/tt-metal-cache
```
Then re-run your tests.

---

## Hanging on PCIe Device Lock

**Symptom**: Your workload hangs trying to acquire a lock on a PCIe device, even though you have a SLURM reservation.

**Cause**: Someone else is using the machine via bare metal SSH (e.g., local-syseng account for bringup work) outside of SLURM.

**Diagnosis**: SSH into the machine and run `who` to see logged in users.

**Solution**: Coordinate with the other user. If they're using bare metal SSH for bringup, ask them to finish or yield the machine. Report conflicts in #exabox-infra.

---

## Machine Rebooted with PCIe Errors

**Symptom**: Machine unexpectedly reboots. Kernel logs (dmesg) show PCIe hardware errors.

Example from kernel logs:
```
[Hardware Error]:  fru_text: PcieError
[Hardware Error]:   section_type: PCIe error
[Hardware Error]:   port_type: 4, root port
[Hardware Error]:   device_id: 0000:c0:01.4
pcieport 0000:c0:01.4: AER: aer_status: 0x00002000, aer_mask: 0x00000000
pcieport 0000:c0:01.4:    [13] NonFatalErr
pcieport 0000:c0:01.4: AER: aer_layer=Transaction Layer, aer_agent=Receiver ID
```

**Cause**: PCIe errors on root ports, possibly from device communication issues.

**Solution**: File an issue on exabox-infra repo and report the machine. Check `dmesg` for the full error log.

---

## UMD Firmware Version Mismatch - Links Not Detected

**Symptom**: Validation reports massive numbers of missing connections (e.g., 180 missing port/cable connections). Every single Trace and Linking Board connection appears to not be trained. However, when checked with the eth status script, all ports show `PORT_UP` and `LINK_TRAIN_PASS`.

Example validation output:
```
Physical Discovery found 180 missing port/cable connections:
  - PhysicalPortEndpoint{hostname='bh-glx-c01u02', ... port_type=TRACE, port_id=1} <-> ...
  - PhysicalPortEndpoint{hostname='bh-glx-c01u02', ... port_type=TRACE, port_id=3} <-> ...
```

But eth script shows ports are actually up:
```
eth_postcode    - PASS
serdes_postcode - PASS
macpcs_postcode - PASS
port_status  - PORT_UP
train_status - LINK_TRAIN_PASS
```

**Cause**: UMD only checks major.minor firmware version, ignoring patch version. If ERISC FW is a patch build (e.g., 1.7.103 when UMD expects 1.7.1 for FW Bundle 19.3.0), UMD ignores the links entirely.

UMD log showing the mismatch:
```
2026-01-05 23:41:02.445 | info | UMD | Firmware bundle version 19.3.0 on the system is newer than the latest fully tested version 19.1.0 for blackhole architecture.
```

The root cause was that patch builds (like 1.7.103) didn't update the version number in the image, so UMD saw version 19.3.0 but expected specific ERISC version 1.7.1.

**How links are validated**: The validation script compares eth connectivity reported by the driver (UMD) against a golden representation (FSD). UMD doesn't report a link if it's not trained. Specifically, it expects `PORT_UP` status at address `0x7CC04` on all links.

**Solution**: Use a UMD version that matches the firmware version on the cluster, or update UMD to accept the patch version.

---

## Missing ASIC After Reboot

**Symptom**: Only 31 chips visible after `tt-smi reset` instead of 32. One ASIC doesn't come up.

Example: bh-glx-c01u08 showing only 31 chips, missing tray_id=2, asic_location=7.

**Context**: This happened after the machine initially had some QSFP connections not showing up, then rebooted itself.

**Cause**: ASIC failed to initialize after software reboot. Software reboot (`tt-smi reset` or OS reboot) is not sufficient to recover the chip.

**Solution**: Full power cycle via BMC (not just software reboot). Contact someone with BMC access to power cycle the machine.

After power cycle, verify all 32 chips come up:
```bash
tt-smi -l  # should show 32 devices
```

---

## GDDR Issue on Chip

**Symptom**: One chip showing GDDR issues in tt-smi. Example: UBB2 ASIC8 having GDDR issue.

**Context**: This appeared out of nowhere - all prior testing showed everything clean. No sensitivity to FW version was found.

**Cause**: GDDR training failure on that specific chip.

**Solution**: Try virtual AC power cycle first to see if it's recoverable. If GDDR issue persists after power cycle, it's a hardware problem that may require chip/board replacement.

---

## tt-smi Reset Fails with ARC Timeout

**Note**: As of tt-smi 3.1.1, use `tt-smi -r` instead of `tt-smi --glx-reset` for faster resets. The new reset goes through the driver instead of BMC, doesn't need sudo, and works in docker/VMs.

**Symptom**: `tt-smi -glx_reset` (or `tt-smi -r`) fails with ARC timing out.

**Cause**: Can be transient or caused by underlying GDDR issues on a specific chip.

**Resolution steps**:
1. Try a soft reboot first (faster than power cycle)
2. If reboot doesn't help, try a full power cycle
3. If still failing after power cycle, it's likely a hardware issue (often GDDR related)

**Diagnosis**: If issue persists, have syseng check for GDDR BIST failures. Common signature: UBBx ASICy GDDRz failing BIST, related to CA.

**Solution for GDDR BIST failure**:
- Syseng may attempt to revive with debug firmware reload
- If no improvement, UBB tray swap required

---

## Z Ports Showing Down

**Symptom**: Diagnostic script shows all Z ports (ETH10+) are down.

**Cause**: Z ports aren't cabled in your current topology.

**Solution**: This is expected behavior. In an 8x16 topology (4 Galaxies), Z ports aren't used. Only ETH00-ETH09 should be up.

Only investigate if Z ports are supposed to be connected in your topology.

---

## Tensix Cores Unresponsive During Validation

**Symptom**: Traffic tests that load management firmware onto tensix and ethernet cores fail. SW reports all tensix cores as unresponsive across all hosts. The workload waits for a response as part of initialization (by polling) and never gets it.

**Observation**: Skipping the step where firmware is loaded onto tensix and immediately launching TX and RX kernels on ethernet cores works fine. Ethernet links are healthy across multiple runs when tensix is bypassed.

**Cause**: Unknown - possibly firmware/driver interaction issue. The firmware team noted this is strange because they don't have anything that runs on the Tensixes. Each host was broken in the same way.

**Solution**: Power cycle resolved the issue, though root cause remains unknown. This is concerning but not an immediate blocker.

**Debugging approach**: Start with debug on the SW side - do basic reads and writes over PCIe to ensure data lands in L1 as expected. Loop in syseng as needed depending on what you see.

**Workaround**: If ethernet-only tests pass, the physical links are healthy. The issue is in the tensix firmware loading path, not the ethernet interconnect.

**Escalation**: Loop in firmware team if this blocks testing.

---

## Transient Ethernet Connectivity Loss

**Symptom**: Missing connections between hosts that were previously stable. Ethernet was working earlier but now shows missing links.

Example: Missing connections between c01u02 and c01u08 - physical cables checked and found no issues.

**Cause**: Transient issue - links failed to train after some event (reset, temperature change, etc.) but no physical damage.

**Diagnosis**:
1. Check physical connectivity first (cables seated properly)
2. If physical looks fine, it's likely a transient issue

**Solution**: Power cycle both affected machines via BMC. This resolved ethernet connectivity issues on the 8x16 cluster.

---

## QSFP Connections Missing Between Hosts

**Symptom**: Validation shows missing QSFP_DD port connections between specific hosts. All missing connections are between the same pair of machines.

Example output:
```
Physical Discovery found 12 missing port/cable connections:
  - PhysicalPortEndpoint{hostname='bh-glx-c01u08', ... tray_id=3, port_type=QSFP_DD, port_id=1} <-> PhysicalPortEndpoint{hostname='bh-glx-c02u08', ... tray_id=4, port_type=QSFP_DD, port_id=1}
  - PhysicalPortEndpoint{hostname='bh-glx-c01u08', ... tray_id=3, port_type=QSFP_DD, port_id=2} <-> ...
  ... (multiple QSFP_DD connections between same host pairs)
```

**Cause**: QSFP cables physically unseated, not fully connected, or bad cable. Can happen during transport, vibration, or accidental contact.

**Resolution steps** (in order):

1. Identify the affected hosts and trays from the error output
2. Have someone physically reseat the QSFP cables at the reported ports
3. Re-run validation
4. If still missing, power cycle both affected machines
5. If still missing after power cycle, swap the cable
6. If issue persists after cable swap, escalate to syseng for hardware investigation

In the example above, QSFPs on C02U08 were found to be down. Reseating the cables resolved the issue.

---

## Trace Connections Hanging After Reset (~30% Failure Rate)

**Symptom**: Physical validation tests show ~70% pass rate. Trace connections on certain hosts hang after some resets. Consistent failure signature across runs.

**Root Cause**: ERISC 0 (responsible for link recovery through base FW) was also running user TX/RX kernels. The BH context switch mechanism does not run link recovery when called. If a link failed training or dynamically went down, recovery never gets invoked while user kernels are running.

**Solution**:
1. Switch execution to ERISC 1 only (single ERISC mode)
2. Add 5 second delay after each `tt-smi reset` to wait for training to stabilize

**Result**: With single ERISC + 5s delay after reset, 48/50 runs passed physical validation (96%).

**Remaining issues**:
- If links take long to stabilize post-reset, they can be reported as missing. Try longer sleeps.
- Occasional data mismatches still occur (under investigation)

**Long-term fix**: Fabric team adding coordinated context switch with link recovery API. Until then, run validation and fabric tests with single ERISC.

**Environment variable**: To force single ERISC mode:
```bash
export TT_METAL_DISABLE_MULTI_AERISC=1
```

---

## Do NOT Update Firmware on Cluster Machines

**Warning**: Exabox cluster machines are running debug firmware (built off 19.3.1 with debug eth fw not merged to main).

**Do not run firmware updates** - updating to released firmware will make the machine unusable in a cluster configuration.

If firmware updates are needed, coordinate with syseng and scaleout teams first.

---

## ERISC Workarounds and Known Issues

Reference for ERISC-related issues discovered during bringup (from GitHub issue #25427):

**ND hangs during init (Mailbox API to Metal FW)**
- Went away after shifting binaries by 32 bytes
- Workaround: Once RISC gets to metal FW, save registers and soft reset both ERISC0 and ERISC1
- Suspected cause: Stuck data in the instruction cache (i$)
- Code: `tt_metal/hw/firmware/src/tt-1xx/active_erisc.cc` lines 117-195

**Interrupts causing retrain counts to increase**
- Interrupts were disabled as workaround
- Code: `tt_metal/hw/firmware/src/tt-1xx/active_erisc.cc` line 203

**NOC usage conflicts**
- Base FW runs on ERISC0 and uses NOC0
- Kernel creation API enforces: ERISC0 → NOC0, ERISC1 → NOC1
- This prevents conflicts between base FW and user kernels

---

## Data Mismatch During Traffic Tests

**Symptom**: Validation reports data mismatch - data sent doesn't match data received. Can occur even when all links show as trained.

Example from logs:
```
8 words of data mismatch
```

**Cause**: Still under investigation. Can occur alongside missing links or independently.

**Workaround**: Re-run the test. If it consistently fails on the same link, it's likely a physical issue (cable/port). If failures are scattered, may be timing-related.

---

## Machine Won't Power On After BMC Power Cycle

**Symptom**: After remote power cycle, machine doesn't come back up. BMC shows it going back to "Off" state immediately.

BMC logs example:
```
Jan 08 02:27:58 s7tk power-control[4462]: Host0: Moving to "Wait for Power Supply Power OK" state
Jan 08 02:28:58 s7tk power-control[4462]: Host0: Moving to "Off" state
```

ipmitool shows UBB tray POST failures:
```
ipmitool -I lanplus -H <bmc-ip> -U root -P '<password>' sel elist
62bf | 01/07/2026 | 13:36:05 MST | Processor UBB3_ASIC0_Stat | FRB2/Hang in POST failure | Asserted
62c0 | 01/07/2026 | 13:36:05 MST | Processor UBB3_ASIC1_Stat | FRB2/Hang in POST failure | Asserted
...
```

**Cause**: PDU circuit breaker tripped due to di/dt (current spike during power cycling).

**Solution**:
1. Physically check the PDU - look for tripped circuit breakers
2. Reset the breaker
3. Power cycle the machine again

This requires physical access to the datacenter.

**Prevention**: Consider power cycling one node at a time to avoid di/dt issues.

---

## Tensix Stall Issue (Requires Power Cycle)

**Symptom**: Cluster validation or tests hang on tensix operations. Multiple hosts affected in the same way.

**Context**: Observed on both 8x16 (C01/C02) and 4x32 (B08/B09) clusters. Goes away with power cycle but root cause unclear.

**Cause**: Unknown - possibly something in SW or the handoff process during cluster bringup. Lower priority for debugging since power cycle resolves it.

**Solution**: Power cycle the affected hosts via BMC. This clears the stall and allows cluster validation to proceed.

**Concern**: If this happens repeatedly on the same system after handoff to SW, it needs more investigation.

---

## Single ERISC Fabric Workloads Hang Immediately

**Symptom**: When running fabric workloads with single ERISC mode (`TT_METAL_DISABLE_MULTI_AERISC=1`), the workload hangs immediately.

**Cause**: Likely SW regression. Pipelines aren't regularly tested with single ERISC on BH, so something probably regressed.

**Workaround**: Run fabric workloads with 2 ERISCs instead. 2D fabric stress tests pass with dual ERISC.

**Note**: Physical validation can still use single ERISC, but fabric tests need dual ERISC.

---

## Intermittent Tensix Cores Hung at Device Startup

**Symptom**: Arbitrary tensix cores are hung/unresponsive at device startup. Happens intermittently.

**Cause**: Under investigation.

**Workaround**: Power cycle and retry. Issue is intermittent.

---
